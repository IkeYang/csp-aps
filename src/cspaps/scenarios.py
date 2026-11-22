"""Annual scenario sets with spatial DNI maps: S = {(sun_s, h_s, D_s(x, y))}.

Generation from the weather year (``scenarios.source = "weather"``)
-------------------------------------------------------------------
1. Every valid hour t of the weather year (sun above ``annual.min_sun_elevation_deg``,
   DNI > 0) has sun direction s_t, station DNI G_t and duration w_t.
2. Map regime. Each hour is given one map of the library: ``by_day`` cycles the
   maps day by day (day k uses map k mod M), ``single`` uses the first map.
   The regime decides only *where* the DNI falls on the site, not how much:
   factor maps have site mean 1, so the site-mean DNI of hour t stays G_t.
3. Within each regime r the hours are clustered into K_r representative states
   (DNI-energy-weighted k-means on the sun vector, ``sun.sun_bins``). K_r is the
   largest-remainder share of ``n_scenarios`` by the regime's DNI energy (at
   least 1). A state gets the energy-weighted mean direction, the energy-weighted
   mean DNI G_s = sum G_t^2 w_t / sum G_t w_t and the hours
   h_s = sum G_t w_t / G_s, so G_s h_s equals the DNI energy of its members:
       sum_s h_s mean_site(D_s) = sum_s h_s G_s = sum_t w_t G_t      (exact)
4. D_s(x, y) = G_s f_{r(s)}(x, y) for factor maps (absolute maps: D_s = g_r).

The same regime assignment gives a denser set with the same map energies (used
for the sampling-error check of the representative scenarios).

A scenario set is saved as ``scenarios.csv`` + ``dni_maps.csv`` / ``dni_maps.json``
and identified by a fingerprint (hash of the table and the maps), so layouts are
only compared on the same set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

from . import sun
from .dni_maps import DNIMap, load_library, model_maps, save_library, uniform_map


@dataclass
class ScenarioSet:
    samples: sun.Samples  # elevation, azimuth, scenario DNI, hours, maps, map_index, ids
    maps: dict[str, DNIMap]
    meta: dict = dc_field(default_factory=dict)
    example_times: list | None = None  # an hour of the weather year represented by each scenario

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def ids(self) -> list[str]:
        return list(self.samples.ids)

    def map_id(self, i: int) -> str:
        return self.samples.map_of(i).map_id

    def index(self, scenario_id: str) -> int:
        try:
            return self.ids.index(str(scenario_id))
        except ValueError:
            raise KeyError(f"unknown scenario {scenario_id!r}; known: {self.ids}") from None

    def dni_at(self, i: int, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.broadcast_to(np.asarray(self.samples.dni_at(i, xy), dtype=float), (len(xy),)).copy()

    def table(self) -> pd.DataFrame:
        s = self.samples
        df = pd.DataFrame({
            "scenario_id": self.ids,
            "elevation_deg": np.round(s.elevation, 6), "azimuth_deg": np.round(s.azimuth, 6),
            "hours": np.round(s.weight_h, 6), "dni_w_m2": np.round(s.dni, 6),
            "map_id": [self.map_id(i) for i in range(len(self))],
            "dni_energy_kwh_m2": np.round(s.dni * s.weight_h / 1e3, 6),
        })
        if s.members is not None:
            df["hourly_members"] = s.members
        if self.example_times is not None:
            df["example_time"] = [str(t) if t is not None else "" for t in self.example_times]
        return df

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        t = self.table()[["scenario_id", "elevation_deg", "azimuth_deg", "hours", "dni_w_m2", "map_id"]]
        h.update(t.to_csv(index=False, float_format="%.6f").encode())
        for k in sorted({self.map_id(i) for i in range(len(self))}):
            h.update(self.maps[k].digest().encode())
        return h.hexdigest()[:16]

    def record(self) -> dict:
        used = sorted({self.map_id(i) for i in range(len(self))})
        hours = self.table().groupby("map_id")["hours"].sum().to_dict()
        energy = self.table().groupby("map_id")["dni_energy_kwh_m2"].sum().to_dict()
        return {"fingerprint": self.fingerprint(), "n_scenarios": len(self), "total_hours": float(self.samples.weight_h.sum()),
                "annual_dni_kwh_m2": round(self.samples.dni_hours, 3), "maps": used,
                "hours_per_map": {k: round(v, 3) for k, v in hours.items()},
                "dni_energy_per_map_kwh_m2": {k: round(v, 3) for k, v in energy.items()},
                "map_records": {k: self.maps[k].record() for k in used}, **self.meta}

    def subset_maps(self) -> dict[str, DNIMap]:
        used = {self.map_id(i) for i in range(len(self))}
        return {k: v for k, v in self.maps.items() if k in used}

    def save(self, folder) -> dict:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = save_library(self.subset_maps(), folder)
        files["scenarios_csv"] = folder / "scenarios.csv"
        self.table().to_csv(files["scenarios_csv"], index=False)
        files["scenario_set_json"] = folder / "scenario_set.json"
        files["scenario_set_json"].write_text(json.dumps(self.record(), indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        return files


def _regimes(case, maps: dict[str, DNIMap]) -> list[str]:
    spec = case.scenarios
    if spec.map_sequence:
        missing = [m for m in spec.map_sequence if m not in maps and m != "uniform"]
        if missing:
            raise ValueError(f"scenarios.map_sequence names unknown maps {missing}; available {sorted(maps)}")
        seq = list(spec.map_sequence)
    else:
        seq = [k for k in maps if k != "uniform"] or ["uniform"]
        if spec.map_include_uniform and "uniform" in maps:
            seq.append("uniform")
    if spec.map_assignment == "single":
        seq = seq[:1]
    return seq


def _allocate(energies: np.ndarray, n_total: int) -> np.ndarray:
    """Largest-remainder share of n_total by energy, at least one state per regime with energy."""
    e = np.asarray(energies, dtype=float)
    pos = e > 0
    k = np.zeros(e.size, dtype=int)
    if not pos.any():
        return k
    n_total = max(int(n_total), int(pos.sum()))
    quota = n_total * e / e.sum()
    k = np.where(pos, np.maximum(1, np.floor(quota).astype(int)), 0)
    while k.sum() < n_total:
        j = int(np.argmax(np.where(pos, quota - k, -np.inf)))
        k[j] += 1
    while k.sum() > n_total:
        cand = np.flatnonzero(k > 1)
        if not cand.size:
            break
        j = cand[int(np.argmin((quota - k)[cand]))]
        k[j] -= 1
    return k


def regime_of_hours(case, hourly: sun.Samples, n_regimes: int) -> np.ndarray:
    """Map regime of every hour: day of the year modulo the number of maps (``by_day``)."""
    if n_regimes <= 1 or hourly.times is None:
        return np.zeros(len(hourly), dtype=int)
    days = np.asarray((hourly.times.normalize() - hourly.times.normalize()[0]).days, dtype=int)
    start = hourly.times[0].normalize().day_of_year - 1
    return (days + start) % n_regimes


def build_scenarios(case, n_scenarios: int | None = None, hourly: sun.Samples | None = None,
                    dni_bands: int | None = None) -> ScenarioSet:
    """The case's scenario set (from the weather year, or the scenario table of ``scenarios.file``)."""
    spec = case.scenarios
    maps = model_maps(case)
    if spec.source == "file":
        return load_table(case, case.resolve(spec.file), maps)
    n_scenarios = int(n_scenarios or spec.n_scenarios)
    bands = int(dni_bands or spec.dni_bands)
    hourly = hourly if hourly is not None else sun.hourly_samples(case)
    regimes = _regimes(case, maps)
    if "uniform" in regimes and "uniform" not in maps:
        maps["uniform"] = uniform_map()
    reg = regime_of_hours(case, hourly, len(regimes))
    energy = np.array([float((hourly.dni * hourly.weight_h)[reg == r].sum()) for r in range(len(regimes))])
    k_r = _allocate(energy, n_scenarios)
    parts, map_index, examples = [], [], []
    for r, k in enumerate(k_r):
        if k <= 0:
            continue
        sub = hourly.subset(np.flatnonzero(reg == r))
        bins = sun.sun_bins(sub, int(k), seed=spec.seed, dni_bands=bands)
        parts.append(bins)
        map_index += [r] * len(bins)
        # an hour of this regime with the closest sun direction (illustration only)
        v_sub = sun.sun_vector(sub.elevation, sub.azimuth)
        v_bin = sun.sun_vector(bins.elevation, bins.azimuth)
        near = np.argmax(v_bin @ v_sub.T, axis=1)
        examples += [sub.times[j] if sub.times is not None else None for j in near]
    if not parts:
        raise ValueError("no valid hours in the weather year (sun above the minimum elevation with DNI > 0)")
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
    members = np.concatenate([p.members if p.members is not None else np.zeros(len(p), int) for p in parts])
    map_list = [maps[m] for m in regimes]
    samples = sun.Samples(cat("elevation"), cat("azimuth"), cat("dni"), cat("weight_h"), None, "scenarios",
                          members, map_list, np.asarray(map_index, dtype=int), None)
    order = np.lexsort((samples.azimuth, samples.map_index))
    samples = samples.subset(order)
    examples = [examples[j] for j in order]
    samples.ids = [f"S{k + 1:02d}" for k in range(len(samples))]
    _quantise(samples)
    meta = {"source": "weather year: " + (str(case.weather.file) if case.weather.source == "file"
                                          else f"clear sky ({case.sun.dni_model})"),
            "generation": f"DNI-energy-weighted k-means per map regime, {bands} DNI band(s)",
            "map_model": spec.map_model, "map_assignment": spec.map_assignment, "regimes": regimes,
            "states_per_regime": {m: int(k) for m, k in zip(regimes, k_r)},
            "hourly_dni_kwh_m2": round(hourly.dni_hours, 3), "n_hours": len(hourly),
            "min_sun_elevation_deg": case.annual.min_sun_elevation_deg}
    return ScenarioSet(samples, maps, meta, examples)


def _quantise(samples: sun.Samples) -> None:
    """Round the scenario table to its saved precision, so a saved and reloaded set is identical."""
    for name in ("elevation", "azimuth", "dni", "weight_h"):
        setattr(samples, name, np.round(np.asarray(getattr(samples, name), dtype=float), 6))


def load_table(case, path, maps: dict[str, DNIMap] | None = None) -> ScenarioSet:
    """Scenario table CSV: scenario_id,elevation_deg,azimuth_deg,hours,dni_w_m2,map_id[,time]."""
    maps = dict(maps if maps is not None else model_maps(case))
    df = pd.read_csv(path)
    need = {"scenario_id", "elevation_deg", "azimuth_deg", "hours", "dni_w_m2", "map_id"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: scenario table needs columns {sorted(need)}")
    if df["scenario_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: scenario IDs must be unique")
    if "uniform" in set(df["map_id"].astype(str)) and "uniform" not in maps:
        maps["uniform"] = uniform_map()
    unknown = sorted(set(df["map_id"].astype(str)) - set(maps))
    if unknown:
        raise ValueError(f"{path}: unknown map ids {unknown}; available {sorted(maps)}")
    for col in ("elevation_deg", "azimuth_deg", "hours", "dni_w_m2"):
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"{path}: column {col} must be finite")
    if (df["hours"] < 0).any() or (df["dni_w_m2"] < 0).any():
        raise ValueError(f"{path}: hours and DNI must be nonnegative")
    order = list(dict.fromkeys(df["map_id"].astype(str)))
    map_list = [maps[m] for m in order]
    idx = np.array([order.index(m) for m in df["map_id"].astype(str)], dtype=int)
    samples = sun.Samples(df["elevation_deg"].to_numpy(float), df["azimuth_deg"].to_numpy(float),
                          df["dni_w_m2"].to_numpy(float), df["hours"].to_numpy(float), None, "scenario_table",
                          df["hourly_members"].to_numpy(int) if "hourly_members" in df else None, map_list, idx,
                          [str(v) for v in df["scenario_id"]])
    _quantise(samples)
    examples = list(df["example_time"].fillna("")) if "example_time" in df else (
        list(df["time"].fillna("")) if "time" in df else None)
    return ScenarioSet(samples, maps, {"source": f"scenario table {Path(path).name}"}, examples)


def load_folder(folder) -> ScenarioSet:
    """A scenario set saved by ``ScenarioSet.save`` (scenarios.csv + dni_maps.*)."""
    folder = Path(folder)
    maps = load_library(folder)
    df = pd.read_csv(folder / "scenarios.csv")
    order = list(dict.fromkeys(df["map_id"].astype(str)))
    idx = np.array([order.index(m) for m in df["map_id"].astype(str)], dtype=int)
    samples = sun.Samples(df["elevation_deg"].to_numpy(float), df["azimuth_deg"].to_numpy(float),
                          df["dni_w_m2"].to_numpy(float), df["hours"].to_numpy(float), None, "scenarios",
                          df["hourly_members"].to_numpy(int) if "hourly_members" in df else None,
                          [maps[m] for m in order], idx, [str(v) for v in df["scenario_id"]])
    meta_path = folder / "scenario_set.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    examples = list(df["example_time"].fillna("")) if "example_time" in df else None
    out = ScenarioSet(samples, maps, {k: v for k, v in meta.items() if k in ("source", "generation", "map_model",
                                                                          "map_assignment", "regimes")}, examples)
    if meta.get("fingerprint") and meta["fingerprint"] != out.fingerprint():
        raise ValueError(f"{folder}: scenario set fingerprint mismatch (files edited after saving?)")
    return out


def is_spatial(case) -> bool:
    """True when the case's DNI is not uniform over the site (any map model but a pure uniform one)."""
    spec = case.scenarios
    return spec.source == "file" or spec.map_model != "uniform"


def attach_regimes(case, hourly: sun.Samples) -> sun.Samples:
    """The hourly samples with the map regime of every hour (unchanged when the DNI is uniform)."""
    if not is_spatial(case) or case.scenarios.source == "file":
        return hourly
    maps = model_maps(case)
    regimes = _regimes(case, maps)
    if "uniform" in regimes and "uniform" not in maps:
        maps["uniform"] = uniform_map()
    out = hourly.subset(np.arange(len(hourly)))
    out.maps = [maps[r] for r in regimes]
    out.map_index = regime_of_hours(case, hourly, len(regimes))
    out.label = hourly.label
    return out


def scenario_samples(case, n_bins: int, dni_bands: int = 1, hourly: sun.Samples | None = None) -> sun.Samples:
    """Map-aware representative states for the layout search (same regimes as the formal set)."""
    return build_scenarios(case, n_bins, hourly, dni_bands).samples


def site_mean_dni(case, sset: ScenarioSet) -> np.ndarray:
    """(S,) mean DNI over the site raster nodes of every scenario (= scenario DNI for normalised factor maps)."""
    from .dni_maps import site_raster

    cell = case.scenarios.map_cell_m
    xs, ys, inside = site_raster(case, cell)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx[inside], gy[inside]]) if inside.any() else np.column_stack([gx.ravel(), gy.ravel()])
    return np.array([float(sset.dni_at(i, pts).mean()) for i in range(len(sset))])


def describe_regime_rule(case) -> str:
    spec = case.scenarios
    if spec.map_assignment == "single":
        return "every hour uses the first map"
    return "day k of the year uses map k mod M (M = number of regimes)"
