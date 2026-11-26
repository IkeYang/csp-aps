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

import calendar
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
    # the weather year behind the set (with the map of every hour) and the scenario of every hour (-1: none);
    # used by the control-variate estimate of the annual value (``layout_value.control_variate``)
    hourly: sun.Samples | None = dc_field(default=None, repr=False)
    assignment: np.ndarray | None = dc_field(default=None, repr=False)

    def hourly_fingerprint(self) -> str | None:
        """Hash of the hours behind the set (None when the set was imported as a table)."""
        if self.hourly is None or self.assignment is None:
            return None
        h = hashlib.sha256()
        hs = self.hourly
        for part in (hs.elevation, hs.azimuth, hs.dni, hs.weight_h):
            h.update(np.round(np.asarray(part, float), 6).astype("<f8").tobytes())
        h.update(np.asarray(self.assignment, "<i8").tobytes())
        if hs.map_index is not None:
            h.update(np.asarray(hs.map_index, "<i8").tobytes())
        return h.hexdigest()[:16]

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
            "actual_hours": np.round(s.actual_hours, 6),
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
        # fixed line ending: the same scenario set has the same fingerprint on Windows and Linux
        h.update(t.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode())
        for k in sorted({self.map_id(i) for i in range(len(self))}):
            h.update(self.maps[k].digest().encode())
        return h.hexdigest()[:16]

    def record(self) -> dict:
        used = sorted({self.map_id(i) for i in range(len(self))})
        hours = self.table().groupby("map_id")["hours"].sum().to_dict()
        energy = self.table().groupby("map_id")["dni_energy_kwh_m2"].sum().to_dict()
        return {"fingerprint": self.fingerprint(), "n_scenarios": len(self), "total_hours": float(self.samples.weight_h.sum()),
                "actual_hours": float(np.sum(self.samples.actual_hours)),
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
        if self.hourly is not None and self.assignment is not None:
            # the member hours of every scenario (for the control-variate estimate after reloading)
            hs, a = self.hourly, np.asarray(self.assignment)
            keep = np.flatnonzero(a >= 0)
            files["hourly_csv"] = folder / "hourly_members.csv"
            pd.DataFrame({
                "time": [str(t) for t in hs.times[keep]] if hs.times is not None else "",
                "elevation_deg": hs.elevation[keep], "azimuth_deg": hs.azimuth[keep], "dni_w_m2": hs.dni[keep],
                "hours": hs.weight_h[keep],
                "map_id": [hs.maps[m].map_id for m in hs.map_index[keep]] if hs.maps is not None else "uniform",
                "scenario_id": [self.ids[j] for j in a[keep]],
            }).to_csv(files["hourly_csv"], index=False, float_format="%.17g", lineterminator="\n")
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


CLEAR_SKY_BASIS = "clear_sky_design"


def is_clear_sky(case) -> bool:
    return case.scenarios.source == "clear_sky_days"


def clear_sky_day_set(case, n_days: int | None = None) -> ScenarioSet:
    """The clear-sky design baseline: every working hour of the representative days is a scenario.

    One clear-sky DNI per hour for the whole field (``sun.clear_sky_design_days``), the uniform map, weight =
    days represented x step. Every layout is scored on exactly these dates, hours and weights. The hours of
    the whole clear-sky year are assigned to the scenario of their day segment at the same clock time (the
    nearest working time of the representative day when that one is below the threshold), for the control
    variate."""
    a, spec = case.annual, case.scenarios
    design, day_idx, days = sun.clear_sky_design_days(case, n_days)
    if not len(design):
        raise ValueError("no working hour on the representative days (sun elevation / DNI threshold)")
    step = float(a.time_step_min)
    slot = lambda t: np.asarray((t.hour * 60 + t.minute) // step, dtype=int)  # noqa: E731 - clock slot of the day
    uni = uniform_map()
    ids = [f"D{int(k) + 1:02d}-{t.strftime('%H%M')}" for k, t in zip(day_idx, design.times)]
    samples = sun.Samples(design.elevation, design.azimuth, design.dni, design.weight_h, design.times, "scenarios",
                          np.zeros(len(design), dtype=int), [uni], np.zeros(len(design), dtype=int), ids,
                          design.weight_h.copy())
    # the clear-sky year behind the days: segment of every hour -> its representative day's scenario
    year = sun.clear_sky_year(case)
    n_year = 365 + calendar.isleap(a.year)
    n = len(days)
    edges = np.floor(np.arange(n + 1) * n_year / n).astype(int)
    doy = np.asarray(year.times.day_of_year, dtype=int) - 1
    seg = np.clip(np.searchsorted(edges, doy, side="right") - 1, 0, n - 1)
    assign = np.full(len(year), -1, dtype=int)
    rep_slot = slot(design.times)
    year_slot = slot(year.times)
    for k in range(n):
        mine = np.flatnonzero(day_idx == k)
        hrs = np.flatnonzero(seg == k)
        if not mine.size or not hrs.size:
            continue
        order = np.argsort(rep_slot[mine], kind="stable")
        sl, sc_idx = rep_slot[mine][order], mine[order]
        pos = np.clip(np.searchsorted(sl, year_slot[hrs]), 0, len(sl) - 1)
        left = np.clip(pos - 1, 0, len(sl) - 1)
        near = np.where(np.abs(sl[left] - year_slot[hrs]) < np.abs(sl[pos] - year_slot[hrs]), left, pos)
        assign[hrs] = sc_idx[near]
    samples.members = np.bincount(assign[assign >= 0], minlength=len(samples))
    year.maps, year.map_index = [uni], np.zeros(len(year), dtype=int)
    _quantise(samples)
    tl = spec.linke_turbidity
    meta = {"source": "clear sky: representative days of the year",
            "basis": CLEAR_SKY_BASIS,
            "basis_zh": "晴空年度（代表日逐时）",
            "generation": (f"{n} representative days (equal date segments, middle day, weight = days in the "
                           f"segment), every {step:g}-min step with sun >= {a.min_sun_elevation_deg:g} deg and "
                           f"DNI >= {a.min_dni_w_m2:g} W/m2"),
            "clear_sky_model": spec.clear_sky_model,
            "linke_turbidity": ("site climatology (pvlib, monthly, interpolated to the day)" if tl is None
                                else tl),
            "days": [{"day": k + 1, "date": str(d.date()), "days_represented": int(w),
                      "working_steps": int(np.sum(day_idx == k))} for k, (d, w) in enumerate(days)],
            "time_step_min": step, "representative_dni": "one clear-sky DNI per hour, whole field",
            "hours_basis": "days represented x step",
            "actual_hours_total": round(float(np.sum(samples.weight_h)), 3),
            "map_model": "uniform", "regimes": ["uniform"],
            "hourly_dni_kwh_m2": round(year.dni_hours, 3), "n_hours": len(year),
            "min_sun_elevation_deg": a.min_sun_elevation_deg, "min_dni_w_m2": a.min_dni_w_m2,
            "year": a.year}
    return ScenarioSet(samples, {"uniform": uni}, meta, list(design.times), year, assign)


def design_hourly(case) -> sun.Samples:
    """The hours a layout search works with: the clear-sky year for the clear-sky baseline, else the weather
    year (with the map regime of every hour)."""
    if is_clear_sky(case):
        return sun.clear_sky_year(case)
    if case.scenarios.source == "file":
        return build_scenarios(case).samples
    return attach_regimes(case, sun.hourly_samples(case))


def low_fidelity_set(case, n_states: int, hourly: sun.Samples | None = None) -> ScenarioSet:
    """A small set of states for quick layout scores (Bayesian search): the formal set's states merged by
    the DNI-energy-weighted k-means of ``sun.sun_bins`` (clear-sky baseline), or a weather set with
    ``n_states`` scenarios."""
    if not is_clear_sky(case):
        lo = case.copy()
        lo.scenarios.n_scenarios = int(n_states)
        return build_scenarios(lo, hourly=None if case.scenarios.source == "file" else hourly)
    full = clear_sky_day_set(case)
    s = full.samples
    plain = sun.Samples(s.elevation, s.azimuth, s.dni, s.weight_h, s.times, "clear_sky_days")
    bins = sun.sun_bins(plain, int(n_states), seed=case.scenarios.seed, weighting=case.annual.representative_dni)
    uni = full.maps["uniform"]
    samples = sun.Samples(bins.elevation, bins.azimuth, bins.dni, bins.weight_h, None, "scenarios", bins.members,
                          [uni], np.zeros(len(bins), dtype=int), [f"L{k + 1:02d}" for k in range(len(bins))],
                          bins.actual_hours)
    _quantise(samples)
    meta = {**full.meta, "generation": f"{len(bins)} states merged from the clear-sky design days (low fidelity)",
            "low_fidelity": True}
    return ScenarioSet(samples, {"uniform": uni}, meta, None)


def build_scenarios(case, n_scenarios: int | None = None, hourly: sun.Samples | None = None,
                    dni_bands: int | None = None, n_days: int | None = None) -> ScenarioSet:
    """The case's scenario set: the clear-sky design days (``scenarios.source: clear_sky_days``; ``n_days``
    overrides the number of days), the weather year, or the scenario table of ``scenarios.file``."""
    spec = case.scenarios
    if is_clear_sky(case):
        return clear_sky_day_set(case, n_days)
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
    assign = np.full(len(hourly), -1, dtype=int)
    offset = 0
    shared = bool(spec.shared_directions) and (len(regimes) > 1 or bands > 1)
    if shared:
        parts, map_index, examples, assign = _shared_direction_states(case, hourly, reg, len(regimes), n_scenarios,
                                                                      bands)
        k_r = []  # the per-map clustering below is skipped
    for r, k in enumerate(k_r):
        if k <= 0:
            continue
        idx_r = np.flatnonzero(reg == r)
        sub = hourly.subset(idx_r)
        bins = sun.sun_bins(sub, int(k), seed=spec.seed, dni_bands=bands, weighting=case.annual.representative_dni)
        a = getattr(bins, "assignment", None)
        if a is not None:
            assign[idx_r] = np.where(a >= 0, a + offset, -1)
        offset += len(bins)
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
                          members, map_list, np.asarray(map_index, dtype=int), None,
                          np.concatenate([p.actual_hours for p in parts]))
    order = np.lexsort((samples.azimuth, samples.map_index))
    samples = samples.subset(order)
    examples = [examples[j] for j in order]
    new_pos = np.empty(len(order), dtype=int)
    new_pos[order] = np.arange(len(order))
    assign = np.where(assign >= 0, new_pos[np.maximum(assign, 0)], -1)
    hourly_maps = hourly.subset(np.arange(len(hourly)))
    hourly_maps.maps, hourly_maps.map_index = map_list, reg.astype(int)
    samples.ids = [f"S{k + 1:02d}" for k in range(len(samples))]
    _quantise(samples)
    meta = {"source": "weather year: " + (str(case.weather.file) if case.weather.source == "file"
                                          else f"clear sky ({case.sun.dni_model})"),
            "generation": (f"DNI-energy-weighted k-means of the sun directions over all maps, hours split by map, "
                           f"{bands} DNI band(s)" if shared else
                           f"DNI-energy-weighted k-means per map regime, {bands} DNI band(s)"),
            "representative_dni": case.annual.representative_dni,
            "hours_basis": ("actual hours of the merged records, time-average DNI"
                            if case.annual.representative_dni == "time_mean"
                            else "irradiance-equivalent hours, energy-weighted DNI"),
            "actual_hours_total": round(float(np.sum(hourly.weight_h)), 3),
            "map_model": spec.map_model, "map_assignment": spec.map_assignment, "regimes": regimes,
            "states_per_regime": {m: int(k) for m, k in zip(regimes, k_r)},
            "hourly_dni_kwh_m2": round(hourly.dni_hours, 3), "n_hours": len(hourly),
            "min_sun_elevation_deg": case.annual.min_sun_elevation_deg}
    return ScenarioSet(samples, maps, meta, examples, hourly_maps, assign)


def _shared_direction_states(case, hourly: sun.Samples, reg: np.ndarray, n_regimes: int, n_scenarios: int,
                             bands: int):
    """Scenarios that share their sun directions across maps and DNI bands.

    The directions are clustered once over the whole year (about n_scenarios / n_maps of them, so the set
    keeps about n_scenarios scenarios); every (direction, map) with hours becomes a scenario with the
    direction of the cluster, the hours of that map in it and their representative DNI (``sun._aggregate``
    within the group, so the DNI energy of every map is kept)."""
    k_dir = max(min(4, n_scenarios), int(round(n_scenarios / max(n_regimes, 1))))  # at least 4 directions
    dirs = sun.sun_bins(hourly, k_dir, seed=case.scenarios.seed, dni_bands=1,
                        weighting=case.annual.representative_dni)
    a_dir = np.asarray(dirs.assignment)
    parts, map_index, examples = [], [], []
    assign = np.full(len(hourly), -1, dtype=int)
    offset = 0
    for d in range(len(dirs)):
        for r in range(n_regimes):
            idx = np.flatnonzero((a_dir == d) & (reg == r))
            if not idx.size:
                continue
            sub = hourly.subset(idx)
            agg = sun._aggregate(sub, np.zeros(idx.size, dtype=int), np.ones(3), bands, "scenarios",
                                 case.annual.representative_dni)
            if not len(agg):
                continue
            agg.elevation = np.full(len(agg), float(dirs.elevation[d]))  # the shared direction of the cluster
            agg.azimuth = np.full(len(agg), float(dirs.azimuth[d]))
            a = np.asarray(agg.assignment)
            assign[idx] = np.where(a >= 0, a + offset, -1)
            offset += len(agg)
            parts.append(agg)
            map_index += [r] * len(agg)
            examples += [sub.times[int(np.argmax(sub.dni))] if sub.times is not None else None] * len(agg)
    return parts, map_index, examples, assign


def _quantise(samples: sun.Samples) -> None:
    """Round the scenario table to its saved precision, so a saved and reloaded set is identical."""
    for name in ("elevation", "azimuth", "dni", "weight_h", "hours"):
        if getattr(samples, name) is not None:
            setattr(samples, name, np.round(np.asarray(getattr(samples, name), dtype=float), 6))


def check_scenario_table(case, df: pd.DataFrame, maps: dict[str, DNIMap], map_file: str = "地图库") -> list[dict]:
    """Problems of a scenario table, one entry per scenario and field (nothing raised).

    Columns: scenario_id, hours (h), dni_w_m2 (W/m2), map_id, and either elevation_deg + azimuth_deg or a
    local ``time`` from which the sun direction is computed for the site. Every message names the scenario.
    """
    out = []
    has_angles = {"elevation_deg", "azimuth_deg"} <= set(df.columns)
    need = ["scenario_id", "hours", "dni_w_m2", "map_id"]
    missing = [c for c in need if c not in df.columns]
    if not has_angles and "time" not in df.columns:
        missing.append("elevation_deg + azimuth_deg 或 time")
    if missing:
        return [{"scenario": None, "field": ", ".join(missing),
                 "message": f"工况表缺少列：{', '.join(missing)}。需要 scenario_id、太阳方向（elevation_deg、azimuth_deg，°）"
                            "或当地时间 time、hours（代表小时，h）、dni_w_m2（W/m²）、map_id。"}]
    if df.empty:
        return [{"scenario": None, "field": "hours", "message": "工况表为空。"}]
    ids = df["scenario_id"].astype(str)
    if df["scenario_id"].isna().any() or ids.str.strip().eq("").any():
        out.append({"scenario": None, "field": "scenario_id", "message": "工况编号不能为空。"})
    for sid in ids[ids.duplicated()].unique():
        out.append({"scenario": sid, "field": "scenario_id", "message": f"工况名 {sid} 重复出现，请让每个工况名唯一。"})
    numeric = [("hours", "h", True), ("dni_w_m2", "W/m²", True)]
    if has_angles:
        numeric += [("elevation_deg", "°", False), ("azimuth_deg", "°", False)]
    for col, unit, nonneg in numeric:
        num = pd.to_numeric(df[col], errors="coerce")
        bad = ~np.isfinite(num) | (num < 0 if nonneg else False)
        for i in np.flatnonzero(bad.to_numpy()):
            out.append({"scenario": ids.iloc[i], "field": col,
                        "message": f"工况 {ids.iloc[i]} 的 {col} 是 {df[col].iloc[i]}。请输入{'非负' if nonneg else ''}数，单位 {unit}。"})
    if not has_angles:
        t = pd.to_datetime(df["time"], errors="coerce")
        for i in np.flatnonzero(t.isna().to_numpy()):
            out.append({"scenario": ids.iloc[i], "field": "time",
                        "message": f"工况 {ids.iloc[i]} 的时间 {df['time'].iloc[i]} 无法识别，请写成 2024-06-21 13:00（当地时间）。"})
    known = set(maps) | {"uniform"}
    for i, m in enumerate(df["map_id"].astype(str)):
        if m not in known:
            out.append({"scenario": ids.iloc[i], "field": "map_id",
                        "message": f"工况 {ids.iloc[i]} 引用的地图 {m} 在{map_file}中找不到。请重新选择地图文件，或改正工况表里的地图名。"})
    return out


def load_table(case, path, maps: dict[str, DNIMap] | None = None) -> ScenarioSet:
    """Scenario table CSV: scenario_id, elevation_deg + azimuth_deg (or a local ``time``), hours, dni_w_m2,
    map_id [, actual_hours, hourly_members]."""
    from .tables import read_user_csv

    maps = dict(maps if maps is not None else model_maps(case))
    df = read_user_csv(path)
    problems = check_scenario_table(case, df, maps, case.scenarios.map_file or "地图库")
    if problems:
        raise ValueError(f"{Path(path).name}: " + " ".join(p["message"] for p in problems[:5]))
    if "uniform" in set(df["map_id"].astype(str)) and "uniform" not in maps:
        maps["uniform"] = uniform_map()
    if not {"elevation_deg", "azimuth_deg"} <= set(df.columns):  # sun direction from the local time at the site
        el, az = sun.solar_position(sun.to_times(df["time"].astype(str), case.site.timezone), case.site)
        df["elevation_deg"], df["azimuth_deg"] = el, az
    order = list(dict.fromkeys(df["map_id"].astype(str)))
    map_list = [maps[m] for m in order]
    idx = np.array([order.index(m) for m in df["map_id"].astype(str)], dtype=int)
    samples = sun.Samples(df["elevation_deg"].to_numpy(float), df["azimuth_deg"].to_numpy(float),
                          df["dni_w_m2"].to_numpy(float), df["hours"].to_numpy(float), None, "scenario_table",
                          df["hourly_members"].to_numpy(int) if "hourly_members" in df else None, map_list, idx,
                          [str(v) for v in df["scenario_id"]],
                          df["actual_hours"].to_numpy(float) if "actual_hours" in df else None)
    _quantise(samples)
    examples = list(df["example_time"].fillna("")) if "example_time" in df else (
        list(df["time"].fillna("")) if "time" in df else None)
    return ScenarioSet(samples, maps, {"source": f"scenario table {Path(path).name}"}, examples)


def validation_scenarios(case, train, n_scenarios, hourly, seed_offset=0):
    """Identify genuinely new operating states, even if a table relabels or reweights them."""
    other = case.copy()
    other.scenarios.seed += seed_offset
    if other.scenarios.validation_file:
        validation = load_table(other, other.resolve(other.scenarios.validation_file))
        source = "separate_table"
    elif other.scenarios.source == "file":
        validation, source = train, "same_table"
    elif is_clear_sky(other):  # twice as many days: their middle days fall between the design days
        validation = clear_sky_day_set(other, 2 * int(other.scenarios.n_days))
        source = "clear_sky_denser_days"
    else:
        validation = build_scenarios(other, n_scenarios, hourly)
        source = "same_weather_year"

    def state_keys(sset):
        keys = []
        s = sset.samples
        for i in range(len(sset)):
            m = s.map_of(i)
            dni = m.values * (s.dni[i] if m.unit == "factor" else 1.0)
            h = hashlib.sha256()
            for part in (m.x, m.y, dni):
                h.update(np.round(part, 6).astype("<f8").tobytes())
            keys.append((round(s.elevation[i], 6), round(s.azimuth[i], 6), h.digest()))
        return keys

    known = set(state_keys(train))
    overlap = sum(k in known for k in state_keys(validation))
    info = {"n": len(validation), "fingerprint": validation.fingerprint(), "seed": other.scenarios.seed,
            "source": source, "shared_states": overlap, "new_states": len(validation) - overlap}
    return validation, info


def load_folder(folder) -> ScenarioSet:
    """A scenario set saved by ``ScenarioSet.save`` (scenarios.csv + dni_maps.*)."""
    folder = Path(folder)
    maps = load_library(folder)
    df = pd.read_csv(folder / "scenarios.csv", float_precision="round_trip")  # exact floats
    order = list(dict.fromkeys(df["map_id"].astype(str)))
    idx = np.array([order.index(m) for m in df["map_id"].astype(str)], dtype=int)
    samples = sun.Samples(df["elevation_deg"].to_numpy(float), df["azimuth_deg"].to_numpy(float),
                          df["dni_w_m2"].to_numpy(float), df["hours"].to_numpy(float), None, "scenarios",
                          df["hourly_members"].to_numpy(int) if "hourly_members" in df else None,
                          [maps[m] for m in order], idx, [str(v) for v in df["scenario_id"]],
                          df["actual_hours"].to_numpy(float) if "actual_hours" in df else None)
    meta_path = folder / "scenario_set.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    examples = list(df["example_time"].fillna("")) if "example_time" in df else None
    out = ScenarioSet(samples, maps, {k: v for k, v in meta.items() if k in ("source", "generation", "map_model",
                                                                          "map_assignment", "regimes")}, examples)
    if meta.get("fingerprint") and meta["fingerprint"] != out.fingerprint():
        raise ValueError(f"{folder}: scenario set fingerprint mismatch (files edited after saving?)")
    hp = folder / "hourly_members.csv"
    if hp.exists():
        h = pd.read_csv(hp, float_precision="round_trip")
        known = [m for m in dict.fromkeys(h["map_id"].astype(str)) if m in maps]
        h = h[h["map_id"].astype(str).isin(known)]
        hourly = sun.Samples(h["elevation_deg"].to_numpy(float), h["azimuth_deg"].to_numpy(float),
                             h["dni_w_m2"].to_numpy(float), h["hours"].to_numpy(float), None, "hourly",
                             None, [maps[m] for m in known],
                             np.array([known.index(m) for m in h["map_id"].astype(str)], dtype=int))
        out.hourly = hourly
        out.assignment = np.array([out.index(s) for s in h["scenario_id"].astype(str)], dtype=int)
    return out


def is_spatial(case) -> bool:
    """True when the case's DNI is not uniform over the site (any map model but a pure uniform one; never for
    the clear-sky design baseline)."""
    spec = case.scenarios
    if spec.source == "clear_sky_days":
        return False
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
