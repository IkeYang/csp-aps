"""Sun position, DNI, weather files and annual sampling (incl. representative sun positions)."""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

from .config import Case, Site, Sun


def to_times(times, timezone: str) -> pd.DatetimeIndex:
    """Parse times; naive values are interpreted in the site's local timezone."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times)))
    if idx.tz is None:
        idx = idx.tz_localize(timezone)
    return idx


def solar_position(times: pd.DatetimeIndex, site: Site) -> tuple[np.ndarray, np.ndarray]:
    """Return (apparent elevation, compass azimuth) in degrees (NREL SPA via pvlib)."""
    sp = pvlib.solarposition.get_solarposition(
        times, site.latitude, site.longitude, altitude=site.elevation_m
    )
    return sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy()


def sun_vector(elevation_deg, azimuth_deg) -> np.ndarray:
    """Unit vector(s) pointing from the ground towards the sun, ENU frame."""
    el = np.radians(np.asarray(elevation_deg, dtype=float))
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.stack(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1
    )


def vector_to_angles(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    el = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    az = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0
    return el, az


def air_mass(elevation_deg) -> np.ndarray:
    """Relative optical air mass, Kasten & Young (1989)."""
    h = np.asarray(elevation_deg, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)
    return np.where(h > 0, am, np.inf)


def dni_meinel(elevation_deg, site_elevation_m: float = 0.0) -> np.ndarray:
    """Clear-sky DNI (W/m2): Meinel model with Laue's altitude correction."""
    h_km = site_elevation_m / 1000.0
    am = air_mass(elevation_deg)
    with np.errstate(over="ignore", invalid="ignore"):
        dni = 1353.0 * ((1.0 - 0.14 * h_km) * 0.7 ** (am**0.678) + 0.14 * h_km)
    return np.where(np.asarray(elevation_deg) > 0, dni, 0.0)


def dni(elevation_deg, sun: Sun, site: Site) -> np.ndarray:
    """Clear-sky DNI from the configured model."""
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    if sun.dni_model == "meinel":
        return dni_meinel(elevation_deg, site.elevation_m)
    if sun.dni_model == "constant":
        return np.where(elevation_deg > 0, sun.dni_constant_w_m2, 0.0)
    raise ValueError(f"Unknown dni_model '{sun.dni_model}' (use 'meinel' or 'constant')")


# --------------------------------------------------------------------------- weather

_WEATHER_CACHE: dict[tuple, pd.Series] = {}


def load_weather(case: Case) -> pd.Series:
    """DNI series (W/m2) from an EPW or CSV file, indexed by tz-aware interval midpoints.

    EPW: hour-ending stamps are shifted back 30 min; the year is set to
    ``annual.year``. CSV: columns ``time,dni``; naive times are site-local and
    taken as given (instants or interval midpoints). Cached per file version.
    """
    path = case.resolve(case.weather.file)
    key = (str(path), path.stat().st_mtime, case.annual.year, case.site.timezone)
    if key not in _WEATHER_CACHE:
        _WEATHER_CACHE[key] = _read_weather(case, path)
    return _WEATHER_CACHE[key]


def _read_weather(case: Case, path) -> pd.Series:
    if path.suffix.lower() == ".epw":
        data, _ = pvlib.iotools.read_epw(path, coerce_year=case.annual.year)
        dni_s = data["dni"].astype(float)
        dni_s.index = dni_s.index - pd.Timedelta(minutes=30)
        dni_s = dni_s.tz_convert(case.site.timezone)
    else:
        df = pd.read_csv(path)
        if not {"time", "dni"} <= set(df.columns):
            raise ValueError(f"{path}: weather CSV needs columns time,dni")
        idx = to_times(df["time"], case.site.timezone)
        dni_s = pd.Series(df["dni"].to_numpy(dtype=float), index=idx)
    return dni_s.clip(lower=0.0).sort_index()


# --------------------------------------------------------------------------- annual sampling

@dataclass
class Samples:
    """Sun states to evaluate; annual energy = sum(power(dni) * weight_h)."""

    elevation: np.ndarray
    azimuth: np.ndarray
    dni: np.ndarray
    weight_h: np.ndarray
    times: pd.DatetimeIndex | None = None
    label: str = ""
    members: np.ndarray | None = None  # hourly samples merged into each state (binned methods)

    def __len__(self) -> int:
        return int(self.elevation.size)

    @property
    def dni_hours(self) -> float:
        """Annual DNI energy per m2 normal to the sun, kWh/m2."""
        return float(np.sum(self.dni * self.weight_h) / 1e3)


def _filter(case: Case, s: Samples) -> Samples:
    keep = (s.elevation >= case.annual.min_sun_elevation_deg) & (s.dni > 0)
    return Samples(s.elevation[keep], s.azimuth[keep], s.dni[keep], s.weight_h[keep],
                   s.times[keep] if s.times is not None else None, s.label)


def hourly_samples(case: Case) -> Samples:
    """Every time step of the year (weather file, or the clear-sky model)."""
    if case.weather.source == "file":
        series = load_weather(case)
        times = series.index
        step_h = float(pd.Series(times).diff().median() / pd.Timedelta(hours=1))
        el, az = solar_position(times, case.site)
        s = Samples(el, az, series.to_numpy(), np.full(len(times), step_h), times, "hourly")
    else:
        spec = case.annual
        step = pd.Timedelta(minutes=spec.time_step_min)
        start = pd.Timestamp(year=spec.year, month=1, day=1)
        n = int(round(pd.Timedelta(days=365 + calendar.isleap(spec.year)) / step))
        times = to_times(start + (np.arange(n) + 0.5) * step, case.site.timezone)
        el, az = solar_position(times, case.site)
        s = Samples(el, az, dni(el, case.sun, case.site), np.full(n, spec.time_step_min / 60.0),
                    times, "hourly")
    return _filter(case, s)


def representative_day_samples(case: Case) -> Samples:
    """One day per month (``day_of_month``), weighted by the days in that month."""
    spec = case.annual
    step = pd.Timedelta(minutes=spec.time_step_min)
    n_steps = int(round(24 * 60 / spec.time_step_min))
    times, weights = [], []
    for month in range(1, 13):
        day = pd.Timestamp(year=spec.year, month=month, day=spec.day_of_month)
        days_in_month = calendar.monthrange(spec.year, month)[1]
        for k in range(n_steps):
            times.append(day + (k + 0.5) * step)
            weights.append(days_in_month * spec.time_step_min / 60.0)
    idx = to_times(times, case.site.timezone)
    el, az = solar_position(idx, case.site)
    if case.weather.source == "file":
        series = load_weather(case)
        pos = series.index.get_indexer(idx, method="nearest")
        dni_v = series.to_numpy()[pos]
    else:
        dni_v = dni(el, case.sun, case.site)
    return _filter(case, Samples(el, az, dni_v, np.asarray(weights), idx, "representative_days"))


def sun_bins(samples: Samples, n_bins: int, seed: int = 0, iters: int = 100, z_weight: float = 3.0,
             dni_bands: int = 1) -> Samples:
    """Cluster sun positions into ``n_bins`` representative positions (weighted k-means).

    Weights are the DNI energy of each sample. Clustering runs on the sun
    vector with its vertical component scaled by ``z_weight`` so that low sun
    elevations, where shading changes fast, are resolved more finely. Each
    bin gets the energy-weighted mean direction, the energy-weighted mean DNI,
    and a weight in hours such that dni_bin * weight_bin equals the DNI energy
    of its members. For any quantity linear in DNI the annual sum is therefore
    preserved; the only approximation is evaluating each bin at one direction.
    With the default 30 bins the annual energy is within ~0.5 % of hourly.
    ``dni_bands`` > 1 splits every bin by DNI quantiles: needed when the power
    is not linear in DNI (flux-limited APS), where merging a 300 W/m2 and a
    1200 W/m2 state overestimates the energy.
    """
    scale = np.array([1.0, 1.0, z_weight])
    v = sun_vector(samples.elevation, samples.azimuth) * scale
    w = samples.dni * samples.weight_h
    n_bins = int(min(n_bins, len(samples)))
    if n_bins <= 0:
        return Samples(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), None, "sun_bins")
    rng = np.random.default_rng(seed)
    # weighted k-means++ initialisation
    centers = [v[rng.choice(len(v), p=w / w.sum())]]
    for _ in range(1, n_bins):
        d2 = np.min(((v[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        p = w * d2
        if p.sum() <= 0:
            break
        centers.append(v[rng.choice(len(v), p=p / p.sum())])
    c = np.asarray(centers)
    for _ in range(iters):
        label = np.argmin(((v[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.zeros_like(c)
        np.add.at(new, label, w[:, None] * v)
        mass = np.bincount(label, weights=w, minlength=len(c))
        empty = mass <= 0
        new[~empty] /= mass[~empty, None]
        new[empty] = c[empty]
        if np.allclose(new, c, atol=1e-10):
            c = new
            break
        c = new
    label = np.argmin(((v[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    return _aggregate(samples, label, scale, dni_bands, "sun_bins")


def _aggregate(samples: Samples, label: np.ndarray, scale: np.ndarray, dni_bands: int, name: str) -> Samples:
    """One representative state per label (optionally split into DNI bands)."""
    v = sun_vector(samples.elevation, samples.azimuth) * scale
    w = samples.dni * samples.weight_h
    if dni_bands > 1:
        # split every bin by DNI quantiles so that flux-limited states are not averaged with weak ones
        new = np.zeros_like(label)
        nxt = 0
        for lab in np.unique(label):
            idx = np.flatnonzero(label == lab)
            edges = np.quantile(samples.dni[idx], np.linspace(0, 1, dni_bands + 1)[1:-1]) if idx.size > 1 else []
            band = np.searchsorted(edges, samples.dni[idx], side="right") if len(edges) else np.zeros(idx.size, int)
            _, band = np.unique(band, return_inverse=True)
            new[idx] = nxt + band
            nxt += int(band.max()) + 1
        label = new
    n_lab = int(label.max()) + 1 if label.size else 0
    energy = np.bincount(label, weights=w, minlength=n_lab)  # Wh/m2
    keep = energy > 0
    cen = np.zeros((n_lab, 3))
    np.add.at(cen, label, w[:, None] * v)
    cen = cen[keep] / energy[keep, None]
    dni_rep = np.bincount(label, weights=w * samples.dni, minlength=n_lab)[keep] / energy[keep]
    el, az = vector_to_angles(cen / scale)
    out = Samples(el, az, dni_rep, energy[keep] / dni_rep, None, name)
    out.members = np.bincount(label, minlength=n_lab)[keep]
    return out


def sun_grid(samples: Samples, el_step_deg: float = 10.0, az_step_deg: float = 20.0, dni_bands: int = 1) -> Samples:
    """Bin sun positions on a fixed elevation x azimuth grid (energy-weighted centroid per bin)."""
    lab_el = np.floor(samples.elevation / el_step_deg).astype(int)
    lab_az = np.floor(samples.azimuth / az_step_deg).astype(int)
    _, label = np.unique(lab_el * 1000 + lab_az, return_inverse=True)
    return _aggregate(samples, label, np.ones(3), dni_bands, "sun_grid")


def annual_samples(case: Case, method: str | None = None, n_bins: int | None = None,
                   dni_bands: int | None = None) -> Samples:
    a = case.annual
    method = method or a.method
    bands = a.dni_bands if dni_bands is None else dni_bands
    if method == "hourly":
        return hourly_samples(case)
    if method == "representative_days":
        return representative_day_samples(case)
    if method == "sun_bins":
        return sun_bins(hourly_samples(case), n_bins or a.n_sun_bins, dni_bands=bands)
    if method == "sun_grid":
        return sun_grid(hourly_samples(case), a.grid_elevation_step_deg, a.grid_azimuth_step_deg, bands)
    raise ValueError(f"Unknown annual method '{method}'")


def sampling_record(case: Case, samples: Samples, aiming: str) -> dict:
    """What an annual number was computed with (for traceability)."""
    return {
        "method": samples.label, "n_states": len(samples), "aiming": aiming,
        "weather": case.weather.file if case.weather.source == "file" else f"clear_sky ({case.sun.dni_model})",
        "annual_dni_kwh_m2": round(samples.dni_hours, 2),
        "min_sun_elevation_deg": case.annual.min_sun_elevation_deg,
        "weights": "hours per state (DNI x hours = DNI energy of the merged hours)",
    }
