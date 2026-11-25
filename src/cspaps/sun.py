"""Sun position, DNI, weather files and annual sampling (incl. representative sun positions)."""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

from .config import Case, Site, Sun


def to_times(times, timezone: str) -> pd.DatetimeIndex:
    """Parse times; naive values are interpreted in the site's local timezone.

    Daylight-saving zones: a local time that does not exist (clocks jump forward) is shifted forward; a
    repeated local time (clocks fall back) is resolved from the order of the series when it contains both
    occurrences, otherwise read as standard time.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times)))
    if idx.tz is None:
        try:
            idx = idx.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
        except ValueError:  # a repeated hour appears once (or the order is not monotonic): standard time
            idx = idx.tz_localize(timezone, ambiguous=np.zeros(len(idx), dtype=bool), nonexistent="shift_forward")
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


def weather_coverage(case: Case) -> dict | None:
    """The time span of the case's weather file (None without one): first and last record, the typical step,
    the tolerance within which an instant may use its nearest record, and a suggested instant inside the file
    (the record with the highest DNI) for time pickers."""
    if case.weather.source != "file" or not case.weather.file:
        return None
    s = load_weather(case)
    if s.empty:
        return None
    step = pd.Series(s.index).diff().median() if len(s) > 1 else pd.Timedelta(hours=1)
    step = step if pd.notna(step) and step > pd.Timedelta(0) else pd.Timedelta(hours=1)
    return {"first": s.index[0], "last": s.index[-1], "step": step, "n": int(len(s)),
            "tolerance": max(pd.Timedelta(hours=1), 1.5 * step), "suggested": s.idxmax()}


def weather_dni_at(case: Case, idx) -> float:
    """DNI of the weather file at the first instant of ``idx`` (nearest record). An instant further than the
    coverage tolerance from every record raises: the file holds no information about it (a June 2025 instant
    must not silently take a 2026 record)."""
    series = load_weather(case)
    j = int(series.index.get_indexer(idx, method="nearest")[0])
    cov = weather_coverage(case)
    gap = abs(series.index[j] - idx[0])
    if cov is not None and gap > cov["tolerance"]:
        raise ValueError(
            f"{idx[0]:%Y-%m-%d %H:%M} is outside the weather file {case.weather.file} "
            f"({cov['first']:%Y-%m-%d %H:%M} – {cov['last']:%Y-%m-%d %H:%M}); the nearest record is "
            f"{gap.total_seconds() / 3600:.1f} h away. Choose a time inside the file, or use clear-sky DNI.")
    return float(series.iloc[j])


def _read_weather(case: Case, path) -> pd.Series:
    if path.suffix.lower() == ".epw":
        data, _ = pvlib.iotools.read_epw(path, coerce_year=case.annual.year)
        dni_s = data["dni"].astype(float)
        dni_s.index = dni_s.index - pd.Timedelta(minutes=30)
        dni_s = dni_s.tz_convert(case.site.timezone)
    else:
        from .tables import read_user_csv

        df = read_user_csv(path)
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
    maps: list | None = None  # spatial DNI maps (dni_maps.DNIMap); None = the DNI is uniform over the site
    map_index: np.ndarray | None = None  # (S,) map of each state (index into ``maps``)
    ids: list | None = None  # scenario IDs (scenario sets)
    hours: np.ndarray | None = None  # (S,) actual hours represented (sum of member durations); None = weight_h

    def __len__(self) -> int:
        return int(self.elevation.size)

    @property
    def actual_hours(self) -> np.ndarray:
        """(S,) real time each state stands for (h). Equals ``weight_h`` for time-mean representative DNI."""
        return self.weight_h if self.hours is None else self.hours

    @property
    def spatial(self) -> bool:
        return self.maps is not None and self.map_index is not None

    def dni_at(self, i: int, xy) -> float | np.ndarray:
        """DNI of state i at the points xy (N, 2): the scalar DNI, or the state's map sampled at the points."""
        if not self.spatial:
            return float(self.dni[i])
        m = self.maps[int(self.map_index[i])]
        v = m.sample(np.asarray(xy, dtype=float).reshape(-1, 2))
        return v * float(self.dni[i]) if m.unit == "factor" else v

    def map_of(self, i: int):
        return self.maps[int(self.map_index[i])] if self.spatial else None

    @property
    def max_factor(self) -> float:
        """Largest DNI / scenario DNI ratio over the maps (1 without maps); bounds any heliostat's DNI."""
        if not self.spatial:
            return 1.0
        out = 1.0
        for i in range(len(self)):
            m = self.map_of(i)
            ref = 1.0 if m.unit == "factor" else max(float(self.dni[i]), 1e-12)
            out = max(out, float(m.values.max()) / ref)
        return out

    def subset(self, idx) -> "Samples":
        idx = np.asarray(idx)
        return Samples(self.elevation[idx], self.azimuth[idx], self.dni[idx], self.weight_h[idx],
                       self.times[idx] if self.times is not None else None, self.label,
                       self.members[idx] if self.members is not None else None, self.maps,
                       self.map_index[idx] if self.map_index is not None else None,
                       [self.ids[k] for k in np.atleast_1d(np.arange(len(self))[idx])] if self.ids is not None else None,
                       self.hours[idx] if self.hours is not None else None)

    @property
    def dni_hours(self) -> float:
        """Annual DNI energy per m2 normal to the sun, kWh/m2."""
        return float(np.sum(self.dni * self.weight_h) / 1e3)


def _filter(case: Case, s: Samples) -> Samples:
    keep = _working(case, s)
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


CLEAR_SKY_MODELS = ("ineichen", "meinel")


def clear_sky_dni(times: pd.DatetimeIndex, elevation_deg, site: Site, model: str = "ineichen",
                  linke_turbidity=None) -> np.ndarray:
    """Clear-sky DNI (W/m2) for the site at these times (one value per time, the same for every mirror).

    "ineichen": Ineichen-Perez (pvlib) with the extraterrestrial DNI of the date, the site altitude
    (pressure-corrected air mass) and the Linke turbidity of the site: ``linke_turbidity`` None = pvlib's
    monthly climatology of the site interpolated to the day (the local typical clear-sky atmosphere), a
    number = that value all year, 12 numbers = one per month. "meinel": Meinel with Laue's altitude
    correction (elevation only)."""
    el = np.asarray(elevation_deg, dtype=float)
    if model == "meinel":
        return dni_meinel(el, site.elevation_m)
    if model != "ineichen":
        raise ValueError(f"unknown clear-sky model '{model}' (use {', '.join(CLEAR_SKY_MODELS)})")
    from pvlib import atmosphere, clearsky, irradiance

    zen = 90.0 - el
    with np.errstate(invalid="ignore", divide="ignore"):
        am_rel = atmosphere.get_relative_airmass(np.where(el > 0, zen, 89.9), model="kastenyoung1989")
    am_abs = atmosphere.get_absolute_airmass(am_rel, atmosphere.alt2pres(site.elevation_m))
    if linke_turbidity is None:
        tl = clearsky.lookup_linke_turbidity(times, site.latitude, site.longitude).to_numpy(float)
    elif np.ndim(linke_turbidity) == 0:
        tl = np.full(len(times), float(linke_turbidity))
    else:
        tl = np.asarray(linke_turbidity, dtype=float)[np.asarray(times.month) - 1]
    cs = clearsky.ineichen(np.where(el > 0, zen, 89.9), np.asarray(am_abs, float), tl,
                           altitude=site.elevation_m, dni_extra=np.asarray(irradiance.get_extra_radiation(times), float))
    return np.where(el > 0, np.nan_to_num(np.asarray(cs["dni"], float)), 0.0)


def design_days(year: int, n_days: int) -> list[tuple[pd.Timestamp, int]]:
    """``n_days`` representative days of the year: the year cut into n equal date segments, the middle day
    of each with the number of days it stands for (they add up to the days of the year). Equal segments
    follow the solar declination and the day length through every season."""
    n_year = 365 + calendar.isleap(year)
    n = max(1, min(int(n_days), n_year))
    edges = np.floor(np.arange(n + 1) * n_year / n).astype(int)
    start = pd.Timestamp(year=year, month=1, day=1)
    return [(start + pd.Timedelta(days=int((a + b - 1) // 2)), int(b - a)) for a, b in zip(edges[:-1], edges[1:])]


def _day_times(day: pd.Timestamp, step_min: float, timezone: str) -> pd.DatetimeIndex:
    step = pd.Timedelta(minutes=step_min)
    n = int(round(24 * 60 / step_min))
    return to_times(day + (np.arange(n) + 0.5) * step, timezone)


def _clear(case: Case, times: pd.DatetimeIndex, weights, label: str) -> Samples:
    sc = case.scenarios
    el, az = solar_position(times, case.site)
    d = clear_sky_dni(times, el, case.site, sc.clear_sky_model, sc.linke_turbidity)
    return Samples(el, az, d, np.asarray(weights, dtype=float), times, label)


def clear_sky_design_days(case: Case, n_days: int | None = None) -> tuple[Samples, np.ndarray, list]:
    """Every working hour of the representative clear-sky days (``design_days``).

    Each state is one time step of a representative day (interval midpoint, site-local clock), with the
    clear-sky DNI of that date and time shared by the whole field; it is kept when the sun is at least
    ``annual.min_sun_elevation_deg`` high and the DNI reaches ``annual.min_dni_w_m2`` (the same working
    threshold for every hour, low winter mornings included). Weight = days represented x step (h).
    Returns (samples, day index of each state, [(date, days represented)])."""
    a = case.annual
    days = design_days(a.year, int(n_days or case.scenarios.n_days))
    times, weights, day_idx = [], [], []
    for k, (day, n_rep) in enumerate(days):
        t = _day_times(day, a.time_step_min, case.site.timezone)
        times.append(t)
        weights.append(np.full(len(t), n_rep * a.time_step_min / 60.0))
        day_idx.append(np.full(len(t), k))
    idx = times[0].append(list(times[1:])) if len(times) > 1 else times[0]
    s = _clear(case, idx, np.concatenate(weights), "clear_sky_days")
    keep = _working(case, s)
    out = Samples(s.elevation[keep], s.azimuth[keep], s.dni[keep], s.weight_h[keep], s.times[keep], s.label)
    return out, np.concatenate(day_idx)[keep], days


def clear_sky_year(case: Case) -> Samples:
    """Every working time step of the year on the clear-sky model of the design baseline (search, control
    variate), weight = the step (h)."""
    a = case.annual
    step = pd.Timedelta(minutes=a.time_step_min)
    start = pd.Timestamp(year=a.year, month=1, day=1)
    n = int(round(pd.Timedelta(days=365 + calendar.isleap(a.year)) / step))
    times = to_times(start + (np.arange(n) + 0.5) * step, case.site.timezone)
    s = _clear(case, times, np.full(n, a.time_step_min / 60.0), "clear_sky_hourly")
    keep = _working(case, s)
    return Samples(s.elevation[keep], s.azimuth[keep], s.dni[keep], s.weight_h[keep], s.times[keep], s.label)


def _working(case: Case, s: Samples) -> np.ndarray:
    a = case.annual
    return (s.elevation >= a.min_sun_elevation_deg) & (s.dni > 0) & (s.dni >= a.min_dni_w_m2)


REPRESENTATIVE_DNI = ("time_mean", "energy_weighted")


def sun_bins(samples: Samples, n_bins: int, seed: int = 0, iters: int = 100, z_weight: float = 3.0,
             dni_bands: int = 1, weighting: str = "time_mean") -> Samples:
    """Cluster sun positions into ``n_bins`` representative positions (weighted k-means).

    Weights are the DNI energy of each sample. Clustering runs on the sun
    vector with its vertical component scaled by ``z_weight`` so that low sun
    elevations, where shading changes fast, are resolved more finely. Each
    bin gets the energy-weighted mean direction, the energy-weighted mean DNI,
    and a weight in hours such that dni_bin * weight_bin equals the DNI energy
    of its members. For any quantity linear in DNI the annual sum is therefore
    preserved; the only approximation is evaluating each bin at one direction.

    ``weighting`` chooses the representative DNI and the weight of a bin:
      time_mean        d = sum G w / sum w (time-average DNI), weight = sum w (the actual hours)
      energy_weighted  g = sum G^2 w / sum G w, weight = sum G w / g ("irradiance-equivalent hours")
    Both keep d x weight = the DNI energy of the members. The actual hours are kept in ``Samples.hours``.
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
    return _aggregate(samples, label, scale, dni_bands, "sun_bins", weighting)


def _aggregate(samples: Samples, label: np.ndarray, scale: np.ndarray, dni_bands: int, name: str,
               weighting: str = "time_mean") -> Samples:
    """One representative state per label (optionally split into DNI bands); see ``sun_bins`` for ``weighting``."""
    if weighting not in REPRESENTATIVE_DNI:
        raise ValueError(f"representative DNI weighting must be one of {REPRESENTATIVE_DNI}")
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
    hours = np.bincount(label, weights=samples.weight_h, minlength=n_lab)[keep]
    if weighting == "time_mean":
        dni_rep = energy[keep] / hours
    else:
        dni_rep = np.bincount(label, weights=w * samples.dni, minlength=n_lab)[keep] / energy[keep]
    el, az = vector_to_angles(cen / scale)
    out = Samples(el, az, dni_rep, energy[keep] / dni_rep, None, name)
    out.members = np.bincount(label, minlength=n_lab)[keep]
    out.hours = hours
    # representative state of every input sample (-1: a label without DNI energy), for estimators that
    # correct each state with its members (control variates)
    new_index = np.full(n_lab, -1, dtype=int)
    new_index[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    out.assignment = new_index[label]
    return out


def sun_grid(samples: Samples, el_step_deg: float = 10.0, az_step_deg: float = 20.0, dni_bands: int = 1,
             weighting: str = "time_mean") -> Samples:
    """Bin sun positions on a fixed elevation x azimuth grid (energy-weighted centroid per bin)."""
    lab_el = np.floor(samples.elevation / el_step_deg).astype(int)
    lab_az = np.floor(samples.azimuth / az_step_deg).astype(int)
    _, label = np.unique(lab_el * 1000 + lab_az, return_inverse=True)
    return _aggregate(samples, label, np.ones(3), dni_bands, "sun_grid", weighting)


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
        return sun_bins(hourly_samples(case), n_bins or a.n_sun_bins, dni_bands=bands, weighting=a.representative_dni)
    if method == "sun_grid":
        return sun_grid(hourly_samples(case), a.grid_elevation_step_deg, a.grid_azimuth_step_deg, bands,
                        a.representative_dni)
    raise ValueError(f"Unknown annual method '{method}'")


def sampling_record(case: Case, samples: Samples, aiming: str) -> dict:
    """What an annual number was computed with (for traceability)."""
    return {
        "method": samples.label, "n_states": len(samples), "aiming": aiming,
        "weather": case.weather.file if case.weather.source == "file" else f"clear_sky ({case.sun.dni_model})",
        "annual_dni_kwh_m2": round(samples.dni_hours, 2),
        "actual_hours": round(float(np.sum(samples.actual_hours)), 2),
        "weight_hours": round(float(np.sum(samples.weight_h)), 2),
        "representative_dni": case.annual.representative_dni,
        "min_sun_elevation_deg": case.annual.min_sun_elevation_deg,
        "weights": ("actual hours per state, time-average DNI" if case.annual.representative_dni == "time_mean"
                    else "irradiance-equivalent hours per state (DNI x hours = DNI energy of the merged hours)"),
    }
