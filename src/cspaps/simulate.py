"""Field evaluation at one sun state, snapshots in time, and annual energy.

Per-heliostat power chain (W):
    P_sun  = DNI_i * A_mirror                   (DNI at the heliostat, may vary with clouds)
    P_cos  = P_sun  * eta_cos
    P_ref  = P_cos  * reflectivity * cleanliness
    P_shd  = P_ref  * eta_shading                (sun rays blocked by neighbours / tower)
    P_blk  = P_shd  * eta_blocking               (reflected rays blocked by neighbours)
    P_att  = P_blk  * eta_attenuation            (power reaching the image plane)
    P_rec  = P_att  * eta_intercept              (power hitting the receiver; 0 if defocused)
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
import pandas as pd

from . import optics, shading, sun
from .config import Case
from .field import Field
from .receiver import ReceiverGrid, build_grid, default_aim, surface_point

STAGES = ("sun", "cosine", "reflectivity", "shading", "blocking", "attenuation", "intercept")
LOSSES = STAGES[1:]


@dataclass
class Snapshot:
    time: pd.Timestamp | None
    sun_elevation: float
    sun_azimuth: float
    dni: np.ndarray  # (N,) W/m2 at each heliostat
    aim_azimuth: np.ndarray
    aim_z: np.ndarray
    active: np.ndarray  # (N,) bool, False = defocused / standby
    eta: dict[str, np.ndarray]  # per-heliostat efficiencies, keys = LOSSES
    power: dict[str, np.ndarray]  # per-heliostat power after each stage, W
    flux: np.ndarray  # (n_height, n_azimuth) W/m2
    grid: ReceiverGrid = dc_field(repr=False)
    limit_kw_m2: float = 1000.0

    @property
    def n(self) -> int:
        return int(self.aim_z.size)

    @property
    def eta_total(self) -> np.ndarray:
        return np.prod([self.eta[k] for k in LOSSES], axis=0)

    @property
    def p_receiver(self) -> float:
        return float(self.power["intercept"].sum())

    def flux_stats(self) -> dict:
        kw = self.flux / 1e3
        lit = kw[kw > 0.05 * kw.max()] if kw.max() > 0 else kw.ravel()
        ring = kw.sum(axis=0)  # height-integrated, per azimuth column
        return {
            "peak_flux_kw_m2": round(float(kw.max()), 2),
            "mean_flux_kw_m2": round(float(kw.mean()), 2),
            "peak_to_limit": round(float(kw.max()) / self.limit_kw_m2, 4),
            "peak_to_mean_lit": round(float(kw.max() / lit.mean()), 3) if lit.size and lit.mean() > 0 else 0.0,
            "circumferential_cv": round(float(ring.std() / ring.mean()), 4) if ring.mean() > 0 else 0.0,
            "flux_map_power_mw": round(float(self.flux.sum()) * self.grid.cell_area / 1e6, 4),
        }

    def summary(self) -> dict:
        totals = {k: float(v.sum()) for k, v in self.power.items()}
        p_sun = totals["sun"]
        out = {
            "time": self.time.isoformat() if self.time is not None else None,
            "sun_elevation_deg": round(self.sun_elevation, 3),
            "sun_azimuth_deg": round(self.sun_azimuth, 3),
            "dni_mean_w_m2": round(float(self.dni.mean()), 1) if self.n else 0.0,
            "n_heliostats": self.n,
            "n_defocused": int((~self.active).sum()),
            "power_mw": {k: round(v / 1e6, 4) for k, v in totals.items()},
            "field_efficiency": {
                k: round(totals[k] / totals[prev], 5) if totals[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "total_optical_efficiency": round(totals["intercept"] / p_sun, 5) if p_sun > 0 else 0.0,
        }
        out.update(self.flux_stats())
        return out


def evaluate(
    case: Case,
    fld: Field,
    sun_elevation: float,
    sun_azimuth: float,
    dni,
    aim: tuple[np.ndarray, np.ndarray] | None = None,
    active: np.ndarray | None = None,
    grid: ReceiverGrid | None = None,
    with_flux: bool = True,
    time: pd.Timestamp | None = None,
    shading_points: int | None = None,
) -> Snapshot:
    """Evaluate the field for one sun state.

    ``dni`` is a scalar or a per-heliostat array (W/m2). ``aim`` = (azimuth_deg,
    z) per heliostat (default: single-point aiming); ``active`` False marks
    defocused heliostats (no power on the receiver).
    """
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    n = fld.n
    dni_v = np.broadcast_to(np.asarray(dni, dtype=float), (n,)).copy()
    aim_az, aim_z = aim if aim is not None else default_aim(rec, fld.xyz)
    aim_az = np.asarray(aim_az, dtype=float)
    aim_z = np.asarray(aim_z, dtype=float)
    active = np.ones(n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    zeros = np.zeros(n)
    if sun_elevation <= 0 or n == 0 or not np.any(dni_v > 0):
        return Snapshot(time, sun_elevation, sun_azimuth, dni_v, aim_az, aim_z, active,
                        {k: zeros.copy() for k in LOSSES}, {k: zeros.copy() for k in STAGES},
                        np.zeros(grid.shape), grid, rec.max_flux_kw_m2)

    s = sun.sun_vector(sun_elevation, sun_azimuth)
    aim_xyz = surface_point(rec, aim_az, aim_z)
    normals, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz, aim_xyz, s)

    if case.shading.enabled:
        sb = shading.shading_blocking(
            fld.xyz, normals, s, t, hel, rec, fld.pairs,
            n_points=shading_points or case.shading.n_points,
            tower_shadow=case.shading.tower_shadow,
        )
    else:
        sb = {"shading": np.ones(n), "blocking": np.ones(n)}

    sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
    eta = {
        "cosine": cos_inc,
        "reflectivity": np.full(n, hel.reflectivity * hel.cleanliness),
        "shading": sb["shading"],
        "blocking": sb["blocking"],
        "attenuation": optics.attenuation(slant, case.atmosphere),
        "intercept": optics.intercept_fraction(rec, aim_xyz, t, sigma) * active,
    }
    power = {"sun": dni_v * hel.area}
    for prev, k in zip(STAGES[:-1], STAGES[1:]):
        power[k] = power[prev] * eta[k]

    if with_flux:
        p_att = power["attenuation"] * active
        flux_flat, _ = optics.flux_on_receiver(grid, aim_xyz[active], t[active], sigma[active], p_att[active])
    else:
        flux_flat = np.zeros(grid.points.shape[0])
    return Snapshot(time, float(sun_elevation), float(sun_azimuth), dni_v, aim_az, aim_z, active,
                    eta, power, grid.to_map(flux_flat), grid, rec.max_flux_kw_m2)


def sun_state(case: Case, time) -> tuple[pd.Timestamp, float, float, float]:
    """(timestamp, elevation, azimuth, clear-sky DNI) at a local time."""
    idx = sun.to_times(time, case.site.timezone)
    el, az = sun.solar_position(idx, case.site)
    if case.weather.source == "file":
        series = sun.load_weather(case)
        dni_v = float(series.iloc[series.index.get_indexer(idx, method="nearest")[0]])
    else:
        dni_v = float(sun.dni(el[0], case.sun, case.site))
    return idx[0], float(el[0]), float(az[0]), dni_v


def snapshot(
    case: Case,
    fld: Field,
    time,
    aim: tuple[np.ndarray, np.ndarray] | None = None,
    grid: ReceiverGrid | None = None,
    dni=None,
    with_flux: bool = True,
    active: np.ndarray | None = None,
) -> Snapshot:
    """Evaluate the field at a local time. ``dni`` (scalar or per-heliostat) overrides the model."""
    ts, el, az, dni_model = sun_state(case, time)
    return evaluate(case, fld, el, az, dni_model if dni is None else dni, aim=aim, active=active,
                    grid=grid, with_flux=with_flux, time=ts)


# --------------------------------------------------------------------------- annual

@dataclass
class AnnualResult:
    energy_mwh: dict[str, float]  # field totals after each stage
    heliostat_energy_mwh: np.ndarray  # (N,) energy on the receiver per heliostat
    heliostat_efficiency: np.ndarray  # (N,) annual optical efficiency per heliostat
    timeseries: pd.DataFrame  # one row per evaluated sun state
    method: str = ""
    aiming: str = "center"
    runtime_s: float = 0.0

    def summary(self) -> dict:
        e = self.energy_mwh
        ts = self.timeseries
        over = ts.loc[ts["over_limit"], "weight_h"].sum() if "over_limit" in ts else float("nan")
        return {
            "method": self.method,
            "aiming": self.aiming,
            "n_sun_states": int(len(ts)),
            "energy_mwh": {k: round(v, 2) for k, v in e.items()},
            "annual_efficiency": {
                k: round(e[k] / e[prev], 5) if e[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "total_optical_efficiency": round(e["intercept"] / e["sun"], 5) if e["sun"] > 0 else 0.0,
            "max_peak_flux_kw_m2": round(float(ts["peak_flux_kw_m2"].max()), 2) if len(ts) else 0.0,
            "hours_over_flux_limit": round(float(over), 1),
            "runtime_s": round(self.runtime_s, 2),
        }


def annual(
    case: Case,
    fld: Field,
    method: str | None = None,
    aiming: str | None = None,
    grid: ReceiverGrid | None = None,
    samples: "sun.Samples | None" = None,
    with_flux: bool = True,
    shading_points: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AnnualResult:
    """Annual energy on the receiver.

    method: "sun_bins" (representative sun positions, fast), "representative_days"
    or "hourly". aiming: "center" or an APS method ("kfactor", "greedy", "milp")
    which then enforces the flux limit at every sun state.
    """
    t0 = _time.perf_counter()
    method = method or case.annual.method
    aiming = aiming or case.annual.aiming
    grid = grid or build_grid(case.receiver)
    samples = samples or sun.annual_samples(case, method)
    limit = case.receiver.max_flux_kw_m2
    energy = {k: 0.0 for k in STAGES}
    per_rec = np.zeros(fld.n)
    per_sun = np.zeros(fld.n)
    rows = []
    for idx in range(len(samples)):
        el, az, dni_v, w = (samples.elevation[idx], samples.azimuth[idx],
                            samples.dni[idx], samples.weight_h[idx])
        ts = samples.times[idx] if samples.times is not None else None
        if aiming == "center":
            snap = evaluate(case, fld, el, az, dni_v, grid=grid, with_flux=with_flux, time=ts,
                            shading_points=shading_points)
        else:
            from .aps import optimize_aims  # local import: aps depends on this module

            res = optimize_aims(case, fld, el, az, dni_v, method=aiming, grid=grid,
                                shading_points=shading_points)
            snap = evaluate(case, fld, el, az, dni_v, aim=res.aim, active=res.active, grid=grid,
                            with_flux=True, time=ts, shading_points=shading_points)
        for k in STAGES:
            energy[k] += snap.power[k].sum() * w / 1e6  # W*h -> MWh
        per_rec += snap.power["intercept"] * w / 1e6
        per_sun += snap.power["sun"] * w / 1e6
        peak = snap.flux.max() / 1e3
        rows.append({
            "time": ts, "sun_elevation_deg": el, "sun_azimuth_deg": az, "weight_h": w,
            "dni_w_m2": dni_v, "p_receiver_mw": snap.p_receiver / 1e6,
            "optical_efficiency": snap.p_receiver / snap.power["sun"].sum() if snap.power["sun"].sum() > 0 else 0.0,
            "peak_flux_kw_m2": peak, "over_limit": bool(with_flux and peak > limit),
            "n_defocused": int((~snap.active).sum()),
        })
        if progress:
            progress(idx + 1, len(samples))
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(per_sun > 0, per_rec / per_sun, 0.0)
    return AnnualResult(energy, per_rec, eff, pd.DataFrame(rows), method, aiming,
                        _time.perf_counter() - t0)
