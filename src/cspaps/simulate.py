"""Snapshot (one instant) and annual simulations with an efficiency breakdown.

Per-heliostat power chain (W):
    P_sun  = DNI * A_mirror                      (DNI on the mirror aperture)
    P_cos  = P_sun  * eta_cos
    P_ref  = P_cos  * reflectivity * cleanliness
    P_sb   = P_ref  * eta_shading_blocking       (1.0 until implemented)
    P_att  = P_sb   * eta_attenuation            (power reaching the image plane)
    P_rec  = P_att  * eta_intercept              (power hitting the receiver surface)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import optics, sun
from .config import Case
from .field import Field
from .receiver import ReceiverGrid, build_grid, default_aim, surface_point

STAGES = ("sun", "cosine", "reflectivity", "shading_blocking", "attenuation", "intercept")


@dataclass
class Snapshot:
    time: pd.Timestamp
    sun_elevation: float
    sun_azimuth: float
    dni: float
    aim_azimuth: np.ndarray
    aim_z: np.ndarray
    eta: dict[str, np.ndarray]  # per-heliostat efficiencies, keys = STAGES[1:]
    power: dict[str, np.ndarray]  # per-heliostat power after each stage, W
    flux: np.ndarray  # (n_height, n_azimuth) W/m2
    grid: ReceiverGrid = field(repr=False)

    @property
    def eta_total(self) -> np.ndarray:
        return np.prod([self.eta[k] for k in STAGES[1:]], axis=0)

    def summary(self) -> dict:
        totals = {k: float(v.sum()) for k, v in self.power.items()}
        p_sun = totals["sun"]
        return {
            "time": self.time.isoformat(),
            "sun_elevation_deg": round(self.sun_elevation, 3),
            "sun_azimuth_deg": round(self.sun_azimuth, 3),
            "dni_w_m2": round(self.dni, 1),
            "n_heliostats": int(self.aim_z.size),
            "power_mw": {k: round(v / 1e6, 4) for k, v in totals.items()},
            "field_efficiency": {
                k: round(totals[k] / totals[prev], 5) if totals[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "total_optical_efficiency": round(totals["intercept"] / p_sun, 5) if p_sun > 0 else 0.0,
            "peak_flux_kw_m2": round(float(self.flux.max()) / 1e3, 2),
            "mean_flux_kw_m2": round(float(self.flux.mean()) / 1e3, 2),
            # Should match power_mw.intercept; the gap is the flux-grid discretisation error
            "flux_map_power_mw": round(float(self.flux.sum()) * self.grid.cell_area / 1e6, 4),
        }


def _zero_snapshot(time, el, az, dni, n, grid, aim_az, aim_z) -> Snapshot:
    zeros = np.zeros(n)
    return Snapshot(
        time, el, az, dni, aim_az, aim_z,
        eta={k: zeros.copy() for k in STAGES[1:]},
        power={k: zeros.copy() for k in STAGES},
        flux=np.zeros(grid.shape), grid=grid,
    )


def snapshot(
    case: Case,
    fld: Field,
    time,
    aim: tuple[np.ndarray, np.ndarray] | None = None,
    grid: ReceiverGrid | None = None,
    dni: float | None = None,
    with_flux: bool = True,
) -> Snapshot:
    """Evaluate the field at one instant.

    ``aim`` = (azimuth_deg, z) per heliostat; defaults to single-point aiming.
    ``dni`` overrides the configured DNI model (W/m2); later this becomes a
    per-heliostat array for cloud shadows. ``with_flux=False`` skips the flux
    map (efficiencies are unaffected) for fast energy-only evaluations.
    """
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    t_idx = sun.to_times(time, case.site.timezone)
    el, az = (float(a[0]) for a in sun.solar_position(t_idx, case.site))
    aim_az, aim_z = aim if aim is not None else default_aim(rec, fld.xyz)
    aim_az = np.asarray(aim_az, dtype=float)
    aim_z = np.asarray(aim_z, dtype=float)
    dni_val = float(sun.dni(el, case.sun, case.site)) if dni is None else float(dni)
    if el <= 0 or dni_val <= 0 or fld.n == 0:
        return _zero_snapshot(t_idx[0], el, az, dni_val, fld.n, grid, aim_az, aim_z)

    s = sun.sun_vector(el, az)
    aim_xyz = surface_point(rec, aim_az, aim_z)
    _, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz, aim_xyz, s)

    eta = {
        "cosine": cos_inc,
        "reflectivity": np.full(fld.n, hel.reflectivity * hel.cleanliness),
        "shading_blocking": np.ones(fld.n),  # TODO: shading & blocking model
        "attenuation": optics.attenuation(slant),
    }
    p_sun = np.full(fld.n, dni_val * hel.area)
    p_att = p_sun * eta["cosine"] * eta["reflectivity"] * eta["shading_blocking"] * eta["attenuation"]
    sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
    eta["intercept"] = optics.intercept_fraction(rec, aim_xyz, t, sigma)
    if with_flux:
        flux_flat, _ = optics.flux_on_receiver(grid, aim_xyz, t, sigma, p_att)
    else:
        flux_flat = np.zeros(grid.points.shape[0])

    power = {"sun": p_sun}
    for prev, k in zip(STAGES[:-1], STAGES[1:]):
        power[k] = power[prev] * eta[k]
    return Snapshot(t_idx[0], el, az, dni_val, aim_az, aim_z, eta, power, grid.to_map(flux_flat), grid)


@dataclass
class AnnualResult:
    energy_mwh: dict[str, float]  # field totals after each stage
    heliostat_energy_mwh: np.ndarray  # (N,) energy on the receiver per heliostat
    heliostat_efficiency: np.ndarray  # (N,) annual optical efficiency per heliostat
    timeseries: pd.DataFrame  # one row per simulated instant

    def summary(self) -> dict:
        e = self.energy_mwh
        return {
            "energy_mwh": {k: round(v, 2) for k, v in e.items()},
            "annual_efficiency": {
                k: round(e[k] / e[prev], 5) if e[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "total_optical_efficiency": round(e["intercept"] / e["sun"], 5) if e["sun"] > 0 else 0.0,
            "max_peak_flux_kw_m2": round(float(self.timeseries["peak_flux_kw_m2"].max()), 2),
            "hours_over_flux_limit": round(float(self.timeseries.loc[self.timeseries["over_limit"], "weight_h"].sum()), 1),
        }


def annual_times(case: Case) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Representative day per month, step midpoints; weights in hours per year."""
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
    idx = sun.to_times(times, case.site.timezone)
    return idx, np.asarray(weights)


def annual(case: Case, fld: Field, grid: ReceiverGrid | None = None, progress=None) -> AnnualResult:
    """Annual energy with single-point aiming (the 'simplified aiming rule')."""
    grid = grid or build_grid(case.receiver)
    times, weights = annual_times(case)
    el, _ = sun.solar_position(times, case.site)
    use = el >= case.annual.min_sun_elevation_deg
    aim = default_aim(case.receiver, fld.xyz)
    limit = case.receiver.max_flux_kw_m2

    energy = {k: 0.0 for k in STAGES}
    per_helio_rec = np.zeros(fld.n)
    per_helio_sun = np.zeros(fld.n)
    rows = []
    sel = np.flatnonzero(use)
    for i, j in enumerate(sel):
        snap = snapshot(case, fld, times[j], aim=aim, grid=grid)
        w = weights[j]
        for k in STAGES:
            energy[k] += snap.power[k].sum() * w / 1e6  # W*h -> MWh
        per_helio_rec += snap.power["intercept"] * w / 1e6
        per_helio_sun += snap.power["sun"] * w / 1e6
        peak = snap.flux.max() / 1e3
        rows.append({
            "time": snap.time, "weight_h": w, "sun_elevation_deg": snap.sun_elevation,
            "dni_w_m2": snap.dni, "p_receiver_mw": snap.power["intercept"].sum() / 1e6,
            "peak_flux_kw_m2": peak, "over_limit": bool(peak > limit),
        })
        if progress:
            progress(i + 1, sel.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(per_helio_sun > 0, per_helio_rec / per_helio_sun, 0.0)
    return AnnualResult(energy, per_helio_rec, eff, pd.DataFrame(rows))
