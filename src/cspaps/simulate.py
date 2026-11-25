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
from .receiver import ReceiverGrid, build_grid, default_aim, effective_absorptivity, surface_point
from .uniformity import region_of_grid, uniformity_metrics

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
    flux: np.ndarray  # (n_height, n_azimuth) incident flux, W/m2
    grid: ReceiverGrid = dc_field(repr=False)
    region: np.ndarray | None = dc_field(default=None, repr=False)  # (M,) preheater cells H
    beams: dict | None = dc_field(default=None, repr=False)  # aim_xyz, t, sigma, power of active beams
    margin: float = 0.0
    alpha_eff: np.ndarray | None = dc_field(default=None, repr=False)  # (N,) beam-spread absorptivity

    tower_fraction: np.ndarray | None = dc_field(default=None, repr=False)

    @property
    def n(self) -> int:
        return int(self.aim_z.size)

    @property
    def eta_total(self) -> np.ndarray:
        return np.prod([self.eta[k] for k in LOSSES], axis=0)

    @property
    def p_receiver(self) -> float:
        """Intercepted (incident) power on the receiver surface, W."""
        return float(self.power["intercept"].sum())

    @property
    def absorbed_flux(self) -> np.ndarray:
        return self.flux * self.grid.to_map(self.grid.alpha)

    @property
    def p_absorbed(self) -> float:
        """Absorbed power, W: intercepted power x flux-weighted absorptivity (flux map), or x the
        beam-spread absorptivity of each heliostat when no flux map was computed."""
        f = self.flux.sum()
        if f <= 0:
            if self.alpha_eff is not None:
                return float((self.power["intercept"] * self.alpha_eff).sum())
            return float(self.p_receiver * self.grid.alpha.mean())
        return float(self.p_receiver * (self.absorbed_flux.sum() / f))

    def flux_stats(self) -> dict:
        kw = self.flux / 1e3
        ratio = self.grid.to_map(self.flux.ravel() / self.grid.limit)
        j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        lim_kw = self.grid.to_map(self.grid.limit)[j] / 1e3
        region = self.grid.to_map(self.region) if self.region is not None else None
        u = uniformity_metrics(self.absorbed_flux, region)
        return {
            "peak_flux_kw_m2": round(float(kw.max()), 2),
            "mean_flux_kw_m2": round(float(kw.mean()), 2),
            "peak_to_limit": round(float(ratio.max()), 4),
            "hotspot_panel": int(self.grid.to_map(self.grid.panel)[j]) + 1,
            "hotspot_azimuth_deg": round(float(self.grid.azimuth_deg[j[1]]), 2),
            "hotspot_z_m": round(float(self.grid.z[j[0]]), 3),
            "margin_at_hotspot_kw_m2": round(float(lim_kw - kw[j]), 2),
            "cv_2d": round(u["cv_2d"], 4),
            "cv_circumferential": round(u["cv_circ"], 4),
            "cv_height": round(u["cv_height"], 4),
            "flux_map_power_mw": round(float(self.flux.sum()) * self.grid.cell_area / 1e6, 4),
            "flux_basis": "coarse grid; see flux_check for the verified peak",
        }

    def thermal_objective(self, spec) -> dict:
        """Area-weighted absorbed flux on H, normalised to intercepted-energy quadrature."""
        q = self.absorbed_flux
        mask = self.grid.to_map(self.region) if self.region is not None else np.ones(q.shape, bool)
        region = q[mask]
        total = self.p_absorbed
        scale = total / q.sum() if q.sum() > 0 else 0.0  # W per cell per sampled W/m²
        load = float(region.sum() * scale)
        spread = float(region.std() * region.size * scale) if region.size else 0.0
        cv = spread / load if load > 0 else None
        valid = (not spec.enabled or
                 (load >= spec.min_load_mw * 1e6 and
                  (spec.mode == "weighted" or (cv is not None and cv <= spec.max_cv + 1e-10))))
        value = total - spec.weight * spread if spec.enabled and spec.mode == "weighted" else total
        return {"objective_w": float(value), "power_w": total, "power_h_w": load,
                "spread_w": spread, "cv_2d": cv, "uniformity_met": bool(valid)}

    def verify(self, rec, margin: float | None = None, max_cells: int = 400_000):
        """Fine-grid feasibility check of this snapshot's beams (uses the DNI it was evaluated with)."""
        from .verify import verify_beams

        b = self.beams
        if b is None:
            raise ValueError("snapshot has no beam data (evaluated with with_flux=False?)")
        chk = verify_beams(rec, b["aim_xyz"], b["t"], b["sigma"], b["power"],
                           self.margin if margin is None else margin, max_cells=max_cells)
        return chk.remap(b["index"])

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
            "intercepted_power_mw": round(totals["intercept"] / 1e6, 4),
            "absorbed_power_mw": round(self.p_absorbed / 1e6, 4),
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
    blockers: np.ndarray | None = None,
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
    region = region_of_grid(case, grid)
    if sun_elevation <= 0 or n == 0 or not np.any(dni_v > 0):
        beams = ({"aim_xyz": np.empty((0, 3)), "t": np.empty((0, 3)), "sigma": np.empty(0),
                  "power": np.empty(0), "index": np.empty(0, dtype=int)} if with_flux else None)
        return Snapshot(time, sun_elevation, sun_azimuth, dni_v, aim_az, aim_z, active,
                        {k: zeros.copy() for k in LOSSES}, {k: zeros.copy() for k in STAGES},
                        np.zeros(grid.shape), grid, region, beams, case.aps.safety_margin, tower_fraction=zeros.copy())

    s = sun.sun_vector(sun_elevation, sun_azimuth)
    aim_xyz = surface_point(rec, aim_az, aim_z)
    normals, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz, aim_xyz, s)

    if case.shading.enabled:
        sb = shading.shading_blocking(
            fld.xyz, normals, s, t, hel, rec, fld.pairs,
            n_points=shading_points or case.shading.n_points,
            tower_shadow=case.shading.tower_shadow, blockers=blockers,
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

    beams = None
    if with_flux:
        p_att = power["attenuation"] * active
        on = np.flatnonzero(active & (p_att > 0))
        flux_flat, _ = optics.flux_on_receiver(grid, aim_xyz[on], t[on], sigma[on], p_att[on])
        beams = {"aim_xyz": aim_xyz[on], "t": t[on], "sigma": sigma[on], "power": p_att[on], "index": on}
    else:
        flux_flat = np.zeros(grid.points.shape[0])
    alpha_eff = effective_absorptivity(rec, aim_az, sigma)
    return Snapshot(time, float(sun_elevation), float(sun_azimuth), dni_v, aim_az, aim_z, active,
                    eta, power, grid.to_map(flux_flat), grid, region, beams, case.aps.safety_margin, alpha_eff,
                    sb.get("tower_fraction", zeros))


def beam_spots(case: Case, xyz: np.ndarray, aim_azimuth, aim_z, sun_elevation: float, sun_azimuth: float,
               k: float = 2.0) -> dict:
    """Size of every mirror's spot on the receiver for a plan (variable geometry from the case).

    Beam std ``sigma_m`` in the image plane (HFLCAL: sun shape, slope / tracking / specular errors,
    astigmatism at the slant range), its k-sigma ellipse on the surface (``optics.surface_spot``) and the
    intercepted fraction of the beam (``optics.intercept_fraction``)."""
    rec = case.receiver
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    aim_az = np.asarray(aim_azimuth, dtype=float).reshape(-1)
    aim_zz = np.asarray(aim_z, dtype=float).reshape(-1)
    aim_xyz = surface_point(rec, aim_az, aim_zz)
    _, t, slant, cos_inc = optics.heliostat_geometry(xyz, aim_xyz, sun.sun_vector(sun_elevation, sun_azimuth))
    sigma = optics.beam_sigma(slant, cos_inc, case.heliostat, case.sun)
    out = optics.surface_spot(rec, aim_az, t, sigma, k)
    out["slant_m"] = slant
    out["intercept"] = optics.intercept_fraction(rec, aim_xyz, t, sigma)
    return out


def occlusion_detail(case: Case, snap: Snapshot, xyz: np.ndarray, i: int, n_points: int = 8,
                     max_radius: float = 1000.0) -> dict:
    """Where mirror ``i`` loses light to its neighbours: ``n_points`` x ``n_points`` sample points on it, each
    "clear", "shaded" (its ray to the sun hits a neighbouring mirror), "tower" (... the tower or receiver) or
    "blocked" (its reflected ray hits a neighbouring mirror), and the neighbours (within the shading / blocking
    interaction radius, posed as they track their own aims) that shade or block it. Same ray tests as
    ``shading.shading_blocking`` (finer sampling, so the fractions differ from the solver's by the sampling)."""
    rec, hel = case.receiver, case.heliostat
    xyz = np.asarray(xyz, dtype=float)
    s = sun.sun_vector(snap.sun_elevation, snap.sun_azimuth)
    aims = surface_point(rec, np.asarray(snap.aim_azimuth, float), np.asarray(snap.aim_z, float))
    normals, t, _, _ = optics.heliostat_geometry(xyz, aims, s)
    dz = float(np.ptp(xyz[:, 2])) if len(xyz) > 1 else 0.0
    t_el = float(np.degrees(np.arcsin(np.clip(t[i, 2], -1.0, 1.0))))
    radius = min(max(shading.interaction_radius(hel, snap.sun_elevation, dz),
                     shading.interaction_radius(hel, t_el, dz)), max_radius)
    d_h = np.hypot(xyz[:, 0] - xyz[i, 0], xyz[:, 1] - xyz[i, 1])
    nb = np.flatnonzero((d_h <= radius) & (np.arange(len(xyz)) != i))
    e_w, e_h = shading.mirror_frames(normals[i:i + 1])
    pts = shading.mirror_points(xyz[i:i + 1], e_w, e_h, hel, shading.grid_offsets(n_points))[0]  # (P, 3)
    status = np.full(len(pts), "clear", dtype=object)
    shaders = blockers = np.zeros(0, dtype=int)
    if nb.size:
        w_nb, h_nb = shading.mirror_frames(normals[nb])
        args = (xyz[nb][None], normals[nb][None], w_nb[None], h_nb[None], 0.5 * hel.width, 0.5 * hel.height)
        hit_s = shading.ray_hits_rect(pts[:, None, :], s[None, None, :], *args)
        hit_b = shading.ray_hits_rect(pts[:, None, :], t[i][None, None, :], *args)
        shaders, blockers = nb[hit_s.any(axis=0)], nb[hit_b.any(axis=0)]
        status[hit_b.any(axis=1)] = "blocked"
        status[hit_s.any(axis=1)] = "shaded"
    if case.shading.tower_shadow:
        status[shading.ray_hits_structure(pts[None], s[None, None, :], rec)[0]] = "tower"
    if not case.shading.enabled:
        status[:] = "clear"
    p = len(pts)
    return {"points": pts, "status": status.astype(str), "neighbours": nb, "normals": normals[nb],
            "shaders": shaders, "blockers": blockers, "radius_m": float(radius), "n_points": p,
            "lost_shading": float(np.isin(status, ("shaded", "tower")).sum() / p),
            "lost_blocking": float((status == "blocked").sum() / p)}


def mirror_detail(case: Case, snap: Snapshot, xyz: np.ndarray, i: int, initial=None) -> dict:
    """Everything about mirror ``i`` of an evaluated snapshot, for the GUI's mirror inspector.

    Geometry (normal bisecting sun and target, incidence, slant range, compass bearing from the tower), the
    stage efficiencies and powers of the snapshot (``eta``, ``power``; optical efficiency = product of the
    stages = intercepted / sun power), the beam and its 2-sigma spot on the receiver, the panel it hits, its
    share of the receiver power, its rank in the field, and the field means of every stage (active mirrors)
    to compare with. ``initial`` = (azimuth, z, active) arrays of the initial state add the aim shift and the
    mirror rotation (``aps.move_stats``)."""
    from .receiver import panel_of

    rec, hel = case.receiver, case.heliostat
    xyz = np.asarray(xyz, dtype=float)
    p = xyz[i]
    s = sun.sun_vector(snap.sun_elevation, snap.sun_azimuth)
    aim_az, aim_z = float(snap.aim_azimuth[i]), float(snap.aim_z[i])
    aim_xyz = surface_point(rec, np.array([aim_az]), np.array([aim_z]))[0]
    normal, t, slant, cos_inc = optics.heliostat_geometry(p[None], aim_xyz[None], s)
    sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
    spot = optics.surface_spot(rec, np.array([aim_az]), t, sigma, 2.0)
    eta = {k: float(snap.eta[k][i]) for k in LOSSES}
    power = {k: float(snap.power[k][i]) for k in STAGES}
    total = float(np.prod(list(eta.values())))
    active = bool(snap.active[i])
    alpha = float(snap.alpha_eff[i]) if snap.alpha_eff is not None else float(rec.absorptivity)
    on = np.asarray(snap.active, bool)
    eta_tot = snap.eta_total
    lit = on & (snap.power["sun"] > 0)
    rank = float((eta_tot[lit] < eta_tot[i]).mean()) if lit.any() and active else None
    rx, ry = p[0] - rec.center[0], p[1] - rec.center[1]
    n = normal[0]
    out = {
        "index": int(i), "xyz": p, "distance_m": float(np.hypot(rx, ry)),
        "bearing_deg": float(np.degrees(np.arctan2(rx, ry)) % 360.0), "active": active, "dni_w_m2": float(snap.dni[i]),
        "sun_vector": s, "aim_azimuth_deg": aim_az % 360.0, "aim_z": aim_z, "aim_xyz": aim_xyz,
        "target_dir": t[0], "normal": n, "normal_azimuth_deg": float(np.degrees(np.arctan2(n[0], n[1])) % 360.0),
        "normal_elevation_deg": float(np.degrees(np.arcsin(np.clip(n[2], -1.0, 1.0)))),
        "incidence_deg": float(np.degrees(np.arccos(np.clip(cos_inc[0], -1.0, 1.0)))), "slant_m": float(slant[0]),
        "eta": eta, "eta_total": total, "power_w": power, "absorbed_w": power["intercept"] * alpha, "alpha": alpha,
        "sigma_m": float(sigma[0]), "spot_width_m": float(spot["width_m"][0]), "spot_height_m": float(spot["height_m"][0]),
        "spot_incidence_deg": float(spot["incidence_deg"][0]), "spot_cov": spot["cov"][0],
        "panel": int(panel_of(rec, aim_az)) + 1,
        "share": float(power["intercept"] / snap.p_receiver) if snap.p_receiver > 0 else 0.0,
        "rank": rank, "tower_fraction": float(snap.tower_fraction[i]) if snap.tower_fraction is not None else 0.0,
        "field_mean": {k: float(snap.eta[k][lit].mean()) if lit.any() else float("nan") for k in LOSSES},
        "field_mean_total": float(eta_tot[lit].mean()) if lit.any() else float("nan"),
    }
    if initial is not None:
        from .aps import move_stats

        a0, z0, act0 = (np.atleast_1d(np.asarray(v)[i]) for v in initial)
        per, _ = move_stats(case, p[None], snap.sun_elevation, snap.sun_azimuth, (a0, z0), act0.astype(bool),
                            (np.array([aim_az]), np.array([aim_z])), np.array([active]))
        out["initial"] = {"aim_azimuth_deg": float(a0[0]) % 360.0, "aim_z": float(z0[0]), "active": bool(act0[0]),
                          "aim_xyz": surface_point(rec, a0.astype(float), z0.astype(float))[0],
                          "shift_m": float(per["shift_m"][0]), "rotation_deg": float(per["rotation_deg"][0])}
    return out


def sun_state(case: Case, time) -> tuple[pd.Timestamp, float, float, float]:
    """(timestamp, elevation, azimuth, clear-sky DNI) at a local time."""
    idx = sun.to_times(time, case.site.timezone)
    el, az = sun.solar_position(idx, case.site)
    if case.weather.source == "file":
        dni_v = sun.weather_dni_at(case, idx)  # raises outside the file's time span
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
    sampling: dict = dc_field(default_factory=dict)
    state_power_w: np.ndarray | None = dc_field(default=None, repr=False)  # (S, N) intercepted power per state

    def summary(self) -> dict:
        e = self.energy_mwh
        ts = self.timeseries
        over = ts.loc[ts["over_limit"], "weight_h"].sum() if "over_limit" in ts else float("nan")
        feasible = ts["operational_feasible"] if "operational_feasible" in ts else pd.Series(False, index=ts.index)
        feasible_energy = float((ts.loc[feasible, "p_absorbed_mw"] * ts.loc[feasible, "weight_h"]).sum()) if len(ts) else 0.0
        uniformity_failed = float(ts.loc[~ts["uniformity_feasible"], "weight_h"].sum()) if "uniformity_feasible" in ts else 0.0
        flux_failed = float(ts.loc[~ts["flux_feasible"], "weight_h"].sum()) if "flux_feasible" in ts else float(over)
        return {
            "method": self.method,
            "aiming": self.aiming,
            "sampling": self.sampling,
            "n_sun_states": int(len(ts)),
            "energy_mwh": {k: round(v, 2) for k, v in e.items()},
            "annual_efficiency": {
                k: round(e[k] / e[prev], 5) if e[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "total_optical_efficiency": round(e["intercept"] / e["sun"], 5) if e["sun"] > 0 else 0.0,
            "max_peak_flux_kw_m2": round(float(ts["peak_flux_kw_m2"].max()), 2) if len(ts) else 0.0,
            "max_peak_to_limit": round(float(ts["peak_to_limit"].max()), 4) if len(ts) else 0.0,
            "hours_over_flux_limit": round(float(over), 1),
            "absorbed_energy_mwh": round(float((ts["p_absorbed_mw"] * ts["weight_h"]).sum()), 2) if len(ts) else 0.0,
            "feasible_absorbed_energy_mwh": round(feasible_energy, 2),
            "hours_uniformity_failed": round(uniformity_failed, 2),
            "hours_flux_or_verification_failed": round(flux_failed, 2),
            "feasible_sun_states": int(feasible.sum()),
            "peak_basis": ("verified fine grid" if self.aiming != "center" or self.sampling.get("flux_checked", True)
                           else "coarse grid (not checked)"),
            "runtime_s": round(self.runtime_s, 2),
        }


def _annual_state(case, fld, grid, el, az, dni_v, ts, aiming, with_flux, shading_points, blockers=None,
                  check_flux=True):
    """Evaluate one sun state; returns (stage power arrays, row dict)."""
    if aiming == "center":
        snap = evaluate(case, fld, el, az, dni_v, grid=grid, with_flux=with_flux, time=ts,
                        shading_points=shading_points, blockers=blockers)
        peak = snap.flux.max() / 1e3
        ratio = float((snap.flux.ravel() / grid.limit).max())
        status = ""
        if with_flux and check_flux and snap.beams is not None and len(snap.beams["power"]):
            # feasibility from the fine-grid check with full beams (the coarse map can miss narrow peaks)
            chk = snap.verify(case.receiver, max_cells=case.aps.verify_max_cells)
            peak, ratio, status = chk.peak_flux_kw_m2, chk.peak_ratio, chk.status
        uniformity_status = "not_checked"
        cv_2d = cv_circ = cv_height = power_h_mw = float("nan")
    else:
        from .aps import optimize_aims  # local import: aps depends on this module

        res = optimize_aims(case, fld, el, az, dni_v, method=aiming, grid=grid, shading_points=shading_points,
                            joint_trials=min(2, case.aps.coupled_trials))
        snap = evaluate(case, fld, el, az, dni_v, aim=res.aim, active=res.active, grid=grid,
                        with_flux=True, time=ts, shading_points=shading_points)
        actual = (res.check if res.info.get("joint_pose_checked") else
                  snap.verify(case.receiver, max_cells=case.aps.verify_max_cells))
        peak, ratio = actual.peak_flux_kw_m2, actual.peak_ratio
        status = actual.status
        uniformity_status = res.info.get("uniformity_status", "disabled") if case.uniformity.enabled else "disabled"
        cv_2d = res.info.get("cv_2d_H", float("nan"))
        cv_circ = res.info.get("cv_circ_H", float("nan"))
        cv_height = res.info.get("cv_height_H", float("nan"))
        power_h_mw = res.info.get("power_H_mw", float("nan"))
    flux_on = with_flux or aiming != "center"
    over_limit = bool(flux_on and ratio > 1.0 + 1e-9)
    # one feasibility rule everywhere: the safety margin is part of the constraint, so a plan that uses part of
    # the margin ("margin_reduced": within the hard limit, above (1 - margin) x limit) is not counted as feasible
    allowed = 1.0 - case.aps.safety_margin
    margin_used = bool(flux_on and ratio > allowed + 1e-9)
    checked = aiming != "center" or (with_flux and check_flux)
    flux_feasible = bool(checked and (status == "ok" if status else (with_flux and not margin_used)))
    if case.uniformity.enabled and flux_on:
        mask = region_of_grid(case, grid, case.uniformity)
        metrics = uniformity_metrics(snap.absorbed_flux, grid.to_map(mask))
        cv_2d = metrics["cv_2d"]
        cv_circ = metrics["cv_circ"]
        cv_height = metrics["cv_height"]
        thermal = snap.thermal_objective(case.uniformity)
        power_h_mw = thermal["power_h_w"] / 1e6
        uniformity_status = "feasible" if thermal["uniformity_met"] else "not_reached"
        if power_h_mw < case.uniformity.min_load_mw:
            uniformity_status = "below_min_load"
    uniformity_feasible = (not case.uniformity.enabled or
                           uniformity_status in ("feasible", "weighted"))
    min_load_ok = (not case.uniformity.enabled or power_h_mw >= case.uniformity.min_load_mw - 1e-9)
    uniformity_feasible = uniformity_feasible and min_load_ok
    operational_feasible = flux_feasible and uniformity_feasible
    circ_spread = 0.0
    if flux_on:
        from .layout_uniformity import sector_of_columns

        q = snap.absorbed_flux
        sector = np.broadcast_to(sector_of_columns(case, grid.azimuth_deg), q.shape)
        mask = grid.to_map(snap.region)
        sector = sector[mask]
        count = np.bincount(sector)
        if count.size and q.sum() > 0:
            means = np.bincount(sector, weights=q[mask])[count > 0] / count[count > 0]
            variance = np.average((means - q[mask].mean()) ** 2, weights=count[count > 0])
            circ_spread = mask.sum() * np.sqrt(variance) * snap.p_absorbed / q.sum()
    row = {
        "time": ts, "sun_elevation_deg": el, "sun_azimuth_deg": az,
        "dni_w_m2": float(np.mean(dni_v)) if np.size(dni_v) else 0.0, "p_receiver_mw": snap.p_receiver / 1e6, "p_absorbed_mw": snap.p_absorbed / 1e6,
        "optical_efficiency": snap.p_receiver / snap.power["sun"].sum() if snap.power["sun"].sum() > 0 else 0.0,
        "peak_flux_kw_m2": peak, "peak_to_limit": ratio, "over_limit": over_limit, "margin_used": margin_used,
        "n_defocused": int((~snap.active).sum()), "aps_status": status,
        "flux_feasible": bool(flux_feasible), "uniformity_status": uniformity_status,
        "uniformity_cv_2d": cv_2d, "uniformity_cv_circ": cv_circ, "uniformity_cv_height": cv_height,
        "uniformity_power_h_mw": power_h_mw, "uniformity_feasible": bool(uniformity_feasible),
        "operational_feasible": bool(operational_feasible),
        "feasible_absorbed_mw": snap.p_absorbed / 1e6 if operational_feasible else 0.0,
        "circ_spread_mw": circ_spread / 1e6,
        "surface_spread_mw": snap.thermal_objective(case.uniformity)["spread_w"] / 1e6 if flux_on else 0.0,
    }
    return {k: snap.power[k] for k in STAGES}, row


def _annual_chunk(args):
    case, xyz, ids, items, aiming, with_flux, shading_points, blockers, check_flux = args
    fld = Field(xyz, ids)
    grid = build_grid(case.receiver)
    return [(i, *_annual_state(case, fld, grid, el, az, d, ts, aiming, with_flux, shading_points, blockers,
                               check_flux)) for i, el, az, d, ts in items]


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
    workers: int | None = None,
    keep_state_power: bool = False,
    blockers: np.ndarray | None = None,
    check_flux: bool = True,
) -> AnnualResult:
    """Annual energy on the receiver.

    method: "sun_bins" (representative sun positions, fast), "sun_grid",
    "representative_days" or "hourly". aiming: "center" or an APS method
    ("kfactor", "greedy", "milp") which then enforces the flux limit (verified)
    at every sun state. Sun states are independent and run in parallel with
    ``workers`` processes (default ``annual.workers``). ``keep_state_power``
    keeps the (states x heliostats) intercepted power in ``state_power_w``.
    ``blockers`` (centre aiming only) limits the mirrors that shade / block others
    (see ``shading.shading_blocking``). ``check_flux`` (centre aiming with flux maps): verify every state on
    the fine grid for the feasibility columns; False skips the check (energy / spread studies) and then no
    state counts as flux-feasible.
    """
    from .parallel import n_workers, pool

    t0 = _time.perf_counter()
    method = method or case.annual.method
    aiming = aiming or case.annual.aiming
    grid = grid or build_grid(case.receiver)
    samples = samples or sun.annual_samples(case, method)
    n_s = len(samples)
    # per-heliostat DNI when the samples carry spatial maps (each mirror reads the map at its position)
    items = [(i, float(samples.elevation[i]), float(samples.azimuth[i]), samples.dni_at(i, fld.xyz[:, :2]),
              samples.times[i] if samples.times is not None else None) for i in range(n_s)]
    nw = min(n_workers(case.annual.workers if workers is None else workers), max(n_s, 1))
    results: list = [None] * n_s
    if nw > 1 and n_s > 1:
        chunks = [items[k::nw] for k in range(nw)]
        tasks = [(case, fld.xyz, fld.ids, c, aiming, with_flux, shading_points, blockers, check_flux) for c in chunks]
        with pool(nw, probe=(_annual_chunk, tasks[0])) as p:
            done = 0
            for out in p.imap_unordered(_annual_chunk, tasks):
                for i, powers, row in out:
                    results[i] = (powers, row)
                done += len(out)
                if progress:
                    progress(done, n_s)
    else:
        for i, el, az, d, ts in items:
            results[i] = _annual_state(case, fld, grid, el, az, d, ts, aiming, with_flux, shading_points, blockers,
                                       check_flux)
            if progress:
                progress(i + 1, n_s)
    energy = {k: 0.0 for k in STAGES}
    per_rec = np.zeros(fld.n)
    per_sun = np.zeros(fld.n)
    rows = []
    for i, (powers, row) in enumerate(results):
        w = float(samples.weight_h[i])
        for k in STAGES:
            energy[k] += powers[k].sum() * w / 1e6  # W*h -> MWh
        per_rec += powers["intercept"] * w / 1e6
        per_sun += powers["sun"] * w / 1e6
        rows.append({**row, "weight_h": w})
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(per_sun > 0, per_rec / per_sun, 0.0)
    res = AnnualResult(energy, per_rec, eff, pd.DataFrame(rows), method, aiming, _time.perf_counter() - t0)
    res.sampling = sun.sampling_record(case, samples, aiming)
    res.sampling["flux_checked"] = bool(aiming != "center" or (with_flux and check_flux))
    if keep_state_power:
        res.state_power_w = (np.stack([powers["intercept"] for powers, _ in results]) if results
                             else np.zeros((0, fld.n)))
    return res
