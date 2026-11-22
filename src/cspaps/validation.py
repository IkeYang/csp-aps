"""Independent ray-trace checks of the analytic model on the stress states of the improvement plan.

Four states, each with a verified task-1 plan (``layout_value.solve_instant``):

  low_sun          sun elevation ~10 deg in the morning: strongest cosine, shading/blocking and astigmatism
  cloud_edge       a straight soft cloud edge through the tower base: per-mirror DNI from 20 % to 100 %
  mixed_high_load  solar noon of the summer solstice, lambda_APS = 0: many mixed aim heights / offsets overlap
  near_limit       limits tightened until centre aiming exceeds them by 30 %; APS pushes the peak to the margin

For every state the same plan (aims, active mirrors, per-mirror DNI) is evaluated twice on the same grid:
the analytic model (HFLCAL beams, ``simulate.evaluate``) and the internal Monte-Carlo ray tracer
(``mcrt.trace``, independent seeds). Reported separately:

  physical error   analytic vs MCRT: power, peak, peak-to-limit, hotspot location, map NRMSE,
                   with the MCRT standard errors, grid, seeds and rays
  solver gap       plan objective vs its upper bound on the candidate set (``inner_gap``)

The two are different questions: the solver gap is measured inside the analytic model; the physical
error says how far the analytic model itself is from the ray trace. A SolTrace export of every plan
(per-mirror DNI as a reflectivity factor) is written as well, for a check with an external code.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from . import sun
from .config import Case
from .field import Field
from .mcrt import trace
from .receiver import build_grid

SUITE = ("low_sun", "cloud_edge", "mixed_high_load", "near_limit")
SUITE_ZH = {"low_sun": "低太阳高度", "cloud_edge": "云影边缘", "mixed_high_load": "高负荷混合瞄准",
            "near_limit": "逼近热流限制"}
SUITE_WHY = {
    "low_sun": "上午太阳高度约 10°：余弦损失、镜间阴影遮挡和像散最强，检验光斑形状与遮挡模型。",
    "cloud_edge": "一条直线云影边缘穿过塔基，阴影一侧 DNI 只有晴空的 20%，过渡带 40 m：检验逐镜 DNI 下的功率与热点位置。",
    "mixed_high_load": "夏至正午、只追求功率（λ = 0）：多种瞄准高度与偏移混合叠加，检验叠加光斑的峰值。",
    "near_limit": "把热流限制收紧到中心瞄准超限 30%，APS 将峰值压到安全裕度边缘：检验“未超限”是否被光线追迹确认。",
}


@dataclass
class SuiteState:
    key: str
    time: pd.Timestamp
    elevation: float
    azimuth: float
    dni: np.ndarray  # (N,) DNI of every mirror, W/m2
    case: Case  # the case the plan was solved for (near_limit: tightened limits)
    lam_aps: float
    plan: object  # layout_value.PlanResult
    note: dict = dc_field(default_factory=dict)


# --------------------------------------------------------------------------- choosing the states

def _day(case: Case, date, step_min: int = 5):
    idx = pd.date_range(pd.Timestamp(date).normalize(), periods=24 * 60 // step_min, freq=f"{step_min}min",
                        tz=case.site.timezone)
    el, az = sun.solar_position(idx, case.site)
    return idx, np.asarray(el, float), np.asarray(az, float)


def summer_solstice(case: Case, year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-06-21" if case.site.latitude >= 0 else f"{year}-12-21")


def winter_solstice(case: Case, year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-12-21" if case.site.latitude >= 0 else f"{year}-06-21")


def low_sun_time(case: Case, year: int, target_deg: float = 10.0):
    """First morning time of the winter solstice with the sun at ``target_deg`` (5-min steps)."""
    idx, el, az = _day(case, winter_solstice(case, year))
    i = int(np.argmax(el >= target_deg))
    if el[i] < target_deg:  # polar night: take the highest sun of the day
        i = int(np.argmax(el))
    return idx[i], float(el[i]), float(az[i])


def solar_noon(case: Case, date):
    idx, el, az = _day(case, date)
    i = int(np.argmax(el))
    return idx[i], float(el[i]), float(az[i])


def clear_dni(case: Case, el: float) -> float:
    """Clear-sky DNI of the case's model (the suite does not depend on the weather record)."""
    return float(np.asarray(sun.dni(np.array([el]), case.sun, case.site)).ravel()[0])


def edge_transmittance(xy: np.ndarray, toward_deg: float = 90.0, offset_m: float = 0.0, width_m: float = 40.0,
                       t_shadow: float = 0.2) -> np.ndarray:
    """Straight soft cloud edge: shadow on the side the compass direction ``toward_deg`` points to.

    u = distance from the edge line (through the tower base when ``offset_m`` = 0) along that direction;
    the transmittance falls linearly from 1 to ``t_shadow`` over ``width_m`` centred on the line."""
    a = np.radians(toward_deg)
    u = xy[:, 0] * np.sin(a) + xy[:, 1] * np.cos(a) - offset_m
    s = np.clip(u / max(width_m, 1e-9) + 0.5, 0.0, 1.0)
    return 1.0 - (1.0 - t_shadow) * s


def tighten_limits(case: Case, factor: float) -> Case:
    c = case.copy()
    c.receiver.max_flux_kw_m2 = float(c.receiver.max_flux_kw_m2) * factor
    if c.receiver.panel_max_flux_kw_m2 is not None:
        c.receiver.panel_max_flux_kw_m2 = [float(v) * factor for v in c.receiver.panel_max_flux_kw_m2]
    return c


def suite_states(case: Case, fld: Field, keys=SUITE, year: int = 2025, lam_aps: float | None = None,
                 overload: float = 1.3, progress=None) -> list[SuiteState]:
    """Build the stress states and solve their task-1 plans (formal inner settings, deterministic)."""
    from .layout_value import inner_settings, solve_instant
    from .simulate import evaluate

    say = progress or (lambda m: None)
    out = []
    lam = inner_settings(case, lam_aps).lam_aps
    for key in keys:
        if key not in SUITE:
            raise ValueError(f"unknown validation state {key!r}; choose from {SUITE}")
        c, lam_k, note = case, lam, {}
        if key == "low_sun":
            ts, el, az = low_sun_time(case, year)
            dni = np.full(fld.n, clear_dni(case, el))
        else:
            ts, el, az = solar_noon(case, summer_solstice(case, year))
            dni = np.full(fld.n, clear_dni(case, el))
        if key == "cloud_edge":
            toward = (az + 90.0) % 360.0  # edge line along the sun's azimuth: half the field in shadow
            tau = edge_transmittance(fld.xyz[:, :2], toward)
            dni = dni * tau
            note = {"edge_toward_deg": round(toward, 2), "edge_width_m": 40.0, "shadow_transmittance": 0.2,
                    "edge_through": "tower base", "n_shaded": int(np.sum(tau < 0.6)), "n_clear": int(np.sum(tau > 0.99))}
        if key in ("mixed_high_load", "near_limit"):
            lam_k = 0.0
        if key == "near_limit":
            snap = evaluate(case, fld, el, az, dni)
            ratio_c = float(snap.verify(case.receiver).peak_ratio)
            factor = ratio_c / overload
            c = tighten_limits(case, factor)
            note = {"limit_factor": round(factor, 6), "center_aim_peak_to_limit_before": round(ratio_c, 4),
                    "center_aim_peak_to_limit_after": round(ratio_c / factor, 4),
                    "max_flux_kw_m2": round(c.receiver.max_flux_kw_m2, 3)}
        say(f"plan for {key}")
        settings = inner_settings(c, lam_k)
        plan = solve_instant(c, fld, el, az, dni, settings)
        out.append(SuiteState(key, ts, el, az, dni, c, lam_k, plan, note))
    return out


# --------------------------------------------------------------------------- analytic vs MCRT

def _cell_distance(rec, grid, a: tuple[int, int], b: tuple[int, int]) -> float:
    """Distance on the cylinder surface between two (height, azimuth) cells, m."""
    dphi = (grid.azimuth_deg[a[1]] - grid.azimuth_deg[b[1]] + 180.0) % 360.0 - 180.0
    return float(np.hypot(rec.radius * np.radians(dphi), grid.z[a[0]] - grid.z[b[0]]))


def compare_state(state: SuiteState, fld: Field, rays: int | None = None, seeds: int | None = None,
                  seed0: int | None = None, grid=None) -> dict:
    """Analytic vs MCRT for the state's plan on one grid; physical error and solver gap kept apart."""
    from .simulate import evaluate

    c = state.case
    rec = c.receiver
    grid = grid or build_grid(rec)
    p = state.plan
    aim = (p.aim_azimuth, p.aim_z)
    snap = evaluate(c, fld, state.elevation, state.azimuth, state.dni, aim=aim, active=p.active, grid=grid,
                    time=state.time)
    chk = snap.verify(rec)
    k = int(seeds or c.mcrt.n_seeds)
    s0 = int(c.mcrt.seed if seed0 is None else seed0)
    nr = int(rays or c.mcrt.rays_per_heliostat)
    runs = [trace(c, fld, state.elevation, state.azimuth, state.dni, aim=aim, active=p.active, grid=grid,
                  rays_per_heliostat=nr, seed=s0 + j) for j in range(max(1, k))]
    lim = grid.to_map(grid.limit)  # allowed incident flux per cell (panel limits, height profile, basis, bias)
    fa = snap.flux
    maps = np.array([r.flux for r in runs])
    fm = maps.mean(axis=0)
    n = len(runs)

    def se(x):
        return float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    ra, rm = fa / lim, fm / lim
    ia = np.unravel_index(int(np.argmax(ra)), ra.shape)
    im = np.unravel_index(int(np.argmax(rm)), rm.shape)
    ratio_runs = maps[:, im[0], im[1]] / lim[im]
    p_m = np.array([r.power["intercept"] for r in runs])
    lit = fm > 0.05 * fm.max() if fm.max() > 0 else np.ones_like(fm, bool)
    nrmse = float(np.sqrt(np.mean((fa[lit] - fm[lit]) ** 2)) / fm[lit].mean()) if lit.any() and fm[lit].mean() > 0 else 0.0
    pa = float(snap.p_receiver)
    pm = float(p_m.mean())
    ratio_m, ratio_m_se = float(rm[im]), se(ratio_runs)
    physical = {
        "power_on_receiver_mw": {"analytic": pa / 1e6, "mcrt": pm / 1e6, "mcrt_std_error": se(p_m / 1e6),
                                 "rel_diff": (pa / pm - 1.0) if pm > 0 else None},
        "peak_flux_kw_m2": {"analytic": float(fa.max()) / 1e3, "mcrt_mean_map": float(fm.max()) / 1e3,
                            "mcrt_std_error": se(maps.reshape(n, -1).max(axis=1) / 1e3),
                            "rel_diff": float(fa.max() / fm.max() - 1.0) if fm.max() > 0 else None},
        "peak_to_limit": {"analytic_grid": float(ra[ia]), "mcrt_grid": ratio_m, "mcrt_std_error": ratio_m_se,
                          "analytic_verified_fine": float(chk.peak_ratio),
                          "allowed": float(chk.allowed_ratio)},
        "hotspot": {"analytic": {"azimuth_deg": float(grid.azimuth_deg[ia[1]]), "z_m": float(grid.z[ia[0]])},
                    "mcrt": {"azimuth_deg": float(grid.azimuth_deg[im[1]]), "z_m": float(grid.z[im[0]])},
                    "verified_fine": {"azimuth_deg": float(chk.hotspot_azimuth_deg), "z_m": float(chk.hotspot_z),
                                      "panel": int(chk.hotspot_panel)},
                    "distance_m": _cell_distance(rec, grid, ia, im)},
        "flux_map_nrmse": nrmse,
        "flux_map_correlation": float(np.corrcoef(fa.ravel(), fm.ravel())[0, 1]) if fa.std() > 0 and fm.std() > 0 else None,
        "difference_direction": "analytic minus MCRT",
        "mcrt": {"seeds": [s0 + j for j in range(n)], "rays_per_heliostat": nr, "n_rays": int(sum(r.n_rays for r in runs)),
                 "engine": "cspaps-mcrt"},
        "grid": [int(grid.azimuth_deg.size), int(grid.z.size)],
        "verification_grid": list(chk.grid),
        # the MCRT mean map + 2 standard errors stays below the limit (cell-averaged, same grid)
        "mcrt_within_limit": bool(ratio_m + 2 * (0.0 if np.isnan(ratio_m_se) else ratio_m_se) <= 1.0),
    }
    solver = {"status": p.status, "lambda_aps": state.lam_aps, "objective_mw": p.objective_w / 1e6,
              "upper_bound_mw": p.upper_bound_w / 1e6 if p.upper_bound_w is not None else None,
              "gap": p.gap, "bound_method": p.bound_method, "proven_optimal": p.proven_optimal,
              "flux_status": p.flux_status, "peak_to_allowed": p.peak_ratio,
              "n_active": p.n_active, "n_defocused": p.n_defocused,
              "candidate_set": "azimuth offsets x height levels of the case's APS settings (+ defocus)",
              "solve_time_s": p.solve_time_s}
    dni = state.dni
    return {
        "key": state.key, "label": SUITE_ZH[state.key], "why": SUITE_WHY[state.key],
        "time": state.time.isoformat(), "elevation_deg": state.elevation, "azimuth_deg": state.azimuth,
        "dni_w_m2": {"min": float(dni.min()), "mean": float(dni.mean()), "max": float(dni.max())},
        "state": state.note, "physical": physical, "solver": solver,
        "maps": {"analytic_w_m2": fa, "mcrt_w_m2": fm, "limit_w_m2": np.asarray(lim, float)},
    }


def row(res: dict) -> dict:
    """One flat table row per state (for CSV / GUI)."""
    ph, so = res["physical"], res["solver"]
    return {
        "state": res["key"], "工况": res["label"], "time": res["time"], "elevation_deg": round(res["elevation_deg"], 2),
        "dni_min_w_m2": round(res["dni_w_m2"]["min"], 1), "dni_max_w_m2": round(res["dni_w_m2"]["max"], 1),
        "power_analytic_mw": ph["power_on_receiver_mw"]["analytic"], "power_mcrt_mw": ph["power_on_receiver_mw"]["mcrt"],
        "power_mcrt_se_mw": ph["power_on_receiver_mw"]["mcrt_std_error"],
        "power_rel_diff": ph["power_on_receiver_mw"]["rel_diff"],
        "peak_analytic_kw_m2": ph["peak_flux_kw_m2"]["analytic"], "peak_mcrt_kw_m2": ph["peak_flux_kw_m2"]["mcrt_mean_map"],
        "peak_mcrt_se_kw_m2": ph["peak_flux_kw_m2"]["mcrt_std_error"], "peak_rel_diff": ph["peak_flux_kw_m2"]["rel_diff"],
        "peak_to_limit_analytic_grid": ph["peak_to_limit"]["analytic_grid"],
        "peak_to_limit_mcrt_grid": ph["peak_to_limit"]["mcrt_grid"],
        "peak_to_limit_mcrt_se": ph["peak_to_limit"]["mcrt_std_error"],
        "peak_to_limit_verified_fine": ph["peak_to_limit"]["analytic_verified_fine"],
        "mcrt_within_limit": ph["mcrt_within_limit"],
        "hotspot_distance_m": ph["hotspot"]["distance_m"], "flux_map_nrmse": ph["flux_map_nrmse"],
        "grid": "x".join(str(v) for v in ph["grid"]), "seeds": ",".join(str(s) for s in ph["mcrt"]["seeds"]),
        "rays_per_heliostat": ph["mcrt"]["rays_per_heliostat"],
        "plan_status": so["status"], "lambda_aps": so["lambda_aps"], "solver_gap": so["gap"],
        "bound_method": so["bound_method"], "objective_mw": so["objective_mw"], "upper_bound_mw": so["upper_bound_mw"],
    }
