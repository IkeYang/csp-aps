"""Annual value of a layout from its own per-scenario APS plans (task 1 inside task 2).

For layout L and scenario s (sun direction, hours h_s, DNI map D_s) heliostat i
reads D_s(x_i, y_i). The inner problem is task 1 with the inner weight lambda_A:

    a_s(L) = argmax_a  P_s(a) - lambda_A U_s(a)       over verified plans (incl. all defocused)

P_s = absorbed power (W) of plan a with its own joint mirror poses (shading and
blocking recomputed for those poses); U_s = A_H sigma_H(q) = area of the region
H times the area-weighted standard deviation of the absorbed flux on H, on a
fixed evaluation grid that does not depend on the layout. The outer value uses
the *same* plans and its own weight lambda_L:

    V(L) = sum_s h_s [P_s - lambda_L U_s] = E(L) - lambda_L Pi(L)          (MWh_th)
    E(L) = sum_s h_s P_s,  Pi(L) = sum_s h_s U_s

so V is affine in lambda_L for fixed plans (dV/dlambda_L = -Pi) and equals
sum_s h_s J_s when lambda_L = lambda_A. Each scenario's penalty is charged once.
Comparing layouts A and B on the same scenario set:

    V_A - V_B = (E_A - E_B) - lambda_L (Pi_A - Pi_B)
              = "more heat"  + "more uniform heating".

Status of a scenario and what it adds to V:
    feasible            verified plan with active mirrors     h_s (P_s - lambda_L U_s)
    feasible_shutdown   the best verified plan defocuses all  0  (P = U = 0)
    no_sun              sun below the horizon or zero DNI     0
    no_feasible_found   no verified plan within the budget,   0  (receiver held off; V is then a lower
                        infeasibility not proven                  value of the model)
    proven_infeasible   no plan meets the constraints          0  (receiver off)
    not_solved          not computed                          V incomplete: the layout is not ranked
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time as _time
from dataclasses import dataclass, field as dc_field, replace
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Case
from .field import Field
from .receiver import ReceiverGrid, build_grid, default_aim
from .tables import safe_filename

STATUS_RULES = {
    "feasible": "verified plan with active heliostats; adds h_s (P_s - lambda_L U_s)",
    "feasible_shutdown": "best verified plan defocuses every heliostat (P = U = 0); adds 0",
    "no_sun": "sun below the horizon or no DNI on the field; adds 0",
    "no_feasible_found": "no plan passed the fine-grid check within the budget and infeasibility is not proven; "
                         "the receiver is held off; adds 0 (conservative)",
    "proven_infeasible": "proven that no plan meets the flux / uniformity / minimum-load constraints; adds 0",
    "not_solved": "not computed; the annual value is incomplete and the layout is not ranked",
}
STATUS_ZH = {"feasible": "可行运行", "feasible_shutdown": "可行停机", "no_sun": "无日照", "no_feasible_found": "未找到可行方案",
             "proven_infeasible": "已证不可行", "not_solved": "未求解"}
COUNTED = ("feasible",)


# --------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class InnerSettings:
    """Everything that defines the inner (task 1) solve; fixed across the layouts being compared."""

    lam_aps: float
    method: str = "greedy"
    joint_trials: int = 2
    certify_rounds: int = 0
    certify_iters: int = 60
    grid_shape: tuple[int, int] = (0, 0)  # (n_height, n_azimuth) of the evaluation grid
    azimuth_offsets_deg: tuple[float, ...] = ()
    n_height_levels: int = 7
    height_span: float = 0.8
    safety_margin: float = 0.05
    allow_defocus: bool = True
    region_panels: tuple[int, ...] | None = None
    region_z: tuple[float, float] | None = None
    min_load_mw: float = 0.0
    # "responses": the spot responses of one sun direction are built once and copied for every scenario of it
    #              (plans bit-identical to independent solves);
    # "full":      also plans reused across DNI levels when the limits never acted, and warm starts along the
    #              sun path (deterministic for the set, faster; independent cold solves may differ by ties);
    # "off":       every scenario built and solved from scratch
    reuse: str = "responses"

    def record(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        d["digest"] = self.digest()
        return d

    def digest(self) -> str:
        payload = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()},
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def inner_weight(case: Case) -> float:
    """Operating weight lambda_APS: by default the same lambda as the evaluation (one preference for both
    layers); ``layout_opt.aps_weight`` sets it separately (advanced)."""
    lo = case.layout_opt
    return float(lo.uniformity_weight if lo.aps_weight is None else lo.aps_weight)


def eval_grid_shape(case: Case) -> tuple[int, int]:
    """Receiver grid fine enough for the smallest beam any allowed position can produce (layout independent).

    With ``aps.auto_grid`` off the configured receiver grid is used as it is."""
    from .optics import beam_sigma

    rec, hel = case.receiver, case.heliostat
    if not case.aps.auto_grid:
        return int(rec.n_height), int(rec.n_azimuth)
    r_min = max(case.field.exclusion_radius, rec.radius + hel.diagonal)
    slant = math.hypot(max(r_min - rec.radius, 1.0), rec.center[2] - hel.pivot_height)
    sigma = float(beam_sigma(np.array([slant]), np.array([1.0]), hel, case.sun)[0])
    h = sigma / 1.5
    n_az = max(rec.n_azimuth, int(math.ceil(2 * math.pi * rec.radius / h)))
    n_z = max(rec.n_height, int(math.ceil(rec.height / h)))
    cap = case.aps.max_solver_cells
    if n_az * n_z > cap:
        f = math.sqrt(cap / (n_az * n_z))
        n_az, n_z = max(rec.n_azimuth, int(n_az * f)), max(rec.n_height, int(n_z * f))
    return int(n_z), int(n_az)


def inner_settings(case: Case, lam_aps: float | None = None) -> InnerSettings:
    lo, aps, u = case.layout_opt, case.aps, case.uniformity
    method = aps.method if aps.method in ("greedy", "milp", "subfield") else "greedy"
    return InnerSettings(
        lam_aps=float(inner_weight(case) if lam_aps is None else lam_aps), method=method,
        joint_trials=int(lo.inner_joint_trials), certify_rounds=int(lo.inner_certify_rounds),
        certify_iters=int(lo.inner_certify_iters), grid_shape=eval_grid_shape(case),
        azimuth_offsets_deg=tuple(float(v) for v in aps.azimuth_offsets_deg), n_height_levels=int(aps.n_height_levels),
        height_span=float(aps.height_span), safety_margin=float(aps.safety_margin), allow_defocus=bool(aps.allow_defocus),
        region_panels=tuple(u.panels) if u.panels else None, region_z=tuple(u.z_range) if u.z_range else None,
        min_load_mw=float(u.min_load_mw), reuse=str(lo.inner_reuse))


def inner_case(case: Case, settings: InnerSettings) -> Case:
    """Copy of the case set up for the inner solve: weighted objective P - lambda_A U on region H."""
    c = case.copy()
    c.uniformity.enabled = True
    c.uniformity.mode = "weighted"
    c.uniformity.weight = settings.lam_aps
    c.uniformity.panels = list(settings.region_panels) if settings.region_panels else None
    c.uniformity.z_range = list(settings.region_z) if settings.region_z else None
    c.uniformity.min_load_mw = settings.min_load_mw
    c.aps.method = settings.method
    c.aps.auto_grid = False  # the evaluation grid is fixed and already fine enough
    c.aps.azimuth_offsets_deg = list(settings.azimuth_offsets_deg)
    c.aps.n_height_levels = settings.n_height_levels
    c.aps.height_span = settings.height_span
    c.aps.safety_margin = settings.safety_margin
    c.aps.allow_defocus = settings.allow_defocus
    c.aps.verify = True
    return c


# --------------------------------------------------------------------------- one instant / scenario

@dataclass
class PlanResult:
    """A verified task-1 plan for one layout at one instant, with what it achieves."""

    status: str
    p_abs_w: float  # absorbed power of the plan (0 unless status counts)
    u_w: float  # A_H sigma_H of the plan
    p_int_w: float
    p_h_w: float
    cv_2d: float | None
    cv_circ: float | None
    circ_u_w: float  # circumferential reference metric A_H sigma_circ
    objective_w: float  # P - lambda_A U
    upper_bound_w: float | None
    gap: float | None
    bound_method: str | None
    proven_optimal: bool
    peak_flux_kw_m2: float
    peak_ratio: float  # verified peak / allowed incl. margin
    flux_status: str
    n_active: int
    n_defocused: int
    dni: np.ndarray  # (N,) DNI read by every heliostat
    aim_azimuth: np.ndarray
    aim_z: np.ndarray
    active: np.ndarray
    mirror_power_w: np.ndarray  # (N,) absorbed power per heliostat (intercepted x beam absorptivity)
    flux: np.ndarray | None  # (n_height, n_azimuth) incident flux on the evaluation grid, W/m2
    solve_time_s: float
    info: dict = dc_field(default_factory=dict)
    choice: np.ndarray | None = None  # candidate index per heliostat (-1 = defocused), for warm starts

    @property
    def counts(self) -> bool:
        return self.status in COUNTED


def _circ_spread(case, snap) -> float:
    """A_H x std of the sector means of the absorbed flux (the layout-decided part, reference metric)."""
    from .layout_uniformity import sector_of_columns

    q = snap.absorbed_flux
    if q.sum() <= 0:
        return 0.0
    mask = snap.grid.to_map(snap.region)
    sector = np.broadcast_to(sector_of_columns(case, snap.grid.azimuth_deg), q.shape)[mask]
    count = np.bincount(sector)
    if not count.size:
        return 0.0
    means = np.bincount(sector, weights=q[mask])[count > 0] / count[count > 0]
    var = np.average((means - q[mask].mean()) ** 2, weights=count[count > 0])
    return float(mask.sum() * np.sqrt(var) * snap.p_absorbed / q.sum())


def inner_response(case: Case, fld: Field, elevation: float, azimuth: float, settings: InnerSettings,
                   grid: ReceiverGrid):
    """The spot responses of one sun direction (DNI-independent), as ``solve_instant`` builds them."""
    from .aps import build_response

    c = inner_case(case, settings)
    return build_response(c, fld, elevation, azimuth, grid, c.aps)


def solve_instant(case: Case, fld: Field, elevation: float, azimuth: float, dni, settings: InnerSettings | None = None,
                  grid: ReceiverGrid | None = None, keep_flux: bool = True, warm=None,
                  dni_constraint=None, response=None) -> PlanResult:
    """Task 1 for one instant with the formal settings (deterministic: same inputs -> same plan).

    ``dni_constraint`` (per heliostat, >= dni) is the DNI the flux limits are checked with, e.g. the
    clear-sky DNI under a passing cloud so the plan stays safe if the cloud suddenly clears."""
    from .aps import build_problem, solve
    from .certify import improve_and_certify
    from .coupled import search_joint_poses
    from .simulate import evaluate

    t0 = _time.perf_counter()
    settings = settings or inner_settings(case)
    c = inner_case(case, settings)
    grid = grid or build_grid(c.receiver, settings.grid_shape[1], settings.grid_shape[0])
    n = fld.n
    dni_v = np.broadcast_to(np.asarray(dni, dtype=float), (n,)).copy()
    zeros = np.zeros(n)
    if not np.isfinite(dni_v).all() or np.any(dni_v < 0):
        raise ValueError("DNI must be finite and nonnegative")
    if elevation <= 0 or n == 0 or not np.any(dni_v > 0):
        aim_az, aim_z = default_aim(c.receiver, fld.xyz)
        return PlanResult("no_sun", 0.0, 0.0, 0.0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, "no_sun", True, 0.0, 0.0,
                          "ok", n, 0, dni_v, aim_az, aim_z, np.ones(n, bool), zeros.copy(),
                          np.zeros(grid.shape) if keep_flux else None, _time.perf_counter() - t0)
    if response is not None:  # a copy: the solve adds hotspot points to its response set
        from .aps import make_problem

        prob = make_problem(copy.deepcopy(response), dni_v, dni_constraint, margin=c.aps.safety_margin)
    else:
        prob = build_problem(c, fld, elevation, azimuth, dni_v, dni_constraint, grid=grid)
    res = solve(prob, settings.method, c.aps, init=warm)
    solver_info = {k: res.info[k] for k in ("subfield_serial_s", "subfield_parallel_s", "milp_status",
                                             "final_repair_rounds", "flux_limited", "limits_active") if k in res.info}
    initial = res.choice.copy()
    dual = None
    if settings.certify_rounds > 0 and (prob.k + 1) ** min(prob.n, 20) > c.aps.coupled_exact_limit:
        res, dual = improve_and_certify(prob, res, c.aps, rounds=settings.certify_rounds,
                                        max_iter=settings.certify_iters, time_limit_s=1e9)
    res, cert = search_joint_poses(prob, fld, res, warm=initial if warm is None else warm,
                                   trials=settings.joint_trials, tighten_bound=True)
    snap = evaluate(c, fld, elevation, azimuth, dni_v, aim=res.aim, active=res.active, grid=grid)
    thermal = snap.thermal_objective(c.uniformity)
    if cert["status"] == "certified":
        status = "feasible" if res.active.any() else "feasible_shutdown"
    elif cert["status"] == "certified_infeasible":
        status = "proven_infeasible"
    else:
        status = "no_feasible_found"
    alpha = snap.alpha_eff if snap.alpha_eff is not None else np.full(n, c.receiver.absorptivity)
    mirror = snap.power["intercept"] * alpha
    ok = status == "feasible"
    p_abs = float(thermal["power_w"]) if ok else 0.0
    u = float(thermal["spread_w"]) if ok else 0.0
    chk = res.check
    upper = cert.get("upper_bound_mw")
    info = {"bound_status": cert["status"], "evaluated_joint_poses": cert.get("evaluated_joint_poses"),
            "raw_power_w": float(thermal["power_w"]), "raw_spread_w": float(thermal["spread_w"]), **solver_info}
    # the flux limits never shaped this plan: same direction, DNI scaled by a factor <= 1 -> same decisions
    info["limits_active"] = bool(solver_info.get("limits_active", True) or cert.get("flux_rejections")
                                 or settings.certify_rounds > 0 or chk is None or chk.status != "ok")
    if chk is not None:
        info.update(hotspot_panel=int(chk.hotspot_panel), hotspot_azimuth_deg=float(chk.hotspot_azimuth_deg),
                    hotspot_z_m=float(chk.hotspot_z), peak_to_limit=float(chk.peak_ratio))
    if dual is not None:
        info["dual_bound_mw"] = dual.get("upper_bound_mw")
        info["dual_gap"] = dual.get("gap")
    return PlanResult(
        status, p_abs, u, float(snap.p_receiver) if ok else 0.0, float(thermal["power_h_w"]) if ok else 0.0,
        thermal["cv_2d"] if ok else None, None, _circ_spread(c, snap) if ok else 0.0,
        p_abs - settings.lam_aps * u,
        upper * 1e6 if upper is not None else None, cert.get("gap"), cert.get("method"),
        bool(cert.get("global_optimum_proven_on_candidate_set")),
        float(chk.peak_flux_kw_m2) if chk is not None else 0.0,
        float(chk.peak_ratio / max(chk.allowed_ratio, 1e-12)) if chk is not None else 0.0,
        chk.status if chk is not None else "unverified", int(res.active.sum()), int((~res.active).sum()), dni_v,
        np.asarray(res.aim[0], float), np.asarray(res.aim[1], float), np.asarray(res.active, bool),
        np.asarray(mirror, float), snap.flux if keep_flux else None, _time.perf_counter() - t0, info,
        np.asarray(res.choice, int))


# names of the annual numbers by the basis of the scenario set (README 0.9.0): on the clear-sky design baseline V and
# E are the "clear-sky annual score" and "clear-sky annual absorbed heat"
VALUE_LABELS = {
    "clear_sky_design": {"value": "晴空年度综合得分", "energy": "晴空年度吸热量", "penalty": "晴空年度均匀性扣分",
                         "value_en": "clear-sky annual score", "energy_en": "clear-sky annual absorbed heat",
                         "basis": "晴空年度：代表日逐时晴空 DNI，全场同一 DNI"},
    None: {"value": "年度价值", "energy": "全年吸收热量", "penalty": "均匀性扣分",
           "value_en": "annual value", "energy_en": "annual absorbed heat", "basis": "气象年代表工况"},
}


def value_labels(basis: str | None) -> dict:
    return VALUE_LABELS.get(basis, VALUE_LABELS[None])


@dataclass
class ScenarioRecord:
    scenario_id: str
    map_id: str
    hours: float
    elevation_deg: float
    azimuth_deg: float
    dni_w_m2: float  # scenario DNI (site mean for normalised factor maps)
    plan: PlanResult
    actual_hours: float | None = None  # real time the scenario stands for (h); hours is the weight of V
    map_digest: str | None = None  # content hash of the DNI map (identifies the exact map used)
    # control variate (``control_variate``): cheap-model energy at the representative state (h_s C(s)) and summed
    # over the member hours of the scenario; their ratio corrects the scenario's contribution to the annual value
    cheap_rep_mwh: float | None = None
    cheap_members_mwh: float | None = None
    cv_factor: float | None = None

    def row(self, lam_aps: float, lam_l: float) -> dict:
        p = self.plan
        counts = p.counts
        dni = p.dni
        return {
            "scenario_id": self.scenario_id, "map_id": self.map_id, "hours": self.hours,
            "elevation_deg": self.elevation_deg, "azimuth_deg": self.azimuth_deg, "dni_w_m2": self.dni_w_m2,
            "dni_mirror_mean_w_m2": float(dni.mean()) if dni.size else 0.0,
            "dni_mirror_min_w_m2": float(dni.min()) if dni.size else 0.0,
            "dni_mirror_max_w_m2": float(dni.max()) if dni.size else 0.0,
            "status": p.status, "counts_in_value": counts,
            "p_abs_mw": p.p_abs_w / 1e6, "u_mw": p.u_w / 1e6,
            "contribution_mwh": self.hours * (p.p_abs_w - lam_l * p.u_w) / 1e6 if counts else 0.0,
            "energy_mwh": self.hours * p.p_abs_w / 1e6 if counts else 0.0,
            "penalty_mwh": self.hours * p.u_w / 1e6 if counts else 0.0,
            "lambda_aps": lam_aps, "lambda_layout": lam_l,
            "j_inner_mw": p.objective_w / 1e6, "upper_bound_mw": p.upper_bound_w / 1e6 if p.upper_bound_w is not None else None,
            "inner_gap": p.gap, "bound_method": p.bound_method, "proven_optimal": p.proven_optimal,
            "p_intercepted_mw": p.p_int_w / 1e6, "p_region_mw": p.p_h_w / 1e6, "cv_2d": p.cv_2d,
            "circ_u_mw": p.circ_u_w / 1e6, "peak_flux_kw_m2": p.peak_flux_kw_m2, "peak_to_allowed": p.peak_ratio,
            "flux_status": p.flux_status, "n_active": p.n_active, "n_defocused": p.n_defocused,
            "peak_to_limit": p.info.get("peak_to_limit"), "hotspot_panel": p.info.get("hotspot_panel"),
            "hotspot_azimuth_deg": p.info.get("hotspot_azimuth_deg"), "hotspot_z_m": p.info.get("hotspot_z_m"),
            "dni_map_digest": self.map_digest, "actual_hours": self.actual_hours,
            "cheap_rep_mwh": self.cheap_rep_mwh, "cheap_members_mwh": self.cheap_members_mwh,
            "cv_factor": self.cv_factor,
            "solve_time_s": round(p.solve_time_s, 3),
            "additional_joint_trials": p.info.get("additional_joint_trials", 0),
        }


def annual_value(table: pd.DataFrame, lam_l: float) -> dict:
    """V, E, Pi (MWh) from a per-scenario table (columns hours, p_abs_mw, u_mw, status) -- the scoring rule."""
    if not len(table):
        return {"value_mwh": 0.0, "energy_mwh": 0.0, "penalty_mwh": 0.0, "complete": True}
    counts = table["status"].isin(COUNTED).to_numpy()
    h = table["hours"].to_numpy(float)
    p = np.where(counts, table["p_abs_mw"].to_numpy(float), 0.0)
    u = np.where(counts, table["u_mw"].to_numpy(float), 0.0)
    e = float(h @ p)
    pen = float(h @ u)
    solved = ~(table["status"] == "not_solved").to_numpy()
    complete = bool(solved.all())
    out = {"value_mwh": e - lam_l * pen if complete else float("nan"), "energy_mwh": e, "penalty_mwh": pen,
           "complete": complete, "value_partial_mwh": e - lam_l * pen,
           "n_scenarios": int(len(table)), "n_solved": int(solved.sum()),
           "coverage": float(solved.mean()), "hours_coverage": float(h[solved].sum() / h.sum()) if h.sum() > 0 else 1.0}
    if "cv_factor" in table and table["cv_factor"].notna().all():
        f = table["cv_factor"].to_numpy(float)
        e_cv, pen_cv = float(h @ (p * f)), float(h @ (u * f))
        out.update(value_cv_mwh=e_cv - lam_l * pen_cv if complete else float("nan"), energy_cv_mwh=e_cv,
                   penalty_cv_mwh=pen_cv)
    return out


@dataclass
class LayoutValue:
    name: str
    xyz: np.ndarray
    ids: np.ndarray
    records: list[ScenarioRecord]
    lam_aps: float
    lam_layout: float
    scenario_fingerprint: str
    settings: InnerSettings
    runtime_s: float = 0.0
    tags: dict = dc_field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.xyz))

    def table(self, lam_l: float | None = None) -> pd.DataFrame:
        lam = self.lam_layout if lam_l is None else float(lam_l)
        return pd.DataFrame([r.row(self.lam_aps, lam) for r in self.records])

    def totals(self, lam_l: float | None = None) -> dict:
        lam = self.lam_layout if lam_l is None else float(lam_l)
        t = self.table(lam)
        v = annual_value(t, lam)
        status = t["status"].value_counts().to_dict() if len(t) else {}
        hours = t.groupby("status")["hours"].sum().to_dict() if len(t) else {}
        fe = t[t["counts_in_value"]] if len(t) else t
        jj = fe["j_inner_mw"].to_numpy(float) if len(fe) else np.zeros(0)
        hh = fe["hours"].to_numpy(float) if len(fe) else np.zeros(0)
        inner_j = float(hh @ jj)
        # upper bound on sum_s h_s max_a J_s (this layout, this candidate model): every scenario with a bound adds
        # h max(UB, 0); proven infeasible and sunless scenarios add 0; an unsolved scenario leaves it open
        bounded = t["status"].isin(("feasible", "feasible_shutdown", "no_feasible_found")) if len(t) else t
        ub_all = (t.loc[bounded, "upper_bound_mw"].astype(float).clip(lower=0.0) if len(t) else pd.Series(dtype=float))
        inner_ub = (float((t.loc[bounded, "hours"] * ub_all).sum()) if len(t) and ub_all.notna().all()
                    else float("nan"))
        if len(t) and (t["status"] == "not_solved").any():
            inner_ub = float("nan")
        e_circ = float((t["hours"] * t["circ_u_mw"] * t["counts_in_value"]).sum()) if len(t) else 0.0
        energy = v["energy_mwh"]
        cv_w = (float((t["hours"] * t["p_abs_mw"] * t["cv_2d"].fillna(0)).sum() / max((t["hours"] * t["p_abs_mw"]).sum(), 1e-12))
                if len(t) else None)
        return {
            "name": self.name, "n_heliostats": self.n, "value_mwh": v["value_mwh"], "energy_mwh": energy,
            "penalty_mwh": v["penalty_mwh"], "weighted_penalty_mwh": lam * v["penalty_mwh"],
            "complete": v["complete"], "lambda_aps": self.lam_aps, "lambda_layout": lam,
            "value_partial_mwh": v["value_partial_mwh"], "n_scenarios": v["n_scenarios"], "n_solved": v["n_solved"],
            "coverage": v["coverage"], "hours_coverage": v["hours_coverage"],
            "ranking_status": "ranked" if v["complete"] else "pending",
            "status_counts": status, "status_hours": {k: round(float(x), 3) for k, x in hours.items()},
            "inner_objective_mwh": inner_j, "inner_upper_bound_mwh": inner_ub,
            "inner_gap_annual": (1.0 - inner_j / inner_ub) if inner_ub > 0 else None,
            # with equal weights V is the hour sum of J_s, so [V, sum h UB_s] brackets the layout's best value
            "value_interval_mwh": ([inner_j, inner_ub] if abs(lam - self.lam_aps) < 1e-12 and np.isfinite(inner_ub)
                                   and not status.get("no_feasible_found") else None),
            "max_inner_gap": float(t["inner_gap"].dropna().max()) if len(t) and t["inner_gap"].notna().any() else None,
            "circ_penalty_mwh": e_circ, "cv_2d_power_weighted": cv_w,
            "value_cv_mwh": v.get("value_cv_mwh"), "energy_cv_mwh": v.get("energy_cv_mwh"),
            "penalty_cv_mwh": v.get("penalty_cv_mwh"),
            "cv_correction_pct": (100.0 * (v["value_cv_mwh"] / v["value_mwh"] - 1.0)
                                  if v.get("value_cv_mwh") is not None and v["complete"] and v["value_mwh"] else None),
            "scenario_fingerprint": self.scenario_fingerprint, "settings_digest": self.settings.digest(),
            "runtime_s": round(self.runtime_s, 2), **self.tags,
        }

    def plan(self, scenario_id: str) -> PlanResult:
        for r in self.records:
            if r.scenario_id == scenario_id:
                return r.plan
        raise KeyError(scenario_id)

    def mirror_energy_mwh(self) -> np.ndarray:
        """(N,) annual absorbed energy of each heliostat under its scenario plans (counted scenarios)."""
        out = np.zeros(self.n)
        for r in self.records:
            if r.plan.counts:
                out += r.hours * r.plan.mirror_power_w / 1e6
        return out

    def export(self, folder, with_aims: bool = True) -> dict:
        """Per-scenario CSV, per-scenario aim plans and a JSON summary: enough to recompute V by hand."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        t = self.table()
        if with_aims:
            aim_dir = folder / "aims"
            aim_dir.mkdir(exist_ok=True)
            names = []
            for r in self.records:
                p = r.plan
                path = aim_dir / f"{safe_filename(r.scenario_id)}.csv"
                pd.DataFrame({"id": self.ids, "x": self.xyz[:, 0], "y": self.xyz[:, 1], "z": self.xyz[:, 2],
                              "dni_w_m2": np.round(p.dni, 4), "active": p.active.astype(int),
                              "aim_azimuth_deg": np.round(p.aim_azimuth, 6), "aim_z": np.round(p.aim_z, 6),
                              "absorbed_kw": np.round(p.mirror_power_w / 1e3, 5)}).to_csv(path, index=False)
                names.append(f"aims/{path.name}")
            t["aims_file"] = names
        files["scenario_csv"] = folder / "scenario_values.csv"
        t.to_csv(files["scenario_csv"], index=False)
        files["layout_csv"] = folder / "layout.csv"
        pd.DataFrame({"id": self.ids, "x": self.xyz[:, 0], "y": self.xyz[:, 1], "z": self.xyz[:, 2]}).to_csv(
            files["layout_csv"], index=False, float_format="%.17g")
        files["summary_json"] = folder / "layout_value.json"
        summary = {**self.totals(), "settings": self.settings.record(), "status_rules": STATUS_RULES,
                   "formula": "V = sum_s hours * (p_abs_mw - lambda_layout * u_mw) over rows with counts_in_value"}
        files["summary_json"].write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        return files


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def recompute_from_csv(path, lam_l: float | None = None) -> dict:
    """V, E, Pi from an exported ``scenario_values.csv`` alone (the audit path)."""
    t = pd.read_csv(path)
    lam = float(t["lambda_layout"].iloc[0]) if lam_l is None and len(t) else float(lam_l or 0.0)
    return {**annual_value(t, lam), "lambda_layout": lam}


def control_variate(case: Case, value: "LayoutValue", sset, workers: int | None = None) -> bool:
    """Attach the control-variate factors of every scenario to ``value`` (False when the set has no hours).

    Cheap model C: centre aiming without flux maps (``simulate.annual``, every mirror reads its own DNI from
    the hour's map). For scenario s with member hours t (weights w_t):

        f_s = sum_t w_t C(t) / (h_s C(s))          C(s) = the cheap model at the representative state

    and the annual value is estimated as V_cv = sum_s h_s J_s f_s (likewise E and Pi). The representative
    state's error in the expensive model is assumed proportional to its error in the cheap model (separate
    ratio estimator per scenario); f_s = 1 when the cheap model is linear over the members. The formal V is
    unchanged; ``layout_opt.rank_by`` decides which one ranks."""
    from .simulate import annual

    if sset.hourly is None or sset.assignment is None or len(value.records) != len(sset):
        return False
    a = np.asarray(sset.assignment)
    keep = np.flatnonzero(a >= 0)
    if not keep.size:
        return False
    fld = Field(value.xyz, value.ids)
    nw = workers if workers is not None else case.layout_opt.workers
    hours = sset.hourly.subset(keep)
    res_h = annual(case, fld, samples=hours, aiming="center", with_flux=False, workers=nw)
    members = np.bincount(a[keep], weights=res_h.timeseries["p_absorbed_mw"].to_numpy(float) * hours.weight_h,
                          minlength=len(sset))
    res_r = annual(case, fld, samples=sset.samples, aiming="center", with_flux=False, workers=nw)
    rep = res_r.timeseries["p_absorbed_mw"].to_numpy(float) * sset.samples.weight_h
    for i, r in enumerate(value.records):
        f = float(members[i] / rep[i]) if rep[i] > 1e-9 else 1.0
        r.cheap_rep_mwh, r.cheap_members_mwh = float(rep[i]), float(members[i])
        r.cv_factor = float(np.clip(f, 0.5, 2.0))
    value.tags["control_variate"] = {"model": "centre aim, no flux map", "hours": int(keep.size),
                                     "hourly_fingerprint": sset.hourly_fingerprint()}
    return True


def _price_chunk(args):
    case, xyz, ids, cand, sset, chain, choices, settings, lam_l, max_iter = args
    from .aps import build_response, make_problem
    from .certify import aps_upper_bound, removal_losses

    c = inner_case(case, settings)
    grid = build_grid(c.receiver, settings.grid_shape[1], settings.grid_shape[0])
    n, m = len(xyz), len(cand)
    both = Field(np.vstack([xyz, cand]))
    layout = np.r_[np.ones(n, bool), np.zeros(m, bool)]
    s = sset.samples
    out, key_rs, rs = [], None, None
    for i in chain:
        el, az = float(s.elevation[i]), float(s.azimuth[i])
        key = _direction(sset, i)
        if key != key_rs:  # DNI-independent: one response set per sun direction
            rs, key_rs = build_response(c, both, el, az, grid, c.aps, blockers=layout), key
        dni = sset.dni_at(i, both.xyz[:, :2])
        prob = make_problem(rs, dni, margin=c.aps.safety_margin)
        choice = np.r_[np.asarray(choices[i], int), np.full(m, -1)]
        b = aps_upper_bound(prob, choice, "weighted", lam_l, None, settings.min_load_mw * 1e6, max_iter=max_iter,
                            time_limit_s=1e9, price_mask=~layout)
        loss = removal_losses(prob, choice, lam_l)
        found = b["objective_found_mw"]
        out.append((i, b["priced_gain_w"][n:], loss[:n], b["upper_bound_mw"] * 1e6,
                    None if found is None else found * 1e6))
    return out


def dual_prices(case: Case, value: "LayoutValue", cand_xyz: np.ndarray, sset, settings: InnerSettings | None = None,
                lam_layout: float | None = None, workers: int | None = None, max_iter: int = 150,
                progress: Callable[[str], None] | None = None) -> dict:
    """Flux-price (Lagrangian dual) estimate of adding each candidate position / removing each mirror (improvement 4).

    For every counted scenario s the upper bound of task 1 (``certify.aps_upper_bound``, objective
    P - lambda_L U in the candidate model, the layout's plan as the incumbent) yields multipliers: nu per flux
    constraint point (the "congestion price" of that receiver cell) and the linearised uniformity weights.
    A candidate position j -- shaded and blocked by the layout's mirrors, aimed at any of its candidate aim
    points -- is priced by its reduced cost G_js = max(0, max_k [c_jk - DNI_j R_jk . (w_o + nu / lim)]): its own
    power minus the price of the flux it would add. Annual gain_j = sum_s h_s G_js. Weak duality gives, per
    scenario, J_s(L + j) <= UB_s + G_js (j not shading / blocking the layout), so
    gain_j + gap bounds what adding j alone can add (gap = sum_s h_s (UB_s - J_s(plan))).
    loss_i = sum_s h_s (J_s(plan) - J_s(plan without i)): what removing mirror i costs with the other aims kept
    (exact in the model, no credit for neighbours it shaded). Returns MWh arrays; a swap i -> j is promising
    when gain_j - loss_i > 0 (it is accepted only on the re-solved V).
    """
    from .parallel import n_workers, pool

    t0 = _time.perf_counter()
    settings = settings or value.settings
    lam_l = float(value.lam_layout if lam_layout is None else lam_layout)
    cand = np.asarray(cand_xyz, dtype=float).reshape(-1, 3)
    n, m = value.n, len(cand)
    gain, loss = np.zeros(m), np.zeros(n)
    gap, n_used = 0.0, 0
    if m == 0 or len(value.records) != len(sset):
        return {"gain_mwh": gain, "loss_mwh": loss, "gap_mwh": 0.0, "scenarios": 0, "time_s": 0.0}
    use = [i for i, r in enumerate(value.records)
           if r.plan.counts and r.plan.choice is not None and r.elevation_deg > 0]
    choices = {i: value.records[i].plan.choice for i in use}
    groups: dict = {}
    for i in use:
        groups.setdefault(_direction(sset, i), []).append(i)
    chains = sorted(groups.values(), key=len, reverse=True)
    nw = min(n_workers(case.layout_opt.workers if workers is None else workers), max(len(chains), 1))
    tasks = [(case, value.xyz, value.ids, cand, sset, ch, {i: choices[i] for i in ch}, settings, lam_l, max_iter)
             for ch in chains]
    results = []
    if nw > 1 and len(tasks) > 1:
        p = pool(nw, probe=(_price_chunk, tasks[0]))
        try:
            for k, part in enumerate(p.imap_unordered(_price_chunk, tasks), start=1):
                results += part
                if progress:
                    progress(f"flux prices: {k}/{len(tasks)} sun directions")
        finally:
            p.close()
            p.join()
    else:
        for task in tasks:
            results += _price_chunk(task)
    for i, g, d, ub, found in results:
        h = value.records[i].hours
        gain += h * g / 1e6
        loss += h * d / 1e6
        if found is not None:
            gap += h * max(ub - found, 0.0) / 1e6
        n_used += 1
    return {"gain_mwh": gain, "loss_mwh": loss, "gap_mwh": float(gap), "scenarios": n_used,
            "lambda_layout": lam_l, "time_s": _time.perf_counter() - t0}


def price_swaps(prices: dict, m: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Up to ``m`` (drop, add) pairs by predicted gain: best-priced candidates against the cheapest mirrors to
    lose, kept while gain_j - loss_i > 0. Returns (drop indices, add indices, predicted change MWh)."""
    order_add = np.argsort(-prices["gain_mwh"], kind="stable")
    order_drop = np.argsort(prices["loss_mwh"], kind="stable")
    k = 0
    while (k < min(m, len(order_add), len(order_drop))
           and prices["gain_mwh"][order_add[k]] - prices["loss_mwh"][order_drop[k]] > 0):
        k += 1
    pred = float(prices["gain_mwh"][order_add[:k]].sum() - prices["loss_mwh"][order_drop[:k]].sum())
    return order_drop[:k], order_add[:k], pred


# --------------------------------------------------------------------------- whole layout

def _direction(sset, i: int) -> tuple[float, float]:
    s = sset.samples
    return round(float(s.elevation[i]), 9), round(float(s.azimuth[i]), 9)


FULL_REUSE_CHAINS = 8  # warm-start chains of reuse "full": fixed, so plans never depend on the number of cores


def scenario_chains(sset, settings: InnerSettings) -> list[list[int]]:
    """How the scenarios are solved: lists solved in order by one worker each.

    Scenarios of one sun direction stay together (their spot responses are built once); inside a direction
    the same map comes together, highest DNI first (a plan the limits never shaped is reused for the lower
    DNI levels). With ``reuse = "full"`` the directions are ordered along the sun path and cut into
    ``FULL_REUSE_CHAINS`` contiguous chains of about equal weight, each warm-starting from its previous
    scenario. The chains depend on the scenario set only, so the plans are the same on any computer."""
    s = sset.samples
    groups: dict = {}
    for i in range(len(sset)):
        groups.setdefault(_direction(sset, i), []).append(i)
    for key in groups:
        groups[key].sort(key=lambda i: (int(s.map_index[i]) if s.map_index is not None else 0, -float(s.dni[i])))
    weight = {k: float(sum(s.weight_h[i] * s.dni[i] for i in v)) for k, v in groups.items()}
    if settings.reuse != "full":
        return [groups[k] for k in sorted(groups, key=lambda k: -weight[k])]  # big first
    keys = sorted(groups, key=lambda k: (round(k[0] / 10.0), k[1]))  # bands of 10 deg elevation, then azimuth
    n_chains = max(1, min(FULL_REUSE_CHAINS, len(keys)))
    total = sum(weight.values()) or 1.0
    chains, cur, acc = [], [], 0.0
    for k in keys:
        cur += groups[k]
        acc += weight[k]
        if acc >= total * (len(chains) + 1) / n_chains and len(chains) < n_chains - 1:
            chains.append(cur)
            cur = []
    if cur:
        chains.append(cur)
    return chains


def _rescaled_plan(case: Case, fld: Field, el: float, az: float, dni: np.ndarray, src: PlanResult, factor: float,
                   settings: InnerSettings, grid, keep_flux: bool, source_id: str) -> PlanResult:
    """``src`` (solved where the limits never acted) at a DNI scaled by ``factor`` <= 1: same aims."""
    from .simulate import evaluate

    t0 = _time.perf_counter()
    c = inner_case(case, settings)
    aim = (src.aim_azimuth, src.aim_z)
    snap = evaluate(c, fld, el, az, dni, aim=aim, active=src.active, grid=grid)
    thermal = snap.thermal_objective(c.uniformity)
    chk = snap.verify(c.receiver)
    ok = src.status == "feasible"
    p_abs = float(thermal["power_w"]) if ok else 0.0
    u = float(thermal["spread_w"]) if ok else 0.0
    alpha = snap.alpha_eff if snap.alpha_eff is not None else np.full(fld.n, c.receiver.absorptivity)
    info = dict(src.info)
    info.update(reused_from=source_id, dni_factor=float(factor), raw_power_w=float(thermal["power_w"]),
                raw_spread_w=float(thermal["spread_w"]), hotspot_panel=int(chk.hotspot_panel),
                hotspot_azimuth_deg=float(chk.hotspot_azimuth_deg), hotspot_z_m=float(chk.hotspot_z),
                peak_to_limit=float(chk.peak_ratio))
    for k in ("dual_bound_mw",):
        if info.get(k) is not None:
            info[k] = info[k] * factor
    return replace(src, p_abs_w=p_abs, u_w=u, p_int_w=float(snap.p_receiver) if ok else 0.0,
                   p_h_w=float(thermal["power_h_w"]) if ok else 0.0, cv_2d=thermal["cv_2d"] if ok else None,
                   circ_u_w=_circ_spread(c, snap) if ok else 0.0, objective_w=p_abs - settings.lam_aps * u,
                   upper_bound_w=src.upper_bound_w * factor if src.upper_bound_w is not None else None,
                   peak_flux_kw_m2=float(chk.peak_flux_kw_m2),
                   peak_ratio=float(chk.peak_ratio / max(chk.allowed_ratio, 1e-12)), flux_status=chk.status,
                   dni=np.asarray(dni, float).copy(), mirror_power_w=np.asarray(snap.power["intercept"] * alpha, float),
                   flux=snap.flux if keep_flux else None, solve_time_s=_time.perf_counter() - t0, info=info)


def solve_chain(case: Case, fld: Field, sset, chain: list[int], settings: InnerSettings, grid=None,
                keep_flux: bool = True) -> list[tuple[int, "ScenarioRecord"]]:
    """Solve the scenarios of ``chain`` in order, reusing what ``settings.reuse`` allows (see InnerSettings)."""
    grid = grid or build_grid(case.receiver, settings.grid_shape[1], settings.grid_shape[0])
    s = sset.samples
    out = []
    key_rs, rs = None, None
    first: dict = {}  # (direction, map) -> (scenario index, plan) of its highest DNI
    prev = None
    for i in chain:
        el, az = float(s.elevation[i]), float(s.azimuth[i])
        key = _direction(sset, i)
        if settings.reuse in ("responses", "full") and key != key_rs and el > 0:
            rs, key_rs = inner_response(case, fld, el, az, settings, grid), key
        dni = sset.dni_at(i, fld.xyz[:, :2])
        plan = None
        g = (key, sset.map_id(i))
        if settings.reuse == "full" and g in first:
            j, src = first[g]
            pos = src.dni > 0
            ratio = dni[pos] / src.dni[pos] if pos.any() else np.zeros(0)
            if (ratio.size and np.allclose(ratio, ratio[0], rtol=1e-9, atol=0.0) and ratio[0] <= 1.0 + 1e-12
                    and np.all(dni[~pos] == 0) and src.status == "feasible" and not src.info.get("limits_active", True)):
                plan = _rescaled_plan(case, fld, el, az, dni, src, float(ratio[0]), settings, grid, keep_flux,
                                      sset.ids[j])
        if plan is None:
            warm = prev.choice if settings.reuse == "full" and prev is not None and prev.choice is not None else None
            plan = solve_instant(case, fld, el, az, dni, settings, grid, keep_flux, warm=warm,
                                 response=rs if settings.reuse in ("responses", "full") and el > 0 else None)
        first.setdefault(g, (i, plan))
        prev = plan
        out.append((i, ScenarioRecord(sset.ids[i], sset.map_id(i), float(s.weight_h[i]), el, az, float(s.dni[i]),
                                      plan, float(s.actual_hours[i]), s.map_of(i).digest())))
    return out


def _scenario_chunk(args):
    case, xyz, ids, sset, chain, settings, keep_flux = args
    return solve_chain(case, Field(xyz, ids), sset, chain, settings, None, keep_flux)


def solve_scenario(case: Case, fld: Field, sset, i: int, settings: InnerSettings | None = None,
                   grid: ReceiverGrid | None = None, keep_flux: bool = True) -> ScenarioRecord:
    s = sset.samples
    plan = solve_instant(case, fld, float(s.elevation[i]), float(s.azimuth[i]), sset.dni_at(i, fld.xyz[:, :2]),
                         settings, grid, keep_flux)
    return ScenarioRecord(sset.ids[i], sset.map_id(i), float(s.weight_h[i]), float(s.elevation[i]),
                          float(s.azimuth[i]), float(s.dni[i]), plan, float(s.actual_hours[i]),
                          sset.samples.map_of(i).digest())


def evaluate_layout_value(case: Case, xyz: np.ndarray, sset, name: str = "layout", ids=None,
                          settings: InnerSettings | None = None, lam_layout: float | None = None,
                          workers: int | None = None, keep_flux: bool = True,
                          progress: Callable[[str], None] | None = None,
                          cancel: Callable[[], bool] | None = None, tags: dict | None = None) -> LayoutValue:
    """Task 1 for every scenario of ``sset`` on this layout; the per-scenario records and V(L)."""
    from .parallel import n_workers, pool

    t0 = _time.perf_counter()
    settings = settings or inner_settings(case)
    lam_l = float(case.layout_opt.uniformity_weight if lam_layout is None else lam_layout)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    ids = np.arange(1, len(xyz) + 1) if ids is None else np.asarray(ids)
    n_s = len(sset)
    nw = min(n_workers(case.layout_opt.workers if workers is None else workers), max(n_s, 1))
    results: list = [None] * n_s
    stop = cancel or (lambda: False)
    if stop():  # cancelled before this layout started: every scenario is reported as not solved
        n_s_run = 0
    else:
        n_s_run = n_s
    chains = scenario_chains(sset, settings)
    if nw > 1 and n_s_run > 1 and len(chains) > 1:
        tasks = [(case, xyz, ids, sset, chain, settings, keep_flux) for chain in chains]
        p = pool(min(nw, len(tasks)), probe=(_scenario_chunk, tasks[0]))
        try:
            done = 0
            for out in p.imap_unordered(_scenario_chunk, tasks):
                for i, rec in out:
                    results[i] = rec
                    done += 1
                if progress:
                    progress(f"APS scenarios {name}: {done}/{n_s}")
                if stop():
                    p.terminate()  # keep what is finished; the rest is reported as not solved
                    break
        finally:
            p.close()
            p.join()
    elif n_s_run:
        fld = Field(xyz, ids)
        grid = build_grid(case.receiver, settings.grid_shape[1], settings.grid_shape[0])
        done = 0
        for chain in chains:
            if stop():
                break
            for i, rec in solve_chain(case, fld, sset, chain, settings, grid, keep_flux):
                results[i] = rec
                done += 1
            if progress:
                progress(f"APS scenarios {name}: {done}/{n_s}")
    return _finish_value(case, name, xyz, ids, results, sset, settings, lam_l, t0, tags, stop, progress, workers)


def _finish_value(case, name, xyz, ids, results, sset, settings, lam_l, t0, tags, stop, progress, workers):
    """LayoutValue from the solved records (unsolved ones reported as such) + its control variate."""
    s = sset.samples
    for i in range(len(sset)):
        if results[i] is None:  # cancelled
            n = len(xyz)
            z = np.zeros(n)
            plan = PlanResult("not_solved", 0.0, 0.0, 0.0, 0.0, None, None, 0.0, 0.0, None, None, None, False, 0.0,
                              0.0, "unverified", 0, n, z, z, z, np.zeros(n, bool), z, None, 0.0)
            results[i] = ScenarioRecord(sset.ids[i], sset.map_id(i), float(s.weight_h[i]), float(s.elevation[i]),
                                        float(s.azimuth[i]), float(s.dni[i]), plan, float(s.actual_hours[i]),
                                        sset.samples.map_of(i).digest())
    from .backend import backend_name

    basis = sset.meta.get("basis")
    labels = value_labels(basis)
    out = LayoutValue(name, xyz, ids, results, settings.lam_aps, lam_l, sset.fingerprint(), settings,
                      _time.perf_counter() - t0, {**(tags or {}), "backend": backend_name(), "basis": basis,
                                                  "value_label": labels["value"], "energy_label": labels["energy"]})
    if case.layout_opt.control_variate and not stop() and all(r.plan.status != "not_solved" for r in results):
        if progress:
            progress(f"control variate {name}: centre-aim energy of the member hours")
        t1 = _time.perf_counter()
        if control_variate(case, out, sset, workers):
            out.runtime_s += _time.perf_counter() - t1
    return out


def _layout_chunk(args):
    case, xyz, ids, sset, chain, settings, keep_flux, name = args
    return name, solve_chain(case, Field(xyz, ids), sset, chain, settings, None, keep_flux)


def evaluate_layout_values(case: Case, layouts: dict, sset, settings: InnerSettings | None = None,
                           lam_layout: float | None = None, workers: int | None = None, keep_flux: bool = True,
                           progress: Callable[[str], None] | None = None, cancel: Callable[[], bool] | None = None,
                           tags: dict | None = None) -> dict[str, LayoutValue]:
    """Several layouts on one scenario set with one pool: every (layout, scenario chain) is one task, so all
    cores stay busy even when there are fewer scenarios than cores. ``layouts``: name -> (xyz, ids or None);
    ``tags``: name -> dict. Each value equals ``evaluate_layout_value`` of that layout."""
    from .parallel import n_workers, pool

    settings = settings or inner_settings(case)
    lam_l = float(case.layout_opt.uniformity_weight if lam_layout is None else lam_layout)
    nw = n_workers(case.layout_opt.workers if workers is None else workers)
    stop = cancel or (lambda: False)
    names = list(layouts)
    tags = tags or {}
    norm = {}
    for n in names:
        xyz, ids = layouts[n]
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        norm[n] = (xyz, np.arange(1, len(xyz) + 1) if ids is None else np.asarray(ids))
    if nw <= 1 or len(names) <= 1:
        return {n: evaluate_layout_value(case, norm[n][0], sset, n, ids=norm[n][1], settings=settings, lam_layout=lam_l,
                                         workers=nw, keep_flux=keep_flux, progress=progress, cancel=cancel,
                                         tags=tags.get(n)) for n in names}
    t0 = _time.perf_counter()
    chains = scenario_chains(sset, settings)
    tasks = [(case, norm[n][0], norm[n][1], sset, chain, settings, keep_flux, n) for n in names for chain in chains]
    results = {n: [None] * len(sset) for n in names}
    if not stop():
        p = pool(min(nw, len(tasks)), probe=(_layout_chunk, tasks[0]))
        try:
            done = {n: 0 for n in names}
            for name, out in p.imap_unordered(_layout_chunk, tasks):
                for i, rec in out:
                    results[name][i] = rec
                    done[name] += 1
                if progress:
                    progress(f"APS scenarios {name}: {done[name]}/{len(sset)} "
                             f"(all layouts {sum(done.values())}/{len(sset) * len(names)})")
                if stop():
                    p.terminate()
                    break
        finally:
            p.close()
            p.join()
    return {n: _finish_value(case, n, norm[n][0], norm[n][1], results[n], sset, settings, lam_l, t0, tags.get(n),
                             stop, progress, nw) for n in names}


def compare_values(values: list[LayoutValue], lam_l: float | None = None, reference: str | None = None,
                   rank_by: str = "value") -> pd.DataFrame:
    """Ranking by V (or by its control-variate estimate, ``rank_by="value_cv"``) with the decomposition of
    every difference into heat and uniformity."""
    if not values:
        return pd.DataFrame()
    fps = {v.scenario_fingerprint for v in values}
    digs = {v.settings.digest() for v in values}
    if len({v.n for v in values}) > 1:
        raise ValueError("比较布局的镜数必须相同")
    if len(fps) > 1 or len(digs) > 1:
        raise ValueError("layouts must be compared on the same scenario set and inner settings")
    lam = values[0].lam_layout if lam_l is None else float(lam_l)
    if not 0 <= lam < math.inf:
        raise ValueError("The uniformity weight must be finite and nonnegative")
    rows = [v.totals(lam) for v in values]
    df = pd.DataFrame(rows)
    df["rankable"] = df["complete"]
    key = "value_mwh"
    if rank_by == "value_cv" and "value_cv_mwh" in df and df["value_cv_mwh"].notna().all():
        key = "value_cv_mwh"
    df["ranked_by"] = key
    df = df.sort_values(["rankable", key], ascending=[False, False], kind="stable").reset_index(drop=True)
    df["rank"] = np.where(df["rankable"], np.arange(1, len(df) + 1), np.nan)
    ref_name = reference if reference in set(df["name"]) else df.loc[0, "name"]
    ref = df.loc[df["name"] == ref_name].iloc[0]
    df["delta_value_mwh"] = df["value_mwh"] - ref["value_mwh"]
    df["delta_energy_mwh"] = df["energy_mwh"] - ref["energy_mwh"]
    df["delta_uniformity_mwh"] = -lam * (df["penalty_mwh"] - ref["penalty_mwh"])
    df["reference"] = ref_name
    # bound intervals (equal weights): a layout whose upper end reaches the leader's found value could still win
    # with a larger inner budget -- these get more budget before the ranking is called final
    lead_lo = df.loc[0, "value_mwh"] if bool(df.loc[0, "rankable"]) else float("nan")

    def overlap(row):
        iv = row["value_interval_mwh"]
        if row.name == 0 or iv is None or not np.isfinite(lead_lo):
            return None
        return bool(iv[1] >= lead_lo)

    df["interval_overlaps_leader"] = df.apply(overlap, axis=1)
    intervals = df["value_interval_mwh"].tolist()
    if not df["complete"].all():
        decision = "incomplete"
    elif len(df) == 1:
        decision = "single_layout"
    elif any(iv is None for iv in intervals) or any(
            r.plan.status == "no_feasible_found" for v in values for r in v.records):
        decision = "feasible_scores_only"
    elif all(abs(iv[1] - iv[0]) < 1e-10 for iv in intervals) and abs(intervals[1][0] - intervals[0][0]) < 1e-10:
        decision = "certified_tie"
    elif not np.isfinite(intervals).all():
        decision = "feasible_scores_only"
    elif intervals[0][0] > max(iv[1] for iv in intervals[1:]) + 1e-10:
        decision = "certified_leader"
    else:
        decision = "overlapping_bounds"
    df["comparison_status"] = decision
    return df


def refine_competing_plans(case, values, sset, progress=None, cancel=None):
    """Spend the same extra state budget per contender, ordered by hours times unresolved power."""
    budget = case.layout_opt.ranking_refine_states
    if not budget or len(values) < 2:
        return
    rank = compare_values(list(values.values()))
    if rank.iloc[0]["comparison_status"] != "overlapping_bounds":
        return
    lead = rank.iloc[0]["value_mwh"]
    for value in values.values():
        interval = value.totals()["value_interval_mwh"]
        if interval is None or interval[1] < lead:
            continue
        candidates = [r for r in value.records if r.plan.upper_bound_w is not None and
                      not r.plan.proven_optimal and r.hours > 0 and
                      r.plan.status in ("feasible", "feasible_shutdown", "no_feasible_found")]
        candidates.sort(key=lambda r: r.hours * (r.plan.upper_bound_w - r.plan.objective_w), reverse=True)
        field = Field(value.xyz, value.ids)
        settings = replace(value.settings, joint_trials=max(8, 2 * value.settings.joint_trials),
                           certify_rounds=max(2, value.settings.certify_rounds))
        for row in candidates[:budget]:
            if cancel and cancel():
                return
            if progress:
                progress(f"APS refinement: {value.name}, {row.scenario_id}")
            old = row.plan
            new = solve_instant(case, field, row.elevation_deg, row.azimuth_deg, old.dni, settings,
                                keep_flux=old.flux is not None, warm=old.choice)
            value.runtime_s += new.solve_time_s
            # Both joint bounds cover the identical feasible set. Retain the strongest of each result.
            feasible = ("feasible", "feasible_shutdown")
            chosen = new if new.status in feasible and (old.status not in feasible or
                                                       new.objective_w >= old.objective_w) else old
            bounds = [ub for ub in (old.upper_bound_w, new.upper_bound_w) if ub is not None]
            upper = min(bounds)
            if chosen.status in feasible and upper < chosen.objective_w - 1e-6:
                raise ArithmeticError("Refined joint bound below the incumbent")
            chosen.upper_bound_w = upper
            if chosen.status in feasible:
                chosen.gap = max(0., upper - chosen.objective_w) / max(abs(upper), 1e-12)
            if new.upper_bound_w != old.upper_bound_w:
                chosen.bound_method = "best_joint_bound_across_runs"
            chosen.info["additional_joint_trials"] = settings.joint_trials
            row.plan = chosen


def rescore(values: list[LayoutValue], lam_l: float, reference: str | None = None) -> pd.DataFrame:
    """The ranking for another outer weight on the *same* plans (no re-solve; P_s, U_s unchanged)."""
    return compare_values(values, lam_l, reference)


# --------------------------------------------------------------------------- small exhaustive checks

def enumerate_layouts(case: Case, candidates_xyz: np.ndarray, n_select: int, sset,
                      settings: InnerSettings | None = None, lam_layout: float | None = None) -> pd.DataFrame:
    """V(L) of every N-subset of a small candidate set (brute force of the outer layer)."""
    from .layout_opt import design_spacing, validate_layout

    settings = settings or inner_settings(case)
    rows = []
    cand = np.asarray(candidates_xyz, dtype=float).reshape(-1, 3)
    for combo in combinations(range(len(cand)), n_select):
        xyz = cand[list(combo)]
        if validate_layout(case, xyz, n_select, design_spacing(case))["status"] != "ok":
            continue
        val = evaluate_layout_value(case, xyz, sset, name=str(combo), settings=settings,
                                    lam_layout=lam_layout, workers=1, keep_flux=False)
        t = val.totals()
        rows.append({"subset": combo, "value_mwh": t["value_mwh"], "energy_mwh": t["energy_mwh"],
                     "penalty_mwh": t["penalty_mwh"], "proven_optimal_all": all(r.plan.proven_optimal or
                                                                                 r.plan.status == "no_sun"
                                                                                 for r in val.records)})
    columns = ["subset", "value_mwh", "energy_mwh", "penalty_mwh", "proven_optimal_all"]
    return pd.DataFrame(rows, columns=columns).sort_values("value_mwh", ascending=False).reset_index(drop=True)


def displacement(baseline_xyz: np.ndarray, layout_xyz: np.ndarray) -> dict:
    """How far the mirrors moved: optimal one-to-one matching (Hungarian) of two equal-size layouts."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    a = np.asarray(baseline_xyz)[:, :2]
    b = np.asarray(layout_xyz)[:, :2]
    if len(a) != len(b) or not len(a):
        return {"comparable": False}
    cost = cdist(a, b)
    r, c = linear_sum_assignment(cost)
    d = cost[r, c]
    same = d < 1e-3
    return {"comparable": True, "match_from": r, "match_to": c, "distance_m": d,
            "mean_move_m": float(d.mean()), "median_move_m": float(np.median(d)), "max_move_m": float(d.max()),
            "n_unchanged": int(same.sum()), "n_moved": int((~same).sum()),
            "moved_over_one_pitch": int((d > np.sqrt(np.sum(np.ptp(a, axis=0) ** 2)) / max(len(a), 1)).sum())}


def baseline_layout(case: Case, n: int) -> tuple[np.ndarray, np.ndarray, str] | None:
    """The case's own field cut to the N heliostats nearest to the tower (rings fill from the inside)."""
    from .field import build_field

    try:
        fld = build_field(case)
    except Exception:  # noqa: BLE001 - no usable initial field
        return None
    if fld.n < n or n <= 0:
        return None
    rec = case.receiver
    r = np.hypot(fld.xyz[:, 0] - rec.center[0], fld.xyz[:, 1] - rec.center[1])
    keep = np.sort(np.argsort(r, kind="stable")[:n])
    note = ("initial field as given" if fld.n == n else
            f"initial field: the {n} of {fld.n} heliostats nearest to the tower")
    return fld.xyz[keep], fld.ids[keep], note


def with_xyz(value: LayoutValue, **kw) -> LayoutValue:
    return replace(value, **kw)
