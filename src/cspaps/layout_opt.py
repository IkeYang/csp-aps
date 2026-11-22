"""Outer loop: heliostat field layout optimisation.

Objective: maximise the annual absorbed energy for a fixed number of
heliostats (``layout_opt.n_heliostats``) inside the site (boundary, exclusion
zone, minimum spacing), minus lambda_L (``layout_opt.uniformity_weight``) times
the circumferential spread of the receiver flux, the part of the
non-uniformity that the layout decides and no aiming strategy can remove
(``layout_uniformity``). With lambda_L > 0 the N positions are chosen by an
exchange search instead of the N highest energies, which surrounds the tower.

Search
  1. A parametric generator (radial staggered or biomimetic spiral) fills the
     site with collision-free candidate positions for a parameter vector.
  2. Every candidate is evaluated over the year with the fast single-point
     aiming rule at the *search sampling* (default 30 DNI-weighted
     representative sun positions), incl. shading, blocking and attenuation.
  3. The best N candidates are kept and the kept field is re-evaluated.
  4. Differential evolution over the generator parameters, then a local
     pattern search around the best point.
Review (the advisor's "受限全年收益复排")
  5. The best distinct layouts are re-evaluated with the *review sampling*
     (more states, each split into DNI bands so flux-limited states are not
     averaged with weak ones) with the full APS enforcing the flux limits
     (verified), plus the centre-aim sampling error against the hourly year.
     They are ranked by this constrained annual energy; rank changes are
     reported.
  6. Swap refinement on the winner: the heliostats with the lowest APS annual
     yield are swapped for the best unused candidates; a swap is kept only if
     the APS annual energy increases.
  7. Final checks: count, boundary, exclusion zone, minimum spacing. A site
     that cannot hold N heliostats is reported as a capacity shortfall.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field, replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import sun
from .config import Case
from .field import BOUNDARY_TOL_M, Field, generate, site_mask
from .parallel import n_workers, pool
from .receiver import build_grid
from .simulate import AnnualResult, annual

DEFAULT_BOUNDS = {
    "radial_staggered": {
        "radial_spacing": [0.866, 1.6],
        "radial_growth": [0.0, 1.5],
        "azimuthal_spacing": [1.0, 2.0],
    },
    "biomimetic": {
        "a": [0.55, 1.2],
        "b": [0.5, 0.62],
    },
}


@dataclass
class LayoutEval:
    params: dict
    n_candidates: int
    xyz: np.ndarray
    energy_mwh: float  # absorbed-energy score for the configured search objective
    result: AnnualResult | None = dc_field(default=None, repr=False)
    candidates_xyz: np.ndarray | None = dc_field(default=None, repr=False)
    candidate_energy: np.ndarray | None = dc_field(default=None, repr=False)
    aps_energy_mwh: float | None = None
    aps_result: AnnualResult | None = dc_field(default=None, repr=False)
    center_energy_mwh: float | None = None
    search_aiming: str = "center"
    uniformity: dict | None = None  # layout_uniformity.layout_metrics of the selected field (lambda_L > 0)
    selection: dict | None = None  # exchange search info (swaps, time)
    value: object | None = dc_field(default=None, repr=False)  # layout_value.LayoutValue (formal ranking)

    @property
    def n(self) -> int:
        return int(len(self.xyz))


def make_samples(case: Case, method: str, n_bins: int, dni_bands: int = 1, hourly: sun.Samples | None = None,
                 seed: int = 0) -> sun.Samples:
    """Sun states for the search / review. With spatial DNI maps every state carries its map regime
    (``scenarios``): the hours of each regime are binned separately, so the maps' DNI energy is preserved."""
    from .scenarios import attach_regimes, build_scenarios, is_spatial

    hourly = hourly if hourly is not None else sun.hourly_samples(case)
    if is_spatial(case):
        if case.scenarios.source == "file":
            return build_scenarios(case).samples
        hourly = hourly if hourly.spatial else attach_regimes(case, hourly)
        if method == "hourly":
            return hourly
        return build_scenarios(case, n_bins, hourly, dni_bands).samples
    if method == "hourly":
        return hourly
    if method == "sun_grid":
        a = case.annual
        return sun.sun_grid(hourly, a.grid_elevation_step_deg, a.grid_azimuth_step_deg, dni_bands)
    return sun.sun_bins(hourly, n_bins, seed=seed, dni_bands=dni_bands)


def _spread_mwh(result, case, operational=False):
    ts = result.timeseries
    column = "surface_spread_mw" if case.layout_opt.uniformity_metric == "surface" else "circ_spread_mw"
    if operational:
        ts = ts.loc[ts.operational_feasible]
    return float(ts[column] @ ts.weight_h) if len(ts) else 0.0


def _surface_score(case):
    return case.layout_opt.uniformity_metric == "surface" or case.uniformity.z_range is not None


def _hourly_theoretical_score(case, xyz, hourly, lam, shading_points):
    """Evaluate one fixed layout over the weather year, bounding working memory by time blocks."""
    from .layout_uniformity import build_model

    full = _surface_score(case)
    res = annual(case, Field(xyz), samples=hourly, aiming="center", with_flux=full,
                 shading_points=shading_points, keep_state_power=lam > 0 and not full)
    energy = float(res.timeseries["p_absorbed_mw"] @ hourly.weight_h) if len(hourly) else 0.0
    if full:
        return energy, _spread_mwh(res, case)
    spread = 0.0
    if lam > 0:
        for start in range(0, len(hourly), 24):
            sl = slice(start, start + 24)
            subset = hourly.subset(np.arange(len(hourly))[sl])
            model = build_model(case, xyz, subset, res.state_power_w[sl], lam)
            spread += model.terms(np.ones(len(xyz)))["spread_mwh"]
    return energy, spread


def evaluate_layout(
    case: Case,
    kind: str,
    params: dict,
    samples: sun.Samples,
    n_select: int | None,
    shading_points: int | None = None,
    aiming: str = "center",
    lam: float | None = None,
    polish: bool = False,
) -> LayoutEval:
    """Generate candidates, choose N of them and score the layout.

    Score (``energy_mwh``) = annual absorbed energy (centre aiming, or the
    feasible APS energy for an APS search) minus lambda_L times the annual
    circumferential spread A_H sigma_circ (layout_uniformity). With lambda_L = 0
    the N candidates with the highest annual energy are kept; otherwise the
    exchange search of ``layout_uniformity.select`` maximises the score.
    """
    from . import layout_uniformity as lu

    lam = case.layout_opt.uniformity_weight if lam is None else float(lam)
    try:
        xyz = generate(case, kind, params)
    except ValueError:
        return LayoutEval(params, 0, np.zeros((0, 3)), 0.0)
    if len(xyz) == 0:
        return LayoutEval(params, 0, xyz, 0.0)
    keep_power = lam > 0
    center_all = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                        shading_points=shading_points, workers=1, keep_state_power=True)
    cand_xyz, cand_e = xyz, center_all.heliostat_energy_mwh
    n_cand = len(xyz)
    selected = bool(n_select and n_cand > n_select)
    sel_info = None
    if selected:
        iters = case.layout_opt.selection_iterations
        sel, sel_info = select_consistent(case, xyz, samples, n_select, lam, shading_points,
                                          max(iters, 6) if polish else iters, dense=center_all, polish=polish)
        keep = np.flatnonzero(sel)
        xyz = xyz[keep]
    center_res = (annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                         shading_points=shading_points, workers=1, keep_state_power=keep_power)
                  if selected else center_all)
    center_energy = center_res.summary()["absorbed_energy_mwh"]
    uni = lu.layout_metrics(case, xyz, samples, center_res.state_power_w, lam) if keep_power else None
    penalty = lam * uni["circ_spread_mwh"] if uni else 0.0
    if aiming == "center" and lam > 0 and _surface_score(case):
        center_res = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=True,
                            shading_points=shading_points, workers=1)
        center_energy = center_res.summary()["absorbed_energy_mwh"]
        penalty = lam * _spread_mwh(center_res, case)
    if aiming == "center":
        return LayoutEval(params, n_cand, xyz, center_energy - penalty, center_res, cand_xyz, cand_e,
                          center_energy_mwh=center_energy, search_aiming="center", uniformity=uni,
                          selection=sel_info)
    res = annual(case, Field(xyz), samples=samples, aiming=aiming, with_flux=True,
                 shading_points=shading_points, workers=1)
    aps_energy = res.summary()["feasible_absorbed_energy_mwh"]
    penalty = lam * _spread_mwh(res, case, operational=True)
    return LayoutEval(params, n_cand, xyz, aps_energy - penalty, res, cand_xyz, cand_e,
                      aps_energy_mwh=aps_energy, aps_result=res, center_energy_mwh=center_energy,
                      search_aiming=aiming, uniformity=uni, selection=sel_info)


def select_consistent(case: Case, xyz: np.ndarray, samples: sun.Samples, n_select: int, lam: float,
                      shading_points: int | None = None, iterations: int = 4,
                      dense: AnnualResult | None = None, polish: bool = False) -> tuple[np.ndarray, dict]:
    """Choose N of the candidate positions with candidate powers that are consistent with the choice.

    Pricing every candidate inside the full candidate field charges it shading
    and blocking from candidates that are never built. That penalty depends on
    the local candidate density -- heaviest in the densest rings, lightest at
    generator zone changes and outer rings -- so it can make far, isolated
    positions look better than closer ones. Instead: price all candidates with
    only the selected heliostats as blockers (selected ones get their real
    power, the others their power if added) and reselect from the current
    selection. A pass is kept only if the objective on the real powers improves;
    otherwise the number of allowed changes is halved (damping), so the passes
    cannot cycle. ``polish`` adds the full swap neighbourhood (review stage).
    """
    from . import layout_uniformity as lu

    fld = Field(xyz)

    def price(blockers):
        return annual(case, fld, samples=samples, aiming="center", with_flux=False, shading_points=shading_points,
                      workers=1, keep_state_power=True, blockers=blockers).state_power_w

    power = dense.state_power_w if dense is not None and dense.state_power_w is not None else price(None)
    sel, info0 = lu.select(lu.build_model(case, xyz, samples, power, lam), n_select)
    swaps, history, converged = info0["swaps"], [], False
    if iterations <= 0:
        return sel, {"swaps": swaps, "iterations": 0, "converged": False, "objective_history_mwh": [],
                     "pricing": "candidate field (no re-pricing)"}
    power = price(sel)
    model = lu.build_model(case, xyz, samples, power, lam)
    val = model.objective(sel)  # the selected heliostats' powers are their real powers
    history.append(round(val, 3))
    max_change = n_select
    for _ in range(int(iterations)):
        cand, info = lu.select(model, n_select, init=sel, max_swaps=max_change, polish=polish)
        if np.array_equal(cand, sel):
            converged = True
            break
        c_power = price(cand)
        c_model = lu.build_model(case, xyz, samples, c_power, lam)
        c_val = c_model.objective(cand)
        if c_val > val + 1e-9 * max(abs(val), 1.0):
            sel, power, model, val = cand, c_power, c_model, c_val
            swaps += info["swaps"]
            history.append(round(val, 3))
        else:
            max_change = max(1, int(np.sum(cand != sel)) // 4)  # half of the attempted swaps
            if int(np.sum(cand != sel)) <= 2:
                converged = True
                break
    return sel, {"swaps": swaps, "iterations": len(history), "converged": converged,
                 "objective_history_mwh": history, "pricing": "candidates priced against the selected layout"}


def layout_tradeoff(case: Case, kind: str, params: dict, samples: sun.Samples, n_select: int, lambdas,
                    shading_points: int | None = None, certify: bool = True) -> list[dict]:
    """Annual energy vs circumferential uniformity of the layout for several lambda_L (fixed generator)."""
    from . import layout_uniformity as lu

    xyz = generate(case, kind, params)
    cand = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                  shading_points=shading_points, workers=1, keep_state_power=True)
    rows = []
    for lam in sorted(float(v) for v in lambdas):
        sel, info = select_consistent(case, xyz, samples, n_select, lam, shading_points,
                                      max(case.layout_opt.selection_iterations, 6), dense=cand, polish=True)
        model = lu.build_model(case, xyz, samples, annual(
            case, Field(xyz), samples=samples, aiming="center", with_flux=False, shading_points=shading_points,
            workers=1, keep_state_power=True, blockers=sel).state_power_w, lam)
        lay = xyz[sel]
        full = _surface_score(case)
        res = annual(case, Field(lay), samples=samples, aiming="center", with_flux=full,
                     shading_points=shading_points, workers=1, keep_state_power=True)
        met = lu.layout_metrics(case, lay, samples, res.state_power_w, lam)
        row = {"lambda": lam, "energy_mwh": res.summary()["absorbed_energy_mwh"],
               "cv_circ_weighted": met["cv_circ_weighted"], "centroid_offset_m": met["centroid_offset_m"],
               "north_fraction": met["north_fraction"], "objective_mwh": met["objective_mwh"],
               "sector_share": met["sector_share"], "n_detached": met["n_detached"], "swaps": info["swaps"], "xyz": lay}
        if full:
            spread = _spread_mwh(res, case)
            row["objective_mwh"] = row["energy_mwh"] - lam * spread
            if case.layout_opt.uniformity_metric == "surface":
                row["surface_spread_mwh"] = spread
            row["uniformity_metric"] = case.layout_opt.uniformity_metric
        if certify and lam > 0 and not full:
            row["selection_gap"] = lu.selection_bound(model, n_select, sel, max_iter=120, time_limit_s=5.0)["gap"]
        rows.append(row)
    return rows


class _Objective:
    """Picklable objective for scipy's differential evolution (supports workers > 1)."""

    def __init__(self, case: Case, kind: str, names: list[str], samples: sun.Samples, aiming: str):
        self.case, self.kind, self.names, self.samples, self.aiming = case, kind, names, samples, aiming

    def params(self, x) -> dict:
        p = dict(self.case.layout_opt.fixed)
        p.update({k: float(v) for k, v in zip(self.names, x)})
        return p

    def __call__(self, x) -> float:
        lo = self.case.layout_opt
        ev = evaluate_layout(self.case, self.kind, self.params(x), self.samples, lo.n_heliostats,
                             lo.shading_points, self.aiming)
        if lo.n_heliostats and ev.n < lo.n_heliostats:
            # Spread of C nonnegative equal-area columns is at most sqrt(C) times
            # the mean. Rank every shortfall below any feasible weighted objective.
            max_e = self.samples.dni_hours * self.samples.max_factor * lo.n_heliostats * self.case.heliostat.area / 1000
            floor = max_e * (1 + lo.uniformity_weight * np.sqrt(self.case.receiver.n_azimuth)) + 1
            return floor * (1 + lo.n_heliostats - ev.n)
        return -ev.energy_mwh


def _refine_positions(case, ev, samples, rng):
    """Improve a feasible field in continuous XY space, keeping count and spacing at every move."""
    from . import layout_uniformity as lu

    spec = case.layout_opt
    if not spec.position_trials or not ev.n or ev.n != spec.n_heliostats or ev.search_aiming != "center":
        return
    spacing = case.heliostat.diagonal * (spec.min_spacing_factor or spec.fixed.get("spacing_factor", 1.1))
    original = ev.xyz.copy()
    accepted = 0
    for step in range(spec.position_trials):
        xyz = ev.xyz.copy()
        ids = rng.choice(ev.n, size=min(ev.n, 1 + step % 5), replace=False)
        angles = rng.uniform(0, 2 * np.pi, len(ids))
        radius = spacing * (0.1, 0.4, 1.0)[step % 3]
        if ev.n > 1:
            nearest = cKDTree(xyz[:, :2]).query(xyz[ids, :2], k=2)[0][:, 1]
            # A step below half the available clearance stays legal even when
            # both members of a tight pair move towards one another.
            radius = np.minimum(radius, np.maximum(0.02, (nearest - spacing) * 0.48))[:, None]
        xyz[ids, :2] += radius * np.column_stack((np.cos(angles), np.sin(angles)))
        if not site_mask(case, xyz[ids])[0].all() or cKDTree(xyz[:, :2]).query_pairs(spacing * (1 - 1e-9)):
            continue
        full = _surface_score(case)
        result = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=full,
                        shading_points=spec.shading_points, workers=1, keep_state_power=spec.uniformity_weight > 0)
        energy = result.summary()["absorbed_energy_mwh"]
        uni = (lu.layout_metrics(case, xyz, samples, result.state_power_w, spec.uniformity_weight)
               if spec.uniformity_weight > 0 else None)
        spread = _spread_mwh(result, case) if full else (uni["circ_spread_mwh"] if uni else 0)
        score = energy - spec.uniformity_weight * spread
        if score > ev.energy_mwh:
            ev.xyz, ev.result, ev.uniformity = xyz, result, uni
            ev.energy_mwh, ev.center_energy_mwh = score, energy
            accepted += 1
    if accepted:
        # Keep unused candidate positions that remain compatible with every moved mirror.
        moved = (ev.xyz != original).any(axis=1)
        distances, _ = cKDTree(ev.xyz[moved, :2]).query(ev.candidates_xyz[:, :2])
        pool = ev.candidates_xyz[distances >= spacing * (1 - 1e-9)]
        ev.candidates_xyz = np.unique(np.vstack([pool, ev.xyz]), axis=0)
        ev.candidate_energy = None
    ev.selection = {**(ev.selection or {}), "position_trials": spec.position_trials,
                    "position_improvements": accepted}


def validate_layout(case: Case, xyz: np.ndarray, n_required: int | None, spacing_m: float) -> dict:
    """Final site checks of a layout; status ok / capacity_shortfall / constraint_violation."""
    keep, reason = site_mask(case, xyz)
    out_idx = np.flatnonzero(~keep)
    min_d = float("inf")
    close = 0
    if len(xyz) > 1:
        tree = cKDTree(xyz[:, :2])
        d, _ = tree.query(xyz[:, :2], k=2)
        min_d = float(d[:, 1].min())
        close = int(len(tree.query_pairs(spacing_m * (1 - 1e-9))))
    n = int(len(xyz))
    shortfall = max(0, int(n_required) - n) if n_required else 0
    violations = []
    if out_idx.size:
        violations.append(f"{out_idx.size} heliostats outside the site: " + "; ".join(reason[i] for i in out_idx[:5]))
    if close:
        violations.append(f"{close} pairs closer than the minimum spacing {spacing_m:.3f} m")
    status = "constraint_violation" if violations else ("capacity_shortfall" if shortfall else "ok")
    return {
        "status": status, "n_heliostats": n, "n_required": n_required, "shortfall": shortfall,
        "min_distance_m": round(min_d, 4) if np.isfinite(min_d) else None,
        "required_min_spacing_m": round(spacing_m, 4), "pairs_too_close": close,
        "outside_site": int(out_idx.size), "boundary_tolerance_m": BOUNDARY_TOL_M, "violations": violations,
    }


def _clean_uniformity(u: dict | None) -> dict | None:
    if not u:
        return None
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in u.items()}


@dataclass
class LayoutOptResult:
    best: LayoutEval
    history: pd.DataFrame  # every evaluated parameter vector
    ranking: pd.DataFrame  # review of the top layouts (constrained annual energy, sampling errors)
    samples: sun.Samples
    runtime_s: float
    validation: dict = dc_field(default_factory=dict)
    swaps: list[dict] = dc_field(default_factory=list)
    review_samples: sun.Samples | None = None
    capacity: dict = dc_field(default_factory=dict)
    candidates: list[LayoutEval] = dc_field(default_factory=list, repr=False)  # reviewed layouts, search order
    bound: dict = dc_field(default_factory=dict)  # certificate: upper bound on any layout's annual energy
    potential: object | None = dc_field(default=None, repr=False)  # certify.PotentialMap of the site
    formal: dict = dc_field(default_factory=dict, repr=False)  # formal ranking by V(L) (see formal_ranking)

    @property
    def field(self) -> Field:
        return Field(self.best.xyz)

    @property
    def status(self) -> str:
        return self.validation.get("status", "unknown")

    def summary(self) -> dict:
        b = self.best
        aps_summary = b.aps_result.summary() if b.aps_result is not None else None
        out = self._summary(b, aps_summary)
        if self.formal:
            out["formal"] = formal_summary(self.formal)
            out["ranking_basis"] = "aps_value: V(L) = sum_s h_s (P_s - lambda_L U_s) from per-scenario APS plans"
            win = out["formal"]["winner_totals"]
            out["annual_energy_mwh_with_aps"] = round(win["energy_mwh"], 1)
            out["annual_value_mwh"] = round(win["value_mwh"], 1) if np.isfinite(win["value_mwh"]) else None
            out["operational_status"] = ("no_feasible_annual_states" if not win["status_counts"].get("feasible")
                                         else "formal_aps_value")
        else:
            out["ranking_basis"] = "legacy review objective"
        return out

    def _summary(self, b, aps_summary) -> dict:
        return {
            "status": self.status,
            "operational_status": ("no_feasible_annual_states" if aps_summary and aps_summary["feasible_sun_states"] == 0
                                   else "reviewed_feasible_states" if aps_summary else "not_reviewed_with_aps"),
            "generator_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in b.params.items()},
            "n_heliostats": b.n,
            "n_required": self.validation.get("n_required"),
            "n_candidates": b.n_candidates,
            "search_aiming": b.search_aiming,
            "search_method": "differential_evolution_plus_local_search_and_APS_review",
            "global_optimality_proven": False,
            "annual_energy_mwh_search_objective": round(b.energy_mwh, 1),
            "annual_energy_mwh_center_aiming": round(b.center_energy_mwh, 1) if b.center_energy_mwh is not None else None,
            "annual_energy_mwh_with_aps": round(b.aps_energy_mwh, 1) if b.aps_energy_mwh is not None else None,
            "annual_energy_mwh_raw_aps": aps_summary["absorbed_energy_mwh"] if aps_summary else None,
            "annual_hours_uniformity_failed": aps_summary["hours_uniformity_failed"] if aps_summary else None,
            "annual_hours_flux_or_verification_failed": aps_summary["hours_flux_or_verification_failed"] if aps_summary else None,
            "n_evaluations": int(len(self.history)),
            "search_sampling": {"method": self.samples.label, "n_states": len(self.samples)},
            "review_sampling": ({"method": self.review_samples.label, "n_states": len(self.review_samples)}
                                if self.review_samples is not None else None),
            "swaps": self.swaps,
            "validation": self.validation,
            "capacity": self.capacity,
            "upper_bound": self.bound,
            "uniformity": _clean_uniformity(b.uniformity),
            "existence": self.existence(),
            "runtime_s": round(self.runtime_s, 1),
        }

    def existence(self) -> dict:
        """What is proven about problem 2 for this site and N."""
        b = self.bound or {}
        cap = b.get("packing_capacity_bound")
        n_req = self.validation.get("n_required")
        if cap is not None and n_req and n_req > cap:
            return {"optimum_exists": False,
                    "statement": f"no layout exists: at most {cap} heliostats fit in the site with the minimum spacing "
                                 "(Fejes Toth packing bound)"}
        if self.validation.get("status") == "ok":
            return {"optimum_exists": True,
                    "scope": "finite evaluated design set",
                    "statement": "A feasible layout exists and the best of the finite evaluated layouts is attained. "
                                 "Continuous-site attainment requires an upper-semicontinuous objective on the closed site.",
                    "found_mwh": b.get("found_energy_mwh"), "upper_bound_mwh": b.get("upper_bound_mwh"),
                    "gap": b.get("gap")}
        return {"optimum_exists": None, "statement": "no feasible layout with N heliostats was found and the packing "
                                                     "bound does not exclude one"}


def optimize_layout(
    case: Case,
    progress: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> LayoutOptResult:
    from scipy.optimize import differential_evolution

    t0 = _time.perf_counter()
    spec = case.layout_opt
    kind = spec.generator
    bounds = dict(DEFAULT_BOUNDS.get(kind, {}))
    bounds.update(spec.bounds or {})
    for k in spec.fixed:
        bounds.pop(k, None)
    if not bounds:
        raise ValueError("layout_opt: nothing to optimise (all parameters fixed)")
    names = list(bounds)
    say = progress or (lambda msg: None)
    stop = cancel or (lambda: False)
    from . import layout_uniformity as lu

    lam = float(spec.uniformity_weight)
    from .scenarios import attach_regimes

    formal = spec.ranking == "aps_value" and spec.refine_top > 0  # refine_top = 0: fast search only, no review
    hourly = attach_regimes(case, sun.hourly_samples(case))  # map regime of every hour (spatial DNI)
    search_aiming = spec.search_aiming
    if search_aiming == "auto":
        # formal ranking: the search is a fast centre-aim pre-screen; APS runs on the formal scenario set
        search_aiming = "greedy" if case.uniformity.enabled and not formal else "center"
    search_bands = spec.review_dni_bands if search_aiming != "center" else 1
    samples = make_samples(case, spec.search_sampling, spec.n_sun_bins, search_bands, hourly, spec.seed)
    say(f"search sampling: {samples.label}, {len(samples)} states, aiming={search_aiming}; optimising {names}")

    obj = _Objective(case, kind, names, samples, search_aiming)
    records: list[dict] = []

    def record(x, energy, stage):
        records.append({"evaluation": len(records) + 1, "stage": stage, **obj.params(x), "energy_mwh": energy})

    gen = {"n": 0}

    class Cancelled(Exception):
        pass

    def callback(intermediate_result):
        gen["n"] += 1
        say(f"generation {gen['n']}: best so far {-intermediate_result.fun:.1f} MWh")
        if stop():
            return True  # stops differential evolution

    nw = n_workers(spec.workers)
    pl = pool(nw) if nw > 1 else None
    stage = {"name": "global"}

    def mapper(f, xs):
        if stop():
            raise Cancelled()
        xs = [np.asarray(x, dtype=float) for x in xs]
        vals = pl.map(f, xs) if pl is not None else [f(x) for x in xs]
        for x, v in zip(xs, vals):
            record(x, -v, stage["name"])
        return vals

    lo = np.array([bounds[k][0] for k in names], dtype=float)
    hi = np.array([bounds[k][1] for k in names], dtype=float)
    cancelled = False
    try:
        if nw > 1:
            workers, func = mapper, obj
        else:
            def func(x):
                return mapper(obj, [x])[0]
            workers = 1
        differential_evolution(
            func, list(zip(lo, hi)), maxiter=spec.maxiter, popsize=spec.popsize,
            seed=spec.seed, polish=False, tol=1e-4, init="sobol", workers=workers,
            updating="deferred" if nw > 1 else "immediate", callback=callback,
        )
        stage["name"] = "local"
        best_i = int(np.argmax([r["energy_mwh"] for r in records]))
        x_best = np.array([records[best_i][k] for k in names], dtype=float)
        e_best = records[best_i]["energy_mwh"]
        step = 0.05 * (hi - lo)
        for r in range(spec.polish_rounds):
            trial = []
            for j in range(len(names)):
                for sgn in (-1.0, 1.0):
                    x = x_best.copy()
                    x[j] = np.clip(x[j] + sgn * step[j], lo[j], hi[j])
                    if not np.allclose(x, x_best):
                        trial.append(x)
            if not trial:
                break
            energies = -np.asarray(mapper(obj, trial))
            k = int(np.argmax(energies))
            if energies[k] > e_best:
                x_best, e_best = trial[k], float(energies[k])
                say(f"local refinement {r + 1}: improved to {e_best:.1f} MWh")
            else:
                step = step / 2
                say(f"local refinement {r + 1}: no improvement, step halved")
    except Cancelled:
        cancelled = True
        say("cancelled: reviewing the best layouts found so far")
    finally:
        if pl is not None:
            pl.close()
            pl.join()
    if not records:
        raise RuntimeError("layout optimisation was cancelled before any layout was evaluated")
    history = pd.DataFrame(records).sort_values("energy_mwh", ascending=False, kind="stable").reset_index(drop=True)
    history["search_rank"] = np.arange(1, len(history) + 1)

    # ---- review of the best distinct layouts
    top = history.drop_duplicates(subset=names).head(max(1, spec.refine_top))
    grid = build_grid(case.receiver)
    review_center = make_samples(case, spec.review_sampling, spec.review_n_sun_bins, 1, hourly, spec.seed)
    review_aps = make_samples(case, spec.review_sampling, spec.review_n_sun_bins, spec.review_dni_bands, hourly, spec.seed)
    review_aiming = case.aps.method if case.aps.method != "center" else search_aiming
    # formal ranking replaces the 0.4 constrained-APS review (APS runs on the formal scenario set instead)
    use_aps = spec.refine_top > 0 and review_aiming != "center" and not formal
    evals: list[LayoutEval] = []
    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        params = {**spec.fixed, **{k: float(row[k]) for k in names}}
        say(f"review {rank}/{len(top)}: selecting the positions (full swap neighbourhood)")
        ev = evaluate_layout(case, kind, params, samples, spec.n_heliostats, spec.shading_points,
                             aiming=search_aiming, polish=True)
        original = replace(ev)
        _refine_positions(case, ev, samples, np.random.default_rng(spec.seed + rank))
        scored = None
        if spec.refine_top > 0 and ev.xyz is not original.xyz:
            # A move profitable on the search quadrature can lose energy over the year.
            # Keep the unmoved layout as an incumbent under the final ranking objective.
            reference = hourly if spec.review_check_hourly else review_center
            base_score = _hourly_theoretical_score(case, original.xyz, reference, lam, spec.shading_points)
            scored = _hourly_theoretical_score(case, ev.xyz, reference, lam, spec.shading_points)
            if scored[0] - lam * scored[1] <= base_score[0] - lam * base_score[1]:
                original.selection = {**(ev.selection or {}), "position_review_accepted": False}
                ev, scored = original, base_score
            else:
                ev.selection["position_review_accepted"] = True
        info = {**{k: params[k] for k in names}, "search_rank": rank, "n_heliostats": ev.n,
                "n_candidates": ev.n_candidates, "search_aiming": search_aiming,
                "energy_search_objective_mwh": ev.energy_mwh,
                "energy_search_center_mwh": ev.center_energy_mwh,
                "energy_search_center_absorbed_mwh": ev.center_energy_mwh}
        if ev.n and spec.refine_top > 0:
            say(f"review {rank}/{len(top)}: centre aiming on {len(review_center)} states")
            rc = annual(case, Field(ev.xyz), samples=review_center, aiming="center", grid=grid,
                        shading_points=spec.shading_points, keep_state_power=True)
            info["energy_review_center_mwh"] = rc.summary()["absorbed_energy_mwh"]
            uni = lu.layout_metrics(case, ev.xyz, review_center, rc.state_power_w, lam, grid)
            ev.uniformity = uni
            info["circ_cv_review"] = uni["cv_circ_weighted"]
            info["circ_spread_review_mwh"] = uni["circ_spread_mwh"]
            info["objective_spread_review_mwh"] = (_spread_mwh(rc, case) if _surface_score(case)
                                                   else uni["circ_spread_mwh"])
            info["centroid_offset_m"] = uni["centroid_offset_m"]
            info["north_fraction"] = uni["north_fraction"]
            info["constrained_states_center"] = int(rc.timeseries["over_limit"].sum())
            info["hours_over_limit_center"] = float(rc.timeseries.loc[rc.timeseries["over_limit"], "weight_h"].sum())
            if spec.review_check_hourly:
                say(f"review {rank}/{len(top)}: hourly year with centre aiming (sampling error)")
                e_h, spread_h = scored or _hourly_theoretical_score(case, ev.xyz, hourly, lam, spec.shading_points)
                info["energy_hourly_center_mwh"] = e_h
                info["objective_spread_hourly_mwh"] = spread_h
                info["search_sampling_error_pct"] = 100 * (float(ev.center_energy_mwh or 0.0) / e_h - 1) if e_h else None
                info["review_sampling_error_pct"] = 100 * (info["energy_review_center_mwh"] / e_h - 1) if e_h else None
            if use_aps and not stop():
                say(f"review {rank}/{len(top)}: constrained APS on {len(review_aps)} states")
                ra = annual(case, Field(ev.xyz), samples=review_aps, aiming=review_aiming, grid=grid,
                            shading_points=spec.shading_points)
                ev.aps_result = ra
                ra_summary = ra.summary()
                ev.aps_energy_mwh = ra_summary["feasible_absorbed_energy_mwh"]
                info["energy_constrained_aps_mwh"] = ev.aps_energy_mwh
                info["energy_raw_aps_absorbed_mwh"] = ra_summary["absorbed_energy_mwh"]
                info["aps_hours_uniformity_failed"] = ra_summary["hours_uniformity_failed"]
                info["aps_hours_flux_or_verification_failed"] = ra_summary["hours_flux_or_verification_failed"]
                cv_rows = ra.timeseries["uniformity_cv_2d"].dropna()
                info["aps_max_cv_2d"] = float(cv_rows.max()) if len(cv_rows) else None
                info["aps_max_peak_kw_m2"] = float(ra.timeseries["peak_flux_kw_m2"].max())
                info["aps_states_not_ok"] = int((~ra.timeseries["operational_feasible"]).sum())
                info["aps_max_defocused"] = int(ra.timeseries["n_defocused"].max())
        if "energy_review_center_mwh" in info:
            if search_aiming != "center" and ev.aps_result is not None:
                spread = _spread_mwh(ev.aps_result, case, operational=True)
                info["objective_review_mwh"] = ev.aps_energy_mwh - lam * spread
            else:
                energy = info.get("energy_hourly_center_mwh", info["energy_review_center_mwh"])
                spread = info.get("objective_spread_hourly_mwh", info["objective_spread_review_mwh"])
                ev.center_energy_mwh = energy
                info["objective_review_mwh"] = energy - lam * spread
        evals.append(ev)
        rows.append(info)
    ranking = pd.DataFrame(rows)
    key_col = "objective_review_mwh" if "objective_review_mwh" in ranking else "energy_search_objective_mwh"
    ranking = ranking.sort_values(key_col, ascending=False).reset_index(drop=True)
    ranking["final_rank"] = np.arange(1, len(ranking) + 1)
    ranking["rank_change"] = ranking["search_rank"] - ranking["final_rank"]
    best_rank = int(ranking.loc[0, "search_rank"]) - 1
    best = evals[best_rank]

    # ---- swap refinement on the winner (accepted on constrained annual energy). With lambda_L > 0 the
    # selection is already a swap-local optimum of the uniformity-weighted objective (exchange search).
    swaps = []
    if (search_aiming != "center" and use_aps and lam <= 0 and best.aps_result is not None and best.candidates_xyz is not None
            and spec.swap_rounds > 0):
        cur_xyz, cur_res = best.xyz, best.aps_result
        m = max(1, int(round(spec.swap_fraction * len(cur_xyz))))
        for r in range(spec.swap_rounds):
            if stop():
                break
            sel = {tuple(np.round(p, 6)) for p in cur_xyz}
            in_layout = np.array([tuple(np.round(p, 6)) in sel for p in best.candidates_xyz])
            unused = np.flatnonzero(~in_layout)
            if unused.size == 0:
                break
            # price unused candidates as if added to the current layout (not inside the full candidate field)
            price = annual(case, Field(best.candidates_xyz), samples=samples, aiming="center", with_flux=False,
                           shading_points=spec.shading_points, workers=1, blockers=in_layout).heliostat_energy_mwh
            add = unused[np.argsort(-price[unused])[:m]]
            drop = np.argsort(cur_res.heliostat_energy_mwh)[:len(add)]
            new_xyz = np.vstack([np.delete(cur_xyz, drop, axis=0), best.candidates_xyz[add]])
            say(f"swap round {r + 1}: trying {len(add)} swaps")
            new_res = annual(case, Field(new_xyz), samples=review_aps, aiming=review_aiming, grid=grid,
                             shading_points=spec.shading_points)
            new_feasible = new_res.summary()["feasible_absorbed_energy_mwh"]
            cur_feasible = cur_res.summary()["feasible_absorbed_energy_mwh"]
            gain = new_feasible - cur_feasible
            swaps.append({"round": r + 1, "swapped": int(len(add)), "gain_mwh": round(float(gain), 2),
                          "accepted": bool(gain > 0)})
            if gain > 0:
                cur_xyz, cur_res = new_xyz, new_res
            else:
                break
        best.xyz, best.aps_result = cur_xyz, cur_res
        best.aps_energy_mwh = cur_res.summary()["feasible_absorbed_energy_mwh"]
        center_final = annual(case, Field(best.xyz), samples=review_center, aiming="center", grid=grid,
                              with_flux=False, shading_points=spec.shading_points)
        best.center_energy_mwh = center_final.summary()["absorbed_energy_mwh"]
        best_row = ranking.index[ranking["search_rank"] == best_rank + 1]
        if len(best_row):
            row_i = best_row[0]
            ranking.loc[row_i, "energy_constrained_aps_mwh"] = best.aps_energy_mwh
            ranking.loc[row_i, "energy_review_center_mwh"] = best.center_energy_mwh
            ranking.loc[row_i, "energy_raw_aps_absorbed_mwh"] = cur_res.summary()["absorbed_energy_mwh"]
            ranking.loc[row_i, "aps_states_not_ok"] = int((~cur_res.timeseries["operational_feasible"]).sum())
            cv_rows = cur_res.timeseries["uniformity_cv_2d"].dropna()
            if len(cv_rows):
                ranking.loc[row_i, "aps_max_cv_2d"] = float(cv_rows.max())
            ranking.loc[row_i, "objective_review_mwh"] = best.aps_energy_mwh
            ranking = ranking.sort_values("objective_review_mwh", ascending=False).reset_index(drop=True)
            ranking["final_rank"] = np.arange(1, len(ranking) + 1)
            ranking["rank_change"] = ranking["search_rank"] - ranking["final_rank"]

    # ---- formal ranking: every shortlisted layout (and the initial field) gets its own APS plan per scenario
    formal_info = {}
    if formal and not stop():
        formal_info = formal_ranking(case, evals, hourly, samples, say, stop)
        win = formal_info.get("winner")
        if win is not None:
            value = formal_info["values"][win]
            owner = formal_info["owners"].get(win)
            if owner is None:  # the initial field won
                best = LayoutEval({"source": "initial field"}, value.n, value.xyz, value.totals()["value_mwh"],
                                  search_aiming="formal_aps_value")
            else:  # centre-aim results of the search belong to the unswapped positions
                best = replace(owner, xyz=value.xyz, result=None, aps_result=None)
            best.value = value
            best.aps_energy_mwh = value.totals()["energy_mwh"]

    # ---- final checks and capacity
    dm = case.heliostat.diagonal * (spec.min_spacing_factor or spec.fixed.get("spacing_factor", 1.1))
    validation = validate_layout(case, best.xyz, spec.n_heliostats, dm)
    capacity = {}
    if validation["status"] == "capacity_shortfall":
        dense = {**spec.fixed, **{k: float(lo[i]) for i, k in enumerate(names)}}  # lower bounds = densest
        try:
            max_n = len(generate(case, kind, dense))
        except ValueError:
            max_n = best.n
        capacity = {"max_placeable_with_densest_parameters": int(max(max_n, best.n)),
                    "message": f"site holds at most about {max(max_n, best.n)} heliostats with the "
                               f"{kind} generator; {validation['shortfall']} short of {spec.n_heliostats}"}
    if cancelled:
        validation["cancelled"] = True
    bound, pmap = {}, None
    if formal_info and spec.upper_bound and spec.n_heliostats:
        from .certify import layout_upper_bound, potential_map

        say("certificate: lone-heliostat potential map and packing bound on the formal scenario set")
        sset = formal_info["scenario_set"]
        pmap = potential_map(case, sset.samples)
        tot = best.value.totals() if best.value is not None else None
        bound = layout_upper_bound(case, spec.n_heliostats, dm, samples=sset.samples, pmap=pmap, layout_xyz=best.xyz,
                                   found_energy_mwh=tot["energy_mwh"] if tot else None)
        bound["sampling"] = {"method": "formal scenario set", "n_states": len(sset), "fingerprint": sset.fingerprint()}
        bound["found_energy_basis"] = "per-scenario APS plans of the winner (joint poses, verified), absorbed energy E"
        if tot and bound.get("upper_bound_mwh"):
            bound["value_found_mwh"] = tot["value_mwh"]
            bound["gap_value_conservative"] = max(0.0, 1.0 - tot["value_mwh"] / bound["upper_bound_mwh"])
            bound["applies_to"] = (f"the formal scenario set ({len(sset)} scenarios, maps {sorted(sset.subset_maps())}); "
                                   "any layout of N heliostats in the site, any aim points; V <= E <= bound")
    elif spec.upper_bound and spec.n_heliostats and not formal:
        from .certify import hourly_site_upper_bound, layout_upper_bound, potential_map

        say("certificate: lone-heliostat potential map and packing bound")
        bound_samples = review_aps if use_aps else review_center
        pmap = potential_map(case, bound_samples)
        e_center = annual(case, Field(best.xyz), samples=bound_samples, aiming="center", with_flux=False,
                          shading_points=spec.shading_points).summary()["absorbed_energy_mwh"]
        bound = layout_upper_bound(case, spec.n_heliostats, dm, samples=bound_samples, pmap=pmap,
                                   layout_xyz=best.xyz, found_energy_mwh=e_center)
        bound["sampling"] = {"method": bound_samples.label, "n_states": len(bound_samples)}
        bound["found_energy_basis"] = "centre aiming, shading and blocking, no flux limit (theoretical)"
        if best.aps_energy_mwh is not None and use_aps and bound.get("upper_bound_mwh"):
            bound["found_aps_feasible_mwh"] = float(best.aps_energy_mwh)
            bound["gap_aps_feasible"] = max(0.0, 1.0 - best.aps_energy_mwh / bound["upper_bound_mwh"])
        if validation["status"] == "ok":
            bound["objective_note"] = ("energy bound; the uniformity-weighted objective J = E - lambda_L * spread "
                                       "is <= E, so J <= this bound as well (conservative: no tighter general bound)")
            best_row = ranking.loc[ranking["final_rank"] == 1]
            if len(best_row) and "objective_review_mwh" in best_row and bound.get("upper_bound_mwh"):
                j_obj = float(best_row["objective_review_mwh"].iloc[0])
                bound["objective_review_mwh"] = j_obj
                hourly_rank = search_aiming == "center" and "energy_hourly_center_mwh" in best_row
                alpha = max(case.receiver.panel_absorptivity or [case.receiver.absorptivity])
                objective_upper = (hourly_site_upper_bound(case, best.n, dm, hourly, pmap)
                                   if hourly_rank else bound["upper_bound_mwh"])
                if search_aiming == "center" and not hourly_rank and use_aps and spec.review_dni_bands > 1:
                    objective_upper = (review_center.dni_hours / 1000 * best.n * case.heliostat.area
                                       * case.heliostat.reflectivity * case.heliostat.cleanliness * alpha)
                bound["objective_upper_bound_mwh"] = float(objective_upper)
                bound["objective_method"] = "hourly" if hourly_rank else spec.review_sampling
                bound["objective_aiming"] = search_aiming
                bound["uniformity_metric"] = spec.uniformity_metric
                bound["objective_states"] = len(hourly if hourly_rank else review_center if search_aiming == "center" else review_aps)
                bound["gap_objective_conservative"] = (1.0 - j_obj / objective_upper) if objective_upper > 0 else None
    uniformity = {"weight": lam}
    if best.n:
        say("certificate: uniformity metrics and selection bound")
        rc = annual(case, Field(best.xyz), samples=review_center, aiming="center", with_flux=False,
                    shading_points=spec.shading_points, keep_state_power=True)
        uniformity.update(lu.layout_metrics(case, best.xyz, review_center, rc.state_power_w, lam, grid))
        # the selection step of the search is certified on the best layout that came out of it
        sel_ev = best if best.candidates_xyz is not None else next(
            (e for e in evals if e.candidates_xyz is not None and e.n == spec.n_heliostats), None)
        if lam > 0 and sel_ev is not None and spec.n_heliostats:
            key = {tuple(np.round(p, 6)) for p in sel_ev.xyz}
            sel = np.array([tuple(np.round(p, 6)) in key for p in sel_ev.candidates_xyz])
            cand = annual(case, Field(sel_ev.candidates_xyz), samples=samples, aiming="center", with_flux=False,
                          shading_points=spec.shading_points, workers=1, keep_state_power=True, blockers=sel)
            model = lu.build_model(case, sel_ev.candidates_xyz, samples, cand.state_power_w, lam, grid)
            if sel.sum() == min(spec.n_heliostats, len(sel)):
                cert = lu.selection_bound(model, int(sel.sum()), sel)
                cert["model"] = ("candidate positions of this generator parameter vector, centre aim, "
                                 "shading / blocking priced against the final layout")
                cert["layout"] = "winner" if sel_ev is best else "best search layout (the initial field won)"
                uniformity["selection_certificate"] = cert
        uniformity["selection"] = best.selection
        best.uniformity = uniformity
    return LayoutOptResult(best, history, ranking, samples, _time.perf_counter() - t0, validation, swaps,
                           review_aps if use_aps else review_center, capacity, evals, bound, pmap, formal_info)


def formal_ranking(case: Case, evals: list[LayoutEval], hourly: sun.Samples, samples: sun.Samples,
                   say: Callable[[str], None] = lambda m: None, stop: Callable[[], bool] = lambda: False) -> dict:
    """Rank layouts by V(L) = sum_s h_s (P_s - lambda_L U_s) with their own APS plan in every scenario.

    Layouts: the shortlisted search layouts (``evals``, N heliostats) and, with ``layout_opt.baseline =
    initial``, the case's initial field. Every layout is solved on the same scenario set with the same
    inner settings. The leading search layout is then polished by swaps (lowest APS-energy heliostats
    out, best-priced unused candidates in) that are kept only if V increases; optionally the winner and
    the initial field are re-evaluated on a denser scenario set (sampling error of the representative set).
    """
    from .field import Field
    from .layout_value import baseline_layout, compare_values, displacement, evaluate_layout_value, inner_settings
    from .scenarios import build_scenarios

    spec = case.layout_opt
    n_req = spec.n_heliostats or (evals[0].n if evals else 0)
    sset = build_scenarios(case, hourly=hourly)
    settings = inner_settings(case)
    say(f"formal ranking: {len(sset)} scenarios (maps {sorted(sset.subset_maps())}), lambda_APS = "
        f"{settings.lam_aps:g}, lambda_L = {spec.uniformity_weight:g}")
    values: dict = {}
    owners: dict = {}
    seen = set()
    for k, ev in enumerate(evals, start=1):
        if stop() or not ev.n or ev.n != n_req:
            continue
        key = np.round(np.sort(ev.xyz[:, :2], axis=0), 4).tobytes()
        if key in seen:
            continue
        seen.add(key)
        name = f"search_{k}"
        say(f"formal ranking: APS on {len(sset)} scenarios for {name}")
        values[name] = evaluate_layout_value(case, ev.xyz, sset, name, settings=settings, progress=say, cancel=stop,
                                             tags={"kind": "search", "search_rank": k})
        owners[name] = ev
    base_note = None
    if spec.baseline == "initial" and not stop():
        b = baseline_layout(case, n_req)
        if b is not None:
            base_note = b[2]
            say(f"formal ranking: APS on {len(sset)} scenarios for the initial field")
            values["initial"] = evaluate_layout_value(case, b[0], sset, "initial", ids=b[1], settings=settings,
                                                      progress=say, cancel=stop,
                                                      tags={"kind": "initial", "note": base_note})
            owners["initial"] = None
        else:
            base_note = "initial field not comparable (fewer heliostats than N or not buildable)"
    swaps = []
    search_names = [k for k in values if k != "initial"]
    if search_names and spec.formal_swap_rounds > 0:
        lead = max(search_names, key=lambda k: values[k].totals()["value_mwh"])
        ev, cur = owners[lead], values[lead]
        m = max(1, int(round(spec.swap_fraction * n_req)))
        for r in range(spec.formal_swap_rounds):
            if stop() or ev.candidates_xyz is None:
                break
            key = {tuple(np.round(p, 6)) for p in cur.xyz}
            in_layout = np.array([tuple(np.round(p, 6)) in key for p in ev.candidates_xyz])
            unused = np.flatnonzero(~in_layout)
            if unused.size == 0:
                break
            price = annual(case, Field(ev.candidates_xyz), samples=samples, aiming="center", with_flux=False,
                           shading_points=spec.shading_points, workers=1, blockers=in_layout).heliostat_energy_mwh
            add = unused[np.argsort(-price[unused])[:m]]
            drop = np.argsort(cur.mirror_energy_mwh(), kind="stable")[:len(add)]
            new_xyz = np.vstack([np.delete(cur.xyz, drop, axis=0), ev.candidates_xyz[add]])
            name = f"{lead}+swap{r + 1}"
            say(f"formal swap round {r + 1}: {len(add)} heliostats, APS on {len(sset)} scenarios")
            new = evaluate_layout_value(case, new_xyz, sset, name, settings=settings, progress=say, cancel=stop,
                                        tags={"kind": "search_swap", "search_rank": cur.tags.get("search_rank")})
            gain = new.totals()["value_mwh"] - cur.totals()["value_mwh"]
            swaps.append({"round": r + 1, "layout": name, "swapped": int(len(add)), "gain_value_mwh": float(gain),
                          "gain_energy_mwh": new.totals()["energy_mwh"] - cur.totals()["energy_mwh"],
                          "accepted": bool(gain > 0)})
            if gain > 0:
                values[name], owners[name], cur = new, ev, new
            else:
                break
    if not values:
        return {"scenario_set": sset, "settings": settings, "values": {}, "owners": {}, "winner": None,
                "ranking": pd.DataFrame(), "swaps": swaps, "baseline_note": base_note}
    ranking = compare_values(list(values.values()), reference="initial" if "initial" in values else None)
    winner = str(ranking.loc[0, "name"]) if bool(ranking.loc[0, "rankable"]) else None
    out = {"scenario_set": sset, "settings": settings, "values": values, "owners": owners, "winner": winner,
           "ranking": ranking, "swaps": swaps, "baseline_note": base_note}
    if winner is not None and "initial" in values and winner != "initial":
        out["displacement"] = displacement(values["initial"].xyz, values[winner].xyz)
    if (spec.cross_check_maps and case.scenarios.map_model == "clouds" and winner is not None
            and "initial" in values and winner != "initial" and not stop()):
        other = case.copy()
        other.scenarios.seed += 1000
        cross_set = build_scenarios(other, hourly=hourly)
        rows = []
        for name in (winner, "initial"):
            say(f"independent cloud maps: {name} on {len(cross_set)} scenarios")
            cv = evaluate_layout_value(other, values[name].xyz, cross_set, name, ids=values[name].ids,
                                       settings=settings, progress=say, cancel=stop, keep_flux=False)
            rows.append({"name": name, "value_mwh": cv.totals()["value_mwh"], "energy_mwh": cv.totals()["energy_mwh"],
                         "penalty_mwh": cv.totals()["penalty_mwh"],
                         "value_formal_mwh": values[name].totals()["value_mwh"]})
        lead = rows[0]["value_mwh"] - rows[1]["value_mwh"]
        out["cross_maps"] = {"seed": other.scenarios.seed, "fingerprint": cross_set.fingerprint(), "rows": rows,
                             "lead_formal_mwh": rows[0]["value_formal_mwh"] - rows[1]["value_formal_mwh"],
                             "lead_independent_mwh": lead, "winner_still_ahead": bool(lead > 0)}
    if spec.dense_check_scenarios > 0 and winner is not None and not stop():
        dense_set = build_scenarios(case, spec.dense_check_scenarios, hourly)
        rows = []
        for name in dict.fromkeys([winner, "initial", *[n for n in ranking["name"] if n not in (winner, "initial")][:1]]):
            if name not in values or stop():
                continue
            say(f"sampling check: {name} on {len(dense_set)} scenarios")
            dv = evaluate_layout_value(case, values[name].xyz, dense_set, name, ids=values[name].ids,
                                       settings=settings, progress=say, cancel=stop, keep_flux=False)
            a, b = values[name].totals(), dv.totals()
            rows.append({"name": name, "value_rep_mwh": a["value_mwh"], "value_dense_mwh": b["value_mwh"],
                         "energy_rep_mwh": a["energy_mwh"], "energy_dense_mwh": b["energy_mwh"],
                         "penalty_rep_mwh": a["penalty_mwh"], "penalty_dense_mwh": b["penalty_mwh"],
                         "value_error_pct": 100 * (a["value_mwh"] / b["value_mwh"] - 1) if b["value_mwh"] else None,
                         "energy_error_pct": 100 * (a["energy_mwh"] / b["energy_mwh"] - 1) if b["energy_mwh"] else None})
        order_rep = [r["name"] for r in sorted(rows, key=lambda r: -r["value_rep_mwh"])]
        order_dense = [r["name"] for r in sorted(rows, key=lambda r: -r["value_dense_mwh"])]
        out["dense"] = {"n_scenarios": len(dense_set), "fingerprint": dense_set.fingerprint(), "rows": rows,
                        "same_order": order_rep == order_dense, "order_representative": order_rep,
                        "order_dense": order_dense}
    return out


def formal_summary(formal: dict) -> dict:
    """JSON-ready digest of a formal ranking."""
    sset, settings = formal["scenario_set"], formal["settings"]
    ranking = formal["ranking"]
    rows = []
    for _, r in ranking.iterrows():
        rows.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()
                     if k in ("rank", "name", "value_mwh", "energy_mwh", "penalty_mwh", "weighted_penalty_mwh",
                              "delta_value_mwh", "delta_energy_mwh", "delta_uniformity_mwh", "complete",
                              "inner_gap_annual", "max_inner_gap", "status_counts", "status_hours", "circ_penalty_mwh",
                              "cv_2d_power_weighted", "kind", "note", "reference")})
    win = formal.get("winner")
    out = {"scenario_set": {k: v for k, v in sset.record().items() if k != "map_records"},
           "inner_settings": settings.record(), "lambda_aps": settings.lam_aps,
           "lambda_layout": float(ranking["lambda_layout"].iloc[0]) if len(ranking) else None,
           "ranking": rows, "winner": win, "swaps": formal.get("swaps", []),
           "baseline_note": formal.get("baseline_note"), "dense_check": formal.get("dense"),
           "cross_maps": formal.get("cross_maps"),
           "winner_totals": formal["values"][win].totals() if win else {"value_mwh": float("nan"), "energy_mwh": 0.0,
                                                                         "status_counts": {}}}
    d = formal.get("displacement")
    if d and d.get("comparable"):
        out["displacement_vs_initial"] = {k: d[k] for k in ("mean_move_m", "median_move_m", "max_move_m",
                                                             "n_unchanged", "n_moved")}
    return out
