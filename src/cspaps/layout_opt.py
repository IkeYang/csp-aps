"""Outer loop: heliostat field layout optimisation.

Objective: maximise the annual energy on the receiver for a fixed number of
heliostats (``layout_opt.n_heliostats``) inside the site (boundary, exclusion
zone, minimum spacing).

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
from dataclasses import dataclass, field as dc_field
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

    @property
    def n(self) -> int:
        return int(len(self.xyz))


def make_samples(case: Case, method: str, n_bins: int, dni_bands: int = 1, hourly: sun.Samples | None = None,
                 seed: int = 0) -> sun.Samples:
    hourly = hourly if hourly is not None else sun.hourly_samples(case)
    if method == "hourly":
        return hourly
    if method == "sun_grid":
        a = case.annual
        return sun.sun_grid(hourly, a.grid_elevation_step_deg, a.grid_azimuth_step_deg, dni_bands)
    return sun.sun_bins(hourly, n_bins, seed=seed, dni_bands=dni_bands)


def evaluate_layout(
    case: Case,
    kind: str,
    params: dict,
    samples: sun.Samples,
    n_select: int | None,
    shading_points: int | None = None,
    aiming: str = "center",
) -> LayoutEval:
    try:
        xyz = generate(case, kind, params)
    except ValueError:
        return LayoutEval(params, 0, np.zeros((0, 3)), 0.0)
    if len(xyz) == 0:
        return LayoutEval(params, 0, xyz, 0.0)
    center_all = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                        shading_points=shading_points, workers=1)
    cand_xyz, cand_e = xyz, center_all.heliostat_energy_mwh
    n_cand = len(xyz)
    selected = bool(n_select and n_cand > n_select)
    if selected:
        # Center-aim single-mirror potential cheaply seeds a fixed-size candidate layout.
        keep = np.sort(np.argsort(-center_all.heliostat_energy_mwh, kind="stable")[:n_select])
        xyz = xyz[keep]
    center_res = (annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                         shading_points=shading_points, workers=1) if selected else center_all)
    center_energy = center_res.summary()["absorbed_energy_mwh"]
    if aiming == "center":
        return LayoutEval(params, n_cand, xyz, center_energy, center_res, cand_xyz, cand_e,
                          center_energy_mwh=center_energy, search_aiming="center")
    res = annual(case, Field(xyz), samples=samples, aiming=aiming, with_flux=True,
                 shading_points=shading_points, workers=1)
    aps_energy = res.summary()["feasible_absorbed_energy_mwh"]
    return LayoutEval(params, n_cand, xyz, aps_energy, res, cand_xyz, cand_e,
                      aps_energy_mwh=aps_energy, aps_result=res, center_energy_mwh=center_energy,
                      search_aiming=aiming)


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
        return -ev.energy_mwh


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

    @property
    def field(self) -> Field:
        return Field(self.best.xyz)

    @property
    def status(self) -> str:
        return self.validation.get("status", "unknown")

    def summary(self) -> dict:
        b = self.best
        aps_summary = b.aps_result.summary() if b.aps_result is not None else None
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
            "runtime_s": round(self.runtime_s, 1),
        }


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
    hourly = sun.hourly_samples(case)
    search_aiming = spec.search_aiming
    if search_aiming == "auto":
        search_aiming = "greedy" if case.uniformity.enabled else "center"
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
    use_aps = spec.refine_top > 0 and review_aiming != "center"
    evals: list[LayoutEval] = []
    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        params = {**spec.fixed, **{k: float(row[k]) for k in names}}
        ev = evaluate_layout(case, kind, params, samples, spec.n_heliostats, spec.shading_points,
                             aiming=search_aiming)
        info = {**{k: params[k] for k in names}, "search_rank": rank, "n_heliostats": ev.n,
                "n_candidates": ev.n_candidates, "search_aiming": search_aiming,
                "energy_search_objective_mwh": ev.energy_mwh,
                "energy_search_center_mwh": ev.center_energy_mwh,
                "energy_search_center_absorbed_mwh": ev.center_energy_mwh}
        if ev.n and spec.refine_top > 0:
            say(f"review {rank}/{len(top)}: centre aiming on {len(review_center)} states")
            rc = annual(case, Field(ev.xyz), samples=review_center, aiming="center", grid=grid,
                        shading_points=spec.shading_points)
            info["energy_review_center_mwh"] = rc.summary()["absorbed_energy_mwh"]
            info["constrained_states_center"] = int(rc.timeseries["over_limit"].sum())
            info["hours_over_limit_center"] = float(rc.timeseries.loc[rc.timeseries["over_limit"], "weight_h"].sum())
            if spec.review_check_hourly:
                say(f"review {rank}/{len(top)}: hourly year with centre aiming (sampling error)")
                rh = annual(case, Field(ev.xyz), samples=hourly, aiming="center", grid=grid, with_flux=False,
                            shading_points=spec.shading_points)
                e_h = rh.summary()["absorbed_energy_mwh"]
                info["energy_hourly_center_mwh"] = e_h
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
        evals.append(ev)
        rows.append(info)
    ranking = pd.DataFrame(rows)
    key_col = "energy_constrained_aps_mwh" if "energy_constrained_aps_mwh" in ranking else "energy_search_objective_mwh"
    ranking = ranking.sort_values(key_col, ascending=False).reset_index(drop=True)
    ranking["final_rank"] = np.arange(1, len(ranking) + 1)
    ranking["rank_change"] = ranking["search_rank"] - ranking["final_rank"]
    best_rank = int(ranking.loc[0, "search_rank"]) - 1
    best = evals[best_rank]

    # ---- swap refinement on the winner (accepted on constrained annual energy)
    swaps = []
    if use_aps and best.aps_result is not None and best.candidates_xyz is not None and spec.swap_rounds > 0:
        cur_xyz, cur_res = best.xyz, best.aps_result
        m = max(1, int(round(spec.swap_fraction * len(cur_xyz))))
        for r in range(spec.swap_rounds):
            if stop():
                break
            sel = {tuple(np.round(p, 6)) for p in cur_xyz}
            unused = np.array([i for i, p in enumerate(best.candidates_xyz) if tuple(np.round(p, 6)) not in sel])
            if unused.size == 0:
                break
            add = unused[np.argsort(-best.candidate_energy[unused])[:m]]
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
            ranking = ranking.sort_values("energy_constrained_aps_mwh", ascending=False).reset_index(drop=True)
            ranking["final_rank"] = np.arange(1, len(ranking) + 1)
            ranking["rank_change"] = ranking["search_rank"] - ranking["final_rank"]

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
    return LayoutOptResult(best, history, ranking, samples, _time.perf_counter() - t0, validation, swaps,
                           review_aps if use_aps else review_center, capacity, evals)
