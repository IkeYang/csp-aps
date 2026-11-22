"""Solver benchmarks under the same conditions (improvement plan, "在相同条件下比较求解器").

Inner (task 1, one layout): the same scenario states, the same candidate model, grid, margin and weight,
solved by

  greedy           the current APS (greedy + fine-grid verification + repair + joint-pose recheck)
  milp             fixed-response 0/1 programme (HiGHS, time limit = budget) + the same verification / recheck
  subfield         HALOS-style decomposition: azimuth sectors solved independently on their share of the limit
  greedy_rounding  greedy + relaxation-guided rounding and a dual bound (iteration budget)

Every plan is verified on the fine grid; only "feasible" plans count. The common reference per state is the
lowest joint-pose upper bound any method produced. Fixed-response duals are diagnostic only. Every method
is measured against the same number.

Outer (layout): the same site, N, scenario set, weights and wall-time budget for

  de               the existing parametric search (differential evolution on the generator parameters,
                   centre-aim pre-screen, selection polish, position refinement)
  multistart_free  multi-start free-position optimisation: from several starts, every mirror moves along the
                   gradient of the lone-mirror annual energy (physics gradient, ``certify.potential_map``) with a
                   repulsion from close neighbours; a move is kept when the centre-aim surrogate improves, and
                   site / spacing constraints hold after every move

Both outputs are scored by the formal value V(L) on the training scenario set and on a held-out set (other
seed: other representative sun states and other cloud maps). Several seeds give the run-to-run spread.
"""

from __future__ import annotations

import time as _time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import Case
from .field import Field, site_mask

INNER_METHODS = ("greedy", "milp", "subfield", "greedy_rounding")
INNER_ZH = {"greedy": "贪心 APS（现行）", "milp": "固定响应整数规划", "subfield": "子场分解（可并行）",
            "greedy_rounding": "贪心 + 松弛取整"}
OUTER_METHODS = ("de", "multistart_free")
OUTER_ZH = {"de": "差分进化（现行）", "multistart_free": "多起点自由镜位"}


# --------------------------------------------------------------------------- inner

def inner_states(case: Case, fld: Field, n_states: int | None = None, sset=None) -> list[dict]:
    """Scenario states of the formal set (largest h_s x DNI first): label, sun, per-mirror DNI."""
    from .scenarios import build_scenarios

    sset = sset or build_scenarios(case)
    s = sset.samples
    order = np.argsort(-s.weight_h * s.dni, kind="stable")
    if n_states:
        order = order[:int(n_states)]
    return [{"label": sset.ids[i], "elevation": float(s.elevation[i]), "azimuth": float(s.azimuth[i]),
             "dni": sset.dni_at(i, fld.xyz[:, :2]), "hours": float(s.weight_h[i]), "map_id": sset.map_id(i)}
            for i in order]


def inner_benchmark(case: Case, fld: Field, states: list[dict], methods=INNER_METHODS, budget_s: float = 60.0,
                    lam_aps: float | None = None, progress=None) -> pd.DataFrame:
    from .layout_value import inner_settings, solve_instant

    say = progress or (lambda m: None)
    base = inner_settings(case, lam_aps)
    c = case.copy()
    c.aps.milp_time_limit_s = float(budget_s)
    c.aps.milp_max_nnz = max(int(c.aps.milp_max_nnz), 20_000_000)  # let the exact method try on the full field
    rows = []
    for st in states:
        for m in methods:
            if m == "greedy_rounding":
                settings = replace(base, method="greedy", certify_rounds=8, certify_iters=300)
            else:
                settings = replace(base, method=m)
            say(f"inner benchmark {st['label']}: {m}")
            t0 = _time.perf_counter()
            try:
                p = solve_instant(c, fld, st["elevation"], st["azimuth"], st["dni"], settings, keep_flux=False)
            except ValueError as exc:  # e.g. MILP too large for the candidate model
                rows.append({"state": st["label"], "method": m, "status": "not_run", "note": str(exc)[:160],
                             "time_s": _time.perf_counter() - t0, "hours": st["hours"], "within_budget": False})
                continue
            t = _time.perf_counter() - t0
            info = p.info
            # subfield: the sectors are independent; the ideal parallel wall time replaces their serial sum
            par = (t - info["subfield_serial_s"] + info["subfield_parallel_s"]) if "subfield_serial_s" in info else None
            rows.append({
                "state": st["label"], "method": m, "status": p.status, "flux_status": p.flux_status,
                "objective_mw": p.objective_w / 1e6, "p_abs_mw": p.p_abs_w / 1e6, "u_mw": p.u_w / 1e6,
                "upper_bound_mw": p.upper_bound_w / 1e6 if p.upper_bound_w is not None else None,
                "own_gap": p.gap, "bound_method": p.bound_method,
                "dual_bound_mw": info.get("dual_bound_mw"), "peak_to_allowed": p.peak_ratio,
                "n_defocused": p.n_defocused, "time_s": t, "parallel_time_s": par,
                "within_budget": bool(t <= budget_s * 1.05 + 1.0), "hours": st["hours"], "map_id": st["map_id"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Fixed-response duals price a different shading geometry. Only joint-pose bounds are comparable.
    for col in ("upper_bound_mw", "objective_mw", "within_budget"):
        if col not in df:
            df[col] = np.nan
    df["state_best_bound_mw"] = df.groupby("state")["upper_bound_mw"].transform("min")
    counts = df["status"].isin(("feasible", "feasible_shutdown"))
    incumbent = df["objective_mw"].where(counts).groupby(df["state"]).transform("max")
    invalid = df["state_best_bound_mw"] < incumbent - 1e-9 * incumbent.abs().clip(lower=1)
    if invalid.any():
        raise ArithmeticError(f"Joint-pose bound below a verified plan in {list(df.loc[invalid, 'state'].unique())}")
    df["state_best_bound_mw"] = np.maximum(df["state_best_bound_mw"], incumbent.fillna(-np.inf))
    df["gap_to_best_bound"] = np.where(counts, (df["state_best_bound_mw"] - df["objective_mw"])
                                       / df["state_best_bound_mw"].abs().clip(lower=1e-12), np.nan)
    best = df["objective_mw"].where(counts).groupby(df["state"]).transform("max")
    df["loss_to_best_found"] = np.where(counts, (best - df["objective_mw"]) / best.abs().clip(lower=1e-12), np.nan)
    return df


def inner_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: share of feasible states, mean / max gap to the common bound, time."""
    if df.empty:
        return df
    g = df.groupby("method", sort=False)
    out = pd.DataFrame({
        "states": g.size(),
        "feasible": g["status"].apply(lambda s: int((s == "feasible").sum())),
        "objective_sum_mw": g["objective_mw"].sum(min_count=1),
        "hours_weighted_objective_mwh": g.apply(lambda d: float((d["objective_mw"].fillna(0) * d["hours"]
                                                                  * d["status"].eq("feasible")).sum())),
        "mean_gap_to_best_bound": g["gap_to_best_bound"].mean(),
        "max_gap_to_best_bound": g["gap_to_best_bound"].max(),
        "mean_loss_to_best_found": g["loss_to_best_found"].mean(),
        "time_s": g["time_s"].sum(),
        "all_within_budget": g["within_budget"].all() if "within_budget" in df else None,
    }).reset_index()
    out["方法"] = out["method"].map(INNER_ZH).fillna(out["method"])
    return out


# --------------------------------------------------------------------------- outer

def surrogate_score(case: Case, xyz: np.ndarray, samples) -> float:
    """Centre-aim annual score of the layout search (energy - lambda_L x spread), MWh."""
    from . import layout_uniformity as lu
    from .layout_opt import _spread_mwh, _surface_score
    from .simulate import annual

    spec = case.layout_opt
    lam = float(spec.uniformity_weight)
    full = lam > 0 and _surface_score(case)
    res = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=full,
                 shading_points=spec.shading_points, workers=1, keep_state_power=lam > 0, check_flux=False)
    energy = res.summary()["absorbed_energy_mwh"]
    if lam <= 0:
        return float(energy)
    if full:
        return float(energy - lam * _spread_mwh(res, case))
    uni = lu.layout_metrics(case, xyz, samples, res.state_power_w, lam)
    return float(energy - lam * uni["circ_spread_mwh"])


def _spacing(case: Case) -> float:
    from .layout_opt import design_spacing

    return design_spacing(case)


def _potential_gradient(pmap):
    """Gradient (MWh/m) of the lone-mirror annual energy on the site raster (outside cells: the site minimum)."""
    e = np.asarray(pmap.energy_mwh, float)
    e = np.where(np.isfinite(e), e, np.nanmin(e) if np.isfinite(e).any() else 0.0)
    gy, gx = np.gradient(e, pmap.y, pmap.x)
    return gx, gy


def _sample_grad(pmap, gx, gy, xy):
    ix = np.clip(np.round((xy[:, 0] - pmap.x[0]) / pmap.cell_m).astype(int), 0, pmap.x.size - 1)
    iy = np.clip(np.round((xy[:, 1] - pmap.y[0]) / pmap.cell_m).astype(int), 0, pmap.y.size - 1)
    return np.column_stack([gx[iy, ix], gy[iy, ix]])


def free_position_search(case: Case, start_xyz: np.ndarray, samples, pmap, rng, deadline: float,
                         max_evals: int | None = None, log=None) -> tuple[np.ndarray, float, int]:
    """Physics-gradient local search on free mirror positions (site and spacing kept after every move)."""
    spacing = _spacing(case)
    gx, gy = _potential_gradient(pmap)
    xyz = np.asarray(start_xyz, float).copy()
    best = surrogate_score(case, xyz, samples)
    evals = 1
    step = 0.5 * spacing
    n = len(xyz)
    while _time.perf_counter() < deadline and (max_evals is None or evals < max_evals):
        if step < 0.02 * spacing:  # converged: kick (iterated local search) and continue from the incumbent
            step = 0.5 * spacing
        g = _sample_grad(pmap, gx, gy, xyz[:, :2])
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        tree = cKDTree(xyz[:, :2])
        d, nb = tree.query(xyz[:, :2], k=min(n, 5))
        push = np.zeros((n, 2))
        if n > 1:
            for col in range(1, d.shape[1]):  # repulsion from neighbours closer than 1.5 x spacing
                close = d[:, col] < 1.5 * spacing
                v = xyz[:, :2] - xyz[nb[:, col], :2]
                w = np.where(close, (1.5 * spacing - d[:, col]) / (0.5 * spacing), 0.0)
                push += w[:, None] * v / np.maximum(d[:, col], 1e-9)[:, None]
        direction = g + 0.5 * push + 0.3 * rng.normal(size=(n, 2))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        moving = rng.random(n) < 0.3
        trial = xyz.copy()
        trial[moving, :2] += step * direction[moving]
        ok_site = site_mask(case, trial)[0]
        trial[~ok_site] = xyz[~ok_site]
        # spacing: undo moves that create a close pair (moved mirrors first), until none is left
        for _ in range(5):
            pairs = np.array(sorted(cKDTree(trial[:, :2]).query_pairs(spacing * (1 - 1e-9))), dtype=int).reshape(-1, 2)
            if not len(pairs):
                break
            undo = np.unique(np.where(moving[pairs[:, 0]], pairs[:, 0], pairs[:, 1]))
            trial[undo] = xyz[undo]
        if cKDTree(trial[:, :2]).query_pairs(spacing * (1 - 1e-9)) or np.allclose(trial, xyz):
            step *= 0.7
            continue
        score = surrogate_score(case, trial, samples)
        evals += 1
        if score > best:
            xyz, best = trial, score
            if log:
                log(f"free positions: {best:.1f} MWh (step {step:.2f} m)")
        else:
            step *= 0.7
    return xyz, best, evals


def _multistart_free(case: Case, budget_s: float, seed: int, samples, pmap, log=None, n_starts: int = 3,
                     max_evals: int | None = None) -> dict:
    from .layout_opt import DEFAULT_BOUNDS, evaluate_layout
    from .layout_value import baseline_layout

    spec = case.layout_opt
    n = int(spec.n_heliostats)
    rng = np.random.default_rng(seed)
    t0 = _time.perf_counter()
    starts = []
    b = baseline_layout(case, n)
    if b is not None:
        starts.append(("initial", b[0]))
    bounds = dict(DEFAULT_BOUNDS.get(spec.generator, {}))
    bounds.update(spec.bounds or {})
    for k in spec.fixed:
        bounds.pop(k, None)
    tries = 0
    while len(starts) < n_starts and tries < 4 * n_starts and _time.perf_counter() - t0 < 0.3 * budget_s:
        tries += 1
        params = dict(spec.fixed)
        params.update({k: float(rng.uniform(*v)) for k, v in bounds.items()})
        ev = evaluate_layout(case, spec.generator, params, samples, n, spec.shading_points)
        if ev.n == n:
            starts.append((f"generator {tries}", ev.xyz))
    best = None
    evals = 0
    for j, (label, xyz0) in enumerate(starts):
        left = budget_s - (_time.perf_counter() - t0)
        if left <= 0:
            break
        deadline = _time.perf_counter() + left / max(1, len(starts) - j)
        per = None if max_evals is None else max(2, (max_evals - evals) // max(1, len(starts) - j))
        xyz, score, k = free_position_search(case, xyz0, samples, pmap, rng, deadline, per, log)
        evals += k
        if best is None or score > best[1]:
            best = (xyz, score, label)
    if best is None:
        raise RuntimeError("multi-start free-position search found no start layout with N heliostats")
    return {"xyz": best[0], "surrogate_mwh": best[1], "start": best[2], "evaluations": evals,
            "n_starts": len(starts), "time_s": _time.perf_counter() - t0}


def _de(case: Case, budget_s: float, seed: int) -> dict:
    from .layout_opt import optimize_layout

    c = case.copy()
    lo = c.layout_opt
    lo.seed = int(seed)
    lo.maxiter = 10_000  # the wall-time budget stops the search (then the best layout is reviewed)
    lo.refine_top, lo.formal_swap_rounds, lo.dense_check_scenarios = 0, 0, 0
    lo.review_check_hourly, lo.upper_bound = False, False
    t0 = _time.perf_counter()
    # stop the population search at 80 % of the budget: the selection polish / position review of the best
    # layout belongs to the method and runs after it
    res = optimize_layout(c, cancel=lambda: _time.perf_counter() - t0 > 0.8 * budget_s)
    return {"xyz": res.best.xyz, "surrogate_mwh": float(res.best.energy_mwh), "start": "sobol population",
            "evaluations": int(len(res.history)), "n_starts": 1, "time_s": _time.perf_counter() - t0}


def outer_benchmark(case: Case, methods=OUTER_METHODS, seeds=(1, 2), budget_s: float = 120.0,
                    held_out_seed_offset: int = 1000, held_out_extra: int = 5, max_evals: int | None = None,
                    progress=None) -> dict:
    """Both layout searches under the same wall-time budget, scored by V(L) on training and held-out sets."""
    from . import sun
    from .certify import potential_map
    from .layout_opt import make_samples, validate_layout
    from .layout_value import evaluate_layout_value, inner_settings
    from .scenarios import attach_regimes, build_scenarios, validation_scenarios

    say = progress or (lambda m: None)
    spec = case.layout_opt
    if not spec.n_heliostats:
        raise ValueError("layout_opt.n_heliostats must be set for the layout benchmark")
    hourly = (build_scenarios(case).samples if case.scenarios.source == "file"
              else attach_regimes(case, sun.hourly_samples(case)))
    samples = make_samples(case, spec.search_sampling, spec.n_sun_bins, 1, hourly, spec.seed)
    pmap = potential_map(case, samples)
    train = build_scenarios(case, hourly=hourly)
    held, validation = validation_scenarios(case, train, len(train) + held_out_extra, hourly, held_out_seed_offset)
    settings = inner_settings(case)
    spacing = _spacing(case)
    rows = []
    layouts = {}
    for m in methods:
        for sd in seeds:
            say(f"layout benchmark: {m}, seed {sd}")
            if m == "de":
                r = _de(case, budget_s, sd)
            elif m == "multistart_free":
                r = _multistart_free(case, budget_s, sd, samples, pmap, max_evals=max_evals)
            else:
                raise ValueError(f"unknown layout method {m!r}")
            name = f"{m}_s{sd}"
            val = validate_layout(case, r["xyz"], spec.n_heliostats, spacing)
            say(f"layout benchmark: V(L) of {name} on {len(train)} + {len(held)} scenarios")
            v_train = evaluate_layout_value(case, r["xyz"], train, name, settings=settings, keep_flux=False)
            tt = v_train.totals()
            if validation["new_states"]:
                th = evaluate_layout_value(case, r["xyz"], held, name, settings=settings, keep_flux=False).totals()
            else:
                th = {"value_mwh": np.nan, "coverage": np.nan}
            layouts[name] = r["xyz"]
            rows.append({"name": name, "method": m, "seed": sd, "time_s": r["time_s"], "evaluations": r["evaluations"],
                         "start": r["start"], "surrogate_mwh": r["surrogate_mwh"], "site_status": val["status"],
                         "min_distance_m": val["min_distance_m"], "outside_site": val["outside_site"],
                         "pairs_too_close": val["pairs_too_close"],
                         "value_mwh": tt["value_mwh"], "energy_mwh": tt["energy_mwh"], "penalty_mwh": tt["penalty_mwh"],
                         "coverage": tt.get("coverage"), "ranking_status": tt.get("ranking_status"),
                         "held_out_value_mwh": th["value_mwh"], "held_out_coverage": th.get("coverage")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank_train"] = df["value_mwh"].where(df.site_status == "ok").rank(ascending=False, method="min")
        df["rank_held_out"] = df["held_out_value_mwh"].where(df.site_status == "ok").rank(ascending=False, method="min")
    summary = (df.groupby("method", sort=False)
               .agg(runs=("name", "size"), value_mean_mwh=("value_mwh", "mean"), value_std_mwh=("value_mwh", "std"),
                    held_out_mean_mwh=("held_out_value_mwh", "mean"), held_out_std_mwh=("held_out_value_mwh", "std"),
                    best_rank_held_out=("rank_held_out", "min"), all_sites_ok=("site_status", lambda s: bool((s == "ok").all())),
                    time_mean_s=("time_s", "mean"), evaluations_mean=("evaluations", "mean"))
               .reset_index()) if not df.empty else df
    if not df.empty:
        summary["方法"] = summary["method"].map(OUTER_ZH).fillna(summary["method"])
    return {"runs": df, "summary": summary, "layouts": layouts, "budget_s": budget_s, "seeds": list(seeds),
            "train": {"n": len(train), "fingerprint": train.fingerprint()},
            "held_out": validation,
            "settings_digest": settings.digest()}
