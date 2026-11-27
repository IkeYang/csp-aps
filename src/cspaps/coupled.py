"""Offline joint-pose search and bounds for the executed heliostat geometry."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import prod

import numpy as np

from . import optics, sun
from .simulate import evaluate
from .uniformity import uniformity_metrics


def _unshaded_beams(prob, field):
    """Beam and intercepted power at current DNI, with all shading/blocking factors set to one."""
    case = prob.case
    if prob.rs.sun_elevation <= 0:
        zeros = np.zeros((prob.n, prob.k))
        return zeros, zeros
    direction = sun.sun_vector(prob.rs.sun_elevation, prob.rs.sun_azimuth)
    _, t, slant, cosine = optics.heliostat_geometry(field.xyz[:, None, :], prob.aim_xyz, direction)
    h = case.heliostat
    beam = prob.dni_obj[:, None] * h.area * h.reflectivity * h.cleanliness * cosine * optics.attenuation(slant, case.atmosphere)
    return beam, beam * optics.intercept_fraction(case.receiver, prob.aim_xyz, t, prob.sigma)


def _weighted_upper(prob, beam, intercepted, allowed, snap, weight):
    """Supporting hyperplane of the flux-spread norm, relaxed over joint shading factors.

    For v of zero mean and norm <= 1, ||q_H - mean(q_H)|| >= v.q_H.
    Relax each joint blocking factor to an independent number in [0,1].
    The positive quadrature normalisation of any mixture is a weighted mean
    of its individual intercepted / sampled power ratios, hence <= max ratio.
    """
    grid = prob.grid
    mask = snap.region
    q = snap.absorbed_flux.ravel()[mask]
    direction = q - q.mean() if q.size else q
    length = np.linalg.norm(direction)
    if not length:
        return np.inf
    direction /= length
    support = grid.alpha.copy()
    support[mask] *= 1 - weight * np.sqrt(q.size) * direction
    largest_ratio, linear = 0.0, 0.0
    from .backend import enabled, row_stats_gpu

    gpu = (row_stats_gpu(grid.points, grid.normals, prob.aim_xyz.reshape(-1, 3), prob.rs.t.reshape(-1, 3),
                         prob.sigma.ravel(), beam.ravel(), support) if enabled() else None)
    if gpu is not None:  # the same bound with the image sums formed on the GPU
        sample_power = gpu[0].reshape(prob.n, prob.k) * grid.cell_area
        coeff = gpu[1].reshape(prob.n, prob.k) * grid.cell_area
        lit = allowed & (intercepted > 0)
        if np.any(lit & (sample_power <= 0)):
            return np.inf
        if np.any(lit):
            largest_ratio = float((intercepted[lit] / sample_power[lit]).max())
        linear = float(np.maximum(np.where(allowed, coeff, 0.0).max(axis=1), 0.0).sum())
        return linear * largest_ratio + 1e-10 * max(1.0, float(beam.sum()))
    stride = max(1, 1_000_000 // (prob.k * len(grid.alpha)))
    for start in range(0, prob.n, stride):
        sl = slice(start, start + stride)
        images = optics.footprints(grid, prob.aim_xyz[sl].reshape(-1, 3), prob.rs.t[sl].reshape(-1, 3),
                                   prob.sigma[sl].ravel(), beam[sl].ravel())
        images = images.reshape(len(beam[sl]), prob.k, len(grid.alpha))
        sample_power = images.sum(axis=2) * grid.cell_area
        lit = allowed[sl] & (intercepted[sl] > 0)
        if np.any(lit & (sample_power <= 0)):
            return np.inf
        if np.any(lit):
            largest_ratio = max(largest_ratio, float((intercepted[sl][lit] / sample_power[lit]).max()))
        coeff = (images @ support) * grid.cell_area
        linear += float(np.maximum(np.where(allowed[sl], coeff, 0.0).max(axis=1), 0.0).sum())
    return linear * largest_ratio + 1e-10 * max(1.0, float(beam.sum()))


def search_joint_poses(prob, field, seed, *, warm=None, exact_limit=None, trials=None, shading_points=None,
                       tighten_bound=True, tracking_reference=None, move_cost=None):
    """Evaluate joint poses with their actual neighbours. Certify the sampled model.

    ``move_cost`` (N, K) W, from ``aps.move_cost``: plans are selected by the objective minus their move price,
    so the search keeps the plan near its initial state unless a move pays. The certificate stays one of the
    objective without the price: the exhaustive bound is the best unpriced objective over all joint poses,
    the gap is that of the selected plan's unpriced objective (what staying near the initial state costs).

    A pose counts as verified when its fine-grid peak is within the allowed level (limit x (1 - margin)).
    The solver may end its plan a hair above that level after its verification rounds ("margin_reduced":
    above the allowed level, below the limit), and the rest of the pipeline accepts such a plan. The bounded
    (non-exhaustive) search then accepts poses up to the seed's own verified peak, never more: the result uses
    no more of the safety margin than the solver's plan and has at least its objective, instead of a nearly
    empty plan that passes the stricter check only because it carries almost no flux. The exhaustive search
    keeps the strict rule (it finds the best pose within the margin).

    The independent, unshaded per-mirror maxima dominate every feasible combination:
    shading factors lie in [0,1], and the uniformity penalty is nonnegative. For small
    products visit all choices. For large products keep the best feasible incumbent
    from the proposal solver, warm start and a bounded neighbourhood search.
    """
    case, rs = prob.case, prob.rs
    spec, uni = case.aps, case.uniformity
    exact_limit = spec.coupled_exact_limit if exact_limit is None else exact_limit
    trials = spec.coupled_trials if trials is None else trials
    mode = uni.mode if uni.enabled else "power"
    allowed = np.broadcast_to(prob.available[:, None], (prob.n, prob.k)).copy()
    if prob.reach is not None:
        allowed &= prob.reach
    choices = []
    for i, mask in enumerate(allowed):
        row = np.flatnonzero(mask).tolist()
        if spec.allow_defocus or not prob.available[i]:
            row.append(-1)
        choices.append(row)
    count = prod(map(len, choices))
    exact = count <= exact_limit
    beam, intercepted = _unshaded_beams(prob, field)
    unshaded = intercepted * max(case.receiver.panel_absorptivity or [case.receiver.absorptivity])
    best_single = np.where(allowed, unshaded, 0.0).max(axis=1)
    upper = float(best_single.sum())
    minimum = uni.min_load_mw * 1e6 if uni.enabled else 0.0
    impossible_load = minimum > upper + 1e-7
    seen = set()
    best = None
    best_value = -np.inf
    fallback = None
    flux_rejections = [0]
    separate_forecast = not np.array_equal(prob.dni_con, prob.dni_obj)

    reference = prob.tracking_reference() if tracking_reference is None else tracking_reference
    best_motion = None
    best_raw = None  # unpriced objective of the selected plan
    top_raw = [-np.inf]  # best unpriced objective of any legal, verified joint pose (exhaustive bound)
    peak_cap = [None]  # the seed's verified peak ratio when it ends inside the safety margin
    from .aps import plan_move_cost

    def verified(check) -> bool:
        return check.ok or (peak_cap[0] is not None and check.status == "margin_reduced"
                            and check.peak_ratio <= peak_cap[0])

    def consider(ch, is_seed=False):
        nonlocal best, best_value, best_motion, best_raw, fallback
        ch = np.asarray(ch, dtype=int)
        key = ch.tobytes()
        if key in seen:
            return
        seen.add(key)
        on = ch >= 0
        legal = (np.all(allowed[np.flatnonzero(on), ch[on]])
                 and (spec.allow_defocus or np.all(on | ~prob.available)))
        args = dict(aim=prob.aims(ch), active=on, grid=prob.grid, shading_points=shading_points)
        snap = evaluate(case, field, rs.sun_elevation, rs.sun_azimuth, prob.dni_obj, **args)
        value = snap.thermal_objective(uni)
        priced = value["objective_w"] - plan_move_cost(move_cost, ch)
        # Lexicographic tie-break on commands. The thermal score and its certificate stay in watts.
        motion = (np.count_nonzero(on != (reference >= 0)), np.count_nonzero(ch != reference))
        better = priced > best_value or (priced == best_value and (best_motion is None or motion < best_motion))
        # a pose that may raise the best unpriced objective is still checked: the exhaustive bound needs it
        raw_better = value["objective_w"] > top_raw[0]
        if fallback is not None and (not legal or not value["uniformity_met"] or not (better or raw_better)):
            return
        forecast = (evaluate(case, field, rs.sun_elevation, rs.sun_azimuth, prob.dni_con, **args)
                    if separate_forecast else snap)
        check = forecast.verify(case.receiver, max_cells=spec.verify_max_cells)
        if is_seed and not exact and check.status == "margin_reduced":
            peak_cap[0] = check.peak_ratio
        flux_rejections[0] += int(not verified(check))
        entry = (ch.copy(), snap, value, check)
        if fallback is None:
            fallback = entry
        if legal and value["uniformity_met"] and verified(check):
            top_raw[0] = max(top_raw[0], value["objective_w"])
            if better:
                best, best_value, best_motion, best_raw = entry, priced, motion, value["objective_w"]

    consider(seed.choice, is_seed=True)
    if warm is not None:
        consider(warm)
    if exact and not impossible_load:
        for ch in product(*choices):
            consider(ch)
    elif count and not impossible_load:
        if spec.allow_defocus:
            consider(np.full(prob.n, -1))
        center = np.full(prob.n, prob.center_k)
        consider(np.where(prob.available, center, -1))
        brightest = np.argmax(np.where(allowed, unshaded, -np.inf), axis=1)
        consider(np.where(allowed.any(axis=1), brightest, -1))
        rng = np.random.default_rng(0)
        for j in range(trials):
            current = (best or fallback)[0].copy()
            movable = np.flatnonzero([len(row) > 1 for row in choices])
            if not len(movable):
                break
            # Alternate single-coordinate moves with blocks to escape a local optimum.
            size = 1 if j % 2 == 0 else min(len(movable), max(2, int(np.sqrt(len(movable)))))
            for i in rng.choice(movable, size=size, replace=False):
                current[i] = rng.choice(choices[i])
            consider(current)

    if best is not None:
        # Reactivating a shaded centre target preserves the pose used for an inactive mirror.
        # Also try its previous target; re-evaluate neighbours and verify before accepting.
        centre = prob.tracking_reference()
        retained = np.where((best[0] < 0) & (reference >= 0), centre, best[0])
        retained = prob.keep_shadowed_targets(best[0], retained)
        consider(retained)
        consider(prob.keep_shadowed_targets(best[0], reference))
    choice, snap, value, check = best or fallback
    # Minimum-load failure can make every candidate plan infeasible even though a fully
    # tower-shadowed heliostat has no optical path to the receiver. Keep its tracking pose
    # if restoring it is flux-safe and does not lower the thermal objective.
    if best is None and case.aps.allow_defocus:
        reference = prob.tracking_reference() if tracking_reference is None else tracking_reference
        retained = prob.keep_shadowed_targets(choice, reference)
        if not np.array_equal(retained, choice):
            args = dict(aim=prob.aims(retained), active=retained >= 0, grid=prob.grid,
                        shading_points=shading_points)
            retained_snap = evaluate(case, field, rs.sun_elevation, rs.sun_azimuth, prob.dni_obj, **args)
            retained_value = retained_snap.thermal_objective(uni)
            forecast = (evaluate(case, field, rs.sun_elevation, rs.sun_azimuth, prob.dni_con, **args)
                        if separate_forecast else retained_snap)
            retained_check = forecast.verify(case.receiver, max_cells=spec.verify_max_cells)
            retained_on = retained >= 0
            retained_legal = np.all(allowed[np.flatnonzero(retained_on), retained[retained_on]])
            if (verified(retained_check) and retained_value["objective_w"] >= value["objective_w"]
                    and retained_legal):
                choice, snap, value, check = retained, retained_snap, retained_value, retained_check
    feasible = best is not None
    proven_empty = not feasible and (exact or impossible_load)
    energy_upper = upper
    priced_value = best_value
    if feasible:
        best_value = best_raw  # the certificate is one of the unpriced objective
    bound_mode = "exhaustive_joint_poses" if exact and not impossible_load else "unshaded_energy"
    if exact and feasible:
        upper = top_raw[0]
    elif feasible and tighten_bound and mode == "weighted" and uni.weight > 0:
        supported = _weighted_upper(prob, beam, intercepted, allowed, snap, uni.weight)
        if supported < upper:
            upper, bound_mode = supported, "joint_flux_support"
    gap = ((upper - best_value) / max(abs(upper), 1e-12) if feasible else None)
    if gap is not None and gap < -1e-8:
        raise ArithmeticError("Joint-pose incumbent exceeds its unshaded energy bound")
    gap = max(0.0, gap) if gap is not None else None
    # exhaustive: proven unless a move price made the search keep a plan below the best unpriced pose
    proven = feasible and ((exact and move_cost is None) or upper - best_value <= 1e-8)
    status = "certified" if feasible else ("certified_infeasible" if proven_empty else "bound_only")
    thermal_status = ("disabled" if not uni.enabled else
                      "below_min_load" if value["power_h_w"] < minimum else
                      "weighted" if mode == "weighted" else "feasible" if value["uniformity_met"] else "not_reached")
    info = dict(seed.info)
    info.pop("certificate", None)
    metrics = uniformity_metrics(snap.absorbed_flux, snap.grid.to_map(snap.region))
    info.update(joint_pose_checked=True, objective_absorbed_power_w=value["power_w"], power_H_mw=value["power_h_w"] / 1e6,
                cv_2d_H=value["cv_2d"], uniformity_status=thermal_status, uniformity_met=value["uniformity_met"],
                global_optimality_proven=proven, optimality_gap=gap, upper_bound_mw=upper / 1e6,
                cv_circ_H=metrics["cv_circ"] if np.isfinite(metrics["cv_circ"]) else None,
                cv_height_H=metrics["cv_height"] if np.isfinite(metrics["cv_height"]) else None,
                solution_class=("global_optimum_on_candidate_set" if proven else
                                "verified_feasible_with_certified_gap" if feasible else "no_verified_incumbent"))
    cert = dict(status=status, mode=mode, method=bound_mode, unshaded_energy_upper_mw=energy_upper / 1e6,
                epsilon=uni.max_cv if mode == "cv_limit" else None, min_load_mw=minimum / 1e6,
                objective_found_mw=best_value / 1e6 if feasible else None, upper_bound_mw=upper / 1e6,
                gap=gap, optimum_exists=True if feasible else False if proven_empty else None,
                global_optimum_proven_on_candidate_set=proven, n_candidates=prob.k,
                evaluated_joint_poses=len(seen), objective_grid=list(prob.grid.shape),
                model="mixed mirror poses, sampled shading, area-weighted absorbed flux and verified peak",
                history=[{"iteration": 1, "upper_bound_mw": upper / 1e6}])
    cert["lambda"] = uni.weight if mode == "weighted" else None
    if move_cost is not None:
        cert["move_cost_mw"] = (best_value - priced_value) / 1e6 if feasible else None
        cert["priced_objective_mw"] = priced_value / 1e6 if feasible else None
        cert["priced_optimum_proven"] = bool(exact and feasible)
    cert["flux_rejections"] = int(flux_rejections[0])  # joint poses refused by the fine-grid flux check
    # From any pivot all receiver targets lie in the angular cone subtended by
    # its enclosing sphere. Cosine is maximal at the cone's edge nearest the sun.
    v = np.asarray(case.receiver.center) - field.xyz
    d = np.linalg.norm(v, axis=1)
    radius = np.hypot(case.receiver.radius, case.receiver.height / 2)
    turn = np.where(d > radius, np.arcsin(np.clip(radius / np.maximum(d, 1e-15), 0, 1)), np.pi)
    angle = np.arccos(np.clip(v @ sun.sun_vector(rs.sun_elevation, rs.sun_azimuth) / np.maximum(d, 1e-15), -1, 1))
    cosine = np.cos(np.maximum(angle - turn, 0.0) / 2)
    material = case.heliostat.area * case.heliostat.reflectivity * case.heliostat.cleanliness
    alpha = max(case.receiver.panel_absorptivity or [case.receiver.absorptivity])
    continuous_upper = float((cosine * prob.dni_obj * prob.available).sum() * material * alpha) if rs.sun_elevation > 0 else 0.0
    cert["continuous_upper_bound_mw"] = continuous_upper / 1e6
    cert["continuous_gap"] = (max(0.0, (continuous_upper - best_value) / max(continuous_upper, 1e-12))
                              if feasible else None)
    result = replace(seed, choice=choice, aim=prob.aims(choice), active=choice >= 0,
                     power_w=snap.p_receiver, peak_ratio=check.peak_ratio / check.allowed_ratio,
                     status=("infeasible" if not feasible and check.status == "over_limit" else check.status),
                     info=info, check=check)
    return result, cert
