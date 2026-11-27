"""Optimality certificates: rigorous upper bounds for problem 1 (APS) and problem 2 (layout).

Problem 1 -- aim points for a fixed field at one sun state (candidate model of aps.py)
-------------------------------------------------------------------------------------
Variables x_ik in {0, 1}, sum_k x_ik <= 1 (all zero = defocused). With the unit
responses of the ResponseSet everything is linear in x:

    P(x)   = sum_ik c_ik x_ik                         absorbed power (objective DNI)
    q(x)   = absorbed flux in the cells of H           (|H| cells of area a)
    S(x)   = A_H * sigma_H(q) = a sqrt(|H|) || q - mean(q) ||     convex, >= 0
    P_H(x) = a * sum_H q                               absorbed power on H (linear)
    flux:  (A x)_m <= 1    every constraint point m  (rows normalised by the allowed flux)

    power     max P(x)
    weighted  max P(x) - lambda S(x)                           (= P - lambda CV_2D P_H)
    cv_limit  max P(x)  s.t.  S(x) <= eps P_H(x),  P_H(x) >= P_min

Existence: the candidate set is finite and "all defocused" meets every flux
limit, so the power and weighted problems always have an optimum; cv_limit has
one whenever it is feasible (with P_min > 0 it may not be).

Bound: with multipliers nu, mu, rho >= 0 the Lagrangian

    L(x) = P(x) - (lambda + mu) S(x) + mu eps P_H(x) + rho (P_H(x) - P_min) - nu.(A x - 1)

is concave in x and >= the objective at every feasible x, so max over the
relaxed set D (product of simplices, x in [0, 1]) of L is an upper bound
(weak duality). Concavity gives, at any point x_t,

    max_D L <= L(x_t) + max_{y in D} g_t.(y - x_t),   g_t a supergradient of L at x_t

and the inner maximum is separable per heliostat (best candidate if its
coefficient is positive, else defocus). We run Frank-Wolfe steps on x and
projected subgradient steps on the multipliers; *every* iterate yields a valid
bound, the reported one is the smallest. Hence

    found objective  <=  global optimum (candidate model)  <=  upper bound.

Problem 2 -- layout of N heliostats in the site, annual absorbed energy
-----------------------------------------------------------------------
e0(x) = annual absorbed energy of one heliostat at x with no neighbours
(no shading / blocking). Neighbours and flux limits only remove energy, so
E(X) <= sum_i e0(x_i). Layer-cake + packing:

    sum_i e0(x_i) = int_0^inf #{i : e0(x_i) >= t} dt <= int_0^inf min(N, cap(L_t)) dt

with L_t = {x in site : e0(x) >= t}. For points with mutual distance >= d the
Voronoi cells truncated by discs of radius d/sqrt(3) are disjoint and each has
area >= sqrt(3) d^2 / 2 (Fejes Toth), so cap(S) <= area(S (+) B(d/sqrt 3)) / (sqrt(3) d^2 / 2).
On the raster, ``cell_energy_upper_bound`` bounds e0 over every point of each
cell and every aim point (interval bounds on cosine, attenuation, intercept and
absorptivity), so no step relies on sampling.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

# --------------------------------------------------------------------------- problem 1


@dataclass
class _Model:
    """Sparse linear model of one AimProblem for the bound."""

    R: object  # (P, N*K) CSC unit-DNI response at every constraint point
    dcon: np.ndarray  # (N*K,) constraint DNI per column
    dobj: np.ndarray  # (N*K,) objective DNI per column
    lim: np.ndarray  # (P,) allowed incident flux incl. margin
    h_idx: np.ndarray  # (|H|,) constraint-point indices of H (grid cells)
    alpha_h: np.ndarray  # (|H|,) absorptivity on H
    c: np.ndarray  # (N*K,) absorbed power per candidate, W
    h_lin: np.ndarray  # (N*K,) dP_H / dx, W
    ub: np.ndarray  # (N*K,) 1 if the candidate may be used
    n: int
    k: int
    a: float  # cell area, m2
    same_dni: bool

    @property
    def sc(self) -> float:
        return self.a * math.sqrt(max(self.h_idx.size, 1))


class CertificateTooLarge(MemoryError):
    """The bound's sparse model would not fit into the free memory (after the system's reserve)."""

    def __init__(self, message: str, need_bytes: int = 0, room_bytes: int = 0):
        super().__init__(message)
        self.need_bytes, self.room_bytes = int(need_bytes), int(max(room_bytes, 0))


BYTES_PER_NONZERO = 48  # COO (int64 row, int64 column, float64 value) + CSC copy + per-heliostat pieces
# the formal layout value skips certificate roundings above this model size (a rule on the inputs alone, so the
# value stays reproducible whatever memory happens to be free)
FORMAL_CERTIFY_MAX_BYTES = 8 * 2 ** 30


def model_bytes(prob) -> int:
    """Memory to build the bound's model of ``prob``: every candidate x support cell is one non-zero."""
    return int(sum(int(c.size) for c in prob.rs.cells) * prob.k * BYTES_PER_NONZERO)


def check_model_memory(prob) -> None:
    """Raise ``CertificateTooLarge`` when building the model would eat into the memory the computer keeps free
    (``parallel.memory_bytes``: free memory minus max(2 GB, 15 %)); unknown memory: no check."""
    from .parallel import memory_bytes

    mem = memory_bytes()
    if mem is None:
        return
    avail, total = mem
    room = avail - max(2 * 2 ** 30, 0.15 * total)
    need = model_bytes(prob)
    if need > room:
        raise CertificateTooLarge(
            f"the optimality certificate needs about {need / 2 ** 30:.1f} GB ({prob.n} heliostats x {prob.k} candidate "
            f"aims on {prob.rs.points.shape[0]} constraint points), more than the {max(room, 0) / 2 ** 30:.1f} GB "
            "free after the system's reserve", need, room)


def _build_model(prob, h_mask: np.ndarray) -> _Model:
    check_model_memory(prob)  # before allocating anything: a too large model must not freeze the computer
    rs = prob.rs
    n, k = prob.n, prob.k
    rows, cols, vals = [], [], []
    for i in range(n):
        idx = rs.cells[i]
        if idx.size == 0:
            continue
        r = rs.resp[i]
        kk, ss = np.nonzero(r)
        rows.append(idx[ss])
        cols.append(i * k + kk)
        vals.append(r[kk, ss].astype(float))
    p = rs.points.shape[0]
    if rows:
        R = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(p, n * k)).tocsc()
    else:
        R = coo_matrix((p, n * k)).tocsc()
    dcon = np.repeat(prob.dni_con * prob.available, k).astype(float)
    dobj = np.repeat(prob.dni_obj * prob.available, k).astype(float)
    h_idx = np.flatnonzero(h_mask)
    alpha_h = rs.alpha[h_idx]
    w = np.zeros(p)
    w[h_idx] = alpha_h
    a = float(rs.grid.cell_area)
    h_lin = a * dobj * (R.T @ w)
    ub = np.repeat(prob.available.astype(float), k)
    if prob.reach is not None:
        ub = ub * np.asarray(prob.reach, dtype=float).ravel()
    return _Model(R, dcon, dobj, prob.limit.astype(float), h_idx, alpha_h,
                  prob.absorbed_power.ravel().astype(float), h_lin, ub, n, k, a,
                  bool(np.allclose(prob.dni_con, prob.dni_obj)))


def _vertex(model: _Model, choice: np.ndarray) -> np.ndarray:
    x = np.zeros(model.n * model.k)
    on = np.flatnonzero(choice >= 0)
    x[on * model.k + choice[on]] = 1.0
    return x


def _fluxes(model: _Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fo = model.R @ (model.dobj * x)
    fc = fo if model.same_dni else model.R @ (model.dcon * x)
    return fc, fo


def _fluxes_sparse(model: _Model, y_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fluxes of a vertex (one candidate per listed column) using only those columns."""
    if y_cols.size == 0:
        z = np.zeros(model.R.shape[0])
        return z, z.copy()
    sub = model.R[:, y_cols]
    fo = sub @ model.dobj[y_cols]
    fc = fo if model.same_dni else sub @ model.dcon[y_cols]
    return fc, fo


def _spread(model: _Model, fo: np.ndarray) -> tuple[np.ndarray, float]:
    q = model.alpha_h * fo[model.h_idx]
    d = q - q.mean() if q.size else q
    return d, float(np.linalg.norm(d))


def objective_value(prob, choice: np.ndarray, mode: str, lam: float = 0.0,
                    h_mask: np.ndarray | None = None) -> dict:
    """P, S, P_H and CV of a discrete plan in the bound's model (solver grid)."""
    from .uniformity import region_of_grid

    h_mask = region_of_grid(prob.case, prob.grid) if h_mask is None else h_mask
    model = _build_model(prob, h_mask)
    x = _vertex(model, np.asarray(choice))
    _, fo = _fluxes(model, x)
    _, nd = _spread(model, fo)
    p = float(model.c @ x)
    s = model.sc * nd
    p_h = float(model.h_lin @ x)
    return {"power_w": p, "spread_w": s, "power_h_w": p_h, "cv_2d": s / p_h if p_h > 0 else float("nan"),
            "objective_w": p - lam * s if mode == "weighted" else p}


def aps_upper_bound(
    prob,
    choice: np.ndarray,
    mode: str = "weighted",
    lam: float = 0.0,
    eps: float | None = None,
    min_load_w: float = 0.0,
    h_mask: np.ndarray | None = None,
    mu0: float = 0.0,
    max_iter: int = 300,
    time_limit_s: float = 30.0,
    tol: float = 2e-4,
    price_mask: np.ndarray | None = None,
) -> dict:
    """Upper bound on the optimum of problem 1 and the optimality gap of ``choice``.

    ``mode`` is "power", "weighted" (objective P - lam * A_H sigma_H) or
    "cv_limit" (max P with CV_2D <= eps and P_H >= min_load_w). The bound is
    for the candidate set and the flux constraint points of ``prob`` (grid
    cells and verified hotspots), i.e. valid for every plan that the
    candidate model allows.

    ``price_mask`` (bool, N) marks heliostats that are *not* part of the layout being bounded (candidate
    positions, off in ``choice``): the bound is for the other heliostats only, and the multipliers of the
    best iteration price each masked heliostat. Its reduced cost
    ``G_j = max(0, max_k [c_jk + (mu eps + rho) h_jk - DNI_j R_jk . (w_o + nu / lim)])`` (``priced_gain_w``)
    bounds what adding it alone can add: optimum(layout + j) <= upper_bound + G_j (weak duality; the
    Lagrangian is concave and x_j = 0 at the linearisation point). It assumes j does not change the other
    heliostats' responses (no shading / blocking of the layout by j).
    """
    from .uniformity import region_of_grid

    t0 = _time.perf_counter()
    if mode not in ("power", "weighted", "cv_limit"):
        raise ValueError(f"unknown bound mode '{mode}'")
    if mode == "cv_limit" and eps is None:
        raise ValueError("cv_limit bound needs eps")
    h_mask = region_of_grid(prob.case, prob.grid) if h_mask is None else np.asarray(h_mask, dtype=bool)
    model = _build_model(prob, h_mask)
    n, k = model.n, model.k
    lam = float(lam) if mode == "weighted" else 0.0
    eps_v = float(eps) if mode == "cv_limit" else 0.0
    use_mu = mode == "cv_limit"
    use_rho = mode in ("cv_limit", "weighted") and min_load_w > 0
    priced = np.zeros(n, dtype=bool) if price_mask is None else np.asarray(price_mask, dtype=bool)
    if priced.shape != (n,):
        raise ValueError("price_mask must have one entry per heliostat")
    if np.any(np.asarray(choice)[priced] >= 0):
        raise ValueError("priced heliostats must be off in the plan")
    ub_loop = model.ub.copy()
    ub_loop[np.repeat(priced, k)] = 0.0
    c_layout = np.where(np.repeat(priced, k), 0.0, model.c)
    p_ref = max(float(np.maximum(c_layout.reshape(n, k), 0.0).max(axis=1, initial=0.0).sum()), 1.0)
    g_best = None

    # value of the given plan
    x0 = _vertex(model, np.asarray(choice))
    fc, fo = _fluxes(model, x0)
    _, nd0 = _spread(model, fo)
    p0, s0, ph0 = float(model.c @ x0), model.sc * nd0, float(model.h_lin @ x0)
    flux_ok0 = bool(np.all(fc <= model.lim * (1 + 1e-9)))
    if mode == "weighted":
        plan_feasible = flux_ok0 and ph0 >= min_load_w * (1 - 1e-9)
        found = p0 - lam * s0 if plan_feasible else None
    elif mode == "power":
        plan_feasible = flux_ok0
        found = p0 if plan_feasible else None
    else:
        plan_feasible = flux_ok0 and s0 <= eps_v * ph0 * (1 + 1e-9) and ph0 >= min_load_w * (1 - 1e-9) and ph0 > 0
        found = p0 if plan_feasible else None

    x = x0.copy()
    nu = np.zeros(model.R.shape[0])
    mu = max(float(mu0), 0.0) if use_mu else 0.0
    rho = 0.0
    best = math.inf
    best_t = 0
    hist = []
    eta_nu, eta_mu = 0.02 * p_ref, 1.0
    it = 0
    for it in range(1, max_iter + 1):
        d, nd = _spread(model, fo)
        kappa = lam + mu
        p = float(model.c @ x)
        ph = float(model.h_lin @ x)
        ax = fc / model.lim
        lval = (p - kappa * model.sc * nd + mu * eps_v * ph + rho * (ph - min_load_w)
                - float(nu @ (ax - 1.0)))
        # supergradient of L
        lin = model.c + (mu * eps_v + rho) * model.h_lin
        w_o = np.zeros(model.R.shape[0])
        if kappa > 0 and nd > 0:
            w_o[model.h_idx] = kappa * model.sc * model.alpha_h * d / nd
        w_c = nu / model.lim
        if model.same_dni:
            g = lin - model.dobj * (model.R.T @ (w_o + w_c))
        else:
            both = model.R.T @ np.column_stack([w_o, w_c])
            g = lin - model.dobj * both[:, 0] - model.dcon * both[:, 1]
        G = np.where(ub_loop > 0, g, -np.inf).reshape(n, k)
        kb = np.argmax(G, axis=1)
        gb = G[np.arange(n), kb]
        take = np.flatnonzero(gb > 0)
        y_cols = take * k + kb[take]
        ub_t = lval + float(gb[take].sum()) - float(g @ x)
        if ub_t < best:
            best, best_t = ub_t, it
            if priced.any():
                g_best = g.copy()
        if it == 1 or it % 10 == 0:
            hist.append((it, best))
        if found is not None and best - found <= tol * max(abs(best), 1.0):
            break
        if _time.perf_counter() - t0 > time_limit_s:
            break
        # Frank-Wolfe step on x (exact line search on the concave 1-D function)
        fcy, foy = _fluxes_sparse(model, y_cols)
        qy = model.alpha_h * foy[model.h_idx]
        dy = qy - qy.mean() if qy.size else qy
        dd = dy - d
        lin_dir = float(lin[y_cols].sum()) - float(lin @ x) - float(w_c @ (fcy - fc))
        uu, uv, vv = float(d @ d), float(d @ dd), float(dd @ dd)

        def deriv(gm: float) -> float:
            if kappa == 0:
                return lin_dir
            nn = math.sqrt(max(uu + 2 * gm * uv + gm * gm * vv, 1e-300))
            return lin_dir - kappa * model.sc * (uv + gm * vv) / nn

        if deriv(0.0) <= 0:
            gam = 0.0
        elif deriv(1.0) >= 0:
            gam = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if deriv(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            gam = 0.5 * (lo + hi)
        if gam > 0:
            x *= (1 - gam)
            x[y_cols] += gam
            fc = (1 - gam) * fc + gam * fcy
            fo = fc if model.same_dni else (1 - gam) * fo + gam * foy
        # multiplier steps (projected subgradient on the dual)
        step = 1.0 / math.sqrt(it)
        nu = np.maximum(0.0, nu + eta_nu * step * (fc / model.lim - 1.0))
        if use_mu:
            _, nd_new = _spread(model, fo)
            ph_new = float(model.h_lin @ x)
            mu = max(0.0, mu + eta_mu * step * (model.sc * nd_new - eps_v * ph_new) / p_ref)
        if use_rho:
            rho = max(0.0, rho + eta_mu * step * (min_load_w - float(model.h_lin @ x)) / p_ref)
    if not hist or hist[-1][0] != it:
        hist.append((it, best))
    relaxed = x.reshape(n, k)
    if found is not None:
        gap = max(0.0, 1.0 - found / best) if best > 0 else 0.0
        status = "certified"
    elif mode == "cv_limit" and best < 0:
        gap, status = None, "certified_infeasible"
    else:
        gap, status = None, "bound_only"
    out = {
        "mode": mode, "lambda": lam if mode == "weighted" else None, "epsilon": eps if mode == "cv_limit" else None,
        "min_load_mw": min_load_w / 1e6 if mode == "cv_limit" else None,
        "plan_feasible_in_model": bool(plan_feasible),
        "objective_found_mw": None if found is None else found / 1e6,
        "upper_bound_mw": best / 1e6,
        "gap": gap,
        "status": status,
        "iterations": it, "best_iteration": best_t,
        "time_s": round(_time.perf_counter() - t0, 3),
        "history": [{"iteration": a, "upper_bound_mw": b / 1e6} for a, b in hist],
        "n_heliostats": n, "n_candidates": k, "n_constraint_points": int(model.R.shape[0]),
        "n_cells_H": int(model.h_idx.size), "multiplier_mu": mu if use_mu else None,
        "model": "candidate aim points of the APS, flux limits at the solver grid and verified hotspots",
        "_relaxed": relaxed,
    }
    if priced.any():
        gain = np.zeros(n)
        if g_best is not None:
            gp = np.where(model.ub > 0, g_best, -np.inf).reshape(n, k)
            gain[priced] = np.maximum(gp[priced].max(axis=1), 0.0)
        out["priced_gain_w"] = gain
    return out


def removal_losses(prob, choice: np.ndarray, lam: float = 0.0, h_mask: np.ndarray | None = None) -> np.ndarray:
    """Objective lost (W) in the bound's model when one heliostat of the plan is switched off alone.

    The objective is P - lam * S (``objective_value``, "weighted" / "power"); switching a mirror off never
    violates a flux limit, so the reduced plan stays feasible. Exact per heliostat (0 when defocused);
    neighbours that the mirror shaded or blocked are not credited (the model's responses are fixed).
    """
    from .uniformity import region_of_grid

    h_mask = region_of_grid(prob.case, prob.grid) if h_mask is None else np.asarray(h_mask, dtype=bool)
    model = _build_model(prob, h_mask)
    choice = np.asarray(choice)
    x = _vertex(model, choice)
    _, fo = _fluxes(model, x)
    d0, nd0 = _spread(model, fo)
    j0 = float(model.c @ x) - lam * model.sc * nd0
    loss = np.zeros(model.n)
    lam = float(lam)
    for i in np.flatnonzero(choice >= 0):
        col = int(i * model.k + choice[i])
        p = float(model.c[col])
        if lam > 0 and model.h_idx.size:
            colv = model.R[:, col].toarray().ravel() * model.dobj[col]
            dq = model.alpha_h * colv[model.h_idx]
            nd = float(np.linalg.norm(d0 - (dq - dq.mean())))
            loss[i] = j0 - (float(model.c @ x) - p - lam * model.sc * nd)
        else:
            loss[i] = p
    return loss


def certify_result(prob, res, case=None, max_iter: int = 300, time_limit_s: float = 30.0) -> dict:
    """Bound for an AimResult in the mode its case asks for (uniformity settings)."""
    from .uniformity import region_of_grid

    case = case or prob.case
    uni = case.uniformity
    h_mask = region_of_grid(case, prob.grid, uni)
    if not uni.enabled:
        mode, lam, eps = "power", 0.0, None
    elif uni.mode == "weighted":
        mode, lam, eps = "weighted", float(uni.weight), None
    else:
        mode, lam, eps = "cv_limit", 0.0, float(uni.max_cv)
    mu0 = float(res.info.get("uniformity_weight") or 0.0) if mode == "cv_limit" else 0.0
    return aps_upper_bound(prob, res.choice, mode, lam, eps, uni.min_load_mw * 1e6, h_mask, mu0=mu0,
                           max_iter=max_iter, time_limit_s=time_limit_s)


def _mode_of(case) -> tuple[str, float, float | None]:
    uni = case.uniformity
    if not uni.enabled:
        return "power", 0.0, None
    if uni.mode == "weighted":
        return "weighted", float(uni.weight), None
    return "cv_limit", 0.0, float(uni.max_cv)


def _model_value(prob, choice, mode, lam, eps, min_load_w, h_mask) -> float | None:
    """Objective of a plan in the bound's model, None if it violates the mode's constraints."""
    if np.any(prob.flux(choice) > prob.limit * (1 + 1e-9)):
        return None
    v = objective_value(prob, choice, mode, lam, h_mask)
    if v["power_h_w"] < min_load_w * (1 - 1e-9):
        return None
    if mode == "cv_limit":
        if not (v["power_h_w"] > 0 and v["spread_w"] <= eps * v["power_h_w"] * (1 + 1e-9)
                and v["power_h_w"] >= min_load_w * (1 - 1e-9)):
            return None
    return v["objective_w"]


def improve_and_certify(prob, res, spec=None, rounds: int = 8, max_iter: int = 300, time_limit_s: float = 20.0,
                        seed: int = 0, prev=None, prev_active=None):
    """Primal-dual improvement of an APS result with an optimality certificate.

    1. Upper bound and relaxed solution from ``aps_upper_bound`` (Lagrangian + Frank-Wolfe).
    2. Relaxation-guided rounding (argmax and ``rounds`` randomised roundings of the
       relaxed solution), flux repair (greedy warm start), local search in the
       mode's objective.
    3. The best plan that improves the objective is verified on the fine grid
       (hotspots added as constraints, repaired); otherwise the input plan is kept. It may not use more of
       the safety margin than the input plan: strictly within it when the input was, else no higher peak.
    4. A second bound run from the final plan; the reported bound is the smaller one.
    With an initial state (``prev``) and ``aps.move_penalty_per_m`` > 0, plans are compared by the objective
    minus their move price (``aps.move_cost``): a rounding of the relaxation may not undo the move pricing. The
    bound stays the bound of the objective without the price (>= the priced optimum, so still valid); the
    certificate reports the plan's price as ``move_cost_mw``.
    Returns (AimResult, certificate dict).
    """
    from .aps import finalize, move_cost, plan_move_cost, solve_greedy, uniformity_info
    from .receiver import surface_point
    from .uniformity import _local_search, _State, region_of_grid

    t0 = _time.perf_counter()
    case = prob.case
    spec = spec or case.aps
    uni = case.uniformity
    mode, lam, eps = _mode_of(case)
    min_load_w = uni.min_load_mw * 1e6 if uni.enabled else 0.0
    h_mask = region_of_grid(case, prob.grid, uni)
    prev_xyz = surface_point(case.receiver, *prev) if prev is not None else None
    mu = float(res.info.get("uniformity_weight") or 0.0) if mode == "cv_limit" else lam
    base_choice = np.asarray(res.choice).copy()
    b1 = aps_upper_bound(prob, base_choice, mode, lam, eps, min_load_w, h_mask, mu0=mu,
                         max_iter=max_iter, time_limit_s=time_limit_s)
    relaxed = b1.pop("_relaxed")
    cost = move_cost(prob, prev_xyz, prev_active, spec)

    def priced(choice):
        v = _model_value(prob, choice, mode, lam, eps, min_load_w, h_mask)
        return None if v is None else v - plan_move_cost(cost, choice)

    base_val = priced(base_choice)
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")
    rng = np.random.default_rng(seed)

    def polish(start: np.ndarray) -> np.ndarray:
        ch, _ = solve_greedy(prob, spec, prev_xyz, prev_active, init=start, full_passes=0 if uni.enabled else None)
        if uni.enabled:
            st = _State(prob, h_mask, ch)
            _local_search(prob, st, mu, 25, prob.absorbed_power, order, prev_xyz, prev_active)
            ch = st.choice
        return ch

    tried = []
    tot = relaxed.sum(axis=1)
    tried.append(np.where(tot >= 0.5, relaxed.argmax(axis=1), -1))
    cum = np.cumsum(relaxed, axis=1)
    for _ in range(max(0, rounds)):
        u = rng.random(relaxed.shape[0])
        pick = (cum < u[:, None]).sum(axis=1)
        tried.append(np.where(pick < relaxed.shape[1], pick, -1))
    best_choice, best_val = base_choice, base_val
    n_improved = 0
    for start in tried:
        start = np.where(prob.available, start, -1)
        ch = polish(start)
        v = priced(ch)
        if v is not None and (best_val is None or v > best_val + 1e-9 * max(abs(best_val), 1.0)):
            best_choice, best_val = ch, v
            n_improved += 1
    final = res
    accepted = False
    if best_choice is not base_choice:
        choice = best_choice
        check = prob.verify(choice) if spec.verify else None
        rounds_v = 0
        while spec.verify and not check.ok and check.hotspots and rounds_v < spec.verify_max_rounds:
            prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
            choice = polish(choice)
            check = prob.verify(choice)
            rounds_v += 1
        v = priced(choice)
        base_now = priced(base_choice)
        # no more of the safety margin than the input plan: within it if the input was ("ok"), else no higher peak
        base_check = res.check
        margin_ok = (check.status == "ok" or (check.status == "margin_reduced" and base_check is not None
                                               and base_check.status != "ok"
                                               and check.peak_ratio <= base_check.peak_ratio))
        ok = (not spec.verify or margin_ok) and v is not None and (base_now is None or v > base_now)
        if ok:
            info = {k: v_ for k, v_ in res.info.items() if not k.startswith("uniformity") and not k.startswith("cv_")
                    and k not in ("power_H_mw", "global_optimality_proven", "solution_class")}
            if uni.enabled:
                info.update(uniformity_info(prob, choice, uni))
                if mode == "cv_limit":
                    info["uniformity_weight"] = mu
            final = finalize(prob, res.method, choice, info, check, spec, t0)
            accepted = True
    b2 = aps_upper_bound(prob, final.choice, mode, lam, eps, min_load_w, h_mask, mu0=mu,
                         max_iter=max(50, max_iter // 3), time_limit_s=max(2.0, time_limit_s / 3))
    b2.pop("_relaxed", None)
    ub = min(b1["upper_bound_mw"], b2["upper_bound_mw"])
    cert = dict(b2)
    cert["upper_bound_mw"] = ub
    found = cert["objective_found_mw"]
    if found is not None:
        cert["gap"] = max(0.0, 1.0 - found / ub) if ub > 0 else 0.0
        cert["status"] = "certified"
    elif mode == "cv_limit" and ub < 0:
        cert["status"], cert["gap"] = "certified_infeasible", None
    cert["objective_before_mw"] = None if base_val is None else (base_val + plan_move_cost(cost, base_choice)) / 1e6
    cert["move_cost_mw"] = plan_move_cost(cost, final.choice) / 1e6
    cert["improved_by_relaxation"] = bool(accepted)
    cert["roundings_tried"] = len(tried)
    cert["iterations"] = b1["iterations"] + b2["iterations"]
    cert["history"] = b1["history"]
    cert["time_s"] = round(_time.perf_counter() - t0, 3)
    cert["global_optimum_proven_on_candidate_set"] = bool(cert.get("gap") is not None and cert["gap"] <= 1e-6)
    final.info["certificate"] = {k: v_ for k, v_ in cert.items() if k != "history"}
    final.info["optimality_gap"] = cert.get("gap")
    final.info["upper_bound_mw"] = ub
    if final.status == "infeasible":
        # the plan failed the fine-grid check: a gap in the candidate model says nothing about it
        cert["status"], cert["gap"] = "plan_failed_verification", None
        cert["global_optimum_proven_on_candidate_set"] = False
        final.info["certificate"] = {k: v_ for k, v_ in cert.items() if k != "history"}
        final.info["optimality_gap"] = None
        final.info["solution_class"] = "infeasible_after_verification"
        final.info["global_optimality_proven"] = False
    elif cert.get("gap") is not None:
        final.info["solution_class"] = ("global_optimum_on_candidate_set" if cert["global_optimum_proven_on_candidate_set"]
                                        else "verified_feasible_with_certified_gap")
        final.info["global_optimality_proven"] = cert["global_optimum_proven_on_candidate_set"]
    return final, cert


def pareto_front(prob, lambdas, base_choice: np.ndarray | None = None, spec=None, certify: bool = True,
                 rounds: int = 2, time_limit_s: float = 6.0) -> list[dict]:
    """Power vs uniformity trade-off over weights lambda (weighted mode, fixed field and DNI).

    Every point is the full single-instant pipeline: coordinate search from the
    previous point (continuation in lambda), relaxation rounding, fine-grid
    verification and, with ``certify``, the upper bound / gap for that lambda.
    """
    import dataclasses

    from .aps import finalize, solve_greedy, uniformity_info
    from .uniformity import region_of_grid, solve_uniform

    spec = spec or prob.case.uniformity
    case0 = prob.case
    aps_spec = case0.aps
    if base_choice is None:
        base_choice, _ = solve_greedy(prob, aps_spec)
    out = []
    start = np.asarray(base_choice)
    try:
        for lam in sorted(float(v) for v in lambdas):
            t0 = _time.perf_counter()
            c = case0.copy()
            c.uniformity = dataclasses.replace(spec, enabled=True, mode="weighted", weight=lam)
            prob.rs.case = c
            h_mask = region_of_grid(c, prob.grid, c.uniformity)
            r = solve_uniform(prob, start, c.uniformity)
            check = prob.verify(r.choice) if aps_spec.verify else None
            info = uniformity_info(prob, r.choice, c.uniformity, r)
            res = finalize(prob, "greedy", r.choice, info, check, aps_spec, t0)
            cert = None
            if certify:
                res, cert = improve_and_certify(prob, res, aps_spec, rounds=rounds, time_limit_s=time_limit_s,
                                                max_iter=200)
            val = objective_value(prob, res.choice, "weighted", lam, h_mask)
            row = {"lambda": lam, "power_abs_mw": val["power_w"] / 1e6, "power_h_mw": val["power_h_w"] / 1e6,
                   "cv_2d": val["cv_2d"], "objective_mw": val["objective_w"] / 1e6, "status": res.status,
                   "peak_flux_kw_m2": res.check.peak_flux_kw_m2 if res.check is not None else None,
                   "n_defocused": res.n_defocused, "solve_time_s": round(_time.perf_counter() - t0, 3),
                   "choice": res.choice}
            if cert is not None:
                row.update({"upper_bound_mw": cert["upper_bound_mw"], "gap": cert["gap"]})
            out.append(row)
            start = res.choice
    finally:
        prob.rs.case = case0
    return out


# --------------------------------------------------------------------------- problem 2

def hex_cell_area(spacing_m: float) -> float:
    """Area of the regular hexagon with inradius d/2 (minimum truncated Voronoi cell), m2."""
    return math.sqrt(3.0) / 2.0 * spacing_m ** 2


def packing_capacity(mask: np.ndarray, cell_m: float, spacing_m: float) -> int:
    """Upper bound on the number of points with mutual distance >= spacing in a raster set.

    Each set cell is a square of side ``cell_m`` and points may lie anywhere in
    it. Their truncated Voronoi cells lie within d / sqrt(3) of the squares; the
    area of that region is over-counted by all raster cells whose centres lie
    within d / sqrt(3) + sqrt(2) cell_m of a set-cell centre, and divided by the
    hexagon area sqrt(3) d^2 / 2 (Fejes Toth's truncated Voronoi cell bound).
    """
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return 0
    # points lie anywhere in the set cells (half diagonal from a centre); a counted cell must also
    # cover any partially overlapped part of the dilated region (another half diagonal)
    r = spacing_m / math.sqrt(3.0) + cell_m * math.sqrt(2.0)
    pad = int(math.ceil(r / cell_m)) + 1
    m = np.pad(mask, pad)
    dist = distance_transform_edt(~m) * cell_m
    area = float((dist <= r).sum()) * cell_m ** 2
    return int(math.floor(area / hex_cell_area(spacing_m) + 1e-9))


@dataclass
class PotentialMap:
    x: np.ndarray  # (nx,) cell-centre x, m
    y: np.ndarray  # (ny,) cell-centre y, m
    cell_m: float
    energy_mwh: np.ndarray  # (ny, nx) annual absorbed energy of a lone heliostat at the cell centre (display)
    inside: np.ndarray  # (ny, nx) bool: the cell touches the site
    ub_mwh: np.ndarray | None = None  # (ny, nx) bound on the lone energy anywhere in the cell (certificate)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """Nearest-cell value at points (N, 2) (display only; bounds use the exact model)."""
        ix = np.clip(np.round((xy[:, 0] - self.x[0]) / self.cell_m).astype(int), 0, self.x.size - 1)
        iy = np.clip(np.round((xy[:, 1] - self.y[0]) / self.cell_m).astype(int), 0, self.y.size - 1)
        return self.energy_mwh[iy, ix]


def lone_heliostat_energy(case, xy: np.ndarray, samples, chunk: int = 4000) -> np.ndarray:
    """Annual absorbed energy (MWh) of a heliostat alone at each xy: cosine, reflectivity,
    attenuation, intercept (centre aim) and absorptivity; no shading, blocking or flux limit."""
    from . import optics, sun
    from .receiver import default_aim, effective_absorptivity, surface_point

    rec, hel = case.receiver, case.heliostat
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.zeros(len(xy))
    if not len(xy):
        return out
    for start in range(0, len(xy), chunk):
        p = xy[start:start + chunk]
        xyz = np.column_stack([p, np.full(len(p), hel.pivot_height)])
        az, z = default_aim(rec, xyz)
        aim = surface_point(rec, az, z)
        e = np.zeros(len(p))
        spatial = getattr(samples, "spatial", False)
        for k, (el, sa, dni, w) in enumerate(zip(samples.elevation, samples.azimuth, samples.dni, samples.weight_h)):
            if el <= 0 or w <= 0 or (not spatial and dni <= 0):
                continue
            if spatial:  # the DNI map of the state, read at each position
                dni = samples.dni_at(k, p)
            s = sun.sun_vector(el, sa)
            _, t, slant, cos_inc = optics.heliostat_geometry(xyz, aim, s)
            sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
            beam = hel.area * hel.reflectivity * hel.cleanliness * cos_inc * optics.attenuation(slant, case.atmosphere)
            e += dni * w * beam * optics.intercept_fraction(rec, aim, t, sigma) * effective_absorptivity(rec, az, sigma)
        out[start:start + chunk] = e / 1e6
    return out


TAIL_K = 5.0  # beam mass beyond 5 sigma laterally: 2 (1 - Phi(5)) = 5.7e-7


def cell_energy_upper_bound(case, centers_xy: np.ndarray, half_m: float, samples, chunk: int = 4000,
                            any_aim: bool = True) -> np.ndarray:
    """Annual energy (MWh) that a lone heliostat placed *anywhere* in each square cell cannot exceed.

    Cell = centre +- half_m in x and y; rho = half diagonal. For every point x of
    the cell the heliostat -> aim vector v(x) differs from v(c) (centre aim of the
    cell centre) by at most eps = rho (1 + R / r_min) (heliostat shift plus the
    shift of the facing point on the cylinder) plus, with ``any_aim``, the largest
    distance of any visible aim point from the facing mid-height point,
    sqrt(2 R^2 + (H/2)^2) -- so the bound holds for every aiming strategy (APS
    included), not only centre aiming. The direction of v turns by at most
    delta = asin(eps / |v(c)|). Per sun state:
      cosine        cos(theta / 2) <= cos(max(0, theta_c - delta) / 2)
      attenuation   decreasing in slant range -> value at |v(c)| - eps
      intercept     silhouette within the rectangle |u| <= R, |v| <= V(beta), V = H/2 cos beta + R sin beta,
                    Gaussian mass of that rectangle at the smallest beam sigma D_min * sigma_0
      absorptivity  largest panel absorptivity within the azimuths the beam can reach (the visible half
                    with any_aim, else facing azimuths +- asin(5 sigma_max / R)), plus the 5-sigma tail at
                    the largest absorptivity of all panels.
    Shading, blocking and flux limits only remove energy, so the bound also holds with neighbours.
    With spatial DNI maps (``samples.spatial``) the DNI of each state is replaced by the largest map
    value anywhere in the cell (``DNIMap.box_max``), so the bound holds for every position in the cell.
    """
    from scipy.special import ndtr

    from . import optics, sun
    from .receiver import _per_panel, default_aim, surface_point

    rec, hel = case.receiver, case.heliostat
    xy = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    out = np.zeros(len(xy))
    if not len(xy):
        return out
    R, H = rec.radius, rec.height
    rho = half_m * math.sqrt(2.0)
    s0 = math.sqrt((case.sun.effective_sigma_mrad * 1e-3) ** 2 + (2e-3 * hel.slope_error_mrad) ** 2
                   + (2e-3 * hel.tracking_error_mrad) ** 2 + (1e-3 * hel.specular_error_mrad) ** 2)
    d_m = math.sqrt(hel.area)
    alpha = _per_panel(rec, rec.panel_absorptivity, rec.absorptivity)
    alpha_max = float(alpha.max())
    uniform = bool(np.allclose(alpha, alpha[0]))
    width = 360.0 / rec.n_panels
    starts = rec.panel_start_azimuth_deg + width * np.arange(rec.n_panels)
    tail = 2.0 * (1.0 - ndtr(TAIL_K))
    const = hel.area * hel.reflectivity * hel.cleanliness
    states = [(k, float(el), float(sa), float(dni) * float(w)) for k, (el, sa, dni, w) in
              enumerate(zip(samples.elevation, samples.azimuth, samples.dni, samples.weight_h))
              if el > 0 and w > 0 and (samples.spatial or dni > 0)]
    spatial = getattr(samples, "spatial", False)
    for start in range(0, len(xy), chunk):
        p = xy[start:start + chunk]
        xyz = np.column_stack([p, np.full(len(p), hel.pivot_height)])
        az, z = default_aim(rec, xyz)
        v = surface_point(rec, az, z) - xyz
        dc = np.linalg.norm(v, axis=1)
        r_c = np.hypot(p[:, 0] - rec.center[0], p[:, 1] - rec.center[1])
        r_min = r_c - rho
        regular = r_min > R  # the cell does not reach the receiver footprint
        eps = rho * (1.0 + R / np.maximum(r_min, 1e-9)) + (math.sqrt(2.0 * R * R + 0.25 * H * H) if any_aim else 0.0)
        delta = np.where(regular & (eps < dc), np.arcsin(np.clip(eps / np.maximum(dc, 1e-9), 0.0, 1.0)), math.pi)
        d_min = np.maximum(dc - eps, 1e-3)
        d_max = dc + eps
        beta_c = np.arcsin(np.clip(v[:, 2] / np.maximum(dc, 1e-9), -1.0, 1.0))
        # largest half silhouette height over beta in [beta_c - delta, beta_c + delta]
        amp, beta_star = math.hypot(0.5 * H, R), math.atan2(R, 0.5 * H)
        b_lo, b_hi = beta_c - delta, beta_c + delta
        v_half = np.where((b_lo <= beta_star) & (beta_star <= b_hi), amp,
                          np.maximum(0.5 * H * np.cos(b_lo) + R * np.sin(b_lo), 0.5 * H * np.cos(b_hi) + R * np.sin(b_hi)))
        v_half = np.maximum(v_half, 0.0)
        sig_min = d_min * s0
        icpt = (2 * ndtr(R / sig_min) - 1.0) * (2 * ndtr(v_half / sig_min) - 1.0)
        icpt = np.where(regular, np.clip(icpt, 0.0, 1.0), 1.0)
        att = np.where(regular, optics.attenuation(d_min, case.atmosphere), 1.0)
        if uniform:
            a_ub = np.full(len(p), float(alpha[0]))
        else:
            focus = hel.focusing
            df_max = (np.ones_like(d_max) if focus == "slant_range" else np.zeros_like(d_max) if focus == "flat"
                      else d_max / float(focus))
            sig_ast = d_m * np.maximum(df_max, 1.0) / (4.0 * d_min)
            sig_max = d_max * np.sqrt(s0 ** 2 + sig_ast ** 2)
            dphi = np.degrees(np.arcsin(np.clip(rho / np.maximum(r_c, 1e-9), 0.0, 1.0)))
            # the beam centre may sit anywhere on the visible half with any_aim
            reach = dphi + (90.0 if any_aim else np.degrees(np.arcsin(np.clip(TAIL_K * sig_max / R, 0.0, 1.0))))
            reach = np.where(regular, reach, 180.0)
            # panel p intersects [az - reach, az + reach] if its sector start lies within reach + width
            rel = (starts[None, :] + 0.5 * width - az[:, None] + 180.0) % 360.0 - 180.0  # sector centre offsets
            hit = np.abs(rel) <= reach[:, None] + 0.5 * width
            a_ub = np.where(hit, alpha[None, :], 0.0).max(axis=1)
        e = np.zeros(len(p))
        for k, el, sa, weight in states:
            if spatial:
                # the largest map value anywhere in the cell (exact bound of the bilinear map) times the hours
                m = samples.map_of(k)
                peak = m.box_max(p, half_m)
                weight = float(samples.weight_h[k]) * (peak * float(samples.dni[k]) if m.unit == "factor" else peak)
            s = sun.sun_vector(el, sa)
            theta_c = np.arccos(np.clip((v @ s) / np.maximum(dc, 1e-9), -1.0, 1.0))
            cos_ub = np.where(regular, np.cos(np.maximum(theta_c - delta, 0.0) / 2.0), 1.0)
            frac = cos_ub * att * icpt
            e += weight * const * (frac * a_ub + (0.0 if uniform else alpha_max * tail))
        out[start:start + chunk] = e / 1e6
    return out


def potential_map(case, samples, cell_m: float | None = None, max_cells: int = 60_000) -> PotentialMap:
    """e0 on a raster covering the site (boundary polygon, exclusion zone)."""
    from .field import default_r_max, site_mask

    rec = case.receiver
    if case.field.boundary is not None:
        b = np.asarray(case.field.boundary, dtype=float)
        x0, y0 = b.min(axis=0)
        x1, y1 = b.max(axis=0)
    else:
        r = default_r_max(case)
        x0, y0, x1, y1 = rec.center[0] - r, rec.center[1] - r, rec.center[0] + r, rec.center[1] + r
    area = (x1 - x0) * (y1 - y0)
    if cell_m is None:
        cell_m = max(case.heliostat.diagonal / 4.0, math.sqrt(area / max_cells))
    xs = np.arange(x0 + 0.5 * cell_m, x1, cell_m)
    ys = np.arange(y0 + 0.5 * cell_m, y1, cell_m)
    xs = np.concatenate([[xs[0] - cell_m], xs, [xs[-1] + cell_m]])  # one ring of cells around the box
    ys = np.concatenate([[ys[0] - cell_m], ys, [ys[-1] + cell_m]])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, case.heliostat.pivot_height)])
    # every raster cell that touches the site (centre within half a cell diagonal), so that each
    # site point lies in a covered cell
    inside = site_mask(case, pts, tol=cell_m * math.sqrt(0.5))[0]
    e = np.full(gx.size, np.nan)
    e[inside] = lone_heliostat_energy(case, pts[inside, :2], samples)
    ub = np.full(gx.size, np.nan)
    ub[inside] = cell_energy_upper_bound(case, pts[inside, :2], 0.5 * cell_m, samples)
    return PotentialMap(xs, ys, float(cell_m), e.reshape(gx.shape), inside.reshape(gx.shape), ub.reshape(gx.shape))


def hourly_site_upper_bound(case, n_heliostats, spacing_m, hourly, pmap):
    """Packing bound for every recorded hour, every site position and every aim on the cylinder.

    cos(incidence) = sqrt((1 + s.t)/2) is concave in s.t. Within each solar-direction
    bin, Jensen's inequality bounds the DNI-weighted mean by its value at the mean
    sun vector (retain its length). The half diagonal of a site cell plus the radius
    of the receiver's enclosing sphere bounds all changes in the mirror-to-aim vector.
    """
    from dataclasses import replace
    from . import sun

    xx, yy = np.meshgrid(pmap.x, pmap.y)
    cx, cy, cz = case.receiver.center
    delta = np.stack([cx - xx, cy - yy, np.full_like(xx, cz - case.heliostat.pivot_height)], axis=-1)
    distance = np.linalg.norm(delta, axis=-1)
    radius = pmap.cell_m / math.sqrt(2) + math.hypot(case.receiver.radius, case.receiver.height / 2)
    cone = np.where(distance > radius, np.arcsin(np.minimum(1, radius / np.maximum(distance, 1e-15))), math.pi)
    direction = delta / np.maximum(distance[..., None], 1e-15)
    vectors = sun.sun_vector(hourly.elevation, hourly.azimuth)
    weights = hourly.dni * hourly.weight_h
    bins = np.floor(hourly.azimuth / 15).astype(int) + 24 * np.floor(hourly.elevation / 10).astype(int)
    spatial = getattr(hourly, "spatial", False)
    regime = np.asarray(hourly.map_index, dtype=int) if spatial else np.zeros(len(hourly), dtype=int)
    bins = bins + 100_000 * regime  # with spatial maps the DNI weight of a cell is the map maximum in it
    factor_of = {}
    if spatial:
        centres = np.column_stack([xx.ravel(), yy.ravel()])
        for r in np.unique(regime):
            m = hourly.maps[int(r)]
            factor_of[int(r)] = (m.box_max(centres, pmap.cell_m / 2.0).reshape(xx.shape), m.unit)
    envelope = np.zeros_like(distance)
    for label in np.unique(bins):
        subset = bins == label
        mass = float(weights[subset].sum())
        if mass <= 0:
            continue
        mean = (weights[subset] @ vectors[subset]) / mass
        length = np.linalg.norm(mean)
        angle = np.arccos(np.clip((direction @ mean) / max(length, 1e-15), -1, 1))
        dot_max = np.minimum(1.0, length * np.cos(np.maximum(0, angle - cone)))
        scale = 1.0
        if spatial:
            fmax, unit = factor_of[int(regime[subset][0])]
            if unit != "factor":
                raise ValueError("hourly bound needs factor maps (absolute maps come with a scenario table)")
            scale = fmax
        envelope += mass * scale * np.sqrt((1 + dot_max) / 2)
    h = case.heliostat
    alpha = max(case.receiver.panel_absorptivity or [case.receiver.absorptivity])
    envelope *= h.area * h.reflectivity * h.cleanliness * alpha / 1e6
    hourly_map = replace(pmap, ub_mwh=envelope * (1 + 1e-12))
    bound = layout_upper_bound(case, n_heliostats, spacing_m, pmap=hourly_map)
    return bound["upper_bound_mwh"]


def layout_upper_bound(case, n_heliostats: int, spacing_m: float, samples=None, pmap: PotentialMap | None = None,
                       n_levels: int = 400, layout_xyz: np.ndarray | None = None,
                       found_energy_mwh: float | None = None) -> dict:
    """Upper bound on the annual absorbed energy of any layout of N heliostats in the site.

    Every site point lies in a raster cell that touches the site, and
    ``cell_energy_upper_bound`` bounds the lone-heliostat energy anywhere in
    that cell, so #{i : e(x_i) >= t} <= cap(union of cells with bound >= t).
    ``samples`` must be the sun states of ``pmap``. Without a site boundary the
    site is the square of half-width ``field.default_r_max`` around the tower.
    A found energy above the bound is reported as ``bound_violation`` (it would
    mean a modelling inconsistency), never clipped.
    """
    from . import sun as sun_mod

    t0 = _time.perf_counter()
    if pmap is None:
        samples = samples if samples is not None else sun_mod.annual_samples(case, "sun_bins", n_bins=30)
        pmap = potential_map(case, samples)
    if pmap.ub_mwh is None:
        raise ValueError("potential map has no cell bounds; build it with certify.potential_map")
    env = np.where(pmap.inside, pmap.ub_mwh, -np.inf)
    vals = env[np.isfinite(env)]
    if vals.size == 0 or n_heliostats <= 0:
        return {"status": "empty_site", "upper_bound_mwh": 0.0}
    e_max = float(vals.max())
    # layer cake: integrate min(N, cap(L_t)) over t in [0, e_max]; cap(L_t) is non-increasing in t, so the
    # value at the left end of every level interval gives an upper Riemann sum
    levels = np.linspace(0.0, e_max, n_levels + 1)
    site_capacity = packing_capacity(env >= 0.0, pmap.cell_m, spacing_m)
    caps = [min(n_heliostats, site_capacity)]
    for t in levels[1:-1]:
        caps.append(min(n_heliostats, packing_capacity(env >= t, pmap.cell_m, spacing_m)))
    caps = np.asarray(caps, dtype=float)
    ub = float((caps * np.diff(levels)).sum())
    out = {
        "status": "ok" if site_capacity >= n_heliostats else "no_layout_exists",
        "upper_bound_mwh": ub,
        "lone_heliostat_max_mwh": e_max,
        "packing_capacity_bound": int(site_capacity),
        "n_heliostats": int(n_heliostats),
        "min_spacing_m": float(spacing_m),
        "raster_cell_m": pmap.cell_m,
        "n_levels": n_levels,
        "time_s": round(_time.perf_counter() - t0, 3),
        "model": "per-cell bound on the lone-heliostat annual energy (no shading/blocking, centre aim, "
                 "interval bounds on cosine, attenuation, intercept and absorptivity) + Fejes Toth packing bound",
    }
    if layout_xyz is not None and len(layout_xyz) and samples is not None:
        e_lone = lone_heliostat_energy(case, np.asarray(layout_xyz)[:, :2], samples)
        out["layout_lone_sum_mwh"] = float(e_lone.sum())
        out["interference_free_gap"] = 1.0 - out["layout_lone_sum_mwh"] / ub if ub > 0 else None
    if found_energy_mwh is not None:
        out["found_energy_mwh"] = float(found_energy_mwh)
        out["gap"] = 1.0 - found_energy_mwh / ub if ub > 0 else None
        if ub > 0 and found_energy_mwh > ub * (1 + 1e-9):
            out["status"] = "bound_violation"
    return out
