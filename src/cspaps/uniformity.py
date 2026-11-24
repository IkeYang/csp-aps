"""Uniform heating of the preheater region H: metrics, epsilon-constrained APS, sweeps, bounds.

Notation (advisor review, section 5): grid cells j with area A_j and
absorptivity alpha_j; incident flux Q_j, absorbed flux q_j = alpha_j Q_j;
H = preheater cells (all cells of H count, dark ones too).

    q_bar_H = sum_H A_j q_j / A_H
    CV_2D   = sqrt(sum_H A_j/A_H (q_j - q_bar_H)^2) / q_bar_H
    CV_circ = CV over azimuth columns of the height-averaged q
    CV_z    = CV over height rows of the azimuth-averaged q

The epsilon-constrained APS maximises absorbed power subject to the flux
limits, CV_2D <= epsilon and A_H q_bar_H >= P_H,min. It is solved by a
Lagrangian local search: maximise P_abs - mu * A_H * std_H(q) for a weight mu
found by bisection (the smallest mu that meets epsilon). Results are best
known feasible solutions; the edge-band bound below can prove that a target
is unreachable for a given candidate model, and ``exact_benchmark`` (needs
pyscipopt) gives proven optima on small problems.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np

from .config import Case, UniformitySpec
from .receiver import ReceiverGrid, region_mask


# --------------------------------------------------------------------------- metrics

def uniformity_metrics(q_map: np.ndarray, mask_map: np.ndarray | None = None) -> dict:
    """CV_2D / CV_circ / CV_height of a (n_height, n_azimuth) map over the masked region (equal areas)."""
    q = np.asarray(q_map, dtype=float)
    m = np.ones_like(q, dtype=bool) if mask_map is None else np.asarray(mask_map, dtype=bool)
    if not m.any():
        return {"cv_2d": float("nan"), "cv_circ": float("nan"), "cv_height": float("nan"), "mean": 0.0}
    vals = q[m]
    mean = vals.mean()

    def cv(x):
        x = np.asarray(x, dtype=float)
        return float(x.std() / x.mean()) if x.size and x.mean() > 0 else float("nan")

    cols = [q[m[:, c], c].mean() for c in range(q.shape[1]) if m[:, c].any()]
    rows = [q[r, m[r, :]].mean() for r in range(q.shape[0]) if m[r, :].any()]
    return {"cv_2d": cv(vals), "cv_circ": cv(cols), "cv_height": cv(rows), "mean": float(mean)}


def region_of_grid(case: Case, grid: ReceiverGrid, spec: UniformitySpec | None = None) -> np.ndarray:
    """(M,) mask of grid cells in H."""
    return region_mask(case.receiver, spec if spec is not None else case.uniformity, grid.cell_az, grid.cell_z)


# --------------------------------------------------------------------------- constrained APS

WEIGHTED_MAX_PASSES = 25  # weighted mode: coordinate passes until no move (cap)

@dataclass
class UniformResult:
    choice: np.ndarray
    power_abs_w: float
    power_h_w: float
    cv_2d: float
    mu: float
    status: str  # "feasible" | "not_reached" (best effort) | "below_min_load"
    history: list[dict] = dc_field(default_factory=list)
    solve_time_s: float = 0.0


class _State:
    """Incremental bookkeeping of absorbed flux on H and constraint flux for local search."""

    def __init__(self, prob, h_mask: np.ndarray, choice: np.ndarray):
        rs = prob.rs
        self.prob = prob
        self.n_grid = rs.n_grid
        self.area = rs.grid.cell_area
        self.alpha = rs.alpha
        self.h = np.zeros(rs.points.shape[0], dtype=bool)
        self.h[: self.n_grid] = h_mask
        self.a_h = self.area * h_mask.sum()
        self.choice = choice.copy()
        self.flux_con = prob.flux(self.choice)
        self.q = prob.flux(self.choice, basis="objective") * self.alpha  # absorbed flux at points
        self.s1 = float((self.area * self.q[self.h]).sum())
        self.s2 = float((self.area * self.q[self.h] ** 2).sum())

    def std(self, s1=None, s2=None) -> float:
        s1 = self.s1 if s1 is None else s1
        s2 = self.s2 if s2 is None else s2
        return math.sqrt(max(s2 / self.a_h - (s1 / self.a_h) ** 2, 0.0))

    def cv(self) -> float:
        mean = self.s1 / self.a_h
        return self.std() / mean if mean > 0 else float("inf")


def _local_search(prob, state: _State, mu: float, passes: int, absorb_power: np.ndarray, order,
                  prev_aim_xyz: np.ndarray | None = None, prev_active: np.ndarray | None = None) -> None:
    """Coordinate moves maximising absorbed power minus the uniformity penalty.

    Runs the compiled kernel (``kernels.local_search``, bit-identical to the loop below) when Numba is available."""
    from . import kernels

    limit = prob.limit
    p_ref = absorb_power[:, prob.center_k]
    if kernels.enabled():
        move_cost = None
        if prev_aim_xyz is not None and prev_active is not None:
            # the numpy loop's penalty, (penalty * p_ref_i) * |aim - previous aim| * active_i, for all i at once
            move = np.linalg.norm(prob.aim_xyz - np.asarray(prev_aim_xyz)[:, None, :], axis=2)
            move_cost = (prob.case.aps.move_penalty_per_m * p_ref[:, None]) * move * np.asarray(prev_active, bool)[:, None]
        kernels.local_search(prob, state, mu, passes, absorb_power, order, move_cost)
        return
    for _ in range(passes):
        moved = 0
        for i in order:
            if not prob.available[i]:
                continue
            idx = prob.cells[i]
            if idx.size == 0:
                continue
            old = state.choice[i]
            Fc = prob.rs.resp[i] * np.float32(prob.dni_con[i])  # (K, S) constraint basis
            Fq = prob.rs.resp[i] * np.float32(prob.dni_obj[i]) * state.alpha[idx][None, :]  # absorbed
            base_con = state.flux_con[idx] - (Fc[old] if old >= 0 else 0.0)
            base_q = state.q[idx] - (Fq[old] if old >= 0 else 0.0)
            hm = state.h[idx]
            # candidate rows: K aims + defocus (zeros)
            Kc = np.vstack([Fc, np.zeros((1, Fc.shape[1]), np.float32)])
            Kq = np.vstack([Fq, np.zeros((1, Fq.shape[1]), np.float32)])
            new_con = base_con[None, :] + Kc
            touch = Kc > 1e-3 * np.maximum(Kc.max(axis=1, keepdims=True), 1e-30)
            feasible = np.all((new_con <= limit[idx][None, :] * (1 + 1e-9)) | ~touch, axis=1)
            if not feasible[:-1].all():
                prob.flux_limited = int(getattr(prob, "flux_limited", 0)) + 1
            feasible[-1] = prob.case.aps.allow_defocus or old < 0
            if prob.reach is not None:
                feasible[:-1] &= prob.reach[i]
            new_q = base_q[None, :] + Kq
            old_q_h = state.q[idx][hm]
            d1 = state.area * (new_q[:, hm].sum(axis=1) - old_q_h.sum())
            d2 = state.area * ((new_q[:, hm] ** 2).sum(axis=1) - (old_q_h ** 2).sum())
            s1n, s2n = state.s1 + d1, state.s2 + d2
            std_n = np.sqrt(np.maximum(s2n / state.a_h - (s1n / state.a_h) ** 2, 0.0))
            p_new = np.append(absorb_power[i], 0.0)
            score = p_new - mu * state.a_h * std_n
            if prev_aim_xyz is not None and prev_active is not None:
                move = np.linalg.norm(prob.aim_xyz[i] - prev_aim_xyz[i][None, :], axis=1)
                score[:-1] -= prob.case.aps.move_penalty_per_m * p_ref[i] * move * bool(prev_active[i])
            score[~feasible] = -np.inf
            best = int(np.argmax(score))
            cur = state.prob.k if old < 0 else old
            tolerance = 1e-9 * max(abs(score[cur]), 1.0) if np.isfinite(score[cur]) else 0.0
            if best != cur and np.isfinite(score[best]) and score[best] > score[cur] + tolerance:
                new_c = -1 if best == prob.k else best
                state.flux_con[idx] = new_con[best]
                state.q[idx] = new_q[best]
                state.s1, state.s2 = float(s1n[best]), float(s2n[best])
                state.choice[i] = new_c
                moved += 1
        if moved == 0:
            break


def solve_uniform(prob, base_choice: np.ndarray, spec: UniformitySpec | None = None,
                  max_cv: float | None = None, prev_aim_xyz: np.ndarray | None = None,
                  prev_active: np.ndarray | None = None) -> UniformResult:
    """Maximise absorbed power with CV_2D(H) <= max_cv, starting from a flux-feasible plan."""
    t0 = _time.perf_counter()
    spec = spec or prob.case.uniformity
    eps = spec.max_cv if max_cv is None else max_cv
    h_mask = region_of_grid(prob.case, prob.grid, spec)
    if not h_mask.any():
        raise ValueError("uniformity region H contains no receiver grid cells")
    absorb_power = prob.absorbed_power
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")

    def run(mu: float, start: np.ndarray, passes: int | None = None) -> _State:
        st = _State(prob, h_mask, start)
        _local_search(prob, st, mu, spec.passes if passes is None else passes, absorb_power, order,
                      prev_aim_xyz, prev_active)
        return st

    hist = []
    tried: list[_State] = []

    def run_rec(mu: float, start: np.ndarray, passes: int | None = None) -> _State:
        st = run(mu, start, passes)
        tried.append(st)
        hist.append({"mu": mu, "cv_2d": st.cv(), "power_abs_mw": _p_abs(st, absorb_power) / 1e6})
        return st

    weighted = spec.mode == "weighted" and max_cv is None
    if weighted:
        mu_best = float(spec.weight)
        # a single fixed weight: run the coordinate search to a local optimum (it stops early)
        best = run_rec(mu_best, base_choice, max(spec.passes, WEIGHTED_MAX_PASSES))
    else:
        st0 = run_rec(0.0, base_choice)
        if st0.cv() <= eps:
            best, mu_best = st0, 0.0
        else:
            # bracket: the weight that makes std as costly as the total power per unit std
            hi = 1.0 / max(min(st0.cv(), 1e6), 1e-6)
            st_hi = run_rec(hi, st0.choice)
            k = 0
            while st_hi.cv() > eps and k < 8:
                hi *= 4.0
                st_hi = run_rec(hi, st_hi.choice)
                k += 1
            best, mu_best = (st_hi, hi) if st_hi.cv() <= eps else (None, hi)
            lo = 0.0
            if best is not None:
                for _ in range(spec.mu_iters):
                    mid = 0.5 * (lo + hi)
                    st = run_rec(mid, st0.choice)
                    if st.cv() <= eps:
                        if _p_abs(st, absorb_power) >= _p_abs(best, absorb_power):
                            best, mu_best = st, mid
                        hi = mid
                    else:
                        lo = mid
            else:
                # not reached: return the most uniform powered plan
                powered = [st for st in tried if st.s1 > 0]
                best = min(powered, key=lambda st: st.cv()) if powered else st0
    p_abs = _p_abs(best, absorb_power)
    p_h = best.s1
    min_required_w = max(spec.min_load_mw * 1e6, 0. if weighted else 1e-6)
    if p_h < min_required_w:
        status = "below_min_load"
    elif weighted:
        status = "weighted"
    else:
        status = "feasible" if best.cv() <= eps else "not_reached"
    return UniformResult(best.choice, p_abs, p_h, best.cv(), mu_best, status, hist, _time.perf_counter() - t0)


def _p_abs(state: _State, absorb_power: np.ndarray) -> float:
    on = state.choice >= 0
    return float(absorb_power[np.flatnonzero(on), state.choice[on]].sum())


def _candidate_alpha(prob) -> np.ndarray:
    """Per-candidate panel absorptivity, retained for external analysis callers."""
    from .receiver import absorptivity_at

    return absorptivity_at(prob.case.receiver, prob.cand_az)


def sweep(prob, base_choice: np.ndarray, eps_values, spec: UniformitySpec | None = None,
          exact: bool = False, exact_time_limit_s: float = 60.0) -> list[dict]:
    """Power-vs-uniformity curve.

    For every epsilon: the heuristic (status "best_known_feasible" or
    "not_reached") and, with ``exact=True`` (small problems, pyscipopt), the
    SCIP result ("proven_optimal", "infeasible_except_shutdown",
    "time_limited_feasible"). Proven optima are non-decreasing in epsilon.
    """
    spec = spec or prob.case.uniformity
    out = []
    for eps in sorted(eps_values):
        r = solve_uniform(prob, base_choice, spec, max_cv=eps)
        row = {"epsilon": float(eps), "power_abs_mw": r.power_abs_w / 1e6, "cv_2d": r.cv_2d,
               "status": "best_known_feasible" if r.status == "feasible" else r.status,
               "solve_time_s": r.solve_time_s}
        if exact:
            ex = exact_benchmark(prob, eps, spec, exact_time_limit_s)
            row.update({"exact_status": ex["status"], "exact_power_abs_mw": ex.get("power_abs_mw"),
                        "exact_upper_bound_mw": ex.get("upper_bound_mw")})
            if ex.get("power_abs_mw"):
                row["gap_to_exact"] = 1.0 - row["power_abs_mw"] / ex["power_abs_mw"] if row["status"] == "best_known_feasible" else None
        out.append(row)
    return out


# --------------------------------------------------------------------------- edge-band bound

def edge_band_bound(prob, power_floor_w: float, band_fraction: float = 0.125) -> dict:
    """Upper bound on the power share reaching the top+bottom bands, and the implied CV_2D lower bound.

    B = top and bottom bands, each ``band_fraction`` of the height (area share a).
    For any plan with intercepted power >= L (= power_floor_w), peak and
    uniformity constraints relaxed:
        P_B <= U(lambda) = sum_i max_k (b_ik + lambda p_ik) - lambda L   for all lambda >= 0
    with p_ik / b_ik the intercepted power on the whole surface / in B
    (exact band intercepts, defocus = 0). u = min U / L bounds the band power
    share; comparing the two area groups gives CV_2D >= max(0, a - u) / sqrt(a (1 - a)).
    Uses the shading of the centre aim and the given candidate set.
    """
    from scipy.optimize import minimize_scalar

    from .optics import intercept_fraction

    rec = prob.case.receiver
    rs = prob.rs
    z0, z1 = rec.center[2] - 0.5 * rec.height, rec.center[2] + 0.5 * rec.height
    hb = band_fraction * rec.height
    scale = (prob.dni_obj * prob.available)[:, None] * rs.unit_beam
    b = scale * (intercept_fraction(rec, rs.aim_xyz, rs.t, rs.sigma, z_range=(z0, z0 + hb))
                 + intercept_fraction(rec, rs.aim_xyz, rs.t, rs.sigma, z_range=(z1 - hb, z1)))
    p = prob.power
    b = np.hstack([b, np.zeros((b.shape[0], 1))]) / 1e6  # MW, + defocus column
    p = np.hstack([p, np.zeros((p.shape[0], 1))]) / 1e6
    L = power_floor_w / 1e6
    a = 2 * band_fraction

    def U(lam: float) -> float:
        return float(np.max(b + lam * p, axis=1).sum() - lam * L)

    if L > p.max(axis=1).sum() + 1e-12:
        return {"feasible": False, "reason": "power floor above the maximum intercepted power",
                "power_floor_mw": L}
    lams = np.concatenate([[0.0], np.logspace(-4, 3, 120)])
    vals = np.array([U(x) for x in lams])
    j = int(np.argmin(vals))
    lo, hi = lams[max(j - 1, 0)], lams[min(j + 1, lams.size - 1)]
    res = minimize_scalar(U, bounds=(lo, hi), method="bounded")
    u_best = min(vals[j], res.fun)
    lam_best = res.x if res.fun < vals[j] else lams[j]
    u = u_best / L if L > 0 else float("nan")
    cv_lb = max(0.0, a - u) / math.sqrt(a * (1 - a))
    return {
        "power_floor_mw": L, "band_fraction_each": band_fraction, "area_share_a": a,
        "band_power_upper_bound_mw": u_best, "band_share_upper_bound_u": u, "lambda": float(lam_best),
        "cv_2d_lower_bound": cv_lb, "n_candidates": prob.k,
        "assumptions": "HFLCAL response, centre-aim shading/blocking, candidate set as configured, "
                       "peak and uniformity constraints relaxed",
    }


# --------------------------------------------------------------------------- exact benchmark (optional)

def exact_benchmark(prob, eps: float, spec: UniformitySpec | None = None, time_limit_s: float = 60.0) -> dict:
    """Mixed-integer SOCP with SCIP (pyscipopt): proven optimum / infeasible / time-limited feasible.

    Only for small problems (tens of heliostats). Variables x_ik in {0, 1}; the
    uniformity constraint sum_H A (q - q_bar)^2 <= eps^2 q_bar^2 A_H is convex
    (a second-order cone in the affine q).
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("exact_benchmark needs pyscipopt: pip install pyscipopt") from exc
    spec = spec or prob.case.uniformity
    h_mask = region_of_grid(prob.case, prob.grid, spec)
    rs = prob.rs
    area = rs.grid.cell_area
    absorb = prob.absorbed_power
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit_s)
    x = {(i, k): m.addVar(vtype="B") for i in range(prob.n) for k in range(prob.k) if prob.available[i]}
    for i in range(prob.n):
        ks = [x[i, k] for k in range(prob.k) if (i, k) in x]
        if ks:
            m.addCons(quicksum(ks) <= 1)
    # flux limits and absorbed flux on H (objective DNI)
    con_terms: dict[int, list] = {}
    q_terms: dict[int, list] = {}
    for i in range(prob.n):
        if not prob.available[i]:
            continue
        idx = prob.cells[i]
        Fc = prob.rs.resp[i] * prob.dni_con[i]
        Fq = prob.rs.resp[i] * prob.dni_obj[i] * rs.alpha[idx][None, :]
        for k in range(prob.k):
            for s_, j in enumerate(idx):
                if Fc[k, s_] > 0:
                    con_terms.setdefault(int(j), []).append(float(Fc[k, s_]) * x[i, k])
                if j < rs.n_grid and h_mask[j] and Fq[k, s_] > 0:
                    q_terms.setdefault(int(j), []).append(float(Fq[k, s_]) * x[i, k])
    lim = prob.limit
    for j, terms in con_terms.items():
        m.addCons(quicksum(terms) <= float(lim[j]))
    h_cells = np.flatnonzero(h_mask)
    a_h = area * h_cells.size
    q = {j: m.addVar(lb=0.0) for j in h_cells}
    for j in h_cells:
        m.addCons(q[j] == quicksum(q_terms.get(int(j), [])))
    qbar = m.addVar(lb=0.0)
    m.addCons(qbar * a_h == quicksum(area * q[j] for j in h_cells))
    m.addCons(quicksum(area * (q[j] - qbar) * (q[j] - qbar) for j in h_cells) <= eps * eps * a_h * qbar * qbar)
    if spec.min_load_mw > 0:
        m.addCons(qbar * a_h >= spec.min_load_mw * 1e6)
    m.setObjective(quicksum(float(absorb[i, k]) * v for (i, k), v in x.items()), "maximize")
    m.optimize()
    st = m.getStatus()
    out = {"epsilon": eps, "scip_status": st}
    if m.getNSols() > 0:
        sol = m.getBestSol()
        choice = np.full(prob.n, -1)
        for (i, k), v in x.items():
            if m.getSolVal(sol, v) > 0.5:
                choice[i] = k
        out.update({"power_abs_mw": float(m.getSolObjVal(sol)) / 1e6, "choice": choice,
                    "upper_bound_mw": float(m.getDualbound()) / 1e6})
    out["status"] = {"optimal": "proven_optimal", "infeasible": "infeasible"}.get(
        st, "time_limited_feasible" if m.getNSols() > 0 else "unknown")
    if out["status"] == "proven_optimal" and out.get("power_abs_mw", 0.0) <= 1e-9:
        out["status"] = "infeasible_except_shutdown"  # only "all defocused" meets the constraints
    return out
