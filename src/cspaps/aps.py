"""Aim-point strategy (APS) optimisation for a cylindrical receiver.

Problem (one sun state, fixed field):
    maximise   sum_i P_i(k_i)                       power on the receiver
    subject to sum_i F_i,k_i(m) <= (1 - margin) * F_max   for every receiver cell m
where k_i is the aim-point candidate chosen for heliostat i (or "defocused").
Each candidate is a point on the cylinder surface (receiver azimuth offset x
height level). Flux superposes linearly, so every candidate's footprint
F_i,k is precomputed once with the analytic HFLCAL model.

Solvers:
    center   all heliostats on the surface point facing them at mid-height
    kfactor  aiming-factor rule: z = zc +- (H/2 - k sigma_i), alternating up/down;
             the largest k (least spillage) that meets the flux limit is found
             by bisection. Fast baseline.
    greedy   image-size-priority construction (largest beams first, each placed
             on the feasible candidate with the highest power, ties broken
             towards the lowest local flux) followed by coordinate-descent
             passes. Default; scales to large fields, supports per-heliostat
             DNI (clouds), a forecast DNI for the constraint, and a move
             penalty against the previous aims for smooth real-time operation.
    milp     exact 0/1 programme solved by HiGHS (small problems only). The LP
             relaxation also provides an upper bound to measure the greedy gap.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import optics, shading, sun
from .config import APSSpec, Case
from .field import Field
from .receiver import ReceiverGrid, build_grid, default_aim, heliostat_azimuth, surface_point

# Footprint cells below this fraction of the beam's own peak are dropped (~3.5 sigma);
# the omitted tail is ~0.2 % of the beam power and is covered by the safety margin.
FOOTPRINT_CUTOFF = 2e-3


@dataclass
class AimProblem:
    case: Case
    grid: ReceiverGrid
    cand_az: np.ndarray  # (N, K) receiver azimuth of each candidate, deg
    cand_z: np.ndarray  # (N, K) height of each candidate, m
    aim_xyz: np.ndarray  # (N, K, 3)
    power: np.ndarray  # (N, K) objective power on the receiver, W
    sigma: np.ndarray  # (N, K) beam sigma, m
    limit: np.ndarray  # (M,) allowed flux, W/m2
    center_k: int
    cells: list[np.ndarray]  # per heliostat: receiver cells its candidates touch
    fp: list[np.ndarray]  # per heliostat: (K, len(cells)) flux footprints, W/m2 (constraint DNI)
    build_time_s: float = 0.0

    @property
    def n(self) -> int:
        return self.cand_az.shape[0]

    @property
    def k(self) -> int:
        return self.cand_az.shape[1]

    def flux(self, choice: np.ndarray) -> np.ndarray:
        """Constraint-basis flux (M,) for a choice vector (-1 = defocused)."""
        flux = np.zeros(self.limit.size)
        for i in np.flatnonzero(choice >= 0):
            flux[self.cells[i]] += self.fp[i][choice[i]]
        return flux

    def total_power(self, choice: np.ndarray) -> float:
        on = choice >= 0
        return float(self.power[np.flatnonzero(on), choice[on]].sum())

    def aims(self, choice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.where(choice >= 0, choice, self.center_k)
        rows = np.arange(self.n)
        return self.cand_az[rows, k], self.cand_z[rows, k]


@dataclass
class AimResult:
    method: str
    aim: tuple[np.ndarray, np.ndarray]  # (azimuth_deg, z) per heliostat
    active: np.ndarray  # (N,) bool
    power_w: float  # objective power on the receiver
    peak_ratio: float  # peak constraint-basis flux / allowed flux (<= 1 is feasible)
    solve_time_s: float
    choice: np.ndarray | None = None
    info: dict = dc_field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.peak_ratio <= 1.0 + 1e-9

    @property
    def n_defocused(self) -> int:
        return int((~self.active).sum())

    def summary(self) -> dict:
        return {
            "method": self.method,
            "power_mw": round(self.power_w / 1e6, 4),
            "peak_to_allowed": round(self.peak_ratio, 4),
            "feasible": self.feasible,
            "n_defocused": self.n_defocused,
            "solve_time_s": round(self.solve_time_s, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.info.items()},
        }


def candidate_grid(spec: APSSpec, rec) -> tuple[np.ndarray, np.ndarray, int]:
    """(azimuth offsets (K,), heights (K,), centre index) of the candidate set."""
    offs = np.asarray(spec.azimuth_offsets_deg, dtype=float) if spec.azimuth_offsets_deg else np.zeros(1)
    L = max(1, int(spec.n_height_levels))
    zc = rec.center[2]
    span = 0.5 * rec.height * float(np.clip(spec.height_span, 0.0, 1.0))
    z = zc + (np.linspace(-span, span, L) if L > 1 else np.zeros(1))
    oo, zz = np.meshgrid(offs, z, indexing="ij")
    oo, zz = oo.ravel(), zz.ravel()
    center = int(np.argmin(np.abs(oo) * 1e3 + np.abs(zz - zc)))
    return oo, zz, center


def build_problem(
    case: Case,
    fld: Field,
    sun_elevation: float,
    sun_azimuth: float,
    dni,
    dni_constraint=None,
    grid: ReceiverGrid | None = None,
    spec: APSSpec | None = None,
    shading_points: int | None = None,
    chunk: int = 48,
) -> AimProblem:
    """Precompute candidate powers and flux footprints.

    ``dni`` (scalar or per heliostat) sets the objective; ``dni_constraint``
    (default: same) sets the flux used in the constraint -- pass the highest
    DNI expected before the next update to stay safe when clouds move away.
    """
    t0 = _time.perf_counter()
    spec = spec or case.aps
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    n = fld.n
    dni_obj = np.broadcast_to(np.asarray(dni, dtype=float), (n,))
    dni_con = dni_obj if dni_constraint is None else np.broadcast_to(np.asarray(dni_constraint, float), (n,))
    dni_con = np.maximum(dni_con, dni_obj)

    offs, zs, center = candidate_grid(spec, rec)
    k = offs.size
    base_az = heliostat_azimuth(rec, fld.xyz)
    cand_az = (base_az[:, None] + offs[None, :]) % 360.0
    cand_z = np.broadcast_to(zs, (n, k)).copy()
    aim_xyz = surface_point(rec, cand_az, cand_z)  # (N, K, 3)
    s = sun.sun_vector(sun_elevation, sun_azimuth)
    normals, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz[:, None, :], aim_xyz, s)

    if case.shading.enabled and n:
        sb = shading.shading_blocking(
            fld.xyz, normals[:, center], s, t[:, center], hel, rec, fld.pairs,
            n_points=shading_points or case.shading.n_points, tower_shadow=case.shading.tower_shadow,
        )["combined"]
    else:
        sb = np.ones(n)
    sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
    base = hel.area * hel.reflectivity * hel.cleanliness * sb[:, None] * cos_inc * optics.attenuation(slant, case.atmosphere)
    intercept = optics.intercept_fraction(rec, aim_xyz, t, sigma)
    power = dni_obj[:, None] * base * intercept
    beam_con = dni_con[:, None] * base

    cells: list[np.ndarray] = []
    fp: list[np.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        f = optics.footprints(
            grid, aim_xyz[start:stop].reshape(-1, 3), t[start:stop].reshape(-1, 3),
            sigma[start:stop].reshape(-1), beam_con[start:stop].reshape(-1),
        ).reshape(stop - start, k, -1)
        for blk in f:
            peak = blk.max()
            sup = np.flatnonzero(blk.max(axis=0) > FOOTPRINT_CUTOFF * peak) if peak > 0 else np.zeros(0, dtype=np.int64)
            cells.append(sup)
            fp.append(blk[:, sup].astype(np.float32))
    limit = np.full(grid.points.shape[0], rec.max_flux_kw_m2 * 1e3 * (1.0 - spec.safety_margin))
    return AimProblem(case, grid, cand_az, cand_z, aim_xyz, power, sigma, limit, center, cells, fp,
                      _time.perf_counter() - t0)


# --------------------------------------------------------------------------- solvers

def _peak_ratio(prob: AimProblem, choice: np.ndarray) -> float:
    return float((prob.flux(choice) / prob.limit).max()) if prob.n else 0.0


def solve_center(prob: AimProblem) -> np.ndarray:
    return np.full(prob.n, prob.center_k)


def solve_kfactor(prob: AimProblem, k_max: float = 4.0, iters: int = 14) -> tuple[np.ndarray, float]:
    """Aiming-factor rule; returns (choice, k). Larger k = aims closer to mid-height."""
    rec = prob.case.receiver
    zc, half = rec.center[2], 0.5 * rec.height
    sig = prob.sigma[:, prob.center_k]
    offs = prob.cand_az - prob.cand_az[:, [prob.center_k]]
    on_axis = np.abs((offs + 180) % 360 - 180) < 1e-9  # candidates on the facing azimuth
    levels = prob.cand_z[0]
    # alternate up/down within each 10-degree sector, ordered by beam size
    sector = np.floor(prob.cand_az[:, prob.center_k] / 10.0).astype(int)
    order = np.lexsort((sig, sector))
    sign = np.empty(prob.n)
    sign[order] = np.where(np.arange(prob.n) % 2 == 0, 1.0, -1.0)

    def choice_for(kf: float) -> np.ndarray:
        target = zc + sign * np.maximum(0.0, half - kf * sig)
        cost = np.abs(levels[None, :] - target[:, None]) + np.where(on_axis, 0.0, 1e6)
        return np.argmin(cost, axis=1)

    hi_choice = choice_for(k_max)
    if _peak_ratio(prob, hi_choice) <= 1.0:
        return hi_choice, k_max
    lo, hi = 0.0, k_max
    if _peak_ratio(prob, choice_for(lo)) > 1.0:
        return choice_for(lo), lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _peak_ratio(prob, choice_for(mid)) <= 1.0:
            lo = mid
        else:
            hi = mid
    return choice_for(lo), lo


def solve_greedy(
    prob: AimProblem,
    spec: APSSpec,
    prev_aim_xyz: np.ndarray | None = None,
    prev_active: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    n = prob.n
    choice = np.full(n, -1)
    flux = np.zeros(prob.limit.size)
    p_ref = prob.power[:, prob.center_k]
    if prev_aim_xyz is not None and spec.move_penalty_per_m > 0:
        move = np.linalg.norm(prob.aim_xyz - prev_aim_xyz[:, None, :], axis=-1)  # (N, K)
        if prev_active is not None:
            move = move * np.asarray(prev_active)[:, None]
        score_all = prob.power - spec.move_penalty_per_m * p_ref[:, None] * move
    else:
        score_all = prob.power
    tol = spec.tie_tolerance

    def choose(i: int) -> int:
        idx = prob.cells[i]
        if idx.size == 0:
            return int(np.argmax(score_all[i]))
        F = prob.fp[i]
        lim = prob.limit[idx]
        new = flux[idx][None, :] + F
        touch = F > 1e-3 * F.max(axis=1, keepdims=True)
        ratio = np.where(touch, new / lim, 0.0).max(axis=1)
        feasible = ratio <= 1.0
        score = score_all[i]
        if feasible.any():
            best = score[feasible].max()
            ties = feasible & (score >= best - tol * max(p_ref[i], 1e-9))
            return int(np.flatnonzero(ties)[np.argmin(ratio[ties])])
        return -1 if spec.allow_defocus else int(np.argmin(ratio))

    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")  # image-size priority
    for i in order:
        c = choose(i)
        choice[i] = c
        if c >= 0:
            flux[prob.cells[i]] += prob.fp[i][c]
    passes_done = 0
    for _ in range(max(0, spec.passes)):
        changed = 0
        for i in order:
            old = choice[i]
            if old >= 0:
                flux[prob.cells[i]] -= prob.fp[i][old]
            c = choose(i)
            choice[i] = c
            if c >= 0:
                flux[prob.cells[i]] += prob.fp[i][c]
            changed += int(c != old)
        passes_done += 1
        if changed == 0:
            break
    return choice, {"passes": passes_done}


def _milp_matrices(prob: AimProblem):
    from scipy.sparse import coo_matrix

    n, k = prob.n, prob.k
    rows, cols, vals = [], [], []
    for i in range(n):
        idx = prob.cells[i]
        if idx.size == 0:
            continue
        kk, ss = np.nonzero(prob.fp[i] > 0)
        rows.append(idx[ss])
        cols.append(i * k + kk)
        vals.append(prob.fp[i][kk, ss].astype(float))
    m = prob.limit.size
    if rows:
        a_flux = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(m, n * k)).tocsr()
    else:
        a_flux = coo_matrix((m, n * k)).tocsr()
    a_one = coo_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), np.arange(n * k))),
                       shape=(n, n * k)).tocsr()
    return a_flux, a_one


def solve_milp(prob: AimProblem, spec: APSSpec, relax: bool = False, max_nnz: int = 4_000_000):
    """Exact 0/1 programme (or its LP relaxation) with HiGHS. Returns (choice, info)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    nnz = sum(f.size for f in prob.fp)
    if nnz > max_nnz:
        raise ValueError(
            f"MILP too large ({nnz:,} flux coefficients > {max_nnz:,}); use aps.method 'greedy' "
            "or fewer candidates / a coarser receiver grid"
        )
    a_flux, a_one = _milp_matrices(prob)
    n, k = prob.n, prob.k
    ub_one = np.ones(n)  # at most one aim point per heliostat
    lb_one = np.zeros(n) if spec.allow_defocus else np.ones(n)
    res = milp(
        c=-prob.power.ravel(),
        integrality=np.zeros(n * k) if relax else np.ones(n * k),
        bounds=Bounds(0, 1),
        constraints=[LinearConstraint(a_one, lb_one, ub_one),
                     LinearConstraint(a_flux, -np.inf, prob.limit)],
        options={"time_limit": spec.milp_time_limit_s, "disp": False},
    )
    info = {"status": int(res.status), "message": str(res.message)}
    if res.x is None:
        return np.full(n, -1), {**info, "objective_w": 0.0}
    x = res.x.reshape(n, k)
    info["objective_w"] = float(-res.fun)
    if relax:
        return None, info
    choice = np.where(x.max(axis=1) > 0.5, np.argmax(x, axis=1), -1)
    return choice, info


def lp_upper_bound(prob: AimProblem, spec: APSSpec | None = None) -> float:
    """Upper bound (W) on the power any aim assignment can achieve under the flux limit."""
    spec = spec or prob.case.aps
    _, info = solve_milp(prob, spec, relax=True)
    return info["objective_w"]


# --------------------------------------------------------------------------- entry point

def optimize_aims(
    case: Case,
    fld: Field,
    sun_elevation: float,
    sun_azimuth: float,
    dni,
    dni_constraint=None,
    method: str | None = None,
    grid: ReceiverGrid | None = None,
    prev: tuple[np.ndarray, np.ndarray] | None = None,
    prev_active: np.ndarray | None = None,
    shading_points: int | None = None,
    spec: APSSpec | None = None,
    problem: AimProblem | None = None,
) -> AimResult:
    """Choose an aim point for every heliostat at one sun state."""
    spec = spec or case.aps
    method = method or spec.method
    t0 = _time.perf_counter()
    if method == "center" and problem is None:
        az, z = default_aim(case.receiver, fld.xyz)
        return AimResult("center", (az, z), np.ones(fld.n, dtype=bool), float("nan"), float("nan"),
                         _time.perf_counter() - t0)
    prob = problem or build_problem(case, fld, sun_elevation, sun_azimuth, dni, dni_constraint,
                                    grid, spec, shading_points)
    info: dict = {"build_time_s": prob.build_time_s, "n_candidates": prob.k}
    if method == "center":
        choice = solve_center(prob)
    elif method == "kfactor":
        choice, kf = solve_kfactor(prob)
        info["k_factor"] = kf
    elif method == "greedy":
        prev_xyz = surface_point(case.receiver, *prev) if prev is not None else None
        choice, extra = solve_greedy(prob, spec, prev_xyz, prev_active)
        info.update(extra)
    elif method == "milp":
        choice, extra = solve_milp(prob, spec)
        info.update(extra)
    else:
        raise ValueError(f"Unknown APS method '{method}'")
    aim = prob.aims(choice)
    return AimResult(method, aim, choice >= 0, prob.total_power(choice), _peak_ratio(prob, choice),
                     _time.perf_counter() - t0, choice, info)
