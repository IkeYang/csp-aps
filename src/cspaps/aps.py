"""Aim-point strategy (APS) optimisation for a cylindrical receiver.

Problem (one sun state, fixed field):
    maximise   sum_i D_i * p_i(k_i)                       intercepted power
    subject to sum_i Dc_i * K_i,k_i(m) <= (1 - margin) * F_lim(m)   every check point m
where k_i is the aim-point candidate of heliostat i (or "defocused"), D_i the
current DNI, Dc_i the DNI used for the constraint (current or forecast
maximum) and K the flux response per unit DNI. F_lim is the incident-flux
limit of the panel at m (panel limits, height profile, absorbed basis and
model bias included, see receiver.py).

The DNI-independent part -- geometry, shading, intercept and the sparse unit
responses K -- is a ``ResponseSet``. It depends only on the field, receiver,
candidate set and sun position, so an online controller builds it once and
reuses it while the sun moves little; every DNI update only rescales rows.

Every solution is verified on an adaptive fine grid with full beams
(verify.py). Hotspots above the allowed flux are added to the response set as
extra constraint points and the problem is re-solved (constraint generation).
The final status is "ok", "margin_reduced" (within the limit but not the
margin) or "infeasible", and is the one reported by the CLI and the GUI.

Solvers:
    center   all heliostats on the facing mid-height point (reference)
    kfactor  aiming-factor rule z = zc +- (H/2 - k sigma_i), largest feasible k by bisection
    greedy   image-size-priority construction + coordinate-descent passes; warm start
             from the last executed plan with violation repair and priority passes
    milp     exact 0/1 programme with HiGHS (small problems); its LP relaxation
             gives an upper bound to measure the greedy gap
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import optics, shading, sun
from .config import APSSpec, Case
from .field import Field
from .receiver import (
    ReceiverGrid, absorptivity_at, build_grid, default_aim, heliostat_azimuth, incident_limit_w_m2,
    surface_normal, surface_point,
)
from .verify import FluxCheck, verify_beams

# Footprint cells below this fraction of the beam's own peak are dropped (~3.5 sigma);
# the omitted tail is ~0.2 % of the beam power. Final checks use full beams (verify.py).
FOOTPRINT_CUTOFF = 2e-3


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


@dataclass
class ResponseSet:
    """DNI-independent candidate data for one field / receiver / sun position."""

    case: Case
    grid: ReceiverGrid
    sun_elevation: float
    sun_azimuth: float
    cand_az: np.ndarray  # (N, K) receiver azimuth of each candidate, deg
    cand_z: np.ndarray  # (N, K) height of each candidate, m
    aim_xyz: np.ndarray  # (N, K, 3)
    t: np.ndarray  # (N, K, 3) heliostat -> aim unit vectors
    sigma: np.ndarray  # (N, K) beam sigma, m
    unit_beam: np.ndarray  # (N, K) beam power at the image plane per unit DNI, W per W/m2
    unit_power: np.ndarray  # (N, K) intercepted power per unit DNI
    sb: np.ndarray  # (N,) shading x blocking (centre-aim geometry)
    center_k: int
    points: np.ndarray  # (P, 3) constraint points: grid cells first, then added hotspots
    normals: np.ndarray  # (P, 3)
    point_az: np.ndarray  # (P,)
    point_z: np.ndarray  # (P,)
    limit_nom: np.ndarray  # (P,) incident limit without margin, W/m2
    alpha: np.ndarray  # (P,) absorptivity
    cells: list[np.ndarray]  # per heliostat: constraint points its candidates touch (sorted)
    resp: list[np.ndarray]  # per heliostat: (K, len(cells)) flux per unit DNI, float32
    resp_peak: np.ndarray  # (N,) max of each heliostat's response
    build_time_s: float = 0.0
    version: int = 0

    @property
    def n(self) -> int:
        return self.cand_az.shape[0]

    @property
    def k(self) -> int:
        return self.cand_az.shape[1]

    @property
    def n_grid(self) -> int:
        return self.grid.points.shape[0]

    def add_points(self, azimuth_deg, z, chunk: int = 256) -> int:
        """Add constraint points (e.g. verified hotspots). Returns the number added."""
        az = np.atleast_1d(np.asarray(azimuth_deg, dtype=float)) % 360.0
        zz = np.atleast_1d(np.asarray(z, dtype=float))
        if az.size == 0:
            return 0
        rec = self.case.receiver
        pts, nrm = surface_point(rec, az, zz), surface_normal(az)
        offset = self.points.shape[0]
        n, k = self.n, self.k
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            f = optics.footprints_at(pts, nrm, self.aim_xyz[start:stop].reshape(-1, 3),
                                     self.t[start:stop].reshape(-1, 3), self.sigma[start:stop].reshape(-1),
                                     self.unit_beam[start:stop].reshape(-1)).reshape(stop - start, k, -1)
            for j, blk in enumerate(f):
                i = start + j
                keep = np.flatnonzero(blk.max(axis=0) > FOOTPRINT_CUTOFF * self.resp_peak[i]) if self.resp_peak[i] > 0 else []
                if len(keep):
                    self.cells[i] = np.concatenate([self.cells[i], offset + np.asarray(keep)])
                    self.resp[i] = np.hstack([self.resp[i], blk[:, keep].astype(np.float32)])
        self.points = np.vstack([self.points, pts])
        self.normals = np.vstack([self.normals, nrm])
        self.point_az = np.concatenate([self.point_az, az])
        self.point_z = np.concatenate([self.point_z, zz])
        self.limit_nom = np.concatenate([self.limit_nom, incident_limit_w_m2(rec, az, zz)])
        self.alpha = np.concatenate([self.alpha, absorptivity_at(rec, az)])
        self.version += 1
        return int(az.size)

    def sun_shift_deg(self, sun_elevation: float, sun_azimuth: float) -> float:
        a = sun.sun_vector(self.sun_elevation, self.sun_azimuth)
        b = sun.sun_vector(sun_elevation, sun_azimuth)
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def solver_grid(rec, grid: ReceiverGrid, sigma_min: float, max_cells: int) -> ReceiverGrid:
    """The given grid, refined until cells are <= sigma_min / 1.5 (within ``max_cells``)."""
    import math

    h = sigma_min / 1.5
    n_az = max(grid.azimuth_deg.size, int(math.ceil(2 * math.pi * rec.radius / h)))
    n_z = max(grid.z.size, int(math.ceil(rec.height / h)))
    if n_az * n_z > max_cells:
        f = math.sqrt(max_cells / (n_az * n_z))
        n_az, n_z = max(grid.azimuth_deg.size, int(n_az * f)), max(grid.z.size, int(n_z * f))
    if (n_az, n_z) == (grid.azimuth_deg.size, grid.z.size):
        return grid
    return build_grid(rec, n_az, n_z)


def build_response(
    case: Case,
    fld: Field,
    sun_elevation: float,
    sun_azimuth: float,
    grid: ReceiverGrid | None = None,
    spec: APSSpec | None = None,
    shading_points: int | None = None,
    chunk: int = 48,
) -> ResponseSet:
    """Geometry, shading, intercept and sparse unit-DNI flux responses of all candidates."""
    t0 = _time.perf_counter()
    spec = spec or case.aps
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    n = fld.n
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
    unit_beam = hel.area * hel.reflectivity * hel.cleanliness * sb[:, None] * cos_inc * optics.attenuation(slant, case.atmosphere)
    unit_power = unit_beam * optics.intercept_fraction(rec, aim_xyz, t, sigma)
    if spec.auto_grid and n:
        grid = solver_grid(rec, grid, float(sigma.min()), spec.max_solver_cells)

    cells: list[np.ndarray] = []
    resp: list[np.ndarray] = []
    peaks = np.zeros(n)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        f = optics.footprints(
            grid, aim_xyz[start:stop].reshape(-1, 3), t[start:stop].reshape(-1, 3),
            sigma[start:stop].reshape(-1), unit_beam[start:stop].reshape(-1),
        ).reshape(stop - start, k, -1)
        for j, blk in enumerate(f):
            peak = blk.max()
            peaks[start + j] = peak
            sup = np.flatnonzero(blk.max(axis=0) > FOOTPRINT_CUTOFF * peak) if peak > 0 else np.zeros(0, dtype=np.int64)
            cells.append(sup)
            resp.append(blk[:, sup].astype(np.float32))
    return ResponseSet(
        case, grid, float(sun_elevation), float(sun_azimuth), cand_az, cand_z, aim_xyz, t, sigma,
        unit_beam, unit_power, sb, center, grid.points.copy(), grid.normals.copy(), grid.cell_az, grid.cell_z,
        grid.limit.copy(), grid.alpha.copy(), cells, resp, peaks, _time.perf_counter() - t0,
    )


@dataclass
class AimProblem:
    """A response set plus the DNI state of one control cycle."""

    rs: ResponseSet
    dni_obj: np.ndarray  # (N,) DNI for the objective (current)
    dni_con: np.ndarray  # (N,) DNI for the flux constraint (>= dni_obj)
    available: np.ndarray  # (N,) bool, False = heliostat out of service
    margin: float
    reach: np.ndarray | None = None  # (N, K) bool: candidate reachable within the slew budget
    _fp: list | None = dc_field(default=None, repr=False)
    _fp_version: int = -1

    # -- pass-throughs used by the solvers
    @property
    def case(self) -> Case:
        return self.rs.case

    @property
    def grid(self) -> ReceiverGrid:
        return self.rs.grid

    @property
    def n(self) -> int:
        return self.rs.n

    @property
    def k(self) -> int:
        return self.rs.k

    @property
    def center_k(self) -> int:
        return self.rs.center_k

    @property
    def sigma(self) -> np.ndarray:
        return self.rs.sigma

    @property
    def cand_az(self) -> np.ndarray:
        return self.rs.cand_az

    @property
    def cand_z(self) -> np.ndarray:
        return self.rs.cand_z

    @property
    def aim_xyz(self) -> np.ndarray:
        return self.rs.aim_xyz

    @property
    def cells(self) -> list[np.ndarray]:
        return self.rs.cells

    @property
    def build_time_s(self) -> float:
        return self.rs.build_time_s

    @property
    def power(self) -> np.ndarray:
        """(N, K) objective power on the receiver, W."""
        return self.rs.unit_power * (self.dni_obj * self.available)[:, None]

    @property
    def limit(self) -> np.ndarray:
        """(P,) allowed incident flux incl. the safety margin, W/m2."""
        return self.rs.limit_nom * (1.0 - self.margin)

    def F(self, i: int) -> np.ndarray:
        """Constraint-basis footprints (K, S) of heliostat i, W/m2."""
        return self.rs.resp[i] * np.float32(self.dni_con[i])

    @property
    def fp(self) -> list[np.ndarray]:
        if self._fp is None or self._fp_version != self.rs.version:
            self._fp = [self.F(i) for i in range(self.n)]
            self._fp_version = self.rs.version
        return self._fp

    def flux(self, choice: np.ndarray, basis: str = "constraint") -> np.ndarray:
        """Flux (P,) at the constraint points for a choice vector (-1 = defocused)."""
        scale = self.dni_con if basis == "constraint" else self.dni_obj
        flux = np.zeros(self.rs.points.shape[0])
        for i in np.flatnonzero(choice >= 0):
            flux[self.rs.cells[i]] += self.rs.resp[i][choice[i]] * scale[i]
        return flux

    def total_power(self, choice: np.ndarray) -> float:
        on = choice >= 0
        return float(self.power[np.flatnonzero(on), choice[on]].sum())

    def aims(self, choice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.where(choice >= 0, choice, self.center_k)
        rows = np.arange(self.n)
        return self.cand_az[rows, k], self.cand_z[rows, k]

    def verify(self, choice: np.ndarray, max_cells: int | None = None) -> FluxCheck:
        on = np.flatnonzero(choice >= 0)
        k = choice[on]
        return verify_beams(
            self.case.receiver, self.rs.aim_xyz[on, k], self.rs.t[on, k], self.rs.sigma[on, k],
            self.rs.unit_beam[on, k] * self.dni_con[on], self.margin,
            max_cells=max_cells or self.case.aps.verify_max_cells,
        ).remap(on)


def _vec(x, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()


def make_problem(rs: ResponseSet, dni, dni_constraint=None, available=None, margin: float | None = None) -> AimProblem:
    n = rs.n
    dni_obj = _vec(dni, n)
    dni_con = dni_obj.copy() if dni_constraint is None else np.maximum(_vec(dni_constraint, n), dni_obj)
    avail = np.ones(n, dtype=bool) if available is None else np.asarray(available, dtype=bool).copy()
    m = rs.case.aps.safety_margin if margin is None else margin
    return AimProblem(rs, dni_obj, dni_con, avail, float(m))


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
    available=None,
) -> AimProblem:
    """Response set + DNI state in one call (offline use)."""
    rs = build_response(case, fld, sun_elevation, sun_azimuth, grid, spec, shading_points)
    return make_problem(rs, dni, dni_constraint, available, (spec or case.aps).safety_margin)


# --------------------------------------------------------------------------- results

@dataclass
class AimResult:
    method: str
    aim: tuple[np.ndarray, np.ndarray]  # (azimuth_deg, z) per heliostat
    active: np.ndarray  # (N,) bool
    power_w: float  # objective power on the receiver
    peak_ratio: float  # verified peak flux / allowed flux incl. margin (<= 1: margin kept)
    solve_time_s: float
    choice: np.ndarray | None = None
    info: dict = dc_field(default_factory=dict)
    status: str = "unverified"  # "ok" | "margin_reduced" | "infeasible" | "unverified"
    check: FluxCheck | None = None

    @property
    def feasible(self) -> bool:
        """Within the hard flux limit (the safety margin may be partly used)."""
        if self.check is not None:
            return self.check.peak_ratio <= 1.0 + 1e-9
        return self.peak_ratio <= 1.0 + 1e-9

    @property
    def n_defocused(self) -> int:
        return int((~self.active).sum())

    def summary(self) -> dict:
        out = {
            "method": self.method,
            "status": self.status,
            "power_mw": round(self.power_w / 1e6, 4),
            "peak_to_allowed": round(self.peak_ratio, 4),
            "feasible": self.feasible,
            "n_defocused": self.n_defocused,
            "solve_time_s": round(self.solve_time_s, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.info.items()},
        }
        if self.check is not None:
            out["flux_check"] = self.check.summary()
        return out


# --------------------------------------------------------------------------- solvers

def _peak_ratio(prob: AimProblem, choice: np.ndarray) -> float:
    return float((prob.flux(choice) / prob.limit).max()) if prob.n else 0.0


def solve_center(prob: AimProblem) -> np.ndarray:
    return np.where(prob.available, prob.center_k, -1)


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
        return np.where(prob.available, np.argmin(cost, axis=1), -1)

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
    init: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    full_passes: int | None = None,
    deadline: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Greedy APS.

    Cold (``init`` None): image-size-priority construction, then ``spec.passes``
    coordinate-descent passes over all heliostats.
    Warm (``init`` given, e.g. the last executed plan): repair every
    violation by moving the heliostats that cause it, then improvement passes
    over ``priority`` (heliostats whose DNI changed, near hotspots, defocused),
    then ``full_passes`` passes over everyone (default 0 when warm).
    ``deadline`` (time.perf_counter value) stops improvement passes early; the
    construction / repair always completes, so the plan is flux-feasible.
    """
    n = prob.n
    limit = prob.limit
    p_ref = prob.power[:, prob.center_k]
    if prev_aim_xyz is not None and spec.move_penalty_per_m > 0:
        move = np.linalg.norm(prob.aim_xyz - prev_aim_xyz[:, None, :], axis=-1)  # (N, K)
        if prev_active is not None:
            move = move * np.asarray(prev_active)[:, None]
        score_all = prob.power - spec.move_penalty_per_m * p_ref[:, None] * move
    else:
        score_all = prob.power
    tol = spec.tie_tolerance
    avail = prob.available
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")  # image-size priority
    stats = {"moves": 0}

    def choose(i: int) -> int:
        if not avail[i]:
            return -1
        idx = prob.cells[i]
        if idx.size == 0 or prob.dni_con[i] <= 0:
            return int(np.argmax(score_all[i]))
        F = prob.F(i)
        lim = limit[idx]
        new = flux[idx][None, :] + F
        touch = F > 1e-3 * F.max(axis=1, keepdims=True)
        ratio = np.where(touch, new / lim, 0.0).max(axis=1)
        feasible = ratio <= 1.0
        if reach is not None:
            feasible &= reach[i]
        score = score_all[i]
        if feasible.any():
            best = score[feasible].max()
            ties = feasible & (score >= best - tol * max(p_ref[i], 1e-9))
            return int(np.flatnonzero(ties)[np.argmin(ratio[ties])])
        return -1 if spec.allow_defocus else int(np.argmin(ratio))

    reach = prob.reach
    timed_out = {"v": False}

    def late() -> bool:
        if deadline is not None and _time.perf_counter() > deadline:
            timed_out["v"] = True
        return timed_out["v"]

    def reassign(i: int) -> None:
        old = choice[i]
        if old >= 0:
            flux[prob.cells[i]] -= prob.F(i)[old]
        c = choose(i)
        choice[i] = c
        if c >= 0:
            flux[prob.cells[i]] += prob.F(i)[c]
        stats["moves"] += int(c != old)

    if init is None:
        choice = np.full(n, -1)
        flux = np.zeros(limit.size)
        for i in order:
            reassign(i)
        n_full = spec.passes if full_passes is None else full_passes
        repairs = 0
    else:
        choice = np.where(avail, np.asarray(init, dtype=int), -1)
        flux = prob.flux(choice)
        repairs = 0
        for _ in range(3):  # repair: heliostats feeding violated points must move or defocus
            viol = flux > limit * (1 + 1e-9)
            if not viol.any():
                break
            culprits = []
            for i in np.flatnonzero(choice >= 0):
                idx = prob.cells[i]
                hit = viol[idx]
                if hit.any():
                    culprits.append((-float(prob.F(i)[choice[i]][hit].sum()), i))
            for _, i in sorted(culprits):
                reassign(i)
                repairs += 1
        if priority is not None:
            for i in order[np.isin(order, np.flatnonzero(priority))]:
                if late():
                    break
                reassign(i)
        n_full = 0 if full_passes is None else full_passes
    passes_done = 0
    for _ in range(max(0, n_full)):
        if late():
            break
        before = stats["moves"]
        for i in order:
            reassign(i)
        passes_done += 1
        if stats["moves"] == before:
            break
    return choice, {"passes": passes_done, "repairs": repairs, "timed_out": timed_out["v"]}


def _milp_matrices(prob: AimProblem):
    from scipy.sparse import coo_matrix

    n, k = prob.n, prob.k
    rows, cols, vals = [], [], []
    for i in range(n):
        idx = prob.cells[i]
        if idx.size == 0:
            continue
        F = prob.F(i)
        kk, ss = np.nonzero(F > 0)
        rows.append(idx[ss])
        cols.append(i * k + kk)
        vals.append(F[kk, ss].astype(float))
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

    nnz = sum(r.size for r in prob.rs.resp)
    if nnz > max_nnz:
        raise ValueError(
            f"MILP too large ({nnz:,} flux coefficients > {max_nnz:,}); use aps.method 'greedy' "
            "or fewer candidates / a coarser receiver grid"
        )
    a_flux, a_one = _milp_matrices(prob)
    n, k = prob.n, prob.k
    ub_one = np.ones(n)  # at most one aim point per heliostat
    lb_one = np.where(prob.available & (not spec.allow_defocus), 1.0, 0.0)
    ub_x = np.repeat(prob.available.astype(float), k)
    if prob.reach is not None:
        ub_x = ub_x * prob.reach.ravel()
    res = milp(
        c=-prob.power.ravel(),
        integrality=np.zeros(n * k) if relax else np.ones(n * k),
        bounds=Bounds(np.zeros(n * k), ub_x),
        constraints=[LinearConstraint(a_one, lb_one, ub_one),
                     LinearConstraint(a_flux, -np.inf, prob.limit)],
        options={"time_limit": spec.milp_time_limit_s, "disp": False},
    )
    info = {"milp_status": int(res.status), "milp_message": str(res.message)}
    if res.x is None:
        return np.full(n, -1), {**info, "objective_w": 0.0}
    x = res.x.reshape(n, k)
    info["objective_w"] = float(-res.fun)
    if relax:
        return None, info
    choice = np.where(x.max(axis=1) > 0.5, np.argmax(x, axis=1), -1)
    return choice, info


def lp_upper_bound(prob: AimProblem, spec: APSSpec | None = None) -> float:
    """Upper bound (W) on the power any aim assignment can achieve under the flux limit
    at the current constraint points (valid for this candidate set and response model)."""
    spec = spec or prob.case.aps
    _, info = solve_milp(prob, spec, relax=True)
    return info["objective_w"]


# --------------------------------------------------------------------------- entry point

_STATUS = {"ok": "ok", "margin_reduced": "margin_reduced", "over_limit": "infeasible"}


def solve(
    prob: AimProblem,
    method: str,
    spec: APSSpec | None = None,
    prev: tuple[np.ndarray, np.ndarray] | None = None,
    prev_active: np.ndarray | None = None,
    init: np.ndarray | None = None,
    priority: np.ndarray | None = None,
) -> AimResult:
    """Solve, verify on a fine grid, add hotspots as constraints and re-solve if needed."""
    spec = spec or prob.case.aps
    t0 = _time.perf_counter()
    rec = prob.case.receiver
    prev_xyz = surface_point(rec, *prev) if prev is not None else None
    info: dict = {"build_time_s": prob.build_time_s, "n_candidates": prob.k}

    def run(warm: np.ndarray | None):
        if method == "center":
            return solve_center(prob), {}
        if method == "kfactor":
            ch, kf = solve_kfactor(prob)
            return ch, {"k_factor": kf}
        if method == "greedy":
            return solve_greedy(prob, spec, prev_xyz, prev_active, init=warm, priority=priority)
        if method == "milp":
            return solve_milp(prob, spec)
        raise ValueError(f"Unknown APS method '{method}'")

    choice, extra = run(init)
    info.update(extra)
    check = None
    rounds = added = 0
    if spec.verify:
        check = prob.verify(choice)
        while (method != "center" and not check.ok and check.hotspots and rounds < spec.verify_max_rounds):
            added += prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
            # re-solve: greedy repairs from the current plan, the others restart
            choice, extra = run(choice if method == "greedy" else None)
            info.update(extra)
            check = prob.verify(choice)
            rounds += 1
        info["verification_rounds"] = rounds
        info["hotspot_points_added"] = added
    uni = prob.case.uniformity
    if uni.enabled and method in ("greedy", "kfactor", "milp"):
        from .uniformity import solve_uniform

        ures = solve_uniform(prob, choice, uni)
        info.update({"uniformity_status": ures.status, "cv_2d_H": round(ures.cv_2d, 4),
                     "uniformity_weight": ures.mu, "power_H_mw": round(ures.power_h_w / 1e6, 4)})
        choice = ures.choice
        if ures.status == "below_min_load":
            choice = np.full(prob.n, -1)  # preheater cannot reach its minimum load: receiver off
        if spec.verify:
            check = prob.verify(choice)
            r2 = 0
            while not check.ok and check.hotspots and r2 < spec.verify_max_rounds:
                prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
                choice = solve_uniform(prob, solve_greedy(prob, spec, init=choice)[0], uni).choice
                check = prob.verify(choice)
                r2 += 1
    if spec.verify:
        status = _STATUS[check.status]
        peak_ratio = check.peak_ratio / max(check.allowed_ratio, 1e-12)
    else:
        status = "unverified"
        peak_ratio = _peak_ratio(prob, choice)
    return AimResult(method, prob.aims(choice), choice >= 0, prob.total_power(choice), peak_ratio,
                     _time.perf_counter() - t0, choice, info, status, check)


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
    available=None,
) -> AimResult:
    """Choose an aim point for every heliostat at one sun state (offline / one-shot use).

    For a running plant use ``engine.APSEngine``, which caches the response set
    between cycles and warm-starts from the last executed plan.
    """
    spec = spec or case.aps
    method = method or spec.method
    t0 = _time.perf_counter()
    if method == "center" and problem is None and not spec.verify:
        az, z = default_aim(case.receiver, fld.xyz)
        return AimResult("center", (az, z), np.ones(fld.n, dtype=bool), float("nan"), float("nan"),
                         _time.perf_counter() - t0)
    prob = problem or build_problem(case, fld, sun_elevation, sun_azimuth, dni, dni_constraint,
                                    grid, spec, shading_points, available)
    res = solve(prob, method, spec, prev, prev_active)
    res.solve_time_s = _time.perf_counter() - t0
    return res
