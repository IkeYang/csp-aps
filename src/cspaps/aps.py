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
    sb: np.ndarray  # (N,) shading x blocking of the centre candidate (per candidate: sb_k)
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
    sb_k: np.ndarray | None = None  # (N, K) shading x blocking of every candidate aim (its own mirror normal / beam)

    tower_dark: np.ndarray | None = None  # (N, K) all aperture samples in tower shadow for this aim
    non_tower_occluded: np.ndarray | None = None  # (N, K, P) cached other-mirror blockage masks

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
    store_occlusion_samples: bool = False,
    blockers: np.ndarray | None = None,
) -> ResponseSet:
    """Geometry, shading, intercept and sparse unit-DNI flux responses of all candidates.

    ``blockers`` (bool, N) limits the mirrors that shade / block others (candidate positions priced "as if
    added" to a layout: every mirror sees the layout's mirrors only)."""
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
        # each candidate has its own mirror normal and reflected beam, so shading and blocking are evaluated
        # per candidate (they differ by up to tens of percent at low sun near the field edge)
        n_pts = int(shading_points or case.shading.n_points)
        sb_k = np.empty((n, k))
        tower_dark = np.zeros((n, k), dtype=bool)
        # the cached sample masks must match the sampling used here (n_pts x n_pts points per mirror)
        non_tower_occluded = np.empty((n, k, n_pts ** 2), dtype=bool) if store_occlusion_samples else None
        for j in range(k):
            shadow = shading.shading_blocking(fld.xyz, normals[:, j], s, t[:, j], hel, rec, fld.pairs,
                                              n_points=n_pts, tower_shadow=case.shading.tower_shadow,
                                              blockers=blockers)
            sb_k[:, j] = shadow["combined"]
            tower_dark[:, j] = shadow["tower_fraction"] >= 1.0
            if non_tower_occluded is not None:
                non_tower_occluded[:, j] = shadow["non_tower_occluded"]
    else:
        sb_k = np.ones((n, k))
        tower_dark = None
        non_tower_occluded = None
    sb = sb_k[:, center]
    sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
    unit_beam = hel.area * hel.reflectivity * hel.cleanliness * sb_k * cos_inc * optics.attenuation(slant, case.atmosphere)
    unit_power = unit_beam * optics.intercept_fraction(rec, aim_xyz, t, sigma)
    if spec.auto_grid and n:
        grid = solver_grid(rec, grid, float(sigma.min()), spec.max_solver_cells)

    cells: list[np.ndarray] = []
    resp: list[np.ndarray] = []
    peaks = np.zeros(n)
    from .backend import enabled, sparse_responses_gpu

    gpu = (sparse_responses_gpu(grid.points, grid.normals, aim_xyz, t, sigma, unit_beam, FOOTPRINT_CUTOFF)
           if enabled() and n else None)
    if gpu is not None:  # formed, thresholded and compacted on the GPU: only the support values come back
        peaks, cells, resp = gpu
    for start in range(0, n if gpu is None else 0, chunk):
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
        grid.limit.copy(), grid.alpha.copy(), cells, resp, peaks, _time.perf_counter() - t0, sb_k=sb_k, tower_dark=tower_dark,
        non_tower_occluded=non_tower_occluded,
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
    # how often a flux limit excluded a candidate aim (greedy and uniformity moves). Zero together with a
    # first-round fine-grid pass means the limits never shaped the plan: for the same direction and a DNI
    # scaled by any factor <= 1 the solver takes the same decisions (``layout_value`` reuses such plans).
    flux_limited: int = 0

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
        """(N, K) intercepted incident power on the receiver, W."""
        return self.rs.unit_power * (self.dni_obj * self.available)[:, None]

    @property
    def absorbed_power(self) -> np.ndarray:
        """Candidate absorbed power integrated over receiver cells, W."""
        scale = (self.dni_obj * self.available)[:, None]
        alpha = self.rs.grid.alpha
        if alpha.size and np.allclose(alpha, alpha[0]):
            return self.power * float(alpha[0])
        result = np.zeros((self.n, self.k), dtype=float)
        n_grid = self.rs.n_grid
        area = self.rs.grid.cell_area
        for i, idx in enumerate(self.rs.cells):
            on_grid = idx < n_grid
            if not on_grid.any():
                continue
            grid_idx = idx[on_grid]
            flux = self.rs.resp[i][:, on_grid] * scale[i]
            result[i] = area * (flux * alpha[grid_idx][None, :]).sum(axis=1)
        return result

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

    def tracking_reference(self, prev=None, prev_active=None):
        """Target to retain on a thermal tie. With no history use the nominal centre target."""
        if prev is None:
            return np.full(self.n, self.center_k)
        delta = (self.cand_az - np.asarray(prev[0])[:, None] + 180.) % 360. - 180.
        distance = np.hypot(np.radians(delta) * self.case.receiver.radius, self.cand_z - np.asarray(prev[1])[:, None])
        ref = distance.argmin(axis=1)
        return np.where(prev_active, ref, -1) if prev_active is not None else ref

    def keep_shadowed_targets(self, choice, reference):
        """Avoid a switch or aim move if both choices deliver zero light due to the tower."""
        dark = self.rs.tower_dark
        if dark is None or not dark.any():
            return choice
        row = np.arange(self.n)
        # An inactive mirror uses its centre orientation in the joint optical model.
        old_k = np.where(choice >= 0, choice, self.center_k)
        new_k = np.where(reference >= 0, reference, self.center_k)
        keep = self.available & (choice != reference) & dark[row, old_k] & dark[row, new_k]
        if self.reach is not None:
            keep &= (reference < 0) | self.reach[row, new_k]
        keep &= (reference >= 0) | self.case.aps.allow_defocus
        return np.where(keep, reference, choice)

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
    power_w: float  # intercepted incident power on the receiver, W
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


SPREAD_K_SIGMA = 2.0  # images kept this many vertical standard deviations inside the receiver edge
TIE_MOVE_STEP_M = 0.25  # aim shifts compared in steps of this size when ties prefer the smaller move


def spread_levels(prob: AimProblem, k_sigma: float = SPREAD_K_SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """On-axis candidate columns ``ks`` (L,), their heights (L,), and the admissible mask (N, L): the heights
    at which a mirror's image (``k_sigma`` vertical standard deviations on the surface, i.e. stretched by the
    beam's incidence, ``optics.surface_spot``) stays on the receiver. The centre height is always admissible."""
    from .optics import surface_spot

    rec = prob.case.receiver
    zc, half = rec.center[2], 0.5 * rec.height
    offs = (prob.cand_az[0] - prob.cand_az[0, prob.center_k] + 180.0) % 360.0 - 180.0
    ks = np.flatnonzero(np.abs(offs) < 1e-9)
    ks = ks[np.argsort(prob.cand_z[0, ks], kind="stable")]
    levels = prob.cand_z[0, ks]
    c = prob.center_k
    spot = surface_spot(rec, prob.cand_az[:, c], prob.rs.t[:, c], prob.sigma[:, c], 1.0)
    sig_v = np.sqrt(spot["cov"][:, 1, 1])
    room = np.maximum(0.0, half - k_sigma * sig_v)
    adm = np.abs(levels[None, :] - zc) <= room[:, None] + 1e-9
    adm[:, int(np.flatnonzero(ks == c)[0])] = True
    return ks, levels, adm


def solve_spread(prob: AimProblem, k_sigma: float = SPREAD_K_SIGMA, sector_deg: float | None = None) -> np.ndarray:
    """Balanced initial aiming ("spread"): every mirror aims on its own facing azimuth, and in each azimuth
    sector of the receiver the mirrors' power is shared out over the candidate heights in proportion to the
    allowed flux at that height (``limit_nom``, so a lower limit near an edge receives less). A mirror may
    only use heights at which its image stays on the receiver (``spread_levels``): large far-field images
    aim near mid-height, small near-field images fill the upper and lower bands.

    Assignment, per sector: the most constrained mirrors first (fewest admissible heights, then the larger
    power), each to the admissible height with the lowest load per capacity (ties: closer to mid-height).
    Deterministic, no flux model, no optimisation: a reasonable starting state that uses the whole receiver
    height, not a power-optimal plan (the solver starts from it and prices moves away from it)."""
    n = prob.n
    if n == 0:
        return np.zeros(0, dtype=int)
    rec = prob.case.receiver
    ks, levels, adm = spread_levels(prob, k_sigma)
    zc = rec.center[2]
    # capacity of each height: the allowed flux there (mean over the receiver circumference)
    g = prob.rs.grid
    lim = g.to_map(g.limit)  # (n_z, n_az)
    cap = np.interp(levels, g.z, lim.mean(axis=1))
    cap = np.maximum(cap / max(cap.max(), 1e-12), 1e-6)
    if sector_deg is None:
        sector_deg = 360.0 / max(2 * rec.n_panels, 8)
    facing = prob.cand_az[:, prob.center_k] % 360.0
    sector = np.floor(facing / sector_deg).astype(int)
    weight = prob.rs.unit_power[:, prob.center_k] * np.maximum(prob.dni_obj, 0.0)
    weight = np.where(weight > 0, weight, prob.rs.unit_beam[:, prob.center_k])  # dark mirrors still spread
    n_adm = adm.sum(axis=1)
    dist_mid = np.abs(levels - zc)
    choice = np.full(n, prob.center_k)
    for s in np.unique(sector):
        members = np.flatnonzero(sector == s)
        members = members[np.lexsort((-weight[members], n_adm[members]))]
        load = np.zeros(levels.size)
        for i in members:
            score = np.where(adm[i], load / cap, np.inf)
            best = score.min()
            pick = np.flatnonzero(score <= best + 1e-12 * max(best, 1.0))
            j = int(pick[np.argmin(dist_mid[pick])])
            choice[i] = ks[j]
            load[j] += weight[i]
    return np.where(prob.available, choice, -1)


def initial_plan(prob: AimProblem, mode: str = "center") -> np.ndarray | None:
    """The initial aim plan of an offline solve without a working state: "spread" (``solve_spread``),
    "kfactor" (``solve_kfactor``), "center", or "none" (no initial plan: cold start, no move pricing)."""
    if mode == "none":
        return None
    if mode == "spread":
        return solve_spread(prob)
    if mode == "kfactor":
        return solve_kfactor(prob)[0]
    if mode == "center":
        return solve_center(prob)
    raise ValueError(f"Unknown initial aim '{mode}' (center, spread, kfactor, none)")


def move_cost(prob: AimProblem, prev_xyz: np.ndarray | None, prev_active=None, spec: APSSpec | None = None):
    """(N, K) price in W of moving each mirror from its initial aim to each candidate, as in the greedy score:
    ``move_penalty_per_m`` x the mirror's centre-aim absorbed power x the aim shift (m). Mirrors inactive in
    the initial state move for free. None without an initial state or with a zero price."""
    spec = spec or prob.case.aps
    if prev_xyz is None or spec.move_penalty_per_m <= 0:
        return None
    p_ref = prob.absorbed_power[:, prob.center_k]
    move = np.linalg.norm(prob.aim_xyz - prev_xyz[:, None, :], axis=-1)
    if prev_active is not None:
        move = move * np.asarray(prev_active)[:, None]
    return spec.move_penalty_per_m * p_ref[:, None] * move


def plan_move_cost(cost: np.ndarray | None, choice: np.ndarray) -> float:
    """Move price of a plan (W); defocusing is free."""
    if cost is None:
        return 0.0
    on = np.flatnonzero(np.asarray(choice) >= 0)
    return float(cost[on, np.asarray(choice)[on]].sum())


def _normals_towards(xyz, aim_xyz, sun_elevation: float, sun_azimuth: float) -> np.ndarray:
    normals, *_ = optics.heliostat_geometry(xyz, aim_xyz, sun.sun_vector(sun_elevation, sun_azimuth))
    return normals


def move_limits(prob: AimProblem, xyz: np.ndarray, prev, prev_active=None, spec: APSSpec | None = None):
    """(N, K) bool: candidates within ``max_aim_shift_m`` (aim shift on the receiver) and ``max_rotation_deg``
    (mirror-normal rotation at this sun) of the initial state; None without an initial state or limits.
    The candidate nearest to the initial aim always stays allowed (the initial aim need not be a candidate),
    and mirrors inactive in the initial state are not limited (their pose is unknown)."""
    spec = spec or prob.case.aps
    if prev is None or (spec.max_aim_shift_m <= 0 and spec.max_rotation_deg <= 0):
        return None
    rec = prob.case.receiver
    prev_xyz = surface_point(rec, prev[0], prev[1])
    shift = np.linalg.norm(prob.aim_xyz - prev_xyz[:, None, :], axis=-1)
    ok = np.ones((prob.n, prob.k), dtype=bool)
    if spec.max_aim_shift_m > 0:
        ok &= shift <= spec.max_aim_shift_m + 1e-9
    if spec.max_rotation_deg > 0:
        el, az = prob.rs.sun_elevation, prob.rs.sun_azimuth
        n_prev = _normals_towards(xyz, prev_xyz, el, az)
        n_cand = _normals_towards(np.asarray(xyz)[:, None, :], prob.aim_xyz, el, az)
        ang = np.degrees(np.arccos(np.clip(np.einsum("nkc,nc->nk", n_cand, n_prev), -1.0, 1.0)))
        ok &= ang <= spec.max_rotation_deg + 1e-9
    if prev_active is not None:
        ok |= ~np.asarray(prev_active, dtype=bool)[:, None]
    ok[np.arange(prob.n), shift.argmin(axis=1)] = True
    return ok


def move_stats(case: Case, xyz: np.ndarray, sun_elevation: float, sun_azimuth: float, prev, prev_active,
               aim, active, moved_tol_m: float = 0.5) -> tuple[dict, dict]:
    """How far every mirror moves from its initial state to a plan: (per-mirror arrays, summary).

    ``shift_m`` = distance between the initial and the new aim point on the receiver; ``rotation_deg`` = angle
    between the mirror normals that track the two aims at this sun (what the drives must turn). A mirror counts
    as moved above ``moved_tol_m``; switched on / off compares the active flags."""
    rec = case.receiver
    xyz = np.asarray(xyz, dtype=float)
    p_xyz = surface_point(rec, np.asarray(prev[0], float), np.asarray(prev[1], float))
    a_xyz = surface_point(rec, np.asarray(aim[0], float), np.asarray(aim[1], float))
    shift = np.linalg.norm(a_xyz - p_xyz, axis=1)
    n0 = _normals_towards(xyz, p_xyz, sun_elevation, sun_azimuth)
    n1 = _normals_towards(xyz, a_xyz, sun_elevation, sun_azimuth)
    rot = np.degrees(np.arccos(np.clip(np.einsum("nc,nc->n", n0, n1), -1.0, 1.0)))
    was = np.ones(len(shift), bool) if prev_active is None else np.asarray(prev_active, dtype=bool)
    now = np.asarray(active, dtype=bool)
    both = was & now
    moved = both & (shift > moved_tol_m)
    spec = case.aps
    within = np.ones(len(shift), bool)
    if spec.max_aim_shift_m > 0:
        within &= ~both | (shift <= spec.max_aim_shift_m + 1e-6)
    if spec.max_rotation_deg > 0:
        within &= ~both | (rot <= spec.max_rotation_deg + 1e-6)

    def q(v, f):
        return round(float(f(v)), 4) if v.size else 0.0

    s_both, r_both = shift[both], rot[both]
    summary = {
        "n_mirrors": int(len(shift)), "n_moved": int(moved.sum()), "moved_tol_m": moved_tol_m,
        "n_switched_on": int((~was & now).sum()), "n_switched_off": int((was & ~now).sum()),
        "mean_shift_m": q(s_both, np.mean), "p95_shift_m": q(s_both, lambda v: np.percentile(v, 95)),
        "max_shift_m": q(s_both, np.max), "mean_rotation_deg": q(r_both, np.mean),
        "p95_rotation_deg": q(r_both, lambda v: np.percentile(v, 95)), "max_rotation_deg": q(r_both, np.max),
        "move_penalty_per_m": spec.move_penalty_per_m, "max_aim_shift_m": spec.max_aim_shift_m,
        "max_rotation_deg_limit": spec.max_rotation_deg, "within_limits": bool(within.all()),
        "n_outside_limits": int((~within).sum()),
    }
    if case.engine.max_slew_deg_s > 0:
        summary["max_slew_time_s"] = round(float(r_both.max() / case.engine.max_slew_deg_s), 2) if r_both.size else 0.0
    return {"shift_m": shift, "rotation_deg": rot, "moved": moved, "within_limits": within}, summary


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
    objective_power = prob.absorbed_power
    p_ref = objective_power[:, prob.center_k]
    if prev_aim_xyz is not None and spec.move_penalty_per_m > 0:
        move = np.linalg.norm(prob.aim_xyz - prev_aim_xyz[:, None, :], axis=-1)  # (N, K)
        if prev_active is not None:
            move = move * np.asarray(prev_active)[:, None]
        score_all = objective_power - spec.move_penalty_per_m * p_ref[:, None] * move
    else:
        score_all = objective_power
    tol = spec.tie_tolerance
    # with an initial state, near-equal candidates (within the tie tolerance) prefer the smaller move first
    tie_first = None
    if prev_aim_xyz is not None:
        stay = np.linalg.norm(prob.aim_xyz - prev_aim_xyz[:, None, :], axis=-1)
        if prev_active is not None:
            stay = stay * np.asarray(prev_active)[:, None]
        tie_first = np.round(stay / TIE_MOVE_STEP_M)
    avail = prob.available
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")  # image-size priority
    stats = {"moves": 0, "flux_limited": 0}

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
        if not feasible.all():
            stats["flux_limited"] += 1
        if reach is not None:
            feasible &= reach[i]
        score = score_all[i]
        if feasible.any():
            best = score[feasible].max()
            ties = np.flatnonzero(feasible & (score >= best - tol * max(p_ref[i], 1e-9)))
            if tie_first is not None:
                rank = tie_first[i, ties]
                ties = ties[rank == rank.min()]
            return int(ties[np.argmin(ratio[ties])])
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

    from . import kernels

    def reassign_all(seq) -> None:
        """``reassign`` over ``seq``: compiled when available (bit-identical), else the loop."""
        if kernels.enabled():
            kernels.greedy_pass(prob, seq, choice, flux, score_all, p_ref, tol, stats, tie_first)
        else:
            for i in seq:
                reassign(i)

    if init is None:
        choice = np.full(n, -1)
        flux = np.zeros(limit.size)
        reassign_all(order)
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
        reassign_all(order)
        passes_done += 1
        if stats["moves"] == before:
            break
    owner = getattr(prob, "_prob", prob)  # a subfield view counts on its parent problem
    owner.flux_limited = int(getattr(owner, "flux_limited", 0)) + stats["flux_limited"]
    return choice, {"passes": passes_done, "repairs": repairs, "timed_out": timed_out["v"]}


class _SubProblem:
    """View of an AimProblem restricted to some heliostats with its own limit per constraint point."""

    def __init__(self, prob: AimProblem, available: np.ndarray, limit: np.ndarray):
        self._prob, self.available, self.limit = prob, available, limit

    def __getattr__(self, name):
        return getattr(self._prob, name)


def subfield_of(prob: AimProblem, n_sub: int) -> np.ndarray:
    """Sector (0..n_sub-1) of every heliostat by the compass azimuth it faces on the receiver (centre aim)."""
    az = prob.cand_az[:, prob.center_k] % 360.0
    return np.minimum((az / (360.0 / max(1, n_sub))).astype(int), max(1, n_sub) - 1)


def solve_subfield(prob: AimProblem, spec: APSSpec, n_sub: int | None = None, prev_aim_xyz: np.ndarray | None = None,
                   prev_active: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Subfield decomposition (HALOS-style): sectors solved independently, so they can run in parallel.

    Sector k may use, at every constraint point p, the share
        s_k(p) = sum_{i in k} max_c F_i,c(p) / sum_i max_c F_i,c(p)
    of the allowed flux (what its mirrors could deliver there at most). The shares sum to 1, so the union
    of the sector plans keeps sum_k flux_k(p) <= sum_k s_k(p) L(p) = L(p) at every response point; the
    caller then verifies on the fine grid and repairs as for the other methods. ``info`` reports the
    serial time (sum over sectors) and the ideal parallel time (slowest sector)."""
    n_sub = int(n_sub or spec.subfield_sectors)
    sector = subfield_of(prob, n_sub)
    npts = prob.limit.size
    reach = np.zeros((n_sub, npts))
    for i in np.flatnonzero(prob.available):
        idx = prob.cells[i]
        if idx.size:
            reach[sector[i], idx] += prob.F(i).max(axis=0)
    total = reach.sum(axis=0)
    share = np.where(total > 0, reach / np.where(total > 0, total, 1.0), 1.0 / n_sub)
    choice = np.full(prob.n, -1)
    times, sizes = [], []
    for k in range(n_sub):
        members = prob.available & (sector == k)
        sizes.append(int(members.sum()))
        if not members.any():
            times.append(0.0)
            continue
        t0 = _time.perf_counter()
        sub = _SubProblem(prob, members, prob.limit * share[k])
        ch, _ = solve_greedy(sub, spec, prev_aim_xyz, prev_active)
        choice[members] = ch[members]
        times.append(_time.perf_counter() - t0)
    return choice, {"subfield_sectors": n_sub, "subfield_sizes": sizes, "subfield_serial_s": float(sum(times)),
                    "subfield_parallel_s": float(max(times) if times else 0.0)}


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


def solve_milp(prob: AimProblem, spec: APSSpec, relax: bool = False, max_nnz: int | None = None, cost=None):
    """Exact 0/1 programme (or its LP relaxation) with HiGHS. Returns (choice, info).

    ``cost`` (N, K) W, from ``move_cost``: maximise absorbed power minus the move price."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    max_nnz = int(getattr(spec, "milp_max_nnz", 4_000_000) if max_nnz is None else max_nnz)
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
    gain = prob.absorbed_power if cost is None else prob.absorbed_power - cost
    res = milp(
        c=-gain.ravel(),
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
    """Upper bound (W) on absorbed power for any aim assignment under the flux limit
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
            return solve_milp(prob, spec, cost=move_cost(prob, prev_xyz, prev_active, spec))
        if method == "subfield":
            return solve_subfield(prob, spec, prev_aim_xyz=prev_xyz, prev_active=prev_active)
        raise ValueError(f"Unknown APS method '{method}'")

    choice, extra = run(init)
    info.update(extra)
    check = None
    rounds = added = repairs = 0
    if spec.verify:
        check = prob.verify(choice)
        # an exact re-solve per round is cheap on the small problems MILP is meant for: allow more rounds
        max_rounds = max(spec.verify_max_rounds, 12) if method == "milp" and spec.verify_max_rounds > 0 else spec.verify_max_rounds
        while (method != "center" and not check.ok and check.hotspots and rounds < max_rounds):
            added += prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
            # re-solve: greedy repairs from the current plan, the others restart
            choice, extra = run(choice if method == "greedy" else None)
            info.update(extra)
            check = prob.verify(choice)
            rounds += 1
        # A restarted exact solve can move flux to new hotspots every round (cutting planes need not converge
        # in a fixed number of rounds). Finish with repair from the current plan: only heliostats feeding a
        # violated point move or defocus, so the plan reaches the limit.
        # (rule-based methods such as kfactor keep their own failure state: they are not silently replaced)
        while (method in ("milp", "greedy", "subfield") and check.status == "over_limit" and check.hotspots
               and repairs < max(4, 2 * spec.verify_max_rounds)):
            added += prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
            choice, _ = solve_greedy(prob, spec, prev_xyz, prev_active, init=choice, full_passes=0)
            check = prob.verify(choice)
            repairs += 1
        info["verification_rounds"] = rounds
        info["repair_rounds"] = repairs
        info["hotspot_points_added"] = added
    uni = prob.case.uniformity
    if uni.enabled and method in ("greedy", "kfactor", "milp", "subfield"):
        from .uniformity import solve_uniform

        ures = solve_uniform(prob, choice, uni, prev_aim_xyz=prev_xyz, prev_active=prev_active)
        choice = ures.choice
        if ures.status == "below_min_load" and spec.allow_defocus:
            choice = np.full(prob.n, -1)  # shut down only when permitted; otherwise keep mirrors aimed
        if spec.verify:
            check = prob.verify(choice)
            r2 = 0
            while not check.ok and check.hotspots and r2 < spec.verify_max_rounds:
                prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
                repair, _ = solve_greedy(prob, spec, prev_xyz, prev_active, init=choice)
                ures = solve_uniform(prob, repair, uni, prev_aim_xyz=prev_xyz, prev_active=prev_active)
                choice = ures.choice
                check = prob.verify(choice)
                r2 += 1
            info["uniformity_repair_rounds"] = r2
        info.update(uniformity_info(prob, choice, uni, ures))
    # Last word: repair from the current plan. A uniformity move (or a re-solve) can leave a fine-grid peak
    # between the allowed ratio and the limit; only heliostats feeding a violated point move or defocus,
    # so the plan keeps its power and ends verified "ok" (the strict margin counts nothing less).
    if spec.verify and check is not None and method in ("milp", "greedy", "subfield"):
        final = 0
        while not check.ok and check.hotspots and final < max(4, 2 * spec.verify_max_rounds):
            added += prob.rs.add_points([h[0] for h in check.hotspots], [h[1] for h in check.hotspots])
            choice, _ = solve_greedy(prob, spec, prev_xyz, prev_active, init=choice, full_passes=0)
            check = prob.verify(choice)
            final += 1
        info["final_repair_rounds"] = final
        info["hotspot_points_added"] = added
        if final and uni.enabled:
            info.update(uniformity_info(prob, choice, uni))
    if info.get("uniformity_status") != "below_min_load":
        retained = prob.keep_shadowed_targets(choice, prob.tracking_reference(prev, prev_active))
        if not np.array_equal(retained, choice):
            choice = retained
            check = prob.verify(choice) if spec.verify else None
    info["flux_limited"] = int(prob.flux_limited)
    info["limits_active"] = bool(prob.flux_limited or info.get("verification_rounds") or info.get("repair_rounds")
                                 or info.get("uniformity_repair_rounds") or info.get("final_repair_rounds")
                                 or info.get("hotspot_points_added") or check is None or not check.ok)
    return finalize(prob, method, choice, info, check, spec, t0)


def uniformity_info(prob: AimProblem, choice: np.ndarray, uni, ures=None) -> dict:
    """Uniformity metrics and status of a plan on the solver grid (region H, absorbed flux)."""
    from .uniformity import region_of_grid, uniformity_metrics

    h_mask = region_of_grid(prob.case, prob.grid, uni)
    absorbed = prob.flux(choice, basis="objective")[:prob.rs.n_grid] * prob.rs.grid.alpha[:prob.rs.n_grid]
    h_power = float(absorbed[h_mask].sum() * prob.rs.grid.cell_area)
    metrics = uniformity_metrics(prob.rs.grid.to_map(absorbed), prob.rs.grid.to_map(h_mask))
    if h_power < max(uni.min_load_mw * 1e6, 0. if uni.mode == "weighted" else 1e-6):
        uni_status = "below_min_load"
    elif uni.mode == "weighted":
        uni_status = "weighted"
    else:
        uni_status = "feasible" if metrics["cv_2d"] <= uni.max_cv + 1e-9 else "not_reached"

    def r4(v):
        return round(v, 4) if np.isfinite(v) else None

    out = {"uniformity_status": uni_status, "uniformity_mode": uni.mode,
           "uniformity_target_cv": uni.max_cv if uni.mode == "cv_limit" else None,
           "uniformity_weight": uni.weight if uni.mode == "weighted" else (ures.mu if ures is not None else None),
           "cv_2d_H": r4(metrics["cv_2d"]), "cv_circ_H": r4(metrics["cv_circ"]),
           "cv_height_H": r4(metrics["cv_height"]), "power_H_mw": round(h_power / 1e6, 4),
           "uniformity_met": uni_status in ("feasible", "weighted")}
    if ures is not None:
        out.update({"uniformity_search_status": ures.status, "uniformity_search_s": round(ures.solve_time_s, 4)})
    return out


def finalize(prob: AimProblem, method: str, choice: np.ndarray, info: dict, check: FluxCheck | None,
             spec: APSSpec, t0: float) -> AimResult:
    """AimResult with status from the verified check (or the coarse peak when verification is off)."""
    uni = prob.case.uniformity
    verified = check is not None and check.status in ("ok", "margin_reduced")
    # "proven optimal" needs the exact solver's own optimum (no repair after it) *and* a passed fine-grid check
    proven = bool(method == "milp" and info.get("milp_status") == 0 and not uni.enabled
                  and not info.get("repair_rounds") and (verified or not spec.verify))
    info["global_optimality_proven"] = proven
    if spec.verify and check is not None and not verified:
        info["solution_class"] = "infeasible_after_verification"
        info["global_optimality_proven"] = False
    elif "solution_class" not in info or info["solution_class"] == "global_optimum_on_candidate_set":
        info["solution_class"] = ("global_optimum_on_candidate_set" if info["global_optimality_proven"]
                                  else "verified_feasible_heuristic")
    if spec.verify and check is not None:
        status = _STATUS[check.status]
        peak_ratio = check.peak_ratio / max(check.allowed_ratio, 1e-12)
    else:
        status = "unverified"
        peak_ratio = _peak_ratio(prob, choice)
    active_choice = choice >= 0
    info["objective_absorbed_power_w"] = float(
        prob.absorbed_power[np.flatnonzero(active_choice), choice[active_choice]].sum())
    return AimResult(method, prob.aims(choice), active_choice, prob.total_power(choice), peak_ratio,
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
    joint_trials: int | None = None,
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
    if method in ("greedy", "milp") and spec.verify and prob.n:
        from .coupled import search_joint_poses

        res, cert = search_joint_poses(prob, fld, res, trials=joint_trials,
                                      shading_points=shading_points, tighten_bound=joint_trials is None,
                                      tracking_reference=prob.tracking_reference(prev, prev_active))
        res.info["certificate"] = cert
    res.solve_time_s = _time.perf_counter() - t0
    return res
