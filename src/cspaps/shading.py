"""Shading and blocking between heliostats, plus the tower shadow.

Each mirror is sampled with an n x n grid of points. A point is *shaded* if
the ray from it towards the sun hits a neighbouring mirror (or the tower),
and *blocked* if the reflected ray towards the aim point hits a neighbouring
mirror. The lost fraction is the union of both, so overlaps are not counted
twice. Mirrors are rectangles with azimuth-elevation tracking: the width axis
stays horizontal.

Neighbour candidates come from a static radius search (cached on the field)
and are filtered per instant to those lying in a narrow corridor along the
ray direction, so the ray tests stay cheap.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import Heliostat, Receiver


def mirror_frames(normals: np.ndarray, xp=np) -> tuple[np.ndarray, np.ndarray]:
    """In-plane unit axes (width: horizontal, height) of each mirror (``xp``: numpy or cupy)."""
    z = xp.asarray([0.0, 0.0, 1.0])
    e_w = xp.cross(z, normals)
    norm = xp.linalg.norm(e_w, axis=-1, keepdims=True)
    e_w = xp.where(norm > 1e-9, e_w / xp.maximum(norm, 1e-12), xp.asarray([1.0, 0.0, 0.0]))
    e_h = xp.cross(normals, e_w)
    return e_w, e_h


def grid_offsets(n: int) -> np.ndarray:
    """(n*n, 2) cell-midpoint offsets in [-0.5, 0.5] (fractions of width, height)."""
    f = (np.arange(n) + 0.5) / n - 0.5
    a, b = np.meshgrid(f, f)
    return np.column_stack([a.ravel(), b.ravel()])


def mirror_points(centers, e_w, e_h, helio: Heliostat, offsets) -> np.ndarray:
    """Sample points on each mirror, (N, P, 3)."""
    return (
        centers[:, None, :]
        + offsets[None, :, 0:1] * helio.width * e_w[:, None, :]
        + offsets[None, :, 1:2] * helio.height * e_h[:, None, :]
    )


def ray_hits_rect(o, d, c, n, e_w, e_h, half_w, half_h, xp=np) -> np.ndarray:
    """Does the ray o + s d (s > 0) hit the rectangle (centre c, normal n, axes e_w, e_h)?

    All vector arguments broadcast against each other with a trailing axis of 3 (``xp``: numpy or cupy).
    """
    denom = xp.sum(d * n, axis=-1)
    safe = xp.where(xp.abs(denom) > 1e-12, denom, 1.0)
    s = xp.sum((c - o) * n, axis=-1) / safe
    p = o + s[..., None] * d - c
    a = xp.sum(p * e_w, axis=-1)
    b = xp.sum(p * e_h, axis=-1)
    return (xp.abs(denom) > 1e-12) & (s > 1e-6) & (xp.abs(a) <= half_w) & (xp.abs(b) <= half_h)


def shadow_cylinders(rec) -> list[tuple[float, float, float]]:
    """The tower and the receiver as vertical cylinders (radius, z_bottom, z_top), m.

    Tower: ``receiver.tower_radius`` from the ground to ``receiver.tower_top`` (the tower height variable;
    by default the bottom of the receiver). Receiver: its own radius from its bottom to its top. A tower
    diameter of 0 switches the tower shadow off."""
    if rec is None or rec.tower_radius <= 0:
        return []
    out = []
    if rec.tower_top > 0:
        out.append((float(rec.tower_radius), 0.0, float(rec.tower_top)))
    if rec.radius > 0 and rec.top > rec.bottom:
        out.append((float(rec.radius), float(rec.bottom), float(rec.top)))
    return out


def ray_hits_structure(o, d, rec, xp=np) -> np.ndarray:
    """Does the ray o + s d (s > 0) hit the tower or the receiver on it (``shadow_cylinders``)?"""
    o = xp.asarray(o, dtype=float)
    hit = xp.zeros(np.broadcast_shapes(o.shape, np.shape(d))[:-1], dtype=bool)
    for radius, z0, z1 in shadow_cylinders(rec):
        hit |= ray_hits_tower(o, d, rec.center[:2], radius, z1, z0, xp=xp)
    return hit


def ray_hits_tower(o, d, center_xy, radius: float, z_top: float, z_bottom: float = 0.0, xp=np) -> np.ndarray:
    """Does the ray o + s d (s > 0) hit a vertical cylinder from z_bottom to z_top?"""
    ox = o[..., 0] - center_xy[0]
    oy = o[..., 1] - center_xy[1]
    dx, dy = d[..., 0], d[..., 1]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    sq = xp.sqrt(xp.where(ok, disc, 0.0))
    a_safe = xp.where(a > 1e-12, a, 1.0)
    s1 = (-b - sq) / (2 * a_safe)
    s2 = (-b + sq) / (2 * a_safe)
    hit = xp.zeros(ok.shape, dtype=bool)
    for s in (s1, s2):
        z = o[..., 2] + s * d[..., 2]
        hit |= ok & (s > 1e-6) & (z >= z_bottom) & (z <= z_top)
    # End caps also cover vertical rays and origins directly below or inside the cylinder.
    dz = d[..., 2]
    for plane in (z_bottom, z_top):
        s = (plane - o[..., 2]) / xp.where(xp.abs(dz) > 1e-12, dz, 1.)
        hit |= (xp.abs(dz) > 1e-12) & (s > 1e-6) & ((ox + s*dx)**2 + (oy + s*dy)**2 <= radius**2)
    return hit


def neighbor_pairs(xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs (i, j), i != j, with horizontal distance <= radius (int32 indices below 2^31 mirrors:
    a low sun over a large field gives hundreds of millions of pairs)."""
    itype = np.int32 if len(xy) < 2 ** 31 else np.int64
    if len(xy) < 2:
        return np.zeros(0, dtype=itype), np.zeros(0, dtype=itype)
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray").astype(itype, copy=False)
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    del pairs
    order = np.argsort(i, kind="stable")
    return i[order], j[order]


def interaction_radius(helio: Heliostat, sun_elevation_deg: float, dz: float = 0.0) -> float:
    """Max horizontal distance at which one mirror can shade another."""
    el = np.radians(max(sun_elevation_deg, 2.0))
    return (helio.height + dz) / np.tan(el) + helio.diagonal


class PairCache:
    """Static neighbour pairs of a field, recomputed only when a larger radius is needed."""

    def __init__(self, xyz: np.ndarray):
        self.xyz = xyz
        self.radius = -1.0
        self.i = self.j = np.zeros(0, dtype=np.int32)
        self.d = np.zeros(0)

    def get(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        if radius > self.radius:
            self.radius = radius * 1.25  # head-room so small increases do not recompute
            self.i = self.j = self.d = None  # the old pairs go before the new ones are built
            self.i, self.j = neighbor_pairs(self.xyz[:, :2], self.radius)
            self.d = np.empty(self.i.size)
            for s in range(0, self.i.size, PAIR_BLOCK):  # same values as one hypot, without the (P, 2) temporary
                e = s + PAIR_BLOCK
                self.d[s:e] = np.hypot(*(self.xyz[self.j[s:e], :2] - self.xyz[self.i[s:e], :2]).T)
        keep = self.d <= radius
        return self.i[keep], self.j[keep]


PAIR_BLOCK = 1_000_000  # neighbour pairs per step of the CPU corridor test (a few (block, 3) float64 arrays)


def _corridor(rel: np.ndarray, d: np.ndarray, diag: float, max_along: np.ndarray, xp=np) -> np.ndarray:
    along = xp.sum(rel * d, axis=-1)
    lateral = xp.linalg.norm(rel - along[:, None] * d, axis=-1)
    return (along > -diag) & (along < max_along) & (lateral < diag)


def _shading_blocking_gpu(cp, scatter_add, xyz, normals, sun_vec, t, helio, rec, i_all, j_all, r_shade, dz,
                          n_points, tower_shadow) -> dict[str, np.ndarray]:
    """``shading_blocking`` after the (CPU) neighbour-pair selection, on the device: same geometry and rules;
    the hit masks are OR-ed per mirror with a scatter-add of the per-pair hits. The pairs stream through the
    device in blocks (a low sun over a large field gives hundreds of millions of them), so the device memory
    stays within ``block_elements`` whatever their number; the OR makes the result independent of the blocks."""
    from .backend import block_elements

    n = xyz.shape[0]
    X, Nrm, T, S = cp.asarray(xyz), cp.asarray(normals), cp.asarray(t), cp.asarray(sun_vec)
    e_w, e_h = mirror_frames(Nrm, xp=cp)
    offsets = grid_offsets(n_points)
    off = cp.asarray(offsets)
    pts = mirror_points(X, e_w, e_h, helio, off)
    diag = helio.diagonal
    hw, hh = 0.5 * helio.width, 0.5 * helio.height
    p = offsets.shape[0]
    block = block_elements(cp)
    chunk = max(1, block // (12 * p))  # pairs per ray test: (chunk, P, 3) float64 arrays
    pair_block = max(chunk, block // 8)  # pairs per corridor step: a few (pair_block, 3) float64 arrays

    def add_hits(counts, I, J, d):
        for start in range(0, int(I.size), chunk):
            ii, jj = I[start:start + chunk], J[start:start + chunk]
            dd = d[ii][:, None, :] if d.ndim == 2 else d[None, None, :]
            hits = ray_hits_rect(pts[ii], dd, X[jj][:, None, :], Nrm[jj][:, None, :], e_w[jj][:, None, :],
                                 e_h[jj][:, None, :], hw, hh, xp=cp)
            scatter_add(counts, ii, hits.astype(cp.int32))

    shade_counts = cp.zeros((n, p), dtype=cp.int32)
    block_counts = cp.zeros((n, p), dtype=cp.int32)
    for start in range(0, int(np.asarray(i_all).size), pair_block):
        I = cp.asarray(np.asarray(i_all[start:start + pair_block], dtype=np.int64))
        J = cp.asarray(np.asarray(j_all[start:start + pair_block], dtype=np.int64))
        rel = X[J] - X[I]
        keep_s = _corridor(rel, cp.broadcast_to(S, rel.shape), diag, cp.full(rel.shape[0], r_shade + diag), xp=cp)
        add_hits(shade_counts, I[keep_s], J[keep_s], S)
        t_pair = T[I]
        max_along = (helio.height + dz) / cp.maximum(t_pair[:, 2], 0.03) + diag
        keep_b = _corridor(rel, t_pair, diag, max_along, xp=cp)
        add_hits(block_counts, I[keep_b], J[keep_b], T)
        del I, J, rel, t_pair, max_along, keep_s, keep_b
    shaded = shade_counts > 0
    blocked = block_counts > 0
    tower_hits = cp.zeros(shaded.shape, dtype=bool)
    tower_fraction = cp.zeros(n)
    if tower_shadow and rec is not None and rec.tower_radius > 0:
        tower_hits = ray_hits_structure(pts, S[None, None, :], rec, xp=cp)
        tower_fraction = tower_hits.mean(axis=1)
    other = shaded | blocked
    eta_shade = 1.0 - (shaded | tower_hits).sum(axis=1) / p
    eta_comb = 1.0 - (other | tower_hits).sum(axis=1) / p
    eta_block = cp.where(eta_shade > 0, eta_comb / cp.where(eta_shade > 0, eta_shade, 1.0), 1.0)
    return {"shading": cp.asnumpy(eta_shade), "blocking": cp.asnumpy(eta_block), "combined": cp.asnumpy(eta_comb),
            "tower_fraction": cp.asnumpy(tower_fraction), "non_tower_occluded": cp.asnumpy(other)}


def _hit_mask(points, i, j, d, xyz, normals, e_w, e_h, helio, n, chunk=4096, out=None) -> np.ndarray:
    """(N, P) mask: point of mirror i hits any listed neighbour j along direction d (OR-ed into ``out``)."""
    mask = np.zeros(points.shape[:2], dtype=bool) if out is None else out
    hw, hh = 0.5 * helio.width, 0.5 * helio.height
    for start in range(0, i.size, chunk):
        ii, jj = i[start:start + chunk], j[start:start + chunk]
        dd = d[ii][:, None, :] if d.ndim == 2 else d[None, None, :]
        hits = ray_hits_rect(points[ii], dd, xyz[jj][:, None, :], normals[jj][:, None, :],
                             e_w[jj][:, None, :], e_h[jj][:, None, :], hw, hh)
        np.logical_or.at(mask, ii, hits)
    return mask


def shading_blocking(
    xyz: np.ndarray,
    normals: np.ndarray,
    sun_vec: np.ndarray,
    t: np.ndarray,
    helio: Heliostat,
    rec: Receiver | None,
    pairs: PairCache,
    n_points: int = 8,
    tower_shadow: bool = True,
    blockers: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-heliostat shading, blocking and combined efficiencies.

    ``blockers`` (bool, N) limits the mirrors that can shade or block others,
    e.g. to the heliostats of a layout while the others are candidates being
    priced "as if added": every mirror is then evaluated against the blockers only.

    Returns dict with 'shading', 'blocking', 'combined' efficiencies (N,) where
    combined = 1 - |shaded U blocked| / P and blocking is reported relative to
    the unshaded part (so shading * blocking == combined).
    """
    n = xyz.shape[0]
    ones = np.ones(n)
    if n == 0:
        return {"shading": ones, "blocking": ones, "combined": ones, "tower_fraction": np.zeros(n)}
    from .backend import enabled

    if enabled():  # the ray tests on the GPU (pairs selected on the CPU, as below)
        gpu = _shading_gpu(xyz, normals, sun_vec, t, helio, rec, pairs, n_points, tower_shadow, blockers)
        if gpu is not None:
            return gpu
    e_w, e_h = mirror_frames(normals)
    offsets = grid_offsets(n_points)
    pts = mirror_points(xyz, e_w, e_h, helio, offsets)
    diag = helio.diagonal
    dz = float(np.ptp(xyz[:, 2])) if n > 1 else 0.0

    sun_el = np.degrees(np.arcsin(np.clip(sun_vec[2], -1, 1)))
    r_shade = interaction_radius(helio, sun_el, dz)
    t_el = np.degrees(np.arcsin(np.clip(t[:, 2], -1, 1)))
    r_block = interaction_radius(helio, float(t_el.min()), dz)
    i_all, j_all = pairs.get(max(r_shade, r_block))
    if blockers is not None:
        keep = np.asarray(blockers, dtype=bool)[j_all]
        i_all, j_all = i_all[keep], j_all[keep]
    # the pairs in blocks: memory bounded whatever their number (hundreds of millions at a low sun)
    shaded_mirrors = np.zeros(pts.shape[:2], dtype=bool)
    blocked = np.zeros(pts.shape[:2], dtype=bool)
    for start in range(0, i_all.size, PAIR_BLOCK):
        i_b, j_b = i_all[start:start + PAIR_BLOCK], j_all[start:start + PAIR_BLOCK]
        rel = xyz[j_b] - xyz[i_b]
        keep_s = _corridor(rel, np.broadcast_to(sun_vec, rel.shape), diag, np.full(rel.shape[0], r_shade + diag))
        _hit_mask(pts, i_b[keep_s], j_b[keep_s], sun_vec, xyz, normals, e_w, e_h, helio, n, out=shaded_mirrors)
        t_pair = t[i_b]
        max_along = (helio.height + dz) / np.maximum(t_pair[:, 2], 0.03) + diag
        keep_b = _corridor(rel, t_pair, diag, max_along)
        _hit_mask(pts, i_b[keep_b], j_b[keep_b], t, xyz, normals, e_w, e_h, helio, n, out=blocked)
    tower_hits = np.zeros(shaded_mirrors.shape, dtype=bool)
    tower_fraction = np.zeros(n)
    if tower_shadow and rec is not None and rec.tower_radius > 0:
        tower_hits = ray_hits_structure(pts, sun_vec[None, None, :], rec)
        tower_fraction = tower_hits.mean(axis=1)

    p = offsets.shape[0]
    other_occlusion = shaded_mirrors | blocked
    eta_shade = 1.0 - (shaded_mirrors | tower_hits).sum(axis=1) / p
    eta_comb = 1.0 - (other_occlusion | tower_hits).sum(axis=1) / p
    with np.errstate(invalid="ignore", divide="ignore"):
        eta_block = np.where(eta_shade > 0, eta_comb / eta_shade, 1.0)
    return {"shading": eta_shade, "blocking": eta_block, "combined": eta_comb,
            "tower_fraction": tower_fraction, "non_tower_occluded": other_occlusion}


def _shading_gpu(xyz, normals, sun_vec, t, helio, rec, pairs, n_points, tower_shadow, blockers):
    """``shading_blocking`` on the GPU when the backend is on and the ray tests are many; else None (also when
    the device runs out of memory: ``backend.falls_back_on_oom``)."""
    from . import backend

    return backend.falls_back_on_oom(_shading_gpu_run)(xyz, normals, sun_vec, t, helio, rec, pairs, n_points,
                                                         tower_shadow, blockers)


def _shading_gpu_run(xyz, normals, sun_vec, t, helio, rec, pairs, n_points, tower_shadow, blockers):
    from . import backend

    dz = float(np.ptp(xyz[:, 2])) if xyz.shape[0] > 1 else 0.0
    sun_el = np.degrees(np.arcsin(np.clip(sun_vec[2], -1, 1)))
    r_shade = interaction_radius(helio, sun_el, dz)
    t_el = np.degrees(np.arcsin(np.clip(t[:, 2], -1, 1)))
    r_block = interaction_radius(helio, float(t_el.min()), dz)
    i_all, j_all = pairs.get(max(r_shade, r_block))
    if blockers is not None:
        keep = np.asarray(blockers, dtype=bool)[j_all]
        i_all, j_all = i_all[keep], j_all[keep]
    cp = backend._gpu_for(int(i_all.size) * n_points * n_points)
    if cp is None:
        return None
    import cupyx

    with backend._device(cp):
        return _shading_blocking_gpu(cp, cupyx.scatter_add, np.asarray(xyz, float), np.asarray(normals, float),
                                     np.asarray(sun_vec, float), np.asarray(t, float), helio, rec, i_all, j_all,
                                     r_shade, dz, n_points, tower_shadow)


def shading_blocking_bruteforce(
    xyz: np.ndarray,
    normals: np.ndarray,
    sun_vec: np.ndarray,
    t: np.ndarray,
    helio: Heliostat,
    rec: Receiver | None,
    n_points: int = 8,
    tower_shadow: bool = True,
) -> dict[str, np.ndarray]:
    """Reference: every ordered pair is tested (no radius search, no corridor filter)."""
    n = xyz.shape[0]
    e_w, e_h = mirror_frames(normals)
    offsets = grid_offsets(n_points)
    pts = mirror_points(xyz, e_w, e_h, helio, offsets)
    i_all, j_all = np.nonzero(~np.eye(n, dtype=bool))
    shaded_mirrors = _hit_mask(pts, i_all, j_all, sun_vec, xyz, normals, e_w, e_h, helio, n)
    tower_hits = np.zeros(shaded_mirrors.shape, dtype=bool)
    if tower_shadow and rec is not None and rec.tower_radius > 0:
        tower_hits = ray_hits_structure(pts, sun_vec[None, None, :], rec)
    blocked = _hit_mask(pts, i_all, j_all, t, xyz, normals, e_w, e_h, helio, n)
    p = offsets.shape[0]
    other_occlusion = shaded_mirrors | blocked
    eta_shade = 1.0 - (shaded_mirrors | tower_hits).sum(axis=1) / p
    eta_comb = 1.0 - (other_occlusion | tower_hits).sum(axis=1) / p
    with np.errstate(invalid="ignore", divide="ignore"):
        eta_block = np.where(eta_shade > 0, eta_comb / eta_shade, 1.0)
    return {"shading": eta_shade, "blocking": eta_block, "combined": eta_comb,
            "non_tower_occluded": other_occlusion}


def compare_screening(xyz, normals, sun_vec, t, helio, rec, n_points=8, tower_shadow=True) -> dict:
    """Corridor screening vs brute force: missed shading/blocking and timing."""
    import time as _t

    t0 = _t.perf_counter()
    fast = shading_blocking(xyz, normals, sun_vec, t, helio, rec, PairCache(xyz), n_points, tower_shadow)
    t_fast = _t.perf_counter() - t0
    t0 = _t.perf_counter()
    ref = shading_blocking_bruteforce(xyz, normals, sun_vec, t, helio, rec, n_points, tower_shadow)
    t_ref = _t.perf_counter() - t0
    diff = fast["combined"] - ref["combined"]
    return {
        "n_heliostats": int(xyz.shape[0]),
        "max_abs_diff_combined": float(np.abs(diff).max()) if diff.size else 0.0,
        "n_heliostats_differing": int(np.sum(np.abs(diff) > 1e-12)),
        "missed_loss_total": float(np.clip(diff, 0, None).sum()),  # > 0: screening missed shading/blocking
        "field_loss_screened": float((1 - fast["combined"]).mean()),
        "field_loss_reference": float((1 - ref["combined"]).mean()),
        "time_screened_s": t_fast, "time_bruteforce_s": t_ref,
    }
