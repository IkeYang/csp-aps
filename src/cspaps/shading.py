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


def mirror_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane unit axes (width: horizontal, height) of each mirror."""
    z = np.array([0.0, 0.0, 1.0])
    e_w = np.cross(z, normals)
    norm = np.linalg.norm(e_w, axis=-1, keepdims=True)
    e_w = np.where(norm > 1e-9, e_w / np.maximum(norm, 1e-12), np.array([1.0, 0.0, 0.0]))
    e_h = np.cross(normals, e_w)
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


def ray_hits_rect(o, d, c, n, e_w, e_h, half_w, half_h) -> np.ndarray:
    """Does the ray o + s d (s > 0) hit the rectangle (centre c, normal n, axes e_w, e_h)?

    All vector arguments broadcast against each other with a trailing axis of 3.
    """
    denom = np.sum(d * n, axis=-1)
    safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    s = np.sum((c - o) * n, axis=-1) / safe
    p = o + s[..., None] * d - c
    a = np.sum(p * e_w, axis=-1)
    b = np.sum(p * e_h, axis=-1)
    return (np.abs(denom) > 1e-12) & (s > 1e-6) & (np.abs(a) <= half_w) & (np.abs(b) <= half_h)


def ray_hits_tower(o, d, center_xy, radius: float, z_top: float) -> np.ndarray:
    """Does the ray o + s d (s > 0) hit a vertical cylinder from z = 0 to z_top?"""
    ox = o[..., 0] - center_xy[0]
    oy = o[..., 1] - center_xy[1]
    dx, dy = d[..., 0], d[..., 1]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    a_safe = np.where(a > 1e-12, a, 1.0)
    s1 = (-b - sq) / (2 * a_safe)
    s2 = (-b + sq) / (2 * a_safe)
    hit = np.zeros(ok.shape, dtype=bool)
    for s in (s1, s2):
        z = o[..., 2] + s * d[..., 2]
        hit |= ok & (s > 1e-6) & (z >= 0) & (z <= z_top)
    return hit


def neighbor_pairs(xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs (i, j), i != j, with horizontal distance <= radius."""
    if len(xy) < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
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
        self.i = self.j = np.zeros(0, dtype=np.int64)

    def get(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        if radius > self.radius:
            self.radius = radius * 1.25  # head-room so small increases do not recompute
            self.i, self.j = neighbor_pairs(self.xyz[:, :2], self.radius)
        d = np.hypot(*(self.xyz[self.j, :2] - self.xyz[self.i, :2]).T)
        keep = d <= radius
        return self.i[keep], self.j[keep]


def _corridor(rel: np.ndarray, d: np.ndarray, diag: float, max_along: np.ndarray) -> np.ndarray:
    along = np.sum(rel * d, axis=-1)
    lateral = np.linalg.norm(rel - along[:, None] * d, axis=-1)
    return (along > -diag) & (along < max_along) & (lateral < diag)


def _hit_mask(points, i, j, d, xyz, normals, e_w, e_h, helio, n, chunk=4096) -> np.ndarray:
    """(N, P) mask: point of mirror i hits any listed neighbour j along direction d."""
    mask = np.zeros(points.shape[:2], dtype=bool)
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
        return {"shading": ones, "blocking": ones, "combined": ones}
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
    rel = xyz[j_all] - xyz[i_all]

    keep_s = _corridor(rel, np.broadcast_to(sun_vec, rel.shape), diag, np.full(rel.shape[0], r_shade + diag))
    shaded = _hit_mask(pts, i_all[keep_s], j_all[keep_s], sun_vec, xyz, normals, e_w, e_h, helio, n)
    if tower_shadow and rec is not None and rec.tower_radius > 0:
        shaded |= ray_hits_tower(pts, sun_vec[None, None, :], rec.center[:2], rec.tower_radius, rec.top)

    t_pair = t[i_all]
    max_along = (helio.height + dz) / np.maximum(t_pair[:, 2], 0.03) + diag
    keep_b = _corridor(rel, t_pair, diag, max_along)
    blocked = _hit_mask(pts, i_all[keep_b], j_all[keep_b], t, xyz, normals, e_w, e_h, helio, n)

    p = offsets.shape[0]
    eta_shade = 1.0 - shaded.sum(axis=1) / p
    eta_comb = 1.0 - (shaded | blocked).sum(axis=1) / p
    with np.errstate(invalid="ignore", divide="ignore"):
        eta_block = np.where(eta_shade > 0, eta_comb / eta_shade, 1.0)
    return {"shading": eta_shade, "blocking": eta_block, "combined": eta_comb}


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
    shaded = _hit_mask(pts, i_all, j_all, sun_vec, xyz, normals, e_w, e_h, helio, n)
    if tower_shadow and rec is not None and rec.tower_radius > 0:
        shaded |= ray_hits_tower(pts, sun_vec[None, None, :], rec.center[:2], rec.tower_radius, rec.top)
    blocked = _hit_mask(pts, i_all, j_all, t, xyz, normals, e_w, e_h, helio, n)
    p = offsets.shape[0]
    eta_shade = 1.0 - shaded.sum(axis=1) / p
    eta_comb = 1.0 - (shaded | blocked).sum(axis=1) / p
    with np.errstate(invalid="ignore", divide="ignore"):
        eta_block = np.where(eta_shade > 0, eta_comb / eta_shade, 1.0)
    return {"shading": eta_shade, "blocking": eta_block, "combined": eta_comb}


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
