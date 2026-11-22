"""Heliostat field: positions, layout generators and CSV I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from .config import Case, Heliostat
from .shading import PairCache

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # 2 pi / phi^2


# Boundary / exclusion checks accept points this close outside (m), so that
# heliostats placed exactly on a boundary survive CSV round trips.
BOUNDARY_TOL_M = 1e-3


@dataclass
class Field:
    xyz: np.ndarray  # (N, 3) heliostat pivot positions, m
    ids: np.ndarray | None = None  # (N,) heliostat IDs (default 1..N), kept through CSV round trips
    dropped: list[dict] = dc_field(default_factory=list, repr=False)  # heliostats removed by site checks
    _pairs: PairCache | None = dc_field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        if not np.isfinite(self.xyz).all():
            raise ValueError("Field: every mirror needs finite x,y,z coordinates")
        if self.ids is None:
            self.ids = np.arange(1, self.xyz.shape[0] + 1)
        self.ids = np.asarray(self.ids, dtype=np.int64)
        if self.ids.shape[0] != self.xyz.shape[0]:
            raise ValueError("Field: ids and xyz have different lengths")
        if np.unique(self.ids).size != self.ids.size:
            raise ValueError("Field: heliostat IDs must be unique")

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def pairs(self) -> PairCache:
        if self._pairs is None:
            self._pairs = PairCache(self.xyz)
        return self._pairs

    def subset(self, mask_or_idx) -> "Field":
        return Field(self.xyz[mask_or_idx], self.ids[mask_or_idx])

    def index_of(self, ids) -> np.ndarray:
        """Positions of the given heliostat IDs (raises on unknown IDs)."""
        lookup = {int(h): i for i, h in enumerate(self.ids)}
        try:
            return np.array([lookup[int(h)] for h in np.atleast_1d(ids)], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"unknown heliostat ID {exc.args[0]}") from None


def points_in_polygon(xy: np.ndarray, polygon, tol: float = 0.0) -> np.ndarray:
    """Points (N, 2) inside a closed polygon [[x, y], ...], or within ``tol`` m of its edges."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError("boundary must be a list of at least 3 [x, y] points")
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    inside = MplPath(poly).contains_points(xy)
    if tol > 0 and len(xy):
        inside |= distance_to_polygon(xy, poly) <= tol
    return inside


def distance_to_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """Distance of each point to the polygon outline, m."""
    poly = np.asarray(polygon, dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = xy[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-30), 0.0, 1.0)
    d = ap - t[..., None] * ab
    return np.sqrt(np.sum(d * d, axis=-1)).min(axis=1)


def radial_staggered(
    heliostat: Heliostat,
    tower_xy: tuple[float, float],
    tower_height: float,
    r_min: float,
    r_max: float,
    spacing_factor: float = 1.1,
    radial_spacing: float = 0.866,
    radial_growth: float = 0.0,
    azimuthal_spacing: float = 1.0,
) -> np.ndarray:
    """Radially staggered layout (DELSOL / Collado style), returns (N, 2) xy.

    DM = heliostat diagonal * spacing_factor is the collision-free diameter.
    Within an azimuthal zone every ring has the same number of heliostats and
    consecutive rings are offset by half a pitch. Radial step between rings:
        dr = DM * (radial_spacing + radial_growth * (r - r_min) / tower_height)
    A new zone (more heliostats per ring) starts when the pitch exceeds twice
    the minimum azimuthal pitch DM * azimuthal_spacing. All rules guarantee
    neighbour distances >= DM.
    """
    if radial_spacing < math.sqrt(3) / 2 - 1e-3:
        raise ValueError("radial_spacing must be >= 0.866 to avoid collisions")
    if azimuthal_spacing < 1.0:
        raise ValueError("azimuthal_spacing must be >= 1.0 to avoid collisions")
    dm = heliostat.diagonal * spacing_factor
    pitch_min = dm * azimuthal_spacing
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    r = r_min
    n_ring = max(3, int(2 * math.pi * r / pitch_min))
    ring_in_zone = 0
    while r <= r_max:
        if 2 * math.pi * r / n_ring > 2 * pitch_min:  # start a new, denser zone
            r += dm * max(0.0, 1.0 - radial_spacing)  # zone boundary needs a full DM gap
            n_ring = int(2 * math.pi * r / pitch_min)
            ring_in_zone = 0
            if r > r_max:
                break
        offset = 0.5 if ring_in_zone % 2 else 0.0
        ang = (np.arange(n_ring) + offset) * (2 * math.pi / n_ring)
        xs.append(tower_xy[0] + r * np.sin(ang))
        ys.append(tower_xy[1] + r * np.cos(ang))
        ring_in_zone += 1
        dr = dm * (radial_spacing + radial_growth * (r - r_min) / max(tower_height, 1.0))
        r += dr
    if not xs:
        return np.empty((0, 2))
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])


def remove_collisions(xy: np.ndarray, min_dist: float) -> np.ndarray:
    """Drop points closer than min_dist to an earlier point (earlier points win)."""
    if len(xy) < 2:
        return xy
    pairs = cKDTree(xy).query_pairs(min_dist * (1 - 1e-9), output_type="ndarray")
    if pairs.size == 0:
        return xy
    later: list[list[int]] = [[] for _ in range(len(xy))]
    for a, b in zip(pairs.min(axis=1).tolist(), pairs.max(axis=1).tolist()):
        later[a].append(b)
    dropped = np.zeros(len(xy), dtype=bool)
    for p in range(len(xy)):  # greedy in index order: a kept point removes later conflicts
        if not dropped[p]:
            dropped[later[p]] = True
    return xy[~dropped]


def biomimetic(
    heliostat: Heliostat,
    tower_xy: tuple[float, float],
    tower_height: float,
    r_min: float,
    r_max: float,
    spacing_factor: float = 1.1,
    a: float = 4.5,
    b: float = 0.65,
) -> np.ndarray:
    """Phyllotaxis spiral layout (Noone et al. 2012): theta_k = k * golden angle, r_k = a k^b.

    ``a`` in units of DM (collision-free diameter), ``b`` < 1 makes the field
    sparser with distance. Collisions are removed afterwards.
    """
    dm = heliostat.diagonal * spacing_factor
    a_m = a * dm
    if a_m <= 0 or b <= 0:
        raise ValueError("biomimetic: a and b must be > 0")
    k_max = int(math.ceil((r_max / a_m) ** (1.0 / b))) + 1
    if k_max > 2_000_000:
        raise ValueError("biomimetic: a too small for this r_max")
    k = np.arange(1, k_max + 1)
    r = a_m * k**b
    keep = (r >= r_min) & (r <= r_max)
    theta = k[keep] * GOLDEN_ANGLE
    r = r[keep]
    xy = np.column_stack([tower_xy[0] + r * np.sin(theta), tower_xy[1] + r * np.cos(theta)])
    return remove_collisions(xy, dm)


GENERATORS = {"radial_staggered": radial_staggered, "biomimetic": biomimetic}


def generate(case: Case, kind: str, params: dict[str, Any]) -> np.ndarray:
    """Run a layout generator and apply the boundary / exclusion zone. Returns (N, 3)."""
    if kind not in GENERATORS:
        raise ValueError(f"Unknown layout generator '{kind}' (use {sorted(GENERATORS)})")
    rec = case.receiver
    params = dict(params)
    if "r_max" not in params:
        params["r_max"] = default_r_max(case)
    params.setdefault("r_min", max(case.field.exclusion_radius, rec.radius + case.heliostat.diagonal))
    # Every subset (including later swaps) inherits the site's minimum centre distance.
    required = case.layout_opt.min_spacing_factor or case.layout_opt.fixed.get("spacing_factor", 1.1)
    params["spacing_factor"] = max(params.get("spacing_factor", 1.1), required)
    xy = GENERATORS[kind](case.heliostat, (rec.center[0], rec.center[1]), rec.center[2], **params)
    xy = remove_collisions(xy, required * case.heliostat.diagonal)
    xyz = np.column_stack([xy, np.full(len(xy), case.heliostat.pivot_height)])
    xyz = xyz[site_mask(case, xyz)[0]]
    count = case.layout_opt.n_heliostats
    if count and len(xyz) < count and case.field.boundary is not None:
        xyz = _fill_shortfall(case, xyz, count, required * case.heliostat.diagonal)
    return xyz


def _fill_shortfall(case: Case, incumbent: np.ndarray, count: int, spacing: float) -> np.ndarray:
    """Try translated triangular lattices aligned with the site or with world axes."""
    polygon = np.asarray(case.field.boundary)
    longest = np.diff(polygon, axis=0, append=polygon[:1])
    dx, dy = longest[np.argmax(np.linalg.norm(longest, axis=1))]
    for angle in {0.0, math.atan2(dy, dx)}:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        local = polygon @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        pitch = spacing * math.sqrt(3) / 2
        nx, ny = np.ceil((hi - lo) / [spacing, pitch]).astype(int) + 2
        for shift_x, shift_y in ((0., 0.), (.5, .5), (.25, .75), (.75, .25)):
            strips = []
            # Bound memory on long narrow plots. Site clipping accepts boundary pivots.
            for row in range(ny):
                y = lo[1] + (row + shift_y) * pitch
                x = lo[0] + (np.arange(nx) + (row % 2) * .5 + shift_x - 1) * spacing
                xy = np.column_stack([x, np.full(nx, y)]) @ rotation.T
                xyz = np.column_stack([xy, np.full(nx, case.heliostat.pivot_height)])
                strips.append(xyz[site_mask(case, xyz)[0]])
            trial = np.concatenate(strips)
            if len(trial) > len(incumbent):
                incumbent = trial
            if len(incumbent) >= count:
                return incumbent
    return incumbent


def default_r_max(case: Case) -> float:
    """Distance from the tower to the farthest boundary vertex (or 8 x tower height)."""
    rec = case.receiver
    if case.field.boundary is not None:
        b = np.asarray(case.field.boundary, dtype=float)
        return float(np.hypot(b[:, 0] - rec.center[0], b[:, 1] - rec.center[1]).max())
    return 8.0 * rec.center[2]


def site_mask(case: Case, xyz: np.ndarray, tol: float = BOUNDARY_TOL_M) -> tuple[np.ndarray, list[str]]:
    """(keep mask, reason per point) for the site boundary and the exclusion zone (shared by all paths)."""
    spec, rec = case.field, case.receiver
    keep = np.ones(len(xyz), dtype=bool)
    reason = [""] * len(xyz)
    if spec.boundary is not None and len(xyz):
        inside = points_in_polygon(xyz[:, :2], spec.boundary, tol)
        for i in np.flatnonzero(~inside):
            reason[i] = "outside site boundary"
        keep &= inside
    if spec.exclusion_radius > 0 and len(xyz):
        r = np.hypot(xyz[:, 0] - rec.center[0], xyz[:, 1] - rec.center[1])
        ok = r >= spec.exclusion_radius - tol
        for i in np.flatnonzero(~ok):
            reason[i] = reason[i] or f"inside exclusion radius ({r[i]:.4f} m < {spec.exclusion_radius} m)"
        keep &= ok
    return keep, reason


def clip_to_site(case: Case, xyz: np.ndarray) -> np.ndarray:
    return xyz[site_mask(case, xyz)[0]]


def read_layout(path: str | Path, pivot_height: float) -> tuple[np.ndarray, np.ndarray | None]:
    """(xyz, ids or None) from a layout CSV with columns [id,]x,y[,z]."""
    from .tables import read_user_csv

    df = read_user_csv(path)
    if not {"x", "y"} <= set(df.columns):
        raise ValueError(f"{path}: CSV needs a header with columns x,y[,z] (optionally id)")
    z = df["z"].to_numpy(float) if "z" in df else np.full(len(df), pivot_height)
    xyz = np.column_stack([df["x"].to_numpy(float), df["y"].to_numpy(float), z])
    ids = df["id"].to_numpy(np.int64) if "id" in df else None
    return xyz, ids


def load_layout_csv(path: str | Path, pivot_height: float) -> np.ndarray:
    return read_layout(path, pivot_height)[0]


def save_layout_csv(field: Field, path: str | Path) -> None:
    """id,x,y,z with micrometre precision (positions survive a round trip unchanged)."""
    import pandas as pd

    pd.DataFrame({"id": field.ids, "x": field.xyz[:, 0], "y": field.xyz[:, 1], "z": field.xyz[:, 2]}
                 ).to_csv(path, index=False, float_format="%.6f")


def build_field(case: Case) -> Field:
    """Field from the case; heliostats removed by the site checks are listed in ``Field.dropped``."""
    spec = case.field
    if spec.layout_file:
        xyz, ids = read_layout(case.resolve(spec.layout_file), case.heliostat.pivot_height)
        ids = np.arange(1, len(xyz) + 1) if ids is None else ids
    elif spec.generator:
        gen = dict(spec.generator)
        kind = gen.pop("type", "radial_staggered")
        if kind not in GENERATORS:
            raise ValueError(f"Unknown layout generator '{kind}' (use {sorted(GENERATORS)})")
        rec = case.receiver
        gen.setdefault("r_max", default_r_max(case))
        gen.setdefault("r_min", max(spec.exclusion_radius, rec.radius + case.heliostat.diagonal))
        xy = GENERATORS[kind](case.heliostat, (rec.center[0], rec.center[1]), rec.center[2], **gen)
        xyz = np.column_stack([xy, np.full(len(xy), case.heliostat.pivot_height)])
        keep, _ = site_mask(case, xyz)
        xyz = xyz[keep]  # generated points outside the site are simply not part of the layout
        ids = np.arange(1, len(xyz) + 1)
    else:
        raise ValueError("field: give either 'layout_file' or 'generator'")
    keep, reason = site_mask(case, xyz)
    dropped = [{"id": int(ids[i]), "x": float(xyz[i, 0]), "y": float(xyz[i, 1]), "reason": reason[i]}
               for i in np.flatnonzero(~keep)]
    return Field(xyz[keep], ids[keep], dropped)
