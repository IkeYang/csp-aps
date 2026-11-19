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


@dataclass
class Field:
    xyz: np.ndarray  # (N, 3) heliostat pivot positions, m
    _pairs: PairCache | None = dc_field(default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def pairs(self) -> PairCache:
        if self._pairs is None:
            self._pairs = PairCache(self.xyz)
        return self._pairs

    def subset(self, mask_or_idx) -> "Field":
        return Field(self.xyz[mask_or_idx])


def points_in_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """Boolean mask of points (N, 2) lying inside a closed polygon [[x, y], ...]."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError("boundary must be a list of at least 3 [x, y] points")
    return MplPath(poly).contains_points(xy)


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
    xy = GENERATORS[kind](case.heliostat, (rec.center[0], rec.center[1]), rec.center[2], **params)
    xyz = np.column_stack([xy, np.full(len(xy), case.heliostat.pivot_height)])
    return clip_to_site(case, xyz)


def default_r_max(case: Case) -> float:
    """Distance from the tower to the farthest boundary vertex (or 8 x tower height)."""
    rec = case.receiver
    if case.field.boundary is not None:
        b = np.asarray(case.field.boundary, dtype=float)
        return float(np.hypot(b[:, 0] - rec.center[0], b[:, 1] - rec.center[1]).max())
    return 8.0 * rec.center[2]


def clip_to_site(case: Case, xyz: np.ndarray) -> np.ndarray:
    spec, rec = case.field, case.receiver
    keep = np.ones(len(xyz), dtype=bool)
    if spec.boundary is not None and len(xyz):
        keep &= points_in_polygon(xyz[:, :2], spec.boundary)
    if spec.exclusion_radius > 0:
        r = np.hypot(xyz[:, 0] - rec.center[0], xyz[:, 1] - rec.center[1])
        keep &= r >= spec.exclusion_radius
    return xyz[keep]


def load_layout_csv(path: str | Path, pivot_height: float) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    if not {"x", "y"} <= set(names):
        raise ValueError(f"{path}: CSV needs a header with columns x,y[,z]")
    x = np.atleast_1d(data["x"])
    y = np.atleast_1d(data["y"])
    z = np.atleast_1d(data["z"]) if "z" in names else np.full(x.shape, pivot_height)
    return np.column_stack([x, y, z]).astype(float)


def save_layout_csv(field: Field, path: str | Path) -> None:
    np.savetxt(path, field.xyz, delimiter=",", header="x,y,z", comments="", fmt="%.3f")


def build_field(case: Case) -> Field:
    spec = case.field
    if spec.layout_file:
        xyz = load_layout_csv(case.resolve(spec.layout_file), case.heliostat.pivot_height)
        return Field(clip_to_site(case, xyz))
    if spec.generator:
        gen = dict(spec.generator)
        kind = gen.pop("type", "radial_staggered")
        return Field(generate(case, kind, gen))
    raise ValueError("field: give either 'layout_file' or 'generator'")
