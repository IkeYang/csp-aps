"""Heliostat field: positions, layout generators and CSV I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from .config import Case, Heliostat


@dataclass
class Field:
    xyz: np.ndarray  # (N, 3) heliostat pivot positions, m

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])


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
            r += dm * (1.0 - radial_spacing)  # zone boundary needs a full DM gap
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
    rec = case.receiver
    if spec.layout_file:
        xyz = load_layout_csv(case.resolve(spec.layout_file), case.heliostat.pivot_height)
    elif spec.generator:
        gen = dict(spec.generator)
        kind = gen.pop("type", "radial_staggered")
        if kind != "radial_staggered":
            raise ValueError(f"Unknown layout generator '{kind}'")
        xy = radial_staggered(
            case.heliostat, (rec.center[0], rec.center[1]), rec.center[2], **gen
        )
        xyz = np.column_stack([xy, np.full(len(xy), case.heliostat.pivot_height)])
    else:
        raise ValueError("field: give either 'layout_file' or 'generator'")

    keep = np.ones(len(xyz), dtype=bool)
    if spec.boundary is not None:
        keep &= points_in_polygon(xyz[:, :2], spec.boundary)
    if spec.exclusion_radius > 0:
        r = np.hypot(xyz[:, 0] - rec.center[0], xyz[:, 1] - rec.center[1])
        keep &= r >= spec.exclusion_radius
    return Field(xyz[keep])
