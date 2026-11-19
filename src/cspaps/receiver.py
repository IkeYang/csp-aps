"""Cylindrical receiver surface discretisation and aim-point geometry.

An aim point is described by (azimuth, z): the compass azimuth of a point on
the cylinder surface as seen from the axis, and its absolute height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Receiver


@dataclass
class ReceiverGrid:
    azimuth_deg: np.ndarray  # (n_azimuth,) cell-centre compass azimuths
    z: np.ndarray  # (n_height,) cell-centre heights, absolute, m
    points: np.ndarray  # (n_height * n_azimuth, 3) cell centres on the surface
    normals: np.ndarray  # (n_height * n_azimuth, 3) outward unit normals
    cell_area: float  # m2, identical for every cell

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z.size, self.azimuth_deg.size)

    def to_map(self, values: np.ndarray) -> np.ndarray:
        """Reshape a flat per-cell array to (n_height, n_azimuth)."""
        return np.asarray(values).reshape(self.shape)


def surface_point(rec: Receiver, azimuth_deg, z) -> np.ndarray:
    """Point(s) on the cylinder surface at the given compass azimuth and height."""
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    z = np.broadcast_to(np.asarray(z, dtype=float), az.shape)
    cx, cy, _ = rec.center
    return np.stack(
        [cx + rec.radius * np.sin(az), cy + rec.radius * np.cos(az), z], axis=-1
    )


def build_grid(rec: Receiver, n_azimuth: int | None = None, n_height: int | None = None) -> ReceiverGrid:
    n_az = n_azimuth or rec.n_azimuth
    n_z = n_height or rec.n_height
    d_az = 360.0 / n_az
    az = (np.arange(n_az) + 0.5) * d_az
    z0 = rec.center[2] - 0.5 * rec.height
    dz = rec.height / n_z
    z = z0 + (np.arange(n_z) + 0.5) * dz
    az_g, z_g = np.meshgrid(az, z)  # (n_z, n_az)
    points = surface_point(rec, az_g, z_g).reshape(-1, 3)
    a = np.radians(az_g.ravel())
    normals = np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)
    cell_area = rec.radius * np.radians(d_az) * dz
    return ReceiverGrid(az, z, points, normals, cell_area)


def heliostat_azimuth(rec: Receiver, xyz: np.ndarray) -> np.ndarray:
    """Compass azimuth of each heliostat as seen from the receiver axis."""
    dx = xyz[:, 0] - rec.center[0]
    dy = xyz[:, 1] - rec.center[1]
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def default_aim(rec: Receiver, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-point aiming: the surface point facing each heliostat, mid-height."""
    az = heliostat_azimuth(rec, xyz)
    return az, np.full(az.shape, rec.center[2])
