"""Cylindrical receiver: surface discretisation, tube panels, flux limits and aim-point geometry.

An aim point is described by (azimuth, z): the compass azimuth of a point on
the cylinder surface as seen from the axis, and its absolute height.

Panels are equal azimuth sectors; panel p (1-based) spans
[start + (p-1) * 360/n, start + p * 360/n). Each panel may have its own flux
limit, absorptivity and a model-bias factor (measured / modelled flux) fed back
from operation. All limits are converted to an *incident* flux limit per
point, because incident flux is what the optical model predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Receiver, UniformitySpec


def panel_of(rec: Receiver, azimuth_deg) -> np.ndarray:
    """0-based panel index for compass azimuth(s)."""
    rel = (np.asarray(azimuth_deg, dtype=float) - rec.panel_start_azimuth_deg) % 360.0
    return np.minimum((rel / (360.0 / rec.n_panels)).astype(int), rec.n_panels - 1)


def panel_bounds(rec: Receiver) -> list[tuple[float, float]]:
    w = 360.0 / rec.n_panels
    return [((rec.panel_start_azimuth_deg + p * w) % 360.0, (rec.panel_start_azimuth_deg + (p + 1) * w) % 360.0)
            for p in range(rec.n_panels)]


def _per_panel(rec: Receiver, values, default) -> np.ndarray:
    return np.asarray(values if values is not None else [default] * rec.n_panels, dtype=float)


def absorptivity_at(rec: Receiver, azimuth_deg) -> np.ndarray:
    return _per_panel(rec, rec.panel_absorptivity, rec.absorptivity)[panel_of(rec, azimuth_deg)]


def nominal_limit_kw_m2(rec: Receiver, azimuth_deg, z) -> np.ndarray:
    """Configured limit (on its own basis: incident or absorbed), kW/m2."""
    lim = _per_panel(rec, rec.panel_max_flux_kw_m2, rec.max_flux_kw_m2)[panel_of(rec, azimuth_deg)]
    if rec.limit_height_profile is not None:
        prof = np.asarray(rec.limit_height_profile, dtype=float)
        z0 = rec.center[2] - 0.5 * rec.height
        frac = np.clip((np.asarray(z, dtype=float) - z0) / rec.height, 0.0, 1.0)
        xp = np.linspace(0.0, 1.0, prof.size) if prof.size > 1 else np.array([0.0])
        lim = lim * (np.interp(frac, xp, prof) if prof.size > 1 else prof[0])
    return np.broadcast_to(lim, np.broadcast(np.asarray(azimuth_deg), np.asarray(z)).shape).astype(float)


def incident_limit_w_m2(rec: Receiver, azimuth_deg, z) -> np.ndarray:
    """Allowed *modelled incident* flux, W/m2: basis conversion and panel bias applied."""
    lim = nominal_limit_kw_m2(rec, azimuth_deg, z) * 1e3
    if rec.flux_limit_basis == "absorbed":
        lim = lim / absorptivity_at(rec, azimuth_deg)
    bias = _per_panel(rec, rec.panel_flux_bias, 1.0)[panel_of(rec, azimuth_deg)]
    return lim / bias


def region_mask(rec: Receiver, spec: UniformitySpec | None, azimuth_deg, z) -> np.ndarray:
    """Points belonging to the preheater region H."""
    az = np.asarray(azimuth_deg, dtype=float)
    mask = np.ones(np.broadcast(az, np.asarray(z)).shape, dtype=bool)
    if spec is None:
        return mask
    if spec.panels is not None:
        mask &= np.isin(panel_of(rec, az) + 1, np.asarray(spec.panels))
    if spec.z_range is not None:
        zz = np.asarray(z, dtype=float)
        mask &= (zz >= spec.z_range[0]) & (zz <= spec.z_range[1])
    return mask


@dataclass
class ReceiverGrid:
    azimuth_deg: np.ndarray  # (n_azimuth,) cell-centre compass azimuths
    z: np.ndarray  # (n_height,) cell-centre heights, absolute, m
    points: np.ndarray  # (n_height * n_azimuth, 3) cell centres on the surface
    normals: np.ndarray  # (n_height * n_azimuth, 3) outward unit normals
    cell_area: float  # m2, identical for every cell
    panel: np.ndarray  # (M,) 0-based panel index
    limit: np.ndarray  # (M,) allowed modelled incident flux, W/m2 (no safety margin)
    alpha: np.ndarray  # (M,) absorptivity

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z.size, self.azimuth_deg.size)

    @property
    def cell_az(self) -> np.ndarray:
        return np.tile(self.azimuth_deg, self.z.size)

    @property
    def cell_z(self) -> np.ndarray:
        return np.repeat(self.z, self.azimuth_deg.size)

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


def surface_normal(azimuth_deg) -> np.ndarray:
    a = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)


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
    normals = surface_normal(az_g.ravel())
    cell_area = rec.radius * np.radians(d_az) * dz
    return ReceiverGrid(az, z, points, normals, cell_area,
                        panel=panel_of(rec, az_g.ravel()),
                        limit=incident_limit_w_m2(rec, az_g.ravel(), z_g.ravel()),
                        alpha=absorptivity_at(rec, az_g.ravel()))


def heliostat_azimuth(rec: Receiver, xyz: np.ndarray) -> np.ndarray:
    """Compass azimuth of each heliostat as seen from the receiver axis."""
    dx = xyz[:, 0] - rec.center[0]
    dy = xyz[:, 1] - rec.center[1]
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def default_aim(rec: Receiver, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-point aiming: the surface point facing each heliostat, mid-height."""
    az = heliostat_azimuth(rec, xyz)
    return az, np.full(az.shape, rec.center[2])
