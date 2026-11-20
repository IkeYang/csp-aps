"""Flux feasibility verification on an adaptive fine grid (full beams, no truncation).

The optimiser works on a coarse receiver grid with truncated footprints. A
coarse grid can miss the peak of a narrow beam (the peak falls between cell
centres), so every solution is re-checked here:

1. grid spacing h <= sigma_min / 3 (sigma_min = narrowest active beam), capped
   at ``max_cells``;
2. full Gaussian superposition of all active beams with the constraint DNI;
3. around the hottest local maxima a 9 x 9 sub-grid at spacing h / 4 refines
   the peak (the sampled maximum of a Gaussian at spacing sigma/12 is within
   0.4 % of the true peak);
4. ratios are taken against the per-point incident limit (panels, height
   profile, absorbed basis and panel bias included).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.ndimage import maximum_filter

from .config import Receiver
from .optics import footprints_at
from .receiver import build_grid, incident_limit_w_m2, panel_of, surface_normal, surface_point


@dataclass
class FluxCheck:
    status: str  # "ok" (<= allowed incl. margin), "margin_reduced" (<= limit), "over_limit"
    peak_ratio: float  # max modelled incident flux / limit (no margin)
    allowed_ratio: float  # 1 - safety margin
    peak_flux_kw_m2: float
    hotspot_azimuth_deg: float
    hotspot_z: float
    hotspot_panel: int  # 1-based
    hotspot_limit_kw_m2: float  # incident-equivalent limit at the hotspot
    exceedance_kw_m2: float  # peak above the limit (0 if within)
    grid: tuple[int, int]  # (n_azimuth, n_height) of the verification grid
    contributors: list[tuple[int, float]] = dc_field(default_factory=list)  # (heliostat index, kW/m2)
    hotspots: list[tuple[float, float, float]] = dc_field(default_factory=list)  # (az, z, ratio) above allowed

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def remap(self, mapping) -> "FluxCheck":
        """Translate contributor indices (positions in the beam list) to heliostat indices."""
        self.contributors = [(int(mapping[i]), f) for i, f in self.contributors]
        return self

    def summary(self) -> dict:
        return {
            "status": self.status,
            "peak_flux_kw_m2": round(self.peak_flux_kw_m2, 2),
            "peak_to_limit": round(self.peak_ratio, 4),
            "allowed_ratio": round(self.allowed_ratio, 4),
            "margin_to_limit_kw_m2": round(self.hotspot_limit_kw_m2 - self.peak_flux_kw_m2, 2),
            "hotspot": {"azimuth_deg": round(self.hotspot_azimuth_deg, 2), "z_m": round(self.hotspot_z, 3),
                        "panel": self.hotspot_panel, "limit_kw_m2": round(self.hotspot_limit_kw_m2, 2)},
            "exceedance_kw_m2": round(self.exceedance_kw_m2, 2),
            "verification_grid": list(self.grid),
            "top_contributors": [{"heliostat": int(i), "flux_kw_m2": round(f, 2)} for i, f in self.contributors],
        }


def _flux_at(points, normals, aim, t, sigma, power, chunk_elems=4_000_000) -> np.ndarray:
    out = np.zeros(points.shape[0])
    if aim.shape[0] == 0:
        return out
    step = max(1, chunk_elems // max(points.shape[0], 1))
    for s in range(0, aim.shape[0], step):
        sl = slice(s, s + step)
        out += footprints_at(points, normals, aim[sl], t[sl], sigma[sl], power[sl]).sum(axis=0)
    return out


def verify_beams(
    rec: Receiver,
    aim_xyz: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    power: np.ndarray,
    safety_margin: float = 0.0,
    max_cells: int = 400_000,
    n_refine: int = 12,
) -> FluxCheck:
    """Check the superposed flux of the given beams (power = beam power at the image plane, W)."""
    allowed = 1.0 - safety_margin
    on = power > 0
    aim_xyz, t, sigma, power = aim_xyz[on], t[on], sigma[on], power[on]
    circ = 2 * math.pi * rec.radius
    if power.size == 0:
        lim = float(incident_limit_w_m2(rec, 0.0, rec.center[2])) / 1e3
        return FluxCheck("ok", 0.0, allowed, 0.0, 0.0, rec.center[2], 1, lim, 0.0, (0, 0))
    coarse_h = min(circ / rec.n_azimuth, rec.height / rec.n_height)
    h = min(float(sigma.min()) / 3.0, coarse_h)
    n_az = int(math.ceil(circ / h))
    n_z = int(math.ceil(rec.height / h))
    if n_az * n_z > max_cells:
        s = math.sqrt(max_cells / (n_az * n_z))
        n_az, n_z = max(rec.n_azimuth, int(n_az * s)), max(rec.n_height, int(n_z * s))
    grid = build_grid(rec, n_az, n_z)
    flux = _flux_at(grid.points, grid.normals, aim_xyz, t, sigma, power)
    ratio = grid.to_map(flux / grid.limit)

    # local maxima (azimuth wraps around), hottest first
    is_max = (ratio == maximum_filter(ratio, size=3, mode=("nearest", "wrap"))) & (ratio > 0)
    iz, ia = np.nonzero(is_max)
    order_all = np.argsort(-ratio[iz, ia])
    order = order_all[: max(n_refine, 1)]
    d_az = 360.0 / n_az
    dz = rec.height / n_z
    z0, z1 = rec.center[2] - 0.5 * rec.height, rec.center[2] + 0.5 * rec.height
    sub = np.linspace(-1.0, 1.0, 9)
    best = (-1.0, 0.0, 0.0)
    spots = []
    for k in order:
        az_c, z_c = grid.azimuth_deg[ia[k]], grid.z[iz[k]]
        az_s, z_s = np.meshgrid(az_c + sub * d_az, np.clip(z_c + sub * dz, z0, z1))
        az_s, z_s = az_s.ravel() % 360.0, z_s.ravel()
        f = _flux_at(surface_point(rec, az_s, z_s), surface_normal(az_s), aim_xyz, t, sigma, power)
        r = f / incident_limit_w_m2(rec, az_s, z_s)
        j = int(np.argmax(r))
        spots.append((float(az_s[j]), float(z_s[j]), float(r[j])))
        if r[j] > best[0]:
            best = (float(r[j]), float(az_s[j]), float(z_s[j]))
    peak_ratio, az_h, z_h = best
    p_h = surface_point(rec, az_h, z_h)[None, :]
    contrib = footprints_at(p_h, surface_normal(np.array([az_h])), aim_xyz, t, sigma, power)[:, 0]
    lim_h = float(incident_limit_w_m2(rec, az_h, z_h))
    peak_flux = float(contrib.sum())
    idx_on = np.flatnonzero(on)
    top = np.argsort(-contrib)[:5]
    status = "ok" if peak_ratio <= allowed + 1e-9 else ("margin_reduced" if peak_ratio <= 1.0 + 1e-9 else "over_limit")
    return FluxCheck(
        status=status, peak_ratio=peak_ratio, allowed_ratio=allowed, peak_flux_kw_m2=peak_flux / 1e3,
        hotspot_azimuth_deg=az_h, hotspot_z=z_h, hotspot_panel=int(panel_of(rec, az_h)) + 1,
        hotspot_limit_kw_m2=lim_h / 1e3, exceedance_kw_m2=max(0.0, (peak_flux - lim_h) / 1e3),
        grid=(n_az, n_z),
        contributors=[(int(idx_on[i]), float(contrib[i]) / 1e3) for i in top if contrib[i] > 0],
        hotspots=_hotspots(spots, grid, ratio, iz, ia, order_all[n_refine:], allowed, max_extra=400),
    )


def _hotspots(spots, grid, ratio, iz, ia, rest, allowed, max_extra):
    """Refined hotspots plus every other fine-grid local maximum above the allowed ratio."""
    out = [s for s in spots if s[2] > allowed + 1e-9]
    for k in rest[:max_extra]:
        r = float(ratio[iz[k], ia[k]])
        if r <= allowed + 1e-9:
            break
        out.append((float(grid.azimuth_deg[ia[k]]), float(grid.z[iz[k]]), r))
    return out
