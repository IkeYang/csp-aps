"""Monte-Carlo ray tracing (MCRT) -- an independent check of the analytic model.

Not used inside any optimisation loop. Each heliostat launches rays from
random points on its mirror:
  1. sun direction sampled from the sunshape (Gaussian or pillbox disc)
  2. shading: the ray towards the sun is tested against neighbouring mirrors and the tower
  3. local mirror normal: ideal on-axis spherical focusing at the configured focal
     length, plus random slope and tracking errors (per-axis Gaussian)
  4. specular reflection; blocking test against neighbouring mirrors
  5. atmospheric attenuation; intersection with the cylinder's lateral surface
  6. hits are binned on the same receiver grid as the analytic flux map
Unlike HFLCAL it captures the true anisotropy of errors at oblique incidence,
non-Gaussian sunshapes and the exact mirror image, so it is the reference for
validating and calibrating the analytic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import optics, sun
from .config import Case
from .field import Field
from .receiver import ReceiverGrid, build_grid, default_aim, surface_point
from .shading import _corridor, mirror_frames, ray_hits_rect, ray_hits_tower
from .simulate import STAGES, Snapshot


@dataclass
class MCRTResult:
    flux: np.ndarray  # (n_height, n_azimuth) W/m2
    power: dict[str, float]  # field totals after each stage, W
    heliostat_power: np.ndarray  # (N,) W on the receiver
    n_rays: int
    grid: ReceiverGrid = dc_field(repr=False)
    seed: int = 0
    rays_per_heliostat: int = 0

    def summary(self) -> dict:
        p = self.power
        return {
            "n_rays": self.n_rays,
            "power_mw": {k: round(v / 1e6, 4) for k, v in p.items()},
            "field_efficiency": {
                k: round(p[k] / p[prev], 5) if p[prev] > 0 else 0.0
                for prev, k in zip(STAGES[:-1], STAGES[1:])
            },
            "peak_flux_kw_m2": round(float(self.flux.max()) / 1e3, 2),
        }


def _perp_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(v, z)
    n = np.linalg.norm(e1, axis=-1, keepdims=True)
    e1 = np.where(n > 1e-9, e1 / np.maximum(n, 1e-12), np.array([1.0, 0.0, 0.0]))
    e2 = np.cross(v, e1)
    return e1, e2


def _sample_sun(s: np.ndarray, shape: tuple, case: Case, rng) -> np.ndarray:
    e1, e2 = _perp_basis(s[None, :])
    if case.sun.shape == "pillbox":
        r = case.sun.half_angle_mrad * 1e-3 * np.sqrt(rng.random(shape))
        phi = 2 * np.pi * rng.random(shape)
        tx, ty = r * np.cos(phi), r * np.sin(phi)
    else:
        sig = case.sun.sigma_mrad * 1e-3
        tx, ty = rng.normal(0, sig, shape), rng.normal(0, sig, shape)
    d = s + tx[..., None] * e1[0] + ty[..., None] * e2[0]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _rect_mask(origins, dirs, ii, jj, xyz, normals, e_w, e_h, hw, hh, chunk=512) -> np.ndarray:
    """(N, R) bool: ray r of heliostat i hits neighbour j for any listed pair (i, j)."""
    mask = np.zeros(origins.shape[:2], dtype=bool)
    for start in range(0, ii.size, chunk):
        a, b = ii[start:start + chunk], jj[start:start + chunk]
        hits = ray_hits_rect(origins[a], dirs[a], xyz[b][:, None, :], normals[b][:, None, :],
                             e_w[b][:, None, :], e_h[b][:, None, :], hw, hh)
        np.logical_or.at(mask, a, hits)
    return mask


def _hit_cylinder(o, d, rec):
    """First intersection with the lateral surface; returns (hit mask, point)."""
    cx, cy, _ = rec.center
    ox, oy = o[..., 0] - cx, o[..., 1] - cy
    dx, dy = d[..., 0], d[..., 1]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - rec.radius**2
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    s1 = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2 * np.where(a > 1e-12, a, 1.0))
    p = o + s1[..., None] * d
    zb = rec.center[2] - 0.5 * rec.height
    hit = ok & (s1 > 0) & (p[..., 2] >= zb) & (p[..., 2] <= zb + rec.height)
    return hit, p


def trace(
    case: Case,
    fld: Field,
    sun_elevation: float,
    sun_azimuth: float,
    dni,
    aim: tuple[np.ndarray, np.ndarray] | None = None,
    active: np.ndarray | None = None,
    grid: ReceiverGrid | None = None,
    rays_per_heliostat: int | None = None,
    seed: int | None = None,
    chunk: int = 128,
) -> MCRTResult:
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    n = fld.n
    nr = int(rays_per_heliostat or case.mcrt.rays_per_heliostat)
    rng = np.random.default_rng(case.mcrt.seed if seed is None else seed)
    dni_v = np.broadcast_to(np.asarray(dni, dtype=float), (n,))
    aim_az, aim_z = aim if aim is not None else default_aim(rec, fld.xyz)
    active = np.ones(n, dtype=bool) if active is None else np.asarray(active, bool)
    s = sun.sun_vector(sun_elevation, sun_azimuth)
    aim_xyz = surface_point(rec, np.asarray(aim_az, float), np.asarray(aim_z, float))
    normals, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz, aim_xyz, s)
    if hel.focusing == "slant_range":
        focal = slant
    elif hel.focusing == "flat":
        focal = np.full(n, np.inf)
    else:
        focal = np.full(n, float(hel.focusing))
    att = optics.attenuation(slant, case.atmosphere)
    rho = hel.reflectivity * hel.cleanliness
    per_helio_track = case.mcrt.tracking_mode == "per_heliostat"
    sig_n = 1e-3 * (hel.slope_error_mrad if per_helio_track else np.hypot(hel.slope_error_mrad, hel.tracking_error_mrad))
    sig_spec = 1e-3 * hel.specular_error_mrad
    if per_helio_track:  # one fixed pointing bias per heliostat for this realisation
        bias = 1e-3 * hel.tracking_error_mrad * rng.normal(size=(n, 2))
        normals = normals + bias[:, :1] * mirror_frames(normals)[0] + bias[:, 1:] * mirror_frames(normals)[1]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    e_w, e_h = mirror_frames(normals)
    hw, hh = 0.5 * hel.width, 0.5 * hel.height
    diag = hel.diagonal * 1.1
    dz = float(np.ptp(fld.xyz[:, 2])) if n > 1 else 0.0
    t_el_min = float(np.degrees(np.arcsin(np.clip(t[:, 2].min(), -1, 1)))) if n else 90.0
    radius = (hel.height + dz) / np.tan(np.radians(max(min(sun_elevation, t_el_min), 2.0))) + diag
    pair_i, pair_j = fld.pairs.get(radius)
    rel = fld.xyz[pair_j] - fld.xyz[pair_i]

    totals = {k: 0.0 for k in STAGES}
    per_h = np.zeros(n)
    energy = np.zeros(grid.points.shape[0])
    n_az, n_z = grid.azimuth_deg.size, grid.z.size
    z0 = rec.center[2] - 0.5 * rec.height
    for start in range(0, n, chunk):
        sl = np.arange(start, min(n, start + chunk))
        c = sl.size
        a = (rng.random((c, nr)) - 0.5) * hel.width
        b = (rng.random((c, nr)) - 0.5) * hel.height
        o = fld.xyz[sl, None, :] + a[..., None] * e_w[sl, None, :] + b[..., None] * e_h[sl, None, :]
        s_ray = _sample_sun(s, (c, nr), case, rng)
        w = dni_v[sl, None] * hel.area / nr * np.clip(np.einsum("crk,ck->cr", s_ray, normals[sl]), 0, None)
        totals["sun"] += float(dni_v[sl].sum() * hel.area)
        totals["cosine"] += float(w.sum())
        w = w * rho
        totals["reflectivity"] += float(w.sum())

        # shading (towards the sun) by neighbours and the tower
        in_chunk = (pair_i >= sl[0]) & (pair_i <= sl[-1])
        keep = in_chunk & _corridor(rel, np.broadcast_to(s, rel.shape), diag, np.full(rel.shape[0], np.inf))
        shaded = _rect_mask(o, s_ray, pair_i[keep] - sl[0], pair_j[keep], fld.xyz, normals, e_w, e_h, hw, hh)
        if case.shading.enabled and case.shading.tower_shadow and rec.tower_radius > 0:
            shaded |= ray_hits_tower(o, s_ray, rec.center[:2], rec.tower_radius, rec.top)
        if not case.shading.enabled:
            shaded[:] = False
        w = np.where(shaded, 0.0, w)
        totals["shading"] += float(w.sum())

        # local normal: spherical focusing + slope/tracking errors
        n_loc = normals[sl, None, :] - (a[..., None] * e_w[sl, None, :] + b[..., None] * e_h[sl, None, :]) / (2 * focal[sl, None, None])
        n_loc += sig_n * (rng.normal(size=(c, nr))[..., None] * e_w[sl, None, :]
                          + rng.normal(size=(c, nr))[..., None] * e_h[sl, None, :])
        n_loc /= np.linalg.norm(n_loc, axis=-1, keepdims=True)
        d_in = -s_ray
        r = d_in - 2 * np.sum(d_in * n_loc, axis=-1, keepdims=True) * n_loc
        if sig_spec > 0:  # specularity error on the reflected ray
            r1, r2 = _perp_basis(r.reshape(-1, 3))
            r = r + sig_spec * (rng.normal(size=(c, nr))[..., None] * r1.reshape(c, nr, 3)
                                + rng.normal(size=(c, nr))[..., None] * r2.reshape(c, nr, 3))
            r /= np.linalg.norm(r, axis=-1, keepdims=True)

        keep_b = in_chunk & _corridor(rel, t[pair_i], diag, np.full(rel.shape[0], np.inf))
        blocked = _rect_mask(o, r, pair_i[keep_b] - sl[0], pair_j[keep_b], fld.xyz, normals, e_w, e_h, hw, hh)
        if not case.shading.enabled:
            blocked[:] = False
        w = np.where(blocked, 0.0, w)
        totals["blocking"] += float(w.sum())
        w = w * att[sl, None]
        totals["attenuation"] += float(w.sum())

        w = w * active[sl, None]
        hit, p = _hit_cylinder(o, r, rec)
        w_hit = np.where(hit, w, 0.0)
        totals["intercept"] += float(w_hit.sum())
        per_h[sl] = w_hit.sum(axis=1)
        az_hit = np.degrees(np.arctan2(p[..., 0] - rec.center[0], p[..., 1] - rec.center[1])) % 360.0
        ia = np.clip((az_hit / 360.0 * n_az).astype(int), 0, n_az - 1)
        iz = np.clip(((p[..., 2] - z0) / rec.height * n_z).astype(int), 0, n_z - 1)
        energy += np.bincount((iz * n_az + ia)[hit], weights=w_hit[hit], minlength=energy.size)
    flux = grid.to_map(energy / grid.cell_area)
    used_seed = case.mcrt.seed if seed is None else seed
    return MCRTResult(flux, totals, per_h, n * nr, grid, int(used_seed), nr)


def trace_repeated(case: Case, fld: Field, sun_elevation, sun_azimuth, dni, aim=None, active=None, grid=None,
                   rays_per_heliostat: int | None = None, n_seeds: int | None = None, seed0: int | None = None):
    """Independent realisations -> mean flux map and statistical uncertainty.

    Returns (mean MCRTResult, stats dict with the standard error of the total
    power and of the peak, and the relative standard error at the mean-map peak).
    """
    k = int(n_seeds or case.mcrt.n_seeds)
    s0 = case.mcrt.seed if seed0 is None else seed0
    runs = [trace(case, fld, sun_elevation, sun_azimuth, dni, aim, active, grid, rays_per_heliostat, s0 + j)
            for j in range(max(1, k))]
    flux = np.mean([r.flux for r in runs], axis=0)
    power = {key: float(np.mean([r.power[key] for r in runs])) for key in runs[0].power}
    per_h = np.mean([r.heliostat_power for r in runs], axis=0)
    mean = MCRTResult(flux, power, per_h, sum(r.n_rays for r in runs), runs[0].grid, s0, runs[0].rays_per_heliostat)
    p_rec = np.array([r.power["intercept"] for r in runs])
    peaks = np.array([r.flux.max() for r in runs])
    j = np.unravel_index(int(np.argmax(flux)), flux.shape)
    at_peak = np.array([r.flux[j] for r in runs])
    n = len(runs)
    se = (lambda x: float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    stats = {
        "engine": "cspaps-mcrt", "seeds": [s0 + j for j in range(n)], "rays_per_heliostat": runs[0].rays_per_heliostat,
        "grid": list(flux.shape[::-1]),
        "power_on_receiver_mw": {"mean": float(p_rec.mean() / 1e6), "std_error": se(p_rec / 1e6)},
        "peak_flux_kw_m2": {"mean_of_runs": float(peaks.mean() / 1e3), "std_error": se(peaks / 1e3),
                            "of_mean_map": float(flux.max() / 1e3)},
        "relative_std_error_at_peak_cell": float(se(at_peak) / at_peak.mean()) if at_peak.mean() > 0 else float("nan"),
        "note": "the maximum of a noisy map is biased high; compare the mean map and its standard error",
    }
    return mean, stats


def convergence(case: Case, fld: Field, sun_elevation, sun_azimuth, dni, aim=None, active=None,
                ray_counts=(250, 500, 1000, 2000, 4000), n_seeds: int = 3) -> list[dict]:
    """Peak and power versus rays per heliostat (each with n_seeds realisations)."""
    out = []
    for nr in ray_counts:
        _, st = trace_repeated(case, fld, sun_elevation, sun_azimuth, dni, aim, active, None, nr, n_seeds)
        out.append({"rays_per_heliostat": nr, "power_mw": st["power_on_receiver_mw"]["mean"],
                    "power_se_mw": st["power_on_receiver_mw"]["std_error"],
                    "peak_kw_m2": st["peak_flux_kw_m2"]["of_mean_map"],
                    "peak_se_kw_m2": st["peak_flux_kw_m2"]["std_error"],
                    "rel_se_at_peak": st["relative_std_error_at_peak_cell"]})
    return out


def compare(analytic: Snapshot, ray: MCRTResult) -> dict:
    """Stage-by-stage and flux-map comparison: analytic (HFLCAL) vs MCRT."""
    pa = {k: float(v.sum()) for k, v in analytic.power.items()}
    pm = ray.power
    stages = {}
    for prev, k in zip(STAGES[:-1], STAGES[1:]):
        ea = pa[k] / pa[prev] if pa[prev] > 0 else 0.0
        em = pm[k] / pm[prev] if pm[prev] > 0 else 0.0
        stages[k] = {"analytic": round(ea, 5), "mcrt": round(em, 5), "diff": round(ea - em, 5)}
    fa, fm = analytic.flux, ray.flux
    lit = fm > 0.05 * fm.max() if fm.max() > 0 else np.ones_like(fm, dtype=bool)
    rms = float(np.sqrt(np.mean((fa[lit] - fm[lit]) ** 2)) / fm[lit].mean()) if lit.any() else 0.0
    return {
        "power_on_receiver_mw": {"analytic": round(pa["intercept"] / 1e6, 4),
                                 "mcrt": round(pm["intercept"] / 1e6, 4),
                                 "rel_diff": round(pa["intercept"] / pm["intercept"] - 1, 5) if pm["intercept"] else None},
        "efficiency_by_stage": stages,
        "peak_flux_kw_m2": {"analytic": round(float(fa.max()) / 1e3, 1), "mcrt": round(float(fm.max()) / 1e3, 1)},
        "flux_map_nrmse": round(rms, 4),
        "flux_map_correlation": round(float(np.corrcoef(fa.ravel(), fm.ravel())[0, 1]), 4),
        "hotspot": {"analytic_azimuth_deg": round(float(ray.grid.azimuth_deg[np.unravel_index(int(np.argmax(fa)), fa.shape)[1]]), 2),
                    "mcrt_azimuth_deg": round(float(ray.grid.azimuth_deg[np.unravel_index(int(np.argmax(fm)), fm.shape)[1]]), 2),
                    "analytic_z_m": round(float(ray.grid.z[np.unravel_index(int(np.argmax(fa)), fa.shape)[0]]), 3),
                    "mcrt_z_m": round(float(ray.grid.z[np.unravel_index(int(np.argmax(fm)), fm.shape)[0]]), 3)},
        "difference_direction": "analytic minus MCRT",
        "n_rays": ray.n_rays,
        "seed": ray.seed,
        "rays_per_heliostat": ray.rays_per_heliostat,
        "grid": [int(ray.grid.azimuth_deg.size), int(ray.grid.z.size)],
    }
