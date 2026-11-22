"""Analytic optics: cosine, attenuation, HFLCAL beam spread, flux on a cylinder.

Flux model (HFLCAL, Schwarzboezl et al. 2009): the reflected beam of each
heliostat is a circular Gaussian in the image plane (normal to the central
ray, through the aim point) with standard deviation

    sigma = D * sqrt(sigma_sun^2 + (2 sigma_slope)^2 + (2 sigma_track)^2 + sigma_spec^2 + sigma_ast^2)

where D is the slant range. Each receiver surface point is projected along
the central ray onto the image plane, and the image-plane flux is multiplied
by the cosine between the ray and the surface normal. Back-facing points get
nothing. With this projection the flux integrated over the receiver equals
the beam power times the exact Gaussian intercept fraction.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .config import Atmosphere, Heliostat, Receiver, Sun
from .receiver import ReceiverGrid


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def attenuation(slant_range_m, atm: Atmosphere | None = None) -> np.ndarray:
    """Atmospheric transmittance between heliostat and receiver.

    vittitoe_biggs_clear: 0.99321 - 0.1176 S + 0.0197 S^2 (S <= 1 km), exp(-0.1106 S) beyond
    vittitoe_biggs_hazy:  0.98707 - 0.2748 S + 0.03394 S^2 (S <= 1 km), continued
                          exponentially beyond 1 km with the slope at 1 km
    beer_lambert:         exp(-beta S), beta = extinction_per_km
    with S the slant range in km.
    """
    model = atm.model if atm is not None else "vittitoe_biggs_clear"
    s = np.asarray(slant_range_m, dtype=float) / 1000.0
    if model == "vittitoe_biggs_clear":
        return np.where(s <= 1.0, 0.99321 - 0.1176 * s + 0.0197 * s**2, np.exp(-0.1106 * s))
    if model == "vittitoe_biggs_hazy":
        at_1km = 0.98707 - 0.2748 + 0.03394
        slope = (-0.2748 + 2 * 0.03394) / at_1km
        near = 0.98707 - 0.2748 * s + 0.03394 * s**2
        return np.where(s <= 1.0, near, at_1km * np.exp(slope * (s - 1.0)))
    if model == "beer_lambert":
        return np.exp(-atm.extinction_per_km * s)
    raise ValueError(f"Unknown atmosphere model '{model}'")


def heliostat_geometry(helio_xyz: np.ndarray, aim_xyz: np.ndarray, sun_vec: np.ndarray):
    """Tracking geometry for each heliostat.

    Returns (normal, target_dir, slant_range, cos_incidence). ``target_dir``
    is the unit vector from the heliostat to its aim point; the mirror normal
    bisects the sun vector and the target direction. Works on any leading
    shape (..., 3).
    """
    to_aim = aim_xyz - helio_xyz
    slant = np.linalg.norm(to_aim, axis=-1)
    t = to_aim / slant[..., None]
    n = _unit(sun_vec + t)
    cos_inc = np.clip(np.einsum("...k,k->...", n, sun_vec), 0.0, 1.0)
    return n, t, slant, cos_inc


def beam_sigma(slant_range: np.ndarray, cos_inc: np.ndarray, helio: Heliostat, sun: Sun) -> np.ndarray:
    """HFLCAL total beam standard deviation in metres at the image plane."""
    d = np.asarray(slant_range, dtype=float)
    d_m = np.sqrt(helio.area)
    if helio.focusing == "slant_range":
        d_over_f = np.ones_like(d)
    elif helio.focusing == "flat":
        d_over_f = np.zeros_like(d)
    else:
        d_over_f = d / float(helio.focusing)
    h_t = d_m * np.abs(d_over_f - cos_inc)
    w_s = d_m * np.abs(d_over_f * cos_inc - 1.0)
    sigma_ast = np.sqrt(0.5 * (h_t**2 + w_s**2)) / (4.0 * d)
    s2 = (
        (sun.effective_sigma_mrad * 1e-3) ** 2
        + (2e-3 * helio.slope_error_mrad) ** 2
        + (2e-3 * helio.tracking_error_mrad) ** 2
        + (1e-3 * helio.specular_error_mrad) ** 2
        + sigma_ast**2
    )
    return d * np.sqrt(s2)


def image_plane_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (horizontal, 'up') basis of the image plane normal to each ray t."""
    z = np.array([0.0, 0.0, 1.0])
    e_u = np.cross(t, z)
    norm = np.linalg.norm(e_u, axis=-1, keepdims=True)
    # Vertical rays are not physical here, but keep the maths finite.
    e_u = np.where(norm > 1e-9, e_u / np.maximum(norm, 1e-12), np.array([1.0, 0.0, 0.0]))
    e_v = np.cross(e_u, t)
    return e_u, e_v


def intercept_fraction(
    rec: Receiver,
    aim_xyz: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    n_quad: int = 48,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Fraction of each Gaussian beam that hits the cylinder's lateral surface.

    Independent of the flux-map resolution. The visible (front) half of the
    cylinder projects onto the image plane as a band: the vertical surface
    line at horizontal image coordinate U in [-R, R] (measured from the axis)
    maps to a segment u = const, v in [v_bot(U), v_top(U)]. The intercept is
        integral pdf(u) * (cdf(v_top) - cdf(v_bot)) dU,
    evaluated with Gauss-Legendre nodes restricted to +-8 sigma around the
    beam centre so that narrow beams are always resolved. ``z_range`` limits
    the target to a horizontal band of the surface (the band is convex and
    directly visible, so the same formula applies).
    """
    shape = aim_xyz.shape[:-1]
    aim_xyz = aim_xyz.reshape(-1, 3)
    t = t.reshape(-1, 3)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    n = aim_xyz.shape[0]
    if n == 0:
        return np.zeros(shape)
    e_u, e_v = image_plane_basis(t)
    R = rec.radius
    z_lo, z_hi = z_range if z_range is not None else (rec.center[2] - 0.5 * rec.height, rec.center[2] + 0.5 * rec.height)
    c_bot = np.array([rec.center[0], rec.center[1], z_lo])
    h = -t.copy()
    h[:, 2] = 0.0
    h /= np.linalg.norm(h, axis=1, keepdims=True)  # horizontal, axis -> heliostat
    a_u = np.einsum("nk,nk->n", aim_xyz - c_bot, e_u)  # beam centre in U coordinates
    lo = np.maximum(-R, a_u - 8 * sigma)
    hi = np.minimum(R, a_u + 8 * sigma)
    ok = hi > lo
    x, w = np.polynomial.legendre.leggauss(n_quad)
    mid, half = 0.5 * (hi + lo), 0.5 * np.where(ok, hi - lo, 0.0)
    U = mid[:, None] + half[:, None] * x[None, :]  # (N, Q)
    depth = np.sqrt(np.clip(R * R - U * U, 0.0, None))
    p_bot = c_bot + h[:, None, :] * depth[..., None] + e_u[:, None, :] * U[..., None]
    r = p_bot - aim_xyz[:, None, :]
    u = U - a_u[:, None]
    v_bot = np.einsum("nqk,nk->nq", r, e_v)
    v_top = v_bot + (z_hi - z_lo) * e_v[:, 2:3]
    s = sigma[:, None]
    pdf = np.exp(-0.5 * (u / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    band = np.abs(ndtr(v_top / s) - ndtr(v_bot / s))
    frac = (pdf * band) @ w * half
    return np.clip(np.where(ok, frac, 0.0), 0.0, 1.0).reshape(shape)


def surface_spot(rec: Receiver, aim_azimuth_deg, t: np.ndarray, sigma, k: float = 2.0) -> dict:
    """Footprint of each Gaussian beam on the receiver surface around its aim point (local tangent plane).

    Tangent axes at compass azimuth phi: e_phi = (cos phi, -sin phi, 0) (towards increasing azimuth) and
    e_z. A surface offset w = (u, v) lies at image-plane distance |w - (w.t) t|, so with
    a = (t.e_phi, t.e_z) the flux falls off as exp(-w^T (I - a a^T) w / (2 sigma^2)) (the same Gaussian as
    ``footprints_at``). The footprint covariance is therefore
        Sigma = sigma^2 (I - a a^T)^-1 = sigma^2 (I + a a^T / cos^2 i),   cos i = |t.n| = sqrt(1 - |a|^2),
    i.e. the circular image is stretched by 1 / cos i along a. Returned per beam: ``sigma_m``,
    ``incidence_deg``, the k-sigma ellipse (``semi_major_m`` = k sigma / cos i, ``semi_minor_m`` = k sigma,
    ``major_angle_deg`` from the circumferential axis towards +z), its bounding extents ``width_m``
    (circumferential) and ``height_m`` (vertical), and ``cov`` (N, 2, 2) in (u, v) metres."""
    phi = np.radians(np.atleast_1d(np.asarray(aim_azimuth_deg, dtype=float)))
    t = np.asarray(t, dtype=float).reshape(-1, 3)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), phi.shape).astype(float)
    e_phi = np.stack([np.cos(phi), -np.sin(phi), np.zeros_like(phi)], axis=-1)
    n_out = np.stack([np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    a = np.stack([np.einsum("nk,nk->n", t, e_phi), t[:, 2]], axis=-1)
    cos_i = np.clip(np.abs(np.einsum("nk,nk->n", t, n_out)), 1e-3, 1.0)
    cov = sig[:, None, None] ** 2 * (np.eye(2)[None] + a[:, :, None] * a[:, None, :] / cos_i[:, None, None] ** 2)
    major = np.where(np.hypot(a[:, 0], a[:, 1]) > 1e-12, np.degrees(np.arctan2(a[:, 1], a[:, 0])), 0.0)
    return {"sigma_m": sig, "incidence_deg": np.degrees(np.arccos(cos_i)), "k": float(k),
            "semi_major_m": k * sig / cos_i, "semi_minor_m": k * sig, "major_angle_deg": major,
            "width_m": 2 * k * np.sqrt(cov[:, 0, 0]), "height_m": 2 * k * np.sqrt(cov[:, 1, 1]), "cov": cov}


def spot_outline(rec: Receiver, aim_azimuth_deg, aim_z, cov, k: float = 2.0, n: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """k-sigma ellipses of ``surface_spot`` in unrolled receiver coordinates (azimuth deg, z m), (N, n) each."""
    th = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(th), np.sin(th)])  # (2, n)
    chol = np.linalg.cholesky(np.asarray(cov, dtype=float))  # (N, 2, 2)
    w = k * chol @ circle  # (N, 2, n) metres on the tangent plane
    az = np.asarray(aim_azimuth_deg, dtype=float)[:, None] + np.degrees(w[:, 0] / rec.radius)
    return az, np.asarray(aim_z, dtype=float)[:, None] + w[:, 1]


def footprints(
    grid: ReceiverGrid,
    aim_xyz: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    power: np.ndarray,
) -> np.ndarray:
    """Flux (W/m2) of each beam on every receiver cell, dense (R, M).

    aim points (R, 3), ray directions heliostat->aim (R, 3), beam sigma (R,)
    in m, beam power arriving at the image plane (R,) in W.
    """
    return footprints_at(grid.points, grid.normals, aim_xyz, t, sigma, power)


def footprints_at(
    points: np.ndarray,
    normals: np.ndarray,
    aim_xyz: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    power: np.ndarray,
) -> np.ndarray:
    """Flux (W/m2) of each beam at arbitrary surface points (P, 3) with outward normals, (R, P)."""
    e_u, e_v = image_plane_basis(t)
    # Projections as matrix products (BLAS) instead of forming (R, P, 3) offsets.
    pts = points
    u = e_u @ pts.T - np.sum(aim_xyz * e_u, axis=1)[:, None]  # (R, P)
    v = e_v @ pts.T - np.sum(aim_xyz * e_v, axis=1)[:, None]
    cos_s = np.clip(-(t @ normals.T), 0.0, None)
    inv2s2 = (0.5 / sigma**2)[:, None]
    q = (u * u + v * v) * inv2s2
    out = np.exp(-q.astype(np.float32)).astype(np.float64)
    out *= cos_s
    out *= (power / (2 * np.pi * sigma**2))[:, None]
    return out


def flux_on_receiver(
    grid: ReceiverGrid,
    aim_xyz: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    power: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Superpose all heliostat images on the receiver surface.

    Returns (flux per cell (M,) in W/m2, intercepted power per heliostat (N,) in W).
    Flux is sampled at cell centres; the per-heliostat intercepted power from
    this sum is only as accurate as the grid -- use ``intercept_fraction`` for
    efficiencies.
    """
    n = aim_xyz.shape[0]
    flux = np.zeros(grid.points.shape[0])
    intercepted = np.zeros(n)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        f = footprints(grid, aim_xyz[sl], t[sl], sigma[sl], power[sl])
        flux += f.sum(axis=0)
        intercepted[sl] = f.sum(axis=1) * grid.cell_area
    return flux, intercepted
