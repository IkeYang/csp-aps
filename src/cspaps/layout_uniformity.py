"""Receiver uniformity that the *layout* decides (problem 2).

Physics. A heliostat at compass azimuth phi_i (seen from the receiver axis)
lights only the half of the cylinder that faces it: in its image plane the
cylinder is a band of half-width R, and the surface point at azimuth phi lies
at the lateral coordinate u = R sin(phi - phi_i) (|phi - phi_i| < 90 deg).
With a lateral Gaussian beam of standard deviation sigma the share of the
intercepted power that lands between azimuths phi_a and phi_b is

    [Phi(R sin(phi_b - phi_i) / sigma) - Phi(R sin(phi_a - phi_i) / sigma)] / [Phi(R / sigma) - Phi(-R / sigma)].

The APS moves aim points along the height and only a few degrees around the
cylinder, so it can even out the flux *within* each azimuth column but cannot
move power from the north side of the receiver to the south side. The
column-mean flux (the circumferential profile) is therefore a property of the
layout. With equal-area columns j in the uniformity region H

    sigma_t^2 (2-D) = sigma_circ,t^2 (between columns) + mean within-column variance,

so the circumferential part is the share of the problem-1 penalty that no
aiming strategy can remove. Problem 2 with uniformity maximises

    J(X) = sum_t w_t [ P_abs,t(X) - lambda_L * A_H * sigma_circ,t(X) ]
         = sum_t w_t P_abs,t (1 - lambda_L CV_circ,t)        (H = whole receiver)

the same form as the weighted APS objective (P - lambda A_H sigma_H).

Selection. For a fixed candidate set (one generator parameter vector) the
choice of N positions is max_x J(x), sum x = N, x in {0, 1}; J is concave in
the relaxed x, so a Frank-Wolfe bound certifies the selection (top-N linear
oracle), and an exchange local search finds the selection.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from . import optics, sun
from .config import Case
from .receiver import ReceiverGrid, build_grid, default_aim, heliostat_azimuth, region_mask, surface_point


def column_shares(case: Case, xyz: np.ndarray, samples, grid: ReceiverGrid | None = None) -> np.ndarray:
    """(S, N, C) share of each heliostat's intercepted power per receiver azimuth column (centre aim)."""
    rec, hel = case.receiver, case.heliostat
    grid = grid or build_grid(rec)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    n, c = len(xyz), grid.azimuth_deg.size
    s_n = len(samples)
    out = np.zeros((s_n, n, c))
    if n == 0 or s_n == 0:
        return out
    phi = heliostat_azimuth(rec, xyz)
    width = 360.0 / c
    edges = (np.arange(c + 1) * width)  # column edges, deg (grid columns start at 0)
    rel = (edges[None, :] - phi[:, None] + 180.0) % 360.0 - 180.0  # (N, C+1) relative edge azimuths
    # column j spans rel[:, j] .. rel[:, j] + width (unwrapped)
    lo = rel[:, :-1]
    hi = lo + width
    lo_c = np.radians(np.clip(lo, -90.0, 90.0))
    hi_c = np.radians(np.clip(hi, -90.0, 90.0))
    u_lo, u_hi = rec.radius * np.sin(lo_c), rec.radius * np.sin(hi_c)  # (N, C)
    az, z = default_aim(rec, xyz)
    aim = surface_point(rec, az, z)
    for t in range(s_n):
        el, sa = float(samples.elevation[t]), float(samples.azimuth[t])
        if el <= 0:
            continue
        s = sun.sun_vector(el, sa)
        _, _, slant, cos_inc = optics.heliostat_geometry(xyz, aim, s)
        sig = np.maximum(optics.beam_sigma(slant, cos_inc, hel, case.sun), 1e-6)[:, None]
        f = ndtr(u_hi / sig) - ndtr(u_lo / sig)
        tot = f.sum(axis=1, keepdims=True)
        out[t] = np.where(tot > 0, f / np.maximum(tot, 1e-300), 0.0)
    return out


@dataclass
class CircModel:
    """Annual absorbed energy and circumferential penalty of a heliostat selection."""

    e: np.ndarray  # (S, N) absorbed power per state, W
    b: np.ndarray  # (S, N, C_H) column-mean absorbed flux contribution on H columns, W/m2
    w: np.ndarray  # (S,) state weights, h
    a_h: float  # area of H, m2
    lam: float  # lambda_L

    @property
    def n(self) -> int:
        return self.e.shape[1]

    @property
    def n_cols(self) -> int:
        return self.b.shape[2]

    def profile(self, x: np.ndarray) -> np.ndarray:
        """(S, C_H) column-mean absorbed flux for a 0/1 or fractional selection."""
        return np.einsum("snc,n->sc", self.b, np.asarray(x, dtype=float))

    def terms(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        p = self.e @ x  # (S,) absorbed power
        g = self.profile(x)
        std = g.std(axis=1)
        mean = g.mean(axis=1)
        cv = np.where(mean > 0, std / np.maximum(mean, 1e-300), np.nan)
        pen = self.a_h * std  # (S,) W
        e_mwh = float((self.w * p).sum() / 1e6)
        pen_mwh = float((self.w * pen).sum() / 1e6)
        wp = self.w * p
        cv_w = float(np.nansum(wp * cv) / wp.sum()) if wp.sum() > 0 else float("nan")
        return {"energy_mwh": e_mwh, "spread_mwh": pen_mwh, "objective_mwh": e_mwh - self.lam * pen_mwh,
                "cv_circ_weighted": cv_w, "cv_circ_states": cv, "power_w": p, "profile": g}

    def objective(self, x: np.ndarray) -> float:
        return self.terms(x)["objective_mwh"]


def build_model(case: Case, xyz: np.ndarray, samples, state_power_w: np.ndarray, lam: float,
                grid: ReceiverGrid | None = None) -> CircModel:
    """Model from per-state intercepted power (``annual(..., keep_state_power=True)``)."""
    rec = case.receiver
    grid = grid or build_grid(rec)
    shares = column_shares(case, xyz, samples, grid)  # (S, N, C)
    col_az = grid.azimuth_deg
    alpha = grid.alpha.reshape(grid.shape)[0]  # absorptivity per column (panels are azimuth sectors)
    h_cols = region_mask(rec, case.uniformity, col_az, np.full(col_az.size, np.mean(case.uniformity.z_range) if case.uniformity.z_range else rec.center[2])) \
        if case.uniformity.panels is not None else np.ones(col_az.size, dtype=bool)
    col_area = rec.radius * np.radians(360.0 / col_az.size) * rec.height
    p = np.asarray(state_power_w, dtype=float)  # (S, N) intercepted
    absorbed = p[:, :, None] * shares * alpha[None, None, :]  # (S, N, C) W
    e = absorbed.sum(axis=2)
    b = absorbed[:, :, h_cols] / col_area
    # every column takes the mean of its sector: the std over equal-area columns is then the area-weighted
    # std of the sector means (the between-sector part that the layout decides)
    sec = sector_of_columns(case, col_az)[h_cols]
    avg = (sec[:, None] == sec[None, :]).astype(float)
    avg /= avg.sum(axis=0, keepdims=True)
    b = np.einsum("snc,cd->snd", b, avg)
    return CircModel(e, b, np.asarray(samples.weight_h, dtype=float), float(col_area * h_cols.sum()), float(lam))


def sector_of_columns(case: Case, col_az: np.ndarray) -> np.ndarray:
    """Sector index of each receiver column for the layout uniformity (panels, or equal sectors)."""
    from .receiver import panel_of

    rec = case.receiver
    n = case.layout_opt.uniformity_sectors
    if n <= 0 and rec.n_panels >= 6:
        return panel_of(rec, col_az)
    n = n if n > 0 else 12
    return (((np.asarray(col_az) - rec.panel_start_azimuth_deg) % 360.0) // (360.0 / n)).astype(int)


def _swap_gains(model: CircModel, sel: np.ndarray, g: np.ndarray):
    """Marginal objective change (MWh) of adding each unselected / removing each selected heliostat."""
    c_h = model.n_cols
    sb = model.b.sum(axis=2)  # (S, N)
    sb2 = (model.b ** 2).sum(axis=2)
    sg = g.sum(axis=1)[:, None]  # (S, 1)
    sg2 = (g ** 2).sum(axis=1)[:, None]
    gb = np.einsum("sc,snc->sn", g, model.b)
    std0 = np.sqrt(np.maximum(sg2 / c_h - (sg / c_h) ** 2, 0.0))
    sign = np.where(sel, -1.0, 1.0)[None, :]
    s1 = sg + sign * sb
    s2 = sg2 + 2 * sign * gb + sb2
    std1 = np.sqrt(np.maximum(s2 / c_h - (s1 / c_h) ** 2, 0.0))
    d = sign * model.e - model.lam * model.a_h * (std1 - std0)
    return (model.w[:, None] * d).sum(axis=0) / 1e6  # (N,)


def _linear_scores(model: CircModel, g: np.ndarray) -> np.ndarray:
    """Gradient of J (MWh per heliostat) at the profile g: energy minus the linearised penalty change."""
    d = g - g.mean(axis=1, keepdims=True)
    std = np.sqrt((d ** 2).mean(axis=1))
    coef = np.where(std > 0, model.lam * model.a_h / (model.n_cols * np.maximum(std, 1e-300)), 0.0)
    return (model.w[:, None] * (model.e - coef[:, None] * np.einsum("sc,snc->sn", d, model.b))).sum(axis=0) / 1e6


def _objective_fast(model: CircModel, p: np.ndarray, g: np.ndarray) -> float:
    """J (MWh) from the per-state absorbed power p (S,) and the column profile g (S, C_H)."""
    return float((model.w * (p - model.lam * model.a_h * g.std(axis=1))).sum() / 1e6)


def select(model: CircModel, n_select: int, init: np.ndarray | None = None, max_swaps: int | None = None,
           top: int = 8, polish: bool = False, polish_pool: int = 300) -> tuple[np.ndarray, dict]:
    """Swap local search for max J over subsets of size n_select.

    Energy only (lambda_L = 0): the N largest energies, or with ``init`` and
    ``max_swaps`` at most that many best-for-worst swaps from ``init``.
    Otherwise each round short-lists additions / removals by their single-move
    gain and by the linearised objective (in which the penalty change of a swap
    inside one sector cancels), evaluates the short-listed pairs exactly
    (incremental O(S C) per pair) and applies the best improving swap. With
    ``polish`` it then tries, for every selected heliostat, every one of the
    ``polish_pool`` best-ranked replacements exactly, until no swap improves.
    """
    t0 = _time.perf_counter()
    n = model.n
    n_select = min(int(n_select), n)
    energy = (model.w[:, None] * model.e).sum(axis=0)
    top_n = np.zeros(n, dtype=bool)
    top_n[np.argsort(-energy, kind="stable")[:n_select]] = True
    if n_select in (0, n) or (model.lam <= 0 and (init is None or max_swaps is None)):
        return top_n, {"swaps": 0, "time_s": round(_time.perf_counter() - t0, 3)}
    sel = top_n if init is None else np.asarray(init, dtype=bool).copy()
    max_swaps = max_swaps if max_swaps is not None else 4 * n_select
    if model.lam <= 0:  # damped energy-only step: best unused for worst used, at most max_swaps
        add = np.flatnonzero(~sel)
        rem = np.flatnonzero(sel)
        add = add[np.argsort(-energy[add], kind="stable")]
        rem = rem[np.argsort(energy[rem], kind="stable")]
        k = 0
        while k < min(max_swaps, add.size, rem.size) and energy[add[k]] > energy[rem[k]] * (1 + 1e-12):
            sel[add[k]], sel[rem[k]] = True, False
            k += 1
        return sel, {"swaps": k, "time_s": round(_time.perf_counter() - t0, 3)}
    p = model.e @ sel.astype(float)
    g = model.profile(sel)
    cur = _objective_fast(model, p, g)
    swaps = 0

    def apply(a: int, r: int) -> None:
        nonlocal p, g, swaps
        sel[a], sel[r] = True, False
        p = p + model.e[:, a] - model.e[:, r]
        g = g + model.b[:, a] - model.b[:, r]
        swaps += 1

    while swaps < max_swaps:
        gains = _swap_gains(model, sel, g)
        lin = _linear_scores(model, g)
        add_all = np.flatnonzero(~sel)
        rem_all = np.flatnonzero(sel)
        # screen with single-move gains and with the linearised objective, then evaluate pairs exactly
        add = np.union1d(add_all[np.argsort(-gains[add_all])[:top]], add_all[np.argsort(-lin[add_all])[:top]])
        rem = np.union1d(rem_all[np.argsort(-gains[rem_all])[:top]], rem_all[np.argsort(lin[rem_all])[:top]])
        best, pair = 1e-9 * max(abs(cur), 1.0), None
        for a in add:
            pa, ga = p + model.e[:, a], g + model.b[:, a]
            for r in rem:
                v = _objective_fast(model, pa - model.e[:, r], ga - model.b[:, r]) - cur
                if v > best:
                    best, pair = v, (a, r)
        if pair is None:
            if not polish:
                break
            # full pass: for each selected heliostat (worst linear score first) the best replacement
            lin = _linear_scores(model, g)
            pool = np.flatnonzero(~sel)
            pool = pool[np.argsort(-lin[pool])[:polish_pool]]
            improved = False
            for r in np.flatnonzero(sel)[np.argsort(lin[sel])]:
                pool = pool[~sel[pool]]
                if pool.size == 0 or swaps >= max_swaps:
                    break
                pp = (p - model.e[:, r])[None, :] + model.e[:, pool].T  # (A, S)
                gg = (g - model.b[:, r])[None] + np.moveaxis(model.b[:, pool], 1, 0)  # (A, S, C)
                j = (model.w[None, :] * (pp - model.lam * model.a_h * gg.std(axis=2))).sum(axis=1) / 1e6
                k = int(np.argmax(j))
                if j[k] > cur + 1e-9 * max(abs(cur), 1.0):
                    apply(int(pool[k]), int(r))
                    cur = float(j[k])
                    improved = True
            if not improved:
                break
            continue
        apply(*pair)
        cur += best
    return sel, {"swaps": swaps, "time_s": round(_time.perf_counter() - t0, 3)}


def selection_bound(model: CircModel, n_select: int, sel: np.ndarray, max_iter: int = 200,
                    time_limit_s: float = 10.0) -> dict:
    """Frank-Wolfe upper bound on max J over subsets of size n_select of this candidate set."""
    t0 = _time.perf_counter()
    n_select = min(int(n_select), model.n)
    x = np.asarray(sel, dtype=float).copy()
    found = model.objective(sel)
    best = math.inf
    c_h = model.n_cols
    it = 0
    for it in range(1, max_iter + 1):
        g = model.profile(x)
        mean = g.mean(axis=1, keepdims=True)
        d = g - mean
        std = np.sqrt((d ** 2).mean(axis=1))
        val = float((model.w * (model.e @ x - model.lam * model.a_h * std)).sum() / 1e6)
        coef = np.where(std > 0, model.lam * model.a_h / (c_h * np.maximum(std, 1e-300)), 0.0)  # (S,)
        grad = (model.w[:, None] * (model.e - coef[:, None] * np.einsum("sc,snc->sn", d, model.b))).sum(axis=0) / 1e6
        top = np.argsort(-grad)[:n_select]
        y = np.zeros_like(x)
        y[top] = 1.0
        ub = val + float(grad @ (y - x))
        best = min(best, ub)
        if best - found <= 1e-5 * max(abs(best), 1.0) or _time.perf_counter() - t0 > time_limit_s:
            break
        # exact line search on the concave 1-D function (ternary search)
        dvec = y - x
        lo, hi = 0.0, 1.0
        for _ in range(40):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if model.objective(x + m1 * dvec) < model.objective(x + m2 * dvec):
                lo = m1
            else:
                hi = m2
        x = x + 0.5 * (lo + hi) * dvec
    gap = max(0.0, 1.0 - found / best) if best > 0 else None
    return {"objective_found_mwh": found, "upper_bound_mwh": best, "gap": gap, "iterations": it,
            "time_s": round(_time.perf_counter() - t0, 3),
            "model": "candidate positions of this generator parameter vector, centre aim, shading in the candidate field"}


def detached_heliostats(xyz: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Indices of heliostats outside the largest cluster (link length = factor x median neighbour distance)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    xy = np.asarray(xyz, dtype=float).reshape(-1, 3)[:, :2]
    if len(xy) < 3:
        return np.zeros(0, dtype=int)
    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]
    graph = tree.sparse_distance_matrix(tree, factor * float(np.median(nn)), output_type="coo_matrix")
    _, label = connected_components(graph, directed=False)
    main = np.argmax(np.bincount(label))
    return np.flatnonzero(label != main)


def layout_metrics(case: Case, xyz: np.ndarray, samples, state_power_w: np.ndarray, lam: float,
                   grid: ReceiverGrid | None = None, n_sectors: int = 12) -> dict:
    """Energy, circumferential penalty / CV and the field balance around the tower for one layout."""
    rec = case.receiver
    model = build_model(case, xyz, samples, state_power_w, lam, grid)
    t = model.terms(np.ones(model.n))
    e_ann = (model.w[:, None] * model.e).sum(axis=0) / 1e6  # (N,) MWh
    rel = np.asarray(xyz)[:, :2] - np.asarray(rec.center[:2])
    tot = e_ann.sum()
    centroid = (rel * e_ann[:, None]).sum(axis=0) / tot if tot > 0 else np.zeros(2)
    r_mean = float(np.hypot(rel[:, 0], rel[:, 1]).mean()) if len(rel) else 0.0
    g_ann = (model.w[:, None] * t["profile"]).sum(axis=0) / max(model.w.sum(), 1e-12)  # time-mean column flux
    grid = grid or build_grid(rec)
    col_az = grid.azimuth_deg
    if case.uniformity.panels is not None:
        col_az = col_az[region_mask(rec, case.uniformity, col_az, np.full(col_az.size, np.mean(case.uniformity.z_range) if case.uniformity.z_range else rec.center[2]))]
    sec = (np.floor(((col_az + 180.0 / n_sectors) % 360.0) / (360.0 / n_sectors))).astype(int)
    sector_energy = np.bincount(sec, weights=g_ann, minlength=n_sectors)
    share = sector_energy / sector_energy.sum() if sector_energy.sum() > 0 else sector_energy
    phi = heliostat_azimuth(rec, np.asarray(xyz))
    north = int(((phi >= 270) | (phi < 90)).sum())
    detached = detached_heliostats(np.asarray(xyz))
    return {
        "uniformity_weight": lam,
        "energy_mwh": t["energy_mwh"], "circ_spread_mwh": t["spread_mwh"], "objective_mwh": t["objective_mwh"],
        "cv_circ_weighted": t["cv_circ_weighted"],
        "cv_circ_annual": float(g_ann.std() / g_ann.mean()) if g_ann.mean() > 0 else None,
        "centroid_offset_m": float(np.hypot(*centroid)), "centroid_xy_m": [float(centroid[0]), float(centroid[1])],
        "centroid_offset_rel": float(np.hypot(*centroid) / r_mean) if r_mean > 0 else None,
        "north_fraction": north / max(len(phi), 1),
        "n_detached": int(detached.size), "detached_xy_m": np.asarray(xyz)[detached, :2].round(2).tolist(),
        "sector_share": [float(v) for v in share],
        "sector_center_deg": [float(k * 360.0 / n_sectors) for k in range(n_sectors)],
    }
