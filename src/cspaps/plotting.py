"""Figures. Every function returns the matplotlib Figure and saves it when ``path`` is given."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Receiver  # noqa: E402
from .field import Field  # noqa: E402
from .receiver import ReceiverGrid  # noqa: E402

STRATEGY_STYLE = {
    "center": ("#8c8c8c", "No APS (all aim at centre)"),
    "aps_reactive": ("#d95f02", "APS, current DNI only"),
    "aps_predictive": ("#1b9e77", "APS + DNI forecast"),
}


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_layout(fld: Field, rec: Receiver, path: str | Path | None = None, values: np.ndarray | None = None,
                label: str = "", boundary=None, title: str = "", cmap: str = "viridis", vmin=None, vmax=None,
                exclusion: float = 0.0):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    size = max(1.5, min(8.0, 6000.0 / max(fld.n, 1)))
    if values is None:
        ax.scatter(fld.xyz[:, 0], fld.xyz[:, 1], s=size, c="tab:blue")
    else:
        sc = ax.scatter(fld.xyz[:, 0], fld.xyz[:, 1], s=size, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
    if boundary is not None:
        b = np.asarray(list(boundary) + [boundary[0]], dtype=float)
        ax.plot(b[:, 0], b[:, 1], "k--", lw=0.8)
    if exclusion and exclusion > 0:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rec.center[0] + exclusion * np.sin(th), rec.center[1] + exclusion * np.cos(th), ":", color="tab:red", lw=0.8)
    ax.plot(rec.center[0], rec.center[1], marker="^", color="tab:red", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_title(title or f"Heliostat field, N = {fld.n}")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def _flux_axes(ax, fig, flux_w_m2, grid, rec, vmax=None, show_limit=True):
    kw = flux_w_m2 / 1e3
    dz = grid.z[1] - grid.z[0] if grid.z.size > 1 else rec.height
    extent = [0.0, 360.0, grid.z[0] - dz / 2, grid.z[-1] + dz / 2]
    im = ax.imshow(kw, origin="lower", aspect="auto", extent=extent, cmap="inferno", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Incident flux (kW/m²)")
    ratio = grid.to_map(flux_w_m2.ravel() / grid.limit)
    if show_limit and ratio.max() > 1.0:
        az_g, z_g = np.meshgrid(grid.azimuth_deg, grid.z)
        ax.contour(az_g, z_g, ratio, levels=[1.0], colors="cyan", linewidths=1)
    if rec.n_panels > 1:
        from .receiver import panel_bounds

        for a0, _ in panel_bounds(rec):
            ax.axvline(a0 % 360.0, color="white", lw=0.4, alpha=0.5)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    ax.set_xlabel("Receiver azimuth")
    ax.set_ylabel("Height z (m)")


def plot_flux_map(flux_w_m2: np.ndarray, grid: ReceiverGrid, rec: Receiver, path: str | Path | None = None,
                  title: str = "", aims: tuple[np.ndarray, np.ndarray] | None = None, hotspot=None):
    """Receiver flux unrolled: x = compass azimuth on the cylinder, y = height.

    Cyan contour = over the (panel) limit; white lines = panel edges; the
    marker shows the verified hotspot (fine-grid check) when given.
    """
    fig, ax = plt.subplots(figsize=(10, 3.8))
    _flux_axes(ax, fig, flux_w_m2, grid, rec)
    if aims is not None:
        ax.scatter(np.asarray(aims[0]) % 360.0, aims[1], s=2, c="#4fc3f7", alpha=0.35, lw=0)
    if hotspot is not None:
        ax.plot(hotspot.hotspot_azimuth_deg, hotspot.hotspot_z, marker="x", color="cyan", ms=9, mew=2)
    peak = flux_w_m2.max() / 1e3
    ax.set_title(title or f"Peak {peak:.0f} kW/m² (cyan line = over limit)", fontsize=9)
    return _finish(fig, path)


def plot_flux_compare(flux_a: np.ndarray, flux_b: np.ndarray, grid: ReceiverGrid, rec: Receiver,
                      path: str | Path | None = None, labels=("Analytic (HFLCAL)", "MCRT")):
    vmax = max(flux_a.max(), flux_b.max()) / 1e3
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, f, lab in zip(axes[:2], (flux_a, flux_b), labels):
        _flux_axes(ax, fig, f, grid, rec, vmax=vmax, show_limit=False)
        ax.set_title(f"{lab}: peak {f.max() / 1e3:.0f} kW/m²")
    diff = (flux_a - flux_b) / 1e3
    lim = np.abs(diff).max() or 1.0
    dz = grid.z[1] - grid.z[0] if grid.z.size > 1 else rec.height
    im = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                        extent=[0, 360, grid.z[0] - dz / 2, grid.z[-1] + dz / 2])
    fig.colorbar(im, ax=axes[2], label="Difference (kW/m²)")
    axes[2].set_title(f"{labels[0]} minus {labels[1]}")
    axes[2].set_xticks(np.arange(0, 361, 45))
    axes[2].set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    axes[2].set_ylabel("Height z (m)")
    return _finish(fig, path)


def plot_realtime(table, path: str | Path | None = None):
    """DNI, power and verified peak-to-limit over the replay (sample-and-hold steps)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    first = table[table["strategy"] == table["strategy"].iloc[0]]
    axes[0].plot(first["time"], first["dni_mean_w_m2"], color="k", label="mean DNI on field")
    axes[0].plot(first["time"], first["dni_min_w_m2"], color="k", ls=":", label="min DNI on field")
    axes[0].set_ylabel("DNI (W/m²)")
    axes[0].legend(loc="lower left", fontsize=8)
    for strat, g in table.groupby("strategy", sort=False):
        color, label = STRATEGY_STYLE.get(strat, (None, strat))
        axes[1].step(g["time"], g["p_intercepted_mw"], where="post", color=color, label=label)
        axes[2].step(g["time"], g["peak_to_limit"], where="post", color=color, label=label)
    axes[1].set_ylabel("Intercepted power (MW)")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1.0, color="red", ls="--", lw=1, label="panel flux limit")
    axes[2].set_ylabel("Verified peak / limit")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Local time")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_sun_bins(hourly, bins, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(hourly.azimuth, hourly.elevation, s=3, c=hourly.dni, cmap="YlOrRd", alpha=0.6)
    fig.colorbar(sc, ax=ax, label="DNI of hourly samples (W/m²)")
    w = bins.dni * bins.weight_h
    ax.scatter(bins.azimuth, bins.elevation, s=40 + 400 * w / w.max(), facecolors="none",
               edgecolors="tab:blue", lw=1.5, label=f"{len(bins)} representative sun positions")
    ax.set_xlabel("Sun azimuth (deg, N = 0, E = 90)")
    ax.set_ylabel("Sun elevation (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_history(history, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    h = history.sort_values("evaluation") if "evaluation" in history else history
    e = h["energy_mwh"].to_numpy()
    idx = np.arange(1, e.size + 1)
    if "stage" in h:
        stage = h["stage"].astype(str).to_numpy()
        styles = {"global": ("tab:blue", "global search (differential evolution)"),
                  "local": ("tab:green", "local refinement"),
                  "initial": ("tab:purple", "initial design (Sobol)"),
                  "bayes": ("tab:orange", "Bayesian proposals (expected improvement)")}
        for key, (color, label) in styles.items():
            sel = stage == key
            if sel.any():
                ax.plot(idx[sel], e[sel] / 1e3, ".", color=color, alpha=0.7, label=label)
        other = ~np.isin(stage, list(styles))
        if other.any():
            ax.plot(idx[other], e[other] / 1e3, ".", color="tab:gray", alpha=0.6, label="other")
    else:
        ax.plot(idx, e / 1e3, ".", alpha=0.6, label="evaluated layout")
    ax.plot(idx, np.maximum.accumulate(e) / 1e3, "-", color="tab:red", label="best so far")
    if e.size:
        best = e.max() / 1e3
        ax.set_ylim(max(0.0, best * 0.7), best * 1.01)
    ax.set_xlabel("Evaluation")
    formal = "objective_basis" in h and h["objective_basis"].astype(str).str.contains("formal V").any()
    ax.set_ylabel("Annual value V, low fidelity (GWh)" if formal else "Annual energy on receiver (GWh)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, path)


def plot_annual_monthly(timeseries, path: str | Path | None = None):
    ts = timeseries.dropna(subset=["time"])
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(ts):
        month = ts["time"].map(lambda t: t.month)
        e = (ts["p_receiver_mw"] * ts["weight_h"]).groupby(month).sum() / 1e3
        ax.bar(e.index, e.values, color="tab:orange")
        ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy on receiver (GWh)")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, path)


def plot_uniformity_curve(curve: list[dict], path: str | Path | None = None):
    """Absorbed power vs allowed CV_2D, markers by status."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    eps = np.array([c["epsilon"] for c in curve])
    pw = np.array([c["power_abs_mw"] for c in curve])
    ok = np.array([c["status"] == "best_known_feasible" for c in curve])
    ax.plot(eps, pw, "-", color="0.6", lw=1)
    ax.plot(eps[ok], pw[ok], "o", color="tab:blue", label="best known feasible (heuristic)")
    if (~ok).any():
        ax.plot(eps[~ok], pw[~ok], "x", color="tab:red", ms=8, label="target not reached (most uniform plan shown)")
    ex = [(c["epsilon"], c.get("exact_power_abs_mw"), c.get("exact_status")) for c in curve if c.get("exact_status")]
    for e, p, st in ex:
        if st == "proven_optimal":
            ax.plot(e, p, "s", mfc="none", color="tab:green", ms=9)
        elif st == "time_limited_feasible" and p is not None:
            ax.plot(e, p, "D", mfc="none", color="tab:orange", ms=8)
        elif st and st.startswith("infeasible"):
            ax.plot(e, 0, "v", color="tab:red", ms=8)
    if ex:
        ax.plot([], [], "s", mfc="none", color="tab:green", label="proven optimal (SCIP)")
        ax.plot([], [], "D", mfc="none", color="tab:orange", label="time-limited feasible (SCIP)")
        ax.plot([], [], "v", color="tab:red", label="proven infeasible except shutdown")
    ax.set_xlabel("Allowed 2-D CV of absorbed flux on H (epsilon)")
    ax.set_ylabel("Absorbed power (MW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_cloud_compare(xyz: np.ndarray, rec: Receiver, grid: ReceiverGrid, states: list[dict],
                       path: str | Path | None = None, boundary=None):
    """Field DNI (top row) and receiver flux with aim points (bottom row) for each cloud-compare state."""
    n = len(states)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.6), squeeze=False)
    vmax_d = max(float(np.max(s["dni"])) for s in states) or 1.0
    vmax_f = max(float(np.max(s["flux"])) for s in states) / 1e3 or 1.0
    for k, s in enumerate(states):
        ax = axes[0, k]
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], s=max(1.5, min(7.0, 3000.0 / max(len(xyz), 1))), c=s["dni"],
                        cmap="cividis", vmin=0, vmax=vmax_d)
        off = ~np.asarray(s["active"], bool)
        if off.any():
            ax.scatter(xyz[off, 0], xyz[off, 1], s=10, marker="x", c="red", lw=0.8)
        if boundary is not None:
            b = np.asarray(list(boundary) + [boundary[0]], dtype=float)
            ax.plot(b[:, 0], b[:, 1], "k--", lw=0.6)
        ax.plot(rec.center[0], rec.center[1], marker="^", color="tab:red", ms=7)
        ax.set_aspect("equal")
        ax.set_title(f"{s.get('label_en', s['label'])}\nmean DNI {np.mean(s['dni']):.0f} W/m²", fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="DNI (W/m²)")
        ax = axes[1, k]
        _flux_axes(ax, fig, s["flux"], grid, rec, vmax=vmax_f)
        on = np.asarray(s["active"], bool)
        ax.scatter(np.asarray(s["aim_azimuth"])[on] % 360.0, np.asarray(s["aim_z"])[on], s=3, c="#4fc3f7",
                   alpha=0.5, lw=0)
        ax.set_title(f"P_abs {s['p_abs_mw']:.2f} MW, U {s['u_mw']:.2f} MW, peak {s['peak_flux_kw_m2']:.0f} kW/m²",
                     fontsize=8.5)
    fig.tight_layout()
    return _finish(fig, path)


def plot_dni_maps(maps: dict, path: str | Path | None = None, boundary=None, tower=(0.0, 0.0)):
    """Site DNI maps (factor or W/m2) side by side."""
    keys = list(maps)
    n = max(len(keys), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for k, key in enumerate(keys):
        m = maps[key]
        ax = axes[0, k]
        im = ax.pcolormesh(m.x, m.y, m.values, shading="nearest", cmap="cividis")
        if boundary is not None:
            b = np.asarray(list(boundary) + [boundary[0]], dtype=float)
            ax.plot(b[:, 0], b[:, 1], "w--", lw=0.7)
        ax.plot(tower[0], tower[1], marker="^", color="tab:red", ms=7)
        ax.set_aspect("equal")
        ax.set_title(f"{key} ({m.unit})", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    return _finish(fig, path)
