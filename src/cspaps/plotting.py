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
                label: str = "", boundary=None, title: str = "", cmap: str = "viridis", vmin=None, vmax=None):
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
    fig.colorbar(im, ax=ax, label="Flux (kW/m²)")
    if show_limit and kw.max() > rec.max_flux_kw_m2:
        az_g, z_g = np.meshgrid(grid.azimuth_deg, grid.z)
        ax.contour(az_g, z_g, kw, levels=[rec.max_flux_kw_m2], colors="cyan", linewidths=1)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    ax.set_xlabel("Receiver azimuth")
    ax.set_ylabel("Height z (m)")


def plot_flux_map(flux_w_m2: np.ndarray, grid: ReceiverGrid, rec: Receiver, path: str | Path | None = None,
                  title: str = "", aims: tuple[np.ndarray, np.ndarray] | None = None):
    """Receiver flux unrolled: x = compass azimuth on the cylinder, y = height."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    _flux_axes(ax, fig, flux_w_m2, grid, rec)
    if aims is not None:
        ax.scatter(np.asarray(aims[0]) % 360.0, aims[1], s=2, c="#4fc3f7", alpha=0.35, lw=0)
    peak = flux_w_m2.max() / 1e3
    ax.set_title(title or f"Peak {peak:.0f} kW/m² (limit {rec.max_flux_kw_m2:.0f}; cyan line = over limit)")
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


def plot_realtime(table, limit_kw_m2: float, path: str | Path | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    first = table[table["strategy"] == table["strategy"].iloc[0]]
    axes[0].plot(first["time"], first["dni_mean_w_m2"], color="k", label="mean DNI on field")
    axes[0].plot(first["time"], first["dni_min_w_m2"], color="k", ls=":", label="min DNI on field")
    axes[0].set_ylabel("DNI (W/m²)")
    axes[0].legend(loc="lower left", fontsize=8)
    for strat, g in table.groupby("strategy", sort=False):
        color, label = STRATEGY_STYLE.get(strat, (None, strat))
        axes[1].plot(g["time"], g["p_receiver_mw"], color=color, label=label)
        axes[2].plot(g["time"], g["peak_flux_kw_m2"], color=color, label=label)
    axes[1].set_ylabel("Power on receiver (MW)")
    axes[1].legend(fontsize=8)
    axes[2].axhline(limit_kw_m2, color="red", ls="--", lw=1, label="flux limit")
    axes[2].set_ylabel("Peak flux (kW/m²)")
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
        local = (h["stage"] == "local").to_numpy()
        ax.plot(idx[~local], e[~local] / 1e3, ".", alpha=0.6, label="global search (differential evolution)")
        ax.plot(idx[local], e[local] / 1e3, ".", color="tab:green", alpha=0.8, label="local refinement")
    else:
        ax.plot(idx, e / 1e3, ".", alpha=0.6, label="evaluated layout")
    ax.plot(idx, np.maximum.accumulate(e) / 1e3, "-", color="tab:red", label="best so far")
    if e.size:
        best = e.max() / 1e3
        ax.set_ylim(max(0.0, best * 0.7), best * 1.01)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Annual energy on receiver (GWh)")
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
