"""Figures: field layout and unrolled receiver flux map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Receiver  # noqa: E402
from .field import Field  # noqa: E402
from .receiver import ReceiverGrid  # noqa: E402


def plot_layout(
    fld: Field,
    rec: Receiver,
    path: str | Path,
    values: np.ndarray | None = None,
    label: str = "",
    boundary=None,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    if values is None:
        ax.scatter(fld.xyz[:, 0], fld.xyz[:, 1], s=3, c="tab:blue")
    else:
        sc = ax.scatter(fld.xyz[:, 0], fld.xyz[:, 1], s=4, c=values, cmap="viridis")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
    if boundary is not None:
        b = np.asarray(boundary + [boundary[0]], dtype=float)
        ax.plot(b[:, 0], b[:, 1], "k--", lw=0.8)
    ax.plot(rec.center[0], rec.center[1], marker="^", color="tab:red", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_title(title or f"Heliostat field, N = {fld.n}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flux_map(
    flux_w_m2: np.ndarray,
    grid: ReceiverGrid,
    rec: Receiver,
    path: str | Path,
    title: str = "",
) -> None:
    """Receiver flux unrolled: x = compass azimuth on the cylinder, y = height."""
    kw = flux_w_m2 / 1e3
    d_az = grid.azimuth_deg[1] - grid.azimuth_deg[0]
    dz = grid.z[1] - grid.z[0] if grid.z.size > 1 else rec.height
    extent = [0.0, 360.0, grid.z[0] - dz / 2, grid.z[-1] + dz / 2]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(kw, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    fig.colorbar(im, ax=ax, label="Flux (kW/m²)")
    if kw.max() > rec.max_flux_kw_m2:
        az_g, z_g = np.meshgrid(grid.azimuth_deg, grid.z)
        ax.contour(az_g, z_g, kw, levels=[rec.max_flux_kw_m2], colors="cyan", linewidths=1)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    ax.set_xlabel(f"Receiver azimuth (cell {d_az:.1f}°)")
    ax.set_ylabel("Height z (m)")
    peak = kw.max()
    ax.set_title(
        title or f"Peak {peak:.0f} kW/m² (limit {rec.max_flux_kw_m2:.0f}, cyan = over limit)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
