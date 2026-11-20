"""Interactive (plotly) figures for the GUI. Same conventions as plotting.py."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .config import Case
from .receiver import ReceiverGrid, panel_bounds

# Inferno with a dark-violet first stop: Streamlit's front end recolours the exact
# '#000004' of the stock scale, which made zero flux look lavender.
FLUX_SCALE = [[0.0, "rgb(10,7,32)"], [0.111, "#1b0c41"], [0.222, "#4a0c6b"], [0.333, "#781c6d"],
              [0.444, "#a52c60"], [0.556, "#cf4446"], [0.667, "#ed6925"], [0.778, "#fb9b06"],
              [0.889, "#f7d13d"], [1.0, "#fcffa4"]]


def _base_layout(fig, title: str, height: int = 520):
    fig.update_layout(title=dict(text=title, font=dict(size=14), y=0.98), height=height,
                      margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, font=dict(size=11)))
    return fig


def field_map(case: Case, xyz: np.ndarray, ids: np.ndarray, values=None, label: str = "", colorscale="Viridis",
              cmin=None, cmax=None, selected: int | None = None, title: str = "", hover: list[str] | None = None,
              candidates: np.ndarray | None = None, inactive: np.ndarray | None = None):
    rec, spec = case.receiver, case.field
    fig = go.Figure()
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color="black", dash="dash", width=1),
                                 name="site boundary", hoverinfo="skip"))
    if spec.exclusion_radius > 0:
        th = np.linspace(0, 2 * np.pi, 180)
        fig.add_trace(go.Scatter(x=rec.center[0] + spec.exclusion_radius * np.sin(th),
                                 y=rec.center[1] + spec.exclusion_radius * np.cos(th), mode="lines",
                                 line=dict(color="red", dash="dot", width=1), name="exclusion zone", hoverinfo="skip"))
    if candidates is not None and len(candidates):
        fig.add_trace(go.Scattergl(x=candidates[:, 0], y=candidates[:, 1], mode="markers",
                                   marker=dict(size=3, color="lightgray"), name="candidate positions", hoverinfo="skip"))
    size = float(np.clip(9000.0 / max(len(xyz), 1), 3, 9))
    marker = dict(size=size)
    if values is not None:
        marker.update(color=values, colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
                      colorbar=dict(title=label, thickness=12))
    else:
        marker.update(color="#1f77b4")
    text = hover or [f"ID {i}" for i in ids]
    fig.add_trace(go.Scattergl(x=xyz[:, 0], y=xyz[:, 1], mode="markers", marker=marker, customdata=ids,
                               text=text, hovertemplate="%{text}<extra></extra>", name="heliostats"))
    if inactive is not None and np.any(inactive):
        k = np.flatnonzero(inactive)
        fig.add_trace(go.Scattergl(x=xyz[k, 0], y=xyz[k, 1], mode="markers", customdata=ids[k],
                                   marker=dict(size=size + 4, symbol="x-thin", line=dict(width=2, color="red")),
                                   name="defocused", hoverinfo="skip"))
    if selected is not None:
        k = np.flatnonzero(ids == selected)
        if k.size:
            fig.add_trace(go.Scatter(x=xyz[k, 0], y=xyz[k, 1], mode="markers", name=f"ID {selected}",
                                     marker=dict(size=size + 10, color="rgba(0,0,0,0)", line=dict(width=2, color="magenta")),
                                     hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="tower",
                             marker=dict(symbol="triangle-up", size=13, color="red"), hoverinfo="skip"))
    fig.update_xaxes(title="East x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="North y (m)")
    return _base_layout(fig, title)


def receiver_map(case: Case, grid: ReceiverGrid, flux_w_m2: np.ndarray, title: str = "", zmax=None,
                 hotspot: dict | None = None, aims=None, selected_cell: tuple[float, float] | None = None,
                 height: int = 380):
    kw = np.asarray(flux_w_m2) / 1e3
    ratio = grid.to_map(np.asarray(flux_w_m2).ravel() / grid.limit)
    limit_kw = grid.to_map(grid.limit) / 1e3
    panel = grid.to_map(grid.panel) + 1
    custom = np.dstack([limit_kw, ratio, panel])
    fig = go.Figure(go.Heatmap(
        x=grid.azimuth_deg, y=grid.z, z=kw, colorscale=FLUX_SCALE, zmin=0, zmax=zmax, customdata=custom,
        colorbar=dict(title="kW/m²", thickness=12),
        hovertemplate="azimuth %{x:.1f}°, z %{y:.2f} m<br>flux %{z:.0f} kW/m²<br>limit %{customdata[0]:.0f} "
                      "(ratio %{customdata[1]:.2f}), panel %{customdata[2]:.0f}<extra></extra>"))
    if ratio.max() > 1.0:
        fig.add_trace(go.Contour(x=grid.azimuth_deg, y=grid.z, z=ratio, showscale=False, hoverinfo="skip",
                                 contours=dict(start=1.0, end=1.0, size=1, coloring="none"),
                                 line=dict(color="cyan", width=2), name="over limit"))
    if case.receiver.n_panels > 1:
        for a0, _ in panel_bounds(case.receiver):
            fig.add_vline(x=a0 % 360.0, line=dict(color="white", width=1, dash="dot"))
    if aims is not None:
        fig.add_trace(go.Scattergl(x=np.asarray(aims[0]) % 360.0, y=aims[1], mode="markers", hoverinfo="skip",
                                   marker=dict(size=3, color="#4fc3f7", opacity=0.4), name="aim points"))
    if hotspot:
        h = hotspot.get("hotspot", hotspot)
        fig.add_trace(go.Scatter(x=[h["azimuth_deg"]], y=[h["z_m"]], mode="markers", name="verified hotspot",
                                 marker=dict(symbol="x", size=14, color="cyan", line=dict(width=2)), hoverinfo="skip"))
    if selected_cell is not None:
        fig.add_trace(go.Scatter(x=[selected_cell[0]], y=[selected_cell[1]], mode="markers", name="selected",
                                 marker=dict(symbol="circle-open", size=16, color="magenta", line=dict(width=2)),
                                 hoverinfo="skip"))
    fig.update_xaxes(title="receiver azimuth (deg, N = 0, E = 90)", range=[0, 360],
                     tickvals=list(range(0, 361, 45)), ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    fig.update_yaxes(title="height z (m)")
    return _base_layout(fig, title, height)


def replay_timeline(table, metric: str, label: str, selected_time=None, limit_line: float | None = None):
    fig = go.Figure()
    colors = {"center": "#8c8c8c", "aps_reactive": "#d95f02", "aps_predictive": "#1b9e77"}
    names = {"center": "不做 APS", "aps_reactive": "APS（仅当前 DNI）", "aps_predictive": "APS + DNI 预报"}
    for strat, g in table.groupby("strategy", sort=False):
        fig.add_trace(go.Scatter(x=g["time"], y=g[metric], mode="lines+markers", line_shape="hv",
                                 name=names.get(strat, strat), line=dict(color=colors.get(strat))))
    if limit_line is not None:
        fig.add_hline(y=limit_line, line=dict(color="red", dash="dash"))
    if selected_time is not None:
        fig.add_vline(x=selected_time, line=dict(color="magenta", width=1))
    fig.update_yaxes(title=label)
    return _base_layout(fig, "", 260)


def uniformity_curves(curves: dict[str, list[dict]]):
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_trace(go.Scatter(x=[c["epsilon"] for c in curve], y=[c["power_abs_mw"] for c in curve],
                                 mode="lines+markers", name=name,
                                 text=[c["status"] for c in curve], hovertemplate="ε %{x:.3f}<br>%{y:.3f} MW<br>%{text}"))
    fig.update_xaxes(title="允许的二维 CV (ε)")
    fig.update_yaxes(title="吸收功率 (MW)")
    return _base_layout(fig, "功率与均匀度", 360)
