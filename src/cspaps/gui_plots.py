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


FONT = "Inter, 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif"
INK, MUTED, TEAL, AMBER, NAVY, RED = "#16232b", "#5d6d74", "#14806f", "#e09b2d", "#0f2a3a", "#c2453a"


def _base_layout(fig, title: str, height: int = 520):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK), y=0.98, x=0.01), height=height,
                      margin=dict(l=10, r=10, t=40 if title else 16, b=10), template="plotly_white",
                      font=dict(family=FONT, size=12, color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfcfc",
                      legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, font=dict(size=11)))
    return fig


def _tower_shadow(fig, case, xyz, sun_angles, *, scene=False):
    """Silhouette of the finite tower cylinder at the nominal mirror pivot height (ENU)."""
    if sun_angles is None or not (case.shading.enabled and case.shading.tower_shadow):
        return
    elevation, azimuth = np.radians(sun_angles)
    rec = case.receiver
    z = case.heliostat.pivot_height
    radius = rec.tower_radius
    if not 0 < elevation <= np.pi / 2 or radius <= 0 or z >= rec.top:
        return
    center = np.asarray(rec.center[:2])
    away = -np.array([np.sin(azimuth), np.cos(azimuth)])
    cross = np.array([-away[1], away[0]])
    length = (rec.top - z) / np.tan(elevation)
    # Clip an exceptionally long sunrise/sunset shadow beyond the visible field.
    extent = np.vstack((center, np.asarray(xyz)[:, :2]))
    if case.field.boundary is not None and not scene:
        extent = np.vstack((extent, case.field.boundary))
    clip = float(((extent - center) @ away).max()) + 2 * radius
    theta = np.linspace(-np.pi / 2, np.pi / 2, 49)
    arc = radius * (np.cos(theta)[:, None] * away + np.sin(theta)[:, None] * cross)
    far = center + length * away + arc if length <= 2 * clip else center + clip * away + arc[[0, -1]]
    points = np.vstack((far, center - arc))
    x, y = points.T
    name = f"塔影（镜轴高度 {z:g} m）"
    if scene:
        idx = np.arange(1, len(x) - 1)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=np.full(len(x), z), i=np.zeros(len(idx), int), j=idx, k=idx + 1,
                                name=name, color="#594577", opacity=0.42, flatshading=True, showlegend=True,
                                hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter(x=np.r_[x, x[0]], y=np.r_[y, y[0]], mode="lines", fill="toself",
                                 line=dict(color="#66547e", width=1.5), fillcolor="rgba(99,76,130,.27)",
                                 name=name, hoverinfo="skip"))


def tower_shadow_fraction(case: Case, xyz, sun_elevation: float, sun_azimuth: float) -> np.ndarray:
    """Projected tower blockage of mirror apertures under the nominal centre aiming pose."""
    from . import optics, shading, sun
    from .receiver import default_aim, surface_point

    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if not len(xyz):
        return np.zeros(0)
    if not (case.shading.enabled and case.shading.tower_shadow and case.receiver.tower_radius > 0
            and sun_elevation > 0):
        return np.zeros(len(xyz))
    aim_az, aim_z = default_aim(case.receiver, xyz)
    direction = sun.sun_vector(sun_elevation, sun_azimuth)
    normals, *_ = optics.heliostat_geometry(xyz, surface_point(case.receiver, aim_az, aim_z), direction)
    points = shading.mirror_points(xyz, *shading.mirror_frames(normals), case.heliostat,
                                   shading.grid_offsets(case.shading.n_points))
    hit = shading.ray_hits_tower(points, direction, case.receiver.center[:2], case.receiver.tower_radius,
                                 case.receiver.top)
    return hit.mean(axis=1)


def field_map(case: Case, xyz: np.ndarray, ids: np.ndarray, values=None, label: str = "", colorscale="Viridis",
              cmin=None, cmax=None, selected: int | None = None, title: str = "", hover: list[str] | None = None,
              candidates: np.ndarray | None = None, inactive: np.ndarray | None = None, preview: bool = False,
              sun_angles=None, shadow_fraction=None):
    rec, spec = case.receiver, case.field
    fig = go.Figure()
    _tower_shadow(fig, case, xyz, sun_angles)
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color="black", dash="dash", width=1),
                                 name="场地边界", hoverinfo="skip"))
    if spec.exclusion_radius > 0:
        th = np.linspace(0, 2 * np.pi, 180)
        fig.add_trace(go.Scatter(x=rec.center[0] + spec.exclusion_radius * np.sin(th),
                                 y=rec.center[1] + spec.exclusion_radius * np.cos(th), mode="lines",
                                 line=dict(color="red", dash="dot", width=1), name="塔周禁排区", hoverinfo="skip"))
    if candidates is not None and len(candidates):
        fig.add_trace(go.Scattergl(x=candidates[:, 0], y=candidates[:, 1], mode="markers",
                                   marker=dict(size=3, color="lightgray"), name="候选镜位", hoverinfo="skip"))
    size = float(np.clip(100.0 / np.sqrt(max(len(xyz), 1)), 2.5, 8))
    marker = dict(size=min(size, 4) if preview else size)
    if values is not None:
        marker.update(color=values, colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
                      colorbar=dict(title=label, thickness=12))
    else:
        marker.update(color="#77cfc0" if preview else "#176f68")
    text = hover or [f"ID {i}" for i in ids]
    if shadow_fraction is not None:
        text = [f"{label}<br>塔影遮挡 {fraction:.0%}" for label, fraction in zip(text, shadow_fraction)]
    fig.add_trace(go.Scattergl(x=xyz[:, 0], y=xyz[:, 1], mode="markers", marker=marker, customdata=ids,
                               text=text, hovertemplate="%{text}<extra></extra>", name="定日镜"))
    if shadow_fraction is not None:
        held = np.asarray(shadow_fraction) == 1.
        if inactive is not None:
            held &= ~np.asarray(inactive, bool)
        if held.any():
            fig.add_trace(go.Scatter(x=xyz[held, 0], y=xyz[held, 1], mode="markers", name="全塔影 · 保持瞄准",
                                     marker=dict(symbol="circle-open", size=size + 6, color="#624589", line_width=2),
                                     customdata=np.asarray(ids)[held], hovertemplate="镜 %{customdata} · 全塔影 · 保持瞄准<extra></extra>"))
    if inactive is not None and np.any(inactive):
        k = np.flatnonzero(inactive)
        fig.add_trace(go.Scattergl(x=xyz[k, 0], y=xyz[k, 1], mode="markers", customdata=ids[k],
                                   marker=dict(size=size + 4, symbol="x-thin", line=dict(width=2, color="red")),
                                   name="散焦镜面", hoverinfo="skip"))
    if selected is not None:
        k = np.flatnonzero(ids == selected)
        if k.size:
            fig.add_trace(go.Scatter(x=xyz[k, 0], y=xyz[k, 1], mode="markers", name=f"ID {selected}",
                                     marker=dict(size=size + 10, color="rgba(0,0,0,0)", line=dict(width=2, color="magenta")),
                                     hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔 / 吸热器",
                             marker=dict(symbol="triangle-up", size=13, color="red"), hoverinfo="skip"))
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    _base_layout(fig, title, height=395 if preview else 520)
    if preview:
        fig.update_layout(paper_bgcolor="#122f33", plot_bgcolor="#122f33",
                          font=dict(color="#e1eeea"), title=dict(x=.03, y=.96),
                          margin=dict(l=48, r=20, t=48, b=55))
        fig.update_xaxes(gridcolor="#28464a", zerolinecolor="#4e696a", automargin=True)
        fig.update_yaxes(gridcolor="#28464a", zerolinecolor="#4e696a", automargin=True)
        fig.update_traces(line_color="#8ba5a4", selector=dict(name="场地边界"))
        fig.update_traces(line_color="#eab864", selector=dict(name="塔周禁排区"))
        fig.update_traces(marker_color="#ffd089", selector=dict(name="塔 / 吸热器"))
    return fig


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
        hovertemplate="方位 %{x:.1f}°，高度 %{y:.2f} m<br>入射通量 %{z:.0f} kW/m²<br>上限 %{customdata[0]:.0f} "
                      "(占上限 %{customdata[1]:.1%})，管屏 %{customdata[2]:.0f}<extra></extra>"))
    if ratio.max() > 1.0:
        fig.add_trace(go.Contour(x=grid.azimuth_deg, y=grid.z, z=ratio, showscale=False, hoverinfo="skip",
                                 contours=dict(start=1.0, end=1.0, size=1, coloring="none"),
                                 line=dict(color="cyan", width=2), name="超通量边界"))
    if case.receiver.n_panels > 1:
        for a0, _ in panel_bounds(case.receiver):
            fig.add_vline(x=a0 % 360.0, line=dict(color="white", width=1, dash="dot"))
    if aims is not None:
        fig.add_trace(go.Scattergl(x=np.asarray(aims[0]) % 360.0, y=aims[1], mode="markers", hoverinfo="skip",
                                   marker=dict(size=3, color="#4fc3f7", opacity=0.4), name="镜面瞄准点"))
    if hotspot:
        h = hotspot.get("hotspot", hotspot)
        fig.add_trace(go.Scatter(x=[h["azimuth_deg"]], y=[h["z_m"]], mode="markers", name="细网格复核热点",
                                 marker=dict(symbol="x", size=14, color="cyan", line=dict(width=2)), hoverinfo="skip"))
    if selected_cell is not None:
        fig.add_trace(go.Scatter(x=[selected_cell[0]], y=[selected_cell[1]], mode="markers", name="当前选中位置",
                                 marker=dict(symbol="circle-open", size=16, color="magenta", line=dict(width=2)),
                                 hoverinfo="skip"))
    fig.update_xaxes(title="吸热器周向方位（°，北 0° / 东 90°）", range=[0, 360],
                     tickvals=list(range(0, 361, 45)), ticktext=["北", "东北", "东", "东南", "南", "西南", "西", "西北", "北"])
    fig.update_yaxes(title="高度 z (m)")
    return _base_layout(fig, title, height)


def replay_timeline(table, metric: str, label: str, selected_time=None, limit_line: float | None = None):
    fig = go.Figure()
    colors = {"center": "#66736f", "aps_reactive": "#c36a36", "aps_predictive": "#187f70"}
    names = {"center": "中心瞄准基准", "aps_reactive": "APS · 当前 DNI", "aps_predictive": "APS · 含 DNI 预测"}
    for strat, g in table.groupby("strategy", sort=False):
        fig.add_trace(go.Scatter(x=g["time"], y=g[metric], mode="lines+markers", line_shape="hv",
                                 name=names.get(strat, strat), line=dict(color=colors.get(strat))))
    if limit_line is not None:
        fig.add_hline(y=limit_line, line=dict(color="#bd3f34", dash="dash"),
                      annotation_text="设备上限", annotation_position="top left")
    if selected_time is not None:
        fig.add_vline(x=selected_time, line=dict(color="magenta", width=1))
    fig.update_yaxes(title=label)
    return _base_layout(fig, "", 260)


def uniformity_curves(curves: dict[str, list[dict]]):
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_trace(go.Scatter(x=[c["epsilon"] for c in curve], y=[c["power_abs_mw"] for c in curve],
                                 mode="lines+markers", name=name,
                                 text=[c["status"] for c in curve],
                                 hovertemplate="二维 CV 上限 %{x:.3f}<br>吸收功率 %{y:.3f} MWₜₕ<br>状态 %{text}<extra></extra>"))
    fig.update_xaxes(title="考核区二维热流变异系数（CV）")
    fig.update_yaxes(title="满足该 CV 目标时的吸收功率（MWₜₕ）")
    return _base_layout(fig, "吸热功率与热流均匀度", 360)


def _sparse_labels(x, y, labels, min_sep: float = 0.07) -> list[str]:
    """Keep a text label only where it does not crowd the previously labelled point (normalised axes)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sx = np.ptp(x) or 1.0
    sy = np.ptp(y) or 1.0
    out, last = [], None
    for i, lab in enumerate(labels):
        p = np.array([x[i] / sx, y[i] / sy])
        if last is None or i == len(labels) - 1 or np.hypot(*(p - last)) >= min_sep:
            out.append(lab)
            last = p
        else:
            out.append("")
    if len(out) > 1 and out[-1] and last is not None:
        prev = [i for i, t in enumerate(out[:-1]) if t]
        if prev and np.hypot(x[-1] / sx - x[prev[-1]] / sx, y[-1] / sy - y[prev[-1]] / sy) < min_sep:
            out[prev[-1]] = "" if prev[-1] != 0 else out[prev[-1]]
    return out


# --------------------------------------------------------------------------- certificates & overview

def plant_scene(case: Case, xyz: np.ndarray, values=None, label: str = "", height: int = 460):
    """3-D view of the plant: heliostats, tower and the receiver cylinder."""
    rec = case.receiver
    cx, cy, cz = rec.center
    fig = go.Figure()
    marker = dict(size=2.6, color=values if values is not None else TEAL, colorscale="Viridis",
                  showscale=values is not None, colorbar=dict(title=label, thickness=10, len=0.6))
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", marker=marker,
                               name="定日镜", hovertemplate="x %{x:.0f} m, y %{y:.0f} m<extra></extra>"))
    th = np.linspace(0, 2 * np.pi, 40)
    zt = np.array([0.0, cz - 0.5 * rec.height])
    tr = rec.tower_radius
    fig.add_trace(go.Surface(x=cx + tr * np.outer(np.sin(th), np.ones(2)), y=cy + tr * np.outer(np.cos(th), np.ones(2)),
                             z=np.outer(np.ones_like(th), zt), showscale=False, colorscale=[[0, "#9aa7ad"], [1, "#9aa7ad"]],
                             hoverinfo="skip", name="塔"))
    zr = np.array([cz - 0.5 * rec.height, cz + 0.5 * rec.height])
    fig.add_trace(go.Surface(x=cx + rec.radius * np.outer(np.sin(th), np.ones(2)),
                             y=cy + rec.radius * np.outer(np.cos(th), np.ones(2)), z=np.outer(np.ones_like(th), zr),
                             showscale=False, colorscale=[[0, AMBER], [1, AMBER]], hoverinfo="skip", name="吸热器"))
    span = float(np.ptp(xyz[:, :2], axis=0).max()) if len(xyz) else 400.0
    fig.update_layout(scene=dict(xaxis_title="东 x (m)", yaxis_title="北 y (m)", zaxis_title="z (m)",
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=max(0.25, 1.6 * cz / max(span, 1.0))),
                                 camera=dict(eye=dict(x=-1.25, y=-1.55, z=0.75))),
                      showlegend=False)
    return _base_layout(fig, "", height)


def receiver_3d(case: Case, grid: ReceiverGrid, flux_w_m2: np.ndarray, height: int = 420):
    """Receiver cylinder coloured by incident flux (kW/m2)."""
    rec = case.receiver
    cx, cy, _ = rec.center
    kw = np.asarray(flux_w_m2).reshape(grid.shape) / 1e3
    az = np.radians(np.concatenate([grid.azimuth_deg, grid.azimuth_deg[:1] + 360.0]))
    kw = np.hstack([kw, kw[:, :1]])
    zz, aa = np.meshgrid(grid.z, az, indexing="ij")
    fig = go.Figure(go.Surface(x=cx + rec.radius * np.sin(aa), y=cy + rec.radius * np.cos(aa), z=zz, surfacecolor=kw,
                               colorscale=FLUX_SCALE, cmin=0, colorbar=dict(title="kW/m²", thickness=10),
                               hovertemplate="%{surfacecolor:.0f} kW/m²<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="东 x", yaxis_title="北 y", zaxis_title="z (m)", aspectmode="data",
                                 camera=dict(eye=dict(x=0.9, y=-1.7, z=0.35))))
    return _base_layout(fig, "", height)


def pareto_chart(rows: list[dict], selected: float | None = None, height: int = 380):
    """Absorbed power vs CV_2D for the weights lambda (each point a certified plan)."""
    fig = go.Figure()
    cv = [r["cv_2d"] for r in rows]
    pw = [r["power_abs_mw"] for r in rows]
    lam = [r["lambda"] for r in rows]
    gaps = [r.get("gap") for r in rows]
    text = [f"λ = {l:g}" for l in lam]
    hover = [f"λ = {l:g}<br>二维 CV {c:.3f}<br>吸收功率 {p:.3f} MW" + (f"<br>最优性差距 ≤ {g:.2%}" if g is not None else "")
             for l, c, p, g in zip(lam, cv, pw, gaps)]
    fig.add_trace(go.Scatter(x=cv, y=pw, mode="lines+markers+text", text=_sparse_labels(cv, pw, text),
                             textposition="top right",
                             line=dict(color=TEAL, width=2), marker=dict(size=9, color=TEAL, line=dict(color="white", width=1)),
                             hovertext=hover, hoverinfo="text", name="已复核方案"))
    if selected is not None and selected in lam:
        j = lam.index(selected)
        fig.add_trace(go.Scatter(x=[cv[j]], y=[pw[j]], mode="markers", marker=dict(size=16, color="rgba(0,0,0,0)",
                                 line=dict(color=AMBER, width=3)), name="当前权重", hoverinfo="skip"))
    fig.update_xaxes(title="考核区二维热流变异系数 CV（越小越均匀）", autorange="reversed")
    fig.update_yaxes(title="吸收功率 (MWₜₕ)")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, "", height)


def bound_chart(history: list[dict], found_mw: float | None, height: int = 300):
    """Upper bound vs iteration with the found objective."""
    fig = go.Figure()
    it = [h["iteration"] for h in history]
    ub = [h["upper_bound_mw"] for h in history]
    fig.add_trace(go.Scatter(x=it, y=ub, mode="lines", line=dict(color=NAVY, width=2), name="可证上界",
                             hovertemplate="迭代 %{x}<br>上界 %{y:.4f} MW<extra></extra>"))
    if found_mw is not None:
        fig.add_hline(y=found_mw, line=dict(color=TEAL, width=2, dash="dash"),
                      annotation_text=f"本方案 {found_mw:.4f} MW", annotation_position="bottom right")
    fig.update_xaxes(title="对偶迭代次数")
    fig.update_yaxes(title="目标函数 (MW)")
    return _base_layout(fig, "", height)


def potential_map_chart(case: Case, pmap, xyz: np.ndarray | None = None, values=None, label: str = "",
                        title: str = "", height: int = 560):
    """Annual energy of a lone heliostat over the site, with the layout on top."""
    rec, spec = case.receiver, case.field
    fig = go.Figure(go.Heatmap(x=pmap.x, y=pmap.y, z=np.where(pmap.inside, pmap.energy_mwh, np.nan),
                               colorscale=[[0, "#f3f6f4"], [0.5, "#a9d6c6"], [1, "#1d6f62"]],
                               colorbar=dict(title=dict(text="潜力<br>MWh/年", side="top"), thickness=10, x=1.0),
                               hovertemplate="x %{x:.0f} m, y %{y:.0f} m<br>单镜年吸收潜力 %{z:.1f} MWh<extra></extra>",
                               name="单镜年潜力"))
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color=INK, dash="dash", width=1),
                                 name="场地边界", hoverinfo="skip"))
    if spec.exclusion_radius > 0:
        th = np.linspace(0, 2 * np.pi, 180)
        fig.add_trace(go.Scatter(x=rec.center[0] + spec.exclusion_radius * np.sin(th),
                                 y=rec.center[1] + spec.exclusion_radius * np.cos(th), mode="lines",
                                 line=dict(color=RED, dash="dot", width=1), name="禁排区", hoverinfo="skip"))
    if xyz is not None and len(xyz):
        marker = dict(size=float(np.clip(110.0 / np.sqrt(len(xyz)), 3, 8)), line=dict(width=0.5, color="white"))
        if values is not None:
            marker.update(color=values, colorscale="YlOrBr", showscale=True,
                          colorbar=dict(title=dict(text=label.replace("年均", "年均<br>"), side="top"), thickness=10, x=1.13))
        else:
            marker.update(color=AMBER)
        fig.add_trace(go.Scattergl(x=xyz[:, 0], y=xyz[:, 1], mode="markers", marker=marker, name="定日镜",
                                   hovertemplate="x %{x:.1f} m, y %{y:.1f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔",
                             marker=dict(symbol="triangle-up", size=14, color=RED), hoverinfo="skip"))
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    return _base_layout(fig, title, height)


def loss_waterfall(power_mw: dict, height: int = 330):
    """Power chain from the sun to the receiver as a waterfall (MW)."""
    names = {"sun": "镜面入射", "cosine": "余弦", "reflectivity": "反射/清洁", "shading": "阴影",
             "blocking": "遮挡", "attenuation": "大气衰减", "intercept": "截断"}
    keys = list(power_mw)
    vals = [power_mw[keys[0]]] + [power_mw[k] - power_mw[p] for p, k in zip(keys[:-1], keys[1:])] + [power_mw[keys[-1]]]
    fig = go.Figure(go.Waterfall(x=[names.get(keys[0], keys[0])] + [names.get(k, k) for k in keys[1:]] + ["到达吸热器"],
                                 measure=["absolute"] + ["relative"] * (len(keys) - 1) + ["total"], y=vals,
                                 decreasing=dict(marker=dict(color="#d98c7f")), totals=dict(marker=dict(color=TEAL)),
                                 increasing=dict(marker=dict(color=TEAL)),
                                 text=[f"{v:+.2f}" if i not in (0, len(vals) - 1) else f"{v:.2f}" for i, v in enumerate(vals)],
                                 textposition="outside", hovertemplate="%{x}: %{y:.3f} MW<extra></extra>"))
    fig.update_yaxes(title="功率 (MW)")
    return _base_layout(fig, "", height)


def history_chart(history, height: int = 300):
    """Layout search: energy of every evaluated parameter vector and the running best."""
    import pandas as pd

    h = history.sort_values("evaluation") if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["evaluation"], y=h["energy_mwh"], mode="markers", marker=dict(size=6, color="#9fb7b0"),
                             name="每次布局评价", hovertemplate="第 %{x} 次: %{y:.0f} MWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["evaluation"], y=h["energy_mwh"].cummax(), mode="lines", line=dict(color=TEAL, width=2),
                             name="迄今最优"))
    fig.update_xaxes(title="布局评价次数")
    fig.update_yaxes(title="搜索目标 J (MWhₜₕ)")
    return _base_layout(fig, "", height)


def receiver_rose(share, centers_deg, compare=None, height: int = 380):
    """Share of the annual absorbed energy per receiver azimuth sector (compass rose)."""
    width = 360.0 / max(len(centers_deg), 1)
    fig = go.Figure()
    if compare is not None:
        fig.add_trace(go.Barpolar(r=[100 * v for v in compare[1]], theta=centers_deg, width=[width * 0.9] * len(centers_deg),
                                  marker=dict(color="rgba(194,69,58,0.25)", line=dict(color=RED, width=1)),
                                  name=compare[0], hovertemplate="%{theta:.0f}°：%{r:.1f}%<extra>" + compare[0] + "</extra>"))
    fig.add_trace(go.Barpolar(r=[100 * v for v in share], theta=centers_deg, width=[width * 0.9] * len(centers_deg),
                              marker=dict(color=TEAL, opacity=0.85, line=dict(color="white", width=1)), name="本方案",
                              hovertemplate="%{theta:.0f}°：%{r:.1f}%<extra>本方案</extra>"))
    fig.update_layout(polar=dict(barmode="overlay",
                                 angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                                  tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                                  ticktext=["北", "东北", "东", "东南", "南", "西南", "西", "西北"]),
                                 radialaxis=dict(ticksuffix="%", angle=60, tickfont=dict(size=10))),
                      showlegend=compare is not None)
    return _base_layout(fig, "", height)


def layout_tradeoff_chart(rows: list[dict], selected: float | None = None, height: int = 360):
    """Annual energy vs circumferential CV of the layout for the weights lambda_L."""
    surface = bool(rows and "surface_spread_mwh" in rows[0])
    cv = [r["surface_spread_mwh"] if surface else r["cv_circ_weighted"] for r in rows]
    e = [r["energy_mwh"] for r in rows]
    lam = [r["lambda"] for r in rows]
    hover = [f"λ_L = {l:g}<br>管屏间 CV {c:.3f}<br>年吸收 {v:,.0f} MWh<br>北侧镜子 {100 * r['north_fraction']:.0f}%"
             for l, c, v, r in zip(lam, cv, e, rows)]
    fig = go.Figure(go.Scatter(x=cv, y=e, mode="lines+markers+text",
                               text=_sparse_labels(cv, e, [f"λ_L = {l:g}" for l in lam]),
                               textposition="top right", hovertext=hover, hoverinfo="text",
                               line=dict(color=NAVY, width=2), marker=dict(size=9, color=AMBER,
                                                                          line=dict(color="white", width=1))))
    if selected is not None and selected in lam:
        j = lam.index(selected)
        fig.add_trace(go.Scatter(x=[cv[j]], y=[e[j]], mode="markers", hoverinfo="skip",
                                 marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(color=TEAL, width=3))))
    fig.update_xaxes(title="二维不均匀惩罚量 (MWhₜₕ)" if surface else "管屏间吸热 CV（逐工况能量加权）",
                     autorange="reversed")
    fig.update_yaxes(title="全年理论吸收热量 (MWhₜₕ)")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, "", height)


def refinement_chart(levels: list[dict], height: int = 320):
    """Found objective and upper bound on nested candidate grids (convergence towards continuous aims)."""
    k = [r["n_candidates"] for r in levels]
    labels = [f"{r['n_azimuth']}×{r['n_height']}" for r in levels]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[r["upper_bound_mw"] for r in levels], mode="lines+markers", name="可证上界",
                             line=dict(color=NAVY, width=2, dash="dash"), text=labels,
                             hovertemplate="%{text}：上界 %{y:.4f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=[r["objective_found_mw"] for r in levels], mode="lines+markers+text", name="本方案",
                             line=dict(color=TEAL, width=2), text=labels, textposition="bottom right",
                             hovertemplate="%{text}：%{y:.4f} MW<extra></extra>"))
    fig.update_xaxes(title="每面镜的候选瞄准点数", type="log")
    fig.update_yaxes(title="目标函数 (MW)")
    return _base_layout(fig, "", height)


# --------------------------------------------------------------------------- aims, clouds and annual value

DNI_SCALE = [[0.0, "#1f3a5f"], [0.35, "#5f7a99"], [0.6, "#b9a77a"], [0.8, "#eaa53a"], [1.0, "#d9480f"]]
# shadow = dark slate blue, full sun = amber / orange: readable on a white background
DIFF_SCALE = [[0.0, "#2b5d9b"], [0.5, "#f7f7f5"], [1.0, "#b8432f"]]


def _cylinder(rec, n: int = 48):
    th = np.linspace(0, 2 * np.pi, n)
    return th, np.sin(th), np.cos(th)


def _mirror_colors(values, cmin=None, cmax=None):
    v = np.asarray(values, dtype=float)
    lo = float(np.nanmin(v)) if cmin is None else cmin
    hi = float(np.nanmax(v)) if cmax is None else cmax
    return v, lo, hi if hi > lo else lo + 1.0


def aim_scene_3d(case: Case, xyz: np.ndarray, dni, aim_az, aim_z, active, max_lines: int = 140,
                 title: str = "", height: int = 520, cmin=None, cmax=None, sun_angles=None, beam_power=None, shadow_fraction=None):
    """Whole plant in 3-D: heliostats coloured by DNI, a sample of beams from each mirror centre to its aim point."""
    from .receiver import surface_point

    rec = case.receiver
    cx, cy, cz = rec.center
    xyz = np.asarray(xyz, float)
    act = np.asarray(active, bool)
    v, lo, hi = _mirror_colors(dni, cmin, cmax)
    fig = go.Figure()
    _tower_shadow(fig, case, xyz, sun_angles, scene=True)
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="定日镜（按 DNI 着色）",
                               marker=dict(size=2.8, color=v, colorscale=DNI_SCALE, cmin=lo, cmax=hi, showscale=True,
                                           colorbar=dict(title="DNI<br>W/m²", thickness=10, len=0.55, x=1.0)),
                               hovertemplate="x %{x:.0f} m, y %{y:.0f} m<br>DNI %{marker.color:.0f} W/m²<extra></extra>"))
    if (~act).any():
        k = np.flatnonzero(~act)
        fig.add_trace(go.Scatter3d(x=xyz[k, 0], y=xyz[k, 1], z=xyz[k, 2], mode="markers", name="散焦",
                                   marker=dict(size=3.5, color=RED, symbol="x"), hoverinfo="skip"))
    if shadow_fraction is not None:
        held = act & (np.asarray(shadow_fraction) == 1.)
        if held.any():
            fig.add_trace(go.Scatter3d(x=xyz[held, 0], y=xyz[held, 1], z=xyz[held, 2], mode="markers",
                                       name="全塔影 · 保持瞄准", marker=dict(symbol="circle-open", size=5, color="#624589"),
                                       hovertemplate="全塔影 · 保持瞄准<extra></extra>"))
    aims = surface_point(rec, np.asarray(aim_az, float), np.asarray(aim_z, float))
    on = np.flatnonzero(act & (np.asarray(beam_power if beam_power is not None else dni) > 0))
    if on.size:
        order = on[np.argsort(np.degrees(np.arctan2(xyz[on, 0] - cx, xyz[on, 1] - cy)) % 360.0, kind="stable")]
        pick = order[np.linspace(0, order.size - 1, min(max_lines, order.size)).astype(int)]
        lx, ly, lz = [], [], []
        for i in pick:
            lx += [xyz[i, 0], aims[i, 0], None]
            ly += [xyz[i, 1], aims[i, 1], None]
            lz += [xyz[i, 2], aims[i, 2], None]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name=f"反射光束（抽样 {pick.size} 条）",
                                   line=dict(color="rgba(224,155,45,0.45)", width=2), hoverinfo="skip"))
    th, s, c = _cylinder(rec)
    zt = np.array([0.0, cz - 0.5 * rec.height])
    fig.add_trace(go.Surface(x=cx + rec.tower_radius * np.outer(s, np.ones(2)), y=cy + rec.tower_radius * np.outer(c, np.ones(2)),
                             z=np.outer(np.ones_like(th), zt), showscale=False, colorscale=[[0, "#9aa7ad"], [1, "#9aa7ad"]],
                             hoverinfo="skip", name="塔"))
    zr = np.array([cz - 0.5 * rec.height, cz + 0.5 * rec.height])
    fig.add_trace(go.Surface(x=cx + rec.radius * np.outer(s, np.ones(2)), y=cy + rec.radius * np.outer(c, np.ones(2)),
                             z=np.outer(np.ones_like(th), zr), showscale=False, colorscale=[[0, AMBER], [1, AMBER]],
                             hoverinfo="skip", name="吸热器"))
    span = float(np.ptp(xyz[:, :2], axis=0).max()) if len(xyz) else 400.0
    fig.update_layout(scene=dict(xaxis_title="东 x (m)", yaxis_title="北 y (m)", zaxis_title="z (m)",
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=max(0.25, 1.6 * cz / max(span, 1.0))),
                                 camera=dict(eye=dict(x=-1.3, y=-1.5, z=0.8))),
                      legend=dict(orientation="h", y=0.02))
    return _base_layout(fig, title, height)


def receiver_aims_3d(case: Case, grid: ReceiverGrid, flux_w_m2, xyz: np.ndarray, aim_az, aim_z, active, dni=None,
                     ids=None, title: str = "", height: int = 520, zmax=None, stub_m: float | None = None, beam_power=None):
    """Receiver close-up: flux on the cylinder, every mirror's aim point and a short stub towards that mirror."""
    from .receiver import surface_point

    rec = case.receiver
    cx, cy, _ = rec.center
    kw = np.asarray(flux_w_m2).reshape(grid.shape) / 1e3
    az = np.radians(np.concatenate([grid.azimuth_deg, grid.azimuth_deg[:1] + 360.0]))
    kw2 = np.hstack([kw, kw[:, :1]])
    zz, aa = np.meshgrid(grid.z, az, indexing="ij")
    fig = go.Figure(go.Surface(x=cx + rec.radius * np.sin(aa), y=cy + rec.radius * np.cos(aa), z=zz, surfacecolor=kw2,
                               colorscale=FLUX_SCALE, cmin=0, cmax=zmax, opacity=0.97, name="入射热流",
                               colorbar=dict(title="kW/m²", thickness=10, len=0.6, x=1.02),
                               hovertemplate="%{surfacecolor:.0f} kW/m²<extra></extra>"))
    act = np.asarray(active, bool)
    on = np.flatnonzero(act)
    aims = surface_point(rec, np.asarray(aim_az, float), np.asarray(aim_z, float))
    lit = on if beam_power is None else on[np.asarray(beam_power)[on] > 0]
    if lit.size:
        d = np.asarray(xyz, float)[lit] - aims[lit]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        stub = stub_m or 0.45 * rec.diameter
        tip = aims[lit] + stub * d
        lx = np.column_stack([aims[lit, 0], tip[:, 0], np.full(lit.size, np.nan)]).ravel()
        ly = np.column_stack([aims[lit, 1], tip[:, 1], np.full(lit.size, np.nan)]).ravel()
        lz = np.column_stack([aims[lit, 2], tip[:, 2], np.full(lit.size, np.nan)]).ravel()
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="rgba(79,195,247,0.18)", width=2),
                                   name="来光方向（指向镜子）", hoverinfo="skip"))
    if on.size:
        vals = np.asarray(dni, float)[on] if dni is not None else None
        txt = [f"镜 {int(ids[i]) if ids is not None else i + 1}<br>瞄准 方位 {np.asarray(aim_az)[i] % 360:.1f}°, 高 {np.asarray(aim_z)[i]:.2f} m"
               + (f"<br>DNI {np.asarray(dni)[i]:.0f} W/m²" if dni is not None else "") for i in on]
        fig.add_trace(go.Scatter3d(x=aims[on, 0], y=aims[on, 1], z=aims[on, 2], mode="markers", name="镜面中心瞄准点",
                                   marker=dict(size=3, color=vals if vals is not None else "#4fc3f7",
                                               colorscale=DNI_SCALE, showscale=False),
                                   text=txt, hovertemplate="%{text}<extra></extra>"))
    span = rec.diameter * 1.4
    fig.update_layout(scene=dict(xaxis=dict(title="东 x", range=[cx - span, cx + span]),
                                 yaxis=dict(title="北 y", range=[cy - span, cy + span]), zaxis_title="z (m)",
                                 aspectmode="data", camera=dict(eye=dict(x=0.9, y=-1.6, z=0.45))),
                      legend=dict(orientation="h", y=0.02))
    return _base_layout(fig, title, height)


def receiver_aims_map(case: Case, grid: ReceiverGrid, flux_w_m2, aim_az, aim_z, active, ids=None, values=None,
                      value_label: str = "DNI (W/m²)", prev=None, title: str = "", height: int = 420, zmax=None,
                      hotspot: dict | None = None, show_flux: bool = True, cmin=None, cmax=None):
    """Unrolled receiver: flux heat map and every active mirror's aim point (coloured by a mirror attribute);
    with ``prev`` = (az, z, active) the moved aims get arrows from their previous position."""
    fig = receiver_map(case, grid, flux_w_m2, title, zmax=zmax, hotspot=hotspot, height=height) if show_flux else go.Figure()
    if show_flux:  # flux colour bar in the upper half, mirror colour bar in the lower half
        fig.update_traces(colorbar=dict(len=0.48, y=0.76, thickness=10), selector=dict(type="heatmap"))
    act = np.asarray(active, bool)
    on = np.flatnonzero(act)
    az = np.asarray(aim_az, float) % 360.0
    zz = np.asarray(aim_z, float)
    ids = np.arange(1, act.size + 1) if ids is None else np.asarray(ids)
    if prev is not None:
        paz, pz, pact = np.asarray(prev[0], float) % 360.0, np.asarray(prev[1], float), np.asarray(prev[2], bool)
        moved = np.flatnonzero(act & pact & ((np.abs(((paz - az + 180) % 360) - 180) > 1e-6) | (np.abs(pz - zz) > 1e-6)))
        if moved.size:
            # unwrap across north: draw the short way round (the segment may leave the 0-360 window)
            end = paz[moved] + ((az[moved] - paz[moved] + 180.0) % 360.0 - 180.0)
            lx = np.column_stack([paz[moved], end, np.full(moved.size, np.nan)]).ravel()
            ly = np.column_stack([pz[moved], zz[moved], np.full(moved.size, np.nan)]).ravel()
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(255,255,255,0.35)", width=0.8),
                                     name=f"瞄准点移动（{moved.size} 面）", hoverinfo="skip"))
        gone = np.flatnonzero(pact & ~act)
        if gone.size:
            fig.add_trace(go.Scatter(x=paz[gone], y=pz[gone], mode="markers", name=f"改为散焦（{gone.size} 面）",
                                     marker=dict(symbol="x", size=7, color=RED), hoverinfo="skip"))
    if on.size:
        marker = dict(size=6, line=dict(width=0.6, color="white"))
        if values is not None:
            v = np.asarray(values, float)[on]
            marker.update(color=v, colorscale=DNI_SCALE, showscale=True, cmin=cmin, cmax=cmax,
                          colorbar=dict(title=value_label, thickness=10, len=0.42, y=0.2))
        else:
            marker.update(color="#4fc3f7")
        txt = [f"镜 {ids[i]}<br>方位 {az[i]:.1f}°，高 {zz[i]:.2f} m" +
               (f"<br>{value_label} {np.asarray(values)[i]:.1f}" if values is not None else "") for i in on]
        fig.add_trace(go.Scatter(x=az[on], y=zz[on], mode="markers", marker=marker, name="镜面中心的瞄准点",
                                   text=txt, hovertemplate="%{text}<extra></extra>", customdata=ids[on]))
    fig.update_xaxes(title="吸热器周向方位（°，北 0° / 东 90°）", range=[0, 360],
                     tickvals=list(range(0, 361, 45)), ticktext=["北", "东北", "东", "东南", "南", "西南", "西", "西北", "北"])
    fig.update_yaxes(title="高度 z (m)")
    return _base_layout(fig, title, height)


def flux_diff_map(case: Case, grid: ReceiverGrid, flux_from, flux_to, title: str = "", height: int = 360):
    d = (np.asarray(flux_to) - np.asarray(flux_from)) / 1e3
    m = float(np.abs(d).max()) or 1.0
    fig = go.Figure(go.Heatmap(x=grid.azimuth_deg, y=grid.z, z=d, colorscale=DIFF_SCALE, zmin=-m, zmax=m,
                               colorbar=dict(title="Δ kW/m²", thickness=12),
                               hovertemplate="方位 %{x:.1f}°，高度 %{y:.2f} m<br>变化 %{z:+.1f} kW/m²<extra></extra>"))
    fig.update_xaxes(title="吸热器周向方位（°）", range=[0, 360], tickvals=list(range(0, 361, 45)),
                     ticktext=["北", "东北", "东", "东南", "南", "西南", "西", "西北", "北"])
    fig.update_yaxes(title="高度 z (m)")
    return _base_layout(fig, title, height)


def field_dni_map(case: Case, xyz, ids, dni, dni_map=None, clouds=None, inactive=None, title: str = "",
                  cmin=None, cmax=None, height: int = 480, map_scale: float = 1.0, sun_angles=None,
                  shadow_fraction=None):
    """Mirrors coloured by their DNI over an optional site raster (DNI map) and cloud outlines."""
    rec, spec = case.receiver, case.field
    fig = go.Figure()
    v, lo, hi = _mirror_colors(dni, cmin, cmax)
    if dni_map is not None:
        fig.add_trace(go.Heatmap(x=dni_map.x, y=dni_map.y, z=dni_map.values * map_scale, colorscale=DNI_SCALE,
                                 zmin=lo, zmax=hi, showscale=False, opacity=0.32, name="DNI 地图", hoverinfo="skip"))
    _tower_shadow(fig, case, xyz, sun_angles)
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color=INK, dash="dash", width=1),
                                 name="场地边界", hoverinfo="skip"))
    for k, cl in enumerate(clouds or []):
        th = np.linspace(0, 2 * np.pi, 90)
        fig.add_trace(go.Scatter(x=cl.x + cl.radius * np.sin(th), y=cl.y + cl.radius * np.cos(th), mode="lines",
                                 line=dict(color="#5d6d74", width=1.5, dash="dot"), name="云影边界" if k == 0 else None,
                                 showlegend=k == 0, hoverinfo="skip"))
    size = float(np.clip(100.0 / np.sqrt(max(len(xyz), 1)), 3, 9))
    ids = np.asarray(ids)
    fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="markers", name="定日镜",
                               marker=dict(size=size, color=v, colorscale=DNI_SCALE, cmin=lo, cmax=hi, showscale=True,
                                           line=dict(width=0.5 if dni_map is not None else 0.3,
                                                     color="#16232b" if dni_map is not None else "white"),
                                           colorbar=dict(title="DNI (W/m²)", thickness=12)),
                               customdata=ids, text=[f"镜 {i}<br>DNI {d:.0f} W/m²" for i, d in zip(ids, v)],
                               hovertemplate="%{text}<extra></extra>"))
    if shadow_fraction is not None:
        fraction = np.asarray(shadow_fraction)
        held = fraction == 1.
        if held.any():
            fig.add_trace(go.Scatter(x=xyz[held, 0], y=xyz[held, 1], mode="markers",
                                     name="全塔影 · 保持瞄准",
                                     marker=dict(symbol="circle-open", size=size + 6, color="#624589",
                                                 line=dict(width=2)),
                                     customdata=ids[held],
                                     text=[f"镜 {i} · 全塔影 · 保持瞄准 · 遮挡 100%" for i in ids[held]],
                                     hovertemplate="%{text}<extra></extra>"))
        partial = (fraction > 0) & (fraction < 1)
        if partial.any():
            fig.add_trace(go.Scatter(x=xyz[partial, 0], y=xyz[partial, 1], mode="markers",
                                     name="部分塔影", marker=dict(symbol="circle-open", size=size + 5,
                                                                  color="#9875ba"),
                                     customdata=ids[partial],
                                     text=[f"镜 {i} · 塔影遮挡 {v:.0%}" for i, v in zip(ids[partial], fraction[partial])],
                                     hovertemplate="%{text}<extra></extra>"))
    if inactive is not None and np.any(inactive):
        k = np.flatnonzero(inactive)
        fig.add_trace(go.Scatter(x=xyz[k, 0], y=xyz[k, 1], mode="markers", name="散焦",
                                   marker=dict(size=size + 4, symbol="x-thin", line=dict(width=2, color=RED)),
                                   hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔",
                             marker=dict(symbol="triangle-up", size=13, color=RED), hoverinfo="skip"))
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    return _base_layout(fig, title, height)


def state_bars(states: list[dict], height: int = 340):
    """Absorbed power, uniformity penalty and objective of the cloud-compare states."""
    labels = [s["short"] for s in states]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[s["p_abs_mw"] for s in states], name="吸收功率 P", marker_color=TEAL,
                         text=[f"{s['p_abs_mw']:.2f}" for s in states], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[s["u_mw"] for s in states], name="不均匀量 U = A_H·σ_H", marker_color="#d98c7f",
                         text=[f"{s['u_mw']:.2f}" for s in states], textposition="outside"))
    fig.add_trace(go.Scatter(x=labels, y=[s["objective_mw"] for s in states], mode="markers+text", name="目标 J = P − λU",
                             marker=dict(symbol="diamond", size=13, color=NAVY), text=[f"{s['objective_mw']:.2f}" for s in states],
                             textposition="top center"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MW")
    return _base_layout(fig, "", height)


def value_decomposition_chart(ranking, labels: dict, height: int = 340):
    """ΔV of every layout against the reference, split into more heat (ΔE) and more uniform heating (−λ_L ΔΠ)."""
    names = list(ranking["name"])
    x = [labels.get(n, n) for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=ranking["delta_energy_mwh"], name="吸热量差 ΔE", marker_color=TEAL,
                         hovertemplate="%{x}<br>ΔE %{y:+,.1f} MWhₜₕ<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=ranking["delta_uniformity_mwh"], name="均匀性差 −λ_L·ΔΠ", marker_color=AMBER,
                         hovertemplate="%{x}<br>−λΔΠ %{y:+,.1f} MWhₜₕ<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=ranking["delta_value_mwh"], mode="markers+text", name="年度价值差 ΔV",
                             marker=dict(symbol="diamond", size=13, color=NAVY),
                             text=[f"{v:+,.0f}" for v in ranking["delta_value_mwh"]], textposition="top center"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title=f"相对 {labels.get(ranking['reference'].iloc[0], ranking['reference'].iloc[0])} (MWhₜₕ)")
    return _base_layout(fig, "", height)


def lambda_sweep_chart(totals: dict, labels: dict, lam_now: float, lam_max: float = 1.0, height: int = 340):
    """V(L) = E - lambda_L * Pi as straight lines: where they cross, the ranking changes."""
    lams = np.linspace(0.0, max(lam_max, lam_now * 1.5, 0.1), 60)
    fig = go.Figure()
    palette = [TEAL, AMBER, NAVY, RED, "#7b5ea7", "#3b8bc2", "#6b8e23"]
    for k, (name, t) in enumerate(totals.items()):
        fig.add_trace(go.Scatter(x=lams, y=t["energy_mwh"] - lams * t["penalty_mwh"], mode="lines",
                                 name=labels.get(name, name), line=dict(width=2.5, color=palette[k % len(palette)]),
                                 hovertemplate="λ_L %{x:.2f}: V %{y:,.0f} MWhₜₕ<extra>" + labels.get(name, name) + "</extra>"))
    fig.add_vline(x=lam_now, line=dict(color=MUTED, dash="dash"), annotation_text=f"当前 λ_L = {lam_now:g}")
    fig.update_xaxes(title="外层权重 λ_L（内层瞄准方案不变）")
    fig.update_yaxes(title="年度价值 V (MWhₜₕ)")
    return _base_layout(fig, "", height)


def scenario_contrib_chart(tables: dict, labels: dict, lam: float, height: int = 360):
    """Per-scenario contributions h_s (P_s - lambda_L U_s) of several layouts."""
    fig = go.Figure()
    palette = [TEAL, AMBER, NAVY, RED]
    for k, (name, t) in enumerate(tables.items()):
        contrib = t["hours"] * (t["p_abs_mw"] - lam * t["u_mw"]) * t["counts_in_value"]
        fig.add_trace(go.Bar(x=t["scenario_id"], y=contrib, name=labels.get(name, name),
                             marker_color=palette[k % len(palette)],
                             customdata=np.column_stack([t["hours"], t["p_abs_mw"], t["u_mw"], t["map_id"]]),
                             hovertemplate="%{x} · %{customdata[3]}<br>h = %{customdata[0]:.1f} h<br>"
                                           "P = %{customdata[1]:.3f} MW, U = %{customdata[2]:.3f} MW<br>"
                                           "贡献 %{y:,.1f} MWhₜₕ<extra>" + labels.get(name, name) + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="工况")
    fig.update_yaxes(title="h·(P − λ_L·U) (MWhₜₕ)")
    return _base_layout(fig, "", height)


def scenario_sky_chart(table, height: int = 340):
    """Representative sun positions sized by hours and coloured by DNI map."""
    fig = go.Figure()
    colors = [TEAL, AMBER, NAVY, RED, "#7b5ea7", "#3b8bc2"]
    palette = {m: colors[k % len(colors)] for k, m in enumerate(sorted(set(table["map_id"])))}
    for m, g in table.groupby("map_id", sort=False):
        fig.add_trace(go.Scatter(x=g["azimuth_deg"], y=g["elevation_deg"], mode="markers+text", name=str(m),
                                 text=g["scenario_id"], textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=8 + 22 * np.sqrt(g["hours"] / table["hours"].max()), color=palette[m],
                                             opacity=0.75, line=dict(color="white", width=1)),
                                 customdata=np.column_stack([g["hours"], g["dni_w_m2"]]),
                                 hovertemplate="%{text}<br>方位 %{x:.0f}°，高度角 %{y:.0f}°<br>代表 %{customdata[0]:.0f} h，"
                                               "DNI %{customdata[1]:.0f} W/m²<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(title="太阳方位角（°，北 0° / 东 90°）", range=[0, 360], tickvals=list(range(0, 361, 45)))
    fig.update_yaxes(title="太阳高度角（°）")
    return _base_layout(fig, "", height)


def dni_map_chart(case: Case, dni_map, title: str = "", height: int = 330, xyz=None):
    rec, spec = case.receiver, case.field
    fig = go.Figure(go.Heatmap(x=dni_map.x, y=dni_map.y, z=dni_map.values, colorscale=DNI_SCALE,
                               colorbar=dict(title="倍数" if dni_map.unit == "factor" else "W/m²", thickness=10),
                               hovertemplate="x %{x:.0f} m, y %{y:.0f} m<br>%{z:.3f}<extra></extra>"))
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color="white", dash="dash", width=1),
                                 hoverinfo="skip", showlegend=False))
    if xyz is not None and len(xyz):
        fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                   marker=dict(size=2.5, color="rgba(255,255,255,0.6)")))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="triangle-up", size=11, color=RED)))
    fig.update_xaxes(title="东 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北 y (m)")
    return _base_layout(fig, title, height)


def layout_overlay(case: Case, base_xyz, new_xyz, match=None, labels=("初始布局", "优化布局"), height: int = 520):
    """Two layouts on the site; with a matching (from, to, distance) every moved mirror gets a line."""
    rec, spec = case.receiver, case.field
    fig = go.Figure()
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color=INK, dash="dash", width=1),
                                 name="场地边界", hoverinfo="skip"))
    if match is not None:
        fr, to, dist = match
        moved = dist > 1e-3
        a, bb = np.asarray(base_xyz)[fr[moved]], np.asarray(new_xyz)[to[moved]]
        lx = np.column_stack([a[:, 0], bb[:, 0], np.full(len(a), np.nan)]).ravel()
        ly = np.column_stack([a[:, 1], bb[:, 1], np.full(len(a), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(93,109,116,0.5)", width=1),
                                 name=f"镜位移动（{int(moved.sum())} 面）", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=base_xyz[:, 0], y=base_xyz[:, 1], mode="markers", name=labels[0],
                               marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color="#8a969c", width=1.2))))
    fig.add_trace(go.Scatter(x=new_xyz[:, 0], y=new_xyz[:, 1], mode="markers", name=labels[1],
                               marker=dict(size=4.5, color=TEAL)))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔",
                             marker=dict(symbol="triangle-up", size=13, color=RED), hoverinfo="skip"))
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    return _base_layout(fig, "", height)


def inner_gap_chart(tables: dict, labels: dict, height: int = 300):
    fig = go.Figure()
    palette = [TEAL, AMBER, NAVY, RED]
    for k, (name, t) in enumerate(tables.items()):
        fig.add_trace(go.Scatter(x=t["scenario_id"], y=100 * t["inner_gap"].astype(float), mode="lines+markers",
                                 name=labels.get(name, name), line=dict(color=palette[k % len(palette)])))
    fig.update_yaxes(title="内层差距（%，相对该工况上界）", rangemode="tozero")
    fig.update_xaxes(title="工况")
    return _base_layout(fig, "", height)


def _nice_length(span: float) -> float:
    target = max(span, 1.0) / 5.0
    for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if v >= target * 0.6:
            return float(v)
    return float(10 ** np.floor(np.log10(target)))


def add_north_and_scale(fig, xs, ys):
    """North arrow (paper corner) and a metric scale bar in data coordinates (equal axis scale)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max(x1 - x0, y1 - y0, 1.0)
    length = _nice_length(span)
    bx, by = x0 + 0.04 * span, y0 - 0.06 * span
    fig.add_trace(go.Scatter(x=[bx, bx + length], y=[by, by], mode="lines+text", line=dict(color=INK, width=4),
                             text=["", f"{length:g} m"], textposition="middle right", hoverinfo="skip",
                             showlegend=False, name="比例尺"))
    fig.add_annotation(x=0.985, y=0.97, xref="paper", yref="paper", ax=0, ay=36, axref="pixel", ayref="pixel",
                       showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2.5, arrowcolor=INK, text="")
    fig.add_annotation(x=0.985, y=0.995, xref="paper", yref="paper", text="<b>北 N</b>", showarrow=False,
                       font=dict(size=12, color=INK))
    return fig


def site_preview(case: Case, layouts: dict, selected: int | None = None, title: str = "", height: int = 520):
    """Plan view to confirm the site: boundary, tower, exclusion zone, one or more layouts, north arrow and scale.

    ``layouts`` maps a label to (xyz, ids); markers carry the heliostat ID (customdata) for click selection.
    """
    rec, spec = case.receiver, case.field
    fig = go.Figure()
    xs, ys = [rec.center[0]], [rec.center[1]]
    if spec.boundary is not None:
        b = np.asarray(list(spec.boundary) + [spec.boundary[0]], dtype=float)
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color=INK, dash="dash", width=1),
                                 name="场地边界", hoverinfo="skip"))
        xs += list(b[:, 0])
        ys += list(b[:, 1])
    if spec.exclusion_radius > 0:
        th = np.linspace(0, 2 * np.pi, 180)
        fig.add_trace(go.Scatter(x=rec.center[0] + spec.exclusion_radius * np.sin(th),
                                 y=rec.center[1] + spec.exclusion_radius * np.cos(th), mode="lines",
                                 line=dict(color=RED, dash="dot", width=1), name="塔周禁排区", hoverinfo="skip"))
    palette = [TEAL, AMBER, NAVY, "#7b5ea7"]
    hel = case.heliostat
    for k, (label, (xyz, ids)) in enumerate(layouts.items()):
        xyz = np.asarray(xyz, float)
        ids = np.asarray(ids)
        xs += list(xyz[:, 0])
        ys += list(xyz[:, 1])
        size = float(np.clip(110.0 / np.sqrt(max(len(xyz), 1)), 4, 11))
        fig.add_trace(go.Scatter(
            x=xyz[:, 0], y=xyz[:, 1], mode="markers", name=label, customdata=ids,
            marker=dict(size=size, color=palette[k % 4], symbol="square", line=dict(width=0.5, color="white")),
            text=[f"{label} · 镜 {i}<br>x {p[0]:.2f} m，y {p[1]:.2f} m，z {p[2]:.2f} m<br>镜面 {hel.width:g} × {hel.height:g} m"
                  for i, p in zip(ids, xyz)], hovertemplate="%{text}<extra></extra>"))
        if selected is not None and np.any(ids == selected):
            j = int(np.flatnonzero(ids == selected)[0])
            fig.add_trace(go.Scatter(x=[xyz[j, 0]], y=[xyz[j, 1]], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=size + 12, color="rgba(0,0,0,0)", line=dict(width=2.5, color="magenta"))))
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔（原点）",
                             marker=dict(symbol="triangle-up", size=14, color=RED), hoverinfo="skip"))
    add_north_and_scale(fig, xs, ys)
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    return _base_layout(fig, title, height)


def dni_points_preview(case: Case, points: np.ndarray, values: np.ndarray, xyz: np.ndarray, dni_at_mirrors: np.ndarray,
                       inside: np.ndarray, bad_points: np.ndarray | None = None, height: int = 460):
    """Measured DNI points, the DNI interpolated to every mirror and the mirrors outside the measured area."""
    fig = go.Figure()
    lo = float(min(np.min(values) if len(values) else 0, np.min(dni_at_mirrors) if len(dni_at_mirrors) else 0))
    hi = float(max(np.max(values) if len(values) else 1, np.max(dni_at_mirrors) if len(dni_at_mirrors) else 1))
    fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="markers", name="定日镜（插值 DNI）",
                             marker=dict(size=6, color=dni_at_mirrors, colorscale=DNI_SCALE, cmin=lo, cmax=hi,
                                         showscale=True, colorbar=dict(title="DNI (W/m²)", thickness=11)),
                             text=[f"插值 DNI {d:.0f} W/m²" + ("" if ok else "<br>测点范围外：取最近测点值")
                                   for d, ok in zip(dni_at_mirrors, inside)], hovertemplate="%{text}<extra></extra>"))
    out = np.flatnonzero(~np.asarray(inside, bool))
    if out.size:
        fig.add_trace(go.Scatter(x=xyz[out, 0], y=xyz[out, 1], mode="markers", name=f"测点范围外（{out.size} 面，外推）",
                                 marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(width=1.5, color=RED)),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="测点",
                             marker=dict(size=12, symbol="diamond", color=values, colorscale=DNI_SCALE, cmin=lo,
                                         cmax=hi, line=dict(width=1.2, color=INK)),
                             text=[f"测点 DNI {v:.0f} W/m²" for v in values], hovertemplate="%{text}<extra></extra>"))
    if bad_points is not None and len(bad_points):
        fig.add_trace(go.Scatter(x=bad_points[:, 0], y=bad_points[:, 1], mode="markers", name="有错误的测点",
                                 marker=dict(size=14, symbol="x", color=RED), hoverinfo="skip"))
    rec = case.receiver
    fig.add_trace(go.Scatter(x=[rec.center[0]], y=[rec.center[1]], mode="markers", name="塔",
                             marker=dict(symbol="triangle-up", size=13, color=RED), hoverinfo="skip"))
    add_north_and_scale(fig, np.concatenate([xyz[:, 0], points[:, 0]]), np.concatenate([xyz[:, 1], points[:, 1]]))
    fig.update_xaxes(title="东向 x (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="北向 y (m)")
    return _base_layout(fig, "", height)


def ab_compare_chart(rows: list[dict], height: int = 300):
    """Side by side: annual heat, uniformity penalty and score of the compared layouts."""
    fig = go.Figure()
    names = [r["label"] for r in rows]
    fig.add_trace(go.Bar(x=names, y=[r["energy_mwh"] for r in rows], name="年吸热量 E (MWh)", marker_color=TEAL,
                         text=[f"{r['energy_mwh']:,.1f}" for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[r["weighted_penalty_mwh"] for r in rows], name="均匀性扣分 λΠ (MWh 等效)",
                         marker_color="#d98c7f", text=[f"{r['weighted_penalty_mwh']:,.1f}" for r in rows],
                         textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[r["value_mwh"] for r in rows], name="综合得分 V (MWh 等效)", marker_color=NAVY,
                         text=[f"{r['value_mwh']:,.1f}" for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MWh / MWh 等效")
    return _base_layout(fig, "", height)
