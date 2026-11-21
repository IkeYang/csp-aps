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


def field_map(case: Case, xyz: np.ndarray, ids: np.ndarray, values=None, label: str = "", colorscale="Viridis",
              cmin=None, cmax=None, selected: int | None = None, title: str = "", hover: list[str] | None = None,
              candidates: np.ndarray | None = None, inactive: np.ndarray | None = None, preview: bool = False):
    rec, spec = case.receiver, case.field
    fig = go.Figure()
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
    fig.add_trace(go.Scattergl(x=xyz[:, 0], y=xyz[:, 1], mode="markers", marker=marker, customdata=ids,
                               text=text, hovertemplate="%{text}<extra></extra>", name="定日镜"))
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
    cv = [r["cv_circ_weighted"] for r in rows]
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
    fig.update_xaxes(title="管屏间吸热 CV（逐工况能量加权，越小越均匀）", autorange="reversed")
    fig.update_yaxes(title="全年理论吸收热量 (MWhₜₕ)")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, "", height)
