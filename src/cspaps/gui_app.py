"""Graphical interface (Streamlit). Start with ``cspaps gui [CASE.yaml]``.

Workspaces: 运行调度 (operations), 布局规划 (layout planning), 光学校验 (optics,
shared tool), plus 全部参数 and 运行记录. Long tasks run in a background thread
with progress and a cancel button; every run is saved in its own folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time as _time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from cspaps import workflows as wf
from cspaps.config import _SECTIONS, case_from_dict
from cspaps.field import Field, build_field, generate, read_layout, save_layout_csv
from cspaps.gui_plots import field_map, receiver_map, replay_timeline, uniformity_curves
from cspaps.optics import footprints_at
from cspaps.receiver import build_grid, incident_limit_w_m2, panel_of, surface_normal, surface_point
from cspaps.simulate import evaluate, sun_state

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "cases" / "example.yaml"
RESULTS = Path.cwd() / "results" / "gui"
UPLOADS = RESULTS / "_uploads"

SECTION_TITLES = {
    "site": "站址", "receiver": "集热器与管屏", "heliostat": "定日镜", "sun": "太阳", "atmosphere": "大气衰减",
    "weather": "气象数据", "shading": "阴影与遮挡", "field": "镜场", "aps": "瞄准策略 APS",
    "uniformity": "预热器均匀受热", "engine": "在线控制引擎", "realtime": "回放（云影）", "annual": "全年计算",
    "layout_opt": "布置优化", "mcrt": "MCRT 校验",
}
LABELS = {
    "site.latitude": "纬度 (°，北为正)", "site.longitude": "经度 (°，东为正)", "site.elevation_m": "海拔 (m)",
    "site.timezone": "时区",
    "receiver.center": "中心点 [x, y, z] (m)", "receiver.diameter": "直径 (m)", "receiver.height": "高度 (m)",
    "receiver.max_flux_kw_m2": "默认允许通量 (kW/m²)", "receiver.n_azimuth": "周向网格数",
    "receiver.n_height": "高度网格数", "receiver.tower_diameter": "塔身直径 (m，空 = 同集热器)",
    "receiver.absorptivity": "吸收率", "receiver.flux_limit_basis": "限值口径 (incident 入射 / absorbed 吸收)",
    "receiver.n_panels": "管屏数", "receiver.panel_start_azimuth_deg": "1 号管屏起始方位 (°)",
    "receiver.panel_max_flux_kw_m2": "各管屏允许通量 (YAML 列表，kW/m²)",
    "receiver.panel_absorptivity": "各管屏吸收率 (YAML 列表)",
    "receiver.panel_flux_bias": "各管屏 实测/模型 通量比 (反馈，YAML 列表)",
    "receiver.limit_height_profile": "允许通量的高度分布系数 (由下至上，YAML 列表)",
    "heliostat.width": "宽 (m)", "heliostat.height": "高 (m)", "heliostat.pivot_height": "旋转中心高度 (m)",
    "heliostat.reflectivity": "反射率", "heliostat.cleanliness": "清洁度",
    "heliostat.slope_error_mrad": "斜率误差 (mrad)", "heliostat.tracking_error_mrad": "跟踪误差 (mrad)",
    "heliostat.specular_error_mrad": "镜面散射误差 (mrad)",
    "heliostat.focusing": "聚焦方式 (slant_range / flat / 焦距 m)",
    "sun.shape": "太阳形状", "sun.sigma_mrad": "高斯 σ (mrad)", "sun.half_angle_mrad": "圆盘半角 (mrad)",
    "sun.dni_model": "晴空 DNI 模型", "sun.dni_constant_w_m2": "常数 DNI (W/m²)",
    "atmosphere.model": "衰减模型", "atmosphere.extinction_per_km": "消光系数 (1/km)",
    "weather.source": "来源", "weather.file": "文件路径 (EPW / CSV)",
    "shading.enabled": "计算阴影遮挡", "shading.n_points": "每边采样点数", "shading.tower_shadow": "计算塔影",
    "field.layout_file": "镜场坐标 CSV（优先于生成器）", "field.generator": "布置生成器参数 (YAML)",
    "field.boundary": "场地边界多边形 (YAML)", "field.exclusion_radius": "塔周禁排半径 (m)",
    "aps.method": "求解方法", "aps.safety_margin": "安全裕度", "aps.n_height_levels": "候选高度数",
    "aps.height_span": "候选高度覆盖比例", "aps.azimuth_offsets_deg": "候选周向偏移 (°)",
    "aps.passes": "改进轮数", "aps.tie_tolerance": "等效功率容差", "aps.move_penalty_per_m": "移动惩罚 (每米)",
    "aps.allow_defocus": "允许散焦", "aps.milp_time_limit_s": "MILP 时限 (s)", "aps.verify": "细网格复核",
    "aps.verify_max_rounds": "热点约束最多轮数", "aps.verify_max_cells": "复核网格上限",
    "uniformity.enabled": "启用均匀度约束", "uniformity.max_cv": "允许二维 CV (ε)",
    "uniformity.panels": "考核管屏 H (YAML 列表，空 = 全部)", "uniformity.z_range": "考核高度范围 [z1, z2]",
    "uniformity.min_load_mw": "预热器最低负荷 (MW)", "uniformity.mu_iters": "权重二分次数",
    "uniformity.passes": "局部搜索轮数",
    "engine.period_s": "控制周期 (s)", "engine.command_latency_s": "指令延迟 (s)",
    "engine.max_dni_age_s": "DNI 最大数据年龄 (s)", "engine.forecast_margin": "预报裕度",
    "engine.no_forecast_policy": "无预报时的约束 DNI", "engine.stale_policy": "数据过期时的约束 DNI",
    "engine.response_refresh_deg": "光斑缓存刷新角 (°)", "engine.reshade_below_elevation_deg": "复核重算遮挡的高度角上限 (°)",
    "engine.max_slew_deg_s": "镜面转速 (°/s，0 = 不限)", "engine.timeout_s": "求解时限 (s)",
    "engine.priority_dni_change": "优先重算的 DNI 变化比例", "engine.fallback": "复核失败时的回退策略",
    "realtime.start": "开始时间", "realtime.duration_min": "时长 (min)", "realtime.step_s": "刷新周期 (s)",
    "realtime.forecast_horizon_s": "预报时长 (s)", "realtime.forecast_step_s": "预报快照间隔 (s)",
    "realtime.clouds": "云列表 (YAML)", "realtime.random_clouds": "随机云设置 (YAML)",
    "realtime.scenario": "场景", "realtime.dni_delay_s": "DNI 数据延迟 (s)", "realtime.dni_dropout_s": "DNI 丢失时段 [起, 止] (s)",
    "realtime.strategies": "对比策略 (YAML)",
    "annual.method": "计算方法", "annual.n_sun_bins": "代表太阳位置数", "annual.dni_bands": "每个代表点的 DNI 分档数",
    "annual.grid_elevation_step_deg": "分箱高度角步长 (°)", "annual.grid_azimuth_step_deg": "分箱方位角步长 (°)",
    "annual.workers": "并行进程 (-1 = 全部)", "annual.year": "年份",
    "annual.day_of_month": "代表日（每月几号）", "annual.time_step_min": "时间步长 (min)",
    "annual.min_sun_elevation_deg": "最低太阳高度角 (°)", "annual.aiming": "瞄准方式",
    "layout_opt.generator": "布置生成器", "layout_opt.n_heliostats": "镜子数量",
    "layout_opt.bounds": "优化参数范围 (YAML)", "layout_opt.fixed": "固定参数 (YAML)",
    "layout_opt.search_sampling": "搜索阶段采样", "layout_opt.n_sun_bins": "搜索代表点数",
    "layout_opt.review_sampling": "复核阶段采样", "layout_opt.review_n_sun_bins": "复核代表点数",
    "layout_opt.review_dni_bands": "复核 DNI 分档数", "layout_opt.review_check_hourly": "复核时与逐时对比误差",
    "layout_opt.min_spacing_factor": "最小镜距系数 (空 = 用 spacing_factor)",
    "layout_opt.swap_rounds": "换位微调轮数", "layout_opt.swap_fraction": "每轮换位比例",
    "layout_opt.shading_points": "阴影采样点数", "layout_opt.maxiter": "进化代数", "layout_opt.popsize": "种群规模系数",
    "layout_opt.seed": "随机种子", "layout_opt.workers": "并行进程 (-1 = 全部)",
    "layout_opt.refine_top": "复核前几名", "layout_opt.polish_rounds": "局部微调轮数",
    "mcrt.rays_per_heliostat": "每面镜子光线数", "mcrt.seed": "随机种子", "mcrt.n_seeds": "重复次数（种子数）",
    "mcrt.tracking_mode": "跟踪误差 (per_ray 统计 / per_heliostat 逐镜偏置)",
}
CHOICES = {
    "sun.shape": ["gaussian", "pillbox"], "sun.dni_model": ["meinel", "constant"],
    "atmosphere.model": ["vittitoe_biggs_clear", "vittitoe_biggs_hazy", "beer_lambert"],
    "weather.source": ["clear_sky", "file"], "aps.method": ["greedy", "kfactor", "center", "milp"],
    "annual.method": ["sun_bins", "sun_grid", "representative_days", "hourly"],
    "annual.aiming": ["center", "greedy", "kfactor"], "layout_opt.generator": ["radial_staggered", "biomimetic"],
    "layout_opt.search_sampling": ["sun_bins", "sun_grid", "hourly"],
    "layout_opt.review_sampling": ["sun_bins", "sun_grid", "hourly"],
    "receiver.flux_limit_basis": ["incident", "absorbed"],
    "engine.no_forecast_policy": ["clear_sky", "current"], "engine.stale_policy": ["clear_sky", "last"],
    "engine.fallback": ["hold_or_defocus", "defocus", "none"],
    "realtime.scenario": ["clouds", "moving_edge", "sudden_clear"],
    "mcrt.tracking_mode": ["per_ray", "per_heliostat"],
}
STATE_ZH = {"ready": "已执行", "fallback": "已回退", "infeasible": "不可执行", "pending": "待复核", "n/a": "—",
            "ok": "可行", "margin_reduced": "可行（裕度不足）"}
STRAT_ZH = {"center": "不做 APS", "aps_reactive": "APS（仅当前 DNI）", "aps_predictive": "APS + DNI 预报"}


# --------------------------------------------------------------------------- state & helpers

def _initial_case_path() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return Path(args.case) if args.case else DEFAULT_CASE


def _load_into_state(data: dict, base_dir: Path) -> None:
    st.session_state["case_dict"] = data
    st.session_state["base_dir"] = str(base_dir)
    st.session_state["ver"] = st.session_state.get("ver", 0) + 1
    for k in [k for k in st.session_state if k.startswith("res_") or k.startswith("job_")]:
        del st.session_state[k]


def _case():
    try:
        return case_from_dict(st.session_state["case_dict"], base_dir=st.session_state["base_dir"]), None
    except Exception as exc:  # noqa: BLE001 - show any validation problem to the user
        return None, str(exc)


def _save_upload(upload) -> Path:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / upload.name
    path.write_bytes(upload.getvalue())
    return path


def _widget(section: str, name: str, value, key: str):
    full = f"{section}.{name}"
    label = LABELS.get(full, name)
    if full in CHOICES:
        opts = CHOICES[full]
        idx = opts.index(value) if value in opts else 0
        return st.selectbox(label, opts, index=idx, key=key)
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if isinstance(value, int):
        return int(st.number_input(label, value=value, step=1, key=key))
    if isinstance(value, float):
        return float(st.number_input(label, value=value, format="%g", key=key))
    if isinstance(value, str):
        return st.text_input(label, value=value, key=key)
    text = st.text_area(label, value=yaml.safe_dump(value, allow_unicode=True, default_flow_style=None).strip()
                        if value is not None else "", key=key, height=100 if isinstance(value, (dict, list)) else 68)
    return yaml.safe_load(text) if text.strip() else None


_DEFAULTS = None


def _defaults():
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = case_from_dict({"site": {"latitude": 0, "longitude": 0},
                                    "receiver": {"center": [0, 0, 1], "diameter": 1, "height": 1},
                                    "heliostat": {"width": 1, "height": 1, "pivot_height": 1}})
    return _DEFAULTS


def _section_form(section: str, expanded: bool = False, fields: list[str] | None = None) -> None:
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    cls = _SECTIONS[section]
    current = dict(data.get(section) or {})
    default_obj = dataclasses.asdict(getattr(_defaults(), section))
    with st.expander(SECTION_TITLES.get(section, section), expanded=expanded):
        cols = st.columns(2)
        new = dict(current)
        names = [f.name for f in dataclasses.fields(cls) if fields is None or f.name in fields]
        for i, name in enumerate(names):
            value = current.get(name, default_obj.get(name))
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(default_obj.get(name), float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            with cols[i % 2]:
                try:
                    new[name] = _widget(section, name, value, f"{ver}:{section}.{name}")
                except yaml.YAMLError as exc:
                    st.error(f"{name}: YAML 格式错误 {exc}")
        data[section] = {k: v for k, v in new.items() if k in current or v != default_obj.get(k)}


def _show_files(result: dict, images: list[str] | None = None) -> None:
    files = result["files"]
    for label in images if images is not None else [k for k in files if str(files[k]).endswith(".png")]:
        if label in files and Path(files[label]).exists():
            st.image(str(files[label]), width="stretch")
    with st.expander("结果数据 (JSON)"):
        st.json(result["summary"])
    cols = st.columns(4)
    for i, (label, path) in enumerate(files.items()):
        path = Path(path)
        if path.exists() and path.is_file():
            with cols[i % 4]:
                st.download_button(f"下载 {path.name}", path.read_bytes(), file_name=path.name,
                                   key=f"dl:{st.session_state['ver']}:{path}")
    if result.get("run_dir"):
        st.caption(f"运行记录：{result['run_dir']}")


def _metrics(pairs: list[tuple[str, object]], per_row: int = 5) -> None:
    _status_header(pairs)


# --------------------------------------------------------------------------- background jobs

class Job:
    def __init__(self, fn, *args, **kwargs):
        self.messages: list[str] = []
        self.result = None
        self.error: str | None = None
        self.cancel = threading.Event()
        self.started = _time.time()
        if "progress" in kwargs:
            kwargs["progress"] = self.messages.append
        if "cancel" in kwargs:
            kwargs["cancel"] = self.cancel.is_set
        self.thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        if os.environ.get("CSPAPS_GUI_SYNC"):  # tests: run in the script thread
            self._run(fn, args, kwargs)
        else:
            self.thread.start()

    def _run(self, fn, args, kwargs):
        try:
            self.result = fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            self.error = traceback.format_exc()

    @property
    def done(self) -> bool:
        return not self.thread.is_alive()

    def collect(self, key: str) -> None:
        if self.error:
            st.session_state[f"err_{key}"] = self.error
        else:
            st.session_state[f"res_{key}"] = self.result
            st.session_state.pop(f"err_{key}", None)


def _job_button(key: str, label: str, fn, *args, cancellable: bool = False, **kwargs) -> None:
    """Start a background job; the result lands in st.session_state['res_<key>']."""
    case, err = _case()
    if err:
        st.error(f"参数有误：{err}")
        return
    job = st.session_state.get(f"job_{key}")
    if job is None or job.done:
        if st.button(label, key=f"btn_{key}", type="primary"):
            if cancellable:
                kwargs["cancel"] = True
            job = Job(fn, case, RESULTS, *args, **kwargs)
            if os.environ.get("CSPAPS_GUI_SYNC"):
                job.collect(key)
            else:
                st.session_state[f"job_{key}"] = job
            st.rerun()
    if st.session_state.get(f"job_{key}") is not None:
        _job_status(key)


@st.fragment(run_every=1.0)
def _job_status(key: str) -> None:
    job = st.session_state.get(f"job_{key}")
    if job is None:
        return
    if not job.done:
        msg = job.messages[-1] if job.messages else "计算中…"
        st.info(f"⏳ {msg}（已用 {(_time.time() - job.started):.0f} s）")
        if st.button("取消", key=f"cancel_{key}"):
            job.cancel.set()
        return
    del st.session_state[f"job_{key}"]
    job.collect(key)
    st.rerun(scope="app")


def _job_error(key: str) -> None:
    if st.session_state.get(f"err_{key}"):
        st.error("计算失败")
        with st.expander("详细错误"):
            st.code(st.session_state[f"err_{key}"])


# --------------------------------------------------------------------------- operations workspace

def _contributors(case, fld, t, dni, aim, active, az, z, top=6):
    ts, el, s_az, _ = sun_state(case, t)
    snap = evaluate(case, fld, el, s_az, dni, aim=aim, active=active)
    b = snap.beams
    if b is None or not len(b["power"]):
        return pd.DataFrame(), 0.0
    p = surface_point(case.receiver, np.array([az]), np.array([z]))
    f = footprints_at(p, surface_normal(np.array([az])), b["aim_xyz"], b["t"], b["sigma"], b["power"])[:, 0]
    order = np.argsort(-f)[:top]
    idx = b["index"][order]
    df = pd.DataFrame({"镜子 ID": fld.ids[idx], "贡献通量 (kW/m²)": np.round(f[order] / 1e3, 2),
                       "占比": np.round(f[order] / max(f.sum(), 1e-9), 3), "DNI (W/m²)": np.round(np.asarray(dni)[idx], 1)})
    return df, float(f.sum() / 1e3)


def _status_header(items: list[tuple[str, object]]) -> None:
    """Status cards that wrap on narrow windows and never truncate values or units."""
    cards = "".join(
        f'<div class="cs-card"><div class="cs-k">{k}</div><div class="cs-v">{v}</div></div>' for k, v in items)
    st.markdown(
        "<style>.cs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin:4px 0 12px}"
        ".cs-card{border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:8px 10px}"
        ".cs-k{font-size:.8rem;opacity:.75}.cs-v{font-size:1.15rem;font-weight:600;overflow-wrap:anywhere;line-height:1.25}"
        "</style>" + f'<div class="cs-grid">{cards}</div>', unsafe_allow_html=True)


def page_operations(case) -> None:
    st.header("运行调度")
    mode = st.radio("数据来源", ["云影回放（模拟数据）", "单时刻（导入实测 / 预报 DNI）"], horizontal=True, key="ops_mode")
    if mode.startswith("云影回放"):
        _ops_replay(case)
    else:
        _ops_single(case)


def _ops_replay(case) -> None:
    with st.expander("回放设置", expanded="res_rt" not in st.session_state):
        _section_form("realtime", expanded=True)
        _section_form("engine")
    _job_button("rt", "运行回放", wf.task_realtime, progress=True)
    _job_error("rt")
    res = st.session_state.get("res_rt")
    if res is None:
        return
    rr, fld = res["result"], res["field"]
    times = [f["time"] for f in rr.frames]
    strategies = list(rr.frames[0]["plans"]) if rr.frames else []
    c1, c2 = st.columns([2, 3])
    strat = c1.radio("显示策略", strategies, format_func=lambda s: STRAT_ZH.get(s, s), horizontal=True, key="ops_strat")
    labels = [f"{t:%H:%M:%S}" for t in times]
    k = labels.index(c2.select_slider("回放时间", options=labels, value=labels[0], key="ops_t"))
    frame, row = rr.frames[k], rr.table[(rr.table["time"] == times[k]) & (rr.table["strategy"] == strat)].iloc[0]
    plan = frame["plans"][strat]
    chk = plan["check"] or {}
    limit_kw = (row["peak_flux_kw_m2"] / row["peak_to_limit"]) if row["peak_to_limit"] > 0 else float("nan")
    _status_header([
        ("时间", f"{times[k]:%Y-%m-%d %H:%M:%S}"),
        ("数据来源", f"模拟云影 ({case.realtime.scenario})"),
        ("数据新鲜度", "延迟 %.0f s" % case.realtime.dni_delay_s if case.realtime.dni_delay_s else "实时"),
        ("当前策略", STRAT_ZH.get(strat, strat)),
        ("指令状态", STATE_ZH.get(row["plan_state"], row["plan_state"])),
        ("截获功率", f"{row['p_intercepted_mw']:.3f} MW"),
        ("吸收功率", f"{row['p_absorbed_mw']:.3f} MW"),
        ("最高通量（复核）", f"{row['peak_flux_kw_m2']:.1f} kW/m²，{int(row['hotspot_panel'])} 号管屏"),
        ("距上限余量", f"{(limit_kw - row['peak_flux_kw_m2']):.1f} kW/m²"),
        ("散焦镜数", f"{int(row['n_defocused'])} 面"),
    ])
    grid = build_grid(case.receiver)
    left, right = st.columns([1, 1])
    with left:
        sel_id = st.session_state.get("ops_sel_id")
        fig = field_map(case, fld.xyz, fld.ids, values=frame["dni"], label="DNI (W/m²)", colorscale="Blues",
                        cmin=0, title="镜场 DNI（图层：模拟）", selected=sel_id,
                        hover=[f"ID {i}<br>DNI {d:.0f} W/m²" for i, d in zip(fld.ids, frame["dni"])],
                        inactive=~plan["active"])
        ev = st.plotly_chart(theme=None, figure_or_data=fig, on_select="rerun", selection_mode=("points",), key=f"ops_field_{k}_{strat}")
        pts = getattr(getattr(ev, "selection", None), "points", None) or []
        if pts and pts[0].get("customdata") is not None:
            st.session_state["ops_sel_id"] = int(np.atleast_1d(pts[0]["customdata"])[0])
    with right:
        sel_cell = st.session_state.get("ops_sel_cell")
        fig = receiver_map(case, grid, plan["flux"], title="接收器展开通量（粗网格；× = 细网格复核热点）",
                           hotspot=chk if chk else None, aims=(plan["aim_azimuth"], plan["aim_z"]), selected_cell=sel_cell)
        ev = st.plotly_chart(theme=None, figure_or_data=fig, on_select="rerun", selection_mode=("points",), key=f"ops_rec_{k}_{strat}")
        pts = getattr(getattr(ev, "selection", None), "points", None) or []
        if pts:
            st.session_state["ops_sel_cell"] = (float(pts[0]["x"]), float(pts[0]["y"]))
    st.plotly_chart(theme=None, figure_or_data=replay_timeline(rr.table, "peak_to_limit", "复核峰值 / 限值", times[k], 1.0), key=f"tl_{k}")
    d1, d2 = st.columns(2)
    with d1:
        sid = st.session_state.get("ops_sel_id")
        if sid is not None:
            i = int(np.flatnonzero(fld.ids == sid)[0])
            prev = rr.frames[k - 1]["plans"][strat] if k > 0 else plan
            st.subheader(f"镜子 {sid}")
            st.table(pd.DataFrame({
                "项目": ["DNI (W/m²)", "原瞄准点 (方位°, 高度 m)", "计划瞄准点 (方位°, 高度 m)", "状态"],
                "值": [f"{frame['dni'][i]:.1f}", f"{prev['aim_azimuth'][i]:.2f}, {prev['aim_z'][i]:.2f}",
                      f"{plan['aim_azimuth'][i]:.2f}, {plan['aim_z'][i]:.2f}",
                      "工作" if plan["active"][i] else ("散焦：回退策略" if row["plan_state"] == "fallback" else "散焦：通量限值")],
            }))
        else:
            st.caption("点击镜场图中的镜子查看明细")
    with d2:
        cell = st.session_state.get("ops_sel_cell")
        if cell is not None:
            az, z = cell
            lim = float(incident_limit_w_m2(case.receiver, az, z)) / 1e3
            df, total = _contributors(case, fld, times[k], frame["dni"], (plan["aim_azimuth"], plan["aim_z"]),
                                      plan["active"], az, z)
            st.subheader(f"热点 方位 {az:.1f}°，高度 {z:.2f} m")
            st.write(f"{int(panel_of(case.receiver, az)) + 1} 号管屏；入射通量 {total:.1f} kW/m²，限值 {lim:.0f} kW/m²")
            st.dataframe(df, hide_index=True)
        else:
            st.caption("点击接收器图查看该点的限值和主要贡献镜")
    st.subheader("策略对比")
    summ = rr.summary()
    comp = pd.DataFrame({STRAT_ZH.get(s, s): {
        "能量 (MWh)": v["energy_mwh"], "最大复核峰值 (kW/m²)": v["max_peak_flux_kw_m2"],
        "超限时长 (min)": v["time_over_limit_min"], "平均调整镜数": v["mean_heliostats_moved"],
        "引擎延迟 P50 (s)": v.get("engine_latency_p50_s", "—"), "引擎延迟 P95 (s)": v.get("engine_latency_p95_s", "—"),
        "回退 / 不可执行次数": str({STATE_ZH.get(a, a): b for a, b in v.get("plan_states", {}).items()})}
        for s, v in summ.items()}).astype(str)
    st.dataframe(comp)
    st.caption(next(iter(summ.values()))["over_limit_basis"] if summ else "")
    with st.expander("下载与全部图表"):
        _show_files(res, images=[])


def _ops_single(case) -> None:
    case_ok = case is not None
    c1, c2, c3 = st.columns(3)
    date = c1.date_input("日期", value=pd.Timestamp("2025-06-21").date(), key="s_date")
    tm = c2.time_input("当地时间", value=pd.Timestamp("13:40").time(), key="s_time", step=60)
    method = c3.selectbox("方法", ["greedy", "kfactor", "center", "milp"], key="s_method")
    st.markdown("**数据导入**（实测 DNI、预报 DNI、上一轮已执行瞄准点）")
    u1, u2, u3 = st.columns(3)
    up_now = u1.file_uploader("当前 DNI（CSV x,y,dni）", type=["csv"], key="s_now")
    up_fc = u2.file_uploader("预报最大 DNI（CSV x,y,dni）", type=["csv"], key="s_fc")
    up_prev = u3.file_uploader("上一轮瞄准点（aims CSV）", type=["csv"], key="s_prev")
    if case_ok:
        fld = build_field(case)
        rows = []
        for label, up in (("当前 DNI", up_now), ("预报 DNI", up_fc)):
            if up is not None:
                df = pd.read_csv(up)
                ok = {"x", "y", "dni"} <= set(df.columns)
                rows.append({"数据": label, "测点数": len(df), "覆盖": f"{len(df)} 点 → {fld.n} 面镜（插值）",
                             "范围 (W/m²)": f"{df['dni'].min():.0f}–{df['dni'].max():.0f}" if ok else "缺列",
                             "校验": "通过" if ok and df["dni"].ge(0).all() else "失败"})
                up.seek(0)
        if up_prev is not None:
            df = pd.read_csv(up_prev)
            matched = int(np.isin(fld.ids, df["id"]).sum()) if "id" in df else 0
            rows.append({"数据": "上一轮瞄准点", "测点数": len(df), "覆盖": f"按 ID 匹配 {matched}/{fld.n}",
                         "范围 (W/m²)": "—", "校验": "通过" if matched == fld.n else "ID 不完整"})
            up_prev.seek(0)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    stamp = f"{date} {tm}"
    _job_button("single", "计算瞄准方案", wf.task_aim, stamp, method=method,
                dni_map=_save_upload(up_now) if up_now else None,
                forecast_dni_map=_save_upload(up_fc) if up_fc else None,
                prev_aims=_save_upload(up_prev) if up_prev else None)
    _job_error("single")
    res = st.session_state.get("res_single")
    if res is None:
        return
    s, snap, fld = res["summary"], res["snapshot"], res["field"]
    executed = st.session_state.get("single_executed") == str(res["run_dir"])
    state = "ready" if executed else ("infeasible" if s["status"] == "infeasible" else "pending")
    chk = s["flux_check"]
    _status_header([
        ("时间", s["time"][:19].replace("T", " ")), ("数据来源", "实测上传" if up_now else "晴空模型"),
        ("当前策略", s["aps"].get("method", method)), ("指令状态", STATE_ZH[state]),
        ("复核状态", STATE_ZH.get(s["status"], s["status"])),
        ("截获功率", f"{s['intercepted_power_mw']:.3f} MW"), ("吸收功率", f"{s['absorbed_power_mw']:.3f} MW"),
        ("最高通量（复核）", f"{chk['peak_flux_kw_m2']:.1f} kW/m²，{chk['hotspot']['panel']} 号管屏"),
        ("距上限余量", f"{chk['margin_to_limit_kw_m2']:.1f} kW/m²"), ("散焦镜数", f"{s['n_defocused']} 面"),
    ])
    if s["status"] == "infeasible":
        st.error(f"复核不通过：热点超限 {chk['exceedance_kw_m2']:.1f} kW/m²，位置 方位 {chk['hotspot']['azimuth_deg']}°、"
                 f"高度 {chk['hotspot']['z_m']} m。控制端应进入保守策略。")
    if not executed and s["status"] != "infeasible":
        if st.button("确认执行（记录为上一轮回执）"):
            st.session_state["single_executed"] = str(res["run_dir"])
            st.rerun()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(theme=None, figure_or_data=field_map(case, fld.xyz, fld.ids, values=snap.dni, label="DNI (W/m²)", colorscale="Blues",
                                  title="镜场 DNI（图层：%s）" % ("实测" if up_now else "模拟/晴空"),
                                  inactive=~snap.active), key="s_field")
    with right:
        st.plotly_chart(theme=None, figure_or_data=receiver_map(case, snap.grid, snap.flux, "接收器展开通量", hotspot=chk,
                                     aims=(snap.aim_azimuth, snap.aim_z)), key="s_rec")
    st.dataframe(pd.DataFrame({"效率": s["field_efficiency"]}).T)
    _show_files(res, images=[])


# --------------------------------------------------------------------------- planning workspace

def page_planning(case) -> None:
    st.header("布局规划")
    tabs = st.tabs(["① 场地与设备", "② 气象与采样", "③ 计算", "④ 方案对比与应用"])
    with tabs[0]:
        c1, c2 = st.columns([2, 3])
        with c1:
            up = st.file_uploader("上传镜场坐标 CSV（列 id,x,y,z 或 x,y）", type=["csv"], key="layout_up")
            if up is not None:
                st.session_state["case_dict"].setdefault("field", {})["layout_file"] = str(_save_upload(up))
            for sec in ("site", "receiver", "heliostat", "field", "shading", "atmosphere", "sun"):
                _section_form(sec, expanded=sec in ("receiver", "field"))
        with c2:
            if case is not None:
                try:
                    fld = build_field(case)
                    st.plotly_chart(theme=None, figure_or_data=field_map(case, fld.xyz, fld.ids, title=f"场地图：边界、禁排区、塔位、当前镜位（{fld.n} 面）"),
                                    key="site_map")
                    if fld.dropped:
                        st.warning(f"{len(fld.dropped)} 面镜子因场地检查被过滤：" +
                                   "；".join(f"ID {d['id']}（{d['reason']}）" for d in fld.dropped[:10]))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"无法生成镜场：{exc}")
    with tabs[1]:
        c1, c2 = st.columns([2, 3])
        with c1:
            wup = st.file_uploader("气象文件（EPW 或 CSV time,dni）", type=["epw", "csv"], key="weather_up")
            if wup is not None:
                st.session_state["case_dict"]["weather"] = {"source": "file", "file": str(_save_upload(wup))}
            _section_form("weather", expanded=True)
            _section_form("annual", expanded=True)
            st.caption("代表太阳位置：常用 20 / 30 / 40 个，或填写任意数量；也可按高度角×方位角分箱，或逐小时遍历。")
        with c2:
            check = st.checkbox("计算采样误差（20/30/40 个代表点及分箱，相对逐小时）", key="bins_check")
            _job_button("bins", "预览代表太阳位置", wf.task_sunbins, check=check)
            _job_error("bins")
            if "res_bins" in st.session_state:
                r = st.session_state["res_bins"]
                st.image(str(r["files"]["bins_png"]), width="stretch")
                if "center_aiming_error_vs_hourly" in r["summary"]:
                    st.dataframe(pd.DataFrame(r["summary"]["center_aiming_error_vs_hourly"]), hide_index=True)
    with tabs[2]:
        c1, c2 = st.columns([2, 3])
        with c1:
            _section_form("layout_opt", expanded=True)
            _section_form("aps")
            _section_form("uniformity")
        with c2:
            st.markdown("**全年能量（当前镜场）**")
            a1, a2 = st.columns(2)
            m = a1.selectbox("方式", ["sun_bins", "sun_grid", "hourly", "representative_days"], key="an_m")
            aim = a2.selectbox("瞄准", ["center", "greedy", "kfactor"], key="an_a", help="greedy/kfactor：每个时刻做受限 APS")
            _job_button("annual", "计算全年能量", wf.task_annual, method=m, aiming=aim, progress=True)
            _job_error("annual")
            if "res_annual" in st.session_state:
                s = st.session_state["res_annual"]["summary"]
                _metrics([("年截获能量 (MWh)", s["energy_mwh"]["intercept"]), ("年吸收能量 (MWh)", s["absorbed_energy_mwh"]),
                          ("年均光学效率", s["total_optical_efficiency"]), ("超限小时", s["hours_over_flux_limit"]),
                          ("计算时间 (s)", s["runtime_s"])])
                st.caption(f"采样：{s['sampling']}")
            st.divider()
            st.markdown("**镜场布置优化**（可随时取消，取消后对已找到的方案做复核）")
            _job_button("opt", "开始布置优化", wf.task_optimize_layout, progress=True, cancellable=True)
            _job_error("opt")
            if "res_opt" in st.session_state:
                s = st.session_state["res_opt"]["summary"]
                st.success(f"完成：状态 {s['status']}，{s['n_heliostats']} 面镜，受限全年能量 "
                           f"{s['annual_energy_mwh_with_aps']} MWh；到 ④ 比较与应用")
                if s.get("capacity"):
                    st.warning(s["capacity"]["message"])
    with tabs[3]:
        _planning_compare(case)


def _planning_compare(case) -> None:
    res = st.session_state.get("res_opt")
    if res is None:
        st.info("先在 ③ 运行布置优化")
        return
    r = res["result"]
    rank = r.ranking.copy()
    show_cols = [c for c in ["final_rank", "search_rank", "rank_change", "n_heliostats", "energy_search_center_mwh",
                             "energy_review_center_mwh", "energy_hourly_center_mwh", "search_sampling_error_pct",
                             "energy_constrained_aps_mwh", "aps_max_peak_kw_m2", "aps_states_not_ok"] if c in rank]
    st.markdown(f"**候选方案**（复核：{r.summary()['review_sampling']}，受限 APS 按 DNI 分档）")
    ev = st.dataframe(rank[show_cols], on_select="rerun", selection_mode="single-row", key="cand_table", hide_index=True)
    rows = getattr(getattr(ev, "selection", None), "rows", None) or [0]
    pick = int(rank.loc[rows[0], "search_rank"]) - 1
    cand = r.candidates[pick] if pick < len(r.candidates) else r.best
    is_best = cand is r.best
    xyz = r.best.xyz if is_best else cand.xyz
    fld = Field(xyz)
    eff = (cand.aps_result or cand.result).heliostat_efficiency if (cand.aps_result or cand.result) is not None else None
    if eff is not None and len(eff) != fld.n:
        eff = None
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(theme=None, figure_or_data=field_map(case, fld.xyz, fld.ids, values=eff, label="年均光学效率",
                                  candidates=cand.candidates_xyz,
                                  title=f"方案 {rows[0] + 1}：{fld.n} 面" + ("（含换位微调，最终方案）" if is_best else "")),
                        key=f"cand_map_{pick}")
    with c2:
        st.json({"布置参数": cand.params, "校验": r.validation if is_best else "（仅最终方案做最终校验）",
                 "换位微调": r.swaps if is_best else "—"})
        st.markdown("**应用前预览**：坐标系 x = 东、y = 北、z = 上（m），原点塔底；"
                    f"镜数 {fld.n}；导出 CSV 列 id,x,y,z（微米精度）。")
        if st.button("确认应用为当前镜场", key="apply_layout"):
            path = Path(res["run_dir"]) / f"applied_layout_{pick + 1}.csv"
            save_layout_csv(fld, path)
            st.session_state["case_dict"].setdefault("field", {})["layout_file"] = str(path.resolve())
            st.session_state["ver"] += 1
            case2, err = _case()
            again = build_field(case2) if case2 else None
            xyz2, ids2 = read_layout(path, 0.0)
            same = again is not None and again.n == fld.n and np.allclose(again.xyz, fld.xyz, atol=1e-6) \
                and np.array_equal(again.ids, fld.ids)
            st.session_state["apply_check"] = {"文件": str(path), "导出镜数": fld.n,
                                               "重新载入镜数": again.n if again else None,
                                               "编号与坐标逐项一致": bool(same),
                                               "被过滤": again.dropped if again else None}
            st.rerun()
        if st.session_state.get("apply_check"):
            st.json(st.session_state["apply_check"])
        new_case = case.copy() if case else None
        if new_case is not None:
            new_case.field.layout_file = f"layout_option_{pick + 1}.csv"
            new_case.field.generator = None
            buf = Path(res["run_dir"]) / f"layout_option_{pick + 1}.csv"
            save_layout_csv(fld, buf)
            st.download_button("导出布局 CSV", buf.read_bytes(), file_name=buf.name, key=f"dl_lay_{pick}")
            st.download_button("导出参数 YAML", new_case.to_yaml().encode("utf-8"),
                               file_name=f"case_option_{pick + 1}.yaml", key=f"dl_case_{pick}")
    st.divider()
    st.markdown("**均匀度与功率曲线对比**（同一时刻、同一 ε 序列、同一候选点模型）")
    u1, u2, u3 = st.columns(3)
    other = u1.selectbox("对比方案", list(range(1, len(r.candidates) + 1)), key="cmp_other")
    t_u = u2.text_input("时刻", "2025-06-21 13:40", key="cmp_t")
    eps_txt = u3.text_input("ε 序列", "0.75, 0.7, 0.65, 0.6, 0.55", key="cmp_eps")
    if st.button("计算两方案的功率–均匀度曲线", key="cmp_btn") and case is not None:
        eps = [float(x) for x in eps_txt.replace("，", ",").split(",") if x.strip()]
        curves = {}
        for name, lay in ((f"方案 {rows[0] + 1}", xyz), (f"方案 {other}", r.candidates[other - 1].xyz)):
            c = case.copy()
            p = UPLOADS / f"cmp_{name}.csv"
            UPLOADS.mkdir(parents=True, exist_ok=True)
            save_layout_csv(Field(lay), p)
            c.field.layout_file, c.field.generator = str(p), None
            with st.spinner(f"{name} …"):
                curves[name] = wf.task_uniformity(c, RESULTS, t_u, eps_values=eps)["summary"]["curve"]
        st.session_state["cmp_curves"] = curves
    if st.session_state.get("cmp_curves"):
        st.plotly_chart(theme=None, figure_or_data=uniformity_curves(st.session_state["cmp_curves"]), key="cmp_plot")
    with st.expander("搜索历史与全部输出"):
        _show_files(res)


# --------------------------------------------------------------------------- optics & other pages

def page_optics(case) -> None:
    st.header("光学校验")
    t1, t2, t3 = st.tabs(["解析 vs 追迹", "SolTrace 对标导出", "遮挡筛选复核"])
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        d = c1.date_input("日期", value=pd.Timestamp("2025-06-21").date(), key="v_date")
        tm = c2.time_input("当地时间", value=pd.Timestamp("13:40").time(), key="v_time", step=300)
        rays = c3.number_input("每面镜子光线数", value=2000, step=500, key="v_rays")
        seeds = c4.number_input("随机种子数", value=3, step=1, min_value=1, key="v_seeds")
        method = st.selectbox("瞄准", ["center", "greedy", "kfactor"], key="v_method")
        conv = st.checkbox("光线数收敛表（500/1000/2000/4000）", key="v_conv")
        _job_button("val", "运行校验", wf.task_validate, f"{d} {tm}", rays=int(rays), method=method, seeds=int(seeds),
                    convergence_rays=[500, 1000, 2000, 4000] if conv else None)
        _job_error("val")
        if "res_val" in st.session_state:
            r = st.session_state["res_val"]
            s = r["summary"]
            st_ = s["mcrt_statistics"]
            _metrics([("功率相对差（解析/追迹 − 1）", s["power_on_receiver_mw"]["rel_diff"]),
                      ("通量图 NRMSE", s["flux_map_nrmse"]), ("相关系数", s["flux_map_correlation"]),
                      ("追迹峰值标准误 (kW/m²)", round(st_["peak_flux_kw_m2"]["std_error"], 2)),
                      ("峰值格相对标准误", round(st_["relative_std_error_at_peak_cell"], 4))])
            st.caption(f"追迹引擎 {st_['engine']}，种子 {st_['seeds']}，每镜 {st_['rays_per_heliostat']} 条光线，"
                       f"网格 {st_['grid'][0]}×{st_['grid'][1]}；差值方向：{s['difference_direction']}；三图同一网格与色标。")
            st.dataframe(pd.DataFrame(s["efficiency_by_stage"]).T)
            if s.get("convergence"):
                st.dataframe(pd.DataFrame(s["convergence"]), hide_index=True)
            _show_files(r)
    with t2:
        st.write("导出同场几何、太阳位置、逐镜姿态和统一误差定义，附 PySolTrace 驱动脚本；在装有 SolTrace v3.4.0 的电脑上运行。")
        c1, c2 = st.columns(2)
        t_s = c1.text_input("时刻", "2025-06-21 13:40", key="st_t")
        m_s = c2.selectbox("瞄准", ["center", "greedy"], key="st_m")
        _job_button("st", "导出对标算例", wf.task_soltrace_export, t_s, method=m_s)
        _job_error("st")
        if "res_st" in st.session_state:
            st.code(st.session_state["res_st"]["summary"]["next_step"])
            _show_files(st.session_state["res_st"], images=[])
    with t3:
        st.write("在低太阳高度角、密集镜位、远端镜和边缘瞄准四种工况下，用全对暴力求交复核当前的空间筛选是否漏判。")
        _job_button("shc", "运行复核", wf.task_shading_check)
        _job_error("shc")
        if "res_shc" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["res_shc"]["summary"]["cases"]), hide_index=True)


def page_parameters(case) -> None:
    st.header("全部参数")
    for sec in _SECTIONS:
        _section_form(sec)
    with st.expander("高级：直接编辑 YAML"):
        text = st.text_area("算例 YAML", yaml.safe_dump(st.session_state["case_dict"], allow_unicode=True, sort_keys=False),
                            height=400, key=f"yaml_{st.session_state['ver']}")
        if st.button("应用 YAML"):
            _load_into_state(yaml.safe_load(text), Path(st.session_state["base_dir"]))
            st.rerun()


def page_runs(case) -> None:
    st.header("运行记录")
    runs = wf.list_runs(RESULTS)
    if runs.empty:
        st.info("还没有运行记录")
        return
    ev = st.dataframe(runs.drop(columns=["path"]), on_select="rerun", selection_mode="single-row", hide_index=True,
                      key="runs_table")
    rows = getattr(getattr(ev, "selection", None), "rows", None) or []
    if rows:
        d = Path(runs.loc[rows[0], "path"])
        st.subheader(d.name)
        for j in sorted(d.glob("*.json")):
            with st.expander(j.name):
                st.json(json.loads(j.read_text(encoding="utf-8")))
        for img in sorted(d.glob("*.png")):
            st.image(str(img), caption=img.name, width="stretch")
        if (d / "inputs_case.yaml").exists():
            if st.button("载入这次运行的输入参数"):
                _load_into_state(yaml.safe_load((d / "inputs_case.yaml").read_text(encoding="utf-8")), d)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="CSP-APS 镜场与瞄准优化", layout="wide")
    if "case_dict" not in st.session_state:
        path = _initial_case_path()
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)
    with st.sidebar:
        st.title("CSP-APS")
        page = st.radio("工作区", ["运行调度", "布局规划", "光学校验", "全部参数", "运行记录"], key="page")
        st.divider()
        up = st.file_uploader("载入算例 YAML", type=["yaml", "yml"])
        if up is not None and st.session_state.get("loaded_upload") != (up.name, up.size):
            st.session_state["loaded_upload"] = (up.name, up.size)
            _load_into_state(yaml.safe_load(up.getvalue().decode("utf-8")), UPLOADS)
            st.rerun()
        if st.button("恢复示例算例"):
            _load_into_state(yaml.safe_load(DEFAULT_CASE.read_text(encoding="utf-8")), DEFAULT_CASE.parent)
            st.rerun()
        case, err = _case()
        if case is not None:
            st.download_button("保存当前参数 (YAML)", case.to_yaml().encode("utf-8"), file_name=f"{case.name}.yaml")
        else:
            st.error(err)
        st.caption(f"结果保存在 {RESULTS}")
    {"运行调度": page_operations, "布局规划": page_planning, "光学校验": page_optics,
     "全部参数": page_parameters, "运行记录": page_runs}[page](case)


main()
