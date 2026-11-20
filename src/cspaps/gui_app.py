"""Graphical interface (Streamlit). Start with ``cspaps gui [CASE.yaml]``.

Workspaces: 运行调度 (operations), 布局规划 (layout planning), 光学校验 (optics,
shared tool), plus 全部参数 and 运行记录. Long tasks run in a background thread
with progress and a cancel button; every run is saved in its own folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from html import escape
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
from cspaps.config import _SECTIONS, case_from_dict, load_case
from cspaps.field import Field, build_field, save_layout_csv
from cspaps.gui_plots import field_map, receiver_map, replay_timeline, uniformity_curves
from cspaps.layout_opt import validate_layout
from cspaps.optics import footprints_at
from cspaps.receiver import build_grid, incident_limit_w_m2, panel_of, surface_normal, surface_point
from cspaps.simulate import evaluate, sun_state

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "cases" / "aksai_demo.yaml"
RESULTS = Path.cwd() / "results" / "gui"
UPLOADS = RESULTS / "_uploads"

SECTION_TITLES = {
    "site": "站址", "receiver": "集热器与管屏", "heliostat": "定日镜", "sun": "太阳", "atmosphere": "大气衰减",
    "weather": "气象数据", "shading": "阴影与遮挡", "field": "镜场", "aps": "瞄准策略 APS",
    "uniformity": "吸热器均匀受热", "engine": "在线控制引擎", "realtime": "回放（云影）", "annual": "全年计算",
    "layout_opt": "布置优化", "mcrt": "蒙特卡洛光线追迹校验",
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
    "aps.auto_grid": "自动细化求解网格", "aps.max_solver_cells": "求解网格上限",
    "uniformity.enabled": "启用均匀度约束", "uniformity.max_cv": "允许二维 CV (ε)",
    "uniformity.mode": "均匀度目标模式", "uniformity.weight": "均匀度权重 λ",
    "uniformity.panels": "考核管屏 H (YAML 列表，空 = 全部)", "uniformity.z_range": "考核高度范围 [z1, z2]",
    "uniformity.min_load_mw": "考核区最低吸收功率 (MW)", "uniformity.mu_iters": "权重二分次数",
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
    "layout_opt.search_aiming": "布局搜索的 APS 目标",
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
    "uniformity.mode": ["cv_limit", "weighted"],
    "annual.method": ["sun_bins", "sun_grid", "representative_days", "hourly"],
    "annual.aiming": ["center", "greedy", "kfactor"], "layout_opt.generator": ["radial_staggered", "biomimetic"],
    "layout_opt.search_sampling": ["sun_bins", "sun_grid", "hourly"],
    "layout_opt.search_aiming": ["auto", "center", "greedy", "kfactor"],
    "layout_opt.review_sampling": ["sun_bins", "sun_grid", "hourly"],
    "receiver.flux_limit_basis": ["incident", "absorbed"],
    "engine.no_forecast_policy": ["clear_sky", "current"], "engine.stale_policy": ["clear_sky", "last"],
    "engine.fallback": ["hold_or_defocus", "defocus", "none"],
    "realtime.scenario": ["clouds", "moving_edge", "sudden_clear"],
    "mcrt.tracking_mode": ["per_ray", "per_heliostat"],
}
CHOICE_LABELS = {
    "sun.shape": {"gaussian": "高斯太阳盘", "pillbox": "均匀圆盘"},
    "sun.dni_model": {"meinel": "Meinel 晴空模型", "constant": "固定 DNI"},
    "atmosphere.model": {"vittitoe_biggs_clear": "清澈大气", "vittitoe_biggs_hazy": "霾天大气",
                         "beer_lambert": "Beer–Lambert 消光"},
    "weather.source": {"clear_sky": "晴空模型", "file": "EPW / DNI 文件"},
    "aps.method": {"greedy": "贪心 APS（推荐）", "kfactor": "瞄准因子法", "center": "中心瞄准基准",
                   "milp": "混合整数精确法（小镜场）"},
    "uniformity.mode": {"cv_limit": "CV 上限约束", "weighted": "固定权重折中"},
    "annual.method": {"sun_bins": "DNI 加权代表太阳位置", "sun_grid": "太阳角度分箱",
                      "representative_days": "每月代表日", "hourly": "逐时复算"},
    "annual.aiming": {"center": "中心瞄准", "greedy": "贪心 APS", "kfactor": "瞄准因子法"},
    "layout_opt.search_aiming": {"auto": "自动（启用均匀度时用 APS）", "center": "中心瞄准快速搜索",
                                 "greedy": "贪心 APS 搜索", "kfactor": "瞄准因子 APS 搜索"},
    "layout_opt.generator": {"radial_staggered": "径向交错布局", "biomimetic": "仿生螺旋布局"},
    "layout_opt.search_sampling": {"sun_bins": "DNI 加权代表太阳位置", "sun_grid": "太阳角度分箱", "hourly": "逐时"},
    "layout_opt.review_sampling": {"sun_bins": "DNI 加权代表太阳位置", "sun_grid": "太阳角度分箱", "hourly": "逐时"},
    "realtime.scenario": {"clouds": "随机云影", "moving_edge": "移动云边", "sudden_clear": "云层突然移开"},
    "engine.no_forecast_policy": {"clear_sky": "晴空 DNI 保守约束", "current": "使用当前 DNI"},
    "engine.stale_policy": {"clear_sky": "晴空 DNI 保守约束", "last": "沿用最近有效数据"},
    "engine.fallback": {"hold_or_defocus": "保持安全旧方案，否则散焦", "defocus": "散焦超限贡献镜", "none": "不自动回退"},
    "receiver.flux_limit_basis": {"incident": "按入射热流限制", "absorbed": "按吸收热流限制"},
    "mcrt.tracking_mode": {"per_ray": "每条光线随机误差", "per_heliostat": "每面镜固定偏差"},
}
HELP = {
    "site.latitude": "场址纬度，北纬为正。用于计算太阳高度角和方位角。",
    "site.longitude": "场址经度，东经为正。当地时间按算例时区解释。",
    "receiver.center": "圆柱吸热器中心的三维坐标 [东向 x, 北向 y, 高度 z]，单位米。",
    "receiver.max_flux_kw_m2": "吸热器表面局部入射热流上限。超限会触发重分配、散焦或不可执行状态。",
    "receiver.panel_absorptivity": "每个管屏把入射热流转成吸收热流的比例。未填写时使用整台吸热器的吸收率。",
    "weather.file": "EPW 文件或两列 CSV（time,dni）。DNI 单位 W/m²，时间需注明时区或按场址时区解释。",
    "aps.method": "瞄准算法为每面镜子选择接收器表面的位置。贪心法适合大镜场，小规模 MILP 可作候选集精确基准。",
    "aps.safety_margin": "相对设备通量上限预留的比例。例如 0.05 表示优化目标不超过上限的 95%。",
    "uniformity.mode": "CV 上限模式直接要求二维 CV 不超过目标。固定权重模式最大化吸收功率 − λ × 考核区面积 × 热流标准差。",
    "uniformity.max_cv": "考核区面积加权二维变异系数上限。CV 越小表示表面热流越均匀。",
    "uniformity.weight": "固定权重 λ。数值越大越重视均匀受热，可能牺牲部分吸收功率。",
    "uniformity.panels": "参与均匀度统计和优化的管屏编号，从 1 开始。留空表示整个圆柱表面。",
    "uniformity.z_range": "考核区域高度 [最低 z, 最高 z]，单位米。留空表示吸热器全高。",
    "uniformity.min_load_mw": "考核区域最低吸收功率。未达到时该运行状态按不满足目标处理。",
    "annual.n_sun_bins": "把全年有效太阳状态压缩成代表点以加速估算。代表点越多，计算越慢，采样误差通常越小。",
    "annual.dni_bands": "每个太阳方向再按 DNI 强弱拆分工况。受通量限制的 APS 建议大于 1。",
    "layout_opt.n_heliostats": "希望最终布置的镜面数量。场地无法容纳时会报告缺额。",
    "layout_opt.search_aiming": "自动模式在启用均匀度时按 APS 年吸收能量搜索，否则用较快的中心瞄准初筛。",
    "layout_opt.refine_top": "对搜索阶段排名靠前的布局运行完整 APS 年收益复核。",
    "engine.max_slew_deg_s": "镜面法向最大转速，用于排除本控制周期内物理上来不及到达的瞄准点。",
    "annual.method": "选用代表工况快速估算，或用全年的逐时天气数据复算。",
    "annual.aiming": "中心瞄准用于快速基准；APS 算法会在每个太阳工况下重新分配瞄准点。",
    "layout_opt.search_sampling": "搜索工况越多通常越准确，但每次布局评估也越慢。",
    "layout_opt.review_sampling": "用更细的太阳和 DNI 工况复核候选布局。",
    "layout_opt.bounds": "生成器参数搜索范围，使用 YAML 映射，例如 radial_spacing: [0.9, 1.4]。",
    "layout_opt.fixed": "固定的镜场生成参数，使用 YAML 映射，例如 spacing_factor: 1.1。",
    "field.generator": "径向交错或仿生螺旋镜场生成参数。先在指定范围内产生候选镜位，再由布局搜索调整系数。",
    "field.boundary": "场地多边形顶点按 x 东向、y 北向填写米制坐标，例如 [[-420,-300],[420,-300],[420,480],[-420,480]]。",
    "layout_opt.min_spacing_factor": "最小中心距系数，乘镜面对角线后得到定日镜之间的最小距离。",
    "layout_opt.swap_rounds": "搜索后的镜位交换轮数；只保留提升 APS 可行年吸收热量的交换。",
}
STATE_ZH = {"ready": "仿真计划通过复核", "fallback": "使用保守回退方案", "infeasible": "无可行运行方案",
            "pending": "待计算", "awaiting_confirmation": "待人工确认", "confirmed": "已记录人工确认", "n/a": "—",
            "ok": "通过复核", "margin_reduced": "设备限值内，安全裕度不足", "unverified": "待复核",
            "uniformity_infeasible": "均匀度或最低负荷未达标"}
UNIFORMITY_ZH = {"weighted": "按权重折中", "feasible": "CV 达标", "not_reached": "CV 超出目标",
                 "below_min_load": "低于最低负荷", "disabled": "未启用", "not_checked": "未考核"}
STRAT_ZH = {"center": "中心瞄准基准", "aps_reactive": "APS · 当前 DNI", "aps_predictive": "APS · 含 DNI 预测"}


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
    help_text = HELP.get(full)
    if full in CHOICES:
        opts = CHOICES[full]
        idx = opts.index(value) if value in opts else 0
        fmt = CHOICE_LABELS.get(full)
        if fmt:
            return st.selectbox(label, opts, index=idx, key=key, format_func=lambda item: fmt.get(item, item), help=help_text)
        return st.selectbox(label, opts, index=idx, key=key, help=help_text)
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key, help=help_text)
    if isinstance(value, int):
        return int(st.number_input(label, value=value, step=1, key=key, help=help_text))
    if isinstance(value, float):
        return float(st.number_input(label, value=value, format="%g", key=key, help=help_text))
    if isinstance(value, str):
        return st.text_input(label, value=value, key=key, help=help_text)
    text = st.text_area(label, value=yaml.safe_dump(value, allow_unicode=True, default_flow_style=None).strip()
                        if value is not None else "", key=key, height=100 if isinstance(value, (dict, list)) else 68,
                        help=help_text or "高级复合参数，使用 YAML 格式。留空表示采用默认值。")
    return yaml.safe_load(text) if text.strip() else None


_DEFAULTS = None


def _defaults():
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = case_from_dict({"site": {"latitude": 0, "longitude": 0},
                                    "receiver": {"center": [0, 0, 1], "diameter": 1, "height": 1},
                                    "heliostat": {"width": 1, "height": 1, "pivot_height": 1}})
    return _DEFAULTS


def _section_form(section: str, expanded: bool = False, fields: list[str] | None = None,
                  title: str | None = None) -> None:
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    cls = _SECTIONS[section]
    current = dict(data.get(section) or {})
    default_obj = dataclasses.asdict(getattr(_defaults(), section))
    with st.expander(title or SECTION_TITLES.get(section, section), expanded=expanded):
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
    download_labels = {"aims_csv": "逐镜瞄准方案 CSV", "flux_png": "吸热器热流图", "summary_json": "计算摘要 JSON",
                       "efficiency_png": "镜场效率图", "layout_csv": "优化镜位 CSV", "history_csv": "搜索历史 CSV",
                       "ranking_csv": "候选排名 CSV", "case_yaml": "算例参数 YAML"}
    cols = st.columns(4)
    for i, (label, path) in enumerate(files.items()):
        path = Path(path)
        if path.exists() and path.is_file():
            with cols[i % 4]:
                st.download_button(f"下载 {download_labels.get(label, path.name)}", path.read_bytes(), file_name=path.name,
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
        self.cancellable = "cancel" in kwargs
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
        if job.cancellable and st.button("取消", key=f"cancel_{key}"):
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
        f'<div class="cs-card"><div class="cs-k">{escape(str(k))}</div><div class="cs-v">{escape(str(v))}</div></div>'
        for k, v in items)
    st.markdown(f'<div class="cs-grid">{cards}</div>', unsafe_allow_html=True)


def _format_number(value, digits: int = 3) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


def _apply_theme() -> None:
    st.markdown(
        "<style>"
        ".stMainBlockContainer{padding-top:3.6rem;max-width:1480px}"
        ".stApp{font-family:Inter,'Microsoft YaHei',sans-serif}"
        "div[data-testid='stToolbar'] button[data-testid='stDeployButton']{display:none!important}"
        "section[data-testid='stSidebar']{border-right:1px solid #738a902e}"
        ".cs-brand{display:flex;align-items:center;gap:.65rem;font:700 1.38rem Inter,sans-serif;letter-spacing:.03em}"
        ".cs-logo{display:inline-block;width:29px;height:29px;border-radius:8px;background:"
        "radial-gradient(circle at 50% 40%,#ffd289 0 17%,transparent 19%),"
        "repeating-radial-gradient(ellipse at 50% 115%,#0b524f 0 4px,#68b8ab 5px 6px,#124e4c 7px 9px)}"
        "section[data-testid='stSidebar'] [role='radiogroup']{gap:.25rem}"
        "section[data-testid='stSidebar'] [role='radiogroup'] label{padding:.48rem .72rem;border-radius:8px}"
        "section[data-testid='stSidebar'] [role='radiogroup'] label:has(input:checked){background:#548e822c;font-weight:650}"
        ".cs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(165px,100%),1fr));gap:10px;margin:.25rem 0 .6rem}"
        ".cs-card{border:1px solid #80948d4a;border-top:3px solid #56998d;border-radius:9px;padding:.8rem 1rem}"
        ".cs-k{font-size:.82rem;opacity:.82}.cs-v{font-size:1.24rem;font-weight:650;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}"
        "div.stButton>button[kind='primary']{background:#14665b;border-color:#14665b;color:white;border-radius:8px}"
        "div.stButton>button[kind='primary']:hover{background:#0b524a;color:white}"
        "div[data-testid='stExpander'] details{border-radius:9px}"
        "@media(max-width:760px){.stMainBlockContainer{padding:3.5rem 1rem 2rem}.cs-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}"
        "</style>", unsafe_allow_html=True)


def page_operations(case) -> None:
    st.header("运行调度")
    st.caption("选择一个时刻生成逐镜瞄准方案，或模拟云层经过镜场时各策略的表现。")
    mode = st.radio("仿真任务", ["云影回放（模拟数据）", "单时刻（导入实测 / 预报 DNI）"], horizontal=True,
                    format_func=lambda item: "云影回放" if item.startswith("云影") else "单时刻瞄准", key="ops_mode")
    if mode.startswith("云影回放"):
        _ops_replay(case)
    else:
        _ops_single(case)


def _ops_replay(case) -> None:
    if case is None:
        st.warning("请先在「全部参数」中修正当前算例。")
        return
    preview, controls = st.columns([1.45, 1], gap="medium")
    with preview:
        if "res_rt" not in st.session_state:
            try:
                field = build_field(case)
                st.plotly_chart(field_map(case, field.xyz, field.ids, title=f"当前镜场 · {field.n:,} 面定日镜",
                                          preview=True), theme=None, key="replay_site_preview")
            except (OSError, ValueError) as exc:
                st.error(f"镜场坐标无法读取。{exc}")
        else:
            st.markdown("**回放工况**")
            last_case = load_case(Path(st.session_state["res_rt"]["run_dir"]) / "inputs_case.yaml")
            st.write(f"{last_case.realtime.start} · {last_case.site.timezone}")
    with controls:
        with st.container(border=True):
            st.subheader("回放设置")
            _section_form("realtime", expanded=True, fields=["scenario", "duration_min", "step_s"], title="云影与时间")
            _job_button("rt", "运行回放", wf.task_realtime, progress=True)
            st.caption("使用当前算例即可运行，结果可按时刻查看。")
    case, _ = _case()
    _job_error("rt")
    advanced = [f.name for f in dataclasses.fields(_SECTIONS["realtime"])
                if f.name not in ("scenario", "duration_min", "step_s")]
    _section_form("realtime", fields=advanced, title="回放时间、预测与云层参数")
    _section_form("engine", title="控制周期与回退策略")
    res = st.session_state.get("res_rt")
    if res is None:
        return
    case = last_case
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
        ("数据来源", CHOICE_LABELS["realtime.scenario"].get(case.realtime.scenario, case.realtime.scenario)),
        ("数据新鲜度", "延迟 %.0f s" % case.realtime.dni_delay_s if case.realtime.dni_delay_s else "实时"),
        ("当前策略", STRAT_ZH.get(strat, strat)),
        ("指令状态", STATE_ZH.get(row["plan_state"], row["plan_state"])),
        ("截获功率", f"{row['p_intercepted_mw']:.3f} MW"),
        ("吸收功率", f"{row['p_absorbed_mw']:.3f} MW"),
        ("最高通量（复核）", f"{row['peak_flux_kw_m2']:.1f} kW/m²，{int(row['hotspot_panel'])} 号管屏"),
        ("距上限余量", f"{(limit_kw - row['peak_flux_kw_m2']):.1f} kW/m²"),
        ("散焦镜数", f"{int(row['n_defocused'])} 面"),
        ("二维 / 周向 / 高度 CV", " / ".join(
            f"{row[k]:.3f}" if np.isfinite(row[k]) else "—"
            for k in ("uniformity_cv_2d", "uniformity_cv_circ", "uniformity_cv_height"))),
        ("均匀度状态", UNIFORMITY_ZH.get(row["uniformity_status"], row["uniformity_status"])),
        ("该时段可行", "是" if row["operational_feasible"] else "否"),
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
    st.plotly_chart(theme=None, figure_or_data=replay_timeline(rr.table, "peak_to_limit", "峰值 / 设备限值", times[k], 1.0), key=f"tl_{k}")
    d1, d2 = st.columns(2)
    with d1:
        sid = st.session_state.get("ops_sel_id")
        if sid is not None and np.any(fld.ids == sid):
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
            st.subheader(f"选中位置 · 方位 {az:.1f}°，高度 {z:.2f} m")
            st.write(f"{int(panel_of(case.receiver, az)) + 1} 号管屏；入射通量 {total:.1f} kW/m²，限值 {lim:.0f} kW/m²")
            st.dataframe(df, hide_index=True)
        else:
            st.caption("点击接收器图查看该点的限值和主要贡献镜")
    st.subheader("策略对比")
    summ = rr.summary()
    comp = pd.DataFrame({STRAT_ZH.get(s, s): {
        "截获热量 (MWhₜₕ)": v["energy_mwh"], "原始吸收热量 (MWhₜₕ)": v["absorbed_energy_mwh"],
        "可行吸收热量 (MWhₜₕ)": v["feasible_absorbed_energy_mwh"],
        "最高复核通量 (kW/m²)": v["max_peak_flux_kw_m2"],
        "均匀度不合格 (min)": round(v["hours_uniformity_failed"] * 60, 2),
        "超限时长 (min)": v["time_over_limit_min"], "平均调整镜数": v["mean_heliostats_moved"],
        "引擎延迟 P50 (s)": v.get("engine_latency_p50_s", "—"), "引擎延迟 P95 (s)": v.get("engine_latency_p95_s", "—"),
        "回退 / 不可执行次数": str({STATE_ZH.get(a, a): b for a, b in v.get("plan_states", {}).items()})}
        for s, v in summ.items()}).astype(str)
    st.dataframe(comp)
    st.caption("每个采样点代表到下一点之间的时段。可行吸收热量只累计通过通量、均匀度和最低负荷检查的时段。")
    with st.expander("下载与全部图表"):
        _show_files(res, images=[])


def _ops_single(case) -> None:
    if case is not None:
        uni = st.session_state["case_dict"].setdefault("uniformity", {})
        goal, setting = st.columns([2, 1])
        current = case.uniformity.mode if case.uniformity.enabled else "power"
        modes = {"power": "优先吸收功率", "weighted": "兼顾功率与均匀度", "cv_limit": "设定均匀度上限"}
        ver = st.session_state["ver"]
        mode = goal.radio("调度目标", list(modes), index=list(modes).index(current), format_func=modes.get,
                          key=f"single_goal:{ver}", horizontal=True)
        uni["enabled"] = mode != "power"
        if uni["enabled"]:
            uni["mode"] = mode
            with setting:
                name = "weight" if mode == "weighted" else "max_cv"
                uni[name] = _widget("uniformity", name, getattr(case.uniformity, name), f"{ver}:uniformity.{name}")
        if mode == "weighted":
            st.caption("提高 λ 会让求解器更重视吸热器表面受热均匀，可能减少吸收功率。")
        elif mode == "cv_limit":
            st.caption("设置可接受的二维 CV 上限。数值越小，对表面均匀受热的要求越高。")
    _section_form("uniformity", fields=["panels", "z_range", "min_load_mw"], title="考核管屏、高度与最低负荷")
    case, case_error = _case()
    case_ok = case is not None
    if case_error:
        st.error(case_error)
    c1, c2, c3 = st.columns(3)
    default_time = pd.Timestamp(case.realtime.start if case is not None else "2025-06-21 13:40")
    case_key = case.name if case is not None else "invalid"
    date = c1.date_input("日期", value=default_time.date(), key=f"s_date:{case_key}")
    tm = c2.time_input("当地时间", value=default_time.time(), key=f"s_time:{case_key}", step=60)
    methods = CHOICES["aps.method"]
    method = c3.selectbox("瞄准算法", methods, index=methods.index(case.aps.method) if case_ok else 0,
                          format_func=lambda item: CHOICE_LABELS["aps.method"].get(item, item),
                          help=HELP["aps.method"], key=f"s_method:{case_key}")
    if method == "center":
        st.caption("中心瞄准用于对照，各定日镜瞄准吸热器朝向该镜一侧的中部。此策略按当前 DNI 复核通量。")
    with st.expander("导入逐镜 DNI、预报或上一轮瞄准点（可选）"):
        st.caption("留空时使用算例中的站址气象数据。DNI 是垂直于太阳光方向的直射辐照度，单位 W/m²。")
        u1, u2, u3 = st.columns(3)
        up_now = u1.file_uploader("当前镜场 DNI（CSV x,y,dni）", type=["csv"], key="s_now",
                                  help="直射法向辐照度，单位 W/m²。空间测点会插值到每面定日镜。")
        up_fc = u2.file_uploader("预测时段最高 DNI（CSV x,y,dni）", type=["csv"], key="s_fc",
                                 help="用于避免云层移开后预测时段内吸热器局部热流超限。")
        up_prev = u3.file_uploader("上一周期已执行的瞄准方案", type=["csv"], key="s_prev",
                                   help="导入上一周期方案 CSV，按定日镜 ID 匹配后用于限制动作幅度。")
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
    run = Path(res["run_dir"])
    case = load_case(run / "inputs_case.yaml")
    inputs = json.loads((run / "run.json").read_text(encoding="utf-8"))["arguments"]
    aps, chk = s["aps"], s["flux_check"]
    method = aps["method"]
    if inputs.get("dni_map"):
        source = f"上传的 DNI 分布 · {Path(inputs['dni_map']).name}"
    elif case.weather.source == "file":
        weather_file = Path(case.weather.file).name.removeprefix("input_weather_")
        source = f"站址气象文件 · {weather_file}"
    else:
        source = "晴空模型"
    uni_status = aps.get("uniformity_status", "not_checked" if case.uniformity.enabled else "disabled")
    eligible = (s["status"] in ("ok", "margin_reduced") and chk["status"] in ("ok", "margin_reduced")
                and uni_status in ("weighted", "feasible", "disabled"))
    executed = st.session_state.get("single_executed") == str(run)
    state = STATE_ZH.get(s["status"], s["status"])
    if chk["status"] == "over_limit":
        state = "本方案通量超限"
    elif uni_status == "not_checked" and s["status"] == "ok":
        state = "均匀度待考核"
    st.subheader("上次计算结果")
    st.caption(f"{s['time'][:19].replace('T', ' ')} · {CHOICE_LABELS['aps.method'][method]} · {source}")
    _status_header([
        ("吸收功率", f"{s['absorbed_power_mw']:.3f} MW"),
        ("考核区二维 CV", _format_number(aps.get("cv_2d_H", s.get("cv_2d")), 4)),
        ("复核热点通量", f"{chk['peak_flux_kw_m2']:.1f} kW/m²"),
        ("方案结论", state),
    ])
    target = "功率优先"
    if case.uniformity.enabled:
        target = (f"CV ≤ {case.uniformity.max_cv:.3f}" if case.uniformity.mode == "cv_limit"
                  else f"λ = {case.uniformity.weight:g}")
    st.caption(f"均匀度目标 {target} · {UNIFORMITY_ZH.get(uni_status, uni_status)}。二维 CV 是考核区吸收热流的标准差与均值之比，越小越均匀。")
    if chk["status"] == "over_limit":
        st.error(f"第 {chk['hotspot']['panel']} 号管屏的复核热点超出设备限值 {chk['exceedance_kw_m2']:.1f} kW/m²。")
    elif s["status"] == "uniformity_infeasible":
        st.warning("本方案未达到均匀度或最低负荷要求，可调整瞄准策略后重新计算。")
    if eligible:
        if executed:
            st.success("已记录人工确认（本次仿真会话）")
        elif st.button("记录为人工已确认"):
            st.session_state["single_executed"] = str(run)
            st.rerun()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(field_map(case, fld.xyz, fld.ids, values=snap.dni, label="DNI (W/m²)", colorscale="Blues",
                                  cmin=0, cmax=max(1.0, float(np.max(snap.dni, initial=0))),
                                  title="各定日镜接收的直射辐照度", inactive=~snap.active), theme=None, key="s_field")
    with right:
        st.plotly_chart(receiver_map(case, snap.grid, snap.flux, "吸热器表面入射热流", hotspot=chk,
                                     aims=(snap.aim_azimuth[snap.active], snap.aim_z[snap.active])),
                        theme=None, key="s_rec")
    with st.expander("复核依据与光学效率"):
        quality = CHOICE_LABELS["aps.method"][method]
        if aps.get("global_optimality_proven") and eligible:
            quality = "候选瞄准点上的全局最优已证明"
        _status_header([
            ("求解方法", quality), ("散焦镜数", f"{s['n_defocused']} 面"),
            ("截获功率", f"{s['intercepted_power_mw']:.3f} MW"),
            ("复核热点", f"第 {chk['hotspot']['panel']} 号管屏 · 高度 {chk['hotspot']['z_m']} m"),
            ("热点距设备上限", f"{chk['margin_to_limit_kw_m2']:.1f} kW/m²"),
            ("周向 CV", _format_number(s.get("cv_circumferential"))),
            ("高度 CV", _format_number(s.get("cv_height"))),
        ])
        st.dataframe(pd.DataFrame({"效率": s["field_efficiency"]}).T)
    _show_files(res, images=[])


# --------------------------------------------------------------------------- planning workspace

def page_planning(case) -> None:
    st.header("布局规划")
    st.caption("在指定场地内比较镜位布置。启用 APS 复核后，按满足约束的年吸收热量给候选方案排名。")
    tabs = st.tabs(["① 场地与设备", "② 气象与采样", "③ 计算", "④ 方案对比与应用"])
    with tabs[0]:
        c1, c2 = st.columns([2, 3])
        with c1:
            up = st.file_uploader("上传镜场坐标 CSV（列 id,x,y,z 或 x,y）", type=["csv"], key="layout_up")
            if up is not None:
                st.session_state["case_dict"].setdefault("field", {})["layout_file"] = str(_save_upload(up))
            for sec in ("site", "receiver", "heliostat", "field", "shading", "atmosphere", "sun"):
                _section_form(sec, expanded=sec in ("receiver", "field"))
            case, _ = _case()
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
            case, _ = _case()
            st.caption("代表工况加速初筛；提高工况数或按 DNI 分档能更细致地覆盖全年运行变化。逐时复算最细，但计算最慢。")
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
            common = ["generator", "n_heliostats", "search_aiming", "refine_top"]
            _section_form("layout_opt", expanded=True, fields=common, title="镜数、布局与复核")
            advanced = [f.name for f in dataclasses.fields(_SECTIONS["layout_opt"]) if f.name not in common]
            _section_form("layout_opt", fields=advanced, title="搜索范围、采样精度与计算资源")
            _section_form("aps")
            _section_form("uniformity")
            case, _ = _case()
        with c2:
            st.markdown("**全年能量（当前镜场）**")
            if case is not None:
                sampling = CHOICE_LABELS["annual.method"][case.annual.method]
                aiming = CHOICE_LABELS["annual.aiming"][case.annual.aiming]
                st.caption(f"{sampling} · {aiming}。在「② 气象与采样」中修改。")
            _job_button("annual", "计算全年能量", wf.task_annual, progress=True)
            _job_error("annual")
            if "res_annual" in st.session_state:
                s = st.session_state["res_annual"]["summary"]
                _metrics([("年吸收热量 (MWhₜₕ)", s["absorbed_energy_mwh"]),
                          ("满足全部约束 (MWhₜₕ)", s["feasible_absorbed_energy_mwh"]),
                          ("年截获热量 (MWhₜₕ)", s["energy_mwh"]["intercept"]),
                          ("最高通量 (kW/m²)" if s["aiming"] == "center" else "最高复核通量 (kW/m²)", s["max_peak_flux_kw_m2"]),
                          ("均匀度不达标时长 (h)", s["hours_uniformity_failed"]),
                          ("通量/复核失败时长 (h)", s["hours_flux_or_verification_failed"]),
                          ("年均光学效率", s["total_optical_efficiency"]), ("计算时间 (s)", s["runtime_s"])])
                st.caption("年度吸收热量为光学热量仿真值。满足全部约束热量仅累计通过通量复核、均匀度和最低负荷检查的工况。")
                st.caption(f"本次结果采用{CHOICE_LABELS['annual.method'][s['method']]}，共 {s['n_sun_states']} 个工况。"
                           + ("中心瞄准基准使用粗网格计算通量。" if s["aiming"] == "center" else ""))
            st.divider()
            st.markdown("**镜场布置优化**（可随时取消，取消后对已找到的方案做复核）")
            _job_button("opt", "开始布置优化", wf.task_optimize_layout, progress=True, cancellable=True)
            _job_error("opt")
            if "res_opt" in st.session_state:
                s = st.session_state["res_opt"]["summary"]
                score = s.get("annual_energy_mwh_with_aps")
                score_label = "可行 APS 年吸收能量" if score is not None else "搜索目标年吸收能量"
                score = score if score is not None else s.get("annual_energy_mwh_search_objective")
                site_status = {"ok": "场地约束通过", "capacity_shortfall": "镜数不足",
                               "constraint_violation": "场地约束未通过"}.get(s["status"], s["status"])
                st.markdown("**本次搜索结果**")
                _status_header([("入选镜数", f"{s['n_heliostats']} 面"), (score_label, f"{score} MWhₜₕ"),
                                ("场地校验", site_status)])
                st.caption("在「④ 方案对比与应用」查看候选排名和各工况的复核结果。本次布局采用启发式搜索。")
                if s.get("operational_status") == "no_feasible_annual_states":
                    st.warning("排名第一的布局在本次年度复核中没有通过全部约束的工况。")
                if s.get("capacity"):
                    st.warning(f"按最密生成参数估计可放置 {s['capacity']['max_placeable_with_densest_parameters']} 面镜，低于目标镜数。")
    with tabs[3]:
        _planning_compare(case)


def _planning_compare(case) -> None:
    res = st.session_state.get("res_opt")
    if res is None:
        st.info("先在 ③ 运行布置优化")
        return
    if case is None:
        st.error("当前算例参数无效，无法比较布局。")
        return
    r = res["result"]
    rank = r.ranking.copy()
    show_cols = [c for c in ["final_rank", "search_rank", "rank_change", "n_heliostats",
                             "energy_search_center_absorbed_mwh", "energy_search_objective_mwh",
                             "energy_review_center_mwh", "energy_hourly_center_mwh", "search_sampling_error_pct",
                             "energy_constrained_aps_mwh", "energy_raw_aps_absorbed_mwh", "aps_max_cv_2d",
                             "aps_max_peak_kw_m2", "aps_hours_uniformity_failed", "aps_states_not_ok"] if c in rank]
    review = res["summary"]["review_sampling"]
    st.markdown(f"**候选方案** · {CHOICE_LABELS['layout_opt.search_aiming'].get(r.best.search_aiming, r.best.search_aiming)}"
                f" · {review['n_states']} 个复核工况")
    display_names = {
        "final_rank": "复核排名", "search_rank": "搜索排名", "rank_change": "排名变化", "n_heliostats": "镜数",
        "energy_search_center_absorbed_mwh": "搜索热量 (MWhₜₕ)",
        "energy_search_objective_mwh": "APS 搜索目标 (MWhₜₕ)",
        "energy_review_center_mwh": "中心瞄准复核 (MWhₜₕ)",
        "energy_hourly_center_mwh": "逐时中心基准 (MWhₜₕ)",
        "search_sampling_error_pct": "搜索采样误差 (%)", "energy_constrained_aps_mwh": "可行 APS 热量 (MWhₜₕ)",
        "energy_raw_aps_absorbed_mwh": "原始 APS 热量 (MWhₜₕ)", "aps_max_cv_2d": "最大二维 CV",
        "aps_max_peak_kw_m2": "最高复核通量 (kW/m²)",
        "aps_hours_uniformity_failed": "均匀度失败工况权重 (h)", "aps_states_not_ok": "未通过工况数",
    }
    view = rank[show_cols].rename(columns=display_names)
    ev = st.dataframe(view, on_select="rerun", selection_mode="single-row", key="cand_table", hide_index=True)
    rows = getattr(getattr(ev, "selection", None), "rows", None) or [0]
    selected_row = rank.iloc[rows[0]]
    search_rank = int(selected_row["search_rank"])
    final_rank = int(selected_row["final_rank"])
    pick = search_rank - 1
    cand = r.candidates[pick]
    is_best = cand is r.best
    xyz = r.best.xyz if is_best else cand.xyz
    fld = Field(xyz)
    spacing_factor = case.layout_opt.min_spacing_factor or case.layout_opt.fixed.get("spacing_factor", 1.1)
    selected_validation = validate_layout(case, xyz, case.layout_opt.n_heliostats, case.heliostat.diagonal * spacing_factor)
    aps_table = cand.aps_result.timeseries if cand.aps_result is not None else None
    feasible_states = int(aps_table["operational_feasible"].sum()) if aps_table is not None else 0
    aps_states = len(aps_table) if aps_table is not None else 0
    eff = (cand.aps_result or cand.result).heliostat_efficiency if (cand.aps_result or cand.result) is not None else None
    if eff is not None and len(eff) != fld.n:
        eff = None
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(theme=None, figure_or_data=field_map(case, fld.xyz, fld.ids, values=eff, label="年均光学效率",
                                  candidates=cand.candidates_xyz,
                                  title=f"复核排名 {final_rank} / 搜索排名 {search_rank}：{fld.n} 面" +
                                        ("（含换位优化）" if is_best else "")),
                        key=f"cand_map_{pick}")
    with c2:
        st.markdown(f"**选中布局**　复核排名 {final_rank} · 搜索排名 {search_rank}")
        site_status = {"ok": "场地约束通过", "capacity_shortfall": "镜数不足", "constraint_violation": "场地约束未通过"}
        st.write(f"{site_status[selected_validation['status']]} · {fld.n} 面镜 · 最小镜距 "
                 f"{selected_validation['min_distance_m']} m")
        if aps_table is None:
            st.warning("该布局仅完成搜索阶段评估，尚未进行 APS 年度复核。")
        elif feasible_states < aps_states:
            st.warning(f"APS 代表工况 {feasible_states}/{aps_states} 个满足全部通量和均匀度约束。")
            if feasible_states == 0:
                st.error("本次年度复核没有找到可行运行工况，不能将该候选标记为可用布局。")
        else:
            st.success(f"APS 代表工况 {feasible_states}/{aps_states} 个满足全部约束。")
        st.caption("镜场坐标采用 x 东、y 北、z 上（m），原点为塔底。载入只切换仿真镜场，不向设备下发指令。")
        selectable = selected_validation["status"] == "ok" and (aps_table is None or feasible_states > 0)
        if st.button("载入此布局继续仿真", key="apply_layout", disabled=not selectable):
            path = Path(res["run_dir"]) / f"applied_layout_{pick + 1}.csv"
            save_layout_csv(fld, path)
            st.session_state["case_dict"].setdefault("field", {})["layout_file"] = str(path.resolve())
            st.session_state["ver"] += 1
            case2, err = _case()
            again = build_field(case2) if case2 else None
            same = again is not None and again.n == fld.n and np.allclose(again.xyz, fld.xyz, atol=1e-6) \
                and np.array_equal(again.ids, fld.ids)
            st.session_state["apply_check"] = {"文件": str(path), "导出镜数": fld.n,
                                               "重新载入镜数": again.n if again else None,
                                               "编号与坐标逐项一致": bool(same),
                                               "被过滤": again.dropped if again else None}
            st.rerun()
        if st.session_state.get("apply_check"):
            check = st.session_state["apply_check"]
            if check["编号与坐标逐项一致"]:
                st.success(f"已载入并复核：{check['重新载入镜数']} 面镜，编号与坐标一致。")
            else:
                st.error(f"布局载入复核失败，导出 {check['导出镜数']} 面，重新载入 {check['重新载入镜数']} 面。")
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
    st.markdown("**均匀受热与功率比较**（固定镜位和 DNI 工况，扫描二维 CV 上限）")
    u1, u2, u3 = st.columns(3)
    alternatives = [int(i) for i in rank["final_rank"] if i != final_rank]
    other = u1.selectbox("比较候选布局", alternatives or [final_rank], key="cmp_other", disabled=not alternatives,
                         format_func=lambda index: (f"复核排名 {int(rank.iloc[index - 1]['final_rank'])} · "
                                                    f"搜索排名 {int(rank.iloc[index - 1]['search_rank'])}"),
                         help="从本次入围的候选镜场中选一个进行均匀度比较。")
    compare_time = str(case.realtime.start) if case is not None else "2025-06-21 13:40"
    t_u = u2.text_input("比较工况时刻", compare_time, key="cmp_t")
    eps_txt = u3.text_input("二维 CV 目标序列", "0.75, 0.7, 0.65, 0.6, 0.55", key="cmp_eps",
                            help="逗号分隔的考核区二维 CV 上限。每个目标都会求一个吸收功率方案。")
    if st.button("计算两方案的功率–均匀度曲线", key="cmp_btn", disabled=not alternatives) and case is not None:
        eps = [float(x) for x in eps_txt.replace("，", ",").split(",") if x.strip()]
        curves = {}
        other_search_rank = int(rank.iloc[other - 1]["search_rank"])
        for name, lay in ((f"复核排名 {final_rank}", xyz),
                          (f"复核排名 {int(rank.iloc[other - 1]['final_rank'])}",
                           r.candidates[other_search_rank - 1].xyz)):
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
    st.caption("蒙特卡洛光线追迹（MCRT）以随机光线独立检查解析光斑模型。功率差、热流图差和峰值统计误差用于评估模型偏差。")
    t1, t2, t3 = st.tabs(["解析模型与光线追迹", "SolTrace 对标", "遮挡筛选复核"])
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        v_default = pd.Timestamp(case.realtime.start if case is not None else "2025-06-21 13:40")
        v_case_key = case.name if case is not None else "invalid"
        d = c1.date_input("日期", value=v_default.date(), key=f"v_date:{v_case_key}")
        tm = c2.time_input("当地时间", value=v_default.time(), key=f"v_time:{v_case_key}", step=300)
        rays = c3.number_input("每面镜子光线数", value=2000, step=500, key="v_rays")
        seeds = c4.number_input("随机种子数", value=3, step=1, min_value=1, key="v_seeds")
        validation_methods = ["center", "greedy", "kfactor"]
        method = st.selectbox("瞄准算法", validation_methods,
                              format_func=lambda item: CHOICE_LABELS["aps.method"].get(item, item),
                              help=HELP["aps.method"], key="v_method")
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
        st.write("导出相同镜场几何、太阳位置、镜面姿态和误差参数，并生成 PySolTrace 驱动脚本供独立复核。")
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
    st.caption("常用场址、吸热器、镜场和 APS 参数可展开编辑；输入框旁的问号说明参数含义和单位。复合列表与高级设置使用 YAML。")
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
    st.caption("每次计算会保存当时的参数副本、结果表和图表。选择记录后可查看结果或载入这份参数继续仿真。")
    runs = wf.list_runs(RESULTS)
    if runs.empty:
        st.info("还没有运行记录")
        return
    task_names = {"layout": "镜场生成", "snapshot": "单时刻基准", "aps": "APS 瞄准优化",
                  "realtime": "云影回放", "annual": "全年评估", "layout_opt": "布局优化",
                  "uniformity": "均匀受热分析", "validate": "光线追迹校验"}
    status_names = {"done": "已完成", "ok": "约束通过", "infeasible": "不可执行",
                    "uniformity_infeasible": "均匀度未达标", "capacity_shortfall": "场地容量不足",
                    "constraint_violation": "几何约束不满足"}
    shown = runs.drop(columns=["path"]).rename(
        columns={"case": "算例", "run": "运行编号", "task": "计算任务", "status": "状态", "finished": "完成时间"})
    shown["计算任务"] = shown["计算任务"].map(lambda value: task_names.get(value, value))
    shown["状态"] = shown["状态"].map(lambda value: status_names.get(value, value))
    ev = st.dataframe(shown, on_select="rerun", selection_mode="single-row", hide_index=True,
                      key="runs_table")
    rows = getattr(getattr(ev, "selection", None), "rows", None) or []
    if rows:
        d = Path(runs.loc[rows[0], "path"])
        task = runs.loc[rows[0], "task"]
        st.subheader(f"{task_names.get(task, task)} · {d.name}")
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
    _apply_theme()
    if "case_dict" not in st.session_state:
        path = _initial_case_path()
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)
    with st.sidebar:
        st.markdown('<div class="cs-brand"><span class="cs-logo" aria-hidden="true"></span>CSP-APS</div>',
                    unsafe_allow_html=True)
        st.caption("镜场设计与运行仿真")
        page = st.radio("工作区", ["运行调度", "布局规划", "光学校验", "全部参数", "运行记录"], key="page")
        st.divider()
        up = st.file_uploader("载入算例 YAML", type=["yaml", "yml"])
        if up is not None and st.session_state.get("loaded_upload") != (up.name, up.size):
            st.session_state["loaded_upload"] = (up.name, up.size)
            _load_into_state(yaml.safe_load(up.getvalue().decode("utf-8")), UPLOADS)
            st.rerun()
        if st.button("恢复阿克塞演示算例"):
            _load_into_state(yaml.safe_load(DEFAULT_CASE.read_text(encoding="utf-8")), DEFAULT_CASE.parent)
            st.rerun()
        case, err = _case()
        if case is not None:
            with st.container(border=True):
                st.markdown(f"**当前算例**　{case.name}")
                st.caption(f"{case.site.latitude:.3f}°N · {case.site.longitude:.3f}°E · {case.site.timezone}")
                st.caption(f"吸热器 Ø {case.receiver.diameter:g} m × 高 {case.receiver.height:g} m · "
                           f"{case.receiver.n_panels} 管屏 · 通量上限 {case.receiver.max_flux_kw_m2:g} kW/m²")
                if case.uniformity.enabled:
                    objective = (f"二维 CV ≤ {case.uniformity.max_cv:g}" if case.uniformity.mode == "cv_limit"
                                 else f"均匀度权重 λ = {case.uniformity.weight:g}")
                    st.caption(f"APS 均匀受热目标　{objective}")
            st.download_button("保存当前参数 (YAML)", case.to_yaml().encode("utf-8"), file_name=f"{case.name}.yaml")
        else:
            st.error(err)
        st.caption(f"结果保存在 {RESULTS}")
    {"运行调度": page_operations, "布局规划": page_planning, "光学校验": page_optics,
     "全部参数": page_parameters, "运行记录": page_runs}[page](case)


main()
