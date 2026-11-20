"""Graphical interface (Streamlit). Start with ``cspaps gui [CASE.yaml]``.

Workspaces: 总览 (overview), 瞄准优化 (problem 1: aim points at one instant,
with optimality certificate and the power-uniformity trade-off), 布局优化
(problem 2: field layout with an upper bound on any layout), 云影回放,
光学校验, 参数设置, 运行记录 and 方法与理论. Long tasks run in a background
thread with progress and a cancel button; every run is saved in its own folder.
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
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from cspaps import workflows as wf
from cspaps.config import _SECTIONS, case_from_dict, load_case
from cspaps.field import Field, build_field, save_layout_csv
from cspaps.gui_plots import (
    bound_chart, field_map, history_chart, loss_waterfall, pareto_chart, plant_scene, potential_map_chart,
    receiver_3d, receiver_map, replay_timeline, uniformity_curves,
)
from cspaps.layout_opt import validate_layout
from cspaps.optics import footprints_at
from cspaps.receiver import build_grid, incident_limit_w_m2, panel_of, surface_normal, surface_point
from cspaps.simulate import evaluate, sun_state

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "cases" / "aksai_demo.yaml"
RESULTS = Path.cwd() / "results" / "gui"
UPLOADS = RESULTS / "_uploads"

PAGES = ["总览", "瞄准优化", "布局优化", "云影回放", "光学校验", "参数设置", "运行记录", "方法与理论"]
PAGE_LABELS = {"总览": "◎  总览", "瞄准优化": "◉  瞄准优化 · 问题一", "布局优化": "▦  布局优化 · 问题二",
               "云影回放": "☁  云影回放", "光学校验": "◇  光学校验", "参数设置": "⚙  参数设置",
               "运行记录": "☰  运行记录", "方法与理论": "∑  方法与理论"}
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

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
    "aps.certify": "计算最优性证书（上界与差距）", "aps.certify_time_s": "证书迭代时限 (s)",
    "aps.certify_rounds": "松弛随机取整次数", "layout_opt.upper_bound": "计算可证上界",
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
    "aps.certify": "用拉格朗日对偶 + Frank–Wolfe 计算任何方案都不可能超过的目标值上界，并用松弛解引导取整改进方案。",
    "aps.certify_rounds": "对松弛解做随机取整的次数；每次取整后修复通量并局部搜索，保留最好的方案。",
    "layout_opt.upper_bound": "按单镜无干扰年潜力与圆盘最密堆积计算任意布局的年能量上界，给出最优性差距。",
    "heliostat.width": "定日镜反射面宽度（水平方向），单位米。", "heliostat.height": "定日镜反射面高度，单位米。",
    "heliostat.pivot_height": "定日镜旋转中心离地高度，单位米。",
    "receiver.diameter": "圆柱外置式吸热器直径，单位米。", "receiver.height": "吸热器受光段高度，单位米。",
    "field.exclusion_radius": "塔周不允许布置定日镜的半径（检修通道、塔基础等），单位米。",
    "aps.n_height_levels": "每面镜在吸热器高度方向的候选瞄准层数。层数越多越接近连续瞄准，计算越慢。",
    "aps.azimuth_offsets_deg": "候选瞄准点相对正对方位的周向偏移（度），YAML 列表，例如 [-6, 0, 6]。",
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


def _fields(section: str, fields: list[str] | None = None, n_cols: int = 2) -> None:
    """Widgets of some fields of a case section, written back to the case dict."""
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    cls = _SECTIONS[section]
    current = dict(data.get(section) or {})
    default_obj = dataclasses.asdict(getattr(_defaults(), section))
    cols = st.columns(n_cols)
    new = dict(current)
    names = [f.name for f in dataclasses.fields(cls) if fields is None or f.name in fields]
    if fields is not None:
        names = [n for n in fields if n in names]
    for i, name in enumerate(names):
        value = current.get(name, default_obj.get(name))
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(default_obj.get(name), float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        with cols[i % n_cols]:
            try:
                new[name] = _widget(section, name, value, f"{ver}:{section}.{name}")
            except yaml.YAMLError as exc:
                st.error(f"{name}: YAML 格式错误 {exc}")
    data[section] = {k: v for k, v in new.items() if k in current or v != default_obj.get(k)}


def _section_form(section: str, expanded: bool = False, fields: list[str] | None = None,
                  title: str | None = None) -> None:
    with st.expander(title or SECTION_TITLES.get(section, section), expanded=expanded):
        _fields(section, fields)


def _show_files(result: dict, images: list[str] | None = None) -> None:
    files = result["files"]
    for label in images if images is not None else [k for k in files if str(files[k]).endswith(".png")]:
        if label in files and Path(files[label]).exists():
            st.image(str(files[label]), width="stretch")
    with st.expander("结果数据 (JSON)"):
        st.json(result["summary"])
    download_labels = {"aims_csv": "逐镜瞄准方案 CSV", "flux_png": "吸热器热流图", "summary_json": "计算摘要 JSON",
                       "efficiency_png": "镜场效率图", "layout_csv": "优化镜位 CSV", "history_csv": "搜索历史 CSV",
                       "ranking_csv": "候选排名 CSV", "case_yaml": "算例参数 YAML", "bound_csv": "上界收敛记录 CSV",
                       "curve_csv": "权衡曲线 CSV", "potential_npz": "场地潜力图 NPZ"}
    cols = st.columns(4)
    for i, (label, path) in enumerate(files.items()):
        path = Path(path)
        if path.exists() and path.is_file():
            with cols[i % 4]:
                st.download_button(f"下载 {download_labels.get(label, path.name)}", path.read_bytes(), file_name=path.name,
                                   key=f"dl:{st.session_state['ver']}:{path}")
    if result.get("run_dir"):
        st.caption(f"运行记录：{result['run_dir']}")


# --------------------------------------------------------------------------- visual building blocks

def _apply_theme() -> None:
    st.markdown("""<style>
:root{--cs-navy:#0f2a3a;--cs-teal:#14806f;--cs-teal-d:#0e5e52;--cs-amber:#e09b2d;--cs-ink:#16232b;
      --cs-muted:#5d6d74;--cs-line:#dfe6e6;--cs-bg:#f4f6f6}
.stApp{background:var(--cs-bg);font-family:Inter,'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif}
.stMainBlockContainer{padding-top:2.4rem;max-width:1520px}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stDeployButton"],[data-testid="stAppDeployButton"],#MainMenu,footer{display:none!important}
section[data-testid="stSidebar"]{background:linear-gradient(185deg,#0f2a3a 0%,#123a46 70%,#16484f 100%)}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"]{color:#dce9e7}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#ffffff12;border-color:#ffffff40}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{color:#dce9e7!important}
section[data-testid="stSidebar"] .stButton>button,section[data-testid="stSidebar"] .stDownloadButton>button{
  background:#ffffff14;border:1px solid #ffffff38;color:#e8f1ef;width:100%}
section[data-testid="stSidebar"] .stButton>button:hover,section[data-testid="stSidebar"] .stDownloadButton>button:hover{
  background:#ffffff26;border-color:#ffffff66;color:#fff}
section[data-testid="stSidebar"] [role="radiogroup"]{gap:.15rem}
section[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .7rem;border-radius:9px;width:100%}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#ffffff14}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#ffffff22;box-shadow:inset 3px 0 0 var(--cs-amber)}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#fff;font-weight:650}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]>div>div:first-child{display:none}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{background:#ffffff1a;border-color:#ffffff55}
div[class*="st-key-card_"]{background:#fff;border-radius:14px;box-shadow:0 1px 3px #0f2a3a12}
.cs-brand{display:flex;align-items:center;gap:.65rem;margin:.2rem 0 .1rem}
.cs-brand .name{font:750 1.32rem Inter,sans-serif;letter-spacing:.04em;color:#fff}
.cs-brand .sub{font-size:.72rem;color:#9fc0bb;letter-spacing:.08em}
.cs-logo{width:34px;height:34px;border-radius:9px;flex:none;background:
  radial-gradient(circle at 50% 38%,#ffd38a 0 16%,transparent 18%),
  repeating-radial-gradient(ellipse at 50% 118%,#0b524f 0 4px,#6cc0b1 5px 6px,#124e4c 7px 9px)}
.cs-side-card{background:#ffffff10;border:1px solid #ffffff26;border-radius:10px;padding:.7rem .8rem;margin:.4rem 0;
  font-size:.8rem;color:#dce9e7;line-height:1.55}
.cs-side-card b{color:#fff}
.cs-hero{position:relative;overflow:hidden;border-radius:16px;padding:1.35rem 1.6rem 1.2rem;margin:0 0 1.1rem;color:#fff;
  background:linear-gradient(118deg,#0f2a3a 0%,#134b4c 58%,#1b6c60 100%);box-shadow:0 6px 18px #0f2a3a22}
.cs-hero:after{content:"";position:absolute;right:-70px;top:-90px;width:280px;height:280px;border-radius:50%;
  background:radial-gradient(circle,#f6c66f 0%,#e09b2d66 30%,transparent 68%)}
.cs-hero .eyebrow{font-size:.74rem;letter-spacing:.14em;color:#a8d4cb;font-weight:650}
.cs-hero .title{font-size:1.7rem;font-weight:760;margin:.15rem 0 .25rem;letter-spacing:.01em}
.cs-hero .desc{font-size:.93rem;color:#d5e7e3;max-width:980px;line-height:1.6}
.cs-chip{display:inline-block;padding:.2rem .65rem;border-radius:999px;background:#ffffff1c;border:1px solid #ffffff33;
  font-size:.76rem;margin:.65rem .4rem 0 0;color:#eef7f5}
.cs-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(175px,100%),1fr));gap:12px;margin:.2rem 0 1rem}
.cs-kpi{position:relative;background:#fff;border:1px solid var(--cs-line);border-radius:12px;padding:.8rem 1rem .75rem 1.1rem;
  box-shadow:0 1px 2px #0f2a3a0d}
.cs-kpi:before{content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:3px;background:var(--cs-teal)}
.cs-kpi.good:before{background:#2a9d6f}.cs-kpi.warn:before{background:#e09b2d}.cs-kpi.bad:before{background:#c2453a}
.cs-kpi.navy:before{background:#2d5d7b}
.cs-kpi .k{font-size:.78rem;color:var(--cs-muted);letter-spacing:.02em}
.cs-kpi .v{font-size:1.42rem;font-weight:700;color:var(--cs-ink);font-variant-numeric:tabular-nums;line-height:1.25;
  margin-top:.12rem;overflow-wrap:anywhere}
.cs-kpi .u{font-size:.82rem;font-weight:500;color:var(--cs-muted);margin-left:.18rem}
.cs-kpi .n{font-size:.75rem;color:var(--cs-muted);margin-top:.18rem;line-height:1.4}
.cs-sec{display:flex;align-items:center;gap:.55rem;font-weight:680;font-size:1.03rem;color:var(--cs-ink);margin:.05rem 0 .35rem}
.cs-step{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--cs-teal);
  color:#fff;font-size:.78rem;font-weight:700;flex:none}
.cs-sec-cap{color:var(--cs-muted);font-size:.84rem;margin:-.1rem 0 .65rem;line-height:1.55}
.cs-callout{border-radius:12px;padding:.85rem 1.1rem;border:1px solid;margin:.3rem 0 1rem;line-height:1.6;font-size:.9rem;color:var(--cs-ink)}
.cs-callout.ok{background:#eaf6f1;border-color:#a7d8c5}.cs-callout.warn{background:#fdf5e7;border-color:#efcf94}
.cs-callout.bad{background:#fbecea;border-color:#eab2ab}.cs-callout.info{background:#edf3f6;border-color:#c4d7e0}
.cs-callout .t{font-weight:700;margin-bottom:.2rem}
.cs-flow{background:#fff;border:1px solid var(--cs-line);border-radius:14px;padding:1.05rem 1.2rem .9rem;height:100%}
.cs-flow .tag{font-size:.72rem;color:var(--cs-teal);font-weight:700;letter-spacing:.1em}
.cs-flow .h{font-size:1.1rem;font-weight:720;margin:.15rem 0 .35rem;color:var(--cs-ink)}
.cs-flow .p{color:var(--cs-muted);font-size:.86rem;line-height:1.6}
.cs-flow ul{margin:.35rem 0 .2rem 1.05rem;padding:0;color:var(--cs-muted);font-size:.84rem;line-height:1.65}
.cs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(165px,100%),1fr));gap:10px;margin:.25rem 0 .6rem}
.cs-card{background:#fff;border:1px solid var(--cs-line);border-top:3px solid var(--cs-teal);border-radius:10px;padding:.7rem .95rem}
.cs-k{font-size:.8rem;color:var(--cs-muted)}.cs-v{font-size:1.15rem;font-weight:660;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
div.stButton>button[kind="primary"],div.stDownloadButton>button[kind="primary"]{background:var(--cs-teal);border-color:var(--cs-teal);
  color:#fff;border-radius:9px;font-weight:650;padding:.5rem 1.2rem}
div.stButton>button[kind="primary"]:hover{background:var(--cs-teal-d);border-color:var(--cs-teal-d);color:#fff}
div.stButton>button{border-radius:9px}
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--cs-line)}
.stTabs [data-baseweb="tab"]{padding:.45rem .9rem;border-radius:8px 8px 0 0}
.stTabs [aria-selected="true"]{color:var(--cs-teal)!important;font-weight:650}
.stTabs [data-baseweb="tab-highlight"]{background:var(--cs-teal)}
div[data-testid="stExpander"] details{border-radius:10px;background:#fff}
.katex{font-size:1.06em}
@media(max-width:760px){.stMainBlockContainer{padding:3.2rem .8rem 2rem}.cs-hero .title{font-size:1.3rem}}
</style>""", unsafe_allow_html=True)


def _hero(eyebrow: str, title: str, desc: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="cs-chip">{escape(c)}</span>' for c in (chips or []))
    st.markdown(f'<div class="cs-hero"><div class="eyebrow">{escape(eyebrow)}</div><div class="title">{escape(title)}</div>'
                f'<div class="desc">{escape(desc)}</div>{chip_html}</div>', unsafe_allow_html=True)


def _kpis(items: list[tuple]) -> None:
    """KPI cards: (label, value[, unit[, note[, tone]]]); tone good|warn|bad|navy."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        unit = it[2] if len(it) > 2 and it[2] else ""
        note = it[3] if len(it) > 3 and it[3] else ""
        tone = it[4] if len(it) > 4 and it[4] else ""
        u = f'<span class="u">{escape(str(unit))}</span>' if unit else ""
        n = f'<div class="n">{escape(str(note))}</div>' if note else ""
        cards.append(f'<div class="cs-kpi {tone}"><div class="k">{escape(str(label))}</div>'
                     f'<div class="v">{escape(str(value))}{u}</div>{n}</div>')
    st.markdown(f'<div class="cs-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def _status_header(items: list[tuple[str, object]]) -> None:
    """Compact status cards (label, value) that wrap on narrow windows."""
    cards = "".join(
        f'<div class="cs-card"><div class="cs-k">{escape(str(k))}</div><div class="cs-v">{escape(str(v))}</div></div>'
        for k, v in items)
    st.markdown(f'<div class="cs-grid">{cards}</div>', unsafe_allow_html=True)


def _metrics(pairs: list[tuple[str, object]], per_row: int = 5) -> None:
    _status_header(pairs)


def _section(step, title: str, caption: str | None = None) -> None:
    badge = f'<span class="cs-step">{escape(str(step))}</span>' if step is not None else ""
    st.markdown(f'<div class="cs-sec">{badge}{escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="cs-sec-cap">{escape(caption)}</div>', unsafe_allow_html=True)


def _callout(kind: str, title: str, body: str) -> None:
    st.markdown(f'<div class="cs-callout {kind}"><div class="t">{escape(title)}</div>{escape(body)}</div>',
                unsafe_allow_html=True)


def _format_number(value, digits: int = 3) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


def _pct(value, digits: int = 2) -> str:
    try:
        v = float(value)
        return f"{100 * v:.{digits}f}%" if np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def _chart(fig, key: str, **kw):
    return st.plotly_chart(fig, theme=None, key=key, config=PLOTLY_CONFIG, **kw)


def _card(name: str):
    """Bordered white card (keyed container, styled by the st-key-card_ class)."""
    return st.container(border=True, key=f"card_{name}")


def _goto(page: str) -> None:
    st.session_state["page"] = page


@st.cache_data(show_spinner=False)
def _annual_dni(case_yaml: str, base_dir: str) -> float | None:
    from cspaps import sun

    try:
        case = case_from_dict(yaml.safe_load(case_yaml), base_dir=base_dir)
        return float(sun.hourly_samples(case).dni_hours)
    except Exception:  # noqa: BLE001 - informational only
        return None


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


def _job_button(key: str, label: str, fn, *args, cancellable: bool = False, help: str | None = None, **kwargs) -> None:
    """Start a background job; the result lands in st.session_state['res_<key>']."""
    case, err = _case()
    if err:
        st.error(f"参数有误：{err}")
        return
    job = st.session_state.get(f"job_{key}")
    if job is None or job.done:
        if st.button(label, key=f"btn_{key}", type="primary", help=help, width="stretch"):
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
        st.info(f"⏳ {_progress_zh(msg)}（已用 {(_time.time() - job.started):.0f} s）")
        if job.cancellable and st.button("取消", key=f"cancel_{key}"):
            job.cancel.set()
        return
    del st.session_state[f"job_{key}"]
    job.collect(key)
    st.rerun(scope="app")


def _progress_zh(msg: str) -> str:
    rules = [("certifying", "计算最优性证书：对偶上界与松弛取整"), ("weight ", "权衡曲线 · "),
             ("search sampling", "布局搜索准备"), ("generation", "差分进化 · 第"), ("local refinement", "局部微调"),
             ("review", "候选复核"), ("swap round", "换位微调"), ("certificate", "计算布局上界与场地潜力图"),
             ("replay step", "回放步"), ("sun state", "全年工况")]
    for key, zh in rules:
        if msg.startswith(key):
            return f"{zh} · {msg}"
    return msg


def _job_error(key: str) -> None:
    if st.session_state.get(f"err_{key}"):
        st.error("计算失败")
        with st.expander("详细错误"):
            st.code(st.session_state[f"err_{key}"])


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

# --------------------------------------------------------------------------- overview

def page_overview(case) -> None:
    if case is None:
        _hero("CSP-APS", "塔式光热镜场与瞄准优化", "当前算例参数无效，请在「参数设置」中修正。")
        return
    rec, hel = case.receiver, case.heliostat
    _hero("CSP-APS · 塔式光热设计与运行优化平台", "镜场布局与瞄准策略优化",
          "以第一性原理光学模型（余弦、反射、阴影遮挡、大气衰减、HFLCAL 光斑与截断）为核心，求解两类优化问题："
          "固定镜场的最优瞄准点分配（问题一）与给定场地的最优镜场布置（问题二）。每个结果都附带可证明的最优性上界，"
          "说明最优解是否存在、当前方案离全局最优还有多远。",
          [case.name, f"{case.site.latitude:.3f}°N  {case.site.longitude:.3f}°E", case.site.timezone,
           "气象：" + ("逐时 DNI 文件" if case.weather.source == "file" else "晴空模型")])
    try:
        fld = build_field(case)
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法生成镜场：{exc}")
        return
    dni_year = _annual_dni(case.to_yaml(), st.session_state["base_dir"])
    _kpis([
        ("定日镜", f"{fld.n:,}", "面", f"单镜 {hel.width:g} × {hel.height:g} m"),
        ("总反射面积", f"{fld.n * hel.area / 1e4:,.2f}", "万 m²", f"反射率 {hel.reflectivity:g} × 清洁度 {hel.cleanliness:g}"),
        ("吸热器", f"Ø{rec.diameter:g} × {rec.height:g}", "m", f"中心高 {rec.center[2]:g} m · {rec.n_panels} 块管屏"),
        ("通量上限", f"{rec.max_flux_kw_m2:g}", "kW/m²", f"安全裕度 {case.aps.safety_margin:.0%}"),
        ("年直射辐照量", f"{dni_year:,.0f}" if dni_year else "—", "kWh/m²", "DNI 全年累计", "navy"),
    ])
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with _card("ov_scene"):
            _section(None, "电站三维视图", "定日镜、塔与圆柱外置式吸热器；坐标 x 东、y 北、z 上，原点为塔底。")
            _chart(plant_scene(case, fld.xyz, height=470), "ov_scene")
    with right:
        st.markdown('<div class="cs-flow"><div class="tag">问题一 · 运行调度</div><div class="h">瞄准点优化 APS</div>'
                    '<div class="p">给定镜场坐标、镜面尺寸、吸热器三维位置和 DNI 分布，为每面定日镜选择吸热器上的瞄准点。</div>'
                    '<ul><li>最大化吸收功率，用权重 λ 或 CV 上限控制受热均匀性</li>'
                    '<li>通量上限逐管屏约束，细网格完整光斑复核</li>'
                    '<li>拉格朗日对偶上界给出最优性差距</li><li>功率–均匀度权衡曲线</li></ul></div>',
                    unsafe_allow_html=True)
        st.button("进入瞄准优化 →", key="go_aps", type="primary", on_click=_goto, args=("瞄准优化",), width="stretch")
        st.markdown('<div style="height:.6rem"></div><div class="cs-flow"><div class="tag">问题二 · 电站设计</div>'
                    '<div class="h">镜场布局优化</div>'
                    '<div class="p">给定场地范围、镜面数量与尺寸、吸热器三维位置和经纬度，寻找全年理论吸热量最大的布置。</div>'
                    '<ul><li>全年逐时气象压缩为 DNI 加权代表工况</li><li>参数化布局 + 差分进化 + 换位微调</li>'
                    '<li>单镜潜力 + 圆盘堆积给出任意布局的能量上界</li><li>候选方案受限 APS 复核</li></ul></div>',
                    unsafe_allow_html=True)
        st.button("进入布局优化 →", key="go_layout", type="primary", on_click=_goto, args=("布局优化",), width="stretch")
    single, opt = st.session_state.get("res_single"), st.session_state.get("res_opt")
    if single or opt:
        _section(None, "本次会话的最新结果")
        items = []
        if single:
            s = single["summary"]
            cert = s.get("certificate") or {}
            items += [("瞄准方案 · 吸收功率", f"{s['absorbed_power_mw']:.2f}", "MW", s["time"][:16].replace("T", " ")),
                      ("瞄准方案 · 最优性差距", f"≤ {_pct(cert.get('gap'))}" if cert.get("gap") is not None else "—", "",
                       "相对可证上界", "good" if cert.get("gap") is not None and cert["gap"] < 0.01 else "")]
        if opt:
            s = opt["summary"]
            ub = s.get("upper_bound") or {}
            items += [("布局 · 镜数", f"{s['n_heliostats']}", "面", "场地校验 " + ("通过" if s["status"] == "ok" else s["status"])),
                      ("布局 · 理论年吸热", f"{ub.get('found_energy_mwh', 0):,.0f}", "MWhₜₕ",
                       f"上界差距 ≤ {_pct(ub.get('gap'))}" if ub.get("gap") is not None else "")]
        _kpis(items)


# --------------------------------------------------------------------------- problem 1: aim points

GOALS = {"power": "吸收功率最大", "weighted": "功率 + 均匀度（权重 λ）", "cv_limit": "均匀度上限（CV ≤ ε）"}
DNI_SOURCES = {"site": "站址气象（全场均匀 DNI）", "cloud": "云影场景（模拟空间分布）", "upload": "导入实测 DNI 分布"}


def page_aps(case) -> None:
    _hero("问题一 · 运行调度", "瞄准点优化（APS）",
          "固定镜场坐标、镜面数量与尺寸、吸热器三维位置，在给定 DNI 分布下为每面定日镜选择吸热器表面的瞄准点。"
          "目标为最大化吸收热功率，同时用超参数控制受热均匀性；所有方案均在细网格上用完整光斑复核通量上限，"
          "并给出可证明的最优性差距。",
          [f"{case.name}" if case else "算例无效"] + ([f"{build_field(case).n} 面定日镜",
                                                       f"吸热器中心 ({', '.join(f'{c:g}' for c in case.receiver.center)}) m"]
                                                      if case else []))
    if case is None:
        st.warning("请先在「参数设置」中修正当前算例。")
        return
    ver = st.session_state["ver"]
    data = st.session_state["case_dict"]
    left, right = st.columns([1, 2.05], gap="large")
    up_now = up_fc = up_prev = None
    with left:
        with _card("aps_conditions"):
            _section(1, "运行工况", "选择时刻与镜场 DNI 分布。太阳位置由经纬度和时区计算。")
            default_time = pd.Timestamp(case.realtime.start)
            c1, c2 = st.columns(2)
            date = c1.date_input("日期", value=default_time.date(), key=f"s_date:{case.name}")
            tm = c2.time_input("当地时间", value=default_time.time(), key=f"s_time:{case.name}", step=300)
            dni_src = st.radio("DNI 分布", list(DNI_SOURCES), format_func=DNI_SOURCES.get, key="aps_dni_src",
                               help="站址气象：全场同一 DNI；云影场景：按回放设置生成云影，DNI 随镜位变化；"
                                    "导入：CSV (x, y, dni) 插值到每面镜。")
            if dni_src == "upload":
                up_now = st.file_uploader("当前镜场 DNI（CSV x,y,dni）", type=["csv"], key="s_now")
            with st.expander("预报 DNI 与上一周期方案（可选）"):
                up_fc = st.file_uploader("预测时段最高 DNI（CSV x,y,dni）", type=["csv"], key="s_fc",
                                         help="用于避免云层移开后预测时段内吸热器局部热流超限。")
                up_prev = st.file_uploader("上一周期已执行的瞄准方案", type=["csv"], key="s_prev",
                                           help="导入上一周期方案 CSV，按定日镜 ID 匹配后用于限制动作幅度。")
        with _card("aps_goal"):
            _section(2, "优化目标", "λ 是均匀性的重要性超参数：λ = 0 只追求功率，λ 越大越重视受热均匀。")
            uni = data.setdefault("uniformity", {})
            current = case.uniformity.mode if case.uniformity.enabled else "power"
            goal = st.radio("目标", list(GOALS), index=list(GOALS).index(current), format_func=GOALS.get,
                            key=f"aps_goal:{ver}", label_visibility="collapsed")
            uni["enabled"] = goal != "power"
            if goal == "weighted":
                uni["mode"] = "weighted"
                uni["weight"] = float(st.slider("均匀度权重 λ", 0.0, 2.0, float(case.uniformity.weight), 0.05,
                                                key=f"{ver}:uniformity.weight", help=HELP["uniformity.weight"]))
                st.latex(r"\max\;P_{abs}-\lambda\,A_H\,\sigma_H")
            elif goal == "cv_limit":
                uni["mode"] = "cv_limit"
                uni["max_cv"] = float(st.number_input("允许的二维 CV 上限 ε", min_value=0.01, max_value=5.0,
                                                      value=float(case.uniformity.max_cv), step=0.05, format="%.3f",
                                                      key=f"{ver}:uniformity.max_cv", help=HELP["uniformity.max_cv"]))
                st.latex(r"\max\;P_{abs}\;\;\text{s.t.}\;\mathrm{CV}_{2D}\le\varepsilon")
            else:
                st.latex(r"\max\;P_{abs}\;\;\text{s.t.}\;Q_j\le(1-m)Q_j^{max}")
            with st.expander("考核区域、最低负荷与安全裕度"):
                _fields("uniformity", ["panels", "z_range", "min_load_mw"])
                _fields("aps", ["safety_margin", "allow_defocus"])
            with st.expander("候选瞄准点、算法与最优性证书"):
                methods = CHOICES["aps.method"]
                method = st.selectbox("瞄准算法", methods, index=methods.index(case.aps.method),
                                      format_func=lambda item: CHOICE_LABELS["aps.method"].get(item, item),
                                      help=HELP["aps.method"], key=f"s_method:{case.name}")
                _fields("aps", ["n_height_levels", "height_span", "azimuth_offsets_deg", "certify",
                                "certify_time_s", "certify_rounds"])
        case, err = _case()
        if err:
            st.error(err)
            return
        stamp = f"{date} {tm}"
        if dni_src == "upload" and up_now is None:
            st.info("请上传当前镜场 DNI 分布，或改用站址气象 / 云影场景。")
        _job_button("single", "▶  计算瞄准方案", wf.task_aim, stamp, method=method,
                    dni_map=_save_upload(up_now) if up_now else None,
                    forecast_dni_map=_save_upload(up_fc) if up_fc else None,
                    prev_aims=_save_upload(up_prev) if up_prev else None,
                    dni_source="cloud" if dni_src == "cloud" else "site", progress=True)
        _job_error("single")
    with right:
        res = st.session_state.get("res_single")
        if res is None:
            with _card("aps_preview"):
                _section(None, "镜场预览", "设定工况与目标后点击「计算瞄准方案」。结果包括热流分布、逐镜瞄准坐标和最优性证书。")
                fld = build_field(case)
                _chart(field_map(case, fld.xyz, fld.ids, title=f"当前镜场 · {fld.n:,} 面定日镜"), "aps_preview")
        else:
            _aps_results(res)
    _pareto_section(case, f"{date} {tm}", dni_src, up_now)


def _aps_results(res: dict) -> None:
    s, snap, fld = res["summary"], res["snapshot"], res["field"]
    run = Path(res["run_dir"])
    case = load_case(run / "inputs_case.yaml")
    inputs = json.loads((run / "run.json").read_text(encoding="utf-8"))["arguments"]
    aps, chk = s["aps"], s["flux_check"]
    cert = s.get("certificate")
    method = aps["method"]
    if inputs.get("dni_map"):
        source = f"上传 DNI 分布 · {Path(inputs['dni_map']).name}"
    elif inputs.get("dni_source") == "cloud":
        source = "云影场景（模拟空间分布）"
    elif case.weather.source == "file":
        source = f"站址气象文件 · {Path(case.weather.file).name.removeprefix('input_weather_')}"
    else:
        source = "晴空模型"
    uni_status = aps.get("uniformity_status", "not_checked" if case.uniformity.enabled else "disabled")
    eligible = (s["status"] in ("ok", "margin_reduced") and chk["status"] in ("ok", "margin_reduced")
                and uni_status in ("weighted", "feasible", "disabled"))
    state = STATE_ZH.get(s["status"], s["status"])
    if chk["status"] == "over_limit":
        state = "本方案通量超限"
    target = "功率优先"
    if case.uniformity.enabled:
        target = (f"CV ≤ {case.uniformity.max_cv:.3f}" if case.uniformity.mode == "cv_limit"
                  else f"λ = {case.uniformity.weight:g}")
    _section(None, "计算结果", f"{s['time'][:19].replace('T', ' ')} · {CHOICE_LABELS['aps.method'][method]} · {source}")
    gap = cert.get("gap") if cert else None
    _kpis([
        ("吸收热功率", f"{s['absorbed_power_mw']:.3f}", "MW", f"截获 {s['intercepted_power_mw']:.3f} MW · DNI 均值 {s['dni_mean_w_m2']:.0f} W/m²"),
        ("考核区二维 CV", _format_number(aps.get("cv_2d_H", s.get("cv_2d")), 4), "",
         f"周向 {_format_number(aps.get('cv_circ_H', s.get('cv_circumferential')))} · 高度 {_format_number(aps.get('cv_height_H', s.get('cv_height')))}"),
        ("峰值通量（细网格复核）", f"{chk['peak_flux_kw_m2']:.0f}", "kW/m²",
         f"{chk['hotspot']['panel']} 号管屏 · 距上限 {chk['margin_to_limit_kw_m2']:.0f} kW/m²",
         "good" if chk["status"] == "ok" else ("warn" if chk["status"] == "margin_reduced" else "bad")),
        ("最优性差距", f"≤ {_pct(gap)}" if gap is not None else "—", "",
         f"可证上界 {cert['upper_bound_mw']:.3f} MW" if cert else "未计算证书",
         "good" if gap is not None and gap < 0.01 else ("warn" if gap is not None else "")),
        ("方案状态", state, "", f"散焦 {s['n_defocused']} 面 · 耗时 {aps.get('solve_time_s', 0):.1f} s",
         "good" if eligible else "bad"),
    ])
    st.caption(f"均匀度目标 {target} · {UNIFORMITY_ZH.get(uni_status, uni_status)}。"
               "二维 CV 是考核区吸收热流的面积加权标准差与均值之比，越小越均匀。")
    if cert:
        exist = s.get("existence") or {}
        mode_txt = {"power": "吸收功率", "weighted": f"P − λ·A_H·σ_H（λ = {cert.get('lambda')}）",
                    "cv_limit": f"吸收功率（CV ≤ {cert.get('epsilon')}）"}[cert["mode"]]
        if cert["status"] == "certified":
            proven = cert.get("global_optimum_proven_on_candidate_set")
            _callout("ok", "最优解存在，且本方案已被证明" + ("为候选集上的全局最优" if proven else "接近全局最优"),
                     f"候选瞄准点集合有限，且“全部散焦”方案满足全部通量约束，因此最优解必然存在。"
                     f"目标函数 {mode_txt}：本方案 {cert['objective_found_mw']:.4f} MW，任何方案都不可能超过 "
                     f"{cert['upper_bound_mw']:.4f} MW（拉格朗日对偶上界），差距 ≤ {_pct(gap, 3)}。")
        elif cert["status"] == "certified_infeasible":
            _callout("bad", "已证明不存在满足要求的方案", exist.get("statement", "") +
                     "。请放宽 CV 上限或降低最低负荷。")
        else:
            _callout("warn", "未找到满足均匀度目标的方案", "当前目标下给出最接近目标的方案；上界为 "
                     f"{cert['upper_bound_mw']:.4f} MW。可放宽 CV 上限或增加候选瞄准点。")
    if chk["status"] == "over_limit":
        st.error(f"第 {chk['hotspot']['panel']} 号管屏的复核热点超出设备限值 {chk['exceedance_kw_m2']:.1f} kW/m²。")
    elif s["status"] == "uniformity_infeasible":
        st.warning("本方案未达到均匀度或最低负荷要求，可调整瞄准策略后重新计算。")
    executed = st.session_state.get("single_executed") == str(run)
    if eligible:
        if executed:
            st.success("已记录人工确认（本次仿真会话）")
        elif st.button("记录为人工已确认"):
            st.session_state["single_executed"] = str(run)
            st.rerun()
    tabs = st.tabs(["吸热器热流", "三维吸热器", "镜场", "最优性证书", "效率与损失", "逐镜方案"])
    with tabs[0]:
        _chart(receiver_map(case, snap.grid, snap.flux, "吸热器展开图 · 入射热流（× 为细网格复核热点）", hotspot=chk,
                            aims=(snap.aim_azimuth[snap.active], snap.aim_z[snap.active]), height=430), "s_rec")
    with tabs[1]:
        _chart(receiver_3d(case, snap.grid, snap.flux), "s_rec3d")
    with tabs[2]:
        color_by = st.radio("着色", ["DNI", "瞄准高度", "光学效率"], horizontal=True, key="s_field_color")
        vals, label, scale = {"DNI": (snap.dni, "DNI (W/m²)", "Blues"),
                              "瞄准高度": (snap.aim_z, "瞄准高度 z (m)", "RdBu"),
                              "光学效率": (snap.eta_total, "光学效率", "Viridis")}[color_by]
        _chart(field_map(case, fld.xyz, fld.ids, values=vals, label=label, colorscale=scale,
                         title=f"镜场 · 按{color_by}着色（× 为散焦镜面）", inactive=~snap.active), "s_field")
    with tabs[3]:
        _certificate_tab(res, cert)
    with tabs[4]:
        _chart(loss_waterfall(s["power_mw"]), "s_waterfall")
        st.dataframe(pd.DataFrame({"效率": s["field_efficiency"]}).T.rename(columns=LOSS_ZH), hide_index=True)
    with tabs[5]:
        aims = pd.read_csv(res["files"]["aims_csv"])
        st.dataframe(aims[["id", "x", "y", "active", "aim_azimuth_deg", "aim_z", "aim_x", "aim_y", "dni_w_m2",
                           "p_receiver_kw"]].rename(columns={
            "id": "镜子 ID", "x": "x (m)", "y": "y (m)", "active": "工作", "aim_azimuth_deg": "瞄准方位 (°)",
            "aim_z": "瞄准高度 (m)", "aim_x": "瞄准点 x", "aim_y": "瞄准点 y", "dni_w_m2": "DNI (W/m²)",
            "p_receiver_kw": "送达功率 (kW)"}), hide_index=True, height=360)
        _show_files(res, images=[])


LOSS_ZH = {"cosine": "余弦", "reflectivity": "反射/清洁", "shading": "阴影", "blocking": "遮挡",
           "attenuation": "大气衰减", "intercept": "截断"}


def _certificate_tab(res: dict, cert: dict | None) -> None:
    if not cert:
        st.info("本次计算未启用最优性证书（瞄准算法为中心瞄准，或在「候选瞄准点、算法与最优性证书」中关闭了证书）。")
        return
    c1, c2 = st.columns([1.3, 1])
    with c1:
        hist = pd.read_csv(res["files"]["bound_csv"]).to_dict("records") if "bound_csv" in res["files"] else []
        if hist:
            _chart(bound_chart(hist, cert.get("objective_found_mw")), "s_bound")
    with c2:
        rows = {
            "证书类型": {"certified": "已证明差距", "certified_infeasible": "已证明不可行",
                     "bound_only": "仅上界"}.get(cert["status"], cert["status"]),
            "本方案目标值 (MW)": _format_number(cert.get("objective_found_mw"), 4),
            "可证上界 (MW)": _format_number(cert.get("upper_bound_mw"), 4),
            "最优性差距": _pct(cert.get("gap"), 3),
            "松弛取整改进": "是" if cert.get("improved_by_relaxation") else "否",
            "取整尝试次数": cert.get("roundings_tried"),
            "对偶迭代次数": cert.get("iterations"),
            "候选点 / 约束点": f"{cert.get('n_candidates')} / {cert.get('n_constraint_points')}",
            "证书耗时 (s)": cert.get("time_s"),
        }
        st.dataframe(pd.DataFrame({"项目": list(rows), "值": [str(v) for v in rows.values()]}), hide_index=True)
    st.markdown("**证明思路**：对每组乘子 ν（通量约束）、μ（CV 约束）、ρ（最低负荷）≥ 0，拉格朗日函数")
    st.latex(r"L(x)=P(x)-(\lambda+\mu)\,S(x)+\mu\,\varepsilon\,P_H(x)+\rho\,(P_H(x)-P_{min})-\nu^\top(Ax-\mathbf 1)")
    st.markdown("对 x 是凹函数，且在任何可行方案处不小于目标值；在放松为 [0,1] 的分配多面体上，凹性给出")
    st.latex(r"\max_x L\;\le\;L(x_t)+\max_{y\in\Delta}\nabla L(x_t)^\top(y-x_t)")
    st.markdown("右端按定日镜可分离、直接求出。每一步 Frank–Wolfe 迭代都得到一个有效上界，取最小者。"
                "上界对本候选瞄准点集合和约束点（求解网格与复核热点）成立；加密候选点后最优值收敛到连续瞄准问题的最优值。")


def _pareto_section(case, stamp: str, dni_src: str, up_now) -> None:
    with _card("pareto"):
        _section(3, "功率–均匀度权衡曲线",
                 "同一时刻、同一 DNI 分布下扫描均匀度权重 λ。每个点都是经过细网格通量复核、带最优性差距的方案，"
                 "可用于选择合适的 λ。")
        c1, c2 = st.columns([2, 1])
        lam_txt = c1.text_input("λ 序列（逗号分隔）", "0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0", key="pareto_lams")
        try:
            lambdas = [float(v) for v in lam_txt.replace("，", ",").split(",") if v.strip()]
        except ValueError:
            st.error("λ 序列格式错误")
            return
        with c2:
            _job_button("pareto", "计算权衡曲线", wf.task_pareto, stamp, lambdas=lambdas,
                        dni_map=_save_upload(up_now) if (dni_src == "upload" and up_now) else None,
                        dni_source="cloud" if dni_src == "cloud" else "site", progress=True)
        _job_error("pareto")
        res = st.session_state.get("res_pareto")
        if res is None:
            return
        rows = res["summary"]["curve"]
        sel = case.uniformity.weight if case and case.uniformity.enabled and case.uniformity.mode == "weighted" else None
        a, b = st.columns([1.5, 1])
        with a:
            _chart(pareto_chart(rows, sel), "pareto_chart")
        with b:
            df = pd.DataFrame(rows)
            view = pd.DataFrame({"λ": df["lambda"], "吸收功率 (MW)": df["power_abs_mw"].round(3),
                                 "二维 CV": df["cv_2d"].round(4),
                                 "峰值 (kW/m²)": df["peak_flux_kw_m2"].round(0) if "peak_flux_kw_m2" in df else None,
                                 "差距": df["gap"].map(lambda g: _pct(g, 2)) if "gap" in df else "—"})
            st.dataframe(view, hide_index=True)
            p0 = rows[0]["power_abs_mw"] if rows else None
            if p0 and len(rows) > 1:
                best = min(rows, key=lambda r: r["cv_2d"])
                st.caption(f"从 λ = {rows[0]['lambda']:g} 到 λ = {best['lambda']:g}：CV {rows[0]['cv_2d']:.3f} → "
                           f"{best['cv_2d']:.3f}，吸收功率 {100 * (best['power_abs_mw'] / p0 - 1):+.1f}%。")

# --------------------------------------------------------------------------- problem 2: layout

PRESETS = {
    "fast": ("快速（约 1 分钟）", {"maxiter": 2, "popsize": 3, "n_sun_bins": 8, "polish_rounds": 1, "refine_top": 1,
                                  "review_n_sun_bins": 4, "swap_rounds": 1}),
    "standard": ("标准（约 2–4 分钟）", {"maxiter": 4, "popsize": 4, "n_sun_bins": 12, "polish_rounds": 2,
                                      "refine_top": 2, "review_n_sun_bins": 6, "swap_rounds": 1}),
    "fine": ("精细（约 10–20 分钟）", {"maxiter": 10, "popsize": 6, "n_sun_bins": 24, "polish_rounds": 4,
                                     "refine_top": 3, "review_n_sun_bins": 12, "swap_rounds": 2}),
}


def _apply_preset() -> None:
    key = st.session_state.get("layout_preset")
    if key in PRESETS:
        st.session_state["case_dict"].setdefault("layout_opt", {}).update(PRESETS[key][1])
        st.session_state["ver"] += 1


def _site_inputs(case) -> None:
    """Site range: rectangle or polygon, exclusion radius."""
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    fspec = data.setdefault("field", {})
    b = case.field.boundary
    is_rect = b is not None and len(b) == 4 and len({p[0] for p in b}) == 2 and len({p[1] for p in b}) == 2
    shape = st.radio("场地形状", ["rect", "polygon"], index=0 if (is_rect or b is None) else 1, horizontal=True,
                     format_func={"rect": "矩形", "polygon": "多边形（顶点列表）"}.get, key=f"site_shape:{ver}")
    if shape == "rect":
        if b is not None:
            xs, ys = [p[0] for p in b], [p[1] for p in b]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        else:
            r = 6.0 * case.receiver.center[2]
            x0, x1, y0, y1 = -r, r, -r, r
        c1, c2 = st.columns(2)
        x0 = c1.number_input("西边界 x_min (m)", value=float(x0), step=10.0, key=f"{ver}:rect.x0")
        x1 = c2.number_input("东边界 x_max (m)", value=float(x1), step=10.0, key=f"{ver}:rect.x1")
        y0 = c1.number_input("南边界 y_min (m)", value=float(y0), step=10.0, key=f"{ver}:rect.y0")
        y1 = c2.number_input("北边界 y_max (m)", value=float(y1), step=10.0, key=f"{ver}:rect.y1")
        if x1 > x0 and y1 > y0:
            fspec["boundary"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        else:
            st.error("边界无效：需要 x_max > x_min 且 y_max > y_min。")
    else:
        _fields("field", ["boundary"], n_cols=1)
    _fields("field", ["exclusion_radius"], n_cols=1)


def _receiver_center_inputs(case) -> None:
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    c = list(case.receiver.center)
    cols = st.columns(3)
    new = [cols[i].number_input(lbl, value=float(c[i]), step=1.0, key=f"{ver}:rc.{i}")
           for i, lbl in enumerate(["中心 x (m)", "中心 y (m)", "中心 z (m)"])]
    data.setdefault("receiver", {})["center"] = new


def page_layout(case) -> None:
    _hero("问题二 · 电站设计", "镜场布局优化",
          "给定场地范围、镜面数量、镜面尺寸、吸热器三维坐标和经纬度，寻找使全年理论吸收热量最大的定日镜布置。"
          "计算同时给出任意布局都不可能超过的能量上界：它回答“最优布置是否存在、当前方案离全局最优多远”。",
          [case.name] if case else ["算例无效"])
    if case is None:
        st.warning("请先在「参数设置」中修正当前算例。")
        return
    left, right = st.columns([1, 2.05], gap="large")
    with left:
        with _card("layout_inputs"):
            _section(1, "场地、设备与位置", "这些就是问题二的全部给定条件。")
            _fields("site", ["latitude", "longitude", "elevation_m", "timezone"])
            _fields("layout_opt", ["n_heliostats"], n_cols=1)
            _fields("heliostat", ["width", "height", "pivot_height"], n_cols=3)
            st.caption("吸热器（圆柱）中心三维坐标，原点为塔底")
            _receiver_center_inputs(case)
            _fields("receiver", ["diameter", "height"])
            _site_inputs(case)
            wup = st.file_uploader("更换气象文件（EPW 或 CSV time,dni）", type=["epw", "csv"], key="weather_up")
            if wup is not None:
                st.session_state["case_dict"]["weather"] = {"source": "file", "file": str(_save_upload(wup))}
        with _card("layout_search"):
            _section(2, "搜索设置", "搜索阶段按全年理论吸收热量（中心瞄准，含阴影遮挡）评价布局，入围方案再用受限 APS 复核。")
            st.radio("计算强度", list(PRESETS), format_func=lambda k: PRESETS[k][0], key="layout_preset",
                     index=1, horizontal=False, on_change=_apply_preset)
            _fields("layout_opt", ["generator", "search_aiming", "refine_top", "upper_bound"])
            with st.expander("搜索范围、采样精度与计算资源"):
                common = {"n_heliostats", "generator", "search_aiming", "refine_top", "upper_bound"}
                _fields("layout_opt", [f.name for f in dataclasses.fields(_SECTIONS["layout_opt"]) if f.name not in common])
            with st.expander("APS 复核使用的瞄准与均匀度设置"):
                _fields("aps", ["method", "safety_margin", "n_height_levels", "azimuth_offsets_deg"])
                _fields("uniformity", ["enabled", "mode", "weight", "max_cv"])
        case, err = _case()
        if err:
            st.error(err)
            return
        _job_button("opt", "▶  开始布局优化", wf.task_optimize_layout, progress=True, cancellable=True)
        _job_error("opt")
    with right:
        res = st.session_state.get("res_opt")
        if res is None:
            with _card("layout_preview"):
                _section(None, "场地预览", "虚线为场地边界，红色点线为塔周禁排区。点击「开始布局优化」后，"
                         "这里会显示单镜年潜力图、优化后的镜位和最优性上界。")
                try:
                    fld = build_field(case)
                    _chart(field_map(case, fld.xyz, fld.ids, title=f"当前镜场：{fld.n} 面（布局优化将按目标镜数 "
                                     f"{case.layout_opt.n_heliostats} 面重新布置）"), "site_map")
                    if fld.dropped:
                        st.warning(f"{len(fld.dropped)} 面镜子因场地检查被过滤。")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"无法生成镜场：{exc}")
        else:
            _layout_results(case, res)
    if st.session_state.get("res_opt") is not None:
        _planning_compare(case)


def _layout_results(case, res: dict) -> None:
    s = res["summary"]
    r = res["result"]
    ub = s.get("upper_bound") or {}
    val = s["validation"]
    site_status = {"ok": "通过", "capacity_shortfall": "镜数不足", "constraint_violation": "未通过"}.get(s["status"], s["status"])
    gap = ub.get("gap")
    _section(None, "优化结果", f"{s['n_heliostats']} 面定日镜 · 评价 {s['n_evaluations']} 个布局 · 用时 {s['runtime_s']:.0f} s")
    _kpis([
        ("全年理论吸收热量", f"{ub.get('found_energy_mwh', s.get('annual_energy_mwh_center_aiming') or 0):,.0f}", "MWhₜₕ",
         "中心瞄准，含阴影与遮挡"),
        ("任意布局的能量上界", f"{ub['upper_bound_mwh']:,.0f}" if ub.get("upper_bound_mwh") else "—", "MWhₜₕ",
         "单镜潜力 + 圆盘堆积", "navy"),
        ("最优性差距", f"≤ {_pct(gap, 1)}" if gap is not None else "—", "", "相对可证上界",
         "good" if gap is not None and gap < 0.08 else ("warn" if gap is not None else "")),
        ("APS 可行年吸收热量", f"{s['annual_energy_mwh_with_aps']:,.0f}" if s.get("annual_energy_mwh_with_aps") else "—",
         "MWhₜₕ", "通过通量与均匀度约束的工况"),
        ("最小镜距", f"{val.get('min_distance_m')}", "m", f"要求 ≥ {val.get('required_min_spacing_m')} m · 场地校验{site_status}",
         "good" if s["status"] == "ok" else "bad"),
    ])
    exist = s.get("existence") or {}
    if exist.get("optimum_exists") is True and ub.get("upper_bound_mwh"):
        _callout("ok", "最优布置存在；本方案与全局最优的距离已被量化",
                 f"镜位可行集（场地内、镜间距 ≥ {ub['min_spacing_m']:.2f} m、镜数 {ub['n_heliostats']}）有界且闭，"
                 f"年能量随镜位连续变化，因此全局最优布置存在（一般不唯一）。本方案全年理论吸收 "
                 f"{ub['found_energy_mwh']:,.0f} MWhₜₕ；任何布局都不超过 {ub['upper_bound_mwh']:,.0f} MWhₜₕ，"
                 f"差距 ≤ {_pct(gap, 1)}。上界忽略阴影遮挡并允许最密堆积，因此真实最优值位于两者之间。")
    elif exist.get("optimum_exists") is False:
        _callout("bad", "已证明不存在满足条件的布置", exist.get("statement", ""))
    elif exist:
        _callout("warn", "尚未找到满足全部条件的布置", exist.get("statement", ""))
    if s.get("operational_status") == "no_feasible_annual_states":
        st.warning("排名第一的布局在年度复核中没有通过全部约束的工况。")
    if s.get("capacity"):
        st.warning(f"按最密生成参数估计可放置 {s['capacity']['max_placeable_with_densest_parameters']} 面镜，低于目标镜数。")
    best = r.best.aps_result or r.best.result
    eff = best.heliostat_efficiency if best is not None and len(best.heliostat_efficiency) == r.best.n else None
    tabs = st.tabs(["单镜潜力图与布局", "三维视图", "搜索收敛", "上界说明"])
    with tabs[0]:
        if r.potential is not None:
            _chart(potential_map_chart(case, r.potential, r.best.xyz, eff, "年均光学效率",
                                       title="底图：单镜全年吸收潜力（无邻镜干扰）· 点：优化镜位"), "pot_map")
        else:
            _chart(field_map(case, r.best.xyz, np.arange(1, r.best.n + 1), values=eff, label="年均光学效率",
                             title="优化镜位"), "opt_map")
    with tabs[1]:
        _chart(plant_scene(case, r.best.xyz, values=eff, label="年均光学效率"), "opt_scene")
    with tabs[2]:
        _chart(history_chart(r.history), "opt_hist")
        st.caption("灰点为每次评价的布局参数，绿线为迄今最优。搜索阶段按代表工况快速评价，入围方案随后复核。")
    with tabs[3]:
        if ub:
            st.latex(r"E(X)\;\le\;\sum_i e_0(x_i)\;=\;\int_0^\infty \#\{i:e_0(x_i)\ge t\}\,dt\;\le\;"
                     r"\int_0^\infty \min\!\big(N,\ \mathrm{cap}(L_t)\big)\,dt")
            st.latex(r"\mathrm{cap}(S)\le \frac{\mathrm{area}\big(S\oplus B(d/\sqrt3)\big)}{\sqrt3\,d^2/2}")
            st.markdown("- $e_0(x)$：位置 $x$ 处单面定日镜在无邻镜阴影遮挡时的全年吸收热量（余弦、反射、衰减、截断）；"
                        "邻镜和通量约束只会减少能量。\n"
                        "- $L_t=\\{x: e_0(x)\\ge t\\}$；镜距 $\\ge d$ 的点，其 Voronoi 胞与半径 $d/\\sqrt3$ 圆盘之交互不重叠、"
                        "面积 $\\ge \\sqrt3 d^2/2$（Fejes Tóth），由此给出区域可容纳镜数上界。\n"
                        f"- 场地最多可容纳约 {ub.get('packing_capacity_bound')} 面镜（上界）；"
                        f"栅格 {ub.get('raster_cell_m', 0):.1f} m；采样 {ub.get('sampling', {}).get('n_states')} 个代表工况。")
            st.dataframe(pd.DataFrame({"项目": ["本方案理论年吸收", "本方案各镜单独潜力之和", "任意布局上界", "最优性差距",
                                              "APS 可行年吸收", "APS 口径差距"],
                                       "值": [f"{ub.get('found_energy_mwh', 0):,.1f} MWh", f"{ub.get('layout_lone_sum_mwh', 0):,.1f} MWh",
                                             f"{ub.get('upper_bound_mwh', 0):,.1f} MWh", _pct(ub.get("gap")),
                                             f"{ub.get('found_aps_feasible_mwh', 0):,.1f} MWh" if ub.get("found_aps_feasible_mwh") else "—",
                                             _pct(ub.get("gap_aps_feasible"))]}), hide_index=True)
        else:
            st.info("本次运行未计算上界（布局优化设置中关闭了「可证上界」）。")

def _planning_compare(case) -> None:
    res = st.session_state.get("res_opt")
    if res is None:
        st.info("先在 ③ 运行布置优化")
        return
    if case is None:
        st.error("当前算例参数无效，无法比较布局。")
        return
    r = res["result"]
    st.divider()
    _section(None, "候选方案对比与应用", "点击表格中的一行查看该候选布局；通过场地校验且有可行工况的布局可以载入继续仿真或导出。")
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
        _chart(uniformity_curves(st.session_state["cmp_curves"]), "cmp_plot")
    with st.expander("搜索历史与全部输出"):
        _show_files(res)



# --------------------------------------------------------------------------- replay

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
        with _card("replay_settings"):
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


# --------------------------------------------------------------------------- theory

def page_theory(case) -> None:
    _hero("方法与理论", "两个优化问题的数学表述、解的存在性与可证明上界",
          "本页说明软件求解的是什么问题、为什么最优解存在、软件如何证明给出的方案离全局最优有多远，以及模型的适用范围。")
    t1, t2, t3, t4 = st.tabs(["光学模型", "问题一 · 瞄准优化", "问题二 · 布局优化", "适用范围"])
    with t1:
        _section(None, "单镜功率链")
        st.latex(r"P_i = \mathrm{DNI}_i\,A\;\eta_{\cos}\;\rho\,c\;\eta_{shd}\;\eta_{blk}\;\eta_{att}\;\eta_{int}")
        st.markdown("- 余弦：镜面法向为太阳方向与镜→瞄准点方向的角平分线；\n"
                    "- 阴影与遮挡：镜面采样点光线与邻镜矩形、塔身求交；\n"
                    "- 大气衰减：Vittitoe–Biggs 或 Beer–Lambert；\n"
                    "- 光斑：HFLCAL 圆高斯，$\\sigma = D\\sqrt{\\sigma_{sun}^2+(2\\sigma_{slope})^2+(2\\sigma_{track})^2+\\sigma_{spec}^2+\\sigma_{ast}^2}$，"
                    "投影到圆柱面并乘入射余弦；截断效率用一维 Gauss–Legendre 积分，与网格无关。")
        st.latex(r"Q_j=\sum_{i,k} D_i\,K_{ikj}\,x_{ik},\qquad q_j=\alpha_j Q_j,\qquad P_{abs}=\sum_j A_j q_j")
        st.markdown("$K_{ikj}$ 为镜 $i$ 瞄准候选点 $k$ 时在吸热器网格 $j$ 上的单位 DNI 热流响应，$x_{ik}\\in\\{0,1\\}$ 为瞄准选择。")
    with t2:
        _section(None, "问题表述")
        st.markdown("给定镜场坐标、镜面数量与尺寸、吸热器三维坐标与 DNI 分布 $D_i$，控制变量为每面定日镜的瞄准点。")
        st.latex(r"\bar q_H=\frac{1}{A_H}\sum_{j\in H}A_jq_j,\qquad \sigma_H=\sqrt{\frac{1}{A_H}\sum_{j\in H}A_j(q_j-\bar q_H)^2},"
                 r"\qquad \mathrm{CV}_{2D}=\sigma_H/\bar q_H")
        st.latex(r"\textbf{加权模式：}\quad \max_{x}\; P_{abs}(x)-\lambda\,A_H\,\sigma_H(x)"
                 r"\quad \text{s.t.}\;\; \sum_k x_{ik}\le 1,\;\; Q_j(x)\le(1-m)\,Q^{max}_j")
        st.latex(r"\textbf{上限模式：}\quad \max_{x}\; P_{abs}(x)\quad \text{s.t.}\;\; \mathrm{CV}_{2D}\le\varepsilon,\;"
                 r"P_H\ge P_{min},\; Q_j\le(1-m)Q^{max}_j")
        st.markdown("超参数 $\\lambda$ 控制均匀性的重要性：$A_H\\sigma_H=\\mathrm{CV}_{2D}\\,P_H$，"
                    "所以目标等价于 $P_{abs}-\\lambda\\,\\mathrm{CV}_{2D}\\,P_H$；考核区为整个吸热器时即 $P(1-\\lambda\\,\\mathrm{CV})$。")
        _section(None, "最优解存在吗？")
        _callout("ok", "存在。", "候选瞄准点集合有限，因此方案集合有限；“全部散焦”方案满足所有通量约束，可行集非空。"
                 "有限非空集合上的最大值必然取到。若把瞄准点视为吸热器表面上的连续变量，可行集是紧集，热流对瞄准点连续，"
                 "由 Weierstrass 定理最优解同样存在。上限模式在设置了最低负荷时可能不可行，此时软件会尝试给出不可行证明。"
                 "最优解一般不唯一（对称镜位可以互换瞄准点）。")
        _section(None, "如何证明离全局最优有多远")
        st.markdown("$S(x)=A_H\\sigma_H$ 是 $x$ 的凸函数（仿射映射的范数），把 $x$ 放松到 $[0,1]$ 后加权模式是凹最大化。"
                    "对乘子 $\\nu,\\mu,\\rho\\ge 0$：")
        st.latex(r"L(x)=P(x)-(\lambda+\mu)S(x)+\mu\varepsilon P_H(x)+\rho\,(P_H(x)-P_{min})-\nu^\top(Ax-\mathbf 1)")
        st.latex(r"\mathrm{OPT}\;\le\;\max_{x\in\Delta}L(x)\;\le\;L(x_t)+\max_{y\in\Delta}\nabla L(x_t)^\top (y-x_t)")
        st.markdown("右端按镜可分离（每面镜取系数最大的候选点或散焦），每次 Frank–Wolfe 迭代都给出有效上界；"
                    "乘子用投影次梯度更新。于是 **本方案目标值 ≤ 全局最优 ≤ 上界**，差距直接显示在结果卡片中。")
        _section(None, "求解流程")
        st.markdown("1. 光斑尺寸优先的贪心构造（满足通量约束）→ 坐标下降；\n"
                    "2. 均匀度：增量更新 $\\sum q, \\sum q^2$ 的坐标搜索（加权模式）或对 $\\mu$ 二分（上限模式）；\n"
                    "3. 对偶上界，同时得到松弛解；\n"
                    "4. 松弛引导取整（确定性 + 随机取整）→ 通量修复 → 局部搜索，保留更优方案；\n"
                    "5. 自适应细网格、完整光斑复核，超限热点加入约束重解；\n"
                    "6. 小规模问题可用 SCIP 求混合整数二阶锥规划的精确最优作为基准。")
    with t3:
        _section(None, "问题表述")
        st.markdown("给定场地 $\\Omega$、镜数 $N$、镜面尺寸、吸热器三维坐标和经纬度（决定全年太阳轨迹与 DNI）：")
        st.latex(r"\max_{X=(x_1,\dots,x_N)}\;E(X)=\sum_{t} w_t\sum_i P_i(X,t)\quad \text{s.t.}\;\;x_i\in\Omega,\;\;"
                 r"\|x_i-x_j\|\ge d\;\;(i\ne j)")
        st.markdown("$E$ 包含余弦、阴影、遮挡、衰减、截断；全年按 DNI 加权的代表太阳位置积分（能量守恒）。")
        _section(None, "最优布置存在吗？")
        _callout("ok", "存在（只要镜子放得下）。", "可行集 $\\{X: x_i\\in\\Omega, \\|x_i-x_j\\|\\ge d\\}$ 是有界闭集，"
                 "即紧集；年能量对镜位连续，由 Weierstrass 定理最大值必然取到。若圆盘堆积上界表明场地容纳不下 N 面镜，"
                 "软件给出“不存在可行布置”的证明。由于对称性，最优布置一般不唯一。")
        _section(None, "上界：任意布局都不可能超过的能量")
        st.latex(r"E(X)\le\sum_i e_0(x_i)=\int_0^\infty\#\{i:e_0(x_i)\ge t\}\,dt\le\int_0^\infty\min\big(N,\mathrm{cap}(L_t)\big)\,dt")
        st.markdown("- $e_0(x)$：单镜在无邻镜干扰、中心瞄准下的全年吸收热量（对称吸热器轮廓上，居中的高斯光斑截断最大——Anderson 不等式）；\n"
                    "- 镜距 $\\ge d$ 的点集，Voronoi 胞与半径 $d/\\sqrt3$ 圆盘之交互不重叠，面积 $\\ge\\sqrt3d^2/2$（Fejes Tóth），"
                    "得到 $\\mathrm{cap}(S)\\le \\mathrm{area}(S\\oplus B(d/\\sqrt3))/(\\sqrt3 d^2/2)$；\n"
                    "- 栅格取 3×3 邻域最大值作包络，保证不低估。")
        _section(None, "搜索方法")
        st.markdown("径向交错 / 仿生螺旋参数化生成器铺满场地 → 每个参数向量按单镜年收益选出最好的 N 个镜位 → "
                    "差分进化 + 局部模式搜索 → 入围方案用更细采样和受限 APS 复核排名 → 低收益镜位与未用候选换位 → "
                    "镜数、边界、禁排区、最小镜距最终校验 → 上界与差距。")
    with t4:
        st.markdown("- 证书是针对**模型**的：候选瞄准点集合、求解网格与复核热点、代表工况采样。加密候选点和工况后，结论随之收敛。\n"
                    "- HFLCAL 把光斑视为圆高斯，大入射角下偏保守约 1%；可用内置 MCRT 或导出 SolTrace 对标校验。\n"
                    "- 未建模吸热器内部传热；温度约束由逐管屏通量上限、高度分布系数和偏差系数代替。\n"
                    "- 布局上界忽略阴影遮挡并允许最密堆积，因此偏乐观；真实全局最优介于本方案与上界之间。\n"
                    "- 演示算例的气象为阿克塞地区 NASA POWER 2024 年逐时 DNI，镜场与吸热器尺寸为模拟设计输入，"
                    "客户提供真实数据后直接替换即可。")


def page_replay(case) -> None:
    _hero("运行仿真", "云影回放",
          "模拟云层经过镜场时在线 APS 引擎的逐周期决策：对比中心瞄准、仅用当前 DNI、含 DNI 预测三种策略的吸热量、"
          "超限时长、均匀度与端到端延迟。", [case.name] if case else [])
    _ops_replay(case)


# --------------------------------------------------------------------------- optics & other pages

def page_optics(case) -> None:
    _hero("模型校验", "光学校验", "蒙特卡洛光线追迹（MCRT）以随机光线独立检查解析光斑模型；功率差、热流图差和峰值统计误差"
          "用于评估模型偏差。也可导出 SolTrace 对标算例，或用暴力求交复核阴影遮挡筛选。")
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
    _hero("算例", "参数设置", "全部输入都在一个 YAML 算例里。按设备与计算任务分组编辑；输入框旁的问号说明含义和单位，"
          "复合列表使用 YAML。拼错的参数名会直接报错，不会被静默忽略。")
    for sec in _SECTIONS:
        _section_form(sec)
    with st.expander("高级：直接编辑 YAML"):
        text = st.text_area("算例 YAML", yaml.safe_dump(st.session_state["case_dict"], allow_unicode=True, sort_keys=False),
                            height=400, key=f"yaml_{st.session_state['ver']}")
        if st.button("应用 YAML"):
            _load_into_state(yaml.safe_load(text), Path(st.session_state["base_dir"]))
            st.rerun()


def page_runs(case) -> None:
    _hero("可追溯", "运行记录", "每次计算都保存当时的参数副本、结果表和图表。选择一条记录查看结果，或载入这份参数继续仿真。")
    runs = wf.list_runs(RESULTS)
    if runs.empty:
        st.info("还没有运行记录")
        return
    task_names = {"layout": "镜场生成", "snapshot": "单时刻基准", "aps": "APS 瞄准优化",
                  "realtime": "云影回放", "annual": "全年评估", "layout_opt": "布局优化",
                  "uniformity": "均匀受热分析", "validate": "光线追迹校验",
                  "pareto": "功率–均匀度曲线", "soltrace_export": "SolTrace 对标导出", "shading_check": "遮挡筛选复核",
                  "sunbins": "代表太阳位置"}
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
    st.set_page_config(page_title="CSP-APS · 镜场与瞄准优化", page_icon="☀", layout="wide",
                       initial_sidebar_state="expanded")
    _apply_theme()
    if "case_dict" not in st.session_state:
        path = _initial_case_path()
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = PAGES[0]
    with st.sidebar:
        st.markdown('<div class="cs-brand"><span class="cs-logo" aria-hidden="true"></span>'
                    '<div><div class="name">CSP-APS</div><div class="sub">HELIOSTAT FIELD · AIMING · LAYOUT</div></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        page = st.radio("工作区", PAGES, key="page", format_func=PAGE_LABELS.get, label_visibility="collapsed")
        st.divider()
        case, err = _case()
        if case is not None:
            u = case.uniformity
            goal = ("功率优先" if not u.enabled else
                    (f"二维 CV ≤ {u.max_cv:g}" if u.mode == "cv_limit" else f"均匀度权重 λ = {u.weight:g}"))
            st.markdown(
                f'<div class="cs-side-card"><b>当前算例</b>　{escape(case.name)}<br>'
                f'{case.site.latitude:.3f}°N · {case.site.longitude:.3f}°E · {escape(case.site.timezone)}<br>'
                f'吸热器 Ø{case.receiver.diameter:g} × {case.receiver.height:g} m · {case.receiver.n_panels} 管屏<br>'
                f'通量上限 {case.receiver.max_flux_kw_m2:g} kW/m² · 瞄准目标 {escape(goal)}</div>',
                unsafe_allow_html=True)
            st.download_button("保存当前参数 (YAML)", case.to_yaml().encode("utf-8"), file_name=f"{case.name}.yaml")
        else:
            st.error(err)
        up = st.file_uploader("载入算例 YAML", type=["yaml", "yml"])
        if up is not None and st.session_state.get("loaded_upload") != (up.name, up.size):
            st.session_state["loaded_upload"] = (up.name, up.size)
            _load_into_state(yaml.safe_load(up.getvalue().decode("utf-8")), UPLOADS)
            st.rerun()
        if st.button("恢复阿克塞演示算例"):
            _load_into_state(yaml.safe_load(DEFAULT_CASE.read_text(encoding="utf-8")), DEFAULT_CASE.parent)
            st.rerun()
        st.caption(f"结果目录 {RESULTS}")
    {"总览": page_overview, "瞄准优化": page_aps, "布局优化": page_layout, "云影回放": page_replay,
     "光学校验": page_optics, "参数设置": page_parameters, "运行记录": page_runs, "方法与理论": page_theory}[page](case)


main()
