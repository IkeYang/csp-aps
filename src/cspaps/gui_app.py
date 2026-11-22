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
import hashlib
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
from cspaps.clouds import DNI_DEFINITION
from cspaps.gui_plots import (
    ab_compare_chart, aim_scene_3d, dni_points_preview, site_preview, dni_map_chart, field_dni_map, field_map, flux_diff_map, history_chart, inner_gap_chart,
    lambda_sweep_chart, layout_overlay, layout_tradeoff_chart, loss_waterfall, pareto_chart, plant_scene,
    potential_map_chart, receiver_aims_3d, receiver_aims_map, receiver_map, receiver_rose, refinement_chart,
    replay_timeline, scenario_contrib_chart, scenario_sky_chart, state_bars, uniformity_curves,
    value_decomposition_chart,
)
from cspaps.layout_opt import validate_layout
from cspaps.optics import footprints_at
from cspaps.resources import demo_case_path
from cspaps.receiver import build_grid, incident_limit_w_m2, panel_of, surface_normal, surface_point
from cspaps.simulate import evaluate, sun_state

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = demo_case_path()  # repository copy, or the copy packaged in the wheel
RESULTS = Path.cwd() / "results" / "gui"
UPLOADS = RESULTS / "_uploads"

PAGES = ["总览", "瞄准优化", "布局优化", "云影回放", "光学校验", "参数设置", "运行记录", "方法与理论"]
PAGE_LABELS = {"总览": "◎  总览", "瞄准优化": "◉  瞄准优化 · 问题一", "布局优化": "▦  布局优化 · 问题二",
               "云影回放": "☁  云影对比与回放", "光学校验": "◇  光学校验", "参数设置": "⚙  参数设置",
               "运行记录": "☰  运行记录", "方法与理论": "∑  方法与理论"}
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
# Blocking input problems found while drawing the current page (the script reruns from the top on every
# interaction, so this list is fresh each time); any entry disables the run buttons.
INPUT_ERRORS: list[str] = []

SECTION_TITLES = {
    "site": "站址", "receiver": "集热器与管屏", "heliostat": "定日镜", "sun": "太阳", "atmosphere": "大气衰减",
    "weather": "气象数据", "shading": "阴影与遮挡", "field": "镜场", "aps": "瞄准策略 APS",
    "uniformity": "吸热器均匀受热", "engine": "在线控制引擎", "realtime": "回放（云影）", "annual": "全年计算",
    "layout_opt": "布置优化", "scenarios": "年度工况与空间 DNI 地图", "mcrt": "蒙特卡洛光线追迹校验",
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
    "aps.allow_defocus": "允许散焦", "aps.milp_time_limit_s": "MILP 时限 (s)", "aps.milp_max_nnz": "MILP 系数上限", "aps.verify": "细网格复核",
    "aps.verify_max_rounds": "热点约束最多轮数", "aps.verify_max_cells": "复核网格上限",
    "aps.auto_grid": "自动细化求解网格", "aps.max_solver_cells": "求解网格上限",
    "aps.certify": "计算最优性证书（上界与差距）", "aps.certify_time_s": "证书迭代时限 (s)",
    "aps.subfield_sectors": "子场分解扇区数",
    "aps.certify_rounds": "松弛随机取整次数", "layout_opt.upper_bound": "计算可证上界",
    "layout_opt.uniformity_weight": "布局均匀度权重 λ_L",
    "layout_opt.selection_iterations": "镜位定价迭代次数（搜索阶段）", "layout_opt.uniformity_sectors": "均匀度扇区数（0 = 按管屏）",
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
    "layout_opt.search_aiming": "布局评价使用的瞄准策略", "layout_opt.uniformity_metric": "布局受热均匀性指标",
    "aps.coupled_exact_limit": "联合姿态穷举规模上限", "aps.coupled_trials": "联合姿态局部搜索次数",
    "layout_opt.position_trials": "连续镜位微调次数",
    "layout_opt.review_sampling": "复核阶段采样", "layout_opt.review_n_sun_bins": "复核代表点数",
    "layout_opt.review_dni_bands": "复核 DNI 分档数", "layout_opt.review_check_hourly": "复核时与逐时对比误差",
    "layout_opt.min_spacing_factor": "最小镜距系数 (空 = 用 spacing_factor)",
    "layout_opt.swap_rounds": "换位微调轮数", "layout_opt.swap_fraction": "每轮换位比例",
    "layout_opt.shading_points": "阴影采样点数", "layout_opt.maxiter": "进化代数", "layout_opt.popsize": "种群规模系数",
    "layout_opt.seed": "随机种子", "layout_opt.workers": "并行进程 (-1 = 全部)",
    "layout_opt.refine_top": "复核前几名", "layout_opt.polish_rounds": "局部微调轮数",
    "layout_opt.ranking": "正式排名口径", "layout_opt.aps_weight": "运行权重 λ_APS（空 = 与评价权重 λ_L 相同）",
    "layout_opt.baseline": "初始布局对照", "layout_opt.formal_swap_rounds": "正式排名后的换位轮数（按 V 接受）",
    "layout_opt.formal_position_trials": "按年度 APS 实算的自由镜位试探次数",
    "layout_opt.ranking_refine_states": "区间重叠时每布局追加求解的工况数",
    "scenarios.validation_file": "复核工况表（与设计工况共用地图库）",
    "layout_opt.inner_joint_trials": "每个工况的联合姿态搜索次数", "layout_opt.inner_certify_rounds": "每个工况的松弛取整次数（0 = 关）",
    "layout_opt.inner_certify_iters": "每个工况的对偶上界迭代数", "layout_opt.dense_check_scenarios": "更密工况复核的工况数（0 = 关）",
    "layout_opt.cross_check_maps": "云影地图：用独立云图复核胜出布局",
    "annual.representative_dni": "代表工况的 DNI 与小时数口径",
    "scenarios.source": "工况来源", "scenarios.file": "工况表 CSV", "scenarios.n_scenarios": "代表工况数",
    "scenarios.dni_bands": "每个太阳位置的 DNI 分档数", "scenarios.map_model": "空间 DNI 地图模型",
    "scenarios.map_file": "DNI 地图库 CSV（map_id,x,y,value）", "scenarios.map_unit": "地图单位",
    "scenarios.map_normalize": "地图归一化", "scenarios.map_assignment": "地图分配到逐时数据的规则",
    "scenarios.map_sequence": "使用的地图及顺序（YAML 列表，空 = 全部）", "scenarios.map_include_uniform": "加入全场均匀 DNI 作兼容工况",
    "scenarios.map_cell_m": "地图栅格尺寸 (m)", "scenarios.map_contrast": "亮暗对比 ±c", "scenarios.map_axis_deg": "亮侧方向（罗盘方位 °）",
    "scenarios.map_edge_m": "半场分界过渡宽度 (m)", "scenarios.cloud_maps": "云影地图张数", "scenarios.cloud_n": "每张地图的云数",
    "scenarios.cloud_radius_m": "云影半径范围 [小, 大] (m)", "scenarios.cloud_transmittance": "云影透过率范围 [小, 大]",
    "scenarios.seed": "随机种子",
    "mcrt.rays_per_heliostat": "每面镜子光线数", "mcrt.seed": "随机种子", "mcrt.n_seeds": "重复次数（种子数）",
    "mcrt.tracking_mode": "跟踪误差 (per_ray 统计 / per_heliostat 逐镜偏置)",
}
CHOICES = {
    "sun.shape": ["gaussian", "pillbox"], "sun.dni_model": ["meinel", "constant"],
    "atmosphere.model": ["vittitoe_biggs_clear", "vittitoe_biggs_hazy", "beer_lambert"],
    "weather.source": ["clear_sky", "file"], "aps.method": ["greedy", "kfactor", "center", "milp", "subfield"],
    "uniformity.mode": ["cv_limit", "weighted"],
    "annual.method": ["sun_bins", "sun_grid", "representative_days", "hourly"],
    "annual.aiming": ["center", "greedy", "kfactor"], "layout_opt.generator": ["radial_staggered", "biomimetic"],
    "layout_opt.search_sampling": ["sun_bins", "sun_grid", "hourly"],
    "layout_opt.uniformity_metric": ["circumferential", "surface"],
    "layout_opt.search_aiming": ["auto", "center", "greedy", "kfactor"],
    "layout_opt.review_sampling": ["sun_bins", "sun_grid", "hourly"],
    "receiver.flux_limit_basis": ["incident", "absorbed"],
    "engine.no_forecast_policy": ["clear_sky", "current"], "engine.stale_policy": ["clear_sky", "last"],
    "engine.fallback": ["hold_or_defocus", "defocus", "none"],
    "realtime.scenario": ["clouds", "moving_edge", "sudden_clear"],
    "mcrt.tracking_mode": ["per_ray", "per_heliostat"],
    "layout_opt.ranking": ["aps_value", "legacy"], "layout_opt.baseline": ["initial", "none"],
    "scenarios.source": ["weather", "file"], "scenarios.map_model": ["uniform", "half_swap", "gradient", "clouds", "file"],
    "scenarios.map_unit": ["factor", "w_m2"], "scenarios.map_normalize": ["site_mean", "none"],
    "scenarios.map_assignment": ["by_day", "single"],
    "annual.representative_dni": ["time_mean", "energy_weighted"],
}
CHOICE_LABELS = {
    "layout_opt.uniformity_metric": {"circumferential": "管屏之间（周向）", "surface": "整个考核区（二维热流）"},
    "sun.shape": {"gaussian": "高斯太阳盘", "pillbox": "均匀圆盘"},
    "sun.dni_model": {"meinel": "Meinel 晴空模型", "constant": "固定 DNI"},
    "atmosphere.model": {"vittitoe_biggs_clear": "清澈大气", "vittitoe_biggs_hazy": "霾天大气",
                         "beer_lambert": "Beer–Lambert 消光"},
    "weather.source": {"clear_sky": "晴空模型", "file": "EPW / DNI 文件"},
    "aps.method": {"greedy": "贪心 APS（推荐）", "kfactor": "瞄准因子法", "center": "中心瞄准基准",
                   "milp": "混合整数精确法（小镜场）", "subfield": "子场分解（可并行）"},
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
    "layout_opt.ranking": {"aps_value": "逐工况 APS 年度价值 V(L)（推荐）", "legacy": "0.4 版复核口径"},
    "layout_opt.baseline": {"initial": "把初始镜场一起排名", "none": "不对照"},
    "scenarios.source": {"weather": "由站址气象年生成", "file": "导入工况表"},
    "scenarios.map_model": {"uniform": "全场均匀（兼容）", "half_swap": "左右半场亮暗 + 互换", "gradient": "线性梯度 + 反向",
                            "clouds": "随机云影", "file": "导入地图库"},
    "scenarios.map_unit": {"factor": "倍数（乘工况 DNI）", "w_m2": "绝对 DNI (W/m²)"},
    "scenarios.map_normalize": {"site_mean": "场地平均 = 1（只重新分配）", "none": "不归一化"},
    "scenarios.map_assignment": {"by_day": "逐日轮换", "single": "只用第一张"},
    "annual.representative_dni": {"time_mean": "实际小时 + 时间平均 DNI（推荐）", "energy_weighted": "辐照等效小时 + 能量加权 DNI（0.5.0 口径）"},
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
    "layout_opt.search_aiming": "中心瞄准按理论吸热量扣除均匀度惩罚复排。选用 APS 时按运行可行能量和实际姿态的均匀度复排。",
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
    "layout_opt.uniformity_weight": "吸热器周向受热不均匀的惩罚权重。0 = 只追求全年能量（高纬度下镜场会集中在塔北侧）；"
                                    "越大镜场越环绕吸热器，吸热器各朝向受热越均匀。",
    "heliostat.width": "定日镜反射面宽度（水平方向），单位米。", "heliostat.height": "定日镜反射面高度，单位米。",
    "heliostat.pivot_height": "定日镜旋转中心离地高度，单位米。",
    "receiver.diameter": "圆柱外置式吸热器直径，单位米。", "receiver.height": "吸热器受光段高度，单位米。",
    "field.exclusion_radius": "塔周不允许布置定日镜的半径（检修通道、塔基础等），单位米。",
    "aps.n_height_levels": "每面镜在吸热器高度方向的候选瞄准层数。层数越多越接近连续瞄准，计算越慢。",
    "aps.azimuth_offsets_deg": "候选瞄准点相对正对方位的周向偏移（度），YAML 列表，例如 [-6, 0, 6]。",
    "layout_opt.ranking": "推荐“逐工况 APS 年度价值”：每个参与排名的布局在同一组工况上逐个求解问题一，"
                          "按 V(L) = Σ h_s (P_s − λ_L U_s) 排名。",
    "layout_opt.aps_weight": "内层权重决定运行时怎样瞄准（P − λ_APS·U）。与外层 λ_L 独立设置。",
    "layout_opt.baseline": "初始镜场取最靠近塔的 N 面镜，作为同一口径下的对照布局参与排名。",
    "scenarios.n_scenarios": "按 DNI 能量加权聚类的代表工况数，每个工况带太阳方向、代表小时数和一张场地 DNI 地图。",
    "scenarios.map_model": "空间 DNI 的建模假设。地图在场地坐标中定义，与布局无关：同一位置读同一 DNI。",
    "scenarios.map_contrast": "半场或梯度模型的相对亮暗差，0.3 表示亮侧 1.3 倍、暗侧 0.7 倍（归一化前）。",
    "scenarios.map_normalize": "场地平均 = 1 时地图只在空间上重新分配气象站 DNI，场地全年 DNI 总量不变。",
    "scenarios.map_assignment": "逐日轮换：第 k 天用第 k mod M 张地图，再在每张地图的小时中聚类代表工况。",
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
    site, weather = data.get("site") or {}, data.get("weather") or {}
    st.session_state["weather_site"] = (site.get("latitude"), site.get("longitude"), weather.get("file"))
    st.session_state["ver"] = st.session_state.get("ver", 0) + 1
    for k in [k for k in st.session_state if k.startswith("res_") or k.startswith("job_")]:
        del st.session_state[k]


def _case():
    try:
        return case_from_dict(st.session_state["case_dict"], base_dir=st.session_state["base_dir"]), None
    except Exception as exc:  # noqa: BLE001 - show any validation problem to the user
        return None, str(exc)


def _save_upload(upload) -> Path:
    content = upload.getvalue()
    directory = UPLOADS / hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(upload.name).name
    if not path.exists():
        path.write_bytes(content)
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
                INPUT_ERRORS.append(f"{LABELS.get(f'{section}.{name}', name)} 的 YAML 格式错误")
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
button[data-testid="stNumberInputStepUp"],button[data-testid="stNumberInputStepDown"]{display:none}
div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;row-gap:.6rem}
div[data-testid="stColumn"]{min-width:min(340px,100%)}
div[data-testid="stColumn"] div[data-testid="stColumn"]{min-width:min(118px,100%)}
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

def _input_digest(case, args, options):
    """Identify the actual solve inputs before filenames are rewritten in the run snapshot."""
    def file_hash(path):
        p = Path(path)
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    settings = case.to_dict()
    for section, attr in (("weather", "file"), ("field", "layout_file"), ("scenarios", "file"),
                          ("scenarios", "map_file"), ("scenarios", "validation_file")):
        path = settings[section].get(attr)
        if path:
            settings[section][attr] = file_hash(case.resolve(path))
    inputs = {k: v for k, v in options.items() if k not in ("progress", "cancel") and v is not None}
    for key in ("dni_map", "forecast_dni_map", "prev_aims"):
        if key in inputs:
            inputs[key] = file_hash(inputs[key])
    if "layouts" in inputs:
        inputs["layouts"] = {name: (file_hash(case.resolve(v)) if isinstance(v, (str, Path)) and str(v) != "initial"
                                    else v) for name, v in inputs["layouts"].items()}
    if "scenarios_dir" in inputs:
        folder = Path(inputs["scenarios_dir"])
        inputs["scenarios_dir"] = [file_hash(folder / name) for name in ("scenarios.csv", "dni_maps.csv", "dni_maps.json")]
    payload = json.dumps([settings, args, inputs], sort_keys=True, default=wf._default)
    return hashlib.sha256(payload.encode()).hexdigest()


class Job:
    def __init__(self, fn, *args, input_digest=None, **kwargs):
        self.input_digest = input_digest
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
            if self.input_digest:
                self.result["input_digest"] = self.input_digest
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


def _job_button(key: str, label: str, fn, *args, cancellable: bool = False, help: str | None = None,
                blockers: list[str] | None = None, completed_case=None, **kwargs) -> str | None:
    """Start a background job; the result lands in st.session_state['res_<key>'].

    The button is disabled while the case is invalid or any input problem is
    pending (``INPUT_ERRORS`` + ``blockers``), so a stale value is never solved.
    """
    case, err = _case() if completed_case is None else (completed_case, None)
    problems = (list(INPUT_ERRORS) if completed_case is None else []) + list(blockers or []) + ([f"参数有误：{err}"] if err else [])
    if problems:
        st.button(label, key=f"btn_{key}", type="primary", disabled=True, width="stretch")
        st.error("无法开始计算：" + "；".join(dict.fromkeys(problems)))
        return
    digest = _input_digest(case, args, kwargs) if key in ("lv", "single", "opt", "recalc") else None
    job = st.session_state.get(f"job_{key}")
    if job is None or job.done:
        if st.button(label, key=f"btn_{key}", type="primary", help=help, width="stretch"):
            if cancellable:
                kwargs["cancel"] = True
            job = Job(fn, case, RESULTS, *args, input_digest=digest, **kwargs)
            if os.environ.get("CSPAPS_GUI_SYNC"):
                job.collect(key)
            else:
                st.session_state[f"job_{key}"] = job
            st.rerun()
    if st.session_state.get(f"job_{key}") is not None:
        _job_status(key)
    return digest


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
    import re

    m = re.match(r"APS scenarios (.+): (\d+)/(\d+)$", msg)
    if m:  # "正在评估布局 B，已完成 8/20 个工况"
        return f"正在评估{wf.layout_label(m.group(1))}，已完成 {m.group(2)}/{m.group(3)} 个工况"
    m = re.match(r"(sampling check|independent cloud maps): (.+) on (\d+) scenarios", msg)
    if m:
        what = "更密工况复核" if m.group(1) == "sampling check" else "独立云图复核"
        return f"{what}：{wf.layout_label(m.group(2))}，{m.group(3)} 个工况"
    rules = [("certifying", "计算最优性证书：对偶上界与松弛取整"), ("weight ", "权衡曲线 · "),
             ("formal ranking", "正式排名（逐工况 APS）"), ("formal swap", "按年度价值接受的换位"),
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

# --------------------------------------------------------------------------- site, weather and field inputs

def _weather_site_check(case) -> None:
    """Weather source line; a warning when the site moved away from the weather file's site."""
    lat, lon = case.site.latitude, case.site.longitude
    ws = st.session_state.get("weather_site")
    if case.weather.source == "file":
        name = Path(case.weather.file).name
        origin = (f"（对应站址 {ws[0]:.3f}°N, {ws[1]:.3f}°E）"
                  if ws and ws[2] == case.weather.file and ws[0] is not None else "")
        st.caption(f"气象数据：逐时 DNI 文件 {name}{origin}")
        moved = (ws is not None and ws[2] == case.weather.file and ws[0] is not None
                 and (abs(ws[0] - lat) > 0.05 or abs(ws[1] - lon) > 0.05))
        if moved:
            _callout("warn", "站址已修改，气象文件仍属于原站址",
                     f"当前站址 {lat:.3f}°N, {lon:.3f}°E，气象文件对应 {ws[0]:.3f}°N, {ws[1]:.3f}°E。"
                     "请下载新站址的逐时 DNI、上传实测气象文件，或改用晴空模型。")
    else:
        st.caption("气象数据：晴空模型（Meinel，按海拔修正）")


def _weather_panel(case, key: str) -> None:
    """Weather source of the case; warns when the site moved away from the weather file's site."""
    from cspaps.weather import fetch_power_dni

    case = _case()[0] or case  # include edits made by the widgets above in this run
    data = st.session_state["case_dict"]
    lat, lon = case.site.latitude, case.site.longitude
    _weather_site_check(case)
    c1, c2 = st.columns(2)
    year = case.annual.year
    if c1.button(f"下载 NASA POWER {year} 年逐时 DNI", key=f"power_{key}",
                 help="从 NASA POWER 下载当前经纬度的逐时直射辐照度（需要联网），按站址时区换算到每小时中点。"):
        path = UPLOADS / f"nasa_power_dni_{lat:.4f}_{lon:.4f}_{year}.csv"
        try:
            with st.spinner("正在下载 NASA POWER 逐时 DNI …"):
                info = fetch_power_dni(lat, lon, year, case.site.timezone, path)
            data["weather"] = {"source": "file", "file": str(path)}
            st.session_state["weather_site"] = (lat, lon, str(path))
            st.session_state["weather_msg"] = (f"已下载 {info['n_hours']} 小时，年 DNI {info['annual_dni_kwh_m2']:,.0f} kWh/m²，"
                                               f"峰值 {info['peak_dni_w_m2']:.0f} W/m²")
            st.session_state["ver"] += 1
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - network or service problems are shown to the user
            st.error(f"下载失败：{exc}。可改用晴空模型或上传气象文件。")
    if c2.button("改用晴空模型", key=f"clear_{key}"):
        data["weather"] = {"source": "clear_sky"}
        data.setdefault("sun", {})["dni_model"] = "meinel"
        st.session_state["ver"] += 1
        st.rerun()
    if st.session_state.get("weather_msg"):
        st.success(st.session_state.pop("weather_msg"))
    wup = st.file_uploader("上传气象文件（EPW，或 CSV 两列 time,dni，当地时间、W/m²）", type=["epw", "csv"],
                           key=f"weather_up_{key}")
    if wup is not None and st.session_state.get(f"weather_loaded_{key}") != (wup.name, wup.size):
        path = _save_upload(wup)
        if path.suffix.lower() == ".csv":
            head = pd.read_csv(path, nrows=5)
            if not {"time", "dni"} <= set(head.columns):
                INPUT_ERRORS.append("气象 CSV 需要 time,dni 两列")
                return
        st.session_state[f"weather_loaded_{key}"] = (wup.name, wup.size)
        data["weather"] = {"source": "file", "file": str(path)}
        st.session_state["weather_site"] = (lat, lon, str(path))
        st.session_state["ver"] += 1
        st.rerun()


def _field_inputs(case) -> None:
    """Problem 1 inputs: heliostat coordinates (CSV import), heliostat size, receiver position and size."""
    data = st.session_state["case_dict"]
    fspec = data.setdefault("field", {})
    try:
        fld = build_field(case)
        n_txt = f"{fld.n} 面定日镜"
        if not fld.n or fld.dropped:
            INPUT_ERRORS.append(f"场界内 {fld.n} 面，场界或禁排区排除 {len(fld.dropped)} 面。请核对固定镜场坐标。")
    except Exception as exc:  # noqa: BLE001
        n_txt = f"无法读取：{exc}"
        INPUT_ERRORS.append("镜场坐标无法读取")
    src = (f"导入坐标 {Path(case.field.layout_file).name}" if case.field.layout_file else "布置生成器")
    st.caption(f"镜场来源：{src} · {n_txt}")
    up = st.file_uploader("导入镜场坐标 CSV（列 id,x,y[,z]，单位 m，x 东、y 北、原点为塔底）", type=["csv"], key="field_csv")
    if up is not None and st.session_state.get("field_csv_loaded") != (up.name, up.size):
        df = pd.read_csv(up)
        up.seek(0)
        if not {"x", "y"} <= set(df.columns):
            st.error("镜场 CSV 需要 x、y 两列（可选 id、z）。")
            INPUT_ERRORS.append("镜场 CSV 缺少 x、y 列")
        elif "id" in df and df["id"].duplicated().any():
            st.error("镜场 CSV 的 id 有重复。")
            INPUT_ERRORS.append("镜场 CSV 的 id 重复")
        else:
            fspec["layout_file"] = str(_save_upload(up))
            st.session_state["field_csv_loaded"] = (up.name, up.size)
            st.session_state["ver"] += 1
            st.rerun()
    c1, c2 = st.columns(2)
    template = "id,x,y,z\n1,0.0,80.0,5.0\n2,15.0,95.0,5.0\n3,-15.0,95.0,5.0\n"
    c1.download_button("下载坐标模板", template.encode("utf-8"), file_name="field_template.csv", width="stretch")
    if case.field.layout_file and case.field.generator and c2.button("改回生成器镜场", width="stretch"):
        fspec["layout_file"] = None
        st.session_state.pop("field_csv_loaded", None)
        st.session_state["ver"] += 1
        st.rerun()
    st.caption("吸热器（圆柱）中心三维坐标与尺寸")
    _receiver_center_inputs(case)
    _fields("receiver", ["diameter", "height"])
    st.caption("定日镜尺寸")
    _fields("heliostat", ["width", "height", "pivot_height"], n_cols=3)


# --------------------------------------------------------------------------- overview

def _overview_details(case) -> None:
    rec, hel = case.receiver, case.heliostat
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
    _chart(plant_scene(case, fld.xyz, height=440), "ov_scene")


# --------------------------------------------------------------------------- first-time engineer workflows

TEMPLATES = {
    "heliostats": ("镜位表 heliostats.csv", "id,x,y,z\n1,0.0,80.0,5.0\n2,15.0,95.0,5.0\n",
                   "id 为镜子编号；x 向东、y 向北、z 向上，单位 m，原点为塔底；z 为镜面旋转中心高度，可省略（取定日镜参数里的高度）。"),
    "dni_points": ("DNI 测点表 dni_points.csv", "x,y,dni\n-200,-150,820\n180,210,905\n",
                   "x、y 为测点位置（m，塔底为原点）；dni 为" + DNI_DEFINITION + "。测点范围外的镜子取最近测点的值。"),
    "layouts": ("布局表 layout_A.csv（B 同格式）", "id,x,y,z\n1,-20.0,92.0,5.0\n2,20.0,108.0,5.0\n",
                "两套布局的镜数应相同；坐标约定同镜位表。"),
    "scenarios": ("工况清单 scenarios.csv", "scenario_id,time,hours,dni_w_m2,map_id\n"
                  "S01,2024-03-20 10:00,180,760,clear\nS02,2024-06-21 13:00,220,910,east_cloud\n",
                  "time 为当地时间（也可以改用 elevation_deg、azimuth_deg 两列直接给太阳方向，°）；hours 为该工况代表的实际小时数（h）；"
                  "dni_w_m2 为该工况的 DNI（W/m²）；map_id 指向地图库里的一张图，全场相同日照写 uniform。"),
    "maps": ("DNI 地图库 dni_maps.csv", "map_id,x,y,value\nclear,-500,-500,1.0\nclear,500,-500,1.0\n",
             "每张地图给出场地上的网格点或至少四个角点（x、y，m）；value 为倍数（乘工况 DNI）或绝对 DNI（W/m²），"
             "在“地图单位”里选择。地图与布局无关，镜子按自己的位置插值取值。"),
}
COORD_NOTE = "坐标约定：以塔底为原点，X 向东，Y 向北，Z 向上，单位 m；方位角从正北顺时针计。"


def _templates(keys: list[str], key: str) -> None:
    with _card(f"templates_{key}"):
        _section(None, "使用我的数据：先下载模板",
                 "每个模板含表头、两行样例和单位说明。填好后在下面的步骤里导入；不确定时可以先用示例试算。")
        cols = st.columns(len(keys))
        for col, k in zip(cols, keys):
            title, text, note = TEMPLATES[k]
            with col:
                st.markdown(f"**{title}**")
                st.code(text.strip(), language="text")
                st.caption(note)
                st.download_button("下载模板", text.encode("utf-8"), file_name=title.split()[-1].split("（")[0],
                                   key=f"tpl_{key}_{k}", width="stretch")


def _start_example(task: str) -> None:
    from cspaps.resources import example_path

    if task == "aps":
        path = DEFAULT_CASE
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)
        st.session_state["page"] = "瞄准优化"
    else:
        path = example_path("wrong_site/wrong_site.yaml")
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)
        st.session_state["page"] = "布局优化"
        st.session_state["layout_mode"] = "compare"
        st.session_state["vc_A"] = "case:layout_A.csv"
        st.session_state["vc_B"] = "case:layout_B.csv"
    st.session_state["demo_data"] = True
    st.session_state.pop("own_data", None)


def _start_own(task: str) -> None:
    st.session_state["page"] = "瞄准优化" if task == "aps" else "布局优化"
    st.session_state["own_data"] = task
    st.session_state["demo_data"] = False
    if task == "layout":
        st.session_state["layout_mode"] = "compare"


def _demo_chip() -> list[str]:
    return ["演示数据"] if st.session_state.get("demo_data") else []


def _task_card(tag: str, title: str, need: list[str], get: list[str]) -> None:
    st.markdown(f'<div class="cs-flow"><div class="tag">{escape(tag)}</div><div class="h">{escape(title)}</div>'
                '<div class="p"><b>手里需要有什么</b></div><ul>' + "".join(f"<li>{escape(x)}</li>" for x in need) +
                '</ul><div class="p"><b>完成后能拿到什么</b></div><ul>' + "".join(f"<li>{escape(x)}</li>" for x in get) +
                "</ul></div>", unsafe_allow_html=True)


def _selected_id(event) -> int | None:
    pts = getattr(getattr(event, "selection", None), "points", None) or []
    for p in pts:
        cd = p.get("customdata")
        if cd is not None:
            return int(np.atleast_1d(cd)[0])
    return None


def _site_check(case, layouts: dict, key: str, height: int = 500) -> None:
    """Step 1: plan view with north arrow and scale; click a mirror to check its ID, coordinates and size."""
    sel = st.session_state.get(f"site_sel_{key}")
    ev = st.plotly_chart(site_preview(case, layouts, sel, height=height), theme=None, key=f"site_{key}",
                         on_select="rerun", selection_mode=("points",), config=PLOTLY_CONFIG)
    picked = _selected_id(ev)
    if picked is not None and picked != sel:
        st.session_state[f"site_sel_{key}"] = picked
        st.rerun()
    st.caption(COORD_NOTE + " 虚线为场地边界，红色点线为塔周禁排区。点一面镜可核对编号、坐标和尺寸。")
    if sel is not None:
        rows = []
        for label, (xyz, ids) in layouts.items():
            k = np.flatnonzero(np.asarray(ids) == sel)
            if k.size:
                p = np.asarray(xyz)[k[0]]
                r = float(np.hypot(p[0] - case.receiver.center[0], p[1] - case.receiver.center[1]))
                rows.append({"布局": label, "镜子编号": sel, "x 东 (m)": round(p[0], 3), "y 北 (m)": round(p[1], 3),
                             "z 上 (m)": round(p[2], 3), "距塔 (m)": round(r, 1),
                             "镜面 (m)": f"{case.heliostat.width:g} × {case.heliostat.height:g}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)


@st.cache_data(show_spinner=False)
def _sun_info(case_yaml: str, base_dir: str, stamp: str):
    try:
        case = case_from_dict(yaml.safe_load(case_yaml), base_dir=base_dir)
        ts, el, az, dni = sun_state(case, stamp)
        return float(el), float(az), float(dni)
    except Exception:  # noqa: BLE001 - informational only
        return None


def _dni_upload_check(case, upload, fld) -> list[str]:
    """Preview and row-level check of an uploaded DNI point table; returns blocking problems."""
    from cspaps.clouds import check_dni_table, coverage_of_points, load_dni_map

    try:
        raw = pd.read_csv(upload, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        upload.seek(0)
        return [f"无法读取文件：{exc}"]
    upload.seek(0)
    clean, problems = check_dni_table(raw)
    bad_rows = {p["row"] - 1 for p in problems if p["row"] is not None}
    if problems:
        for p in problems[:6]:
            st.error(p["message"])
        if len(problems) > 6:
            st.error(f"另有 {len(problems) - 6} 处错误。")
        view = raw.copy()
        view.index = np.arange(1, len(view) + 1)
        st.dataframe(view.style.apply(lambda r: ["background-color: #f8d7d3" if r.name - 1 in bad_rows else ""
                                                 for _ in r], axis=1), height=220)
        st.caption("红色为需要修改的行（行号不含表头）。其余设置都已保留，改好文件后重新导入即可继续。")
        return ["DNI 文件有错误行，请按提示修改后重新导入"]
    tmp = UPLOADS / f"_check_{upload.name}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    clean.to_csv(tmp, index=False)
    dni = load_dni_map(tmp, fld.xyz)
    pts = clean[["x", "y"]].to_numpy(float)
    inside = coverage_of_points(pts, fld.xyz[:, :2])
    st.caption(f"已读取 {len(clean)} 个测点，DNI {clean['dni'].min():.0f}–{clean['dni'].max():.0f} W/m²，按位置插值到每面定日镜。")
    if (~inside).any():
        _callout("warn", f"{int((~inside).sum())} 面镜在测点范围外",
                 "这些镜子（红圈）取最近测点的 DNI（外推），结果对它们的日照更不确定。可以补测点覆盖整个镜场。")
    _chart(dni_points_preview(case, pts, clean["dni"].to_numpy(float), fld.xyz, dni, inside), "dni_upload_prev")
    return []


def _preference_controls(case, key: str) -> None:
    """One control for both layers: "more heat <-> more uniform heating"; hard limits shown apart."""
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    lo_d = data.setdefault("layout_opt", {})
    uni = data.setdefault("uniformity", {})
    lam0 = float(case.layout_opt.uniformity_weight)
    st.markdown("**更重视吸热量 ↔ 更重视受热均匀**")
    lam = float(st.slider("λ（0 = 只看吸热量；越大越重视受热均匀）", 0.0, 1.0, lam0, 0.05, key=f"{ver}:pref_{key}",
                          help="得分 = 吸收功率 − λ × 受热不均匀量（A_H·σ_H）。默认运行（怎样瞄准）和评价（怎样排名）用同一个 λ。"))
    lo_d["uniformity_weight"] = lam
    uni.update({"enabled": True, "mode": "weighted", "weight": lam})
    allowed = 1 - case.aps.safety_margin
    st.caption(f"当前 λ = {lam:.2f}。硬约束不随偏好改变：最大允许热流 {case.receiver.max_flux_kw_m2:g} kW/m²，"
               f"安全裕度 {case.aps.safety_margin:.0%}（只允许使用上限的 {allowed:.0%}）。")
    with st.expander("高级：分别设置运行权重和评价权重"):
        sep = st.checkbox("运行权重 λ_APS 与评价权重 λ_L 分开设置", value=case.layout_opt.aps_weight is not None,
                          key=f"{ver}:sep_{key}")
        if sep:
            lo_d["aps_weight"] = float(st.slider("运行权重 λ_APS（每个工况怎样瞄准）", 0.0, 2.0,
                                                 float(case.layout_opt.aps_weight if case.layout_opt.aps_weight is not None
                                                       else lam), 0.05, key=f"{ver}:apsw_{key}"))
            st.caption("只改评价权重时，已存的瞄准方案可以直接重新计分；改运行权重需要重新求解 APS。")
        else:
            lo_d["aps_weight"] = None
        _fields("aps", ["safety_margin"], n_cols=1)


def _trial_section(case, key: str, layout_source, scenarios_dir=None) -> None:
    """Try the current preference on one scenario before computing the whole year."""
    try:
        tbl = _scenario_table(case.to_yaml(), st.session_state["base_dir"])
    except Exception as exc:  # noqa: BLE001
        st.caption(f"工况集暂不可用：{exc}")
        return
    ids = list(tbl["scenario_id"])
    default = int(np.argmax(tbl["dni_w_m2"] * tbl["hours"]))
    c1, c2 = st.columns([1.3, 1])
    sid = c1.selectbox("选一个工况试算", ids, index=default, key=f"trial_sid_{key}",
                       format_func=lambda i: f"{i} · {tbl.set_index('scenario_id').loc[i, 'map_id']}")
    prev = st.session_state.get(f"trial_last_{key}", 0.0)
    lam = float(case.layout_opt.uniformity_weight)
    with c2:
        _job_button(f"trial_{key}", f"试算 λ {prev:.2f} → {lam:.2f}", wf.task_preference_trial, sid,
                    lambdas=(prev, lam), layout=layout_source, scenarios_dir=scenarios_dir)
    _job_error(f"trial_{key}")
    res = st.session_state.get(f"res_trial_{key}")
    if res:
        tr = res["summary"]["trials"]
        a, b = tr[0], tr[-1]
        st.session_state[f"trial_last_{key}"] = b["lambda"]

        def pct(x, y):
            return f"{100 * (y / x - 1):+.1f}%" if x else "—"

        _callout("info", f"{res['summary']['scenario_id']}：λ {a['lambda']:.2f} → {b['lambda']:.2f}",
                 f"吸收功率 {a['p_abs_mw']:.3f} → {b['p_abs_mw']:.3f} MW（{pct(a['p_abs_mw'], b['p_abs_mw'])}）；"
                 f"最热点 {a['peak_flux_kw_m2']:.0f} → {b['peak_flux_kw_m2']:.0f} kW/m²；"
                 f"均匀性 CV {_format_number(a['cv_2d'], 3)} → {_format_number(b['cv_2d'], 3)}（越小越均匀）。")


def _run_summary(items: list[tuple[str, str]]) -> None:
    _status_header(items)


def _layout_options(case) -> dict:
    opts = {"initial": "初始镜场（最靠近塔的 N 面）"}
    base = Path(st.session_state["base_dir"])
    for f in sorted(base.glob("layout*.csv")):
        opts[f"case:{f.name}"] = f"算例目录里的 {f.name}"
    if st.session_state.get("res_opt") is not None:
        opts["opt"] = "本次布局优化的结果"
    opts["upload"] = "上传布局 CSV"
    return opts


def _layout_from_choice(case, tag: str, choice: str):
    """(source for task_layout_value, (xyz, ids) for previews, blocker)."""
    from cspaps.field import read_layout

    if choice == "initial":
        from cspaps.layout_value import baseline_layout

        n = case.layout_opt.n_heliostats or build_field(case).n
        b = baseline_layout(case, n)
        if b is None:
            return None, None, f"初始镜场少于 {n} 面镜"
        return "initial", (b[0], b[1]), None
    if choice == "opt":
        xyz = st.session_state["res_opt"]["result"].best.xyz
        return xyz, (xyz, np.arange(1, len(xyz) + 1)), None
    if choice.startswith("case:"):
        path = Path(st.session_state["base_dir"]) / choice[5:]
        xyz, ids = read_layout(path, case.heliostat.pivot_height)
        return str(path), (xyz, ids if ids is not None else np.arange(1, len(xyz) + 1)), None
    up = st.session_state.get(f"vc_up_{tag}")
    if up is None:
        return None, None, f"请上传布局 {tag}"
    path = _save_upload(up)
    try:
        xyz, ids = read_layout(path, case.heliostat.pivot_height)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"布局 {tag} 无法读取：{exc}"
    return str(path), (xyz, ids if ids is not None else np.arange(1, len(xyz) + 1)), None


def _scenario_source_inputs(case, key: str) -> list[str]:
    """Annual scenarios: generated from the station weather, or a scenario list + map library imported in a batch."""
    from cspaps.dni_maps import load_map_file, model_maps
    from cspaps.scenarios import check_scenario_table

    data = st.session_state["case_dict"]
    sc = data.setdefault("scenarios", {})
    blockers = []
    src = st.radio("工况从哪里来", ["weather", "file"], index=0 if case.scenarios.source == "weather" else 1,
                   horizontal=True, key=f"scen_src_{key}",
                   format_func={"weather": "由站址全年气象自动生成", "file": "导入工况清单与 DNI 地图"}.get)
    if key == "opt":
        check_file = st.file_uploader("独立复核工况 CSV（可选，共用地图库）", type=["csv"], key="validation_scenarios")
        if check_file is not None:
            sc["validation_file"] = str(_save_upload(check_file))
    if src == "file":
        sc["source"] = "file"
        c1, c2 = st.columns(2)
        up_t = c1.file_uploader("工况清单 CSV", type=["csv"], key=f"scen_tab_{key}")
        up_m = c2.file_uploader("DNI 地图库 CSV（可选，全场相同日照可省略）", type=["csv"], key=f"scen_maps_{key}")
        if up_t is not None:
            sc["file"] = str(_save_upload(up_t))
        if up_m is not None:
            sc["map_file"] = str(_save_upload(up_m))
            sc["map_model"] = "file"
        _fields("scenarios", ["map_unit"], n_cols=1)
        if not sc.get("file") and not case.scenarios.file:
            blockers.append("请导入工况清单")
            return blockers
        c2_, _ = _case()
        if c2_ is None:
            return blockers
        try:
            maps = model_maps(c2_) if c2_.scenarios.map_model != "file" else load_map_file(
                c2_, c2_.resolve(c2_.scenarios.map_file), c2_.scenarios.map_unit, c2_.scenarios.map_normalize)
        except Exception as exc:  # noqa: BLE001
            st.error(f"地图库无法读取：{exc}")
            return blockers + ["地图库无法读取"]
        try:
            table = pd.read_csv(c2_.resolve(c2_.scenarios.file), dtype=str, keep_default_na=False)
        except (OSError, ValueError) as exc:
            st.error(f"工况清单无法读取：{exc}")
            return blockers + ["工况清单无法读取"]
        problems = check_scenario_table(c2_, table, maps, Path(c2_.scenarios.map_file or "地图库").name)
        for p in problems[:8]:
            st.error(p["message"])
        if problems:
            st.caption("其他已匹配的工况仍可在下方预览；改正后重新导入对应文件即可，已导入的另一份文件会保留。")
            blockers.append("工况清单或地图库有错误")
    else:
        sc["source"] = "weather"
        _fields("scenarios", ["n_scenarios", "map_model"], n_cols=2)
        model = (sc.get("map_model") or case.scenarios.map_model)
        if model in ("half_swap", "gradient"):
            _fields("scenarios", ["map_contrast", "map_axis_deg"], n_cols=2)
        elif model == "clouds":
            _fields("scenarios", ["cloud_maps", "cloud_n", "seed"], n_cols=3)
        if model == "uniform":
            _callout("info", "本次按全场相同日照估算",
                     "只有站址气象（一个点的 DNI）时，全场每面镜读同一个 DNI。有测点图或地图库时改用“导入工况清单与 DNI 地图”。")
    return blockers


def _scenario_browser(case, layouts: dict, key: str) -> None:
    """Scenario list; click a row to preview its map with every mirror's interpolated DNI; click a mirror for its value."""
    from cspaps.gui_plots import tower_shadow_fraction
    from cspaps.scenarios import build_scenarios

    try:
        sset = build_scenarios(case)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"工况集暂不可用：{exc}")
        return
    t = sset.table()
    times = sset.example_times or [""] * len(t)
    view = pd.DataFrame({"工况": t["scenario_id"],
                         "太阳 高度角 / 方位角 (°)": [f"{e:.1f} / {a:.0f}" for e, a in zip(t["elevation_deg"], t["azimuth_deg"])],
                         "当地时间（示例）": [str(x)[:16] for x in times], "DNI 地图": t["map_id"],
                         "代表小时（实际，h）": t["actual_hours"].round(1), "工况 DNI (W/m²)": t["dni_w_m2"].round(0)})
    rec = sset.record()
    st.caption(f"{len(t)} 个工况 · 合计代表 {rec['actual_hours']:,.0f} 实际小时 · DNI 年能量 {rec['annual_dni_kwh_m2']:,.0f} kWh/m² · "
               f"指纹 {rec['fingerprint'][:8]}（比较布局时冻结同一组工况）")
    ev = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"scen_rows_{key}",
                      height=240)
    rows = getattr(getattr(ev, "selection", None), "rows", None) or [0]
    i = int(rows[0])
    m = sset.samples.map_of(i)
    sid = sset.ids[i]
    l, r = st.columns([1.3, 1])
    with l:
        allxyz = [np.asarray(v[0]) for v in layouts.values()]
        allids = [np.asarray(v[1]) for v in layouts.values()]
        xyz = np.vstack(allxyz) if allxyz else np.zeros((0, 3))
        ids = np.concatenate(allids) if allids else np.zeros(0, int)
        dni = sset.dni_at(i, xyz[:, :2]) if len(xyz) else np.zeros(0)
        ev2 = st.plotly_chart(field_dni_map(case, xyz, ids, dni, dni_map=m,
                                            sun_angles=(sset.samples.elevation[i], sset.samples.azimuth[i]),
                                            shadow_fraction=tower_shadow_fraction(case, xyz, sset.samples.elevation[i],
                                                                                  sset.samples.azimuth[i]),
                                            map_scale=float(sset.samples.dni[i]) if m.unit == "factor" else 1.0,
                                            title=f"{sid} · 地图 {m.map_id}", cmin=0, height=420),
                              theme=None, key=f"scen_map_{key}_{i}", on_select="rerun", selection_mode=("points",),
                              config=PLOTLY_CONFIG)
        pick = _selected_id(ev2)
    with r:
        st.markdown(f"**{sid}**　{m.description or m.map_id}")
        st.markdown(f"- 单位：{'倍数 × 工况 DNI' if m.unit == 'factor' else 'DNI (W/m²)'}；{m.normalization}\n"
                    f"- 插值：{m.interpolation}\n- 坐标系：{m.crs}\n- 来源：{m.source}")
        if len(xyz):
            out = ((xyz[:, 0] < m.x[0]) | (xyz[:, 0] > m.x[-1]) | (xyz[:, 1] < m.y[0]) | (xyz[:, 1] > m.y[-1]))
            if out.any() and m.map_id != "uniform":
                st.warning(f"{int(out.sum())} 面镜在地图栅格以外，按栅格边缘值外推。")
        if pick is not None:
            for label, (lx, lid) in layouts.items():
                k = np.flatnonzero(np.asarray(lid) == pick)
                if k.size:
                    p = np.asarray(lx)[k[0]]
                    st.info(f"{label} · 镜 {pick}（{p[0]:.1f}, {p[1]:.1f}）m：插值 DNI "
                            f"{float(sset.dni_at(i, p[None, :2])[0]):.1f} W/m²")
        else:
            st.caption("点地图上的一面镜，查看它在这个工况下插值得到的 DNI。")


def _compare_workspace(case) -> None:
    """Compare two given layouts on one set of annual scenarios (the plan's main path for engineers)."""
    if st.session_state.get("own_data") == "layout":
        _templates(["layouts", "scenarios", "maps"], "layout")
    opts = _layout_options(case)
    with _card("cmp_site"):
        _section(1, "先确认场地和两套布局", COORD_NOTE)
        c1, c2 = st.columns(2)
        with c1:
            _fields("site", ["latitude", "longitude", "timezone"], n_cols=3)
            _weather_site_check(_case()[0] or case)
        with c2:
            _receiver_center_inputs(case)
        choices, layouts, blockers = {}, {}, []
        c1, c2 = st.columns(2)
        for col, tag in ((c1, "A"), (c2, "B")):
            with col:
                keys = list(opts)
                cur = st.session_state.get(f"vc_{tag}")
                if cur is not None and cur not in keys:
                    del st.session_state[f"vc_{tag}"]
                    cur = None
                if cur is not None:  # set by an example or an earlier choice: no default index
                    choices[tag] = st.selectbox(f"布局 {tag}", keys, format_func=opts.get, key=f"vc_{tag}")
                else:
                    idx = keys.index("initial") if tag == "B" or len(keys) < 3 else 1
                    choices[tag] = st.selectbox(f"布局 {tag}", keys, index=idx, format_func=opts.get, key=f"vc_{tag}")
                if choices[tag] == "upload":
                    st.file_uploader(f"布局 {tag} CSV（id,x,y[,z]）", type=["csv"], key=f"vc_up_{tag}")
        sources = {}
        for tag in ("A", "B"):
            src, pv, err = _layout_from_choice(case, tag, choices[tag])
            if err:
                blockers.append(err)
            else:
                sources[tag], layouts[f"布局 {tag}"] = src, pv
        if layouts:
            _site_check(case, layouts, "cmp")
            n = {k: len(v[0]) for k, v in layouts.items()}
            if len(set(n.values())) > 1:
                _callout("warn", "两套布局镜数不同", "、".join(f"{k} {v} 面" for k, v in n.items()) +
                         "。正式排名要求两套布局均达到目标镜数。")
    with _card("cmp_scen"):
        _section(2, "看看日照落在哪些镜子上", "工况 = 太阳方向 + 代表小时数 + 场地 DNI 地图。" + DNI_DEFINITION + "。")
        blockers += _scenario_source_inputs(case, "cmp")
        c3, _ = _case()
        if c3 is not None and not blockers:
            _scenario_browser(c3, layouts, "cmp")
    with _card("cmp_goal"):
        _section(3, "用工程语言设定目标", "一个偏好同时控制运行（怎样瞄准）和评价（怎样排名）。")
        _preference_controls(case, "cmp")
        c4, _ = _case()
        if c4 is not None and not blockers and "A" in sources:
            _trial_section(c4, "cmp", sources["A"])
    c5, err = _case()
    with _card("cmp_run"):
        _section(4, "确认要算什么", "开始前核对规模与口径；计算中可以取消，已完成的工况会保留。")
        if c5 is not None and not blockers:
            try:
                tbl = _scenario_table(c5.to_yaml(), st.session_state["base_dir"])
                n_s, hrs = len(tbl), float(tbl["actual_hours"].sum())
            except Exception:  # noqa: BLE001
                n_s, hrs = 0, 0.0
            region = ("整个吸热面" if not c5.uniformity.panels and not c5.uniformity.z_range else
                      f"管屏 {c5.uniformity.panels or '全部'}，高度 {c5.uniformity.z_range or '全高'}")
            src = (f"工况清单 {Path(c5.scenarios.file).name}" if c5.scenarios.source == "file" else
                   f"站址气象 {Path(c5.weather.file).name if c5.weather.source == 'file' else '晴空模型'}")
            _run_summary([("镜数", " / ".join(f"{k} {len(v[0])} 面" for k, v in layouts.items())),
                          ("工况数", f"{n_s} 个"), ("合计代表小时（实际）", f"{hrs:,.0f} h"), ("数据来源", src),
                          ("考核区域", region), ("安全裕度", f"{c5.aps.safety_margin:.0%}"),
                          ("偏好 λ", f"{c5.layout_opt.uniformity_weight:.2f}" +
                           ("" if c5.layout_opt.aps_weight is None else f"（运行 {c5.layout_opt.aps_weight:.2f}）"))])
        frozen = None
        if st.session_state.get("res_opt") is not None:
            sdir = Path(st.session_state["res_opt"]["run_dir"]) / "formal" / "scenario_set"
            if sdir.exists() and st.checkbox("沿用本次布局优化冻结的工况集", value=True, key="cmp_frozen"):
                frozen = str(sdir)
        digest = _job_button("lv", "▶  按年度价值比较两套布局", wf.task_layout_value, layouts=sources, scenarios_dir=frozen,
                    blockers=blockers, progress=True, cancellable=True)
        _job_error("lv")
    res = st.session_state.get("res_lv")
    if res is not None:
        rc = res_case(res)
        if res.get("input_digest") != digest:
            _callout("warn", "参数已修改，需要重新计算",
                     f"下面的结果对应上一次的参数：λ = {rc.layout_opt.uniformity_weight:g}，"
                     f"工况 {len(res['scenario_set'])} 个，来源 {res['scenario_set'].meta.get('source', '—')}。")
        for name, problems in res["summary"].get("excluded_layouts", {}).items():
            st.error(f"{wf.layout_label(name)} 未参与排名 · {'，'.join(problems)}")
        if res["ranking"].empty:
            return
        _render_values(rc, res["values"], res["ranking"], res["scenario_set"], "lv", run_dir=res["run_dir"])


def page_overview(case) -> None:
    _hero("CSP-APS · 塔式光热镜场", "这批镜子该瞄哪里？同样数量的镜子建在哪里？",
          "从手里的资料出发选一个任务。每个任务都带一份可以直接运行的示例；用自己的数据时先下载对应的模板。",
          ([case.name] if case else []) + _demo_chip())
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _task_card("任务一 · 镜子已经装好", "计算瞄准点",
                   ["镜位表（编号与坐标）、镜面尺寸", "接收器位置和尺寸", "一个时刻的日照条件（站址气象或 DNI 测点）"],
                   ["逐镜瞄准点", "吸收功率", "接收器热点位置", "距离允许热流上限还有多少"])
        a, b = st.columns(2)
        a.button("用示例试算", key="home_aps_demo", type="primary", on_click=_start_example, args=("aps",), width="stretch")
        b.button("使用我的数据", key="home_aps_own", on_click=_start_own, args=("aps",), width="stretch")
    with c2:
        _task_card("任务二 · 镜子尚未安装", "比较全年布局",
                   ["场地边界、镜数与规格", "接收器与站址", "两套布局，或让软件搜索", "全年代表工况（可带 DNI 地图）"],
                   ["布局 A / B 的全年吸热量", "均匀性扣分与综合得分", "哪些工况决定了胜负"])
        a, b = st.columns(2)
        a.button("用示例试算", key="home_layout_demo", type="primary", on_click=_start_example, args=("layout",),
                 width="stretch")
        b.button("使用我的数据", key="home_layout_own", on_click=_start_own, args=("layout",), width="stretch")
    if st.session_state.get("demo_data"):
        st.caption("当前载入的是演示数据（标记“演示数据”）。镜场与吸热器为模拟设计输入，气象为 NASA POWER 2024 年逐时 DNI。")
    if case is None:
        st.warning("当前算例参数无效，请在「参数设置」中修正。")
        return
    with st.expander("当前算例概况与三维视图"):
        _overview_details(case)
    single, opt = st.session_state.get("res_single"), st.session_state.get("res_opt")
    if single or opt:
        _section(None, "本次会话的最新结果")
        items = []
        if single:
            s = single["summary"]
            items += [("瞄准方案 · 吸收功率", f"{s['absorbed_power_mw']:.2f}", "MW", s["time"][:16].replace("T", " "))]
        if opt and (opt["summary"].get("formal") or {}).get("winner"):
            w = opt["summary"]["formal"]["winner_totals"]
            items += [("布局 · 综合得分", _format_number(w["value_mwh"], 0), "MWh 等效", wf.layout_label(w["name"]))]
        _kpis(items)


# --------------------------------------------------------------------------- problem 1: aim points

GOALS = {"power": "吸收功率最大", "weighted": "功率 + 均匀度（权重 λ）", "cv_limit": "均匀度上限（CV ≤ ε）"}
DNI_SOURCES = {"site": "站址气象（全场均匀 DNI）", "cloud": "云影场景（模拟空间分布）", "upload": "导入实测 DNI 分布"}


def page_aps(case) -> None:
    _hero("问题一 · 运行调度", "瞄准点优化（APS）",
          "在给定镜场与 DNI 工况下，选择每面镜的瞄准位置。"
          "调整均匀性权重或 CV 上限，查看吸收功率、热点"
          "及目标差距。",
          [f"{case.name}" if case else "算例无效"] + ([f"{build_field(case).n} 面定日镜",
                                                       f"吸热器中心 ({', '.join(f'{c:g}' for c in case.receiver.center)}) m"]
                                                      if case else []) + _demo_chip())
    if case is None:
        st.warning("请先在「参数设置」中修正当前算例。")
        return
    if st.session_state.get("own_data") == "aps":
        _templates(["heliostats", "dni_points"], "aps")
    ver = st.session_state["ver"]
    data = st.session_state["case_dict"]
    if st.session_state.get("res_single") is not None:
        right, left = st.tabs(["计算结果", "调整参数"])
    else:
        left, right = st.columns([1, 2.05], gap="large")
    up_now = up_fc = up_prev = None
    with left:
        with _card("aps_conditions"):
            _section(1, "运行工况", "选择时刻与镜场 DNI 分布。太阳位置由经纬度和时区计算。")
            default_time = pd.Timestamp(case.realtime.start)
            c1, c2 = st.columns(2)
            date = c1.date_input("日期", value=default_time.date(), key=f"s_date:{case.name}")
            tm = c2.time_input("当地时间", value=default_time.time(), key=f"s_time:{case.name}", step=300)
            with st.expander("镜场坐标、定日镜与吸热器（给定条件）", expanded=bool(case.field.layout_file)):
                _field_inputs(case)
            info = _sun_info(case.to_yaml(), st.session_state["base_dir"], f"{date} {tm}")
            if info:
                st.caption(f"太阳方向由站址和当地时间计算：高度角 {info[0]:.1f}°，方位角 {info[1]:.0f}°（北 0°、东 90°）。")
            st.caption(DNI_DEFINITION + "。")
            dni_src = st.radio("日照（DNI）怎样分布", list(DNI_SOURCES), format_func=DNI_SOURCES.get, key="aps_dni_src",
                               help="站址气象：全场同一 DNI；云影场景：按回放设置生成云影，DNI 随镜位变化；"
                                    "导入：CSV (x, y, dni) 插值到每面镜。")
            aps_blockers = []
            if dni_src == "site" and info:
                _callout("info", "本次按全场相同日照估算",
                         f"站址气象在该时刻的 DNI 为 {info[2]:.0f} W/m²，每面镜都读这个值。有测点图时改选“导入实测 DNI 分布”。")
            if dni_src == "upload":
                up_now = st.file_uploader("当前镜场 DNI（CSV x,y,dni，W/m²）", type=["csv"], key="s_now")
                if up_now is None:
                    aps_blockers.append("请上传当前镜场 DNI 分布")
                else:
                    try:
                        fld_now = build_field(case)
                        aps_blockers += _dni_upload_check(case, up_now, fld_now)
                    except Exception as exc:  # noqa: BLE001
                        aps_blockers.append(f"镜场无法读取：{exc}")
            else:
                with st.expander("站址与气象数据"):
                    _fields("site", ["latitude", "longitude", "elevation_m", "timezone"])
                    _weather_panel(case, "aps")
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
                st.markdown("**更重视吸热量 ↔ 更重视受热均匀**")
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
            st.caption(f"硬约束不随偏好改变：最大允许热流 {case.receiver.max_flux_kw_m2:g} kW/m²，安全裕度 "
                       f"{case.aps.safety_margin:.0%}（只允许使用上限的 {1 - case.aps.safety_margin:.0%}）。")
            with st.expander("考核区域、最低负荷与安全裕度"):
                _fields("uniformity", ["panels", "z_range", "min_load_mw"])
                _fields("aps", ["safety_margin", "allow_defocus"])
            with st.expander("候选瞄准点、算法与最优性证书"):
                methods = CHOICES["aps.method"]
                method = st.selectbox("瞄准算法", methods, index=methods.index(case.aps.method),
                                      format_func=lambda item: CHOICE_LABELS["aps.method"].get(item, item),
                                      help=HELP["aps.method"], key=f"s_method:{case.name}")
                _fields("aps", ["n_height_levels", "height_span", "azimuth_offsets_deg", "certify",
                                "certify_time_s", "certify_rounds", "coupled_exact_limit", "coupled_trials"])
        case, err = _case()
        if err:
            st.error(err)
            return
        stamp = f"{date} {tm}"
        try:
            n_now = build_field(case).n
        except Exception:  # noqa: BLE001
            n_now = 0
        _run_summary([("镜数", f"{n_now} 面"), ("时刻", stamp), ("日照", DNI_SOURCES[dni_src]),
                      ("考核区域", "整个吸热面" if not case.uniformity.panels and not case.uniformity.z_range else "部分区域"),
                      ("安全裕度", f"{case.aps.safety_margin:.0%}"), ("算法", CHOICE_LABELS["aps.method"].get(method, method))])
        digest = _job_button("single", "▶  计算瞄准方案", wf.task_aim, stamp, method=method, blockers=aps_blockers,
                    dni_map=_save_upload(up_now) if up_now else None,
                    forecast_dni_map=_save_upload(up_fc) if up_fc else None,
                    prev_aims=_save_upload(up_prev) if up_prev else None,
                    dni_source="cloud" if dni_src == "cloud" else "site", progress=True)
        _job_error("single")
    if st.session_state.get("res_recalc") is not None:  # the recomputation replaces the shown result
        st.session_state["res_single"] = st.session_state.pop("res_recalc")
        st.rerun()
    with right:
        res = st.session_state.get("res_single")
        if res is None:
            with _card("aps_preview"):
                _section(None, "先确认场地", "核对塔、禁排区和镜位后再计算。结果会先给出吸收功率、热点所在管屏和热流余量，"
                         "再给出逐镜瞄准表。")
                fld = build_field(case)
                _site_check(case, {"镜场": (fld.xyz, fld.ids)}, "aps")
        else:
            done = load_case(Path(res["run_dir"]) / "inputs_case.yaml")
            if res.get("input_digest") != digest:
                _callout("warn", "参数已修改，需要重新计算",
                         f"下面的结果对应上一次的参数（均匀度目标 "
                         f"{'λ = %g' % done.uniformity.weight if done.uniformity.enabled and done.uniformity.mode == 'weighted' else ('CV ≤ %g' % done.uniformity.max_cv if done.uniformity.enabled else '功率优先')}，"
                         f"安全裕度 {done.aps.safety_margin:.0%}）。可以在“调整参数”里改回，或重新计算。")
            _aps_results(res)
    _pareto_section(case, f"{date} {tm}", dni_src, up_now)
    _scenario_check_section(case)


@st.cache_data(show_spinner=False)
def _scenario_table(case_yaml: str, base_dir: str):
    from cspaps.scenarios import build_scenarios

    case = case_from_dict(yaml.safe_load(case_yaml), base_dir=base_dir)
    return build_scenarios(case).table()


def _scenario_check_section(case) -> None:
    """Task 1 for one annual scenario (sun + site DNI map) on the current field: the check of a layout score."""
    with _card("aps_scenario"):
        _section(4, "年度工况复核（问题二的工况与 DNI 地图）",
                 "从年度工况集中选一个工况：太阳位置与场地 DNI 地图都来自工况集，各镜按自己的位置读 DNI。"
                 "求解设置与布局正式排名相同，所得 P、U 与布局工作区该工况的记录逐位一致。")
        try:
            tbl = _scenario_table(case.to_yaml(), st.session_state["base_dir"])
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(f"无法生成工况集：{exc}")
            return
        c1, c2 = st.columns([1.2, 1])
        sid = c1.selectbox("工况", list(tbl["scenario_id"]), key="scen_pick",
                           format_func=lambda i: (lambda r: f"{i} · {r['map_id']} · 高度角 {r['elevation_deg']:.0f}° · "
                                                            f"{r['hours']:.0f} h")(tbl.set_index("scenario_id").loc[i]))
        c2.caption(f"共 {len(tbl)} 个工况 · 地图 {', '.join(sorted(set(tbl['map_id'])))}")
        _job_button("scen", "▶  求解该工况", wf.task_scenario_aps, sid)
        _job_error("scen")
    res = st.session_state.get("res_scen")
    if res is None:
        return
    from cspaps.layout_value import STATUS_ZH, eval_grid_shape

    rec, fld = res["record"], res["field"]
    p = rec.plan
    c = res_case(res)
    nz, na = eval_grid_shape(c)
    grid = build_grid(c.receiver, na, nz)
    row = res["summary"]["scenario"]
    _kpis([("工况", rec.scenario_id, "", f"{rec.map_id} · 代表 {rec.hours:.1f} h · 太阳高度角 {rec.elevation_deg:.1f}°"),
           ("吸收功率 P", f"{p.p_abs_w / 1e6:.3f}", "MW", STATUS_ZH.get(p.status, p.status)),
           ("U = A_H·σ_H", f"{p.u_w / 1e6:.3f}", "MW", f"二维 CV {_format_number(p.cv_2d, 3)}"),
           ("J = P − λ_APS·U", f"{p.objective_w / 1e6:.3f}", "MW", f"差距 ≤ {_pct(p.gap, 2)}"),
           ("该工况贡献 h·(P − λ_L·U)", f"{row['contribution_mwh']:,.1f}", "MWhₜₕ", f"λ_L = {row['lambda_layout']:g}")])
    l, r = st.columns(2)
    with l:
        _chart(field_dni_map(c, fld.xyz, fld.ids, p.dni, title=f"{rec.scenario_id} 各镜读到的 DNI", cmin=0,
                             inactive=~p.active, height=420, sun_angles=(rec.elevation_deg, rec.azimuth_deg)), "scen_dni")
    with r:
        _chart(receiver_aims_map(c, grid, p.flux, p.aim_azimuth, p.aim_z, p.active, fld.ids, p.dni,
                                 title="吸热器热流与每面镜中心的瞄准点", height=420, cmin=0), "scen_rec")
    _chart(aim_scene_3d(c, fld.xyz, p.dni, p.aim_azimuth, p.aim_z, p.active, title="三维示意（抽样光束）", cmin=0,
                       sun_angles=(rec.elevation_deg, rec.azimuth_deg), beam_power=p.mirror_power_w),
           "scen_3d")
    _show_files(res, images=[])


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
    eligible = (s["status"] == "ok" and chk["status"] == "ok"  # the safety margin is part of the constraint
                and uni_status in ("weighted", "feasible", "disabled"))
    state = STATE_ZH.get(s["status"], s["status"])
    if chk["status"] == "over_limit":
        state = "本方案通量超限"
    target = "功率优先"
    if case.uniformity.enabled:
        target = (f"CV ≤ {case.uniformity.max_cv:.3f}" if case.uniformity.mode == "cv_limit"
                  else f"λ = {case.uniformity.weight:g}")
    _section(None, "计算结果", f"{s['time'][:19].replace('T', ' ')} · {CHOICE_LABELS['aps.method'][method]} · {source}")
    _aps_headline(s, case, res, inputs)
    gap = cert.get("gap") if cert else None
    _kpis([
        ("吸收热功率", f"{s['absorbed_power_mw']:.3f}", "MW", f"截获 {s['intercepted_power_mw']:.3f} MW · DNI 均值 {s['dni_mean_w_m2']:.0f} W/m²"),
        ("考核区二维 CV", _format_number(aps.get("cv_2d_H", s.get("cv_2d")), 4), "",
         f"周向 {_format_number(aps.get('cv_circ_H', s.get('cv_circumferential')))} · 高度 {_format_number(aps.get('cv_height_H', s.get('cv_height')))}"),
        ("峰值通量（细网格复核）", f"{chk['peak_flux_kw_m2']:.0f}", "kW/m²",
         f"{chk['hotspot']['panel']} 号管屏 · 距上限 {chk['margin_to_limit_kw_m2']:.0f} kW/m²",
         "good" if chk["status"] == "ok" else ("warn" if chk["status"] == "margin_reduced" else "bad")),
        ("瞄准目标 J 的差距", f"≤ {_pct(gap)}" if gap is not None else "—", "",
         (f"相对离散候选集（每镜 {len(case.aps.azimuth_offsets_deg)}×{case.aps.n_height_levels} 个瞄准点）"
          f"在 {s['time'][:16].replace('T', ' ')} 的上界 {cert['upper_bound_mw']:.3f} MW；"
          f"相对连续瞄准的保守差距 {_pct(cert.get('continuous_gap'))}") if cert else "未计算证书",
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
            proof = {"exhaustive_joint_poses": "联合姿态已穷举", "joint_flux_support": "热流离散度支撑平面上界",
                     "unshaded_energy": "逐镜无互遮功率上界"}.get(cert.get("method"), "模型上界")
            _callout("ok" if proven else "info", "候选集全局最优" if proven else "已找到可行瞄准方案",
                     f"{proof} · 已检查 {cert.get('evaluated_joint_poses', 0)} 组姿态。"
                     f"{mode_txt} 的方案值为 {cert['objective_found_mw']:.4f} MW，上界为 "
                     f"{cert['upper_bound_mw']:.4f} MW，差距 ≤ {_pct(gap, 3)}。")
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
    aims_top = pd.read_csv(res["files"]["aims_csv"])
    shade = aims_top.get("tower_shadow_fraction", pd.Series(0., index=aims_top.index))
    aims_top["瞄准状态"] = np.select([aims_top["active"] == 0, shade == 1, shade > 0],
                                       ["散焦", "保持瞄准 · 全塔影", "瞄准 · 部分塔影"], default="瞄准")
    st.markdown("**逐镜瞄准表**（完整表在“逐镜方案”中下载）")
    st.dataframe(aims_top[["id", "x", "y", "瞄准状态", "aim_azimuth_deg", "aim_z", "dni_w_m2", "p_receiver_kw"]].rename(columns={
        "id": "镜子 ID", "x": "x 东 (m)", "y": "y 北 (m)", "active": "工作", "aim_azimuth_deg": "瞄准方位 (°)",
        "aim_z": "瞄准高度 (m)", "dni_w_m2": "DNI (W/m²)", "p_receiver_kw": "送达功率 (kW)"}), hide_index=True, height=220)
    tabs = st.tabs(["吸热器热流与瞄准点", "三维瞄准示意", "镜场", "最优性证书", "效率与损失", "逐镜方案"])
    with tabs[0]:
        color_by = st.radio("瞄准点着色", ["该镜 DNI", "镜子到塔的距离", "送达功率"], horizontal=True, key="s_aim_color")
        dist = np.hypot(fld.xyz[:, 0] - case.receiver.center[0], fld.xyz[:, 1] - case.receiver.center[1])
        vals, vlab = {"该镜 DNI": (snap.dni, "DNI (W/m²)"), "镜子到塔的距离": (dist, "距塔 (m)"),
                      "送达功率": (snap.power["intercept"] / 1e3, "送达 (kW)")}[color_by]
        _chart(receiver_aims_map(case, snap.grid, snap.flux, snap.aim_azimuth, snap.aim_z, snap.active, fld.ids, vals, vlab,
                                 title="吸热器展开图 · 入射热流与每面镜中心的瞄准点（× 为细网格复核热点）",
                                 hotspot=chk, height=460), "s_rec")
        st.caption("每个点是一面定日镜的镜面中心光线在吸热器上的落点（瞄准点），悬停查看镜子编号与坐标。"
                   "横轴按罗盘方位展开圆柱面：北 0°、东 90°。")
    with tabs[1]:
        l, r = st.columns(2)
        with l:
            _chart(aim_scene_3d(case, fld.xyz, snap.dni, snap.aim_azimuth, snap.aim_z, snap.active,
                                sun_angles=(snap.sun_elevation, snap.sun_azimuth), beam_power=snap.power["intercept"],
                                shadow_fraction=snap.tower_fraction,
                                title="镜面中心 → 瞄准点（抽样光束，镜子按 DNI 着色）"), "s_scene3d")
        with r:
            _chart(receiver_aims_3d(case, snap.grid, snap.flux, fld.xyz, snap.aim_azimuth, snap.aim_z, snap.active,
                                    snap.dni, fld.ids, beam_power=snap.power["intercept"], title="吸热器近景：入射热流与全部瞄准点"), "s_rec3d")
    with tabs[2]:
        color_by = st.radio("着色", ["DNI", "瞄准高度", "光学效率"], horizontal=True, key="s_field_color")
        vals, label, scale = {"DNI": (snap.dni, "DNI (W/m²)", "Blues"),
                              "瞄准高度": (snap.aim_z, "瞄准高度 z (m)", "RdBu"),
                              "光学效率": (snap.eta_total, "光学效率", "Viridis")}[color_by]
        _chart(field_map(case, fld.xyz, fld.ids, values=vals, label=label, colorscale=scale,
                         title=f"镜场 · 按{color_by}着色（× 为散焦镜面）", inactive=~snap.active,
                         sun_angles=(snap.sun_elevation, snap.sun_azimuth), shadow_fraction=snap.tower_fraction), "s_field")
        if snap.tower_fraction is not None and snap.tower_fraction.any():
            st.caption(f"紫色区域为塔影在镜轴高度处的投影。全遮挡且保持瞄准 {sum(snap.active & (snap.tower_fraction == 1))} 面，"
                       f"部分遮挡 {sum((snap.tower_fraction > 0) & (snap.tower_fraction < 1))} 面。悬停可查看逐镜遮挡比例。")
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


def _aps_headline(s: dict, case, res: dict, inputs: dict) -> None:
    """Result first, in plain words: absorbed power, the change against centre aiming, hotspot and margin."""
    p = s["absorbed_power_mw"]
    chk = s["flux_check"]
    mc = s.get("margin_check") or {}
    allowed = mc.get("allowed_fraction", 1 - case.aps.safety_margin)
    frac = mc.get("peak_fraction", chk.get("peak_to_limit", 0.0))
    peak = chk["peak_flux_kw_m2"]
    limit_kw = peak / frac if frac else case.receiver.max_flux_kw_m2
    if p <= 1e-9:
        tracking = int(res["snapshot"].active.sum())
        text = ("全部镜子散焦，吸收功率为零，CV 无法评价。" if not tracking
                else f"{tracking} 面镜保持瞄准，当前吸收功率为零，CV 无法评价。")
    else:
        text = f"采用目前的瞄准方案，接收器吸收 {p:.2f} MW。"
        ref = s.get("center_reference")
        if ref:
            d = p - ref["absorbed_power_mw"]
            verb = "降至" if peak < ref["peak_flux_kw_m2"] else "升至"
            text += (f"与中心瞄准相比{'少' if d < 0 else '多'}吸收 {abs(d):.2f} MW，最热点从 {ref['peak_flux_kw_m2']:.0f} "
                     f"{verb} {peak:.0f} kW/m²。")
        text += (f"最热点在第 {mc.get('hotspot_panel', chk['hotspot']['panel'])} 号管屏，距项目允许热流"
                 f"（上限的 {allowed:.0%}，{allowed * limit_kw:.0f} kW/m²）还有 {allowed * limit_kw - peak:.0f} kW/m²。")
    _callout("ok" if mc.get("within_margin", True) else "warn", "结果", text)
    if p > 1e-9 and not mc.get("within_margin", True):
        _callout("bad" if not mc.get("within_limit", True) else "warn", "超出项目安全裕度",
                 f"项目允许使用上限的 {allowed:.0%}，当前为 {frac:.0%}。热点在第 {mc.get('hotspot_panel')} 号管屏，"
                 f"方位 {mc.get('hotspot_azimuth_deg', 0):.0f}°、高度 {mc.get('hotspot_z_m', 0):.2f} m（图中 × 标记）。"
                 "年度评价按同样的规则把这类方案计为不可行。")
        _job_button("recalc", "重新计算瞄准（贪心 APS，满足安全裕度）", wf.task_aim, inputs["time"], method="greedy",
                    completed_case=case, dni=inputs.get("dni"), dni_map=inputs.get("dni_map"),
                    forecast_dni_map=inputs.get("forecast_dni_map"), dni_source=inputs.get("dni_source", "site"),
                    progress=True)
        _job_error("recalc")
    with st.expander("CV 是什么意思"):
        st.write("CV（变异系数）= 表面热流的标准差 ÷ 平均值，数值越小，受热越均匀。功率用 MW 表示当时的功率；"
                 "全年吸热量用 MWh；扣除不均匀性后的得分记为“MWh 等效”。")


LOSS_ZH = {"cosine": "余弦", "reflectivity": "反射/清洁", "shading": "阴影", "blocking": "遮挡",
           "attenuation": "大气衰减", "intercept": "截断"}


def _certificate_tab(res: dict, cert: dict | None) -> None:
    if not cert:
        st.info("本次计算未启用最优性证书（瞄准算法为中心瞄准，或在「候选瞄准点、算法与最优性证书」中关闭了证书）。")
        return
    exact = cert.get("method") == "exhaustive_joint_poses"
    rows = [
        ("证明范围", "离散候选瞄准点 · 联合镜面姿态 · 数值通量复核"),
        ("求解依据", "全部可选组合穷举" if exact else
         "无遮挡热流响应与离散度的支撑平面" if cert.get("method") == "joint_flux_support" else "无遮挡最大功率之和"),
        ("已评估组合", cert.get("evaluated_joint_poses", "—")),
        ("每镜候选瞄准点", cert.get("n_candidates")),
        ("均匀度网格（高 × 周向）", " × ".join(map(str, cert.get("objective_grid", [])))),
        ("方案目标 / 上界（MW）", f"{_format_number(cert.get('objective_found_mw'), 4)} / {_format_number(cert.get('upper_bound_mw'), 4)}"),
        ("目标差距", _pct(cert.get("gap"), 3)),
        ("任意连续瞄准位置的能量上界（MW）", _format_number(cert.get("continuous_upper_bound_mw"), 4)),
        ("相对连续瞄准上界的保守差距", _pct(cert.get("continuous_gap"), 3)),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["项目", "结果"]).astype(str), hide_index=True)
    st.caption("每次评价都按各镜自己的瞄准位置重算阴影与遮挡。无互遮能量上界允许各镜独立取得最大功率，"
               "均匀性惩罚非负，因此同样界定加权目标的最大值。")
    st.divider()
    _section(None, "候选点加密收敛性",
             "证书针对离散候选瞄准点。在嵌套加密的候选网格上（每级包含上一级，方位与高度步长减半）从上一级方案出发重新求解并给出上界，"
             "观察本方案与上界随瞄准分辨率的收敛——当候选网格达到瞄准控制的指令分辨率时，证书即覆盖实际可执行的全部方案。"
             "三级（21 / 65 / 225 个候选点）约需 4 分钟。")
    inputs = json.loads((Path(res["run_dir"]) / "run.json").read_text(encoding="utf-8"))["arguments"]
    case = res_case(res)
    span = st.number_input("加密搜索的最大方位偏移 (°)", min_value=0., max_value=89.,
                           value=float(min(89., max(map(abs, case.aps.azimuth_offsets_deg or [0.])))), key="refine_az_span")
    _job_button("refine", "运行加密收敛分析", wf.task_aps_refinement, inputs["time"], azimuth_span=span,
                completed_case=load_case(Path(res["run_dir"]) / "inputs_case.yaml"), dni=inputs.get("dni"),
                forecast_dni_map=inputs.get("forecast_dni_map"), dni_map=inputs.get("dni_map"),
                dni_source=inputs.get("dni_source", "site"), progress=True)
    _job_error("refine")
    ref = st.session_state.get("res_refine")
    if ref:
        lv = ref["summary"]["levels"]
        a, b = st.columns([1.3, 1])
        with a:
            _chart(refinement_chart(lv), "refine_chart")
        with b:
            st.dataframe(pd.DataFrame({
                "候选网格": [f"{r['n_azimuth']}×{r['n_height']}" for r in lv],
                "方位步长 (°)": [r["azimuth_step_deg"] for r in lv], "高度步长 (m)": [r["height_step_m"] for r in lv],
                "本方案 (MW)": [_format_number(r.get("objective_found_mw"), 4) for r in lv],
                "上界 (MW)": [round(r["upper_bound_mw"], 4) for r in lv], "差距": [_pct(r["gap"], 3) for r in lv],
                "二维 CV": [r["cv_2d_H"] for r in lv]}), hide_index=True)
            if len(lv) > 1 and lv[0].get("objective_found_mw") and lv[-1].get("objective_found_mw") is not None:
                st.caption(f"最细网格的方案比默认网格提高 {100 * (lv[-1]['objective_found_mw'] / lv[0]['objective_found_mw'] - 1):+.2f}%；"
                           f"上界逐级增量 " + " → ".join(f"{lv[i + 1]['upper_bound_mw'] - lv[i]['upper_bound_mw']:+.3f}"
                                                   for i in range(len(lv) - 1)) + " MW。")


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
                        blockers=["请上传当前镜场 DNI 分布"] if dni_src == "upload" and up_now is None else None,
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
    "fast": ("快速 · 6 个工况正式排名", {"maxiter": 2, "popsize": 3, "n_sun_bins": 8, "polish_rounds": 1, "refine_top": 1,
                                       "review_n_sun_bins": 4, "swap_rounds": 1, "review_check_hourly": False,
                                       "formal_swap_rounds": 0, "dense_check_scenarios": 0},
             {"n_scenarios": 6}),
    "standard": ("标准 · 12 个工况正式排名", {"maxiter": 4, "popsize": 4, "n_sun_bins": 12, "polish_rounds": 2,
                                          "refine_top": 2, "review_n_sun_bins": 6, "swap_rounds": 1,
                                          "review_check_hourly": False, "formal_swap_rounds": 1, "dense_check_scenarios": 0},
                 {"n_scenarios": 12}),
    "fine": ("精细 · 20 个工况 + 40 工况复核", {"maxiter": 10, "popsize": 6, "n_sun_bins": 24, "polish_rounds": 4,
                                            "refine_top": 3, "review_n_sun_bins": 12, "swap_rounds": 2,
                                            "review_check_hourly": True, "formal_swap_rounds": 2, "dense_check_scenarios": 40},
             {"n_scenarios": 20}),
}


def _apply_preset() -> None:
    key = st.session_state.get("layout_preset")
    if key in PRESETS:
        st.session_state["case_dict"].setdefault("layout_opt", {}).update(PRESETS[key][1])
        st.session_state["case_dict"].setdefault("scenarios", {}).update(PRESETS[key][2])
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
            INPUT_ERRORS.append("场地边界无效（需要 x_max > x_min、y_max > y_min）")
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
          "给定场地范围、镜数与镜面尺寸、吸热器三维坐标、经纬度和全年空间 DNI 工况，寻找年度价值 "
          "V(L) = Σ h_s (P_s − λ_L·U_s) 最高的定日镜布置：每个参与排名的布局在每个工况下都用自己的 APS 方案计分。"
          "同时给出任意布局都不可能超过的上界，回答“最优布置是否存在、当前方案离全局最优多远”。",
          ([case.name] if case else ["算例无效"]) + _demo_chip())
    if case is None:
        st.warning("请先在「参数设置」中修正当前算例。")
        return
    mode = st.radio("我要做什么", ["compare", "search"], horizontal=True, key="layout_mode",
                    format_func={"compare": "比较两套已有布局（A / B）", "search": "自动搜索更好的布局"}.get)
    if mode == "compare":
        _compare_workspace(case)
        return
    if st.session_state.get("own_data") == "layout":
        _templates(["layouts", "scenarios", "maps"], "layout_search")
    if st.session_state.get("res_opt") is not None:
        right, left = st.tabs(["计算结果", "调整参数"])
    else:
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
            _weather_panel(case, "layout")
        with _card("layout_goal"):
            _section(2, "用工程语言设定目标", "一个偏好同时控制运行（怎样瞄准）和评价（怎样排名）。")
            _preference_controls(case, "opt")
            _fields("layout_opt", ["uniformity_metric"], n_cols=1)
            st.caption("上面的初筛指标只用于中心瞄准的快速初筛（周向 = 布局决定、瞄准策略无法消除的部分）；"
                       "正式排名的不均匀量与问题一相同，是考核区二维吸收热流的 A_H·σ_H。")
        scen_blockers = []
        with _card("layout_scen"):
            _section(3, "看看日照落在哪些镜子上",
                     "工况 = 太阳方向 + 代表小时数 + 场地 DNI 地图。" + DNI_DEFINITION + "。镜子在哪里就读哪里的 DNI。")
            scen_blockers = _scenario_source_inputs(case, "opt")
            c0, _ = _case()
            if c0 is not None and not scen_blockers:
                try:
                    from cspaps.layout_value import baseline_layout

                    b0 = baseline_layout(c0, c0.layout_opt.n_heliostats or build_field(c0).n)
                    _scenario_browser(c0, {"初始镜场": (b0[0], b0[1])} if b0 else {}, "opt")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"工况预览不可用：{exc}")
            with st.expander("工况与地图的高级设置"):
                _fields("scenarios", ["map_include_uniform", "map_assignment", "map_sequence", "map_cell_m",
                                      "dni_bands"], n_cols=2)
                _fields("annual", ["representative_dni"], n_cols=1)
                _fields("layout_opt", ["formal_swap_rounds", "formal_position_trials", "ranking_refine_states", "dense_check_scenarios", "inner_joint_trials",
                                       "inner_certify_rounds", "inner_certify_iters", "cross_check_maps"], n_cols=2)
        with _card("layout_search"):
            _section(4, "搜索设置", "初筛按中心瞄准的全年理论热量减周向不均匀惩罚评价参数化布局；入围布局与初始布局"
                     "再在同一组工况上逐个求解 APS，按 V(L) 正式排名。")
            st.radio("计算强度", list(PRESETS), format_func=lambda k: PRESETS[k][0], key="layout_preset",
                     index=1, horizontal=False, on_change=_apply_preset)
            _fields("layout_opt", ["generator", "search_aiming", "refine_top", "upper_bound"])
            with st.expander("搜索范围、采样精度与计算资源"):
                common = {"n_heliostats", "generator", "search_aiming", "refine_top", "upper_bound", "uniformity_weight",
                          "uniformity_metric", "ranking", "baseline", "aps_weight", "formal_swap_rounds",
                          "dense_check_scenarios", "formal_position_trials", "ranking_refine_states", "inner_joint_trials", "inner_certify_rounds", "inner_certify_iters", "cross_check_maps"}
                _fields("layout_opt", [f.name for f in dataclasses.fields(_SECTIONS["layout_opt"]) if f.name not in common])
            with st.expander("APS 复核使用的瞄准设置"):
                _fields("aps", ["method", "n_height_levels", "azimuth_offsets_deg"])
                st.caption("均匀性权重由第 2 步的偏好滑块决定（运行与评价同一个 λ）；安全裕度也在第 2 步设置。")
        case, err = _case()
        if err:
            st.error(err)
            return
        with _card("layout_run"):
            _section(5, "确认要算什么", "开始前核对规模与口径；计算中可以取消，已完成的工况会保留，未完成的布局标为排名待定。")
            try:
                tbl = _scenario_table(case.to_yaml(), st.session_state["base_dir"])
                n_s, hrs = len(tbl), float(tbl["actual_hours"].sum())
            except Exception:  # noqa: BLE001
                n_s, hrs = 0, 0.0
            lo = case.layout_opt
            _run_summary([("镜数", f"{lo.n_heliostats} 面"), ("正式排名工况", f"{n_s} 个"),
                          ("合计代表小时（实际）", f"{hrs:,.0f} h"),
                          ("数据来源", f"工况清单 {Path(case.scenarios.file).name}" if case.scenarios.source == "file"
                           else f"站址气象 + 地图模型 {case.scenarios.map_model}"),
                          ("考核区域", "整个吸热面" if not case.uniformity.panels and not case.uniformity.z_range else "部分区域"),
                          ("安全裕度", f"{case.aps.safety_margin:.0%}"),
                          ("参加排名", f"搜索前 {lo.refine_top} 名" + (" + 初始镜场" if lo.baseline == "initial" else "")),
                          ("偏好 λ", f"{lo.uniformity_weight:.2f}")])
            digest = _job_button("opt", "▶  开始布局优化", wf.task_optimize_layout, progress=True, cancellable=True,
                        blockers=scen_blockers)
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
            completed = load_case(Path(res["run_dir"]) / "inputs_case.yaml")
            if res.get("input_digest") != digest:
                _callout("warn", "参数已修改，需要重新计算",
                         f"下面的图表、指标和导出文件对应上一次的参数：镜数 {completed.layout_opt.n_heliostats}，"
                         f"λ = {completed.layout_opt.uniformity_weight:g}，正式工况 {completed.scenarios.n_scenarios} 个，"
                         f"地图模型 {completed.scenarios.map_model}。")
            f = res["result"].formal
            if f and f.get("values"):
                s_ = res["summary"]
                _section(None, "正式排名 · 逐工况 APS 年度价值",
                         f"{s_['n_heliostats']} 面镜 · {len(f['scenario_set'])} 个工况 · λ_APS = {f['settings'].lam_aps:g} · "
                         f"λ_L = {completed.layout_opt.uniformity_weight:g} · 用时 {s_['runtime_s']:.0f} s · "
                         f"场地校验 {'通过' if s_['status'] == 'ok' else s_['status']}")
                if f.get("validation_set") and not f["validation_set"]["new_states"]:
                    st.info("复核表与设计表使用相同工况。请导入另一张复核工况表以检查排序稳定性。")
                _render_values(completed, f["values"], f["ranking"], f["scenario_set"], "opt", bound=res["result"].bound,
                               dense=f.get("dense"), run_dir=res["run_dir"], swaps=f.get("swaps"),
                               cross=f.get("cross_maps"))
                with st.expander("初筛（中心瞄准）、搜索过程、场地潜力与周向受热"):
                    _layout_results(completed, res)
            else:
                _layout_results(completed, res)
    if res is not None:
        _planning_compare(completed)


def _layout_results(case, res: dict) -> None:
    s = res["summary"]
    r = res["result"]
    ub = s.get("upper_bound") or {}
    val = s["validation"]
    site_status = {"ok": "通过", "capacity_shortfall": "镜数不足", "constraint_violation": "未通过"}.get(s["status"], s["status"])
    gap = ub.get("gap_objective_conservative")
    uni = s.get("uniformity") or {}
    method = ub.get("objective_method")
    sampling = "逐时" if method == "hourly" else "代表工况"
    basis = f"{sampling} · {ub.get('objective_states', s['review_sampling']['n_states'])} 个工况"
    first = r.ranking.iloc[0]
    objective = first.get("objective_review_mwh", first.get("energy_search_objective_mwh"))
    energy = first.get("energy_hourly_center_mwh", first.get("energy_review_center_mwh", s.get("annual_energy_mwh_center_aiming")))
    _section(None, "优化结果", f"{s['n_heliostats']} 面镜 · {s['n_evaluations']} 次参数评价 · 用时 {s['runtime_s']:.0f} s")
    _kpis([
        ("中心瞄准年吸热量", _format_number(energy, 0), "MWhₜₕ",
         "逐时气象复算" if "energy_hourly_center_mwh" in first else "代表工况估算"),
        ("加权布局目标 J", _format_number(objective, 0), "MWhₜₕ", basis + (" · 二维" if case.layout_opt.uniformity_metric == "surface" else " · 周向")),
        ("J 的保守差距", f"≤ {_pct(gap, 1)}" if gap is not None else "—", "",
         f"同工况上界 {_format_number(ub.get('objective_upper_bound_mwh'), 0)} MWhₜₕ"),
        ("受限 APS 年吸热量", _format_number(s.get("annual_energy_mwh_with_aps"), 0),
         "MWhₜₕ", f"独立运行复核 · {s['review_sampling']['n_states']} 个工况"),
        ("最小镜距", f"{val.get('min_distance_m')}", "m", f"要求 ≥ {val.get('required_min_spacing_m')} m · 场地校验{site_status}",
         "good" if s["status"] == "ok" else "bad"),
    ] + ([
        ("管屏间吸热 CV", _format_number(uni.get("cv_circ_weighted"), 3), "", f"布局权重 λ_L = {uni.get('weight', 0):g} · 逐工况能量加权",
         "good" if (uni.get("cv_circ_weighted") or 9) < 0.25 else "warn"),
        ("镜场质心偏移", f"{uni.get('centroid_offset_m', 0):.0f}", "m",
         f"北侧镜子 {100 * uni.get('north_fraction', 0):.0f}% · 脱离主镜场 {uni.get('n_detached', 0)} 面 · "
         "0 m 表示集热器位于镜场中心",
         "good" if (uni.get("centroid_offset_rel") or 9) < 0.25 else "warn"),
    ] if uni else []))
    if uni and uni.get("n_detached"):
        _callout("warn", f"{uni['n_detached']} 面定日镜脱离主镜场",
                 "这些镜子与主镜场之间的距离超过中位镜距的 3 倍：" +
                 "；".join(f"({x:.0f}, {y:.0f}) m" for x, y in uni.get("detached_xy_m", [])[:6]))
    if uni and uni.get("weight", 0) <= 0 and uni.get("north_fraction", 0) > 0.75:
        _callout("warn", "镜场集中在塔的一侧", f"{100 * uni['north_fraction']:.0f}% 的定日镜在塔北侧，吸热器南侧几乎不受热，"
                 f"管屏间吸热 CV {uni['cv_circ_weighted']:.2f}。这部分不均匀无法靠瞄准策略消除——在 ② 中提高布局均匀度权重 λ_L，"
                 "让镜场环绕吸热器。")
    exist = s.get("existence") or {}
    if ub.get("status") == "bound_violation":
        _callout("bad", "上界校验失败：本方案的能量超过了计算出的上界",
                 "这说明模型之间存在不一致，本次的差距不能作为证明。请保存运行记录并反馈。")
    elif exist.get("optimum_exists") is True and ub.get("upper_bound_mwh"):
        _callout("info", "场地与镜距校验通过",
                 f"当前布置已满足 {s['n_heliostats']} 面镜的几何约束。候选按同一能量与均匀性目标复排。"
                 f"布局权重 λ_L = {uni.get('weight', 0):g}，{basis}。")
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
    tabs = st.tabs(["单镜潜力图与布局", "吸热器周向受热", "能量–均匀度权衡", "三维视图", "搜索收敛", "上界说明"])
    with tabs[0]:
        if r.potential is not None:
            _chart(potential_map_chart(case, r.potential, r.best.xyz, eff, "年均光学效率",
                                       title="底图：单镜全年吸收潜力（无邻镜干扰）· 点：优化镜位"), "pot_map")
        else:
            _chart(field_map(case, r.best.xyz, np.arange(1, r.best.n + 1), values=eff, label="年均光学效率",
                             title="优化镜位"), "opt_map")
    with tabs[1]:
        if uni.get("sector_share"):
            a, b = st.columns([1.1, 1])
            with a:
                tr = st.session_state.get("res_ltrade")
                compare = None
                if tr and tr["summary"]["curve"] and tr["summary"]["curve"][0]["lambda"] == 0:
                    compare = ("仅能量最大（λ_L = 0）", tr["summary"]["curve"][0].get("sector_share"))
                _chart(receiver_rose(uni["sector_share"], uni["sector_center_deg"],
                                     compare if compare and compare[1] else None), "opt_rose")
            with b:
                st.markdown("**吸热器各朝向分得的全年吸收热量**（30° 扇区，按方位：北 0°、东 90°）")
                st.markdown(f"- 管屏间吸热 CV（逐工况、能量加权）：**{uni['cv_circ_weighted']:.3f}**；全年累计：{_format_number(uni.get('cv_circ_annual'))}\n"
                            f"- 镜场能量质心距塔 **{uni['centroid_offset_m']:.0f} m**（相对平均镜距 {100 * uni['centroid_offset_rel']:.0f}%）\n"
                            f"- 北侧镜子占 {100 * uni['north_fraction']:.0f}%")
                st.caption("圆柱外置式吸热器的每一侧只被朝向它的定日镜照射。瞄准策略可以沿高度、在小角度内调整热流，"
                           "但不能把北侧的热流搬到南侧，所以周向均匀性必须在布局阶段解决。")
    with tabs[2]:
        _section(None, "布局的能量–均匀度权衡",
                 "固定本次最优生成参数，只改变 λ_L 重新选择镜位（每点数秒）。用于决定愿意用多少年发电量换取周向均匀受热。")
        c1, c2 = st.columns([2, 1])
        lam_txt = c1.text_input("λ_L 序列", "0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0", key="ltrade_lams")
        try:
            lams = [float(v) for v in lam_txt.replace("，", ",").split(",") if v.strip()]
        except ValueError:
            lams = None
            st.error("λ_L 序列格式错误")
        # This tab explores the generator before formal APS refinement.
        best_params = r.best.params if r.best.params and "source" not in r.best.params else None
        with c2:
            if lams:
                _job_button("ltrade", "计算布局权衡曲线", wf.task_layout_tradeoff, lambdas=lams, params=best_params,
                            completed_case=case, progress=True)
        _job_error("ltrade")
        tr = st.session_state.get("res_ltrade")
        if tr:
            rows = tr["summary"]["curve"]
            a, b = st.columns([1.5, 1])
            with a:
                _chart(layout_tradeoff_chart(rows, uni.get("weight")), "ltrade_chart")
            with b:
                df = pd.DataFrame(rows)
                st.dataframe(pd.DataFrame({"λ_L": df["lambda"], "加权目标 J (MWh)": df["objective_mwh"].round(0),
                                           "年吸收 (MWh)": df["energy_mwh"].round(0),
                                           "管屏间 CV": df["cv_circ_weighted"].round(3),
                                           "北侧占比": (100 * df["north_fraction"]).round(0).astype(int).astype(str) + "%",
                                           "质心偏移 (m)": df["centroid_offset_m"].round(0),
                                           "脱离主镜场": df["n_detached"] if "n_detached" in df else 0}), hide_index=True)
                if len(rows) > 1:
                    e0 = rows[0]["energy_mwh"]
                    best = min(rows, key=lambda q: q["cv_circ_weighted"])
                    st.caption(f"λ_L {rows[0]['lambda']:g} → {best['lambda']:g}：管屏间 CV {rows[0]['cv_circ_weighted']:.2f} → "
                               f"{best['cv_circ_weighted']:.2f}，全年能量 {100 * (best['energy_mwh'] / e0 - 1):+.1f}%。")
    with tabs[3]:
        _chart(plant_scene(case, r.best.xyz, values=eff, label="年均光学效率"), "opt_scene")
    with tabs[4]:
        _chart(history_chart(r.history), "opt_hist")
        st.caption("灰点为每次评价的布局参数，绿线为迄今最优。搜索目标 J = 全年理论吸收热量 − λ_L × 周向不均匀惩罚；"
                   "入围方案随后复核。")
    with tabs[5]:
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
            items = [("本方案理论年吸收", f"{ub.get('found_energy_mwh', 0):,.1f} MWh"),
                     ("本方案各镜单独潜力之和", f"{ub.get('layout_lone_sum_mwh', 0):,.1f} MWh"),
                     ("任意布局、任意瞄准策略的年能量上界", f"{ub.get('upper_bound_mwh', 0):,.1f} MWh"),
                     ("理论能量差距", _pct(ub.get("gap"))),
                     ("受限 APS 可行年吸收", f"{ub.get('found_aps_feasible_mwh', 0):,.1f} MWh" if ub.get("found_aps_feasible_mwh") else "—"),
                     ("APS 口径差距", _pct(ub.get("gap_aps_feasible")))]
            if ub.get("objective_review_mwh") is not None:
                items += [("当前瞄准策略的加权目标 J", f"{ub['objective_review_mwh']:,.1f} MWh"),
                          ("同工况下 J 的上界 (MWh)", _format_number(ub.get("objective_upper_bound_mwh"), 1)),
                          ("J 的保守差距", _pct(ub.get("gap_objective_conservative")))]
            st.dataframe(pd.DataFrame({"项目": [k for k, _ in items], "值": [v for _, v in items]}), hide_index=True)
            if ub.get("objective_review_mwh") is not None:
                st.caption("加权目标 J 没有比能量上界更紧的通用上界：任何布局的管屏间惩罚都 ≥ 0，所以只能用 J ≤ E ≤ 上界 作保守比较，"
                           "其中包含了为均匀受热主动放弃的能量。")
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
                             "aps_max_peak_kw_m2", "aps_hours_uniformity_failed", "aps_states_not_ok",
                             "circ_cv_review", "north_fraction", "objective_review_mwh"] if c in rank]
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
        "circ_cv_review": "管屏间 CV", "north_fraction": "北侧占比", "objective_review_mwh": "复核目标 J (MWhₜₕ)",
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
        formal_vals = (r.formal or {}).get("values") or {}
        fname = f"search_{search_rank}"
        if fname in formal_vals:
            ft = formal_vals[fname].totals()
            n_ok = ft["status_counts"].get("feasible", 0)
            n_all = sum(ft["status_counts"].values())
            st.success(f"已进入正式排名（{wf.layout_label(fname)}）：年度价值 V = {ft['value_mwh']:,.0f} MWhₜₕ，"
                       f"{n_ok}/{n_all} 个工况可行运行。详见上方“正式排名”。")
        elif formal_vals:
            st.info("该布局只参加了中心瞄准初筛，没有进入正式排名。")
        elif aps_table is None:
            st.warning("该布局仅完成搜索阶段评估，尚未进行 APS 年度复核。")
        elif feasible_states < aps_states:
            st.warning(f"APS 代表工况 {feasible_states}/{aps_states} 个满足全部通量和均匀度约束。")
            if feasible_states == 0:
                st.error("本次年度复核没有找到可行运行工况，不能将该候选标记为可用布局。")
        else:
            st.success(f"APS 代表工况 {feasible_states}/{aps_states} 个满足全部约束。")
        st.caption("镜场坐标采用 x 东、y 北、z 上（m），原点为塔底。载入只切换仿真镜场，不向设备下发指令。")
        selectable = selected_validation["status"] == "ok" and (aps_table is None or feasible_states > 0 or bool(formal_vals))
        if st.button("载入此布局继续仿真", key="apply_layout", disabled=not selectable):
            path = Path(res["run_dir"]) / f"applied_layout_{pick + 1}.csv"
            save_layout_csv(fld, path)
            updated = case.to_dict()
            updated["field"]["layout_file"] = str(path.resolve())
            st.session_state["case_dict"] = updated
            st.session_state["base_dir"] = str(case.base_dir)
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



# --------------------------------------------------------------------------- problem 2: annual value V(L)

VALUE_STATUS_ZH = {"feasible": "可行运行", "feasible_shutdown": "可行停机", "no_sun": "无日照",
                   "no_feasible_found": "未找到可行方案", "proven_infeasible": "已证不可行", "not_solved": "未求解"}
VALUE_STATUS_RULE = {"feasible": "计入 h·(P − λ_L·U)", "feasible_shutdown": "计 0（全部散焦也是可执行方案）",
                     "no_sun": "计 0", "no_feasible_found": "计 0（保守：吸热器保持停机，年度价值为下界）",
                     "proven_infeasible": "计 0（约束下无任何可运行方案）",
                     "not_solved": "不计分：该布局年度价值不完整，不参与排名"}


def _labels(values: dict) -> dict:
    return {n: wf.layout_label(n) for n in values}


def _sector_share(value, case, n_sectors: int = 12):
    """Annual absorbed energy per receiver azimuth sector from the layout's own APS plans (flux maps)."""
    from cspaps.layout_value import eval_grid_shape

    nz, na = eval_grid_shape(case)
    grid = build_grid(case.receiver, na, nz)
    acc = np.zeros(na)
    for r in value.records:
        if r.plan.counts and r.plan.flux is not None and r.plan.flux.shape == grid.shape:
            acc += r.hours * (r.plan.flux * grid.to_map(grid.alpha)).sum(axis=0)
    sector = np.minimum((grid.azimuth_deg // (360.0 / n_sectors)).astype(int), n_sectors - 1)
    tot = np.bincount(sector, weights=acc, minlength=n_sectors)
    share = tot / tot.sum() if tot.sum() > 0 else tot
    return list(share), list((np.arange(n_sectors) + 0.5) * 360.0 / n_sectors)


def _value_kpis(values: dict, ranking: pd.DataFrame, bound: dict | None, sset) -> None:
    first = ranking.iloc[0]
    win = values[first["name"]]
    tot = win.totals()
    lam = tot["lambda_layout"]
    items = [
        ("年度价值 V（第 1 名）", _format_number(tot["value_mwh"], 0), "MWhₜₕ", wf.layout_label(first["name"]), "good"),
        ("全年吸收热量 E", _format_number(tot["energy_mwh"], 0), "MWhₜₕ", "Σ h_s·P_s，各工况自己的 APS 方案"),
        ("均匀性扣分 λ_L·Π", _format_number(lam * tot["penalty_mwh"], 0), "MWhₜₕ",
         f"Π = Σ h_s·U_s = {tot['penalty_mwh']:,.0f} · λ_L = {lam:g}"),
        ("内层求解差距（年度加权）", f"≤ {_pct(tot['inner_gap_annual'], 2)}" if tot["inner_gap_annual"] is not None else "—", "",
         f"最大单工况 {_pct(tot['max_inner_gap'], 1)} · λ_APS = {tot['lambda_aps']:g}"),
    ]
    if bound and bound.get("gap_value_conservative") is not None:
        items.append(("外层上界差距（保守）", f"≤ {_pct(bound['gap_value_conservative'], 1)}", "",
                      f"任意布局能量上界 {bound['upper_bound_mwh']:,.0f} MWhₜₕ"))
    items.append(("工况集", f"{len(sset)}", "个", f"地图 {', '.join(sorted(sset.subset_maps()))} · 指纹 {sset.fingerprint()[:8]}", "navy"))
    st_ = win.settings
    items.append(("瞄准策略（每个工况）", "APS", "",
                  f"{'贪心' if st_.method == 'greedy' else '精确法'} + 联合姿态搜索 {st_.joint_trials} 次"
                  + (f" + 松弛取整 {st_.certify_rounds} 次" if st_.certify_rounds else "")
                  + f" · 候选 {len(st_.azimuth_offsets_deg)}×{st_.n_height_levels} · 网格 {st_.grid_shape[0]}×{st_.grid_shape[1]}"
                  + f" · 设置摘要 {st_.digest()[:8]}", "navy"))
    _kpis(items)


def _decision_callout(values: dict, ranking: pd.DataFrame) -> None:
    first = ranking.iloc[0]
    ref = first["reference"]
    lab = wf.layout_label
    status = first["comparison_status"]
    if status == "incomplete":
        _callout("warn", "年度比较尚未完成", "补齐未求解工况后可以判断两套布局的年度价值。")
        return
    if status == "certified_tie":
        _callout("ok", "给定候选点上已证并列最高", "在这组代表工况与候选瞄准点上，领先布局的年度最优值得分相同。")
        return
    if status == "single_layout":
        _callout("ok", "一套布局已完成年度评分", "比较优劣需要另一套符合场地约束的等镜数布局。")
        return
    if status in ("overlapping_bounds", "feasible_scores_only"):
        _callout("warn", "当前可行方案得分领先 · 优劣尚待确认",
                 f"{lab(first['name'])} 的可行得分为 {first['value_mwh']:,.2f} MWh 等效。"
                 + ("候选瞄准模型的年度区间重叠。" if status == "overlapping_bounds"
                    else "此次两层权重或运行状态无法给出可比较的最优值区间。"))
        return
    if first["name"] == ref and len(ranking) > 1:
        second = ranking.iloc[1]
        dv = -second["delta_value_mwh"]
        de, du = -second["delta_energy_mwh"], -second["delta_uniformity_mwh"]
        _callout("ok", f"{lab(first['name'])} 得分最高 · 已证领先受比较布局",
                 f"在当前工况和候选瞄准点上比第 2 名 {lab(second['name'])} 多 {dv:,.1f} MWh 等效：其中吸热量贡献 {de:+,.1f}，更均匀受热贡献 {du:+,.1f}。")
        return
    row = ranking.loc[ranking["name"] == ref].iloc[0] if (ranking["name"] == ref).any() else None
    if row is None:
        return
    dv, de, du = first["delta_value_mwh"], first["delta_energy_mwh"], first["delta_uniformity_mwh"]
    source = "更多吸热" if abs(de) >= abs(du) else "更均匀受热"
    _callout("ok", f"{lab(first['name'])} 得分最高，优势主要来自{source}",
             f"相对 {lab(ref)}：年度价值 {dv:+,.1f} MWhₜₕ = 吸热量 {de:+,.1f} + 均匀性 {du:+,.1f}（−λ_L·ΔΠ）。"
             "每一项都来自各布局在同一组工况上自己的 APS 方案，可在“逐工况明细”中逐项核对。")


def _ranking_table(ranking: pd.DataFrame) -> pd.DataFrame:
    def statuses(d):
        return " · ".join(f"{VALUE_STATUS_ZH.get(k, k)} {v}" for k, v in d.items())

    return pd.DataFrame({
        "排名": ranking["rank"].map(lambda v: "—" if pd.isna(v) else int(v)),
        "布局": ranking["name"].map(wf.layout_label), "镜数": ranking["n_heliostats"],
        "年度价值 V (MWhₜₕ)": ranking["value_mwh"].round(1), "吸收热量 E (MWhₜₕ)": ranking["energy_mwh"].round(1),
        "均匀性扣分 λ_L·Π (MWhₜₕ)": ranking["weighted_penalty_mwh"].round(1),
        "ΔV": ranking["delta_value_mwh"].round(1), "ΔE（吸热）": ranking["delta_energy_mwh"].round(1),
        "−λ_LΔΠ（均匀）": ranking["delta_uniformity_mwh"].round(1),
        "内层差距": ranking["inner_gap_annual"].map(lambda g: _pct(g, 2)),
        "工况状态": ranking["status_counts"].map(statuses),
        "周向扣分对照 Π_circ (MWhₜₕ)": ranking["circ_penalty_mwh"].round(1)})


NEXT_STEP = {
    "no_feasible_found": "本次未找到：可增加预算再试（「工况与地图的高级设置」里的联合姿态搜索次数、松弛取整次数）。",
    "proven_infeasible": "已证无可行方案：核对约束（最低负荷、通量上限、可用镜、是否允许散焦）。",
    "feasible_shutdown": "可行停机：这些小时吸热为零（单独列出）。若不能接受，降低 λ（更重视吸热量）或放宽约束。",
    "not_solved": "未求解：重新运行补齐这些工况后才能排名。",
}


def _hotspot_text(case, p) -> None:
    if p.status in ("no_sun", "not_solved") or p.info.get("hotspot_panel") is None:
        return
    allowed = 1 - case.aps.safety_margin
    frac = p.info.get("peak_to_limit") or 0.0
    tone = "ok" if frac <= allowed + 1e-9 else ("warn" if frac <= 1 + 1e-9 else "bad")
    msg = (f"最热点：第 {p.info['hotspot_panel']} 号管屏，方位 {p.info['hotspot_azimuth_deg']:.0f}°、高度 "
           f"{p.info['hotspot_z_m']:.2f} m，{p.peak_flux_kw_m2:.0f} kW/m²（设备上限的 {100 * frac:.0f}%，项目允许 {100 * allowed:.0f}%）。")
    if p.status == "feasible_shutdown" or p.p_abs_w <= 0:
        msg = ("吸收功率为零（全部散焦），CV 无法评价。" if not p.n_active
               else f"{p.n_active} 面镜保持瞄准，当前吸收功率为零，CV 无法评价。")
    _callout(tone, "热点", msg)


def _ab_block(values: dict, ranking: pd.DataFrame, key: str) -> None:
    """Read first: the compared layouts side by side -- annual heat, uniformity penalty, combined score."""
    lam = float(ranking["lambda_layout"].iloc[0])
    rows = []
    for _, r in ranking.iterrows():
        rows.append({"label": wf.layout_label(r["name"]), "energy_mwh": r["energy_mwh"],
                     "weighted_penalty_mwh": r["weighted_penalty_mwh"],
                     "value_mwh": r["value_mwh"] if r["complete"] else r["value_partial_mwh"], "rank": r["rank"],
                     "complete": r["complete"], "n_solved": r["n_solved"], "n_scenarios": r["n_scenarios"],
                     "interval": r["value_interval_mwh"], "overlap": r.get("interval_overlaps_leader")})
    shown = rows[:4]
    a, b = st.container(), st.container()
    with a:
        _chart(ab_compare_chart(shown), f"{key}_ab")
    with b:
        tbl = pd.DataFrame({
            "布局": [r["label"] for r in shown],
            "名次": [("排名待定" if not r["complete"] else int(r["rank"])) for r in shown],
            "年吸热量 E (MWh)": [round(r["energy_mwh"], 1) for r in shown],
            "均匀性扣分 λΠ (MWh 等效)": [round(r["weighted_penalty_mwh"], 1) for r in shown],
            "综合得分 V (MWh 等效)": [round(r["value_mwh"], 1) for r in shown],
            "可能范围（同模型上界）": [f"{r['interval'][0]:,.1f} – {r['interval'][1]:,.1f}" if r["interval"] else "—"
                                    for r in shown]})
        st.dataframe(tbl, hide_index=True)
        if len(shown) >= 2 and shown[0]["complete"] and shown[1]["complete"]:
            d_e = shown[0]["energy_mwh"] - shown[1]["energy_mwh"]
            d_u = -(shown[0]["weighted_penalty_mwh"] - shown[1]["weighted_penalty_mwh"])
            st.markdown(f"**{shown[0]['label']}** 比 **{shown[1]['label']}** 高 "
                        f"{shown[0]['value_mwh'] - shown[1]['value_mwh']:+,.1f} MWh 等效 = 吸热量 {d_e:+,.1f} MWh + "
                        f"受热均匀 {d_u:+,.1f} MWh 等效（λ = {lam:g}）。")
    pending = [r for r in rows if not r["complete"]]
    for r in pending:
        _callout("warn", f"{r['label']}：排名待定",
                 f"只完成了 {r['n_solved']}/{r['n_scenarios']} 个工况（已完成部分的得分 {r['value_mwh']:,.1f} MWh 等效）。"
                 "补算剩余工况后才参与排名。")
    over = [r for r in rows[1:] if r["overlap"]]
    if over:
        _callout("info", "区间与第一名重叠",
                 "、".join(r["label"] for r in over) + " 的同模型上界高于第一名的得分：增加内层计算预算后排名可能改变，"
                 "这些布局应优先增加预算。")
    with st.expander("得分、单位与 CV 是什么意思"):
        st.markdown("- **年吸热量 E**（MWh）= Σ 代表小时 × 吸收功率。\n"
                    "- **均匀性扣分 λΠ**（MWh 等效）= λ × Σ 代表小时 × 受热不均匀量 U，U = 考核面积 × 热流标准差（MW）。\n"
                    "- **综合得分 V = E − λΠ**（MWh 等效），用于排名。\n"
                    "- **CV**（变异系数）= 表面热流的标准差 ÷ 平均值，数值越小，受热越均匀；全场停机时吸收功率为零，CV 无法评价。")
    status_rows = []
    for n, v in values.items():
        tot = v.totals()
        for stt in ("feasible_shutdown", "no_feasible_found", "proven_infeasible", "not_solved"):
            if tot["status_counts"].get(stt):
                status_rows.append({"布局": wf.layout_label(n), "状态": VALUE_STATUS_ZH[stt],
                                    "工况数": tot["status_counts"][stt], "代表小时": tot["status_hours"].get(stt),
                                    "计分": VALUE_STATUS_RULE[stt], "下一步": NEXT_STEP[stt]})
    if status_rows:
        st.markdown("**需要注意的工况**")
        st.dataframe(pd.DataFrame(status_rows), hide_index=True)


STATUS_MEANING = {
    "feasible": "找到通过细网格复核（不占用安全裕度）的运行方案。",
    "feasible_shutdown": "找到了方案，但最优或唯一可行的做法是全部散焦（吸热为零）——这是可执行的决定，不是计算失败。",
    "no_feasible_found": "本次计算预算内没找到可行运行方案，但没有证明不存在；吸热器按停机计 0，是保守的下界。",
    "proven_infeasible": "已经证明在当前约束下不存在任何可运行方案（例如不允许散焦而热流上限过低）。",
    "not_solved": "计算被取消或尚未完成，该工况没有结果；布局的年度价值不完整。",
}


def _status_legend() -> None:
    """How to read the scenario statuses and what to do next (shown whether or not they occur)."""
    with st.expander("工况状态怎么读：本次未找到 / 已证无可行 / 可行停机"):
        st.dataframe(pd.DataFrame([{"状态": VALUE_STATUS_ZH[k], "含义": STATUS_MEANING[k], "计分": VALUE_STATUS_RULE[k],
                                    "下一步": NEXT_STEP.get(k, "—")} for k in STATUS_MEANING]),
                     hide_index=True, width="stretch")


def _calc_basis(case, values: dict, ranking: pd.DataFrame, sset) -> None:
    """Parameters, data sources and the solve scope of this result, in one place."""
    v0 = next(iter(values.values()))
    stg = v0.settings
    rec = sset.record()
    lam_l = float(ranking["lambda_layout"].iloc[0])
    offs = ", ".join(f"{a:g}°" for a in stg.azimuth_offsets_deg) or "0°"
    region = ("整个吸热面" if not stg.region_panels and not stg.region_z else
              f"管屏 {list(stg.region_panels) if stg.region_panels else '全部'}，高度 {list(stg.region_z) if stg.region_z else '全高'}")
    method = CHOICE_LABELS["aps.method"].get(stg.method, stg.method)
    with st.expander("本次计算的口径：参数、数据来源和求解范围"):
        st.dataframe(pd.DataFrame([
            {"项目": "偏好 λ", "本次取值": f"评价 λ_L = {lam_l:g}；运行 λ_APS = {stg.lam_aps:g}"},
            {"项目": "安全裕度", "本次取值": f"{stg.safety_margin:.0%}（细网格复核的峰值不超过上限的 {1 - stg.safety_margin:.0%} 才算可行）"},
            {"项目": "考核区域", "本次取值": region},
            {"项目": "工况集", "本次取值": f"{rec['n_scenarios']} 个工况，合计 {rec['actual_hours']:,.0f} 实际小时，"
                                       f"DNI 年能量 {rec['annual_dni_kwh_m2']:,.0f} kWh/m²"},
            {"项目": "数据来源", "本次取值": str(rec.get("source", "—"))},
            {"项目": "DNI 地图", "本次取值": "、".join(rec["maps"])},
            {"项目": "代表小时口径", "本次取值": str(rec.get("hours_basis", "工况清单中的小时数"))},
            {"项目": "求解方法", "本次取值": f"{method} + 联合姿态复核 {stg.joint_trials} 次"
                                        + (f" + 松弛取整 {stg.certify_rounds} 轮" if stg.certify_rounds else "")},
            {"项目": "候选瞄准点（求解范围）", "本次取值": f"周向偏移 {offs} × {stg.n_height_levels} 个高度层（覆盖吸热器高度的 "
                                                  f"{stg.height_span:.0%}）" + ("，允许散焦" if stg.allow_defocus else "，不允许散焦")},
            {"项目": "评分网格", "本次取值": f"{stg.grid_shape[1]} 方位 × {stg.grid_shape[0]} 高度"},
            {"项目": "工况集指纹 / 设置指纹", "本次取值": f"{rec['fingerprint']} / {stg.digest()}（相同才可比较）"},
        ]), hide_index=True, width="stretch")
        st.caption("“候选瞄准点”就是求解范围：最优性差距与上界只对这组候选点成立。比较布局时必须用同一工况集指纹和设置指纹。")


def _render_values(case, values: dict, ranking: pd.DataFrame, sset, key: str, bound: dict | None = None,
                   dense: dict | None = None, disp: dict | None = None, run_dir=None, swaps=None,
                   cross: dict | None = None) -> None:
    """Formal comparison of layouts by V(L): ranking, decomposition, per-scenario plans, maps, audit."""
    from cspaps.layout_value import compare_values, eval_grid_shape

    key = f"fv_{key}"  # widget keys of this view never collide with the pre-screen view
    labels = _labels(values)
    _decision_callout(values, ranking)
    _ab_block(values, ranking, key)
    _value_kpis(values, ranking, bound, sset)
    _calc_basis(case, values, ranking, sset)
    _status_legend()
    tabs = st.tabs(["排名与优势分解", "逐工况明细", "工况与 DNI 地图", "与初始布局对照", "最优性与误差", "导出与复算"])
    with tabs[0]:
        a, b = st.columns([1.2, 1])
        with a:
            _chart(value_decomposition_chart(ranking, labels), f"{key}_decomp")
        with b:
            lam_now = float(ranking["lambda_layout"].iloc[0])
            lam = st.slider("外层权重 λ_L 重新计分（不重新求解瞄准方案）", 0.0, max(2.0, 2 * lam_now), lam_now, 0.05,
                            key=f"{key}_lam")
            totals = {n: v.totals() for n, v in values.items()}
            _chart(lambda_sweep_chart(totals, labels, lam, max(1.0, 2 * lam_now)), f"{key}_sweep")
        shown = compare_values(list(values.values()), lam, reference=ranking["reference"].iloc[0]) if lam != lam_now else ranking
        st.dataframe(_ranking_table(shown), hide_index=True)
        st.caption("V(L) = Σ_s h_s (P_s − λ_L·U_s)。内层用 λ_APS 选瞄准方案，外层只按 λ_L 对这组方案的 P_s、U_s 重新计分，"
                   "每个工况的惩罚只扣一次；改变 λ_L 时各直线斜率为 −Π，交点处排名互换。周向扣分仅作对照，不参与排名。")
        if swaps:
            st.dataframe(pd.DataFrame([{"轮次": w["round"], "布局": wf.layout_label(w["layout"]), "换位镜数": w["swapped"],
                                        "ΔV (MWhₜₕ)": round(w["gain_value_mwh"], 2), "ΔE (MWhₜₕ)": round(w["gain_energy_mwh"], 2),
                                        "接受": "是" if w["accepted"] else "否"} for w in swaps]), hide_index=True)
            st.caption("换位：移出 APS 年吸热最少的镜子、换入定价最高的空闲候选镜位；只有在同一工况集上 V 提高才接受。")
    names = list(values)
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        na = c1.selectbox("布局 A", names, index=0, format_func=labels.get, key=f"{key}_la")
        others = [n for n in names if n != na] or names
        default_b = others.index("initial") if "initial" in others else 0
        nb = c2.selectbox("布局 B", others, index=default_b, format_func=labels.get, key=f"{key}_lb")
        lam = float(ranking["lambda_layout"].iloc[0])
        ta, tb = values[na].table(lam), values[nb].table(lam)
        # sort the scenarios by their contribution to the score difference A - B (largest effect first)
        diff = ta["contribution_mwh"] - tb["contribution_mwh"]
        order = np.argsort(-np.abs(diff.to_numpy()), kind="stable")
        ta, tb, diff = ta.iloc[order].reset_index(drop=True), tb.iloc[order].reset_index(drop=True), diff.iloc[order]
        _chart(scenario_contrib_chart({na: ta, nb: tb}, labels, lam), f"{key}_contrib")
        view = pd.DataFrame({
            "对分差的贡献 A − B (MWh 等效)": diff.round(2).to_numpy(),
            "工况": ta["scenario_id"], "DNI 地图": ta["map_id"], "代表小时 h": ta["hours"].round(2),
            "太阳 高度/方位 (°)": [f"{e:.1f} / {z:.0f}" for e, z in zip(ta["elevation_deg"], ta["azimuth_deg"])],
            "工况 DNI": ta["dni_w_m2"].round(0),
            "A 状态": ta["status"].map(VALUE_STATUS_ZH), "A: P (MW)": ta["p_abs_mw"].round(3), "A: U (MW)": ta["u_mw"].round(3),
            "A: 贡献 (MWhₜₕ)": ta["contribution_mwh"].round(1),
            "B 状态": tb["status"].map(VALUE_STATUS_ZH), "B: P (MW)": tb["p_abs_mw"].round(3), "B: U (MW)": tb["u_mw"].round(3),
            "B: 贡献 (MWhₜₕ)": tb["contribution_mwh"].round(1)})
        ev = st.dataframe(view, hide_index=True, height=320, on_select="rerun", selection_mode="single-row",
                          key=f"{key}_rows")
        st.caption(f"A = {labels[na]}，B = {labels[nb]}。按对两方案分差的贡献从大到小排序；贡献 = h·(P − λ_L·U)，"
                   "只对“可行运行”工况计入。点一行在下方用同一色标对照两套镜位、逐镜日照和接收器热流。")
        rows = getattr(getattr(ev, "selection", None), "rows", None) or [0]
        sid = str(ta["scenario_id"].iloc[int(rows[0])])
        st.markdown(f"**工况 {sid}** · 对分差的贡献 {float(diff.iloc[int(rows[0])]):+,.2f} MWh 等效")
        i = sset.index(sid)
        nz, nax = eval_grid_shape(case)
        grid = build_grid(case.receiver, nax, nz)
        m = sset.samples.map_of(i)
        pa, pb = values[na].plan(sid), values[nb].plan(sid)
        zmax = max(float(np.max(p.flux)) if p.flux is not None else 0.0 for p in (pa, pb)) / 1e3 or None
        dmax = max(float(np.max(pa.dni)), float(np.max(pb.dni))) if pa.dni.size else None
        l, r = st.columns(2)
        for col, name, p in ((l, na, pa), (r, nb, pb)):
            with col:
                st.markdown(f"**{labels[name]}** · {VALUE_STATUS_ZH.get(p.status, p.status)}")
                _kpis([("P", f"{p.p_abs_w / 1e6:.3f}", "MW"), ("U", f"{p.u_w / 1e6:.3f}", "MW", f"CV {_format_number(p.cv_2d, 3)}"),
                       ("J = P − λ_APS·U", f"{p.objective_w / 1e6:.3f}", "MW", f"差距 ≤ {_pct(p.gap, 2)}"),
                       ("峰值", f"{p.peak_flux_kw_m2:.0f}", "kW/m²", f"散焦 {p.n_defocused} 面")])
                v = values[name]
                _chart(field_dni_map(case, v.xyz, v.ids, p.dni, dni_map=m,
                                     sun_angles=(sset.samples.elevation[i], sset.samples.azimuth[i]),
                                     map_scale=float(sset.samples.dni[i]) if m.unit == "factor" else 1.0,
                                     title=f"{sid} 镜场 DNI（底图为该工况的场地 DNI 地图）", cmin=0, cmax=dmax, height=400),
                       f"{key}_dni_{name}_{sid}")
                if p.flux is not None:
                    hs = ({"azimuth_deg": p.info["hotspot_azimuth_deg"], "z_m": p.info["hotspot_z_m"]}
                          if p.info.get("hotspot_azimuth_deg") is not None else None)
                    _chart(receiver_aims_map(case, grid, p.flux, p.aim_azimuth, p.aim_z, p.active, v.ids, p.dni,
                                             title="吸热器热流与瞄准点（该布局自己的 APS 方案）", zmax=zmax, height=380,
                                             hotspot=hs, cmin=0, cmax=dmax), f"{key}_rec_{name}_{sid}")
                    _hotspot_text(case, p)
    with tabs[2]:
        tbl = sset.table()
        a, b = st.columns([1, 1])
        with a:
            _chart(scenario_sky_chart(tbl), f"{key}_sky")
        with b:
            rec = sset.record()
            st.markdown(f"- 工况数 **{rec['n_scenarios']}**，代表小时合计 {rec['total_hours']:,.0f} h，DNI 能量 {rec['annual_dni_kwh_m2']:,.0f} kWh/m²\n"
                        f"- 来源：{rec.get('source', '—')}\n- 生成：{rec.get('generation', '—')}\n"
                        f"- 地图分配：{rec.get('map_assignment', '—')}；每张地图的工况数 {rec.get('states_per_regime', '—')}\n"
                        f"- 指纹 `{rec['fingerprint']}`：比较布局时工况集、地图、代表小时数、两项权重和求解精度都相同")
        maps = sset.subset_maps()
        cols = st.columns(min(3, max(len(maps), 1)))
        any_xyz = next(iter(values.values())).xyz
        for k, (mid, m) in enumerate(maps.items()):
            with cols[k % len(cols)]:
                _chart(dni_map_chart(case, m, title=f"{mid}", xyz=any_xyz), f"{key}_map_{mid}")
                st.caption(f"{m.description} · 单位 {m.unit} · {m.normalization}")
        st.dataframe(pd.DataFrame([{"地图": k, "单位": m.unit, "来源": m.source, "坐标系": m.crs, "插值": m.interpolation,
                                    "归一化": m.normalization, "栅格": f"{m.x.size}×{m.y.size} · {m.cell_m:g} m",
                                    "范围": f"{m.values.min():.3f} – {m.values.max():.3f}"} for k, m in maps.items()]),
                     hide_index=True)
        st.dataframe(tbl, hide_index=True, height=280)
    with tabs[3]:
        if "initial" not in values:
            st.info("本次比较没有初始布局。")
        else:
            other = [n for n in ranking["name"] if n != "initial"]
            pick = st.selectbox("对照的布局", other, format_func=labels.get, key=f"{key}_cmp") if other else None
            if pick:
                from cspaps.layout_value import displacement

                d = displacement(values["initial"].xyz, values[pick].xyz)
                ti, tp = values["initial"].totals(), values[pick].totals()
                _kpis([("年度价值 ΔV", f"{tp['value_mwh'] - ti['value_mwh']:+,.1f}", "MWhₜₕ", f"{labels[pick]} − 初始布局"),
                       ("吸热量 ΔE", f"{tp['energy_mwh'] - ti['energy_mwh']:+,.1f}", "MWhₜₕ"),
                       ("均匀性 −λ_LΔΠ", f"{-ti['lambda_layout'] * (tp['penalty_mwh'] - ti['penalty_mwh']):+,.1f}", "MWhₜₕ"),
                       ("移动的镜子", f"{d.get('n_moved', '—')}", "面",
                        f"平均移动 {d.get('mean_move_m', 0):.1f} m · 最大 {d.get('max_move_m', 0):.0f} m" if d.get("comparable") else "")])
                a, b = st.columns([1.3, 1])
                with a:
                    match = (d["match_from"], d["match_to"], d["distance_m"]) if d.get("comparable") else None
                    _chart(layout_overlay(case, values["initial"].xyz, values[pick].xyz, match,
                                          ("初始布局", labels[pick])), f"{key}_overlay")
                with b:
                    sh_p, cen = _sector_share(values[pick], case)
                    sh_i, _ = _sector_share(values["initial"], case)
                    _chart(receiver_rose(sh_p, cen, ("初始布局", sh_i)), f"{key}_rose")
                    st.caption("各朝向（30° 扇区）分得的全年吸收热量占比，来自各布局自己的逐工况 APS 方案热流。")
    with tabs[4]:
        tables = {n: v.table() for n, v in values.items()}
        _chart(inner_gap_chart(tables, labels), f"{key}_gap")
        st.caption("内层差距 = 1 − 本工况方案的 J / 该工况的可证上界（联合姿态支撑平面或无遮挡能量上界）。"
                   "差距越小，P_s、U_s 越接近该布局在这个工况下的最优运行能力。")
        rows = []
        for n, v in values.items():
            tot = v.totals()
            for stt, cnt in tot["status_counts"].items():
                rows.append({"布局": labels[n], "状态": VALUE_STATUS_ZH.get(stt, stt), "工况数": cnt,
                             "代表小时": tot["status_hours"].get(stt), "计入规则": VALUE_STATUS_RULE.get(stt, "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if bound:
            st.markdown(f"**外层上界**：{bound.get('applies_to', '')}。任意 N 面镜布局、任意瞄准点的年吸收热量 ≤ "
                        f"{bound.get('upper_bound_mwh', 0):,.0f} MWhₜₕ（单镜潜力 + 圆盘堆积，按工况地图取每个栅格的 DNI 最大值）；"
                        f"由于 U ≥ 0，V ≤ E ≤ 上界，保守差距 {_pct(bound.get('gap_value_conservative'), 1)}。")
        if dense:
            d = pd.DataFrame(dense["rows"])
            st.markdown(f"**代表工况误差**：同一布局在 {dense['n_scenarios']} 个更密工况上的复核"
                        f"（排序{'一致' if dense['same_order'] else '不一致'}）。")
            st.dataframe(pd.DataFrame({"布局": d["name"].map(wf.layout_label), "V（代表工况）": d["value_rep_mwh"].round(1),
                                       "V（更密工况）": d["value_dense_mwh"].round(1), "V 误差 %": d["value_error_pct"].round(2),
                                       "E 误差 %": d["energy_error_pct"].round(2)}), hide_index=True)
        if cross:
            ok = cross["winner_still_ahead"]
            _callout("ok" if ok else "warn", "独立云图复核" + ("：胜出布局仍领先" if ok else "：胜出布局不再领先"),
                     f"换一组随机云图（种子 {cross['seed']}）重新求解：胜出布局领先初始布局 "
                     f"{cross['lead_formal_mwh']:+,.0f} → {cross['lead_independent_mwh']:+,.0f} MWhₜₕ。"
                     + ("" if ok else "说明优势来自躲开固定位置的云影（对云图过拟合），不是可靠的设计结论；"
                                      "请增加云图张数，或改用持久性的空间 DNI 地图。"))
        st.caption("小镜场的内层（联合姿态穷举）与外层（全部候选组合）都有穷举对照测试（tests/test_layout_value.py）。"
                   "中心瞄准的逐时复核只验证中心瞄准指标，不用于 V(L)。")
    with tabs[5]:
        if run_dir is not None:
            vdir = Path(run_dir) / "formal" / "values"
            if not vdir.exists():
                vdir = Path(run_dir) / "values"
            if vdir.exists():
                for csv in sorted(vdir.glob("*/scenario_values.csv")):
                    st.download_button(f"下载 {wf.layout_label(csv.parent.name.replace('_plus_', '+'))} 的逐工况记录",
                                       csv.read_bytes(), file_name=f"{csv.parent.name}_scenario_values.csv",
                                       key=f"{key}_dl_{csv.parent.name}")
                rk = vdir / "formal_ranking.csv"
                if rk.exists():
                    st.download_button("下载排名表", rk.read_bytes(), file_name=rk.name, key=f"{key}_dl_rank")
                st.caption(f"逐镜瞄准方案：{vdir}/<布局>/aims/<工况>.csv（编号、坐标、DNI、是否工作、瞄准方位与高度、吸收功率）。")
                if st.button("仅凭导出文件复算全部得分", key=f"{key}_recompute"):
                    lam = float(ranking["lambda_layout"].iloc[0])
                    rec = wf.task_rescore(vdir, lam)["summary"]["layouts"]
                    mem = {n.replace("+", "_plus_"): v.totals(lam)["value_mwh"] for n, v in values.items()}
                    st.session_state[f"{key}_recheck"] = [
                        {"布局": wf.layout_label(r["layout"].replace("_plus_", "+")), "复算 V": round(r["value_mwh"], 4),
                         "界面 V": round(mem.get(r["layout"], float("nan")), 4),
                         "一致": "✓" if abs(r["value_mwh"] - mem.get(r["layout"], np.inf)) < 1e-6 * max(1.0, abs(r["value_mwh"]))
                         else "✗"} for r in rec]
                if st.session_state.get(f"{key}_recheck"):
                    st.dataframe(pd.DataFrame(st.session_state[f"{key}_recheck"]), hide_index=True)
        st.latex(r"V(L)=\sum_s h_s\,\big(P_s-\lambda_L\,U_s\big),\qquad U_s=A_H\,\sigma_H(q_s)")
        st.caption("复算只需 scenario_values.csv 的 hours、p_abs_mw、u_mw、status 四列和 λ_L：对 status = feasible 的行求和。")


def res_case(res: dict):
    return load_case(Path(res["run_dir"]) / "inputs_case.yaml")


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
        ("新增散焦指令", f"{int(row.get('n_defocus_commands', 0))} 面"),
        ("全塔影且保持瞄准", f"{int(row.get('n_tracking_in_tower_shadow', 0))} 面"),
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
                        inactive=~plan["active"], sun_angles=frame.get("sun_angles"),
                        shadow_fraction=plan["tower_fraction"])
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
    t1, t2, t3, t5, t4 = st.tabs(["光学模型", "问题一 · 瞄准优化", "问题二 · 布局优化", "年度价值 · 空间 DNI", "适用范围"])
    with t5:
        _section(None, "工况与空间 DNI 地图")
        st.markdown("每个代表工况 $s$ 给出太阳方向、代表小时数 $h_s$ 和一张覆盖整个场地的 DNI 地图 $D_s(x,y)$。"
                    "地图定义在场地坐标里（x 东、y 北，原点为塔底），与布局无关：第 $i$ 面镜读 $D_s(x_i,y_i)$，"
                    "移动镜位就在同一张图上按新坐标重新取值。地图为规则栅格，双线性插值，栅格外取边缘值。")
        st.latex(r"D_s(x,y)=G_s\,f_{r(s)}(x,y),\qquad \frac{1}{|\Omega|}\int_\Omega f_r\,dA=1")
        st.markdown("$G_s$ 来自站址逐时 DNI；$f_r$ 是地图（倍数），平均为 1 时只在空间上重新分配辐照。"
                    "逐时数据按日轮换分配到各地图 $r$，在每张地图的小时内按 DNI 能量加权聚类出代表工况，"
                    "$G_s h_s$ 等于成员小时的 DNI 能量，因此")
        st.latex(r"\sum_s h_s\,\overline{D_s}^{\,\Omega}=\sum_s h_s G_s=\sum_t w_t G_t")
        _section(None, "内层：每个布局、每个工况各自求解问题一")
        st.latex(r"a_s(L)=\arg\max_{a}\;P_s(a)-\lambda_{APS}\,U_s(a),\qquad U_s=A_H\,\sigma_H(q_s)")
        st.markdown("$P_s$ 为该方案联合姿态下（阴影遮挡按各镜自己的姿态重算）的吸收功率，$U_s$ 为考核区二维吸收热流的面积加权"
                    "标准差乘面积，在与布局无关的固定网格上计算；全部散焦是可行的停机方案，所以最优解存在。")
        _section(None, "外层：同一组方案上的年度价值")
        st.latex(r"V(L)=\sum_s h_s\big(P_s-\lambda_L U_s\big)=E(L)-\lambda_L\,\Pi(L),\qquad "
                 r"E=\sum_s h_sP_s,\;\;\Pi=\sum_s h_sU_s")
        st.markdown(r"- λ_L = λ_APS 时 $V=\sum_s h_s J_s$；两者不同时内层权重决定运行策略，外层权重评价它实现的功率与均匀性，"
                    "每个工况的惩罚只扣一次。\n"
                    r"- 方案固定时 $V$ 对 λ_L 是直线，斜率 $-\Pi$：改变 λ_L 不需要重新求解。" "\n"
                    r"- 两个布局之差 $V_A-V_B=(E_A-E_B)-\lambda_L(\Pi_A-\Pi_B)$，前一项是“更多吸热”，后一项是“更均匀受热”。" "\n"
                    "- 先在每个工况考核热流不均匀，再累计年度价值，不同时间的热点不会在年平均图上相互抵消。")
        _section(None, "状态与计分")
        st.dataframe(pd.DataFrame({"状态": [VALUE_STATUS_ZH[k] for k in VALUE_STATUS_RULE],
                                   "计入年度价值": list(VALUE_STATUS_RULE.values())}), hide_index=True)
        _section(None, "最优性结论的范围")
        st.markdown("- **内层**：每个工况给出可证上界与差距（小镜场穷举联合姿态即证最优）。\n"
                    r"- **外层**：任意 N 面镜布局、任意瞄准点，$V\le E\le$ 单镜潜力 + 圆盘堆积上界；地图下每个栅格取 DNI 最大值，"
                    "上界只对所用工况集成立。\n"
                    "- **采样**：在更密工况上复核胜出布局与初始布局，报告代表工况的误差与排序是否一致。")
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
        st.write("离散瞄准点构成有限集合。找到满足通量、均匀度及最低负荷的方案，就能确认最优解存在。"
                 "穷举或能量上界可证明不可行，其余情况保留为可行性待定。")
        _section(None, "联合姿态与最优性")
        st.write("不同镜面姿态会改变相邻镜子的受光。固定响应模型用于生成方案，最终使用联合姿态重算功率、通量与目标。"
                 "小规模问题枚举全部组合，大规模问题采用局部搜索，并用无遮挡的逐镜功率上界量化差距。")
        _section(None, "求解流程")
        st.markdown("1. 光斑尺寸优先的贪心构造（满足通量约束）→ 坐标下降；\n"
                    "2. 均匀度：增量更新 $\\sum q, \\sum q^2$ 的坐标搜索（加权模式）或对 $\\mu$ 二分（上限模式）；\n"
                    "3. 对偶上界，同时得到松弛解；\n"
                    "4. 松弛引导取整（确定性 + 随机取整）→ 通量修复 → 局部搜索，保留更优方案；\n"
                    "5. 自适应细网格、完整光斑复核，超限热点加入约束重解；\n"
                    "6. 小规模问题可用 SCIP 求混合整数二阶锥规划的精确最优作为基准。")
    with t3:
        _section(None, "问题表述")
        st.markdown(r"给定场地 $\\Omega$、镜数 $N$、镜面尺寸、吸热器三维坐标和经纬度（决定全年太阳轨迹与 DNI）：")
        st.latex(r"\max_{X=(x_1,\dots,x_N)}\;E(X)=\sum_{t} w_t\sum_i P_i(X,t)\quad \text{s.t.}\;\;x_i\in\Omega,\;\;"
                 r"\|x_i-x_j\|\ge d\;\;(i\ne j)")
        st.markdown("$E$ 包含余弦、阴影、遮挡、衰减、截断；全年按 DNI 加权的代表太阳位置积分（能量守恒）。")
        _section(None, "为什么布局必须考虑均匀性")
        st.markdown(r"圆柱外置式吸热器的每一侧只被朝向它的定日镜照射：位于方位 $\phi_i$ 的定日镜，其光斑在像平面上落在"
                    r"宽 $2R$ 的吸热器轮廓内，表面方位 $\phi$ 对应横向坐标 $u=R\sin(\phi-\phi_i)$（$|\phi-\phi_i|<90°$）。"
                    "各周向列分得的功率比例为")
        st.latex(r"s_{ij}=\frac{\Phi\!\left(\tfrac{R\sin(\phi_{j+1}-\phi_i)}{\sigma_i}\right)-"
                 r"\Phi\!\left(\tfrac{R\sin(\phi_j-\phi_i)}{\sigma_i}\right)}{\Phi(R/\sigma_i)-\Phi(-R/\sigma_i)}")
        st.markdown("APS 只能沿高度、在几度范围内移动瞄准点，大量光斑相互重叠后可以抹平每块管屏（扇区）内部的起伏，"
                    "但不能把北侧的热流搬到南侧。按管屏分组的方差分解")
        st.latex(r"\sigma^2_{2D}=\underbrace{\sigma^2_{circ}}_{\text{管屏之间，由布局决定}}+"
                 r"\underbrace{\overline{\sigma^2_{管屏内}}}_{\text{APS 可调}}")
        st.markdown("说明周向部分是问题一的均匀度惩罚中任何瞄准策略都去不掉的部分。只按全年能量布置时，"
                    "高纬度场址的镜子会集中在塔北侧（余弦效率高），吸热器南侧几乎不受热。因此问题二的目标取")
        st.latex(r"\max_X\;J(X)=\sum_t w_t\Big[P_t(X)-\lambda_L\,A_H\,\sigma_{circ,t}(X)\Big]"
                 r"=\sum_t w_t\,P_t\,(1-\lambda_L\,\mathrm{CV}_{circ,t})")
        st.markdown(r"与问题一的加权目标同构，$\lambda_L$ 是布局阶段均匀性的重要性超参数；$\lambda_L=0$ 退化为只追求全年能量。"
                    r"固定生成参数时，从候选镜位中选 $N$ 个是 $\max J(x),\ \sum x=N$；$J$ 对松弛变量是凹函数，"
                    "用交换局部搜索求解，并用 Frank–Wolfe 上界（线性子问题为取前 $N$ 大）证明选择的差距。"
                    "候选镜位的功率按“加入当前布局后”的阴影遮挡计算（只有已选镜子遮挡），选择与定价交替迭代，"
                    "只接受使真实目标值提高的一步——避免把从不建造的候选镜位的遮挡算进来，选出孤立的远处镜位。")
        _section(None, "最优布置存在吗？")
        st.write("搜索返回的合规布置是可行性的直接证据，有限候选中必能取到最大值。"
                 "连续场地上的存在性还要求目标在非空紧集上上半连续。采样遮挡可能随镜位跳变。"
                 "当给定镜数超过圆盘堆积容量上界时，可以排除所有布置。")
        _section(None, "上界：任意布局都不可能超过的能量")
        st.latex(r"E(X)\le\sum_i e_0(x_i)=\int_0^\infty\#\{i:e_0(x_i)\ge t\}\,dt\le\int_0^\infty\min\big(N,\mathrm{cap}(L_t)\big)\,dt")
        st.markdown("- $e_0(x)$：单镜在无邻镜干扰时的全年吸收热量（余弦、反射、衰减、截断、按光斑在各管屏上分布加权的吸收率）；\n"
                    "- 镜距 $\\ge d$ 的点集，Voronoi 胞与半径 $d/\\sqrt3$ 圆盘之交互不重叠，面积 $\\ge\\sqrt3d^2/2$（Fejes Tóth），"
                    "得到 $\\mathrm{cap}(S)\\le \\mathrm{area}(S\\oplus B(d/\\sqrt3))/(\\sqrt3 d^2/2)$；\n"
                    "- 栅格上的严格做法：对每个格子求对格内**任意位置、任意瞄准点**都成立的上界——余弦按方向最大偏转、"
                    "衰减按最短斜距、截断按包含吸热器轮廓的矩形在最小光斑下的高斯质量、吸收率按光斑可达管屏的最大值；"
                    "圆盘膨胀半径再加 √2 个格宽。每一步都是不等式，不依赖采样密度。")
        _section(None, "搜索方法")
        st.markdown("径向交错 / 仿生螺旋参数化生成器铺满场地 → 每个参数向量用交换搜索选出使 J 最大的 N 个镜位（λ_L = 0 时即年收益最高的 N 个）→ "
                    "差分进化 + 局部模式搜索 → 入围方案用更细采样和受限 APS 复核排名 → 低收益镜位与未用候选换位 → "
                    "镜数、边界、禁排区、最小镜距最终校验 → 上界与差距。")
    with t4:
        st.markdown("- 证书是针对**模型**的：候选瞄准点集合、求解网格与复核热点、代表工况采样。加密候选点和工况后，结论随之收敛。\n"
                    "- HFLCAL 把光斑视为圆高斯，大入射角下偏保守约 1%；可用内置 MCRT 或导出 SolTrace 对标校验。\n"
                    "- 未建模吸热器内部传热；温度约束由逐管屏通量上限、高度分布系数和偏差系数代替。\n"
                    "- 布局上界忽略阴影遮挡并允许最密堆积，因此偏乐观；真实全局最优介于本方案与上界之间。\n"
                    "- 演示算例的气象为阿克塞地区 NASA POWER 2024 年逐时 DNI，镜场与吸热器尺寸为模拟设计输入，"
                    "客户提供真实数据后直接替换即可。")


# --------------------------------------------------------------------------- single-instant cloud comparison

CLOUD_SHORT = {"clear": "① 云前 · 方案 A", "cloud_keep": "③ 云来 · 沿用 A", "cloud_aps": "④ 云来 · 重新 APS",
               "clear_keep": "⑤ 云散 · 仍用 C", "cloud_safe": "④′ 云来 · 防云散 APS", "clear_safe": "⑤′ 云散 · 用 S"}


def _state_numbers(s: dict, ref: dict | None = None) -> None:
    def delta(key, digits=2, unit=""):
        if ref is None:
            return ""
        d = s[key] - ref[key]
        return f"较云前 {d:+.{digits}f}{unit}"

    tone = "good" if s["flux_status"] == "ok" else ("warn" if s["flux_status"] == "margin_reduced" else "bad")
    _kpis([
        ("吸收热功率 P", f"{s['p_abs_mw']:.3f}", "MW", delta("p_abs_mw", 3, " MW")),
        ("不均匀量 U = A_H·σ_H", f"{s['u_mw']:.3f}", "MW", f"二维 CV {_format_number(s['cv_2d'], 3)} · 周向 {_format_number(s['cv_circ'], 3)}"),
        ("目标 J = P − λU", f"{s['objective_mw']:.3f}", "MW", delta("objective_mw", 3, " MW")),
        ("峰值热流（细网格复核）", f"{s['peak_flux_kw_m2']:.0f}", "kW/m²", f"占设备上限 {100 * s['peak_to_limit']:.0f}%", tone),
        ("工作 / 散焦镜", f"{s['n_active']} / {s['n_defocused']}", "面", f"相对方案 A 改变瞄准 {s['n_changed_vs_a']} 面"),
    ])


def _cloud_compare_inputs(case) -> None:
    with _card("cloud_inputs"):
        _section(1, "设定时刻与随机云", "云影冻结在所选时刻：先求晴空下的瞄准方案 A，再让云落到镜场上，对比“沿用 A”与“重新 APS”。")
        default_time = pd.Timestamp(case.realtime.start)
        c1, c2, c3, c4 = st.columns(4)
        date = c1.date_input("日期", value=default_time.date(), key=f"cc_date:{case.name}")
        tm = c2.time_input("当地时间", value=default_time.time(), key=f"cc_time:{case.name}", step=300)
        n_cl = c3.number_input("云的数量", 1, 8, 2, key="cc_n")
        seed = c4.number_input("随机种子（换一组云）", 0, 9999, 3, key="cc_seed")
        c1, c2, c3 = st.columns(3)
        rad = c1.slider("云影半径范围 (m)", 20, 400, (60, 140), 10, key="cc_rad")
        tau = c2.slider("云影透过率范围", 0.0, 1.0, (0.10, 0.35), 0.05, key="cc_tau",
                        help="云影中心处 DNI 保留的比例，0 = 完全遮挡。")
        u = case.uniformity
        lam0 = float(u.weight) if (u.enabled and u.mode == "weighted") else 0.0
        lam = c3.slider("瞄准均匀度权重 λ", 0.0, 2.0, lam0, 0.05, key="cc_lam",
                        help="问题一的目标 P − λ·A_H·σ_H；λ = 0 只追求吸收功率。")
        _job_button("cloud", "▶  计算 云前 / 云来 / 重新 APS 对比", wf.task_cloud_compare, f"{date} {tm}",
                    n_clouds=int(n_cl), radius_m=tuple(float(v) for v in rad), transmittance=tuple(float(v) for v in tau),
                    seed=int(seed), lam_aps=float(lam), progress=True)
        _job_error("cloud")


def _cloud_results(res: dict) -> None:
    st_ = {s["key"]: s for s in res["states"]}
    for k, s in st_.items():
        s["short"] = CLOUD_SHORT[k]
    case, fld, grid = res["case"], res["field"], res["grid"]
    summ = res["summary"]
    a, b, c, d = st_["clear"], st_["cloud_keep"], st_["cloud_aps"], st_["clear_keep"]
    safe, safe_clear = st_["cloud_safe"], st_["clear_safe"]
    lam = summ["lambda_aps"]
    cmin, cmax = 0.0, float(np.max(a["dni"])) * 1.02
    zmax = max(float(np.max(s["flux"])) for s in res["states"]) / 1e3
    _section(None, "云影对比结果",
             f"{summ['time'][:16].replace('T', ' ')} · 太阳高度角 {summ['sun_elevation_deg']:.1f}° · 晴空 DNI {summ['dni_clear_w_m2']:.0f} W/m² · "
             f"λ = {lam:g} · {summ['n_heliostats']} 面定日镜")
    _kpis([
        ("① 云前吸收功率", f"{a['p_abs_mw']:.2f}", "MW", f"二维 CV {_format_number(a['cv_2d'], 3)}"),
        ("③ 云来 · 不改策略", f"{b['p_abs_mw']:.2f}", "MW",
         f"{100 * (b['p_abs_mw'] / max(a['p_abs_mw'], 1e-9) - 1):+.1f}% · CV {_format_number(b['cv_2d'], 3)}", "warn"),
        ("④ 云来 · 重新 APS", f"{c['p_abs_mw']:.2f}", "MW",
         f"目标 J {c['objective_mw'] - b['objective_mw']:+.3f} MW · CV {_format_number(c['cv_2d'], 3)}", "good"),
        ("被云遮挡", f"{summ['shaded_heliostats']}", "面", f"平均透过率 {100 * summ['mean_transmittance']:.0f}%", "navy"),
        ("⑤ 云突然散去 · 仍用 C", f"{d['peak_flux_kw_m2']:.0f}", "kW/m²", f"占上限 {100 * d['peak_to_limit']:.0f}%",
         "bad" if d["peak_to_limit"] > 1 else ("warn" if d["peak_to_limit"] > 0.95 else "good")),
    ])
    gain = c["objective_mw"] - b["objective_mw"]
    _callout("info", "怎么读这组结果",
             f"云遮住 {summ['shaded_heliostats']} 面镜后，沿用晴空方案的吸收功率降到 {b['p_abs_mw']:.2f} MW，受热也变得不均匀"
             f"（CV {_format_number(a['cv_2d'], 3)} → {_format_number(b['cv_2d'], 3)}）。按云下 DNI 重新求解后，目标 J 提高 {gain:.3f} MW，"
             f"CV 回到 {_format_number(c['cv_2d'], 3)}。如果云在执行方案 C 时突然散去，峰值为设备上限的 {100 * d['peak_to_limit']:.0f}%；"
             f"按晴空 DNI 校核通量的方案 S 在云散后峰值为上限的 {100 * safe_clear['peak_to_limit']:.0f}%。")
    tabs = st.tabs(["① 云来之前", "② 云来了 · DNI 变化", "③ 不改变策略", "④ 重新 APS", "⑤ 云散风险与防云散方案", "全部数值与下载"])
    with tabs[0]:
        _state_numbers(a)
        l, r = st.columns([1, 1.25])
        with l:
            _chart(field_dni_map(case, fld.xyz, fld.ids, a["dni"], sun_angles=(summ["sun_elevation_deg"], summ["sun_azimuth_deg"]),
                                 shadow_fraction=a["tower_fraction"], title="镜场 DNI（晴空，全场均匀）", cmin=cmin,
                                 cmax=cmax, height=430), "cc_dni_a")
        with r:
            _chart(receiver_aims_map(case, grid, a["flux"], a["aim_azimuth"], a["aim_z"], a["active"], fld.ids, a["dni"],
                                     title="吸热器展开 · 热流与每面镜中心的瞄准点（点色 = 该镜 DNI）", zmax=zmax, cmin=cmin, cmax=cmax, height=430),
                   "cc_rec_a")
        l, r = st.columns(2)
        with l:
            _chart(aim_scene_3d(case, fld.xyz, a["dni"], a["aim_azimuth"], a["aim_z"], a["active"],
                                sun_angles=(summ["sun_elevation_deg"], summ["sun_azimuth_deg"]), beam_power=a["mirror_power_kw"],
                                shadow_fraction=a["tower_fraction"],
                                title="三维示意：镜面中心 → 瞄准点（抽样光束）", cmin=cmin, cmax=cmax), "cc_3d_a")
        with r:
            _chart(receiver_aims_3d(case, grid, a["flux"], fld.xyz, a["aim_azimuth"], a["aim_z"], a["active"], a["dni"],
                                    fld.ids, beam_power=a["mirror_power_kw"],
                                    title="吸热器近景：热流与送达镜子的瞄准点", zmax=zmax), "cc_r3d_a")
    with tabs[1]:
        tau = res["transmittance"]
        _kpis([("塔影下的镜子", f"{a['n_tracking_in_tower_shadow']} 面保持瞄准", "全塔影", "不发散焦指令"),
               ("平均 DNI（云前 → 云来）", f"{np.mean(a['dni']):.0f} → {np.mean(b['dni']):.0f}", "W/m²"),
               ("最低 DNI", f"{np.min(b['dni']):.0f}", "W/m²", "云影中心"),
               ("被遮挡（透过率 < 95%）", f"{summ['shaded_heliostats']}", "面", f"占 {100 * summ['shaded_heliostats'] / fld.n:.0f}%"),
               ("云影", f"{len(res['clouds'])}", "块", " · ".join(f"r {cl.radius:.0f} m, τ {cl.transmittance:.2f}"
                                                                  for cl in res["clouds"][:3]))])
        l, r = st.columns(2)
        with l:
            _chart(field_dni_map(case, fld.xyz, fld.ids, a["dni"], sun_angles=(summ["sun_elevation_deg"], summ["sun_azimuth_deg"]),
                                 shadow_fraction=a["tower_fraction"], title="云来之前", cmin=cmin, cmax=cmax, height=430),
                   "cc_dni_a2")
        with r:
            _chart(field_dni_map(case, fld.xyz, fld.ids, b["dni"], sun_angles=(summ["sun_elevation_deg"], summ["sun_azimuth_deg"]),
                                 shadow_fraction=b["tower_fraction"], clouds=res["clouds"], title="云来之后（紫色为塔影，虚线为云影边界）",
                                 cmin=cmin, cmax=cmax, height=430), "cc_dni_b")
        st.caption(f"透过率分布：最小 {tau.min():.2f}，中位 {np.median(tau):.2f}；云影边缘按平滑过渡建模。")
    with tabs[2]:
        _state_numbers(b, a)
        l, r = st.columns([1.25, 1])
        with l:
            _chart(receiver_aims_map(case, grid, b["flux"], b["aim_azimuth"], b["aim_z"], b["active"], fld.ids, b["dni"],
                                     title="沿用方案 A：瞄准点不变（点色 = 云下 DNI）", zmax=zmax, cmin=cmin, cmax=cmax, height=420), "cc_rec_b")
        with r:
            _chart(flux_diff_map(case, grid, a["flux"], b["flux"], "热流变化（云来 · 沿用 A − 云前）"), "cc_diff_ab")
        st.caption("被遮挡镜子的光斑变暗，原本相互补齐的热流出现缺口：总吸热减少，受热不均匀度上升。")
    with tabs[3]:
        _state_numbers(c, a)
        l, r = st.columns([1.25, 1])
        with l:
            _chart(receiver_aims_map(case, grid, c["flux"], c["aim_azimuth"], c["aim_z"], c["active"], fld.ids, c["dni"],
                                     prev=(a["aim_azimuth"], a["aim_z"], a["active"]),
                                     title="重新 APS：新瞄准点（白线 = 从方案 A 的移动）", zmax=zmax, cmin=cmin, cmax=cmax, height=420), "cc_rec_c")
        with r:
            _chart(flux_diff_map(case, grid, b["flux"], c["flux"], "热流变化（重新 APS − 沿用 A）"), "cc_diff_bc")
        l, r = st.columns(2)
        with l:
            _chart(aim_scene_3d(case, fld.xyz, c["dni"], c["aim_azimuth"], c["aim_z"], c["active"],
                                sun_angles=(summ["sun_elevation_deg"], summ["sun_azimuth_deg"]), beam_power=c["mirror_power_kw"],
                                shadow_fraction=c["tower_fraction"],
                                title="三维示意：云下的新方案", cmin=cmin, cmax=cmax), "cc_3d_c")
        with r:
            _chart(receiver_aims_3d(case, grid, c["flux"], fld.xyz, c["aim_azimuth"], c["aim_z"], c["active"], c["dni"],
                                    fld.ids, beam_power=c["mirror_power_kw"], title="吸热器近景：送达镜子的瞄准点", zmax=zmax), "cc_r3d_c")
        st.caption(f"{c['n_changed_vs_a']} 面镜子改变了瞄准点，平均移动 {c['mean_aim_shift_m']:.2f} m（吸热器表面）。"
                   "方案 A 也参与了重新求解的比较，所以重新 APS 的目标值不会低于沿用 A。")
    with tabs[4]:
        l, r = st.columns(2)
        with l:
            st.markdown("**云突然散去，仍执行方案 C**")
            _state_numbers(d, a)
            _chart(receiver_aims_map(case, grid, d["flux"], d["aim_azimuth"], d["aim_z"], d["active"], fld.ids, d["dni"],
                                     title="晴空 DNI 下执行云下方案 C", zmax=zmax, cmin=cmin, cmax=cmax, height=380, hotspot=d["hotspot"]), "cc_rec_d")
        with r:
            st.markdown("**防云散方案 S：通量按晴空 DNI 校核，目标按云下 DNI**")
            _state_numbers(safe_clear, a)
            _chart(receiver_aims_map(case, grid, safe_clear["flux"], safe_clear["aim_azimuth"], safe_clear["aim_z"],
                                     safe_clear["active"], fld.ids, safe_clear["dni"], title="云散后执行方案 S",
                                     zmax=zmax, cmin=cmin, cmax=cmax, height=380, hotspot=safe_clear["hotspot"]), "cc_rec_s")
        st.caption(f"云下方案 S 的吸收功率 {safe['p_abs_mw']:.3f} MW（方案 C {c['p_abs_mw']:.3f} MW）：为云突然散去预留的代价。"
                   "在线引擎的“含 DNI 预测”策略按同样的原则约束通量。")
    with tabs[5]:
        states = res["states"]
        _chart(state_bars(states), "cc_bars")
        tbl = pd.DataFrame([{
            "状态": s["short"], "平均 DNI (W/m²)": round(float(np.mean(s["dni"])), 1), "吸收功率 P (MW)": round(s["p_abs_mw"], 4),
            "截获功率 (MW)": round(s["p_int_mw"], 4), "U = A_H·σ_H (MW)": round(s["u_mw"], 4), "J = P − λU (MW)": round(s["objective_mw"], 4),
            "二维 CV": _format_number(s["cv_2d"], 4), "周向 CV": _format_number(s["cv_circ"], 4),
            "峰值 (kW/m²)": round(s["peak_flux_kw_m2"], 1), "占上限": f"{100 * s['peak_to_limit']:.1f}%",
            "复核": STATE_ZH.get(s["flux_status"], s["flux_status"]) if s["flux_status"] != "over_limit" else "超限",
            "工作镜": s["n_active"], "散焦镜": s["n_defocused"], "相对 A 改变": s["n_changed_vs_a"]} for s in states])
        st.dataframe(tbl, hide_index=True)
        _show_files(res, images=[])


def _cloud_compare(case) -> None:
    if case is None:
        st.warning("请先在「参数设置」中修正当前算例。")
        return
    _cloud_compare_inputs(case)
    res = st.session_state.get("res_cloud")
    if res is not None:
        _cloud_results(res)


def page_replay(case) -> None:
    _hero("运行仿真", "云影与在线 APS",
          "单时刻对比：云来之前的镜场 DNI、各定日镜中心对应的瞄准点和吸热器热流；云来以后不改策略与重新 APS 的热流、总热量和"
          "均匀性。时序回放：在线引擎逐周期决策，对比中心瞄准、仅用当前 DNI、含 DNI 预测三种策略。", [case.name] if case else [])
    t1, t2 = st.tabs(["单时刻云影对比", "时序回放（在线引擎）"])
    with t1:
        _cloud_compare(case)
    with t2:
        _ops_replay(case)


# --------------------------------------------------------------------------- optics & other pages

def page_optics(case) -> None:
    _hero("模型校验", "光学校验", "蒙特卡洛光线追迹（MCRT）以随机光线独立检查解析光斑模型；功率差、热流图差和峰值统计误差"
          "用于评估模型偏差。也可导出 SolTrace 对标算例，或用暴力求交复核阴影遮挡筛选。")
    t1, t4, t5, t2, t3 = st.tabs(["解析模型与光线追迹", "压力工况复核", "求解器对比", "SolTrace 对标", "遮挡筛选复核"])
    with t4:
        _stress_suite(case)
    with t5:
        _solver_benchmark(case)
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


def _pct_num(v, digits: int = 2):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(100 * float(v), digits)


def _stress_suite(case) -> None:
    from cspaps.validation import SUITE, SUITE_WHY, SUITE_ZH

    st.write("用同一方案分别跑解析模型与光线追迹，检验四种最容易出错的工况。**物理误差**（解析 − 追迹，同一方案、同一网格）"
             "与**求解差距**（候选瞄准集上的上界与方案目标之差，在解析模型内部计算）分开列出：前者说明模型本身离光线追迹多远，"
             "后者说明算法离该模型下的最好方案多远。")
    with st.expander("四种工况分别检验什么", expanded=False):
        for k in SUITE:
            st.markdown(f"- **{SUITE_ZH[k]}**：{SUITE_WHY[k]}")
    c1, c2, c3 = st.columns(3)
    keys = c1.multiselect("工况", list(SUITE), default=list(SUITE), format_func=lambda k: SUITE_ZH[k], key="vs_keys")
    rays = c2.number_input("每面镜子光线数", value=1000, step=500, min_value=100, key="vs_rays")
    seeds = c3.number_input("随机种子数", value=3, step=1, min_value=2, key="vs_seeds")
    soltrace = st.checkbox("同时导出 SolTrace 对标算例（逐镜 DNI 写成反射率系数）", value=True, key="vs_soltrace")
    _job_button("vsuite", "运行压力工况复核", wf.task_validation_suite, keys=tuple(keys) or None, rays=int(rays),
                seeds=int(seeds), soltrace=soltrace, blockers=[] if keys else ["请至少选择一个工况"])
    _job_error("vsuite")
    if "res_vsuite" not in st.session_state:
        return
    r = st.session_state["res_vsuite"]
    states = r["summary"]["states"]
    worst = max(states, key=lambda x: abs(x["physical"]["power_on_receiver_mw"]["rel_diff"] or 0))
    ok = all(x["physical"]["mcrt_within_limit"] for x in states)
    st.success(f"{len(states)} 个工况中，解析与追迹的功率差最大 {_pct_num(worst['physical']['power_on_receiver_mw']['rel_diff'])}%"
               f"（{worst['label']}）。" + ("光线追迹在所有工况下都确认热流低于限值（均值 + 2 倍标准误）。" if ok else
                                         "注意：有工况的追迹峰值（均值 + 2 倍标准误）超过限值，见下表。"))
    rows = []
    for x in states:
        ph = x["physical"]
        rows.append({
            "工况": x["label"], "时刻": x["time"][:16].replace("T", " "), "太阳高度 (°)": round(x["elevation_deg"], 1),
            "DNI 范围 (W/m²)": f"{x['dni_w_m2']['min']:.0f}–{x['dni_w_m2']['max']:.0f}",
            "功率 解析 (MW)": round(ph["power_on_receiver_mw"]["analytic"], 3),
            "功率 追迹 (MW)": round(ph["power_on_receiver_mw"]["mcrt"], 3),
            "功率差 (%)": _pct_num(ph["power_on_receiver_mw"]["rel_diff"]),
            "追迹功率标准误 (%)": _pct_num(ph["power_on_receiver_mw"]["mcrt_std_error"] / max(ph["power_on_receiver_mw"]["mcrt"], 1e-12)),
            "峰值 解析 (kW/m²)": round(ph["peak_flux_kw_m2"]["analytic"], 1),
            "峰值 追迹 (kW/m²)": round(ph["peak_flux_kw_m2"]["mcrt_mean_map"], 1),
            "峰值差 (%)": _pct_num(ph["peak_flux_kw_m2"]["rel_diff"], 1),
            "峰值/限值 细网格复核": round(ph["peak_to_limit"]["analytic_verified_fine"], 3),
            "峰值/限值 追迹": f"{ph['peak_to_limit']['mcrt_grid']:.3f} ± {ph['peak_to_limit']['mcrt_std_error']:.3f}",
            "热点距离 (m)": round(ph["hotspot"]["distance_m"], 2),
            "网格": "×".join(str(v) for v in ph["grid"]), "种子": ", ".join(str(v) for v in ph["mcrt"]["seeds"]),
            "每镜光线": ph["mcrt"]["rays_per_heliostat"],
        })
    _section(None, "物理误差：解析模型 vs 光线追迹", "差值方向：解析 − 追迹；追迹为多个种子的平均图，峰值取平均图的最大格；"
             "解析图取格心值、追迹图为格内平均，峰值差含这部分离散差异。")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    srows = []
    for x in states:
        so = x["solver"]
        srows.append({"工况": x["label"], "λ_APS": so["lambda_aps"], "方案状态": so["status"],
                      "目标 P − λU (MW)": round(so["objective_mw"], 3),
                      "上界 (MW)": None if so["upper_bound_mw"] is None else round(so["upper_bound_mw"], 3),
                      "求解差距 (%)": _pct_num(so["gap"]), "上界方法": so["bound_method"],
                      "候选范围": "本算例 APS 设置的周向偏移 × 高度层（含散焦）",
                      "散焦镜数": so["n_defocused"], "求解时间 (s)": round(so["solve_time_s"], 2)})
    _section(None, "求解差距：方案目标 vs 候选集上界", "上界对候选集内任何方案都成立；“无遮挡能量上界”在低太阳高度时较松，"
             "差距大并不表示方案差，而是该上界未计入阴影遮挡。")
    st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch")
    pick = st.selectbox("查看工况的热流图", [x["key"] for x in states],
                        format_func=lambda k: next(x["label"] for x in states if x["key"] == k), key="vs_pick")
    maps = r["maps"][pick]
    grid = build_grid(case.receiver)
    zmax = max(maps["analytic_w_m2"].max(), maps["mcrt_w_m2"].max()) / 1e3
    a, b = st.columns(2)
    with a:
        _chart(receiver_map(case, grid, maps["analytic_w_m2"], "解析模型热流", zmax=zmax), f"vs_a_{pick}")
    with b:
        _chart(receiver_map(case, grid, maps["mcrt_w_m2"], "光线追迹热流（多种子平均）", zmax=zmax), f"vs_m_{pick}")
    _chart(flux_diff_map(case, grid, maps["mcrt_w_m2"], maps["analytic_w_m2"], "解析 − 追迹"), f"vs_d_{pick}")
    x = next(x for x in states if x["key"] == pick)
    if x.get("soltrace_export"):
        st.caption(f"SolTrace 算例：{x['soltrace_export']['folder']}；运行：`{x['soltrace_export']['next_step']}`")
    _show_files(r, images=[])


def _solver_benchmark(case) -> None:
    from cspaps.benchmark import INNER_METHODS, INNER_ZH, OUTER_METHODS, OUTER_ZH

    st.write("在相同条件下比较求解器：同一场地、镜数、DNI 空间图、权重和计算时间。只有通过细网格复核（不占用安全裕度）的方案"
             "才计入；所有内层方法都对同一个上界（各方法得到的最小有效上界）计算差距，外层布局都用正式年度价值 V(L) 评分，"
             "并在另一组留出工况（不同代表太阳位置与云图）上重新排名。")
    part = st.radio("比较", ["inner", "outer"], horizontal=True, key="sb_part",
                    format_func=lambda v: {"inner": "内层：单时刻瞄准", "outer": "外层：布局搜索"}[v])
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("每个方法的时间预算 (s)", value=60 if part == "inner" else 300, step=10, min_value=5,
                             key=f"sb_budget_{part}")
    if part == "inner":
        n_states = c2.number_input("工况数（按 h×DNI 最大取）", value=4, min_value=1, step=1, key="sb_states")
        tighten = c3.checkbox("收紧热流限制使其起作用", value=False, key="sb_tight",
                              help="把所有管屏限值乘以系数，使中心瞄准在夏至正午超限约 30%。")
        factor = None
        if tighten:
            factor = st.slider("限值系数", 0.2, 1.0, 0.6, 0.05, key="sb_factor")
        methods = st.multiselect("方法", list(INNER_METHODS), default=list(INNER_METHODS), key="sb_inner_m",
                                 format_func=lambda m: INNER_ZH[m])
        _job_button("sbench", "运行内层比较", wf.task_solver_benchmark, "inner", int(n_states), float(budget),
                    limit_factor=factor, methods=tuple(methods) or None)
    else:
        seeds = c2.multiselect("随机种子", [1, 2, 3, 4], default=[1, 2], key="sb_seeds")
        methods = st.multiselect("方法", list(OUTER_METHODS), default=list(OUTER_METHODS), key="sb_outer_m",
                                 format_func=lambda m: OUTER_ZH[m])
        st.caption("外层比较每个方法、每个种子都要跑一次布局搜索和两次正式年度价值评估，耗时约为预算的数倍。")
        _job_button("sbench", "运行外层比较", wf.task_solver_benchmark, "outer", budget_s=float(budget),
                    seeds=tuple(seeds) or (1,), methods=tuple(methods) or None)
    _job_error("sbench")
    r = st.session_state.get("res_sbench")
    if not r:
        return
    if "inner_summary" in r:
        agg = r["inner_summary"]
        lim = r["summary"].get("limit_factor")
        _section(None, "内层：每种方法的汇总", f"热流限值 {r['summary']['max_flux_kw_m2']:.0f} kW/m²"
                 + (f"（原值 × {lim:g}）" if lim else "") + "；差距相对于该工况所有方法中最小的有效上界。")
        show = agg.rename(columns={"方法": "方法名", "states": "工况数", "feasible": "可行工况",
                                   "hours_weighted_objective_mwh": "按小时加权目标 (MWh)",
                                   "mean_gap_to_best_bound": "平均差距", "max_gap_to_best_bound": "最大差距",
                                   "mean_loss_to_best_found": "离最好方案平均损失", "time_s": "总时间 (s)",
                                   "all_within_budget": "都在预算内"})
        st.dataframe(show.drop(columns=["method", "objective_sum_mw"], errors="ignore"), hide_index=True, width="stretch")
        with st.expander("逐工况明细"):
            st.dataframe(r["inner"], hide_index=True, width="stretch")
    if "outer_summary" in r:
        verification = r["summary"]["outer"]["held_out"]
        _section(None, "外层：布局搜索", f"设计工况得到年度价值 V。复核集包含 {verification['new_states']} 个新工况，"
                 f"{verification['shared_states']} 个工况与设计集相同。")
        if not verification["new_states"]:
            st.info("同组工况不能验证排名的泛化能力。可在工况参数中指定另一张复核工况表。")
        st.dataframe(r["outer_summary"].drop(columns=["method"], errors="ignore"), hide_index=True, width="stretch")
        with st.expander("每次运行"):
            st.dataframe(r["outer"], hide_index=True, width="stretch")
    _show_files(r, images=[])


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
    task_names = {"layout_value": "年度布局比较", "layout": "镜场生成", "snapshot": "单时刻基准", "aps": "APS 瞄准优化",
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
        if task == "layout_value" and (d / "layout_value.json").is_file():
            key = f"replay_{d.name}_{d.parent.name}"
            if st.button("用冻结的布局与工况重新求解", key=f"start_{key}", disabled=f"job_{key}" in st.session_state):
                st.session_state[f"job_{key}"] = Job(wf.replay_layout_value, d, RESULTS, progress=True, cancel=True)
                st.rerun()
            if f"job_{key}" in st.session_state:
                _job_status(key)
            _job_error(key)
            replay = st.session_state.get(f"res_{key}")
            if replay and not replay["ranking"].empty:
                st.dataframe(_ranking_table(replay["ranking"]), hide_index=True)
                st.caption(f"复算结果目录 {replay['run_dir']}")
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
