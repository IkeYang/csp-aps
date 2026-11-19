"""Graphical interface (Streamlit). Start with ``cspaps gui [CASE.yaml]``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from cspaps import workflows as wf
from cspaps.config import _SECTIONS, case_from_dict, load_case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "cases" / "example.yaml"
RESULTS = Path.cwd() / "results" / "gui"
UPLOADS = RESULTS / "_uploads"

SECTION_TITLES = {
    "site": "站址", "receiver": "集热器", "heliostat": "定日镜", "sun": "太阳", "atmosphere": "大气衰减",
    "weather": "气象数据", "shading": "阴影与遮挡", "field": "镜场", "aps": "瞄准策略 APS",
    "realtime": "实时仿真（云影）", "annual": "全年计算", "layout_opt": "布置优化", "mcrt": "MCRT 校验",
}
LABELS = {
    "site.latitude": "纬度 (°，北为正)", "site.longitude": "经度 (°，东为正)", "site.elevation_m": "海拔 (m)",
    "site.timezone": "时区",
    "receiver.center": "中心点 [x, y, z] (m)", "receiver.diameter": "直径 (m)", "receiver.height": "高度 (m)",
    "receiver.max_flux_kw_m2": "允许最大通量 (kW/m²)", "receiver.n_azimuth": "周向网格数",
    "receiver.n_height": "高度网格数", "receiver.tower_diameter": "塔身直径 (m，空 = 同集热器)",
    "heliostat.width": "宽 (m)", "heliostat.height": "高 (m)", "heliostat.pivot_height": "旋转中心高度 (m)",
    "heliostat.reflectivity": "反射率", "heliostat.cleanliness": "清洁度",
    "heliostat.slope_error_mrad": "斜率误差 (mrad)", "heliostat.tracking_error_mrad": "跟踪误差 (mrad)",
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
    "aps.allow_defocus": "允许散焦", "aps.milp_time_limit_s": "MILP 时限 (s)",
    "realtime.start": "开始时间", "realtime.duration_min": "时长 (min)", "realtime.step_s": "刷新周期 (s)",
    "realtime.forecast_horizon_s": "预报时长 (s)", "realtime.clouds": "云列表 (YAML)",
    "realtime.random_clouds": "随机云设置 (YAML)", "realtime.strategies": "对比策略 (YAML)",
    "annual.method": "计算方法", "annual.n_sun_bins": "代表太阳位置数", "annual.year": "年份",
    "annual.day_of_month": "代表日（每月几号）", "annual.time_step_min": "时间步长 (min)",
    "annual.min_sun_elevation_deg": "最低太阳高度角 (°)", "annual.aiming": "瞄准方式",
    "layout_opt.generator": "布置生成器", "layout_opt.n_heliostats": "镜子数量",
    "layout_opt.bounds": "优化参数范围 (YAML)", "layout_opt.fixed": "固定参数 (YAML)",
    "layout_opt.n_sun_bins": "代表太阳位置数", "layout_opt.shading_points": "阴影采样点数",
    "layout_opt.maxiter": "进化代数", "layout_opt.popsize": "种群规模系数", "layout_opt.seed": "随机种子",
    "layout_opt.workers": "并行进程 (-1 = 全部)", "layout_opt.refine_top": "APS 精算前几名",
    "layout_opt.polish_rounds": "局部微调轮数",
    "mcrt.rays_per_heliostat": "每面镜子光线数", "mcrt.seed": "随机种子",
}
CHOICES = {
    "sun.shape": ["gaussian", "pillbox"],
    "sun.dni_model": ["meinel", "constant"],
    "atmosphere.model": ["vittitoe_biggs_clear", "vittitoe_biggs_hazy", "beer_lambert"],
    "weather.source": ["clear_sky", "file"],
    "aps.method": ["greedy", "kfactor", "center", "milp"],
    "annual.method": ["sun_bins", "representative_days", "hourly"],
    "annual.aiming": ["center", "greedy", "kfactor"],
    "layout_opt.generator": ["radial_staggered", "biomimetic"],
}


def _initial_case_path() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return Path(args.case) if args.case else DEFAULT_CASE


def _load_into_state(data: dict, base_dir: Path) -> None:
    st.session_state["case_dict"] = data
    st.session_state["base_dir"] = str(base_dir)
    st.session_state["ver"] = st.session_state.get("ver", 0) + 1
    for k in [k for k in st.session_state if k.startswith("res_")]:
        del st.session_state[k]


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
                        if value is not None else "", key=key, height=110 if isinstance(value, (dict, list)) else 68)
    return yaml.safe_load(text) if text.strip() else None


def _parameter_forms() -> None:
    data = st.session_state["case_dict"]
    ver = st.session_state["ver"]
    data["name"] = st.text_input("算例名称", value=data.get("name", "case"), key=f"{ver}:name")
    defaults = case_from_dict({"site": {"latitude": 0, "longitude": 0}, "receiver": {"center": [0, 0, 1], "diameter": 1, "height": 1},
                               "heliostat": {"width": 1, "height": 1, "pivot_height": 1}})
    for section, cls in _SECTIONS.items():
        current = dict(data.get(section) or {})
        default_obj = dataclasses.asdict(getattr(defaults, section))
        with st.expander(SECTION_TITLES.get(section, section), expanded=section in ("receiver", "heliostat")):
            cols = st.columns(2)
            new = {}
            for i, f in enumerate(dataclasses.fields(cls)):
                value = current.get(f.name, default_obj.get(f.name))
                if isinstance(value, tuple):
                    value = list(value)
                if isinstance(default_obj.get(f.name), float) and isinstance(value, int) and not isinstance(value, bool):
                    value = float(value)
                with cols[i % 2]:
                    try:
                        new[f.name] = _widget(section, f.name, value, f"{ver}:{section}.{f.name}")
                    except yaml.YAMLError as exc:
                        st.error(f"{f.name}: YAML 格式错误 {exc}")
                        new[f.name] = value
            # keep only keys that differ from the defaults (keeps the saved YAML short), plus required ones
            data[section] = {k: v for k, v in new.items() if k in current or v != default_obj.get(k)}


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


def _show(result: dict, images: list[str] | None = None) -> None:
    files = result["files"]
    for label in images or [k for k in files if str(files[k]).endswith(".png")]:
        if label in files and Path(files[label]).exists():
            st.image(str(files[label]), width="stretch")
    with st.expander("结果数据 (JSON)"):
        st.json(result["summary"])
    cols = st.columns(4)
    for i, (label, path) in enumerate(files.items()):
        path = Path(path)
        if path.exists():
            with cols[i % 4]:
                st.download_button(f"下载 {path.name}", path.read_bytes(), file_name=path.name,
                                   key=f"dl:{st.session_state['ver']}:{path}")


def _run(tab_key: str, label: str, fn, *args, **kwargs) -> None:
    case, err = _case()
    if err:
        st.error(f"参数有误：{err}")
        return
    if st.button(label, key=f"btn_{tab_key}", type="primary"):
        status = st.empty()
        try:
            with st.spinner("计算中…"):
                out = wf.out_dir(RESULTS, case)
                if "progress" in kwargs:
                    kwargs["progress"] = lambda msg: status.text(msg)
                st.session_state[f"res_{tab_key}"] = fn(case, out, *args, **kwargs)
            status.empty()
        except Exception as exc:  # noqa: BLE001
            st.error(f"计算失败：{exc}")
            with st.expander("详细错误"):
                st.code(traceback.format_exc())
            return


def _metrics(pairs: list[tuple[str, object]]) -> None:
    cols = st.columns(len(pairs))
    for c, (k, v) in zip(cols, pairs):
        c.metric(k, v)


def main() -> None:
    st.set_page_config(page_title="CSP-APS 镜场与瞄准优化", layout="wide")
    if "case_dict" not in st.session_state:
        path = _initial_case_path()
        _load_into_state(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)

    with st.sidebar:
        st.header("算例")
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

    st.title("光热镜场布置与实时瞄准策略优化")
    tabs = st.tabs(["参数设置", "镜场布置", "瞄准优化 APS", "云影实时仿真", "全年能量", "布置优化",
                    "MCRT 校验", "代表太阳位置"])

    with tabs[0]:
        st.caption("修改后立即生效；可在左侧保存为 YAML。复杂参数用 YAML 写法，例如列表 [1, 2]。")
        c1, c2 = st.columns(2)
        with c1:
            layout_up = st.file_uploader("上传镜场坐标 CSV（列 x,y[,z]）", type=["csv"], key="layout_up")
            if layout_up is not None:
                st.session_state["case_dict"].setdefault("field", {})["layout_file"] = str(_save_upload(layout_up))
        with c2:
            weather_up = st.file_uploader("上传气象文件（EPW 或 CSV time,dni）", type=["epw", "csv"], key="weather_up")
            if weather_up is not None:
                st.session_state["case_dict"]["weather"] = {"source": "file", "file": str(_save_upload(weather_up))}
        _parameter_forms()

    with tabs[1]:
        _run("layout", "生成镜场", wf.task_layout)
        if "res_layout" in st.session_state:
            r = st.session_state["res_layout"]
            _metrics([("镜子数量", r["summary"]["n_heliostats"]), ("总镜面积 (m²)", r["summary"]["mirror_area_m2"])])
            _show(r)

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("日期", value=pd.Timestamp("2025-06-21").date(), key="aps_date")
        time = c2.time_input("当地时间", value=pd.Timestamp("13:40").time(), key="aps_time", step=300)
        method = c3.selectbox("方法", ["greedy", "kfactor", "center", "milp"], key="aps_method")
        c4, c5 = st.columns(2)
        dni = c4.number_input("DNI (W/m²，0 = 晴空模型)", value=0.0, step=50.0, key="aps_dni")
        dni_map = c5.file_uploader("实测 DNI 分布 CSV（列 x,y,dni，可选）", type=["csv"], key="aps_map")
        stamp = f"{date} {time}"
        _run("aps", "优化瞄准点", wf.task_aim, stamp, method=method, dni=dni or None,
             dni_map=_save_upload(dni_map) if dni_map else None)
        if "res_aps" in st.session_state:
            r = st.session_state["res_aps"]
            s = r["summary"]
            _metrics([("集热器功率 (MW)", s["power_mw"]["intercept"]), ("峰值通量 (kW/m²)", s["peak_flux_kw_m2"]),
                      ("总光学效率", s["total_optical_efficiency"]), ("散焦镜子", s["n_defocused"]),
                      ("求解时间 (s)", s["aps"].get("solve_time_s", 0))])
            st.dataframe(pd.DataFrame({"效率": s["field_efficiency"]}).T, width="stretch")
            _show(r, ["flux_png", "efficiency_png"])

    with tabs[3]:
        st.caption("按“实时仿真”参数模拟云影移动，对比：不做 APS / APS 仅用当前 DNI / APS 加 DNI 预报。")
        _run("rt", "运行实时仿真", wf.task_realtime, progress=True)
        if "res_rt" in st.session_state:
            r = st.session_state["res_rt"]
            st.dataframe(pd.DataFrame(r["summary"]), width="stretch")
            _show(r)

    with tabs[4]:
        c1, c2 = st.columns(2)
        a_method = c1.selectbox("计算方法", ["sun_bins", "representative_days", "hourly"], key="an_method")
        a_aim = c2.selectbox("瞄准方式", ["center", "greedy", "kfactor"], key="an_aim",
                             help="greedy/kfactor 会在每个时刻做 APS（满足通量上限），较慢")
        _run("annual", "计算全年能量", wf.task_annual, method=a_method, aiming=a_aim, progress=True)
        if "res_annual" in st.session_state:
            r = st.session_state["res_annual"]
            s = r["summary"]
            _metrics([("集热器年能量 (MWh)", s["energy_mwh"]["intercept"]),
                      ("年均光学效率", s["total_optical_efficiency"]),
                      ("超通量小时", s["hours_over_flux_limit"]), ("计算时间 (s)", s["runtime_s"])])
            st.dataframe(pd.DataFrame({"年均效率": s["annual_efficiency"]}).T, width="stretch")
            _show(r)

    with tabs[5]:
        st.caption("在场地边界内生成候选位置，按全年能量选出指定数量的镜子，用差分进化优化布置参数，"
                   "最后对前几名用 APS 精算。耗时取决于“布置优化”参数，示例约几分钟。")
        _run("opt", "开始布置优化", wf.task_optimize_layout, progress=True)
        if "res_opt" in st.session_state:
            r = st.session_state["res_opt"]
            s = r["summary"]
            _metrics([("镜子数量", s["n_heliostats"]), ("年能量 中心瞄准 (MWh)", s["annual_energy_mwh_center_aiming"]),
                      ("年能量 APS (MWh)", s["annual_energy_mwh_with_aps"]), ("评价次数", s["n_evaluations"])])
            st.write("最优布置参数：", s["generator_params"])
            st.dataframe(r["result"].ranking, width="stretch")
            if st.button("把优化后的镜场设为当前镜场"):
                st.session_state["case_dict"]["field"] = {**st.session_state["case_dict"].get("field", {}),
                                                         "layout_file": str(Path(r["files"]["layout_csv"]).resolve())}
                st.session_state["ver"] += 1
                st.rerun()
            _show(r)

    with tabs[6]:
        c1, c2, c3 = st.columns(3)
        v_date = c1.date_input("日期", value=pd.Timestamp("2025-06-21").date(), key="v_date")
        v_time = c2.time_input("当地时间", value=pd.Timestamp("13:40").time(), key="v_time", step=300)
        rays = c3.number_input("每面镜子光线数", value=1000, step=500, key="v_rays")
        _run("val", "运行 MCRT 校验", wf.task_validate, f"{v_date} {v_time}", rays=int(rays))
        if "res_val" in st.session_state:
            r = st.session_state["res_val"]
            s = r["summary"]
            _metrics([("功率相对差", s["power_on_receiver_mw"]["rel_diff"]), ("通量图 NRMSE", s["flux_map_nrmse"]),
                      ("通量图相关系数", s["flux_map_correlation"])])
            st.dataframe(pd.DataFrame(s["efficiency_by_stage"]).T, width="stretch")
            _show(r)

    with tabs[7]:
        check = st.checkbox("与逐时全年对比误差（较慢）", value=False)
        _run("bins", "计算代表太阳位置", wf.task_sunbins, check=check)
        if "res_bins" in st.session_state:
            _show(st.session_state["res_bins"])


main()
