import json
import os
from pathlib import Path

import pytest
import yaml

from cspaps.cli import main
from cspaps.workflows import list_runs

from .conftest import ROOT, small_case_dict


@pytest.fixture
def case_file(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(small_case_dict()), encoding="utf-8")
    return path


def _run_dir(out: Path, task: str) -> Path:
    dirs = sorted((out / "tiny").glob(f"{task}_*"))
    assert dirs, f"no run folder for {task}"
    d = dirs[-1]
    assert (d / "inputs_case.yaml").exists() and (d / "run.json").exists()
    return d


def test_cli_layout_snapshot_aps(case_file, tmp_path):
    out = tmp_path / "out"
    assert main(["layout", str(case_file), "--out", str(out)]) == 0
    assert (_run_dir(out, "layout") / "layout.csv").read_text().startswith("id,x,y,z")
    assert main(["snapshot", str(case_file), "--time", "2025-06-21 13:40", "--out", str(out)]) == 0
    data = json.loads((_run_dir(out, "snapshot") / "snapshot_20250621_1340_center.json").read_text())
    assert data["n_heliostats"] > 100 and data["status"] == "ok"
    assert main(["aps", str(case_file), "--time", "2025-06-21 13:40", "--out", str(out)]) == 0
    aims = _run_dir(out, "aps") / "aps_20250621_1340_greedy_aims.csv"
    dni_map = tmp_path / "dni.csv"
    dni_map.write_text("x,y,dni\n-200,-200,600\n200,-200,600\n-200,200,900\n200,200,900\n")
    assert main(["aps", str(case_file), "--time", "2025-06-21 13:45", "--prev-aims", str(aims),
                 "--dni-map", str(dni_map), "--out", str(out)]) == 0
    runs = list_runs(out)
    assert {"layout", "snapshot", "aps"} <= set(runs["task"])


def test_cli_infeasible_exit_code(tmp_path):
    d = small_case_dict()
    d["receiver"]["max_flux_kw_m2"] = 5
    path = tmp_path / "tight.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert main(["snapshot", str(path), "--time", "2025-06-21 13:40", "--out", str(tmp_path / "o")]) == 2


def test_cli_tools(case_file, tmp_path):
    out = tmp_path / "out"
    assert main(["sunbins", str(case_file), "--bins", "8", "--out", str(out)]) == 0
    assert main(["validate", str(case_file), "--time", "2025-06-21 13:40", "--rays", "200", "--seeds", "2",
                 "--out", str(out)]) == 0
    v = _run_dir(out, "validate")
    assert (v / "validate_20250621_1340_center_mcrt_kw_m2.csv").read_text().startswith("# kW/m2; grid")
    assert main(["uniformity", str(case_file), "--time", "2025-06-21 13:00", "--eps", "0.9", "0.7",
                 "--out", str(out)]) == 0
    assert main(["shading-check", str(case_file), "--out", str(out)]) == 0
    assert main(["soltrace-export", str(case_file), "--time", "2025-06-21 13:00", "--rays", "100", "--seeds", "1",
                 "--out", str(out)]) == 0
    assert main(["annual", str(case_file), "--method", "sun_bins", "--out", str(out)]) == 0
    assert main(["aps", str(case_file), "--time", "2025-06-21 13:00", "--weight", "0.3", "--out", str(out)]) == 0
    data = json.loads(next(_run_dir(out, "aps").glob("aps_*_greedy.json")).read_text())
    assert data["certificate"]["status"] == "certified" and data["existence"]["optimum_exists"]
    assert main(["pareto", str(case_file), "--time", "2025-06-21 13:00", "--lambdas", "0", "1", "--out", str(out)]) == 0
    curve = json.loads((_run_dir(out, "pareto") / "pareto.json").read_text())["curve"]
    assert curve[1]["cv_2d"] <= curve[0]["cv_2d"] and all(r["gap"] is not None for r in curve)
    assert main(["layout-tradeoff", str(case_file), "--lambdas", "0", "0.5", "--out", str(out)]) == 0
    rows = json.loads((_run_dir(out, "layout_tradeoff") / "layout_tradeoff.json").read_text())["curve"]
    assert rows[1]["cv_circ_weighted"] < rows[0]["cv_circ_weighted"]


def test_gui_walkthrough(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    d = small_case_dict()
    d["receiver"]["max_flux_kw_m2"] = 350
    d["layout_opt"].update({"review_n_sun_bins": 6, "review_dni_bands": 2, "swap_rounds": 1, "review_check_hourly": False})
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=600)
    at.session_state["case_dict"] = d
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    assert not at.exception  # overview page
    at.radio(key="page").set_value("云影回放").run()
    sections = [item.value for item in at.markdown]
    preview_index = next(i for i, text in enumerate(sections) if "场地预览" in text)
    settings_index = next(i for i, text in enumerate(sections) if "回放设置" in text)
    assert preview_index < settings_index
    assert at.get("plotly_chart")[0].key == "replay_site_preview"
    at.button(key="btn_rt").click().run()
    assert not at.exception and not at.error
    cards = " ".join(m.value for m in at.markdown)
    for label in ("指令状态", "最高通量（复核）", "距上限余量", "散焦镜数", "数据新鲜度"):
        assert label in cards
    at.radio(key="page").set_value("布局优化").run()
    assert at.radio(key="layout_mode").value == "compare"  # the engineer's main path opens first
    at.radio(key="layout_mode").set_value("search").run()
    sections = [item.value for item in at.markdown]
    preview_index = next(i for i, text in enumerate(sections) if "场地预览" in text)
    inputs_index = next(i for i, text in enumerate(sections) if "场地、设备与位置" in text)
    assert preview_index < inputs_index
    assert at.get("plotly_chart")[0].key == "site_map"
    at.slider(key="1:pref_opt").set_value(0.5).run()
    assert at.session_state["case_dict"]["layout_opt"]["uniformity_weight"] == 0.5
    at.button(key="btn_opt").click().run()
    assert not at.exception
    res = at.session_state["res_opt"]
    ub = res["summary"]["upper_bound"]
    assert ub["upper_bound_mwh"] >= ub["found_energy_mwh"] > 0 and res["summary"]["existence"]["optimum_exists"]
    uni = res["summary"]["uniformity"]
    assert uni["weight"] == 0.5 and uni["north_fraction"] < 0.7
    at.text_input(key="ltrade_lams").set_value("0, 0.5").run()
    at.button(key="btn_ltrade").click().run()
    assert not at.exception
    curve = at.session_state["res_ltrade"]["summary"]["curve"]
    assert curve[0]["north_fraction"] > curve[1]["north_fraction"]
    at.button(key="apply_layout").click().run()
    chk = at.session_state["apply_check"]
    assert chk["导出镜数"] == chk["重新载入镜数"] == 120 and chk["编号与坐标逐项一致"]
    for page in ("光学校验", "运行记录", "方法与理论", "参数设置", "总览"):
        at.radio(key="page").set_value(page).run()
        assert not at.exception, page


def test_single_result_keeps_completed_objective(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    data = small_case_dict()
    data["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=120)
    at.session_state["case_dict"] = data
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("瞄准优化").run()
    at.button(key="btn_single").click().run()
    assert not at.exception and not at.error
    sections = [item.value for item in at.markdown]
    preview_index = next(i for i, text in enumerate(sections) if "场地预览" in text)
    inputs_index = next(i for i, text in enumerate(sections) if "运行工况" in text)
    assert preview_index < inputs_index
    assert at.get("plotly_chart")[0].key == "site_aps"
    cert = at.session_state["res_single"]["summary"]["certificate"]
    assert cert["status"] == "certified" and cert["upper_bound_mw"] >= cert["objective_found_mw"]
    at.slider(key="1:uniformity.weight").set_value(2.0).run()
    assert at.session_state["case_dict"]["uniformity"]["weight"] == 2.0
    assert any("均匀度目标 λ = 0.3" in c.value for c in at.caption)
    at.text_input(key="pareto_lams").set_value("0, 0.5").run()
    at.button(key="btn_pareto").click().run()
    assert not at.exception
    curve = at.session_state["res_pareto"]["summary"]["curve"]
    assert [r["lambda"] for r in curve] == [0.0, 0.5] and curve[1]["cv_2d"] <= curve[0]["cv_2d"]
    at.radio(key="aps_goal:1").set_value("cv_limit").run()
    at.number_input(key="1:uniformity.max_cv").set_value(0.01).run()
    at.button(key="btn_single").click().run()
    assert not at.exception
    assert at.session_state["res_single"]["summary"]["status"] == "uniformity_infeasible"
    assert "记录为人工已确认" not in [b.label for b in at.button]


def test_gui_cloud_compare_and_formal_value(monkeypatch, tmp_path):
    """Cloud before / after page, the formal layout ranking view, rescoring and the audit recomputation."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    d = small_case_dict()
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["scenarios"] = {"source": "weather", "n_scenarios": 2, "map_model": "half_swap"}
    d["field"]["generator"]["r_max"] = 80
    d["layout_opt"].update({"n_heliostats": 60, "refine_top": 1, "review_check_hourly": False, "swap_rounds": 0,
                            "formal_swap_rounds": 0})
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=600)
    at.session_state["case_dict"] = d
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("云影回放").run()
    at.button(key="btn_cloud").click().run()
    assert not at.exception and not at.error
    keys = [s["key"] for s in at.session_state["res_cloud"]["states"]]
    assert keys == ["clear", "cloud_keep", "cloud_aps", "clear_keep", "cloud_safe", "clear_safe"]
    labels = " ".join(m.value for m in at.markdown)
    for text in ("① 云前吸收功率", "③ 云来 · 不改策略", "④ 云来 · 重新 APS", "被云遮挡"):
        assert text in labels
    at.radio(key="page").set_value("布局优化").run()
    at.radio(key="layout_mode").set_value("search").run()
    at.button(key="btn_opt").click().run()
    assert not at.exception
    f = at.session_state["res_opt"]["summary"]["formal"]
    assert {r["name"] for r in f["ranking"]} >= {"initial"} and f["winner"]
    at.slider(key="fv_opt_lam").set_value(1.0).run()
    assert not at.exception
    at.button(key="fv_opt_recompute").click().run()
    assert all(r["一致"] == "✓" for r in at.session_state["fv_opt_recheck"])
    at.radio(key="layout_mode").set_value("compare").run()  # the optimised layout against the initial field
    assert at.selectbox(key="vc_A").value == "opt" and at.selectbox(key="vc_B").value == "initial"
    at.button(key="btn_lv").click().run()
    assert not at.exception
    assert at.session_state["res_lv"]["summary"]["equal_heliostat_count"]
    # problem 1 on one annual scenario (sun + site DNI map): the independent check of a layout score
    at.radio(key="page").set_value("瞄准优化").run()
    at.button(key="btn_scen").click().run()
    assert not at.exception and not at.error
    rec = at.session_state["res_scen"]["record"]
    assert rec.scenario_id == "S01" and rec.plan.dni.std() > 0  # the half-swap map gives every mirror its own DNI
