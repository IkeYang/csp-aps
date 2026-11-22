"""The five first-time user tasks of docs/APS-IMPROVEMENT-PLAN.md, driven through the GUI (AppTest).

Each test follows what an engineer who only knows heliostats, DNI and the receiver would click, and checks
that the answer the task asks for is on the screen -- and that it is the right answer.
"""

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import ROOT, small_case_dict

pytest.importorskip("streamlit")


def _app(monkeypatch, tmp_path, case_dict=None, timeout=900):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=timeout)
    if case_dict is not None:
        at.session_state["case_dict"] = case_dict
        at.session_state["base_dir"] = str(ROOT)
        at.session_state["ver"] = 1
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _frames(at) -> list[pd.DataFrame]:
    return [d.value for d in at.dataframe]


def test_task1_home_example_single_instant_hotspot_and_margin(monkeypatch, tmp_path):
    """Pick the right task on the home page, run the example, read the hotspot and the remaining margin."""
    at = _app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception
    assert at.button(key="home_aps_demo") and at.button(key="home_layout_demo")
    at.button(key="home_aps_demo").click().run()
    assert at.session_state["page"] == "瞄准优化" and at.session_state["demo_data"]
    at.button(key="btn_single").click().run()
    assert not at.exception and not at.error
    s = at.session_state["res_single"]["summary"]
    mc, chk = s["margin_check"], s["flux_check"]
    text = _text(at)
    panel = mc.get("hotspot_panel", chk["hotspot"]["panel"])
    assert f"最热点在第 {panel} 号管屏" in text
    allowed_kw = mc["allowed_fraction"] * chk["peak_flux_kw_m2"] / mc["peak_fraction"]
    left = allowed_kw - chk["peak_flux_kw_m2"]
    assert f"还有 {left:.0f} kW/m²" in text and left >= 0  # the margin left is shown and the plan keeps it
    assert mc["within_margin"]


def test_task2_dni_table_with_a_bad_row_is_found_and_fixed(monkeypatch, tmp_path):
    """Import an own DNI table, find the faulty row from the preview, fix it and continue."""
    at = _app(monkeypatch, tmp_path, small_case_dict())
    at.run()
    at.button(key="home_aps_own").click().run()
    assert at.session_state["page"] == "瞄准优化" and at.session_state["own_data"] == "aps"
    at.radio(key="aps_dni_src").set_value("upload").run()
    bad = "x,y,dni\n-120,-120,850\n120,-120,870\n-120,120,\n120,120,900\n0,0,880\n"
    at.file_uploader(key="s_now").set_value(("my_dni.csv", bad.encode(), "text/csv")).run()
    errors = [e.value for e in at.error]
    assert any("第 3 行" in e and "dni" in e for e in errors)  # the faulty row is named
    assert at.button(key="btn_single").disabled  # nothing is solved with a stale / broken table
    good = bad.replace("-120,120,\n", "-120,120,860\n")
    at.file_uploader(key="s_now").set_value(("my_dni.csv", good.encode(), "text/csv")).run()
    assert not any("第 3 行" in e.value for e in at.error)
    assert not at.button(key="btn_single").disabled
    at.button(key="btn_single").click().run()
    assert not at.exception
    aims = pd.read_csv(at.session_state["res_single"]["files"]["aims_csv"])
    assert aims["dni_w_m2"].between(850, 900).all() and aims["dni_w_m2"].std() > 0  # every mirror its own DNI


def _run_layout_example(at):
    at.run()
    at.button(key="home_layout_demo").click().run()
    assert at.session_state["page"] == "布局优化" and at.session_state["layout_mode"] == "compare"
    assert at.selectbox(key="vc_A").value == "case:layout_A.csv" and at.selectbox(key="vc_B").value == "case:layout_B.csv"
    at.button(key="btn_lv").click().run()
    assert not at.exception and not at.error


def test_task3_two_layouts_same_maps_winner_energy_score_and_top_scenario(monkeypatch, tmp_path):
    """Two layouts + one set of annual maps: the winner, its actual annual heat, the score and the scenario
    that contributes most to the difference."""
    at = _app(monkeypatch, tmp_path)
    _run_layout_example(at)
    res = at.session_state["res_lv"]
    rk = res["ranking"].set_index("name")
    assert rk.loc["B", "rank"] == 1 and rk.loc["A", "rank"] == 2  # the plan's wrong-site example
    assert rk.loc["A", "value_mwh"] == pytest.approx(41.449, abs=5e-4)
    assert rk.loc["B", "value_mwh"] == pytest.approx(84.785, abs=5e-4)
    text = _text(at)
    assert "布局 B" in text and "得分最高" in text
    ab = next(f for f in _frames(at) if "年吸热量 E (MWh)" in f.columns)
    row_b = ab.set_index("布局").loc["布局 B"]
    assert row_b["年吸热量 E (MWh)"] == pytest.approx(round(rk.loc["B", "energy_mwh"], 1))
    assert row_b["综合得分 V (MWh 等效)"] == pytest.approx(round(rk.loc["B", "value_mwh"], 1))
    # hours are actual hours (time-average DNI), so E is the real annual heat
    hours = res["values"]["B"].table(0.0)
    assert hours["hours"].to_numpy() == pytest.approx(hours["actual_hours"].to_numpy())
    # the per-scenario table opens on the scenario with the largest contribution to the difference
    contrib = next(f for f in _frames(at) if "对分差的贡献 A − B (MWh 等效)" in f.columns)
    lam = float(res["ranking"]["lambda_layout"].iloc[0])
    names = list(res["values"])
    ta, tb = res["values"][names[0]].table(lam), res["values"][names[1]].table(lam)
    diff = (ta["contribution_mwh"] - tb["contribution_mwh"]).abs()
    assert contrib["工况"].iloc[0] == ta["scenario_id"].iloc[int(diff.to_numpy().argmax())]
    assert abs(contrib["对分差的贡献 A − B (MWh 等效)"].iloc[0]) == pytest.approx(round(diff.max(), 2))


def test_task4_preference_change_and_where_the_basis_is(monkeypatch, tmp_path):
    """Raise the "uniform heating" preference: what changes, and where are the parameters, data sources and
    solve scope of this calculation."""
    at = _app(monkeypatch, tmp_path)
    _run_layout_example(at)
    before = at.session_state["res_lv"]
    assert not any("需要重新计算" in m.value for m in at.markdown)
    basis = next(f for f in _frames(at) if "项目" in f.columns and "本次取值" in f.columns)
    items = set(basis["项目"])
    assert {"偏好 λ", "安全裕度", "数据来源", "DNI 地图", "候选瞄准点（求解范围）", "工况集指纹 / 设置指纹"} <= items
    src = basis.set_index("项目").loc["数据来源", "本次取值"]
    assert "scenarios.csv" in src  # the imported scenario list, not the station weather
    ver = at.session_state["ver"]
    lam0 = float(before["ranking"]["lambda_layout"].iloc[0])
    lam1 = min(1.0, lam0 + 0.5)
    at.slider(key=f"{ver}:pref_cmp").set_value(lam1).run()
    assert any("需要重新计算" in m.value for m in at.markdown)  # the shown result no longer matches the inputs
    at.button(key="btn_trial_cmp").click().run()
    trial = " ".join(m.value for m in at.markdown if "λ " in m.value and "吸收功率" in m.value)
    assert f"→ {lam1:.2f}" in trial and "最热点" in trial and "CV" in trial
    at.button(key="btn_lv").click().run()
    after = at.session_state["res_lv"]
    for name in ("A", "B"):  # exact inner solves: a higher weight never increases the non-uniformity
        u0 = before["values"][name].totals()["penalty_mwh"]
        u1 = after["values"][name].totals()["penalty_mwh"]
        e0 = before["values"][name].totals()["energy_mwh"]
        e1 = after["values"][name].totals()["energy_mwh"]
        assert u1 <= u0 + 1e-9 and e1 <= e0 + 1e-9
    assert float(after["ranking"]["lambda_layout"].iloc[0]) == pytest.approx(lam1)


@pytest.mark.parametrize("allow_defocus,status", [(True, "可行停机"), (False, "已证不可行")])
def test_task5_shutdown_proven_infeasible_and_not_found_are_told_apart(monkeypatch, tmp_path, allow_defocus, status):
    """A flux limit no single mirror can respect: with defocusing allowed the plan is a feasible shutdown,
    without it the constraints are proven infeasible. The table says which, and what to do next."""
    at = _app(monkeypatch, tmp_path)
    at.run()
    at.button(key="home_layout_demo").click().run()
    d = at.session_state["case_dict"]
    d["receiver"]["max_flux_kw_m2"] = 0.05  # 50 W/m2: no mirror can be on in any scenario
    d.setdefault("aps", {})["allow_defocus"] = allow_defocus
    at.session_state["ver"] += 1
    at.run()
    at.button(key="btn_lv").click().run()
    assert not at.exception
    table = next(f for f in _frames(at) if "下一步" in f.columns and "工况数" in f.columns)
    assert set(table["状态"]) == {status}
    advice = table["下一步"].iloc[0]
    assert ("核对约束" in advice) if not allow_defocus else ("降低 λ" in advice or "放宽约束" in advice)
    legend = next(f for f in _frames(at) if "含义" in f.columns and "下一步" in f.columns)
    rows = legend.set_index("状态")
    for s in ("未找到可行方案", "已证不可行", "可行停机"):
        assert s in rows.index
    assert rows.loc["未找到可行方案", "下一步"] != rows.loc["已证不可行", "下一步"] != rows.loc["可行停机", "下一步"]
    assert "增加预算" in rows.loc["未找到可行方案", "下一步"]
    res = at.session_state["res_lv"]
    for v in res["values"].values():
        t = v.totals()
        assert t["energy_mwh"] == 0.0 and t["coverage"] == 1.0  # scored 0, but every scenario was solved
