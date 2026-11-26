"""Aim results in the GUI: the initial-state card, the single-mirror inspector (3-D light path and data), the
field efficiency breakdown and the initial state / moves tab (strict AppTest, small case)."""

import sys
from pathlib import Path

import pytest

from .conftest import ROOT, small_case_dict

GUI = str(Path(ROOT) / "src" / "cspaps" / "gui_app.py")


@pytest.fixture
def solved(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])

    def run(**aps):
        at = AppTest.from_file(GUI, default_timeout=300)
        d = small_case_dict()
        d["receiver"]["height"] = 16.0
        d.setdefault("aps", {}).update(aps)
        at.session_state["case_dict"] = d
        at.session_state["base_dir"] = str(ROOT)
        at.session_state["ver"] = 1
        at.run()
        at.radio(key="page").set_value("瞄准优化").run()
        at.button(key="btn_single").click().run()
        assert not at.exception
        return at

    return run


def _charts(at):
    return [c.key for c in at.get("plotly_chart")]


def _text(at):
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def test_initial_state_card_and_move_settings(solved):
    at = solved()
    text = _text(at)
    assert "初始状态与镜面移动" in text
    choice = at.radio(key="1:aps.initial_aim")
    assert choice.value == "spread" and list(choice.options) == ["均布瞄准（推荐）", "全部瞄中心", "不设初始状态（纯功率）"]
    assert "三种初始方案有什么不同？怎么选" in [e.label for e in at.expander]
    assert "瞄哪儿都差不多的镜子" in text and "初始方案：均布瞄准" in text  # the guide and the result label
    labels = [n.label for n in at.number_input]
    assert any(label.startswith("移动代价") for label in labels)
    assert any(label.startswith("瞄准点最大移动") for label in labels)
    assert any(label.startswith("镜面最大转角") for label in labels)


def test_inspector_shows_the_light_path_and_data(solved):
    at = solved()
    keys = _charts(at)
    assert {"insp_map", "insp_3d", "insp_factors", "insp_spot"} <= set(keys)
    at.radio(key="insp_view").set_value("镜面近景（阴影与遮挡）").run()
    assert not at.exception and "insp_close" in _charts(at)
    text = _text(at)
    assert "单镜详情" in text and "效率链" in text and "cs-spec" in text
    res = at.session_state["res_single"]
    ids = list(res["field"].ids)
    eta = res["snapshot"].eta_total
    # by default the mirror delivering the most power is shown
    p = res["snapshot"].power["intercept"]
    box = next(n for n in at.number_input if n.label == "镜子 ID")
    assert at.session_state.get("insp_id") is None and box.value == int(ids[int(p.argmax())])
    # quick pick: the most efficient mirror
    at.button(key="insp_pick_0").click().run()
    assert not at.exception
    best = int(ids[int(eta.argmax())])
    assert at.session_state["insp_id"] == best
    # typing an id selects it
    box = next(n for n in at.number_input if n.label == "镜子 ID")
    box.set_value(int(ids[3])).run()
    assert at.session_state["insp_id"] == int(ids[3])
    assert "相对初始状态" in "".join(m.value for m in at.markdown)


def test_field_efficiency_breakdown(solved):
    at = solved()
    assert "s_field" in _charts(at)
    text = _text(at)
    assert "为什么有些远处的镜子效率反而更高" in text and "光学效率 = 送达吸热器的功率" in text
    radio = at.radio(key="s_field_color")
    assert {"余弦效率", "阴影 × 遮挡", "大气衰减", "截断效率", "瞄准点移动"} <= set(radio.options)
    radio.set_value("余弦效率").run()
    assert not at.exception


def test_moves_tab_compares_initial_and_final(solved):
    at = solved()
    keys = _charts(at)
    assert {"mv_flux0", "mv_flux1", "mv_aims", "mv_hist"} <= set(keys)
    s = at.session_state["res_single"]["summary"]
    assert s["initial_state"]["source"] == "spread" and s["moves"]["within_limits"]
    assert "初始状态：均布瞄准" in _text(at)


def test_picking_an_initial_plan_changes_the_case_and_the_result(solved):
    at = solved()
    at.radio(key="1:aps.initial_aim").set_value("center").run()
    assert not at.exception and at.session_state["case_dict"]["aps"]["initial_aim"] == "center"
    at.button(key="btn_single").click().run()
    assert not at.exception
    assert at.session_state["res_single"]["summary"]["initial_state"]["source"] == "center"
    text = _text(at)
    assert "初始方案：全部瞄中心" in text and "初始状态：全部瞄中心" in text


def test_without_initial_state_the_page_says_so(solved):
    at = solved(initial_aim="none")
    text = _text(at)
    assert "没有初始状态" in text
    assert "mv_flux0" not in _charts(at)
    assert "initial_state" not in at.session_state["res_single"]["summary"]


def test_charts_are_archived_once_and_shown_offline_in_the_history(solved):
    import json

    at = solved()
    run_dir = Path(at.session_state["res_single"]["run_dir"])
    charts = run_dir / "visualizations"
    keys = {c["key"] for c in json.loads((charts / "index.json").read_text(encoding="utf-8"))["charts"]}
    assert {"s_full_flux", "insp_3d", "mv_flux0"} <= keys
    files = sorted(p.name for p in charts.glob("*.json.gz"))
    at.button(key="insp_pick_0").click().run()  # another mirror: an interaction, not a new result
    assert sorted(p.name for p in charts.glob("*.json.gz")) == files
    at.radio(key="page").set_value("运行记录").run()
    assert not at.exception
    pick = at.selectbox(key="history_run_pick")
    aps = next(i for i, label in enumerate(pick.options) if "APS 瞄准优化" in label)
    pick.set_value(aps).run()
    assert not at.exception
    assert any(str(c.key).startswith("history_figure_") for c in at.get("plotly_chart"))  # Streamlit's plotly
    table = at.dataframe[0].value
    assert "状态" in table and set(table["状态"]) <= {"已完成", "约束通过", "用到安全裕度", "不可执行", "均匀度未达标",
                                                    "场地容量不足", "几何约束不满足"}
