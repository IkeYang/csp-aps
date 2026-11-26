"""The story of a cloud replay (``replay_story``): scorecard, cloud episodes, events, the clearing moment, the
panel timeline -- on a hand-made table with known answers and on a real replay -- and the replay page."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cspaps.realtime import RealtimeResult
from cspaps.replay_story import clearing_spotlight, cloud_episodes, events, panel_timeline, scorecard

from .conftest import ROOT, small_case_dict

T0 = pd.Timestamp("2025-06-21 12:00", tz="Asia/Shanghai")


def _table():
    """10 samples of 30 s; a cloud covers 40 % of the field at samples 2-5 and clears at sample 6. "late" keeps
    its shadow plan: over the limit at samples 6-7. "safe" stays inside but uses the margin at sample 6."""
    n = 10
    cover = np.array([0, 0, .4, .4, .4, .4, 0, 0, 0, 0], float)
    dni = 1000 * (1 - 0.8 * cover)
    rows = []
    for strat, ratio in (("late", [.6, .6, .5, .5, .5, .5, 1.2, 1.1, .7, .7]),
                         ("safe", [.6, .6, .5, .5, .5, .5, .97, .8, .7, .7])):
        for j in range(n):
            rows.append({"time": T0 + pd.Timedelta(seconds=30 * j), "interval_s": 30.0 if j < n - 1 else 0.0,
                         "strategy": strat, "control": j % 2 == 0, "control_index": j // 2,
                         "dni_mean_w_m2": dni[j], "dni_clear_w_m2": 1000.0, "cloud_cover": cover[j],
                         "p_intercepted_mw": 10 * dni[j] / 1000, "p_absorbed_mw": 9 * dni[j] / 1000,
                         "p_absorbed_clear_mw": 9.0, "peak_flux_kw_m2": 1000 * ratio[j], "peak_to_limit": ratio[j],
                         "over_limit": ratio[j] > 1, "exceedance_kw_m2": max(0, 1000 * (ratio[j] - 1)),
                         "hotspot_panel": 1, "n_defocused": 0, "n_moved": 5 if (j == 6 and strat == "safe") else 0,
                         "n_defocus_commands": 0, "plan_state": "ready", "plan_reasons": "",
                         "engine_total_s": 0.1 if j % 2 == 0 else np.nan, "response_reused": True,
                         "uniformity_feasible": True, "operational_feasible": ratio[j] <= 1,
                         "feasible_absorbed_mw": 9 * dni[j] / 1000 if ratio[j] <= 1 else 0.0})
    table = pd.DataFrame(rows)
    frames = [{"time": T0 + pd.Timedelta(seconds=30 * j), "control_index": j // 2, "control": j % 2 == 0,
               "plans": {s: {"check": {"top_contributors": []}, "flux": np.zeros((2, 2))} for s in ("late", "safe")}}
              for j in range(n)]
    return RealtimeResult(table, {}, None, 60.0, 270.0, frames)


class _Case:
    class aps:
        safety_margin = 0.05


def test_scorecard_episode_and_events_on_a_known_table():
    rr = _table()
    eps = cloud_episodes(rr.table)
    assert len(eps) == 1 and eps[0]["start"] == T0 + pd.Timedelta(seconds=60)
    assert eps[0]["end"] == T0 + pd.Timedelta(seconds=180) and eps[0]["duration_s"] == 120
    assert eps[0]["max_cover"] == pytest.approx(0.4)
    sc = scorecard(rr, _Case).set_index("strategy")
    assert sc.loc["late", "time_over_limit_s"] == 60 and sc.loc["safe", "time_over_limit_s"] == 0
    assert sc.loc["safe", "time_in_margin_s"] == 30  # 0.97 > 1 - 0.05
    assert sc.loc["late", "max_peak_to_limit"] == pytest.approx(1.2)
    # absorbed: 9 MW for 6 clear samples (incl. the last one of zero length) and 9 x 0.68 under the cloud
    expected = (9 * 30 * 5 + 9 * 0.68 * 30 * 4) / 3600
    assert sc.loc["late", "absorbed_mwh"] == pytest.approx(expected, rel=1e-3)
    assert sc.loc["late", "cloud_loss_fraction"] == pytest.approx(1 - expected / (9 * 270 / 3600), abs=1e-3)
    assert sc.loc["late", "feasible_absorbed_mwh"] < sc.loc["safe", "feasible_absorbed_mwh"]
    ev = events(rr, _Case)
    kinds = list(zip(ev["kind"], [s if isinstance(s, str) else None for s in ev["strategy"]]))
    assert ("cloud_enter", None) in kinds and ("cloud_leave", None) in kinds
    assert ("dni_drop", None) in kinds and ("dni_rise", None) in kinds
    rise = ev[ev["kind"] == "dni_rise"].iloc[0]
    assert rise["time"] == T0 + pd.Timedelta(seconds=180) and rise["value"] == pytest.approx(0.32)
    over = ev[(ev["kind"] == "over_limit")].iloc[0]
    assert over["strategy"] == "late" and over["value"] == pytest.approx(1.2) and over["detail"] == "60"
    assert ((ev["kind"] == "back_within_limit") & (ev["strategy"] == "late")).sum() == 1
    assert ((ev["kind"] == "margin_used") & (ev["strategy"] == "safe")).sum() == 1
    assert ((ev["kind"] == "replan") & (ev["strategy"] == "safe")).sum() == 1
    assert list(ev["time"]) == sorted(ev["time"])


def test_clearing_spotlight_finds_the_dangerous_moment():
    rr = _table()
    spot = clearing_spotlight(rr, _Case)
    assert spot["before"] == 5 and spot["at"] == 6
    assert spot["dni_ratio_before"] == pytest.approx(0.68) and spot["dni_ratio_at"] == pytest.approx(1.0)
    late, safe = spot["strategies"]["late"], spot["strategies"]["safe"]
    assert late["worst"] == 6 and late["over_limit"] and late["before_peak_to_limit"] == pytest.approx(0.5)
    # the clearing (sample 6) is a control instant: the plan in effect in interval 3 was decided at sample 4 (before),
    # the plan decided at the clearing takes effect at sample 8 -> the window is samples 6-7
    assert spot["window_end"] == T0 + pd.Timedelta(seconds=210)
    assert not safe["over_limit"] and safe["peak_to_limit"] == pytest.approx(0.97)


def test_story_of_a_real_sudden_clearing(small_case):
    from cspaps.field import build_field
    from cspaps.realtime import run_realtime

    small_case.receiver.max_flux_kw_m2 = 350
    small_case.realtime.scenario = "sudden_clear"
    small_case.realtime.duration_min = 6
    fld = build_field(small_case)
    rr = run_realtime(small_case, fld)
    spot = clearing_spotlight(rr, small_case)
    assert spot is not None and spot["dni_ratio_at"] > spot["dni_ratio_before"] + 0.1
    half = rr.table["time"].min() + pd.Timedelta(seconds=0.5 * 6 * 60)
    assert pd.Timestamp(spot["time"]) == half  # the cloud ends at mid-replay (scenario definition)
    sc = scorecard(rr, small_case).set_index("strategy")
    assert sc.loc["aps_predictive", "time_over_limit_s"] == 0
    assert (sc["clear_sky_mwh"] >= sc["absorbed_mwh"] - 1e-9).all()
    pt = panel_timeline(rr, small_case)
    assert set(pt) == set(rr.frames[0]["plans"]) and pt["center"].shape == (len(rr.frames), small_case.receiver.n_panels)
    # the verified peak of the whole receiver bounds the coarse panel maxima from above (finer grid)
    for s, m in pt.items():
        g = rr.table[rr.table["strategy"] == s]["peak_to_limit"].to_numpy()
        assert np.all(m.max(axis=1) <= g * 1.02 + 1e-9)


@pytest.fixture
def replay_app(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=300)
    d = small_case_dict()
    d["receiver"]["max_flux_kw_m2"] = 350
    d["realtime"].update({"scenario": "sudden_clear", "duration_min": 4})
    at.session_state["case_dict"] = d
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("云影回放").run()
    at.button(key="btn_rt").click().run()
    assert not at.exception
    return at


def _keys(at):
    return [str(c.key) for c in at.get("plotly_chart")]


def test_replay_page_tells_the_story(replay_app):
    at = replay_app
    keys = _keys(at)
    for prefix in ("rp_player_", "rp_timeline_", "rp_panels_", "rp_spot_"):
        assert any(k.startswith(prefix) for k in keys), prefix
    text = "\n".join(m.value for m in at.markdown)
    assert "这次过云发生了什么" in text and "cs-scores" in text and "云散瞬间" in text
    assert any("回放" in m.value and "分钟" in m.value for m in at.markdown)
    # an event row jumps the detail to its time
    run_key = next(k for k in keys if k.startswith("rp_player_")).removeprefix("rp_player_")
    slider = f"ops_t_{run_key}"
    events = next(d for d in at.dataframe if "事件" in d.value.columns)
    rise = int(np.flatnonzero(events.value["事件"].to_numpy() == "云散 · 日照骤升")[0])
    target = events.value["时间"].iloc[rise]
    at.session_state[f"rp_events_{run_key}_云与日照-安全"] = {"selection": {"rows": [rise], "columns": []}}
    at.run()
    assert not at.exception
    assert at.session_state[slider] == target
    # the spotlight button jumps there as well
    at.session_state[slider] = at.select_slider(key=slider).options[0]
    at.run()
    spot_btn = next(b for b in at.button if str(b.key).startswith("rp_spot_jump_"))
    spot_btn.click().run()
    assert at.session_state[slider] == target


def test_restored_history_tells_the_same_story(replay_app):
    at = replay_app
    at.radio(key="page").set_value("运行记录").run()
    assert not at.exception
    keys = _keys(at)
    assert any(k.startswith("rp_player_") for k in keys) and any(k.startswith("rp_timeline_") for k in keys)
    assert any("这次过云发生了什么" in m.value for m in at.markdown)
