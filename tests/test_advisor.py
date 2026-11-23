"""Algorithm depth by field size and time budget (fast / standard / deep), time model and pools."""

import json
import pickle
import threading
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest

from cspaps.advisor import (MODES, Calibration, apply_mode, estimate_seconds, fit_power, load_calibration,
                            mode_overrides, recommend, tier_of)


def test_tiers_follow_the_field_size():
    assert tier_of(4, 3)["key"] == "tiny"  # 4^4 = 256 joint poses: exhaustive
    assert tier_of(5, 3)["key"] == "small"
    assert tier_of(473, 21)["key"] == "small" and tier_of(1_000, 21)["key"] == "small"
    assert tier_of(3_000, 21)["key"] == "medium" and tier_of(5_000, 21)["key"] == "medium"
    assert tier_of(12_000, 21)["key"] == "large" and tier_of(20_000, 21)["key"] == "large"
    for n in (4, 473, 3_000, 12_000):
        t = tier_of(n, 3 if n == 4 else 21)
        assert t["label"] and t["text"]


def test_deep_aims_refine_the_standard_candidates(small_case):
    deep = mode_overrides(small_case, "aim", "deep")["aps"]
    std = small_case.aps
    assert deep["n_height_levels"] == 2 * std.n_height_levels - 1
    assert set(std.azimuth_offsets_deg) <= set(deep["azimuth_offsets_deg"])  # nested: every candidate stays
    assert mode_overrides(small_case, "aim", "fast") == {"aps": {"certify": False}}
    c = apply_mode(small_case, "aim", "deep")
    assert c.aps.n_height_levels == deep["n_height_levels"] and small_case.aps.n_height_levels == std.n_height_levels


@pytest.mark.parametrize("task", ["aim", "compare", "search"])
def test_deeper_takes_longer_and_the_deepest_that_fits_is_recommended(small_case, task):
    cal = Calibration()
    t = [estimate_seconds(small_case, task, m, cal, n=2_000, workers=8)["seconds"] for m in MODES]
    assert t[0] < t[1] < t[2]
    for budget, expect in ((t[2] * 10, "deep"), ((t[1] + t[2]) / 2, "standard"), (t[0] * 0.1, "fast")):
        rec = recommend(small_case, task, budget, cal, workers=8, n=2_000)
        assert rec["recommended"] == expect, (budget, rec["reason"])
        assert [o["recommended"] for o in rec["options"]].count(True) == 1
        assert all(o["text"] and o["label"] for o in rec["options"])
    assert "超出" in recommend(small_case, task, 1e-3, cal, workers=8, n=2_000)["reason"]


def test_more_mirrors_take_longer(small_case):
    cal = Calibration()
    for task in ("aim", "compare", "search"):
        small = estimate_seconds(small_case, task, "standard", cal, n=1_000, workers=4)["seconds"]
        big = estimate_seconds(small_case, task, "standard", cal, n=20_000, workers=4)
        assert big["seconds"] > 20 * small and big["extrapolated"]


def test_power_law_fit():
    n = np.array([400.0, 1_000.0, 2_500.0])
    a, p = fit_power(n, 0.01 * n ** 1.2)
    assert p == pytest.approx(1.2) and a == pytest.approx(0.01)
    assert fit_power(n, 0.01 * n ** 2.0)[1] == 1.3  # capped: the cost per mirror levels off for big fields
    assert fit_power(n, 0.01 * n ** 0.5)[1] == 1.0
    a, p = fit_power([1_000.0], [5.0])
    assert (a, p) == (0.005, 1.0)


def test_calibration_file_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    cal = Calibration(aim={m: [1e-3, 1.1] for m in MODES}, source="test", n_max=5_000)
    from dataclasses import asdict

    path.write_text(json.dumps(asdict(cal)), encoding="utf-8")
    monkeypatch.setenv("CSPAPS_CALIBRATION", str(path))
    got = load_calibration()
    assert got.source == "test" and got.aim["deep"] == [1e-3, 1.1] and got.n_max == 5_000
    path.write_text("not json", encoding="utf-8")
    assert load_calibration().source.startswith("built-in")


class _Square:
    def __call__(self, x):
        return x * x


def test_background_pools_use_processes_when_the_work_pickles(monkeypatch):
    from cspaps import parallel

    seen = {}

    def job(probe, key):
        p = parallel.pool(2, probe=probe)
        seen[key] = type(p).__name__
        p.close()
        p.join()

    class Local:  # defined in a function: cannot be pickled, like a class of a hot-reloaded module
        pass

    for probe, key in ((( _Square(), 3), "picklable"), ((Local(), 1), "not_picklable"), (None, "no_probe")):
        t = threading.Thread(target=job, args=(probe, key))
        t.start()
        t.join()
    assert seen["picklable"] == "Pool" and seen["not_picklable"] == "ThreadPool" and seen["no_probe"] == "ThreadPool"
    with pytest.raises(Exception):
        pickle.dumps(Local())
    monkeypatch.setenv("CSPAPS_POOL", "thread")
    p = parallel.pool(2, probe=(_Square(), 3))
    assert isinstance(p, ThreadPool)
    p.close()
    p.join()


def test_cli_advise_and_mode(small_case, tmp_path):
    import yaml

    from cspaps.cli import main
    from tests.conftest import small_case_dict

    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(small_case_dict()), encoding="utf-8")
    assert main(["advise", str(path), "--task", "search", "--budget-min", "20"]) == 0
    out = tmp_path / "out"
    assert main(["layout-value", str(path), "--layouts", "initial", "--mode", "fast", "--out", str(out)]) == 0
    from cspaps import load_case

    used = load_case(next(out.rglob("inputs_case.yaml")))
    assert used.scenarios.n_scenarios == 12 and used.layout_opt.inner_joint_trials == 1


def test_gui_depth_selector_explains_and_applies_only_when_chosen(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from tests.conftest import ROOT, small_case_dict

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=300)
    at.session_state["case_dict"] = small_case_dict()
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("瞄准优化").run()
    r = at.radio(key="depth_aim")
    assert r.value == "custom" and len(r.options) == 4 and sum("推荐" in o for o in r.options) == 1
    assert at.session_state["case_dict"].get("aps", {}).get("n_height_levels") is None  # nothing changed yet
    text = " ".join(m.value for m in at.markdown)
    assert "快速" in text or "深度思考" in text
    r.set_value("deep").run()
    assert not at.exception
    assert at.session_state["case_dict"]["aps"]["n_height_levels"] == 13
    at.radio(key="page").set_value("布局优化").run()
    at.radio(key="layout_mode").set_value("search").run()
    at.radio(key="depth_search").set_value("fast").run()
    assert at.session_state["case_dict"]["layout_opt"]["maxiter"] == 2
    assert at.session_state["case_dict"]["scenarios"]["n_scenarios"] == 6


def test_deep_on_large_fields_spends_time_on_scenarios_not_rounding(small_case):
    small = mode_overrides(small_case, "compare", "deep", n=2_000)["layout_opt"]
    large = mode_overrides(small_case, "compare", "deep", n=12_000)["layout_opt"]
    assert small["inner_certify_rounds"] > 0 and large["inner_certify_rounds"] == 0
    assert mode_overrides(small_case, "compare", "deep", n=12_000)["scenarios"]["n_scenarios"] == 40
    assert mode_overrides(small_case, "aim", "deep", n=12_000)["aps"]["certify_rounds"] == 4
    rec = recommend(small_case, "compare", 3600.0, Calibration(), workers=8, n=12_000)
    deep = next(o for o in rec["options"] if o["mode"] == "deep")
    assert "松弛取整几乎没有收益" in deep["text"]
