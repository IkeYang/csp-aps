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
    assert not at.exception
    at.button(key="btn_rt").click().run()  # replay in the operations workspace
    assert not at.exception and not at.error
    cards = " ".join(m.value for m in at.markdown)
    for label in ("指令状态", "最高通量（复核）", "距上限余量", "散焦镜数", "数据新鲜度"):
        assert label in cards
    at.radio(key="page").set_value("布局规划").run()
    at.button(key="btn_opt").click().run()
    at.button(key="apply_layout").click().run()
    chk = at.session_state["apply_check"]
    assert chk["导出镜数"] == chk["重新载入镜数"] == 120 and chk["编号与坐标逐项一致"]
    at.radio(key="page").set_value("光学校验").run()
    assert not at.exception
    at.radio(key="page").set_value("运行记录").run()
    assert not at.exception
