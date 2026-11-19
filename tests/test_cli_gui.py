import json
from pathlib import Path

import pytest
import yaml

from cspaps.cli import main

from .conftest import small_case_dict


@pytest.fixture
def case_file(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(small_case_dict()), encoding="utf-8")
    return path


def test_cli_layout_snapshot_aps(case_file, tmp_path):
    out = tmp_path / "out"
    assert main(["layout", str(case_file), "--out", str(out)]) == 0
    assert (out / "tiny" / "layout.csv").exists()
    assert main(["snapshot", str(case_file), "--time", "2025-06-21 13:40", "--out", str(out)]) == 0
    data = json.loads((out / "tiny" / "snapshot_20250621_1340_center.json").read_text())
    assert data["n_heliostats"] > 100
    assert main(["aps", str(case_file), "--time", "2025-06-21 13:40", "--out", str(out)]) == 0
    aims = out / "tiny" / "aps_20250621_1340_greedy_aims.csv"
    assert aims.exists()
    # warm start from the previous aims, with a measured DNI map
    dni_map = tmp_path / "dni.csv"
    dni_map.write_text("x,y,dni\n-200,-200,600\n200,-200,600\n-200,200,900\n200,200,900\n")
    assert main(["aps", str(case_file), "--time", "2025-06-21 13:45", "--prev-aims", str(aims),
                 "--dni-map", str(dni_map), "--out", str(out)]) == 0


def test_cli_sunbins_and_validate(case_file, tmp_path):
    out = tmp_path / "out"
    assert main(["sunbins", str(case_file), "--bins", "8", "--out", str(out)]) == 0
    assert (out / "tiny" / "sun_bins.csv").exists()
    assert main(["validate", str(case_file), "--time", "2025-06-21 13:40", "--rays", "200", "--out", str(out)]) == 0
    assert (out / "tiny" / "validate_20250621_1340_flux_compare.png").exists()


def test_cli_annual(case_file, tmp_path):
    out = tmp_path / "out"
    assert main(["annual", str(case_file), "--method", "sun_bins", "--out", str(out)]) == 0
    data = json.loads((out / "tiny" / "annual_sun_bins_center.json").read_text())
    assert data["energy_mwh"]["intercept"] > 0


def test_gui_loads_and_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "cspaps" / "gui_app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 8
    at.button(key="btn_layout").click().run()
    assert not at.exception and not at.error
    assert any(m.label == "镜子数量" for m in at.metric)
