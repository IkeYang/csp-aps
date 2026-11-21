"""NASA POWER parsing (offline), packaged demo case, and GUI input guards."""

import filecmp
from pathlib import Path

import pandas as pd
import pytest

from cspaps import load_case
from cspaps.resources import DATA, demo_case_path
from cspaps.weather import parse_power_csv

from .conftest import ROOT, small_case_dict

PAYLOAD = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Hourly Data
Location: Latitude  39.6318   Longitude 94.3389
Parameter(s):
ALLSKY_SFC_SW_DNI     CERES SYN1deg All Sky Surface Shortwave Downward Direct Normal Irradiance (Wh/m^2)
-END HEADER-
YEAR,MO,DY,HR,ALLSKY_SFC_SW_DNI
2023,12,31,15,0.0
2023,12,31,16,0.0
2024,1,1,0,0.0
2024,1,1,1,166.57
2024,1,1,2,-999
2024,12,31,15,12.5
2024,12,31,16,9.0
"""


def test_power_payload_is_converted_to_local_hour_midpoints():
    df, info = parse_power_csv(PAYLOAD, "Asia/Shanghai", 2024)
    t = dict(zip(df["time"], df["dni"]))
    # UTC 2024-01-01 01:00 (hour start) -> 01:30 UTC -> 09:30 China Standard Time
    assert t["2024-01-01 09:30:00"] == pytest.approx(166.57)
    assert t["2024-01-01 00:30:00"] == 0.0  # UTC 2023-12-31 16:00 belongs to the local year 2024
    assert "2025-01-01 00:30:00" not in t  # UTC 2024-12-31 16:00 is already local 2025
    assert t["2024-01-01 10:30:00"] == 0.0 and info["missing_hours"] == 1
    assert info["n_hours"] == 5


def test_packaged_demo_case_matches_repository_and_loads():
    for name in ("aksai_demo.yaml", "aksai_2024_dni.csv"):
        assert filecmp.cmp(DATA / name, ROOT / "cases" / name, shallow=False), f"{name} packaged copy is stale"
    case = load_case(DATA / "aksai_demo.yaml")
    assert case.weather.source == "file" and (DATA / case.weather.file).exists()
    assert demo_case_path().exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "data/*.yaml" in pyproject and "data/*.csv" in pyproject


def test_invalid_rectangle_blocks_the_layout_run(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    d = small_case_dict()
    d["field"]["boundary"] = [[-150, -150], [150, -150], [150, 150], [-150, 150]]
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=120)
    at.session_state["case_dict"] = d
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("布局优化").run()
    at.number_input(key="1:rect.x0").set_value(400.0).run()  # x_min > x_max
    btn = at.button(key="btn_opt")
    assert btn.disabled
    assert any("场地边界无效" in e.value for e in at.error)
    assert at.session_state["case_dict"]["field"]["boundary"][1][0] == 150  # the old boundary was not replaced
    assert "res_opt" not in at.session_state


def test_upload_dni_without_file_blocks_the_aps_run(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=120)
    at.session_state["case_dict"] = small_case_dict()
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("瞄准优化").run()
    at.radio(key="aps_dni_src").set_value("upload").run()
    assert at.button(key="btn_single").disabled
    assert any("请上传当前镜场 DNI 分布" in e.value for e in at.error)
    at.radio(key="aps_dni_src").set_value("site").run()
    assert not at.button(key="btn_single").disabled


def test_site_change_warns_about_the_weather_file(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    import yaml
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load((ROOT / "cases" / "aksai_demo.yaml").read_text(encoding="utf-8"))
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=120)
    at.session_state["case_dict"] = data
    at.session_state["base_dir"] = str(ROOT / "cases")
    at.session_state["ver"] = 1
    at.session_state["weather_site"] = (data["site"]["latitude"], data["site"]["longitude"], data["weather"]["file"])
    at.run()
    at.radio(key="page").set_value("布局优化").run()
    assert not any("站址已修改" in m.value for m in at.markdown)
    at.number_input(key="1:site.latitude").set_value(30.0).run()
    assert any("站址已修改" in m.value for m in at.markdown)
    at.button(key="clear_layout").click().run()
    assert at.session_state["case_dict"]["weather"]["source"] == "clear_sky"


def test_daylight_saving_weather_file_loads(tmp_path):
    """Adelaide (DST): the repeated 02:00 at the April fall-back and the skipped hour in October."""
    import pandas as pd

    from cspaps import case_from_dict, sun

    times = pd.date_range("2024-04-06 22:00", "2024-04-07 06:00", freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()
    times += pd.date_range("2024-10-06 00:00", "2024-10-06 05:00", freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()
    path = tmp_path / "adelaide.csv"
    pd.DataFrame({"time": times, "dni": 0.0}).to_csv(path, index=False)
    case = case_from_dict({"site": {"latitude": -32.5, "longitude": 137.8, "timezone": "Australia/Adelaide"},
                           "receiver": {"center": [0, 0, 60], "diameter": 6, "height": 6},
                           "heliostat": {"width": 6, "height": 6, "pivot_height": 4},
                           "weather": {"source": "file", "file": str(path)}, "annual": {"year": 2024}},
                          base_dir=tmp_path)
    series = sun.load_weather(case)
    assert len(series) == len(times) and series.index.is_monotonic_increasing
    idx = sun.to_times(["2024-10-06 02:30"], "Australia/Adelaide")  # does not exist: shifted forward
    assert idx[0].hour == 3
