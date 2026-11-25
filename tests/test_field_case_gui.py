"""Field-test feedback (cases/aksai_field_test_2026): loading a case with its files, the start buttons, the demo
mark, and the receiver flux along the power-uniformity trade-off curve."""

import io
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from cspaps import case_from_dict, load_case

from .conftest import ROOT, small_case_dict

FIELD = Path(ROOT) / "cases" / "aksai_field_test_2026"
GUI = str(Path(ROOT) / "src" / "cspaps" / "gui_app.py")
needs_field = pytest.mark.skipif(not (FIELD / "case.yaml").is_file(), reason="field test dataset not present")


class _Upload:
    def __init__(self, path: Path, name: str | None = None, data: bytes | None = None):
        self.name = name or path.name
        self._data = data if data is not None else path.read_bytes()
        self.size = len(self._data)

    def getvalue(self):
        return self._data


# ----------------------------------------------------------------------------- referenced files

@needs_field
def test_case_lists_and_checks_its_files(tmp_path):
    case = load_case(FIELD / "case.yaml")
    kinds = {k for k, _, _ in case.referenced_files()}
    assert kinds == {"field.layout_file", "weather.file"} and not case.missing_files()
    alone = tmp_path / "case.yaml"
    shutil.copy(FIELD / "case.yaml", alone)  # the YAML without its CSVs: what a lone upload produced
    lost = load_case(alone)
    assert {Path(v).name for _, v, _ in lost.missing_files()} == {"layout_full.csv", "nasa_power_dni_2026.csv"}
    from cspaps.field import build_field

    with pytest.raises(FileNotFoundError, match="relative to the YAML's own folder"):
        build_field(lost)


def test_unused_weather_file_is_not_required():
    d = small_case_dict()
    d["weather"] = {"source": "clear_sky", "file": "nowhere.csv"}
    assert case_from_dict(d, base_dir=ROOT).missing_files() == []


# ----------------------------------------------------------------------------- uploads (helpers of the loader)

@pytest.fixture
def gui(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    from cspaps import gui_app

    monkeypatch.setattr(gui_app, "UPLOADS", tmp_path / "_uploads")
    return gui_app


@needs_field
def test_upload_yaml_with_its_csvs_finds_everything(gui):
    files = [_Upload(FIELD / n) for n in ("case.yaml", "layout_full.csv", "nasa_power_dni_2026.csv")]
    folder = gui._save_case_upload(files)
    target = gui._case_in(folder)
    case = load_case(target)
    assert target.name == "case.yaml" and not case.missing_files()
    from cspaps.field import build_field

    assert build_field(case).n > 11000  # the real field, not the 473-mirror demo


@needs_field
def test_upload_zip_of_the_case_folder(gui, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in ("case.yaml", "layout_full.csv", "nasa_power_dni_2026.csv"):
            zf.write(FIELD / n, f"aksai_field_test_2026/{n}")
        zf.writestr("../escape.txt", "outside")  # must not leave the upload folder
    folder = gui._save_case_upload([_Upload(tmp_path / "x.zip", "aksai.zip", buf.getvalue())])
    target = gui._case_in(folder)
    assert target.parent.name == "aksai_field_test_2026" and not load_case(target).missing_files()
    assert not (folder.parent / "escape.txt").exists()


def _app(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    return AppTest.from_file(GUI, default_timeout=300)


@needs_field
def test_lone_yaml_upload_names_the_missing_files(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    folder = tmp_path / "results" / "gui" / "_uploads" / "case_abc"
    folder.mkdir(parents=True)
    shutil.copy(FIELD / "case.yaml", folder / "case.yaml")
    at = _app(monkeypatch, tmp_path)
    at.session_state["case_dict"] = yaml.safe_load((folder / "case.yaml").read_text(encoding="utf-8"))
    at.session_state["base_dir"] = str(folder)
    at.session_state["case_source"] = {"kind": "upload", "path": str(folder / "case.yaml")}
    at.session_state["ver"] = 1
    at.session_state["page"] = "参数设置"
    at.run()
    msg = " ".join(e.value for e in at.error)
    assert "layout_full.csv" in msg and "nasa_power_dni_2026.csv" in msg and "相对路径按 YAML 所在的文件夹" in msg


@needs_field
def test_open_case_by_path_keeps_its_files_and_clears_the_demo_mark(monkeypatch, tmp_path):
    """Issue 3: the demo mark followed a sticky flag; it must follow the loaded case."""
    pytest.importorskip("streamlit")
    at = _app(monkeypatch, tmp_path)
    at.run()
    at.button(key="home_aps_demo").click().run()  # the demo: marked
    assert at.session_state["demo_data"] and at.session_state["case_dict"]["name"] == "aksai_demo"
    at.sidebar.text_input(key="case_path").set_value(str(FIELD / "case.yaml")).run()
    at.sidebar.button(key="open_case").click().run()
    assert not at.exception
    assert at.session_state["case_dict"]["name"] == "aksai_field_test_2026"
    assert Path(at.session_state["base_dir"]) == FIELD and not at.session_state["demo_data"]
    assert not any("找不到" in e.value for e in at.error)
    at.radio(key="page").set_value("总览").run()
    note = " ".join(c.value for c in at.caption)
    assert "layout_full.csv" in note and "nasa_power_dni_2026.csv" in note and "演示" not in note


@needs_field
def test_start_buttons_keep_the_loaded_case(monkeypatch, tmp_path):
    """Issue 2: "用示例试算" replaced the loaded case with the demo; the loaded case now comes first."""
    pytest.importorskip("streamlit")
    at = _app(monkeypatch, tmp_path)
    at.session_state["case_dict"] = yaml.safe_load((FIELD / "case.yaml").read_text(encoding="utf-8"))
    at.session_state["base_dir"] = str(FIELD)
    at.session_state["case_source"] = {"kind": "folder", "path": str(FIELD / "case.yaml")}
    at.session_state["ver"] = 1
    at.run()
    start = at.button(key="home_aps_current")
    assert "aksai_field_test_2026" in start.label
    start.click().run()
    assert at.session_state["page"] == "瞄准优化"
    assert at.session_state["case_dict"]["name"] == "aksai_field_test_2026"  # nothing replaced
    at.radio(key="page").set_value("总览").run()
    at.button(key="home_aps_demo").click().run()  # the demo only on request, and marked as such
    assert at.session_state["case_dict"]["name"] == "aksai_demo" and at.session_state["demo_data"]


# ----------------------------------------------------------------------------- trade-off curve: receiver flux

def test_tradeoff_curve_keeps_the_receiver_flux_of_every_plan(small_case, tmp_path):
    from cspaps import workflows as wf
    from cspaps.receiver import build_grid

    small_case.receiver.n_panels = 4
    res = wf.task_pareto(small_case, tmp_path, "2025-06-21 13:00", lambdas=[0.0, 1.0], certify=False)
    flux, rows, panels = res["flux"], res["summary"]["curve"], res["summary"]["panels"]
    grid = build_grid(small_case.receiver)
    assert flux["incident_kw_m2"].shape == (2, grid.z.size, grid.azimuth_deg.size)
    assert len(panels) == 2 * 4 and {r["panel"] for r in panels} == {1, 2, 3, 4}
    for k, lam in enumerate(flux["lambdas"]):
        mine = [r for r in panels if r["lambda"] == lam]
        # conservation: the panels add up to the absorbed power on the map
        total = float(flux["absorbed_kw_m2"][k].sum()) * 1e3 * grid.cell_area / 1e6
        assert sum(r["absorbed_mw"] for r in mine) == pytest.approx(total, rel=1e-9)
        assert max(r["peak_kw_m2"] for r in mine) == pytest.approx(float(flux["incident_kw_m2"][k].max()))
    assert rows[0]["hotspot_panel"] in {1, 2, 3, 4}
    assert res["files"]["flux_npz"].is_file() and res["files"]["panels_csv"].is_file()
    assert np.load(res["files"]["flux_npz"])["incident_kw_m2"].shape[0] == 2


def test_tradeoff_flux_views_render(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    at = _app(monkeypatch, tmp_path)
    at.session_state["case_dict"] = small_case_dict()
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("瞄准优化").run()
    at.text_input(key="pareto_lams").set_value("0, 0.5").run()
    at.button(key="btn_pareto").click().run()
    assert not at.exception
    keys = {c.key for c in at.get("plotly_chart")} if at.get("plotly_chart") else set()
    assert {"pareto_flux_grid", "pareto_flux_map", "pareto_panel_bars", "pareto_profiles"} <= keys
    at.select_slider(key="pareto_pick").set_value(0.5).run()
    assert not at.exception


# ----------------------------------------------------------------------------- colour scales that do not exaggerate

def test_colour_range_does_not_stretch_tiny_differences():
    from cspaps.gui_plots import DNI_MIN_SPAN, span_range

    lo, hi, widened = span_range([71.0, 71.2, 71.5], min_span=DNI_MIN_SPAN, lower=0.0)
    assert widened and hi - lo == pytest.approx(DNI_MIN_SPAN) and lo < 71.0 < 71.5 < hi
    assert span_range([300.0, 900.0], min_span=DNI_MIN_SPAN, lower=0.0) == (300.0, 900.0, False)
    assert span_range([5.0, 6.0], min_span=DNI_MIN_SPAN, lower=0.0)[:2] == (0.0, DNI_MIN_SPAN)  # not below 0
    assert span_range([71.0, 71.5], cmin=0.0, cmax=1000.0, min_span=DNI_MIN_SPAN) == (0.0, 1000.0, False)


def test_dni_colour_bar_states_the_actual_range(small_case):
    from cspaps.field import build_field
    from cspaps.gui_plots import field_dni_map

    fld = build_field(small_case)
    dni = 71.0 + 0.5 * np.random.default_rng(0).random(fld.n)
    fig = field_dni_map(small_case, fld.xyz, fld.ids, dni)
    m = [t for t in fig.data if getattr(t, "name", "") == "定日镜"][0].marker
    assert m.cmax - m.cmin >= 100.0 - 1e-9
    assert f"实际 {dni.min():.2f}–{dni.max():.2f}" in m.colorbar.title.text  # the real (small) spread is stated


def test_tradeoff_colour_modes_share_one_scale():
    from cspaps.gui_plots import tradeoff_scale

    maps = np.stack([np.full((3, 4), 500.0), np.full((3, 4), 505.0), np.full((3, 4), 650.0)])
    maps[:, 0, 0] = 0.0  # an unlit cell
    flux = {"incident_kw_m2": maps, "lambdas": np.array([0.0, 0.5, 1.0])}
    d, lo, hi, _, unit = tradeoff_scale(flux, "diff", ref=0)
    assert np.allclose(d[0], 0) and lo == -hi == pytest.approx(-150.0)  # one limit for every lambda
    assert d[1].max() == pytest.approx(5.0)  # the small change keeps its real size: pale, not saturated
    assert "λ = 0" in unit
    assert tradeoff_scale(flux, "shared")[1:3] == (0.0, 650.0)
    lo_c = tradeoff_scale(flux, "contrast")[1]
    assert 0.0 < lo_c <= 500.0  # from the lit cells, not from the dark one


def test_tradeoff_colour_option_beside_the_maps(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    at = _app(monkeypatch, tmp_path)
    at.session_state["case_dict"] = small_case_dict()
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("瞄准优化").run()
    at.text_input(key="pareto_lams").set_value("0, 0.5, 1").run()
    at.button(key="btn_pareto").click().run()
    at.radio(key="pareto_scale").set_value("diff").run()
    assert not at.exception
    at.selectbox(key="pareto_ref").set_value(0.5).run()
    at.radio(key="pareto_scale").set_value("contrast").run()
    assert not at.exception
