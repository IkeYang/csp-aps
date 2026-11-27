"""The sidebar's compute resources: CPU / GPU, which GPUs, how many processes (GUI, strict CuPy stand-in)."""

import os
import sys
from pathlib import Path

import pytest

from . import fake_cupy
from .conftest import ROOT, small_case_dict

GUI = str(Path(ROOT) / "src" / "cspaps" / "gui_app.py")
KEYS = ("CSPAPS_GPU", "CSPAPS_GPUS", "CSPAPS_GPU_DEVICES", "CSPAPS_WORKERS", "CSPAPS_GPU_SEARCH", "CSPAPS_GPU_DEVICE")


@pytest.fixture
def app(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    for key in KEYS:  # the app sets them: restore after the test
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])

    def make():
        at = AppTest.from_file(GUI, default_timeout=120)
        at.session_state["case_dict"] = small_case_dict()
        at.session_state["base_dir"] = str(ROOT)
        at.session_state["ver"] = 1
        return at

    return make


def _captions(at) -> str:
    return "\n".join(c.value for c in at.sidebar.caption)


def test_gpus_are_chosen_in_the_sidebar(monkeypatch, app):
    fake_cupy.install(monkeypatch, 4)  # 4 visible GPUs, GPU backend on
    at = app()
    at.run()
    assert not at.exception
    assert at.sidebar.radio(key="compute_device").value == "gpu"
    pick = at.sidebar.multiselect(key="gpu_ids")
    assert pick.value == [0, 1, 2, 3] and pick.format_func(0) == "GPU 0"
    assert "检测到 NVIDIA GPU 4 张" in _captions(at) and "4 × RTX 2080 Ti · 11 GB" in _captions(at)
    pick.set_value([1, 3]).run()
    assert os.environ["CSPAPS_GPU_DEVICES"] == "1,3" and os.environ["CSPAPS_GPU"] == "1"
    assert "将使用：GPU 1、3（2 张）" in _captions(at)
    at.sidebar.checkbox(key="gpu_search").uncheck().run()
    assert os.environ["CSPAPS_GPU_SEARCH"] == "0"
    at.sidebar.radio(key="compute_device").set_value("cpu").run()
    assert os.environ["CSPAPS_GPU"] == "0" and "将使用：仅 CPU" in _captions(at)
    at.sidebar.radio(key="compute_device").set_value("gpu").run()  # back on: every GPU preselected
    assert at.sidebar.multiselect(key="gpu_ids").value == [0, 1, 2, 3] and os.environ["CSPAPS_GPU"] == "1"


def test_worker_count_is_set_in_the_sidebar(monkeypatch, app):
    fake_cupy.install(monkeypatch, 2)
    at = app()
    at.run()
    box = next(n for n in at.sidebar.number_input if n.label.startswith("并行进程数"))
    box.set_value(2).run()
    assert at.session_state["case_dict"]["layout_opt"]["workers"] == 2
    assert at.session_state["case_dict"]["annual"]["workers"] == 2
    assert os.environ["CSPAPS_WORKERS"] == "2"  # the depths use it too
    from cspaps.advisor import recommended_workers

    assert recommended_workers() == 2
    assert "CPU 2 个进程" in _captions(at)


def test_without_cupy_the_sidebar_says_why(monkeypatch, app):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    at = app()
    at.run()
    assert not at.exception
    text = _captions(at)
    assert "检测到 NVIDIA GPU 0 张" in text and "没有安装 CuPy" in text and "将使用：仅 CPU" in text
    assert all(r.key != "compute_device" for r in at.sidebar.radio)  # no GPU choice to make


def test_gui_command_takes_gpus(monkeypatch):
    from cspaps import backend
    from cspaps.cli import build_parser

    args = build_parser().parse_args(["gui", "--gpus", "0,2", "--workers", "6"])
    assert args.gpus == "0,2" and args.workers == 6
    for key in KEYS:
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    backend.set_resources([0, 2])
    assert os.environ["CSPAPS_GPU_DEVICES"] == "0,2" and os.environ["CSPAPS_GPU"] == "1"


def test_the_sidebar_states_the_memory_per_process_for_this_field(monkeypatch, app):
    """The per-process memory follows the loaded field (not a flat 1.5 GB), and the process cap uses it."""
    from cspaps import parallel
    from cspaps.aps import CERTIFICATE_BYTES_PER_CANDIDATE, candidate_count, problem_memory_bytes
    from cspaps.config import case_from_dict
    from cspaps.field import build_field

    case = case_from_dict(small_case_dict(), base_dir=ROOT)
    n, k = build_field(case).n, candidate_count(case.aps)
    per = problem_memory_bytes(n, k)
    gb = 2 ** 30
    monkeypatch.delenv("CSPAPS_WORKER_MEM_GB", raising=False)
    monkeypatch.setattr(parallel, "memory_bytes", lambda: (int(2.4 * gb + 2 * per + per // 2), 16 * gb))
    monkeypatch.setattr(parallel, "cpu_count", lambda: 8)  # any machine: room for 6 processes in the box
    at = app()
    at.run()
    box = next(x for x in at.sidebar.number_input if x.label.startswith("并行进程数"))
    box.set_value(6).run()
    assert not at.exception
    text = _captions(at)
    assert f"当前镜场 {n:,} 面：每个进程求解时约占 {per / gb:.1f} GB" in text
    assert f"最优性证书另需约 {n * k * CERTIFICATE_BYTES_PER_CANDIDATE / gb:.0f} GB" in text
    assert f"最多同时启动 2 个进程（每个按 {per / gb:.1f} GB 估计" in text
