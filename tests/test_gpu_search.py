"""The GPU local search kernel (``gpu_search``): its code run on the CPU (g++) must take the same moves as the
compiled CPU kernel, and the launch path must marshal the arguments correctly (strict CuPy stand-in)."""

import shutil

import numpy as np
import pytest

from cspaps import gpu_search, kernels
from cspaps.uniformity import _State, _local_search, region_of_grid

from . import fake_cupy
from .test_kernels import _problem

pytestmark = [pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++ to run the kernel code on the CPU"),
              pytest.mark.skipif(not kernels.enabled(), reason="compares with the Numba kernel")]


def _pair(prob, start, mu, passes, move_cost=None, nt=gpu_search.THREADS):
    h = region_of_grid(prob.case, prob.grid, prob.case.uniformity)
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")
    ap = prob.absorbed_power
    out = []
    for run in ("cpu", "gpu"):
        prob.flux_limited = 0
        st = _State(prob, h, start)
        if run == "cpu":
            kernels.local_search(prob, st, mu, passes, ap, order, move_cost)
        else:
            gpu_search.emulate(prob, st, mu, passes, ap, order, move_cost, nt=nt)
        out.append((st, prob.flux_limited))
    (a, la), (b, lb) = out
    assert np.array_equal(a.choice, b.choice)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.flux_con, b.flux_con)  # per-cell state: same arithmetic
    assert b.s1 == pytest.approx(a.s1, rel=1e-12) and b.s2 == pytest.approx(a.s2, rel=1e-12)  # sums: other order
    assert la == lb
    return b, lb


@pytest.mark.parametrize("mu", [0.0, 0.3, 5.0])
def test_gpu_kernel_code_takes_the_same_moves(small_case, mu):
    prob, start, _ = _problem(small_case)
    _pair(prob, start, mu, passes=6)


@pytest.mark.parametrize("nt", [32, 64, 256])
def test_gpu_kernel_code_with_binding_limits_reach_and_moves(small_case, nt):
    small_case.aps.move_penalty_per_m = 0.02
    prob, start, _ = _problem(small_case, "2025-12-21 10:00")
    peak = prob.flux(start).max()
    prob.rs.limit_nom = np.full_like(prob.rs.limit_nom, peak / (1.0 - prob.margin) * 1.0001)
    rng = np.random.default_rng(0)
    prob.reach = rng.random((prob.n, prob.k)) < 0.7
    prob.reach[np.arange(prob.n), np.where(start >= 0, start, prob.center_k)] = True
    prev = prob.aim_xyz[np.arange(prob.n), rng.integers(0, prob.k, prob.n)]
    p_ref = prob.absorbed_power[:, prob.center_k]
    move = np.linalg.norm(prob.aim_xyz - prev[:, None, :], axis=2)
    cost = (small_case.aps.move_penalty_per_m * p_ref[:, None]) * move * (rng.random(prob.n) < 0.8)[:, None]
    _, limited = _pair(prob, start, 0.8, passes=8, move_cost=cost, nt=nt)
    assert limited > 0


def test_gpu_kernel_code_after_hotspot_points(small_case):
    prob, start, _ = _problem(small_case)
    rec = small_case.receiver
    prob.rs.add_points(np.linspace(0, 350, 12), np.full(12, rec.center[2] + 0.3 * rec.height))
    _pair(prob, start, 0.5, passes=4)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setenv("CSPAPS_GPU_MIN", "1")
    monkeypatch.delenv("CSPAPS_GPU_DEVICE", raising=False)
    monkeypatch.delenv("CSPAPS_GPU_DEVICES", raising=False)
    monkeypatch.setattr(gpu_search, "_FAILED", [])
    monkeypatch.setattr(gpu_search, "_KERNELS", {})
    return fake_cupy.install(monkeypatch, 2)


def test_local_search_runs_on_the_gpu_path(monkeypatch, small_case, gpu):
    """Through ``uniformity._local_search`` with the GPU backend: one block of THREADS, the kernel's shared
    memory, the device of the process, and the same plan as the CPU kernel."""
    prob, start, _ = _problem(small_case)
    h = region_of_grid(prob.case, prob.grid, prob.case.uniformity)
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")
    ref = _State(prob, h, start)
    kernels.local_search(prob, ref, 0.3, 6, prob.absorbed_power, order, None)
    fake, state = gpu
    monkeypatch.setenv("CSPAPS_GPU_DEVICE", "1")
    st = _State(prob, h, start)
    _local_search(prob, st, 0.3, 6, prob.absorbed_power, order)
    launch = state["launches"][-1]
    assert launch["grid"] == (1,) and launch["block"] == (gpu_search.THREADS,) and launch["device"] == 1
    assert launch["shared_mem"] == gpu_search.shared_bytes(prob.k) and "--fmad=false" in launch["options"]
    assert np.array_equal(st.choice, ref.choice) and np.array_equal(st.q, ref.q)
    assert st.s1 == pytest.approx(ref.s1, rel=1e-12)


def test_gpu_failure_falls_back_to_the_cpu_kernel(monkeypatch, small_case, gpu):
    prob, start, _ = _problem(small_case)
    h = region_of_grid(prob.case, prob.grid, prob.case.uniformity)
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")
    ref = _State(prob, h, start)
    kernels.local_search(prob, ref, 0.3, 4, prob.absorbed_power, order, None)
    fake, state = gpu
    state["raw_kernel_error"] = "nvrtc: compilation failed"
    st = _State(prob, h, start)
    with pytest.warns(RuntimeWarning, match="GPU local search unavailable"):
        _local_search(prob, st, 0.3, 4, prob.absorbed_power, order)
    assert np.array_equal(st.choice, ref.choice) and st.s1 == ref.s1  # the CPU kernel ran: bit-identical
    assert not gpu_search.enabled()  # no further GPU attempts in this process
    state.pop("raw_kernel_error")


def test_gpu_search_can_be_switched_off(monkeypatch, gpu):
    assert gpu_search.enabled()
    monkeypatch.setenv("CSPAPS_GPU_SEARCH", "0")
    assert not gpu_search.enabled()
