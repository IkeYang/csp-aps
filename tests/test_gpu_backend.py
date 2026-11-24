"""GPU code paths, run on a strict numpy stand-in for CuPy (``fake_cupy``): same results as the CPU path,
device-side reductions, and the device assignment of pool workers."""

import multiprocessing as mp

import numpy as np
import pytest

from cspaps import backend, optics
from cspaps.aps import build_problem, build_response
from cspaps.field import build_field
from cspaps.receiver import build_grid
from cspaps.simulate import evaluate, sun_state
from cspaps.verify import verify_beams

from . import fake_cupy

TIME = "2025-03-20 10:00"


@pytest.fixture
def gpu(monkeypatch):
    def on(n_devices=2):
        monkeypatch.setenv("CSPAPS_GPU_MIN", "1")  # every size on the "GPU"
        monkeypatch.delenv("CSPAPS_GPU_DEVICE", raising=False)
        monkeypatch.delenv("CSPAPS_GPU_DEVICES", raising=False)
        return fake_cupy.install(monkeypatch, n_devices)

    return on


def _beams(case, n=None):
    fld = build_field(case)
    _, el, az, dni = sun_state(case, TIME)
    snap = evaluate(case, fld, el, az, dni)
    return case, fld, el, az, dni, snap


def test_flux_sum_and_verification_match_cpu(monkeypatch, small_case, gpu):
    case, fld, el, az, dni, snap = _beams(small_case)
    rs = build_response(case, fld, el, az)
    on = np.arange(fld.n)
    k = np.full(fld.n, rs.center_k)
    args = (rs.aim_xyz[on, k], rs.t[on, k], rs.sigma[on, k], rs.unit_beam[on, k] * dni)
    cpu = verify_beams(case.receiver, *args, 0.05)
    _, state = gpu()
    dev = verify_beams(case.receiver, *args, 0.05)
    assert state["devices_used"]  # the work ran on the (fake) device
    assert dev.status == cpu.status and dev.grid == cpu.grid
    assert dev.peak_ratio == pytest.approx(cpu.peak_ratio, rel=1e-12)
    assert (dev.hotspot_azimuth_deg, dev.hotspot_z) == (cpu.hotspot_azimuth_deg, cpu.hotspot_z)


def test_flux_on_receiver_matches_cpu(small_case, gpu):
    case, fld, el, az, dni, snap = _beams(small_case)
    grid = build_grid(case.receiver, 60, 20)
    rs = build_response(case, fld, el, az, grid)
    a, t, s, p = rs.aim_xyz[:, 0], rs.t[:, 0], rs.sigma[:, 0], rs.unit_beam[:, 0] * 900.0
    flux_c, pw_c = optics.flux_on_receiver(grid, a, t, s, p)
    gpu()
    flux_g, pw_g = optics.flux_on_receiver(grid, a, t, s, p)
    assert isinstance(flux_g, np.ndarray) and not isinstance(flux_g, fake_cupy.DeviceArray)
    np.testing.assert_allclose(flux_g, flux_c, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(pw_g, pw_c, rtol=1e-12, atol=1e-9)


def test_sparse_responses_match_cpu(small_case, gpu):
    case, fld, el, az, dni, _ = _beams(small_case)
    cpu = build_response(case, fld, el, az)
    gpu()
    dev = build_response(case, fld, el, az)
    np.testing.assert_array_equal(dev.resp_peak, cpu.resp_peak)
    assert len(dev.cells) == len(cpu.cells) == fld.n
    for i in range(fld.n):
        np.testing.assert_array_equal(dev.cells[i], cpu.cells[i])
        assert dev.resp[i].dtype == np.float32 and dev.resp[i].flags.c_contiguous
        np.testing.assert_array_equal(dev.resp[i], cpu.resp[i])
    np.testing.assert_allclose(dev.unit_power, cpu.unit_power, rtol=1e-12, atol=1e-15)


def test_sparse_responses_in_several_blocks(monkeypatch, small_case, gpu):
    case, fld, el, az, dni, _ = _beams(small_case)
    cpu = build_response(case, fld, el, az)
    gpu()
    monkeypatch.setenv("CSPAPS_GPU_BLOCK", "100000")  # minimum block: many heliostat blocks
    dev = build_response(case, fld, el, az)
    assert all(np.array_equal(a, b) for a, b in zip(dev.cells, cpu.cells))
    assert all(np.array_equal(a, b) for a, b in zip(dev.resp, cpu.resp))


def test_intercept_fraction_matches_cpu(small_case, gpu):
    case, fld, el, az, dni, _ = _beams(small_case)
    rs = build_response(case, fld, el, az)
    cpu = optics.intercept_fraction(case.receiver, rs.aim_xyz, rs.t, rs.sigma)
    band = optics.intercept_fraction(case.receiver, rs.aim_xyz, rs.t, rs.sigma, z_range=(58.0, 60.0))
    gpu()
    np.testing.assert_allclose(optics.intercept_fraction(case.receiver, rs.aim_xyz, rs.t, rs.sigma), cpu,
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(optics.intercept_fraction(case.receiver, rs.aim_xyz, rs.t, rs.sigma,
                                                         z_range=(58.0, 60.0)), band, rtol=1e-12, atol=1e-14)


def test_weighted_upper_matches_cpu(small_case, gpu):
    from cspaps.coupled import _unshaded_beams, _weighted_upper

    case, fld, el, az, dni, snap = _beams(small_case)
    case.uniformity.enabled = True
    prob = build_problem(case, fld, el, az, dni)
    beam, intercepted = _unshaded_beams(prob, fld)
    allowed = np.ones((prob.n, prob.k), bool)
    snap = evaluate(case, fld, el, az, dni, grid=prob.grid)
    cpu = _weighted_upper(prob, beam, intercepted, allowed, snap, 0.3)
    gpu()
    assert _weighted_upper(prob, beam, intercepted, allowed, snap, 0.3) == pytest.approx(cpu, rel=1e-10)


def test_scenario_solve_on_gpu_matches_cpu(small_case, gpu):
    from cspaps.layout_value import inner_settings, solve_instant

    small_case.uniformity.enabled = True
    small_case.uniformity.mode, small_case.uniformity.weight = "weighted", 0.3
    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, TIME)
    cpu = solve_instant(small_case, fld, el, az, dni, inner_settings(small_case))
    _, state = gpu()
    dev = solve_instant(small_case, fld, el, az, dni, inner_settings(small_case))
    assert len(state["devices_used"]) > 5
    assert dev.status == cpu.status
    assert dev.p_abs_w == pytest.approx(cpu.p_abs_w, rel=1e-9)
    assert dev.u_w == pytest.approx(cpu.u_w, rel=1e-6, abs=1.0)


def test_round_robin_devices_and_memory_share(monkeypatch, gpu):
    gpu(n_devices=3)
    for key in ("CSPAPS_GPU_DEVICE", "CSPAPS_GPU_SHARE"):  # worker_init sets them: restore after the test
        monkeypatch.setenv(key, "")
    counter = mp.Value("i", 0)
    seen = []
    for _ in range(7):
        backend.worker_init(counter, 7)
        seen.append(backend.device_for_this_process())
        assert backend.os.environ["CSPAPS_GPU_SHARE"] == "3"  # ceil(7 workers / 3 devices)
    assert seen == [0, 1, 2, 0, 1, 2, 0]
    monkeypatch.setenv("CSPAPS_GPU_DEVICES", "4,6")  # a subset of the machine's devices
    seen = []
    for _ in range(4):
        backend.worker_init(counter, 4)
        seen.append(int(backend.os.environ["CSPAPS_GPU_DEVICE"]))
    assert sorted(seen) == [4, 4, 6, 6]


def test_memory_share_caps_the_pool_and_blocks(monkeypatch, gpu):
    fake, state = gpu(n_devices=2)
    monkeypatch.setenv("CSPAPS_GPU_DEVICE", "1")
    monkeypatch.setenv("CSPAPS_GPU_SHARE", "4")
    with backend._device(fake) as dev:
        assert dev == 1
    assert state["limits"] == {1: pytest.approx(0.9 / 4)}
    assert backend.block_elements(fake) == 8_000_000  # default block, below the memory cap
    monkeypatch.setenv("CSPAPS_GPU_BLOCK", "1e9")
    # 8 GiB free of 11 GiB, 4 processes on the device: 0.8 * 11/4 GiB / 40 bytes per element
    assert backend.block_elements(fake) == int(0.8 * (11 << 30) / 4 / 40)


def _worker_device(_):
    import os

    return os.getpid(), os.environ.get("CSPAPS_GPU_DEVICE"), os.environ.get("CSPAPS_GPU_SHARE")


def _worker_env(_):
    import os

    return os.environ.get("CSPAPS_GPU"), os.environ.get("CSPAPS_GPU_DEVICES"), os.environ.get("CSPAPS_GPU_DEVICE")


def test_process_pool_assigns_devices(monkeypatch):
    """A real pool with the GPU backend on: every worker is initialised with a device and a share."""
    from cspaps.parallel import pool

    monkeypatch.setenv("CSPAPS_GPU", "1")
    monkeypatch.setenv("CSPAPS_GPU_DEVICES", "0,1")
    with pool(3) as p:
        rows = p.map(_worker_device, range(30), chunksize=1)
    devices = {pid: (dev, share) for pid, dev, share in rows}
    assert {d for d, _ in devices.values()} <= {"0", "1"}
    assert all(share == "2" for _, share in devices.values())


def test_workers_take_the_current_settings(monkeypatch):
    """A fork server started earlier keeps its old environment; workers must still see the parent's settings
    of the moment the pool is created (set and unset)."""
    import threading
    import time

    from cspaps.parallel import pool

    threading.Thread(target=time.sleep, args=(5,), daemon=True).start()  # multi-threaded: forkserver
    monkeypatch.setenv("CSPAPS_GPU", "1")
    monkeypatch.delenv("CSPAPS_GPU_DEVICES", raising=False)
    with pool(2) as p:  # (re)uses the fork server with CSPAPS_GPU_DEVICES unset
        p.map(_worker_env, range(4))
    monkeypatch.setenv("CSPAPS_GPU_DEVICES", "0,1")
    with pool(2) as p:
        rows = set(p.map(_worker_env, range(8), chunksize=1))
    assert all(gpu == "1" and devs == "0,1" and dev in ("0", "1") for gpu, devs, dev in rows)
    monkeypatch.setenv("CSPAPS_GPU", "0")
    with pool(2) as p:  # switched off in the parent: off in the workers, no device assigned
        rows = set(p.map(_worker_env, range(4)))
    assert all(gpu == "0" and dev is None for gpu, _, dev in rows)


@pytest.mark.parametrize("blockers", [False, True])
def test_shading_on_gpu_matches_cpu(small_case, gpu, blockers):
    from cspaps import shading, sun

    case = small_case
    fld = build_field(case)
    rec = case.receiver
    for time in ("2025-12-21 09:00", "2025-06-21 12:00"):
        _, el, az, _ = sun_state(case, time)
        s = sun.sun_vector(el, az)
        aim = np.broadcast_to(np.asarray(rec.center, float), fld.xyz.shape)
        normals, t, _, _ = optics.heliostat_geometry(fld.xyz, aim, s)
        mask = (np.arange(fld.n) % 3 != 0) if blockers else None
        cpu = shading.shading_blocking(fld.xyz, normals, s, t, case.heliostat, rec, fld.pairs, n_points=6,
                                       blockers=mask)
        with pytest.MonkeyPatch.context() as mp_:
            _, state = fake_cupy.install(mp_, 2)
            mp_.setenv("CSPAPS_GPU_MIN", "1")
            dev = shading.shading_blocking(fld.xyz, normals, s, t, case.heliostat, rec, fld.pairs, n_points=6,
                                           blockers=mask)
            assert state["devices_used"]  # the device path ran
        assert cpu["combined"].min() < 1.0  # some mirrors are shaded or blocked
        for key in ("shading", "blocking", "combined", "tower_fraction", "non_tower_occluded"):
            np.testing.assert_array_equal(dev[key], cpu[key])
