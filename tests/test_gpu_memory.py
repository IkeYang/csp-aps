"""Device memory (strict CuPy stand-in with a memory limit).

Field report: the layout search (differential evolution, centre-aimed annual energy of every candidate layout)
stopped with ``cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 4,716,459,008 bytes (allocated so far:
17,434,886,144 bytes, limit set to: 19,609,429,606)`` in the shading / blocking test. A low sun over a large
candidate field gives hundreds of millions of neighbour pairs (the search radius grows as 1 / tan(elevation)), and
the device code held several (pairs, 3) float64 arrays at once. The pairs now stream through in blocks, and any
GPU step that still runs out of device memory is done on the CPU instead of ending the calculation."""

import warnings

import numpy as np
import pytest

from cspaps import backend, optics, shading, sun
from cspaps.config import Heliostat, Receiver

from . import fake_cupy


def _dense_field(n_side=40, spacing=12.0):
    """A square grid of 8 m mirrors north of the tower, 0-6 m of relief."""
    g = np.arange(n_side) * spacing
    x, y = np.meshgrid(g - g.mean(), g + 150.0)
    z = 5.0 + 6.0 * (y - y.min()) / np.ptp(y)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _geometry(xyz, el, az):
    rec = Receiver(center=(0.0, 0.0, 100.0), diameter=10.0, height=12.0)
    hel = Heliostat(width=8.0, height=8.0, pivot_height=5.0)
    s = sun.sun_vector(el, az)
    normals, t, _, _ = optics.heliostat_geometry(xyz, np.broadcast_to(np.asarray(rec.center), xyz.shape), s)
    return rec, hel, s, normals, t


@pytest.fixture
def small_device(monkeypatch):
    """A device whose largest successful allocation is ``max_bytes`` (set by the test), smallest blocks."""
    _, state = fake_cupy.install(monkeypatch, 1)
    monkeypatch.setenv("CSPAPS_GPU_MIN", "1")
    monkeypatch.setenv("CSPAPS_GPU_BLOCK", "100000")
    monkeypatch.setattr(backend, "OOM_FALLBACKS", [])
    return state


def test_low_sun_shading_streams_its_pairs_within_the_device_memory(small_device):
    xyz = _dense_field()
    rec, hel, s, normals, t = _geometry(xyz, 3.0, 150.0)
    pairs = shading.PairCache(xyz)
    i, _ = pairs.get(shading.interaction_radius(hel, 3.0, float(np.ptp(xyz[:, 2]))))
    small_device["max_bytes"] = 4 * 2 ** 20
    assert i.size * 3 * 8 > 4 * small_device["max_bytes"]  # one (pairs, 3) float64 array would not fit
    dev = shading.shading_blocking(xyz, normals, s, t, hel, rec, pairs, n_points=4)
    assert small_device["devices_used"] and small_device["oom"] == 0 and not backend.OOM_FALLBACKS
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CSPAPS_GPU", "0")
        cpu = shading.shading_blocking(xyz, normals, s, t, hel, rec, pairs, n_points=4)
    assert cpu["combined"].min() < 0.9  # a low sun: much shading
    for key in ("shading", "blocking", "combined", "tower_fraction", "non_tower_occluded"):
        np.testing.assert_array_equal(dev[key], cpu[key])


def test_a_gpu_step_that_runs_out_of_device_memory_is_done_on_the_cpu(small_device):
    xyz = _dense_field(20)
    rec, hel, s, normals, t = _geometry(xyz, 10.0, 150.0)
    small_device["max_bytes"] = 1000  # nothing fits
    with pytest.warns(RuntimeWarning, match="GPU memory full"):
        dev = shading.shading_blocking(xyz, normals, s, t, hel, rec, shading.PairCache(xyz), n_points=4)
    assert small_device["oom"] >= 1 and small_device["freed"] >= 1 and backend.OOM_FALLBACKS
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CSPAPS_GPU", "0")
        cpu = shading.shading_blocking(xyz, normals, s, t, hel, rec, shading.PairCache(xyz), n_points=4)
    for key in ("shading", "blocking", "combined"):
        np.testing.assert_array_equal(dev[key], cpu[key])
    # every GPU entry point: None = "use the CPU path"
    pts = np.column_stack([np.zeros(50), np.full(50, 5.0), np.linspace(94, 106, 50)])
    nrm = np.tile([0.0, 1.0, 0.0], (50, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert backend.flux_sum_gpu(pts, nrm, pts[:10], np.tile([0.0, -1.0, 0.0], (10, 1)), np.ones(10),
                                    np.ones(10)) is None
    # other errors are not hidden
    small_device["max_bytes"] = None
    with pytest.raises(TypeError):
        backend.flux_sum_gpu(pts, nrm, pts[:10], None, np.ones(10), np.ones(10))


def test_the_pair_cache_is_compact_and_exact():
    xyz = _dense_field(25)
    pc = shading.PairCache(xyz)
    i, j = pc.get(60.0)
    assert i.dtype == np.int32 and j.dtype == np.int32
    d = np.hypot(*(xyz[:, None, :2] - xyz[None, :, :2]).transpose(2, 0, 1))
    want = {(a, b) for a, b in zip(*np.nonzero((d <= 60.0) & ~np.eye(len(xyz), dtype=bool)))}
    assert set(zip(i.tolist(), j.tolist())) == want
    i2, j2 = pc.get(30.0)  # from the cache
    want2 = {(a, b) for a, b in zip(*np.nonzero((d <= 30.0) & ~np.eye(len(xyz), dtype=bool)))}
    assert set(zip(i2.tolist(), j2.tolist())) == want2
