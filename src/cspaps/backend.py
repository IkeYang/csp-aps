"""Optional GPU backend (CuPy) for the dense spot kernel and the reductions that follow it.

``optics.footprints_at`` -- spot responses of the aim candidates, fine-grid verification, flux maps, joint-pose
evaluations -- is a dense exp over beams x surface points. Every caller reduces the dense block right away (sum
over beams, sum per beam, support cells above a cutoff, a weighted sum), so on the GPU the block is formed *and
reduced* on the device and only the small result crosses the bus:

* ``flux_sum_gpu``             superposed flux at the points (fine-grid verification, ``verify``)
* ``flux_and_power_gpu``       flux at the cells and power per beam (``optics.flux_on_receiver``, ``evaluate``)
* ``sparse_responses_gpu``     peak, support cells and float32 support values per heliostat (``aps.build_response``)
* ``row_stats_gpu``            sum and weighted sum per beam (``coupled._weighted_upper``)
* ``footprints_gpu``           the dense block itself (other callers)

Switch on with ``CSPAPS_GPU=1`` (CuPy installed: ``pip install -e ".[gpu]"``). Dense blocks below
``CSPAPS_GPU_MIN`` elements (default 2 million), reduced ones below an eighth of that, stay on the CPU. Blocks are sized to ``CSPAPS_GPU_BLOCK`` elements (default 8
million, ~40 bytes each while a block is formed) and to the memory share of the process.

Several GPUs: every process of a process pool gets its own device, round-robin over ``CSPAPS_GPU_DEVICES``
(default: all visible devices; ``parallel.pool`` assigns them in its worker initializer), and the processes
sharing one device split its memory (``CSPAPS_GPU_SHARE``). Run as many workers as CPU cores: the sequential
search loops run on the cores, the dense kernels on the GPUs. With several workers per GPU, NVIDIA MPS
(``nvidia-cuda-mps-control -d``) lets their kernels run concurrently instead of time-sliced.

The CPU path is the existing numpy code, unchanged, so results without a GPU stay bit-identical. GPU results
agree to float32 rounding (the exp is evaluated in float32 on both paths), not bit for bit: a run records the
backend it used (``backend_name``, ``describe``) and comparisons should not mix the two.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import warnings

import numpy as np


def enabled() -> bool:
    """GPU backend switched on (``CSPAPS_GPU=1``) and not limited to zero GPUs (``CSPAPS_GPUS=0``)."""
    if os.environ.get("CSPAPS_GPUS", "").strip() == "0":
        return False
    return os.environ.get("CSPAPS_GPU", "").strip().lower() in ("1", "true", "yes", "on")


def gpu_inventory() -> dict:
    """The visible CUDA devices, whether or not the backend is on (for the interface):
    ``{"devices": [{"id", "name", "memory_gb"}], "error": None | "no_cupy" | "no_driver" | "no_device",
    "detail": str}``."""
    try:
        import cupy
    except Exception as exc:  # noqa: BLE001 - not installed / broken install
        return {"devices": [], "error": "no_cupy", "detail": repr(exc)}
    try:
        n = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - no driver, driver too old for this CuPy, no device node
        return {"devices": [], "error": "no_driver", "detail": repr(exc)}
    devs = []
    for i in range(n):
        try:
            props = cupy.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            mem = float(props.get("totalGlobalMem", 0)) / 2 ** 30
        except Exception:  # noqa: BLE001
            name, mem = "?", 0.0
        devs.append({"id": i, "name": name, "memory_gb": round(mem, 1)})
    return {"devices": devs, "error": None if devs else "no_device", "detail": ""}


def detected_gpus() -> int:
    """Visible CUDA devices (0 without CuPy / driver / device), whether or not the backend is on."""
    return len(gpu_inventory()["devices"])


def selected_gpus() -> list[int]:
    """The devices the calculations of this process will use ([] = CPU only)."""
    if not enabled():
        return []
    visible = {d["id"] for d in gpu_inventory()["devices"]}
    return [d for d in devices() if d in visible]


def _cupy():
    """CuPy when the GPU backend is switched on and a device is visible, else None (never raises)."""
    if not enabled():
        return None
    try:
        import cupy

        return cupy if cupy.cuda.runtime.getDeviceCount() > 0 else None
    except Exception:  # noqa: BLE001 - no CuPy, no driver, no device: CPU
        return None


def devices() -> list[int]:
    """The GPUs to use: ``CSPAPS_GPU_DEVICES`` (a list, e.g. ``0,2,5``), else the first ``CSPAPS_GPUS``
    visible ones, else all visible devices."""
    raw = os.environ.get("CSPAPS_GPU_DEVICES", "").strip()
    if raw:
        return [int(v) for v in raw.replace(";", ",").split(",") if v.strip()]
    cp = _cupy()
    if cp is None:
        return []
    visible = list(range(cp.cuda.runtime.getDeviceCount()))
    count = os.environ.get("CSPAPS_GPUS", "").strip()
    if count and count.lower() not in ("all", "-1"):
        return visible[:max(0, int(count))]
    return visible


def set_resources(gpus=None) -> None:
    """Choose the GPUs for this process and the pools it starts (CLI ``--gpus``, GUI): a number of GPUs
    (0 = CPU only, "all" / -1 = every visible GPU) or a list of device ids."""
    if gpus is not None:
        if isinstance(gpus, (list, tuple)):
            os.environ["CSPAPS_GPU_DEVICES"] = ",".join(str(int(g)) for g in gpus)
            os.environ.pop("CSPAPS_GPUS", None)
            os.environ["CSPAPS_GPU"] = "1" if gpus else "0"
        else:
            text = str(gpus).strip().lower()
            os.environ.pop("CSPAPS_GPU_DEVICES", None)
            os.environ["CSPAPS_GPUS"] = "all" if text in ("all", "-1") else str(max(0, int(text)))
            os.environ["CSPAPS_GPU"] = "0" if os.environ["CSPAPS_GPUS"] == "0" else "1"


def device_for_this_process() -> int | None:
    """The device of this process: ``CSPAPS_GPU_DEVICE`` (set per pool worker by ``worker_init``), else the
    pool-worker number round-robin over ``devices()``, else the first device."""
    devs = devices()
    if not devs:
        return None
    raw = os.environ.get("CSPAPS_GPU_DEVICE", "").strip()
    if raw:
        return int(raw)
    import multiprocessing as mp

    ident = getattr(mp.current_process(), "_identity", ())
    return devs[(ident[-1] - 1) % len(devs)] if ident else devs[0]


def worker_init(counter, n_workers: int) -> None:
    """Pool-worker initializer: the next device round-robin and the memory share of each process on it."""
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    devs = devices()
    if not devs:
        return
    os.environ["CSPAPS_GPU_DEVICE"] = str(devs[index % len(devs)])
    os.environ["CSPAPS_GPU_SHARE"] = str(max(1, math.ceil(n_workers / len(devs))))


def backend_name() -> str:
    return "cupy" if _cupy() is not None else "numpy"


def describe() -> dict:
    """Backend, devices and kernels of this process (recorded with results)."""
    from . import kernels

    out = {"backend": backend_name(), "cpu_kernels": kernels.name()}
    cp = _cupy()
    if cp is not None:
        dev = device_for_this_process()
        try:
            props = cp.cuda.runtime.getDeviceProperties(dev or 0)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        except Exception:  # noqa: BLE001
            name = "?"
        out.update(device=dev, device_name=name, devices=devices())
    return out


def min_elements() -> int:
    try:
        return int(float(os.environ.get("CSPAPS_GPU_MIN", 2_000_000)))
    except ValueError:
        return 2_000_000


def _share() -> int:
    try:
        return max(1, int(os.environ.get("CSPAPS_GPU_SHARE", "1")))
    except ValueError:
        return 1


_CONFIGURED: set = set()
GPU_MEMORY_FRACTION = 0.85  # of each GPU's memory for all calculation processes on it: the rest stays free


@contextlib.contextmanager
def _device(cp):
    """This process's device as the current one; on first use caps the memory pool at its share of
    ``GPU_MEMORY_FRACTION`` of the GPU's memory."""
    dev = device_for_this_process() or 0
    with cp.cuda.Device(dev):
        if dev not in _CONFIGURED:
            _CONFIGURED.add(dev)
            share = _share()  # also with one process per GPU: the display and other programs keep some memory
            with contextlib.suppress(Exception):
                cp.get_default_memory_pool().set_limit(fraction=GPU_MEMORY_FRACTION / share)
        yield dev


def block_elements(cp=None) -> int:
    """Elements of one dense block on the device: ``CSPAPS_GPU_BLOCK``, capped by this process's memory share
    (forming a block keeps ~5 float64 temporaries alive)."""
    try:
        block = int(float(os.environ.get("CSPAPS_GPU_BLOCK", 8_000_000)))
    except ValueError:
        block = 8_000_000
    if cp is not None:
        with contextlib.suppress(Exception):
            free, total = cp.cuda.runtime.memGetInfo()
            block = min(block, int(0.8 * min(free, total / _share()) / 40))
    return max(block, 100_000)


def footprints_xp(xp, points, normals, aim_xyz, t, sigma, power):
    """``optics.footprints_at`` written for any array module (numpy or cupy); inputs are xp arrays."""
    z = xp.asarray([0.0, 0.0, 1.0])
    e_u = xp.cross(t, z)
    norm = xp.linalg.norm(e_u, axis=-1, keepdims=True)
    e_u = xp.where(norm > 1e-9, e_u / xp.maximum(norm, 1e-12), xp.asarray([1.0, 0.0, 0.0]))
    e_v = xp.cross(e_u, t)
    u = e_u @ points.T - xp.sum(aim_xyz * e_u, axis=1)[:, None]
    v = e_v @ points.T - xp.sum(aim_xyz * e_v, axis=1)[:, None]
    cos_s = xp.maximum(-(t @ normals.T), 0.0)
    q = (u * u + v * v) * (0.5 / sigma ** 2)[:, None]
    out = xp.exp(-q.astype(xp.float32)).astype(xp.float64)
    out *= cos_s
    out *= (power / (2 * np.pi * sigma ** 2))[:, None]
    return out


def _dev(cp, a, dtype=np.float64):
    return cp.asarray(np.ascontiguousarray(a, dtype=dtype))


def _beams(cp, aim_xyz, t, sigma, power, sl):
    return (_dev(cp, aim_xyz[sl]), _dev(cp, t[sl]), _dev(cp, sigma[sl]), _dev(cp, power[sl]))


OOM_FALLBACKS: list[str] = []  # GPU steps that ran out of device memory and were done on the CPU instead


def _is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "outofmemory" in text.replace(" ", "") or "out of memory" in text or "memoryallocation" in text


def free_device_memory() -> None:
    """Hand the memory pools' cached blocks back to the device (after a failed allocation)."""
    cp = _cupy()
    if cp is None:
        return
    for pool in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
        with contextlib.suppress(Exception):
            getattr(cp, pool)().free_all_blocks()


def falls_back_on_oom(fn):
    """A GPU entry point (returns None for "use the CPU path"): when the device runs out of memory, its pools
    are freed and None is returned, so this step runs on the CPU and the calculation goes on (results agree to
    rounding). Other errors propagate."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - CuPy's OutOfMemoryError / CUDA allocation errors
            if not _is_out_of_memory(exc):
                raise
            free_device_memory()
            if not OOM_FALLBACKS:
                warnings.warn(f"GPU memory full in {fn.__name__} ({exc}); this step runs on the CPU",
                              RuntimeWarning, stacklevel=2)
            OOM_FALLBACKS.append(fn.__name__)
            return None

    return wrapper


def _gpu_for(n_elements: int, dense: bool = False):
    """CuPy when the GPU backend is on and the work is large enough, else None.

    A dense result crosses the bus whole (``CSPAPS_GPU_MIN`` elements to pay for it); a reduced one is small, so
    the GPU pays off from an eighth of that."""
    if n_elements < (min_elements() if dense else min_elements() // 8):
        return None
    return _cupy()


@falls_back_on_oom
def footprints_gpu(points, normals, aim_xyz, t, sigma, power):
    """The footprint matrix on the GPU, or None when the CPU path should be used."""
    cp = _gpu_for(aim_xyz.shape[0] * points.shape[0], dense=True)
    if cp is None:
        return None
    with _device(cp):
        args = [cp.asarray(np.asarray(a, dtype=np.float64)) for a in (points, normals, aim_xyz, t, sigma, power)]
        return cp.asnumpy(footprints_xp(cp, *args))


@falls_back_on_oom
def flux_sum_gpu(points, normals, aim_xyz, t, sigma, power):
    """Superposed flux of all beams at the points, (P,), reduced on the device; None -> CPU path."""
    n_b, n_p = aim_xyz.shape[0], points.shape[0]
    cp = _gpu_for(n_b * n_p)
    if cp is None:
        return None
    with _device(cp):
        pts, nrm = _dev(cp, points), _dev(cp, normals)
        step = max(1, block_elements(cp) // max(n_p, 1))
        out = cp.zeros(n_p)
        for s in range(0, n_b, step):
            out += footprints_xp(cp, pts, nrm, *_beams(cp, aim_xyz, t, sigma, power, slice(s, s + step))).sum(axis=0)
        return cp.asnumpy(out)


@falls_back_on_oom
def flux_and_power_gpu(points, normals, aim_xyz, t, sigma, power, cell_area):
    """(flux at the points (P,), flux sum per beam x ``cell_area`` (R,)), reduced on the device; None -> CPU."""
    n_b, n_p = aim_xyz.shape[0], points.shape[0]
    cp = _gpu_for(n_b * n_p)
    if cp is None:
        return None
    with _device(cp):
        pts, nrm = _dev(cp, points), _dev(cp, normals)
        step = max(1, block_elements(cp) // max(n_p, 1))
        flux = cp.zeros(n_p)
        per_beam = cp.zeros(n_b)
        for s in range(0, n_b, step):
            f = footprints_xp(cp, pts, nrm, *_beams(cp, aim_xyz, t, sigma, power, slice(s, s + step)))
            flux += f.sum(axis=0)
            per_beam[s:s + f.shape[0]] = f.sum(axis=1) * cell_area
        return cp.asnumpy(flux), cp.asnumpy(per_beam)


@falls_back_on_oom
def row_stats_gpu(points, normals, aim_xyz, t, sigma, power, weights):
    """Per beam: the flux sum over the points and the ``weights``-weighted sum, (R,) each; None -> CPU path."""
    n_b, n_p = aim_xyz.shape[0], points.shape[0]
    cp = _gpu_for(n_b * n_p)
    if cp is None:
        return None
    with _device(cp):
        pts, nrm, w = _dev(cp, points), _dev(cp, normals), _dev(cp, weights)
        step = max(1, block_elements(cp) // max(n_p, 1))
        sums, dots = cp.zeros(n_b), cp.zeros(n_b)
        for s in range(0, n_b, step):
            f = footprints_xp(cp, pts, nrm, *_beams(cp, aim_xyz, t, sigma, power, slice(s, s + step)))
            sums[s:s + f.shape[0]] = f.sum(axis=1)
            dots[s:s + f.shape[0]] = f @ w
        return cp.asnumpy(sums), cp.asnumpy(dots)


@falls_back_on_oom
def sparse_responses_gpu(points, normals, aim_xyz, t, sigma, power, cutoff):
    """Sparse candidate responses of N heliostats x K candidates, built on the device.

    ``aim_xyz``/``t`` (N, K, 3), ``sigma``/``power`` (N, K). Per heliostat, as ``aps.build_response`` does on the
    CPU: the peak of its (K, P) block, the cells where some candidate exceeds ``cutoff`` x peak (sorted) and the
    (K, S) float32 values there. Only the support values cross the bus. Returns (peaks, cells, resp) or None."""
    n, k = sigma.shape
    n_p = points.shape[0]
    cp = _gpu_for(n * k * n_p)
    if cp is None:
        return None
    peaks = np.zeros(n)
    cells: list[np.ndarray] = []
    resp: list[np.ndarray] = []
    with _device(cp):
        pts, nrm = _dev(cp, points), _dev(cp, normals)
        step = max(1, block_elements(cp) // max(k * n_p, 1))
        for s in range(0, n, step):
            e = min(n, s + step)
            m = e - s
            f = footprints_xp(cp, pts, nrm, _dev(cp, aim_xyz[s:e].reshape(-1, 3)), _dev(cp, t[s:e].reshape(-1, 3)),
                              _dev(cp, sigma[s:e].reshape(-1)), _dev(cp, power[s:e].reshape(-1))).reshape(m, k, n_p)
            peak = f.reshape(m, -1).max(axis=1)
            mask = (f.max(axis=1) > cutoff * peak[:, None]) & (peak[:, None] > 0)
            rows, cols = cp.nonzero(mask)  # row-major: heliostat by heliostat, cells ascending
            vals = f.astype(cp.float32).transpose(0, 2, 1)[rows, cols]  # (nnz, K)
            peaks[s:e] = cp.asnumpy(peak)
            rows, cols, vals = cp.asnumpy(rows), cp.asnumpy(cols).astype(np.int64), cp.asnumpy(vals)
            counts = np.bincount(rows, minlength=m)
            ends = np.cumsum(counts)
            for j in range(m):
                a, b = ends[j] - counts[j], ends[j]
                cells.append(cols[a:b])
                resp.append(np.ascontiguousarray(vals[a:b].T))
    return peaks, cells, resp


def intercept_xp(xp, ndtr, aim_xyz, t, sigma, radius, c_bot, z_span, x, w):
    """``optics.intercept_fraction`` for any array module (same formula and quadrature); xp inputs, (n,) out."""
    z = xp.asarray([0.0, 0.0, 1.0])
    e_u = xp.cross(t, z)
    norm = xp.linalg.norm(e_u, axis=-1, keepdims=True)
    e_u = xp.where(norm > 1e-9, e_u / xp.maximum(norm, 1e-12), xp.asarray([1.0, 0.0, 0.0]))
    e_v = xp.cross(e_u, t)
    h = -t.copy()
    h[:, 2] = 0.0
    h /= xp.linalg.norm(h, axis=1, keepdims=True)
    a_u = xp.sum((aim_xyz - c_bot) * e_u, axis=1)
    lo = xp.maximum(-radius, a_u - 8 * sigma)
    hi = xp.minimum(radius, a_u + 8 * sigma)
    ok = hi > lo
    mid, half = 0.5 * (hi + lo), 0.5 * xp.where(ok, hi - lo, 0.0)
    U = mid[:, None] + half[:, None] * x[None, :]
    depth = xp.sqrt(xp.maximum(radius * radius - U * U, 0.0))
    p_bot = c_bot + h[:, None, :] * depth[..., None] + e_u[:, None, :] * U[..., None]
    r = p_bot - aim_xyz[:, None, :]
    u = U - a_u[:, None]
    v_bot = xp.sum(r * e_v[:, None, :], axis=2)
    v_top = v_bot + z_span * e_v[:, 2:3]
    s = sigma[:, None]
    pdf = xp.exp(-0.5 * (u / s) ** 2) / (math.sqrt(2 * math.pi) * s)
    band = xp.abs(ndtr(v_top / s) - ndtr(v_bot / s))
    frac = (pdf * band) @ w * half
    return xp.minimum(xp.maximum(xp.where(ok, frac, 0.0), 0.0), 1.0)


@falls_back_on_oom
def intercept_fraction_gpu(aim_xyz, t, sigma, radius, c_bot, z_span, n_quad):
    """Gaussian intercept fractions of (n,) beams on the device; None -> CPU path."""
    n = sigma.shape[0]
    cp = _gpu_for(4 * n * n_quad)
    if cp is None:
        return None
    from cupyx.scipy.special import ndtr

    x, w = np.polynomial.legendre.leggauss(n_quad)
    out = np.empty(n)
    with _device(cp):
        xd, wd, cb = _dev(cp, x), _dev(cp, w), _dev(cp, c_bot)
        step = max(1, block_elements(cp) // (12 * n_quad))
        for s in range(0, n, step):
            sl = slice(s, s + step)
            out[sl] = cp.asnumpy(intercept_xp(cp, ndtr, _dev(cp, aim_xyz[sl]), _dev(cp, t[sl]), _dev(cp, sigma[sl]),
                                              float(radius), cb, float(z_span), xd, wd))
    return out


def self_check(n_beams: int = 4000, n_points: int = 4000, seed: int = 0) -> dict:
    """GPU vs CPU on random beams: largest relative difference and the time of each, per device (run on the GPU
    machine: ``CSPAPS_GPU=1 python -m cspaps.backend``)."""
    import time

    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2 * np.pi, n_points)
    pts = np.column_stack([3.5 * np.sin(az), 3.5 * np.cos(az), rng.uniform(96, 104, n_points)])
    nrm = np.column_stack([np.sin(az), np.cos(az), np.zeros(n_points)])
    src = np.column_stack([rng.uniform(-300, 300, n_beams), rng.uniform(-300, 300, n_beams), np.full(n_beams, 5.0)])
    aim = pts[rng.integers(0, n_points, n_beams)]
    t = aim - src
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    sigma, power = rng.uniform(0.5, 2.0, n_beams), rng.uniform(1e4, 6e4, n_beams)
    t0 = time.perf_counter()
    cpu = footprints_xp(np, pts, nrm, aim, t, sigma, power)
    cpu_sum = cpu.sum(axis=0)
    t_cpu = time.perf_counter() - t0
    out = {**describe(), "elements": n_beams * n_points, "cpu_s": round(t_cpu, 3)}
    cp = _cupy()
    if cp is None:
        out["note"] = "no CuPy / GPU: set CSPAPS_GPU=1 and install the gpu extra (pip install -e \".[gpu]\")"
        return out
    per_device = []
    for dev in devices():
        os.environ["CSPAPS_GPU_DEVICE"] = str(dev)
        with cp.cuda.Device(dev):
            small = [cp.asarray(a[:8]) for a in (pts, nrm, aim, t, sigma, power)]
            cp.asnumpy(footprints_xp(cp, *small))  # warm-up: context and kernel compilation
            t0 = time.perf_counter()
            gpu_sum = flux_sum_gpu(pts, nrm, aim, t, sigma, power)  # transfers included, as in use
            t_gpu = time.perf_counter() - t0
        scale = float(np.abs(cpu_sum).max()) or 1.0
        per_device.append({"device": dev, "gpu_s": round(t_gpu, 4), "speedup": round(t_cpu / max(t_gpu, 1e-9), 1),
                           "max_abs_diff_rel_to_peak": float(np.abs(gpu_sum - cpu_sum).max() / scale)})
    os.environ.pop("CSPAPS_GPU_DEVICE", None)
    out["devices_checked"] = per_device
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), indent=2, ensure_ascii=False))
