"""Optional GPU backend (CuPy) for the flux kernel every spot computation goes through.

``optics.footprints_at`` -- spot responses of the aim candidates, fine-grid verification, flux maps -- is a
dense exp over beams x surface points and dominates large fields. With ``CSPAPS_GPU=1`` and CuPy installed it
runs on the GPU when the work is large enough (``CSPAPS_GPU_MIN`` elements, default 2 million) to pay for the
transfers; worker processes spread over the devices of ``CSPAPS_GPU_DEVICES`` (default: all) by process id.

The CPU path is the existing numpy code, unchanged, so results without a GPU stay bit-identical. GPU results
agree to float32 rounding (the exp is evaluated in float32 on both paths), not bit for bit: a run records the
backend it used (``backend_name``) and comparisons should not mix the two.
"""

from __future__ import annotations

import os

import numpy as np


def enabled() -> bool:
    return os.environ.get("CSPAPS_GPU", "").strip().lower() in ("1", "true", "yes", "on")


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
    raw = os.environ.get("CSPAPS_GPU_DEVICES", "").strip()
    if raw:
        return [int(v) for v in raw.replace(";", ",").split(",") if v.strip()]
    cp = _cupy()
    return list(range(cp.cuda.runtime.getDeviceCount())) if cp is not None else []


def device_for_this_process() -> int | None:
    """The device of this (worker) process: process id modulo the number of devices."""
    devs = devices()
    return devs[os.getpid() % len(devs)] if devs else None


def backend_name() -> str:
    return "cupy" if _cupy() is not None else "numpy"


def min_elements() -> int:
    try:
        return int(float(os.environ.get("CSPAPS_GPU_MIN", 2_000_000)))
    except ValueError:
        return 2_000_000


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


def footprints_gpu(points, normals, aim_xyz, t, sigma, power):
    """The footprint matrix on the GPU, or None when the CPU path should be used."""
    if aim_xyz.shape[0] * points.shape[0] < min_elements():
        return None
    cp = _cupy()
    if cp is None:
        return None
    dev = device_for_this_process()
    with cp.cuda.Device(dev if dev is not None else 0):
        args = [cp.asarray(np.asarray(a, dtype=np.float64)) for a in (points, normals, aim_xyz, t, sigma, power)]
        return cp.asnumpy(footprints_xp(cp, *args))


def self_check(n_beams: int = 4000, n_points: int = 4000, seed: int = 0) -> dict:
    """GPU vs CPU on random beams: largest relative difference and the time of each (run on the GPU machine:
    ``CSPAPS_GPU=1 python -m cspaps.backend``)."""
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
    t_cpu = time.perf_counter() - t0
    out = {"backend": backend_name(), "devices": devices(), "elements": n_beams * n_points, "cpu_s": round(t_cpu, 3)}
    cp = _cupy()
    if cp is None:
        out["note"] = "no CuPy / GPU: set CSPAPS_GPU=1 and install the gpu extra (pip install -e \".[gpu]\")"
        return out
    with cp.cuda.Device(device_for_this_process() or 0):
        small = [cp.asarray(a[:8]) for a in (pts, nrm, aim, t, sigma, power)]
        cp.asnumpy(footprints_xp(cp, *small))  # warm-up: context and kernel compilation
        t0 = time.perf_counter()
        args = [cp.asarray(a) for a in (pts, nrm, aim, t, sigma, power)]  # transfers included, as in use
        gpu = cp.asnumpy(footprints_xp(cp, *args))
        t_gpu = time.perf_counter() - t0
    scale = float(np.abs(cpu).max()) or 1.0
    out.update(gpu_s=round(t_gpu, 3), speedup=round(t_cpu / max(t_gpu, 1e-9), 1),
               max_abs_diff_rel_to_peak=float(np.abs(gpu - cpu).max() / scale))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), indent=2, ensure_ascii=False))
