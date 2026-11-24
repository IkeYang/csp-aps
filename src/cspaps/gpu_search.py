"""The uniformity local search as one CUDA thread block (GPU backend).

``uniformity._local_search`` visits the heliostats one after another: each accepted move changes the flux state
the next heliostat is scored against, so the visiting order cannot be parallelised. Each *step*, however, scores
K + 1 candidate rows on the heliostat's S support cells -- on a 20,000-mirror field ~70 % of the CPU time of a
scenario. Here one thread block runs the whole search: warps take the candidate rows, lanes the support cells,
and a step costs a handful of barriers instead of K x S serial operations. Several scenarios (worker processes)
share a GPU; with NVIDIA MPS their blocks run at the same time.

Same moves, order, tie rule, tolerances and float32 / float64 arithmetic as the CPU code; the row sums are
added in a fixed but different order (lanes, then rows), so results agree with the CPU kernels to rounding, not
bit for bit -- like every GPU result (``backend``) -- and are deterministic on a given GPU model.

The kernel source compiles as CUDA (NVRTC via CuPy) and as plain C++: thread-parallel sections are explicit loops
over thread ids separated by barriers (``FOR_THREADS`` / ``SYNC``), so ``emulate`` runs the identical code on the
CPU (g++) and the tests check it there. Any GPU failure (compilation, launch, memory) falls back to the CPU
kernels and is reported once. ``CSPAPS_GPU_SEARCH=0`` keeps the search on the CPU.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

THREADS = 256  # threads per block (a multiple of 32)
MAX_ROWS = 48  # K + 1 candidate rows that fit the shared-memory layout (K <= 47)

SOURCE = r"""
#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
  #define KERNEL extern "C" __global__
  #define SYNC() __syncthreads()
  #define T_FIRST ((int)threadIdx.x)
  #define T_STEP ((int)blockDim.x)
  #define SHARED_BUFFER extern __shared__ double smem[];
#else
  #include <math.h>
  #define KERNEL extern "C"
  #define SYNC() ((void)0)
  #define T_FIRST 0
  #define T_STEP 1
  static double smem_cpu[16384];
  #define SHARED_BUFFER double* smem = smem_cpu;
#endif
/* every thread id t of the block: on the GPU each thread runs its own t, on the CPU one loop runs them all */
#define FOR_THREADS(t) for (int t = T_FIRST; t < nt; t += T_STEP)

KERNEL void local_search(
    const long long* order, long long n_order, const unsigned char* available,
    const long long* ptr, const long long* cells, const float* resp, int n_k, long long n,
    const double* dni_con, const double* dni_obj, const double* alpha, const double* limit,
    const unsigned char* in_h, double area, double a_h, const double* absorb_power,
    const unsigned char* reach, int has_reach, const double* move_cost, int has_move,
    int allow_defocus, double mu, int passes,
    double* flux_con, double* q, long long* choice, double* sums, long long* limited_out,
    float* thr, double* base_con, double* base_q, int nt)
{
    SHARED_BUFFER
    const int rows = n_k + 1;
    double* part1 = smem;                 /* (rows, 32) lane partial sums of the new H flux */
    double* part2 = part1 + rows * 32;    /* (rows, 32) ... of its square */
    double* bad = part2 + rows * 32;      /* (rows, 32) 1 = a touched cell over its limit */
    double* tp1 = bad + rows * 32;        /* (nt) thread partials of the current H flux */
    double* tp2 = tp1 + nt;
    double* rs1 = tp2 + nt;               /* (rows) row sums */
    double* rs2 = rs1 + rows;
    double* rbad = rs2 + rows;
    double* shd = rbad + rows;            /* scalars: s1, s2, old_sum1, old_sum2 */
    long long* sh = (long long*)(shd + 8);/* scalars: [4] move, [5] moved, [6] limited */

    /* the "touched" threshold of every candidate row: float32 1e-3 * max(row max, 1e-30) */
    FOR_THREADS(t) {
        for (long long row = t; row < n * n_k; row += nt) {
            long long i = row / n_k;
            int k = (int)(row % n_k);
            long long a = ptr[i], n_s = ptr[i + 1] - a;
            float dcon = (float)dni_con[i];
            const float* r = resp + n_k * a + (long long)k * n_s;
            float peak = 0.0f;
            for (long long s = 0; s < n_s; ++s) {
                float v = r[s] * dcon;
                if (s == 0 || v > peak) peak = v;
            }
            thr[row] = 1e-3f * (peak > 1e-30f ? peak : 1e-30f);
        }
    }
    FOR_THREADS(t) {
        if (t == 0) { shd[0] = sums[0]; shd[1] = sums[1]; sh[6] = 0; }
    }
    SYNC();
    const int nw = nt / 32;
    for (int pass = 0; pass < passes; ++pass) {
        FOR_THREADS(t) { if (t == 0) sh[5] = 0; }
        SYNC();
        for (long long oi = 0; oi < n_order; ++oi) {
            const long long i = order[oi];
            if (!available[i]) continue;
            const long long a = ptr[i], n_s = ptr[i + 1] - a;
            if (n_s == 0) continue;
            const long long old = choice[i];
            const float dcon = (float)dni_con[i], dobj = (float)dni_obj[i];
            const float* R = resp + n_k * a;
            /* the state without heliostat i, and the current H flux */
            FOR_THREADS(t) {
                double p1 = 0.0, p2 = 0.0;
                for (long long s = t; s < n_s; s += nt) {
                    long long c = cells[a + s];
                    double bc = flux_con[c], bq = q[c];
                    if (old >= 0) {
                        float r = R[old * n_s + s];
                        bc = bc - (double)(r * dcon);
                        bq = bq - (double)(r * dobj) * alpha[c];
                    }
                    base_con[s] = bc;
                    base_q[s] = bq;
                    if (in_h[c]) { double qc = q[c]; p1 += qc; p2 += qc * qc; }
                }
                tp1[t] = p1;
                tp2[t] = p2;
            }
            SYNC();
            /* candidate rows: warp w takes rows w, w + nw, ...; its lanes the cells */
            FOR_THREADS(t) {
                int w = t / 32, lane = t % 32;
                for (int k = w; k < rows; k += nw) {
                    double p1 = 0.0, p2 = 0.0, b = 0.0;
                    if (k < n_k) {
                        const float* Rk = R + (long long)k * n_s;
                        float th = thr[i * n_k + k];
                        for (long long s = lane; s < n_s; s += 32) {
                            float r = Rk[s];
                            float v = r * dcon;
                            long long c = cells[a + s];
                            if (v > th && !(base_con[s] + (double)v <= limit[c] * (1.0 + 1e-9))) { b = 1.0; break; }
                            if (in_h[c]) { double nq = base_q[s] + (double)(r * dobj) * alpha[c]; p1 += nq; p2 += nq * nq; }
                        }
                    } else {
                        for (long long s = lane; s < n_s; s += 32) {
                            if (in_h[cells[a + s]]) { double nq = base_q[s] + 0.0; p1 += nq; p2 += nq * nq; }
                        }
                    }
                    part1[k * 32 + lane] = p1;
                    part2[k * 32 + lane] = p2;
                    bad[k * 32 + lane] = b;
                }
            }
            SYNC();
            /* row sums (thread k: row k), current H sums (thread `rows`) */
            FOR_THREADS(t) {
                if (t < rows) {
                    double r1 = 0.0, r2 = 0.0, b = 0.0;
                    for (int u = 0; u < 32; ++u) {
                        r1 += part1[t * 32 + u];
                        r2 += part2[t * 32 + u];
                        if (bad[t * 32 + u] != 0.0) b = 1.0;
                    }
                    rs1[t] = r1; rs2[t] = r2; rbad[t] = b;
                } else if (t == rows) {
                    double o1 = 0.0, o2 = 0.0;
                    for (int u = 0; u < nt; ++u) { o1 += tp1[u]; o2 += tp2[u]; }
                    shd[2] = o1; shd[3] = o2;
                }
            }
            SYNC();
            /* the move: best feasible row (first maximum), accepted if it beats the current one by the tolerance */
            FOR_THREADS(t) {
                if (t == 0) {
                    const double s1 = shd[0], s2 = shd[1], o1 = shd[2], o2 = shd[3];
                    const int cur = old < 0 ? n_k : (int)old;
                    int best = -1, cur_ok = 0, any_limited = 0;
                    double best_score = 0.0, cur_score = 0.0, s1_best = 0.0, s2_best = 0.0;
                    for (int k = 0; k < rows; ++k) {
                        int ok;
                        if (k < n_k) {
                            ok = rbad[k] == 0.0;
                            if (!ok) any_limited = 1;
                            if (ok && has_reach && !reach[i * n_k + k]) ok = 0;
                        } else {
                            ok = allow_defocus || old < 0;
                        }
                        if (!ok) continue;
                        double s1n = s1 + area * (rs1[k] - o1);
                        double s2n = s2 + area * (rs2[k] - o2);
                        double m1 = s1n / a_h;
                        double var = s2n / a_h - m1 * m1;
                        double sd = sqrt(var > 0.0 ? var : 0.0);
                        double sc = (k < n_k ? absorb_power[i * n_k + k] : 0.0) - mu * a_h * sd;
                        if (has_move && k < n_k) sc -= move_cost[i * n_k + k];
                        if (k == cur) { cur_ok = 1; cur_score = sc; }
                        if (best < 0 || sc > best_score) { best = k; best_score = sc; s1_best = s1n; s2_best = s2n; }
                    }
                    if (any_limited) sh[6] += 1;
                    long long move = -2;
                    if (best >= 0 && best != cur) {
                        double tol = 0.0;
                        if (cur_ok) { double m = fabs(cur_score); tol = 1e-9 * (m > 1.0 ? m : 1.0); }
                        if (!cur_ok || best_score > cur_score + tol) {
                            move = best;
                            shd[0] = s1_best;
                            shd[1] = s2_best;
                            choice[i] = best == n_k ? -1 : best;
                            sh[5] += 1;
                        }
                    }
                    sh[4] = move;
                }
            }
            SYNC();
            const long long mv = sh[4];
            if (mv != -2) {
                FOR_THREADS(t) {
                    for (long long s = t; s < n_s; s += nt) {
                        long long c = cells[a + s];
                        if (mv < n_k) {
                            float r = R[mv * n_s + s];
                            flux_con[c] = base_con[s] + (double)(r * dcon);
                            q[c] = base_q[s] + (double)(r * dobj) * alpha[c];
                        } else {
                            flux_con[c] = base_con[s] + 0.0;
                            q[c] = base_q[s] + 0.0;
                        }
                    }
                }
            }
            SYNC();
        }
        if (sh[5] == 0) break;
    }
    FOR_THREADS(t) {
        if (t == 0) { sums[0] = shd[0]; sums[1] = shd[1]; limited_out[0] = sh[6]; }
    }
}
"""


def shared_bytes(n_k: int, nt: int = THREADS) -> int:
    rows = n_k + 1
    return 8 * (3 * rows * 32 + 2 * nt + 3 * rows + 8 + 8)


def enabled() -> bool:
    """GPU local search: GPU backend on and not switched off with ``CSPAPS_GPU_SEARCH=0``."""
    from .backend import enabled as gpu_enabled

    if os.environ.get("CSPAPS_GPU_SEARCH", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return gpu_enabled() and not _FAILED


_FAILED: list = []  # set once a GPU attempt failed in this process: the CPU kernels take over
_KERNELS: dict = {}


def _kernel(cp, dev: int):
    if dev not in _KERNELS:
        _KERNELS[dev] = cp.RawKernel(SOURCE, "local_search", options=("--fmad=false",))
    return _KERNELS[dev]


def _inputs(prob, state, absorb_power, order, move_cost):
    """The kernel's host arrays (shared by the GPU launch and the CPU emulation)."""
    from .kernels import packed_responses

    ptr, cells, resp = packed_responses(prob.rs)
    n, k = int(prob.n), int(prob.k)
    reach = prob.reach
    return {
        "order": np.ascontiguousarray(order, dtype=np.int64),
        "available": np.ascontiguousarray(prob.available, dtype=np.uint8),
        "ptr": ptr, "cells": cells, "resp": resp, "n_k": k, "n": n,
        "dni_con": np.ascontiguousarray(prob.dni_con, dtype=float),
        "dni_obj": np.ascontiguousarray(prob.dni_obj, dtype=float),
        "alpha": np.ascontiguousarray(state.alpha, dtype=float),
        "limit": np.ascontiguousarray(prob.limit, dtype=float),
        "in_h": np.ascontiguousarray(state.h, dtype=np.uint8),
        "area": float(state.area), "a_h": float(state.a_h),
        "absorb_power": np.ascontiguousarray(absorb_power, dtype=float),
        "reach": np.ascontiguousarray(reach if reach is not None else np.zeros((1, 1)), dtype=np.uint8),
        "has_reach": int(reach is not None),
        "move_cost": np.ascontiguousarray(move_cost if move_cost is not None else np.zeros((1, 1)), dtype=float),
        "has_move": int(move_cost is not None),
        "allow_defocus": int(bool(prob.case.aps.allow_defocus)),
        "s_max": int(np.diff(ptr).max()) if n else 0,
    }


_ORDER = ["order", "n_order", "available", "ptr", "cells", "resp", "n_k", "n", "dni_con", "dni_obj", "alpha",
          "limit", "in_h", "area", "a_h", "absorb_power", "reach", "has_reach", "move_cost", "has_move",
          "allow_defocus", "mu", "passes", "flux_con", "q", "choice", "sums", "limited", "thr", "base_con",
          "base_q", "nt"]


def local_search(prob, state, mu: float, passes: int, absorb_power, order, move_cost) -> bool:
    """Run ``uniformity._local_search`` on this process's GPU (state updated in place). False: not run (too
    small, no GPU, or a GPU failure) -- the caller uses the CPU kernels."""
    from . import backend

    if not enabled() or prob.k + 1 > MAX_ROWS or prob.n == 0:
        return False
    try:
        x = _inputs(prob, state, absorb_power, order, move_cost)
        cp = backend._gpu_for(int(x["resp"].size))
        if cp is None:
            return False
        with backend._device(cp) as dev:
            d = {key: cp.asarray(v) for key, v in x.items() if isinstance(v, np.ndarray)}
            flux_con, q = cp.asarray(state.flux_con), cp.asarray(state.q)
            choice = cp.asarray(np.ascontiguousarray(state.choice, dtype=np.int64))
            sums = cp.asarray(np.array([state.s1, state.s2], dtype=float))
            limited = cp.zeros(1, dtype=np.int64)
            thr = cp.empty(max(1, x["n"] * x["n_k"]), dtype=np.float32)
            base_con, base_q = cp.empty(max(1, x["s_max"])), cp.empty(max(1, x["s_max"]))
            vals = {**d, "n_order": np.int64(x["order"].size), "n_k": np.int32(x["n_k"]), "n": np.int64(x["n"]),
                    "area": np.float64(x["area"]), "a_h": np.float64(x["a_h"]), "has_reach": np.int32(x["has_reach"]),
                    "has_move": np.int32(x["has_move"]), "allow_defocus": np.int32(x["allow_defocus"]),
                    "mu": np.float64(mu), "passes": np.int32(passes), "flux_con": flux_con, "q": q,
                    "choice": choice, "sums": sums, "limited": limited, "thr": thr, "base_con": base_con,
                    "base_q": base_q, "nt": np.int32(THREADS)}
            _kernel(cp, dev)((1,), (THREADS,), tuple(vals[key] for key in _ORDER),
                             shared_mem=shared_bytes(x["n_k"]))
            state.flux_con[:] = cp.asnumpy(flux_con)
            state.q[:] = cp.asnumpy(q)
            state.choice[:] = cp.asnumpy(choice)
            s = cp.asnumpy(sums)
            n_limited = int(cp.asnumpy(limited)[0])
    except Exception as exc:  # noqa: BLE001 - any GPU problem: CPU kernels from now on
        _FAILED.append(repr(exc))
        warnings.warn(f"GPU local search unavailable ({exc!r}); using the CPU kernels", RuntimeWarning, stacklevel=2)
        return False
    state.s1, state.s2 = float(s[0]), float(s[1])
    if n_limited:
        prob.flux_limited = int(getattr(prob, "flux_limited", 0)) + n_limited
    return True


# --------------------------------------------------------------------------- CPU emulation (tests)

_CPU_BUILDS: dict = {}


def compile_cpu(source: str = SOURCE, name: str = "local_search"):
    """A kernel source compiled as C++ (g++, no FMA contraction): every thread id runs in turn between the
    barriers. Returns the ctypes function."""
    key = (hash(source), name)
    if key not in _CPU_BUILDS:
        d = Path(tempfile.mkdtemp(prefix="cspaps_gpu_kernel_"))
        src, lib = d / "kernel.cpp", d / ("kernel.dll" if sys.platform == "win32" else "kernel.so")
        src.write_text(source)
        subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-o", str(lib), str(src)],
                       check=True, capture_output=True)
        f = getattr(ctypes.CDLL(str(lib)), name)
        f.restype = None
        _CPU_BUILDS[key] = f
    return _CPU_BUILDS[key]


def _emulator():
    return compile_cpu(SOURCE, "local_search")


def emulate(prob, state, mu: float, passes: int, absorb_power, order, move_cost=None, nt: int = THREADS) -> None:
    """Run the GPU kernel's code on the CPU (state updated in place) -- for tests on machines without a GPU."""
    x = _inputs(prob, state, absorb_power, order, move_cost)
    choice = np.ascontiguousarray(state.choice, dtype=np.int64)
    sums = np.array([state.s1, state.s2], dtype=float)
    limited = np.zeros(1, dtype=np.int64)
    thr = np.empty(max(1, x["n"] * x["n_k"]), dtype=np.float32)
    base_con, base_q = np.empty(max(1, x["s_max"])), np.empty(max(1, x["s_max"]))
    P = ctypes.c_void_p
    arrays = {**{k: v for k, v in x.items() if isinstance(v, np.ndarray)}, "flux_con": state.flux_con, "q": state.q,
              "choice": choice, "sums": sums, "limited": limited, "thr": thr, "base_con": base_con, "base_q": base_q}
    scalars = {"n_order": ctypes.c_longlong(x["order"].size), "n_k": ctypes.c_int(x["n_k"]),
               "n": ctypes.c_longlong(x["n"]), "area": ctypes.c_double(x["area"]), "a_h": ctypes.c_double(x["a_h"]),
               "has_reach": ctypes.c_int(x["has_reach"]), "has_move": ctypes.c_int(x["has_move"]),
               "allow_defocus": ctypes.c_int(x["allow_defocus"]), "mu": ctypes.c_double(mu),
               "passes": ctypes.c_int(passes), "nt": ctypes.c_int(nt)}
    for a in arrays.values():
        assert a.flags.c_contiguous
    args = [P(arrays[k].ctypes.data) if k in arrays else scalars[k] for k in _ORDER]
    _emulator()(*args)
    state.choice[:] = choice
    state.s1, state.s2 = float(sums[0]), float(sums[1])
    if limited[0]:
        prob.flux_limited = int(getattr(prob, "flux_limited", 0)) + int(limited[0])
