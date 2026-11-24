"""Compiled CPU kernels (Numba) for the sequential solver loops.

The uniformity local search (``uniformity._local_search``) is a coordinate descent: every accepted move changes
the flux state that the next heliostat is scored against, so the moves cannot be spread over GPU threads. In
numpy each step costs a few dozen small array calls (K + 1 candidate rows x the heliostat's support cells); with
25 passes over every heliostat this loop took ~70 % of a scenario on a 1,500-mirror field. Compiled, the same
moves run in one tight loop and an infeasible candidate row stops at its first violated cell.

Bit-identical to the numpy code: same visiting order, same float32 / float64 arithmetic and the same summation
order. numpy sums a contiguous float64 array pairwise (blocks of 128 with eight partial sums; numpy <= 2.2 also
splits arrays longer than its 8192-element buffer) -- ``_np_sum`` reproduces that -- and the boolean-masked
candidate rows ``new_q[:, hm]`` (an F-ordered copy) sequentially. Both are probed once at run time against the
installed numpy; if they do not match, or Numba is missing, or ``CSPAPS_JIT=0``, the numpy path runs.
"""

from __future__ import annotations

import functools
import os

import numpy as np

try:  # optional dependency: the numpy reference path is used without it
    import numba as _nb
except Exception:  # noqa: BLE001 - not installed / broken install
    _nb = None


def _jit(f):
    return _nb.njit(cache=True, nogil=True)(f) if _nb is not None else f


# --------------------------------------------------------------------------- numpy-exact summation

@_jit
def _pw_block(a, lo, n):
    """numpy's ``pairwise_sum`` for n <= 128 (eight partial sums)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    stop = n - n % 8
    while i < stop:
        j = lo + i
        r0 += a[j]
        r1 += a[j + 1]
        r2 += a[j + 2]
        r3 += a[j + 3]
        r4 += a[j + 4]
        r5 += a[j + 5]
        r6 += a[j + 6]
        r7 += a[j + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@_jit
def _pw(a, lo, n, stk, part):
    """numpy's ``pairwise_sum``: halves (aligned down to 8) until blocks of <= 128, left + right at every node.

    Walks the same tree without recursion (Numba cannot cache recursive functions): ``stk`` (3, depth) int64
    holds each open node's start, length and stage (1: left half running, 2: right half running), ``part``
    the finished left half."""
    if n <= 128:
        return _pw_block(a, lo, n)
    top = 0
    stk[0, 0] = lo
    stk[1, 0] = n
    while True:
        cl = stk[0, top]
        cn = stk[1, top]
        if cn > 128:  # open a node: its left half first
            stk[2, top] = 1
            n2 = cn // 2
            n2 -= n2 % 8
            top += 1
            stk[0, top] = cl
            stk[1, top] = n2
            continue
        ret = _pw_block(a, cl, cn)
        top -= 1
        while top >= 0 and stk[2, top] == 2:  # both halves done: left + right
            ret = part[top] + ret
            top -= 1
        if top < 0:
            return ret
        part[top] = ret  # left half done: run the right half
        stk[2, top] = 2
        n2 = stk[1, top] // 2
        n2 -= n2 % 8
        stk[0, top + 1] = stk[0, top] + n2
        stk[1, top + 1] = stk[1, top] - n2
        top += 1


@_jit
def _np_sum(a, n, block, stk, part):
    """``a[:n].sum()`` exactly as numpy computes it; ``block`` = 0 (one pairwise sum) or the buffer size."""
    if block <= 0 or n <= block:
        return _pw(a, 0, n, stk, part)
    res = _pw(a, 0, block, stk, part)
    lo = block
    while lo < n:
        m = min(block, n - lo)
        res += _pw(a, lo, m, stk, part)
        lo += m
    return res


def _workspace():
    return np.zeros((3, 64), dtype=np.int64), np.zeros(64)


@functools.lru_cache(maxsize=1)
def _sum_block() -> int | None:
    """The summation variant of the installed numpy (0 or 8192), None if ``_np_sum`` cannot reproduce it."""
    if _nb is None:
        return None
    rng = np.random.default_rng(12345)
    tests = [rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 6, n) for n in (0, 5, 8, 127, 128, 129, 1000, 1846,
                                                                             8192, 8193, 9001, 20000, 40000)]
    stk, part = _workspace()
    # the candidate-row sums: numpy reduces the F-ordered A[:, mask] sequentially (checked, else no kernel)
    for rows, n in ((22, 800), (5, 9), (3, 1), (22, 3000)):
        a = rng.standard_normal((rows, n)) * 10.0 ** rng.uniform(-3, 6, (rows, n))
        sub = a[:, rng.random(n) < 0.7]
        for ref, row in zip(sub.sum(axis=1), sub):
            acc = row[0] if row.size else 0.0
            for v in row[1:]:
                acc += v
            if acc != ref:
                return None
    for block in (0, 8192):
        try:
            if all(_np_sum(x, x.size, block, stk, part) == x.sum()
                   and _np_sum(np.vstack([x, x])[1], x.size, block, stk, part) == np.vstack([x, x]).sum(axis=1)[1]
                   for x in tests):
                return block
        except Exception:  # noqa: BLE001 - compilation failure: numpy path
            return None
    return None


def enabled() -> bool:
    """Compiled kernels in use: Numba importable, numpy's summation reproduced, not off by ``CSPAPS_JIT=0``."""
    if os.environ.get("CSPAPS_JIT", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return _sum_block() is not None


def name() -> str:
    return "numba" if enabled() else "numpy"


# --------------------------------------------------------------------------- packed responses

def packed_responses(rs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ragged per-heliostat responses as flat arrays, cached on the response set per version.

    ``ptr`` (N + 1,): heliostat i owns cells ``cells[ptr[i]:ptr[i + 1]]`` and the row-major (K, S_i) block
    ``resp[K * ptr[i]: K * ptr[i + 1]]``."""
    key = (rs.version, len(rs.cells), id(rs.resp))
    cached = getattr(rs, "_packed", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    lens = np.fromiter((c.size for c in rs.cells), dtype=np.int64, count=len(rs.cells))
    ptr = np.zeros(len(rs.cells) + 1, dtype=np.int64)
    np.cumsum(lens, out=ptr[1:])
    if ptr[-1]:
        cells = np.concatenate([np.asarray(c, dtype=np.int64) for c in rs.cells])
        resp = np.concatenate([np.ascontiguousarray(r, dtype=np.float32).ravel() for r in rs.resp])
    else:
        cells, resp = np.zeros(0, np.int64), np.zeros(0, np.float32)
    out = (ptr, cells, resp)
    rs._packed = (key, out)
    return out


# --------------------------------------------------------------------------- uniformity local search

@_jit
def _local_search(order, available, ptr, cells, resp, n_k, dni_con, dni_obj, alpha, limit, h, area, a_h,
                  absorb_power, reach, move_cost, allow_defocus, mu, passes, flux_con, q, choice, sums, block,
                  stk, part):
    """In-place coordinate passes of ``uniformity._local_search``; returns how many steps a flux limit excluded a
    candidate in. Operation by operation the numpy loop: float32 constraint basis, float64 state, numpy sums."""
    f32_cut = np.float32(1e-3)
    f32_tiny = np.float32(1e-30)
    n = ptr.size - 1
    nnz = ptr[n]
    # static per call: the "touched" threshold of every candidate row (numpy: Kc > 1e-3 * max(rowmax, 1e-30), in
    # float32) and the per-support-cell data, gathered once so the passes read contiguous memory
    thr = np.empty(n * n_k, dtype=np.float32)
    for i in range(n):
        a = ptr[i]
        n_s = ptr[i + 1] - a
        dcon = np.float32(dni_con[i])
        for k in range(n_k):
            row = n_k * a + k * n_s
            peak = np.float32(0.0)
            for s in range(n_s):
                v = resp[row + s] * dcon
                if s == 0 or v > peak:
                    peak = v
            thr[i * n_k + k] = f32_cut * (peak if peak > f32_tiny else f32_tiny)
    in_h = np.empty(nnz, dtype=np.bool_)
    alpha_p = np.empty(nnz)
    lim_p = np.empty(nnz)
    for j in range(nnz):
        c = cells[j]
        in_h[j] = h[c]
        alpha_p[j] = alpha[c]
        lim_p[j] = limit[c] * (1 + 1e-9)
    s_max = 0
    for i in range(n):
        s_max = max(s_max, ptr[i + 1] - ptr[i])
    base_con = np.empty(s_max)
    base_q = np.empty(s_max)
    buf1 = np.empty(s_max)
    buf2 = np.empty(s_max)
    score = np.empty(n_k + 1)
    s1n = np.empty(n_k + 1)
    s2n = np.empty(n_k + 1)
    has_reach = reach.shape[0] > 0
    has_move = move_cost.shape[0] > 0
    limited = 0
    s1 = sums[0]
    s2 = sums[1]
    for _ in range(passes):
        moved = 0
        for i in order:
            if not available[i]:
                continue
            a = ptr[i]
            n_s = ptr[i + 1] - a
            if n_s == 0:
                continue
            old = choice[i]
            dcon = np.float32(dni_con[i])
            dobj = np.float32(dni_obj[i])
            r0 = n_k * a
            # the state without heliostat i, and the H part of the current state (numpy: old_q_h)
            nh = 0
            for s in range(n_s):
                c = cells[a + s]
                bc = flux_con[c]
                bq = q[c]
                if old >= 0:
                    r = resp[r0 + old * n_s + s]
                    bc = bc - np.float64(r * dcon)
                    bq = bq - np.float64(r * dobj) * alpha_p[a + s]
                base_con[s] = bc
                base_q[s] = bq
                if in_h[a + s]:
                    qc = q[c]
                    buf1[nh] = qc
                    buf2[nh] = qc * qc
                    nh += 1
            old_sum1 = _np_sum(buf1, nh, block, stk, part)
            old_sum2 = _np_sum(buf2, nh, block, stk, part)
            # every candidate row: flux limits on the cells it touches (an infeasible row's score is -inf, so it
            # stops at the first violation) and the H sums of the new absorbed flux. numpy forms new_q[:, hm] as an
            # F-ordered copy and reduces it along the strided axis: sequential sums starting from the first element
            any_limited = False
            for k in range(n_k + 1):
                ok = True
                m = 0
                sum1 = 0.0
                sum2 = 0.0
                if k < n_k:
                    row = r0 + k * n_s
                    t = thr[i * n_k + k]
                    for s in range(n_s):
                        r = resp[row + s]
                        v = r * dcon
                        if v > t and not (base_con[s] + np.float64(v) <= lim_p[a + s]):
                            ok = False
                            break
                        if in_h[a + s]:
                            nq = base_q[s] + np.float64(r * dobj) * alpha_p[a + s]
                            if m == 0:
                                sum1 = nq
                                sum2 = nq * nq
                            else:
                                sum1 += nq
                                sum2 += nq * nq
                            m += 1
                    if not ok:
                        any_limited = True
                    elif has_reach and not reach[i, k]:
                        ok = False
                else:
                    # defocus row: zeros touch no cell, so only the defocus rule decides
                    ok = allow_defocus or old < 0
                    for s in range(n_s):
                        if in_h[a + s]:
                            nq = base_q[s] + 0.0
                            if m == 0:
                                sum1 = nq
                                sum2 = nq * nq
                            else:
                                sum1 += nq
                                sum2 += nq * nq
                            m += 1
                if not ok:
                    score[k] = -np.inf
                    continue
                d1 = area * (sum1 - old_sum1)
                d2 = area * (sum2 - old_sum2)
                s1n[k] = s1 + d1
                s2n[k] = s2 + d2
                m1 = s1n[k] / a_h
                var = s2n[k] / a_h - m1 * m1
                std = np.sqrt(var if var > 0.0 else 0.0)
                p_new = absorb_power[i, k] if k < n_k else 0.0
                sc = p_new - mu * a_h * std
                if has_move and k < n_k:
                    sc -= move_cost[i, k]
                score[k] = sc
            if any_limited:
                limited += 1
            best = 0
            for k in range(1, n_k + 1):
                if score[k] > score[best]:
                    best = k
            cur = n_k if old < 0 else old
            cur_score = score[cur]
            tolerance = 1e-9 * max(abs(cur_score), 1.0) if np.isfinite(cur_score) else 0.0
            if best != cur and np.isfinite(score[best]) and score[best] > cur_score + tolerance:
                if best < n_k:
                    row = r0 + best * n_s
                    for s in range(n_s):
                        c = cells[a + s]
                        r = resp[row + s]
                        flux_con[c] = base_con[s] + np.float64(r * dcon)
                        q[c] = base_q[s] + np.float64(r * dobj) * alpha_p[a + s]
                else:
                    for s in range(n_s):
                        c = cells[a + s]
                        flux_con[c] = base_con[s] + 0.0
                        q[c] = base_q[s] + 0.0
                s1 = s1n[best]
                s2 = s2n[best]
                choice[i] = -1 if best == n_k else best
                moved += 1
        if moved == 0:
            break
    sums[0] = s1
    sums[1] = s2
    return limited


def local_search(prob, state, mu: float, passes: int, absorb_power: np.ndarray, order,
                 move_cost: np.ndarray | None) -> None:
    """Compiled ``uniformity._local_search`` on a ``uniformity._State`` (updated in place)."""
    rs = prob.rs
    ptr, cells, resp = packed_responses(rs)
    reach = np.ascontiguousarray(prob.reach, dtype=np.bool_) if prob.reach is not None else np.zeros((0, 0), np.bool_)
    sums = np.array([state.s1, state.s2], dtype=float)
    choice = np.ascontiguousarray(state.choice, dtype=np.int64)
    limited = _local_search(
        np.ascontiguousarray(order, dtype=np.int64), np.ascontiguousarray(prob.available, dtype=np.bool_),
        ptr, cells, resp, int(prob.k), np.ascontiguousarray(prob.dni_con, dtype=float),
        np.ascontiguousarray(prob.dni_obj, dtype=float), np.ascontiguousarray(state.alpha, dtype=float),
        np.ascontiguousarray(prob.limit, dtype=float), np.ascontiguousarray(state.h, dtype=np.bool_),
        float(state.area), float(state.a_h), np.ascontiguousarray(absorb_power, dtype=float), reach,
        np.zeros((0, 0)) if move_cost is None else np.ascontiguousarray(move_cost, dtype=float),
        bool(prob.case.aps.allow_defocus), float(mu), int(passes), state.flux_con, state.q, choice, sums,
        int(_sum_block()), *_workspace())
    state.choice[:] = choice
    state.s1, state.s2 = float(sums[0]), float(sums[1])
    if limited:
        prob.flux_limited = int(getattr(prob, "flux_limited", 0)) + int(limited)


# --------------------------------------------------------------------------- greedy APS passes

@_jit
def _greedy_pass(order, available, ptr, cells, resp, n_k, dni_con, limit, score_all, p_ref, tol, reach,
                 allow_defocus, choice, flux, counts):
    """``aps.solve_greedy``'s ``reassign`` over ``order``, in place: take the heliostat's beam out, ``choose`` the
    best flux-feasible candidate (ties: lowest peak ratio), put it back. counts += (moves, flux-limited steps).
    Same float32 constraint basis, comparisons and divisions as the numpy code (no sums), so bit-identical."""
    f32_cut = np.float32(1e-3)
    ratio = np.empty(n_k)
    feasible = np.empty(n_k, dtype=np.bool_)
    has_reach = reach.shape[0] > 0
    for i in order:
        a = ptr[i]
        n_s = ptr[i + 1] - a
        r0 = n_k * a
        dcon = np.float32(dni_con[i])
        old = choice[i]
        if old >= 0:
            row = r0 + old * n_s
            for s in range(n_s):
                c = cells[a + s]
                flux[c] = flux[c] - np.float64(resp[row + s] * dcon)
        # choose(i)
        if not available[i]:
            best_k = -1
        elif n_s == 0 or dni_con[i] <= 0:
            best_k = 0
            for k in range(1, n_k):
                if score_all[i, k] > score_all[i, best_k]:
                    best_k = k
        else:
            any_infeasible = False
            for k in range(n_k):
                row = r0 + k * n_s
                peak = resp[row] * dcon
                for s in range(1, n_s):
                    v = resp[row + s] * dcon
                    if v > peak:
                        peak = v
                thr = f32_cut * peak
                r_max = 0.0
                for s in range(n_s):
                    v = resp[row + s] * dcon
                    if v > thr:
                        c = cells[a + s]
                        x = (flux[c] + np.float64(v)) / limit[c]
                    else:
                        x = 0.0
                    if s == 0 or x > r_max:
                        r_max = x
                ratio[k] = r_max
                feasible[k] = r_max <= 1.0
                if not feasible[k]:
                    any_infeasible = True
            if any_infeasible:
                counts[1] += 1
            any_feasible = False
            best = -np.inf
            for k in range(n_k):
                if has_reach and not reach[i, k]:
                    feasible[k] = False
                if feasible[k]:
                    if not any_feasible or score_all[i, k] > best:
                        best = score_all[i, k]
                    any_feasible = True
            if any_feasible:
                cut = best - tol * max(p_ref[i], 1e-9)
                best_k = -1
                for k in range(n_k):
                    if feasible[k] and score_all[i, k] >= cut:
                        if best_k < 0 or ratio[k] < ratio[best_k]:
                            best_k = k
            elif allow_defocus:
                best_k = -1
            else:
                best_k = 0
                for k in range(1, n_k):
                    if ratio[k] < ratio[best_k]:
                        best_k = k
        choice[i] = best_k
        if best_k >= 0:
            row = r0 + best_k * n_s
            for s in range(n_s):
                c = cells[a + s]
                flux[c] = flux[c] + np.float64(resp[row + s] * dcon)
        if best_k != old:
            counts[0] += 1


def greedy_pass(prob, order, choice: np.ndarray, flux: np.ndarray, score_all: np.ndarray, p_ref: np.ndarray,
                tol: float, stats: dict) -> None:
    """Compiled ``reassign`` over ``order`` (``aps.solve_greedy``); ``choice``, ``flux`` and ``stats`` in place."""
    ptr, cells, resp = packed_responses(prob.rs)
    reach = np.ascontiguousarray(prob.reach, dtype=np.bool_) if prob.reach is not None else np.zeros((0, 0), np.bool_)
    counts = np.zeros(2, dtype=np.int64)
    _greedy_pass(np.ascontiguousarray(order, dtype=np.int64), np.ascontiguousarray(prob.available, dtype=np.bool_),
                 ptr, cells, resp, int(prob.k), np.ascontiguousarray(prob.dni_con, dtype=float),
                 np.ascontiguousarray(prob.limit, dtype=float), np.ascontiguousarray(score_all, dtype=float),
                 np.ascontiguousarray(p_ref, dtype=float), float(tol), reach, bool(prob.case.aps.allow_defocus),
                 choice, flux, counts)
    stats["moves"] += int(counts[0])
    stats["flux_limited"] += int(counts[1])
