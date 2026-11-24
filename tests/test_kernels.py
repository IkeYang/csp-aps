"""Compiled kernels (Numba) must reproduce the numpy reference bit for bit."""

import numpy as np
import pytest

from cspaps.aps import build_problem, solve_greedy
from cspaps.field import build_field
from cspaps.simulate import sun_state
from cspaps.uniformity import _State, _local_search, region_of_grid

kernels = pytest.importorskip("cspaps.kernels")
pytestmark = pytest.mark.skipif(not kernels.enabled(), reason="Numba not installed or numpy summation not reproduced")


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 127, 128, 129, 255, 1000, 1846, 8192, 8193, 9001, 20000])
def test_numpy_exact_sum(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 6, n)
    stk, part = kernels._workspace()
    assert kernels._np_sum(x, n, kernels._sum_block(), stk, part) == x.sum()


def _problem(case, time="2025-06-21 09:30", **aps):
    for k, v in aps.items():
        setattr(case.aps, k, v)
    fld = build_field(case)
    _, el, az, dni = sun_state(case, time)
    prob = build_problem(case, fld, el, az, dni)
    choice = solve_greedy(prob, case.aps)
    choice = getattr(choice, "choice", choice[0] if isinstance(choice, tuple) else choice)
    return prob, np.asarray(choice), fld


def _run(monkeypatch, prob, start, mu, passes, **kw):
    states = []
    counts = []
    h_mask = region_of_grid(prob.case, prob.grid, prob.case.uniformity)
    order = np.argsort(-prob.sigma[:, prob.center_k], kind="stable")
    for jit in ("0", "1"):
        monkeypatch.setenv("CSPAPS_JIT", jit)
        prob.flux_limited = 0
        st = _State(prob, h_mask, start)
        _local_search(prob, st, mu, passes, prob.absorbed_power, order, **kw)
        states.append(st)
        counts.append(prob.flux_limited)
    a, b = states
    assert np.array_equal(a.choice, b.choice)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.flux_con, b.flux_con)
    assert a.s1 == b.s1 and a.s2 == b.s2
    assert counts[0] == counts[1]
    return a, counts[0]


@pytest.mark.parametrize("mu", [0.0, 0.3, 5.0])
def test_local_search_kernel_is_bit_identical(monkeypatch, small_case, mu):
    prob, start, _ = _problem(small_case)
    st, _ = _run(monkeypatch, prob, start, mu, passes=6)
    assert not np.array_equal(st.choice, start) or mu == 0.0  # the search moved something


def test_local_search_kernel_with_binding_limits_reach_and_moves(monkeypatch, small_case):
    """Tight flux limits (candidates excluded and counted), a slew budget and a move penalty."""
    small_case.aps.move_penalty_per_m = 0.02
    prob, start, fld = _problem(small_case, "2025-12-21 10:00")
    # limits just above the start plan's own peak: moves that concentrate flux are excluded
    peak = prob.flux(start).max()
    prob.rs.limit_nom = np.full_like(prob.rs.limit_nom, peak / (1.0 - prob.margin) * 1.0001)
    rng = np.random.default_rng(0)
    prob.reach = rng.random((prob.n, prob.k)) < 0.7
    prob.reach[np.arange(prob.n), np.where(start >= 0, start, prob.center_k)] = True
    prev_xyz = prob.aim_xyz[np.arange(prob.n), rng.integers(0, prob.k, prob.n)]
    prev_active = rng.random(prob.n) < 0.8
    st, limited = _run(monkeypatch, prob, start, 0.8, passes=8, prev_aim_xyz=prev_xyz, prev_active=prev_active)
    assert limited > 0  # the limits excluded candidates: the counter path is exercised


def test_local_search_kernel_after_hotspot_points(monkeypatch, small_case):
    """Constraint points added after verification (outside the grid, not in H) take part like grid cells."""
    prob, start, _ = _problem(small_case)
    rec = small_case.receiver
    prob.rs.add_points(np.linspace(0, 350, 12), np.full(12, rec.center[2] + 0.3 * rec.height))
    _run(monkeypatch, prob, start, 0.5, passes=4)


def test_compiled_kernel_does_not_change_formal_scores(monkeypatch, small_case):
    from cspaps.layout_value import inner_settings, solve_instant

    small_case.uniformity.enabled = True
    small_case.uniformity.mode = "weighted"
    small_case.uniformity.weight = 0.3
    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, "2025-03-20 15:00")
    out = []
    for jit in ("0", "1"):
        monkeypatch.setenv("CSPAPS_JIT", jit)
        r = solve_instant(small_case, fld, el, az, dni, inner_settings(small_case))
        out.append(r)
    a, b = out
    assert a.status == b.status and a.p_abs_w == b.p_abs_w and a.u_w == b.u_w
    assert np.array_equal(a.choice, b.choice)
