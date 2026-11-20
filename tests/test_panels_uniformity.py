import math

import numpy as np
import pytest

from cspaps.aps import build_problem, optimize_aims, solve
from cspaps.field import build_field
from cspaps.receiver import build_grid, incident_limit_w_m2, panel_of
from cspaps.simulate import evaluate, sun_state
from cspaps.uniformity import edge_band_bound, solve_uniform, sweep, uniformity_metrics

TIME = "2025-06-21 13:00"


def test_panel_index_and_limits(small_case):
    rec = small_case.receiver
    rec.n_panels, rec.panel_start_azimuth_deg = 4, -45.0
    rec.panel_max_flux_kw_m2 = [800.0, 600.0, 400.0, 200.0]
    assert panel_of(rec, [0.0, 44.9, 45.0, 135.0, 225.0, 315.0]).tolist() == [0, 0, 1, 2, 3, 0]
    np.testing.assert_allclose(incident_limit_w_m2(rec, [0.0, 90.0, 180.0, 270.0], 60.0), [8e5, 6e5, 4e5, 2e5])
    rec.flux_limit_basis, rec.absorptivity = "absorbed", 0.8  # absorbed limit -> higher incident limit
    assert float(incident_limit_w_m2(rec, 0.0, 60.0)) == pytest.approx(8e5 / 0.8)
    rec.panel_flux_bias = [2.0, 1.0, 1.0, 1.0]  # plant measures twice the modelled flux on panel 1
    assert float(incident_limit_w_m2(rec, 0.0, 60.0)) == pytest.approx(8e5 / 0.8 / 2.0)
    rec.limit_height_profile = [0.5, 1.0]
    z_bot = rec.center[2] - 0.5 * rec.height
    assert float(incident_limit_w_m2(rec, 90.0, z_bot)) == pytest.approx(0.5 * 6e5 / 0.8)


def test_two_panels_with_different_limits(small_case):
    """Advisor P1-03: two panels with different allowed flux; limits and hotspot location are respected."""
    rec = small_case.receiver
    rec.n_panels, rec.panel_start_azimuth_deg = 2, -90.0  # panel 1 = north half, panel 2 = south half
    rec.panel_max_flux_kw_m2 = [500.0, 150.0]
    small_case.aps.auto_grid = True  # solver grid refined to the narrowest beam
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    res = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    assert res.status == "ok"
    fine = evaluate(small_case, fld, el, az, dni, aim=res.aim, active=res.active, grid=build_grid(rec, 240, 80))
    kw = fine.flux / 1e3
    south = fine.grid.to_map(fine.grid.panel) == 1
    assert kw[south].max() <= 150.0 + 1e-6
    assert kw[~south].max() <= 500.0 + 1e-6
    assert kw[~south].max() > 150.0  # the north panel uses its higher limit
    s = fine.summary()
    assert s["absorbed_power_mw"] == pytest.approx(s["intercepted_power_mw"] * rec.absorptivity, abs=2e-4)
    center = evaluate(small_case, fld, el, az, dni)  # without APS the south panel is over its limit
    assert center.summary()["peak_to_limit"] > 1.0


def test_uniformity_metrics_definitions():
    flat = np.full((4, 6), 3.0)
    m = uniformity_metrics(flat)
    assert m["cv_2d"] == pytest.approx(0.0) and m["cv_circ"] == pytest.approx(0.0)
    two = np.vstack([np.full((2, 6), 1.0), np.full((2, 6), 3.0)])  # bottom 1, top 3: only a height gradient
    m = uniformity_metrics(two)
    assert m["cv_2d"] == pytest.approx(0.5)
    assert m["cv_height"] == pytest.approx(0.5)
    assert m["cv_circ"] == pytest.approx(0.0)  # hidden in the circumferential view
    mask = np.zeros_like(two, bool)
    mask[2:, :] = True  # H = top half only
    assert uniformity_metrics(two, mask)["cv_2d"] == pytest.approx(0.0)


def test_uniformity_constraint_trades_little_power(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    prob = build_problem(small_case, fld, el, az, dni)
    base = solve(prob, "greedy")
    st0 = solve_uniform(prob, base.choice, max_cv=10.0)
    target = 0.9 * st0.cv_2d
    r = solve_uniform(prob, base.choice, max_cv=target)
    assert r.status == "feasible" and r.cv_2d <= target + 1e-9
    assert r.power_abs_w >= 0.97 * st0.power_abs_w
    assert (prob.flux(r.choice) <= prob.limit * (1 + 1e-9)).all()  # flux limits kept
    curve = sweep(prob, base.choice, [target, 0.95 * st0.cv_2d, 10.0])
    powers = [c["power_abs_mw"] for c in curve]
    assert powers == sorted(powers)  # looser epsilon never gives less power here


def test_uniformity_in_aps_solve(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    free = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    small_case.uniformity.enabled = True
    snap = evaluate(small_case, fld, el, az, dni, aim=free.aim, active=free.active)
    small_case.uniformity.max_cv = 0.9 * snap.summary()["cv_2d"]
    res = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    assert res.info["uniformity_status"] == "feasible"
    assert res.status in ("ok", "margin_reduced")
    small_case.uniformity.min_load_mw = 1e6  # unreachable minimum load -> receiver off
    off = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    assert off.info["uniformity_status"] == "below_min_load" and off.n_defocused == fld.n


def test_edge_band_bound(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    prob = build_problem(small_case, fld, el, az, dni)
    base = solve(prob, "greedy")
    b = edge_band_bound(prob, 0.995 * base.power_w)
    assert 0.0 <= b["band_share_upper_bound_u"] <= 1.0
    assert b["area_share_a"] == pytest.approx(0.25)
    assert b["cv_2d_lower_bound"] == pytest.approx(max(0.0, 0.25 - b["band_share_upper_bound_u"]) / math.sqrt(0.25 * 0.75))
    # any concrete plan respects the bound: its band share cannot exceed u
    snap = evaluate(small_case, fld, el, az, dni, aim=base.aim, active=base.active, grid=build_grid(small_case.receiver, 72, 48))
    q = snap.flux
    nb = 6  # 1/8 of 48 rows at each end
    share = (q[:nb].sum() + q[-nb:].sum()) / q.sum()
    assert share <= b["band_share_upper_bound_u"] * 1.02 + 1e-3
    loose = edge_band_bound(prob, 0.2 * base.power_w)  # a lower power floor allows more edge power
    assert loose["band_share_upper_bound_u"] >= b["band_share_upper_bound_u"]
    assert edge_band_bound(prob, 10 * base.power_w)["feasible"] is False


def test_exact_benchmark_monotone(small_case):
    pytest.importorskip("pyscipopt")
    from cspaps.uniformity import exact_benchmark

    rec = small_case.receiver
    rec.n_azimuth, rec.n_height, rec.n_panels, rec.panel_start_azimuth_deg = 24, 8, 4, -45.0
    small_case.aps.n_height_levels, small_case.aps.azimuth_offsets_deg = 3, [-6.0, 0.0, 6.0]
    small_case.uniformity.panels = [1]
    full = build_field(small_case)
    fld = full.subset(np.flatnonzero((full.xyz[:, 1] > 60) & (np.abs(full.xyz[:, 0]) < 30))[:6])
    ts, el, az, dni = sun_state(small_case, TIME)
    prob = build_problem(small_case, fld, el, az, dni)
    results = [exact_benchmark(prob, eps, time_limit_s=60) for eps in (0.7, 0.9, 1.2)]
    powers = [r.get("power_abs_mw", 0.0) if r["status"] == "proven_optimal" else 0.0 for r in results]
    assert all(r["status"] in ("proven_optimal", "infeasible_except_shutdown") for r in results)
    assert powers == sorted(powers)  # proven optimum non-decreasing as epsilon relaxes
