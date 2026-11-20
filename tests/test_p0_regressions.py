"""Regression cases for the three P0 issues of the advisor review (baseline 0d9292d)."""

import numpy as np
import pandas as pd
import pytest

from cspaps.aps import optimize_aims
from cspaps.field import Field, build_field, save_layout_csv
from cspaps.layout_opt import optimize_layout
from cspaps.realtime import RealtimeResult, replay_times
from cspaps.receiver import build_grid
from cspaps.simulate import evaluate, sun_state


# ---------------------------------------------------------------- P0-01 flux feasibility

@pytest.fixture
def p0_01_case(example_case):
    """Advisor case: heliostat at (0, 80, 5) m, 2025-06-21 13:40, shading off, centre candidate only,
    limit 10 kW/m2, 12 x 8 receiver grid."""
    c = example_case
    c.shading.enabled = False
    c.aps.azimuth_offsets_deg = [0.0]
    c.aps.n_height_levels = 1
    c.receiver.max_flux_kw_m2 = 10
    c.receiver.n_azimuth, c.receiver.n_height = 12, 8
    return c, Field(np.array([[0.0, 80.0, 5.0]]))


def test_p0_01_coarse_grid_misses_the_peak(p0_01_case):
    case, fld = p0_01_case
    ts, el, az, dni = sun_state(case, "2025-06-21 13:40")
    coarse = evaluate(case, fld, el, az, dni, grid=build_grid(case.receiver, 12, 8))
    fine = evaluate(case, fld, el, az, dni, grid=build_grid(case.receiver, 360, 160))
    assert coarse.flux.max() / 1e3 == pytest.approx(4.18, abs=0.05)  # reported "feasible" before the fix
    assert fine.flux.max() / 1e3 == pytest.approx(19.49, abs=0.1)


def test_p0_01_centre_aim_is_intercepted(p0_01_case):
    case, fld = p0_01_case
    ts, el, az, dni = sun_state(case, "2025-06-21 13:40")
    center = optimize_aims(case, fld, el, az, dni, method="center")
    assert center.status == "infeasible" and not center.feasible
    assert center.check.peak_flux_kw_m2 >= 19.49 - 0.1  # verified peak (refined, >= the 360 x 160 value)
    assert center.check.exceedance_kw_m2 > 9.0
    assert center.check.hotspot_panel == 1
    greedy = optimize_aims(case, fld, el, az, dni, method="greedy")
    assert greedy.status == "ok" and greedy.n_defocused == 1  # the only feasible plan: defocus
    kf = optimize_aims(case, fld, el, az, dni, method="kfactor")  # cannot defocus -> clear failure state
    assert kf.status == "infeasible"


def test_p0_01_verified_peak_matches_fine_grid(example_case):
    fld = build_field(example_case)
    ts, el, az, dni = sun_state(example_case, "2025-06-21 13:40")
    res = optimize_aims(example_case, fld, el, az, dni, method="greedy")
    fine = evaluate(example_case, fld, el, az, dni, aim=res.aim, active=res.active,
                    grid=build_grid(example_case.receiver, 360, 160))
    assert res.status == "ok"
    assert fine.flux.max() / 1e3 <= res.check.peak_flux_kw_m2 * 1.002  # check is at least as high as 360 x 160
    assert res.check.peak_flux_kw_m2 <= example_case.receiver.max_flux_kw_m2 * (1 - example_case.aps.safety_margin)


# ---------------------------------------------------------------- P0-02 layout round trip

def test_p0_02_optimised_layout_survives_export_and_reload(small_case, tmp_path):
    res = optimize_layout(small_case)
    fld = Field(res.best.xyz)
    assert fld.n == 120
    # the layout contains heliostats exactly on the 40 m exclusion boundary (r_min = exclusion radius)
    r = np.hypot(fld.xyz[:, 0], fld.xyz[:, 1])
    path = tmp_path / "layout.csv"
    save_layout_csv(fld, path)
    small_case.field.layout_file = str(path)
    small_case.field.generator = None
    again = build_field(small_case)
    assert again.n == 120 and not again.dropped
    np.testing.assert_array_equal(again.ids, fld.ids)
    np.testing.assert_allclose(again.xyz, fld.xyz, atol=1e-6)
    assert r.min() >= small_case.field.exclusion_radius - 1e-3


def test_p0_02_heliostat_on_the_exclusion_boundary(small_case, tmp_path):
    ang = np.radians([10.0, 100.0, 200.0])
    xyz = np.column_stack([40.0 * np.sin(ang), 40.0 * np.cos(ang), np.full(3, 4.0)])  # exactly on r = 40 m
    xyz_rounded = np.round(xyz, 3)  # what a 3-decimal export produced
    path = tmp_path / "edge.csv"
    pd.DataFrame({"id": [7, 8, 9], "x": xyz_rounded[:, 0], "y": xyz_rounded[:, 1], "z": xyz_rounded[:, 2]}).to_csv(path, index=False)
    small_case.field.layout_file = str(path)
    small_case.field.generator = None
    fld = build_field(small_case)
    assert fld.n == 3 and fld.ids.tolist() == [7, 8, 9]
    # a heliostat really inside the exclusion zone is removed and reported
    pd.DataFrame({"id": [1, 2], "x": [0.0, 0.0], "y": [39.0, 60.0], "z": [4.0, 4.0]}).to_csv(path, index=False)
    fld = build_field(small_case)
    assert fld.ids.tolist() == [2]
    assert fld.dropped and fld.dropped[0]["id"] == 1 and "exclusion" in fld.dropped[0]["reason"]


# ---------------------------------------------------------------- P0-03 replay time accounting

def _constant_replay(duration_s: float, step_s: float, power_mw: float, over: bool) -> RealtimeResult:
    t0 = pd.Timestamp("2025-06-21 12:00", tz="Asia/Shanghai")
    times, intervals = replay_times(t0, duration_s, step_s)
    table = pd.DataFrame({
        "time": times, "interval_s": intervals, "strategy": "center", "dni_mean_w_m2": 900.0, "dni_min_w_m2": 900.0,
        "p_intercepted_mw": power_mw, "p_absorbed_mw": power_mw, "peak_flux_kw_m2": 1100.0 if over else 800.0,
        "peak_to_limit": 1.1 if over else 0.8, "over_limit": over, "exceedance_kw_m2": 100.0 if over else 0.0,
        "hotspot_panel": 1, "n_defocused": 0, "n_moved": 0, "plan_state": "n/a", "plan_reasons": "",
        "engine_total_s": np.nan, "response_reused": np.nan,
    })
    return RealtimeResult(table, {}, None, step_s, duration_s)


@pytest.mark.parametrize("duration,expected_mwh,n_samples", [(180, 3.0, 4), (150, 2.5, 4)])
def test_p0_03_energy_uses_intervals(duration, expected_mwh, n_samples):
    res = _constant_replay(duration, 60, 60.0, over=True)
    s = res.summary()["center"]
    assert len(res.table) == n_samples
    assert s["energy_mwh"] == pytest.approx(expected_mwh)
    assert s["time_over_limit_s"] == pytest.approx(duration)  # all over the limit -> whole replay
    assert s["replay_duration_s"] == pytest.approx(duration)
    assert "sample-and-hold" in s["over_limit_basis"]


def test_p0_03_replay_times():
    t0 = pd.Timestamp("2025-06-21 12:00", tz="Asia/Shanghai")
    times, d = replay_times(t0, 150, 60)
    assert [(t - t0).total_seconds() for t in times] == [0, 60, 120, 150]
    assert d.tolist() == [60, 60, 30, 0]
