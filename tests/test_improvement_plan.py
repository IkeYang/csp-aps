"""Acceptance tests of docs/APS-IMPROVEMENT-PLAN.md, section "先修正会改变排名的计算"."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from cspaps import case_from_dict, load_case, sun
from cspaps import workflows as wf
from cspaps.field import Field
from cspaps.layout_value import (
    annual_value, compare_values, evaluate_layout_value, inner_case, inner_settings, recompute_from_csv,
)
from cspaps.receiver import build_grid
from cspaps.scenarios import build_scenarios
from cspaps.simulate import annual, evaluate, sun_state

from .conftest import ROOT, small_case_dict

WRONG = ROOT / "cases" / "wrong_site"


# ----------------------------------------------------------------------------- hours

def test_actual_hours_and_energy_weight_are_recorded_separately():
    """20 clusters of the Aksai demo year: 4,028 actual hours vs 2,800.62 irradiance-equivalent hours."""
    case = load_case(ROOT / "cases" / "aksai_demo.yaml")
    hourly = sun.hourly_samples(case)
    t = sun.sun_bins(hourly, 20, seed=7, weighting="time_mean")
    e = sun.sun_bins(hourly, 20, seed=7, weighting="energy_weighted")
    assert len(hourly) == 4028 and hourly.weight_h.sum() == pytest.approx(4028.0)
    assert t.weight_h.sum() == pytest.approx(4028.0) and np.allclose(t.actual_hours, t.weight_h)
    assert e.weight_h.sum() == pytest.approx(2800.62, abs=0.01) and e.actual_hours.sum() == pytest.approx(4028.0)
    # both conventions keep the DNI energy of every cluster; only the representative DNI differs
    assert t.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-12) == e.dni_hours
    np.testing.assert_allclose(t.dni * t.weight_h, e.dni * e.weight_h, rtol=1e-12)
    assert np.all(t.dni <= e.dni + 1e-9)  # time mean <= energy-weighted mean (Cauchy-Schwarz)
    sset = build_scenarios(case, 20)
    rec = sset.record()
    assert rec["actual_hours"] == pytest.approx(4028.0) and rec["total_hours"] == pytest.approx(4028.0)
    assert "actual_hours" in sset.table() and rec["hours_basis"].startswith("actual hours")


# ----------------------------------------------------------------------------- the wrong-site example

def _brute_force(case, xyz, el, az, dni, lam):
    """All 3 x 3 joint poses (no defocus), evaluated and verified on the fixed receiver grid."""
    from cspaps.aps import build_problem

    grid = build_grid(case.receiver)
    fld = Field(xyz)
    prob = build_problem(case, fld, el, az, dni, grid=grid)
    best = -np.inf
    for ch in product(range(prob.k), repeat=len(xyz)):
        ch = np.asarray(ch)
        snap = evaluate(case, fld, el, az, dni, aim=prob.aims(ch), active=np.ones(len(xyz), bool), grid=grid)
        if snap.verify(case.receiver).ok:
            th = snap.thermal_objective(case.uniformity)
            best = max(best, th["power_w"] - lam * th["spread_w"])
    return best


@pytest.mark.parametrize("yaml_name,expected", [
    ("wrong_site.yaml", {"A": 41.449, "B": 84.785}),       # every mirror reads the map at its position: B wins
    ("wrong_site_mean.yaml", {"A": 72.817, "B": 59.142}),  # the field mean of every map: A wins
])
def test_wrong_site_example_reproduces_the_plan_table(yaml_name, expected, tmp_path):
    case = load_case(WRONG / yaml_name)
    res = wf.task_layout_value(case, tmp_path, {"A": "layout_A.csv", "B": "layout_B.csv"})
    rank = res["ranking"]
    got = {r["name"]: r["value_mwh"] for _, r in rank.iterrows()}
    assert got == pytest.approx(expected, abs=5e-4)
    assert list(rank["name"]) == sorted(expected, key=expected.get, reverse=True)
    sset = res["scenario_set"]
    assert len(sset) == 20 and sset.record()["actual_hours"] == pytest.approx(4028.0)
    # 40 "layout x scenario" optima, enumerated independently of the solver
    lam = 0.05
    c = inner_case(case, inner_settings(case))
    s = sset.samples
    for name, v in res["values"].items():
        t = v.table()
        assert (t["status"] == "feasible").all() and t["proven_optimal"].all()
        for i in range(len(sset)):
            dni = sset.dni_at(i, v.xyz[:, :2])
            j = _brute_force(c, v.xyz, float(s.elevation[i]), float(s.azimuth[i]), dni, lam)
            assert t["j_inner_mw"].iloc[i] * 1e6 == pytest.approx(j, rel=1e-9)
        assert t["hours"].sum() == pytest.approx(4028.0)
    # the exported records alone give the same scores (full precision) and the same ranking
    out = Path(res["run_dir"]) / "values"
    re = {n: recompute_from_csv(out / n / "scenario_values.csv")["value_mwh"] for n in ("A", "B")}
    assert re == {n: res["values"][n].totals()["value_mwh"] for n in ("A", "B")}
    assert sorted(re, key=re.get, reverse=True) == list(rank["name"])


# ----------------------------------------------------------------------------- one feasibility rule

def _single_mirror_case(margin_target=0.98):
    d = small_case_dict()
    d["field"] = {"layout_file": None, "generator": {"type": "radial_staggered", "r_min": 60, "r_max": 61}}
    case = case_from_dict(d, base_dir=ROOT)
    case.aps.safety_margin = 0.05
    case.aps.n_height_levels, case.aps.azimuth_offsets_deg = 1, [0.0]
    fld = Field(np.array([[0.0, 70.0, case.heliostat.pivot_height]]))
    ts, el, az, dni = sun_state(case, "2025-06-21 12:30")
    snap = evaluate(case, fld, el, az, dni)
    peak = snap.verify(case.receiver).peak_flux_kw_m2
    case.receiver.max_flux_kw_m2 = peak / margin_target  # the plan reaches 98 % of the nominal limit
    return case, fld, el, az, dni


def test_single_mirror_boundary_case_uses_the_margin_everywhere():
    case, fld, el, az, dni = _single_mirror_case()
    snap = evaluate(case, fld, el, az, dni)
    chk = snap.verify(case.receiver)
    assert chk.status == "margin_reduced" and chk.peak_ratio == pytest.approx(0.98, rel=1e-3)
    samples = sun.Samples(np.array([el]), np.array([az]), np.array([dni]), np.array([3.0]))
    center = annual(case, fld, samples=samples, aiming="center", workers=1)
    row = center.timeseries.iloc[0]
    assert row["margin_used"] and not row["over_limit"] and not row["flux_feasible"]
    assert center.summary()["feasible_absorbed_energy_mwh"] == 0.0
    aps = annual(case, fld, samples=samples, aiming="greedy", workers=1).timeseries.iloc[0]
    assert (not aps["operational_feasible"]) or aps["peak_to_limit"] <= 0.95 + 1e-9
    # layout value: the same rule (only plans within the margin count; the mirror is defocused)
    sset = build_scenarios(case, 1)
    sset.samples.elevation[:], sset.samples.azimuth[:], sset.samples.dni[:] = el, az, dni
    v = evaluate_layout_value(case, fld.xyz, sset, "one", workers=1)
    p = v.records[0].plan
    assert p.status in ("feasible", "feasible_shutdown") and (p.status != "feasible" or p.peak_ratio <= 1.0 + 1e-9)
    assert p.status == "feasible_shutdown" and v.totals()["energy_mwh"] == 0.0


# ----------------------------------------------------------------------------- statuses, coverage, intervals

def _case(**scen):
    d = small_case_dict()
    d["scenarios"] = {"n_scenarios": 4, "map_model": "half_swap", **scen}
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["layout_opt"]["uniformity_weight"] = 0.3
    return case_from_dict(d, base_dir=ROOT)


def _ring(n, r, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.sin(a), r * np.cos(a), np.full(n, 4.0)])


def test_cancelled_evaluation_reports_coverage_and_pending_rank():
    case = _case()
    sset = build_scenarios(case)
    done = {"n": 0}

    def progress(msg):
        done["n"] = int(msg.rsplit(":", 1)[1].split("/")[0])  # "APS scenarios B: 2/4"

    part = evaluate_layout_value(case, _ring(20, 60.0), sset, "B", workers=1, progress=progress,
                                 cancel=lambda: done["n"] >= 2)  # the user cancels after two scenarios
    full = evaluate_layout_value(case, _ring(20, 70.0), sset, "A", workers=1)
    t = part.totals()
    assert t["ranking_status"] == "pending" and t["n_solved"] == 2 and t["coverage"] == pytest.approx(0.5)
    assert np.isnan(t["value_mwh"]) and t["value_partial_mwh"] > 0
    rank = compare_values([full, part])
    assert list(rank["name"]) == ["A", "B"] and rank["rank"].iloc[0] == 1 and np.isnan(rank["rank"].iloc[1])


def test_value_interval_brackets_the_found_value():
    case = _case()
    sset = build_scenarios(case)
    a = evaluate_layout_value(case, _ring(20, 60.0), sset, "A", workers=1)
    b = evaluate_layout_value(case, _ring(20, 85.0, 0.1), sset, "B", workers=1)
    for v in (a, b):
        lo, hi = v.totals()["value_interval_mwh"]
        assert lo == pytest.approx(v.totals()["value_mwh"]) and hi >= lo
    rank = compare_values([a, b])
    second = rank.iloc[1]
    assert second["interval_overlaps_leader"] == (second["value_interval_mwh"][1] >= rank.iloc[0]["value_mwh"])
    assert a.totals(0.5)["value_interval_mwh"] is None  # the bound is on sum h J_s: equal weights only


def test_opposite_hotspots_are_penalised_in_each_scenario():
    """Morning and afternoon hotspots on opposite sides do not cancel: Pi = sum h U_s >= U of the mean map."""
    case = _case(map_model="half_swap", map_contrast=0.6, n_scenarios=4)
    sset = build_scenarios(case)
    v = evaluate_layout_value(case, _ring(24, 65.0), sset, "L", workers=1)
    grid = build_grid(case.receiver, v.settings.grid_shape[1], v.settings.grid_shape[0])
    alpha = grid.to_map(grid.alpha)
    acc, hours, pen = np.zeros(grid.shape), 0.0, 0.0
    for r in v.records:
        if r.plan.counts:
            acc += r.hours * r.plan.flux * alpha
            hours += r.hours
            pen += r.hours * r.plan.u_w
    mean = acc / hours
    q = mean.ravel()
    u_mean_map = q.std() * q.size * grid.cell_area  # A_H sigma_H of the hour-averaged map
    assert pen / hours > 1.2 * u_mean_map  # averaging first would hide a large part of the non-uniformity
    t = v.table()
    assert annual_value(t, 0.3)["penalty_mwh"] == pytest.approx(pen / 1e6, rel=1e-9)


def test_shared_lambda_is_the_default():
    case = _case()
    case.layout_opt.uniformity_weight = 0.17
    case.uniformity.weight = 0.9  # problem 1's own weight does not leak into the layout evaluation
    assert inner_settings(case).lam_aps == 0.17
    case.layout_opt.aps_weight = 0.4
    assert inner_settings(case).lam_aps == 0.4
