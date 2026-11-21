"""Acceptance cases with mutual shading, zero optical input, and binding site constraints."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from cspaps import load_case
from cspaps.aps import build_problem
from cspaps.field import build_field
from cspaps.layout_opt import optimize_layout
from cspaps.resources import demo_case_path
from cspaps.simulate import evaluate, sun_state
from cspaps.workflows import task_aim


@pytest.fixture
def pair(tmp_path):
    case = load_case(demo_case_path())
    case.receiver.center = (0, 0, 55)
    case.receiver.height, case.receiver.diameter = 20, 12
    case.aps.n_height_levels, case.aps.azimuth_offsets_deg = 3, [0.0]
    case.aps.height_span = 0.75
    case.field.exclusion_radius = 10
    path = tmp_path / "field.csv"
    pd.DataFrame({"x": [1.659256, 16.787252], "y": [-73.187585, -74.519679], "z": [5, 5]}).to_csv(path, index=False)
    case.field.layout_file = str(path)
    return case


TIME = "2024-03-19 08:30"


@pytest.mark.parametrize("weight", [0.0, 0.01])
def test_coupled_optimum_matches_all_mixed_poses(pair, tmp_path, weight):
    pair.uniformity.weight = weight
    field = build_field(pair)
    _, el, az, _ = sun_state(pair, TIME)
    prob = build_problem(pair, field, el, az, 650)
    scores = {}
    for choices in product(range(-1, 3), repeat=2):
        ch = np.array(choices)
        snap = evaluate(pair, field, el, az, 650, aim=prob.aims(ch), active=ch >= 0, grid=prob.grid)
        if snap.verify(pair.receiver).ok:
            # Normalise surface quadrature to the independently integrated intercepted energy.
            q = snap.absorbed_flux
            spread = snap.p_absorbed * q.std() / q.mean() if q.mean() else 0.0
            scores[choices] = snap.p_absorbed - weight * spread
    result = task_aim(pair, tmp_path, TIME, dni=650, method="milp")
    best = max(scores.values())
    cert = result["certificate"]
    assert scores[tuple(result["aim_result"].choice)] == pytest.approx(best)
    assert cert["objective_found_mw"] * 1e6 == pytest.approx(best)
    assert cert["upper_bound_mw"] * 1e6 == pytest.approx(best)
    assert cert["global_optimum_proven_on_candidate_set"]


def test_weighted_impossible_load_and_zero_dni(pair, tmp_path):
    pair.uniformity.min_load_mw = 1
    s = task_aim(pair, tmp_path, TIME, dni=650)["summary"]
    assert s["existence"]["optimum_exists"] is False
    assert s["certificate"]["objective_found_mw"] is None
    pair.uniformity.min_load_mw = 0
    s = task_aim(pair, tmp_path, TIME, dni=0)["summary"]
    assert s["status"] == "ok" and s["absorbed_power_mw"] == 0
    assert s["flux_check"]["peak_flux_kw_m2"] == 0 and s["aps"]["cv_2d_H"] is None


def test_required_spacing_enters_search(small_case):
    c = small_case
    c.field.boundary = [[-105, -105], [105, -105], [105, 105], [-105, 105]]
    c.field.exclusion_radius = 50
    c.layout_opt.n_heliostats, c.layout_opt.min_spacing_factor = 12, 3
    c.layout_opt.maxiter, c.layout_opt.popsize, c.layout_opt.refine_top = 1, 2, 0
    result = optimize_layout(c)
    assert result.validation["status"] == "ok"
    assert result.best.n == 12
    assert result.validation["min_distance_m"] >= 3 * c.heliostat.diagonal - 1e-4


def test_relaxed_joint_bound_dominates_enumeration(pair, tmp_path):
    pair.uniformity.weight = 0.01
    pair.receiver.panel_absorptivity = [0.62, 0.93] * 6
    exact = task_aim(pair, tmp_path, TIME, dni=650)["certificate"]
    pair.aps.coupled_exact_limit = 0
    limited = task_aim(pair, tmp_path, TIME, dni=650)["certificate"]
    assert limited["upper_bound_mw"] >= exact["objective_found_mw"] - 1e-10
    assert limited["continuous_upper_bound_mw"] >= exact["objective_found_mw"]
    assert limited["objective_found_mw"] <= exact["objective_found_mw"] + 1e-10


def test_hourly_layout_bound_uses_every_weather_record(pair, tmp_path):
    from cspaps import sun
    from cspaps.certify import hourly_site_upper_bound, potential_map
    from cspaps.layout_opt import _hourly_theoretical_score

    path = tmp_path / "weather.csv"
    pd.DataFrame({"time": pd.date_range("2024-06-21 08:00", periods=12, freq="h"),
                  "dni": [200, 500, 800, 900, 500, 700, 900, 300, 650, 420, 260, 80]}).to_csv(path, index=False)
    pair.weather.file = str(path)
    hours = sun.hourly_samples(pair)
    pmap = potential_map(pair, sun.sun_bins(hours, 2), max_cells=300)
    field = build_field(pair)
    upper = hourly_site_upper_bound(pair, field.n, 12, hours, pmap)
    energy, spread = _hourly_theoretical_score(pair, field.xyz, hours, 0.3, 5)
    assert 0 < energy - 0.3 * spread <= energy <= upper
    assert len(hours) > 2


def test_surface_layout_penalty_uses_entire_flux_map(pair):
    from cspaps import sun
    from cspaps.layout_opt import _hourly_theoretical_score

    pair.layout_opt.uniformity_metric = "surface"
    _, el, az, _ = sun_state(pair, TIME)
    sample = sun.Samples(np.array([el]), np.array([az]), np.array([650.]), np.array([1.5]))
    fld = build_field(pair)
    snap = evaluate(pair, fld, el, az, 650)
    energy, spread = _hourly_theoretical_score(pair, fld.xyz, sample, 0.3, pair.shading.n_points)
    assert energy == pytest.approx(1.5 * snap.p_absorbed / 1e6)
    assert spread == pytest.approx(energy * snap.absorbed_flux.std() / snap.absorbed_flux.mean())
