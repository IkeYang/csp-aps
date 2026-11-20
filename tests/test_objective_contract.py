from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cspaps import load_case
from cspaps.aps import build_problem, solve_greedy
from cspaps.field import Field, build_field
from cspaps.receiver import build_grid
from cspaps.simulate import annual, evaluate, sun_state
from cspaps.sun import Samples
from cspaps.uniformity import solve_uniform


def test_absorbed_objective_integrates_across_panel_boundaries(small_case):
    case = small_case
    case.shading.enabled = False
    case.receiver.n_panels = 2
    case.receiver.panel_start_azimuth_deg = 0.0
    case.receiver.panel_absorptivity = [0.1, 0.9]
    case.aps.azimuth_offsets_deg = [0.0]
    case.aps.n_height_levels = 1
    case.aps.auto_grid = False
    field = Field(np.array([[0.0, 80.0, case.heliostat.pivot_height]]))
    grid = build_grid(case.receiver, 144, 64)
    problem = build_problem(case, field, 45.0, 180.0, 900.0, grid=grid)

    aim = problem.aims(np.array([0]))
    fine = evaluate(case, field, 45.0, 180.0, 900.0, aim=aim,
                    grid=build_grid(case.receiver, 288, 128))

    assert problem.absorbed_power[0, 0] == pytest.approx(fine.p_absorbed, rel=0.01)
    assert problem.absorbed_power[0, 0] < problem.power[0, 0] * 0.8


def test_weighted_uniformity_respects_reachable_aims(small_case):
    case = small_case
    case.shading.enabled = False
    case.uniformity.enabled = True
    case.uniformity.mode = "weighted"
    case.uniformity.weight = 0.3
    field = build_field(case)
    _, elevation, azimuth, dni = sun_state(case, "2025-06-21 13:00")
    problem = build_problem(case, field, elevation, azimuth, dni)
    base, _ = solve_greedy(problem, case.aps)
    problem.reach = np.zeros((problem.n, problem.k), dtype=bool)
    active = np.flatnonzero(base >= 0)
    problem.reach[active, base[active]] = True

    result = solve_uniform(problem, base, case.uniformity)

    selected = np.flatnonzero(result.choice >= 0)
    assert result.status == "weighted"
    assert np.all(problem.reach[selected, result.choice[selected]])
    assert np.all(problem.flux(result.choice) <= problem.limit * (1 + 1e-9))


def test_uniformity_weight_controls_power_cv_tradeoff(small_case):
    case = small_case
    case.shading.enabled = False
    case.uniformity.enabled = True
    case.uniformity.mode = "weighted"
    field = build_field(case)
    _, elevation, azimuth, dni = sun_state(case, "2025-06-21 13:00")
    problem = build_problem(case, field, elevation, azimuth, dni)
    base, _ = solve_greedy(problem, case.aps)

    case.uniformity.weight = 0.0
    power_first = solve_uniform(problem, base, case.uniformity)
    case.uniformity.weight = 2.0
    uniformity_weighted = solve_uniform(problem, base, case.uniformity)

    assert power_first.status == uniformity_weighted.status == "weighted"
    assert uniformity_weighted.cv_2d < power_first.cv_2d
    assert uniformity_weighted.power_abs_w < power_first.power_abs_w


def test_annual_layout_score_zeroes_uniformity_failed_hours():
    case = load_case(Path(__file__).resolve().parents[1] / "cases" / "aksai_2024.yaml")
    case.shading.enabled = False
    case.aps.n_height_levels = 3
    case.aps.azimuth_offsets_deg = [0.0]
    case.aps.passes = 1
    case.aps.verify_max_rounds = 2
    case.uniformity.enabled = True
    case.uniformity.mode = "cv_limit"
    case.uniformity.max_cv = 0.01
    field = Field(np.array([
        [0.0, 200.0, 5.0],
        [28.0, 215.0, 5.0],
        [-25.0, 235.0, 5.0],
        [5.0, 250.0, 5.0],
    ]))
    time = pd.DatetimeIndex([pd.Timestamp("2024-06-21 12:00", tz="Asia/Shanghai")])
    samples = Samples(np.array([45.0]), np.array([180.0]), np.array([900.0]), np.array([2.0]),
                      times=time, label="synthetic_aksai")

    result = annual(case, field, samples=samples, aiming="greedy", workers=1)
    row = result.timeseries.iloc[0]
    summary = result.summary()

    assert row["flux_feasible"]
    assert row["uniformity_status"] == "not_reached"
    assert row["uniformity_cv_2d"] > case.uniformity.max_cv
    assert row["p_absorbed_mw"] > 0.0
    assert not row["operational_feasible"]
    assert row["feasible_absorbed_mw"] == 0.0
    assert summary["absorbed_energy_mwh"] > 0.0
    assert summary["feasible_absorbed_energy_mwh"] == 0.0
    assert summary["hours_uniformity_failed"] == pytest.approx(2.0)
