"""Field report: the trade-off curve from the "center" initial plan put lambda = 0 at 25.8 MW while every other
point was about 370 MW. The solver had filled the receiver centre to the allowed flux; on the fine grid its plan
read a hair above that level ("margin_reduced": above limit x (1 - margin), below the limit), which the pipeline
accepts. The joint-pose search accepted only poses strictly within the margin, so it discarded the plan and
climbed from "every mirror defocused" to a nearly empty one -- and called it certified. Regression tests."""

import numpy as np
import pytest

from cspaps.aps import build_problem, move_cost, plan_move_cost, solve, solve_greedy
from cspaps.coupled import search_joint_poses
from cspaps.field import build_field
from cspaps.receiver import build_grid, surface_point
from cspaps.simulate import evaluate, sun_state
from cspaps.workflows import initial_state, task_pareto

TIME = "2025-06-21 11:00"


def _binding(case, mode):
    """Small case whose flux limit binds at lambda = 0."""
    case.receiver.height = 16.0
    case.receiver.max_flux_kw_m2 = 200
    case.aps.initial_aim = mode
    return case


@pytest.mark.parametrize("mode", ["center", "none"])
def test_lambda_zero_is_the_most_powerful_point_when_the_limit_binds(small_case, tmp_path, mode):
    case = _binding(small_case, mode)
    curve = task_pareto(case, tmp_path, TIME, lambdas=[0.0, 0.05, 0.3], certify=True)["summary"]["curve"]
    p = [r["power_abs_mw"] for r in curve]
    assert p[0] >= max(p[1:]) * (1 - 1e-3)  # lambda = 0 only asks for power
    assert curve[0]["peak_flux_kw_m2"] <= case.receiver.max_flux_kw_m2  # still within the device limit


def test_joint_pose_search_keeps_a_solver_plan_that_ends_inside_the_margin(small_case):
    """The solver may end its plan inside the safety margin after its verification rounds (field case: 950.01
    kW/m2 against 950 allowed); the pipeline accepts it. Such a seed is made here by solving without a margin
    and searching with 5 %: the search keeps it (or better) and uses no more of the margin."""
    case = _binding(small_case, "center")
    case.aps.safety_margin = 0.0
    fld = build_field(case)
    _, el, az, dni = sun_state(case, TIME)
    prob = build_problem(case, fld, el, az, dni, grid=build_grid(case.receiver))
    prev, prev_active, _ = initial_state(case, prob, "greedy")
    prev_xyz = surface_point(case.receiver, *prev)
    warm, _ = solve_greedy(prob, case.aps, prev_xyz, prev_active)
    seed = solve(prob, "greedy", case.aps, prev, prev_active, init=warm)
    case.aps.safety_margin = 0.05
    snap = evaluate(case, fld, el, az, prob.dni_obj, aim=prob.aims(seed.choice), active=seed.choice >= 0, grid=prob.grid)
    seed_check = snap.verify(case.receiver, max_cells=case.aps.verify_max_cells)
    seed_value = snap.thermal_objective(case.uniformity)["objective_w"]
    assert seed_check.status == "margin_reduced"  # the situation of the field report
    prices = move_cost(prob, prev_xyz, prev_active)
    out, cert = search_joint_poses(prob, fld, seed, warm=warm, move_cost=prices,
                                   tracking_reference=prob.tracking_reference(prev, prev_active))
    assert np.count_nonzero(out.choice >= 0) >= 0.95 * np.count_nonzero(seed.choice >= 0)
    assert cert["priced_objective_mw"] * 1e6 >= seed_value - plan_move_cost(prices, seed.choice) - 1e-6
    # no more of the safety margin than the solver's own plan, and still below the device limit
    assert out.check.peak_ratio <= seed_check.peak_ratio + 1e-12 and out.check.status != "over_limit"
    assert cert["status"] == "certified"


def test_a_seed_within_the_margin_still_needs_the_strict_check(small_case):
    """Unchanged rule when the solver's plan is strictly within the margin: every accepted pose is too."""
    small_case.receiver.max_flux_kw_m2 = 350
    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, TIME)
    prob = build_problem(small_case, fld, el, az, dni, grid=build_grid(small_case.receiver))
    seed = solve(prob, "greedy", small_case.aps)
    assert seed.check.ok
    out, cert = search_joint_poses(prob, fld, seed)
    assert out.check.ok and cert["status"] == "certified"
