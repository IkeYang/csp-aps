import numpy as np
import pytest

from cspaps.aps import build_problem, candidate_grid, lp_upper_bound, optimize_aims, solve_greedy
from cspaps.field import Field, build_field
from cspaps.receiver import build_grid
from cspaps.simulate import evaluate, sun_state

TIME = "2025-06-21 13:00"


@pytest.fixture
def tight(small_case):
    """Small field with a flux limit that centre aiming clearly violates."""
    small_case.receiver.max_flux_kw_m2 = 350
    return small_case


def _eval(case, fld, res, dni=None):
    ts, el, az, dni_m = sun_state(case, TIME)
    return evaluate(case, fld, el, az, dni_m if dni is None else dni, aim=res.aim, active=res.active)


def test_candidate_grid_has_center(small_case):
    offs, zs, c = candidate_grid(small_case.aps, small_case.receiver)
    assert offs.size == zs.size == len(small_case.aps.azimuth_offsets_deg) * small_case.aps.n_height_levels
    assert offs[c] == 0 and zs[c] == pytest.approx(small_case.receiver.center[2])


def test_center_violates_and_aps_satisfies_limit(tight):
    fld = build_field(tight)
    ts, el, az, dni = sun_state(tight, TIME)
    limit = tight.receiver.max_flux_kw_m2
    center = evaluate(tight, fld, el, az, dni)
    assert center.flux.max() / 1e3 > limit
    for method in ("kfactor", "greedy"):
        res = optimize_aims(tight, fld, el, az, dni, method=method)
        snap = _eval(tight, fld, res)
        assert res.feasible, method
        # the solver works with a 5 % margin and truncated footprints; the real flux must stay below the limit
        assert snap.flux.max() / 1e3 <= limit, method
        assert snap.p_receiver > 0.85 * center.p_receiver, method


def test_greedy_not_worse_than_kfactor(tight):
    fld = build_field(tight)
    ts, el, az, dni = sun_state(tight, TIME)
    prob = build_problem(tight, fld, el, az, dni)
    g = optimize_aims(tight, fld, el, az, dni, method="greedy", problem=prob)
    k = optimize_aims(tight, fld, el, az, dni, method="kfactor", problem=prob)
    assert g.power_w >= k.power_w * 0.999


def test_milp_and_lp_bound_on_tiny_problem(tight):
    full = build_field(tight)
    north = np.flatnonzero(full.xyz[:, 1] > 60)[:14]
    fld = Field(full.xyz[north])
    tight.receiver.max_flux_kw_m2 = 40
    tight.aps.n_height_levels = 3
    tight.aps.azimuth_offsets_deg = [-5.0, 0.0, 5.0]
    tight.aps.milp_time_limit_s = 20
    grid = build_grid(tight.receiver)
    ts, el, az, dni = sun_state(tight, TIME)
    prob = build_problem(tight, fld, el, az, dni, grid=grid)
    greedy = optimize_aims(tight, fld, el, az, dni, method="greedy", problem=prob)
    milp = optimize_aims(tight, fld, el, az, dni, method="milp", problem=prob)
    bound = lp_upper_bound(prob)
    assert milp.feasible and greedy.feasible
    if milp.info["global_optimality_proven"]:  # the exact optimum on the (enlarged) constraint set
        assert milp.info["objective_absorbed_power_w"] >= greedy.info["objective_absorbed_power_w"] - 1e-6 * bound
    assert bound >= milp.info["objective_absorbed_power_w"] * (1 - 1e-6)
    assert greedy.info["objective_absorbed_power_w"] >= 0.85 * milp.info["objective_absorbed_power_w"]


def test_defocus_when_limit_unreachable(small_case):
    small_case.receiver.max_flux_kw_m2 = 20
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    res = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    assert res.n_defocused > 0 and res.feasible
    small_case.aps.allow_defocus = False
    res2 = optimize_aims(small_case, fld, el, az, dni, method="greedy")
    assert res2.n_defocused == 0 and not res2.feasible


def test_forecast_constraint_is_respected(tight):
    """Constraint DNI above the current DNI: flux stays safe when the DNI rises to it."""
    fld = build_field(tight)
    ts, el, az, dni = sun_state(tight, TIME)
    shaded = np.where(fld.xyz[:, 1] > 0, 0.3 * dni, dni)  # northern half under a cloud now
    res = optimize_aims(tight, fld, el, az, shaded, dni_constraint=np.full(fld.n, dni), method="greedy")
    after = _eval(tight, fld, res, dni=dni)  # the cloud has gone
    assert after.flux.max() / 1e3 <= tight.receiver.max_flux_kw_m2
    reactive = optimize_aims(tight, fld, el, az, shaded, method="greedy")
    assert _eval(tight, fld, reactive, dni=dni).flux.max() > after.flux.max()


def test_move_penalty_reduces_moves(tight):
    fld = build_field(tight)
    ts, el, az, dni = sun_state(tight, TIME)
    base = optimize_aims(tight, fld, el, az, dni, method="greedy")
    ts2, el2, az2, dni2 = sun_state(tight, "2025-06-21 13:20")
    free = optimize_aims(tight, fld, el2, az2, dni2, method="greedy", prev=base.aim, prev_active=base.active)
    tight.aps.move_penalty_per_m = 0.05
    smooth = optimize_aims(tight, fld, el2, az2, dni2, method="greedy", prev=base.aim, prev_active=base.active)

    def moved(r):
        return int(np.sum((np.abs(r.aim[0] - base.aim[0]) > 1e-6) | (np.abs(r.aim[1] - base.aim[1]) > 1e-6)))
    assert moved(smooth) <= moved(free)
    assert smooth.feasible


def test_greedy_flux_bookkeeping(tight):
    fld = build_field(tight)
    ts, el, az, dni = sun_state(tight, TIME)
    prob = build_problem(tight, fld, el, az, dni)
    choice, _ = solve_greedy(prob, tight.aps)
    assert (prob.flux(choice) <= prob.limit * (1 + 1e-9)).all()


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
def test_verification_always_ends_feasible_and_labels_are_honest(tight, rounds):
    """Audit: with too few re-solve rounds the MILP plan failed the fine-grid check yet was labelled optimal."""
    full = build_field(tight)
    fld = Field(full.xyz[np.flatnonzero(full.xyz[:, 1] > 60)[:14]])
    tight.receiver.max_flux_kw_m2 = 40
    tight.aps.n_height_levels, tight.aps.azimuth_offsets_deg = 3, [-5.0, 0.0, 5.0]
    tight.aps.verify_max_rounds = rounds
    ts, el, az, dni = sun_state(tight, TIME)
    prob = build_problem(tight, fld, el, az, dni, grid=build_grid(tight.receiver))
    optimize_aims(tight, fld, el, az, dni, method="greedy", problem=prob)  # shares hotspot points with the MILP
    milp = optimize_aims(tight, fld, el, az, dni, method="milp", problem=prob)
    assert milp.status in ("ok", "margin_reduced") and milp.feasible
    if milp.info["global_optimality_proven"]:
        assert milp.info["repair_rounds"] == 0 and milp.info["solution_class"] == "global_optimum_on_candidate_set"
    tight.aps.verify = True
    # an unverifiable plan is never "proven optimal"
    tight.aps.verify_max_rounds = 0
    tight.aps.allow_defocus = False
    fail = optimize_aims(tight, fld, el, az, dni, method="milp", problem=prob)
    if fail.status == "infeasible":
        assert not fail.info["global_optimality_proven"]
        assert fail.info["solution_class"] == "infeasible_after_verification"


def test_shading_is_evaluated_per_candidate_aim(small_case):
    """Each candidate aim has its own mirror normal and beam, so its own shading / blocking."""
    from cspaps import shading, sun as sun_mod

    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, "2025-12-21 09:30")  # low sun: blocking depends on the aim
    prob = build_problem(small_case, fld, el, az, dni)
    rs = prob.rs
    assert rs.sb_k.shape == (fld.n, prob.k)
    np.testing.assert_allclose(rs.sb, rs.sb_k[:, rs.center_k])
    s = sun_mod.sun_vector(el, az)
    from cspaps import optics
    j = 0 if rs.center_k != 0 else 1
    normals, t, *_ = optics.heliostat_geometry(fld.xyz, rs.aim_xyz[:, j], s)
    direct = shading.shading_blocking(fld.xyz, normals, s, t, small_case.heliostat, small_case.receiver, fld.pairs,
                                      n_points=small_case.shading.n_points)["combined"]
    np.testing.assert_allclose(rs.sb_k[:, j], direct)
