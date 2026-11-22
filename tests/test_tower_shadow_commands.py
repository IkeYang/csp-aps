import numpy as np
import pandas as pd
import pytest

from cspaps.aps import build_problem, solve
from cspaps.coupled import search_joint_poses
from cspaps.engine import APSEngine, ControlRequest, Receipt
from cspaps.field import Field
from cspaps.layout_value import solve_instant
from cspaps.simulate import evaluate


@pytest.fixture
def shadow_field(small_case):
    c = small_case
    c.receiver.tower_diameter = 10.
    c.aps.n_height_levels, c.aps.azimuth_offsets_deg = 3, [0.]
    c.uniformity.enabled, c.uniformity.mode, c.uniformity.weight = True, "weighted", 0.
    c.aps.allow_defocus = True
    return c, Field(np.array([[0., 60., 4.], [5., 90., 4.], [55., 70., 4.]]))


def test_shadow_tracking_and_joint_certificate(shadow_field):
    case, field = shadow_field
    prob = build_problem(case, field, 20., 180., 800.)
    snap = evaluate(case, field, 20., 180., 800.)
    assert snap.tower_fraction[0] == 1 and 0 < snap.tower_fraction[1] < 1 and snap.tower_fraction[2] == 0
    plan = solve_instant(case, field, 20., 180., 800.)
    assert plan.active.all() and plan.n_defocused == 0
    assert plan.mirror_power_w[0] == 0 and plan.mirror_power_w[1] > 0
    seed = solve(prob, "greedy", case.aps)
    seed.choice[0] = -1  # force a thermally indistinguishable but unnecessarily parked starting plan
    result, cert = search_joint_poses(prob, field, seed)
    assert result.active.all() and result.check.ok
    assert cert["global_optimum_proven_on_candidate_set"]
    # A genuinely binding limit still requires defocus on illuminated mirrors.
    case.receiver.max_flux_kw_m2 = 0.001
    constrained = solve_instant(case, field, 20., 180., 800.)
    assert constrained.active.tolist() == [True, False, False]
    assert constrained.flux_status == "ok" and constrained.p_abs_w == 0


def test_shadow_preserves_executed_target_and_respects_availability(shadow_field):
    case, field = shadow_field
    field = Field(field.xyz[:1])
    engine = APSEngine(case, field)
    t = pd.Timestamp("2025-03-20 10:00", tz="Asia/Shanghai")
    az, z = np.array([1.23]), np.array([case.receiver.center[2] + 0.37])
    req = ControlRequest(t, np.array([800.]), t, receipt=Receipt(1, az, z, np.array([True])),
                         sun_elevation=20., sun_azimuth=180.)
    held = engine.step(req)
    assert held.check.ok and held.active[0] and held.n_moved == 0 and held.power_absorbed_w == 0
    assert held.summary()["n_defocus_commands"] == 0
    assert held.aim_azimuth[0] == az[0] and held.aim_z[0] == z[0]
    req.receipt = Receipt(2, az, z, np.array([False]))
    parked = engine.step(req)
    assert not parked.active[0] and parked.n_moved == 0 and parked.n_defocus_commands == 0
    req.sun_azimuth = 90.  # out of shadow: evaluate actual power again
    assert engine.step(req).power_absorbed_w > 0
    req.available = np.array([False])
    assert not engine.step(req).active[0]


def test_shadow_is_not_a_defocus_command_and_disallowed_shutdown_stays_aimed(shadow_field):
    case, field = shadow_field
    # A daytime fully shaded heliostat is aimed and makes no power.
    plan = solve_instant(case, Field(field.xyz[:1]), 20., 180., 800.)
    assert plan.active[0] and plan.mirror_power_w[0] == 0

    # An unreachable minimum load does not turn the shadowed mirror into a defocus command.
    case.uniformity.min_load_mw = 1.
    impossible = solve_instant(case, Field(field.xyz[:1]), 20., 180., 800.)
    assert impossible.status in ("no_feasible_found", "proven_infeasible") and impossible.active[0]
    case.uniformity.min_load_mw = 0.

    # When defocus is forbidden, a uniformity minimum-load failure cannot emit an off command.
    case.aps.allow_defocus = False
    case.uniformity.enabled = True
    case.uniformity.min_load_mw = 1.
    engine = APSEngine(case, Field(field.xyz[:1]))
    t = pd.Timestamp("2025-03-20 10:00", tz="Asia/Shanghai")
    result = engine.step(ControlRequest(t, np.array([800.]), t, sun_elevation=20., sun_azimuth=180.))
    assert result.state == "infeasible" and result.active[0]
    assert result.n_defocus_commands == 0 and result.power_absorbed_w == 0

    # At night the mirrors retain their tracking state too.
    idle = solve_instant(case, Field(field.xyz[:1]), -1., 180., 0.)
    assert idle.status == "no_sun" and idle.n_active == 1 and idle.n_defocused == 0

    # With the tower clear, a binding flux limit cannot generate an off command when defocus is disabled.
    case.uniformity.enabled = False
    case.uniformity.min_load_mw = 0.
    case.receiver.max_flux_kw_m2 = 0.001
    exposed = APSEngine(case, Field(np.array([[90., 0., 4.]])))
    constrained = exposed.step(ControlRequest(t, np.array([800.]), t, sun_elevation=60., sun_azimuth=90.))
    assert constrained.state == "infeasible" and constrained.active[0]
    assert constrained.n_defocus_commands == 0


def test_projected_shadow_and_beam_visibility(shadow_field):
    pytest.importorskip("plotly")
    from cspaps.gui_plots import aim_scene_3d, field_map

    case, field = shadow_field
    snap = evaluate(case, field, 20., 180., 800.)
    kw = dict(sun_angles=(20., 180.))
    plan = field_map(case, field.xyz, field.ids, **kw)
    poly = next(t for t in plan.data if t.name.startswith("塔影"))
    assert max(poly.y) == pytest.approx((case.receiver.top - case.heliostat.pivot_height)
                                         / np.tan(np.radians(20.)) + case.receiver.tower_radius)
    scene = aim_scene_3d(case, field.xyz, snap.dni, snap.aim_azimuth, snap.aim_z, snap.active,
                         beam_power=snap.power["intercept"], **kw)
    assert any(t.type == "mesh3d" and np.all(t.z == case.heliostat.pivot_height) for t in scene.data)
    beams = next(t for t in scene.data if t.name and "反射光束" in t.name)
    assert list(beams.x)[::3] == pytest.approx([5., 55.])
    for elevation, enabled in ((-5., True), (20., False)):
        case.shading.tower_shadow = enabled
        fig = field_map(case, field.xyz, field.ids, sun_angles=(elevation, 180.))
        assert not any(t.name.startswith("塔影") for t in fig.data)
