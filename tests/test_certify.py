"""Optimality certificates: the bounds must dominate exact optima (brute force / SCIP) and real layouts."""

import itertools
import math

import numpy as np
import pytest

from cspaps import sun
from cspaps.aps import build_problem, solve
from cspaps.certify import (
    aps_upper_bound, improve_and_certify, layout_upper_bound, lone_heliostat_energy, objective_value,
    packing_capacity, potential_map, removal_losses,
)
from cspaps.field import Field, build_field, generate
from cspaps.layout_opt import evaluate_layout
from cspaps.receiver import build_grid
from cspaps.simulate import annual, sun_state
from cspaps.uniformity import region_of_grid

TIME = "2025-06-21 13:00"


def _tiny_problem(case, n=6, levels=3, offsets=(0.0,), tight=0.45, single_factor=1.15):
    """A few neighbouring heliostats, few candidates, a flux limit that binds at centre aim."""
    case.aps.n_height_levels, case.aps.azimuth_offsets_deg, case.aps.height_span = levels, list(offsets), 0.8
    case.shading.enabled = False
    case.receiver.n_panels, case.receiver.panel_start_azimuth_deg = 8, -22.5
    case.uniformity.panels = [1]  # H = the panel facing the selected (northern) heliostats
    fld = build_field(case)
    az = np.abs(np.degrees(np.arctan2(fld.xyz[:, 0], fld.xyz[:, 1])))
    pick = np.argsort(az)[:n]  # a column due north: their beams stack on the same spot
    sub = fld.subset(pick)
    ts, el, az, dni = sun_state(case, TIME)
    grid = build_grid(case.receiver, 48, 16)
    prob = build_problem(case, sub, el, az, dni, grid=grid)
    stacked = prob.flux(np.full(prob.n, prob.center_k)).max()
    single = max(float(r.max()) * prob.dni_con[i] for i, r in enumerate(prob.rs.resp))
    # every beam alone is feasible (with room for the fine-grid peak), stacking them all is not
    limit = max(tight * stacked, single_factor * single)
    case.receiver.max_flux_kw_m2 = limit / 1e3 / (1 - case.aps.safety_margin)
    prob = build_problem(case, sub, el, az, dni, grid=grid)
    return prob


def _enumerate(prob, mode, lam=0.0, eps=None, off=None):
    """Exact optimum of the candidate model by brute force over all (K + 1)^N plans (``off``: kept defocused)."""
    h = region_of_grid(prob.case, prob.grid)
    alpha = prob.rs.alpha[: prob.rs.n_grid][h]
    area = prob.rs.grid.cell_area
    best = -math.inf
    off = np.zeros(prob.n, dtype=bool) if off is None else off
    for combo in itertools.product(*[range(-1, 0) if o else range(-1, prob.k) for o in off]):
        ch = np.array(combo)
        if np.any(prob.flux(ch) > prob.limit * (1 + 1e-9)):
            continue
        q = prob.flux(ch, basis="objective")[: prob.rs.n_grid][h] * alpha
        p = float(prob.absorbed_power[np.flatnonzero(ch >= 0), ch[ch >= 0]].sum())
        s = area * math.sqrt(q.size) * float(np.linalg.norm(q - q.mean()))
        p_h = area * float(q.sum())
        if mode == "cv_limit":
            if p_h <= 0 or s > eps * p_h:
                continue
            val = p
        else:
            val = p - lam * s
        best = max(best, val)
    return best


@pytest.mark.parametrize("mode,lam", [("power", 0.0), ("weighted", 0.3), ("weighted", 1.0)])
def test_flux_prices_bound_the_value_of_an_added_heliostat(small_case, mode, lam):
    """Dual prices (improvement 4): with heliostat j priced (not in the layout), the bound is for the others and
    bound + G_j dominates the brute-force optimum with j added; removal losses are exact in the model."""
    prob = _tiny_problem(small_case)
    h = region_of_grid(prob.case, prob.grid)
    for j in (0, prob.n - 1):
        priced = np.zeros(prob.n, dtype=bool)
        priced[j] = True
        heur = solve(prob, "greedy", small_case.aps).choice.copy()
        heur[priced] = -1  # switching a mirror off keeps the plan feasible
        b = aps_upper_bound(prob, heur, mode, lam, h_mask=h, max_iter=400, time_limit_s=30, price_mask=priced)
        g = b["priced_gain_w"]
        assert g[j] >= 0 and np.all(g[~priced] == 0)
        assert b["upper_bound_mw"] * 1e6 >= _enumerate(prob, mode, lam, off=priced) * (1 - 1e-9) - 1e-6
        assert b["upper_bound_mw"] * 1e6 + g[j] >= _enumerate(prob, mode, lam) * (1 - 1e-9) - 1e-6
    with pytest.raises(ValueError):
        aps_upper_bound(prob, np.zeros(prob.n, dtype=int), mode, lam, h_mask=h, price_mask=np.ones(prob.n, bool))
    heur = solve(prob, "greedy", small_case.aps).choice
    loss = removal_losses(prob, heur, lam, h)
    j0 = objective_value(prob, heur, mode, lam, h)["objective_w"]
    assert np.all(loss[heur < 0] == 0)
    for i in np.flatnonzero(heur >= 0):
        ch = heur.copy()
        ch[i] = -1
        assert loss[i] == pytest.approx(j0 - objective_value(prob, ch, mode, lam, h)["objective_w"], rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("mode,lam", [("power", 0.0), ("weighted", 0.1), ("weighted", 0.3), ("weighted", 1.0)])
def test_aps_bound_dominates_brute_force(small_case, mode, lam):
    prob = _tiny_problem(small_case)
    heur = solve(prob, "greedy", small_case.aps).choice  # may add verified hotspots as constraint points
    opt = _enumerate(prob, mode, lam)  # same constraint points as the bound
    h = region_of_grid(prob.case, prob.grid)
    b = aps_upper_bound(prob, heur, mode, lam, h_mask=h, max_iter=400, time_limit_s=30)
    found = objective_value(prob, heur, mode, lam, h)["objective_w"]
    assert found <= opt * (1 + 1e-9) + 1e-6
    assert b["upper_bound_mw"] * 1e6 >= opt * (1 - 1e-9) - 1e-6  # valid upper bound
    assert opt > 0 and np.any(heur >= 0)
    assert b["status"] == "certified" and 0.0 <= b["gap"] < 1.0
    # the bound is informative on this tiny, strongly coupled case
    assert b["upper_bound_mw"] * 1e6 <= opt * 1.35 + 1.0


def test_aps_bound_cv_limit_against_brute_force_and_scip(small_case):
    pytest.importorskip("pyscipopt")
    from cspaps.uniformity import exact_benchmark

    prob = _tiny_problem(small_case, n=5)
    small_case.uniformity.enabled, small_case.uniformity.mode = True, "cv_limit"
    for eps in (0.9, 1.6):
        opt = _enumerate(prob, "cv_limit", eps=eps)  # plans with a lit H (CV defined)
        ex = exact_benchmark(prob, eps, small_case.uniformity, time_limit_s=60)
        assert ex["status"] in ("proven_optimal", "infeasible_except_shutdown")
        scip = (ex.get("power_abs_mw") or 0.0) * 1e6
        # SCIP's cone also admits plans that leave H dark, so its optimum is >= the brute force one
        if np.isfinite(opt):
            assert scip >= opt * (1 - 1e-6) - 1.0
        start = ex["choice"] if "choice" in ex else np.full(prob.n, -1)
        b = aps_upper_bound(prob, start, "cv_limit", eps=eps, max_iter=800, time_limit_s=30)
        assert b["upper_bound_mw"] * 1e6 >= scip * (1 - 1e-6) - 1e-6
        if np.isfinite(opt):
            assert b["upper_bound_mw"] * 1e6 >= opt * (1 - 1e-9) - 1e-6


def test_cv_limit_infeasibility_certificate(small_case):
    prob = _tiny_problem(small_case, n=4)
    # minimum load far above anything the field can deliver -> negative dual bound
    b = aps_upper_bound(prob, np.full(prob.n, -1), "cv_limit", eps=0.5, min_load_w=1e9, max_iter=600,
                        time_limit_s=30)
    assert b["status"] == "certified_infeasible"


def test_improve_and_certify_full_field(small_case):
    small_case.uniformity.enabled, small_case.uniformity.mode, small_case.uniformity.weight = True, "weighted", 0.5
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, TIME)
    prob = build_problem(small_case, fld, el, az, dni, grid=build_grid(small_case.receiver))
    res = solve(prob, "greedy", small_case.aps)
    before = objective_value(prob, res.choice, "weighted", 0.5)["objective_w"]
    new, cert = improve_and_certify(prob, res, rounds=2, time_limit_s=10)
    after = objective_value(prob, new.choice, "weighted", 0.5)["objective_w"]
    assert after >= before - 1e-6
    assert new.status in ("ok", "margin_reduced")
    assert cert["status"] == "certified"
    assert cert["upper_bound_mw"] * 1e6 >= after - 1e-6
    assert new.info["optimality_gap"] == pytest.approx(cert["gap"])
    assert new.info["solution_class"] in ("verified_feasible_with_certified_gap", "global_optimum_on_candidate_set")


def test_packing_capacity_dominates_hex_lattice():
    cell, d = 1.0, 10.0
    mask = np.zeros((150, 200), dtype=bool)
    mask[:, :] = True  # 200 m x 150 m rectangle
    cap = packing_capacity(mask, cell, d)
    # densest regular packing inside the rectangle (hexagonal lattice, both orientations)
    best = 0
    for dx, dy in ((d, d * math.sqrt(3) / 2), (d * math.sqrt(3) / 2, d)):
        n = 0
        rows = int(149 // dy) + 1
        for r_ in range(rows):
            off = 0.5 * dx if r_ % 2 else 0.0
            n += int((199 - off) // dx) + 1
        best = max(best, n)
    assert cap >= best
    assert cap <= 1.5 * best  # and not absurdly loose
    single = np.zeros((5, 5), dtype=bool)
    single[2, 2] = True
    assert packing_capacity(single, cell, d) >= 1


def test_lone_energy_matches_annual_without_shading(small_case):
    small_case.shading.enabled = False
    fld = build_field(small_case)
    smp = sun.sun_bins(sun.hourly_samples(small_case), 6)
    e = lone_heliostat_energy(small_case, fld.xyz[:, :2], smp)
    a = annual(small_case, fld, samples=smp, aiming="center", with_flux=False)
    assert e.sum() == pytest.approx(a.summary()["absorbed_energy_mwh"], rel=1e-6)


def test_layout_bound_dominates_generated_layouts(small_case):
    smp = sun.sun_bins(sun.hourly_samples(small_case), 6)
    small_case.field.boundary = [[-120, -120], [120, -120], [120, 120], [-120, 120]]
    pm = potential_map(small_case, smp)
    n = 120
    d = small_case.heliostat.diagonal * 1.1
    b = layout_upper_bound(small_case, n, d, samples=smp, pmap=pm)
    assert b["status"] == "ok"
    for params in ({"radial_spacing": 0.866, "radial_growth": 0.0, "azimuthal_spacing": 1.0},
                   {"radial_spacing": 1.2, "radial_growth": 0.5, "azimuthal_spacing": 1.4}):
        ev = evaluate_layout(small_case, "radial_staggered", {**params, "spacing_factor": 1.1}, smp, n, 4)
        assert ev.n == n
        e_real = annual(small_case, Field(ev.xyz), samples=smp, aiming="center", with_flux=False).summary()[
            "absorbed_energy_mwh"]
        e_lone = lone_heliostat_energy(small_case, ev.xyz[:, :2], smp).sum()
        assert e_real <= e_lone * (1 + 1e-9) <= b["upper_bound_mwh"] * (1 + 1e-9)
    # monotone in N, and N above the packing capacity is proven impossible
    assert layout_upper_bound(small_case, n // 2, d, samples=smp, pmap=pm)["upper_bound_mwh"] <= b["upper_bound_mwh"]
    big = layout_upper_bound(small_case, 10 * b["packing_capacity_bound"], d, samples=smp, pmap=pm)
    assert big["status"] == "no_layout_exists"
    densest = generate(small_case, "radial_staggered", {"radial_spacing": 0.866, "radial_growth": 0.0,
                                                        "azimuthal_spacing": 1.0, "spacing_factor": 1.1})
    assert len(densest) <= b["packing_capacity_bound"]


# --------------------------------------------------------------------------- audit regressions (layout bound rigour)

def _random_cell_points(pm, case, n, seed=0):
    from cspaps.field import site_mask

    rng = np.random.default_rng(seed)
    iy, ix = np.nonzero(pm.inside)
    k = rng.integers(0, len(ix), n)
    pts = np.column_stack([pm.x[ix[k]] + (rng.random(n) - 0.5) * pm.cell_m,
                           pm.y[iy[k]] + (rng.random(n) - 0.5) * pm.cell_m])
    ok = site_mask(case, np.column_stack([pts, np.full(n, case.heliostat.pivot_height)]))[0]
    return pts[ok], iy[k][ok], ix[k][ok]


@pytest.mark.parametrize("alpha", [None, [0.95, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3, 0.95, 0.3, 0.3, 0.3]])
def test_cell_bound_holds_anywhere_in_the_cell(small_case, alpha):
    """The per-cell bound dominates the lone-heliostat energy at random points inside the cells."""
    small_case.receiver.n_panels = 8
    small_case.receiver.panel_absorptivity = alpha
    small_case.field.boundary = [[-150, -150], [150, -150], [150, 150], [-150, 150]]
    smp = sun.sun_bins(sun.hourly_samples(small_case), 4)
    pm = potential_map(small_case, smp, cell_m=8.0)
    pts, iy, ix = _random_cell_points(pm, small_case, 4000)
    e = lone_heliostat_energy(small_case, pts, smp)
    assert np.all(e <= pm.ub_mwh[iy, ix] * (1 + 1e-9))


@pytest.mark.parametrize("alpha", [None, [0.95, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])
def test_single_heliostat_cannot_beat_layout_bound(small_case, alpha):
    """Audit counterexample: the best one-heliostat layout must not exceed the N = 1 bound."""
    from cspaps.field import site_mask

    small_case.receiver.n_panels = 8
    small_case.receiver.panel_absorptivity = alpha
    small_case.field.boundary = [[-150, -150], [150, -150], [150, 150], [-150, 150]]
    smp = sun.sun_bins(sun.hourly_samples(small_case), 4)
    pm = potential_map(small_case, smp, cell_m=8.0)  # coarse raster: peaks fall between centres
    ub = layout_upper_bound(small_case, 1, small_case.heliostat.diagonal * 1.1, samples=smp, pmap=pm)
    centre = np.where(pm.inside, pm.energy_mwh, -np.inf)
    a, b = np.unravel_index(np.argmax(centre), centre.shape)
    g = np.linspace(-2 * pm.cell_m, 2 * pm.cell_m, 41)
    gx, gy = np.meshgrid(pm.x[b] + g, pm.y[a] + g)
    loc = np.column_stack([gx.ravel(), gy.ravel()])
    loc = loc[site_mask(small_case, np.column_stack([loc, np.full(len(loc), 4.0)]))[0]]
    lone = lone_heliostat_energy(small_case, loc, smp)
    p = loc[int(np.argmax(lone))]
    one = Field(np.array([[p[0], p[1], small_case.heliostat.pivot_height]]))
    real = annual(small_case, one, samples=smp, aiming="center", with_flux=True).summary()["absorbed_energy_mwh"]
    assert lone.max() <= ub["upper_bound_mwh"] and real <= ub["upper_bound_mwh"] * (1 + 1e-3)
    checked = layout_upper_bound(small_case, 1, small_case.heliostat.diagonal * 1.1, samples=smp, pmap=pm,
                                 found_energy_mwh=real)
    assert checked["status"] == "ok" and checked["gap"] >= 0


def test_fast_energy_path_uses_beam_spread_absorptivity(small_case):
    """Without a flux map the absorbed power follows the panels each beam actually hits (not the mean)."""
    small_case.receiver.n_panels = 8
    small_case.receiver.panel_absorptivity = [0.95, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    fld = build_field(small_case)
    smp = sun.sun_bins(sun.hourly_samples(small_case), 4)
    fine = build_grid(small_case.receiver, 144, 24)  # columns resolve the panel edges
    fast = annual(small_case, fld, samples=smp, aiming="center", with_flux=False).summary()["absorbed_energy_mwh"]
    full = annual(small_case, fld, samples=smp, aiming="center", grid=fine).summary()["absorbed_energy_mwh"]
    assert fast == pytest.approx(full, rel=2e-3)
    # a heliostat facing the middle of panel 1 (0-45 deg) lights the 0.95 panel; the old
    # mean-absorptivity shortcut gave 0.38 for it
    a = np.radians(22.5)
    one = Field(np.array([[80.0 * np.sin(a), 80.0 * np.cos(a), small_case.heliostat.pivot_height]]))
    e1 = annual(small_case, one, samples=smp, aiming="center", with_flux=False)
    assert e1.summary()["absorbed_energy_mwh"] / e1.energy_mwh["intercept"] > 0.9


def test_bound_violation_is_reported_not_clipped(small_case):
    smp = sun.sun_bins(sun.hourly_samples(small_case), 4)
    small_case.field.boundary = [[-120, -120], [120, -120], [120, 120], [-120, 120]]
    pm = potential_map(small_case, smp)
    b = layout_upper_bound(small_case, 10, small_case.heliostat.diagonal * 1.1, samples=smp, pmap=pm)
    bad = layout_upper_bound(small_case, 10, small_case.heliostat.diagonal * 1.1, samples=smp, pmap=pm,
                             found_energy_mwh=2 * b["upper_bound_mwh"])
    assert bad["status"] == "bound_violation" and bad["gap"] < 0



def test_final_check_uses_the_executed_aims_and_refinement_is_monotone(small_case, tmp_path):
    """task_aim reports the check of the final aims (re-shaded); nested refinement never loses objective."""
    from cspaps import workflows as wf

    small_case.uniformity.enabled, small_case.uniformity.mode, small_case.uniformity.weight = True, "weighted", 0.3
    s = wf.task_aim(small_case, tmp_path, "2025-12-21 10:00", method="greedy", certify=False)["summary"]
    assert "final aim points" in s["flux_check"]["basis"] and "solver_flux_check" in s
    assert s["status"] in ("ok", "margin_reduced", "uniformity_infeasible")
    small_case.aps.certify_time_s = 4.0
    ref = wf.task_aps_refinement(small_case, tmp_path, "2025-06-21 13:00", levels=[(3, 3), (5, 5)])["summary"]["levels"]
    assert [r["n_candidates"] for r in ref] == [9, 25]
    assert ref[1]["objective_found_mw"] >= ref[0]["objective_found_mw"] - 1e-9
    assert all(r["upper_bound_mw"] >= r["objective_found_mw"] - 1e-9 for r in ref)
