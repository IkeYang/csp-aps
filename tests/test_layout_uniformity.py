"""Layout-determined (circumferential) receiver uniformity for problem 2."""

import itertools

import numpy as np
import pytest

from cspaps import sun
from cspaps import layout_uniformity as lu
from cspaps.field import Field, build_field, generate
from cspaps.layout_opt import evaluate_layout, layout_tradeoff, optimize_layout
from cspaps.receiver import build_grid
from cspaps.simulate import annual, evaluate


def _samples(case, n=6):
    return sun.sun_bins(sun.hourly_samples(case), n)


def test_column_shares_match_full_flux_model(small_case):
    """Analytic lateral-Gaussian column shares vs column sums of the HFLCAL flux map."""
    grid = build_grid(small_case.receiver, 72, 16)
    fld = build_field(small_case)
    smp = _samples(small_case)
    for t in (0, 3):
        snap = evaluate(small_case, fld, smp.elevation[t], smp.azimuth[t], smp.dni[t], grid=grid)
        full = snap.flux.sum(axis=0) * grid.cell_area
        shares = lu.column_shares(small_case, fld.xyz, smp, grid)[t]
        model = (snap.power["intercept"][:, None] * shares).sum(axis=0)
        assert model.sum() == pytest.approx(full.sum(), rel=2e-3)
        assert np.abs(model - full).max() <= 0.01 * full.max()


def test_column_shares_single_heliostat_geometry(small_case):
    """A heliostat due north lights the north half only, symmetrically, and its shares sum to one."""
    grid = build_grid(small_case.receiver, 36, 8)
    xyz = np.array([[0.0, 150.0, small_case.heliostat.pivot_height]])
    smp = _samples(small_case, 4)
    sh = lu.column_shares(small_case, xyz, smp, grid)
    for t in range(len(smp)):
        np.testing.assert_allclose(sh[t, 0].sum(), 1.0, rtol=1e-9)
        rel = (grid.azimuth_deg + 180.0) % 360.0 - 180.0
        assert np.all(sh[t, 0][np.abs(rel) > 90.0] == 0.0)
        np.testing.assert_allclose(sh[t, 0], sh[t, 0][::-1], atol=1e-12)  # columns symmetric about north


def test_selection_bound_and_search_against_brute_force(small_case):
    smp = _samples(small_case, 4)
    xyz = build_field(small_case).xyz
    rng = np.random.default_rng(3)
    pick = rng.choice(len(xyz), 11, replace=False)
    sub = xyz[pick]
    res = annual(small_case, Field(sub), samples=smp, aiming="center", with_flux=False, keep_state_power=True)
    for lam in (0.2, 1.0):
        model = lu.build_model(small_case, sub, smp, res.state_power_w, lam)
        n = 4
        best = max(model.objective(np.isin(np.arange(len(sub)), c)) for c in itertools.combinations(range(len(sub)), n))
        sel, _ = lu.select(model, n)
        found = model.objective(sel)
        b = lu.selection_bound(model, n, sel, max_iter=400)
        assert sel.sum() == n
        assert found <= best + 1e-9
        assert b["upper_bound_mwh"] >= best - 1e-6  # valid bound
        assert found >= best - 0.03 * abs(best)  # exchange search is near the brute-force optimum


def test_uniformity_weight_centres_the_field(small_case):
    """Energy-only selection crowds the north side; lambda_L > 0 surrounds the tower."""
    smp = _samples(small_case)
    params = {"radial_spacing": 0.95, "radial_growth": 0.3, "azimuthal_spacing": 1.2, "spacing_factor": 1.1}
    rows = layout_tradeoff(small_case, "radial_staggered", params, smp, 120, [0.0, 0.5], shading_points=4)
    e0, e1 = rows
    assert e0["north_fraction"] > 0.75
    assert e1["north_fraction"] < e0["north_fraction"] - 0.15
    assert e1["cv_circ_weighted"] < 0.5 * e0["cv_circ_weighted"]
    assert e1["centroid_offset_m"] < 0.5 * e0["centroid_offset_m"]
    assert e1["energy_mwh"] < e0["energy_mwh"]  # uniformity costs some annual energy
    # evaluate_layout uses the same selection and scores J = E - lambda_L * spread
    ev = evaluate_layout(small_case, "radial_staggered", params, smp, 120, 4, lam=0.5)
    assert ev.uniformity["north_fraction"] == pytest.approx(e1["north_fraction"])
    assert ev.energy_mwh == pytest.approx(ev.center_energy_mwh - 0.5 * ev.uniformity["circ_spread_mwh"])
    xyz_all = generate(small_case, "radial_staggered", params)
    assert len(ev.xyz) == 120 and len(xyz_all) > 120


def test_optimize_layout_reports_uniformity(small_case):
    small_case.layout_opt.uniformity_weight = 0.5
    small_case.layout_opt.review_n_sun_bins = 6
    small_case.layout_opt.review_check_hourly = False
    res = optimize_layout(small_case)
    s = res.summary()
    u = s["uniformity"]
    assert u["weight"] == 0.5 and u["north_fraction"] < 0.7
    cert = u["selection_certificate"]
    assert cert["upper_bound_mwh"] >= cert["objective_found_mwh"] - 1e-6 and cert["gap"] < 0.05
    assert "circ_cv_review" in res.ranking and "objective_review_mwh" in res.ranking
    assert s["status"] == "ok" and s["existence"]["optimum_exists"]
