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
    b = s["upper_bound"]
    assert b["objective_review_mwh"] <= b["upper_bound_mwh"] and 0 <= b["gap_objective_conservative"] < 1
    assert s["status"] == "ok" and s["existence"]["optimum_exists"]


# --------------------------------------------------------------------------- consistent candidate pricing

def test_pricing_against_the_selection_is_exact(small_case):
    """With the selected heliostats as the only blockers, each selected heliostat gets its real power."""
    smp = _samples(small_case, 4)
    xyz = generate(small_case, "radial_staggered", {"radial_spacing": 0.9, "radial_growth": 0.2,
                                                    "azimuthal_spacing": 1.1, "spacing_factor": 1.1})
    rng = np.random.default_rng(1)
    sel = np.zeros(len(xyz), dtype=bool)
    sel[rng.choice(len(xyz), 120, replace=False)] = True
    priced = annual(small_case, Field(xyz), samples=smp, aiming="center", with_flux=False, blockers=sel,
                    keep_state_power=True)
    alone = annual(small_case, Field(xyz[sel]), samples=smp, aiming="center", with_flux=False, keep_state_power=True)
    np.testing.assert_allclose(priced.state_power_w[:, sel], alone.state_power_w, rtol=1e-12, atol=1e-9)
    dense = annual(small_case, Field(xyz), samples=smp, aiming="center", with_flux=False, keep_state_power=True)
    assert np.all(priced.heliostat_energy_mwh >= dense.heliostat_energy_mwh - 1e-9)  # fewer blockers, never less


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_consistent_selection_is_swap_optimal_and_compact(small_case, lam):
    """At its own prices no single swap improves the selection, passes never lose, and no group is detached."""
    from cspaps.layout_opt import select_consistent

    smp = _samples(small_case, 6)
    xyz = generate(small_case, "radial_staggered", {"radial_spacing": 0.87, "radial_growth": 0.2,
                                                    "azimuthal_spacing": 1.0, "spacing_factor": 1.1})
    sel, info = select_consistent(small_case, xyz, smp, 120, lam, shading_points=4, iterations=8, polish=True)
    assert sel.sum() == 120
    hist = info["objective_history_mwh"]
    assert all(b >= a for a, b in zip(hist, hist[1:]))  # damped passes: no cycling
    assert lu.detached_heliostats(xyz[sel]).size == 0
    priced = annual(small_case, Field(xyz), samples=smp, aiming="center", with_flux=False, blockers=sel,
                    keep_state_power=True, shading_points=4)
    m = lu.build_model(small_case, xyz, smp, priced.state_power_w, lam)
    cur, p, g = m.objective(sel), m.e @ sel, m.profile(sel)
    add = np.flatnonzero(~sel)
    best = -np.inf
    for r in np.flatnonzero(sel):  # exhaustive single-swap neighbourhood
        pp = (p - m.e[:, r])[None, :] + m.e[:, add].T
        gg = (g - m.b[:, r])[None] + np.moveaxis(m.b[:, add], 1, 0)
        best = max(best, ((m.w[None, :] * (pp - lam * m.a_h * gg.std(axis=2))).sum(axis=1) / 1e6).max())
    if lam > 0:
        assert best <= cur + 1e-6 * abs(cur)


def test_aksai_selection_has_no_detached_heliostats():
    """Regression (audit): dense-field pricing pulled 3 heliostats 85 m beyond the north edge of the field."""
    from cspaps import load_case
    from cspaps.layout_opt import make_samples, select_consistent

    from .conftest import ROOT

    case = load_case(ROOT / "cases" / "aksai_demo.yaml")
    lo = case.layout_opt
    smp = make_samples(case, lo.search_sampling, lo.n_sun_bins, 1, seed=lo.seed)
    params = {"spacing_factor": 1.1, "radial_spacing": 0.8689, "radial_growth": 0.1831, "azimuthal_spacing": 1.0005}
    xyz = generate(case, "radial_staggered", params)
    dense = annual(case, Field(xyz), samples=smp, aiming="center", with_flux=False, shading_points=lo.shading_points,
                   keep_state_power=True)
    old, _ = lu.select(lu.build_model(case, xyz, smp, dense.state_power_w, 0.3), 473)
    assert lu.detached_heliostats(xyz[old]).size == 3  # the reported defect
    new, info = select_consistent(case, xyz, smp, 473, 0.3, lo.shading_points, iterations=6, dense=dense, polish=True)
    assert lu.detached_heliostats(xyz[new]).size == 0

    def real(sel):
        a = annual(case, Field(xyz[sel]), samples=smp, aiming="center", with_flux=False,
                   shading_points=lo.shading_points, keep_state_power=True)
        return lu.layout_metrics(case, xyz[sel], smp, a.state_power_w, 0.3)["objective_mwh"]

    assert real(new) > real(old)
    r = np.hypot(xyz[new, 0], xyz[new, 1])
    assert r.max() < 260
