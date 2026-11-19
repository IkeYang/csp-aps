import numpy as np
import pytest

from cspaps import sun
from cspaps.layout_opt import evaluate_layout, optimize_layout


def test_evaluate_layout_selects_best(small_case):
    bins = sun.sun_bins(sun.hourly_samples(small_case), 6)
    params = {"radial_spacing": 0.9, "radial_growth": 0.2, "azimuthal_spacing": 1.0}
    ev = evaluate_layout(small_case, "radial_staggered", params, bins, n_select=100, shading_points=4)
    assert ev.n == 100 and ev.n_candidates > 100
    full = evaluate_layout(small_case, "radial_staggered", params, bins, n_select=None, shading_points=4)
    # the kept heliostats are the most productive ones
    assert ev.energy_mwh / ev.n > full.energy_mwh / full.n
    bad = evaluate_layout(small_case, "radial_staggered", {**params, "radial_spacing": 0.1}, bins, 100)
    assert bad.energy_mwh == 0.0  # invalid parameters score zero instead of crashing


def test_optimize_layout_runs(small_case):
    res = optimize_layout(small_case)
    s = res.summary()
    assert s["n_heliostats"] == small_case.layout_opt.n_heliostats
    assert s["annual_energy_mwh_center_aiming"] > 0
    assert s["annual_energy_mwh_with_aps"] is not None
    assert s["annual_energy_mwh_with_aps"] <= s["annual_energy_mwh_center_aiming"] * 1.001
    assert len(res.history) >= 2
    assert res.history["energy_mwh"].is_monotonic_decreasing
    assert set(res.history["stage"]) == {"global", "local"}
    assert np.all(np.isfinite(res.field.xyz))


def test_optimize_layout_parallel(small_case):
    small_case.layout_opt.workers = 2
    small_case.layout_opt.refine_top = 0
    res = optimize_layout(small_case)
    assert res.summary()["annual_energy_mwh_with_aps"] is None
    assert len(res.history) >= 2
