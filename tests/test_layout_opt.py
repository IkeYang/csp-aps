import numpy as np
import pandas as pd
import pytest

from cspaps import load_case, sun
from cspaps.field import build_field
from cspaps.layout_opt import evaluate_layout, optimize_layout
from cspaps.sun import load_weather
from cspaps.workflows import task_optimize_layout


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
    # formal ranking: E of the winner's own per-scenario APS plans stays below the bound for any layout
    assert s["annual_energy_mwh_with_aps"] <= s["upper_bound"]["upper_bound_mwh"]
    assert s["annual_value_mwh"] <= s["annual_energy_mwh_with_aps"] + 1e-6
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


def test_review_ranking_and_validation(small_case):
    small_case.layout_opt.ranking = "legacy"  # the 0.4 constrained-APS review path
    small_case.layout_opt.refine_top = 2
    small_case.layout_opt.review_n_sun_bins = 6
    small_case.layout_opt.review_dni_bands = 2
    small_case.layout_opt.review_check_hourly = True
    small_case.layout_opt.swap_rounds = 1
    res = optimize_layout(small_case)
    r = res.ranking
    for col in ("search_rank", "final_rank", "rank_change", "energy_constrained_aps_mwh",
                "search_sampling_error_pct", "energy_hourly_center_mwh"):
        assert col in r
    assert res.validation["status"] == "ok" and res.validation["pairs_too_close"] == 0
    assert res.best.n == small_case.layout_opt.n_heliostats
    assert len(res.candidates) == len(r)


def test_capacity_shortfall(small_case):
    small_case.field.boundary = [[-60, 30], [60, 30], [60, 110], [-60, 110]]
    small_case.layout_opt.n_heliostats = 200
    small_case.layout_opt.refine_top = 0
    small_case.layout_opt.swap_rounds = 0
    res = optimize_layout(small_case)
    assert res.status == "capacity_shortfall"
    assert res.validation["shortfall"] == 200 - res.best.n > 0
    assert res.capacity["max_placeable_with_densest_parameters"] < 200


def test_optimized_case_export_keeps_weather_and_layout_loadable(small_case, tmp_path):
    weather = tmp_path / "weather.csv"
    pd.DataFrame({"time": ["2024-06-21 12:30", "2024-06-21 13:30", "2024-06-21 14:30"],
                  "dni": [800.0, 900.0, 700.0]}).to_csv(weather, index=False)
    small_case.weather.source = "file"
    small_case.weather.file = str(weather)
    small_case.annual.year = 2024
    small_case.layout_opt.n_heliostats = 24
    small_case.layout_opt.maxiter = 1
    small_case.layout_opt.popsize = 2
    small_case.layout_opt.refine_top = 0
    small_case.layout_opt.swap_rounds = 0
    small_case.layout_opt.search_aiming = "center"
    result = task_optimize_layout(small_case, tmp_path / "results")

    exported = load_case(result["files"]["case_yaml"])
    field = build_field(exported)
    series = load_weather(exported)
    assert field.n == small_case.layout_opt.n_heliostats
    assert len(series) == 3
    assert exported.weather.file == "input_weather_weather.csv"
