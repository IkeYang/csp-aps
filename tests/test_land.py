"""Land use in the layout objective: J(L) = E_abs - lambda_L Pi - kappa A(L), A = convex hull of the mirror centres."""

import numpy as np
import pandas as pd
import pytest

from cspaps import case_from_dict
from cspaps.layout_value import (compare_values, evaluate_layout_value, land_area_ha, recommend_land_weight,
                                 rescore)
from cspaps.scenarios import build_scenarios

from .conftest import ROOT, small_case_dict
from .test_improvement_plan import _case, _ring


def test_hull_area_analytic():
    sq = np.array([[x, y, 0.0] for x in (0, 100) for y in (0, 100)] + [[50, 50, 0], [10, 90, 3]])
    assert land_area_ha(sq) == pytest.approx(1.0)  # interior points and heights do not count
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    ring = np.column_stack([300 * np.cos(t), 300 * np.sin(t), np.zeros_like(t)])
    assert land_area_ha(ring) == pytest.approx(np.pi * 300 ** 2 / 1e4, rel=1e-5)
    assert land_area_ha(np.array([[0, 0, 0], [5, 0, 0], [9, 0, 0.0]])) == 0.0  # collinear
    assert land_area_ha(np.zeros((2, 3))) == 0.0


@pytest.fixture(scope="module")
def two_layouts():
    """A compact and a wide ring of 20 mirrors on the same scenarios: the wide one gets more heat."""
    case = _case()
    sset = build_scenarios(case)
    a = evaluate_layout_value(case, _ring(20, 60.0), sset, "compact", workers=1)
    b = evaluate_layout_value(case, _ring(20, 85.0, 0.1), sset, "wide", workers=1)
    return case, sset, a, b


def test_objective_subtracts_land_once_per_layout(two_layouts):
    _, _, a, b = two_layouts
    for v in (a, b):
        t0, t = v.totals(), v.totals(land_weight=120.0)
        assert t0["land_ha"] == pytest.approx(land_area_ha(v.xyz)) and t0["objective_mwh"] == t0["value_mwh"]
        assert t["objective_mwh"] == pytest.approx(t["value_mwh"] - 120.0 * t["land_ha"])
        assert t["value_mwh"] == t0["value_mwh"]  # the annual sum (and every scenario plan) is unchanged
        if t["objective_interval_mwh"] is not None:
            assert t["objective_interval_mwh"][0] == pytest.approx(t["value_interval_mwh"][0] - 120.0 * t["land_ha"])


def test_ranking_switches_at_the_reported_kappa(two_layouts):
    _, _, a, b = two_layouts
    ta, tb = a.totals(), b.totals()
    (big, small) = (b, a) if tb["land_ha"] > ta["land_ha"] else (a, b)
    tb_, ts_ = big.totals(), small.totals()
    if not tb_["value_mwh"] > ts_["value_mwh"]:
        pytest.skip("the larger layout is not the better one at kappa = 0 in this small case")
    k_star = (tb_["value_mwh"] - ts_["value_mwh"]) / (tb_["land_ha"] - ts_["land_ha"])
    r0 = compare_values([a, b], land_weight=0.0)
    assert r0.loc[0, "name"] == big.name and r0.loc[1, "kappa_switch"] == pytest.approx(k_star)
    assert compare_values([a, b], land_weight=0.99 * k_star).loc[0, "name"] == big.name
    assert compare_values([a, b], land_weight=1.01 * k_star).loc[0, "name"] == small.name
    r = rescore([a, b], a.lam_layout, land_weight=1.01 * k_star)  # no re-solve: same plans, same areas
    assert r.loc[0, "name"] == small.name and r["ranked_by"].iloc[0] == "objective_mwh"
    rec = recommend_land_weight(_case(), [a, b])
    assert rec["switch_points"][0]["kappa"] == pytest.approx(k_star) and rec["switch_points"][0]["layout"] == small.name


def test_land_limit_is_a_hard_constraint(two_layouts):
    _, _, a, b = two_layouts
    big, small = (a, b) if a.land_ha > b.land_ha else (b, a)
    limit = 0.5 * (a.land_ha + b.land_ha)
    for v in (a, b):
        v.max_land_ha = limit
    try:
        r = compare_values([a, b])
        row = r.set_index("name")
        assert bool(row.loc[small.name, "rankable"]) and not bool(row.loc[big.name, "rankable"])
        assert r.loc[0, "name"] == small.name and np.isnan(row.loc[big.name, "rank"])
    finally:
        for v in (a, b):
            v.max_land_ha = 0.0


def test_recommended_kappa_from_prices_or_the_field(two_layouts):
    case, _, a, b = two_layouts
    case.layout_opt.land_cost_per_ha_year = 15000.0  # per ha and year
    case.layout_opt.heat_value_per_mwh = 120.0  # per MWh_th
    rec = recommend_land_weight(case, [a, b])
    assert rec["method"] == "price" and rec["kappa"] == pytest.approx(125.0)
    case.layout_opt.land_cost_per_ha_year = 0.0
    rec = recommend_land_weight(case, [a, b], reference="compact")
    t = a.totals()
    assert rec["method"] == "elasticity" and rec["reference"] == "compact"
    assert rec["kappa"] == pytest.approx(0.05 * t["energy_mwh"] / a.land_ha)


def test_search_score_includes_the_land_term():
    from cspaps.layout_opt import LAND_LIMIT_OFFSET, evaluate_layout, land_penalty_mwh, make_samples

    case = case_from_dict(small_case_dict(), base_dir=ROOT)
    samples = make_samples(case, "sun_bins", 4)
    params = {"r_min": 40, "r_max": 110}
    e0 = evaluate_layout(case, "radial_staggered", params, samples, 60, 4)
    case.layout_opt.land_weight = 100.0
    e1 = evaluate_layout(case, "radial_staggered", params, samples, 60, 4)
    assert np.array_equal(e0.xyz, e1.xyz)  # same layout, scored by J
    assert e1.energy_mwh == pytest.approx(e0.energy_mwh - 100.0 * land_area_ha(e0.xyz))
    case.layout_opt.max_land_ha = 0.5 * land_area_ha(e0.xyz)
    assert land_penalty_mwh(case, e0.xyz) > LAND_LIMIT_OFFSET  # over the limit: below every layout within it


def test_negative_land_inputs_are_rejected():
    for key in ("land_weight", "max_land_ha", "land_cost_per_ha_year", "heat_value_per_mwh"):
        d = small_case_dict()
        d["layout_opt"][key] = -1.0
        with pytest.raises(ValueError, match=key):
            case_from_dict(d, base_dir=ROOT)


def test_rescore_from_exported_files(two_layouts, tmp_path):
    from cspaps import workflows as wf

    _, _, a, b = two_layouts
    for v in (a, b):
        v.export(tmp_path / v.name, with_aims=False)
    res = wf.task_rescore(tmp_path, a.lam_layout, land_weight=200.0)["summary"]
    rows = pd.DataFrame(res["layouts"]).set_index("layout")
    for v in (a, b):
        t = v.totals(land_weight=200.0)
        assert rows.loc[v.name, "objective_mwh"] == pytest.approx(t["objective_mwh"], rel=1e-9)
    assert res["ranked_by"] == "objective_mwh"
