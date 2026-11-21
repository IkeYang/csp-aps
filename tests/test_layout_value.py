"""Acceptance tests of the annual layout value with spatial DNI maps and per-scenario APS.

Requirement items (docs/APS-LAYOUT-REQUIREMENTS.md, section 4):
  1. scenarios carry reusable site DNI maps (same position -> same DNI, moved mirror -> new DNI)
  2. independent task-1 calls reproduce the layout score and ranking
  3. 2-D penalty and two weights: rescoring the outer weight is exact on fixed plans
  4. exports alone recompute every score; statuses and their accounting
  5. small enumerable field: inner and outer layers against brute force; bounds
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cspaps import case_from_dict, sun
from cspaps import workflows as wf
from cspaps.certify import cell_energy_upper_bound, layout_upper_bound, lone_heliostat_energy, potential_map
from cspaps.dni_maps import model_maps
from cspaps.field import Field, build_field
from cspaps.layout_value import (
    annual_value, compare_values, enumerate_layouts, evaluate_layout_value, inner_case, inner_settings,
    recompute_from_csv, rescore, solve_instant,
)
from cspaps.receiver import build_grid
from cspaps.scenarios import build_scenarios, load_folder, site_mean_dni
from cspaps.simulate import annual, evaluate

from .conftest import ROOT, small_case_dict


def _case(**scen):
    d = small_case_dict()
    d["scenarios"] = {"n_scenarios": 4, "map_model": "half_swap", **scen}
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["layout_opt"]["uniformity_weight"] = 0.3
    return case_from_dict(d, base_dir=ROOT)


def _ring(n, r, phase=0.0, z=4.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.sin(a), r * np.cos(a), np.full(n, z)])


# ----------------------------------------------------------------------------- item 1: maps

def test_half_swap_maps_are_layout_independent():
    case = _case(map_contrast=0.4, map_axis_deg=90.0)
    maps = model_maps(case)
    east, west = maps["east_bright"], maps["west_bright"]
    assert east.source == "model:half_swap" and east.unit == "factor" and "bilinear" in east.interpolation
    a = np.array([[-80.0, 10.0], [80.0, 10.0]])  # layout A: one mirror west, one east
    b = np.array([[-80.0, 10.0], [0.0, 95.0]])  # layout B shares the west position
    assert east.sample(a)[0] == east.sample(b)[0]  # same position -> same DNI, whatever the layout
    assert east.sample(a)[1] > 1.2 > 0.8 > east.sample(a)[0]  # east half brighter in "east_bright"
    np.testing.assert_allclose(west.sample(a), east.sample(a)[::-1], rtol=1e-9)  # swapped map
    moved = np.array([[80.0, 10.0]])  # the west mirror moved to the east reads the new value
    assert east.sample(moved)[0] == pytest.approx(east.sample(a)[1])
    # site mean is 1 (the map only redistributes the station DNI)
    from cspaps.dni_maps import site_raster

    xs, ys, inside = site_raster(case, case.scenarios.map_cell_m)
    gx, gy = np.meshgrid(xs, ys)
    assert east.sample(np.column_stack([gx[inside], gy[inside]])).mean() == pytest.approx(1.0, rel=1e-12)


def test_box_max_bounds_bilinear_map():
    case = _case(map_model="clouds", cloud_maps=2, cloud_n=4, cloud_radius_m=[20, 50])
    rng = np.random.default_rng(1)
    for m in model_maps(case).values():
        c = rng.uniform(-120, 120, (300, 2))
        half = 7.0
        ub = m.box_max(c, half)
        for _ in range(5):
            p = c + rng.uniform(-half, half, c.shape)
            assert np.all(m.sample(p) <= ub + 1e-12)


def test_scenarios_preserve_site_dni_energy_and_round_trip(tmp_path):
    case = _case(n_scenarios=6)
    sset = build_scenarios(case)
    hourly = sun.hourly_samples(case)
    assert sset.samples.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-6)
    np.testing.assert_allclose(site_mean_dni(case, sset), sset.samples.dni, rtol=1e-9)  # site mean = scenario DNI
    assert set(sset.table()["map_id"]) == {"east_bright", "west_bright"}
    sset.save(tmp_path / "s")
    again = load_folder(tmp_path / "s")
    assert again.fingerprint() == sset.fingerprint()
    xy = np.random.default_rng(0).uniform(-100, 100, (40, 2))
    for i in range(len(sset)):
        np.testing.assert_array_equal(again.dni_at(i, xy), sset.dni_at(i, xy))


def test_uniform_maps_reproduce_the_scalar_path():
    case = _case(map_model="uniform", n_scenarios=4)
    sset = build_scenarios(case)
    fld = Field(_ring(24, 70.0))
    scalar = sun.Samples(sset.samples.elevation, sset.samples.azimuth, sset.samples.dni, sset.samples.weight_h)
    a = annual(case, fld, samples=sset.samples, with_flux=False, workers=1).energy_mwh
    b = annual(case, fld, samples=scalar, with_flux=False, workers=1).energy_mwh
    assert a == pytest.approx(b, rel=1e-12)


def test_map_aware_layout_bound_dominates_any_layout():
    case = _case(n_scenarios=4, map_contrast=0.5)
    case.field.boundary = [[-120, -120], [120, -120], [120, 120], [-120, 120]]
    sset = build_scenarios(case)
    pmap = potential_map(case, sset.samples, cell_m=8.0)
    rng = np.random.default_rng(3)
    # the per-cell bound holds at random positions inside each sampled cell, with the map's DNI there
    centres = rng.uniform(-110, 110, (60, 2))
    ub = cell_energy_upper_bound(case, centres, 4.0, sset.samples)
    pts = centres + rng.uniform(-4, 4, centres.shape)
    assert np.all(lone_heliostat_energy(case, pts, sset.samples) <= ub * (1 + 1e-9))
    lay = _ring(30, 75.0)
    bound = layout_upper_bound(case, 30, case.heliostat.diagonal * 1.1, samples=sset.samples, pmap=pmap,
                               layout_xyz=lay)
    e = annual(case, Field(lay), samples=sset.samples, with_flux=False, workers=1).summary()["absorbed_energy_mwh"]
    assert bound["upper_bound_mwh"] >= bound["layout_lone_sum_mwh"] >= e > 0


# ----------------------------------------------------------------------------- items 2-4: value, rescoring, export

@pytest.fixture(scope="module")
def two_layouts():
    case = _case(n_scenarios=4)
    sset = build_scenarios(case)
    a = _ring(30, 62.0)
    b = np.vstack([_ring(15, 62.0), _ring(15, 90.0, phase=0.1)])
    settings = inner_settings(case)
    va = evaluate_layout_value(case, a, sset, "A", settings=settings, workers=1)
    vb = evaluate_layout_value(case, b, sset, "B", settings=settings, workers=2)
    return case, sset, settings, va, vb


def test_independent_task1_reproduces_scores_and_ranking(two_layouts, tmp_path):
    case, sset, settings, va, vb = two_layouts
    sdir = tmp_path / "scen"
    sset.save(sdir)
    rank = compare_values([va, vb])
    recomputed = {}
    for v in (va, vb):
        csv = tmp_path / f"{v.name}.csv"
        pd.DataFrame({"id": v.ids, "x": v.xyz[:, 0], "y": v.xyz[:, 1], "z": v.xyz[:, 2]}).to_csv(csv, index=False)
        rows = []
        for sid in sset.ids:
            r = wf.task_scenario_aps(case, tmp_path / "runs", sid, scenarios_dir=sdir, layout=csv)
            row = r["summary"]["scenario"]
            mine = v.table().set_index("scenario_id").loc[sid]
            assert row["p_abs_mw"] == mine["p_abs_mw"] and row["u_mw"] == mine["u_mw"]  # bit-identical plans
            assert row["status"] == mine["status"]
            rows.append(row)
        recomputed[v.name] = annual_value(pd.DataFrame(rows), case.layout_opt.uniformity_weight)["value_mwh"]
    assert recomputed[va.name] == pytest.approx(va.totals()["value_mwh"], abs=1e-9)
    assert recomputed[vb.name] == pytest.approx(vb.totals()["value_mwh"], abs=1e-9)
    order = sorted(recomputed, key=recomputed.get, reverse=True)
    assert order == list(rank["name"])


def test_outer_weight_rescoring_is_exact(two_layouts):
    case, sset, settings, va, vb = two_layouts
    lam_a = settings.lam_aps
    t0 = va.table()
    # equal weights: V is the hour-weighted sum of the inner objectives J_s = P_s - lambda_A U_s
    j = (t0["hours"] * t0["j_inner_mw"] * t0["counts_in_value"]).sum()
    assert va.totals(lam_a)["value_mwh"] == pytest.approx(j, rel=1e-12)
    for lam in (0.0, 0.1, 1.0, 2.5):
        t = va.table(lam)
        pd.testing.assert_series_equal(t["p_abs_mw"], t0["p_abs_mw"])  # plans untouched
        pd.testing.assert_series_equal(t["u_mw"], t0["u_mw"])
        tot = va.totals(lam)
        assert tot["value_mwh"] == pytest.approx(
            va.totals(lam_a)["value_mwh"] - (lam - lam_a) * tot["penalty_mwh"], rel=1e-12, abs=1e-9)
        assert tot["value_mwh"] == pytest.approx(tot["energy_mwh"] - lam * tot["penalty_mwh"], rel=1e-12)
    r = rescore([va, vb], 1.0)
    for _, row in r.iterrows():
        assert row["delta_value_mwh"] == pytest.approx(row["delta_energy_mwh"] + row["delta_uniformity_mwh"], abs=1e-9)


def test_exports_recompute_every_score(two_layouts, tmp_path):
    case, sset, settings, va, vb = two_layouts
    files = wf.export_values({"A": va, "B": vb}, tmp_path / "values", compare_values([va, vb]))
    for v in (va, vb):
        csv = tmp_path / "values" / v.name / "scenario_values.csv"
        rec = recompute_from_csv(csv)
        assert rec["value_mwh"] == pytest.approx(v.totals()["value_mwh"], rel=1e-9)
        assert rec["energy_mwh"] == pytest.approx(v.totals()["energy_mwh"], rel=1e-9)
        t = pd.read_csv(csv)
        assert {"scenario_id", "map_id", "hours", "p_abs_mw", "u_mw", "status", "counts_in_value",
                "aims_file"} <= set(t.columns)
        aims = pd.read_csv(csv.parent / t["aims_file"].iloc[0])
        plan = v.records[0].plan
        np.testing.assert_allclose(aims["aim_z"], np.round(plan.aim_z, 6))
        assert aims["active"].sum() == plan.n_active
        # the exported per-mirror absorbed power adds up to P_s for counted scenarios
        if t["counts_in_value"].iloc[0]:
            assert aims["absorbed_kw"].sum() / 1e3 == pytest.approx(t["p_abs_mw"].iloc[0], rel=2e-3)
    other = wf.task_rescore(tmp_path / "values", 0.0)["summary"]["layouts"]
    assert {r["layout"] for r in other} == {"A", "B"}
    assert files["ranking_csv"].exists()


def test_status_accounting_rules():
    t = pd.DataFrame({"hours": [10.0, 5.0, 3.0, 2.0], "p_abs_mw": [2.0, 1.0, 9.0, 9.0], "u_mw": [1.0, 0.5, 9.0, 9.0],
                      "status": ["feasible", "feasible_shutdown", "no_feasible_found", "proven_infeasible"]})
    v = annual_value(t, 0.5)
    assert v["energy_mwh"] == 20.0 and v["penalty_mwh"] == 10.0 and v["value_mwh"] == 15.0 and v["complete"]
    t.loc[1, "status"] = "not_solved"
    assert not annual_value(t, 0.5)["complete"] and np.isnan(annual_value(t, 0.5)["value_mwh"])


def test_impossible_minimum_load_is_proven_infeasible():
    case = _case()
    case.uniformity.min_load_mw = 1e4
    fld = Field(_ring(6, 60.0))
    p = solve_instant(case, fld, 50.0, 180.0, 900.0, inner_settings(case))
    assert p.status == "proven_infeasible" and p.p_abs_w == 0.0 and p.u_w == 0.0 and not p.counts


# ----------------------------------------------------------------------------- item 5: brute force

def _brute_force_inner(case, fld, el, az, dni, settings):
    """Every joint pose of a tiny problem, evaluated and verified independently of the solver."""
    from cspaps.aps import build_problem

    c = inner_case(case, settings)
    grid = build_grid(c.receiver, settings.grid_shape[1], settings.grid_shape[0])
    prob = build_problem(c, fld, el, az, dni, grid=grid)
    best, rejected = -np.inf, 0
    for ch in product(*[list(range(prob.k)) + [-1]] * fld.n):
        ch = np.asarray(ch)
        snap = evaluate(c, fld, el, az, dni, aim=prob.aims(ch), active=ch >= 0, grid=grid)
        if not snap.verify(c.receiver).ok:
            rejected += 1
            continue
        th = snap.thermal_objective(c.uniformity)
        best = max(best, th["power_w"] - settings.lam_aps * th["spread_w"])
    return best, rejected


def test_small_field_inner_and_outer_layers_against_brute_force():
    case = _case(n_scenarios=2, map_contrast=0.5)
    case.uniformity.weight = case.layout_opt.uniformity_weight = 0.05
    case.aps.azimuth_offsets_deg = [0.0]
    case.aps.n_height_levels = 3
    case.aps.coupled_exact_limit = 4096
    case.receiver.max_flux_kw_m2 = 30.0  # the flux limit binds: stacked beams are rejected
    sset = build_scenarios(case)
    settings = inner_settings(case)
    cand = np.array([[0.0, 60.0, 4.0], [8.0, 75.0, 4.0], [-8.0, 75.0, 4.0], [30.0, 62.0, 4.0]])
    # inner: the scenario solve is the proven optimum of the enumerable joint-pose problem
    fld = Field(cand[:3])
    s = sset.samples
    for i in range(len(sset)):
        dni = sset.dni_at(i, fld.xyz[:, :2])
        plan = solve_instant(case, fld, float(s.elevation[i]), float(s.azimuth[i]), dni, settings)
        best, rejected = _brute_force_inner(case, fld, float(s.elevation[i]), float(s.azimuth[i]), dni, settings)
        assert rejected > 0 and plan.status == "feasible" and plan.n_active >= 2  # non-trivial instance
        assert plan.proven_optimal
        assert plan.objective_w == pytest.approx(best, rel=1e-9)
    # outer: all C(4, 3) layouts; the best V is attained and below the map-aware packing bound
    table = enumerate_layouts(case, cand, 3, sset, settings)
    assert len(table) == 4 and table["proven_optimal_all"].all()
    assert table["value_mwh"].nunique() == 4  # the maps and the geometry separate the layouts
    vals = [evaluate_layout_value(case, cand[list(sub)], sset, str(sub), settings=settings, workers=1)
            for sub in table["subset"]]
    ranking = compare_values(vals)
    assert ranking.loc[0, "value_mwh"] == pytest.approx(table["value_mwh"].max(), rel=1e-12)
    case.field.boundary = [[-80, -80], [80, -80], [80, 80], [-80, 80]]
    bound = layout_upper_bound(case, 3, case.heliostat.diagonal * 1.1, samples=sset.samples,
                               pmap=potential_map(case, sset.samples, cell_m=6.0))
    assert bound["upper_bound_mwh"] >= table["energy_mwh"].max() >= table["value_mwh"].max()


# ----------------------------------------------------------------------------- cloud before / after

def test_cloud_compare_states(tmp_path):
    d = small_case_dict()
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["receiver"]["max_flux_kw_m2"] = 260.0
    d["field"]["generator"]["r_max"] = 75
    case = case_from_dict(d, base_dir=ROOT)
    r = wf.task_cloud_compare(case, tmp_path, "2025-06-21 12:30", n_clouds=2, seed=2)
    st = {s["key"]: s for s in r["states"]}
    assert r["summary"]["shaded_heliostats"] > 0
    assert st["cloud_keep"]["p_abs_mw"] < st["clear"]["p_abs_mw"]  # the cloud removes heat
    # re-solving never does worse than keeping plan A (plan A competes in the re-solve)
    assert st["cloud_aps"]["objective_mw"] >= st["cloud_keep"]["objective_mw"] - 1e-9
    # the clearing-safe plan stays within the hard flux limit when the cloud clears
    fld = r["field"]
    s = st["cloud_safe"]
    c = inner_case(case, inner_settings(case, 0.3))
    snap = evaluate(c, fld, r["summary"]["sun_elevation_deg"], r["summary"]["sun_azimuth_deg"],
                    r["summary"]["dni_clear_w_m2"], aim=(s["aim_azimuth"], s["aim_z"]), active=s["active"],
                    grid=r["grid"])
    assert snap.verify(case.receiver).peak_ratio <= 1.0 + 1e-9
    for k in ("summary_json", "states_csv", "aims_csv", "compare_png"):
        assert Path(r["files"][k]).exists()


# ----------------------------------------------------------------------------- layout optimisation and CLI

def test_optimize_layout_formal_ranking_with_initial_field():
    from cspaps.layout_opt import optimize_layout

    case = _case(n_scenarios=3)
    case.layout_opt.refine_top = 2
    case.layout_opt.dense_check_scenarios = 4
    case.layout_opt.review_check_hourly = False
    res = optimize_layout(case)
    s = res.summary()
    f = s["formal"]
    names = [r["name"] for r in f["ranking"]]
    assert "initial" in names and any(n.startswith("search_") for n in names)
    assert f["winner"] == names[0] and s["ranking_basis"].startswith("aps_value")
    vals = res.formal["values"]
    best = vals[f["winner"]].totals()
    assert best["value_mwh"] == max(v.totals()["value_mwh"] for v in vals.values())
    assert np.allclose(np.sort(res.best.xyz, axis=0), np.sort(vals[f["winner"]].xyz, axis=0))
    assert res.best.n == case.layout_opt.n_heliostats and res.validation["status"] == "ok"
    # every layout was solved on the same scenario set with the same inner settings
    assert len({v.scenario_fingerprint for v in vals.values()}) == 1
    assert len({v.settings.digest() for v in vals.values()}) == 1
    ub = s["upper_bound"]
    assert ub["upper_bound_mwh"] >= ub["found_energy_mwh"] >= ub["value_found_mwh"]
    assert f["dense_check"]["n_scenarios"] == 4 and f["dense_check"]["rows"]
    for sw in f["swaps"]:
        assert sw["accepted"] == (sw["gain_value_mwh"] > 0)


def test_cli_scenarios_value_and_rescore(tmp_path):
    import json

    import yaml

    from cspaps.cli import main

    d = small_case_dict()
    d["scenarios"] = {"n_scenarios": 2, "map_model": "half_swap"}
    d["field"]["generator"]["r_max"] = 70
    d["layout_opt"]["n_heliostats"] = 40
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["scenarios", str(path), "--out", str(out)]) == 0
    sdir = next((out / "tiny").glob("scenarios_*")) / "scenario_set"
    assert (sdir / "scenarios.csv").exists() and (sdir / "dni_maps.csv").exists()
    lay = tmp_path / "other.csv"
    fld = build_field(case_from_dict(d, base_dir=ROOT))
    r = np.hypot(fld.xyz[:, 0], fld.xyz[:, 1])
    keep = np.argsort(-r)[:40]
    pd.DataFrame({"id": fld.ids[keep], "x": fld.xyz[keep, 0], "y": fld.xyz[keep, 1]}).to_csv(lay, index=False)
    assert main(["layout-value", str(path), "--layouts", "initial", str(lay), "--scenarios", str(sdir),
                 "--out", str(out)]) == 0
    run = next((out / "tiny").glob("layout_value_*"))
    summary = json.loads((run / "layout_value.json").read_text(encoding="utf-8"))
    assert summary["equal_heliostat_count"] and len(summary["ranking"]) == 2
    assert main(["scenario-aps", str(path), "--id", "S01", "--scenarios", str(sdir), "--layout", str(lay),
                 "--out", str(out)]) in (0, 2)
    row = json.loads(next((out / "tiny").glob("scenario_aps_*/scenario_S01.json")).read_text(encoding="utf-8"))["scenario"]
    exported = pd.read_csv(run / "values" / "other" / "scenario_values.csv").set_index("scenario_id").loc["S01"]
    assert row["p_abs_mw"] == pytest.approx(exported["p_abs_mw"], rel=1e-12)
    assert main(["rescore", str(run / "values"), "--lambda-layout", "0"]) == 0
    assert main(["cloud-compare", str(path), "--time", "2025-06-21 12:30", "--clouds", "1", "--out", str(out)]) == 0


def test_formal_ranking_cross_checks_cloud_maps():
    """With cloud-shadow maps the winner is re-scored on independent maps (transient shadows must not decide)."""
    from cspaps.layout_opt import LayoutEval, formal_ranking
    from cspaps.layout_value import baseline_layout

    case = _case(map_model="clouds", cloud_maps=2, cloud_n=2, n_scenarios=2)
    case.layout_opt.n_heliostats = 30
    case.layout_opt.formal_swap_rounds = 0
    base = baseline_layout(case, 30)[0]
    # the search layout equals the initial field: equal V, the stable sort ranks the search layout first
    evals = [LayoutEval({"x": 1}, 30, base.copy(), 0.0)]
    hourly = sun.hourly_samples(case)
    f = formal_ranking(case, evals, hourly, build_scenarios(case).samples)
    assert set(f["values"]) == {"search_1", "initial"} and f["winner"] == "search_1"
    assert f["values"]["search_1"].totals()["value_mwh"] == f["values"]["initial"].totals()["value_mwh"]
    c = f["cross_maps"]
    assert c["seed"] == case.scenarios.seed + 1000 and len(c["rows"]) == 2
    assert c["fingerprint"] != f["scenario_set"].fingerprint()  # really other maps
    assert c["lead_independent_mwh"] == 0.0 and not c["winner_still_ahead"]  # identical layouts tie


@pytest.mark.parametrize("name", ["aksai_demo", "hami_2024", "port_augusta_2024"])
def test_site_cases_build_their_scenario_sets(name):
    """The three site studies: weather year, initial field of N heliostats, DNI-energy-preserving scenario set."""
    from cspaps import load_case
    from cspaps.layout_value import baseline_layout

    case = load_case(ROOT / "cases" / f"{name}.yaml")
    hourly = sun.hourly_samples(case)
    assert len(hourly) > 3000  # NASA POWER 2024, sun above 5 degrees
    sset = build_scenarios(case, hourly=hourly)
    assert sset.samples.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-6)
    assert len(sset.subset_maps()) >= 2 and all(m.unit == "factor" for m in sset.maps.values())
    assert baseline_layout(case, case.layout_opt.n_heliostats)[0].shape == (case.layout_opt.n_heliostats, 3)
