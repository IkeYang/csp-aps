"""Improvement plan step 3: stress-state ray-trace checks, the subfield solver and the solver benchmarks."""

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cspaps.aps import build_problem, solve, solve_subfield, subfield_of
from cspaps.benchmark import inner_benchmark, inner_states, outer_benchmark
from cspaps.field import build_field
from cspaps.layout_value import inner_case, inner_settings
from cspaps.receiver import build_grid
from cspaps.soltrace import export_plan
from cspaps.validation import (SUITE, clear_dni, compare_state, edge_transmittance, solar_noon, suite_states,
                               summer_solstice, tighten_limits)


def _noon(case, fld):
    ts, el, az = solar_noon(case, summer_solstice(case, 2025))
    return ts, el, az, np.full(fld.n, clear_dni(case, el))


def test_edge_transmittance_matches_the_linear_ramp():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [-100.0, 0.0], [10.0, 5.0], [0.0, 50.0]])
    t = edge_transmittance(xy, toward_deg=90.0, width_m=40.0, t_shadow=0.2)  # shadow on the east side
    # on the line: half way; deep east: shadow; deep west: clear; 10 m east: 1 - 0.8 x (10/40 + 0.5)
    assert t == pytest.approx([0.6, 0.2, 1.0, 1.0 - 0.8 * 0.75, 0.6])


def test_near_limit_plan_keeps_its_power_after_the_final_repair(small_case):
    """A uniformity move left the fine-grid peak between the margin and the limit; the final repair from the
    current plan (only mirrors feeding a violated point move) must end verified "ok" without shedding power."""
    fld = build_field(small_case)
    hi, near = suite_states(small_case, fld, ("mixed_high_load", "near_limit"))
    assert near.note["center_aim_peak_to_limit_after"] == pytest.approx(1.3)
    p = near.plan
    assert p.status == "feasible" and p.flux_status == "ok"
    assert p.n_active >= 0.9 * fld.n
    assert p.p_abs_w >= 0.9 * hi.plan.p_abs_w
    # the plan is checked against the tightened limit, independently of the solver
    from cspaps.simulate import evaluate
    snap = evaluate(near.case, fld, near.elevation, near.azimuth, near.dni, aim=(p.aim_azimuth, p.aim_z),
                    active=p.active)
    chk = snap.verify(near.case.receiver)
    assert chk.status == "ok" and chk.peak_ratio <= 1 - near.case.aps.safety_margin + 1e-9


def test_subfield_shares_keep_the_union_within_the_limit(small_case):
    """Sum over sectors of share_k = 1 at every point, so the union of the sector plans never exceeds the
    limit on the response points (checked point by point, brute force), and the full solve verifies ok."""
    fld = build_field(small_case)
    ts, el, az, dni = _noon(small_case, fld)
    c = inner_case(tighten_limits(small_case, 0.3), inner_settings(small_case, 0.0))
    grid = build_grid(c.receiver, c.receiver.n_azimuth, c.receiver.n_height)
    prob = build_problem(c, fld, el, az, dni, grid=grid)
    choice, info = solve_subfield(prob, c.aps, 4)
    assert info["subfield_sizes"] == [int(np.sum(subfield_of(prob, 4) == k)) for k in range(4)]
    assert sum(info["subfield_sizes"]) == fld.n and min(info["subfield_sizes"]) > 0
    assert info["subfield_parallel_s"] <= info["subfield_serial_s"] + 1e-12
    flux = prob.flux(choice)
    assert np.all(flux <= prob.limit * (1 + 1e-6))
    assert (choice >= 0).sum() > 0.5 * fld.n
    res = solve(prob, "subfield", c.aps)
    assert res.status == "ok"


def test_stress_states_separate_physical_error_and_solver_gap(small_case):
    fld = build_field(small_case)
    states = suite_states(small_case, fld)
    assert [s.key for s in states] == list(SUITE)
    by = {s.key: s for s in states}
    assert 8.0 <= by["low_sun"].elevation <= 15.0
    d = by["cloud_edge"].dni
    assert d.min() == pytest.approx(0.2 * d.max(), rel=1e-6) and d.max() == pytest.approx(by["mixed_high_load"].dni.max())
    assert by["cloud_edge"].note["n_shaded"] > 0 and by["cloud_edge"].note["n_clear"] > 0
    cell_m = None
    for s in states:
        r = compare_state(s, fld, rays=600, seeds=2, seed0=11)
        ph, so = r["physical"], r["solver"]
        assert so["status"] == "feasible" and so["flux_status"] == "ok"
        assert ph["mcrt"]["seeds"] == [11, 12] and ph["mcrt"]["rays_per_heliostat"] == 600
        assert ph["grid"] == [small_case.receiver.n_azimuth, small_case.receiver.n_height]
        assert abs(ph["power_on_receiver_mw"]["rel_diff"]) < 0.02
        assert ph["power_on_receiver_mw"]["mcrt_std_error"] > 0
        grid = build_grid(s.case.receiver)
        cell_m = max(s.case.receiver.radius * np.radians(360 / grid.azimuth_deg.size), grid.z[1] - grid.z[0])
        assert ph["hotspot"]["distance_m"] <= 2.5 * cell_m
        assert ph["mcrt_within_limit"]
        assert {"physical", "solver"} <= set(r) and "gap" in so and "rel_diff" in ph["power_on_receiver_mw"]
    near = compare_state(by["near_limit"], fld, rays=600, seeds=2)
    assert near["physical"]["peak_to_limit"]["analytic_verified_fine"] > 0.6  # the limit really binds


class _Rec:
    """Records attribute writes; any method call returns another recorder."""

    def __init__(self, name="root"):
        self.__dict__["_name"] = name
        self.__dict__["front"] = SimpleNamespace()
        self.__dict__["back"] = SimpleNamespace()
        self.__dict__["children"] = []

    def __getattr__(self, item):
        def call(*a, **k):
            r = _Rec(f"{self._name}.{item}{a}")
            self.children.append((item, a, r))
            return r
        return call


def test_soltrace_export_scales_reflectivity_by_the_mirror_dni(small_case, tmp_path):
    fld = build_field(small_case)
    ts, el, az, dni = _noon(small_case, fld)
    dni = dni * edge_transmittance(fld.xyz[:, :2], 90.0)
    res = export_plan(small_case, fld, ts, el, az, dni, None, None, tmp_path, with_mcrt=False)
    data = json.loads(res["files"]["benchmark_json"].read_text())
    assert data["dni_w_m2"] == pytest.approx(dni.max())
    f = np.array([h["dni_factor"] for h in data["heliostats"]])
    assert f == pytest.approx(dni / dni.max())
    assert "cspaps_mcrt" not in data and res["summary"]["dni_w_m2"]["per_mirror"]
    spec = importlib.util.spec_from_file_location("drv", res["files"]["pysoltrace_driver"])
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    PT = drv.build(_Rec, lambda *a: a, data, 1000, data["dni_w_m2"])
    assert PT.dni == pytest.approx(dni.max())
    optics = {a[0]: r for item, a, r in PT.children if item == "add_optic"}
    stage = [r for item, a, r in PT.children if item == "add_stage"][0]
    elements = [r for item, a, r in stage.children if item == "add_element"][:fld.n]
    rho = data["optics"]["reflectivity"]
    for h, e in zip(data["heliostats"], elements):
        expected = rho * (1.0 if round(h["dni_factor"], 2) >= 0.995 else round(h["dni_factor"], 2))
        assert e.optic.front.reflectivity == pytest.approx(expected)
    assert "Reflector" in optics and len(optics) > 3  # scaled optics were created for the shaded mirrors
    # reflected power = DNI_max x rho x factor = DNI_i x rho up to the 0.01 grouping of the factors
    eff = np.array([e.optic.front.reflectivity for e in elements]) * data["dni_w_m2"]
    assert eff == pytest.approx(rho * dni, abs=0.006 * rho * dni.max())


def test_inner_benchmark_measures_every_method_against_one_bound(small_case):
    c = tighten_limits(small_case, 0.3)
    fld = build_field(c)
    states = inner_states(c, fld, 1)
    df = inner_benchmark(c, fld, states, ("greedy", "subfield"), budget_s=30)
    assert set(df["method"]) == {"greedy", "subfield"}
    assert (df["status"] == "feasible").all()
    ub = df[["upper_bound_mw", "dual_bound_mw"]].min(axis=1, skipna=True)
    assert np.all(df["state_best_bound_mw"] <= ub + 1e-12)
    assert np.all(df["gap_to_best_bound"] >= -1e-9) and np.all(df["loss_to_best_found"] >= -1e-12)
    assert df.loc[df["method"] == "subfield", "parallel_time_s"].notna().all()


def test_outer_benchmark_scores_both_searches_on_held_out_scenarios(small_case):
    r = outer_benchmark(small_case, seeds=(1,), budget_s=6, max_evals=6)
    runs = r["runs"]
    assert list(runs["method"]) == ["de", "multistart_free"]
    assert (runs["site_status"] == "ok").all() and (runs["pairs_too_close"] == 0).all()
    assert r["held_out"]["fingerprint"] != r["train"]["fingerprint"]
    assert r["held_out"]["n"] == r["train"]["n"] + 5
    assert sorted(runs["rank_train"]) == [1, 2] or runs["rank_train"].nunique() == 1
    assert runs["held_out_value_mwh"].gt(0).all() and runs["coverage"].eq(1.0).all()


def test_gui_stress_states_and_solver_benchmark_tabs(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from tests.conftest import ROOT, small_case_dict

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    at = AppTest.from_file(str(Path(ROOT) / "src" / "cspaps" / "gui_app.py"), default_timeout=600)
    at.session_state["case_dict"] = small_case_dict()
    at.session_state["base_dir"] = str(ROOT)
    at.session_state["ver"] = 1
    at.run()
    at.radio(key="page").set_value("光学校验").run()
    at.multiselect(key="vs_keys").set_value(["cloud_edge", "near_limit"]).run()
    at.number_input(key="vs_rays").set_value(300).run()
    at.number_input(key="vs_seeds").set_value(2).run()
    at.button(key="btn_vsuite").click().run()
    assert not at.exception
    frames = [d.value for d in at.dataframe]
    phys = next(f for f in frames if "热点距离 (m)" in f.columns)
    gap = next(f for f in frames if "求解差距 (%)" in f.columns)
    assert list(phys["工况"]) == ["云影边缘", "逼近热流限制"] and list(gap["工况"]) == ["云影边缘", "逼近热流限制"]
    assert phys["种子"].iloc[0] == "0, 1" and (phys["每镜光线"] == 300).all()
    assert (gap["方案状态"] == "feasible").all()
    st_dir = Path(at.session_state["res_vsuite"]["run_dir"]) / "soltrace" / "cloud_edge" / "benchmark_case.json"
    assert st_dir.exists()
    at.radio(key="sb_part").set_value("inner").run()
    at.number_input(key="sb_states").set_value(1).run()
    at.number_input(key="sb_budget_inner").set_value(20).run()
    at.multiselect(key="sb_inner_m").set_value(["greedy", "subfield"]).run()
    at.button(key="btn_sbench").click().run()
    assert not at.exception
    agg = at.session_state["res_sbench"]["inner_summary"]
    assert list(agg["method"]) == ["greedy", "subfield"] and (agg["feasible"] == 1).all()


def test_cli_validation_suite_and_solver_benchmark(tmp_path):
    import yaml

    from cspaps.cli import main
    from tests.conftest import small_case_dict

    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(small_case_dict()), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["validation-suite", str(path), "--states", "cloud_edge", "--rays", "200", "--seeds", "2",
                 "--no-soltrace", "--out", str(out)]) == 0
    table = pd.read_csv(next(out.rglob("validation_suite.csv")))
    assert list(table["state"]) == ["cloud_edge"] and table["seeds"].iloc[0] == "0,1"
    assert main(["solver-benchmark", str(path), "--part", "inner", "--states", "1", "--budget", "10",
                 "--limit-factor", "0.3", "--weight", "0", "--out", str(out)]) == 0
    rec = json.loads(next(out.rglob("solver_benchmark.json")).read_text(encoding="utf-8"))
    assert rec["limit_factor"] == 0.3 and rec["lambda_aps"] == 0.0
    assert {r["method"] for r in rec["inner"]["summary"]} == {"greedy", "milp", "subfield", "greedy_rounding"}
