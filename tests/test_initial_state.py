"""Initial state of an offline plan and the price / limits of mirror moves away from it.

* the "spread" initial plan uses the whole receiver height, keeps every image on the receiver and balances each
  sector's power over the heights in proportion to the allowed flux;
* with a move price the selected plan is the exact optimum of objective - price (brute force over all joint
  poses), and the certificate stays one of the unpriced objective (valid bound, honest gap);
* the shift / rotation limits are hard constraints of the plan.
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from cspaps import load_case
from cspaps.aps import build_problem, move_stats, solve_spread, spread_levels
from cspaps.field import build_field
from cspaps.receiver import surface_point
from cspaps.resources import demo_case_path
from cspaps.simulate import evaluate, sun_state
from cspaps.workflows import read_aims_csv, task_aim, task_pareto, write_aims_csv

TIME = "2024-03-19 08:30"


@pytest.fixture
def pair(tmp_path):
    case = load_case(demo_case_path())
    case.receiver.center = (0, 0, 55)
    case.receiver.height, case.receiver.diameter = 20, 12
    case.aps.n_height_levels, case.aps.azimuth_offsets_deg = 3, [0.0]
    case.aps.height_span = 0.75
    case.field.exclusion_radius = 10
    path = tmp_path / "field.csv"
    pd.DataFrame({"x": [1.659256, 16.787252], "y": [-73.187585, -74.519679], "z": [5, 5]}).to_csv(path, index=False)
    case.field.layout_file = str(path)
    case.uniformity.weight = 0.0  # objective = absorbed power (weighted mode, as in test_coupled_design)
    return case


def test_spread_plan_fills_the_height_within_the_image_limits(small_case):
    case = small_case
    case.receiver.height = 16.0  # tall enough for several heights to fit the images
    fld = build_field(case)
    _, el, az, dni = sun_state(case, "2025-06-21 11:00")
    prob = build_problem(case, fld, el, az, dni)
    choice = solve_spread(prob)
    assert np.array_equal(choice, solve_spread(prob))  # deterministic
    ks, levels, adm = spread_levels(prob)
    col = {k: j for j, k in enumerate(ks)}
    j = np.array([col[c] for c in choice])
    assert adm[np.arange(prob.n), j].all()  # every image stays on the receiver
    assert np.all(prob.cand_az[np.arange(prob.n), choice] == prob.cand_az[:, prob.center_k])  # on its facing azimuth
    used = np.unique(j)
    assert used.size >= 3  # the plan uses the height, not one band
    # loads balanced: per height, each sector's share stays within one mirror of the even split over the
    # heights its mirrors may use (equal capacity: uniform limit)
    w = prob.rs.unit_power[:, prob.center_k] * prob.dni_obj
    sector = np.floor((prob.cand_az[:, prob.center_k] % 360.0) / (360.0 / max(2 * case.receiver.n_panels, 8)))
    for s in np.unique(sector):
        m = sector == s
        free = m & adm.all(axis=1)  # mirrors that may use every height
        if free.sum() < 2 * levels.size:
            continue
        load = np.bincount(j[m], weights=w[m], minlength=levels.size)
        assert load.max() - load.min() <= w[m].max() * (1 + 1e-9)


def test_lower_limit_receives_less_aim_power(small_case):
    case = small_case
    case.receiver.height = 16.0
    case.receiver.limit_height_profile = [1.0, 1.0, 0.4]  # the top may take less flux
    fld = build_field(case)
    _, el, az, dni = sun_state(case, "2025-06-21 11:00")
    prob = build_problem(case, fld, el, az, dni)
    ks, levels, _ = spread_levels(prob)
    w = prob.rs.unit_power[:, prob.center_k] * prob.dni_obj
    load = np.bincount(np.searchsorted(ks, solve_spread(prob)), weights=w, minlength=levels.size)
    c = levels.size // 2
    assert load[-1] < 0.75 * load[c]


def _working_state(case, fld, prob, choice, path):
    aim = prob.aims(np.asarray(choice))
    write_aims_csv(path, fld, case, aim, np.asarray(choice) >= 0)
    return path


@pytest.mark.parametrize("prev_choice, mu", [((2, 2), 0.005), ((2, 2), 0.02), ((1, 1), 0.02), ((0, 0), 0.001)])
def test_priced_plan_is_the_brute_force_optimum(pair, tmp_path, prev_choice, mu):
    pair.aps.move_penalty_per_m = mu
    fld = build_field(pair)
    _, el, az, _ = sun_state(pair, TIME)
    prob = build_problem(pair, fld, el, az, 650)
    prev_csv = _working_state(pair, fld, prob, prev_choice, tmp_path / "working.csv")
    prev_xyz = surface_point(pair.receiver, *prob.aims(np.asarray(prev_choice)))
    cost = mu * prob.absorbed_power[:, prob.center_k][:, None] * np.linalg.norm(prob.aim_xyz - prev_xyz[:, None], axis=-1)
    raw, priced = {}, {}
    for ch in product(range(-1, 3), repeat=2):
        c = np.array(ch)
        snap = evaluate(pair, fld, el, az, 650, aim=prob.aims(c), active=c >= 0, grid=prob.grid)
        if snap.verify(pair.receiver).ok:
            raw[ch] = snap.p_absorbed
            priced[ch] = raw[ch] - sum(cost[i, k] for i, k in enumerate(ch) if k >= 0)
    result = task_aim(pair, tmp_path, TIME, dni=650, method="greedy", prev_aims=str(prev_csv))
    chosen = tuple(int(v) for v in result["aim_result"].choice)
    cert = result["certificate"]
    assert priced[chosen] == pytest.approx(max(priced.values()), rel=1e-9)
    assert cert["priced_objective_mw"] * 1e6 == pytest.approx(max(priced.values()), rel=1e-9)
    assert cert["objective_found_mw"] * 1e6 == pytest.approx(raw[chosen], rel=1e-9)
    assert cert["upper_bound_mw"] * 1e6 == pytest.approx(max(raw.values()), rel=1e-9)  # bound of the unpriced objective
    # (the working-state CSV stores aims to 1e-6: a residual shift of ~1e-7 m costs micro-watts)
    assert cert["move_cost_mw"] * 1e6 == pytest.approx(raw[chosen] - priced[chosen], rel=1e-6, abs=1e-3)
    unpriced_best = max(raw, key=raw.get)
    assert cert["global_optimum_proven_on_candidate_set"] == (raw[chosen] >= raw[unpriced_best] - 1e-6)
    assert cert["priced_optimum_proven"]
    s = result["summary"]
    assert s["initial_state"]["source"] == "working_state"
    assert s["moves"]["n_mirrors"] == 2
    if (prev_choice, mu) == ((2, 2), 0.005):
        assert chosen != unpriced_best  # the case where the price changes the plan


def test_without_initial_state_nothing_is_priced(pair, tmp_path):
    pair.aps.initial_aim = "none"
    result = task_aim(pair, tmp_path, TIME, dni=650, method="milp")
    assert "initial_state" not in result["summary"] and "move_cost_mw" not in result["certificate"]


def test_move_limits_are_hard_constraints(small_case, tmp_path):
    case = small_case
    fld = build_field(case)
    _, el, az, dni = sun_state(case, "2025-06-21 11:00")
    prob = build_problem(case, fld, el, az, dni)
    start = np.full(fld.n, prob.center_k)
    prev_csv = _working_state(case, fld, prob, start, tmp_path / "working.csv")
    case.aps.move_penalty_per_m = 0.0  # free moves: only the limit holds the mirrors
    free = task_aim(case, tmp_path, "2025-06-21 11:00", method="greedy", prev_aims=str(prev_csv), certify=False)
    assert free["summary"]["moves"]["max_shift_m"] > 1.0
    case.aps.max_aim_shift_m = 1.0
    res = task_aim(case, tmp_path, "2025-06-21 11:00", method="greedy", prev_aims=str(prev_csv), certify=False)
    m = res["summary"]["moves"]
    assert m["within_limits"] and m["n_outside_limits"] == 0 and m["max_shift_m"] <= 1.0 + 1e-6
    assert res["summary"]["status"] in ("ok", "margin_reduced")
    case.aps.max_aim_shift_m, case.aps.max_rotation_deg = 0.0, 0.1
    res = task_aim(case, tmp_path, "2025-06-21 11:00", method="greedy", prev_aims=str(prev_csv), certify=False)
    m = res["summary"]["moves"]
    assert m["within_limits"] and m["max_rotation_deg"] <= 0.1 + 1e-6
    aims = pd.read_csv(res["files"]["aims_csv"])
    assert {"initial_aim_z", "aim_shift_m", "rotation_deg"} <= set(aims.columns)
    assert aims["rotation_deg"].max() <= 0.1 + 1e-6


def test_move_stats_against_geometry(small_case):
    """Shift = distance of the aim points. Rotation = angle between the bisector normals (s + t) / |s + t| of
    the two aims, computed here from the sun vector; it lies between half the reflected-beam turn delta and
    delta / (2 cos theta) (theta = incidence angle), the first-order bounds of the bisector's turn."""
    from cspaps import sun as sunm

    case = small_case
    fld = build_field(case)
    _, el, az, _ = sun_state(case, "2025-06-21 11:00")
    rec = case.receiver
    prev = (np.full(fld.n, 0.0), np.full(fld.n, rec.center[2]))
    new = (np.full(fld.n, 0.0), np.full(fld.n, rec.center[2] + 1.0))
    per, s = move_stats(case, fld.xyz, el, az, prev, None, new, np.ones(fld.n, bool))
    np.testing.assert_allclose(per["shift_m"], 1.0, atol=1e-9)
    sv = sunm.sun_vector(el, az)
    a = surface_point(rec, *prev) - fld.xyz
    b = surface_point(rec, *new) - fld.xyz
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    na = (sv + a) / np.linalg.norm(sv + a, axis=1, keepdims=True)
    nb = (sv + b) / np.linalg.norm(sv + b, axis=1, keepdims=True)
    expected = np.degrees(np.arccos(np.clip(np.einsum("nc,nc->n", na, nb), -1, 1)))
    np.testing.assert_allclose(per["rotation_deg"], expected, rtol=1e-6, atol=1e-9)
    delta = np.degrees(np.arccos(np.clip(np.einsum("nc,nc->n", a, b), -1, 1)))
    cos_theta = np.einsum("nc,nc->n", na, a)
    assert np.all(per["rotation_deg"] >= delta / 2 * (1 - 1e-3))
    assert np.all(per["rotation_deg"] <= delta / (2 * cos_theta) * (1 + 1e-3))
    assert s["n_moved"] == fld.n and s["max_shift_m"] == pytest.approx(1.0)


def test_trade_off_starts_from_the_initial_state(small_case, tmp_path):
    case = small_case
    case.receiver.height = 16.0
    out = task_pareto(case, tmp_path, "2025-06-21 11:00", lambdas=[0.0, 0.3], certify=False)
    s = out["summary"]
    assert s["initial_state"]["source"] == "spread"
    assert "initial_incident_kw_m2" in out["flux"]
    assert all(r["mean_shift_m"] is not None for r in s["curve"])
    free = case.copy()
    free.aps.move_penalty_per_m = 0.0
    s0 = task_pareto(free, tmp_path, "2025-06-21 11:00", lambdas=[0.0], certify=False)["summary"]
    # the price keeps lambda = 0 closer to the initial plan than free moves do
    assert s["curve"][0]["mean_shift_m"] <= s0["curve"][0]["mean_shift_m"] + 1e-9


def test_working_state_round_trip(pair, tmp_path):
    fld = build_field(pair)
    _, el, az, _ = sun_state(pair, TIME)
    prob = build_problem(pair, fld, el, az, 650)
    path = _working_state(pair, fld, prob, (2, -1), tmp_path / "w.csv")
    (a, z), active = read_aims_csv(path, fld)
    np.testing.assert_allclose(z, prob.cand_z[np.arange(2), [2, prob.center_k]])
    assert active.tolist() == [True, False]


def test_ties_prefer_the_smaller_move(small_case):
    """With non-binding limits every candidate is feasible: each mirror ends on a candidate within the tie
    tolerance of its best priced score, and no other such candidate is a (quantised) smaller move."""
    from cspaps.aps import TIE_MOVE_STEP_M, solve_greedy

    case = small_case
    case.receiver.height = 16.0
    fld = build_field(case)
    _, el, az, dni = sun_state(case, "2025-06-21 11:00")
    prob = build_problem(case, fld, el, az, dni)
    prob.rs.limit_nom = np.full_like(prob.rs.limit_nom, 1e12)
    start = solve_spread(prob)
    prev_xyz = prob.aim_xyz[np.arange(prob.n), start]
    choice, _ = solve_greedy(prob, case.aps, prev_xyz, None)
    p = prob.absorbed_power
    p_ref = p[:, prob.center_k]
    move = np.linalg.norm(prob.aim_xyz - prev_xyz[:, None], axis=-1)
    score = p - case.aps.move_penalty_per_m * p_ref[:, None] * move
    ties = score >= score.max(axis=1, keepdims=True) - case.aps.tie_tolerance * p_ref[:, None]
    rank = np.where(ties, np.round(move / TIE_MOVE_STEP_M), np.inf)
    rows = np.arange(prob.n)
    assert ties[rows, choice].all()
    assert np.array_equal(rank[rows, choice], rank.min(axis=1))
    # the initial plan's own candidate is kept unless a move gains more than the tolerance
    stays = choice == start
    assert stays.mean() > 0.3


def test_mirror_detail_power_chain_and_geometry(small_case):
    from cspaps import sun as sunm
    from cspaps.simulate import LOSSES, STAGES, mirror_detail

    case = small_case
    fld = build_field(case)
    _, el, az, dni = sun_state(case, "2025-06-21 11:00")
    snap = evaluate(case, fld, el, az, dni)
    total = snap.power["intercept"].sum()
    shares = 0.0
    for i in range(fld.n):
        d = mirror_detail(case, snap, fld.xyz, i)
        shares += d["share"]
        if i % 37:
            continue
        p = d["power_w"]
        # each stage = previous stage x its efficiency; optical efficiency = intercepted / sun power
        for a, b in zip(STAGES[:-1], STAGES[1:]):
            assert p[b] == pytest.approx(p[a] * d["eta"][b], rel=1e-12)
        assert d["eta_total"] == pytest.approx(p["intercept"] / p["sun"], rel=1e-12)
        assert d["eta_total"] == pytest.approx(snap.eta_total[i], rel=1e-12)
        # the normal bisects the sun and target directions; incidence = angle(normal, sun)
        s = sunm.sun_vector(el, az)
        n = (s + d["target_dir"]) / np.linalg.norm(s + d["target_dir"])
        np.testing.assert_allclose(d["normal"], n, atol=1e-12)
        assert np.cos(np.radians(d["incidence_deg"])) == pytest.approx(d["eta"]["cosine"], rel=1e-9)
        assert d["slant_m"] == pytest.approx(np.linalg.norm(d["aim_xyz"] - fld.xyz[i]), rel=1e-12)
        assert set(d["field_mean"]) == set(LOSSES)
    assert shares == pytest.approx(1.0, rel=1e-9)  # the shares add up to the receiver power
    assert total > 0


@pytest.mark.parametrize("time", ["2025-06-21 11:00", "2025-03-20 16:30", "2025-12-21 10:30"])
def test_inspector_occlusion_matches_the_solver(small_case, time):
    """The inspector's per-point shading / blocking (neighbours posed on their own aims) is the evaluation's
    shading and blocking at the same sampling, mirror by mirror."""
    from cspaps.simulate import occlusion_detail

    case = small_case
    fld = build_field(case)
    _, el, az, dni = sun_state(case, time)
    snap = evaluate(case, fld, el, az, dni, shading_points=8)
    lost = 1.0 - snap.eta["shading"] * snap.eta["blocking"]
    some = 0
    for i in range(0, fld.n, 5):
        o = occlusion_detail(case, snap, fld.xyz, i, n_points=8)
        assert o["lost_shading"] + o["lost_blocking"] == pytest.approx(lost[i], abs=1e-12)
        assert o["lost_shading"] == pytest.approx(1.0 - snap.eta["shading"][i], abs=1e-12)
        some += bool(len(o["shaders"]) or len(o["blockers"]))
    assert some > 0  # the instants include occluded mirrors


def test_cli_takes_the_initial_state_and_move_limits(tmp_path):
    import glob
    import json

    import yaml

    from cspaps.cli import main

    from .conftest import small_case_dict

    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(small_case_dict(), allow_unicode=True), encoding="utf-8")
    assert main(["aps", str(path), "--time", "2025-06-21 11:00", "--initial-aim", "center", "--max-shift", "1.5",
                 "--move-penalty", "0.002", "--no-certify", "--out", str(tmp_path / "out")]) == 0
    summary = [f for f in glob.glob(str(tmp_path / "out" / "**" / "*.json"), recursive=True) if f.endswith("_greedy.json")]
    s = json.loads(open(summary[0], encoding="utf-8").read())
    assert s["initial_state"]["source"] == "center"
    assert s["moves"]["max_aim_shift_m"] == 1.5 and s["moves"]["move_penalty_per_m"] == 0.002
    assert s["moves"]["within_limits"] and s["moves"]["max_shift_m"] <= 1.5 + 1e-6
    with pytest.raises(ValueError, match="must be >= 0"):
        main(["aps", str(path), "--time", "2025-06-21 11:00", "--max-shift", "-1", "--out", str(tmp_path / "out")])
