"""Annual-value acceleration (0.9.0), improvements 1-5 of the annual-computation review:

  1. control variate: the scenario set keeps the hour -> scenario assignment; factors from the cheap model
  2. reuse: spot responses per sun direction (bit-identical plans), plans across DNI levels the limits never
     shaped, shared sun directions across maps (DNI energy kept)
  3. Bayesian optimisation of the generator parameters on the low-fidelity formal V; sector thinning
  4. flux prices (Lagrange multipliers of the APS bound) price free positions and mirrors for the swaps
  5. layouts x scenarios in one pool; optional GPU backend with CPU fallback
The bound behind item 4 is tested against brute force in ``test_certify``.
"""

from dataclasses import replace

import numpy as np
import pytest

from cspaps import case_from_dict, sun
from cspaps.bayes import GP, expected_improvement, optimise, suggest
from cspaps.field import radial_staggered, sector_thinning
from cspaps.layout_value import (
    dual_prices, evaluate_layout_value, evaluate_layout_values, inner_settings, price_swaps,
)
from cspaps.scenarios import build_scenarios, load_folder
from cspaps.simulate import annual
from cspaps.field import Field

from .conftest import ROOT, small_case_dict


def _case(**scen):
    d = small_case_dict()
    d["scenarios"] = {"source": "weather", "n_scenarios": 4, "map_model": "half_swap", **scen}
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["layout_opt"]["uniformity_weight"] = 0.3
    return case_from_dict(d, base_dir=ROOT)


def _ring(n, r, phase=0.0, z=4.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.sin(a), r * np.cos(a), np.full(n, z)])


def _totals(sset):
    """Per scenario: member hours and member DNI energy from the hour assignment."""
    a = np.asarray(sset.assignment)
    h = sset.hourly
    keep = a >= 0
    hours = np.bincount(a[keep], weights=h.weight_h[keep], minlength=len(sset))
    energy = np.bincount(a[keep], weights=(h.dni * h.weight_h)[keep], minlength=len(sset))
    return hours, energy


# ----------------------------------------------------------------------------- 1: assignment and control variate

def test_hour_assignment_partitions_the_year_and_round_trips(tmp_path):
    case = _case(n_scenarios=6)
    sset = build_scenarios(case)
    a = np.asarray(sset.assignment)
    assert a.shape == (len(sset.hourly),) and a.max() < len(sset) and set(np.unique(a[a >= 0])) == set(range(len(sset)))
    hours, energy = _totals(sset)
    s = sset.samples
    # every scenario stands for exactly its member hours, and their DNI energy (time-mean DNI x actual hours)
    np.testing.assert_allclose(hours, s.actual_hours, rtol=1e-6)
    np.testing.assert_allclose(energy, s.dni * s.weight_h, rtol=1e-3)
    # members carry the map of their scenario
    assert np.all(sset.hourly.map_index[a >= 0] == s.map_index[a[a >= 0]])
    sset.save(tmp_path / "s")
    again = load_folder(tmp_path / "s")
    np.testing.assert_array_equal(again.assignment, sset.assignment)
    assert again.hourly_fingerprint() == sset.hourly_fingerprint() is not None
    assert again.fingerprint() == sset.fingerprint()


def test_control_variate_factors_follow_the_cheap_model():
    case = _case(n_scenarios=4)
    case.layout_opt.control_variate = True
    sset = build_scenarios(case)
    xyz = _ring(24, 70.0)
    v = evaluate_layout_value(case, xyz, sset, "A", workers=1, keep_flux=False)
    assert v.tags["control_variate"]["hours"] == int(np.sum(np.asarray(sset.assignment) >= 0))
    a = np.asarray(sset.assignment)
    keep = np.flatnonzero(a >= 0)
    hourly = sset.hourly.subset(keep)
    direct = annual(case, Field(xyz), samples=hourly, aiming="center", with_flux=False, workers=1)
    total = float((direct.timeseries["p_absorbed_mw"].to_numpy(float) * hourly.weight_h).sum())
    members = np.array([r.cheap_members_mwh for r in v.records])
    assert members.sum() == pytest.approx(total, rel=1e-9)  # the members partition the year
    t = v.table()
    f = np.array([r.cv_factor for r in v.records])
    assert np.all((f >= 0.5) & (f <= 2.0))
    np.testing.assert_allclose(f, np.clip(t["cheap_members_mwh"] / t["cheap_rep_mwh"], 0.5, 2.0), rtol=1e-12)
    tot = v.totals()
    assert tot["value_cv_mwh"] == pytest.approx(float((t["contribution_mwh"] * f).sum()), rel=1e-9)
    assert tot["energy_cv_mwh"] == pytest.approx(float((t["energy_mwh"] * f).sum()), rel=1e-9)
    case.layout_opt.control_variate = False
    off = evaluate_layout_value(case, xyz, sset, "A", workers=1, keep_flux=False)
    assert "control_variate" not in off.tags and off.totals()["value_mwh"] == tot["value_mwh"]


# ----------------------------------------------------------------------------- 2: reuse

def test_response_reuse_and_batches_are_bit_identical():
    case = _case(n_scenarios=4, dni_bands=2)
    sset = build_scenarios(case)
    a, b = _ring(24, 66.0), np.vstack([_ring(12, 62.0), _ring(12, 90.0, phase=0.1)])
    st = inner_settings(case)
    off = replace(st, reuse="off")
    ref = {k: evaluate_layout_value(case, x, sset, k, settings=off, workers=1, keep_flux=False) for k, x in (("A", a), ("B", b))}
    batch = evaluate_layout_values(case, {"A": (a, None), "B": (b, None)}, sset, settings=st, workers=2, keep_flux=False)
    for k in ("A", "B"):
        p_ref = [(r.plan.p_abs_w, r.plan.u_w, r.plan.status) for r in ref[k].records]
        p_new = [(r.plan.p_abs_w, r.plan.u_w, r.plan.status) for r in batch[k].records]
        assert p_new == p_ref  # same plans, bit for bit, whatever the reuse / pool layout
        assert batch[k].totals()["value_mwh"] == ref[k].totals()["value_mwh"]


def test_full_reuse_only_rescales_plans_the_limits_never_shaped():
    # DNI bands share their direction (shared_directions), so lower bands can reuse the plan of the highest
    case = _case(n_scenarios=3, dni_bands=3, map_model="uniform", shared_directions=True)
    case.receiver.max_flux_kw_m2 = 5000  # the limits never act
    sset = build_scenarios(case)
    xyz = _ring(20, 70.0)
    full, off = replace(inner_settings(case), reuse="full"), replace(inner_settings(case), reuse="off")
    vf = evaluate_layout_value(case, xyz, sset, "F", settings=full, workers=1, keep_flux=False)
    vo = evaluate_layout_value(case, xyz, sset, "O", settings=off, workers=1, keep_flux=False)
    ids = [r.scenario_id for r in vf.records]
    reused = [i for i, r in enumerate(vf.records) if "reused_from" in r.plan.info]
    assert len(reused) == 6  # 3 directions x 3 bands: the two lower bands of every direction
    for i in reused:
        pf = vf.records[i].plan
        j = ids.index(pf.info["reused_from"])
        f = pf.info["dni_factor"]
        assert not pf.info["limits_active"] and 0 < f < 1
        assert vf.records[i].dni_w_m2 == pytest.approx(f * vf.records[j].dni_w_m2, rel=1e-5)
        # the property behind the reuse: cold solves of the same direction take the same decisions at every
        # DNI level, and P, U are homogeneous of degree 1 in DNI
        np.testing.assert_array_equal(vo.records[i].plan.choice, vo.records[j].plan.choice)
        assert pf.p_abs_w == pytest.approx(f * vf.records[j].plan.p_abs_w, rel=1e-12)
        assert pf.u_w == pytest.approx(f * vf.records[j].plan.u_w, rel=1e-12)
    # with responses-only reuse nothing is rescaled
    assert not any("reused_from" in r.plan.info for r in vo.records)
    # the warm-start chains depend on the scenario set only: the same plans with any number of workers
    v2 = evaluate_layout_value(case, xyz, sset, "F2", settings=full, workers=2, keep_flux=False)
    assert [(r.plan.p_abs_w, r.plan.u_w) for r in v2.records] == [(r.plan.p_abs_w, r.plan.u_w) for r in vf.records]


def test_shared_directions_keep_the_dni_energy_of_every_map():
    case = _case(n_scenarios=8, shared_directions=True)
    sset = build_scenarios(case)
    hourly = sun.hourly_samples(case)
    s = sset.samples
    assert s.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-3)
    dirs = {(round(float(e), 6), round(float(z), 6)) for e, z in zip(s.elevation, s.azimuth)}
    assert len(dirs) < len(sset)  # maps share sun directions -> one response set per direction
    reg = sset.hourly.map_index
    for m in np.unique(s.map_index):  # each map keeps its own DNI energy
        e_map = float((sset.hourly.dni * sset.hourly.weight_h)[reg == m].sum())
        assert float((s.dni * s.weight_h)[s.map_index == m].sum()) == pytest.approx(e_map, rel=1e-3)


# ----------------------------------------------------------------------------- 3: sector thinning and Bayesian search

def test_sector_thinning_only_removes_positions(small_case):
    hel = small_case.heliostat
    base = radial_staggered(hel, (0.0, 0.0), 60.0, 40.0, 110.0)
    same = radial_staggered(hel, (0.0, 0.0), 60.0, 40.0, 110.0, sector_n=1.0, sector_e=1.0, sector_s=1.0, sector_w=1.0)
    np.testing.assert_array_equal(base, same)
    thin = radial_staggered(hel, (0.0, 0.0), 60.0, 40.0, 110.0, sector_n=2.0)
    key = {tuple(np.round(p, 9)) for p in base}
    assert all(tuple(np.round(p, 9)) in key for p in thin)  # a subset: spacing is kept
    phi = np.degrees(np.arctan2(base[:, 0], base[:, 1])) % 360
    phi_t = np.degrees(np.arctan2(thin[:, 0], thin[:, 1])) % 360
    north = lambda p: (p < 30) | (p > 330)  # noqa: E731
    south = lambda p: np.abs(p - 180) < 60  # noqa: E731
    assert north(phi_t).sum() < 0.7 * north(phi).sum()
    assert south(phi_t).sum() == south(phi).sum()  # untouched where only the south factor (1) applies
    assert sector_thinning(np.radians(90.0), (1.6, 1.2, 1.0, 1.0)) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        radial_staggered(hel, (0.0, 0.0), 60.0, 40.0, 110.0, sector_e=0.9)


def test_gaussian_process_and_expected_improvement():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (12, 2))
    f = lambda p: -np.sum((p - 0.3) ** 2, axis=-1)  # noqa: E731
    gp = GP(x, f(x)).fit()
    mu, sd = gp.predict(x)
    np.testing.assert_allclose(mu, f(x), atol=0.05 * np.ptp(f(x)))  # nearly interpolates the data
    assert np.all(sd >= 0) and np.all(expected_improvement(mu, sd, f(x).max()) >= 0)
    pts = suggest(x, f(x), q=3, seed=1)
    assert pts.shape == (3, 2) and np.all((pts >= 0) & (pts <= 1))
    assert np.min(np.linalg.norm(pts[:, None] - x[None], axis=2)) > 1e-6  # never re-proposes a point
    tr = optimise(lambda p: f(np.asarray(p)), 2, n_init=6, n_total=16, q=2, seed=0)
    assert len(tr.y) == 16 and tr.stage.count("initial") == 6
    assert max(tr.y) > -0.01  # finds the optimum region of a smooth 2-D objective


def test_bayesian_layout_search_runs_on_the_formal_value(small_case):
    from cspaps.layout_opt import optimize_layout

    lo = small_case.layout_opt
    lo.search_method, lo.bayes_evaluations, lo.bayes_initial, lo.bayes_scenarios = "bayes", 4, 3, 2
    lo.refine_top, lo.n_heliostats, lo.formal_swap_rounds, lo.formal_position_trials = 1, 60, 0, 0
    res = optimize_layout(small_case)
    h = res.history
    assert set(h["stage"]) >= {"initial", "bayes"} and len(h) == 4
    assert h["objective_basis"].str.contains("formal").all()
    assert res.validation["status"] == "ok" and res.best.n == 60


# ----------------------------------------------------------------------------- 4: flux prices

def test_dual_prices_of_free_positions_and_mirrors():
    case = _case(n_scenarios=3)
    sset = build_scenarios(case)
    xyz = _ring(24, 66.0)
    free = np.vstack([_ring(12, 92.0, phase=0.13), _ring(8, 120.0)])
    v = evaluate_layout_value(case, xyz, sset, "A", workers=1, keep_flux=False)
    p = dual_prices(case, v, free, sset, workers=1)
    assert p["gain_mwh"].shape == (len(free),) and p["loss_mwh"].shape == (len(xyz),)
    assert np.all(p["gain_mwh"] >= 0) and np.all(p["loss_mwh"] >= 0) and p["gap_mwh"] >= 0
    assert p["scenarios"] == sum(r.plan.counts for r in v.records)
    # a mirror's removal loss never exceeds its annual energy in the plan (it only takes its P, maybe less U)
    assert p["loss_mwh"].sum() > 0
    drop, add, pred = price_swaps(p, 3)
    assert len(drop) == len(add) <= 3
    assert np.all(p["gain_mwh"][add] - p["loss_mwh"][drop] > 0)
    assert pred == pytest.approx(float(p["gain_mwh"][add].sum() - p["loss_mwh"][drop].sum()))


def test_swap_pricing_is_validated():
    from cspaps.config import _validate as validate

    case = _case()
    assert case.layout_opt.swap_pricing == "dual"
    for key, bad in (("swap_pricing", "lp"), ("search_method", "cmaes"), ("inner_reuse", "all"), ("rank_by", "cv")):
        c = case.copy()
        setattr(c.layout_opt, key, bad)
        with pytest.raises(ValueError):
            validate(c)


# ----------------------------------------------------------------------------- 5: backend

def test_gpu_backend_switches_and_cpu_fallback(monkeypatch):
    from cspaps import backend, optics

    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-3, 3, 50), rng.uniform(-3, 3, 50), rng.uniform(57, 63, 50)])
    nrm = np.tile([0.0, -1.0, 0.0], (50, 1))
    aim = np.tile([0.0, -3.0, 60.0], (7, 1)) + rng.normal(0, 0.5, (7, 3))
    t = aim - np.array([0.0, -80.0, 4.0])
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    sigma, power = rng.uniform(0.4, 1.0, 7), rng.uniform(1e4, 3e4, 7)
    monkeypatch.delenv("CSPAPS_GPU", raising=False)
    cpu = optics.footprints_at(pts, nrm, aim, t, sigma, power)
    np.testing.assert_allclose(backend.footprints_xp(np, pts, nrm, aim, t, sigma, power), cpu, rtol=1e-6, atol=1e-9)
    assert backend.backend_name() == "numpy" and not backend.enabled()
    monkeypatch.setenv("CSPAPS_GPU", "1")
    monkeypatch.setenv("CSPAPS_GPU_MIN", "0")
    assert backend.enabled()
    if backend._cupy() is None:  # no CuPy / device here: the switch falls back to the CPU path, bit for bit
        assert backend.backend_name() == "numpy"
        np.testing.assert_array_equal(optics.footprints_at(pts, nrm, aim, t, sigma, power), cpu)
    from cspaps.parallel import start_method

    import multiprocessing as mp

    if "forkserver" in mp.get_all_start_methods():  # CUDA cannot be used in a child forked after the parent
        assert start_method() == "forkserver"
    monkeypatch.setenv("CSPAPS_GPU_DEVICES", "0;2")
    assert backend.devices() == [0, 2]
    monkeypatch.setenv("CSPAPS_GPU_MIN", "many")
    assert backend.min_elements() == 2_000_000
