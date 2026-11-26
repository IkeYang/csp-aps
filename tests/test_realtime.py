import numpy as np
import pandas as pd
import pytest

from cspaps.clouds import Cloud, CloudField, load_dni_map
from cspaps.field import build_field
from cspaps.realtime import run_realtime
from cspaps.workflows import load_realtime_run, task_realtime


def test_cloud_transmittance_and_motion():
    t0 = pd.Timestamp("2025-06-21 12:00", tz="Asia/Shanghai")
    cf = CloudField([Cloud(0.0, 0.0, 100.0, transmittance=0.2, vx=10.0, edge=10.0)], t0)
    xy = np.array([[0.0, 0.0], [500.0, 0.0], [300.0, 0.0]])
    f0 = cf.transmittance(xy, t0)
    assert f0[0] == pytest.approx(0.2) and f0[1] == pytest.approx(1.0)
    f30 = cf.transmittance(xy, t0 + pd.Timedelta(seconds=30))  # cloud centre now at x = 300
    assert f30[2] == pytest.approx(0.2) and f30[0] == pytest.approx(1.0)
    # the highest DNI over the next minute is at least the current one
    mx = cf.max_dni(xy, t0, 60, 1000.0)
    assert np.all(mx >= cf.dni(xy, t0, 1000.0))
    assert mx[0] == pytest.approx(1000.0)  # the shadow leaves the origin within a minute


def test_load_dni_map(tmp_path):
    path = tmp_path / "dni.csv"
    pd.DataFrame({"x": [0, 100, 0, 100], "y": [0, 0, 100, 100], "dni": [800, 900, 800, 900]}).to_csv(path, index=False)
    v = load_dni_map(path, np.array([[50.0, 50.0, 4.0], [500.0, 0.0, 4.0]]))
    assert v[0] == pytest.approx(850.0)
    assert v[1] == pytest.approx(900.0)  # nearest value outside the measured hull


def test_realtime_runs_and_reports(small_case):
    small_case.receiver.max_flux_kw_m2 = 350
    fld = build_field(small_case)
    res = run_realtime(small_case, fld)
    rt = small_case.realtime
    n_steps = int(rt.duration_min * 60 / rt.step_s) + 1
    per_interval = int(rt.step_s / rt.substep_s)  # control instant + sub-samples
    assert len(res.table) == ((n_steps - 1) * per_interval + 1) * 3
    assert int(res.table["control"].sum()) == n_steps * 3
    s = res.summary()
    assert set(s) == {"center", "aps_reactive", "aps_predictive"}
    assert s["aps_predictive"]["max_peak_to_limit"] <= 1.0
    assert s["center"]["max_peak_to_limit"] > 1.0
    assert s["aps_predictive"]["energy_mwh"] > 0

    # A failed first sample contributes its own interval in hours, regardless of the sample count.
    res.table["uniformity_feasible"] = res.table["time"] != res.table["time"].min()
    for report in res.summary().values():
        assert report["hours_uniformity_failed"] == pytest.approx(rt.substep_s / 3600)


def test_replay_samples_hold_the_plan_between_control_instants():
    from cspaps.realtime import replay_samples, replay_times

    t0 = pd.Timestamp("2025-06-21 12:00", tz="Asia/Shanghai")
    times, d, k, ctl = replay_samples(t0, 150, 60, 20)
    assert [(t - t0).total_seconds() for t in times] == [0, 20, 40, 60, 80, 100, 120, 140, 150]
    assert d.tolist() == [20, 20, 20, 20, 20, 20, 20, 10, 0] and d.sum() == 150
    assert k.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 3] and ctl.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 1]
    for sub in (0, 60, 90):  # no sub-samples: exactly the control instants
        t2, d2, _, c2 = replay_samples(t0, 150, 60, sub)
        tc, dc = replay_times(t0, 150, 60)
        assert t2 == tc and d2.tolist() == dc.tolist() and c2.all()


def test_sub_samples_keep_the_control_decisions(small_case):
    """Sub-samples only add evaluations: the plans and the rows at the control instants are those of a replay
    without sub-samples; between control instants the plan in effect is held (no moves, no engine call)."""
    small_case.receiver.max_flux_kw_m2 = 350
    small_case.realtime.scenario = "sudden_clear"
    fld = build_field(small_case)
    fine = run_realtime(small_case, fld)
    small_case.realtime.substep_s = 0
    coarse = run_realtime(small_case, fld)
    ctl = fine.table[fine.table["control"]].reset_index(drop=True)
    assert len(ctl) == len(coarse.table)
    for col in ("time", "strategy", "p_absorbed_mw", "peak_to_limit", "n_moved", "n_defocus_commands", "plan_state"):
        assert ctl[col].tolist() == coarse.table[col].tolist(), col
    sub = fine.table[~fine.table["control"]]
    assert (sub["n_moved"] == 0).all() and sub["engine_total_s"].isna().all()
    for f in fine.frames:
        k = f["control_index"]
        c = next(g for g in coarse.frames if g["control_index"] == k)
        for strat, plan in f["plans"].items():
            np.testing.assert_array_equal(plan["aim_z"], c["plans"][strat]["aim_z"])
    assert fine.duration_s == coarse.duration_s
    s_f, s_c = fine.summary(), coarse.summary()
    for strat in s_f:
        assert s_f[strat]["replay_duration_s"] == s_c[strat]["replay_duration_s"]
        assert s_f[strat]["heliostat_moves"] == s_c[strat]["heliostat_moves"]


def test_clear_sky_reference_is_the_same_plan_without_clouds(small_case):
    """p_absorbed_clear_mw = the plan in effect evaluated with clear-sky DNI everywhere (brute force)."""
    from cspaps.simulate import evaluate, sun_state

    small_case.realtime.scenario = "sudden_clear"
    small_case.realtime.substep_s = 30
    fld = build_field(small_case)
    res = run_realtime(small_case, fld, strategies=["center", "aps_reactive"])
    rows = res.table.reset_index(drop=True)
    checked = 0
    for j in range(0, len(res.frames), 2):
        f = res.frames[j]
        _, el, az, dni_clear = sun_state(small_case, f["time"])
        for strat, plan in f["plans"].items():
            snap = evaluate(small_case, fld, el, az, dni_clear, aim=(plan["aim_azimuth"], plan["aim_z"]),
                            active=plan["active"])
            row = rows[(rows["time"] == f["time"]) & (rows["strategy"] == strat)].iloc[0]
            assert row["p_absorbed_clear_mw"] == pytest.approx(snap.p_absorbed / 1e6, rel=1e-6)
            assert row["p_absorbed_mw"] <= row["p_absorbed_clear_mw"] * (1 + 1e-9)
            checked += 1
    assert checked >= 4
    covered = rows[rows["cloud_cover"] > 0]
    assert len(covered) and (covered["p_absorbed_mw"] < covered["p_absorbed_clear_mw"]).all()


def test_saved_realtime_run_restores_full_frame_data(small_case, tmp_path):
    run = task_realtime(small_case, tmp_path)
    restored = load_realtime_run(run["run_dir"])
    original, replayed = run["result"], restored["result"]

    assert len(replayed.frames) == len(original.frames)
    assert replayed.table["strategy"].tolist() == original.table["strategy"].tolist()
    assert replayed.table["time"].astype(str).tolist() == original.table["time"].astype(str).tolist()
    for column in ("p_intercepted_mw", "p_absorbed_mw", "peak_flux_kw_m2", "peak_to_limit", "interval_s"):
        np.testing.assert_allclose(replayed.table[column], original.table[column], equal_nan=True)
    for column in ("plan_state", "plan_reasons"):
        saved = replayed.table[column].astype(object).where(replayed.table[column].notna(), "").astype(str)
        source = original.table[column].astype(object).where(original.table[column].notna(), "").astype(str)
        assert saved.tolist() == source.tolist()
    # sample-and-hold time base of one strategy, not the sum over every row of the table
    assert replayed.duration_s == pytest.approx(original.duration_s)
    assert replayed.summary() == original.summary()
    for source, target in zip(original.frames, replayed.frames):
        np.testing.assert_array_equal(target["dni"], source["dni"])
        assert target["time"] == source["time"]
        for strategy, plan in source["plans"].items():
            restored_plan = target["plans"][strategy]
            np.testing.assert_array_equal(restored_plan["aim_azimuth"], plan["aim_azimuth"])
            np.testing.assert_array_equal(restored_plan["aim_z"], plan["aim_z"])
            np.testing.assert_array_equal(restored_plan["active"], plan["active"])
            np.testing.assert_array_equal(restored_plan["tower_fraction"], plan["tower_fraction"])
            np.testing.assert_array_equal(restored_plan["flux"], plan["flux"])
            assert restored_plan["check"] == plan["check"]  # the verified check, hotspot included
