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
    n_steps = int(small_case.realtime.duration_min * 60 / small_case.realtime.step_s) + 1
    assert len(res.table) == n_steps * 3
    s = res.summary()
    assert set(s) == {"center", "aps_reactive", "aps_predictive"}
    assert s["aps_predictive"]["max_peak_to_limit"] <= 1.0
    assert s["center"]["max_peak_to_limit"] > 1.0
    assert s["aps_predictive"]["energy_mwh"] > 0

    # A failed first interval contributes one step in hours, regardless of the sample count.
    res.table["uniformity_feasible"] = res.table["time"] != res.table["time"].min()
    for report in res.summary().values():
        assert report["hours_uniformity_failed"] == pytest.approx(small_case.realtime.step_s / 3600)


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
