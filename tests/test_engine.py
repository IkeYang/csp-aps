import io
import json

import numpy as np
import pandas as pd
import pytest

from cspaps.engine import APSEngine, ControlRequest, ForecastSnapshot, Receipt
from cspaps.field import build_field
from cspaps.realtime import run_realtime
from cspaps.simulate import sun_state
from cspaps.workflows import engine_service

T0 = pd.Timestamp("2025-06-21 13:00", tz="Asia/Shanghai")


@pytest.fixture
def setup(small_case):
    small_case.receiver.max_flux_kw_m2 = 350
    fld = build_field(small_case)
    _, _, _, dni = sun_state(small_case, T0)
    return small_case, fld, dni


def _forecasts(t, n, dni, horizon=120, step=15):
    return [ForecastSnapshot(t + pd.Timedelta(seconds=s), np.full(n, dni)) for s in range(0, horizon + 1, step)]


def test_engine_cycle_warm_start_and_cache(setup):
    case, fld, dni = setup
    eng = APSEngine(case, fld)
    r1 = eng.step(ControlRequest(T0, np.full(fld.n, dni), T0, _forecasts(T0, fld.n, dni)))
    assert r1.state == "ready" and not r1.response_reused
    assert r1.check.ok and r1.ids.tolist() == fld.ids.tolist()
    assert r1.valid_until - r1.valid_from == pd.Timedelta(seconds=case.engine.period_s)
    t2 = T0 + pd.Timedelta(seconds=30)
    d2 = np.full(fld.n, dni)
    d2[fld.xyz[:, 0] < 0] *= 0.4
    rec = Receipt(r1.plan_id, r1.aim_azimuth, r1.aim_z, r1.active)
    r2 = eng.step(ControlRequest(t2, d2, t2, _forecasts(t2, fld.n, dni), receipt=rec))
    assert r2.response_reused  # sun moved < refresh angle: cached responses, hot path
    assert r2.state == "ready" and r2.check.ok
    assert r2.timings["total_s"] < r1.timings["total_s"]
    assert "forecast_max" in r2.dni_basis


def test_engine_stale_missing_and_coverage(setup):
    case, fld, dni = setup
    eng = APSEngine(case, fld)
    old = T0 - pd.Timedelta(seconds=case.engine.max_dni_age_s + 30)
    r = eng.step(ControlRequest(T0, np.full(fld.n, dni), old))
    assert r.state == "fallback" and "dni_stale" in r.reasons and r.dni_basis.startswith("clear_sky")
    r = eng.step(ControlRequest(T0, None))
    assert r.state == "fallback" and "dni_missing" in r.reasons
    short = [ForecastSnapshot(T0 + pd.Timedelta(seconds=10), np.full(fld.n, dni))]  # does not cover the window
    r = eng.step(ControlRequest(T0, np.full(fld.n, dni), T0, short))
    assert "forecast_does_not_cover_window" in r.reasons and r.state == "fallback"


def test_engine_availability_and_reachability(setup):
    case, fld, dni = setup
    avail = np.ones(fld.n, bool)
    avail[:5] = False
    eng = APSEngine(case, fld)
    r = eng.step(ControlRequest(T0, np.full(fld.n, dni), T0, _forecasts(T0, fld.n, dni), available=avail))
    assert not r.active[:5].any()
    assert all(x == "unavailable" for x in r.defocus_reason[:5])
    case.engine.max_slew_deg_s = 1e-6  # practically frozen heliostats: aims cannot change
    eng2 = APSEngine(case, fld)
    first = eng2.step(ControlRequest(T0, np.full(fld.n, dni), T0, _forecasts(T0, fld.n, dni)))
    t2 = T0 + pd.Timedelta(seconds=60)
    rec = Receipt(first.plan_id, first.aim_azimuth, first.aim_z, first.active)
    second = eng2.step(ControlRequest(t2, np.full(fld.n, 0.5 * dni), t2, _forecasts(t2, fld.n, 0.5 * dni), receipt=rec))
    moved = (np.abs(second.aim_z - first.aim_z) > 1e-6) & second.active & first.active
    assert not moved.any()


def test_engine_json_service(setup):
    case, fld, dni = setup
    lines = [
        json.dumps({"time": "2025-06-21 13:00", "dni": {str(i): dni for i in fld.ids.tolist()}, "dni_time": "2025-06-21 13:00",
                    "forecasts": [{"valid_time": f"2025-06-21 13:0{m}", "dni": [dni] * fld.n} for m in range(0, 4)]}),
        "not json",
    ]
    out = []
    n = engine_service(case, lines, out.append)
    assert n == 2
    first = json.loads(out[0])
    assert first["state"] == "ready" and len(first["heliostats"]) == fld.n
    assert first["heliostats"][0]["id"] == int(fld.ids[0])
    assert json.loads(out[1])["state"] == "error"


@pytest.mark.parametrize("scenario", ["moving_edge", "sudden_clear"])
def test_replay_scenarios(small_case, scenario):
    small_case.receiver.max_flux_kw_m2 = 350
    small_case.realtime.scenario = scenario
    small_case.realtime.duration_min = 4
    fld = build_field(small_case)
    res = run_realtime(small_case, fld)
    s = res.summary()
    assert s["aps_predictive"]["time_over_limit_s"] == 0
    assert s["aps_predictive"]["engine_latency_p95_s"] > 0
    assert len(res.frames) == len(res.table) // 3


def test_replay_dni_dropout_triggers_fallback(small_case):
    small_case.receiver.max_flux_kw_m2 = 350
    small_case.realtime.duration_min = 4
    small_case.realtime.dni_dropout_s = [60.0, 180.0]
    fld = build_field(small_case)
    res = run_realtime(small_case, fld, strategies=["aps_predictive"])
    states = res.table["plan_state"].tolist()
    reasons = ";".join(res.table["plan_reasons"].astype(str))
    assert "fallback" in states and "dni_missing" in reasons
    assert res.summary()["aps_predictive"]["time_over_limit_s"] == 0
