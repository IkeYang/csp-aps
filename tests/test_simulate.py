import numpy as np
import pytest

from cspaps import sun
from cspaps.field import build_field
from cspaps.simulate import LOSSES, STAGES, annual, evaluate, snapshot


def test_snapshot_power_chain(small_case):
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-03-20 13:30")
    assert snap.sun_elevation > 30
    for prev, k in zip(STAGES[:-1], STAGES[1:]):
        np.testing.assert_allclose(snap.power[k], snap.power[prev] * snap.eta[k])
        assert np.all((snap.eta[k] >= 0) & (snap.eta[k] <= 1))
    # Flux map integrates to the power on the receiver (coarse 36 x 12 grid)
    assert snap.flux.sum() * snap.grid.cell_area == pytest.approx(snap.p_receiver, rel=0.03)
    s = snap.summary()
    assert 0.4 < s["total_optical_efficiency"] < 0.9


def test_low_sun_has_shading(small_case):
    fld = build_field(small_case)
    low = snapshot(small_case, fld, "2025-12-21 09:30")
    high = snapshot(small_case, fld, "2025-06-21 13:30")
    assert low.summary()["field_efficiency"]["shading"] < high.summary()["field_efficiency"]["shading"]
    small_case.shading.enabled = False
    off = snapshot(small_case, fld, "2025-12-21 09:30")
    assert np.all(off.eta["shading"] == 1) and np.all(off.eta["blocking"] == 1)


def test_snapshot_without_flux(small_case):
    fld = build_field(small_case)
    a = snapshot(small_case, fld, "2025-03-20 13:30")
    b = snapshot(small_case, fld, "2025-03-20 13:30", with_flux=False)
    np.testing.assert_allclose(a.power["intercept"], b.power["intercept"])
    assert b.flux.max() == 0


def test_north_field_beats_south_at_noon(small_case):
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-12-21 13:40")
    north = fld.xyz[:, 1] > 30
    south = fld.xyz[:, 1] < -30
    assert snap.eta["cosine"][north].mean() > snap.eta["cosine"][south].mean() + 0.1


def test_night_is_zero(small_case):
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-06-21 02:00")
    assert snap.p_receiver == 0
    assert snap.flux.max() == 0


def test_per_heliostat_dni_and_defocus(small_case):
    fld = build_field(small_case)
    a = snapshot(small_case, fld, "2025-06-21 13:00", dni=500)
    b = snapshot(small_case, fld, "2025-06-21 13:00", dni=1000)
    assert b.p_receiver == pytest.approx(2 * a.p_receiver)
    dni = np.where(fld.xyz[:, 0] > 0, 1000.0, 0.0)  # west half fully under a cloud
    c = snapshot(small_case, fld, "2025-06-21 13:00", dni=dni)
    assert np.all(c.power["intercept"][fld.xyz[:, 0] <= 0] == 0)
    active = np.ones(fld.n, bool)
    active[:10] = False
    d = snapshot(small_case, fld, "2025-06-21 13:00", dni=1000, active=active)
    assert np.all(d.power["intercept"][:10] == 0)
    assert d.summary()["n_defocused"] == 10


def test_annual_consistency(small_case):
    fld = build_field(small_case)
    res = annual(small_case, fld, method="representative_days")
    e = res.energy_mwh
    assert all(e[prev] >= e[k] for prev, k in zip(STAGES[:-1], STAGES[1:]))
    assert res.heliostat_energy_mwh.sum() == pytest.approx(e["intercept"])
    assert 0.4 < res.summary()["total_optical_efficiency"] < 0.9


def test_sun_bins_close_to_hourly(small_case):
    small_case.annual.time_step_min = 180
    small_case.shading.n_points = 4
    fld = build_field(small_case)
    hourly = sun.hourly_samples(small_case)
    e_h = annual(small_case, fld, samples=hourly, with_flux=False).energy_mwh["intercept"]
    e_b = annual(small_case, fld, samples=sun.sun_bins(hourly, 24), with_flux=False).energy_mwh["intercept"]
    assert e_b == pytest.approx(e_h, rel=0.015)


def test_annual_with_aps_respects_limit(small_case):
    small_case.receiver.max_flux_kw_m2 = 300
    fld = build_field(small_case)
    bins = sun.sun_bins(sun.hourly_samples(small_case), 4)
    center = annual(small_case, fld, samples=bins, aiming="center")
    aps = annual(small_case, fld, samples=bins, aiming="greedy")
    assert center.summary()["hours_over_flux_limit"] > 0
    assert aps.timeseries["peak_flux_kw_m2"].max() <= 300 * 1.01
    assert aps.energy_mwh["intercept"] < center.energy_mwh["intercept"]


def test_evaluate_accepts_explicit_sun(small_case):
    fld = build_field(small_case)
    snap = evaluate(small_case, fld, 60.0, 180.0, 900.0)
    assert snap.time is None and snap.p_receiver > 0
    assert set(snap.eta) == set(LOSSES)
