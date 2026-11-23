"""The clear-sky design baseline (README 0.9.0): the layout score is the clear-sky annual value.

  * N representative days cut the year into equal date segments (weight = days in the segment) and cover the
    solar declination of every season;
  * every working hour of a representative day is a scenario, low winter mornings included;
  * one clear-sky DNI per hour, shared by the whole field (site, date, time, typical clear-sky atmosphere);
  * every layout is solved hour by hour on exactly the same dates, hours and weights; spatial maps, clouds and
    measured weather do not enter the score, and the results are named accordingly.
"""

import numpy as np
import pandas as pd
import pvlib
import pytest

from cspaps import case_from_dict, sun
from cspaps.config import _validate
from cspaps.layout_value import evaluate_layout_value, value_labels
from cspaps.scenarios import build_scenarios, design_hourly, low_fidelity_set, validation_scenarios

from .conftest import ROOT, small_case_dict


def _case(**scen):
    d = small_case_dict()
    d["scenarios"] = {"source": "clear_sky_days", "n_days": 4, **scen}
    d["uniformity"] = {"enabled": True, "mode": "weighted", "weight": 0.3}
    d["layout_opt"]["uniformity_weight"] = 0.3
    return case_from_dict(d, base_dir=ROOT)


def _ring(n, r, z=4.0):
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.sin(a), r * np.cos(a), np.full(n, z)])


def test_default_design_baseline_is_clear_sky_days():
    d = small_case_dict()
    d.pop("scenarios")
    case = case_from_dict(d, base_dir=ROOT)
    assert case.scenarios.source == "clear_sky_days" and case.scenarios.n_days == 20
    assert case.scenarios.clear_sky_model == "ineichen"


@pytest.mark.parametrize("year,n", [(2025, 20), (2024, 20), (2025, 7)])
def test_representative_days_cover_the_year_and_the_seasons(year, n):
    days = sun.design_days(year, n)
    assert len(days) == n and sum(w for _, w in days) == (366 if year % 4 == 0 else 365)
    assert max(w for _, w in days) - min(w for _, w in days) <= 1  # equal segments
    doy = np.array([d.dayofyear for d, _ in days])
    assert np.all(np.diff(doy) > 0)
    if n == 20:  # the solstices fall inside the extreme segments: declination within 1 deg of +-23.44
        dec = np.degrees(pvlib.solarposition.declination_spencer71(doy))
        assert dec.max() > 22.4 and dec.min() < -22.4


def test_every_working_hour_with_a_shared_clear_sky_dni():
    case = _case(n_days=4)
    case.annual.time_step_min = 60
    sset = build_scenarios(case)
    s = sset.samples
    days = sset.meta["days"]
    a = case.annual
    assert sset.meta["basis"] == "clear_sky_design" and len(days) == 4
    # all working hours of each day are kept (from morning to evening), nothing else
    assert np.all(s.elevation >= a.min_sun_elevation_deg) and np.all(s.dni > 0)
    for k, d in enumerate(days):
        t = pd.date_range(d["date"], periods=24, freq="h", tz=case.site.timezone) + pd.Timedelta(minutes=30)
        el, _ = sun.solar_position(t, case.site)
        assert d["working_steps"] == int(np.sum(el >= a.min_sun_elevation_deg))
        assert np.allclose(s.weight_h[[i.startswith(f"D{k + 1:02d}-") for i in sset.ids]], d["days_represented"])
    assert s.weight_h.sum() == pytest.approx(sum(d["days_represented"] * d["working_steps"] for d in days))
    # one DNI per hour for the whole field, whatever the layout or the map settings
    xy = np.random.default_rng(0).uniform(-120, 120, (50, 2))
    for i in range(len(sset)):
        assert np.all(sset.dni_at(i, xy) == s.dni[i])
    case.scenarios.map_model = "half_swap"  # maps do not enter the clear-sky baseline
    assert build_scenarios(case).fingerprint() == sset.fingerprint()


def test_clear_sky_dni_is_the_ineichen_model_of_site_date_and_time():
    case = _case(n_days=4)
    sset = build_scenarios(case)
    times = pd.DatetimeIndex(sset.samples.times)
    site = case.site
    loc = pvlib.location.Location(site.latitude, site.longitude, site.timezone, site.elevation_m)
    ref = loc.get_clearsky(times, model="ineichen")["dni"].to_numpy()  # independent pvlib call
    np.testing.assert_allclose(sset.samples.dni, ref, rtol=2e-3)
    # the date matters (Earth-Sun distance, turbidity), not only the elevation: Meinel would not see it
    el = np.array([30.0, 30.0])
    t = pd.DatetimeIndex(["2025-01-05 11:00", "2025-07-05 11:00"]).tz_localize(site.timezone)
    ine = sun.clear_sky_dni(t, el, site, "ineichen")
    mei = sun.clear_sky_dni(t, el, site, "meinel")
    assert mei[0] == mei[1] and abs(ine[0] / ine[1] - 1) > 0.02
    # a fixed or monthly Linke turbidity (local measurements) replaces the climatology
    clean, hazy = sun.clear_sky_dni(t, el, site, "ineichen", 2.0), sun.clear_sky_dni(t, el, site, "ineichen", 5.0)
    assert np.all(clean > hazy)
    monthly = sun.clear_sky_dni(t, el, site, "ineichen", [2.0] * 6 + [5.0] * 6)
    np.testing.assert_allclose(monthly, [clean[0], hazy[1]])


def test_working_threshold_and_year_assignment():
    case = _case(n_days=4)
    full = build_scenarios(case)
    case.annual.min_dni_w_m2 = 600.0
    cut = build_scenarios(case)
    assert len(cut) < len(full) and np.all(cut.samples.dni >= 600.0)
    # the clear-sky year behind the days: every working hour belongs to a scenario of its date segment
    a = np.asarray(full.assignment)
    assert np.all(a >= 0) and len(a) == len(full.hourly)
    assert np.all(full.hourly.dni > 0) and np.all(full.hourly.elevation >= case.annual.min_sun_elevation_deg)
    assert full.samples.members.sum() == len(full.hourly)
    yr = design_hourly(case)
    assert yr.label == "clear_sky_hourly" and np.all(yr.dni >= 600.0)


def test_clear_sky_layout_value_is_named_and_hour_weighted():
    case = _case(n_days=3)
    sset = build_scenarios(case)
    v = evaluate_layout_value(case, _ring(24, 66.0), sset, "A", workers=1, keep_flux=False)
    t, tot = v.table(), v.totals()
    assert tot["basis"] == "clear_sky_design"
    assert tot["value_label"] == "晴空年度综合得分" and tot["energy_label"] == "晴空年度吸热量"
    assert value_labels("clear_sky_design")["energy"] == "晴空年度吸热量"
    assert value_labels(None)["value"] == "年度价值"
    # V = sum over the working hours of (days represented x step) x (P - lambda U), each hour its own APS plan
    h = sset.samples.weight_h
    np.testing.assert_allclose(t["hours"], h)
    assert tot["value_mwh"] == pytest.approx(float(t["contribution_mwh"].sum()), rel=1e-12)
    assert len({r.plan.status for r in v.records} - {"feasible", "feasible_shutdown"}) == 0


def test_low_fidelity_and_denser_check_sets():
    case = _case(n_days=4)
    full = build_scenarios(case)
    lf = low_fidelity_set(case, 6)
    assert len(lf) <= 6 and lf.samples.dni_hours == pytest.approx(full.samples.dni_hours, rel=1e-6)
    dense, info = validation_scenarios(case, full, 0, None)
    assert info["source"] == "clear_sky_denser_days" and len(dense.meta["days"]) == 8 and info["new_states"] > 0


def test_clear_sky_options_are_validated():
    for key, bad in (("n_days", 0), ("clear_sky_model", "bird"), ("linke_turbidity", [3.0] * 11),
                     ("linke_turbidity", 20.0), ("source", "sunny")):
        c = _case()
        setattr(c.scenarios, key, bad)
        with pytest.raises(ValueError):
            _validate(c)
    c = _case()
    c.annual.min_dni_w_m2 = -1
    with pytest.raises(ValueError):
        _validate(c)


def test_layout_search_scores_on_the_clear_sky_baseline(small_case):
    from cspaps.layout_opt import optimize_layout

    small_case.scenarios.source, small_case.scenarios.n_days = "clear_sky_days", 2
    small_case.scenarios.map_model = "clouds"  # ignored: no cloud maps in the design score, no cloud cross-check
    lo = small_case.layout_opt
    lo.n_heliostats, lo.refine_top, lo.formal_swap_rounds, lo.formal_position_trials = 60, 1, 0, 0
    res = optimize_layout(small_case)
    f = res.formal
    assert f and f["scenario_set"].meta["basis"] == "clear_sky_design"
    assert all(v.totals()["basis"] == "clear_sky_design" for v in f["values"].values())
    assert "cross_maps" not in f and res.validation["status"] == "ok"
