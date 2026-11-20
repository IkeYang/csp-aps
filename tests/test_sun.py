import numpy as np
import pandas as pd
import pytest

from cspaps import sun
from cspaps.config import Site, Sun


def test_sun_vector_directions():
    np.testing.assert_allclose(sun.sun_vector(90, 0), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(sun.sun_vector(0, 90), [1, 0, 0], atol=1e-12)  # east
    np.testing.assert_allclose(sun.sun_vector(0, 180), [0, -1, 0], atol=1e-12)  # south
    el, az = sun.vector_to_angles(sun.sun_vector(37.0, 211.0))
    assert el == pytest.approx(37.0) and az == pytest.approx(211.0)


def test_solstice_noon_elevation():
    # Max elevation at 40 N on the June solstice ~ 90 - 40 + 23.44 = 73.44 deg
    site = Site(latitude=40.0, longitude=94.7, elevation_m=0, timezone="Asia/Shanghai")
    times = pd.date_range("2025-06-21 10:00", "2025-06-21 17:00", freq="1min", tz=site.timezone)
    el, az = sun.solar_position(times, site)
    i = int(np.argmax(el))
    assert el[i] == pytest.approx(73.44, abs=0.1)
    assert az[i] == pytest.approx(180.0, abs=3.0)


def test_naive_times_are_local():
    idx = sun.to_times("2025-06-21 12:00", "Asia/Shanghai")
    assert idx[0].utcoffset() == pd.Timedelta(hours=8)


def test_dni_models():
    site = Site(latitude=40, longitude=0, elevation_m=0)
    assert sun.dni(-5, Sun(), site) == 0.0
    high = sun.dni(80, Sun(), site)
    low = sun.dni(10, Sun(), site)
    assert 800 < high < 1100
    assert 0 < low < high
    assert sun.dni_meinel(30, 2000) > sun.dni_meinel(30, 0)  # higher site -> more DNI
    assert sun.dni(30, Sun(dni_model="constant", dni_constant_w_m2=850), site) == 850


def test_hourly_and_representative_days(small_case):
    hourly = sun.hourly_samples(small_case)  # 2 h steps
    assert np.all(hourly.elevation >= small_case.annual.min_sun_elevation_deg)
    assert hourly.weight_h.sum() == pytest.approx(len(hourly) * 2.0)
    assert 2500 < hourly.dni_hours < 4500  # kWh/m2/yr, clear sky, 1200 m
    rd = sun.representative_day_samples(small_case)
    assert rd.dni_hours == pytest.approx(hourly.dni_hours, rel=0.05)


def test_sun_bins_preserve_dni_energy(small_case):
    hourly = sun.hourly_samples(small_case)
    bins = sun.sun_bins(hourly, 12)
    assert len(bins) == 12
    assert bins.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-9)
    assert np.all((bins.elevation > 0) & (bins.elevation < 90))
    assert np.all(bins.dni > 0)


def test_weather_csv(tmp_path, small_case):
    times = pd.date_range("2025-01-01 00:30", periods=24 * 365, freq="h")
    el, _ = sun.solar_position(times.tz_localize("Asia/Shanghai"), small_case.site)
    dni = np.where(el > 0, 700.0, 0.0)
    path = tmp_path / "w.csv"
    pd.DataFrame({"time": times.strftime("%Y-%m-%d %H:%M"), "dni": dni}).to_csv(path, index=False)
    small_case.weather.source = "file"
    small_case.weather.file = str(path)
    s = sun.hourly_samples(small_case)
    assert np.all(s.dni == 700.0)
    assert s.weight_h[0] == pytest.approx(1.0)


def test_sun_grid_and_dni_bands_preserve_energy(small_case):
    hourly = sun.hourly_samples(small_case)
    grid = sun.sun_grid(hourly, 15.0, 30.0)
    banded = sun.sun_bins(hourly, 10, dni_bands=3)
    for s in (grid, banded):
        assert s.dni_hours == pytest.approx(hourly.dni_hours, rel=1e-9)
        assert s.members.sum() == len(hourly)
    assert len(banded) > 10  # bins split into DNI bands


def test_dni_bands_avoid_merging_strong_and_weak_states(small_case):
    """Advisor: two equal-length states at one sun position, 300 and 1200 W/m2 -> keep them apart."""
    s = sun.Samples(np.array([45.0, 45.0]), np.array([180.0, 180.0]), np.array([300.0, 1200.0]), np.array([1.0, 1.0]))
    merged = sun.sun_bins(s, 1)
    split = sun.sun_bins(s, 1, dni_bands=2)
    assert len(merged) == 1 and len(split) == 2
    assert sorted(split.dni.tolist()) == [300.0, 1200.0]
