import numpy as np
import pandas as pd
import pytest

from cspaps import sun
from cspaps.config import Site, Sun


def test_sun_vector_directions():
    np.testing.assert_allclose(sun.sun_vector(90, 0), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(sun.sun_vector(0, 90), [1, 0, 0], atol=1e-12)  # east
    np.testing.assert_allclose(sun.sun_vector(0, 180), [0, -1, 0], atol=1e-12)  # south


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
    # Higher site -> more DNI
    assert sun.dni_meinel(30, 2000) > sun.dni_meinel(30, 0)
    assert sun.dni(30, Sun(dni_model="constant", dni_constant_w_m2=850), site) == 850
