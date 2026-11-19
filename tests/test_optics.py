import math

import numpy as np
import pytest

from cspaps import optics
from cspaps.config import Atmosphere, Heliostat, Receiver, Sun
from cspaps.receiver import build_grid, default_aim, surface_point


def test_cosine_efficiency():
    helio = np.array([[0.0, 0.0, 0.0]])
    # Sun straight overhead, target due east on the horizon: n bisects -> cos 45 deg
    _, _, d, cos = optics.heliostat_geometry(helio, np.array([[100.0, 0, 0]]), np.array([0, 0, 1.0]))
    assert d[0] == pytest.approx(100.0)
    assert cos[0] == pytest.approx(math.cos(math.radians(45)))
    # Sun direction equals target direction -> normal incidence
    s = np.array([0.0, 0.6, 0.8])
    _, _, _, cos = optics.heliostat_geometry(helio, s[None, :] * 50, s)
    assert cos[0] == pytest.approx(1.0)


def test_attenuation_models():
    assert optics.attenuation(0.0) == pytest.approx(0.99321)
    assert optics.attenuation(500.0) == pytest.approx(0.99321 - 0.0588 + 0.004925, abs=1e-6)
    assert abs(optics.attenuation(999.999) - optics.attenuation(1000.001)) < 0.01
    d = np.linspace(0, 2000, 50)
    clear = optics.attenuation(d, Atmosphere("vittitoe_biggs_clear"))
    hazy = optics.attenuation(d, Atmosphere("vittitoe_biggs_hazy"))
    assert np.all(np.diff(clear) < 0) and np.all(np.diff(hazy) < 0)
    assert np.all(hazy[1:] < clear[1:])
    # hazy branch is continuous at 1 km
    assert optics.attenuation(1000.0, Atmosphere("vittitoe_biggs_hazy")) == pytest.approx(
        optics.attenuation(1000.001, Atmosphere("vittitoe_biggs_hazy")), abs=1e-5)
    bl = optics.attenuation(2000.0, Atmosphere("beer_lambert", extinction_per_km=0.2))
    assert bl == pytest.approx(math.exp(-0.4))


def test_beam_sigma_focused_vs_flat():
    sun = Sun(sigma_mrad=2.51)
    focused = Heliostat(width=8, height=8, pivot_height=5, slope_error_mrad=1.5, tracking_error_mrad=1.0)
    flat = Heliostat(width=8, height=8, pivot_height=5, focusing="flat")
    d = np.array([300.0])
    expected = 300.0 * math.sqrt(2.51e-3**2 + 3e-3**2 + 2e-3**2)  # no astigmatism at normal incidence
    assert optics.beam_sigma(d, np.array([1.0]), focused, sun)[0] == pytest.approx(expected)
    assert optics.beam_sigma(d, np.array([0.9]), flat, sun)[0] > optics.beam_sigma(d, np.array([0.9]), focused, sun)[0]


@pytest.fixture
def receiver():
    return Receiver(center=(0.0, 0.0, 50.0), diameter=8.0, height=10.0)


def _single_beam(receiver, sigma, n_az=720, n_z=400, helio=(0.0, 200.0, 50.0)):
    grid = build_grid(receiver, n_azimuth=n_az, n_height=n_z)
    hxyz = np.array([helio])
    az, z = default_aim(receiver, hxyz)
    aim = surface_point(receiver, az, z)
    t = (aim - hxyz) / np.linalg.norm(aim - hxyz, axis=1, keepdims=True)
    flux, intercepted = optics.flux_on_receiver(grid, aim, t, np.array([sigma]), np.array([1e6]))
    return grid, flux, intercepted, aim


def test_default_aim_faces_heliostat(receiver):
    az, z = default_aim(receiver, np.array([[0.0, 200.0, 5.0], [150.0, 0.0, 5.0], [0.0, -90.0, 5.0]]))
    np.testing.assert_allclose(az, [0.0, 90.0, 180.0], atol=1e-9)
    np.testing.assert_allclose(z, 50.0)


def _intercept(receiver, aim, helio, sigma):
    t = (aim - helio) / np.linalg.norm(aim - helio, axis=1, keepdims=True)
    return optics.intercept_fraction(receiver, aim, t, np.array([sigma]))[0]


@pytest.mark.parametrize("sigma", [0.3, 0.8, 2.0, 4.0])
def test_intercept_matches_analytic(receiver, sigma):
    """Horizontal ray: silhouette is a D x H rectangle -> product of erfs."""
    _, flux, intercepted, aim = _single_beam(receiver, sigma)
    expected = math.erf(receiver.radius / (math.sqrt(2) * sigma)) * math.erf(
        0.5 * receiver.height / (math.sqrt(2) * sigma)
    )
    helio = np.array([[0.0, 200.0, 50.0]])
    assert _intercept(receiver, aim, helio, sigma) == pytest.approx(expected, rel=1e-6)
    if sigma >= 0.8:  # grid sum is only accurate when sigma >> cell size
        assert intercepted[0] / 1e6 == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("helio,aim_offset", [
    ((120.0, 80.0, 5.0), (0.0, 0.0)),     # low heliostat: tilted ray
    ((-60.0, 30.0, 5.0), (0.0, 3.0)),     # close and steep, aim above centre
    ((10.0, -300.0, 5.0), (25.0, -2.0)),  # aim off to the side and low
])
def test_intercept_matches_fine_grid(receiver, helio, aim_offset):
    """Quadrature intercept vs brute-force integration of a very fine flux grid."""
    helio = np.array([helio])
    az, z = default_aim(receiver, helio)
    aim = surface_point(receiver, az + aim_offset[0], z + aim_offset[1])
    t = (aim - helio) / np.linalg.norm(aim - helio, axis=1, keepdims=True)
    sigma = np.array([1.2])
    grid = build_grid(receiver, n_azimuth=1440, n_height=500)
    _, by_grid = optics.flux_on_receiver(grid, aim, t, sigma, np.array([1.0]))
    by_quad = optics.intercept_fraction(receiver, aim, t, sigma)
    assert 0.3 < by_quad[0] < 1.0
    assert by_quad[0] == pytest.approx(by_grid[0], rel=2e-3)


def test_flux_conservation_and_peak(receiver):
    sigma = 1.5
    grid, flux, intercepted, _ = _single_beam(receiver, sigma)
    assert flux.sum() * grid.cell_area == pytest.approx(intercepted.sum(), rel=1e-12)
    assert flux.max() == pytest.approx(1e6 / (2 * math.pi * sigma**2), rel=5e-3)
    back = grid.to_map(flux)[:, np.abs(grid.azimuth_deg - 180) < 60]
    assert back.max() == 0.0  # the far side of the cylinder gets nothing


def test_flux_peak_is_at_aim_point(receiver):
    grid, flux, _, aim = _single_beam(receiver, 1.0, helio=(150.0, 0.0, 10.0))
    k = int(np.argmax(flux))
    assert np.linalg.norm(grid.points[k] - aim[0]) < 0.3


def test_intercept_accepts_candidate_shapes(receiver):
    helio = np.array([[100.0, 50.0, 5.0], [-80.0, -40.0, 5.0]])
    az, z = default_aim(receiver, helio)
    aims = surface_point(receiver, az[:, None] + np.array([-5.0, 0.0, 5.0]), np.broadcast_to(z[:, None], (2, 3)))
    t = aims - helio[:, None, :]
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    out = optics.intercept_fraction(receiver, aims, t, np.full((2, 3), 1.0))
    assert out.shape == (2, 3)
    assert np.all(out[:, 1] >= out[:, 0] - 1e-9) and np.all(out[:, 1] >= out[:, 2] - 1e-9)
