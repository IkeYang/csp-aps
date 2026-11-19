import math

import numpy as np
import pytest

from cspaps.config import Heliostat, Receiver
from cspaps.shading import (
    PairCache, grid_offsets, mirror_frames, ray_hits_rect, ray_hits_tower, shading_blocking,
)

HEL = Heliostat(width=6.0, height=6.0, pivot_height=4.0)


def test_mirror_frames_orthonormal():
    n = np.array([[0.0, -1.0, 1.0], [0.3, 0.2, 0.9]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    e_w, e_h = mirror_frames(n)
    for a, b in [(e_w, e_h), (e_w, n), (e_h, n)]:
        np.testing.assert_allclose(np.sum(a * b, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(e_w[:, 2], 0, atol=1e-12)  # width axis stays horizontal
    assert np.all(e_h[:, 2] > 0)


def test_ray_rect_and_tower():
    o = np.array([0.0, 0.0, 0.0])
    hit = ray_hits_rect(o, np.array([0, 1.0, 0]), np.array([0, 10.0, 0]), np.array([0, 1.0, 0]),
                        np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 1.0, 1.0)
    miss = ray_hits_rect(o, np.array([0, 1.0, 0]), np.array([3.0, 10.0, 0]), np.array([0, 1.0, 0]),
                         np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 1.0, 1.0)
    behind = ray_hits_rect(o, np.array([0, -1.0, 0]), np.array([0, 10.0, 0]), np.array([0, 1.0, 0]),
                           np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 1.0, 1.0)
    assert hit and not miss and not behind
    assert ray_hits_tower(np.array([0, -50.0, 5.0]), np.array([0, 1.0, 0.1]), (0, 0), 4.0, 100.0)
    assert not ray_hits_tower(np.array([0, -50.0, 5.0]), np.array([0, 1.0, 3.0]), (0, 0), 4.0, 100.0)


def _two_mirrors(dy, sun_el_deg, n_points=20):
    """Two identical mirrors facing the same way, B directly 'sunward' of A."""
    el = math.radians(sun_el_deg)
    s = np.array([0.0, -math.cos(el), math.sin(el)])  # sun in the south
    normal = s.copy()  # both mirrors face the sun exactly
    xyz = np.array([[0.0, 0.0, 4.0], [0.0, -dy, 4.0]])  # mirror 1 is south of mirror 0
    normals = np.array([normal, normal])
    t = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # reflected rays straight up: no blocking
    eta = shading_blocking(xyz, normals, s, t, HEL, None, PairCache(xyz), n_points=n_points, tower_shadow=False)
    return eta, s, normal


@pytest.mark.parametrize("dy,el", [(8.0, 30.0), (10.0, 20.0), (7.0, 40.0)])
def test_parallel_mirror_shading_matches_geometry(dy, el):
    eta, s, n = _two_mirrors(dy, el)
    # Shadow of mirror 1 on mirror 0's plane is mirror 1 shifted by the projected offset.
    rel = np.array([0.0, -dy, 0.0])
    e_w, e_h = mirror_frames(n[None, :])
    off = rel - np.dot(rel, n) / np.dot(s, n) * s  # project along the sun direction
    a, b = abs(np.dot(off, e_w[0])), abs(np.dot(off, e_h[0]))
    overlap = max(0.0, HEL.width - a) * max(0.0, HEL.height - b) / HEL.area
    assert eta["shading"][0] == pytest.approx(1 - overlap, abs=0.06)
    assert eta["shading"][1] == 1.0  # the sunward mirror is never shaded
    assert np.all(eta["blocking"] == 1.0)


def test_isolated_mirrors_and_high_sun():
    eta, _, _ = _two_mirrors(60.0, 30.0)
    assert np.all(eta["combined"] == 1.0)
    eta, _, _ = _two_mirrors(8.0, 89.0)
    assert np.all(eta["shading"] == 1.0)


def test_blocking_of_reflected_rays():
    # Both mirrors vertical, facing north; tower far to the north; mirror 1 directly north of 0.
    xyz = np.array([[0.0, 0.0, 4.0], [0.0, 7.0, 4.0]])
    n = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    t = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    s = np.array([0.0, 0.0, 1.0])
    eta = shading_blocking(xyz, n, s, t, HEL, None, PairCache(xyz), n_points=10, tower_shadow=False)
    assert eta["blocking"][0] == pytest.approx(0.0)  # fully blocked
    assert eta["blocking"][1] == 1.0


def test_tower_shadow():
    rec = Receiver(center=(0.0, 0.0, 60.0), diameter=8.0, height=6.0, tower_diameter=8.0)
    el = math.radians(20.0)
    s = np.array([0.0, -math.cos(el), math.sin(el)])  # sun in the south, low
    # straight behind the tower / partly behind it / far to the east
    xyz = np.array([[0.0, 60.0, 4.0], [5.0, 60.0, 4.0], [200.0, 60.0, 4.0]])
    n = np.array([s, s, s])
    t = np.array([[0.0, 0.0, 1.0]] * 3)
    eta = shading_blocking(xyz, n, s, t, HEL, rec, PairCache(xyz), n_points=10, tower_shadow=True)
    assert eta["shading"][0] == 0.0  # 8 m tower shadow covers the whole 6 m mirror
    # shadow band |x| < 4 covers x in [2, 4] of the mirror spanning [2, 8]: 3 of 10 sample columns
    assert eta["shading"][1] == pytest.approx(0.7)
    assert eta["shading"][2] == 1.0


def test_grid_offsets():
    off = grid_offsets(4)
    assert off.shape == (16, 2)
    assert off.min() == pytest.approx(-0.375) and off.max() == pytest.approx(0.375)
