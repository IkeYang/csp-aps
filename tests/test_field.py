import numpy as np
import pytest

from cspaps.config import Heliostat
from cspaps.field import (
    Field, build_field, load_layout_csv, points_in_polygon, radial_staggered, save_layout_csv,
)


def _min_distance(xy):
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min()


@pytest.mark.parametrize("growth", [0.0, 0.5])
def test_radial_staggered_no_collisions(growth):
    h = Heliostat(width=6, height=6, pivot_height=4)
    xy = radial_staggered(h, (0, 0), 60, r_min=40, r_max=200, spacing_factor=1.1,
                          radial_growth=growth, azimuthal_spacing=1.0)
    dm = h.diagonal * 1.1
    assert len(xy) > 100
    assert _min_distance(xy) >= dm * (1 - 1e-6)
    r = np.hypot(xy[:, 0], xy[:, 1])
    assert r.min() >= 40 - 1e-9 and r.max() <= 200 + 1e-9


def test_radial_staggered_rejects_overlap():
    h = Heliostat(width=6, height=6, pivot_height=4)
    with pytest.raises(ValueError):
        radial_staggered(h, (0, 0), 60, r_min=40, r_max=100, radial_spacing=0.5)


def test_points_in_polygon():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    mask = points_in_polygon(np.array([[5, 5], [15, 5], [-1, -1]]), poly)
    assert mask.tolist() == [True, False, False]


def test_build_field_applies_boundary(small_case):
    full = build_field(small_case)
    small_case.field.boundary = [[-200, 0], [200, 0], [200, 200], [-200, 200]]
    north = build_field(small_case)
    assert 0 < north.n < full.n
    assert np.all(north.xyz[:, 1] >= 0)
    assert np.all(north.xyz[:, 2] == small_case.heliostat.pivot_height)


def test_layout_csv_roundtrip(tmp_path):
    fld = Field(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    path = tmp_path / "layout.csv"
    save_layout_csv(fld, path)
    np.testing.assert_allclose(load_layout_csv(path, pivot_height=0.0), fld.xyz)
    path.write_text("x,y\n10,20\n")
    np.testing.assert_allclose(load_layout_csv(path, pivot_height=5.0), [[10, 20, 5]])
