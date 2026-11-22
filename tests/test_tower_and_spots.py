"""Tower shadow from the variable tower / receiver / pivot heights, and the beam spot size on the receiver."""

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from cspaps import optics, sun
from cspaps.field import Field
from cspaps.gui_plots import receiver_aims_map, shadow_plane_height, tower_shadow_outline
from cspaps.receiver import build_grid, surface_point
from cspaps.shading import ray_hits_structure, ray_hits_tower, shadow_cylinders
from cspaps.simulate import beam_spots, evaluate


def _plane_points(z, half=60.0, step=0.7):
    g = np.arange(-half, half + 1e-9, step)
    x, y = np.meshgrid(g, g)
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


@pytest.mark.parametrize("el,az", [(25.0, 180.0), (40.0, 75.0), (12.0, 300.0)])
@pytest.mark.parametrize("tower_d,tower_h,z_plane", [(None, None, 4.0), (10.0, 52.0, 4.0), (4.0, 70.0, 9.0)])
def test_shadow_outline_matches_the_ray_test(small_case, el, az, tower_d, tower_h, z_plane):
    """Points inside the drawn shadow are exactly the points whose ray to the sun hits the tower or the
    receiver (brute force), up to the polygon discretisation at the edge."""
    rec = small_case.receiver
    rec.tower_diameter, rec.tower_height = tower_d, tower_h
    pts = _plane_points(z_plane, half=max(60.0, (rec.top - z_plane) / np.tan(np.radians(el)) + 10.0), step=0.9)
    s = sun.sun_vector(el, az)
    hit = ray_hits_structure(pts, s, rec)
    inside = np.zeros(len(pts), bool)
    for poly in tower_shadow_outline(small_case, el, az, z_plane):
        inside |= MplPath(poly).contains_points(pts[:, :2])
    assert hit.any()
    wrong = hit != inside
    # disagreements only within a thin band at the edge (polygon chords vs circles, cell size)
    if wrong.any():
        d_axis = np.abs(np.cross(pts[wrong, :2] - np.asarray(rec.center[:2]), -s[:2] / np.linalg.norm(s[:2])))
        radii = [r for r, *_ in shadow_cylinders(rec)]
        assert np.all(np.min(np.abs(d_axis[:, None] - np.array(radii)[None]), axis=1) < 0.15) or wrong.mean() < 2e-3


def test_tower_height_is_a_variable(small_case):
    """A ray passing 4 m from the axis at 58 m height: the receiver (radius 3 m, bottom at 57 m) misses it,
    the 10 m tower hits it only if the tower reaches above 58 m."""
    rec = small_case.receiver  # centre z 60, height 6 -> receiver from 57 to 63 m, radius 3 m
    rec.tower_diameter = 10.0
    o = np.array([[4.0, -50.0, 58.0]])
    d = np.array([0.0, 1.0, 0.0])  # horizontal ray through x = 4, z = 58
    rec.tower_height = None  # default: tower top = receiver bottom (57 m)
    assert rec.tower_top == pytest.approx(57.0) and not ray_hits_structure(o, d, rec)[0]
    rec.tower_height = 60.0
    assert ray_hits_structure(o, d, rec)[0]
    rec.tower_diameter = 0.0  # 0 = no tower shadow at all
    assert not ray_hits_structure(o, d, rec)[0] and shadow_cylinders(rec) == []


def test_default_tower_reproduces_the_single_cylinder(small_case):
    """tower_diameter / tower_height unset: tower + receiver = one cylinder of receiver radius up to the top."""
    rec = small_case.receiver
    rng = np.random.default_rng(3)
    o = np.column_stack([rng.uniform(-80, 80, 4000), rng.uniform(-80, 80, 4000), rng.uniform(0, 8, 4000)])
    d = rng.normal(size=(4000, 3))
    d[:, 2] = np.abs(d[:, 2])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    old = ray_hits_tower(o, d, rec.center[:2], rec.radius, rec.top)
    assert np.array_equal(ray_hits_structure(o, d, rec), old)


def test_tower_shadow_on_mirrors_follows_tower_and_pivot_heights(small_case):
    """Shadow fraction of a mirror just outside the default shadow appears when the tower is raised, and the
    drawn plane follows the mirrors' own height."""
    c = small_case
    c.receiver.tower_diameter = 8.0
    el, az = 30.0, 180.0  # sun in the south: shadow to the north
    tan = np.tan(np.radians(el))
    z = c.heliostat.pivot_height
    reach = (c.receiver.top - z) / tan  # end of the shadow of the receiver (on top of the default tower)
    fld = Field(np.array([[0.0, reach + 9.0, z]]))  # just beyond it (the tilted 6 m mirror reaches back ~5 m)
    f0 = evaluate(c, fld, el, az, 900.0).tower_fraction[0]
    c.receiver.tower_height = c.receiver.top + 4.0  # a tower 4 m higher than the receiver top
    f1 = evaluate(c, fld, el, az, 900.0).tower_fraction[0]
    assert f0 == 0 and f1 > 0.4
    assert shadow_plane_height(c, fld.xyz) == pytest.approx(z)
    assert shadow_plane_height(c, np.array([[0, 0, 7.5], [1, 1, 7.5], [2, 2, 9.0]])) == pytest.approx(7.5)
    assert shadow_plane_height(c, None) == pytest.approx(c.heliostat.pivot_height)
    # the drawn shadow gets shorter by (z2 - z1) / tan(el) when the mirrors stand higher
    far = lambda zp: max(float(np.max(p @ np.array([0.0, 1.0]))) for p in tower_shadow_outline(c, el, az, zp))  # noqa: E731
    assert far(z) - far(z + 3.0) == pytest.approx(3.0 / tan, rel=1e-6)


def test_surface_spot_matches_the_numerical_footprint(small_case):
    """Second moments of the model's own beam footprint on the tangent plane = sigma^2 (I + a a^T / cos^2 i)."""
    rec = small_case.receiver
    rng = np.random.default_rng(1)
    for _ in range(5):
        phi = rng.uniform(0, 360)
        aim = surface_point(rec, np.array([phi]), np.array([rec.center[2] + rng.uniform(-1, 1)]))[0]
        n_out = np.array([np.sin(np.radians(phi)), np.cos(np.radians(phi)), 0.0])
        e_phi = np.array([np.cos(np.radians(phi)), -np.sin(np.radians(phi)), 0.0])
        # incoming direction with a random incidence up to 60 deg
        t = -n_out + rng.uniform(-0.8, 0.8) * e_phi + rng.uniform(-0.6, 0.6) * np.array([0, 0, 1.0])
        t /= np.linalg.norm(t)
        sigma = rng.uniform(0.2, 0.6)
        g = np.linspace(-6, 6, 241)
        u, v = np.meshgrid(g, g)
        pts = aim + u.ravel()[:, None] * e_phi + v.ravel()[:, None] * np.array([0, 0, 1.0])
        q = optics.footprints_at(pts, np.tile(n_out, (len(pts), 1)), aim[None], t[None], np.array([sigma]),
                                 np.array([1.0]))[0]
        w = q / q.sum()
        uu, vv = u.ravel(), v.ravel()
        cov_num = np.array([[np.sum(w * uu * uu), np.sum(w * uu * vv)], [np.sum(w * uu * vv), np.sum(w * vv * vv)]])
        spot = optics.surface_spot(rec, np.array([phi]), t[None], sigma)
        assert spot["cov"][0] == pytest.approx(cov_num, rel=2e-3, abs=1e-5)
        cos_i = abs(t @ n_out)
        assert spot["semi_major_m"][0] == pytest.approx(2 * sigma / cos_i)
        assert spot["semi_minor_m"][0] == pytest.approx(2 * sigma)
        assert spot["incidence_deg"][0] == pytest.approx(np.degrees(np.arccos(cos_i)))


def test_aim_map_shows_spot_sizes(small_case):
    c = small_case
    from cspaps.field import build_field

    fld = build_field(c)
    el, az = 40.0, 160.0
    snap = evaluate(c, fld, el, az, 850.0)
    spots = beam_spots(c, fld.xyz, snap.aim_azimuth, snap.aim_z, el, az)
    # the model's own beams: same sigma and intercept as the evaluation
    b = snap.beams
    assert spots["sigma_m"][b["index"]] == pytest.approx(b["sigma"])
    assert spots["intercept"] == pytest.approx(snap.eta["intercept"], abs=1e-9)
    # sigma = slant x sqrt(sun^2 + errors^2 + astigmatism^2): grows with the slant range
    assert np.corrcoef(spots["slant_m"], spots["sigma_m"])[0, 1] > 0.5
    err = np.sqrt((c.sun.effective_sigma_mrad * 1e-3) ** 2 + (2e-3 * c.heliostat.slope_error_mrad) ** 2
                  + (2e-3 * c.heliostat.tracking_error_mrad) ** 2)
    assert np.all(spots["sigma_m"] >= spots["slant_m"] * err * (1 - 1e-9))
    grid = build_grid(c.receiver)
    fig = receiver_aims_map(c, grid, snap.flux, snap.aim_azimuth, snap.aim_z, snap.active, fld.ids, snap.dni,
                            spots=spots)
    names = [t.name for t in fig.data]
    outlines = [t for t in fig.data if t.name and "光斑轮廓" in t.name]
    assert outlines[0].visible is True  # a sample is drawn by default
    if fld.n > 40:
        assert "抽样 40 面" in outlines[0].name and outlines[1].visible == "legendonly"
        assert np.isnan(outlines[0].x).sum() >= 40 and np.isnan(outlines[1].x).sum() >= fld.n
    pts = next(t for t in fig.data if t.name and t.name.startswith("镜面中心的瞄准点"))
    assert "光斑 σ" in pts.text[0] and "截获" in pts.text[0]
    sizes = np.asarray(pts.marker.size)
    i_big, i_small = np.argmax(spots["sigma_m"]), np.argmin(spots["sigma_m"])
    assert sizes[i_big] > sizes[i_small]
    assert any("瞄准点" in (n or "") for n in names)
