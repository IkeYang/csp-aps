import pytest

from cspaps.field import build_field
from cspaps.mcrt import compare, trace
from cspaps.simulate import evaluate, sun_state


@pytest.mark.parametrize("time", ["2025-06-21 13:00", "2025-12-21 10:00"])
def test_mcrt_agrees_with_analytic(small_case, time):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, time)
    snap = evaluate(small_case, fld, el, az, dni)
    ray = trace(small_case, fld, el, az, dni, rays_per_heliostat=1500, seed=1)
    c = compare(snap, ray)
    assert abs(c["power_on_receiver_mw"]["rel_diff"]) < 0.03
    for stage in ("cosine", "reflectivity", "attenuation"):
        assert abs(c["efficiency_by_stage"][stage]["diff"]) < 2e-3
    for stage in ("shading", "blocking", "intercept"):
        assert abs(c["efficiency_by_stage"][stage]["diff"]) < 0.02
    assert c["flux_map_correlation"] > 0.95


def test_mcrt_is_reproducible(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, "2025-06-21 13:00")
    a = trace(small_case, fld, el, az, dni, rays_per_heliostat=200, seed=7)
    b = trace(small_case, fld, el, az, dni, rays_per_heliostat=200, seed=7)
    assert a.power == b.power
