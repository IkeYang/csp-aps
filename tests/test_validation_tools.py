import json
import py_compile

import numpy as np
import pytest

from cspaps import optics, sun
from cspaps.field import build_field
from cspaps.mcrt import compare, convergence, trace, trace_repeated
from cspaps.receiver import default_aim, surface_point
from cspaps.shading import compare_screening
from cspaps.simulate import evaluate, sun_state
from cspaps.soltrace import export_benchmark


def test_screening_matches_bruteforce(small_case):
    fld = build_field(small_case)
    for t in ("2025-12-21 09:30", "2025-06-21 13:00"):
        ts, el, az, _ = sun_state(small_case, t)
        s = sun.sun_vector(el, az)
        a_az, a_z = default_aim(small_case.receiver, fld.xyz)
        n, tt, *_ = optics.heliostat_geometry(fld.xyz, surface_point(small_case.receiver, a_az, a_z + 2.5), s)
        r = compare_screening(fld.xyz, n, s, tt, small_case.heliostat, small_case.receiver, 6)
        assert r["max_abs_diff_combined"] == 0.0 and r["missed_loss_total"] == 0.0


def test_mcrt_repeated_statistics_and_convergence(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, "2025-06-21 13:00")
    mean, st = trace_repeated(small_case, fld, el, az, dni, rays_per_heliostat=300, n_seeds=3, seed0=5)
    assert st["seeds"] == [5, 6, 7] and st["rays_per_heliostat"] == 300
    assert st["power_on_receiver_mw"]["std_error"] > 0
    assert 0 < st["relative_std_error_at_peak_cell"] < 0.2
    snap = evaluate(small_case, fld, el, az, dni)
    c = compare(snap, mean)
    assert c["grid"] == [small_case.receiver.n_azimuth, small_case.receiver.n_height]
    assert c["difference_direction"] == "analytic minus MCRT"
    conv = convergence(small_case, fld, el, az, dni, ray_counts=(200, 800), n_seeds=2)
    assert conv[1]["rel_se_at_peak"] < conv[0]["rel_se_at_peak"]  # more rays -> less noise


def test_mcrt_specular_and_tracking_bias(small_case):
    fld = build_field(small_case)
    ts, el, az, dni = sun_state(small_case, "2025-06-21 13:00")
    base = trace(small_case, fld, el, az, dni, rays_per_heliostat=600, seed=1)
    small_case.heliostat.specular_error_mrad = 3.0
    wide = trace(small_case, fld, el, az, dni, rays_per_heliostat=600, seed=1)
    assert wide.flux.max() < base.flux.max()  # extra scatter spreads the image
    assert optics.beam_sigma(np.array([100.0]), np.array([1.0]), small_case.heliostat, small_case.sun)[0] > \
        optics.beam_sigma(np.array([100.0]), np.array([1.0]), type(small_case.heliostat)(6, 6, 4), small_case.sun)[0]
    small_case.heliostat.specular_error_mrad = 0.0
    small_case.mcrt.tracking_mode = "per_heliostat"
    biased = trace(small_case, fld, el, az, dni, rays_per_heliostat=600, seed=1)
    assert biased.power["intercept"] == pytest.approx(base.power["intercept"], rel=0.05)


def test_soltrace_export(small_case, tmp_path):
    fld = build_field(small_case)
    res = export_benchmark(small_case, fld, "2025-06-21 13:00", tmp_path, mcrt_rays=200, mcrt_seeds=2)
    data = json.loads(res["files"]["benchmark_json"].read_text())
    assert len(data["heliostats"]) == fld.n and data["heliostats"][0]["id"] == int(fld.ids[0])
    assert data["conventions"]["attenuation"] == "excluded on both sides"
    assert np.asarray(data["analytic"]["flux_w_m2"]).shape == (small_case.receiver.n_height, small_case.receiver.n_azimuth)
    n = np.asarray(data["heliostats"][0]["normal"])
    assert np.linalg.norm(n) == pytest.approx(1.0)
    py_compile.compile(str(res["files"]["pysoltrace_driver"]), doraise=True)
    assert abs(res["summary"]["power_on_receiver_mw"]["analytic"] / res["summary"]["power_on_receiver_mw"]["cspaps_mcrt"] - 1) < 0.05
