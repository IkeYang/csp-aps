import json

import numpy as np
import pytest

from cspaps.cli import main
from cspaps.field import build_field
from cspaps.simulate import STAGES, annual, snapshot


def test_snapshot_power_chain(small_case):
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-03-20 13:30")
    assert snap.sun_elevation > 30
    for prev, k in zip(STAGES[:-1], STAGES[1:]):
        np.testing.assert_allclose(snap.power[k], snap.power[prev] * snap.eta[k])
        assert np.all((snap.eta[k] >= 0) & (snap.eta[k] <= 1))
    # Flux map integrates to the power on the receiver (coarse 36 x 12 grid)
    assert snap.flux.sum() * snap.grid.cell_area == pytest.approx(snap.power["intercept"].sum(), rel=0.03)
    s = snap.summary()
    assert 0.4 < s["total_optical_efficiency"] < 0.9


def test_snapshot_without_flux(small_case):
    fld = build_field(small_case)
    a = snapshot(small_case, fld, "2025-03-20 13:30")
    b = snapshot(small_case, fld, "2025-03-20 13:30", with_flux=False)
    np.testing.assert_allclose(a.power["intercept"], b.power["intercept"])
    assert b.flux.max() == 0


def test_north_field_beats_south_at_noon(small_case):
    """Northern hemisphere: heliostats north of the tower have better cosine at noon."""
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-12-21 13:40")
    north = fld.xyz[:, 1] > 30
    south = fld.xyz[:, 1] < -30
    assert snap.eta["cosine"][north].mean() > snap.eta["cosine"][south].mean() + 0.1


def test_night_is_zero(small_case):
    fld = build_field(small_case)
    snap = snapshot(small_case, fld, "2025-06-21 02:00")
    assert snap.power["intercept"].sum() == 0
    assert snap.flux.max() == 0


def test_dni_override_scales_linearly(small_case):
    fld = build_field(small_case)
    a = snapshot(small_case, fld, "2025-06-21 13:00", dni=500)
    b = snapshot(small_case, fld, "2025-06-21 13:00", dni=1000)
    assert b.power["intercept"].sum() == pytest.approx(2 * a.power["intercept"].sum())


def test_annual_consistency(small_case):
    fld = build_field(small_case)
    res = annual(small_case, fld)
    e = res.energy_mwh
    assert all(e[prev] >= e[k] for prev, k in zip(STAGES[:-1], STAGES[1:]))
    assert res.heliostat_energy_mwh.sum() == pytest.approx(e["intercept"])
    # 12 representative days, 2 h steps, only daylight instants above 5 deg
    assert 0 < len(res.timeseries) < 12 * 12
    assert 0.4 < res.summary()["total_optical_efficiency"] < 0.9


def test_cli_snapshot(tmp_path, example_case):
    case_path = example_case.base_dir / "example.yaml"
    assert main(["snapshot", str(case_path), "--time", "2025-06-21 13:40", "--out", str(tmp_path)]) == 0
    out = tmp_path / "example"
    data = json.loads((out / "snapshot_20250621_1340.json").read_text())
    assert data["n_heliostats"] > 500
    assert (out / "snapshot_20250621_1340_flux.png").stat().st_size > 0
