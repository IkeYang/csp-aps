import pytest
import yaml

from cspaps import case_from_dict

from .conftest import small_case_dict


def test_example_loads(example_case):
    assert example_case.receiver.radius == pytest.approx(3.5)
    assert example_case.heliostat.area == pytest.approx(64.0)
    assert example_case.field.generator["type"] == "radial_staggered"
    assert example_case.aps.method == "greedy"
    assert example_case.annual.method == "sun_bins"


def test_unknown_key_rejected():
    d = small_case_dict()
    d["receiver"]["diametr"] = 5.0
    with pytest.raises(ValueError, match="diametr"):
        case_from_dict(d)


def test_unknown_section_rejected():
    d = small_case_dict()
    d["recevier"] = {}
    with pytest.raises(ValueError, match="recevier"):
        case_from_dict(d)


def test_missing_section_rejected():
    d = small_case_dict()
    del d["heliostat"]
    with pytest.raises(ValueError, match="heliostat"):
        case_from_dict(d)


@pytest.mark.parametrize("section,key,value", [
    ("receiver", "diameter", -1.0),
    ("heliostat", "reflectivity", 1.5),
    ("aps", "method", "magic"),
    ("aps", "safety_margin", 1.0),
    ("annual", "method", "weekly"),
    ("sun", "shape", "square"),
])
def test_invalid_values_rejected(section, key, value):
    d = small_case_dict()
    d.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        case_from_dict(d)


def test_yaml_roundtrip(small_case):
    again = case_from_dict(yaml.safe_load(small_case.to_yaml()))
    assert again.to_dict() == small_case.to_dict()


def test_sun_effective_sigma(small_case):
    small_case.sun.shape = "pillbox"
    small_case.sun.half_angle_mrad = 4.65
    assert small_case.sun.effective_sigma_mrad == pytest.approx(2.325)
