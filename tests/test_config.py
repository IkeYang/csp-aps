import pytest

from cspaps import case_from_dict

from .conftest import small_case_dict


def test_example_loads(example_case):
    assert example_case.receiver.radius == pytest.approx(3.5)
    assert example_case.heliostat.area == pytest.approx(64.0)
    assert example_case.field.generator["type"] == "radial_staggered"


def test_unknown_key_rejected():
    d = small_case_dict()
    d["receiver"]["diametr"] = 5.0
    with pytest.raises(ValueError, match="diametr"):
        case_from_dict(d)


def test_missing_section_rejected():
    d = small_case_dict()
    del d["heliostat"]
    with pytest.raises(ValueError, match="heliostat"):
        case_from_dict(d)
