from pathlib import Path

import pytest

from cspaps import case_from_dict, load_case

ROOT = Path(__file__).resolve().parents[1]


def small_case_dict() -> dict:
    return {
        "name": "tiny",
        "site": {"latitude": 40.0, "longitude": 94.7, "elevation_m": 1200, "timezone": "Asia/Shanghai"},
        "receiver": {"center": [0, 0, 60], "diameter": 6.0, "height": 6.0,
                     "n_azimuth": 36, "n_height": 12},
        "heliostat": {"width": 6.0, "height": 6.0, "pivot_height": 4.0},
        "field": {
            "generator": {"type": "radial_staggered", "r_min": 40, "r_max": 120},
            "exclusion_radius": 40,
        },
        "annual": {"time_step_min": 120},
    }


@pytest.fixture
def small_case():
    return case_from_dict(small_case_dict(), base_dir=ROOT)


@pytest.fixture
def example_case():
    return load_case(ROOT / "cases" / "example.yaml")
