from pathlib import Path

import pytest

from cspaps import case_from_dict, load_case

ROOT = Path(__file__).resolve().parents[1]


def small_case_dict() -> dict:
    return {
        "name": "tiny",
        "site": {"latitude": 40.0, "longitude": 94.7, "elevation_m": 1200, "timezone": "Asia/Shanghai"},
        "receiver": {"center": [0, 0, 60], "diameter": 6.0, "height": 6.0,
                     "n_azimuth": 36, "n_height": 12, "max_flux_kw_m2": 1000},
        "heliostat": {"width": 6.0, "height": 6.0, "pivot_height": 4.0},
        "field": {
            "generator": {"type": "radial_staggered", "r_min": 40, "r_max": 110},
            "exclusion_radius": 40,
        },
        "shading": {"n_points": 6},
        "aps": {"auto_grid": False},  # keep the tests fast; verification still guards feasibility
        "annual": {"time_step_min": 120, "n_sun_bins": 12},
        "realtime": {"start": "2025-06-21 12:30", "duration_min": 3, "step_s": 60,
                     "random_clouds": {"n": 2, "seed": 1, "radius_m": [40, 60], "speed_m_s": 6}},
        "layout_opt": {"n_heliostats": 120, "maxiter": 1, "popsize": 2, "refine_top": 1,
                       "n_sun_bins": 6, "shading_points": 4, "workers": 1, "polish_rounds": 1},
        "mcrt": {"rays_per_heliostat": 400},
    }


@pytest.fixture
def small_case():
    return case_from_dict(small_case_dict(), base_dir=ROOT)


@pytest.fixture
def example_case():
    return load_case(ROOT / "cases" / "example.yaml")
