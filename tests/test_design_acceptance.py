import numpy as np
import pandas as pd
import pytest

from cspaps import case_from_dict, load_case
from cspaps.benchmark import inner_benchmark
from cspaps.field import Field, build_field
from cspaps.layout_opt import (LayoutEval, design_spacing, formal_ranking, resolve_layout_opt_case,
                               validate_layout)
from cspaps.scenarios import build_scenarios, validation_scenarios
from cspaps.simulate import sun_state
from cspaps.workflows import replay_layout_value, task_layout_value
from tests.conftest import ROOT, small_case_dict


def test_imported_layouts_require_buildable_positions_and_exact_count(tmp_path):
    case = load_case(ROOT / "cases/wrong_site/wrong_site.yaml")
    layouts = {"valid": "layout_A.csv", "outside": [[-127, -129, 5], [127, -129, 5]],
               "crowded": [[11, 81, 5], [13, 82, 5]], "extra": [[0, 90, 5], [40, 90, 5], [80, 90, 5]]}
    result = task_layout_value(case, tmp_path, layouts)
    assert result["ranking"]["name"].tolist() == ["valid"]
    assert set(result["summary"]["excluded_layouts"]) == {"outside", "crowded", "extra"}
    snapshot = load_case(result["run_dir"] / "inputs_case.yaml")
    assert build_scenarios(snapshot).fingerprint() == result["scenario_set"].fingerprint()


def test_imported_field_defaults_to_same_count_and_ranks_initial_layout(tmp_path):
    layout_path = tmp_path / "measured_layout.csv"
    pd.DataFrame({"id": [101, 102], "x": [50.0, 58.0], "y": [0.0, 0.0], "z": [4.0, 4.0]}).to_csv(
        layout_path, index=False)
    data = small_case_dict()
    data["field"]["layout_file"] = str(layout_path)
    data["layout_opt"].update({"n_heliostats": None, "min_spacing_factor": None, "fixed": {},
                               "formal_swap_rounds": 0, "formal_position_trials": 0,
                               "ranking_refine_states": 0, "cross_check_maps": False})
    case = case_from_dict(data, base_dir=tmp_path)

    resolved = resolve_layout_opt_case(case)

    assert case.layout_opt.n_heliostats is None  # resolving defaults does not mutate the caller
    assert resolved.layout_opt.n_heliostats == 2
    assert design_spacing(resolved) == pytest.approx(8.0)
    original = build_field(resolved)
    assert validate_layout(resolved, original.xyz, 2, design_spacing(resolved), original.ids)["status"] == "ok"

    scenarios = build_scenarios(resolved)
    candidate = np.array([[-60.0, 0.0, 4.0], [60.0, 0.0, 4.0]])
    ranked = formal_ranking(resolved, [LayoutEval({}, 2, candidate, 0.0)],
                            scenarios.samples, scenarios.samples)
    assert set(ranked["values"]) == {"search_1", "initial"}


def test_formal_dedup_keeps_xy_pairing():
    case = load_case(ROOT / "cases/wrong_site/wrong_site.yaml")
    case.layout_opt.baseline = "none"
    case.layout_opt.formal_swap_rounds = 0
    case.layout_opt.formal_position_trials = 0
    a = np.array([[-95., -118., 5.], [24., 62., 5.]])
    b = a.copy()
    b[:, 1] = a[::-1, 1]
    candidates = [LayoutEval({}, 2, xyz, 0.) for xyz in (a, b, b[::-1])]
    samples = build_scenarios(case).samples
    result = formal_ranking(case, candidates, samples, samples)
    assert list(result["ranking"]["name"]) == ["search_2", "search_1"]


def test_fixed_response_bound_cannot_certify_joint_shading():
    case = load_case(ROOT / "cases/aksai_demo.yaml")
    case.receiver.center = (0, 0, 55)
    case.receiver.height, case.receiver.diameter = 20, 12
    case.aps.n_height_levels, case.aps.azimuth_offsets_deg = 3, [0.]
    case.aps.height_span, case.aps.auto_grid = 0.75, False
    case.aps.coupled_exact_limit, case.layout_opt.inner_joint_trials = 0, 40
    case.aps.allow_defocus = False
    fld = Field(np.array([[1.659256, -73.187585, 5.], [16.787252, -74.519679, 5.]]))
    _, el, az, _ = sun_state(case, "2024-03-19 08:30")
    states = [dict(label="pair", elevation=el, azimuth=az, dni=np.full(2, 650.), hours=1., map_id="uniform")]
    runs = inner_benchmark(case, fld, states, methods=("greedy", "greedy_rounding"), lam_aps=0.)
    assert runs.objective_mw.max() > runs.dual_bound_mw.min()
    assert runs.state_best_bound_mw.min() >= runs.objective_mw.max() - 1e-10


def test_validation_detects_renamed_training_states(tmp_path):
    case = load_case(ROOT / "cases/wrong_site/wrong_site.yaml")
    train = build_scenarios(case)
    _, info = validation_scenarios(case, train, 40, train.samples, 1000)
    assert info["new_states"] == 0
    df = pd.read_csv(case.resolve(case.scenarios.file)).iloc[::-1].copy()
    df["scenario_id"] = "validation_" + df.scenario_id
    path = tmp_path / "other.csv"
    df.to_csv(path, index=False)
    case.scenarios.validation_file = str(path)
    _, info = validation_scenarios(case, train, 40, train.samples)
    assert info["new_states"] == 0 and info["shared_states"] == 20
    df.loc[df.index[0], "azimuth_deg"] += 1.
    df.to_csv(path, index=False)
    _, info = validation_scenarios(case, train, 40, train.samples)
    assert info["new_states"] == 1 and info["shared_states"] == 19


def test_replay_uses_frozen_inputs_after_originals_move(tmp_path):
    import shutil

    source = tmp_path / "source"
    shutil.copytree(ROOT / "cases/wrong_site", source)
    case = load_case(source / "wrong_site.yaml")
    run = task_layout_value(case, tmp_path / "runs", {"A": "layout_A.csv", "B": "layout_B.csv"})
    assert run["summary"]["comparison_status"] == "certified_leader"
    source.rename(tmp_path / "moved")
    replay = replay_layout_value(run["run_dir"], tmp_path / "replay")
    assert replay["scenario_set"].fingerprint() == run["scenario_set"].fingerprint()
    assert replay["ranking"].name.tolist() == ["B", "A"]
    assert list(replay["ranking"].value_mwh) == pytest.approx(list(run["ranking"].value_mwh))
