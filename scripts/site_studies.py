"""Site studies for the annual layout value with spatial DNI (Aksai, Hami, Port Augusta).

For each site case: the formal layout ranking (per-scenario APS on the scenario set with DNI maps, initial
field ranked too, swap polishing, denser-scenario check), the same winner / initial field on uniform DNI maps,
the energy-only vs uniformity-weighted pre-screen layouts, and a single-instant cloud comparison.

    python scripts/site_studies.py [aksai_demo hami_2024 port_augusta_2024] [--out results/site_studies]
    python scripts/site_studies.py hami_2024 --tag _uniform --map-model uniform      # design on uniform DNI
    python scripts/site_studies.py hami_2024 --robustness hami_2024 hami_2024_uniform  # winners on two cloud-map sets

Writes <out>/<case>/... run folders and <out>/site_studies.json (the numbers quoted in the README); the
robustness check writes <out>/<case>_robustness.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cspaps import load_case
from cspaps import workflows as wf
from cspaps.tables import export_name

ROOT = Path(__file__).resolve().parents[1]


def study(name: str, out: Path, n_scenarios: int, dense: int, map_model: str | None = None,
          cloud_maps: int | None = None) -> dict:
    case = load_case(ROOT / "cases" / f"{name}.yaml")
    if map_model:
        case.scenarios.map_model = map_model
    if cloud_maps:
        case.scenarios.cloud_maps = cloud_maps
    lo = case.layout_opt
    lo.refine_top, lo.formal_swap_rounds, lo.dense_check_scenarios = 3, 2, dense
    lo.review_check_hourly, lo.workers = False, 4
    case.scenarios.n_scenarios = n_scenarios
    t0 = time.perf_counter()
    say = lambda m: print(f"  [{name} {time.perf_counter() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    opt = wf.task_optimize_layout(case, out, progress=say)
    res, s = opt["result"], opt["summary"]
    f = res.formal
    values, ranking = f["values"], f["ranking"]
    win = f["winner"]
    rows = []
    for _, r in ranking.iterrows():
        rows.append({k: (r[k] if not isinstance(r[k], (np.floating, float)) or np.isfinite(r[k]) else None)
                     for k in ("rank", "name", "value_mwh", "energy_mwh", "penalty_mwh", "weighted_penalty_mwh",
                               "delta_value_mwh", "delta_energy_mwh", "delta_uniformity_mwh", "inner_gap_annual",
                               "max_inner_gap", "circ_penalty_mwh", "cv_2d_power_weighted")})
        rows[-1]["status_counts"] = r["status_counts"]
    search_order = [n for n in ranking["name"] if n.startswith("search_") and "+" not in n]
    # the same two layouts on uniform DNI (compatibility scenarios): how much the maps change the comparison
    uni = case.copy()
    uni.scenarios.map_model = "uniform"
    say("uniform-DNI comparison of the winner and the initial field")
    lv = wf.task_layout_value(uni, out, {"winner": values[win].xyz, "initial": "initial"}, progress=say)
    uni_rank = lv["ranking"]
    # transient features (cloud shadows): the formal ranking re-scores winner and initial field on independent maps
    cross = f.get("cross_maps")
    # lambda_L = 0 (energy only) vs the case weight on the centre-aim pre-screen: where the mirrors go
    say("layout trade-off lambda_L 0 / case weight")
    tr = wf.task_layout_tradeoff(case, out, lambdas=[0.0, lo.uniformity_weight],
                                 params={k: v for k, v in res.best.params.items() if k != "source"}
                                 if "source" not in res.best.params else None)
    say("cloud comparison")
    cc = wf.task_cloud_compare(case, out, case.realtime.start, n_clouds=2, seed=3)
    states = {st["key"]: {k: st[k] for k in ("p_abs_mw", "u_mw", "objective_mw", "cv_2d", "peak_flux_kw_m2",
                                              "peak_to_limit", "n_defocused", "n_changed_vs_a")} for st in cc["states"]}
    north = [{"lambda": r["lambda"], "north_fraction": r["north_fraction"], "energy_mwh": r["energy_mwh"],
              "cv_circ_weighted": r["cv_circ_weighted"], "centroid_offset_m": r["centroid_offset_m"]}
             for r in tr["summary"]["curve"]]
    ub = s["upper_bound"]
    return {
        "case": name, "latitude": case.site.latitude, "runtime_s": round(time.perf_counter() - t0, 1),
        "run_dir": str(opt["run_dir"]), "scenario_set": s["formal"]["scenario_set"],
        "lambda_aps": f["settings"].lam_aps, "lambda_layout": lo.uniformity_weight,
        "ranking": rows, "winner": win, "search_order_prescreen": search_order, "swaps": f["swaps"],
        "dense_check": f.get("dense"), "displacement_vs_initial": s["formal"].get("displacement_vs_initial"),
        "bound": {k: ub.get(k) for k in ("upper_bound_mwh", "found_energy_mwh", "value_found_mwh", "gap",
                                          "gap_value_conservative", "packing_capacity_bound")},
        "uniform_dni": [{"name": r["name"], "value_mwh": r["value_mwh"], "energy_mwh": r["energy_mwh"],
                         "penalty_mwh": r["penalty_mwh"]} for _, r in uni_rank.iterrows()],
        "cross_maps": cross, "map_settings": {k: getattr(case.scenarios, k) for k in (
            "map_model", "cloud_maps", "cloud_n", "seed", "map_contrast", "map_axis_deg")},
        "prescreen_tradeoff": north, "cloud_compare": {"time": cc["summary"]["time"],
                                                       "shaded": cc["summary"]["shaded_heliostats"],
                                                       "mean_transmittance": cc["summary"]["mean_transmittance"],
                                                       "states": states},
        "validation": s["validation"]["status"],
    }


def robustness(name: str, keys: list[str], out: Path) -> dict:
    """Winners of earlier runs (keys of site_studies.json) and the initial field on the case's maps and on
    independent maps (seed + 1000): does a lead survive other cloud positions?"""
    done = json.loads((out / "site_studies.json").read_text(encoding="utf-8"))
    layouts = {}
    for key in keys:
        r = done[key]
        layouts[key] = str(Path(r["run_dir"]) / "formal" / "values" / export_name(r["winner"]) / "layout.csv")
    layouts["initial"] = "initial"
    res = {}
    base = load_case(ROOT / "cases" / f"{name}.yaml")
    for seed in (base.scenarios.seed, base.scenarios.seed + 1000):
        case = base.copy()
        case.scenarios.seed = seed
        case.layout_opt.workers = 4
        lv = wf.task_layout_value(case, out, layouts)
        res[str(seed)] = [{"name": r["name"], "value_mwh": r["value_mwh"], "energy_mwh": r["energy_mwh"],
                           "penalty_mwh": r["penalty_mwh"]} for _, r in lv["ranking"].iterrows()]
    (out / f"{name}_robustness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cases", nargs="*", default=["aksai_demo", "hami_2024", "port_augusta_2024"])
    p.add_argument("--out", default=str(ROOT / "results" / "site_studies"))
    p.add_argument("--scenarios", type=int, default=20)
    p.add_argument("--dense", type=int, default=40)
    p.add_argument("--tag", default="", help="suffix of the result key (keeps earlier runs of the same case)")
    p.add_argument("--map-model", default=None, help="override scenarios.map_model (e.g. uniform)")
    p.add_argument("--robustness", nargs="+", default=None, help="result keys whose winners are re-scored")
    p.add_argument("--cloud-maps", type=int, default=None, help="override scenarios.cloud_maps")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.robustness:
        print(json.dumps(robustness(a.cases[0], a.robustness, out), indent=1))
        return
    path = out / "site_studies.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in a.cases:
        print(f"=== {name}", flush=True)
        done[name + a.tag] = study(name, out, a.scenarios, a.dense, a.map_model, a.cloud_maps)
        path.write_text(json.dumps(done, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
