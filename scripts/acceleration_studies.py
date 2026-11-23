"""Evidence for the 0.9.0 annual-computation improvements (README 0.9.0).

    python scripts/acceleration_studies.py [--parts swaps reuse outer] [--case aksai_demo] [--workers 4]
        [--outer-budget 600] [--out results/annual_acceleration]

  swaps   energy- vs flux-price (dual) swap selection on the initial field cut to N mirrors (free positions:
          the case's other positions within 1.6 x the layout radius); predicted and re-solved V change
  reuse   wall time of one layout's formal value with reuse off / responses / full, per-map vs shared sun
          directions, and three layouts serially vs in one pool
  outer   differential evolution vs Bayesian search under the same wall-time budget, V on training and
          held-out scenarios (``benchmark.outer_benchmark``)

The control-variate check against hourly APS is ``scripts/validate_annual_estimators.py``.
Writes <out>/<case>_<part>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cspaps import load_case
from cspaps.field import Field, build_field
from cspaps.layout_value import (baseline_layout, dual_prices, evaluate_layout_value, evaluate_layout_values,
                                 inner_settings, price_swaps)
from cspaps.scenarios import build_scenarios
from cspaps.simulate import annual

ROOT = Path(__file__).resolve().parents[1]


def _case(name: str, workers: int):
    case = load_case(ROOT / "cases" / f"{name}.yaml")
    case.layout_opt.workers = workers
    return case


def swaps(name: str, workers: int, sizes=(80, 150), n_scen: int = 8) -> list[dict]:
    rows = []
    for n_mirrors in sizes:
        case = _case(name, workers)
        case.layout_opt.control_variate = False
        fld = build_field(case)
        xyz, ids, _ = baseline_layout(case, n_mirrors)
        key = {tuple(np.round(p, 6)) for p in xyz}
        r = np.hypot(fld.xyz[:, 0], fld.xyz[:, 1])
        pool_xyz = fld.xyz[r < 1.6 * np.hypot(xyz[:, 0], xyz[:, 1]).max()]
        in_layout = np.array([tuple(np.round(p, 6)) in key for p in pool_xyz])
        unused = np.flatnonzero(~in_layout)
        m = max(1, int(round(case.layout_opt.swap_fraction * n_mirrors)))
        sset = build_scenarios(case, n_scen)
        st = inner_settings(case)
        cur = evaluate_layout_value(case, xyz, sset, "base", ids=ids, settings=st, keep_flux=False)
        v0 = cur.totals()["value_mwh"]
        t0 = time.perf_counter()
        pr = dual_prices(case, cur, pool_xyz[unused], sset, st)
        t_price = time.perf_counter() - t0
        drop_d, add_d, pred = price_swaps(pr, m)
        price = annual(case, Field(pool_xyz), samples=sset.samples, aiming="center", with_flux=False,
                       workers=workers, blockers=in_layout).heliostat_energy_mwh
        add_e = unused[np.argsort(-price[unused])[:m]]
        drop_e = np.argsort(cur.mirror_energy_mwh(), kind="stable")[:len(add_e)]
        row = {"n_heliostats": n_mirrors, "free_positions": int(unused.size), "scenarios": len(sset), "swap_size": m,
               "value_mwh": v0, "pricing_time_s": round(t_price, 1), "dual_predicted_mwh": pred,
               "dual_bound_gap_mwh": pr["gap_mwh"]}
        for label, drop, add in (("energy", drop_e, add_e), ("dual", drop_d, unused[add_d])):
            if len(add) == 0:
                row[f"{label}_gain_mwh"] = 0.0
                continue
            new_xyz = np.vstack([np.delete(xyz, drop, axis=0), pool_xyz[add]])
            v = evaluate_layout_value(case, new_xyz, sset, label, settings=st, keep_flux=False)
            row[f"{label}_gain_mwh"] = v.totals()["value_mwh"] - v0
        print(row, flush=True)
        rows.append(row)
    return rows


def reuse(name: str, workers: int, n_scen: int = 20) -> dict:
    out = {}
    for shared in (False, True):
        case = _case(name, workers)
        case.layout_opt.control_variate = False
        case.scenarios.dni_bands = 2
        case.scenarios.shared_directions = shared
        sset = build_scenarios(case, n_scen)
        xyz, ids, _ = baseline_layout(case, case.layout_opt.n_heliostats or 473)
        dirs = {(float(e), float(a)) for e, a in zip(sset.samples.elevation, sset.samples.azimuth)}
        for mode in (("off", "responses", "full") if not shared else ("responses", "full")):
            st = replace(inner_settings(case), reuse=mode)
            t0 = time.perf_counter()
            v = evaluate_layout_value(case, xyz, sset, mode, ids=ids, settings=st, keep_flux=False)
            key = f"{'shared' if shared else 'per_map'}_{mode}"
            out[key] = {"scenarios": len(sset), "directions": len(dirs), "time_s": round(time.perf_counter() - t0, 1),
                        "rescaled_plans": sum("reused_from" in r.plan.info for r in v.records),
                        "value_mwh": round(v.totals()["value_mwh"], 2)}
            print(key, out[key], flush=True)
    case = _case(name, workers)
    case.layout_opt.control_variate = False
    sset = build_scenarios(case, 6)
    fld = build_field(case)
    n = min(300, fld.n)
    rng = np.random.default_rng(0)
    lays = {f"L{k}": (fld.xyz[np.sort(rng.choice(fld.n, n, replace=False))], None) for k in range(3)}
    st = inner_settings(case)
    t0 = time.perf_counter()
    for k, (x, _) in lays.items():
        evaluate_layout_value(case, x, sset, k, settings=st, keep_flux=False)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    evaluate_layout_values(case, lays, sset, settings=st, keep_flux=False)
    out["three_layouts_six_scenarios"] = {"serial_s": round(t_serial, 1), "one_pool_s": round(time.perf_counter() - t0, 1)}
    print(out["three_layouts_six_scenarios"], flush=True)
    return out


def outer(name: str, workers: int, budget: float, seeds=(1, 2)) -> dict:
    from cspaps.benchmark import outer_benchmark

    case = _case(name, workers)
    t0 = time.perf_counter()
    res = outer_benchmark(case, methods=("de", "bayes"), seeds=seeds, budget_s=budget,
                          progress=lambda m: print(f"[{time.perf_counter() - t0:6.0f}s] {m}", flush=True))
    print(res["summary"].to_string(), flush=True)
    return {"budget_s": budget, "runs": res["runs"].to_dict("records"), "summary": res["summary"].to_dict("records"),
            "train": res["train"], "settings_digest": res["settings_digest"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="aksai_demo")
    ap.add_argument("--parts", nargs="+", default=["swaps", "reuse", "outer"], choices=["swaps", "reuse", "outer"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outer-budget", type=float, default=600.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "annual_acceleration"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for part in a.parts:
        print(f"=== {a.case}: {part}", flush=True)
        if part == "swaps":
            res = swaps(a.case, a.workers)
        elif part == "reuse":
            res = reuse(a.case, a.workers)
        else:
            res = outer(a.case, a.workers, a.outer_budget)
        (out / f"{a.case}_{part}.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
