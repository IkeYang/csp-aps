"""Solver benchmarks and ray-trace checks for the record (improvement plan step 3).

    python scripts/solver_benchmark.py [--case aksai_demo] [--out results/solver_benchmark]
        [--inner-budget 60] [--outer-budget 300] [--seeds 1 2] [--parts inner inner_binding suite outer]

Writes run folders under <out>/<case>/ and <out>/<case>_benchmark.json with
  inner           four task-1 solvers on the 6 largest scenarios (case limits and lambda)
  inner_binding   the same states, power only (lambda_APS = 0), every flux limit scaled so that centre aiming
                  exceeds it by 30 % at the summer-solstice noon
  outer           differential evolution vs multi-start free positions (same budget, seeds, V on train / held out)
  suite           analytic vs MCRT on low sun, cloud edge, high-load mixed aiming and near the limit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cspaps import load_case
from cspaps import workflows as wf
from cspaps.field import build_field
from cspaps.simulate import evaluate
from cspaps.validation import clear_dni, solar_noon, summer_solstice

ROOT = Path(__file__).resolve().parents[1]


def binding_factor(case, overload: float = 1.3) -> float:
    """Limit scale so that centre aiming at the summer-solstice noon exceeds the limit by ``overload``."""
    fld = build_field(case)
    ts, el, az = solar_noon(case, summer_solstice(case, 2025))
    snap = evaluate(case, fld, el, az, clear_dni(case, el))
    return float(snap.verify(case.receiver).peak_ratio) / overload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="aksai_demo")
    ap.add_argument("--out", default=str(ROOT / "results" / "solver_benchmark"))
    ap.add_argument("--inner-budget", type=float, default=60.0)
    ap.add_argument("--outer-budget", type=float, default=300.0)
    ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-scenarios", type=int, default=None, help="formal scenario count for the layout benchmark")
    ap.add_argument("--parts", nargs="+", default=["inner", "inner_binding", "suite", "outer"],
                    choices=["inner", "inner_binding", "suite", "outer"])
    ap.add_argument("--rays", type=int, default=2000, help="MCRT rays per heliostat of the ray-trace checks")
    ap.add_argument("--mcrt-seeds", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    case = load_case(ROOT / "cases" / f"{args.case}.yaml")
    case.layout_opt.workers = 4
    t0 = time.perf_counter()
    say = lambda m: print(f"  [{args.case} {time.perf_counter() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    path = out / f"{args.case}_benchmark.json"
    record: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}  # parts run before are kept
    record["case"] = args.case
    if "inner" in args.parts:
        r = wf.task_solver_benchmark(case, out, "inner", args.n_states, args.inner_budget, progress=say)
        record["inner"] = r["summary"]["inner"]
    if "inner_binding" in args.parts:
        # power only (lambda_APS = 0) with every limit scaled so that centre aiming exceeds it by 30 %
        f = binding_factor(case)
        say(f"binding limits: factor {f:.4f}, lambda_APS = 0")
        r = wf.task_solver_benchmark(case, out, "inner", args.n_states, args.inner_budget, limit_factor=f, lam_aps=0.0,
                                     progress=say)
        record["inner_binding"] = {**r["summary"]["inner"], "limit_factor": f, "lambda_aps": 0.0,
                                   "max_flux_kw_m2": r["summary"]["max_flux_kw_m2"]}
    if "suite" in args.parts:
        s = wf.task_validation_suite(case, out, rays=args.rays, seeds=args.mcrt_seeds, progress=say)
        record["suite"] = s["summary"]["table"]
    if "outer" in args.parts:
        c = case.copy()
        if args.n_scenarios:
            c.scenarios.n_scenarios = args.n_scenarios
        r = wf.task_solver_benchmark(c, out, "outer", budget_s=args.outer_budget, seeds=tuple(args.seeds), progress=say)
        record["outer"] = r["summary"]["outer"]
    record.setdefault("runtime_s", {})
    if not isinstance(record["runtime_s"], dict):
        record["runtime_s"] = {}
    record["runtime_s"]["+".join(args.parts)] = time.perf_counter() - t0
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
                    encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
