"""Speed of the annual-value inner loop: numpy reference vs compiled CPU kernels vs GPU, and multi-GPU throughput.

Run on the target machine (e.g. 8 x RTX 2080 Ti / 3080):

    python scripts/gpu_benchmark.py cases/aksai_demo.yaml --heliostats 473 1500 --scenarios 6
    python scripts/gpu_benchmark.py cases/aksai_demo.yaml --heliostats 3000 --throughput 64 --workers -1

Part 1 (one process, the same scenarios in every configuration, each in a fresh process so the environment
variables apply from the start):
  reference   CSPAPS_JIT=0, no GPU         -- the numpy code
  cpu-jit     compiled CPU kernels          -- bit-identical to the reference (checked)
  gpu         compiled kernels + CSPAPS_GPU=1 (needs CuPy) -- agrees to float32 rounding (checked)
Part 2 (``--throughput N``): N scenarios of one layout over a process pool (``--workers``, -1 = all cores), CPU
only and with every GPU; scenarios per minute. Results go to ``results/gpu_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "reference": {"CSPAPS_JIT": "0", "CSPAPS_GPU": "0"},
    "cpu-jit": {"CSPAPS_JIT": "1", "CSPAPS_GPU": "0"},
    "gpu": {"CSPAPS_JIT": "1", "CSPAPS_GPU": "1"},
}


def _case(path: str, n: int):
    from cspaps import load_case

    case = load_case(path)
    gen = case.field.generator or {}
    if n and case.field.layout_file is None and gen.get("r_max"):
        from cspaps.field import build_field

        base = build_field(case).n
        gen["r_max"] = float(gen["r_max"]) * (n / max(base, 1)) ** 0.5  # area grows with the mirror count
    return case


def child(args) -> None:
    """One configuration in this process: time every scenario; print one JSON line."""
    import numpy as np

    from cspaps import backend, kernels
    from cspaps.field import build_field
    from cspaps.layout_value import evaluate_layout_value, inner_settings, solve_scenario
    from cspaps.scenarios import build_scenarios

    case = _case(args.case, args.n)
    fld = build_field(case)
    sset = build_scenarios(case)
    settings = inner_settings(case)
    out = {"config": args.config, "heliostats": fld.n, **backend.describe(), "cpu_kernels": kernels.name()}
    if args.throughput:
        # about ``throughput`` scenarios: representative clear-sky days of ~11 working hours each
        sub = build_scenarios(case, n_days=max(1, round(args.throughput / 11)))
        t0 = time.perf_counter()
        v = evaluate_layout_value(case, fld.xyz, sub, "bench", ids=fld.ids, settings=settings, workers=args.workers)
        dt = time.perf_counter() - t0
        out.update(scenarios=len(sub), seconds=round(dt, 1), scenarios_per_min=round(60 * len(sub) / dt, 2),
                   value_mwh=v.totals().get("value_mwh"))
    else:
        idx = np.linspace(0, len(sset) - 1, args.scenarios).astype(int)
        solve_scenario(case, fld, sset, int(idx[0]), settings)  # warm-up: kernel compilation, CUDA context
        times, plans = [], []
        for i in idx:
            t0 = time.perf_counter()
            rec = solve_scenario(case, fld, sset, int(i), settings)
            times.append(time.perf_counter() - t0)
            plans.append([rec.plan.status, rec.plan.p_abs_w, rec.plan.u_w])
        out.update(seconds_per_scenario=round(sum(times) / len(times), 3), plans=plans)
    print("RESULT " + json.dumps(out, default=str), flush=True)


def run(config: str, args, extra: list[str]) -> dict | None:
    env = {**os.environ, **CONFIGS[config]}
    cmd = [sys.executable, __file__, args.case, "--child", config, *extra]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print(f"  {config}: failed\n{p.stderr[-2000:]}", file=sys.stderr)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--heliostats", type=int, nargs="+", default=[0], help="field sizes (0 = the case's field)")
    ap.add_argument("--scenarios", type=int, default=6)
    ap.add_argument("--throughput", type=int, default=0, help="scenarios for the multi-process throughput test")
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--child", default=None)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    if args.child:
        args.config = args.child
        return child(args)

    from cspaps import backend

    os.environ["CSPAPS_GPU"] = "1"
    have_gpu = backend.backend_name() == "cupy"
    print(json.dumps(backend.describe(), ensure_ascii=False))
    report = {"single_process": [], "throughput": []}
    for n in args.heliostats:
        rows = {}
        for config in CONFIGS:
            if config == "gpu" and not have_gpu:
                continue
            r = run(config, args, ["--n", str(n), "--scenarios", str(args.scenarios)])
            if r:
                rows[config] = r
        ref = rows.get("reference")
        for config, r in rows.items():
            same = None
            if ref and config != "reference":
                pairs = list(zip(ref["plans"], r["plans"]))
                same = ("bit-identical" if all(a == b for a, b in pairs) else
                        f"max rel. diff of P {max(abs(a[1] - b[1]) / max(abs(a[1]), 1) for a, b in pairs):.1e}")
            speed = ref["seconds_per_scenario"] / r["seconds_per_scenario"] if ref else None
            row = {"heliostats": r["heliostats"], "config": config, "s_per_scenario": r["seconds_per_scenario"],
                   "speedup": round(speed, 2) if speed else None, "vs_reference": same}
            report["single_process"].append(row)
            print(json.dumps(row, ensure_ascii=False))
        if args.throughput:
            for config in ("cpu-jit", "gpu") if have_gpu else ("cpu-jit",):
                r = run(config, args, ["--n", str(n), "--throughput", str(args.throughput),
                                       "--workers", str(args.workers)])
                if r:
                    report["throughput"].append(r)
                    print(json.dumps({k: r[k] for k in ("heliostats", "config", "scenarios", "seconds",
                                                         "scenarios_per_min")}, ensure_ascii=False))
    out = Path("results") / "gpu_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
