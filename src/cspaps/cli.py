"""Command line interface: ``cspaps <command> CASE.yaml [options]``.

Exit codes: 0 = done, 2 = the result is not executable / not valid
(APS status "infeasible", layout capacity shortfall or constraint violation).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import workflows as wf
from .config import load_case

BAD_STATUS = {"infeasible", "capacity_shortfall", "constraint_violation", "missed_losses"}


def _print(result: dict) -> int:
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
    for label, path in result["files"].items():
        print(f"  {label}: {path}")
    print(f"  run folder: {result.get('run_dir')}")
    status = result["summary"].get("status")
    if status in BAD_STATUS:
        print(f"STATUS: {status}", file=sys.stderr)
        return 2
    return 0


def _progress(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cspaps", description="Heliostat field layout & aim-point strategy optimisation")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("case", help="case YAML file")
        sp.add_argument("--out", default="results", help="output folder (default: results)")
        return sp

    add("layout", "generate the field layout (CSV + plot)")
    for name, help_ in [("snapshot", "one instant: efficiency breakdown + flux map (default: centre aiming)"),
                        ("aps", "control engine, one instant: optimise and verify aim points")]:
        sp = add(name, help_)
        sp.add_argument("--time", required=True, help='local time, e.g. "2025-06-21 13:40"')
        sp.add_argument("--method", choices=["center", "kfactor", "greedy", "milp"], default=None)
        sp.add_argument("--dni", type=float, default=None, help="uniform DNI override (W/m2)")
        sp.add_argument("--dni-map", default=None, help="CSV x,y,dni: measured DNI at the heliostats")
        sp.add_argument("--forecast-dni-map", default=None, help="CSV x,y,dni: highest DNI expected before the next update")
        sp.add_argument("--prev-aims", default=None, help="aims CSV of the last executed plan (matched by id)")
    add("realtime", "replay real-time APS under cloud shadows (scenario from the case)")
    sp = add("annual", "annual energy on the receiver")
    sp.add_argument("--method", choices=["sun_bins", "sun_grid", "representative_days", "hourly"], default=None)
    sp.add_argument("--aiming", choices=["center", "kfactor", "greedy", "milp"], default=None)
    add("optimize-layout", "optimise the field layout for constrained annual energy")
    sp = add("uniformity", "power vs uniformity (CV) curve and edge-band bound at one instant")
    sp.add_argument("--time", required=True)
    sp.add_argument("--eps", type=float, nargs="*", default=None, help="CV targets (default: around the greedy CV)")
    sp.add_argument("--exact", action="store_true", help="also solve the exact MISOCP with SCIP (small problems)")
    sp = add("validate", "Monte-Carlo ray tracing check of the analytic model")
    sp.add_argument("--time", required=True)
    sp.add_argument("--rays", type=int, default=None, help="rays per heliostat")
    sp.add_argument("--seeds", type=int, default=None, help="independent realisations")
    sp.add_argument("--method", choices=["center", "kfactor", "greedy"], default="center")
    sp.add_argument("--convergence", type=int, nargs="*", default=None, help="ray counts for a convergence table")
    sp = add("soltrace-export", "export a benchmark case + PySolTrace driver for a SolTrace comparison")
    sp.add_argument("--time", required=True)
    sp.add_argument("--method", choices=["center", "kfactor", "greedy"], default="center")
    sp.add_argument("--rays", type=int, default=None)
    sp.add_argument("--seeds", type=int, default=None)
    add("shading-check", "check shading/blocking screening against brute force")
    sp = add("sunbins", "representative sun positions for the annual evaluation")
    sp.add_argument("--bins", type=int, default=None)
    sp.add_argument("--check", action="store_true", help="annual-energy error of 20/30/40 bins and the grid vs hourly")
    sp = sub.add_parser("engine", help="online control engine as a JSON-lines service on stdin/stdout")
    sp.add_argument("case")
    sp = sub.add_parser("gui", help="open the graphical interface in the browser")
    sp.add_argument("case", nargs="?", default=None)
    sp.add_argument("--port", type=int, default=8501)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        app = Path(__file__).with_name("gui_app.py")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--"]
        if args.case:
            cmd += ["--case", str(Path(args.case).resolve())]
        return subprocess.call(cmd)
    case = load_case(args.case)
    if args.command == "engine":
        wf.engine_service(case, sys.stdin, lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush()))
        return 0
    out = args.out
    if args.command == "layout":
        res = wf.task_layout(case, out)
    elif args.command in ("snapshot", "aps"):
        method = args.method or ("center" if args.command == "snapshot" else case.aps.method)
        res = wf.task_aim(case, out, args.time, method=method, dni=args.dni, dni_map=args.dni_map,
                          forecast_dni_map=args.forecast_dni_map, prev_aims=args.prev_aims, tag=args.command)
    elif args.command == "realtime":
        res = wf.task_realtime(case, out, progress=_progress)
    elif args.command == "annual":
        res = wf.task_annual(case, out, method=args.method, aiming=args.aiming, progress=_progress)
    elif args.command == "optimize-layout":
        res = wf.task_optimize_layout(case, out, progress=_progress)
    elif args.command == "uniformity":
        res = wf.task_uniformity(case, out, args.time, eps_values=args.eps, exact=args.exact)
    elif args.command == "validate":
        res = wf.task_validate(case, out, args.time, rays=args.rays, method=args.method, seeds=args.seeds,
                               convergence_rays=args.convergence)
    elif args.command == "soltrace-export":
        res = wf.task_soltrace_export(case, out, args.time, args.method, args.rays, args.seeds)
    elif args.command == "shading-check":
        res = wf.task_shading_check(case, out)
    elif args.command == "sunbins":
        res = wf.task_sunbins(case, out, n_bins=args.bins, check=args.check)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return _print(res)


if __name__ == "__main__":
    raise SystemExit(main())
