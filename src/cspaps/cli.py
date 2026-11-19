"""Command line interface: ``cspaps <command> CASE.yaml [options]``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import workflows as wf
from .config import load_case


def _print(result: dict) -> None:
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    for label, path in result["files"].items():
        print(f"  {label}: {path}")


def _progress(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cspaps", description="Heliostat field layout & aim-point strategy optimisation")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("case", help="case YAML file")
        sp.add_argument("--out", default="results", help="output folder (default: results)")
        return sp

    add("layout", "generate the field layout (CSV + plot)")

    for name, help_ in [("snapshot", "one instant: efficiency breakdown + flux map (default: centre aiming)"),
                        ("aps", "real-time control engine: optimise aim points for one instant")]:
        sp = add(name, help_)
        sp.add_argument("--time", required=True, help='local time, e.g. "2025-06-21 13:40"')
        sp.add_argument("--method", choices=["center", "kfactor", "greedy", "milp"], default=None)
        sp.add_argument("--dni", type=float, default=None, help="uniform DNI override (W/m2)")
        sp.add_argument("--dni-map", default=None, help="CSV x,y,dni: measured DNI at the heliostats (clouds)")
        sp.add_argument("--forecast-dni-map", default=None, help="CSV x,y,dni: highest DNI expected before the next update")
        sp.add_argument("--prev-aims", default=None, help="aims CSV from the previous update (smooth moves)")

    add("realtime", "simulate real-time APS under moving cloud shadows")
    sp = add("annual", "annual energy on the receiver")
    sp.add_argument("--method", choices=["sun_bins", "representative_days", "hourly"], default=None)
    sp.add_argument("--aiming", choices=["center", "kfactor", "greedy", "milp"], default=None)
    add("optimize-layout", "optimise the field layout for annual energy")
    sp = add("validate", "Monte-Carlo ray tracing check of the analytic model")
    sp.add_argument("--time", required=True)
    sp.add_argument("--rays", type=int, default=None, help="rays per heliostat")
    sp.add_argument("--method", choices=["center", "kfactor", "greedy"], default="center")
    sp = add("sunbins", "representative sun positions for the annual evaluation")
    sp.add_argument("--bins", type=int, default=None)
    sp.add_argument("--check", action="store_true", help="compare with the full hourly year")
    sp = sub.add_parser("gui", help="open the graphical interface in the browser")
    sp.add_argument("case", nargs="?", default=None)
    sp.add_argument("--port", type=int, default=8501)

    args = p.parse_args(argv)
    if args.command == "gui":
        app = Path(__file__).with_name("gui_app.py")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--"]
        if args.case:
            cmd += ["--case", str(Path(args.case).resolve())]
        return subprocess.call(cmd)

    case = load_case(args.case)
    out = wf.out_dir(args.out, case)
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
    elif args.command == "validate":
        res = wf.task_validate(case, out, args.time, rays=args.rays, method=args.method)
    elif args.command == "sunbins":
        res = wf.task_sunbins(case, out, n_bins=args.bins, check=args.check)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
