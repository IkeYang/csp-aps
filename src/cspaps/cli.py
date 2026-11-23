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

BAD_STATUS = {"infeasible", "uniformity_infeasible", "capacity_shortfall", "constraint_violation", "missed_losses",
              "incomplete", "proven_infeasible", "no_feasible_found"}


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
        sp.add_argument("--method", choices=["center", "kfactor", "greedy", "milp", "subfield"], default=None)
        sp.add_argument("--dni", type=float, default=None, help="uniform DNI override (W/m2)")
        sp.add_argument("--dni-map", default=None, help="CSV x,y,dni: measured DNI at the heliostats")
        sp.add_argument("--forecast-dni-map", default=None, help="CSV x,y,dni: highest DNI expected before the next update")
        sp.add_argument("--prev-aims", default=None, help="aims CSV of the last executed plan (matched by id)")
        if name == "aps":
            sp.add_argument("--mode", choices=["fast", "standard", "deep"], default=None,
                            help="depth: fast (one greedy pass) | standard (joint poses + rounding + gap) | "
                                 "deep (denser candidates, more rounds); see `cspaps advise`")
            sp.add_argument("--no-certify", action="store_true",
                            help="skip the optimality certificate (upper bound and relaxation rounding)")
            sp.add_argument("--weight", type=float, default=None,
                            help="uniformity weight lambda: maximise P_abs - lambda * A_H * std_H (enables uniformity)")
            sp.add_argument("--max-cv", type=float, default=None,
                            help="maximise P_abs subject to CV_2D <= this value (enables uniformity)")
    sp = add("aps-refine", "problem 1 on nested, finer candidate grids: convergence of the certified optimum")
    sp.add_argument("--time", required=True)
    sp.add_argument("--azimuth-span", type=float, help="maximum azimuth offset in either direction (deg)")
    sp.add_argument("--dni-map", help="CSV x,y,dni for the fixed field")
    sp.add_argument("--weight", type=float, help="uniformity weight for all candidate grids")
    sp.add_argument("--levels", default="3x7,5x13,9x25", help="azimuth x height candidates per level, e.g. 3x7,5x13")
    sp = add("pareto", "power vs uniformity trade-off over the weight lambda (with optimality gaps)")
    sp.add_argument("--time", required=True)
    sp.add_argument("--lambdas", type=float, nargs="*", default=None)
    sp.add_argument("--dni-map", default=None)
    sp.add_argument("--no-certify", action="store_true")
    add("realtime", "replay real-time APS under cloud shadows (scenario from the case)")
    sp = add("annual", "annual energy on the receiver")
    sp.add_argument("--method", choices=["sun_bins", "sun_grid", "representative_days", "hourly"], default=None)
    sp.add_argument("--aiming", choices=["center", "kfactor", "greedy", "milp", "subfield"], default=None)
    sp = add("optimize-layout", "optimise the field layout for constrained annual energy")
    sp.add_argument("--mode", choices=["fast", "standard", "deep"], default=None,
                    help="depth of the search and the formal ranking (see `cspaps advise --task search`)")
    sp.add_argument("--uniformity-weight", type=float, default=None,
                    help="lambda_L: weight of the circumferential receiver non-uniformity (0 = energy only)")
    sp = add("layout-tradeoff", "annual energy vs circumferential uniformity of the layout over lambda_L")
    sp.add_argument("--lambdas", type=float, nargs="*", default=None)
    sp = add("scenarios", "build and save the annual scenario set (sun, hours, spatial DNI map per scenario)")
    sp.add_argument("--n", type=int, default=None, help="number of scenarios (default scenarios.n_scenarios)")
    sp = add("scenario-aps", "task 1 for one scenario of a scenario set on one layout (independent score check)")
    sp.add_argument("--id", required=True, help="scenario ID, e.g. S03")
    sp.add_argument("--scenarios", default=None, help="saved scenario-set folder (default: build from the case)")
    sp.add_argument("--layout", default=None, help="layout CSV id,x,y[,z] or 'initial' (default: the case field)")
    sp = add("layout-value", "rank layouts by V(L) = sum_s h_s (P_s - lambda_L U_s) with per-scenario APS")
    sp.add_argument("--mode", choices=["fast", "standard", "deep"], default=None,
                    help="depth of every scenario solve and of the scenario set (see `cspaps advise --task compare`)")
    sp.add_argument("--layouts", nargs="*", default=None, help="layout CSV files (and/or 'initial')")
    sp.add_argument("--scenarios", default=None, help="saved scenario-set folder (default: build from the case)")
    sp.add_argument("--lambda-layout", type=float, default=None, help="outer weight lambda_L (default layout_opt)")
    sp = add("validation-suite", "ray-trace checks: low sun, cloud edge, high-load mixed aiming, near the flux limit")
    sp.add_argument("--states", nargs="*", default=None,
                    choices=["low_sun", "cloud_edge", "mixed_high_load", "near_limit"])
    sp.add_argument("--rays", type=int, default=None, help="MCRT rays per heliostat (default mcrt.rays_per_heliostat)")
    sp.add_argument("--seeds", type=int, default=None, help="independent MCRT seeds (default mcrt.n_seeds)")
    sp.add_argument("--no-soltrace", action="store_true", help="skip the SolTrace export of every plan")
    sp = add("solver-benchmark", "solvers under the same budget: task-1 methods (inner) or layout searches (outer)")
    sp.add_argument("--part", choices=["inner", "outer", "both"], default="inner")
    sp.add_argument("--states", type=int, default=6, help="inner: number of scenarios (largest h x DNI first)")
    sp.add_argument("--budget", type=float, default=60.0, help="wall-time budget per method (and seed), s")
    sp.add_argument("--seeds", type=int, nargs="*", default=[1, 2], help="outer: random seeds")
    sp.add_argument("--limit-factor", type=float, default=None, help="scale every flux limit (make them bind)")
    sp.add_argument("--weight", type=float, default=None, help="inner lambda_APS (default: the case's)")
    sp = sub.add_parser("replay-layout", help="re-solve a saved layout comparison with its frozen inputs")
    sp.add_argument("folder")
    sp.add_argument("--out", default="results")
    sp = add("advise", "which algorithm depth fits the field size and the time budget (fast / standard / deep)")
    sp.add_argument("--task", choices=["aim", "compare", "search"], default="aim")
    sp.add_argument("--budget-min", type=float, default=15.0, help="time budget per run, minutes")
    sp.add_argument("--workers", type=int, default=None, help="parallel workers (default: all cores but one)")
    sp.add_argument("--heliostats", type=int, default=None, help="what-if: number of heliostats instead of the case's")
    sp = add("calibrate", "measure the time model of the algorithm depths on this computer")
    sp.add_argument("--sizes", type=int, nargs="+", default=[400, 1200], help="field sizes to time (heliostats)")
    sp = sub.add_parser("rescore", help="recompute V(L) from exported scenario_values.csv files for another lambda_L")
    sp.add_argument("folder")
    sp.add_argument("--lambda-layout", type=float, required=True)
    sp = add("cloud-compare", "one instant before / after a cloud: keep the plan vs re-solve APS")
    sp.add_argument("--time", required=True)
    sp.add_argument("--clouds", type=int, default=2, help="number of cloud shadows")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--radius", type=float, nargs=2, default=[60.0, 140.0])
    sp.add_argument("--transmittance", type=float, nargs=2, default=[0.1, 0.35])
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
    sp = add("fetch-weather", "download hourly DNI for the case's site from NASA POWER (writes CSV + case copy)")
    sp.add_argument("--year", type=int, default=None, help="calendar year (default: annual.year)")
    sp = sub.add_parser("engine", help="online control engine as a JSON-lines service on stdin/stdout")
    sp.add_argument("case")
    sp = sub.add_parser("gui", help="open the graphical interface in the browser")
    sp.add_argument("case", nargs="?", default=None)
    sp.add_argument("--port", type=int, help="browser port (auto-select a free port when omitted)")
    return p


def _safe_console() -> None:
    """Never fail on printing: a Windows console or a redirected stdout may use a code page (cp936, cp1252)
    that cannot encode every character of a result (e.g. Chinese labels, superscripts, Greek letters)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # replaced streams (pytest capture, IDE consoles)
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        app = Path(__file__).with_name("gui_app.py")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app),
               "--theme.base", "light", "--theme.primaryColor", "#14806f", "--theme.backgroundColor", "#f4f6f6",
               "--theme.secondaryBackgroundColor", "#eef2f1", "--theme.textColor", "#16232b",
               "--browser.gatherUsageStats", "false"]
        if args.port is not None:
            cmd += ["--server.port", str(args.port)]
        cmd.append("--")
        if args.case:
            cmd += ["--case", str(Path(args.case).resolve())]
        return subprocess.call(cmd)
    if args.command == "replay-layout":
        return _print(wf.replay_layout_value(args.folder, args.out, progress=_progress))
    if args.command == "rescore":
        res = wf.task_rescore(args.folder, args.lambda_layout)
        print(json.dumps(res["summary"], indent=2, ensure_ascii=False, default=str))
        return 0
    case = load_case(args.case)
    if args.command == "engine":
        wf.engine_service(case, sys.stdin, lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush()))
        return 0
    out = args.out
    if getattr(args, "mode", None):
        from .advisor import apply_mode

        from .advisor import _field_size

        task = {"aps": "aim", "layout-value": "compare", "optimize-layout": "search"}[args.command]
        case = apply_mode(case, task, args.mode, n=_field_size(case, task))
        _progress(f"depth: {args.mode} ({task})")
    if args.command == "advise":
        from .advisor import recommend

        rec = recommend(case, args.task, 60.0 * args.budget_min, workers=args.workers, n=args.heliostats)
        print(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.command == "calibrate":
        from .advisor import calibrate, calibration_path

        cal = calibrate(case, sizes=args.sizes, progress=_progress)
        print(json.dumps({"written": str(calibration_path()), "aim": cal.aim, "inner": cal.inner,
                          "prescreen": cal.prescreen, "measured": cal.measured}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "layout":
        res = wf.task_layout(case, out)
    elif args.command in ("snapshot", "aps"):
        method = args.method or ("center" if args.command == "snapshot" else case.aps.method)
        certify = None
        if args.command == "aps":
            if args.weight is not None:
                case.uniformity.enabled, case.uniformity.mode, case.uniformity.weight = True, "weighted", args.weight
            if args.max_cv is not None:
                case.uniformity.enabled, case.uniformity.mode, case.uniformity.max_cv = True, "cv_limit", args.max_cv
            certify = False if args.no_certify else None
        res = wf.task_aim(case, out, args.time, method=method, dni=args.dni, dni_map=args.dni_map,
                          forecast_dni_map=args.forecast_dni_map, prev_aims=args.prev_aims, tag=args.command,
                          certify=certify, progress=_progress)
    elif args.command == "aps-refine":
        levels = [tuple(int(v) for v in lv.lower().split("x")) for lv in args.levels.split(",") if lv.strip()]
        if args.weight is not None:
            if not 0 <= args.weight < float("inf"):
                raise ValueError("weight must be finite and nonnegative")
            case.uniformity.enabled, case.uniformity.mode, case.uniformity.weight = True, "weighted", args.weight
        res = wf.task_aps_refinement(case, out, args.time, levels=levels, progress=_progress,
                                     azimuth_span=args.azimuth_span, dni_map=args.dni_map)
    elif args.command == "pareto":
        res = wf.task_pareto(case, out, args.time, lambdas=args.lambdas, dni_map=args.dni_map,
                             certify=not args.no_certify, progress=_progress)
    elif args.command == "realtime":
        res = wf.task_realtime(case, out, progress=_progress)
    elif args.command == "annual":
        res = wf.task_annual(case, out, method=args.method, aiming=args.aiming, progress=_progress)
    elif args.command == "optimize-layout":
        if args.uniformity_weight is not None:
            case.layout_opt.uniformity_weight = args.uniformity_weight
        res = wf.task_optimize_layout(case, out, progress=_progress)
    elif args.command == "fetch-weather":
        res = wf.task_fetch_weather(case, out, year=args.year)
    elif args.command == "layout-tradeoff":
        res = wf.task_layout_tradeoff(case, out, lambdas=args.lambdas, progress=_progress)
    elif args.command == "scenarios":
        res = wf.task_scenarios(case, out, n=args.n)
    elif args.command == "scenario-aps":
        res = wf.task_scenario_aps(case, out, args.id, scenarios_dir=args.scenarios, layout=args.layout)
    elif args.command == "layout-value":
        names = args.layouts or ["initial"]
        layouts = {("initial" if v == "initial" else Path(v).stem): v for v in names}
        if len(layouts) != len(names):
            raise ValueError("布局文件名重复，请为待比较的布局使用不同的文件名。")
        res = wf.task_layout_value(case, out, layouts, scenarios_dir=args.scenarios, lam_layout=args.lambda_layout,
                                   progress=_progress)
    elif args.command == "cloud-compare":
        res = wf.task_cloud_compare(case, out, args.time, n_clouds=args.clouds, seed=args.seed,
                                    radius_m=tuple(args.radius), transmittance=tuple(args.transmittance),
                                    progress=_progress)
    elif args.command == "uniformity":
        res = wf.task_uniformity(case, out, args.time, eps_values=args.eps, exact=args.exact)
    elif args.command == "validate":
        res = wf.task_validate(case, out, args.time, rays=args.rays, method=args.method, seeds=args.seeds,
                               convergence_rays=args.convergence)
    elif args.command == "soltrace-export":
        res = wf.task_soltrace_export(case, out, args.time, args.method, args.rays, args.seeds)
    elif args.command == "shading-check":
        res = wf.task_shading_check(case, out)
    elif args.command == "validation-suite":
        res = wf.task_validation_suite(case, out, keys=args.states, rays=args.rays, seeds=args.seeds,
                                       soltrace=not args.no_soltrace, progress=_progress)
    elif args.command == "solver-benchmark":
        res = wf.task_solver_benchmark(case, out, args.part, args.states, args.budget, tuple(args.seeds),
                                       limit_factor=args.limit_factor, lam_aps=args.weight, progress=_progress)
    elif args.command == "sunbins":
        res = wf.task_sunbins(case, out, n_bins=args.bins, check=args.check)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return _print(res)


if __name__ == "__main__":
    raise SystemExit(main())
