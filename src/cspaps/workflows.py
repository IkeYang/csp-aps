"""Task runners shared by the CLI and the GUI.

Every run gets its own folder ``<out>/<case>/<task>_<YYYYmmdd-HHMMSS>/`` with
``inputs_case.yaml`` (the exact inputs), ``run.json`` (task, arguments,
version, status, timings) and all outputs, so earlier results can be reviewed.
Each runner returns ``{"summary": dict, "files": {label: path}, "run_dir": Path, ...}``.
"""

from __future__ import annotations

import itertools
import json
import shutil
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__, sun
from .tables import export_name, safe_filename
from .aps import build_problem, optimize_aims, solve, solve_greedy
from .certify import improve_and_certify
from .clouds import load_dni_map
from .config import Case, load_case
from .coupled import search_joint_poses
from .engine import APSEngine, request_from_dict, response_to_dict
from .field import Field, build_field, generate, save_layout_csv
from .layout_opt import design_spacing, optimize_layout, validate_layout
from .mcrt import compare, convergence, trace_repeated
from .plotting import (
    plot_annual_monthly, plot_flux_compare, plot_flux_map, plot_history, plot_layout,
    plot_realtime, plot_sun_bins, plot_uniformity_curve,
)
from .realtime import run_realtime
from .receiver import build_grid, surface_point
from .shading import compare_screening
from .simulate import LOSSES, annual, evaluate, sun_state
from .uniformity import edge_band_bound, region_of_grid, sweep, uniformity_metrics

Progress = Callable[[str], None]


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _json(path: Path, data) -> Path:
    path.write_text(json.dumps(_clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def _clean(d):
    return json.loads(json.dumps(d, default=_default), parse_constant=lambda _: None)


def out_dir(base: str | Path, case: Case, task: str = "run") -> Path:
    """A new run folder; the case inputs are saved in it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(base) / safe_filename(case.name)
    root.mkdir(parents=True, exist_ok=True)
    for k in itertools.count(1):
        out = root / (f"{task}_{stamp}" if k == 1 else f"{task}_{stamp}-{k}")
        try:
            out.mkdir()
            break
        except FileExistsError:
            continue
    snapshot = case.copy()
    snapshot.base_dir = out
    for section, attr in (("weather", "file"), ("field", "layout_file"),
                          ("scenarios", "file"), ("scenarios", "map_file"), ("scenarios", "validation_file")):
        value = getattr(getattr(case, section), attr)
        if not value:
            continue
        source = case.resolve(value)
        if source.is_file():
            dest = out / f"input_{section}_{attr}_{source.name}" if section == "scenarios" else out / f"input_{section}_{source.name}"
            shutil.copy2(source, dest)
            setattr(getattr(snapshot, section), attr, dest.name)
        else:
            setattr(getattr(snapshot, section), attr, str(source.resolve()))
    (out / "inputs_case.yaml").write_text(snapshot.to_yaml(), encoding="utf-8")
    return out


def finish(out: Path, task: str, summary: dict, args: dict | None = None, t0: float | None = None) -> None:
    status = summary.get("status") or (summary.get("aps") or {}).get("status") or "done"
    _json(out / "run.json", {"task": task, "version": __version__, "finished": datetime.now(),
                             "status": status, "arguments": args or {},
                             "runtime_s": round(_time.perf_counter() - t0, 3) if t0 else None})


def list_runs(base: str | Path, case_name: str | None = None) -> pd.DataFrame:
    """Earlier runs found under ``base`` (newest first)."""
    rows = []
    for rj in Path(base).glob(f"{case_name or '*'}/*/run.json"):
        try:
            info = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({"case": rj.parent.parent.name, "run": rj.parent.name, "task": info.get("task"),
                     "status": info.get("status"), "finished": info.get("finished"), "path": str(rj.parent)})
    df = pd.DataFrame(rows, columns=["case", "run", "task", "status", "finished", "path"])
    return df.sort_values("finished", ascending=False).reset_index(drop=True)


def write_aims_csv(path: Path, fld: Field, case: Case, aim, active, extra: dict | None = None) -> Path:
    xyz = surface_point(case.receiver, aim[0], aim[1])
    df = pd.DataFrame({
        "id": fld.ids, "x": fld.xyz[:, 0], "y": fld.xyz[:, 1], "z": fld.xyz[:, 2],
        "active": np.asarray(active, dtype=int), "aim_azimuth_deg": np.round(aim[0], 6),
        "aim_z": np.round(aim[1], 6), "aim_x": np.round(xyz[:, 0], 6), "aim_y": np.round(xyz[:, 1], 6),
    })
    for k, v in (extra or {}).items():
        df[k] = v
    df.to_csv(path, index=False)
    return path


def read_aims_csv(path, fld: Field | None = None) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Aims from a CSV; rows are matched to the field by ``id`` when both are available."""
    df = pd.read_csv(path)
    if fld is not None and "id" in df:
        df = df.set_index("id").loc[fld.ids].reset_index()
    active = df["active"].to_numpy(bool) if "active" in df else np.ones(len(df), bool)
    return (df["aim_azimuth_deg"].to_numpy(float), df["aim_z"].to_numpy(float)), active


def _dropped_note(fld: Field) -> dict:
    return {"n_dropped": len(fld.dropped), "dropped": fld.dropped[:20]} if fld.dropped else {}


# --------------------------------------------------------------------------- tasks

def task_layout(case: Case, base: str | Path) -> dict:
    t0 = _time.perf_counter()
    out = out_dir(base, case, "layout")
    fld = build_field(case)
    files = {"layout_csv": out / "layout.csv", "layout_png": out / "layout.png"}
    save_layout_csv(fld, files["layout_csv"])
    plot_layout(fld, case.receiver, files["layout_png"], boundary=case.field.boundary,
                exclusion=case.field.exclusion_radius)
    summary = {"n_heliostats": fld.n, "mirror_area_m2": round(fld.n * case.heliostat.area, 1), **_dropped_note(fld)}
    _json(out / "summary.json", summary)
    finish(out, "layout", summary, t0=t0)
    return {"summary": summary, "files": files, "field": fld, "run_dir": out}


def task_aim(
    case: Case,
    base: str | Path,
    time,
    method: str | None = None,
    dni: float | None = None,
    dni_map: str | Path | None = None,
    forecast_dni_map: str | Path | None = None,
    prev_aims: str | Path | None = None,
    tag: str = "aps",
    certify: bool | None = None,
    progress: Progress | None = None,
    dni_source: str = "site",
) -> dict:
    """Single instant: optimise aim points and evaluate them (status = verified flux check).

    With ``certify`` (default ``aps.certify``) the plan is improved by relaxation
    rounding and an upper bound on the optimum is attached (``summary["certificate"]``).
    """
    t_start = _time.perf_counter()
    fld = build_field(case)
    if not fld.n or fld.dropped:
        raise ValueError(f"Fixed field needs all supplied coordinates inside the site: {fld.n} accepted, "
                         f"{len(fld.dropped)} excluded. Check boundary and tower exclusion radius.")
    out = out_dir(base, case, tag)
    def save_input(path, label):
        if path is None:
            return None
        dest = out / f"input_{label}_{Path(path).name}"
        shutil.copyfile(path, dest)
        return str(dest.resolve())

    dni_map = save_input(dni_map, "dni")
    forecast_dni_map = save_input(forecast_dni_map, "forecast")
    prev_aims = save_input(prev_aims, "previous")
    method = method or case.aps.method
    ts, el, az, dni_model = sun_state(case, time)
    dni_v = load_dni_map(dni_map, fld.xyz) if dni_map else (dni_model if dni is None else float(dni))
    if not np.isfinite(dni_v).all() or np.any(np.asarray(dni_v) < 0):
        raise ValueError("DNI must be finite and nonnegative")
    if dni_source == "cloud" and not dni_map:
        from .clouds import cloud_dni_snapshot

        dni_v = cloud_dni_snapshot(case, fld, ts, float(np.mean(dni_v)))
    dni_con = load_dni_map(forecast_dni_map, fld.xyz) if forecast_dni_map else None
    prev, prev_active = read_aims_csv(prev_aims, fld) if prev_aims else (None, None)
    grid = build_grid(case.receiver)
    cert = None
    do_certify = case.aps.certify if certify is None else certify
    if method == "center":
        from .receiver import default_aim

        aim, active, res = default_aim(case.receiver, fld.xyz), np.ones(fld.n, bool), None
    else:
        t_solve = _time.perf_counter()
        prob = build_problem(case, fld, el, az, dni_v, dni_con, grid=grid)
        res = solve(prob, method, case.aps, prev, prev_active)
        proposal_check = res.check
        if do_certify and method in ("greedy", "kfactor", "milp", "subfield") and fld.n:
            if progress:
                progress("certifying: upper bound and relaxation rounding")
            initial = res.choice.copy()
            if (prob.k + 1) ** min(prob.n, 20) > case.aps.coupled_exact_limit and np.any(prob.dni_obj > 0):
                res, _ = improve_and_certify(prob, res, case.aps, rounds=case.aps.certify_rounds,
                                             time_limit_s=case.aps.certify_time_s, prev=prev, prev_active=prev_active)
            res, cert = search_joint_poses(prob, fld, res, warm=initial,
                                           tracking_reference=prob.tracking_reference(prev, prev_active))
        else:
            res.info["global_optimality_proven"] = False
            res.info["solution_class"] = "joint_pose_verification"
        res.solve_time_s = _time.perf_counter() - t_solve
        aim, active = res.aim, res.active
    snap = evaluate(case, fld, el, az, dni_v, aim=aim, active=active, grid=grid, time=ts)
    # Final check of the plan as it will be executed: the final aim points with their own shading and
    # blocking (re-evaluated, not the solver's response model), full beams, fine grid, constraint DNI.
    if dni_con is not None:
        dni_check = np.maximum(np.broadcast_to(np.asarray(dni_con, dtype=float), (fld.n,)), snap.dni)
        snap_con = evaluate(case, fld, el, az, dni_check, aim=aim, active=active, grid=grid, time=ts)
    else:
        snap_con = snap
    chk_con = snap_con.verify(case.receiver)
    status = {"ok": "ok", "margin_reduced": "margin_reduced", "over_limit": "infeasible"}[chk_con.status]
    if res is not None and status == "infeasible" and res.info.get("solution_class") != "infeasible_after_verification":
        res.info["solution_class"] = "infeasible_after_verification"
        res.info["global_optimality_proven"] = False
    thermal = snap.thermal_objective(case.uniformity) if cert is None else None
    if res is not None and thermal is not None:
        u = case.uniformity
        metrics = uniformity_metrics(snap.absorbed_flux, snap.grid.to_map(snap.region))
        res.info.update(uniformity_met=thermal["uniformity_met"], cv_2d_H=thermal["cv_2d"],
                        cv_circ_H=metrics["cv_circ"], cv_height_H=metrics["cv_height"],
                        power_H_mw=thermal["power_h_w"] / 1e6, objective_absorbed_power_w=snap.p_absorbed)
        if u.enabled:
            res.info["uniformity_status"] = ("weighted" if thermal["uniformity_met"] and u.mode == "weighted"
                                             else "feasible" if thermal["uniformity_met"] else "not_reached")
    uniform_ok = thermal["uniformity_met"] if thermal is not None else res.info["uniformity_met"]
    if not uniform_ok:
        status = "uniformity_infeasible"
    if cert is not None and cert["optimum_exists"] is not True and status in ("ok", "margin_reduced"):
        status = "infeasible"
    stamp = f"{ts:%Y%m%d_%H%M}_{method}"
    files = {
        "summary_json": out / f"{tag}_{stamp}.json",
        "aims_csv": out / f"{tag}_{stamp}_aims.csv",
        "flux_png": out / f"{tag}_{stamp}_flux.png",
        "efficiency_png": out / f"{tag}_{stamp}_efficiency.png",
    }
    extra = {f"eta_{k}": np.round(snap.eta[k], 6) for k in LOSSES}
    extra["dni_w_m2"] = np.round(snap.dni, 2)
    extra["tower_shadow_fraction"] = snap.tower_fraction
    extra["p_receiver_kw"] = np.round(snap.power["intercept"] / 1e3, 4)
    write_aims_csv(files["aims_csv"], fld, case, aim, active, extra)
    check_summary = chk_con.summary()
    check_summary["top_contributors"] = [{"heliostat_id": int(fld.ids[c["heliostat"]]), "flux_kw_m2": c["flux_kw_m2"]}
                                         for c in check_summary["top_contributors"]]
    summary = {"status": status, "aps": res.summary() if res is not None else {"method": "center"},
               "flux_check": check_summary, **snap.summary(), **_dropped_note(fld)}
    margin = case.aps.safety_margin
    summary["margin_check"] = {"allowed_fraction": 1.0 - margin, "peak_fraction": chk_con.peak_ratio,
                               "within_margin": chk_con.status == "ok", "within_limit": chk_con.status != "over_limit",
                               "hotspot_panel": int(chk_con.hotspot_panel),
                               "hotspot_azimuth_deg": chk_con.hotspot_azimuth_deg, "hotspot_z_m": chk_con.hotspot_z}
    if method != "center":
        # reference for the result sentence: the same instant and DNI with every mirror on the centre point
        from .receiver import default_aim

        snap_c = evaluate(case, fld, el, az, dni_v, aim=default_aim(case.receiver, fld.xyz), grid=grid, time=ts)
        chk_c = snap_c.verify(case.receiver)
        summary["center_reference"] = {"absorbed_power_mw": snap_c.p_absorbed / 1e6,
                                       "peak_flux_kw_m2": chk_c.peak_flux_kw_m2, "peak_fraction": chk_c.peak_ratio,
                                       "status": chk_c.status}
    summary["flux_check"]["basis"] = ("final aim points, shading and blocking re-evaluated for them, full beams, "
                                      "fine grid, constraint DNI")
    if res is not None and proposal_check is not None:
        summary["solver_flux_check"] = {"status": proposal_check.status, "peak_flux_kw_m2": round(proposal_check.peak_flux_kw_m2, 2),
                                        "basis": "solver response model (candidate aims)"}
    summary["n_tracking_in_tower_shadow"] = int(np.sum(active & (snap.tower_fraction == 1.)))
    summary["aps"].pop("flux_check", None)
    summary["aps"].pop("certificate", None)
    if cert is not None:
        summary["certificate"] = {k: v for k, v in cert.items() if k != "history"}
        summary["existence"] = existence_statement_aps(case, cert)
        files["bound_csv"] = out / f"{tag}_{stamp}_bound_history.csv"
        pd.DataFrame(cert["history"]).to_csv(files["bound_csv"], index=False)
    _json(files["summary_json"], summary)
    plot_flux_map(snap.flux, grid, case.receiver, files["flux_png"], aims=aim, hotspot=chk_con,
                  title=f"{ts:%Y-%m-%d %H:%M}  {method}  [{status}]  verified peak {chk_con.peak_flux_kw_m2:.0f} kW/m², "
                        f"{snap.p_receiver / 1e6:.2f} MW intercepted")
    plot_layout(fld, case.receiver, files["efficiency_png"], values=snap.eta_total,
                label="Optical efficiency", boundary=case.field.boundary,
                title=f"Heliostat optical efficiency, {ts:%Y-%m-%d %H:%M}")
    finish(out, tag, summary, {"time": str(time), "method": method, "dni": dni, "dni_map": dni_map,
                               "forecast_dni_map": forecast_dni_map, "prev_aims": prev_aims,
                               "dni_source": dni_source, "certify": cert is not None}, t_start)
    return {"summary": _clean(summary), "files": files, "snapshot": snap, "aim_result": res, "field": fld,
            "check": chk_con, "run_dir": out, "certificate": cert}


def existence_statement_aps(case: Case, cert: dict) -> dict:
    """Plain statement of what is proven about problem 1 for this instant."""
    if "optimum_exists" in cert:
        exists = cert["optimum_exists"]
        statement = {True: "Verified feasible assignment in a finite set of joint aim poses.",
                     False: "No admissible assignment meets all constraints under this model.",
                     None: "Feasibility is unresolved within the joint-pose search budget."}
        return {"optimum_exists": exists, "statement": statement[exists],
                "found_mw": cert["objective_found_mw"], "upper_bound_mw": cert["upper_bound_mw"], "gap": cert["gap"]}
    mode = cert["mode"]
    if cert["status"] == "certified_infeasible":
        return {"optimum_exists": False, "statement": "no plan meets the CV target and the minimum load "
                "under the flux limits (the Lagrangian bound is negative)"}
    statement = ("finite candidate set and the all-defocused plan meets every flux limit, so an optimum exists"
                 if mode in ("power", "weighted") else
                 "finite candidate set; an optimum exists because a plan meeting the CV target was found")
    if mode == "cv_limit" and cert.get("objective_found_mw") is None:
        return {"optimum_exists": None, "statement": "no plan meeting the CV target was found; the bound did not "
                "prove infeasibility either"}
    return {"optimum_exists": True, "statement": statement,
            "found_mw": cert.get("objective_found_mw"), "upper_bound_mw": cert.get("upper_bound_mw"),
            "gap": cert.get("gap")}


def task_aps_refinement(case: Case, base: str | Path, time, levels=None, dni_map: str | Path | None = None,
                        dni_source: str = "site", progress: Progress | None = None, dni: float | None = None,
                        forecast_dni_map: str | Path | None = None, azimuth_span: float | None = None) -> dict:
    """Problem 1 on nested, increasingly fine candidate grids: how the certified optimum converges.

    The certificate of ``task_aim`` holds for the discrete candidate set. Real aim
    points are continuous (or limited by the controller resolution), so this task
    re-solves on nested grids -- every level contains the previous one, azimuth and
    height spacing halve -- warm-starting from the previous plan (the found value can
    only rise) and reports the found objective and the upper bound per level.
    """
    t0 = _time.perf_counter()
    out = out_dir(base, case, "aps_refinement")
    fld = build_field(case)
    ts, el, az, dni_model = sun_state(case, time)
    dni_v = load_dni_map(dni_map, fld.xyz) if dni_map else (dni_model if dni is None else dni)
    forecast = load_dni_map(forecast_dni_map, fld.xyz) if forecast_dni_map else None
    if dni_source == "cloud" and not dni_map:
        from .clouds import cloud_dni_snapshot

        dni_v = cloud_dni_snapshot(case, fld, ts, float(dni_model))
    levels = levels or [(3, 7), (5, 13), (9, 25)]
    if any(a < 1 or z < 1 for a, z in levels):
        raise ValueError("Candidate grid dimensions must be positive")
    for coarse, fine in zip(levels, levels[1:]):
        for m, n in zip(coarse, fine):
            if n < m or (m > 1 and (n - 1) % (m - 1)) or (m == 1 and n % 2 == 0):
                raise ValueError("Refinement levels must retain every preceding aim point")
    off_max = max((abs(float(o)) for o in case.aps.azimuth_offsets_deg), default=0.0) if azimuth_span is None else azimuth_span
    if not np.isfinite(off_max) or not 0 <= off_max <= 90:
        raise ValueError("Azimuth span must be in [0, 90] degrees")
    from .layout_value import eval_grid_shape
    n_z_grid, n_az_grid = eval_grid_shape(case)
    grid = build_grid(case.receiver, n_az_grid, n_z_grid)
    rows, prev = [], None
    for j, (n_az, n_z) in enumerate(levels):
        if progress:
            progress(f"refinement level {j + 1}/{len(levels)}: {n_az} x {n_z} candidates")
        c = case.copy()
        c.aps.auto_grid = False
        c.aps.azimuth_offsets_deg = np.linspace(-off_max, off_max, n_az).tolist() if off_max and n_az > 1 else [0.]
        c.aps.n_height_levels = int(n_z)
        t_lvl = _time.perf_counter()
        prob = build_problem(c, fld, el, az, dni_v, forecast, grid=grid)
        init = None
        if prev is not None:  # nested grids: the previous plan's aims are candidates of this level
            d_az = (prob.cand_az - np.asarray(prev[0])[:, None] + 180.0) % 360.0 - 180.0
            dist = np.abs(d_az) * 1e3 + np.abs(prob.cand_z - np.asarray(prev[1])[:, None])
            init = np.where(prev[2], np.argmin(dist, axis=1), -1)
        res = solve(prob, case.aps.method if case.aps.method != "center" else "greedy", c.aps, init=init)
        budget = min(120.0, c.aps.certify_time_s * max(1.0, (prob.k / 21.0) ** 0.75))
        if (prob.k + int(c.aps.allow_defocus)) ** min(prob.n, 20) > c.aps.coupled_exact_limit:
            res, _ = improve_and_certify(prob, res, c.aps, rounds=c.aps.certify_rounds, time_limit_s=budget)
        res, cert = search_joint_poses(prob, fld, res, warm=init)
        prev = (res.aim[0], res.aim[1], res.active)
        rows.append({"n_azimuth": len(c.aps.azimuth_offsets_deg), "n_height": n_z, "n_candidates": int(prob.k),
                     "azimuth_step_deg": (2 * off_max / (n_az - 1)) if n_az > 1 else None,
                     "height_step_m": (c.receiver.height * c.aps.height_span / (n_z - 1)) if n_z > 1 else None,
                     "objective_found_mw": cert.get("objective_found_mw"), "upper_bound_mw": cert["upper_bound_mw"],
                     "gap": cert.get("gap"), "status": res.status,
                     "power_abs_mw": res.info.get("objective_absorbed_power_w", 0.0) / 1e6,
                     "cv_2d_H": res.info.get("cv_2d_H"), "peak_flux_kw_m2": res.check.peak_flux_kw_m2 if res.check else None,
                     "time_s": round(_time.perf_counter() - t_lvl, 1)})
    files = {"summary_json": out / "aps_refinement.json", "table_csv": out / "aps_refinement.csv"}
    pd.DataFrame(rows).to_csv(files["table_csv"], index=False)
    summary = {"time": ts.isoformat(), "levels": rows, "azimuth_span_deg": off_max,
               "evaluation_grid": [n_z_grid, n_az_grid],
               "note": "certificates hold for each discrete candidate set; nested refinement shows the convergence "
                       "towards continuous aim points"}
    _json(files["summary_json"], summary)
    finish(out, "aps_refinement", summary, {"time": str(time), "levels": levels}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}


def task_pareto(case: Case, base: str | Path, time, lambdas=None, dni_map: str | Path | None = None,
                certify: bool = True, progress: Progress | None = None, dni_source: str = "site") -> dict:
    """Power vs uniformity trade-off over the weight lambda at one instant (fixed field and DNI)."""
    t0 = _time.perf_counter()
    out = out_dir(base, case, "pareto")
    fld = build_field(case)
    ts, el, az, dni_model = sun_state(case, time)
    dni_v = load_dni_map(dni_map, fld.xyz) if dni_map else dni_model
    if dni_source == "cloud" and not dni_map:
        from .clouds import cloud_dni_snapshot

        dni_v = cloud_dni_snapshot(case, fld, ts, float(dni_model))
    grid = build_grid(case.receiver)
    base_case = case.copy()
    base_case.uniformity.enabled = False
    prob = build_problem(base_case, fld, el, az, dni_v, grid=grid)
    base_choice, _ = solve_greedy(prob, case.aps)
    prob.rs.case = case
    lambdas = list(lambdas) if lambdas is not None else [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows, previous = [], base_choice
    for j, lam in enumerate(sorted(lambdas)):
        if progress:
            progress(f"weight {j + 1}/{len(lambdas)}: lambda = {lam:g}")
        c = case.copy()
        c.uniformity.enabled, c.uniformity.mode, c.uniformity.weight = True, "weighted", lam
        prob.rs.case = c
        res = solve(prob, "greedy", c.aps, init=previous)
        if certify:
            res, _ = improve_and_certify(prob, res, c.aps, rounds=2, max_iter=200, time_limit_s=6.0)
        res, bound = search_joint_poses(prob, fld, res, warm=previous)
        previous = res.choice
        rows.append({"lambda": lam, "power_abs_mw": res.info["objective_absorbed_power_w"] / 1e6,
                     "power_h_mw": res.info["power_H_mw"], "cv_2d": res.info["cv_2d_H"],
                     "objective_mw": bound["objective_found_mw"], "status": res.status,
                     "peak_flux_kw_m2": res.check.peak_flux_kw_m2, "n_defocused": res.n_defocused,
                     "upper_bound_mw": bound["upper_bound_mw"] if certify else None,
                     "gap": bound["gap"] if certify else None})
    files = {"summary_json": out / "pareto.json", "curve_csv": out / "pareto_curve.csv"}
    pd.DataFrame(rows).to_csv(files["curve_csv"], index=False)
    summary = {"time": ts.isoformat(), "dni_mean_w_m2": float(np.mean(dni_v)), "n_heliostats": fld.n,
               "region_H": {"panels": case.uniformity.panels, "z_range": case.uniformity.z_range},
               "curve": rows}
    _json(files["summary_json"], summary)
    finish(out, "pareto", summary, {"time": str(time), "lambdas": lambdas, "dni_map": dni_map,
                                    "dni_source": dni_source}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}


def task_realtime(case: Case, base: str | Path, progress: Progress | None = None) -> dict:
    t0 = _time.perf_counter()
    out = out_dir(base, case, "realtime")
    fld = build_field(case)
    res = run_realtime(case, fld, progress=(lambda i, n: progress(f"replay step {i}/{n}")) if progress else None)
    files = {
        "summary_json": out / "realtime.json",
        "table_csv": out / "realtime_timeseries.csv",
        "timeseries_png": out / "realtime.png",
        "frames_npz": out / "realtime_frames.npz",
    }
    summary = {"scenario": case.realtime.scenario, "strategies": res.summary(),
               "engine": res.extra.get("engine_stats")}
    _json(files["summary_json"], summary)
    res.table.to_csv(files["table_csv"], index=False)
    plot_realtime(res.table, files["timeseries_png"])
    save_frames(files["frames_npz"], res, fld)
    for strat, snap in res.worst.items():
        p = out / f"realtime_worst_flux_{strat}.png"
        plot_flux_map(snap.flux, snap.grid, case.receiver, p, aims=(snap.aim_azimuth, snap.aim_z),
                      title=f"{strat}: worst moment {snap.time:%H:%M:%S}, coarse-grid peak {snap.flux.max() / 1e3:.0f} kW/m²")
        files[f"worst_flux_{strat}_png"] = p
    worst_t = res.table.loc[res.table["dni_min_w_m2"].idxmin(), "time"]
    cloud_png = out / "realtime_cloud_shadow.png"
    plot_layout(fld, case.receiver, cloud_png, values=res.clouds.transmittance(fld.xyz[:, :2], worst_t),
                label="Cloud transmittance", cmap="Blues_r", vmin=0, vmax=1.3, boundary=case.field.boundary,
                title=f"Cloud shadow at {pd.Timestamp(worst_t):%H:%M:%S}")
    files["cloud_png"] = cloud_png
    finish(out, "realtime", summary, t0=t0)
    return {"summary": _clean(summary), "files": files, "result": res, "field": fld, "run_dir": out}


def save_frames(path: Path, res, fld: Field) -> None:
    """Per-sample DNI, aims and flux maps of a replay (for the GUI timeline)."""
    arrays = {"ids": fld.ids, "xyz": fld.xyz, "times": np.array([str(f["time"]) for f in res.frames]),
              "dni": np.array([f["dni"] for f in res.frames])}
    for strat in res.frames[0]["plans"] if res.frames else []:
        arrays[f"{strat}__aim_azimuth"] = np.array([f["plans"][strat]["aim_azimuth"] for f in res.frames])
        arrays[f"{strat}__aim_z"] = np.array([f["plans"][strat]["aim_z"] for f in res.frames])
        arrays[f"{strat}__active"] = np.array([f["plans"][strat]["active"] for f in res.frames])
        arrays[f"{strat}__flux"] = np.array([f["plans"][strat]["flux"] for f in res.frames])
    np.savez_compressed(path, **arrays)


def task_annual(case: Case, base: str | Path, method: str | None = None, aiming: str | None = None,
                progress: Progress | None = None) -> dict:
    t0 = _time.perf_counter()
    out = out_dir(base, case, "annual")
    fld = build_field(case)
    res = annual(case, fld, method=method, aiming=aiming,
                 progress=(lambda i, n: progress(f"sun state {i}/{n}")) if progress else None)
    tag = f"annual_{res.method}_{res.aiming}"
    files = {
        "summary_json": out / f"{tag}.json",
        "timeseries_csv": out / f"{tag}_timeseries.csv",
        "heliostats_csv": out / f"{tag}_heliostats.csv",
        "efficiency_png": out / f"{tag}_efficiency.png",
    }
    summary = {"case": case.name, "n_heliostats": fld.n,
               "mirror_area_m2": round(fld.n * case.heliostat.area, 1), **res.summary(), **_dropped_note(fld)}
    _json(files["summary_json"], summary)
    res.timeseries.to_csv(files["timeseries_csv"], index=False)
    pd.DataFrame({"id": fld.ids, "x": fld.xyz[:, 0], "y": fld.xyz[:, 1], "z": fld.xyz[:, 2],
                  "energy_mwh": res.heliostat_energy_mwh, "annual_efficiency": res.heliostat_efficiency}
                 ).to_csv(files["heliostats_csv"], index=False)
    plot_layout(fld, case.receiver, files["efficiency_png"], values=res.heliostat_efficiency,
                label="Annual optical efficiency", boundary=case.field.boundary,
                title="Annual optical efficiency per heliostat")
    if res.timeseries["time"].notna().any():
        files["monthly_png"] = out / f"{tag}_monthly.png"
        plot_annual_monthly(res.timeseries, files["monthly_png"])
    finish(out, "annual", summary, {"method": method, "aiming": aiming}, t0)
    return {"summary": _clean(summary), "files": files, "result": res, "run_dir": out}


def task_optimize_layout(case: Case, base: str | Path, progress: Progress | None = None,
                         cancel: Callable[[], bool] | None = None) -> dict:
    t0 = _time.perf_counter()
    out = out_dir(base, case, "layout_opt")
    res = optimize_layout(case, progress=progress, cancel=cancel)
    fld = Field(res.best.xyz)
    files = {
        "summary_json": out / "layout_opt.json",
        "layout_csv": out / "layout_optimized.csv",
        "history_csv": out / "layout_opt_history.csv",
        "ranking_csv": out / "layout_opt_ranking.csv",
        "layout_png": out / "layout_optimized.png",
        "history_png": out / "layout_opt_history.png",
        "case_yaml": out / "case_optimized.yaml",
    }
    save_layout_csv(fld, files["layout_csv"])
    res.history.to_csv(files["history_csv"], index=False)
    res.ranking.to_csv(files["ranking_csv"], index=False)
    best = res.best.aps_result or res.best.result
    plot_layout(fld, case.receiver, files["layout_png"],
                values=best.heliostat_efficiency if best is not None and len(best.heliostat_efficiency) == fld.n else None,
                label="Annual optical efficiency", boundary=case.field.boundary,
                exclusion=case.field.exclusion_radius, title=f"Optimised layout, N = {fld.n} [{res.status}]")
    plot_history(res.history, files["history_png"])
    if res.potential is not None:
        files["potential_npz"] = out / "site_potential.npz"
        np.savez_compressed(files["potential_npz"], x=res.potential.x, y=res.potential.y,
                            energy_mwh=res.potential.energy_mwh, cell_m=res.potential.cell_m)
    new_case = load_case(out / "inputs_case.yaml")
    new_case.field.layout_file = files["layout_csv"].name
    new_case.field.generator = None
    new_case.name = f"{case.name}_optimized"
    new_case.base_dir = out
    files["case_yaml"].write_text(new_case.to_yaml(), encoding="utf-8")
    summary = res.summary()
    summary["search_settings"] = {k: getattr(case.layout_opt, k) for k in (
        "generator", "n_heliostats", "bounds", "fixed", "search_sampling", "n_sun_bins", "review_sampling",
        "search_aiming", "review_n_sun_bins", "review_dni_bands", "maxiter", "popsize", "seed", "polish_rounds", "refine_top",
        "swap_rounds", "swap_fraction", "uniformity_weight", "ranking", "aps_weight", "baseline", "formal_swap_rounds",
        "formal_position_trials", "ranking_refine_states", "inner_joint_trials", "inner_certify_rounds", "dense_check_scenarios")}
    if res.formal and res.formal.get("values"):
        fdir = out / "formal"
        files.update(save_scenario_set(res.formal["scenario_set"], fdir / "scenario_set", case))
        files.update(export_values(res.formal["values"], fdir / "values", res.formal["ranking"]))
        summary["formal"]["exports"] = {
            "scenario_set": "formal/scenario_set (scenarios.csv, dni_maps.csv/json)",
            "per_layout": "formal/values/<layout>/scenario_values.csv + aims/<scenario>.csv + layout.csv",
            "recompute": "V = sum(hours * (p_abs_mw - lambda_layout * u_mw)) over rows with counts_in_value"}
    _json(files["summary_json"], summary)
    finish(out, "layout_opt", summary, t0=t0)
    return {"summary": _clean(summary), "files": files, "result": res, "run_dir": out,
            "input_parameters": case.to_dict()}


def task_fetch_weather(case: Case, base: str | Path, year: int | None = None) -> dict:
    """Hourly DNI for the case's site from NASA POWER, plus a copy of the case that uses it."""
    from .weather import fetch_power_dni

    t0 = _time.perf_counter()
    out = out_dir(base, case, "weather")
    year = int(year or case.annual.year)
    site = case.site
    path = out / f"nasa_power_dni_{site.latitude:.4f}_{site.longitude:.4f}_{year}.csv"
    info = fetch_power_dni(site.latitude, site.longitude, year, site.timezone, path)
    new_case = case.copy()
    new_case.weather.source, new_case.weather.file = "file", path.name
    new_case.annual.year = year
    new_case.base_dir = out
    files = {"weather_csv": path, "case_yaml": out / f"{case.name}_weather_{year}.yaml"}
    files["case_yaml"].write_text(new_case.to_yaml(), encoding="utf-8")
    info["file"] = path.name
    _json(out / "weather.json", info)
    finish(out, "fetch_weather", info, {"year": year}, t0)
    return {"summary": _clean(info), "files": files, "run_dir": out}


def task_layout_tradeoff(case: Case, base: str | Path, lambdas=None, params: dict | None = None,
                         progress: Progress | None = None) -> dict:
    """Annual energy vs circumferential uniformity of the layout over lambda_L (fixed generator parameters).

    ``params`` are the generator parameters (default: the case's field generator,
    else the middle of the layout_opt bounds); the search sampling of layout_opt is used.
    """
    from .layout_opt import DEFAULT_BOUNDS, layout_tradeoff, make_samples

    t0 = _time.perf_counter()
    out = out_dir(base, case, "layout_tradeoff")
    lo = case.layout_opt
    kind = lo.generator
    if params is None:
        gen = dict(case.field.generator or {})
        if gen and gen.pop("type", kind) == kind:
            params = {k: v for k, v in gen.items() if k not in ("r_min", "r_max")}
        else:
            bounds = {**DEFAULT_BOUNDS.get(kind, {}), **(lo.bounds or {})}
            params = {k: 0.5 * (v[0] + v[1]) for k, v in bounds.items()}
        params = {**lo.fixed, **params}
    lambdas = list(lambdas) if lambdas is not None else [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    if progress:
        progress(f"layout trade-off: {len(lambdas)} weights, {lo.n_sun_bins} sun states")
    samples = make_samples(case, lo.search_sampling, lo.n_sun_bins, 1, seed=lo.seed)
    n_select = lo.n_heliostats or len(generate(case, kind, params))
    rows = layout_tradeoff(case, kind, params, samples, n_select, lambdas, lo.shading_points)
    files = {"summary_json": out / "layout_tradeoff.json", "curve_csv": out / "layout_tradeoff.csv"}
    layouts = {}
    for r in rows:
        layouts[r["lambda"]] = r.pop("xyz")
        p = out / f"layout_lambda_{r['lambda']:g}.csv"
        save_layout_csv(Field(layouts[r["lambda"]]), p)
        r["layout_csv"] = p.name
    pd.DataFrame(rows).to_csv(files["curve_csv"], index=False)
    summary = {"generator": kind, "params": params, "n_heliostats": n_select,
               "sampling": {"method": samples.label, "n_states": len(samples)}, "curve": rows}
    _json(files["summary_json"], summary)
    finish(out, "layout_tradeoff", summary, {"lambdas": lambdas, "params": params}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out, "layouts": layouts}


def task_uniformity(case: Case, base: str | Path, time, eps_values=None, exact: bool = False,
                    power_floor_fraction: float = 0.995, band_fraction: float = 0.125) -> dict:
    """Power-vs-uniformity curve, CV metrics and the edge-band reachability bound at one instant."""
    t0 = _time.perf_counter()
    out = out_dir(base, case, "uniformity")
    fld = build_field(case)
    ts, el, az, dni = sun_state(case, time)
    grid = build_grid(case.receiver)
    baseline_case = case.copy()
    baseline_case.uniformity.enabled = False
    prob = build_problem(baseline_case, fld, el, az, dni, grid=grid)
    objective_prob = prob
    base_choice, base_info = solve_greedy(prob, case.aps)
    base_aim = prob.aims(base_choice)
    snap0 = evaluate(baseline_case, fld, el, az, dni, aim=base_aim, active=base_choice >= 0, grid=grid)
    h_mask = region_of_grid(case, grid, case.uniformity)
    m0 = uniformity_metrics(snap0.absorbed_flux, grid.to_map(h_mask))
    if eps_values is None:
        top = max(m0["cv_2d"], 0.05)
        eps_values = [round(top * f, 3) for f in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6)]
    curve = sweep(objective_prob, base_choice, eps_values, spec=case.uniformity, exact=exact)
    rec = case.receiver
    full_region = case.uniformity.panels is None and case.uniformity.z_range is None
    alpha_uniform = rec.panel_absorptivity is None or np.allclose(rec.panel_absorptivity, rec.panel_absorptivity[0])
    if full_region and alpha_uniform:
        bound = edge_band_bound(prob, power_floor_fraction * prob.total_power(base_choice), band_fraction)
    else:
        bound = {"feasible": None, "status": "not_applicable",
                 "reason": "edge-band bound is derived for the full receiver and uniform absorptivity"}
    for row in curve:
        if row["epsilon"] < bound.get("cv_2d_lower_bound", -1):
            row["proof"] = f"unreachable at >= {power_floor_fraction:.1%} of the greedy power (edge-band bound)"
    files = {"summary_json": out / "uniformity.json", "curve_csv": out / "uniformity_curve.csv",
             "curve_png": out / "uniformity_curve.png"}
    pd.DataFrame(curve).to_csv(files["curve_csv"], index=False)
    plot_uniformity_curve(curve, files["curve_png"])
    summary = {"time": ts.isoformat(), "region_H": {"panels": case.uniformity.panels, "z_range": case.uniformity.z_range},
               "greedy_baseline": {"power_intercepted_mw": prob.total_power(base_choice) / 1e6,
                                   "power_absorbed_mw": snap0.p_absorbed / 1e6,
                                   "solver": base_info, **m0},
               "curve": curve, "edge_band_bound": bound}
    _json(files["summary_json"], summary)
    finish(out, "uniformity", summary, {"time": str(time), "eps": list(eps_values), "exact": exact}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}


def task_validate(case: Case, base: str | Path, time, rays: int | None = None, method: str = "center",
                  seeds: int | None = None, convergence_rays: list[int] | None = None) -> dict:
    t0 = _time.perf_counter()
    out = out_dir(base, case, "validate")
    fld = build_field(case)
    ts, el, az, dni = sun_state(case, time)
    grid = build_grid(case.receiver)
    aim, active = None, None
    if method != "center":
        r = optimize_aims(case, fld, el, az, dni, method=method, grid=grid)
        aim, active = r.aim, r.active
    snap = evaluate(case, fld, el, az, dni, aim=aim, active=active, grid=grid, time=ts)
    ray, stats = trace_repeated(case, fld, el, az, dni, aim=(snap.aim_azimuth, snap.aim_z), active=snap.active,
                                grid=grid, rays_per_heliostat=rays, n_seeds=seeds)
    cmp = compare(snap, ray)
    cmp["mcrt_statistics"] = stats
    cmp["verified_analytic_peak_kw_m2"] = snap.verify(case.receiver).peak_flux_kw_m2
    if convergence_rays:
        cmp["convergence"] = convergence(case, fld, el, az, dni, aim=(snap.aim_azimuth, snap.aim_z),
                                         active=snap.active, ray_counts=convergence_rays, n_seeds=seeds or case.mcrt.n_seeds)
    stamp = f"validate_{ts:%Y%m%d_%H%M}_{method}"
    files = {"summary_json": out / f"{stamp}.json", "compare_png": out / f"{stamp}_flux_compare.png",
             "analytic_csv": out / f"{stamp}_analytic_kw_m2.csv", "mcrt_csv": out / f"{stamp}_mcrt_kw_m2.csv"}
    header = (f"kW/m2; grid {grid.azimuth_deg.size} azimuth x {grid.z.size} height; rows = height bottom->top, "
              f"columns = azimuth from N clockwise; MCRT seeds {stats['seeds']}, {stats['rays_per_heliostat']} rays/heliostat")
    np.savetxt(files["analytic_csv"], snap.flux / 1e3, delimiter=",", fmt="%.4f", header=header)
    np.savetxt(files["mcrt_csv"], ray.flux / 1e3, delimiter=",", fmt="%.4f", header=header)
    _json(files["summary_json"], cmp)
    plot_flux_compare(snap.flux, ray.flux, grid, case.receiver, files["compare_png"],
                      labels=("Analytic (HFLCAL)", f"MCRT ({len(stats['seeds'])} seeds x {stats['rays_per_heliostat']} rays)"))
    finish(out, "validate", cmp, {"time": str(time), "rays": rays, "method": method, "seeds": seeds}, t0)
    return {"summary": _clean(cmp), "files": files, "run_dir": out}


def task_soltrace_export(case: Case, base: str | Path, time, method: str = "center", rays: int | None = None,
                         seeds: int | None = None) -> dict:
    from .soltrace import export_benchmark

    t0 = _time.perf_counter()
    out = out_dir(base, case, "soltrace")
    fld = build_field(case)
    res = export_benchmark(case, fld, time, out, method, rays, seeds)
    _json(out / "export_summary.json", res["summary"])
    finish(out, "soltrace_export", res["summary"], {"time": str(time), "method": method}, t0)
    return {"summary": _clean(res["summary"]), "files": res["files"], "run_dir": out}


def task_validation_suite(case: Case, base: str | Path, keys=None, rays: int | None = None, seeds: int | None = None,
                          year: int = 2025, soltrace: bool = True, progress: Progress | None = None) -> dict:
    """Ray-trace checks of the stress states (low sun, cloud edge, high-load mixed aiming, near the limit).

    Physical error (analytic vs MCRT on the same plan and grid) and the solver gap are reported separately;
    every plan is also exported for SolTrace (per-mirror DNI as a reflectivity factor)."""
    from .soltrace import export_plan
    from .validation import SUITE, compare_state, row, suite_states

    t0 = _time.perf_counter()
    out = out_dir(base, case, "validation_suite")
    say = progress or (lambda m: None)
    fld = build_field(case)
    keys = tuple(keys or SUITE)
    states = suite_states(case, fld, keys, year=year, progress=say)
    grid = build_grid(case.receiver)
    results, rows, files = [], [], {}
    for st_ in states:
        say(f"ray trace {st_.key}")
        r = compare_state(st_, fld, rays=rays, seeds=seeds, grid=build_grid(st_.case.receiver))
        maps = r.pop("maps")
        header = (f"kW/m2; {st_.key}; grid {grid.azimuth_deg.size} azimuth x {grid.z.size} height; rows = height "
                  f"bottom->top, columns = azimuth from N clockwise; MCRT seeds {r['physical']['mcrt']['seeds']}, "
                  f"{r['physical']['mcrt']['rays_per_heliostat']} rays/heliostat")
        for k in ("analytic", "mcrt"):
            fp = out / f"{st_.key}_{k}_kw_m2.csv"
            np.savetxt(fp, maps[f"{k}_w_m2"] / 1e3, delimiter=",", fmt="%.4f", header=header)
            files[f"{st_.key}_{k}_csv"] = fp
        fp = out / f"{st_.key}_limit_kw_m2.csv"
        np.savetxt(fp, maps["limit_w_m2"] / 1e3, delimiter=",", fmt="%.4f", header=header.replace(st_.key, f"{st_.key} limit"))
        files[f"{st_.key}_limit_csv"] = fp
        png = out / f"{st_.key}_flux_compare.png"
        plot_flux_compare(maps["analytic_w_m2"], maps["mcrt_w_m2"], build_grid(st_.case.receiver), st_.case.receiver, png,
                          labels=(f"Analytic, {st_.key}", f"MCRT ({len(r['physical']['mcrt']['seeds'])} seeds)"))
        files[f"{st_.key}_png"] = png
        if soltrace:
            p = st_.plan
            ex = export_plan(st_.case, fld, st_.time, st_.elevation, st_.azimuth, st_.dni, (p.aim_azimuth, p.aim_z),
                             p.active, out / "soltrace" / st_.key, method=f"suite:{st_.key}", with_mcrt=False)
            r["soltrace_export"] = {"folder": str(out / "soltrace" / st_.key), "next_step": ex["summary"]["next_step"],
                                    "dni_per_mirror": ex["summary"]["dni_w_m2"]["per_mirror"]}
        r["maps_files"] = {k: str(v) for k, v in files.items() if k.startswith(st_.key)}
        r["_maps"] = maps
        results.append(r)
        rows.append(row(r))
    table = pd.DataFrame(rows)
    files["summary_csv"] = out / "validation_suite.csv"
    table.to_csv(files["summary_csv"], index=False, encoding="utf-8-sig")
    files["summary_json"] = out / "validation_suite.json"
    summary = {"states": [{k: v for k, v in r.items() if k != "_maps"} for r in results], "table": rows,
               "conventions": {"physical_error": "analytic minus MCRT on the same plan, grid and DNI per mirror",
                               "solver_gap": "(upper bound - objective) / upper bound on the candidate set, analytic model",
                               "mcrt_within_limit": "MCRT mean map + 2 standard errors <= the limit (same grid)",
                               "dni": "clear-sky model of the case (not the weather record); cloud edge scales it per mirror"}}
    _json(files["summary_json"], summary)
    finish(out, "validation_suite", {"table": rows}, {"keys": list(keys), "rays": rays, "seeds": seeds, "year": year}, t0)
    return {"summary": _clean(summary), "table": table, "maps": {r["key"]: r["_maps"] for r in results},
            "files": files, "run_dir": out}


def task_solver_benchmark(case: Case, base: str | Path, part: str = "inner", n_states: int = 6, budget_s: float = 60.0,
                          seeds=(1, 2), limit_factor: float | None = None, methods=None,
                          max_evals: int | None = None, lam_aps: float | None = None,
                          progress: Progress | None = None) -> dict:
    """Solvers under the same conditions: inner (task 1 on the largest scenarios) and / or outer (layout search).

    ``limit_factor`` scales every flux limit (e.g. 0.45) so that the limits bind; the record says so."""
    from . import benchmark as bm
    from .validation import tighten_limits

    t0 = _time.perf_counter()
    out = out_dir(base, case, "solver_benchmark")
    say = progress or (lambda m: None)
    c = tighten_limits(case, float(limit_factor)) if limit_factor else case
    summary: dict = {"case": case.name, "budget_s": budget_s, "limit_factor": limit_factor,
                     "max_flux_kw_m2": c.receiver.max_flux_kw_m2, "lambda_aps": lam_aps}
    files = {}
    result: dict = {}
    if part in ("inner", "both"):
        fld = build_field(c)
        states = bm.inner_states(c, fld, n_states)
        df = bm.inner_benchmark(c, fld, states, methods or bm.INNER_METHODS, budget_s, lam_aps=lam_aps, progress=say)
        agg = bm.inner_summary(df)
        files["inner_csv"] = out / "inner_benchmark.csv"
        files["inner_summary_csv"] = out / "inner_benchmark_summary.csv"
        df.to_csv(files["inner_csv"], index=False, encoding="utf-8-sig")
        agg.to_csv(files["inner_summary_csv"], index=False, encoding="utf-8-sig")
        summary["inner"] = {"n_heliostats": fld.n, "states": [s["label"] for s in states],
                            "summary": agg.to_dict("records"), "runs": df.to_dict("records")}
        result["inner"], result["inner_summary"] = df, agg
    if part in ("outer", "both"):
        r = bm.outer_benchmark(c, methods or bm.OUTER_METHODS, tuple(seeds), budget_s, max_evals=max_evals, progress=say)
        files["outer_csv"] = out / "layout_benchmark.csv"
        files["outer_summary_csv"] = out / "layout_benchmark_summary.csv"
        r["runs"].to_csv(files["outer_csv"], index=False, encoding="utf-8-sig")
        r["summary"].to_csv(files["outer_summary_csv"], index=False, encoding="utf-8-sig")
        for name, xyz in r["layouts"].items():
            fp = out / f"layout_{name}.csv"
            save_layout_csv(Field(xyz), fp)
            files[f"layout_{name}"] = fp
        summary["outer"] = {"runs": r["runs"].to_dict("records"), "summary": r["summary"].to_dict("records"),
                            "train": r["train"], "held_out": r["held_out"], "seeds": r["seeds"],
                            "settings_digest": r["settings_digest"], "n_heliostats": c.layout_opt.n_heliostats}
        result["outer"], result["outer_summary"] = r["runs"], r["summary"]
    files["summary_json"] = out / "solver_benchmark.json"
    _json(files["summary_json"], summary)
    finish(out, "solver_benchmark", {"part": part}, {"part": part, "n_states": n_states, "budget_s": budget_s,
                                                       "seeds": list(seeds), "limit_factor": limit_factor,
                                                       "lambda_aps": lam_aps}, t0)
    return {"summary": _clean(summary), **result, "files": files, "run_dir": out}


def task_shading_check(case: Case, base: str | Path) -> dict:
    """Corridor screening vs brute force on the advisor's four stress cases."""
    from . import optics
    from .receiver import default_aim

    t0 = _time.perf_counter()
    out = out_dir(base, case, "shading_check")
    fld = build_field(case)
    r = np.hypot(fld.xyz[:, 0] - case.receiver.center[0], fld.xyz[:, 1] - case.receiver.center[1])
    year = case.annual.year
    hourly = sun.hourly_samples(case)
    low_i = int(np.argmin(hourly.elevation))
    low_time = hourly.times[low_i]
    near = fld.xyz[r <= np.quantile(r, 0.3)]
    far = fld.xyz[r >= np.quantile(r, 0.8)]
    try:
        dense = generate(case, case.layout_opt.generator, {"radial_spacing": 0.866, "radial_growth": 0.0,
                                                           "azimuthal_spacing": 1.0, "r_max": float(np.quantile(r, 0.4))}
                         if case.layout_opt.generator == "radial_staggered" else {"a": 0.55, "b": 0.5})
    except ValueError:
        dense = near
    cases = [("low_sun_near_field", near, low_time, None), ("far_field", far, low_time, None),
             ("dense_layout_low_sun", dense, low_time, None),
             ("edge_aiming_noon", near, f"{year}-03-20 12:30", case.receiver.top - 0.1)]
    rows = []
    for name, xyz, t, z_aim in cases:
        ts, el, az, _ = sun_state(case, t)
        s = sun.sun_vector(el, az)
        a_az, a_z = default_aim(case.receiver, xyz)
        if z_aim is not None:
            a_z = np.full_like(a_z, z_aim)
        n, tt, *_ = optics.heliostat_geometry(xyz, surface_point(case.receiver, a_az, a_z), s)
        res = compare_screening(xyz, n, s, tt, case.heliostat, case.receiver, case.shading.n_points,
                                case.shading.tower_shadow)
        rows.append({"case": name, "time": ts, "sun_elevation_deg": round(el, 2), **res})
    files = {"summary_json": out / "shading_check.json", "table_csv": out / "shading_check.csv"}
    pd.DataFrame(rows).to_csv(files["table_csv"], index=False)
    summary = {"status": "ok" if all(r_["missed_loss_total"] <= 1e-12 for r_ in rows) else "missed_losses",
               "cases": rows}
    _json(files["summary_json"], summary)
    finish(out, "shading_check", summary, t0=t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}


def task_sunbins(case: Case, base: str | Path, n_bins: int | None = None, check: bool = False,
                 methods: list[str] | None = None) -> dict:
    """Representative sun positions; with ``check`` the annual-energy error of 20/30/40 bins and the grid vs hourly."""
    t0 = _time.perf_counter()
    out = out_dir(base, case, "sunbins")
    hourly = sun.hourly_samples(case)
    bins = sun.sun_bins(hourly, n_bins or case.annual.n_sun_bins, dni_bands=case.annual.dni_bands,
                        weighting=case.annual.representative_dni)
    files = {"bins_csv": out / "sun_bins.csv", "bins_png": out / "sun_bins.png", "summary_json": out / "sun_bins.json"}
    pd.DataFrame({"elevation_deg": bins.elevation, "azimuth_deg": bins.azimuth, "dni_w_m2": bins.dni,
                  "weight_h": bins.weight_h, "actual_hours": bins.actual_hours,
                  "dni_energy_kwh_m2": bins.dni * bins.weight_h / 1e3,
                  "hourly_members": bins.members}).to_csv(files["bins_csv"], index=False)
    plot_sun_bins(hourly, bins, files["bins_png"])
    summary = {"n_hourly_samples": len(hourly), "n_bins": len(bins), "annual_dni_kwh_m2": round(hourly.dni_hours, 1),
               "sampling": sun.sampling_record(case, bins, "n/a")}
    if check:
        fld = build_field(case)
        t = _time.perf_counter()
        e_h = annual(case, fld, samples=hourly, with_flux=False).energy_mwh["intercept"]
        rows = [{"method": "hourly", "n_states": len(hourly), "energy_mwh": e_h, "error_pct": 0.0,
                 "runtime_s": _time.perf_counter() - t}]
        variants = [("sun_bins", k) for k in (20, 30, 40)] + [("sun_grid", None)]
        for name, k in variants:
            smp = sun.sun_bins(hourly, k, weighting=case.annual.representative_dni) if name == "sun_bins" else sun.sun_grid(
                hourly, case.annual.grid_elevation_step_deg, case.annual.grid_azimuth_step_deg,
                weighting=case.annual.representative_dni)
            t = _time.perf_counter()
            e = annual(case, fld, samples=smp, with_flux=False, workers=1).energy_mwh["intercept"]
            rows.append({"method": name if k is None else f"{name}_{k}", "n_states": len(smp), "energy_mwh": e,
                         "error_pct": 100 * (e / e_h - 1), "runtime_s": _time.perf_counter() - t})
        summary["center_aiming_error_vs_hourly"] = rows
        files["check_csv"] = out / "sun_bins_check.csv"
        pd.DataFrame(rows).to_csv(files["check_csv"], index=False)
    _json(files["summary_json"], summary)
    finish(out, "sunbins", summary, t0=t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}


# --------------------------------------------------------------------------- engine service

def engine_service(case: Case, lines, write: Callable[[str], None]) -> int:
    """JSON-lines service: one request per input line, one response per output line.

    Request: {"time": "...", "dni": [..] or {"id": dni}, "dni_time": "...",
              "forecasts": [{"valid_time": "...", "dni": [..]}], "available": [..],
              "receipt": {"plan_id": k, "aim_azimuth": [..], "aim_z": [..], "active": [..], "executed": true}}
    Response: plan, state, reasons, flux check, timings and per-heliostat aims (by ID).
    """
    fld = build_field(case)
    eng = APSEngine(case, fld)
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            req = request_from_dict(json.loads(line), fld, case.site.timezone)
            write(json.dumps(response_to_dict(eng.step(req)), ensure_ascii=False, default=_default))
        except Exception as exc:  # noqa: BLE001 - every request gets an answer
            write(json.dumps({"state": "error", "error": f"{type(exc).__name__}: {exc}"}))
        n += 1
    return n


# --------------------------------------------------------------------------- annual value with spatial DNI maps

LAYOUT_NAMES_ZH = {"initial": "初始布局"}


def layout_label(name: str) -> str:
    """Display name of a formal-ranking layout (search_2+swap1 -> 搜索候选 2 + 换位 1)."""
    if name in LAYOUT_NAMES_ZH:
        return LAYOUT_NAMES_ZH[name]
    if len(name) == 1 and name.isalpha():
        return f"布局 {name}"
    if name.startswith("position_"):
        return f"自由镜位 {name.removeprefix('position_')}"
    if name.startswith("search_"):
        base, _, swap = name.partition("+swap")
        txt = f"搜索候选 {base.removeprefix('search_')}"
        return f"{txt} + 换位 {swap}" if swap else txt
    return name


def _scenario_set_for(case: Case, scenarios_dir=None):
    from .scenarios import build_scenarios, load_folder

    return load_folder(scenarios_dir) if scenarios_dir else build_scenarios(case)


def _layout_source(case: Case, source) -> tuple[np.ndarray, np.ndarray, str]:
    """(xyz, ids, note) of a layout given as 'initial', a CSV path or an (N, 3) array."""
    from .field import read_layout
    from .layout_value import baseline_layout

    if isinstance(source, str) and source == "initial":
        n = case.layout_opt.n_heliostats or build_field(case).n
        b = baseline_layout(case, n)
        if b is None:
            raise ValueError("the case's initial field has fewer heliostats than layout_opt.n_heliostats")
        return b
    if isinstance(source, (str, Path)):
        path = case.resolve(source)
        if not path.is_file() and Path(source).is_file():
            path = Path(source)
        xyz, ids = read_layout(path, case.heliostat.pivot_height)
        return xyz, (np.arange(1, len(xyz) + 1) if ids is None else ids), f"layout CSV {Path(source).name}"
    xyz = np.asarray(source, dtype=float).reshape(-1, 3)
    return xyz, np.arange(1, len(xyz) + 1), "layout array"


def save_scenario_set(sset, folder: Path, case: Case) -> dict:
    from .plotting import plot_dni_maps

    files = sset.save(folder)
    files["maps_png"] = folder / "dni_maps.png"
    plot_dni_maps(sset.subset_maps(), files["maps_png"], boundary=case.field.boundary, tower=case.receiver.center[:2])
    return files


def task_scenarios(case: Case, base: str | Path, n: int | None = None) -> dict:
    """Build the case's scenario set (sun, hours, DNI map per scenario) and save it for reuse and audit."""
    from .scenarios import build_scenarios

    t0 = _time.perf_counter()
    out = out_dir(base, case, "scenarios")
    # n: scenarios of a weather set, or representative days of the clear-sky baseline
    sset = build_scenarios(case, n, n_days=n if case.scenarios.source == "clear_sky_days" else None)
    files = save_scenario_set(sset, out / "scenario_set", case)
    summary = {"status": "done", **{k: v for k, v in sset.record().items() if k != "map_records"},
               "scenarios": sset.table().to_dict(orient="records"),
               "maps": {k: m.record() for k, m in sset.subset_maps().items()}}
    _json(out / "scenarios.json", summary)
    finish(out, "scenarios", summary, {"n": n}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out, "scenario_set": sset}


def task_scenario_aps(case: Case, base: str | Path, scenario_id: str, scenarios_dir=None, layout=None,
                      lam_aps: float | None = None) -> dict:
    """Task 1 for one scenario of a scenario set on one layout -- the independent check of a layout score.

    Uses exactly the inner settings of the layout evaluation (``layout_value.inner_settings``), so the
    plan, P_s and U_s equal the row of that layout's ``scenario_values.csv``.
    """
    from .layout_value import inner_settings, solve_scenario

    t0 = _time.perf_counter()
    out = out_dir(base, case, "scenario_aps")
    sset = _scenario_set_for(case, scenarios_dir)
    xyz, ids, note = _layout_source(case, layout) if layout is not None else (build_field(case).xyz, build_field(case).ids,
                                                                             "case field")
    fld = Field(xyz, ids)
    settings = inner_settings(case, lam_aps)
    i = sset.index(scenario_id)
    rec = solve_scenario(case, fld, sset, i, settings)
    row = rec.row(settings.lam_aps, case.layout_opt.uniformity_weight)
    p = rec.plan
    files = {"summary_json": out / f"scenario_{scenario_id}.json", "aims_csv": out / f"scenario_{scenario_id}_aims.csv",
             "flux_png": out / f"scenario_{scenario_id}_flux.png"}
    write_aims_csv(files["aims_csv"], fld, case, (p.aim_azimuth, p.aim_z), p.active,
                   {"dni_w_m2": np.round(p.dni, 4), "absorbed_kw": np.round(p.mirror_power_w / 1e3, 5)})
    grid = build_grid(case.receiver, settings.grid_shape[1], settings.grid_shape[0])
    if p.flux is not None:
        plot_flux_map(p.flux, grid, case.receiver, files["flux_png"], aims=(p.aim_azimuth[p.active], p.aim_z[p.active]),
                      title=f"{scenario_id} [{p.status}] P_abs {p.p_abs_w / 1e6:.3f} MW, U {p.u_w / 1e6:.3f} MW")
    summary = {"status": p.status, "scenario": row, "layout": note, "n_heliostats": fld.n,
               "scenario_set_fingerprint": sset.fingerprint(), "inner_settings": settings.record(),
               "map": sset.samples.map_of(i).record()}
    _json(files["summary_json"], summary)
    finish(out, "scenario_aps", summary, {"scenario_id": scenario_id, "scenarios_dir": scenarios_dir,
                                          "layout": str(layout) if layout is not None else None}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out, "record": rec, "field": fld}


def export_values(values: dict, folder: Path, ranking: pd.DataFrame | None = None) -> dict:
    """Per-layout exports (scenario table, aim plans, layout CSV) and the comparison table."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, v in values.items():
        safe = export_name(name)
        sub = v.export(folder / safe)
        files[f"values_{safe}_csv"] = sub["scenario_csv"]
    if ranking is not None and len(ranking):
        files["ranking_csv"] = folder / "formal_ranking.csv"
        cols = [c for c in ranking.columns if c not in ("status_counts", "status_hours")]
        r = ranking[cols].copy()
        r["status_counts"] = ranking["status_counts"].map(lambda d: json.dumps(d, ensure_ascii=False))
        r["label"] = r["name"].map(layout_label)
        r.to_csv(files["ranking_csv"], index=False)
    return files


def task_layout_value(case: Case, base: str | Path, layouts: dict | None = None, scenarios_dir=None,
                      lam_layout: float | None = None, progress: Progress | None = None,
                      cancel: Callable[[], bool] | None = None) -> dict:
    """Evaluate given layouts on one scenario set with their own APS plans and rank them by V(L).

    ``layouts`` maps a name to 'initial', a layout CSV or an (N, 3) array (default: the initial field).
    """
    from .layout_value import (STATUS_RULES, compare_values, displacement, evaluate_layout_values, inner_settings,
                               refine_competing_plans)

    t0 = _time.perf_counter()
    out = out_dir(base, case, "layout_value")
    sset = _scenario_set_for(case, scenarios_dir)
    files = save_scenario_set(sset, out / "scenario_set", case)
    settings = inner_settings(case)
    lam = case.layout_opt.uniformity_weight if lam_layout is None else float(lam_layout)
    if not 0 <= lam < float("inf"):
        raise ValueError("The uniformity weight must be finite and nonnegative")
    layouts = layouts or {"initial": "initial"}
    values, notes, excluded, inputs = {}, {}, {}, {}
    archived_layouts = {}
    input_dir = out / "input_layouts"
    input_dir.mkdir()
    for index, (name, src) in enumerate(layouts.items(), 1):
        destination = input_dir / f"layout_{index}.csv"
        if isinstance(src, str) and src == "initial":
            archived_layouts[name] = "initial"
        else:
            archived_layouts[name] = destination.relative_to(out).as_posix()
            if isinstance(src, (str, Path)):
                path = case.resolve(src)
                if not path.is_file() and Path(src).is_file():
                    path = Path(src)
                if path.is_file():
                    shutil.copy2(path, destination)
                    src = destination.resolve()
            else:
                points = np.asarray(src, dtype=float).reshape(-1, 3)
                pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(destination, index=False, float_format="%.17g")
        try:
            inputs[name] = _layout_source(case, src)
        except (OSError, ValueError) as exc:
            excluded[name] = [str(exc)]
    counts = {len(xyz) for xyz, _, _ in inputs.values()}
    required = case.layout_opt.n_heliostats
    batch = {}
    for name, (xyz, ids, note) in inputs.items():
        check = validate_layout(case, xyz, required, design_spacing(case), ids)
        if check["status"] != "ok":
            excluded[name] = check["violations"]
            continue
        if required is None and len(counts) > 1:
            excluded[name] = ["比较需要等镜数，请指定目标镜数"]
            continue
        notes[name] = note
        batch[name] = (xyz, ids)
    # all layouts x scenarios in one pool (every core busy even with few scenarios)
    values.update(evaluate_layout_values(case, batch, sset, settings=settings, lam_layout=lam, progress=progress,
                                         cancel=cancel, tags={n: {"note": notes[n]} for n in batch}))
    refine_competing_plans(case, values, sset, progress, cancel)
    ranking = compare_values(list(values.values()), lam, reference="initial" if "initial" in values else None,
                             rank_by=case.layout_opt.rank_by)
    files.update(export_values(values, out / "values", ranking))
    status = "constraint_violation" if excluded else "done"
    if any(not v.totals()["complete"] for v in values.values()):
        status = "incomplete"
    summary = {"status": status, "excluded_layouts": excluded, "input_layouts": archived_layouts,
               "comparison_status": ranking.iloc[0]["comparison_status"] if len(ranking) else "no_valid_layout",
               "equal_heliostat_count": len(counts) == 1, "lambda_aps": settings.lam_aps, "lambda_layout": lam,
               "scenario_set": {k: v for k, v in sset.record().items() if k != "map_records"},
               "inner_settings": settings.record(), "ranking": ranking.to_dict("records"),
               "notes": notes, "status_rules": STATUS_RULES}
    names = list(values)
    if len(names) >= 2 and values[names[0]].n == values[names[1]].n:
        d = displacement(values[names[0]].xyz, values[names[1]].xyz)
        summary["displacement"] = {k: d[k] for k in ("mean_move_m", "median_move_m", "max_move_m", "n_unchanged",
                                                     "n_moved")} if d.get("comparable") else None
    files["summary_json"] = out / "layout_value.json"
    _json(files["summary_json"], summary)
    finish(out, "layout_value", summary, {"layouts": {k: str(v) if not isinstance(v, np.ndarray) else "array"
                                                      for k, v in layouts.items()}, "lambda_layout": lam}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out, "values": values, "ranking": ranking,
            "scenario_set": sset}


def replay_layout_value(folder, base, progress=None, cancel=None):
    """Re-solve a saved comparison using only inputs and layouts beneath the exported run folder."""
    folder = Path(folder).resolve()
    summary = json.loads((folder / "layout_value.json").read_text(encoding="utf-8"))
    case = load_case(folder / "inputs_case.yaml")
    layouts = summary.get("input_layouts")
    if layouts:
        layouts = {name: ("initial" if path == "initial" else folder / path) for name, path in layouts.items()}
    else:  # Historical comparisons store their positions with the per-scenario results.
        layouts = {r["name"]: folder / "values" / export_name(r["name"]) / "layout.csv"
                   for r in summary["ranking"]}
    return task_layout_value(case, base, layouts, folder / "scenario_set", summary["lambda_layout"], progress, cancel)


def task_rescore(folder: str | Path, lam_layout: float, land_weight: float = 0.0) -> dict:
    """Recompute every layout's V (and J = V - kappa A) for another outer weight / land weight from exported
    scenario tables and layouts only (no solver)."""
    from .layout_value import recompute_from_csv

    folder = Path(folder)
    if not 0 <= lam_layout < float("inf"):
        raise ValueError("The uniformity weight must be finite and nonnegative")
    if not 0 <= land_weight < float("inf"):
        raise ValueError("The land weight must be finite and nonnegative")
    rows = []
    for csv in sorted(folder.rglob("scenario_values.csv")):
        r = recompute_from_csv(csv, lam_layout, land_weight)
        rows.append({"layout": csv.parent.name, **r, "file": str(csv)})
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    key = "objective_mwh" if len(df) and "objective_mwh" in df and df["objective_mwh"].notna().all() else "value_mwh"
    if len(df):
        df = df.sort_values(key, ascending=False).reset_index(drop=True)
        df["rank"] = df[key].where(df["complete"]).rank(ascending=False, method="min")
    return {"summary": {"lambda_layout": lam_layout, "land_weight": land_weight, "ranked_by": key,
                        "layouts": df.to_dict("records")}, "files": {}, "run_dir": folder}


def task_cloud_compare(case: Case, base: str | Path, time, n_clouds: int = 2, radius_m=(60.0, 140.0),
                       transmittance=(0.1, 0.35), seed: int = 0, clouds: list[dict] | None = None,
                       lam_aps: float | None = None, progress: Progress | None = None) -> dict:
    """One instant before and after a cloud arrives (task 1 with the formal inner settings).

    States:
      clear       clear-sky DNI, plan A = APS for the clear field
      cloud_keep  cloud shadow on the field, plan A kept (no re-optimisation)
      cloud_aps   cloud shadow, plan C = APS re-solved for the shaded DNI
      clear_keep  the cloud has gone, plan C still executed (risk of exceeding the flux limit)
      cloud_safe  cloud shadow, APS with the flux limit checked at clear-sky DNI (safe if the cloud clears)
    """
    from .clouds import Cloud, cloud_transmittance, static_clouds
    from .layout_value import inner_case, inner_settings, solve_instant
    from .plotting import plot_cloud_compare

    t0 = _time.perf_counter()
    out = out_dir(base, case, "cloud_compare")
    say = progress or (lambda m: None)
    fld = build_field(case)
    ts, el, az, dni_clear = sun_state(case, time)
    u = case.uniformity
    lam = (u.weight if (u.enabled and u.mode == "weighted") else 0.0) if lam_aps is None else float(lam_aps)
    settings = inner_settings(case, lam)
    grid = build_grid(case.receiver, settings.grid_shape[1], settings.grid_shape[0])
    cl = ([Cloud(**c) for c in clouds] if clouds else
          static_clouds(fld, n_clouds, tuple(radius_m), tuple(transmittance), seed))
    tau = cloud_transmittance(cl, fld.xyz[:, :2])
    d_clear = np.full(fld.n, float(dni_clear))
    d_cloud = d_clear * tau
    c_eval = inner_case(case, settings)

    def executed(label, key, dni, aim, active, plan=None):
        snap = evaluate(c_eval, fld, el, az, dni, aim=aim, active=active, grid=grid, time=ts)
        th = snap.thermal_objective(c_eval.uniformity)
        chk = snap.verify(case.receiver)
        m = uniformity_metrics(snap.absorbed_flux, grid.to_map(snap.region))
        return {"key": key, "label": label, "dni": dni, "aim_azimuth": np.asarray(aim[0], float),
                "aim_z": np.asarray(aim[1], float), "active": np.asarray(active, bool), "flux": snap.flux,
                "absorbed_flux": snap.absorbed_flux, "tower_fraction": snap.tower_fraction,
                "p_abs_mw": th["power_w"] / 1e6, "p_int_mw": snap.p_receiver / 1e6,
                "u_mw": th["spread_w"] / 1e6, "objective_mw": (th["power_w"] - lam * th["spread_w"]) / 1e6,
                "cv_2d": th["cv_2d"], "cv_circ": m["cv_circ"], "cv_height": m["cv_height"],
                "peak_flux_kw_m2": chk.peak_flux_kw_m2, "peak_to_allowed": chk.peak_ratio / max(chk.allowed_ratio, 1e-12),
                "peak_to_limit": chk.peak_ratio, "flux_status": chk.status,
                "hotspot": {"azimuth_deg": chk.hotspot_azimuth_deg, "z_m": chk.hotspot_z, "panel": int(chk.hotspot_panel)},
                "n_active": int(np.sum(active)), "n_defocused": int(np.sum(~np.asarray(active, bool))),
                "n_tracking_in_tower_shadow": int(np.sum(np.asarray(active) & (snap.tower_fraction == 1.))),
                "mirror_power_kw": snap.power["intercept"] * (snap.alpha_eff if snap.alpha_eff is not None else 1.0) / 1e3,
                "plan_status": plan.status if plan is not None else "kept",
                "inner_gap": plan.gap if plan is not None else None}

    say("plan A: APS for the clear field")
    plan_a = solve_instant(case, fld, el, az, d_clear, settings, grid)
    aim_a, act_a = (plan_a.aim_azimuth, plan_a.aim_z), plan_a.active
    say("plan C: APS re-solved under the cloud shadow")
    plan_c = solve_instant(case, fld, el, az, d_cloud, settings, grid, warm=plan_a.choice)  # A competes too
    aim_c, act_c = (plan_c.aim_azimuth, plan_c.aim_z), plan_c.active
    say("plan S: APS under the cloud, flux checked at clear-sky DNI")
    plan_s = solve_instant(case, fld, el, az, d_cloud, settings, grid, dni_constraint=d_clear, warm=plan_a.choice)
    states = [executed("云来之前 · APS 方案 A", "clear", d_clear, aim_a, act_a, plan_a),
              executed("云来 · 沿用方案 A", "cloud_keep", d_cloud, aim_a, act_a),
              executed("云来 · 重新 APS 方案 C", "cloud_aps", d_cloud, aim_c, act_c, plan_c),
              executed("云散 · 仍执行方案 C", "clear_keep", d_clear, aim_c, act_c),
              executed("云来 · 防云散 APS 方案 S", "cloud_safe", d_cloud, (plan_s.aim_azimuth, plan_s.aim_z),
                       plan_s.active, plan_s),
              executed("云散 · 执行方案 S", "clear_safe", d_clear, (plan_s.aim_azimuth, plan_s.aim_z), plan_s.active)]
    for s, en in zip(states, ("Clear sky, plan A", "Cloud, plan A kept", "Cloud, APS re-solved (C)",
                              "Cloud gone, plan C kept", "Cloud, APS safe for clearing (S)",
                              "Cloud gone, plan S kept")):
        s["label_en"] = en
    pts_a = surface_point(case.receiver, *aim_a)
    for s in states:
        p = surface_point(case.receiver, s["aim_azimuth"], s["aim_z"])
        moved = np.linalg.norm(p - pts_a, axis=1)
        changed = (moved > 1e-6) | (s["active"] != act_a)
        s["n_changed_vs_a"] = int(changed.sum())
        s["mean_aim_shift_m"] = float(moved[changed & s["active"] & act_a].mean()) if np.any(changed & s["active"] & act_a) else 0.0
    rows = [{k: v for k, v in s.items() if not isinstance(v, np.ndarray) and k not in ("hotspot",)} for s in states]
    table = pd.DataFrame(rows)
    files = {"summary_json": out / "cloud_compare.json", "states_csv": out / "cloud_compare_states.csv",
             "compare_png": out / "cloud_compare.png", "aims_csv": out / "cloud_compare_aims.csv"}
    table.to_csv(files["states_csv"], index=False)
    aims = pd.DataFrame({"id": fld.ids, "x": fld.xyz[:, 0], "y": fld.xyz[:, 1], "transmittance": np.round(tau, 5)})
    for s in states:
        aims[f"{s['key']}_dni"] = np.round(s["dni"], 3)
        aims[f"{s['key']}_active"] = s["active"].astype(int)
        aims[f"{s['key']}_tower_shadow_fraction"] = s["tower_fraction"]
        aims[f"{s['key']}_aim_azimuth"] = np.round(s["aim_azimuth"], 5)
        aims[f"{s['key']}_aim_z"] = np.round(s["aim_z"], 5)
        aims[f"{s['key']}_power_kw"] = np.round(s["mirror_power_kw"], 4)
    aims.to_csv(files["aims_csv"], index=False)
    plot_cloud_compare(fld.xyz, case.receiver, grid, states[:3] + states[3:4], files["compare_png"],
                       boundary=case.field.boundary)
    base_row = states[0]
    summary = {"status": "done", "time": ts.isoformat(), "sun_elevation_deg": el, "sun_azimuth_deg": az,
               "dni_clear_w_m2": float(dni_clear), "lambda_aps": lam, "n_heliostats": fld.n,
               "clouds": [c.__dict__ for c in cl], "shaded_heliostats": int((tau < 0.95).sum()),
               "mean_transmittance": float(tau.mean()), "states": rows,
               "loss_keep_vs_clear_mw": base_row["p_abs_mw"] - states[1]["p_abs_mw"],
               "gain_reaps_vs_keep_mw": states[2]["p_abs_mw"] - states[1]["p_abs_mw"],
               "inner_settings": settings.record()}
    _json(files["summary_json"], summary)
    finish(out, "cloud_compare", summary, {"time": str(time), "n_clouds": n_clouds, "seed": seed,
                                           "radius_m": list(radius_m), "transmittance": list(transmittance)}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out, "states": states, "field": fld,
            "grid": grid, "transmittance": tau, "clouds": cl, "case": case}


def task_preference_trial(case: Case, base: str | Path, scenario_id: str | None = None, lambdas=(0.0, None),
                          layout="initial", scenarios_dir=None) -> dict:
    """Try the preference on one scenario: the same layout solved with each lambda (None = the case's lambda).

    Returns absorbed power, hottest flux, uniformity (CV, U) and objective for every lambda, so a user sees
    what moving the "more heat <-> more uniform heating" slider changes before running the whole year.
    """
    from .layout_value import inner_settings, solve_scenario

    t0 = _time.perf_counter()
    out = out_dir(base, case, "preference_trial")
    sset = _scenario_set_for(case, scenarios_dir)
    if scenario_id is None:  # the scenario with the most DNI energy
        scenario_id = sset.ids[int(np.argmax(sset.samples.dni * sset.samples.weight_h))]
    i = sset.index(scenario_id)
    xyz, ids, note = _layout_source(case, layout)
    fld = Field(xyz, ids)
    rows = []
    for lam in lambdas:
        lam = case.layout_opt.uniformity_weight if lam is None else float(lam)
        st = inner_settings(case, lam)
        rec = solve_scenario(case, fld, sset, i, st, keep_flux=False)
        p = rec.plan
        rows.append({"lambda": lam, "status": p.status, "p_abs_mw": p.p_abs_w / 1e6, "u_mw": p.u_w / 1e6,
                     "cv_2d": p.cv_2d, "peak_flux_kw_m2": p.peak_flux_kw_m2, "peak_to_limit": p.info.get("peak_to_limit"),
                     "hotspot_panel": p.info.get("hotspot_panel"), "objective_mw": p.objective_w / 1e6,
                     "n_defocused": p.n_defocused, "inner_gap": p.gap})
    summary = {"status": "done", "scenario_id": scenario_id, "map_id": sset.map_id(i),
               "hours": float(sset.samples.weight_h[i]), "layout": note, "n_heliostats": fld.n,
               "max_flux_kw_m2": case.receiver.max_flux_kw_m2, "safety_margin": case.aps.safety_margin,
               "trials": rows}
    files = {"summary_json": out / "preference_trial.json"}
    _json(files["summary_json"], summary)
    finish(out, "preference_trial", summary, {"scenario_id": scenario_id, "lambdas": list(lambdas)}, t0)
    return {"summary": _clean(summary), "files": files, "run_dir": out}
