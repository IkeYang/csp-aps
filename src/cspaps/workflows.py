"""Task runners shared by the CLI and the GUI.

Every run gets its own folder ``<out>/<case>/<task>_<YYYYmmdd-HHMMSS>/`` with
``inputs_case.yaml`` (the exact inputs), ``run.json`` (task, arguments,
version, status, timings) and all outputs, so earlier results can be reviewed.
Each runner returns ``{"summary": dict, "files": {label: path}, "run_dir": Path, ...}``.
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__, sun
from .aps import build_problem, optimize_aims, solve
from .clouds import load_dni_map
from .config import Case
from .engine import APSEngine, request_from_dict, response_to_dict
from .field import Field, build_field, generate, save_layout_csv
from .layout_opt import optimize_layout
from .mcrt import compare, convergence, trace_repeated
from .plotting import (
    plot_annual_monthly, plot_flux_compare, plot_flux_map, plot_history, plot_layout,
    plot_realtime, plot_sun_bins, plot_uniformity_curve,
)
from .realtime import run_realtime
from .receiver import build_grid, surface_point
from .shading import compare_screening
from .simulate import LOSSES, annual, evaluate, sun_state
from .uniformity import edge_band_bound, sweep, uniformity_metrics

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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    return path


def _clean(d):
    return json.loads(json.dumps(d, default=_default))


def out_dir(base: str | Path, case: Case, task: str = "run") -> Path:
    """A new run folder; the case inputs are saved in it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(base) / case.name
    out = root / f"{task}_{stamp}"
    k = 1
    while out.exists():
        k += 1
        out = root / f"{task}_{stamp}-{k}"
    out.mkdir(parents=True)
    (out / "inputs_case.yaml").write_text(case.to_yaml(), encoding="utf-8")
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
) -> dict:
    """Single instant: optimise aim points and evaluate them (status = verified flux check)."""
    t_start = _time.perf_counter()
    out = out_dir(base, case, tag)
    fld = build_field(case)
    method = method or case.aps.method
    ts, el, az, dni_model = sun_state(case, time)
    dni_v = load_dni_map(dni_map, fld.xyz) if dni_map else (dni_model if dni is None else float(dni))
    dni_con = load_dni_map(forecast_dni_map, fld.xyz) if forecast_dni_map else None
    prev, prev_active = read_aims_csv(prev_aims, fld) if prev_aims else (None, None)
    grid = build_grid(case.receiver)
    if method == "center":
        from .receiver import default_aim

        aim, active, res = default_aim(case.receiver, fld.xyz), np.ones(fld.n, bool), None
    else:
        res = optimize_aims(case, fld, el, az, dni_v, dni_constraint=dni_con, method=method, grid=grid,
                            prev=prev, prev_active=prev_active)
        aim, active = res.aim, res.active
    snap = evaluate(case, fld, el, az, dni_v, aim=aim, active=active, grid=grid, time=ts)
    chk_con = res.check if res is not None and res.check is not None else snap.verify(case.receiver)
    status = res.status if res is not None else {"ok": "ok", "margin_reduced": "margin_reduced",
                                                 "over_limit": "infeasible"}[chk_con.status]
    stamp = f"{ts:%Y%m%d_%H%M}_{method}"
    files = {
        "summary_json": out / f"{tag}_{stamp}.json",
        "aims_csv": out / f"{tag}_{stamp}_aims.csv",
        "flux_png": out / f"{tag}_{stamp}_flux.png",
        "efficiency_png": out / f"{tag}_{stamp}_efficiency.png",
    }
    extra = {f"eta_{k}": np.round(snap.eta[k], 6) for k in LOSSES}
    extra["dni_w_m2"] = np.round(snap.dni, 2)
    extra["p_receiver_kw"] = np.round(snap.power["intercept"] / 1e3, 4)
    write_aims_csv(files["aims_csv"], fld, case, aim, active, extra)
    check_summary = chk_con.summary()
    check_summary["top_contributors"] = [{"heliostat_id": int(fld.ids[c["heliostat"]]), "flux_kw_m2": c["flux_kw_m2"]}
                                         for c in check_summary["top_contributors"]]
    summary = {"status": status, "aps": res.summary() if res is not None else {"method": "center"},
               "flux_check": check_summary, **snap.summary(), **_dropped_note(fld)}
    summary["aps"].pop("flux_check", None)
    _json(files["summary_json"], summary)
    plot_flux_map(snap.flux, grid, case.receiver, files["flux_png"], aims=aim, hotspot=chk_con,
                  title=f"{ts:%Y-%m-%d %H:%M}  {method}  [{status}]  verified peak {chk_con.peak_flux_kw_m2:.0f} kW/m², "
                        f"{snap.p_receiver / 1e6:.2f} MW intercepted")
    plot_layout(fld, case.receiver, files["efficiency_png"], values=snap.eta_total,
                label="Optical efficiency", boundary=case.field.boundary,
                title=f"Heliostat optical efficiency, {ts:%Y-%m-%d %H:%M}")
    finish(out, tag, summary, {"time": str(time), "method": method, "dni": dni, "dni_map": dni_map,
                               "forecast_dni_map": forecast_dni_map, "prev_aims": prev_aims}, t_start)
    return {"summary": _clean(summary), "files": files, "snapshot": snap, "aim_result": res, "field": fld,
            "check": chk_con, "run_dir": out}


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
    new_case = case.copy()
    new_case.field.layout_file = str(files["layout_csv"].resolve())
    new_case.field.generator = None
    new_case.name = f"{case.name}_optimized"
    files["case_yaml"].write_text(new_case.to_yaml(), encoding="utf-8")
    summary = res.summary()
    summary["search_settings"] = {k: getattr(case.layout_opt, k) for k in (
        "generator", "n_heliostats", "bounds", "fixed", "search_sampling", "n_sun_bins", "review_sampling",
        "review_n_sun_bins", "review_dni_bands", "maxiter", "popsize", "seed", "polish_rounds", "refine_top",
        "swap_rounds", "swap_fraction")}
    _json(files["summary_json"], summary)
    finish(out, "layout_opt", summary, t0=t0)
    return {"summary": _clean(summary), "files": files, "result": res, "run_dir": out}


def task_uniformity(case: Case, base: str | Path, time, eps_values=None, exact: bool = False,
                    power_floor_fraction: float = 0.995, band_fraction: float = 0.125) -> dict:
    """Power-vs-uniformity curve, CV metrics and the edge-band reachability bound at one instant."""
    t0 = _time.perf_counter()
    out = out_dir(base, case, "uniformity")
    fld = build_field(case)
    ts, el, az, dni = sun_state(case, time)
    grid = build_grid(case.receiver)
    prob = build_problem(case, fld, el, az, dni, grid=grid)
    base_res = solve(prob, "greedy")
    snap0 = evaluate(case, fld, el, az, dni, aim=base_res.aim, active=base_res.active, grid=grid)
    m0 = uniformity_metrics(snap0.absorbed_flux, grid.to_map(snap0.region))
    if eps_values is None:
        top = max(m0["cv_2d"], 0.05)
        eps_values = [round(top * f, 3) for f in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6)]
    curve = sweep(prob, base_res.choice, eps_values, exact=exact)
    bound = edge_band_bound(prob, power_floor_fraction * base_res.power_w, band_fraction)
    for row in curve:
        if row["epsilon"] < bound.get("cv_2d_lower_bound", -1):
            row["proof"] = f"unreachable at >= {power_floor_fraction:.1%} of the greedy power (edge-band bound)"
    files = {"summary_json": out / "uniformity.json", "curve_csv": out / "uniformity_curve.csv",
             "curve_png": out / "uniformity_curve.png"}
    pd.DataFrame(curve).to_csv(files["curve_csv"], index=False)
    plot_uniformity_curve(curve, files["curve_png"])
    summary = {"time": ts.isoformat(), "region_H": {"panels": case.uniformity.panels, "z_range": case.uniformity.z_range},
               "greedy_baseline": {"power_intercepted_mw": base_res.power_w / 1e6, **m0},
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
    bins = sun.sun_bins(hourly, n_bins or case.annual.n_sun_bins, dni_bands=case.annual.dni_bands)
    files = {"bins_csv": out / "sun_bins.csv", "bins_png": out / "sun_bins.png", "summary_json": out / "sun_bins.json"}
    pd.DataFrame({"elevation_deg": bins.elevation, "azimuth_deg": bins.azimuth, "dni_w_m2": bins.dni,
                  "weight_h": bins.weight_h, "dni_energy_kwh_m2": bins.dni * bins.weight_h / 1e3,
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
            smp = sun.sun_bins(hourly, k) if name == "sun_bins" else sun.sun_grid(
                hourly, case.annual.grid_elevation_step_deg, case.annual.grid_azimuth_step_deg)
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
