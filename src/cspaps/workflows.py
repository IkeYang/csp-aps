"""Task runners shared by the CLI and the GUI.

Each runner takes a Case, does the work, writes its outputs into ``out`` and
returns ``{"summary": dict, "files": {label: path}, ...}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import sun
from .aps import optimize_aims
from .clouds import load_dni_map
from .config import Case
from .field import Field, build_field, save_layout_csv
from .layout_opt import optimize_layout
from .mcrt import compare, trace
from .plotting import (
    plot_annual_monthly, plot_flux_compare, plot_flux_map, plot_history, plot_layout,
    plot_realtime, plot_sun_bins,
)
from .realtime import run_realtime
from .receiver import build_grid, surface_point
from .simulate import LOSSES, annual, evaluate, sun_state

Progress = Callable[[str], None]


def _json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean(d):
    return json.loads(json.dumps(d, default=_default))


def out_dir(base: str | Path, case: Case) -> Path:
    out = Path(base) / case.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_aims_csv(path: Path, fld: Field, case: Case, aim, active, extra: dict | None = None) -> Path:
    xyz = surface_point(case.receiver, aim[0], aim[1])
    df = pd.DataFrame({
        "id": np.arange(fld.n), "x": fld.xyz[:, 0], "y": fld.xyz[:, 1], "z": fld.xyz[:, 2],
        "active": np.asarray(active, dtype=int), "aim_azimuth_deg": np.round(aim[0], 4),
        "aim_z": np.round(aim[1], 4), "aim_x": np.round(xyz[:, 0], 4), "aim_y": np.round(xyz[:, 1], 4),
    })
    for k, v in (extra or {}).items():
        df[k] = v
    df.to_csv(path, index=False)
    return path


def read_aims_csv(path) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    df = pd.read_csv(path)
    active = df["active"].to_numpy(bool) if "active" in df else np.ones(len(df), bool)
    return (df["aim_azimuth_deg"].to_numpy(float), df["aim_z"].to_numpy(float)), active


# --------------------------------------------------------------------------- tasks

def task_layout(case: Case, out: Path) -> dict:
    fld = build_field(case)
    files = {"layout_csv": out / "layout.csv", "layout_png": out / "layout.png"}
    save_layout_csv(fld, files["layout_csv"])
    plot_layout(fld, case.receiver, files["layout_png"], boundary=case.field.boundary)
    summary = {"n_heliostats": fld.n, "mirror_area_m2": round(fld.n * case.heliostat.area, 1)}
    return {"summary": summary, "files": files, "field": fld}


def task_aim(
    case: Case,
    out: Path,
    time,
    method: str | None = None,
    dni: float | None = None,
    dni_map: str | Path | None = None,
    forecast_dni_map: str | Path | None = None,
    prev_aims: str | Path | None = None,
    tag: str = "aps",
) -> dict:
    """Single instant: optimise aim points (the real-time control engine) and evaluate them."""
    fld = build_field(case)
    method = method or case.aps.method
    ts, el, az, dni_model = sun_state(case, time)
    dni_v = load_dni_map(dni_map, fld.xyz) if dni_map else (dni_model if dni is None else float(dni))
    dni_con = load_dni_map(forecast_dni_map, fld.xyz) if forecast_dni_map else None
    prev, prev_active = read_aims_csv(prev_aims) if prev_aims else (None, None)
    grid = build_grid(case.receiver)
    res = optimize_aims(case, fld, el, az, dni_v, dni_constraint=dni_con, method=method, grid=grid,
                        prev=prev, prev_active=prev_active)
    snap = evaluate(case, fld, el, az, dni_v, aim=res.aim, active=res.active, grid=grid, time=ts)
    stamp = f"{tag}_{ts:%Y%m%d_%H%M}_{method}"
    files = {
        "summary_json": out / f"{stamp}.json",
        "aims_csv": out / f"{stamp}_aims.csv",
        "flux_png": out / f"{stamp}_flux.png",
        "efficiency_png": out / f"{stamp}_efficiency.png",
    }
    extra = {f"eta_{k}": np.round(snap.eta[k], 6) for k in LOSSES}
    extra["dni_w_m2"] = np.round(snap.dni, 2)
    extra["p_receiver_kw"] = np.round(snap.power["intercept"] / 1e3, 4)
    write_aims_csv(files["aims_csv"], fld, case, res.aim, res.active, extra)
    summary = {"aps": res.summary() if method != "center" else {"method": "center"}, **snap.summary()}
    _json(files["summary_json"], summary)
    plot_flux_map(snap.flux, grid, case.receiver, files["flux_png"], aims=res.aim,
                  title=f"{ts:%Y-%m-%d %H:%M}  {method}: peak {snap.flux.max() / 1e3:.0f} kW/m² "
                        f"(limit {case.receiver.max_flux_kw_m2:.0f}), {snap.p_receiver / 1e6:.2f} MW")
    plot_layout(fld, case.receiver, files["efficiency_png"], values=snap.eta_total,
                label="Optical efficiency", boundary=case.field.boundary,
                title=f"Heliostat optical efficiency, {ts:%Y-%m-%d %H:%M}")
    return {"summary": _clean(summary), "files": files, "snapshot": snap, "aim_result": res, "field": fld}


def task_realtime(case: Case, out: Path, progress: Progress | None = None) -> dict:
    fld = build_field(case)
    res = run_realtime(case, fld, progress=(lambda i, n: progress(f"step {i}/{n}")) if progress else None)
    files = {
        "summary_json": out / "realtime.json",
        "table_csv": out / "realtime_timeseries.csv",
        "timeseries_png": out / "realtime.png",
    }
    summary = res.summary()
    _json(files["summary_json"], summary)
    res.table.to_csv(files["table_csv"], index=False)
    plot_realtime(res.table, case.receiver.max_flux_kw_m2, files["timeseries_png"])
    for strat, snap in res.worst.items():
        p = out / f"realtime_worst_flux_{strat}.png"
        plot_flux_map(snap.flux, snap.grid, case.receiver, p, aims=(snap.aim_azimuth, snap.aim_z),
                      title=f"{strat}: worst moment {snap.time:%H:%M:%S}, peak {snap.flux.max() / 1e3:.0f} kW/m²")
        files[f"worst_flux_{strat}_png"] = p
    worst_t = res.table.loc[res.table["dni_min_w_m2"].idxmin(), "time"]
    cloud_png = out / "realtime_cloud_shadow.png"
    plot_layout(fld, case.receiver, cloud_png, values=res.clouds.transmittance(fld.xyz[:, :2], worst_t),
                label="Cloud transmittance", cmap="Blues_r", vmin=0, vmax=1.3, boundary=case.field.boundary,
                title=f"Cloud shadow at {pd.Timestamp(worst_t):%H:%M:%S}")
    files["cloud_png"] = cloud_png
    return {"summary": _clean(summary), "files": files, "result": res}


def task_annual(case: Case, out: Path, method: str | None = None, aiming: str | None = None,
                progress: Progress | None = None) -> dict:
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
               "mirror_area_m2": round(fld.n * case.heliostat.area, 1), **res.summary()}
    _json(files["summary_json"], summary)
    res.timeseries.to_csv(files["timeseries_csv"], index=False)
    pd.DataFrame({"x": fld.xyz[:, 0], "y": fld.xyz[:, 1], "z": fld.xyz[:, 2],
                  "energy_mwh": res.heliostat_energy_mwh, "annual_efficiency": res.heliostat_efficiency}
                 ).to_csv(files["heliostats_csv"], index=False)
    plot_layout(fld, case.receiver, files["efficiency_png"], values=res.heliostat_efficiency,
                label="Annual optical efficiency", boundary=case.field.boundary,
                title="Annual optical efficiency per heliostat")
    if res.timeseries["time"].notna().any():
        files["monthly_png"] = out / f"{tag}_monthly.png"
        plot_annual_monthly(res.timeseries, files["monthly_png"])
    return {"summary": _clean(summary), "files": files, "result": res}


def task_optimize_layout(case: Case, out: Path, progress: Progress | None = None) -> dict:
    res = optimize_layout(case, progress=progress)
    fld = res.field
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
                values=best.heliostat_efficiency if best is not None else None,
                label="Annual optical efficiency", boundary=case.field.boundary,
                title=f"Optimised layout, N = {fld.n}")
    plot_history(res.history, files["history_png"])
    new_case = case.copy()
    new_case.field.layout_file = str(files["layout_csv"].resolve())
    new_case.field.generator = None
    new_case.name = f"{case.name}_optimized"
    files["case_yaml"].write_text(new_case.to_yaml(), encoding="utf-8")
    summary = res.summary()
    _json(files["summary_json"], summary)
    return {"summary": _clean(summary), "files": files, "result": res}


def task_validate(case: Case, out: Path, time, rays: int | None = None, method: str = "center") -> dict:
    fld = build_field(case)
    ts, el, az, dni = sun_state(case, time)
    grid = build_grid(case.receiver)
    aim, active = None, None
    if method != "center":
        r = optimize_aims(case, fld, el, az, dni, method=method, grid=grid)
        aim, active = r.aim, r.active
    snap = evaluate(case, fld, el, az, dni, aim=aim, active=active, grid=grid, time=ts)
    ray = trace(case, fld, el, az, dni, aim=aim, active=active, grid=grid, rays_per_heliostat=rays)
    cmp = compare(snap, ray)
    stamp = f"validate_{ts:%Y%m%d_%H%M}"
    files = {"summary_json": out / f"{stamp}.json", "compare_png": out / f"{stamp}_flux_compare.png"}
    _json(files["summary_json"], cmp)
    plot_flux_compare(snap.flux, ray.flux, grid, case.receiver, files["compare_png"])
    return {"summary": _clean(cmp), "files": files}


def task_sunbins(case: Case, out: Path, n_bins: int | None = None, check: bool = False) -> dict:
    hourly = sun.hourly_samples(case)
    bins = sun.sun_bins(hourly, n_bins or case.annual.n_sun_bins)
    files = {"bins_csv": out / "sun_bins.csv", "bins_png": out / "sun_bins.png", "summary_json": out / "sun_bins.json"}
    pd.DataFrame({"elevation_deg": bins.elevation, "azimuth_deg": bins.azimuth, "dni_w_m2": bins.dni,
                  "weight_h": bins.weight_h, "dni_energy_kwh_m2": bins.dni * bins.weight_h / 1e3}
                 ).to_csv(files["bins_csv"], index=False)
    plot_sun_bins(hourly, bins, files["bins_png"])
    summary = {"n_hourly_samples": len(hourly), "n_bins": len(bins),
               "annual_dni_kwh_m2": round(hourly.dni_hours, 1)}
    if check:
        fld = build_field(case)
        e_h = annual(case, fld, samples=hourly, with_flux=False).energy_mwh["intercept"]
        e_b = annual(case, fld, samples=bins, with_flux=False).energy_mwh["intercept"]
        summary.update({"energy_hourly_mwh": round(e_h, 1), "energy_bins_mwh": round(e_b, 1),
                        "bins_error_pct": round(100 * (e_b / e_h - 1), 3)})
    _json(files["summary_json"], summary)
    return {"summary": _clean(summary), "files": files}
