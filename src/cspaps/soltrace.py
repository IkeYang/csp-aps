"""Benchmark export for an external SolTrace / PySolTrace comparison (advisor item P2-03).

``export_benchmark`` writes one self-contained case: sun, optics with explicit
error conventions, every heliostat (id, position, normal, focal length, size,
active), the receiver and tower, the flux grid, and the CSP-APS results on that
grid (analytic HFLCAL map and internal MCRT, attenuation excluded on both sides
because SolTrace does not model it). ``run_pysoltrace.py`` (copied alongside)
builds the same scene in PySolTrace, bins its hits on the same grid and writes
the comparison.
"""

from __future__ import annotations

import copy
import json
import shutil
import time as _time
from pathlib import Path

import numpy as np

from . import optics, sun
from .aps import optimize_aims
from .config import Case
from .field import Field
from .mcrt import trace_repeated
from .receiver import build_grid, surface_point
from .simulate import evaluate, sun_state


def export_benchmark(case: Case, fld: Field, time, out_dir: str | Path, method: str = "center",
                     mcrt_rays: int | None = None, mcrt_seeds: int | None = None) -> dict:
    ts, el, az, dni = sun_state(case, time)
    bench = benchmark_case(case)
    grid = build_grid(bench.receiver)
    aim, active = None, None
    if method != "center":
        r = optimize_aims(bench, fld, el, az, dni, method=method, grid=grid)
        aim, active = r.aim, r.active
    return export_plan(case, fld, ts, el, az, dni, aim, active, out_dir, method, mcrt_rays, mcrt_seeds)


def benchmark_case(case: Case) -> Case:
    """The case as exported for SolTrace: attenuation excluded on both sides."""
    bench = copy.deepcopy(case)
    bench.atmosphere.model = "beer_lambert"
    bench.atmosphere.extinction_per_km = 0.0
    return bench


def export_plan(case: Case, fld: Field, ts, el: float, az: float, dni, aim, active, out_dir: str | Path,
                method: str = "given plan", mcrt_rays: int | None = None, mcrt_seeds: int | None = None,
                with_mcrt: bool = True) -> dict:
    """SolTrace benchmark of a given plan (aims, active mirrors, per-mirror DNI) + the internal MCRT of it.

    Per-mirror DNI goes out as ``dni_factor`` = DNI_i / max DNI; the driver scales each mirror's
    reflectivity by it and sets the SolTrace DNI to the maximum."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bench = benchmark_case(case)
    grid = build_grid(bench.receiver)
    dni_v = np.broadcast_to(np.asarray(dni, dtype=float), (fld.n,)).copy()
    dni_max = float(dni_v.max()) if fld.n else 0.0
    dni = dni_v
    t0 = _time.perf_counter()
    snap = evaluate(bench, fld, el, az, dni, aim=aim, active=active, grid=grid, time=ts)
    t_analytic = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    mc, mc_stats = None, None
    if with_mcrt:
        mc, mc_stats = trace_repeated(bench, fld, el, az, dni, aim=(snap.aim_azimuth, snap.aim_z),
                                      active=snap.active, grid=grid, rays_per_heliostat=mcrt_rays, n_seeds=mcrt_seeds)
    t_mcrt = _time.perf_counter() - t0
    hel, rec = bench.heliostat, bench.receiver
    s = sun.sun_vector(el, az)
    aim_xyz = surface_point(rec, snap.aim_azimuth, snap.aim_z)
    normals, _, slant, _ = optics.heliostat_geometry(fld.xyz, aim_xyz, s)
    if hel.focusing == "slant_range":
        focal = slant
    elif hel.focusing == "flat":
        focal = [None] * fld.n
    else:
        focal = np.full(fld.n, float(hel.focusing))
    sun_sigma = case.sun.sigma_mrad if case.sun.shape == "gaussian" else case.sun.half_angle_mrad
    data = {
        "description": "CSP-APS benchmark case for an external SolTrace comparison",
        "time": ts.isoformat() if ts is not None else None, "aiming": method, "dni_w_m2": dni_max,
        "conventions": {
            "frame": "x = East, y = North, z = Up, m; azimuth compass (N = 0, clockwise)",
            "sun_sigma": "Gaussian: std per axis, mrad; pillbox: half angle, mrad",
            "normal_error_mrad": "per-axis mirror normal error = sqrt(slope^2 + tracking^2)",
            "specular_error_mrad": "per-axis error of the reflected ray",
            "heliostat_surface": "spherical with radius 2 x focal_length; flat if focal_length is null",
            "attenuation": "excluded on both sides",
            "receiver": "closed vertical cylinder, fully absorbing; tower = absorbing cylinder below it",
            "flux_grid": "rows = height bottom->top, columns = azimuth cells starting at N, clockwise",
        },
        "sun": {"vector": s.tolist(), "elevation_deg": el, "azimuth_deg": az, "shape": case.sun.shape,
                "sigma_mrad": sun_sigma},
        "optics": {"reflectivity": hel.reflectivity * hel.cleanliness,
                   "normal_error_mrad": float(np.hypot(hel.slope_error_mrad, hel.tracking_error_mrad)),
                   "specular_error_mrad": hel.specular_error_mrad},
        "receiver": {"center": list(rec.center), "radius": rec.radius, "height": rec.height,
                     "tower_radius": rec.tower_radius if case.shading.tower_shadow else 0.0,
                     "tower_height": rec.center[2] - 0.5 * rec.height},
        "grid": {"n_azimuth": int(grid.azimuth_deg.size), "n_height": int(grid.z.size)},
        "heliostats": [
            {"id": int(h_id), "position": p.tolist(), "normal": n.tolist(), "aim_point": a.tolist(),
             "focal_length": (None if f is None else float(f)), "width": hel.width, "height": hel.height,
             "active": bool(on), "dni_factor": (float(d / dni_max) if dni_max > 0 else 0.0)}
            for h_id, p, n, a, f, on, d in zip(fld.ids, fld.xyz, normals, aim_xyz, focal, snap.active, dni_v)
        ],
        "analytic": {"power_on_receiver_w": snap.p_receiver, "flux_w_m2": snap.flux.tolist(),
                     "peak_flux_kw_m2": float(snap.flux.max()) / 1e3, "run_time_s": t_analytic,
                     "stage_efficiency": snap.summary()["field_efficiency"]},
    }
    if mc is not None:
        data["cspaps_mcrt"] = {"power_on_receiver_w": mc.power["intercept"], "flux_w_m2": mc.flux.tolist(),
                               "statistics": mc_stats, "run_time_s": t_mcrt}
    case_path = out / "benchmark_case.json"
    case_path.write_text(json.dumps(data), encoding="utf-8")
    driver = out / "run_pysoltrace.py"
    shutil.copyfile(Path(__file__).with_name("soltrace_driver.py"), driver)
    summary = {
        "time": ts.isoformat() if ts is not None else None, "aiming": method, "n_heliostats": fld.n,
        "power_on_receiver_mw": {"analytic": snap.p_receiver / 1e6},
        "peak_flux_kw_m2": {"analytic": float(snap.flux.max()) / 1e3},
        "dni_w_m2": {"max": dni_max, "min": float(dni_v.min()) if fld.n else 0.0,
                     "per_mirror": bool(fld.n and np.ptp(dni_v) > 0)},
        "run_time_s": {"analytic": t_analytic},
        "next_step": f"python {driver.name} {case_path.name} --pysoltrace <SolTrace>/app/deploy/api",
    }
    if mc is not None:
        fa, fm = snap.flux, mc.flux
        lit = fm > 0.05 * fm.max() if fm.max() > 0 else np.ones_like(fm, bool)
        summary["power_on_receiver_mw"]["cspaps_mcrt"] = mc.power["intercept"] / 1e6
        summary["peak_flux_kw_m2"]["cspaps_mcrt_mean_map"] = float(fm.max()) / 1e3
        summary["flux_map_nrmse"] = float(np.sqrt(np.mean((fa[lit] - fm[lit]) ** 2)) / fm[lit].mean())
        summary["mcrt_statistics"] = mc_stats
        summary["run_time_s"]["cspaps_mcrt"] = t_mcrt
    return {"summary": summary, "files": {"benchmark_json": case_path, "pysoltrace_driver": driver},
            "snapshot": snap, "mcrt": mc}
