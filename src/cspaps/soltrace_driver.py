"""Run a CSP-APS benchmark case in PySolTrace (SolTrace v3.4.0) and compare with CSP-APS.

This script is copied next to ``benchmark_case.json`` by ``cspaps soltrace-export``.
It only needs numpy and PySolTrace (with its compiled ``coretrace_api`` library):

    python run_pysoltrace.py benchmark_case.json --pysoltrace /path/to/SolTrace/app/deploy/api \
        --rays 2000000 --seeds 3 --threads 8

Conventions (identical on both sides, see "conventions" in the JSON):
* sun: Gaussian sigma (per axis) or pillbox half angle, mrad;
* mirror normal error per axis = sqrt(slope^2 + tracking^2) (SolTrace has no
  tracking error term), applied by SolTrace as its Gaussian slope error;
* specularity error per axis on the reflected ray;
* spherical heliostats with radius 2 f (f = focal length), flat if f is null;
* atmospheric attenuation excluded on both sides;
* receiver = closed vertical cylinder, absorbing (reflectivity 0); the tower
  below it is an absorbing cylinder in the heliostat stage (tower shadow);
* flux binned on the same azimuth x height grid as the analytic map.
Outputs: soltrace_flux_mean.csv, soltrace_summary.json (next to the case file).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np


def build(PySolTrace, Point, case, rays, dni):
    PT = PySolTrace()
    o = case["optics"]
    ref = PT.add_optic("Reflector")
    ref.front.reflectivity = o["reflectivity"]
    ref.front.slope_error = o["normal_error_mrad"]
    ref.front.spec_error = o["specular_error_mrad"]
    ref.front.dist_type = "g"
    ref.back.reflectivity = 0.0
    absb = PT.add_optic("Absorber")
    absb.front.reflectivity = 0.0
    absb.back.reflectivity = 0.0

    sun = PT.add_sun()
    sun.shape = "g" if case["sun"]["shape"] == "gaussian" else "p"
    sun.sigma = case["sun"]["sigma_mrad"]
    sv = case["sun"]["vector"]
    sun.position = Point(100.0 * sv[0], 100.0 * sv[1], 100.0 * sv[2])

    # per-mirror DNI (cloud shadows): SolTrace has one DNI, so each mirror's reflectivity is scaled by its DNI
    # fraction (optics shared by mirrors with the same fraction, 0.01 steps)
    scaled = {}

    def optic_for(factor):
        f = round(float(factor), 2)
        if f >= 0.995:
            return ref
        if f not in scaled:
            op = PT.add_optic(f"Reflector_{int(round(100 * f)):03d}")
            op.front.reflectivity = o["reflectivity"] * f
            op.front.slope_error = o["normal_error_mrad"]
            op.front.spec_error = o["specular_error_mrad"]
            op.front.dist_type = "g"
            op.back.reflectivity = 0.0
            scaled[f] = op
        return scaled[f]

    st = PT.add_stage()
    for h in case["heliostats"]:
        if not h["active"]:
            continue
        el = st.add_element()
        el.optic = optic_for(h.get("dni_factor", 1.0))
        p, n = h["position"], h["normal"]
        el.position = Point(*p)
        el.aim = Point(p[0] + 100.0 * n[0], p[1] + 100.0 * n[1], p[2] + 100.0 * n[2])
        el.zrot = PT.util_calc_zrot_azel(Point(*n))
        if h["focal_length"] is None:
            el.surface_flat()
        else:
            el.surface_spherical(2.0 * h["focal_length"])
        el.aperture_rectangle(h["width"], h["height"])
    rc = case["receiver"]
    cx, cy, cz = rc["center"]
    if rc["tower_radius"] > 0 and rc["tower_height"] > 0:
        tw = st.add_element()
        tw.optic = absb
        r_t, h_t = rc["tower_radius"], rc["tower_height"]
        tw.position = Point(cx - r_t, cy, 0.5 * h_t)
        tw.aim = Point(cx - r_t + 100.0, cy, 0.5 * h_t)
        tw.zrot = 90.0
        tw.surface_cylindrical(r_t)
        tw.aperture_singleax_curve(0.0, 0.0, h_t)

    sta = PT.add_stage()
    rx = sta.add_element()
    rx.optic = absb
    R = rc["radius"]
    rx.position = Point(cx - R, cy, cz)
    rx.aim = Point(cx - R + 100.0, cy, cz)
    rx.zrot = 90.0  # cylinder axis (local y) vertical
    rx.surface_cylindrical(R)
    rx.aperture_singleax_curve(0.0, 0.0, rc["height"])

    PT.num_ray_hits = int(rays)
    PT.max_rays_traced = int(rays) * 100
    PT.is_sunshape = True
    PT.is_surface_errors = True
    PT.dni = dni
    return PT


def flux_map(PT, case):
    df = PT.raydata
    last = df["stage"].max()
    hits = df[(df["stage"] == last) & (df["element"] != 0)]
    rc = case["receiver"]
    cx, cy, cz = rc["center"]
    n_az, n_z = case["grid"]["n_azimuth"], case["grid"]["n_height"]
    az = np.degrees(np.arctan2(hits["loc_x"].to_numpy() - cx, hits["loc_y"].to_numpy() - cy)) % 360.0
    z0 = cz - 0.5 * rc["height"]
    ia = np.clip((az / 360.0 * n_az).astype(int), 0, n_az - 1)
    iz = np.clip(((hits["loc_z"].to_numpy() - z0) / rc["height"] * n_z).astype(int), 0, n_z - 1)
    counts = np.bincount(iz * n_az + ia, minlength=n_az * n_z).reshape(n_z, n_az)
    cell = R_cell_area(rc, n_az, n_z)
    return counts * PT.powerperray / cell, len(hits) * PT.powerperray


def R_cell_area(rc, n_az, n_z):
    return rc["radius"] * (2 * math.pi / n_az) * rc["height"] / n_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--pysoltrace", required=True, help="folder containing pysoltrace.py and coretrace_api")
    ap.add_argument("--rays", type=float, default=2e6, help="ray hits per run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    sys.path.insert(0, args.pysoltrace)
    from pysoltrace import Point, PySolTrace

    with open(args.case, encoding="utf-8") as fh:
        case = json.load(fh)
    maps, powers, times = [], [], []
    for k in range(args.seeds):
        PT = build(PySolTrace, Point, case, args.rays, case["dni_w_m2"])
        t0 = time.perf_counter()
        PT.run(1000 + k, True, args.threads)
        times.append(time.perf_counter() - t0)
        f, p = flux_map(PT, case)
        maps.append(f)
        powers.append(p)
    flux = np.mean(maps, axis=0)
    a = np.asarray(case["analytic"]["flux_w_m2"])
    lit = flux > 0.05 * flux.max()
    n = len(maps)
    peaks = np.array([m.max() for m in maps])
    out = {
        "engine": "SolTrace (PySolTrace)", "seeds": [1000 + k for k in range(n)], "ray_hits_per_run": int(args.rays),
        "grid": [case["grid"]["n_azimuth"], case["grid"]["n_height"]],
        "power_on_receiver_mw": {"soltrace": float(np.mean(powers)) / 1e6,
                                 "soltrace_std_error": float(np.std(powers, ddof=1) / math.sqrt(n)) / 1e6 if n > 1 else None,
                                 "analytic": case["analytic"]["power_on_receiver_w"] / 1e6,
                                 "cspaps_mcrt": case.get("cspaps_mcrt", {}).get("power_on_receiver_w", float("nan")) / 1e6},
        "peak_flux_kw_m2": {"soltrace_mean_map": float(flux.max()) / 1e3,
                            "soltrace_std_error": float(peaks.std(ddof=1) / math.sqrt(n)) / 1e3 if n > 1 else None,
                            "analytic": float(a.max()) / 1e3},
        "hotspot": {"soltrace_cell": [int(i) for i in np.unravel_index(int(np.argmax(flux)), flux.shape)],
                    "analytic_cell": [int(i) for i in np.unravel_index(int(np.argmax(a)), a.shape)],
                    "cell_order": "(height index, azimuth index)"},
        "flux_map_nrmse_analytic_vs_soltrace": float(np.sqrt(np.mean((a[lit] - flux[lit]) ** 2)) / flux[lit].mean()),
        "flux_map_correlation": float(np.corrcoef(a.ravel(), flux.ravel())[0, 1]),
        "difference_direction": "analytic minus SolTrace",
        "soltrace_run_time_s": times,
    }
    out["power_on_receiver_mw"]["rel_diff_analytic"] = out["power_on_receiver_mw"]["analytic"] / out["power_on_receiver_mw"]["soltrace"] - 1
    base = os.path.dirname(os.path.abspath(args.case))
    np.savetxt(os.path.join(base, "soltrace_flux_mean.csv"), flux / 1e3, delimiter=",", fmt="%.4f",
               header="kW/m2, rows = height (bottom->top), columns = azimuth (N=0, clockwise)")
    with open(os.path.join(base, "soltrace_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
