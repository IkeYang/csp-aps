"""How close is the annual value of K representative scenarios to the hourly truth -- with and without the
control variate (improvement 1)?

    python scripts/validate_annual_estimators.py [--case aksai_demo] [--mirrors 100] [--maps uniform clouds]
        [--ks 6 12 20 40] [--workers 4] [--out results/annual_estimators]
    python scripts/validate_annual_estimators.py --clear-sky [--days 10 20 40]    # the clear-sky design baseline

Clear-sky baseline: the truth is task 1 at every working hour of the whole clear-sky year (the same clear-sky
model and threshold), compared with the value on 10 / 20 / 40 representative days.

The reference solves task 1 (formal inner settings) at every hour of the weather year that the scenario sets
represent, with every mirror reading its own DNI from the hour's map: V_ref = sum_t w_t J(t). Then, for every
K, the formal V on K scenarios and its control-variate estimate are compared with V_ref (same layout, same
inner settings, same maps). Writes <out>/<case>_<maps>.json with the errors and the hourly table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cspaps import load_case
from cspaps.field import Field
from cspaps.layout_value import baseline_layout, evaluate_layout_value, inner_settings, solve_instant
from cspaps.parallel import pool
from cspaps.receiver import build_grid
from cspaps.scenarios import build_scenarios

ROOT = Path(__file__).resolve().parents[1]


def _hour_chunk(args):
    case, xyz, hourly, idx, settings = args
    fld = Field(xyz)
    grid = build_grid(case.receiver, settings.grid_shape[1], settings.grid_shape[0])
    out = []
    for i in idx:
        d = np.broadcast_to(np.asarray(hourly.dni_at(i, xyz[:, :2]), float), (len(xyz),)).copy()
        p = solve_instant(case, fld, float(hourly.elevation[i]), float(hourly.azimuth[i]), d, settings, grid,
                          keep_flux=False)
        out.append((i, p.status, p.p_abs_w, p.u_w))
    return out


def hourly_reference(case, xyz, sset, workers: int, say=print) -> pd.DataFrame:
    settings = inner_settings(case)
    hs = sset.hourly
    keep = np.flatnonzero(np.asarray(sset.assignment) >= 0)
    chunks = [keep[k::max(1, 8 * workers)] for k in range(max(1, 8 * workers))]
    rows = {}
    p = pool(workers)
    try:
        done = 0
        for out in p.imap_unordered(_hour_chunk, [(case, xyz, hs, c, settings) for c in chunks if len(c)]):
            for i, status, pa, u in out:
                rows[i] = (status, pa, u)
            done += 1
            say(f"  hourly reference: {done}/{len(chunks)} chunks")
    finally:
        p.close()
        p.join()
    idx = np.array(sorted(rows))
    return pd.DataFrame({"hour": idx, "weight_h": hs.weight_h[idx], "elevation_deg": hs.elevation[idx],
                         "dni_w_m2": hs.dni[idx], "status": [rows[i][0] for i in idx],
                         "p_abs_mw": [rows[i][1] / 1e6 for i in idx], "u_mw": [rows[i][2] / 1e6 for i in idx]})


def study(name: str, n_mirrors: int, maps: str, ks, workers: int, out: Path, clear_sky: bool = False) -> dict:
    case = load_case(ROOT / "cases" / f"{name}.yaml")
    case.scenarios.source = "clear_sky_days" if clear_sky else "weather"
    case.scenarios.map_model = maps
    case.layout_opt.workers = workers
    xyz, ids, _ = baseline_layout(case, n_mirrors)
    t0 = time.perf_counter()
    # the hours are the same for every K (weather year, or the whole clear-sky year)
    big = build_scenarios(case, n_days=max(ks)) if clear_sky else build_scenarios(case, max(ks))
    ref = hourly_reference(case, xyz, big, workers)
    lam = float(case.layout_opt.uniformity_weight)
    feas = ref["status"] == "feasible"
    v_ref = float((ref["weight_h"] * (ref["p_abs_mw"] - lam * ref["u_mw"]) * feas).sum())
    e_ref = float((ref["weight_h"] * ref["p_abs_mw"] * feas).sum())
    rows = []
    for k in ks:
        sset = build_scenarios(case, n_days=k) if clear_sky else build_scenarios(case, k)
        v = evaluate_layout_value(case, xyz, sset, f"K{k}", ids=ids, workers=workers, keep_flux=False)
        t = v.totals()
        rows.append({"k": k, "scenarios": len(sset), "value_mwh": t["value_mwh"], "value_cv_mwh": t["value_cv_mwh"],
                     "error_pct": 100 * (t["value_mwh"] / v_ref - 1), "error_cv_pct": 100 * (t["value_cv_mwh"] / v_ref - 1),
                     "energy_error_pct": 100 * (t["energy_mwh"] / e_ref - 1),
                     "energy_error_cv_pct": 100 * (t["energy_cv_mwh"] / e_ref - 1), "runtime_s": t["runtime_s"]})
        print(f"  K={k}: V {t['value_mwh']:.1f} ({rows[-1]['error_pct']:+.2f}%), "
              f"V_cv {t['value_cv_mwh']:.1f} ({rows[-1]['error_cv_pct']:+.2f}%), ref {v_ref:.1f}", flush=True)
    res = {"case": name, "maps": "clear_sky_days" if clear_sky else maps, "n_mirrors": int(len(xyz)), "lambda": lam,
           "hours": int(len(ref)),
           "value_ref_mwh": v_ref, "energy_ref_mwh": e_ref, "rows": rows, "runtime_s": time.perf_counter() - t0}
    out.mkdir(parents=True, exist_ok=True)
    tag = "clear_sky" if clear_sky else maps
    (out / f"{name}_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ref.to_csv(out / f"{name}_{tag}_hourly.csv", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="aksai_demo")
    ap.add_argument("--mirrors", type=int, default=100)
    ap.add_argument("--maps", nargs="+", default=["uniform", "clouds"])
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 12, 20, 40])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clear-sky", action="store_true", help="the clear-sky design baseline (representative days)")
    ap.add_argument("--days", type=int, nargs="+", default=[10, 20, 40], help="clear sky: representative days")
    ap.add_argument("--out", default=str(ROOT / "results" / "annual_estimators"))
    a = ap.parse_args()
    if a.clear_sky:
        print(f"=== {a.case}, {a.mirrors} mirrors, clear-sky design days {a.days}", flush=True)
        study(a.case, a.mirrors, "uniform", a.days, a.workers, Path(a.out), clear_sky=True)
        return
    for m in a.maps:
        print(f"=== {a.case}, {a.mirrors} mirrors, maps {m}", flush=True)
        study(a.case, a.mirrors, m, a.ks, a.workers, Path(a.out))


if __name__ == "__main__":
    main()
