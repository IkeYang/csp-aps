"""Real-time APS under moving cloud shadows.

Every ``step_s`` the controller reads the sun position and the DNI at each
heliostat, solves the APS, and the new aims are applied at the next step
(one step of latency, as on a real plant). Strategies compared:

    center          no APS: everyone aims at the facing mid-height point
    aps_reactive    APS with the DNI measured now
    aps_predictive  APS whose flux constraint uses the highest DNI expected over
                    the forecast horizon (clouds moving away), objective uses
                    the DNI now. Prevents over-flux when shadows leave.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
import pandas as pd

from .aps import optimize_aims
from .clouds import CloudField, cloud_field
from .config import Case
from .field import Field
from .receiver import build_grid, default_aim
from .simulate import Snapshot, evaluate, sun_state


@dataclass
class RealtimeResult:
    table: pd.DataFrame
    worst: dict[str, Snapshot]  # snapshot with the highest peak flux per strategy
    clouds: CloudField
    step_s: float
    limit_kw_m2: float
    extra: dict = dc_field(default_factory=dict)

    def summary(self) -> dict:
        out = {}
        dt_h = self.step_s / 3600.0
        for strat, g in self.table.groupby("strategy", sort=False):
            out[strat] = {
                "energy_mwh": round(float(g["p_receiver_mw"].sum() * dt_h), 3),
                "max_peak_flux_kw_m2": round(float(g["peak_flux_kw_m2"].max()), 1),
                "max_peak_to_limit": round(float(g["peak_flux_kw_m2"].max()) / self.limit_kw_m2, 3),
                "minutes_over_limit": round(float(g["over_limit"].sum() * self.step_s / 60.0), 1),
                "mean_solve_time_s": round(float(g["solve_time_s"].mean()), 3),
                "max_solve_time_s": round(float(g["solve_time_s"].max()), 3),
                "mean_heliostats_moved": round(float(g["n_moved"].mean()), 1),
                "max_defocused": int(g["n_defocused"].max()),
            }
        return out


def _moved(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], tol_m: float, radius: float) -> int:
    daz = np.radians((a[0] - b[0] + 180.0) % 360.0 - 180.0) * radius
    return int((np.hypot(daz, a[1] - b[1]) > tol_m).sum())


def run_realtime(
    case: Case,
    fld: Field,
    strategies: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RealtimeResult:
    rt = case.realtime
    strategies = strategies or rt.strategies
    for s in strategies:
        if s not in ("center", "aps_reactive", "aps_predictive"):
            raise ValueError(f"Unknown real-time strategy '{s}'")
    t0 = pd.Timestamp(sun_state(case, rt.start)[0])
    n_steps = int(round(rt.duration_min * 60.0 / rt.step_s)) + 1
    times = [t0 + pd.Timedelta(seconds=k * rt.step_s) for k in range(n_steps)]
    clouds = cloud_field(case, fld, t0)
    grid = build_grid(case.receiver)
    xy = fld.xyz[:, :2]
    rec = case.receiver
    center_aim = default_aim(rec, fld.xyz)
    ones = np.ones(fld.n, dtype=bool)
    applied = {s: (center_aim, ones) for s in strategies}
    pending: dict[str, tuple] = {}
    rows, worst = [], {}

    for step, t in enumerate(times):
        ts, el, az, dni_clear = sun_state(case, t)
        dni_now = clouds.dni(xy, ts, dni_clear)
        for strat in strategies:
            solve_time = 0.0
            if strat != "center":
                if step == 0:  # initial solve applied immediately
                    pending[strat] = _decide(case, fld, strat, el, az, dni_now, clouds, xy, ts, dni_clear,
                                             grid, None, None)
                aim, active, solve_time = pending[strat]
                n_moved = _moved(aim, applied[strat][0], 0.01, rec.radius) if step else 0
                applied[strat] = (aim, active)
            else:
                aim, active = center_aim, ones
                n_moved = 0
            snap = evaluate(case, fld, el, az, dni_now, aim=aim, active=active, grid=grid, time=ts)
            peak = float(snap.flux.max()) / 1e3
            rows.append({
                "time": ts, "strategy": strat, "dni_mean_w_m2": float(dni_now.mean()),
                "dni_min_w_m2": float(dni_now.min()), "p_receiver_mw": snap.p_receiver / 1e6,
                "peak_flux_kw_m2": peak, "over_limit": peak > rec.max_flux_kw_m2,
                "n_defocused": int((~active).sum()), "n_moved": n_moved, "solve_time_s": solve_time,
            })
            if strat not in worst or peak > worst[strat].flux.max() / 1e3:
                worst[strat] = snap
            if strat != "center" and step + 1 < n_steps:
                pending[strat] = _decide(case, fld, strat, el, az, dni_now, clouds, xy, ts, dni_clear,
                                         grid, aim, active)
        if progress:
            progress(step + 1, n_steps)
    return RealtimeResult(pd.DataFrame(rows), worst, clouds, rt.step_s, rec.max_flux_kw_m2)


def _decide(case, fld, strat, el, az, dni_now, clouds, xy, ts, dni_clear, grid, prev, prev_active):
    t0 = _time.perf_counter()
    dni_con = None
    if strat == "aps_predictive":
        dni_con = clouds.max_dni(xy, ts, case.realtime.forecast_horizon_s, dni_clear)
    res = optimize_aims(case, fld, el, az, dni_now, dni_constraint=dni_con, grid=grid,
                        prev=prev, prev_active=prev_active,
                        method=case.aps.method if case.aps.method != "center" else "greedy")
    return res.aim, res.active, _time.perf_counter() - t0
