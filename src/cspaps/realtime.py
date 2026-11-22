"""Replay of real-time APS under moving cloud shadows, driven by the control engine.

Time base (sample-and-hold): samples t_0 = 0, t_1 = step, ... and a final
sample at the end of the replay (also when the duration is not a whole
number of steps). Sample k represents the interval [t_k, t_k+1); the final
sample has zero length. Energy = sum P_k * (t_k+1 - t_k); time over the limit =
sum of the intervals whose sample is over the limit. A 3 min replay with 60 s
steps therefore has 3 intervals of 60 s (4 samples), a 150 s replay has
intervals of 60, 60 and 30 s.

A plan computed at t_k is in effect from t_k+1 (one step of latency, as on a
plant); the first plan is applied immediately. Strategies:
    center          no APS
    aps_reactive    engine without forecasts, constraint = DNI measured now
    aps_predictive  engine with spatially synchronised forecast snapshots covering
                    the window in which the plan will be in effect (+ margin)
Measured DNI can be delayed (``dni_delay_s``) or lost (``dni_dropout_s``) to
test the engine's freshness checks and fallbacks. Every applied plan is
evaluated with the *actual* DNI and verified on a fine grid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
import pandas as pd

from .clouds import CloudField, cloud_field
from .config import Case
from .engine import APSEngine, ControlRequest, ForecastSnapshot, Receipt
from .field import Field
from .receiver import build_grid, default_aim, region_mask
from .simulate import Snapshot, evaluate, sun_state
from .uniformity import uniformity_metrics

OVER_LIMIT_BASIS = ("sample-and-hold: the verified peak at sample t_k holds for [t_k, t_k+1); "
                    "time over limit = sum of those intervals")


def replay_times(t0: pd.Timestamp, duration_s: float, step_s: float) -> tuple[list[pd.Timestamp], np.ndarray]:
    """Sample times and the interval each sample represents (s)."""
    offs = list(np.arange(0.0, duration_s, step_s))
    if not offs or abs(offs[-1] - duration_s) > 1e-9:
        offs.append(float(duration_s))
    offs = np.asarray(offs)
    intervals = np.append(np.diff(offs), 0.0)
    return [t0 + pd.Timedelta(seconds=float(o)) for o in offs], intervals


@dataclass
class RealtimeResult:
    table: pd.DataFrame
    worst: dict[str, Snapshot]  # snapshot with the highest verified peak per strategy
    clouds: CloudField
    step_s: float
    duration_s: float
    frames: list[dict] = dc_field(default_factory=list, repr=False)  # per sample: DNI and plans (GUI timeline)
    extra: dict = dc_field(default_factory=dict)

    def summary(self) -> dict:
        out = {}
        for strat, g in self.table.groupby("strategy", sort=False):
            d = g["interval_s"].to_numpy()
            over = g["over_limit"].to_numpy(bool)
            lat = g["engine_total_s"].dropna().to_numpy()
            feasible = g["operational_feasible"].to_numpy(bool) if "operational_feasible" in g else ~over
            item = {
                "energy_mwh": round(float((g["p_intercepted_mw"] * d).sum() / 3600.0), 4),
                "absorbed_energy_mwh": round(float((g["p_absorbed_mw"] * d).sum() / 3600.0), 4),
                "feasible_absorbed_energy_mwh": round(float((g["feasible_absorbed_mw"] * d).sum() / 3600.0), 4)
                    if "feasible_absorbed_mw" in g else round(float((g.loc[feasible, "p_absorbed_mw"] * d[feasible]).sum() / 3600.0), 4),
                "hours_uniformity_failed": float(d[~g["uniformity_feasible"].to_numpy(bool)].sum()) / 3600.0
                    if "uniformity_feasible" in g else 0.0,
                "feasible_intervals": int(feasible.sum()),
                "max_peak_flux_kw_m2": round(float(g["peak_flux_kw_m2"].max()), 1),
                "max_peak_to_limit": round(float(g["peak_to_limit"].max()), 4),
                "max_exceedance_kw_m2": round(float(g["exceedance_kw_m2"].max()), 1),
                "time_over_limit_s": round(float(d[over].sum()), 1),
                "time_over_limit_min": round(float(d[over].sum()) / 60.0, 2),
                "replay_duration_s": round(float(d.sum()), 1),
                "over_limit_basis": OVER_LIMIT_BASIS,
                "mean_heliostats_moved": round(float(g["n_moved"].mean()), 1),
                "max_defocused": int(g["n_defocused"].max()),
            }
            if lat.size:
                item.update({
                    "engine_latency_p50_s": round(float(np.percentile(lat, 50)), 3),
                    "engine_latency_p95_s": round(float(np.percentile(lat, 95)), 3),
                    "engine_latency_max_s": round(float(lat.max()), 3),
                    "cold_builds": int((~g["response_reused"].dropna().astype(bool)).sum()),
                    "timeouts": int(g["plan_reasons"].fillna("").str.contains("solve_timeout").sum()),
                    "plan_states": g["plan_state"].value_counts().to_dict(),
                })
            out[strat] = item
        return out


def _engine_case(case: Case, strat: str) -> Case:
    c = copy.deepcopy(case)
    rt = c.realtime
    c.engine.command_latency_s = rt.step_s  # plan computed at t_k is in effect from t_k+1 ...
    c.engine.period_s = max(rt.forecast_horizon_s, rt.step_s)  # ... and must stay safe over the horizon
    if strat == "aps_reactive":
        c.engine.no_forecast_policy = "current"
    return c


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
    duration = rt.duration_min * 60.0
    times, intervals = replay_times(t0, duration, rt.step_s)
    clouds = cloud_field(case, fld, t0)
    grid = build_grid(case.receiver)
    xy = fld.xyz[:, :2]
    rec = case.receiver
    center_aim = default_aim(rec, fld.xyz)
    ones = np.ones(fld.n, dtype=bool)
    engines = {s: APSEngine(_engine_case(case, s), fld) for s in strategies if s != "center"}
    applied = {s: (center_aim[0], center_aim[1], ones, "n/a", "") for s in strategies}
    pending: dict[str, tuple] = {}
    rows, worst, frames = [], {}, []
    drop = rt.dni_dropout_s

    def clear_dni(t):
        return sun_state(case, t)[3]

    def measured(t):
        """DNI the controller receives at time t: (values, timestamp) or (None, None)."""
        off = (t - t0).total_seconds()
        if drop is not None and drop[0] <= off < drop[1]:
            return None, None
        tm = t - pd.Timedelta(seconds=rt.dni_delay_s)
        return clouds.dni(xy, tm, clear_dni(tm)), tm

    for step, (t, dt) in enumerate(zip(times, intervals)):
        ts, el, az, dni_clear = sun_state(case, t)
        dni_now = clouds.dni(xy, ts, dni_clear)
        frame = {"time": ts, "dni": dni_now, "sun_angles": (el, az), "plans": {}}
        for strat in strategies:
            eng_time = np.nan
            reused = np.nan
            if strat != "center":
                if step == 0:
                    pending[strat] = _decide(engines[strat], strat, case, fld, ts, measured, clouds, xy, None)
                aim_az, aim_z, active, state, reasons, eng_time, reused = pending[strat]
                prev = applied[strat]
                n_defocus_commands = int(np.count_nonzero(prev[2] & ~active)) if step else 0
                d_az = np.radians((aim_az - prev[0] + 180.0) % 360.0 - 180.0) * rec.radius
                n_moved = int(np.sum((np.hypot(d_az, aim_z - prev[1]) > 0.01) | (active != prev[2]))) if step else 0
                applied[strat] = (aim_az, aim_z, active, state, reasons)
            else:
                aim_az, aim_z, active, state, reasons = applied[strat]
                n_moved = 0
                n_defocus_commands = 0
            snap = evaluate(case, fld, el, az, dni_now, aim=(aim_az, aim_z), active=active, grid=grid, time=ts)
            chk = snap.verify(rec) if snap.beams is not None and len(snap.beams["power"]) else None
            peak_ratio = chk.peak_ratio if chk is not None else 0.0
            if case.uniformity.enabled:
                h_mask = region_mask(rec, case.uniformity, grid.cell_az, grid.cell_z)
                uni_metrics = uniformity_metrics(grid.to_map(snap.absorbed_flux), grid.to_map(h_mask))
                power_h_mw = float(snap.absorbed_flux.ravel()[h_mask].sum() * grid.cell_area / 1e6)
                if case.uniformity.mode == "weighted":
                    uni_met = power_h_mw >= case.uniformity.min_load_mw - 1e-9
                    uni_status = "weighted" if uni_met else "below_min_load"
                else:
                    uni_met = (power_h_mw >= case.uniformity.min_load_mw - 1e-9
                               and np.isfinite(uni_metrics["cv_2d"])
                               and uni_metrics["cv_2d"] <= case.uniformity.max_cv)
                    uni_status = "feasible" if uni_met else (
                        "below_min_load" if power_h_mw < case.uniformity.min_load_mw - 1e-9 else "not_reached")
            else:
                uni_metrics = {"cv_2d": float("nan"), "cv_circ": float("nan"), "cv_height": float("nan")}
                power_h_mw, uni_met, uni_status = snap.p_absorbed / 1e6, True, "disabled"
            flux_met = chk is not None and chk.status == "ok"
            operational = bool(flux_met and uni_met)
            rows.append({
                "time": ts, "interval_s": float(dt), "strategy": strat,
                "dni_mean_w_m2": float(dni_now.mean()), "dni_min_w_m2": float(dni_now.min()),
                "p_intercepted_mw": snap.p_receiver / 1e6, "p_absorbed_mw": snap.p_absorbed / 1e6,
                "peak_flux_kw_m2": chk.peak_flux_kw_m2 if chk is not None else 0.0,
                "peak_to_limit": peak_ratio, "over_limit": bool(peak_ratio > 1.0 + 1e-9),
                "exceedance_kw_m2": chk.exceedance_kw_m2 if chk is not None else 0.0,
                "hotspot_panel": chk.hotspot_panel if chk is not None else 0,
                "uniformity_status": uni_status, "uniformity_cv_2d": uni_metrics["cv_2d"],
                "uniformity_cv_circ": uni_metrics["cv_circ"], "uniformity_cv_height": uni_metrics["cv_height"],
                "uniformity_power_h_mw": power_h_mw, "uniformity_feasible": bool(uni_met),
                "flux_feasible": bool(flux_met), "operational_feasible": operational,
                "feasible_absorbed_mw": snap.p_absorbed / 1e6 if operational else 0.0,
                "n_defocused": int((~np.asarray(active)).sum()), "n_moved": n_moved,
                "n_defocus_commands": n_defocus_commands,
                "n_tracking_in_tower_shadow": int(np.sum(np.asarray(active, bool) & (snap.tower_fraction >= 1.0)))
                if snap.tower_fraction is not None else 0,
                "plan_state": state, "plan_reasons": reasons,
                "engine_total_s": eng_time, "response_reused": reused,
            })
            frame["plans"][strat] = {"aim_azimuth": aim_az, "aim_z": aim_z, "active": np.asarray(active),
                                     "tower_fraction": snap.tower_fraction, "flux": snap.flux,
                                     "check": chk.summary() if chk is not None else None,
                                     "state": state, "uniformity_status": uni_status,
                                     "uniformity_metrics": uni_metrics, "operational_feasible": operational}
            if strat not in worst or peak_ratio > worst[strat][0]:
                worst[strat] = (peak_ratio, snap)
            if strat != "center" and step + 1 < len(times):
                receipt = Receipt(None, aim_az, aim_z, np.asarray(active), True)
                pending[strat] = _decide(engines[strat], strat, case, fld, ts, measured, clouds, xy, receipt)
        frames.append(frame)
        if progress:
            progress(step + 1, len(times))
    return RealtimeResult(pd.DataFrame(rows), {k: v[1] for k, v in worst.items()}, clouds, rt.step_s, duration,
                          frames, {"engine_stats": {k: e.stats for k, e in engines.items()}})


def _decide(engine: APSEngine, strat: str, case: Case, fld: Field, ts, measured, clouds, xy, receipt):
    rt = case.realtime
    dni_meas, dni_time = measured(ts)
    forecasts = []
    if strat == "aps_predictive":
        es = engine.case.engine
        start = ts + pd.Timedelta(seconds=es.command_latency_s)
        n_snap = int(np.ceil(es.period_s / rt.forecast_step_s))
        for j in range(n_snap + 1):
            tf = start + pd.Timedelta(seconds=min(j * rt.forecast_step_s, es.period_s))
            forecasts.append(ForecastSnapshot(tf, clouds.dni(xy, tf, sun_state(case, tf)[3])))
    resp = engine.step(ControlRequest(time=ts, dni=dni_meas, dni_time=dni_time, forecasts=forecasts,
                                      receipt=receipt))
    return (resp.aim_azimuth, resp.aim_z, resp.active, resp.state, ";".join(resp.reasons),
            resp.timings["total_s"], bool(resp.response_reused))
