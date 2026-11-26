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
            ctl = g[g["control"].astype(bool)] if "control" in g else g  # decisions happen at control instants
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
                "mean_heliostats_moved": round(float(ctl["n_moved"].mean()), 1),
                "heliostat_moves": int(ctl["n_moved"].sum()),
                "defocus_commands": int(ctl["n_defocus_commands"].sum()) if "n_defocus_commands" in ctl else 0,
                "max_defocused": int(g["n_defocused"].max()),
            }
            if "p_absorbed_clear_mw" in g:
                clear = float((g["p_absorbed_clear_mw"] * d).sum() / 3600.0)
                item["clear_sky_absorbed_energy_mwh"] = round(clear, 4)
                item["cloud_loss_fraction"] = round(1.0 - item["absorbed_energy_mwh"] / clear, 4) if clear > 0 else 0.0
            if lat.size:
                reasons = ctl["plan_reasons"].astype(object).where(ctl["plan_reasons"].notna(), "")
                item.update({
                    "engine_latency_p50_s": round(float(np.percentile(lat, 50)), 3),
                    "engine_latency_p95_s": round(float(np.percentile(lat, 95)), 3),
                    "engine_latency_max_s": round(float(lat.max()), 3),
                    "cold_builds": int((~g["response_reused"].dropna().astype(bool)).sum()),
                    "timeouts": int(reasons.astype(str).str.contains("solve_timeout").sum()),
                    "plan_states": ctl["plan_state"].value_counts().to_dict(),
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


def replay_samples(t0: pd.Timestamp, duration_s: float, step_s: float, substep_s: float = 0.0):
    """Sample times of a replay with sub-samples between the control instants.

    Control instants are ``replay_times(t0, duration_s, step_s)``; with ``0 < substep_s < step_s`` every control
    interval [c_k, c_k+1) is also sampled every ``substep_s`` (the plan decided for it is held, the DNI moves).
    Returns (times, intervals_s, control_index, is_control): sample j represents [t_j, t_j+1) (sample-and-hold),
    lies in control interval ``control_index[j]`` and is a control instant when ``is_control[j]``. Without
    sub-samples this is exactly ``replay_times`` (every sample a control instant)."""
    ctimes, _ = replay_times(t0, duration_s, step_s)
    coffs = [(c - t0).total_seconds() for c in ctimes]
    offs, cidx, is_ctl = [], [], []
    sub = float(substep_s or 0.0)
    for k, c in enumerate(coffs):
        offs.append(c)
        cidx.append(k)
        is_ctl.append(True)
        if k + 1 < len(coffs) and 0.0 < sub < step_s:
            s = c + sub
            while s < coffs[k + 1] - 1e-9:
                offs.append(s)
                cidx.append(k)
                is_ctl.append(False)
                s += sub
    offs = np.asarray(offs, dtype=float)
    intervals = np.append(np.diff(offs), 0.0)
    return ([t0 + pd.Timedelta(seconds=float(o)) for o in offs], intervals, np.asarray(cidx, dtype=int),
            np.asarray(is_ctl, dtype=bool))


def cloud_shapes(clouds: CloudField, t) -> list[dict]:
    """The cloud shadows present at time t: centre, radius, soft-edge width, transmittance and velocity."""
    dt = (pd.Timestamp(t) - clouds.t0).total_seconds()
    out = []
    for c in clouds.clouds:
        if (c.t_start_s is not None and dt < c.t_start_s) or (c.t_end_s is not None and dt >= c.t_end_s):
            continue
        out.append({"x": float(c.x + c.vx * dt), "y": float(c.y + c.vy * dt), "radius": float(c.radius),
                    "edge": float(c.edge), "transmittance": float(c.transmittance), "vx": float(c.vx),
                    "vy": float(c.vy), "ends_in_s": None if c.t_end_s is None else float(c.t_end_s - dt)})
    return out


SHADED_BELOW = 0.9  # a mirror counts as under the cloud when it receives less than this fraction of clear-sky DNI


def _evaluate_sample(case: Case, fld: Field, grid, ts, dni_now, dni_clear: float, plans: dict) -> dict:
    """Every strategy's plan in effect at one sample, evaluated with the actual DNI and verified on the fine grid.

    Per strategy: the row metrics (power, verified peak, uniformity), the clear-sky reference (the same plan and
    geometry with clear-sky DNI: every stage efficiency is DNI-independent, so P_i scales with DNI_i), the
    coarse flux map, tower shadow and the verified check summary."""
    rec = case.receiver
    _, el, az, _ = sun_state(case, ts)
    out = {}
    for strat, (aim_az, aim_z, active) in plans.items():
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
                uni_met = bool(power_h_mw >= case.uniformity.min_load_mw - 1e-9
                               and np.isfinite(uni_metrics["cv_2d"])
                               and uni_metrics["cv_2d"] <= case.uniformity.max_cv)
                uni_status = "feasible" if uni_met else (
                    "below_min_load" if power_h_mw < case.uniformity.min_load_mw - 1e-9 else "not_reached")
        else:
            uni_metrics = {"cv_2d": float("nan"), "cv_circ": float("nan"), "cv_height": float("nan")}
            power_h_mw, uni_met, uni_status = snap.p_absorbed / 1e6, True, "disabled"
        eta_all = np.prod([snap.eta[k] for k in snap.eta], axis=0)
        alpha = snap.alpha_eff if snap.alpha_eff is not None else np.full(fld.n, rec.absorptivity)
        p_clear = dni_clear * case.heliostat.area * eta_all
        out[strat] = {
            "p_intercepted_mw": snap.p_receiver / 1e6, "p_absorbed_mw": snap.p_absorbed / 1e6,
            "p_absorbed_clear_mw": float((p_clear * alpha).sum()) / 1e6,
            "peak_flux_kw_m2": chk.peak_flux_kw_m2 if chk is not None else 0.0, "peak_to_limit": peak_ratio,
            "over_limit": bool(peak_ratio > 1.0 + 1e-9),
            "exceedance_kw_m2": chk.exceedance_kw_m2 if chk is not None else 0.0,
            "hotspot_panel": chk.hotspot_panel if chk is not None else 0,
            "uniformity_status": uni_status, "uniformity_cv_2d": uni_metrics["cv_2d"],
            "uniformity_cv_circ": uni_metrics["cv_circ"], "uniformity_cv_height": uni_metrics["cv_height"],
            "uniformity_power_h_mw": power_h_mw, "uniformity_feasible": bool(uni_met),
            "flux_feasible": bool(chk is not None and chk.status == "ok"),
            "n_defocused": int((~np.asarray(active)).sum()),
            "n_tracking_in_tower_shadow": int(np.sum(np.asarray(active, bool) & (snap.tower_fraction >= 1.0)))
            if snap.tower_fraction is not None else 0,
            "_flux": snap.flux, "_tower": snap.tower_fraction,
            "_check": chk.summary() if chk is not None else None,
            "_uniformity_metrics": uni_metrics,
        }
    return out


def _replay_chunk(task):
    """Pool worker: evaluate a chunk of samples (all strategies) of a replay."""
    from .field import Field as _Field

    case, xyz, ids, items = task
    fld = _Field(xyz, ids)
    grid = build_grid(case.receiver)
    return [(j, _evaluate_sample(case, fld, grid, ts, dni_now, dni_clear, plans))
            for j, ts, dni_now, dni_clear, plans in items]


def run_realtime(
    case: Case,
    fld: Field,
    strategies: list[str] | None = None,
    progress: Callable[..., None] | None = None,
    workers: int | None = None,
) -> RealtimeResult:
    """Replay the scenario: (1) every strategy's plans, decided by its engine at the control instants (one
    control interval of latency, the first plan applied at once), then (2) every sample -- control instants and
    the ``realtime.substep_s`` sub-samples between them, at which the plan is held while the clouds move --
    evaluated with the actual DNI and verified on the fine grid. The samples are independent and run in
    parallel with ``workers`` processes (default ``annual.workers``)."""
    from .parallel import n_workers, pool

    rt = case.realtime
    strategies = strategies or rt.strategies
    for s in strategies:
        if s not in ("center", "aps_reactive", "aps_predictive"):
            raise ValueError(f"Unknown real-time strategy '{s}'")
    t0 = pd.Timestamp(sun_state(case, rt.start)[0])
    duration = rt.duration_min * 60.0
    ctimes, _ = replay_times(t0, duration, rt.step_s)
    times, intervals, cidx, is_ctl = replay_samples(t0, duration, rt.step_s, rt.substep_s)
    clouds = cloud_field(case, fld, t0)
    grid = build_grid(case.receiver)
    xy = fld.xyz[:, :2]
    rec = case.receiver
    center_aim = default_aim(rec, fld.xyz)
    ones = np.ones(fld.n, dtype=bool)
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

    # (1) plans per control interval: plans[strat][k] = (aim_az, aim_z, active, state, reasons, t_engine, reused)
    # one engine at a time (its response cache is GBs for a large field): decide all its plans, keep its
    # statistics, free it before the next one and before the sample workers start
    import gc

    plans, engine_stats = {}, {}
    n_dec = sum(len(ctimes) for s in strategies if s != "center")
    done_dec = 0
    for strat in strategies:
        if strat == "center":
            plans[strat] = [(center_aim[0], center_aim[1], ones, "n/a", "", np.nan, np.nan)] * len(ctimes)
            continue
        engine = APSEngine(_engine_case(case, strat), fld)
        seq = [_decide(engine, strat, case, fld, ctimes[0], measured, clouds, xy, None)]
        for k in range(len(ctimes) - 1):
            done_dec += 1
            if progress:
                progress(done_dec, n_dec, "engine plans")
            prev = seq[-1]
            receipt = Receipt(None, prev[0], prev[1], np.asarray(prev[2]), True)
            seq.append(_decide(engine, strat, case, fld, ctimes[k], measured, clouds, xy, receipt))
        done_dec += 1
        plans[strat] = seq
        engine_stats[strat] = engine.stats
        del engine
        gc.collect()
    moves = {}
    for strat in strategies:
        out = [(0, 0)]
        for k in range(1, len(ctimes)):
            (a0, z0, act0, *_), (a1, z1, act1, *_) = plans[strat][k - 1], plans[strat][k]
            d_az = np.radians((a1 - a0 + 180.0) % 360.0 - 180.0) * rec.radius
            n_moved = int(np.sum((np.hypot(d_az, z1 - z0) > 0.01) | (act1 != act0)))
            out.append((n_moved, int(np.count_nonzero(act0 & ~act1))))
        moves[strat] = out

    # (2) every sample: actual DNI, plan in effect, verified
    items = []
    for j, ts in enumerate(times):
        ts, _, _, dni_clear = sun_state(case, ts)
        dni_now = clouds.dni(xy, ts, dni_clear)
        k = int(cidx[j])
        items.append((j, ts, dni_now, float(dni_clear), {s: plans[s][k][:3] for s in strategies}))
    results: list = [None] * len(items)
    nw = min(n_workers(case.annual.workers if workers is None else workers), max(len(items), 1))
    if nw > 1 and len(items) > 1:
        chunks = [items[c::nw] for c in range(nw)]
        tasks = [(case, fld.xyz, fld.ids, ch) for ch in chunks]
        with pool(nw, probe=(_replay_chunk, tasks[0])) as p:
            done = 0
            for out in p.imap_unordered(_replay_chunk, tasks):
                for j, res in out:
                    results[j] = res
                done += len(out)
                if progress:
                    progress(done, len(items), "replay samples")
    else:
        for j, ts, dni_now, dni_clear, pl in items:
            results[j] = _evaluate_sample(case, fld, grid, ts, dni_now, dni_clear, pl)
            if progress:
                progress(j + 1, len(items), "replay samples")

    rows, frames, worst = [], [], {}
    for j, (_, ts, dni_now, dni_clear, _) in enumerate(items):
        k, ctl = int(cidx[j]), bool(is_ctl[j])
        _, el, az, _ = sun_state(case, ts)
        cover = float(np.mean(dni_now < SHADED_BELOW * dni_clear)) if dni_clear > 0 and fld.n else 0.0
        frame = {"time": ts, "dni": dni_now, "sun_angles": (el, az), "plans": {}, "control": ctl,
                 "control_index": k, "dni_clear": dni_clear, "clouds": cloud_shapes(clouds, ts)}
        for strat in strategies:
            aim_az, aim_z, active, state, reasons, eng_time, reused = plans[strat][k]
            r = results[j][strat]
            n_moved, n_defocus_commands = moves[strat][k] if ctl else (0, 0)
            operational = bool(r["flux_feasible"] and r["uniformity_feasible"])
            rows.append({
                "time": ts, "interval_s": float(intervals[j]), "strategy": strat, "control": ctl,
                "control_index": k, "dni_mean_w_m2": float(dni_now.mean()), "dni_min_w_m2": float(dni_now.min()),
                "dni_clear_w_m2": dni_clear, "cloud_cover": cover,
                **{key: v for key, v in r.items() if not key.startswith("_")},
                "operational_feasible": operational,
                "feasible_absorbed_mw": r["p_absorbed_mw"] if operational else 0.0,
                "n_moved": n_moved, "n_defocus_commands": n_defocus_commands,
                "plan_state": state, "plan_reasons": reasons if ctl else "",
                "engine_total_s": eng_time if ctl else np.nan, "response_reused": reused if ctl else np.nan,
            })
            frame["plans"][strat] = {"aim_azimuth": aim_az, "aim_z": aim_z, "active": np.asarray(active),
                                     "tower_fraction": r["_tower"], "flux": r["_flux"], "check": r["_check"],
                                     "state": state, "uniformity_status": r["uniformity_status"],
                                     "uniformity_metrics": r["_uniformity_metrics"],
                                     "operational_feasible": operational}
            if strat not in worst or r["peak_to_limit"] > worst[strat][0]:
                worst[strat] = (r["peak_to_limit"], j)
        frames.append(frame)
    # the worst sample of each strategy as a full snapshot (for the run's flux plot)
    worst_snaps = {}
    for strat, (_, j) in worst.items():
        _, ts, dni_now, _, pl = items[j]
        _, el, az, _ = sun_state(case, ts)
        aim_az, aim_z, active = pl[strat]
        worst_snaps[strat] = evaluate(case, fld, el, az, dni_now, aim=(aim_az, aim_z), active=active, grid=grid, time=ts)
    return RealtimeResult(pd.DataFrame(rows), worst_snaps, clouds, rt.step_s, duration, frames,
                          {"engine_stats": engine_stats, "substep_s": float(rt.substep_s or 0.0)})


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
