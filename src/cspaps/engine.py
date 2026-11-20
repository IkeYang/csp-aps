"""Online APS control engine: one call per control cycle.

    engine = APSEngine(case, field)
    resp = engine.step(ControlRequest(time=..., dni=..., dni_time=..., forecasts=[...],
                                      available=..., receipt=...))

Per cycle:
1. input checks: DNI completeness and freshness, forecast coverage of the
   window in which the plan will be in effect [t + latency, t + latency + period];
2. constraint DNI per heliostat: max over the forecast snapshots in that window
   (+ margin), never below the current DNI; conservative clear-sky DNI when the
   data are stale/missing or the forecast does not cover the window;
3. response set reused from the cache while the sun has moved less than
   ``response_refresh_deg`` (otherwise rebuilt), plus hotspot points learned
   in earlier cycles;
4. warm start from the last *executed* plan (receipt), repair of violations,
   re-optimisation of priority heliostats (large DNI change, near hotspots,
   defocused, newly available); reachability filter from the slew rate;
5. final check with the *current* sun geometry (shading recomputed at low sun)
   and the constraint DNI on an adaptive fine grid; hotspots become new
   constraints and the plan is repaired; if still not verified the configured
   fallback is applied (hold the previous plan if it verifies, else defocus the
   main contributors) and the state says so.

States: "ready" (verified, execute), "fallback" (a conservative plan replaced
the optimised one; reasons listed), "infeasible" (no verified plan: the control
side must apply its own safety policy). Plant-side interlocks stay independent.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from . import optics, shading, sun
from .aps import AimProblem, ResponseSet, build_response, make_problem, solve_greedy, solve_kfactor
from .config import Case
from .field import Field
from .receiver import build_grid, default_aim, surface_point
from .simulate import sun_state
from .verify import FluxCheck, verify_beams


@dataclass
class ForecastSnapshot:
    valid_time: pd.Timestamp
    dni: np.ndarray  # (N,) per heliostat, field order


@dataclass
class Receipt:
    """What the plant actually executed last cycle."""

    plan_id: int | None
    aim_azimuth: np.ndarray
    aim_z: np.ndarray
    active: np.ndarray
    executed: bool = True


@dataclass
class ControlRequest:
    time: pd.Timestamp
    dni: np.ndarray | None = None  # current DNI per heliostat (field order); None = no data
    dni_time: pd.Timestamp | None = None  # timestamp of the DNI measurement
    forecasts: list[ForecastSnapshot] = dc_field(default_factory=list)
    available: np.ndarray | None = None  # bool per heliostat
    receipt: Receipt | None = None
    sun_elevation: float | None = None  # optional override (else computed from time and site)
    sun_azimuth: float | None = None


@dataclass
class ControlResponse:
    plan_id: int
    time: pd.Timestamp
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp
    state: str  # "ready" | "fallback" | "infeasible"
    reasons: list[str]
    ids: np.ndarray
    aim_azimuth: np.ndarray
    aim_z: np.ndarray
    aim_xyz: np.ndarray
    active: np.ndarray
    defocus_reason: list[str]
    check: FluxCheck | None
    dni_basis: str  # how the constraint DNI was formed
    power_intercepted_w: float  # with the current DNI
    power_absorbed_w: float
    n_moved: int
    response_reused: bool
    timings: dict

    def summary(self) -> dict:
        return {
            "plan_id": self.plan_id, "time": self.time.isoformat(),
            "valid_from": self.valid_from.isoformat(), "valid_until": self.valid_until.isoformat(),
            "state": self.state, "reasons": self.reasons, "dni_basis": self.dni_basis,
            "power_intercepted_mw": round(self.power_intercepted_w / 1e6, 4),
            "power_absorbed_mw": round(self.power_absorbed_w / 1e6, 4),
            "n_defocused": int((~self.active).sum()), "n_moved": self.n_moved,
            "response_reused": self.response_reused,
            "timings_s": {k: round(v, 4) for k, v in self.timings.items()},
            "flux_check": _check_with_ids(self.check, self.ids),
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": self.ids, "active": self.active.astype(int), "aim_azimuth_deg": np.round(self.aim_azimuth, 4),
            "aim_z": np.round(self.aim_z, 4), "aim_x": np.round(self.aim_xyz[:, 0], 4),
            "aim_y": np.round(self.aim_xyz[:, 1], 4), "defocus_reason": self.defocus_reason,
        })


class APSEngine:
    def __init__(self, case: Case, fld: Field, method: str | None = None):
        self.case = case
        self.fld = fld
        self.method = method or (case.aps.method if case.aps.method in ("greedy", "kfactor") else "greedy")
        self.grid = build_grid(case.receiver)
        self._rs: ResponseSet | None = None
        self._plan_id = 0
        self._last_plan: tuple | None = None  # (aim_az, aim_z, active)
        self._last_dni: np.ndarray | None = None
        self._last_good_dni: tuple[pd.Timestamp, np.ndarray] | None = None
        self.stats = {"cold_builds": 0, "reuses": 0}

    # ------------------------------------------------------------------ helpers
    def _clear_sky(self, el: float) -> float:
        return float(sun.dni(el, self.case.sun, self.case.site))

    def _response(self, el: float, az: float) -> tuple[ResponseSet, bool]:
        rs = self._rs
        if rs is not None and rs.sun_shift_deg(el, az) <= self.case.engine.response_refresh_deg:
            self.stats["reuses"] += 1
            return rs, True
        self._rs = build_response(self.case, self.fld, el, az, self.grid)
        self.stats["cold_builds"] += 1
        return self._rs, False

    def _choice_from_plan(self, rs: ResponseSet, aim_az, aim_z, active) -> np.ndarray:
        """Nearest candidate index of an executed plan (-1 where defocused)."""
        d_az = (rs.cand_az - np.asarray(aim_az)[:, None] + 180.0) % 360.0 - 180.0
        dist = np.hypot(np.radians(d_az) * self.case.receiver.radius, rs.cand_z - np.asarray(aim_z)[:, None])
        k = np.argmin(dist, axis=1)
        return np.where(np.asarray(active, bool), k, -1)

    def _current_beams(self, el, az, aim_az, aim_z, active, dni_con, reshade: bool):
        """Beams of a plan with the *current* sun geometry (for the final check)."""
        case, fld, rec, hel = self.case, self.fld, self.case.receiver, self.case.heliostat
        s = sun.sun_vector(el, az)
        aim_xyz = surface_point(rec, aim_az, aim_z)
        normals, t, slant, cos_inc = optics.heliostat_geometry(fld.xyz, aim_xyz, s)
        if case.shading.enabled and reshade:
            sb = shading.shading_blocking(fld.xyz, normals, s, t, hel, rec, fld.pairs,
                                          n_points=case.shading.n_points,
                                          tower_shadow=case.shading.tower_shadow)["combined"]
        else:
            sb = self._rs.sb if self._rs is not None else np.ones(fld.n)
        sigma = optics.beam_sigma(slant, cos_inc, hel, case.sun)
        unit = hel.area * hel.reflectivity * hel.cleanliness * sb * cos_inc * optics.attenuation(slant, case.atmosphere)
        icpt = optics.intercept_fraction(rec, aim_xyz, t, sigma)
        on = np.asarray(active, bool)
        return aim_xyz, t, sigma, unit, icpt, on

    def _check(self, el, az, aim_az, aim_z, active, dni_con, margin) -> tuple[FluxCheck, dict]:
        reshade = el < self.case.engine.reshade_below_elevation_deg
        aim_xyz, t, sigma, unit, icpt, on = self._current_beams(el, az, aim_az, aim_z, active, dni_con, reshade)
        idx = np.flatnonzero(on)
        chk = verify_beams(self.case.receiver, aim_xyz[idx], t[idx], sigma[idx], unit[idx] * dni_con[idx], margin,
                           max_cells=self.case.aps.verify_max_cells).remap(idx)
        return chk, {"unit": unit, "icpt": icpt, "aim_xyz": aim_xyz}

    # ------------------------------------------------------------------ main
    def step(self, req: ControlRequest) -> ControlResponse:
        t_start = _time.perf_counter()
        es, case, n = self.case.engine, self.case, self.fld.n
        reasons: list[str] = []
        tm: dict[str, float] = {}
        now = pd.Timestamp(req.time)
        if req.sun_elevation is not None and req.sun_azimuth is not None:
            el, az = float(req.sun_elevation), float(req.sun_azimuth)
        else:
            _, el, az, _ = sun_state(case, now)
        clear = self._clear_sky(el)
        valid_from = now + pd.Timedelta(seconds=es.command_latency_s)
        valid_until = valid_from + pd.Timedelta(seconds=es.period_s)
        avail = np.ones(n, bool) if req.available is None else np.asarray(req.available, bool).copy()

        # 1-2: current and constraint DNI
        dni_now = None if req.dni is None else np.asarray(req.dni, dtype=float)
        stale = False
        if dni_now is None:
            stale, reasons = True, reasons + ["dni_missing"]
        elif dni_now.shape != (n,) or not np.all(np.isfinite(dni_now)) or np.any(dni_now < 0):
            stale, reasons = True, reasons + ["dni_invalid"]
            dni_now = None
        elif req.dni_time is not None and (now - pd.Timestamp(req.dni_time)).total_seconds() > es.max_dni_age_s:
            stale, reasons = True, reasons + ["dni_stale"]
        if dni_now is not None and not stale:
            self._last_good_dni = (now, dni_now)
        if stale:
            if es.stale_policy == "last" and self._last_good_dni is not None:
                dni_obj = self._last_good_dni[1]
                dni_con = np.maximum(dni_obj, clear)
            else:
                dni_obj = np.full(n, clear) if dni_now is None else dni_now
                dni_con = np.full(n, clear)
            basis = "clear_sky (stale or missing DNI)"
        else:
            dni_obj = dni_now
            window = [f for f in req.forecasts if valid_from <= pd.Timestamp(f.valid_time) <= valid_until]
            covers = bool(req.forecasts) and min(pd.Timestamp(f.valid_time) for f in req.forecasts) <= valid_from \
                and max(pd.Timestamp(f.valid_time) for f in req.forecasts) >= valid_until
            if req.forecasts and covers:
                fmax = np.max([np.asarray(f.dni, float) for f in window], axis=0) if window else dni_now
                dni_con = np.maximum(dni_now, fmax * (1.0 + es.forecast_margin))
                dni_con = np.minimum(dni_con, max(clear * (1.0 + es.forecast_margin), float(dni_now.max())))
                basis = f"forecast_max over {len(window)} snapshots (+{es.forecast_margin:.0%})"
            elif req.forecasts:
                reasons.append("forecast_does_not_cover_window")
                dni_con = np.full(n, max(clear, float(dni_now.max())))
                basis = "clear_sky (forecast coverage)"
            elif es.no_forecast_policy == "current":
                dni_con = dni_now.copy()
                basis = "current"
            else:
                dni_con = np.maximum(dni_now, clear)
                basis = "clear_sky (no forecast)"
        tm["input_s"] = _time.perf_counter() - t_start

        # 3: response set
        t0 = _time.perf_counter()
        rs, reused = self._response(el, az)
        prob: AimProblem = make_problem(rs, dni_obj, dni_con, avail)
        tm["response_s"] = _time.perf_counter() - t0

        # previous executed plan
        prev = None
        if req.receipt is not None and req.receipt.executed:
            prev = (np.asarray(req.receipt.aim_azimuth, float), np.asarray(req.receipt.aim_z, float),
                    np.asarray(req.receipt.active, bool))
        elif self._last_plan is not None:
            prev = self._last_plan
        if es.max_slew_deg_s > 0 and prev is not None:
            prob.reach = self._reachable(rs, prev, el, az)

        # 4: solve (warm when a previous plan exists)
        t0 = _time.perf_counter()
        deadline = t0 + es.timeout_s
        init = priority = None
        if prev is not None and self.method == "greedy":
            init = self._choice_from_plan(rs, prev[0], prev[1], prev[2])
            ref = self._last_dni if self._last_dni is not None else dni_obj
            change = np.abs(dni_obj - ref) / max(float(np.max(ref)), 1.0)
            priority = (change > es.priority_dni_change) | (init < 0) | ~prev[2]
        if self.method == "kfactor":
            choice, _ = solve_kfactor(prob)
            info = {}
        else:
            prev_xyz = surface_point(case.receiver, prev[0], prev[1]) if prev is not None else None
            choice, info = solve_greedy(prob, case.aps, prev_xyz, prev[2] if prev is not None else None,
                                        init=init, priority=priority, deadline=deadline)
        if info.get("timed_out"):
            reasons.append("solve_timeout")
        tm["solve_s"] = _time.perf_counter() - t0

        # 5: check with current geometry; hotspots -> constraints -> repair
        t0 = _time.perf_counter()
        aim_az, aim_z = prob.aims(choice)
        active = choice >= 0
        chk, beams = self._check(el, az, aim_az, aim_z, active, dni_con, prob.margin)
        rounds = 0
        while not chk.ok and chk.hotspots and rounds < case.aps.verify_max_rounds:
            rs.add_points([h[0] for h in chk.hotspots], [h[1] for h in chk.hotspots])
            choice, _ = solve_greedy(prob, case.aps, init=choice)
            aim_az, aim_z = prob.aims(choice)
            active = choice >= 0
            chk, beams = self._check(el, az, aim_az, aim_z, active, dni_con, prob.margin)
            rounds += 1
        state = "ready"
        defocus_reason = ["" if a else "flux limit" for a in active]
        for i in np.flatnonzero(~avail):
            defocus_reason[i] = "unavailable"
        if not chk.ok:
            reasons.append(f"check_{chk.status}")
            fb = self._fallback(el, az, prev, choice, prob, dni_con, chk)
            if fb is not None:
                aim_az, aim_z, active, chk, beams, why = fb
                state = "fallback"
                reasons.append(why)
                defocus_reason = [r if not a else "" for r, a in zip(defocus_reason, active)]
                defocus_reason = [r or ("" if a else "fallback") for r, a in zip(defocus_reason, active)]
            if not chk.ok and chk.status == "over_limit":
                state = "infeasible"
        elif stale or "forecast_does_not_cover_window" in reasons or "solve_timeout" in reasons:
            state = "fallback"  # verified, but built on conservative inputs
        tm["verify_s"] = _time.perf_counter() - t0

        # 6: output
        t0 = _time.perf_counter()
        p_int = float(np.sum(dni_obj * beams["unit"] * beams["icpt"] * active))
        alpha = case.receiver.absorptivity
        n_moved = 0
        if prev is not None:
            d_az = np.radians((aim_az - prev[0] + 180.0) % 360.0 - 180.0) * case.receiver.radius
            n_moved = int(np.sum((np.hypot(d_az, aim_z - prev[1]) > 0.01) | (active != prev[2])))
        self._plan_id += 1
        self._last_plan = (aim_az, aim_z, active)
        self._last_dni = dni_obj
        resp = ControlResponse(
            self._plan_id, now, valid_from, valid_until, state, reasons, self.fld.ids.copy(), aim_az, aim_z,
            beams["aim_xyz"], active, defocus_reason, chk, basis, p_int, p_int * alpha, n_moved, reused, tm,
        )
        tm["output_s"] = _time.perf_counter() - t0
        tm["total_s"] = _time.perf_counter() - t_start
        return resp

    def _reachable(self, rs: ResponseSet, prev, el, az) -> np.ndarray:
        """Candidates whose mirror-normal change from the executed aim fits the slew budget."""
        es = self.case.engine
        s = sun.sun_vector(el, az)
        prev_xyz = surface_point(self.case.receiver, prev[0], prev[1])
        n_prev, *_ = optics.heliostat_geometry(self.fld.xyz, prev_xyz, s)
        n_cand, *_ = optics.heliostat_geometry(self.fld.xyz[:, None, :], rs.aim_xyz, s)
        ang = np.degrees(np.arccos(np.clip(np.einsum("nkc,nc->nk", n_cand, n_prev), -1, 1)))
        budget = es.max_slew_deg_s * (es.command_latency_s + es.period_s)
        return ang <= budget + 1e-9

    def _fallback(self, el, az, prev, choice, prob, dni_con, chk):
        es = self.case.engine
        if es.fallback == "none":
            return None
        if es.fallback == "hold_or_defocus" and prev is not None:
            c2, b2 = self._check(el, az, prev[0], prev[1], prev[2], dni_con, prob.margin)
            if c2.ok:
                return prev[0], prev[1], prev[2], c2, b2, "fallback_hold_previous"
        aim_az, aim_z = prob.aims(choice)
        active = (choice >= 0).copy()
        c2, b2 = chk, None
        for _ in range(max(1, self.fld.n)):
            if c2.ok or not c2.contributors:
                break
            for i, _f in c2.contributors[:3]:
                active[i] = False
            c2, b2 = self._check(el, az, aim_az, aim_z, active, dni_con, prob.margin)
        if b2 is None:
            c2, b2 = self._check(el, az, aim_az, aim_z, active, dni_con, prob.margin)
        return aim_az, aim_z, active, c2, b2, "fallback_defocus_contributors"


def _check_with_ids(chk: FluxCheck | None, ids: np.ndarray) -> dict | None:
    if chk is None:
        return None
    out = chk.summary()
    out["top_contributors"] = [{"heliostat_id": int(ids[c["heliostat"]]), "flux_kw_m2": c["flux_kw_m2"]}
                               for c in out["top_contributors"]]
    return out


def request_from_dict(d: dict, fld: Field, timezone: str) -> ControlRequest:
    """Build a request from JSON: DNI as a list (field order) or a mapping {id: dni}."""
    def arr(v):
        if v is None:
            return None
        if isinstance(v, dict):
            out = np.full(fld.n, np.nan)
            out[fld.index_of([int(k) for k in v])] = [float(x) for x in v.values()]
            return out
        return np.asarray(v, dtype=float)

    def ts(v):
        return None if v is None else sun.to_times(v, timezone)[0]

    receipt = None
    if d.get("receipt"):
        r = d["receipt"]
        receipt = Receipt(r.get("plan_id"), arr(r["aim_azimuth"]), arr(r["aim_z"]),
                          np.asarray(arr(r["active"]), bool), bool(r.get("executed", True)))
    avail = d.get("available")
    return ControlRequest(
        time=ts(d["time"]), dni=arr(d.get("dni")), dni_time=ts(d.get("dni_time")),
        forecasts=[ForecastSnapshot(ts(f["valid_time"]), arr(f["dni"])) for f in d.get("forecasts", [])],
        available=None if avail is None else np.asarray(arr(avail), bool), receipt=receipt,
        sun_elevation=d.get("sun_elevation"), sun_azimuth=d.get("sun_azimuth"),
    )


def response_to_dict(r: ControlResponse) -> dict:
    out = r.summary()
    out["heliostats"] = [
        {"id": int(i), "active": bool(a), "aim_azimuth_deg": round(float(p), 4), "aim_z": round(float(z), 4),
         "aim_xyz": [round(float(v), 4) for v in xyz], "defocus_reason": why}
        for i, a, p, z, xyz, why in zip(r.ids, r.active, r.aim_azimuth, r.aim_z, r.aim_xyz, r.defocus_reason)
    ]
    return out


def default_center(case: Case, fld: Field):
    return default_aim(case.receiver, fld.xyz)
