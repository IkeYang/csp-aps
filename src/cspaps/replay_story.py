"""What happened during a cloud replay, in the terms an operator asks for.

From a ``realtime.RealtimeResult`` (rows per sample and strategy, frames with DNI, plans and flux maps):

* ``scorecard``: per strategy, time over the limit, time in the safety margin, highest verified peak, absorbed
  energy against the clear-sky energy of the same plans (what the clouds cost), mirror moves and defocus
  commands, time without the uniformity target;
* ``cloud_episodes`` / ``events``: when the clouds arrive and leave, the sharpest DNI drop and recovery, and
  per strategy the re-plans, defocus commands, fallbacks, limit exceedances and the worst peak;
* ``clearing_spotlight``: the moment the clouds clear (largest rise of the field's DNI) with the sample before
  and, per strategy, the worst verified sample of the control intervals around it -- the dangerous moment:
  the DNI is back while the plan made for the shadow is still in effect;
* ``panel_timeline``: per strategy, the highest flux / allowed flux of every panel at every sample (coarse map).

Every safety number comes from the rows' fine-grid verification (``peak_to_limit``, ``over_limit``); the coarse
flux maps are only used for the panel timeline, which says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COVER_ON = 0.02  # an episode starts when this fraction of the mirrors is under a cloud ...
DNI_STEP = 0.10  # a DNI drop / rise between two samples counts when the field mean moves this much of clear sky


def _one(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of one strategy (field-level columns are the same for every strategy)."""
    first = table["strategy"].iloc[0]
    return table[table["strategy"] == first].reset_index(drop=True)


def dni_ratio(table: pd.DataFrame) -> np.ndarray:
    """Field-mean DNI over clear-sky DNI per sample."""
    g = _one(table)
    clear = g["dni_clear_w_m2"].to_numpy(float) if "dni_clear_w_m2" in g else np.full(len(g), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(clear > 0, g["dni_mean_w_m2"].to_numpy(float) / clear, np.nan)


def cloud_episodes(table: pd.DataFrame, threshold: float = COVER_ON) -> list[dict]:
    """Stretches with at least ``threshold`` of the mirrors under a cloud: start, end (the first clear sample or
    the end of the replay), the largest cover and when it occurred."""
    g = _one(table)
    if "cloud_cover" not in g:
        return []
    cover = g["cloud_cover"].to_numpy(float)
    on = cover >= threshold
    out, j = [], 0
    while j < len(on):
        if not on[j]:
            j += 1
            continue
        s = j
        while j < len(on) and on[j]:
            j += 1
        e = j  # first sample after the episode (== len when it lasts to the end)
        peak = s + int(np.argmax(cover[s:e]))
        out.append({"start": g["time"].iloc[s], "end": g["time"].iloc[e] if e < len(g) else g["time"].iloc[-1],
                    "until_end": e >= len(g), "max_cover": float(cover[peak]), "max_cover_time": g["time"].iloc[peak],
                    "duration_s": float(g["interval_s"].iloc[s:e].sum())})
    return out


def scorecard(result, case) -> pd.DataFrame:
    """One row per strategy: the questions first (safe? what did the clouds cost? how much did the mirrors move?)."""
    margin = case.aps.safety_margin
    summ = result.summary()
    rows = []
    for strat, g in result.table.groupby("strategy", sort=False):
        d = g["interval_s"].to_numpy(float)
        ratio = g["peak_to_limit"].to_numpy(float)
        s = summ[strat]
        worst = int(np.argmax(ratio)) if len(ratio) else 0
        rows.append({
            "strategy": strat,
            "time_over_limit_s": float(d[ratio > 1.0 + 1e-9].sum()),
            "time_in_margin_s": float(d[(ratio > 1.0 - margin + 1e-9) & (ratio <= 1.0 + 1e-9)].sum()),
            "max_peak_to_limit": float(ratio.max()) if len(ratio) else 0.0,
            "max_peak_flux_kw_m2": float(g["peak_flux_kw_m2"].max()) if len(g) else 0.0,
            "max_peak_time": g["time"].iloc[worst] if len(g) else None,
            "absorbed_mwh": s["absorbed_energy_mwh"],
            "clear_sky_mwh": s.get("clear_sky_absorbed_energy_mwh"),
            "cloud_loss_fraction": s.get("cloud_loss_fraction"),
            "feasible_absorbed_mwh": s["feasible_absorbed_energy_mwh"],
            "heliostat_moves": s.get("heliostat_moves", int(g["n_moved"].sum())),
            "defocus_commands": s.get("defocus_commands", 0),
            "max_defocused": s["max_defocused"],
            "uniformity_failed_s": s["hours_uniformity_failed"] * 3600.0,
        })
    return pd.DataFrame(rows)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where ``mask`` is true."""
    out, j = [], 0
    while j < len(mask):
        if mask[j]:
            s = j
            while j < len(mask) and mask[j]:
                j += 1
            out.append((s, j))
        else:
            j += 1
    return out


def events(result, case, ids=None) -> pd.DataFrame:
    """Time-ordered events. Columns: time, strategy (None = the field), kind, value, detail.

    kinds: cloud_enter / cloud_leave / cloud_max (cover), dni_drop / dni_rise (the sharpest change of the
    field's DNI, fraction of clear sky), replan (mirrors re-aimed at a control instant), defocus (defocus
    commands), fallback (the engine fell back), over_limit / back_within_limit (verified peak above / back
    under the hard limit), margin_used (into the safety margin), worst_peak."""
    table = result.table
    margin = case.aps.safety_margin
    ev = []
    for e in cloud_episodes(table):
        ev.append({"time": e["start"], "strategy": None, "kind": "cloud_enter", "value": e["max_cover"], "detail": ""})
        ev.append({"time": e["max_cover_time"], "strategy": None, "kind": "cloud_max", "value": e["max_cover"],
                   "detail": ""})
        if not e["until_end"]:
            ev.append({"time": e["end"], "strategy": None, "kind": "cloud_leave", "value": 0.0, "detail": ""})
    g = _one(table)
    r = dni_ratio(table)
    if len(r) > 1 and np.isfinite(r).all():
        step = np.diff(r)
        j = int(np.argmin(step))
        if step[j] <= -DNI_STEP:
            ev.append({"time": g["time"].iloc[j + 1], "strategy": None, "kind": "dni_drop", "value": float(step[j]),
                       "detail": f"{r[j]:.2f}->{r[j + 1]:.2f}"})
        j = int(np.argmax(step))
        if step[j] >= DNI_STEP:
            ev.append({"time": g["time"].iloc[j + 1], "strategy": None, "kind": "dni_rise", "value": float(step[j]),
                       "detail": f"{r[j]:.2f}->{r[j + 1]:.2f}"})
    for strat, s in table.groupby("strategy", sort=False):
        s = s.reset_index(drop=True)
        ctl = s["control"].to_numpy(bool) if "control" in s else np.ones(len(s), bool)
        for j in np.flatnonzero(ctl & (s["n_moved"].to_numpy() > 0)):
            ev.append({"time": s["time"].iloc[j], "strategy": strat, "kind": "replan", "value": int(s["n_moved"].iloc[j]),
                       "detail": str(s["plan_reasons"].iloc[j] or "")})
        if "n_defocus_commands" in s:
            for j in np.flatnonzero(ctl & (s["n_defocus_commands"].to_numpy() > 0)):
                ev.append({"time": s["time"].iloc[j], "strategy": strat, "kind": "defocus",
                           "value": int(s["n_defocus_commands"].iloc[j]), "detail": ""})
        for j in np.flatnonzero(ctl & s["plan_state"].astype(str).str.contains("fallback|infeasible").to_numpy()):
            ev.append({"time": s["time"].iloc[j], "strategy": strat, "kind": "fallback", "value": None,
                       "detail": f"{s['plan_state'].iloc[j]} {s['plan_reasons'].iloc[j] or ''}".strip()})
        ratio = s["peak_to_limit"].to_numpy(float)
        for a, b in _runs(ratio > 1.0 + 1e-9):
            worst = a + int(np.argmax(ratio[a:b]))
            ev.append({"time": s["time"].iloc[a], "strategy": strat, "kind": "over_limit", "value": float(ratio[worst]),
                       "detail": f"{float(s['interval_s'].iloc[a:b].sum()):.0f}"})
            if b < len(s):
                ev.append({"time": s["time"].iloc[b], "strategy": strat, "kind": "back_within_limit",
                           "value": float(ratio[b]), "detail": ""})
        for a, b in _runs((ratio > 1.0 - margin + 1e-9) & (ratio <= 1.0 + 1e-9)):
            ev.append({"time": s["time"].iloc[a], "strategy": strat, "kind": "margin_used",
                       "value": float(ratio[a:b].max()), "detail": f"{float(s['interval_s'].iloc[a:b].sum()):.0f}"})
        if len(ratio):
            j = int(np.argmax(ratio))
            ev.append({"time": s["time"].iloc[j], "strategy": strat, "kind": "worst_peak", "value": float(ratio[j]),
                       "detail": f"{float(s['peak_flux_kw_m2'].iloc[j]):.0f}"})
    out = pd.DataFrame(ev, columns=["time", "strategy", "kind", "value", "detail"])
    order = {"cloud_enter": 0, "dni_drop": 1, "cloud_max": 2, "dni_rise": 3, "cloud_leave": 4, "replan": 5,
             "defocus": 6, "fallback": 7, "margin_used": 8, "over_limit": 9, "back_within_limit": 10, "worst_peak": 11}
    if len(out):
        out["_o"] = out["kind"].map(order)
        out = out.sort_values(["time", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return out


def clearing_spotlight(result, case) -> dict | None:
    """The moment the clouds clear: the largest one-sample rise of the field's DNI (at least ``DNI_STEP`` of clear
    sky). The risk window runs from that sample as long as the plan in effect was decided before the clearing
    (a plan decided at control instant c_k-1 is in effect during control interval k; the first plan at once).
    Per strategy: the worst verified sample of the window, its peak against the sample before the clearing,
    and the mirrors that contributed most to its hotspot."""
    table = result.table
    r = dni_ratio(table)
    if len(r) < 2 or not np.isfinite(r).all():
        return None
    step = np.diff(r)
    j = int(np.argmax(step))
    if step[j] < DNI_STEP:
        return None
    before, at = j, j + 1
    frames = result.frames
    g = _one(table)
    t_rise = g["time"].iloc[at]
    if "control_index" in g and "control" in g:
        ctl_time = dict(zip(g.loc[g["control"].astype(bool), "control_index"], g.loc[g["control"].astype(bool), "time"]))
        decided = [ctl_time.get(max(int(k) - 1, 0), g["time"].iloc[0]) for k in g["control_index"]]
        window = [i for i in range(at, len(g)) if decided[i] < t_rise]
    else:  # one sample per control instant: the plan in effect at the rise was decided one step earlier
        window = [at]
    window = window or [at]
    out = {"before": before, "at": at, "time": t_rise, "window_end": g["time"].iloc[window[-1]],
           "dni_ratio_before": float(r[before]), "dni_ratio_at": float(r[at]), "strategies": {}}
    for strat, s in table.groupby("strategy", sort=False):
        s = s.reset_index(drop=True)
        ratio = s["peak_to_limit"].to_numpy(float)
        w = int(window[int(np.argmax(ratio[window]))])
        check = (frames[w]["plans"][strat].get("check") or {}) if frames else {}
        out["strategies"][strat] = {
            "worst": w, "time": s["time"].iloc[w], "peak_to_limit": float(ratio[w]),
            "peak_flux_kw_m2": float(s["peak_flux_kw_m2"].iloc[w]), "over_limit": bool(ratio[w] > 1.0 + 1e-9),
            "before_peak_to_limit": float(ratio[before]), "plan_state": str(s["plan_state"].iloc[w]),
            "hotspot": check.get("hotspot"), "top_contributors": (check.get("top_contributors") or [])[:5],
        }
    return out


def panel_timeline(result, case) -> dict:
    """Per strategy: (n_samples, n_panels) highest flux / allowed flux of every panel on the coarse flux map of
    each sample (the verified peak of the whole receiver is ``peak_to_limit`` in the rows)."""
    from .receiver import build_grid

    grid = build_grid(case.receiver)
    panel = grid.to_map(grid.panel)
    limit = grid.to_map(grid.limit)
    n_p = case.receiver.n_panels
    out = {}
    for strat in (result.frames[0]["plans"] if result.frames else {}):
        m = np.zeros((len(result.frames), n_p))
        for j, f in enumerate(result.frames):
            ratio = np.asarray(f["plans"][strat]["flux"]) / limit
            for p in range(n_p):
                sel = panel == p
                m[j, p] = float(ratio[sel].max()) if sel.any() else 0.0
        out[strat] = m
    return out
