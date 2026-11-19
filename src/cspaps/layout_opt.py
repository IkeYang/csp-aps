"""Outer loop: heliostat field layout optimisation.

Objective: maximise the annual energy on the receiver for a fixed number of
heliostats (``layout_opt.n_heliostats``) inside the site boundary.

1. A parametric generator (radial staggered or biomimetic spiral) fills the
   whole site with candidate positions for a given parameter vector.
2. Every candidate is evaluated over the year at 20-40 representative sun
   positions (DNI-weighted k-means of all daylight hours) with the fast
   single-point aiming rule, including shading, blocking and attenuation.
3. The best ``n_heliostats`` candidates by annual energy are kept and the
   kept field is re-evaluated (shading/blocking change when neighbours go).
4. Differential evolution searches the generator parameters.
5. The best few layouts are re-ranked with APS-aware annual energy: at every
   representative sun position the full APS enforces the flux limit, so the
   extra spillage of spreading the aim points is accounted for.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time as _time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
import pandas as pd

from . import sun
from .config import Case
from .field import Field, generate
from .receiver import build_grid
from .simulate import AnnualResult, annual

DEFAULT_BOUNDS = {
    "radial_staggered": {
        "radial_spacing": [0.866, 1.6],
        "radial_growth": [0.0, 1.5],
        "azimuthal_spacing": [1.0, 2.0],
    },
    "biomimetic": {
        "a": [0.55, 1.2],
        "b": [0.5, 0.62],
    },
}


@dataclass
class LayoutEval:
    params: dict
    n_candidates: int
    xyz: np.ndarray
    energy_mwh: float  # annual energy on the receiver, single-point aiming
    result: AnnualResult | None = dc_field(default=None, repr=False)
    aps_energy_mwh: float | None = None
    aps_result: AnnualResult | None = dc_field(default=None, repr=False)

    @property
    def n(self) -> int:
        return int(len(self.xyz))


def evaluate_layout(
    case: Case,
    kind: str,
    params: dict,
    samples: sun.Samples,
    n_select: int | None,
    shading_points: int | None = None,
) -> LayoutEval:
    try:
        xyz = generate(case, kind, params)
    except ValueError:
        return LayoutEval(params, 0, np.zeros((0, 3)), 0.0)
    if len(xyz) == 0:
        return LayoutEval(params, 0, xyz, 0.0)
    res = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                 shading_points=shading_points)
    n_cand = len(xyz)
    if n_select and n_cand > n_select:
        keep = np.sort(np.argsort(-res.heliostat_energy_mwh, kind="stable")[:n_select])
        xyz = xyz[keep]
        res = annual(case, Field(xyz), samples=samples, aiming="center", with_flux=False,
                     shading_points=shading_points)
    return LayoutEval(params, n_cand, xyz, res.energy_mwh["intercept"], res)


def _start_method() -> str:
    """forkserver/spawn are safe in multi-threaded hosts (GUI) but must re-import ``__main__``;
    fall back to fork when the main module has no file (interactive / stdin scripts)."""
    methods = mp.get_all_start_methods()
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file and os.path.exists(main_file):
        return "forkserver" if "forkserver" in methods else "spawn"
    return "fork" if "fork" in methods else "spawn"


class _Objective:
    """Picklable objective for scipy's differential evolution (supports workers > 1)."""

    def __init__(self, case: Case, kind: str, names: list[str], samples: sun.Samples):
        self.case, self.kind, self.names, self.samples = case, kind, names, samples

    def params(self, x) -> dict:
        p = dict(self.case.layout_opt.fixed)
        p.update({k: float(v) for k, v in zip(self.names, x)})
        return p

    def __call__(self, x) -> float:
        lo = self.case.layout_opt
        ev = evaluate_layout(self.case, self.kind, self.params(x), self.samples, lo.n_heliostats,
                             lo.shading_points)
        return -ev.energy_mwh


@dataclass
class LayoutOptResult:
    best: LayoutEval
    history: pd.DataFrame  # every evaluated parameter vector
    ranking: pd.DataFrame  # APS-aware re-ranking of the top layouts
    samples: sun.Samples
    runtime_s: float

    @property
    def field(self) -> Field:
        return Field(self.best.xyz)

    def summary(self) -> dict:
        b = self.best
        return {
            "generator_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in b.params.items()},
            "n_heliostats": b.n,
            "n_candidates": b.n_candidates,
            "annual_energy_mwh_center_aiming": round(b.energy_mwh, 1),
            "annual_energy_mwh_with_aps": round(b.aps_energy_mwh, 1) if b.aps_energy_mwh is not None else None,
            "n_evaluations": int(len(self.history)),
            "n_sun_bins": len(self.samples),
            "runtime_s": round(self.runtime_s, 1),
        }


def optimize_layout(
    case: Case,
    progress: Callable[[str], None] | None = None,
) -> LayoutOptResult:
    from scipy.optimize import differential_evolution

    t0 = _time.perf_counter()
    spec = case.layout_opt
    kind = spec.generator
    bounds = dict(DEFAULT_BOUNDS.get(kind, {}))
    bounds.update(spec.bounds or {})
    for k in spec.fixed:
        bounds.pop(k, None)
    if not bounds:
        raise ValueError("layout_opt: nothing to optimise (all parameters fixed)")
    names = list(bounds)
    say = progress or (lambda msg: None)
    samples = sun.sun_bins(sun.hourly_samples(case), spec.n_sun_bins, seed=spec.seed)
    say(f"{len(samples)} representative sun positions, optimising {names}")

    obj = _Objective(case, kind, names, samples)
    records: list[dict] = []

    def record(x, energy, stage):
        records.append({"evaluation": len(records) + 1, "stage": stage, **obj.params(x), "energy_mwh": energy})

    gen = {"n": 0}

    def callback(intermediate_result):
        gen["n"] += 1
        say(f"generation {gen['n']}: best so far {-intermediate_result.fun:.1f} MWh")

    n_workers = (os.cpu_count() or 1) if spec.workers == -1 else max(1, int(spec.workers))
    pool = mp.get_context(_start_method()).Pool(n_workers) if n_workers > 1 else None
    stage = {"name": "global"}

    def mapper(f, xs):
        xs = [np.asarray(x, dtype=float) for x in xs]
        vals = pool.map(f, xs) if pool is not None else [f(x) for x in xs]
        for x, v in zip(xs, vals):
            record(x, -v, stage["name"])
        return vals

    lo = np.array([bounds[k][0] for k in names], dtype=float)
    hi = np.array([bounds[k][1] for k in names], dtype=float)
    try:
        if n_workers > 1:
            workers, func = mapper, obj
        else:
            def func(x):
                return mapper(obj, [x])[0]
            workers = 1
        differential_evolution(
            func, list(zip(lo, hi)), maxiter=spec.maxiter, popsize=spec.popsize,
            seed=spec.seed, polish=False, tol=1e-4, init="sobol", workers=workers,
            updating="deferred" if n_workers > 1 else "immediate", callback=callback,
        )
        # local refinement: coordinate pattern search around the best point found
        stage["name"] = "local"
        best_i = int(np.argmax([r["energy_mwh"] for r in records]))
        x_best = np.array([records[best_i][k] for k in names], dtype=float)
        e_best = records[best_i]["energy_mwh"]
        step = 0.05 * (hi - lo)
        for r in range(spec.polish_rounds):
            trial = []
            for j in range(len(names)):
                for sgn in (-1.0, 1.0):
                    x = x_best.copy()
                    x[j] = np.clip(x[j] + sgn * step[j], lo[j], hi[j])
                    if not np.allclose(x, x_best):
                        trial.append(x)
            if not trial:
                break
            energies = -np.asarray(mapper(obj, trial))
            k = int(np.argmax(energies))
            if energies[k] > e_best:
                x_best, e_best = trial[k], float(energies[k])
                say(f"local refinement {r + 1}: improved to {e_best:.1f} MWh")
            else:
                step = step / 2
                say(f"local refinement {r + 1}: no improvement, step halved")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    history = pd.DataFrame(records).sort_values("energy_mwh", ascending=False, kind="stable").reset_index(drop=True)

    # APS-aware re-ranking of the best distinct layouts
    top = history.drop_duplicates(subset=names).head(max(1, spec.refine_top))
    grid = build_grid(case.receiver)
    evals: list[LayoutEval] = []
    rows = []
    for _, row in top.iterrows():
        params = {**spec.fixed, **{k: float(row[k]) for k in names}}
        ev = evaluate_layout(case, kind, params, samples, spec.n_heliostats, spec.shading_points)
        if spec.refine_top > 0 and ev.n > 0 and case.aps.method != "center":
            say(f"APS-aware annual energy for {params}")
            aps_res = annual(case, Field(ev.xyz), samples=samples, aiming=case.aps.method, grid=grid,
                             shading_points=spec.shading_points)
            ev.aps_result = aps_res
            ev.aps_energy_mwh = aps_res.energy_mwh["intercept"]
        evals.append(ev)
        rows.append({**{k: params[k] for k in names}, "n_heliostats": ev.n,
                     "energy_center_mwh": ev.energy_mwh, "energy_aps_mwh": ev.aps_energy_mwh})
    ranking = pd.DataFrame(rows)
    key = (lambda e: e.aps_energy_mwh) if all(e.aps_energy_mwh is not None for e in evals) else (lambda e: e.energy_mwh)
    best = max(evals, key=key)
    if len(ranking):
        ranking = ranking.sort_values("energy_aps_mwh" if best.aps_energy_mwh is not None else "energy_center_mwh",
                                      ascending=False).reset_index(drop=True)
    return LayoutOptResult(best, history, ranking, samples, _time.perf_counter() - t0)
