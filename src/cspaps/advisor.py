"""Which algorithm, how deep: recommendations by field size and time budget.

Every task offers three depths, from the same solvers with different budgets:

  fast      "快速"      a first answer: one greedy APS pass with the fine-grid flux check
  standard  "标准"      everyday use: joint-pose recheck, relaxation rounding and an optimality gap
  deep      "深度思考"  before a decision: denser candidate aims / more scenarios / more rounds of every step

and a scale tier from the number of heliostats (``TIERS``). ``recommend`` estimates the run time of each
depth from a per-heliostat time model (``Calibration``: built-in numbers measured on a 4-core machine, or a
calibration measured on this computer with ``cspaps calibrate``) and recommends the deepest depth that fits
the time budget. Nothing is hidden: ``mode_overrides`` lists every setting a depth changes, and the user can
override any of them afterwards.

Tasks: ``aim`` (problem 1, one instant), ``compare`` (problem 2, given layouts ranked by V(L)),
``search`` (problem 2, layout search + formal ranking).
"""

from __future__ import annotations

import json
import math
import os
import time as _time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

from .config import Case

MODES = ("fast", "standard", "deep")
MODE_ZH = {"fast": "快速", "standard": "标准", "deep": "深度思考"}
MODE_ICON = {"fast": "⚡", "standard": "◎", "deep": "◆"}
TASKS = ("aim", "compare", "search")
TASK_ZH = {"aim": "问题一 · 单时刻瞄准", "compare": "问题二 · 比较已有布局", "search": "问题二 · 搜索新布局"}

# what each depth means, in the user's words (shown next to the option)
MODE_TEXT = {
    "aim": {
        "fast": "先看方案：贪心 APS 一次求解，再用细网格完整光斑复核通量，不计算最优性差距。几秒出结果，适合调参数、看趋势。",
        "standard": "日常使用：在快速方案上按镜子的真实姿态复核镜间遮挡（联合姿态），再做 4 轮松弛取整，给出离最优还差多少。",
        "deep": "定方案前用（深度思考）：候选瞄准点加密一倍（周向 5 列 × 高度约 2 倍层数），松弛取整 16 轮、上界迭代时间加长、"
                "联合姿态多试探。每一步多想几轮，差距通常更小，时间是标准的数倍。",
    },
    "compare": {
        "fast": "粗看谁好：每个工况贪心 APS + 1 次联合姿态试探（晴空基线逐时计算全部代表日；气象年口径用 12 个工况）。"
                "两套布局差距明显时就够了。",
        "standard": "日常比较：每个工况 2 次联合姿态试探，对得分最接近的工况补算（气象年口径 20 个工况）。结论可复算、可导出。",
        "deep": "排名接近或要定稿时用（深度思考）：每个工况 4 次联合姿态试探并做松弛取整，再用加倍的代表日（气象年口径 60 个工况）"
                "核对采样误差。每个工况的瞄准更接近最优，年度分数更稳。",
    },
    "search": {
        "fast": "快速布局：初筛只跑 2 代差分进化，前 1 名与初始镜场正式排名（气象年口径 6 个工况），不做换位。几分钟看到一个可行布局。",
        "standard": "标准布局：初筛 4 代，前 2 名正式排名（气象年口径 12 个工况），按通量价格换位 1 轮（重新求解后 V 提高才接受）。",
        "deep": "深度思考布局：初筛 16 代、更大种群，前 5 名逐工况 APS 正式排名，通量价格换位 3 轮、镜位微调 8 次，"
                "再用加倍的代表日复核（气象年口径 20 + 40 个工况与独立云图）。搜索范围更大、入围布局算得更细，结论最稳，用于定稿。",
    },
}

# scale tiers by the number of heliostats (defaults; the time model decides what fits a budget)
TIERS = (
    ("tiny", 0, "验证级", "姿态组合很少，可以穷举全部组合，给出含镜间遮挡的候选集全局最优。用于验证和教学。"),
    ("small", 1_000, "小规模", "单个工况几秒。每一档都很快，时间预算宽松时直接用深度思考。"),
    ("medium", 5_000, "中等规模", "单个工况十几秒到半分钟。年度比较要算几十个工况，档位之间的时间差别明显。"),
    ("large", math.inf, "大规模", "单个工况分钟级。年度布局依靠多核并行和分级保真（先粗后精），深度思考需要较长预算。"),
)


def tier_of(n: int, n_candidates: int | None = None, exact_limit: int = 256) -> dict:
    """Scale tier of a field: tiny when every joint pose combination can be enumerated."""
    if n_candidates is not None and n > 0 and (n_candidates + 1) ** min(n, 20) <= exact_limit:
        key = "tiny"
    else:
        key = next(k for k, upper, *_ in TIERS[1:] if n <= upper)
    t = next(t for t in TIERS if t[0] == key)
    return {"key": key, "label": t[2], "text": t[3], "n_heliostats": int(n)}


# --------------------------------------------------------------------------- what a depth changes

_SEARCH = {
    "fast": ({"maxiter": 2, "popsize": 3, "n_sun_bins": 8, "polish_rounds": 1, "refine_top": 1, "review_n_sun_bins": 4,
              "swap_rounds": 1, "review_check_hourly": False, "formal_swap_rounds": 0, "dense_check_scenarios": 0,
              "formal_position_trials": 0, "ranking_refine_states": 0, "inner_joint_trials": 1,
              "inner_certify_rounds": 0}, {"n_scenarios": 6}),
    "standard": ({"maxiter": 4, "popsize": 4, "n_sun_bins": 12, "polish_rounds": 2, "refine_top": 2,
                  "review_n_sun_bins": 6, "swap_rounds": 1, "review_check_hourly": False, "formal_swap_rounds": 1,
                  "dense_check_scenarios": 0, "formal_position_trials": 4, "ranking_refine_states": 2,
                  "inner_joint_trials": 2, "inner_certify_rounds": 0}, {"n_scenarios": 12}),
    "deep": ({"maxiter": 16, "popsize": 8, "n_sun_bins": 24, "polish_rounds": 4, "refine_top": 5,
              "review_n_sun_bins": 12, "swap_rounds": 2, "review_check_hourly": True, "formal_swap_rounds": 3,
              "dense_check_scenarios": 40, "formal_position_trials": 8, "ranking_refine_states": 4,
              "inner_joint_trials": 4, "inner_certify_rounds": 2, "inner_certify_iters": 120,
              "cross_check_maps": True}, {"n_scenarios": 20}),
}
_COMPARE = {
    "fast": ({"inner_joint_trials": 1, "inner_certify_rounds": 0, "dense_check_scenarios": 0,
              "ranking_refine_states": 0}, {"n_scenarios": 12}),
    "standard": ({"inner_joint_trials": 2, "inner_certify_rounds": 0, "dense_check_scenarios": 0,
                  "ranking_refine_states": 2}, {"n_scenarios": 20}),
    "deep": ({"inner_joint_trials": 4, "inner_certify_rounds": 4, "inner_certify_iters": 120,
              "dense_check_scenarios": 60, "ranking_refine_states": 4}, {"n_scenarios": 40}),
}


def _denser_candidates(case: Case) -> dict:
    """Nested refinement of the candidate aims: every standard candidate stays a candidate."""
    aps = case.aps
    offs = sorted(float(v) for v in aps.azimuth_offsets_deg)
    if len(offs) >= 2:
        n = 2 * len(offs) - 1
        offs = [offs[0] + (offs[-1] - offs[0]) * i / (n - 1) for i in range(n)]
    return {"n_height_levels": 2 * int(aps.n_height_levels) - 1, "azimuth_offsets_deg": [round(v, 6) for v in offs]}


LARGE_N = 5_000  # above this, relaxation rounding buys next to nothing: deep spends the time elsewhere
DEEP_LARGE_NOTE = ("大规模镜场：贪心解已非常接近同一模型的上界（实测 2,000 余面镜时只差约 0.001%），松弛取整几乎没有收益，"
                   "所以深度思考不做松弛取整，把时间用在更多工况、更密的候选点和更多联合姿态复核上。")


def mode_overrides(case: Case, task: str, mode: str, workers: int | None = None, n: int | None = None) -> dict:
    """Settings a depth sets, as {section: {field: value}} (applied on top of the case).

    ``n`` (heliostats) adapts the deep depth of large fields (``LARGE_N``): no relaxation rounding."""
    if task not in TASKS or mode not in MODES:
        raise ValueError(f"task must be one of {TASKS} and mode one of {MODES}")
    workers = int(workers or recommended_workers())
    large = n is not None and n > LARGE_N
    if task == "aim":
        if mode == "fast":
            return {"aps": {"certify": False}}
        if mode == "standard":
            return {"aps": {"certify": True, "certify_rounds": 4, "certify_time_s": 15.0}}
        return {"aps": {"certify": True, "certify_rounds": 4 if large else 16, "certify_time_s": 120.0,
                        "coupled_trials": 60, **_denser_candidates(case)}}
    lo, sc = (_SEARCH if task == "search" else _COMPARE)[mode]
    lo = {**lo, "workers": workers}
    if mode == "deep" and large:
        lo.update(inner_certify_rounds=0, inner_joint_trials=max(2, lo.get("inner_joint_trials", 2) // 2))
    return {"layout_opt": lo, "scenarios": dict(sc)}


def apply_mode(case: Case, task: str, mode: str, workers: int | None = None, n: int | None = None) -> Case:
    c = case.copy()
    for section, values in mode_overrides(case, task, mode, workers, n).items():
        obj = getattr(c, section)
        for k, v in values.items():
            setattr(obj, k, v)
    return c


def recommended_workers() -> int:
    """Parallel workers for scenario-level work: all usable CPUs but one (``parallel.cpu_count``)."""
    from .parallel import cpu_count

    return max(1, cpu_count() - 1)


# --------------------------------------------------------------------------- time model

@dataclass
class Calibration:
    """Run time of the building blocks as t = a N^p seconds (N heliostats), one worker each.

    aim: one problem-1 run (``task_aim``) at each depth; inner: one scenario of the formal value
    (``solve_instant``); prescreen: one centre-aim annual evaluation per sun state (layout search). The cost
    per mirror grows with the field (farther mirrors make bigger spots on more receiver cells and have more
    shading neighbours), so p > 1; it is kept within [1, 1.3] because it falls again once the spots are
    clipped by the receiver. Built-in numbers: measured on a 4-core test machine with the demo receiver at
    427 and 1,049 heliostats (2026-11); ``calibrate`` replaces them with this computer's."""

    aim: dict = dc_field(default_factory=lambda: {'fast': [0.00135, 1.27], 'standard': [0.006495, 1.3], 'deep': [0.04637, 1.3]})
    inner: dict = dc_field(default_factory=lambda: {'fast': [0.001266, 1.3], 'standard': [0.001181, 1.3], 'deep': [0.004555, 1.3]})
    prescreen: list = dc_field(default_factory=lambda: [1.176e-05, 1.03])  # per sun state
    source: str = "built-in (4-core test machine, 2026-11)"
    cores: int = 4
    n_max: int = 1049  # largest calibrated field: beyond 2x this the estimate is an extrapolation
    measured: dict = dc_field(default_factory=dict)

    @staticmethod
    def _t(coef, n: int) -> float:
        a, p = coef
        return float(a) * max(int(n), 1) ** float(p)

    def t_aim(self, mode: str, n: int) -> float:
        return self._t(self.aim[mode], n)

    def t_inner(self, mode: str, n: int) -> float:
        return self._t(self.inner[mode], n)

    def t_prescreen(self, n: int, states: int) -> float:
        return self._t(self.prescreen, n) * states


def calibration_path() -> Path:
    return Path(os.environ.get("CSPAPS_CALIBRATION", Path.home() / ".cspaps" / "calibration.json"))


def load_calibration(path: str | Path | None = None) -> Calibration:
    p = Path(path) if path else calibration_path()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return Calibration(**{k: v for k, v in data.items() if k in Calibration.__dataclass_fields__})
        except (ValueError, TypeError):
            pass
    return Calibration()


def _field_size(case: Case, task: str) -> int:
    if task == "aim":
        from .field import build_field

        try:
            return build_field(case).n
        except Exception:  # noqa: BLE001 - an invalid field is reported elsewhere
            return int(case.layout_opt.n_heliostats or 0)
    if case.layout_opt.n_heliostats:
        return int(case.layout_opt.n_heliostats)
    from .field import build_field

    return build_field(case).n


def estimate_seconds(case: Case, task: str, mode: str, cal: Calibration | None = None, n: int | None = None,
                     workers: int | None = None, n_layouts: int = 2) -> dict:
    """Wall-clock estimate of one run at this depth: {"seconds", "low", "high", "parts"}."""
    cal = cal or load_calibration()
    n = int(n if n is not None else _field_size(case, task))
    workers = int(workers or recommended_workers())
    c = apply_mode(case, task, mode, workers, n)
    from .parallel import cpu_count

    eff = max(1.0, 0.8 * min(workers, cpu_count()))  # scenario-level parallel efficiency
    parts = {}
    if task == "aim":
        parts["solve"] = cal.t_aim(mode, n)
    else:
        n_s = _scenario_count(c)
        # deep without rounding (large fields) costs about the standard solve per scenario
        t_s = cal.t_inner("standard" if mode == "deep" and n > LARGE_N else mode, n)
        lo = c.layout_opt
        dense = int(lo.dense_check_scenarios or 0)
        if dense and c.scenarios.source == "clear_sky_days":  # the check uses twice as many days
            dense = 2 * n_s
        if task == "compare":
            parts["formal"] = n_layouts * n_s * t_s / min(eff, n_s)
            parts["refine"] = n_layouts * int(lo.ranking_refine_states or 0) * t_s
            parts["dense"] = n_layouts * dense * t_s / min(eff, max(dense, 1)) if dense else 0.0
        else:
            dim = 3 + (4 if lo.sector_spacing else 0)
            if lo.search_method == "bayes":  # every proposal: N selected on the pre-screen, then low-fidelity V
                per = (cal.t_prescreen(int(1.4 * n), lo.n_sun_bins)
                       + int(lo.bayes_scenarios) * cal.t_inner("fast", n))
                parts["prescreen"] = int(lo.bayes_evaluations) * per / eff
            else:
                evals = max(1, lo.popsize) * dim * (max(0, lo.maxiter) + 1) + 2 * dim * lo.polish_rounds
                parts["prescreen"] = evals * cal.t_prescreen(int(1.4 * n), lo.n_sun_bins) / eff
            ranked = int(lo.refine_top) + (1 if lo.baseline == "initial" else 0)
            parts["formal"] = ranked * n_s * t_s / min(eff, n_s)
            # a dual-priced swap round prices the free positions (about one layout) before the re-solve
            swap_cost = 2 if lo.swap_pricing == "dual" else 1
            extra = swap_cost * int(lo.formal_swap_rounds) + int(lo.formal_position_trials or 0)
            parts["swaps_and_moves"] = extra * n_s * t_s / min(eff, n_s)
            parts["refine"] = ranked * int(lo.ranking_refine_states or 0) * t_s
            parts["dense"] = 2 * dense * t_s / min(eff, max(dense, 1)) if dense else 0.0
            parts["cross_maps"] = (2 * n_s * t_s / min(eff, n_s)
                                   if lo.cross_check_maps and c.scenarios.map_model == "clouds" else 0.0)
    total = float(sum(parts.values()))
    extrapolated = n > 2 * cal.n_max
    return {"seconds": total, "low": 0.6 * total, "high": (2.5 if extrapolated else 1.6) * total, "parts": parts,
            "n_heliostats": n, "workers": workers, "extrapolated": extrapolated}


def _scenario_count(case: Case) -> int:
    """Scenarios one layout is solved on: clear-sky design days x working hours, a weather set, or a table."""
    sc = case.scenarios
    if sc.source == "clear_sky_days":
        try:
            from .sun import clear_sky_design_days

            return max(1, len(clear_sky_design_days(case)[0]))
        except Exception:  # noqa: BLE001 - estimate: about 11 working hours a day
            return 11 * int(sc.n_days)
    if sc.source == "file":
        return _file_scenarios(case)
    return int(sc.n_scenarios) * int(sc.dni_bands)


def _file_scenarios(case: Case) -> int:
    try:
        from .tables import read_user_csv

        return max(1, len(read_user_csv(case.resolve(case.scenarios.file))))
    except Exception:  # noqa: BLE001
        return int(case.scenarios.n_scenarios)


def _gpu_note() -> str:
    from .backend import backend_name, devices, enabled
    from .kernels import enabled as compiled

    cpu = ("均匀度局部搜索已编译运行（Numba，结果与 numpy 逐位相同）。" if compiled() else
           "未启用编译内核（未安装 Numba 或 CSPAPS_JIT=0），均匀度局部搜索按 numpy 运行。")
    if backend_name() == "cupy":
        devs = devices()
        return (f"光斑与通量计算使用 GPU（CuPy，{len(devs)} 张卡 {devs}，各工作进程轮流分配），"
                f"时间估计按 CPU 校准，实际会更短。" + cpu)
    if enabled():
        return "已设置 CSPAPS_GPU=1，但没有找到 CuPy 或 GPU，按 CPU 计算。" + cpu
    return "当前按 CPU 多进程计算；安装 CuPy 并设置 CSPAPS_GPU=1 后，光斑与通量计算可在 GPU 上进行。" + cpu


def fmt_duration(s: float) -> str:
    if s < 90:
        return f"{max(1, round(s))} 秒"
    if s < 5400:
        return f"{s / 60:.0f} 分钟"
    return f"{s / 3600:.1f} 小时"


def recommend(case: Case, task: str, budget_s: float = 900.0, cal: Calibration | None = None,
              workers: int | None = None, n_layouts: int = 2, n: int | None = None) -> dict:
    """Every depth with its meaning and time estimate; the deepest that fits the budget is recommended.

    ``n`` overrides the number of heliostats (what-if: "a 15,000-mirror field with this receiver")."""
    cal = cal or load_calibration()
    n = int(n) if n is not None else _field_size(case, task)
    k = len(case.aps.azimuth_offsets_deg) * int(case.aps.n_height_levels)
    tier = tier_of(n, k, case.aps.coupled_exact_limit)
    options = []
    for m in MODES:
        est = estimate_seconds(case, task, m, cal, n, workers, n_layouts)
        text = MODE_TEXT[task][m] + (" " + DEEP_LARGE_NOTE if m == "deep" and n > LARGE_N else "")
        options.append({"mode": m, "label": MODE_ZH[m], "icon": MODE_ICON[m], "text": text,
                        "seconds": est["seconds"], "range": [est["low"], est["high"]],
                        "fits": est["seconds"] <= budget_s, "parts": est["parts"],
                        "extrapolated": est["extrapolated"], "overrides": mode_overrides(case, task, m, workers, n)})
    fitting = [o for o in options if o["fits"]]
    rec = (fitting[-1] if fitting else options[0])["mode"]
    for o in options:
        o["recommended"] = o["mode"] == rec
    if fitting:
        reason = (f"{tier['label']}（{n:,} 面镜）：在 {fmt_duration(budget_s)} 的预算内，"
                  f"「{MODE_ZH[rec]}」预计 {fmt_duration(next(o for o in options if o['mode'] == rec)['seconds'])}，"
                  "是放得下的最深一档。")
    else:
        reason = (f"{tier['label']}（{n:,} 面镜）：连「快速」也预计超出 {fmt_duration(budget_s)} 的预算，"
                  "建议加大预算，或先用较少工况 / 较少镜子试算。")
    return {"task": task, "task_label": TASK_ZH[task], "tier": tier, "budget_s": float(budget_s),
            "options": options, "recommended": rec, "reason": reason, "calibration": cal.source,
            "workers": int(workers or recommended_workers()), "gpu": _gpu_note()}


# --------------------------------------------------------------------------- calibration on this computer

def fit_power(n, t, p_min: float = 1.0, p_max: float = 1.3) -> list:
    """t = a N^p: p from a log-log least-squares fit kept within [p_min, p_max], a matched at the largest N
    (the estimate is exact at the largest calibrated field and conservative to extrapolate)."""
    import numpy as np

    n = np.asarray(n, float)
    t = np.asarray(t, float)
    p = float(np.polyfit(np.log(n), np.log(t), 1)[0]) if n.size > 1 else 1.0
    p = min(max(p, p_min), p_max)
    k = int(np.argmax(n))
    return [float(t[k] / n[k] ** p), p]


def calibrate(case: Case, sizes=(400, 1200), out: str | Path | None = None, progress=None) -> Calibration:
    """Measure the time model on this computer: one sun state, fields of about ``sizes`` heliostats."""
    import tempfile

    import numpy as np

    from . import sun
    from .field import build_field
    from .layout_value import inner_settings, solve_instant
    from .simulate import annual
    from .workflows import task_aim

    say = progress or (lambda m: None)
    rows = []
    for target in sizes:
        c = case.copy()
        c.field.layout_file = None
        r_min = float(c.field.generator.get("r_min", 50.0)) if c.field.generator else 50.0
        # radial staggered density ~ N / (pi (r_max^2 - r_min^2)) at the case spacing: scale r_max to the target
        base = build_field(case)
        r0 = float(np.max(np.hypot(base.xyz[:, 0], base.xyz[:, 1]))) if base.n else 200.0
        r_max = math.sqrt(r_min ** 2 + (r0 ** 2 - r_min ** 2) * target / max(base.n, 1))
        c.field.generator = {**(c.field.generator or {"type": "radial_staggered", "r_min": r_min}), "r_max": r_max}
        c.field.boundary = None
        fld = build_field(c)
        row = {"n": fld.n}
        el, az, dni = 45.0, 180.0, 900.0
        with tempfile.TemporaryDirectory() as tmp:
            for m in MODES:
                say(f"calibration {fld.n} mirrors: problem 1, {m}")
                cm = apply_mode(c, "aim", m)
                cm.field.layout_file = None
                cm.weather.source = "clear_sky"
                t0 = _time.perf_counter()
                task_aim(cm, tmp, "2025-03-20 12:00", dni=dni)
                row[f"aim_{m}"] = _time.perf_counter() - t0
        for m in MODES:
            say(f"calibration {fld.n} mirrors: one scenario of the formal value, {m}")
            cm = apply_mode(c, "search", m)
            t0 = _time.perf_counter()
            solve_instant(cm, fld, el, az, dni, inner_settings(cm), keep_flux=False)
            row[f"inner_{m}"] = _time.perf_counter() - t0
        say(f"calibration {fld.n} mirrors: centre-aim annual pre-screen")
        states = sun.Samples(np.full(4, el), np.linspace(120, 240, 4), np.full(4, dni), np.full(4, 100.0))
        t0 = _time.perf_counter()
        annual(c, fld, samples=states, aiming="center", with_flux=False, workers=1)
        row["prescreen_per_state"] = (_time.perf_counter() - t0) / len(states)
        rows.append(row)

    def fit(key):
        return fit_power([r["n"] for r in rows], [r[key] for r in rows])

    cal = Calibration(aim={m: fit(f"aim_{m}") for m in MODES}, inner={m: fit(f"inner_{m}") for m in MODES},
                      prescreen=fit("prescreen_per_state"),
                      source=f"measured on this computer ({_time.strftime('%Y-%m-%d')}, {os.cpu_count()} cores)",
                      cores=int(os.cpu_count() or 1), n_max=int(max(r["n"] for r in rows)), measured={"rows": rows})
    if out is not False:
        p = Path(out) if out else calibration_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(cal), indent=2, ensure_ascii=False), encoding="utf-8")
    return cal
