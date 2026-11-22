"""Cloud shadows on the field: per-heliostat DNI for real-time APS.

Clouds are modelled directly as their ground shadows: soft-edged discs that
drift with a constant velocity. Inside a shadow DNI is multiplied by the
cloud's transmittance. Measured DNI maps (e.g. from a sky camera or a
pyrheliometer network) can be loaded from CSV instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Case
from .field import Field


@dataclass
class Cloud:
    x: float  # shadow centre at t0, m
    y: float
    radius: float  # m
    transmittance: float = 0.2  # fraction of DNI that still reaches the ground
    vx: float = 0.0  # shadow velocity, m/s
    vy: float = 0.0
    edge: float = 30.0  # width of the soft edge, m
    t_start_s: float | None = None  # the cloud exists from this time (s after t0) ...
    t_end_s: float | None = None  # ... until this time (None = always); models sudden clearing


class CloudField:
    def __init__(self, clouds: list[Cloud], t0: pd.Timestamp):
        self.clouds = list(clouds)
        self.t0 = pd.Timestamp(t0)

    def transmittance(self, xy: np.ndarray, t) -> np.ndarray:
        dt = (pd.Timestamp(t) - self.t0).total_seconds()
        f = np.ones(len(xy))
        for c in self.clouds:
            if (c.t_start_s is not None and dt < c.t_start_s) or (c.t_end_s is not None and dt >= c.t_end_s):
                continue
            d = np.hypot(xy[:, 0] - (c.x + c.vx * dt), xy[:, 1] - (c.y + c.vy * dt))
            # smoothstep from 0 (outside) to 1 (inside) across the edge band
            s = np.clip((c.radius - d) / max(c.edge, 1e-6) + 0.5, 0.0, 1.0)
            s = s * s * (3 - 2 * s)
            f *= 1.0 - (1.0 - c.transmittance) * s
        return f

    def dni(self, xy: np.ndarray, t, dni_clear: float) -> np.ndarray:
        return dni_clear * self.transmittance(xy, t)

    def max_dni(self, xy: np.ndarray, t, horizon_s: float, dni_clear: float, n_samples: int = 7) -> np.ndarray:
        """Highest DNI each heliostat will see over [t, t + horizon] (perfect forecast)."""
        out = self.dni(xy, t, dni_clear)
        if horizon_s <= 0:
            return out
        for dt in np.linspace(0.0, horizon_s, n_samples)[1:]:
            out = np.maximum(out, self.dni(xy, pd.Timestamp(t) + pd.Timedelta(seconds=float(dt)), dni_clear))
        return out


def random_clouds(spec: dict, fld: Field, duration_s: float, center_xy=(0.0, 0.0)) -> list[Cloud]:
    """Random clouds that cross the field during the simulated window.

    spec keys: n (5), seed (0), radius_m [80, 250], transmittance [0.05, 0.4],
    speed_m_s (8), direction_deg (90 = moving towards the east), edge_m (40).
    """
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    n = int(spec.get("n", 5))
    r_lo, r_hi = spec.get("radius_m", [80.0, 250.0])
    tau_lo, tau_hi = spec.get("transmittance", [0.05, 0.4])
    speed = float(spec.get("speed_m_s", 8.0))
    direction = np.radians(float(spec.get("direction_deg", 90.0)))
    edge = float(spec.get("edge_m", 40.0))
    wind = np.array([np.sin(direction), np.cos(direction)])
    lateral = np.array([wind[1], -wind[0]])
    rel = fld.xyz[:, :2] - np.asarray(center_xy) if fld.n else np.zeros((1, 2))
    extent = float(np.hypot(rel[:, 0], rel[:, 1]).max()) + 1.0
    clouds = []
    for _ in range(n):
        radius = rng.uniform(r_lo, r_hi)
        # place the cloud upwind so that it reaches the field at a random moment of the window
        along = -rng.uniform(-extent - radius, extent + radius + speed * duration_s)
        across = rng.uniform(-extent, extent)
        pos = np.asarray(center_xy) + along * wind + across * lateral
        clouds.append(Cloud(float(pos[0]), float(pos[1]), radius, rng.uniform(tau_lo, tau_hi),
                            float(speed * wind[0]), float(speed * wind[1]), edge))
    return clouds


def scenario_clouds(case: Case, fld: Field) -> list[Cloud]:
    """Built-in replay scenarios.

    moving_edge  : a cloud bank much larger than the field drifts across it, so a
                   straight soft edge sweeps over the heliostats
    sudden_clear : a cloud covering half of the field disappears at mid-replay
    """
    rt = case.realtime
    rec = case.receiver
    cx, cy = rec.center[0], rec.center[1]
    extent = float(np.hypot(*(fld.xyz[:, :2] - [cx, cy]).T).max()) if fld.n else 300.0
    duration = rt.duration_min * 60.0
    if rt.scenario == "moving_edge":
        radius = 20 * extent
        speed = 2.0 * extent / max(duration, 1.0)  # the edge crosses the whole field during the replay
        return [Cloud(cx - radius - extent, cy, radius, transmittance=0.1, vx=speed, vy=0.0, edge=60.0)]
    if rt.scenario == "sudden_clear":
        return [Cloud(cx - 0.6 * extent, cy, 0.9 * extent, transmittance=0.1, edge=40.0,
                      t_end_s=0.5 * duration)]
    return []


def cloud_field(case: Case, fld: Field, t0: pd.Timestamp) -> CloudField:
    rt = case.realtime
    clouds = [Cloud(**c) for c in (rt.clouds or [])]
    clouds += scenario_clouds(case, fld)
    if rt.random_clouds and rt.scenario == "clouds":
        rec = case.receiver
        clouds += random_clouds(rt.random_clouds, fld, rt.duration_min * 60.0, rec.center[:2])
    return CloudField(clouds, t0)


def cloud_dni_snapshot(case: Case, fld: Field, t, dni_clear: float) -> np.ndarray:
    """Per-heliostat DNI at time t under the case's cloud scenario (clouds half-way through the replay window)."""
    t = pd.Timestamp(t)
    cf = cloud_field(case, fld, t - pd.Timedelta(seconds=0.5 * case.realtime.duration_min * 60.0))
    return cf.dni(fld.xyz[:, :2], t, dni_clear)


def load_dni_map(path, xy: np.ndarray) -> np.ndarray:
    """Per-heliostat DNI from a CSV of measurements with columns x,y,dni (linear, nearest outside)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    df = pd.read_csv(path)
    clean, problems = check_dni_table(df)
    if problems:
        raise ValueError(f"{Path(path).name}: " + " ".join(p["message"] for p in problems[:5]))
    if not len(clean):
        raise ValueError(f"{Path(path).name}: no DNI points")
    df = clean
    pts = df[["x", "y"]].to_numpy(float)
    val = df["dni"].to_numpy(float)
    if len(df) == len(xy) and np.allclose(pts, xy[:, :2], atol=1e-3):
        return np.clip(val, 0.0, None)
    nearest = NearestNDInterpolator(pts, val)(xy[:, 0], xy[:, 1])
    offsets = pts - pts.mean(axis=0)
    if len(df) >= 3 and np.linalg.matrix_rank(offsets) == 2:
        lin = LinearNDInterpolator(pts, val)(xy[:, 0], xy[:, 1])
        nearest = np.where(np.isfinite(lin), lin, nearest)
    elif len(df) >= 2 and np.any(offsets):
        axis = np.linalg.svd(offsets, full_matrices=False)[2][0]
        projected = pts @ axis
        order = np.argsort(projected)
        nearest = np.interp(xy[:, :2] @ axis, projected[order], val[order])
    return np.clip(nearest, 0.0, None)


def static_clouds(fld: Field, n: int = 2, radius_m=(60.0, 140.0), transmittance=(0.1, 0.35), seed: int = 0,
                  edge_fraction: float = 0.3) -> list[Cloud]:
    """Clouds whose shadows lie on the field at the moment of interest (centres drawn near random heliostats).

    Used by the single-instant before/after comparison: the shadows are frozen (zero velocity).
    """
    rng = np.random.default_rng(int(seed))
    if fld.n == 0 or n <= 0:
        return []
    out = []
    for _ in range(int(n)):
        r = float(rng.uniform(*radius_m))
        base = fld.xyz[int(rng.integers(fld.n)), :2]
        jitter = rng.normal(0.0, 0.25 * r, 2)
        out.append(Cloud(float(base[0] + jitter[0]), float(base[1] + jitter[1]), r,
                         float(rng.uniform(*transmittance)), 0.0, 0.0, max(1.0, edge_fraction * r)))
    return out


def cloud_transmittance(clouds: list[Cloud], xy: np.ndarray) -> np.ndarray:
    """Transmittance of frozen cloud shadows at the points (1 = clear)."""
    t0 = pd.Timestamp("2000-01-01")
    return CloudField(clouds, t0).transmittance(np.asarray(xy, dtype=float).reshape(-1, 2), t0)


DNI_DEFINITION = "直射辐照度（DNI，W/m²），表示垂直于太阳光线的单位面积收到多少直射功率"


def check_dni_table(df: pd.DataFrame, value_col: str = "dni") -> tuple[pd.DataFrame, list[dict]]:
    """Row-level check of a DNI point table (x, y, dni). Returns (rows that pass as floats, problems).

    Every problem names the data row (1-based, header not counted), the column, the raw value and what is
    expected, e.g. row 3: dni is 'missing' -> enter a nonnegative number in W/m2 and import again.
    """
    problems = []
    need = ["x", "y", value_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return df.iloc[0:0], [{"row": None, "column": ", ".join(missing), "value": None,
                               "message": f"文件缺少列 {', '.join(missing)}。需要表头 x,y,{value_col}（x 东、y 北，单位 m；"
                                          f"{value_col} 单位 W/m²）。"}]
    ok = np.ones(len(df), dtype=bool)
    for col in need:
        num = pd.to_numeric(df[col], errors="coerce")
        bad = num.isna() | ~np.isfinite(num.to_numpy(dtype=float, na_value=np.nan))
        if col == value_col:
            bad |= num < 0
        for i in np.flatnonzero(bad.to_numpy()):
            raw = df[col].iloc[i]
            unit = "W/m²" if col == value_col else "m"
            what = "非负数" if col == value_col else "数"
            problems.append({"row": int(i) + 1, "column": col, "value": raw,
                             "message": f"第 {i + 1} 行的 {col} 是 {raw}。请输入{what}，单位 {unit}，再重新导入。"})
        ok &= ~bad.to_numpy()
    clean = df.loc[ok, need].astype(float)
    return clean, problems


def coverage_of_points(points_xy: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """True where a heliostat lies inside the convex hull of the measurement points (interpolated, not extrapolated)."""
    from scipy.spatial import Delaunay

    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    q = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(pts) < 3 or np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return np.zeros(len(q), dtype=bool)
    return Delaunay(pts).find_simplex(q) >= 0
