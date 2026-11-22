"""Spatial DNI maps over the site: rasters in site coordinates, shared by every layout.

A map is a function D(x, y) on the site, independent of the heliostat layout: a
heliostat at (x_i, y_i) reads D(x_i, y_i), so two layouts read the same value
at the same position and a heliostat moved elsewhere reads the new value.

Representation. Values on a regular raster of nodes (x_k, y_l) (m, x east,
y north, origin at the tower base). Sampling is bilinear between the four
surrounding nodes and clamped to the raster edge outside it (nearest edge
value). A bilinear interpolant on a rectangle is a convex combination of the
node values of the raster cells it touches, so the largest node value of
those cells bounds D anywhere in a square (``box_max``, used by the layout
upper bound).

Units. ``factor`` maps multiply the scenario DNI (W/m2); ``w_m2`` maps are the
DNI itself. A factor map normalised with ``site_mean`` has mean 1 over the
raster nodes inside the site (boundary polygon minus exclusion zone): the
station DNI is read as the site-mean DNI and the map only redistributes it.

Models (``scenarios.map_model``; all explicit assumptions, recorded with each map):
    uniform     f = 1 (compatibility: identical to the scalar-DNI path)
    half_swap   f_+ = 1 + c tanh(2u / w), f_- = 1 - c tanh(2u / w): one half of the site brighter,
                the other darker, and the swapped map; u = distance along the compass axis
                ``map_axis_deg`` from the tower, w the transition width
    gradient    f_+- = 1 +- c u / max|u|: linear brightening across the site and its mirror image
    clouds      products of soft-edged cloud-shadow discs (transmittance tau inside, smoothstep edge)
    file        CSV map_id,x,y,value on a regular grid, or scattered points (rasterised:
                Delaunay-linear inside the hull, nearest outside)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

CRS = "site ENU: x east, y north (m), origin at the tower base"
BILINEAR = "bilinear on raster nodes, clamped to the edge value outside the raster"


@dataclass
class DNIMap:
    map_id: str
    x: np.ndarray  # (nx,) node x, ascending, uniform spacing, m
    y: np.ndarray  # (ny,) node y, ascending, uniform spacing, m
    values: np.ndarray  # (ny, nx)
    unit: str = "factor"  # "factor" (x scenario DNI) | "w_m2"
    source: str = ""
    description: str = ""
    interpolation: str = BILINEAR
    crs: str = CRS
    normalization: str = "none"
    meta: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.values = np.asarray(self.values, dtype=float).reshape(self.y.size, self.x.size)
        if self.x.size < 1 or self.y.size < 1:
            raise ValueError(f"DNI map {self.map_id}: empty raster")
        for name, ax in (("x", self.x), ("y", self.y)):
            if ax.size > 1:
                d = np.diff(ax)
                if not np.all(d > 0) or not np.allclose(d, d[0], rtol=1e-6, atol=1e-6):
                    raise ValueError(f"DNI map {self.map_id}: {name} nodes must be ascending and equally spaced")
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError(f"DNI map {self.map_id}: values must be finite and nonnegative")

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _axis(nodes: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if nodes.size == 1:
            z = np.zeros(q.shape, dtype=int)
            return z, z, np.zeros(q.shape)
        step = nodes[1] - nodes[0]
        f = np.clip((q - nodes[0]) / step, 0.0, nodes.size - 1)
        i = np.minimum(np.floor(f).astype(int), nodes.size - 2)
        return i, i + 1, f - i

    def sample(self, xy) -> np.ndarray:
        """Bilinear value at points (N, 2) (clamped to the raster edge)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        i0, i1, tx = self._axis(self.x, xy[:, 0])
        j0, j1, ty = self._axis(self.y, xy[:, 1])
        v = self.values
        return ((1 - ty) * ((1 - tx) * v[j0, i0] + tx * v[j0, i1])
                + ty * ((1 - tx) * v[j1, i0] + tx * v[j1, i1]))

    def box_max(self, centers_xy, half_m: float) -> np.ndarray:
        """Upper bound of the sampled value anywhere in squares centre +- half_m (exact for bilinear)."""
        c = np.asarray(centers_xy, dtype=float).reshape(-1, 2)

        def span(nodes, lo, hi):
            if nodes.size == 1:
                z = np.zeros(lo.shape, dtype=int)
                return z, z
            step = nodes[1] - nodes[0]
            a = np.floor(np.clip((lo - nodes[0]) / step, 0, nodes.size - 1)).astype(int)
            b = np.ceil(np.clip((hi - nodes[0]) / step, 0, nodes.size - 1)).astype(int)
            return a, b

        ia, ib = span(self.x, c[:, 0] - half_m, c[:, 0] + half_m)
        ja, jb = span(self.y, c[:, 1] - half_m, c[:, 1] + half_m)
        out = np.full(len(c), -np.inf)
        wx = int((ib - ia).max()) if len(c) else 0
        wy = int((jb - ja).max()) if len(c) else 0
        for ox in range(wx + 1):
            ii = np.minimum(ia + ox, ib)
            for oy in range(wy + 1):
                out = np.maximum(out, self.values[np.minimum(ja + oy, jb), ii])
        return out

    # ------------------------------------------------------------------ bookkeeping
    @property
    def cell_m(self) -> float:
        return float(self.x[1] - self.x[0]) if self.x.size > 1 else float("nan")

    def to_frame(self) -> pd.DataFrame:
        gx, gy = np.meshgrid(self.x, self.y)
        return pd.DataFrame({"map_id": self.map_id, "x": gx.ravel(), "y": gy.ravel(), "value": self.values.ravel()})

    def record(self) -> dict:
        return {"map_id": self.map_id, "unit": self.unit, "source": self.source, "description": self.description,
                "interpolation": self.interpolation, "crs": self.crs, "normalization": self.normalization,
                "nx": int(self.x.size), "ny": int(self.y.size), "x_range_m": [float(self.x[0]), float(self.x[-1])],
                "y_range_m": [float(self.y[0]), float(self.y[-1])], "cell_m": self.cell_m,
                "min": float(self.values.min()), "max": float(self.values.max()),
                "mean_nodes": float(self.values.mean()), **self.meta}

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.map_id.encode())
        h.update(self.unit.encode())
        for a in (self.x, self.y, self.values):
            h.update(np.round(a, 9).astype(np.float64).tobytes())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- raster of the site

def site_raster(case, cell_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x nodes, y nodes, inside-site mask (ny, nx)) covering the site boundary (or r_max square) plus one cell."""
    from .field import default_r_max, site_mask

    rec = case.receiver
    if case.field.boundary is not None:
        b = np.asarray(case.field.boundary, dtype=float)
        x0, y0 = b.min(axis=0)
        x1, y1 = b.max(axis=0)
    else:
        r = default_r_max(case)
        x0, y0, x1, y1 = rec.center[0] - r, rec.center[1] - r, rec.center[0] + r, rec.center[1] + r
    nx = int(math.ceil((x1 - x0) / cell_m)) + 3
    ny = int(math.ceil((y1 - y0) / cell_m)) + 3
    xs = x0 - cell_m + cell_m * np.arange(nx)
    ys = y0 - cell_m + cell_m * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    inside = site_mask(case, pts)[0].reshape(gx.shape)
    return xs, ys, inside


def _normalise(values: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, float]:
    ref = values[inside] if inside.any() else values.ravel()
    mean = float(ref.mean())
    if mean <= 0:
        raise ValueError("cannot normalise a DNI map whose site mean is zero")
    return values / mean, mean


def _axis_name(deg: float) -> str:
    names = {0: "north", 45: "northeast", 90: "east", 135: "southeast", 180: "south", 225: "southwest",
             270: "west", 315: "northwest"}
    d = round(deg % 360.0, 6)
    return names.get(int(d), f"axis{d:g}") if float(d).is_integer() else f"axis{d:g}"


def model_maps(case) -> dict[str, DNIMap]:
    """Maps of the configured model (``scenarios.map_model``) on the site raster."""
    spec = case.scenarios
    out: dict[str, DNIMap] = {}
    xs, ys, inside = site_raster(case, spec.map_cell_m)
    gx, gy = np.meshgrid(xs, ys)
    cx, cy = case.receiver.center[0], case.receiver.center[1]
    theta = math.radians(spec.map_axis_deg)
    u = (gx - cx) * math.sin(theta) + (gy - cy) * math.cos(theta)
    norm = spec.map_normalize
    c = spec.map_contrast

    def add(map_id, values, desc, meta=None):
        vals = np.clip(values, 0.0, None)
        mean = None
        if norm == "site_mean":
            vals, mean = _normalise(vals, inside)
        out[map_id] = DNIMap(map_id, xs, ys, vals, "factor", f"model:{spec.map_model}", desc,
                             normalization=("site mean = 1 (raw mean %.4f)" % mean) if mean is not None else "none",
                             meta={"model": spec.map_model, **(meta or {})})

    if spec.map_model == "half_swap":
        w = max(spec.map_edge_m, 1e-6)
        side = _axis_name(spec.map_axis_deg)
        opp = _axis_name(spec.map_axis_deg + 180.0)
        add(f"{side}_bright", 1.0 + c * np.tanh(2.0 * u / w),
            f"{side} half brighter (+{c:.0%}), {opp} half darker; soft edge {w:g} m through the tower",
            {"contrast": c, "axis_deg": spec.map_axis_deg, "edge_m": w})
        add(f"{opp}_bright", 1.0 - c * np.tanh(2.0 * u / w),
            f"swapped: {opp} half brighter (+{c:.0%}), {side} half darker",
            {"contrast": c, "axis_deg": (spec.map_axis_deg + 180.0) % 360.0, "edge_m": w})
    elif spec.map_model == "gradient":
        umax = float(np.abs(u[inside]).max()) if inside.any() else float(np.abs(u).max())
        side = _axis_name(spec.map_axis_deg)
        opp = _axis_name(spec.map_axis_deg + 180.0)
        add(f"{side}_gradient", 1.0 + c * u / max(umax, 1e-9),
            f"linear rise towards {side}: +-{c:.0%} at the site edge", {"contrast": c, "axis_deg": spec.map_axis_deg})
        add(f"{opp}_gradient", 1.0 - c * u / max(umax, 1e-9),
            f"linear rise towards {opp}: +-{c:.0%} at the site edge", {"contrast": c})
    elif spec.map_model == "clouds":
        rng = np.random.default_rng(spec.seed)
        x_in, y_in = gx[inside], gy[inside]
        lo_x, hi_x = (x_in.min(), x_in.max()) if inside.any() else (xs[0], xs[-1])
        lo_y, hi_y = (y_in.min(), y_in.max()) if inside.any() else (ys[0], ys[-1])
        for k in range(spec.cloud_maps):
            f = np.ones_like(gx)
            clouds = []
            for _ in range(spec.cloud_n):
                r = rng.uniform(*spec.cloud_radius_m)
                tau = rng.uniform(*spec.cloud_transmittance)
                px, py = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
                edge = 0.3 * r
                d = np.hypot(gx - px, gy - py)
                s = np.clip((r - d) / edge + 0.5, 0.0, 1.0)
                s = s * s * (3 - 2 * s)
                f *= 1.0 - (1.0 - tau) * s
                clouds.append({"x": round(px, 2), "y": round(py, 2), "radius_m": round(r, 2),
                               "transmittance": round(tau, 3)})
            add(f"clouds_{k + 1:02d}", f, f"{spec.cloud_n} soft cloud shadows (seed {spec.seed}, map {k + 1})",
                {"clouds": clouds})
    elif spec.map_model == "file":
        out.update(load_map_file(case, case.resolve(spec.map_file), spec.map_unit, norm))
    if spec.map_model == "uniform" or spec.map_include_uniform or not out:
        out["uniform"] = uniform_map(xs, ys)
    return out


def uniform_map(xs=None, ys=None) -> DNIMap:
    xs = np.array([-1.0, 1.0]) if xs is None else np.asarray(xs)[[0, -1]]
    ys = np.array([-1.0, 1.0]) if ys is None else np.asarray(ys)[[0, -1]]
    if xs[0] == xs[-1]:
        xs = np.array([xs[0], xs[0] + 1.0])
    if ys[0] == ys[-1]:
        ys = np.array([ys[0], ys[0] + 1.0])
    return DNIMap("uniform", xs, ys, np.ones((2, 2)), "factor", "model:uniform",
                  "uniform DNI over the site (compatibility scenario)", normalization="site mean = 1")


def load_map_file(case, path, unit: str = "factor", normalize: str = "site_mean") -> dict[str, DNIMap]:
    """Map library CSV: map_id,x,y,value. Regular grids are used as given; scattered points are rasterised."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    df = pd.read_csv(path)
    if not {"map_id", "x", "y", "value"} <= set(df.columns):
        raise ValueError(f"{path}: DNI map library needs columns map_id,x,y,value")
    out = {}
    xs_site, ys_site, inside_site = site_raster(case, case.scenarios.map_cell_m)
    for map_id, g in df.groupby("map_id", sort=False):
        map_id = str(map_id)
        pts = g[["x", "y"]].to_numpy(float)
        val = g["value"].to_numpy(float)
        if not np.isfinite(pts).all() or not np.isfinite(val).all() or (val < 0).any():
            raise ValueError(f"{path}: map {map_id} needs finite coordinates and nonnegative values")
        ux, uy = np.unique(pts[:, 0]), np.unique(pts[:, 1])
        regular = (ux.size * uy.size == len(g) and len(np.unique(pts, axis=0)) == len(g)
                   and (ux.size < 2 or np.allclose(np.diff(ux), np.diff(ux)[0], rtol=1e-6))
                   and (uy.size < 2 or np.allclose(np.diff(uy), np.diff(uy)[0], rtol=1e-6)))
        if regular:
            ix = np.searchsorted(ux, pts[:, 0])
            iy = np.searchsorted(uy, pts[:, 1])
            grid = np.empty((uy.size, ux.size))
            grid[iy, ix] = val
            if ux.size == 1:
                ux, grid = np.array([ux[0], ux[0] + 1.0]), np.repeat(grid, 2, axis=1)
            if uy.size == 1:
                uy, grid = np.array([uy[0], uy[0] + 1.0]), np.repeat(grid, 2, axis=0)
            xs, ys, interp = ux, uy, BILINEAR
            gx, gy = np.meshgrid(xs, ys)
            q = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
            from .field import site_mask

            inside = site_mask(case, q)[0].reshape(gx.shape)
        else:
            xs, ys, inside = xs_site, ys_site, inside_site
            gx, gy = np.meshgrid(xs, ys)
            grid = NearestNDInterpolator(pts, val)(gx, gy)
            if len(pts) >= 3 and np.linalg.matrix_rank(pts - pts.mean(axis=0)) == 2:
                lin = LinearNDInterpolator(pts, val)(gx, gy)
                grid = np.where(np.isfinite(lin), lin, grid)
            interp = (f"rasterised from {len(pts)} scattered points (Delaunay-linear inside the hull, nearest "
                      f"outside) at {case.scenarios.map_cell_m:g} m; then " + BILINEAR)
        note = "none"
        if unit == "factor" and normalize == "site_mean":
            grid, mean = _normalise(np.clip(grid, 0.0, None), inside)
            note = "site mean = 1 (raw mean %.4f)" % mean
        out[map_id] = DNIMap(map_id, xs, ys, np.clip(grid, 0.0, None), unit, f"file:{Path(path).name}",
                             f"imported map {map_id}", interpolation=interp, normalization=note)
    return out


# --------------------------------------------------------------------------- library I/O

def save_library(maps: dict[str, DNIMap], folder) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frames = [m.to_frame() for m in maps.values()]
    csv = folder / "dni_maps.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv, index=False, float_format="%.17g")
    meta = folder / "dni_maps.json"
    meta.write_text(json.dumps({k: m.record() for k, m in maps.items()}, indent=2, ensure_ascii=False,
                               default=float), encoding="utf-8")
    return {"maps_csv": csv, "maps_json": meta}


def load_library(folder) -> dict[str, DNIMap]:
    folder = Path(folder)
    df = pd.read_csv(folder / "dni_maps.csv")
    meta = json.loads((folder / "dni_maps.json").read_text(encoding="utf-8"))
    out = {}
    for map_id, g in df.groupby("map_id", sort=False):
        map_id = str(map_id)
        xs, ys = np.unique(g["x"].to_numpy(float)), np.unique(g["y"].to_numpy(float))
        grid = np.empty((ys.size, xs.size))
        grid[np.searchsorted(ys, g["y"].to_numpy(float)), np.searchsorted(xs, g["x"].to_numpy(float))] = g["value"]
        m = meta.get(map_id, {})
        extra = {k: v for k, v in m.items() if k not in (
            "map_id", "unit", "source", "description", "interpolation", "crs", "normalization", "nx", "ny",
            "x_range_m", "y_range_m", "cell_m", "min", "max", "mean_nodes")}
        out[map_id] = DNIMap(map_id, xs, ys, grid, m.get("unit", "factor"), m.get("source", ""),
                             m.get("description", ""), m.get("interpolation", BILINEAR), m.get("crs", CRS),
                             m.get("normalization", "none"), extra)
    return out
