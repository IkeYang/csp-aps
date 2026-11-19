"""Case configuration: everything the user can change lives in one YAML file.

Each YAML section maps to one dataclass below. Unknown keys raise an error so
that typos in the input file do not silently fall back to defaults.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Site:
    latitude: float  # deg, north positive
    longitude: float  # deg, east positive
    elevation_m: float = 0.0
    timezone: str = "UTC"  # IANA name, e.g. "Asia/Shanghai"


@dataclass
class Receiver:
    """External cylindrical receiver with a vertical axis, on top of a tower."""

    center: tuple[float, float, float]  # centre of the cylinder (on its axis), m
    diameter: float  # m
    height: float  # m
    max_flux_kw_m2: float = 1000.0  # allowable peak flux density
    n_azimuth: int = 72  # flux-map cells around the circumference
    n_height: int = 32  # flux-map cells along the height
    tower_diameter: float | None = None  # for the tower shadow; None = receiver diameter, 0 = off

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def area(self) -> float:
        return math.pi * self.diameter * self.height

    @property
    def tower_radius(self) -> float:
        d = self.diameter if self.tower_diameter is None else self.tower_diameter
        return 0.5 * d

    @property
    def top(self) -> float:
        return self.center[2] + 0.5 * self.height


@dataclass
class Heliostat:
    width: float  # m
    height: float  # m
    pivot_height: float  # height of the rotation centre above ground, m
    reflectivity: float = 0.94
    cleanliness: float = 0.97
    slope_error_mrad: float = 1.5  # surface normal error, 1 sigma per axis
    tracking_error_mrad: float = 1.0  # normal pointing error, 1 sigma per axis
    # "slant_range" = canted/focused at the slant range to the receiver,
    # "flat" = flat mirror, or a number = fixed focal length in metres.
    focusing: str | float = "slant_range"

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)


@dataclass
class Sun:
    shape: str = "gaussian"  # "gaussian" (sigma_mrad) or "pillbox" (half_angle_mrad)
    sigma_mrad: float = 2.51  # effective sunshape standard deviation (HFLCAL default)
    half_angle_mrad: float = 4.65  # pillbox sun disc half angle
    dni_model: str = "meinel"  # "meinel" (clear-sky) or "constant"
    dni_constant_w_m2: float = 900.0  # used when dni_model == "constant"

    @property
    def effective_sigma_mrad(self) -> float:
        """Per-axis standard deviation used by the analytic (HFLCAL) model."""
        if self.shape == "gaussian":
            return self.sigma_mrad
        if self.shape == "pillbox":
            return 0.5 * self.half_angle_mrad  # uniform disc: per-axis sigma = R / 2
        raise ValueError(f"Unknown sun shape '{self.shape}' (use 'gaussian' or 'pillbox')")


@dataclass
class Atmosphere:
    # "vittitoe_biggs_clear" (~23 km visibility), "vittitoe_biggs_hazy" (~5 km),
    # or "beer_lambert" with a user extinction coefficient.
    model: str = "vittitoe_biggs_clear"
    extinction_per_km: float = 0.1


@dataclass
class Weather:
    source: str = "clear_sky"  # "clear_sky" (uses sun.dni_model) or "file"
    file: str | None = None  # EPW, or CSV with columns time,dni (local time)


@dataclass
class Shading:
    enabled: bool = True
    n_points: int = 8  # mirror sampled with n_points x n_points points
    tower_shadow: bool = True


@dataclass
class FieldSpec:
    layout_file: str | None = None  # CSV with columns x,y[,z]; overrides generator
    generator: dict[str, Any] | None = None
    boundary: list[list[float]] | None = None  # polygon [[x, y], ...] in m
    exclusion_radius: float = 0.0  # no heliostats closer than this to the tower


@dataclass
class APSSpec:
    method: str = "greedy"  # "center" | "kfactor" | "greedy" | "milp"
    safety_margin: float = 0.05  # aim for at most (1 - margin) * max flux
    n_height_levels: int = 7  # candidate aim heights per heliostat
    height_span: float = 0.8  # candidate heights span this fraction of the receiver height
    azimuth_offsets_deg: list[float] = dc_field(default_factory=lambda: [-4.0, 0.0, 4.0])
    passes: int = 3  # improvement passes after the greedy construction
    tie_tolerance: float = 0.005  # candidates within this power fraction count as ties
    move_penalty_per_m: float = 0.0  # real-time smoothing: power fraction lost per m of aim shift
    allow_defocus: bool = True  # put heliostats on standby when no feasible aim exists
    milp_time_limit_s: float = 60.0


@dataclass
class Realtime:
    start: str = "2025-06-21 12:00"
    duration_min: float = 20.0
    step_s: float = 60.0
    forecast_horizon_s: float = 120.0  # constraint uses the max DNI expected over this horizon
    clouds: list[dict[str, Any]] | None = None  # explicit cloud list
    random_clouds: dict[str, Any] | None = None  # random cloud generator settings
    strategies: list[str] = dc_field(default_factory=lambda: ["center", "aps_reactive", "aps_predictive"])


@dataclass
class AnnualSpec:
    method: str = "sun_bins"  # "sun_bins" | "representative_days" | "hourly"
    n_sun_bins: int = 30
    year: int = 2025
    day_of_month: int = 15  # representative_days: one day per month
    time_step_min: float = 60.0
    min_sun_elevation_deg: float = 5.0
    aiming: str = "center"  # "center" (fast) or an APS method name


@dataclass
class LayoutOpt:
    generator: str = "radial_staggered"  # "radial_staggered" | "biomimetic"
    n_heliostats: int | None = None  # keep the best N candidates (None = keep all)
    bounds: dict[str, list[float]] = dc_field(default_factory=dict)  # optimised generator params
    fixed: dict[str, Any] = dc_field(default_factory=dict)  # other generator params
    n_sun_bins: int = 20
    shading_points: int = 5
    maxiter: int = 8
    popsize: int = 6
    seed: int = 1
    workers: int = 1  # parallel processes for the global search (-1 = all CPUs)
    polish_rounds: int = 4  # local pattern-search rounds after the global search
    refine_top: int = 3  # re-rank the best layouts with APS-aware annual energy


@dataclass
class MCRTSpec:
    rays_per_heliostat: int = 1000
    seed: int = 0


@dataclass
class Case:
    name: str
    site: Site
    receiver: Receiver
    heliostat: Heliostat
    sun: Sun = dc_field(default_factory=Sun)
    atmosphere: Atmosphere = dc_field(default_factory=Atmosphere)
    weather: Weather = dc_field(default_factory=Weather)
    shading: Shading = dc_field(default_factory=Shading)
    field: FieldSpec = dc_field(default_factory=FieldSpec)
    aps: APSSpec = dc_field(default_factory=APSSpec)
    realtime: Realtime = dc_field(default_factory=Realtime)
    annual: AnnualSpec = dc_field(default_factory=AnnualSpec)
    layout_opt: LayoutOpt = dc_field(default_factory=LayoutOpt)
    mcrt: MCRTSpec = dc_field(default_factory=MCRTSpec)
    base_dir: Path = dc_field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from the YAML relative to the YAML file's folder."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)

    def copy(self) -> "Case":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        for key in _SECTIONS:
            out[key] = dataclasses.asdict(getattr(self, key))
        out["receiver"]["center"] = list(out["receiver"]["center"])
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)


_SECTIONS = {
    "site": Site,
    "receiver": Receiver,
    "heliostat": Heliostat,
    "sun": Sun,
    "atmosphere": Atmosphere,
    "weather": Weather,
    "shading": Shading,
    "field": FieldSpec,
    "aps": APSSpec,
    "realtime": Realtime,
    "annual": AnnualSpec,
    "layout_opt": LayoutOpt,
    "mcrt": MCRTSpec,
}


def _build(cls, data: dict[str, Any] | None, section: str):
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(
            f"Unknown key(s) in section '{section}': {sorted(unknown)}. "
            f"Allowed: {sorted(names)}"
        )
    try:
        return cls(**data)
    except TypeError as exc:  # missing required keys
        raise ValueError(f"Section '{section}': {exc}") from exc


def _validate(case: Case) -> None:
    rec, hel = case.receiver, case.heliostat
    for name, value in [("receiver.diameter", rec.diameter), ("receiver.height", rec.height),
                        ("heliostat.width", hel.width), ("heliostat.height", hel.height),
                        ("receiver.max_flux_kw_m2", rec.max_flux_kw_m2)]:
        if not value > 0:
            raise ValueError(f"{name} must be > 0 (got {value})")
    if not 0 < hel.reflectivity <= 1 or not 0 < hel.cleanliness <= 1:
        raise ValueError("heliostat.reflectivity and cleanliness must be in (0, 1]")
    if not 0 <= case.aps.safety_margin < 1:
        raise ValueError("aps.safety_margin must be in [0, 1)")
    if case.aps.method not in ("center", "kfactor", "greedy", "milp"):
        raise ValueError(f"aps.method '{case.aps.method}' not in center|kfactor|greedy|milp")
    if case.annual.method not in ("sun_bins", "representative_days", "hourly"):
        raise ValueError(f"annual.method '{case.annual.method}' not in sun_bins|representative_days|hourly")
    if case.weather.source == "file" and not case.weather.file:
        raise ValueError("weather.source is 'file' but weather.file is not set")
    case.sun.effective_sigma_mrad  # raises on an unknown shape


def case_from_dict(data: dict[str, Any], base_dir: str | Path | None = None) -> Case:
    data = dict(data)
    name = str(data.pop("name", "case"))
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown)}")
    for required in ("site", "receiver", "heliostat"):
        if required not in data:
            raise ValueError(f"Missing required section '{required}'")
    parts = {key: _build(cls, data.get(key), key) for key, cls in _SECTIONS.items()}
    parts["receiver"].center = tuple(float(c) for c in parts["receiver"].center)
    case = Case(name=name, base_dir=Path(base_dir or Path.cwd()), **parts)
    _validate(case)
    return case


def load_case(path: str | Path) -> Case:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return case_from_dict(data, base_dir=path.parent)
