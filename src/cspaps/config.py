"""Case configuration: everything the user can change lives in one YAML file.

Each YAML section maps to one dataclass below. Unknown keys raise an error so
that typos in the input file do not silently fall back to defaults.
"""

from __future__ import annotations

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
    """External cylindrical receiver with a vertical axis."""

    center: tuple[float, float, float]  # centre of the cylinder (on its axis), m
    diameter: float  # m
    height: float  # m
    max_flux_kw_m2: float = 1000.0  # allowable peak flux density
    n_azimuth: int = 72  # flux-map cells around the circumference
    n_height: int = 32  # flux-map cells along the height

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def area(self) -> float:
        return math.pi * self.diameter * self.height


@dataclass
class Heliostat:
    width: float  # m
    height: float  # m
    pivot_height: float  # height of the rotation centre above ground, m
    reflectivity: float = 0.94
    cleanliness: float = 0.97
    slope_error_mrad: float = 1.5  # surface normal error, 1 sigma
    tracking_error_mrad: float = 1.0  # normal pointing error, 1 sigma
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
    sigma_mrad: float = 2.51  # effective sunshape standard deviation
    dni_model: str = "meinel"  # "meinel" (clear-sky) or "constant"
    dni_constant_w_m2: float = 900.0  # used when dni_model == "constant"


@dataclass
class FieldSpec:
    layout_file: str | None = None  # CSV with columns x,y[,z]; overrides generator
    generator: dict[str, Any] | None = None
    boundary: list[list[float]] | None = None  # polygon [[x, y], ...] in m
    exclusion_radius: float = 0.0  # no heliostats closer than this to the tower


@dataclass
class AnnualSpec:
    year: int = 2025
    day_of_month: int = 15  # one representative day per month
    time_step_min: float = 60.0
    min_sun_elevation_deg: float = 5.0


@dataclass
class Case:
    name: str
    site: Site
    receiver: Receiver
    heliostat: Heliostat
    sun: Sun = dc_field(default_factory=Sun)
    field: FieldSpec = dc_field(default_factory=FieldSpec)
    annual: AnnualSpec = dc_field(default_factory=AnnualSpec)
    base_dir: Path = dc_field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from the YAML relative to the YAML file's folder."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)


_SECTIONS = {
    "site": Site,
    "receiver": Receiver,
    "heliostat": Heliostat,
    "sun": Sun,
    "field": FieldSpec,
    "annual": AnnualSpec,
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
    return Case(name=name, base_dir=Path(base_dir or Path.cwd()), **parts)


def load_case(path: str | Path) -> Case:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return case_from_dict(data, base_dir=path.parent)
