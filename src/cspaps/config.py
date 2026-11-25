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
    max_flux_kw_m2: float = 1000.0  # allowable peak flux density (default for every panel)
    n_azimuth: int = 72  # flux-map cells around the circumference
    n_height: int = 32  # flux-map cells along the height
    tower_diameter: float | None = None  # for the tower shadow; None = receiver diameter, 0 = off
    tower_height: float | None = None  # tower top above the ground, m; None = bottom of the receiver
    absorptivity: float = 0.95  # absorbed / incident flux
    flux_limit_basis: str = "incident"  # the flux limits apply to "incident" or "absorbed" flux
    # Tube panels: n_panels equal azimuth sectors, panel 1 starts at panel_start_azimuth_deg
    n_panels: int = 1
    panel_start_azimuth_deg: float = 0.0
    panel_max_flux_kw_m2: list[float] | None = None  # per panel limit (None = max_flux_kw_m2)
    panel_absorptivity: list[float] | None = None  # per panel (None = absorptivity)
    panel_flux_bias: list[float] | None = None  # measured / modelled flux per panel (feedback, 1 = trust model)
    limit_height_profile: list[float] | None = None  # relative allowed flux, bottom -> top, interpolated

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

    @property
    def bottom(self) -> float:
        return self.center[2] - 0.5 * self.height

    @property
    def tower_top(self) -> float:
        """Height of the tower top above the ground (the receiver sits on it unless set otherwise)."""
        return self.bottom if self.tower_height is None else float(self.tower_height)


@dataclass
class Heliostat:
    width: float  # m
    height: float  # m
    pivot_height: float  # height of the rotation centre above ground, m
    reflectivity: float = 0.94
    cleanliness: float = 0.97
    slope_error_mrad: float = 1.5  # surface normal error, 1 sigma per axis
    tracking_error_mrad: float = 1.0  # normal pointing error, 1 sigma per axis
    specular_error_mrad: float = 0.0  # specularity: scatter of the reflected ray, 1 sigma per axis
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


# settings holding input file paths, resolved against the YAML's folder (``Case.resolve``)
FILE_SETTINGS = (("weather", "file"), ("field", "layout_file"), ("scenarios", "file"), ("scenarios", "map_file"),
                 ("scenarios", "validation_file"))


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
    method: str = "greedy"  # "center" | "kfactor" | "greedy" | "milp" | "subfield"
    safety_margin: float = 0.05  # aim for at most (1 - margin) * max flux
    n_height_levels: int = 7  # candidate aim heights per heliostat
    height_span: float = 0.8  # candidate heights span this fraction of the receiver height
    azimuth_offsets_deg: list[float] = dc_field(default_factory=lambda: [-4.0, 0.0, 4.0])
    passes: int = 3  # improvement passes after the greedy construction
    tie_tolerance: float = 0.005  # candidates within this power fraction count as ties
    move_penalty_per_m: float = 0.0  # real-time smoothing: power fraction lost per m of aim shift
    allow_defocus: bool = True  # put heliostats on standby when no feasible aim exists
    milp_time_limit_s: float = 60.0
    milp_max_nnz: int = 4_000_000  # refuse MILPs with more flux coefficients (memory / time guard)
    verify: bool = True  # re-check every solution on an adaptive fine grid with full beams
    auto_grid: bool = True  # refine the solver grid so that cells are <= sigma_min / 1.5 (capped)
    max_solver_cells: int = 40_000
    verify_max_rounds: int = 4  # add hotspots as constraints and re-solve at most this often
    verify_max_cells: int = 400_000  # cap of the verification grid size
    # Optimality certificate of single-instant plans (Lagrangian + Frank-Wolfe upper bound,
    # relaxation-guided rounding). Not used inside annual / layout loops.
    certify: bool = True
    certify_time_s: float = 15.0  # time budget of the upper-bound iterations
    certify_rounds: int = 4  # randomised roundings of the relaxed solution
    coupled_exact_limit: int = 256  # enumerate all joint mirror poses up to this many assignments
    coupled_trials: int = 18  # additional joint-pose trials for larger offline problems
    subfield_sectors: int = 4  # method "subfield": azimuth sectors solved independently (parallelisable)


@dataclass
class UniformitySpec:
    """Uniform heating of the preheater region H (absorbed flux, area-weighted)."""

    enabled: bool = False
    mode: str = "cv_limit"  # "cv_limit": maximise power under max_cv; "weighted": fixed P - weight * A_H * std_H(q)
    max_cv: float = 0.5  # epsilon: allowed 2-D coefficient of variation over H
    weight: float = 0.2  # dimensionless uniformity penalty weight for mode="weighted"
    panels: list[int] | None = None  # 1-based panel numbers forming H (None = all panels)
    z_range: list[float] | None = None  # [z_min, z_max] of H in m (None = full height)
    min_load_mw: float = 0.0  # minimum absorbed power on H; below it the receiver is shut down
    mu_iters: int = 12  # bisection steps on the uniformity weight
    passes: int = 3  # local-search passes per weight


@dataclass
class EngineSpec:
    """Online control engine (one APS solve per control cycle)."""

    period_s: float = 60.0  # control cycle: a plan is held for this long
    command_latency_s: float = 5.0  # from the request time until the plan is in effect
    max_dni_age_s: float = 90.0  # older DNI measurements count as stale
    forecast_margin: float = 0.05  # uplift on forecast DNI for forecast error
    no_forecast_policy: str = "clear_sky"  # constraint DNI without forecasts: "clear_sky" | "current"
    stale_policy: str = "clear_sky"  # constraint DNI when the measurement is stale/missing: "clear_sky" | "last"
    response_refresh_deg: float = 0.3  # rebuild the cached responses after this sun movement
    reshade_below_elevation_deg: float = 90.0  # recompute shading in the final check below this sun elevation (90 = always)
    max_slew_deg_s: float = 0.0  # heliostat normal slew rate for the reachability filter (0 = off)
    timeout_s: float = 20.0  # solve budget per cycle
    priority_dni_change: float = 0.05  # warm start re-optimises heliostats whose DNI changed more
    fallback: str = "hold_or_defocus"  # when the check fails: "hold_or_defocus" | "defocus" | "none"


@dataclass
class Realtime:
    start: str = "2025-06-21 12:00"
    duration_min: float = 20.0
    step_s: float = 60.0
    forecast_horizon_s: float = 120.0  # predictive strategy: forecast window after the command takes effect
    forecast_step_s: float = 15.0  # spacing of the forecast snapshots
    clouds: list[dict[str, Any]] | None = None  # explicit cloud list
    random_clouds: dict[str, Any] | None = None  # random cloud generator settings
    scenario: str = "clouds"  # "clouds" | "moving_edge" | "sudden_clear"
    dni_delay_s: float = 0.0  # measured DNI reaches the controller this late
    dni_dropout_s: list[float] | None = None  # [start, end] s after start: no DNI data arrives
    strategies: list[str] = dc_field(default_factory=lambda: ["center", "aps_reactive", "aps_predictive"])


@dataclass
class AnnualSpec:
    method: str = "sun_bins"  # "sun_bins" | "sun_grid" | "representative_days" | "hourly"
    n_sun_bins: int = 30
    dni_bands: int = 1  # split every representative state into this many DNI bands
    grid_elevation_step_deg: float = 10.0  # sun_grid bin size
    grid_azimuth_step_deg: float = 20.0
    workers: int = 1  # parallel processes for annual runs (-1 = all CPUs)
    year: int = 2025
    day_of_month: int = 15  # representative_days: one day per month
    time_step_min: float = 60.0
    min_sun_elevation_deg: float = 5.0
    min_dni_w_m2: float = 0.0  # working threshold on DNI (W/m2) together with the minimum sun elevation
    aiming: str = "center"  # "center" (fast) or an APS method name
    # representative DNI of a merged state: "time_mean" (weight = the actual hours) or "energy_weighted"
    # (sum G^2 w / sum G w, weight = irradiance-equivalent hours, the 0.4 / 0.5.0 convention)
    representative_dni: str = "time_mean"


@dataclass
class LayoutOpt:
    generator: str = "radial_staggered"  # "radial_staggered" | "biomimetic"
    n_heliostats: int | None = None  # keep the best N candidates (None = keep all)
    bounds: dict[str, list[float]] = dc_field(default_factory=dict)  # optimised generator params
    fixed: dict[str, Any] = dc_field(default_factory=dict)  # other generator params
    search_sampling: str = "sun_bins"  # annual sampling during the search: sun_bins | sun_grid | hourly
    search_aiming: str = "auto"  # "auto": greedy when uniformity is enabled, otherwise center; or center | greedy | kfactor
    n_sun_bins: int = 30  # representative sun positions for the search (20 / 30 / 40 / any)
    review_sampling: str = "sun_bins"  # sampling for the review of the best layouts
    review_n_sun_bins: int = 40
    review_dni_bands: int = 3  # DNI bands per state in the constrained (APS) review
    review_check_hourly: bool = True  # also report the sampling error against the hourly year
    min_spacing_factor: float | None = None  # final check: min distance >= factor x diagonal (None = spacing_factor)
    swap_rounds: int = 2  # swap low-yield heliostats for unused candidates, accepted on APS annual energy
    swap_fraction: float = 0.03
    shading_points: int = 5
    maxiter: int = 8
    popsize: int = 6
    seed: int = 1
    workers: int = 1  # parallel processes for the global search (-1 = all CPUs)
    polish_rounds: int = 4  # local pattern-search rounds after the global search
    position_trials: int = 12  # feasible continuous XY perturbations per shortlisted theoretical layout
    refine_top: int = 3  # re-rank the best layouts with APS-aware annual energy
    upper_bound: bool = True  # certify: upper bound on the annual energy of any layout (packing bound)
    # lambda_L: weight of the circumferential (layout-determined) non-uniformity of the receiver flux.
    # Objective J = sum_t w_t [P_abs,t - lambda_L * A_H * sigma_circ,t]; 0 = annual energy only.
    uniformity_weight: float = 0.0
    uniformity_metric: str = "circumferential"  # "surface": area-weighted 2-D flux over the receiver region
    # Fixed-point passes that re-price every candidate against the selected layout (only the selected
    # heliostats shade / block) and reselect; stops when the selection no longer changes.
    selection_iterations: int = 2  # search stage; the review stage uses at least 6 plus the full swap polish
    # Azimuth sectors for the layout uniformity: 0 = the receiver panels (12 x 30 deg if fewer than 6 panels).
    # The layout decides how heat is shared between sectors / panels; the ripple inside a sector is the APS's job.
    uniformity_sectors: int = 0
    # ---- formal ranking: V(L) = sum_s h_s (P_s - lambda_L U_s) from per-scenario APS plans (layout_value.py)
    ranking: str = "aps_value"  # "aps_value" (per-scenario APS on the scenario set) | "legacy" (0.4 review)
    aps_weight: float | None = None  # inner (operating) lambda_APS of P - lambda_APS U; None = uniformity_weight (one lambda)
    baseline: str = "initial"  # "initial": the case's own field (N heliostats nearest the tower) is ranked too | "none"
    formal_position_trials: int = 4  # feasible XY proposals evaluated with full per-scenario APS
    ranking_refine_states: int = 2  # highest uncertainty states per competing layout; 0 disables
    formal_swap_rounds: int = 1  # swap rounds on the winner, accepted only if V(L) increases
    # how the formal swaps pick heliostats: "dual" (flux prices of the per-scenario APS bound: free positions by
    # their reduced cost, mirrors by their removal loss; costs about one layout evaluation per round) | "energy"
    # (lowest APS-energy mirrors out, best centre-aim free positions in; blind to receiver congestion)
    swap_pricing: str = "dual"
    inner_joint_trials: int = 2  # joint-pose neighbourhood trials per scenario solve
    inner_certify_rounds: int = 0  # relaxation roundings + dual bound per scenario (0 = off; fixed iterations)
    inner_certify_iters: int = 60  # dual / Frank-Wolfe iterations of that bound (iteration budget, reproducible)
    dense_check_scenarios: int = 0  # re-evaluate winner and baseline on this many scenarios (0 = off)
    # clouds maps: re-score winner and baseline on independent cloud maps (seed + 1000); transient shadows must
    # not decide where mirrors go
    cross_check_maps: bool = True
    # control variate: correct every scenario with the centre-aim energy of its member hours (cheap, needs the
    # weather year behind the scenario set); "value_cv" ranks by that estimate instead of the formal V
    control_variate: bool = True
    rank_by: str = "value"  # "value" (formal V on the scenario set) | "value_cv" (control-variate estimate)
    # ---- land use: the layout objective is J(L) = E_abs - lambda_L Pi - kappa A(L), A = convex hull of the mirror
    # centres (ha), counted once per layout (outside the annual sum); search and ranking use J, the hourly aiming
    # its own objective. kappa in MWh_th per hectare and year: the annual heat worth one hectare (0 = land free).
    land_weight: float = 0.0
    max_land_ha: float = 0.0  # hard limit A(L) <= max (ha); 0 = none. Layouts over it are not ranked
    # for the recommended kappa (layout_value.recommend_land_weight): annualised land cost per hectare and the
    # value of one MWh_th of useful heat, same currency (0 = unknown -> recommendation from the field itself)
    land_cost_per_ha_year: float = 0.0
    heat_value_per_mwh: float = 0.0
    # reuse inside one layout's scenarios: "responses" (spot responses per sun direction built once; plans identical
    # to independent solves) | "full" (+ plans reused across DNI levels the limits never shaped, warm starts along
    # the sun path; faster) | "off"
    inner_reuse: str = "responses"
    # search: "de" = differential evolution on the centre-aim pre-screen | "bayes" = Bayesian optimisation of the
    # generator parameters directly on the formal V of ``bayes_scenarios`` scenarios (fast inner settings)
    search_method: str = "de"
    bayes_evaluations: int = 30  # layouts evaluated by the Bayesian search (initial design included)
    bayes_initial: int = 0  # Sobol initial design (0 = 2 x dimensions + 2)
    bayes_scenarios: int = 8  # scenarios of the low-fidelity V
    sector_spacing: bool = False  # also search thinning factors towards N / E / S / W (radial_staggered)


@dataclass
class ScenarioSpec:
    """Annual operating scenarios s = (sun direction, hours h_s, site DNI map D_s(x, y)) for the layout value.

    Maps are rasters in site coordinates (x east, y north, m, origin at the tower base), sampled bilinearly
    and clamped at the raster edge. A "factor" map multiplies the scenario DNI; with ``map_normalize`` =
    "site_mean" its mean over the site (boundary minus exclusion zone) is 1, so the map redistributes the
    station DNI in space without changing the site's annual DNI energy.
    """

    # "clear_sky_days" (design baseline): every working hour of ``n_days`` representative days, one clear-sky DNI
    #     per hour shared by the whole field (site, date, time, typical clear-sky atmosphere); the layout score is
    #     the "clear-sky annual" value -- spatial maps, clouds and measured weather do not enter it
    # "weather": representative states of the weather year (+ spatial DNI maps) | "file": scenario table CSV
    source: str = "clear_sky_days"
    n_days: int = 20  # clear_sky_days: representative days (equal date segments, weight = days in the segment)
    clear_sky_model: str = "ineichen"  # clear_sky_days: "ineichen" (date, altitude, Linke turbidity) | "meinel"
    # clear_sky_days, Ineichen: None = the site's monthly Linke-turbidity climatology (pvlib), a number = all year,
    # 12 numbers = one per month (local measurements)
    linke_turbidity: Any = None
    file: str | None = None  # scenario table: scenario_id,elevation_deg,azimuth_deg,hours,dni_w_m2,map_id[,time]
    validation_file: str | None = None  # separate scenario table, using the case's map library
    n_scenarios: int = 20  # representative scenarios generated from the weather year
    dni_bands: int = 1  # split each representative sun position into DNI bands
    # one set of sun directions for every map and DNI band (directions clustered over the whole year, the hours of
    # each direction then split by map and band): the spot responses of a direction serve all its scenarios, and
    # with layout_opt.inner_reuse = full a plan the flux limits never shaped serves its lower DNI bands
    shared_directions: bool = False
    map_model: str = "uniform"  # uniform | half_swap | gradient | clouds | file
    map_file: str | None = None  # map library CSV: map_id,x,y,value (regular grid or scattered points)
    map_unit: str = "factor"  # file maps: "factor" (x scenario DNI) | "w_m2" (absolute DNI)
    map_normalize: str = "site_mean"  # factor maps: "site_mean" | "none"
    map_assignment: str = "by_day"  # weather scenarios: map regime of each day cycles through the maps | "single"
    map_sequence: list[str] | None = None  # map ids used as regimes, in order (None = every map of the model)
    map_include_uniform: bool = False  # add the uniform map as an extra regime (compatibility scenarios)
    map_cell_m: float = 20.0  # raster resolution of modelled maps, m
    map_contrast: float = 0.3  # half_swap / gradient: relative DNI difference, +- (0.3 = 1.3 vs 0.7)
    map_axis_deg: float = 90.0  # compass direction towards the bright side (90 = east half brighter in map 1)
    map_edge_m: float = 40.0  # half_swap: width of the soft transition, m
    cloud_maps: int = 4  # clouds: number of cloud-shadow maps
    cloud_n: int = 3  # clouds: shadows per map
    cloud_radius_m: list[float] = dc_field(default_factory=lambda: [60.0, 160.0])
    cloud_transmittance: list[float] = dc_field(default_factory=lambda: [0.2, 0.6])
    seed: int = 0


@dataclass
class MCRTSpec:
    rays_per_heliostat: int = 1000
    seed: int = 0
    n_seeds: int = 3  # repeat with independent seeds -> statistical uncertainty of power and peak
    tracking_mode: str = "per_ray"  # "per_ray" (statistical, like HFLCAL) | "per_heliostat" (fixed bias per mirror)


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
    uniformity: UniformitySpec = dc_field(default_factory=UniformitySpec)
    engine: EngineSpec = dc_field(default_factory=EngineSpec)
    realtime: Realtime = dc_field(default_factory=Realtime)
    annual: AnnualSpec = dc_field(default_factory=AnnualSpec)
    layout_opt: LayoutOpt = dc_field(default_factory=LayoutOpt)
    scenarios: ScenarioSpec = dc_field(default_factory=ScenarioSpec)
    mcrt: MCRTSpec = dc_field(default_factory=MCRTSpec)
    base_dir: Path = dc_field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from the YAML relative to the YAML file's folder."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)

    def referenced_files(self) -> list[tuple[str, str, Path]]:
        """The input files this case reads: (``section.attr``, value as written, resolved path), only settings in
        use (the weather file with ``weather.source: file``, the scenario table with ``scenarios.source: file``)."""
        out = []
        for section, attr in FILE_SETTINGS:
            value = getattr(getattr(self, section), attr, None)
            if not value:
                continue
            if (section, attr) == ("weather", "file") and self.weather.source != "file":
                continue
            if (section, attr) == ("scenarios", "file") and self.scenarios.source != "file":
                continue
            out.append((f"{section}.{attr}", str(value), self.resolve(value)))
        return out

    def missing_files(self) -> list[tuple[str, str, Path]]:
        """``referenced_files`` that do not exist (e.g. a YAML moved or uploaded without its CSVs)."""
        return [r for r in self.referenced_files() if not r[2].is_file()]

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
    "uniformity": UniformitySpec,
    "engine": EngineSpec,
    "realtime": Realtime,
    "annual": AnnualSpec,
    "layout_opt": LayoutOpt,
    "scenarios": ScenarioSpec,
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
    if rec.n_panels < 1:
        raise ValueError("receiver.n_panels must be >= 1")
    for name in ("panel_max_flux_kw_m2", "panel_absorptivity", "panel_flux_bias"):
        values = getattr(rec, name)
        if values is not None:
            if len(values) != rec.n_panels:
                raise ValueError(f"receiver.{name} needs {rec.n_panels} values (one per panel), got {len(values)}")
            if any(not v > 0 for v in values):
                raise ValueError(f"receiver.{name} values must be > 0")
    if rec.tower_diameter is not None and rec.tower_diameter < 0:
        raise ValueError("receiver.tower_diameter must be >= 0 (0 = no tower shadow)")
    if rec.tower_height is not None and not rec.tower_height > 0:
        raise ValueError(f"receiver.tower_height must be > 0 m (got {rec.tower_height})")
    if not hel.pivot_height > 0:
        raise ValueError(f"heliostat.pivot_height must be > 0 m (got {hel.pivot_height})")
    if rec.bottom <= 0:
        raise ValueError("receiver.center z must be above half the receiver height (the receiver stands on a tower)")
    if rec.flux_limit_basis not in ("incident", "absorbed"):
        raise ValueError("receiver.flux_limit_basis must be 'incident' or 'absorbed'")
    if not 0 < rec.absorptivity <= 1:
        raise ValueError("receiver.absorptivity must be in (0, 1]")
    if rec.panel_absorptivity and any(v > 1 for v in rec.panel_absorptivity):
        raise ValueError("receiver.panel_absorptivity must be in (0, 1]")
    if rec.limit_height_profile is not None and (len(rec.limit_height_profile) < 1
                                                 or any(not v > 0 for v in rec.limit_height_profile)):
        raise ValueError("receiver.limit_height_profile needs positive values")
    u = case.uniformity
    if u.min_load_mw < 0 or case.aps.coupled_exact_limit < 0 or case.aps.coupled_trials < 0:
        raise ValueError("minimum load and coupled-search budgets must be nonnegative")
    if case.layout_opt.min_spacing_factor is not None and not case.layout_opt.min_spacing_factor > 0:
        raise ValueError("layout_opt.min_spacing_factor must be positive")
    if u.mode not in ("cv_limit", "weighted"):
        raise ValueError("uniformity.mode must be cv_limit|weighted")
    if not 0 <= u.weight < math.inf:
        raise ValueError("uniformity.weight must be >= 0")
    if u.panels is not None and any(not 1 <= p <= rec.n_panels for p in u.panels):
        raise ValueError(f"uniformity.panels must be panel numbers 1..{rec.n_panels}")
    if not u.max_cv > 0:
        raise ValueError("uniformity.max_cv must be > 0")
    e = case.engine
    if e.no_forecast_policy not in ("clear_sky", "current") or e.stale_policy not in ("clear_sky", "last"):
        raise ValueError("engine.no_forecast_policy must be clear_sky|current, stale_policy clear_sky|last")
    if e.fallback not in ("hold_or_defocus", "defocus", "none"):
        raise ValueError("engine.fallback must be hold_or_defocus|defocus|none")
    for name in ("search_sampling", "review_sampling"):
        if getattr(case.layout_opt, name) not in ("sun_bins", "sun_grid", "hourly"):
            raise ValueError(f"layout_opt.{name} must be sun_bins|sun_grid|hourly")
    if not 0 <= case.layout_opt.uniformity_weight < math.inf:
        raise ValueError("layout_opt.uniformity_weight must be >= 0")
    for name in ("land_weight", "max_land_ha", "land_cost_per_ha_year", "heat_value_per_mwh"):
        if not 0 <= float(getattr(case.layout_opt, name)) < math.inf:
            raise ValueError(f"layout_opt.{name} must be finite and >= 0")
    if case.layout_opt.uniformity_metric not in ("circumferential", "surface"):
        raise ValueError("layout_opt.uniformity_metric must be circumferential|surface")
    if case.layout_opt.position_trials < 0:
        raise ValueError("layout_opt.position_trials must be nonnegative")
    if case.layout_opt.uniformity_sectors < 0 or case.layout_opt.selection_iterations < 0:
        raise ValueError("layout_opt.uniformity_sectors and selection_iterations must be >= 0")
    if case.layout_opt.search_aiming not in ("auto", "center", "greedy", "kfactor"):
        raise ValueError("layout_opt.search_aiming must be auto|center|greedy|kfactor")
    lo = case.layout_opt
    if lo.ranking not in ("aps_value", "legacy"):
        raise ValueError("layout_opt.ranking must be aps_value|legacy")
    if lo.rank_by not in ("value", "value_cv"):
        raise ValueError("layout_opt.rank_by must be value|value_cv")
    if lo.swap_pricing not in ("energy", "dual"):
        raise ValueError("layout_opt.swap_pricing must be energy|dual")
    if lo.inner_reuse not in ("responses", "full", "off"):
        raise ValueError("layout_opt.inner_reuse must be responses|full|off")
    if lo.search_method not in ("de", "bayes"):
        raise ValueError("layout_opt.search_method must be de|bayes")
    if lo.bayes_evaluations < 2 or lo.bayes_scenarios < 1 or lo.bayes_initial < 0:
        raise ValueError("layout_opt: bayes_evaluations >= 2, bayes_scenarios >= 1, bayes_initial >= 0")
    if lo.baseline not in ("initial", "none"):
        raise ValueError("layout_opt.baseline must be initial|none")
    if lo.aps_weight is not None and not 0 <= lo.aps_weight < math.inf:
        raise ValueError("layout_opt.aps_weight must be >= 0")
    if lo.n_heliostats is not None and (not isinstance(lo.n_heliostats, int) or lo.n_heliostats <= 0):
        raise ValueError("layout_opt.n_heliostats must be a positive integer")
    budgets = (lo.formal_position_trials, lo.ranking_refine_states, lo.formal_swap_rounds, lo.inner_joint_trials,
               lo.inner_certify_rounds, lo.inner_certify_iters, lo.dense_check_scenarios)
    if any(not isinstance(v, int) or v < 0 for v in budgets):
        raise ValueError("layout_opt formal-ranking budgets must be nonnegative")
    sc = case.scenarios
    if sc.source not in ("clear_sky_days", "weather", "file"):
        raise ValueError("scenarios.source must be clear_sky_days|weather|file")
    if not isinstance(sc.n_days, int) or not 1 <= sc.n_days <= 366:
        raise ValueError("scenarios.n_days must be an integer in 1..366")
    if sc.clear_sky_model not in ("ineichen", "meinel"):
        raise ValueError("scenarios.clear_sky_model must be ineichen|meinel")
    tl = sc.linke_turbidity
    if tl is not None:
        vals = [tl] if not isinstance(tl, (list, tuple)) else list(tl)
        if (isinstance(tl, (list, tuple)) and len(vals) != 12) or not all(
                isinstance(v, (int, float)) and 1.0 <= float(v) <= 10.0 for v in vals):
            raise ValueError("scenarios.linke_turbidity must be null, a number in [1, 10] or 12 monthly numbers")
    if case.annual.min_dni_w_m2 < 0:
        raise ValueError("annual.min_dni_w_m2 must be >= 0")
    if sc.source == "file" and not sc.file:
        raise ValueError("scenarios.source is 'file' but scenarios.file is not set")
    if sc.map_model not in ("uniform", "half_swap", "gradient", "clouds", "file"):
        raise ValueError("scenarios.map_model must be uniform|half_swap|gradient|clouds|file")
    if sc.map_model == "file" and not sc.map_file:
        raise ValueError("scenarios.map_model is 'file' but scenarios.map_file is not set")
    if sc.map_unit not in ("factor", "w_m2") or sc.map_normalize not in ("site_mean", "none"):
        raise ValueError("scenarios.map_unit must be factor|w_m2 and map_normalize site_mean|none")
    if sc.map_model == "file" and sc.map_unit == "w_m2" and sc.source != "file":
        raise ValueError("absolute DNI maps (map_unit w_m2) need a scenario table (scenarios.source: file); "
                         "maps assigned to weather hours must be factor maps")
    if sc.map_assignment not in ("by_day", "single"):
        raise ValueError("scenarios.map_assignment must be by_day|single")
    if sc.n_scenarios < 1 or sc.dni_bands < 1 or not sc.map_cell_m > 0 or sc.cloud_maps < 1 or sc.cloud_n < 0:
        raise ValueError("scenarios: n_scenarios, dni_bands, cloud_maps >= 1, map_cell_m > 0, cloud_n >= 0")
    if not 0 <= sc.map_contrast < 1:
        raise ValueError("scenarios.map_contrast must be in [0, 1)")
    if len(sc.cloud_radius_m) != 2 or len(sc.cloud_transmittance) != 2 or min(sc.cloud_radius_m) <= 0 \
            or not all(0 <= v <= 1 for v in sc.cloud_transmittance):
        raise ValueError("scenarios.cloud_radius_m [lo, hi] > 0 and cloud_transmittance [lo, hi] in [0, 1]")
    if case.mcrt.tracking_mode not in ("per_ray", "per_heliostat"):
        raise ValueError("mcrt.tracking_mode must be per_ray|per_heliostat")
    if case.realtime.scenario not in ("clouds", "moving_edge", "sudden_clear"):
        raise ValueError("realtime.scenario must be clouds|moving_edge|sudden_clear")
    if not 0 <= case.aps.safety_margin < 1:
        raise ValueError("aps.safety_margin must be in [0, 1)")
    if case.aps.method not in ("center", "kfactor", "greedy", "milp", "subfield"):
        raise ValueError(f"aps.method '{case.aps.method}' not in center|kfactor|greedy|milp|subfield")
    if case.aps.subfield_sectors < 1:
        raise ValueError("aps.subfield_sectors must be >= 1")
    if case.annual.representative_dni not in ("time_mean", "energy_weighted"):
        raise ValueError("annual.representative_dni must be time_mean|energy_weighted")
    if case.annual.method not in ("sun_bins", "sun_grid", "representative_days", "hourly"):
        raise ValueError(f"annual.method '{case.annual.method}' not in sun_bins|sun_grid|representative_days|hourly")
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
