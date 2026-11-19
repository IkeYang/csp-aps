"""Sun position and direct normal irradiance (DNI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .config import Site, Sun


def to_times(times, timezone: str) -> pd.DatetimeIndex:
    """Parse times; naive values are interpreted in the site's local timezone."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times)))
    if idx.tz is None:
        idx = idx.tz_localize(timezone)
    return idx


def solar_position(times: pd.DatetimeIndex, site: Site) -> tuple[np.ndarray, np.ndarray]:
    """Return (apparent elevation, compass azimuth) in degrees (NREL SPA via pvlib)."""
    sp = pvlib.solarposition.get_solarposition(
        times, site.latitude, site.longitude, altitude=site.elevation_m
    )
    return sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy()


def sun_vector(elevation_deg, azimuth_deg) -> np.ndarray:
    """Unit vector(s) pointing from the ground towards the sun, ENU frame."""
    el = np.radians(np.asarray(elevation_deg, dtype=float))
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.stack(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1
    )


def air_mass(elevation_deg) -> np.ndarray:
    """Relative optical air mass, Kasten & Young (1989)."""
    h = np.asarray(elevation_deg, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)
    return np.where(h > 0, am, np.inf)


def dni_meinel(elevation_deg, site_elevation_m: float = 0.0) -> np.ndarray:
    """Clear-sky DNI (W/m2): Meinel model with Laue's altitude correction."""
    h_km = site_elevation_m / 1000.0
    am = air_mass(elevation_deg)
    with np.errstate(over="ignore", invalid="ignore"):
        dni = 1353.0 * ((1.0 - 0.14 * h_km) * 0.7 ** (am**0.678) + 0.14 * h_km)
    return np.where(np.asarray(elevation_deg) > 0, dni, 0.0)


def dni(elevation_deg, sun: Sun, site: Site) -> np.ndarray:
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    if sun.dni_model == "meinel":
        return dni_meinel(elevation_deg, site.elevation_m)
    if sun.dni_model == "constant":
        return np.where(elevation_deg > 0, sun.dni_constant_w_m2, 0.0)
    raise ValueError(f"Unknown dni_model '{sun.dni_model}' (use 'meinel' or 'constant')")
