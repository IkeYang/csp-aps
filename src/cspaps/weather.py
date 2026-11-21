"""Hourly DNI for a site from NASA POWER (ALLSKY_SFC_SW_DNI), written as the case CSV format time,dni.

NASA POWER returns hourly values in Wh/m2 for UTC hours (YEAR, MO, DY, HR = start
of the hour). An hourly energy in Wh/m2 is the hour-average irradiance in W/m2;
it is stamped at the middle of the hour (start + 30 min) and converted to the
site's local time zone, which is what ``sun.load_weather`` expects. Missing
values (-999) become 0 and are counted.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def parse_power_csv(text: str, timezone: str, year: int) -> tuple[pd.DataFrame, dict]:
    """(DataFrame time,dni in local time for ``year``, info) from a NASA POWER hourly CSV payload."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("YEAR,"))
    except StopIteration as exc:
        raise ValueError("not a NASA POWER hourly CSV (no YEAR,MO,DY,HR header)") from exc
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    col = next((c for c in df.columns if c.upper().startswith("ALLSKY_SFC_SW_DNI")), None)
    if col is None:
        raise ValueError("NASA POWER CSV has no ALLSKY_SFC_SW_DNI column")
    utc = pd.to_datetime({"year": df["YEAR"], "month": df["MO"], "day": df["DY"], "hour": df["HR"]}, utc=True)
    mid = (utc + pd.Timedelta(minutes=30)).dt.tz_convert(timezone)
    dni = pd.to_numeric(df[col], errors="coerce")
    missing = int(((dni < 0) | dni.isna()).sum())
    dni = dni.where(dni >= 0, 0.0).fillna(0.0)
    local = mid.dt.tz_localize(None)
    keep = local.dt.year == int(year)
    out = pd.DataFrame({"time": local[keep].dt.strftime("%Y-%m-%d %H:%M:%S"), "dni": dni[keep].round(2)})
    out = out.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    info = {"n_hours": int(len(out)), "missing_hours": missing, "annual_dni_kwh_m2": round(float(out["dni"].sum()) / 1e3, 1),
            "peak_dni_w_m2": round(float(out["dni"].max()), 1) if len(out) else 0.0,
            "time_basis": "local time, middle of each UTC hour", "source": "NASA POWER ALLSKY_SFC_SW_DNI (hourly)"}
    return out, info


def fetch_power_dni(latitude: float, longitude: float, year: int, timezone: str, out_path: str | Path,
                    timeout_s: float = 60.0) -> dict:
    """Download one local calendar year of hourly DNI and write it as CSV time,dni. Returns the info dict."""
    params = {"parameters": "ALLSKY_SFC_SW_DNI", "community": "RE", "longitude": f"{longitude:.4f}",
              "latitude": f"{latitude:.4f}", "start": f"{int(year) - 1}1231", "end": f"{int(year) + 1}0101",
              "format": "CSV", "time-standard": "UTC"}
    url = f"{POWER_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - fixed https endpoint
        text = resp.read().decode("utf-8", errors="replace")
    df, info = parse_power_csv(text, timezone, year)
    if len(df) < 8000:
        raise ValueError(f"NASA POWER returned only {len(df)} hours for {year}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    info.update({"latitude": latitude, "longitude": longitude, "year": int(year), "timezone": timezone,
                 "file": str(out_path)})
    return info
