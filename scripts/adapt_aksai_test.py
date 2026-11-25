#!/usr/bin/env python3
"""Build a test DNI map from the Aksai SW/NE station records and NASA POWER background."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_station(path: Path) -> dict[datetime, float]:
    rows = read_rows(path)
    if not rows or not {"time", "DNI"} <= set(rows[0]):
        raise ValueError(f"{path}: expected columns time,DNI")
    return {datetime.strptime(row["time"], "%Y/%m/%d %H:%M:%S"): float(row["DNI"])
            for row in rows if row.get("DNI", "").strip()}


def interpolate_background(path: Path, time: datetime) -> float:
    rows = read_rows(path)
    if not rows or not {"time", "dni"} <= set(rows[0]):
        raise ValueError(f"{path}: expected NASA POWER columns time,dni")
    samples = [(datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S"), float(row["dni"]))
               for row in rows if row.get("dni", "").strip()]
    samples.sort()
    times = [row[0] for row in samples]
    i = bisect.bisect_right(times, time)
    if i == 0:
        return samples[0][1]
    if i == len(samples):
        return samples[-1][1]
    before, after = samples[i - 1], samples[i]
    fraction = (time - before[0]).total_seconds() / (after[0] - before[0]).total_seconds()
    return before[1] + fraction * (after[1] - before[1])


def fetch_power_window(path: Path, time: datetime) -> None:
    from cspaps.weather import parse_power_csv

    start = time.date() - timedelta(days=1)
    end = time.date() + timedelta(days=2)
    params = {
        "parameters": "ALLSKY_SFC_SW_DNI",
        "community": "RE",
        "longitude": "94.3960",
        "latitude": "38.9132",
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "CSV",
        "time-standard": "UTC",
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    frame, _ = parse_power_csv(text, "Asia/Shanghai", time.year)
    if frame.empty:
        raise ValueError("NASA POWER returned no hourly DNI for the requested date window")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "dni"])
        writer.writerows(frame.itertuples(index=False, name=None))


def background_covers(path: Path, time: datetime) -> bool:
    if not path.exists():
        return False
    rows = read_rows(path)
    if not rows or not {"time", "dni"} <= set(rows[0]):
        return False
    times = [datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
             for row in rows if row.get("dni", "").strip()]
    return bool(times) and min(times) <= time <= max(times)


def build_map(input_dir: Path, output_dir: Path, requested_time: datetime) -> dict:
    if requested_time.year != 2026:
        raise ValueError("the supplied Aksai station records cover 2026")
    sw = read_station(input_dir / "SW_MAIN.csv")
    ne = read_station(input_dir / "NE_SLAVE.csv")
    common = sorted(sw.keys() & ne.keys())
    if not common:
        raise ValueError("SW_MAIN.csv and NE_SLAVE.csv have no matching timestamps")
    time = min(common, key=lambda value: abs((value - requested_time).total_seconds()))
    if abs((time - requested_time).total_seconds()) > 60:
        raise ValueError("no shared station sample within 60 seconds of the requested time")

    stations = {row["station"]: (float(row["x_east_m"]), float(row["y_north_m"]))
                for row in read_rows(output_dir / "camera_positions.csv")}
    if not {"SW_MAIN", "NE_SLAVE"} <= stations.keys():
        raise ValueError("camera_positions.csv needs SW_MAIN and NE_SLAVE")
    sx, sy = stations["SW_MAIN"]
    nx, ny = stations["NE_SLAVE"]
    dx, dy = nx - sx, ny - sy
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        raise ValueError("SW and NE camera coordinates must be distinct")

    power_file = output_dir / "nasa_power_dni_2026.csv"
    if not background_covers(power_file, time):
        fetch_power_window(power_file, time)
    background = interpolate_background(power_file, time)
    sw_dni, ne_dni = sw[time], ne[time]
    layout = read_rows(output_dir / "layout_full.csv")
    if not layout or not {"id", "x", "y"} <= set(layout[0]):
        raise ValueError("layout_full.csv needs columns id,x,y")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    map_path = output_dir / f"dni_map_{stamp}.csv"
    values = []
    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "dni"])
        for row in layout:
            x, y = float(row["x"]), float(row["y"])
            along = min(1.0, max(0.0, ((x - sx) * dx + (y - sy) * dy) / length2))
            perpendicular = math.hypot(x - sx - along * dx, y - sy - along * dy)
            weight = math.exp(-0.5 * (perpendicular / 600.0) ** 2)
            measured_line = (1.0 - along) * sw_dni + along * ne_dni
            dni = max(0.0, background + weight * (measured_line - background))
            values.append(dni)
            writer.writerow([row["x"], row["y"], f"{dni:.3f}"])

    result = {
        "time_requested_local": requested_time.isoformat(sep=" "),
        "time_used_local": time.isoformat(sep=" "),
        "nearest_sample_seconds": abs((time - requested_time).total_seconds()),
        "source": "SW_MAIN and NE_SLAVE measured DNI anchored to NASA POWER ALLSKY_SFC_SW_DNI hourly background",
        "spatial_model": "linear along SW-NE station axis, Gaussian 600 m perpendicular fade to NASA background",
        "dni_map": map_path.name,
        "heliostat_count": len(values),
        "nasa_background_dni_w_m2": round(background, 3),
        "sw_main_dni_w_m2": round(sw_dni, 3),
        "ne_slave_dni_w_m2": round(ne_dni, 3),
        "map_dni_min_w_m2": round(min(values), 3),
        "map_dni_mean_w_m2": round(sum(values) / len(values), 3),
        "map_dni_max_w_m2": round(max(values), 3),
        "stations_enu_m": {key: list(value) for key, value in stations.items()},
    }
    (output_dir / f"dni_map_{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                          encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="folder containing SW_MAIN.csv and NE_SLAVE.csv")
    parser.add_argument("--output", type=Path, required=True, help="folder containing prepared layout, stations and NASA data")
    parser.add_argument("--time", required=True, help="requested local time, e.g. 2026-06-18 12:00:00")
    args = parser.parse_args()
    requested = datetime.fromisoformat(args.time)
    print(json.dumps(build_map(args.input, args.output, requested), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
