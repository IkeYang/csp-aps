"""Command line interface: ``cspaps <command> CASE.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_case
from .field import build_field, save_layout_csv
from .plotting import plot_flux_map, plot_layout
from .simulate import STAGES, annual, snapshot


def _out_dir(args, case) -> Path:
    out = Path(args.out) / case.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_layout(args) -> int:
    case = load_case(args.case)
    fld = build_field(case)
    out = _out_dir(args, case)
    save_layout_csv(fld, out / "layout.csv")
    plot_layout(fld, case.receiver, out / "layout.png", boundary=case.field.boundary)
    print(f"{fld.n} heliostats -> {out / 'layout.csv'}")
    return 0


def cmd_snapshot(args) -> int:
    case = load_case(args.case)
    fld = build_field(case)
    snap = snapshot(case, fld, args.time, dni=args.dni)
    out = _out_dir(args, case)
    tag = snap.time.strftime("%Y%m%d_%H%M")
    summary = snap.summary()
    _write_json(out / f"snapshot_{tag}.json", summary)
    cols = ["x", "y", "z", "aim_azimuth_deg", "aim_z"] + [f"eta_{k}" for k in STAGES[1:]] + ["eta_total", "p_receiver_kw"]
    table = np.column_stack(
        [fld.xyz, snap.aim_azimuth, snap.aim_z]
        + [snap.eta[k] for k in STAGES[1:]]
        + [snap.eta_total, snap.power["intercept"] / 1e3]
    )
    np.savetxt(out / f"snapshot_{tag}_heliostats.csv", table, delimiter=",",
               header=",".join(cols), comments="", fmt="%.6g")
    plot_flux_map(snap.flux, snap.grid, case.receiver, out / f"snapshot_{tag}_flux.png")
    plot_layout(fld, case.receiver, out / f"snapshot_{tag}_efficiency.png",
                values=snap.eta_total, label="Optical efficiency",
                boundary=case.field.boundary, title=f"Heliostat efficiency at {snap.time:%Y-%m-%d %H:%M}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results -> {out}")
    return 0


def cmd_annual(args) -> int:
    case = load_case(args.case)
    fld = build_field(case)

    def progress(i, n):
        if i % 12 == 0 or i == n:
            print(f"\r  {i}/{n} instants", end="", file=sys.stderr, flush=True)

    res = annual(case, fld, progress=progress)
    print(file=sys.stderr)
    out = _out_dir(args, case)
    summary = {"case": case.name, "n_heliostats": fld.n,
               "mirror_area_m2": round(fld.n * case.heliostat.area, 1), **res.summary()}
    _write_json(out / "annual.json", summary)
    res.timeseries.to_csv(out / "annual_timeseries.csv", index=False)
    np.savetxt(out / "annual_heliostats.csv",
               np.column_stack([fld.xyz, res.heliostat_energy_mwh, res.heliostat_efficiency]),
               delimiter=",", header="x,y,z,energy_mwh,annual_efficiency", comments="", fmt="%.6g")
    plot_layout(fld, case.receiver, out / "annual_efficiency.png",
                values=res.heliostat_efficiency, label="Annual optical efficiency",
                boundary=case.field.boundary, title="Annual optical efficiency per heliostat")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cspaps", description="Heliostat field & aim-point optimization")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("case", help="case YAML file")
        sp.add_argument("--out", default="results", help="output folder (default: results)")

    sp = sub.add_parser("layout", help="generate the field layout, write CSV + plot")
    common(sp)
    sp.set_defaults(func=cmd_layout)

    sp = sub.add_parser("snapshot", help="one instant: efficiencies + receiver flux map")
    common(sp)
    sp.add_argument("--time", required=True, help='local time, e.g. "2025-06-21 13:40"')
    sp.add_argument("--dni", type=float, default=None, help="override DNI (W/m2)")
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("annual", help="annual energy with single-point aiming")
    common(sp)
    sp.set_defaults(func=cmd_annual)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
