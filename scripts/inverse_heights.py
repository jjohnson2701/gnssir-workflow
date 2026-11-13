# ABOUTME: Runs the gnssrefl inverse method (invsnr) for a tidal station over a range of days.
# ABOUTME: Splits the period at incomplete days, merges the output, and masks times with no arcs nearby.
"""Smooth water-height series from the gnssrefl inverse method.

The periodogram method gives one height per satellite arc. The inverse method (gnssrefl
`invsnr`, after Strandberg et al. 2016) fits one smooth height curve to the SNR data of
all arcs together. At GLBX it gave about 7 cm against a tide gauge, against 10 to 11 cm
for a spline through the periodogram heights (one month of tests, 2024).

What this script adds to `invsnr`:
  - it takes the height, elevation and peak-to-noise settings from the station file;
  - `invsnr` accepts one azimuth range, so the widest range of the station mask is used;
  - days with an incomplete SNR file are left out, and the period is run in segments of
    whole days between them;
  - `invsnr` refuses to run when the gap between arcs is longer than the knot spacing.
    The script runs it with the `risky` option and then removes every output time that
    has no periodogram arc within `support_hours` on both sides.

Run `gnssir` for the days first: the arc times come from its result files.

Usage:
    python -m scripts.inverse_heights --station glbx --year 2024 --doy 60 --doy_end 91 --out glbx_inverse.csv
    (REFL_CODE must point to the gnssrefl workspace; invsnr and invsnr_input must be on PATH)
"""
import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from scripts.station_check import load_results


def widest_azimuth_range(azval2):
    """The widest (start, end) pair of a gnssrefl azimuth list."""
    pairs = list(zip(azval2[0::2], azval2[1::2]))
    return max(pairs, key=lambda p: p[1] - p[0])


def whole_day_segments(sizes, doy, doy_end, min_share=0.8):
    """Runs of consecutive days whose SNR file is at least `min_share` of the median size.

    sizes: dict day -> file size (missing days are absent). Returns a list of (first, last).
    """
    present = [sizes[d] for d in range(doy, doy_end + 1) if d in sizes]
    if not present:
        return []
    limit = min_share * float(np.median(present))
    segments, start = [], None
    for d in range(doy, doy_end + 2):
        good = d <= doy_end and sizes.get(d, 0) >= limit
        if good and start is None:
            start = d
        if not good and start is not None:
            segments.append((start, d - 1))
            start = None
    return segments


def supported(times, arc_times, hours):
    """True where an arc exists within `hours` before and within `hours` after each time (days)."""
    arc_times = np.sort(np.asarray(arc_times, float))
    times = np.asarray(times, float)
    if len(arc_times) == 0:
        return np.zeros(len(times), dtype=bool)
    i = np.searchsorted(arc_times, times)
    ok = (i > 0) & (i < len(arc_times))
    w = hours / 24.0
    ok[ok] &= (times[ok] - arc_times[i[ok] - 1] <= w) & (arc_times[i[ok]] - times[ok] <= w)
    return ok


def snr_sizes(refl_code, station, year):
    sizes = {}
    for f in glob.glob(f"{refl_code}/{year}/snr/{station}/{station}???0.{year % 100:02d}.snr*"):
        name = os.path.basename(f)
        sizes[int(name[4:7])] = os.path.getsize(f)
    return sizes


def write_inverse_settings(refl_code, station, cfg, azimuth):
    """Write $REFL_CODE/input/<station>.inv.json from the station file (same content as invsnr_input)."""
    inv = {
        "station": station, "lat": cfg["lat"], "lon": cfg["lon"], "ht": cfg["ht"],
        "rhlims": [cfg["minH"], cfg["maxH"]], "elvlims": [cfg["e1"], cfg["e2"]],
        "azilims": [float(azimuth[0]), float(azimuth[1])], "pktnlim": cfg["PkNoise"],
        "precision": cfg.get("desiredP", 0.005), "l2c_only": True,
    }
    path = Path(refl_code, "input", f"{station}.inv.json")
    path.write_text(json.dumps(inv, indent=4))
    return path


def run(station, year, doy, doy_end, signal="L1+L2+L5", knot_hours=3, decimate=10, delta_out=300,
        support_hours=1.5, refl_code=None, work_dir=None):
    """Run the inverse method. Returns (table, info). table columns: doy_fraction, mjd, rh."""
    refl_code = refl_code or os.environ["REFL_CODE"]
    station = station.lower()
    cfg = json.loads(Path(refl_code, "input", f"{station}.json").read_text())
    azimuth = widest_azimuth_range(cfg.get("azval2", [0, 360]))
    write_inverse_settings(refl_code, station, cfg, azimuth)
    segments = whole_day_segments(snr_sizes(refl_code, station, year), doy, doy_end)
    arcs = load_results(refl_code, station, year, doy, doy_end)
    arcs = arcs[(arcs[:, 5] >= azimuth[0]) & (arcs[:, 5] <= azimuth[1])]
    arc_t = arcs[:, 1] + arcs[:, 4] / 24.0
    parts, failed = [], []
    work = Path(work_dir or Path(refl_code, "Files", station, "inverse"))
    for first, last in segments:
        seg = work / f"seg{first:03d}"
        seg.mkdir(parents=True, exist_ok=True)
        cmd = ["invsnr", station, str(year), str(first), signal, "-doy_end", str(last), "-dec", str(decimate),
               "-plt", "F", "-risky", "T", "-knot_space", str(int(knot_hours)), "-delta_out", str(delta_out),
               "-outfile_name", "out", "-outfile_type", "txt"]
        with open(seg / "run.out", "w") as log:
            subprocess.run(cmd, cwd=seg, stdout=log, stderr=subprocess.STDOUT, check=False)
        out = seg / "out.txt"
        if not out.exists() or out.stat().st_size == 0:
            failed.append((first, last))
            continue
        a = np.loadtxt(out, comments="%", ndmin=2)
        t = a[:, 7] + (a[:, 3] + a[:, 4] / 60.0 + a[:, 5] / 3600.0) / 24.0
        parts.append(np.c_[t, a[:, 8], a[:, 6]])
    table = np.vstack(parts) if parts else np.empty((0, 3))
    inside = (table[:, 2] > cfg["minH"]) & (table[:, 2] < cfg["maxH"])
    keep = supported(table[:, 0], arc_t, support_hours) & inside
    info = {"azimuth": azimuth, "segments": segments, "failed_segments": failed, "epochs": int(len(table)),
            "kept": int(keep.sum()), "outside_window": int((~inside).sum()), "arcs": int(len(arcs)), "work_dir": str(work)}
    return table[keep], info


def main(argv=None):
    p = argparse.ArgumentParser(description="Smooth water-height series from the gnssrefl inverse method.")
    p.add_argument("--station", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    p.add_argument("--doy_end", type=int, required=True)
    p.add_argument("--signal", default="L1+L2+L5", help="L1, L2, L5, L1+L2, L1+L5 or L1+L2+L5; single signals can be unstable")
    p.add_argument("--knot_hours", type=int, default=3, help="spline knot spacing, whole hours")
    p.add_argument("--support_hours", type=float, default=1.5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if not os.environ.get("REFL_CODE"):
        raise SystemExit("Set REFL_CODE to the gnssrefl workspace.")
    table, info = run(a.station, a.year, a.doy, a.doy_end, a.signal, a.knot_hours, support_hours=a.support_hours)
    np.savetxt(a.out, table, delimiter=",", header="doy_fraction,mjd,rh_m", comments="", fmt=["%.6f", "%.6f", "%.4f"])
    print(f"Azimuth range used: {info['azimuth'][0]:g} to {info['azimuth'][1]:g} deg ({info['arcs']} periodogram arcs there).")
    print("Segments of whole days: " + ", ".join(f"{f:03d}-{l:03d}" for f, l in info["segments"]))
    if info["failed_segments"]:
        print("Segments with no output: " + ", ".join(f"{f:03d}-{l:03d}" for f, l in info["failed_segments"]))
    print(f"{info['kept']} of {info['epochs']} output times kept ({info['outside_window']} outside the height window; "
          f"the rest have no arc within {a.support_hours:g} h on both sides).")
    print(f"Written: {a.out}. Logs: {info['work_dir']}")


if __name__ == "__main__":
    main()
