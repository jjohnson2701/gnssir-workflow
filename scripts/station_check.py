# ABOUTME: Three basic checks for a station: sampling rate, which L2 signal is in the file, and
# ABOUTME: the scatter of the heights about a smooth curve (a quality measure that needs no gauge).
"""Station checks.

1. Sampling check. A reflector at height H gives an SNR oscillation with 2H / wavelength
   cycles per unit of sin(elevation). The data must sample that oscillation at least twice
   per cycle. The check reads one SNR file and gives the largest height that the sampling
   can resolve, per signal, for the elevation range and azimuth mask of the station.

2. L2 signal check. A RINEX 3 file can hold several GPS L2 signal-strength observables.
   S2L, S2S and S2X are the civil signal (L2C). S2W, S2P, S2Y and S2D are the legacy signal,
   which is weaker and noisier. The check lists what the file holds, with counts.

3. Scatter about a smooth curve. Fits a cubic spline through the heights of all arcs and
   reports the robust scatter of the residuals. With `rate_correction` the fit also removes
   the height-rate error (edotF * dRH/dt), which is large for tides.

Usage:
    python -m scripts.station_check --station glbx --year 2024 --doy 60 --doy_end 91 \
        [--rinex path/to/file.24o]
    (REFL_CODE must point to the gnssrefl workspace)
"""
import argparse
import glob
import gzip
import json
import os
import re
from pathlib import Path

import numpy as np

C = 299792458.0
# wavelength (m) per gnssrefl frequency code; GLONASS uses the centre of its band
WAVELENGTH = {
    1: C / 1575.42e6, 2: C / 1227.60e6, 20: C / 1227.60e6, 5: C / 1176.45e6,
    101: C / 1602.0e6, 102: C / 1246.0e6,
    201: C / 1575.42e6, 205: C / 1176.45e6, 206: C / 1278.75e6, 207: C / 1207.14e6, 208: C / 1191.795e6,
    302: C / 1561.098e6, 306: C / 1268.52e6, 307: C / 1207.14e6,
}
L2C_CODES = ("S2L", "S2S", "S2X")
LEGACY_L2_CODES = ("S2W", "S2P", "S2Y", "S2D", "S2M", "S2N")
RESULT_COLUMNS = "year doy RH sat UTC az amp emin emax n freq rise edotf pk delT mjd refr".split()


# ----------------------------------------------------------------- 1. sampling
def read_snr(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return np.loadtxt(f)


def in_azimuth_mask(az, azval2):
    az = np.asarray(az)
    ok = np.zeros(az.shape, dtype=bool)
    for a1, a2 in zip(azval2[0::2], azval2[1::2]):
        ok |= (az >= a1) & (az <= a2)
    return ok


def sampling_check(snr, e1, e2, azval2, max_height, freqs):
    """Largest resolvable height per signal for this SNR array (columns: sat, elev, azim, seconds, ...).

    Returns a dict: interval_s, step (95th percentile of |change of sin(elevation)| between
    samples), and per frequency: wavelength, resolvable height, pass flag, and the longest
    sample interval that still resolves `max_height`.
    """
    sat, ele, az, sec = snr[:, 0], snr[:, 1], snr[:, 2], snr[:, 3]
    keep = (ele >= e1) & (ele <= e2) & in_azimuth_mask(az, azval2)
    steps, dts = [], []
    for s in np.unique(sat[keep]):
        m = keep & (sat == s)
        t, x = sec[m], np.sin(np.radians(ele[m]))
        order = np.argsort(t)
        dt, dx = np.diff(t[order]), np.abs(np.diff(x[order]))
        if len(dt) == 0:
            continue
        good = dt <= 2.5 * np.median(dt)  # same arc, no gap
        steps.append(dx[good])
        dts.append(dt[good])
    if not steps:
        raise ValueError("no samples inside the elevation range and azimuth mask")
    steps, dts = np.concatenate(steps), np.concatenate(dts)
    interval = float(np.median(dts))
    step = float(np.percentile(steps, 95))
    out = {"interval_s": interval, "step_sin_elevation": step, "max_height_m": max_height, "signals": {}}
    for f in freqs:
        if f not in WAVELENGTH:
            continue
        h = WAVELENGTH[f] / (4.0 * step)
        out["signals"][f] = {
            "wavelength_m": WAVELENGTH[f],
            "resolvable_height_m": h,
            "ok": bool(h >= max_height),
            "longest_interval_s": interval * h / max_height,
        }
    return out


# ----------------------------------------------------------------- 2. L2 signal
def l2_signal_check(rinex_path, max_epochs=600):
    """Which GPS L2 signal-strength observables a RINEX file holds.

    Reads the header and the first `max_epochs` epochs. Returns a dict with the RINEX
    version, the S2 codes in the header, the count of non-empty values and of satellites
    per code, and a verdict: 'l2c', 'legacy_only', 'unknown_rinex2' or 'no_l2'.
    """
    opener = gzip.open if str(rinex_path).endswith(".gz") else open
    version, gps_types, counts, sats, epochs = None, [], {}, {}, 0
    rinex2_types = []
    with opener(rinex_path, "rt", errors="replace") as f:
        for line in f:
            label = line[60:].strip()
            if label == "RINEX VERSION / TYPE":
                version = float(line[:9])
            elif label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    system = line[0]
                if system == "G":
                    gps_types += line[7:60].split()
            elif label == "# / TYPES OF OBSERV":
                rinex2_types += line[6:60].split()
            elif label == "END OF HEADER":
                break
        if version is not None and version < 3:
            has_s2 = "S2" in rinex2_types
            return {"version": version, "header_codes": ["S2"] if has_s2 else [], "counts": {}, "satellites": {},
                    "epochs_read": 0, "verdict": "unknown_rinex2" if has_s2 else "no_l2"}
        s2 = [(i, c) for i, c in enumerate(gps_types) if c.startswith("S2")]
        for _, c in s2:
            counts[c], sats[c] = 0, set()
        for line in f:
            if line.startswith(">"):
                epochs += 1
                if epochs > max_epochs:
                    break
                continue
            if line[0] != "G":
                continue
            for i, c in s2:
                field = line[3 + 16 * i: 3 + 16 * i + 14].strip()
                if field:
                    try:
                        if float(field) > 0:
                            counts[c] += 1
                            sats[c].add(line[:3])
                    except ValueError:
                        pass
    with_data = [c for c in counts if counts[c] > 0]
    if any(c in L2C_CODES for c in with_data):
        verdict = "l2c"
    elif with_data:
        verdict = "legacy_only"
    else:
        verdict = "no_l2"
    return {"version": version, "header_codes": [c for _, c in s2], "counts": counts,
            "satellites": {c: len(v) for c, v in sats.items()}, "epochs_read": min(epochs, max_epochs), "verdict": verdict}


L2_VERDICT_TEXT = {
    "l2c": "The file holds the civil L2 signal (L2C). gnssrefl 4 reads it when it reads the RINEX 3 file directly.",
    "legacy_only": "The file holds only the legacy L2 signal. Results labelled L2C would come from the weaker legacy signal.",
    "unknown_rinex2": "RINEX 2 holds one L2 signal-strength value (S2). The file does not say which signal it is.",
    "no_l2": "No GPS L2 signal-strength values were found.",
}


# ----------------------------------------------------------------- 3. scatter about a smooth curve
def _spline(t, y, knot_hours):
    from scipy.interpolate import LSQUnivariateSpline

    order = np.argsort(t)
    t, y = t[order], y[order]
    step = knot_hours / 24.0
    knots = np.arange(t[0] + step, t[-1] - step / 2, step)
    # keep a knot only where data exist on both sides near it, so the fit is defined across gaps
    keep = [k for k in knots if np.any((t > k - step) & (t <= k)) and np.any((t > k) & (t <= k + step))]
    return LSQUnivariateSpline(t, y, keep, k=3) if len(t) > len(keep) + 4 else None


def scatter_about_spline(t_days, rh, edotf=None, knot_hours=3.0, rate_correction=True, reject=3.0):
    """Robust scatter of heights about a cubic spline.

    t_days: time of each arc in days. rh: reflector height (m). edotf: the gnssir edotF
    column (hours); needed for the rate correction. Returns a dict with n, n_rejected,
    robust scatter and RMS (m) of the residuals, the corrected heights, and `used`: which
    arcs the last fit kept (False for rejected arcs and for arcs with a missing value).
    """
    t, y = np.asarray(t_days, float), np.asarray(rh, float)
    use = np.isfinite(t) & np.isfinite(y)
    if rate_correction and edotf is not None:
        use &= np.isfinite(np.asarray(edotf, float))
    n_valid = int(use.sum())
    corr = np.zeros_like(y)
    if use.sum() < 20:
        return {"n": n_valid, "n_rejected": 0, "robust_scatter_m": np.nan, "rms_m": np.nan, "rh_corrected": y, "used": use}
    for _ in range(3):
        sp = _spline(t[use], (y - corr)[use], knot_hours)
        if sp is None:
            break
        res = (y - corr) - sp(t)
        sd = 1.4826 * np.median(np.abs(res[use] - np.median(res[use])))
        use = np.isfinite(res) & (np.abs(res - np.median(res[use])) <= reject * sd)
        if rate_correction and edotf is not None:
            corr = np.nan_to_num(np.asarray(edotf, float) * sp.derivative()(t) / 24.0)  # edotF (h) * dRH/dt (m/h)
            use &= np.isfinite(np.asarray(edotf, float))
    if sp is None:
        return {"n": n_valid, "n_rejected": 0, "robust_scatter_m": np.nan, "rms_m": np.nan, "rh_corrected": y, "used": use}
    res = (y - corr) - sp(t)
    return {
        "n": int(use.sum()),
        "n_rejected": int(n_valid - use.sum()),
        "robust_scatter_m": float(1.4826 * np.median(np.abs(res[use] - np.median(res[use])))),
        "rms_m": float(np.sqrt(np.mean(res[use] ** 2))),
        "rh_corrected": y - corr,
        "used": use,
    }


def load_results(refl_code, station, year, doy, doy_end, extension=""):
    rows = []
    sub = f"{extension}/" if extension else ""
    for f in sorted(glob.glob(f"{refl_code}/{year}/results/{station}/{sub}[0-9][0-9][0-9].txt*")):
        d = int(re.search(r"/(\d{3})\.txt", f).group(1))
        if doy <= d <= doy_end:
            a = np.loadtxt(f, comments="%", ndmin=2)
            if a.size:
                rows.append(a[:, :17])
    return np.vstack(rows) if rows else np.empty((0, 17))


# ----------------------------------------------------------------- command line
def main(argv=None):
    p = argparse.ArgumentParser(description="Sampling, L2 signal and scatter checks for one station.")
    p.add_argument("--station", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    p.add_argument("--doy_end", type=int)
    p.add_argument("--rinex", help="RINEX observation file for the L2 signal check")
    p.add_argument("--knot_hours", type=float, default=3.0, help="spline knot spacing; use 24 or more for a lake")
    p.add_argument("--extension", default="")
    a = p.parse_args(argv)
    rc = os.environ.get("REFL_CODE")
    if not rc:
        raise SystemExit("Set REFL_CODE to the gnssrefl workspace.")
    sta, end = a.station.lower(), a.doy_end or a.doy
    name = f"{sta}.{a.extension}.json" if a.extension else f"{sta}.json"
    cfg = json.loads(Path(rc, "input", name).read_text())

    print(f"Station {sta.upper()}, {a.year} days {a.doy:03d} to {end:03d}")
    print("\n1. Sampling check")
    snr_files = sorted(glob.glob(f"{rc}/{a.year}/snr/{sta}/{sta}{a.doy:03d}0.{a.year % 100:02d}.snr*"))
    if snr_files:
        r = sampling_check(read_snr(snr_files[0]), cfg["e1"], cfg["e2"], cfg.get("azval2", [0, 360]), cfg["maxH"], cfg["freqs"])
        print(f"   Sample interval {r['interval_s']:.0f} s. Largest height in the station file: {r['max_height_m']:.1f} m.")
        # the shortest wavelength is the hardest case
        f, s_ = min(r["signals"].items(), key=lambda kv: kv[1]["wavelength_m"])
        print(f"   Hardest signal (wavelength {s_['wavelength_m']:.3f} m): the sampling resolves heights to "
              f"{s_['resolvable_height_m']:.0f} m ({'OK' if s_['ok'] else 'TOO SLOW'}).")
        print(f"   The longest sample interval that still resolves {r['max_height_m']:.1f} m is {s_['longest_interval_s']:.0f} s.")
    else:
        print("   No SNR file for the first day.")

    print("\n2. L2 signal check")
    if a.rinex:
        r = l2_signal_check(a.rinex)
        print(f"   RINEX {r['version']}, first {r['epochs_read']} epochs. " + L2_VERDICT_TEXT[r["verdict"]])
        for c in r["header_codes"]:
            if c in r["counts"]:
                kind = "L2C" if c in L2C_CODES else "legacy"
                print(f"   {c} ({kind}): {r['counts'][c]} values, {r['satellites'][c]} satellites")
    else:
        print("   Not run. Give --rinex.")

    print("\n3. Scatter of the heights about a smooth curve")
    d = load_results(rc, sta, a.year, a.doy, end, a.extension)
    if len(d) >= 20:
        t = d[:, 1] + d[:, 4] / 24.0
        for label, corr in (("without the height-rate correction", False), ("with the height-rate correction", True)):
            r = scatter_about_spline(t, d[:, 2], d[:, 12], a.knot_hours, corr)
            print(f"   {label}: robust scatter {r['robust_scatter_m']:.3f} m, RMS {r['rms_m']:.3f} m, "
                  f"{r['n']} arcs used, {r['n_rejected']} rejected")
        print(f"   Arcs per day: {len(d) / (end - a.doy + 1):.0f}. Spline knots every {a.knot_hours:g} hours.")
    else:
        print(f"   Too few arcs ({len(d)}).")


if __name__ == "__main__":
    main()
