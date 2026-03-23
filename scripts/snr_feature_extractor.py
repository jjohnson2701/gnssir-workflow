# ABOUTME: Extract per-arc SNR features (CLR, area factor, damping, etc.) from raw SNR files
# ABOUTME: Reads gnssrefl SNR files, matches arcs to per-arc parquet, computes literature-based indicators

"""
Per-arc SNR feature extractor for ice detection.

Reads raw SNR files produced by rinex2snr, matches observation segments to the
per-arc parquet from gnssrefl, and computes six indicators from the detrended
SNR arcs:

  CLR   — clarity ratio (Purnell 2024): P1 / mean(other peaks) in LSP
  PR    — peak ratio (Purnell 2024): P1 / P2
  AF    — area factor (Song 2022): integral of CWT power curve at dominant RH
  gamma — damping parameter (Strandberg 2017): envelope decay rate
  MS    — mean raw SNR (dB)
  VS    — variance of detrended SNR

Usage:
    python scripts/snr_feature_extractor.py --station UMNQ --year 2025
    python scripts/snr_feature_extractor.py --station UMNQ --year 2025 --num_cores 8
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle, find_peaks, hilbert, cwt, morlet2

# scipy.signal.cwt is deprecated in 1.12, removed in 1.15. We're on 1.13.
# When upgrading scipy, migrate to pywt.cwt.
warnings.filterwarnings("ignore", message="scipy.signal.cwt is deprecated",
                        category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Frequency code → SNR file column index (0-based)
# From gnssrefl read_snr_files.py: columns are PRN, ele, az, sod, edot, S6, S1, S2, S5, S7, S8
FREQ_TO_COL = {
    1: 6, 101: 6, 201: 6, 301: 6,       # L1 → column 6
    2: 7, 20: 7, 102: 7, 302: 7,         # L2C → column 7
    5: 8, 205: 8,                          # L5 → column 8
    206: 5, 306: 5,                        # E5b/B3 → column 5
    207: 9, 307: 9,                        # E5 → column 9
    208: 10,                               # E6 → column 10
}

# GNSS carrier wavelengths in meters
FREQ_WAVELENGTH = {
    1: 0.19029, 101: 0.19029, 201: 0.19029, 301: 0.19029,   # L1
    2: 0.24421, 20: 0.24421, 102: 0.24421, 302: 0.24421,    # L2
    5: 0.25482, 205: 0.25482,                                 # L5
    206: 0.24834, 306: 0.24834,                                # E5b
    207: 0.25478, 307: 0.25478,                                # E5
    208: 0.23405,                                              # E6
}


# ---------------------------------------------------------------------------
# SNR file I/O
# ---------------------------------------------------------------------------

def read_snr_file(path):
    """Read a gnssrefl SNR file into a numpy array.

    Returns ndarray of shape (n_obs, 11):
        col 0: satellite PRN
        col 1: elevation angle (deg)
        col 2: azimuth (deg)
        col 3: seconds of day
        col 4: edot (elevation rate)
        col 5-10: S6, S1, S2, S5, S7, S8 (dB-Hz)
    """
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            data = np.loadtxt(f)
    else:
        data = np.loadtxt(path)
    return data


# ---------------------------------------------------------------------------
# Arc segmentation
# ---------------------------------------------------------------------------

def segment_satellite_arcs(seconds_of_day, elevation, gap_seconds=300):
    """Segment one satellite's observations into individual arcs.

    Splits on time gaps > gap_seconds OR elevation direction reversals.

    Returns list of dicts, each with:
        start_idx: int — first index of arc in input arrays
        end_idx:   int — one-past-last index (for slicing)
        rise:      int — 1 if elevation increasing, -1 if decreasing
    """
    n = len(seconds_of_day)
    if n < 2:
        return []

    # Track direction: 1=rising, -1=setting, 0=undetermined
    last_dir = 0

    splits = [0]
    for i in range(1, n):
        dt = seconds_of_day[i] - seconds_of_day[i - 1]
        if dt > gap_seconds or dt < 0:
            splits.append(i)
            last_dir = 0
            continue

        d = elevation[i] - elevation[i - 1]
        if abs(d) < 0.01:
            continue  # skip near-zero changes (apex plateau)

        cur_dir = 1 if d > 0 else -1
        if last_dir != 0 and cur_dir != last_dir:
            splits.append(i)
        last_dir = cur_dir

    arcs = []
    for j in range(len(splits)):
        start = splits[j]
        end = splits[j + 1] if j + 1 < len(splits) else n
        if end - start < 5:
            continue

        seg_ele = elevation[start:end]
        net = seg_ele[-1] - seg_ele[0]
        rise = 1 if net > 0 else -1

        arcs.append({
            "start_idx": start,
            "end_idx": end,
            "rise": rise,
        })
    return arcs


# ---------------------------------------------------------------------------
# Arc matching
# ---------------------------------------------------------------------------

def find_matching_segment(arcs, seconds_of_day, elevation,
                          target_utctime, target_rise, e1, e2):
    """Find which arc segment matches a per-arc parquet entry.

    Args:
        arcs: list of dicts from segment_satellite_arcs
        seconds_of_day: full array of SOD for this satellite
        elevation: full array of elevation for this satellite
        target_utctime: UTCtime (hours) from per-arc parquet
        target_rise: rise direction (1 or -1) from per-arc parquet
        e1, e2: elevation angle window for computing windowed mean time

    Returns:
        Index into arcs list, or -1 if no match.
    """
    best_idx = -1
    best_dt = float("inf")

    for i, arc in enumerate(arcs):
        if arc["rise"] != target_rise:
            continue

        seg_sod = seconds_of_day[arc["start_idx"]:arc["end_idx"]]
        seg_ele = elevation[arc["start_idx"]:arc["end_idx"]]

        # Window to [e1, e2]
        mask = (seg_ele >= e1) & (seg_ele <= e2)
        if mask.sum() < 3:
            continue

        windowed_sod = seg_sod[mask]
        mean_utc_hours = np.mean(windowed_sod) / 3600.0

        dt = abs(mean_utc_hours - target_utctime)
        # Handle midnight wrap
        if dt > 12:
            dt = 24 - dt

        if dt < best_dt:
            best_dt = dt
            best_idx = i

    # Require match within 1 hour
    if best_dt > 1.0:
        return -1
    return best_idx


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------

def detrend_arc(elevation, snr_linear, poly_order=4, pele=(5, 30)):
    """Remove polynomial trend from SNR arc (replicates gnssrefl approach).

    Args:
        elevation: array of elevation angles (degrees)
        snr_linear: SNR in linear units (already converted from dB via 10^(dB/20))
        poly_order: polynomial order for trend removal
        pele: (min, max) elevation range for polynomial fitting

    Returns:
        detrended: array same length as input, trend subtracted
    """
    pele_min, pele_max = pele
    fit_mask = (elevation >= pele_min) & (elevation <= pele_max)
    if fit_mask.sum() < poly_order + 1:
        # Not enough points — return centered data
        return snr_linear - np.mean(snr_linear)

    coeffs = np.polyfit(elevation[fit_mask], snr_linear[fit_mask], poly_order)
    trend = np.polyval(coeffs, elevation)
    return snr_linear - trend


# ---------------------------------------------------------------------------
# LSP feature computation
# ---------------------------------------------------------------------------

def compute_lsp_features(sin_elev, detrended, wavelength,
                         min_rh, max_rh, precision):
    """Compute LSP-derived features: CLR, PR, SP, RH.

    Args:
        sin_elev: sin(elevation) array
        detrended: detrended SNR array (same length)
        wavelength: carrier wavelength in meters
        min_rh, max_rh: reflector height search range (meters)
        precision: RH step size (meters)

    Returns:
        dict with keys: CLR, PR, SP, RH
    """
    cf = wavelength / 2
    x = sin_elev / cf

    # Frequency grid (reflector heights)
    rh_grid = np.arange(min_rh, max_rh + precision, precision)
    angular_freq = 2 * np.pi * rh_grid

    # Lomb-Scargle periodogram
    pgram = lombscargle(x, detrended, angular_freq, normalize=False)
    # Convert to amplitude (matching gnssrefl's scaling)
    amp = 2 * np.sqrt(pgram / len(x))

    # Peak detection
    peak_indices, _ = find_peaks(amp)
    if len(peak_indices) == 0:
        # Fallback: use global max
        peak_idx = np.argmax(amp)
        return {"CLR": 1.0, "PR": 1.0, "SP": float(amp[peak_idx]),
                "RH": float(rh_grid[peak_idx])}

    peak_amps = amp[peak_indices]
    sorted_idx = np.argsort(peak_amps)[::-1]
    p1_idx = peak_indices[sorted_idx[0]]
    p1 = peak_amps[sorted_idx[0]]

    # CLR = P1 / mean(all other peaks)
    if len(peak_amps) > 1:
        other_peaks = np.delete(peak_amps, sorted_idx[0])
        clr = float(p1 / np.mean(other_peaks))
    else:
        clr = float(p1)  # only one peak

    # PR = P1 / P2
    if len(peak_amps) >= 2:
        p2 = peak_amps[sorted_idx[1]]
        pr = float(p1 / p2) if p2 > 0 else float(p1)
    else:
        pr = float(p1)

    return {
        "CLR": clr,
        "PR": pr,
        "SP": float(p1),
        "RH": float(rh_grid[p1_idx]),
    }


# ---------------------------------------------------------------------------
# Area factor (Song 2022)
# ---------------------------------------------------------------------------

def compute_area_factor(sin_elev, detrended, wavelength, min_rh, max_rh):
    """Compute wavelet-derived area factor.

    Uses CWT with Morlet wavelet on dSNR(sin(ε)) to get 2D power spectrum.
    Extracts the power curve at the dominant RH frequency and integrates.

    Args:
        sin_elev: sin(elevation) array (must be sorted ascending)
        detrended: detrended SNR array
        wavelength: carrier wavelength in meters
        min_rh, max_rh: RH range in meters

    Returns:
        float: area factor (integrated power at dominant RH)
    """
    # Sort by sin(elev) for consistent CWT input
    sort_idx = np.argsort(sin_elev)
    x = sin_elev[sort_idx]
    y = detrended[sort_idx]

    if len(y) < 20:
        return np.nan

    cf = wavelength / 2
    # The SNR oscillation frequency in the sin(e)/cf domain is the RH value.
    # CWT scale parameter relates to frequency: scale ∝ 1/frequency.
    # For Morlet wavelet with w=5 (default), center frequency ≈ w/(2π).
    # We need scales that map to the RH range.

    # Sampling rate in the sin(e)/cf domain
    dx = np.mean(np.diff(x / cf))
    if dx <= 0:
        return np.nan

    # Morlet center frequency (w parameter)
    w = 5.0
    # Scale to frequency: freq = w / (2π * scale * dx)
    # RH = freq, so scale = w / (2π * RH * dx)
    rh_values = np.linspace(min_rh, max_rh, 100)
    scales = w / (2 * np.pi * rh_values * dx)
    # Filter out invalid scales
    valid = scales > 1
    if valid.sum() < 5:
        return np.nan
    scales = scales[valid]
    rh_values = rh_values[valid]

    # Compute CWT
    # scipy.signal.cwt signature: cwt(data, wavelet, widths)
    # morlet2 signature: morlet2(M, w, s) where M=length, w=omega0, s=scale
    cwtmatr = cwt(y, morlet2, scales, w=w)
    power = np.abs(cwtmatr) ** 2

    # Find dominant RH (scale with max integrated power)
    integrated = np.sum(power, axis=1)
    dominant_idx = np.argmax(integrated)

    # Power curve at dominant RH frequency
    power_curve = power[dominant_idx, :]

    # Area factor = integral of power curve (trapezoidal)
    af = float(np.trapz(power_curve, x / cf))
    return af


# ---------------------------------------------------------------------------
# Damping parameter (Strandberg 2017)
# ---------------------------------------------------------------------------

def compute_damping(elevation_deg, detrended, wavelength):
    """Compute damping parameter γ from envelope of detrended SNR.

    Uses Hilbert transform to get the signal envelope, then fits
    log(envelope) = log(A) - 4k²γ sin²(ε) to extract γ.

    Args:
        elevation_deg: elevation angles in degrees
        detrended: detrended SNR array
        wavelength: carrier wavelength in meters

    Returns:
        float: damping parameter γ (>=0)
    """
    if len(detrended) < 20:
        return np.nan

    # Hilbert envelope
    analytic = hilbert(detrended)
    envelope = np.abs(analytic)

    # Avoid log(0)
    envelope = np.maximum(envelope, 1e-10)

    sin2_e = np.sin(np.radians(elevation_deg)) ** 2
    log_env = np.log(envelope)

    # Linear fit: log(env) = intercept + slope * sin²(ε)
    # slope = -4k²γ → γ = -slope / (4k²)
    k = 2 * np.pi / wavelength
    try:
        coeffs = np.polyfit(sin2_e, log_env, 1)
        slope = coeffs[0]
        gamma = -slope / (4 * k**2)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    # Physical damping is non-negative; clamp
    return max(0.0, float(gamma))


# ---------------------------------------------------------------------------
# Full per-arc feature extraction
# ---------------------------------------------------------------------------

def extract_arc_features(elevation, snr_db, snr_linear, detrended,
                         wavelength, e1, e2, min_rh, max_rh, precision):
    """Compute all 6 features + full_arc flag for one arc on one frequency.

    Args:
        elevation: elevation angles (degrees) for the arc segment
        snr_db: raw SNR in dB for computing MS
        snr_linear: SNR in linear units (for detrending reference)
        detrended: already-detrended SNR
        wavelength: carrier wavelength
        e1, e2: elevation window
        min_rh, max_rh: RH search range
        precision: LSP step

    Returns:
        dict with CLR, PR, AF, gamma, MS, VS, SP, RH, full_arc
    """
    # Window to [e1, e2]
    mask = (elevation >= e1) & (elevation <= e2)
    n_windowed = mask.sum()
    if n_windowed < 15:
        return None

    ele_w = elevation[mask]
    sin_e = np.sin(np.radians(ele_w))
    dsnr = detrended[mask]
    snr_db_w = snr_db[mask]

    # Full-arc check (Song 2022): must span close to [e1, e2]
    ele_range = ele_w.max() - ele_w.min()
    expected_range = e2 - e1
    full_arc = ele_range >= 0.8 * expected_range

    # LSP features
    lsp = compute_lsp_features(sin_e, dsnr, wavelength, min_rh, max_rh, precision)

    # Area factor (only meaningful for full arcs, but compute anyway)
    af = compute_area_factor(sin_e, dsnr, wavelength, min_rh, max_rh)

    # Damping
    gamma = compute_damping(ele_w, dsnr, wavelength)

    # MS (mean raw SNR in dB) and VS (variance of detrended)
    ms = float(np.mean(snr_db_w))
    vs = float(np.var(dsnr))

    return {
        "CLR": lsp["CLR"],
        "PR": lsp["PR"],
        "SP": lsp["SP"],
        "RH": lsp["RH"],
        "AF": af,
        "gamma": gamma,
        "MS": ms,
        "VS": vs,
        "full_arc": full_arc,
    }


# ---------------------------------------------------------------------------
# Day-level processing
# ---------------------------------------------------------------------------

def _load_station_config(station):
    """Load gnssrefl processing parameters for a station."""
    # Try station-specific json first
    stations_cfg = PROJECT_ROOT / "config" / "stations_config.json"
    if stations_cfg.exists():
        with open(stations_cfg) as f:
            all_cfg = json.load(f)
        if station in all_cfg:
            gnssir_path = all_cfg[station].get("gnssir_json_params_path")
            if gnssir_path:
                full_path = PROJECT_ROOT / gnssir_path
                if full_path.exists():
                    with open(full_path) as f:
                        return json.load(f)
    return None


def _snr_file_path(station, year, doy):
    """Construct path to an SNR file."""
    station_lower = station.lower()
    yy = str(year)[-2:]
    doy_str = f"{doy:03d}"
    return (PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code"
            / str(year) / "snr" / station_lower
            / f"{station_lower}{doy_str}0.{yy}.snr66")


def extract_features_for_day(station, year, doy, per_arc_day, config):
    """Extract all SNR features for one day.

    Args:
        station: station ID (e.g., "UMNQ")
        year: processing year
        doy: day of year
        per_arc_day: DataFrame — per_arc parquet rows for this day
        config: dict from station gnssir json (e1, e2, minH, maxH, polyV, pele, desiredP)

    Returns:
        list of dicts, one per (arc, frequency) pair with features + join keys
    """
    snr_path = _snr_file_path(station, year, doy)
    if not snr_path.exists():
        # Try gzipped
        gz_path = Path(str(snr_path) + ".gz")
        if gz_path.exists():
            snr_path = gz_path
        else:
            logger.debug(f"No SNR file for DOY {doy}: {snr_path}")
            return []

    try:
        snr_data = read_snr_file(snr_path)
    except Exception as e:
        logger.warning(f"Failed to read SNR file DOY {doy}: {e}")
        return []

    e1 = config["e1"]
    e2 = config["e2"]
    min_rh = config["minH"]
    max_rh = config["maxH"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))
    precision = config.get("desiredP", 0.005)

    # Group parquet rows by physical arc (sat, UTCtime, rise)
    # Multiple freqs share the same physical arc
    arc_groups = per_arc_day.groupby(["sat", "UTCtime", "rise"])

    results = []
    segment_cache = {}  # PRN → (arcs, sat_data)

    for (sat, utctime, rise), freq_rows in arc_groups:
        # Get or compute arc segments for this satellite
        if sat not in segment_cache:
            sat_mask = snr_data[:, 0] == sat
            sat_data = snr_data[sat_mask]
            if len(sat_data) < 5:
                segment_cache[sat] = ([], None)
            else:
                arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
                segment_cache[sat] = (arcs, sat_data)

        arcs, sat_data = segment_cache[sat]
        if sat_data is None or len(arcs) == 0:
            continue

        # Find matching segment
        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime,
            target_rise=rise,
            e1=e1, e2=e2,
        )
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        # Process each frequency for this physical arc
        for _, row in freq_rows.iterrows():
            freq = row["freq"]
            col_idx = FREQ_TO_COL.get(freq)
            if col_idx is None or col_idx >= arc_data.shape[1]:
                continue

            snr_db_col = arc_data[:, col_idx]
            # Skip if no signal (all zeros)
            if np.all(snr_db_col == 0):
                continue

            wavelength = FREQ_WAVELENGTH.get(freq)
            if wavelength is None:
                continue

            # Convert to linear and detrend
            snr_lin = np.power(10, snr_db_col / 20)
            detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

            # Extract features
            feats = extract_arc_features(
                arc_ele, snr_db_col, snr_lin, detrended,
                wavelength, e1, e2, min_rh, max_rh, precision,
            )
            if feats is None:
                continue

            feats["doy"] = doy
            feats["sat"] = int(sat)
            feats["UTCtime"] = float(utctime)
            feats["rise"] = int(rise)
            feats["freq"] = int(freq)
            results.append(feats)

    return results


# ---------------------------------------------------------------------------
# Station-year processing
# ---------------------------------------------------------------------------

def _process_day_worker(args):
    """Multiprocessing worker for extract_features. Must be module-level for pickle."""
    station, year, doy, day_arcs_records, config = args
    day_arcs = pd.DataFrame.from_records(day_arcs_records)
    return extract_features_for_day(station, year, doy, day_arcs, config)


def extract_features(station, year, num_cores=1):
    """Extract SNR features for a full station-year.

    Reads per-arc parquet as index, processes each day's SNR file,
    and outputs {station}_{year}_snr_features.parquet.
    """
    config = _load_station_config(station)
    if config is None:
        logger.error(f"No config found for {station}")
        return None

    per_arc_path = (PROJECT_ROOT / "results_annual" / station
                    / f"{station}_{year}_per_arc.parquet")
    if not per_arc_path.exists():
        logger.error(f"Per-arc parquet not found: {per_arc_path}")
        return None

    per_arc = pd.read_parquet(per_arc_path)
    doys = sorted(per_arc["doy"].unique())
    logger.info(f"Processing {station} {year}: {len(doys)} days, {len(per_arc)} arcs")

    if num_cores > 1:
        from multiprocessing import Pool

        # Build serializable args per day (list of dicts, not DataFrame)
        day_args = []
        for doy in doys:
            day_arcs = per_arc[per_arc["doy"] == doy]
            day_args.append((station, year, doy, day_arcs.to_dict("records"), config))

        with Pool(num_cores) as pool:
            day_results = pool.map(_process_day_worker, day_args)
        all_results = [r for day in day_results for r in day]
    else:
        all_results = []
        for i, doy in enumerate(doys):
            if (i + 1) % 50 == 0:
                logger.info(f"  Day {i + 1}/{len(doys)} (DOY {doy})")
            day_arcs = per_arc[per_arc["doy"] == doy]
            all_results.extend(
                extract_features_for_day(station, year, doy, day_arcs, config)
            )

    if not all_results:
        logger.error("No features extracted")
        return None

    df = pd.DataFrame(all_results)
    out_path = (PROJECT_ROOT / "results_annual" / station
                / f"{station}_{year}_snr_features.parquet")
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} feature rows to {out_path}")

    # Summary
    logger.info(f"Features: CLR={df['CLR'].median():.2f}, "
                f"AF={df['AF'].median():.2f}, "
                f"gamma={df['gamma'].median():.4f}")
    full_pct = df["full_arc"].mean() * 100
    logger.info(f"Full arcs: {full_pct:.1f}%")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract SNR arc features")
    parser.add_argument("--station", required=True, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--num_cores", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = extract_features(args.station, args.year, num_cores=args.num_cores)
    if df is None:
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SNR Feature Extraction: {args.station} {args.year}")
    print(f"{'='*60}")
    print(f"Total feature rows: {len(df)}")
    print(f"Full arcs: {df['full_arc'].sum()} / {len(df)} "
          f"({df['full_arc'].mean()*100:.1f}%)")
    print(f"\nFeature medians:")
    for col in ["CLR", "PR", "AF", "gamma", "MS", "VS"]:
        if col in df.columns:
            print(f"  {col:>6s}: {df[col].median():.4f}")


if __name__ == "__main__":
    main()
