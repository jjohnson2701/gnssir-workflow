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
from scipy.signal import lombscargle, find_peaks, hilbert
from scipy.signal.windows import tukey
from scipy.stats import siegelslopes


def _morlet2(M, w, s):
    """Morlet wavelet (replacement for removed scipy.signal.morlet2)."""
    t = np.arange(0, M) - (M - 1.0) / 2
    t = t / s
    output = np.exp(1j * w * t) * np.exp(-0.5 * t ** 2) * np.pi ** (-0.25)
    return output


def _cwt(data, wavelet_func, widths, **kwargs):
    """Continuous wavelet transform (replacement for removed scipy.signal.cwt)."""
    N = len(data)
    out = np.empty((len(widths), N), dtype=complex)
    for i, width in enumerate(widths):
        wavelet = wavelet_func(N, kwargs.get("w", 5), width)
        out[i] = np.convolve(data, np.conj(wavelet[::-1]), mode="same")
    return out

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
                "RH": float(rh_grid[peak_idx]),
                "clr_peak_power": float(amp[peak_idx]),
                "clr_total_power": 0.0}

    peak_amps = amp[peak_indices]
    sorted_idx = np.argsort(peak_amps)[::-1]
    p1_idx = peak_indices[sorted_idx[0]]
    p1 = peak_amps[sorted_idx[0]]

    # CLR = P1 / mean(all other peaks)
    # Store components separately for Gate 2 decomposition (feature profiling)
    if len(peak_amps) > 1:
        other_peaks = np.delete(peak_amps, sorted_idx[0])
        clr_total_power = float(np.mean(other_peaks))
        clr = float(p1 / clr_total_power)
    else:
        clr_total_power = 0.0
        clr = float(p1)  # only one peak
    clr_peak_power = float(p1)

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
        "clr_peak_power": clr_peak_power,
        "clr_total_power": clr_total_power,
    }


# ---------------------------------------------------------------------------
# Area factor (Song 2022)
# ---------------------------------------------------------------------------

def compute_area_factor(sin_elev, detrended, wavelength, min_rh, max_rh,
                        baseline_power_curve=None, baseline_sin_grid=None,
                        return_power=False):
    """Compute wavelet-derived area factor.

    Uses CWT with Morlet wavelet on dSNR(sin(ε)) to get 2D power spectrum.
    Extracts the power curve at the dominant RH frequency and integrates.

    If baseline_power_curve is provided, subtracts the per-PRN average power
    curve before integration (Song 2022 Eq. 19), isolating surface-driven
    power changes from the antenna gain pattern.

    Args:
        sin_elev: sin(elevation) array (must be sorted ascending)
        detrended: detrended SNR array
        wavelength: carrier wavelength in meters
        min_rh, max_rh: RH range in meters
        baseline_power_curve: optional per-PRN average power curve to subtract
        baseline_sin_grid: sin(ε) grid the baseline is defined on
        return_power: if True, also return the 2D power matrix and metadata

    Returns:
        float: area factor (integrated power at dominant RH)
        If return_power=True, returns (af, power_info_dict) instead.
    """
    # Sort by sin(elev) for consistent CWT input
    sort_idx = np.argsort(sin_elev)
    x = sin_elev[sort_idx]
    y = detrended[sort_idx]

    nan_result = (np.nan, None) if return_power else np.nan

    if len(y) < 20:
        return nan_result

    cf = wavelength / 2
    # The SNR oscillation frequency in the sin(e)/cf domain is the RH value.
    # CWT scale parameter relates to frequency: scale ∝ 1/frequency.
    # For Morlet wavelet with w=5 (default), center frequency ≈ w/(2π).
    # We need scales that map to the RH range.

    # Sampling rate in the sin(e)/cf domain
    dx = np.mean(np.diff(x / cf))
    if dx <= 0:
        return nan_result

    # Morlet center frequency (w parameter)
    w = 5.0
    # Scale to frequency: freq = w / (2π * scale * dx)
    # RH = freq, so scale = w / (2π * RH * dx)
    rh_values = np.linspace(min_rh, max_rh, 100)
    scales = w / (2 * np.pi * rh_values * dx)
    # Filter out invalid scales
    valid = scales > 1
    if valid.sum() < 5:
        return nan_result
    scales = scales[valid]
    rh_values = rh_values[valid]

    # Compute CWT
    # scipy.signal.cwt signature: cwt(data, wavelet, widths)
    # morlet2 signature: morlet2(M, w, s) where M=length, w=omega0, s=scale
    cwtmatr = _cwt(y, _morlet2, scales, w=w)
    power = np.abs(cwtmatr) ** 2

    # Find dominant RH (scale with max integrated power)
    integrated = np.sum(power, axis=1)
    dominant_idx = np.argmax(integrated)

    # Power curve at dominant RH frequency
    power_curve = power[dominant_idx, :]

    # Subtract per-PRN baseline if provided (Song 2022)
    if baseline_power_curve is not None and baseline_sin_grid is not None:
        from scipy.interpolate import interp1d
        # Baseline is stored on a sin(ε) grid — query at sin(ε) coordinates,
        # NOT sin(ε)/cf. (The power curve has one value per sample; each sample's
        # sin(ε) coordinate is x, its rescaled coordinate is x/cf.)
        bl_interp = interp1d(
            baseline_sin_grid, baseline_power_curve,
            bounds_error=False, fill_value=np.nan,
        )(x)
        # Check baseline domain coverage
        valid_bl = ~np.isnan(bl_interp)
        oob_frac = 1.0 - valid_bl.mean() if len(valid_bl) > 0 else 1.0
        if oob_frac <= 0.5 and valid_bl.sum() >= 5:
            # Enough coverage — subtract baseline where defined, keep raw power elsewhere
            power_curve[valid_bl] = np.maximum(power_curve[valid_bl] - bl_interp[valid_bl], 0.0)
            if oob_frac > 0.2:
                log.debug("AF baseline: %.0f%% of arc outside baseline domain", oob_frac * 100)
        # else: baseline domain too narrow, use uncorrected power curve
        af = float(np.trapz(power_curve, x / cf))
    else:
        # Area factor = integral of power curve (trapezoidal)
        af = float(np.trapz(power_curve, x / cf))

    if return_power:
        power_info = {
            "power": power,
            "rh_values": rh_values,
            "sin_elev": x / cf,
            "dominant_idx": dominant_idx,
        }
        return af, power_info
    return af


# ---------------------------------------------------------------------------
# Damping parameter (Strandberg 2017)
# ---------------------------------------------------------------------------

def compute_damping(elevation_deg, detrended, wavelength, taper_frac=0.1):
    """Compute damping parameter γ from envelope of detrended SNR.

    Uses Hilbert transform to get the signal envelope, then fits
    log(envelope) = log(A) - 4k²γ sin²(ε) to extract γ.

    A Tukey window is applied before the Hilbert transform to suppress
    edge artifacts from spectral leakage (the FFT-based Hilbert assumes
    periodicity). The tapered edges are excluded from the fit.

    The fit uses Siegel repeated-medians regression (scipy.stats.siegelslopes)
    instead of OLS polyfit, making it robust to near-zero envelope values
    at destructive interference nodes.

    Args:
        elevation_deg: elevation angles in degrees
        detrended: detrended SNR array
        wavelength: carrier wavelength in meters
        taper_frac: fraction of each edge to taper (default 0.1 = 10%)

    Returns:
        tuple: (gamma, gamma_r2) where gamma is the damping parameter (>=0)
               and gamma_r2 is the R² of the log-envelope fit (0–1).
               Returns (np.nan, np.nan) on failure.
    """
    nan_result = (np.nan, np.nan)
    n = len(detrended)
    if n < 20:
        return nan_result

    # Tukey window: cosine taper on edges, flat in center
    window = tukey(n, alpha=2 * taper_frac)
    tapered = detrended * window

    # Hilbert envelope on tapered signal
    analytic = hilbert(tapered)
    envelope = np.abs(analytic)

    # Trim tapered edges from the fit (they are artificially suppressed)
    trim = max(1, int(n * taper_frac))
    envelope = envelope[trim:-trim]
    elev_fit = elevation_deg[trim:-trim]

    if len(envelope) < 15:
        return nan_result

    # Avoid log(0)
    envelope = np.maximum(envelope, 1e-10)

    sin2_e = np.sin(np.radians(elev_fit)) ** 2
    log_env = np.log(envelope)

    # Robust linear fit: log(env) = intercept + slope * sin²(ε)
    # slope = -4k²γ → γ = -slope / (4k²)
    k = 2 * np.pi / wavelength
    try:
        slope, intercept = siegelslopes(log_env, sin2_e)
        gamma = -slope / (4 * k**2)
    except (ValueError, RuntimeError):
        return nan_result

    # R² of the fit (on the robust line)
    predicted = intercept + slope * sin2_e
    ss_res = np.sum((log_env - predicted) ** 2)
    ss_tot = np.sum((log_env - np.mean(log_env)) ** 2)
    gamma_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Physical damping is non-negative; clamp
    return (max(0.0, float(gamma)), float(np.clip(gamma_r2, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Phase extraction (Strandberg et al. 2017, Muñoz-Martín et al. 2020)
# ---------------------------------------------------------------------------

def compute_phase(sin_elev, detrended, wavelength, rh):
    """Extract reflection phase from detrended SNR at known RH.

    Uses matched-filter (inner product) approach: project dSNR onto
    cos and sin reference signals at the known oscillation frequency
    ω = 4πRH/λ in the sin(ε) domain.

    The SNR model is:
        dSNR(ε) = A · exp(−γ·sin²ε) · cos(ω·sin(ε) + φ)

    Phase responds to the dielectric properties of the reflecting surface.
    A permittivity change (water→ice, dry ice→wet melt) shifts the Fresnel
    reflection coefficient phase.

    Args:
        sin_elev: sin(elevation) array
        detrended: detrended SNR array (same length)
        wavelength: carrier wavelength in meters
        rh: reflector height from LSP (meters)

    Returns:
        float: phase in radians, wrapped to [−π, π]
    """
    if len(detrended) < 10 or rh <= 0:
        return np.nan

    # Oscillation frequency in the sin(ε) domain
    omega = 4 * np.pi * rh / wavelength

    # Reference signals at the known frequency
    arg = omega * sin_elev
    cos_ref = np.cos(arg)
    sin_ref = np.sin(arg)

    # Inner products (matched filter projection)
    # dSNR ≈ A·cos(ω·sin(ε) + φ)
    #       = A·cos(φ)·cos(ω·sin(ε)) − A·sin(φ)·sin(ω·sin(ε))
    # So: a ∝ A·cos(φ),  b ∝ A·sin(φ)
    a = np.sum(detrended * cos_ref)
    b = np.sum(detrended * sin_ref)

    if a == 0 and b == 0:
        return np.nan

    phase = np.arctan2(-b, a)
    return float(phase)


# ---------------------------------------------------------------------------
# Full per-arc feature extraction
# ---------------------------------------------------------------------------

def extract_arc_features(elevation, snr_db, snr_linear, detrended,
                         wavelength, e1, e2, min_rh, max_rh, precision,
                         af_baseline=None, af_baseline_sin_grid=None):
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
        af_baseline: optional per-PRN power curve baseline for AF correction
        af_baseline_sin_grid: sin(ε) grid the baseline is defined on

    Returns:
        dict with CLR, PR, AF, gamma, phase, MS, VS, SP, RH, full_arc
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

    # AF and gamma are mechanically biased by truncated elevation ranges:
    # AF integrates over a shorter domain, gamma has less lever arm for the fit.
    # Only compute for full arcs (Song 2022 ≥80% coverage).
    if full_arc:
        af = compute_area_factor(
            sin_e, dsnr, wavelength, min_rh, max_rh,
            baseline_power_curve=af_baseline,
            baseline_sin_grid=af_baseline_sin_grid,
        )
        gamma, gamma_r2 = compute_damping(ele_w, dsnr, wavelength)
    else:
        af = np.nan
        gamma = np.nan
        gamma_r2 = np.nan

    # Phase (matched filter at LSP-derived RH)
    phase = compute_phase(sin_e, dsnr, wavelength, lsp["RH"])

    # MS (mean raw SNR in dB) and VS (variance of detrended)
    ms = float(np.mean(snr_db_w))
    vs = float(np.var(dsnr))

    return {
        "CLR": lsp["CLR"],
        "PR": lsp["PR"],
        "SP": lsp["SP"],
        "RH": lsp["RH"],
        "clr_peak_power": lsp["clr_peak_power"],
        "clr_total_power": lsp["clr_total_power"],
        "AF": af,
        "gamma": gamma,
        "gamma_r2": gamma_r2,
        "phase": phase,
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


def extract_features_for_day(station, year, doy, per_arc_day, config,
                             af_baselines=None, af_sin_grid=None):
    """Extract all SNR features for one day.

    Args:
        station: station ID (e.g., "UMNQ")
        year: processing year
        doy: day of year
        per_arc_day: DataFrame — per_arc parquet rows for this day
        config: dict from station gnssir json (e1, e2, minH, maxH, polyV, pele, desiredP)
        af_baselines: optional dict of {(sat, freq): power_curve_array} for AF correction
        af_sin_grid: sin(ε) grid the baselines are defined on

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

            # Look up AF baseline for this (sat, freq) if available
            bl_curve = None
            bl_grid = None
            if af_baselines is not None:
                bl_curve = af_baselines.get((int(sat), int(freq)))
                if bl_curve is not None:
                    bl_grid = af_sin_grid

            # Extract features
            feats = extract_arc_features(
                arc_ele, snr_db_col, snr_lin, detrended,
                wavelength, e1, e2, min_rh, max_rh, precision,
                af_baseline=bl_curve, af_baseline_sin_grid=bl_grid,
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

def _load_af_baselines(station, year):
    """Load precomputed AF baselines if available.

    Returns (baselines_dict, sin_grid) or (None, None).
    baselines_dict maps (sat, freq) → power_curve_array.
    """
    bl_path = (PROJECT_ROOT / "results_annual" / station
               / f"{station}_{year}_af_baselines.npz")
    if not bl_path.exists():
        return None, None

    bl_data = np.load(bl_path)
    baselines = {
        (int(k[0]), int(k[1])): curve
        for k, curve in zip(bl_data["keys"], bl_data["baselines"])
    }
    sin_grid = bl_data["sin_grid"]
    logger.info(f"Loaded AF baselines for {len(baselines)} PRN/freq combinations")
    return baselines, sin_grid


def _process_day_worker(args):
    """Multiprocessing worker for extract_features. Must be module-level for pickle."""
    station, year, doy, day_arcs_records, config, af_baselines, af_sin_grid = args
    day_arcs = pd.DataFrame.from_records(day_arcs_records)
    return extract_features_for_day(
        station, year, doy, day_arcs, config,
        af_baselines=af_baselines, af_sin_grid=af_sin_grid,
    )


def extract_features(station, year, num_cores=1):
    """Extract SNR features for a full station-year.

    Reads arc_table.parquet (Layer 1) as index, processes each day's SNR
    file, and merges features back into arc_table.parquet.

    Falls back to per_arc.parquet for legacy stations.

    If AF baselines exist (from compute_af_baselines.py), applies per-PRN
    power curve correction when computing area factors.
    """
    config = _load_station_config(station)
    if config is None:
        logger.error(f"No config found for {station}")
        return None

    # Resolve Layer 1 file (arc_table preferred, per_arc fallback)
    from scripts.results_handler import resolve_layer1
    layer1_path = resolve_layer1(station, year)
    if layer1_path is None:
        logger.error(f"No per-arc data found for {station} {year}")
        return None
    logger.info(f"Reading Layer 1 from {layer1_path.name}")

    per_arc = pd.read_parquet(layer1_path)
    doys = sorted(per_arc["doy"].unique())
    logger.info(f"Processing {station} {year}: {len(doys)} days, {len(per_arc)} arcs")

    # Load AF baselines if available
    af_baselines, af_sin_grid = _load_af_baselines(station, year)
    if af_baselines is None:
        logger.info("No AF baselines found — computing uncorrected area factors")

    if num_cores > 1:
        from multiprocessing import Pool

        # Build serializable args per day (list of dicts, not DataFrame)
        day_args = []
        for doy in doys:
            day_arcs = per_arc[per_arc["doy"] == doy]
            day_args.append((
                station, year, doy, day_arcs.to_dict("records"), config,
                af_baselines, af_sin_grid,
            ))

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
                extract_features_for_day(
                    station, year, doy, day_arcs, config,
                    af_baselines=af_baselines, af_sin_grid=af_sin_grid,
                )
            )

    if not all_results:
        logger.error("No features extracted")
        return None

    df = pd.DataFrame(all_results)

    # Summary
    logger.info(f"Extracted {len(df)} feature rows")
    logger.info(f"Features: CLR={df['CLR'].median():.2f}, "
                f"AF={df['AF'].dropna().median():.2f}, "
                f"gamma={df['gamma'].dropna().median():.4f}, "
                f"phase={df['phase'].median():.3f} rad")
    full_pct = df["full_arc"].mean() * 100
    logger.info(f"Full arcs: {full_pct:.1f}%")

    # Save example arcs for dashboard single-arc display
    _save_example_arcs(df, station, year, config)

    # --- Merge features into arc_table (Layer 1) ---
    results_dir = PROJECT_ROOT / "results_annual" / station
    arc_table_path = results_dir / f"{station}_{year}_arc_table.parquet"

    # Join keys shared between arc_table and feature rows
    join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]

    # Rename feature RH to avoid collision with gnssrefl RH
    df_feat = df.rename(columns={"RH": "RH_snr"})

    # Drop any existing feature columns from per_arc so fresh values overwrite them.
    # Without this, columns already present in arc_table (CLR, AF, gamma, etc.) are
    # excluded from feat_cols and never updated on re-extraction runs.
    feat_cols = [c for c in df_feat.columns if c not in join_cols]
    per_arc_base = per_arc.drop(
        columns=[c for c in feat_cols if c in per_arc.columns],
        errors="ignore",
    )
    # feat_cols for merge = feature columns + join keys
    merge_cols = join_cols + [c for c in feat_cols if c not in join_cols]
    arc_table = per_arc_base.merge(df_feat[merge_cols], on=join_cols, how="left")

    # Log match rate
    if "CLR" in arc_table.columns:
        n_matched = int(arc_table["CLR"].notna().sum())
        logger.info(
            f"arc_table merge: {n_matched}/{len(arc_table)} arcs "
            f"have SNR features ({n_matched / len(arc_table) * 100:.1f}%)"
        )

    arc_table.to_parquet(arc_table_path, index=False, engine="pyarrow")
    logger.info(
        f"arc_table (Layer 1) updated: {arc_table_path} "
        f"({len(arc_table)} arcs, {arc_table_path.stat().st_size / 1024:.0f} KB)"
    )

    # --- Write separate arc_features.csv (never modifies gnssrefl output) ---
    df_csv = df.copy()
    df_csv["year"] = year
    # Rename phase to phase_deg (degrees) for the CSV output
    if "phase" in df_csv.columns:
        df_csv["phase_deg"] = np.degrees(df_csv["phase"])
    # Carry through raw amplitude from gnssrefl if available via arc_table merge
    if "Amp" in arc_table.columns:
        amp_lookup = arc_table.set_index(join_cols)["Amp"]
        df_csv = df_csv.set_index(join_cols)
        df_csv["amp_raw"] = amp_lookup
        df_csv = df_csv.reset_index()
    # Build final column list (only include columns that exist)
    csv_cols = [c for c in [
        "year", "doy", "sat", "UTCtime", "rise", "freq",
        "CLR", "PR", "AF", "gamma", "gamma_r2",
        "phase_deg", "SP", "MS", "VS", "full_arc",
        "clr_peak_power", "clr_total_power", "amp_raw",
    ] if c in df_csv.columns]
    arc_features_path = results_dir / f"{station}_{year}_arc_features.csv"
    df_csv[csv_cols].to_csv(arc_features_path, index=False, float_format="%.6f")
    logger.info(
        f"arc_features.csv written: {arc_features_path} "
        f"({len(df_csv)} rows, {arc_features_path.stat().st_size / 1024:.0f} KB)"
    )

    return df


# ---------------------------------------------------------------------------
# Example arc saving for dashboard
# ---------------------------------------------------------------------------

def _save_example_arcs(results_df, station, year, config):
    """Save the best ice-like and water-like example arcs for dashboard display.

    Selects the arc with lowest gamma (ice) and highest gamma (water),
    re-reads the SNR files, extracts the detrended signal and Hilbert
    envelope, and saves to a JSON file.
    """
    if "gamma" not in results_df.columns or results_df["gamma"].isna().all():
        return

    valid = results_df.dropna(subset=["gamma"])
    if len(valid) < 2:
        return

    ice_row = valid.loc[valid["gamma"].idxmin()]
    water_row = valid.loc[valid["gamma"].idxmax()]

    e1 = config["e1"]
    e2 = config["e2"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))

    examples = {}
    for label, row in [("ice", ice_row), ("water", water_row)]:
        doy = int(row["doy"])
        sat = int(row["sat"])
        freq = int(row["freq"])
        utctime = float(row["UTCtime"])
        rise = int(row["rise"])

        snr_path = _snr_file_path(station, year, doy)
        if not snr_path.exists():
            gz_path = Path(str(snr_path) + ".gz")
            snr_path = gz_path if gz_path.exists() else snr_path
        if not snr_path.exists():
            continue

        try:
            snr_data = read_snr_file(snr_path)
        except Exception:
            continue

        sat_mask = snr_data[:, 0] == sat
        sat_data = snr_data[sat_mask]
        if len(sat_data) < 5:
            continue

        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime, target_rise=rise,
            e1=e1, e2=e2,
        )
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        col_idx = FREQ_TO_COL.get(freq)
        if col_idx is None or col_idx >= arc_data.shape[1]:
            continue

        snr_db = arc_data[:, col_idx]
        if np.all(snr_db == 0):
            continue

        wavelength = FREQ_WAVELENGTH.get(freq)
        if wavelength is None:
            continue

        snr_lin = np.power(10, snr_db / 20)
        detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

        # Window to [e1, e2]
        mask = (arc_ele >= e1) & (arc_ele <= e2)
        ele_w = arc_ele[mask]
        dsnr_w = detrended[mask]

        if len(dsnr_w) < 15:
            continue

        # Hilbert envelope
        analytic = hilbert(dsnr_w)
        envelope = np.abs(analytic)

        date_str = str(
            pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        )[:10]

        examples[label] = {
            "date": date_str,
            "doy": doy,
            "sat": sat,
            "freq": freq,
            "gamma": float(row["gamma"]),
            "af": float(row.get("AF", 0)),
            "clr": float(row.get("CLR", 0)),
            "elevation": ele_w.tolist(),
            "dsnr": dsnr_w.tolist(),
            "envelope": envelope.tolist(),
        }

    if examples:
        import json as json_mod
        out_path = (PROJECT_ROOT / "results_annual" / station
                    / f"{station}_{year}_example_arcs.json")
        with open(out_path, "w") as f:
            json_mod.dump(examples, f)
        logger.info(f"Saved example arcs to {out_path}")


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
    for col in ["CLR", "PR", "AF", "gamma", "phase", "MS", "VS"]:
        if col in df.columns:
            print(f"  {col:>6s}: {df[col].median():.4f}")


if __name__ == "__main__":
    main()
