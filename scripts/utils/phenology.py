# ABOUTME: Phenological transition detection for ice/water surface state changes
# ABOUTME: Moving t-test, constrained peak finding, and data gap detection

"""
Phenological transition detection for GNSS-IR ice classification.

Provides statistical methods for detecting freeze-up and break-up dates
from ice_score time series:

  moving_t_test    — detects abrupt mean shifts via consecutive-window t-test
  find_transition  — locates most extreme t-score within a DOY window
  detect_data_gaps — finds gaps > threshold in a date series

Originally developed in generate_ice_notebook.py (v7.1), extracted here
for reuse by the Dash dashboard, classifier pipelines, and notebooks.

References:
  Rodionov (2004) — sequential t-test for regime shift detection
  GNSS-IR phenology: ice_score regime shifts indicate freeze-up/break-up
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default phenology windows per station (DOY ranges)
# ---------------------------------------------------------------------------
# Extensible via station config: add "phenology" key with fus_doy_range / bus_doy_range

DEFAULT_PHENOLOGY_WINDOWS = {
    "ROSS": {"fus_doy_range": (300, 60), "bus_doy_range": (30, 150)},
    "MCHN": {"fus_doy_range": (300, 60), "bus_doy_range": (30, 150)},
    "UMNQ": {"fus_doy_range": (270, 30), "bus_doy_range": (120, 210)},
    "NKAR": {"fus_doy_range": (270, 30), "bus_doy_range": (120, 210)},
    "NIAQ": {"fus_doy_range": (270, 30), "bus_doy_range": (120, 210)},
    "LRSK": {"fus_doy_range": (270, 30), "bus_doy_range": (120, 210)},
    # General fallback for Great Lakes / Arctic
    "_default": {"fus_doy_range": (270, 60), "bus_doy_range": (30, 180)},
}


def get_phenology_windows(station):
    """Get freeze-up and break-up DOY search windows for a station.

    Checks station config for a "phenology" key first, falls back to
    DEFAULT_PHENOLOGY_WINDOWS, then the generic default.

    Returns dict with fus_doy_range and bus_doy_range tuples.
    """
    # Try station config
    try:
        from scripts.utils.external_data import load_station_config
        cfg = load_station_config(station)
        pheno = cfg.get("phenology")
        if pheno and "fus_doy_range" in pheno:
            return pheno
    except (ValueError, ImportError):
        pass

    return DEFAULT_PHENOLOGY_WINDOWS.get(
        station, DEFAULT_PHENOLOGY_WINDOWS["_default"]
    )


# ---------------------------------------------------------------------------
# Moving t-test
# ---------------------------------------------------------------------------

def moving_t_test(series, window=30):
    """Detect abrupt mean shifts via consecutive-window t-test.

    For each position i, computes a two-sample t-statistic between
    the window before [i-W, i) and the window after [i, i+W).
    Uses pooled variance (equal-variance assumption).

    Large negative t → mean decreased (freeze-up: ice_score drops).
    Large positive t → mean increased (break-up: ice_score rises).

    Args:
        series: 1D array-like of values (may contain NaN)
        window: number of observations in each half-window

    Returns:
        np.ndarray of t-scores, same length as input (NaN at edges
        and where insufficient data).
    """
    values = np.asarray(series, dtype=float)
    n = len(values)
    t_scores = np.full(n, np.nan)

    for i in range(window, n - window):
        sub1 = values[i - window:i]
        sub2 = values[i:i + window]

        s1 = sub1[~np.isnan(sub1)]
        s2 = sub2[~np.isnan(sub2)]

        if len(s1) < 5 or len(s2) < 5:
            continue

        n1, n2 = len(s1), len(s2)
        v1, v2 = s1.var(ddof=1), s2.var(ddof=1)
        denom = n1 + n2 - 2
        if denom <= 0:
            continue

        s_pooled = np.sqrt((n1 * v1 + n2 * v2) / denom)
        if s_pooled < 1e-10:
            continue

        t = (s2.mean() - s1.mean()) / (s_pooled * np.sqrt(1.0 / n1 + 1.0 / n2))
        t_scores[i] = t

    return t_scores


# ---------------------------------------------------------------------------
# DOY range helpers
# ---------------------------------------------------------------------------

def doy_in_range(doy, lo, hi):
    """Check if DOY is within [lo, hi], handling year-boundary wrap.

    Examples:
        doy_in_range(350, 300, 60)  → True  (winter wrap)
        doy_in_range(20, 300, 60)   → True
        doy_in_range(100, 300, 60)  → False
        doy_in_range(100, 30, 150)  → True  (no wrap)
    """
    if lo <= hi:
        return lo <= doy <= hi
    else:
        return doy >= lo or doy <= hi


# ---------------------------------------------------------------------------
# Constrained transition detection
# ---------------------------------------------------------------------------

def find_transition(t_scores, doys, doy_lo, doy_hi, sign="negative",
                    t_crit=3.36):
    """Find the most extreme t-score within a DOY window.

    Used to locate freeze-up (most negative t in FUS window) or
    break-up (most positive t in BUS window).

    Args:
        t_scores: array of t-statistics from moving_t_test
        doys: array of day-of-year values (same length as t_scores)
        doy_lo, doy_hi: DOY search window (wraps via doy_in_range)
        sign: "negative" for freeze-up, "positive" for break-up
        t_crit: minimum |t| to consider significant (default: p<0.001
                for df≈58, i.e., window=30)

    Returns:
        (index, t_value) or (None, 0) if no significant transition found.
    """
    best_idx = None
    best_t = 0

    for i in range(len(t_scores)):
        if np.isnan(t_scores[i]):
            continue
        if not doy_in_range(int(doys[i]), doy_lo, doy_hi):
            continue

        if sign == "negative" and t_scores[i] < -t_crit:
            if best_idx is None or t_scores[i] < best_t:
                best_t = t_scores[i]
                best_idx = i
        elif sign == "positive" and t_scores[i] > t_crit:
            if best_idx is None or t_scores[i] > best_t:
                best_t = t_scores[i]
                best_idx = i

    return best_idx, best_t


# ---------------------------------------------------------------------------
# Data gap detection
# ---------------------------------------------------------------------------

def detect_data_gaps(dates, threshold_days=7):
    """Find gaps larger than threshold in a date series.

    Args:
        dates: array-like of datetime64 or Timestamps
        threshold_days: minimum gap size to report

    Returns:
        list of dicts with keys: start, end, gap_days, start_idx
    """
    dates = pd.to_datetime(dates)
    if len(dates) < 2:
        return []

    deltas = np.diff(dates.astype("int64") // 10**9) / 86400  # days
    gaps = []
    for i, d in enumerate(deltas):
        if d > threshold_days:
            gaps.append({
                "start": dates[i],
                "end": dates[i + 1],
                "gap_days": float(d),
                "start_idx": i,
            })
    return gaps


def near_gap(idx, gap_indices, tolerance=3):
    """Check if an index is within tolerance of any data gap boundary.

    Args:
        idx: index to check
        gap_indices: list of gap start indices (from detect_data_gaps)
        tolerance: number of positions to consider "near"
    """
    for gi in gap_indices:
        if abs(idx - gi) <= tolerance or abs(idx - (gi + 1)) <= tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# High-level: detect freeze-up and break-up for a station-year
# ---------------------------------------------------------------------------

def detect_transitions(ice_score, dates, station, window=30,
                       significance=0.001):
    """Detect freeze-up and break-up dates from an ice_score time series.

    Runs the full MTT pipeline: moving t-test, constrained peak search
    within the station's phenology windows, and gap-adjacency flagging.

    Args:
        ice_score: 1D array of daily ice scores (0=water, 1=ice)
        dates: array of datetime values (same length as ice_score)
        station: station ID (for phenology window lookup)
        window: MTT half-window size in observations
        significance: p-value threshold for t-critical

    Returns:
        dict with keys:
            fus_date, fus_t, fus_gap_adjacent (freeze-up start)
            bus_date, bus_t, bus_gap_adjacent (break-up start)
            t_scores (full array for plotting)
            gaps (list of detected data gaps)
    """
    from scipy import stats as sp_stats

    ice_score = np.asarray(ice_score, dtype=float)
    dates = pd.to_datetime(dates)
    doys = dates.dayofyear

    # Critical t-value
    df_approx = 2 * window - 2
    t_crit = float(sp_stats.t.ppf(1 - significance, df_approx))

    # Phenology windows
    pheno = get_phenology_windows(station)
    fus_lo, fus_hi = pheno["fus_doy_range"]
    bus_lo, bus_hi = pheno["bus_doy_range"]

    # Run MTT
    t_scores = moving_t_test(ice_score, window=window)

    # Detect gaps
    gaps = detect_data_gaps(dates)
    gap_indices = [g["start_idx"] for g in gaps]

    # Find transitions
    fus_idx, fus_t = find_transition(t_scores, doys, fus_lo, fus_hi,
                                     "negative", t_crit)
    bus_idx, bus_t = find_transition(t_scores, doys, bus_lo, bus_hi,
                                     "positive", t_crit)

    result = {
        "t_scores": t_scores,
        "t_crit": t_crit,
        "gaps": gaps,
        "fus_date": dates[fus_idx] if fus_idx is not None else None,
        "fus_t": fus_t,
        "fus_gap_adjacent": near_gap(fus_idx, gap_indices) if fus_idx is not None else False,
        "bus_date": dates[bus_idx] if bus_idx is not None else None,
        "bus_t": bus_t,
        "bus_gap_adjacent": near_gap(bus_idx, gap_indices) if bus_idx is not None else False,
    }

    if fus_idx is not None:
        flag = " [GAP-ADJACENT]" if result["fus_gap_adjacent"] else ""
        logger.info(f"FUS: {result['fus_date'].strftime('%Y-%m-%d')} "
                    f"(t={fus_t:.2f}){flag}")
    if bus_idx is not None:
        flag = " [GAP-ADJACENT]" if result["bus_gap_adjacent"] else ""
        logger.info(f"BUS: {result['bus_date'].strftime('%Y-%m-%d')} "
                    f"(t={bus_t:.2f}){flag}")

    return result
