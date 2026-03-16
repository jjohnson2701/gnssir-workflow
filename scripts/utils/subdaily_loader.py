# ABOUTME: Loads gnssrefl subdaily spline output with confidence annotations.
# ABOUTME: Classifies each spline point as observed vs interpolated based on observation proximity.

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def load_subdaily_results(spline_path, exclude_gaps=True):
    """
    Load gnssrefl subdaily spline output file.

    The spline output contains evenly-sampled WSE values (default 30-min interval).
    WSE is computed by gnssrefl as Hortho - RH, where Hortho uses EGM96 geoid.

    Args:
        spline_path: Path to {station}_spline_out.txt
        exclude_gaps: If True, exclude rows with 999 gap-fill values

    Returns:
        DataFrame with columns: datetime, rh_m, wse_ortho_m, datum
    """
    rows = []
    with open(spline_path) as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            rh = float(parts[1])
            wse = float(parts[8])
            if exclude_gaps and (rh > 900 or wse > 900 or wse < -900):
                continue
            rows.append({
                "mjd": float(parts[0]),
                "rh_m": rh,
                "year": int(parts[2]),
                "month": int(parts[3]),
                "day": int(parts[4]),
                "hour": int(parts[5]),
                "minute": int(parts[6]),
                "second": int(parts[7]),
                "wse_ortho_m": wse,
            })
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    ).dt.tz_localize("UTC")
    df["datum"] = "orthometric_EGM96"
    return df


def load_corrected_retrievals(if_corrected_path):
    """
    Load full retrieval data with RHdot and IF bias corrections from .withrhdotIF file.

    Returns a DataFrame with all standard gnssir columns plus corrected RH values.
    Column names match combined_raw.csv for drop-in compatibility.

    Args:
        if_corrected_path: Path to {station}_{year}_subdaily_edit.txt.withrhdotIF

    Returns:
        DataFrame with raw columns plus rh_rhdot_corrected, rhdot_correction, rh_if_corrected
    """
    rows = []
    with open(if_corrected_path) as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 25:
                continue
            rows.append({
                "year": int(parts[0]),
                "doy": int(parts[1]),
                "RH": float(parts[2]),
                "sat": int(parts[3]),
                "UTCtime": float(parts[4]),
                "Azim": float(parts[5]),
                "Amp": float(parts[6]),
                "eminO": float(parts[7]),
                "emaxO": float(parts[8]),
                "NumbOf": int(parts[9]),
                "freq": int(parts[10]),
                "rise": int(parts[11]),
                "EdotF": float(parts[12]),
                "PkNoise": float(parts[13]),
                "DelT": float(parts[14]),
                "MJD": float(parts[15]),
                "refr_model": int(parts[16]),
                "rh_rhdot_corrected": float(parts[22]),
                "rhdot_correction": float(parts[23]),
                "rh_if_corrected": float(parts[24]),
            })
    df = pd.DataFrame(rows)
    # Add date column for compatibility with combined_raw.csv consumers
    if len(df) > 0:
        df["date"] = pd.to_datetime(
            df["year"] * 1000 + df["doy"], format="%Y%j"
        ).dt.strftime("%Y-%m-%d")
    return df


def load_observation_times(if_corrected_path):
    """
    Load original observation timestamps from IF+RHdot corrected retrieval file.

    Args:
        if_corrected_path: Path to {station}_{year}_subdaily_edit.txt.withrhdotIF

    Returns:
        DatetimeIndex of observation times (UTC)
    """
    times = []
    with open(if_corrected_path) as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 21:
                continue
            year = int(parts[0])
            month = int(parts[17])
            day = int(parts[18])
            hour = int(parts[19])
            minute = int(parts[20])
            times.append(pd.Timestamp(
                year=year, month=month, day=day,
                hour=hour, minute=minute, tz="UTC",
            ))
    return pd.DatetimeIndex(sorted(times))


def annotate_confidence(spline_df, obs_times, threshold_hours=3.0):
    """
    Annotate each spline point with distance to nearest real observation.

    Args:
        spline_df: DataFrame from load_subdaily_results
        obs_times: DatetimeIndex of actual observation timestamps
        threshold_hours: Points farther than this from any observation are "interpolated"

    Returns:
        DataFrame with added columns: nearest_obs_hours, is_interpolated
    """
    result = spline_df.copy()
    obs_ns = obs_times.values.astype("int64")
    spline_ns = result["datetime"].values.astype("int64")

    nearest_hours = np.empty(len(spline_ns))
    for i, t in enumerate(spline_ns):
        diffs = np.abs(obs_ns - t)
        nearest_hours[i] = diffs.min() / 3.6e12  # nanoseconds to hours

    result["nearest_obs_hours"] = nearest_hours
    result["is_interpolated"] = nearest_hours > threshold_hours
    return result


def detect_observation_gaps(obs_times, min_gap_hours=6.0):
    """
    Find gaps in observation coverage.

    Args:
        obs_times: DatetimeIndex of observation timestamps
        min_gap_hours: Minimum gap duration to report

    Returns:
        List of dicts with start, end, duration_hours for each gap
    """
    if len(obs_times) < 2:
        return []

    sorted_times = obs_times.sort_values()
    diffs = np.diff(sorted_times.values).astype("float64") / 3.6e12  # to hours

    gaps = []
    for i, diff_hours in enumerate(diffs):
        if diff_hours >= min_gap_hours:
            gaps.append({
                "start": sorted_times[i],
                "end": sorted_times[i + 1],
                "duration_hours": diff_hours,
            })
    return gaps
