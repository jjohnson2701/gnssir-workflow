# ABOUTME: Canonical data loading for GNSS-IR per-arc data and reference timeseries.
# ABOUTME: Consolidates config, per_arc parquet, ERDDAP/USGS/CO-OPS loading, and merge_asof matching.

"""
# In progress — not yet integrated
Unified reference data loading module.

Replaces duplicated loading logic across create_polar_animation.py,
generate_erddap_matched.py, and comparison scripts with a single
canonical interface.

Usage:
    from scripts.utils.reference_data import (
        load_station_config, load_per_arc, load_reference_timeseries,
        match_gnssir_to_reference, compute_match_statistics, get_reference_info,
    )

    cfg = load_station_config("GLBX")
    df = load_per_arc("GLBX", 2024, Path("results_annual"), station_config=cfg)
    ref = load_reference_timeseries("GLBX", 2024, Path("results_annual"), station_config=cfg)
    matched = match_gnssir_to_reference(df, ref, max_time_diff_min=30)
    stats = compute_match_statistics(matched)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_station_config(station: str) -> dict:
    """Load a single station's configuration from stations_config.json.

    Args:
        station: 4-character station name (e.g. "GLBX", "FORA").

    Returns:
        dict with all station configuration fields.

    Raises:
        KeyError: if station is not in the config file.
    """
    with open(CONFIG_PATH) as f:
        all_cfg = json.load(f)
    if station not in all_cfg:
        raise KeyError(f"Station '{station}' not found in {CONFIG_PATH}")
    return all_cfg[station]


def get_reference_info(station_config: dict) -> dict:
    """Extract reference source metadata from station config.

    Returns dict with keys: source_type, source_name, source_id,
    latitude, longitude, distance_km.
    """
    ext = station_config.get("external_data_sources", {})
    erddap = ext.get("erddap", {})
    coops = ext.get("noaa_coops", {})
    usgs = station_config.get("usgs_comparison", {})
    coops_top = station_config.get("coops_comparison", {})

    # Priority: ERDDAP (if primary_reference) > CO-OPS > USGS
    if erddap.get("primary_reference") or erddap.get("enabled"):
        name = erddap.get("station_name", "ERDDAP")
        return {
            "source_type": "erddap",
            "source_name": name,
            "source_id": erddap.get("dataset_id", ""),
            "latitude": erddap.get("latitude", station_config.get("latitude_deg")),
            "longitude": erddap.get("longitude", station_config.get("longitude_deg")),
            "distance_km": erddap.get("distance_km", 0),
        }

    if coops.get("enabled") and coops.get("nearest_station"):
        nearest = coops["nearest_station"]
        return {
            "source_type": "coops",
            "source_name": nearest.get("name", "CO-OPS"),
            "source_id": nearest.get("id", ""),
            "latitude": nearest.get("latitude", station_config.get("latitude_deg")),
            "longitude": nearest.get("longitude", station_config.get("longitude_deg")),
            "distance_km": nearest.get("distance_km", 0),
        }

    if coops_top.get("target_station"):
        return {
            "source_type": "coops",
            "source_name": "CO-OPS",
            "source_id": coops_top["target_station"],
            "latitude": coops_top.get("station_latitude", station_config.get("latitude_deg")),
            "longitude": coops_top.get("station_longitude", station_config.get("longitude_deg")),
            "distance_km": 0,
        }

    if usgs.get("target_usgs_site"):
        return {
            "source_type": "usgs",
            "source_name": "USGS",
            "source_id": usgs["target_usgs_site"],
            "latitude": usgs.get("usgs_latitude", station_config.get("latitude_deg")),
            "longitude": usgs.get("usgs_longitude", station_config.get("longitude_deg")),
            "distance_km": 0,
        }

    return {
        "source_type": "none",
        "source_name": "None",
        "source_id": "",
        "latitude": station_config.get("latitude_deg"),
        "longitude": station_config.get("longitude_deg"),
        "distance_km": 0,
    }


# ---------------------------------------------------------------------------
# Per-arc GNSS-IR data
# ---------------------------------------------------------------------------

def load_per_arc(
    station: str,
    year: int,
    results_dir: Path,
    station_config: dict | None = None,
    apply_corrections: bool = True,
) -> pd.DataFrame:
    """Load per-arc parquet (preferred) or combined_raw CSV for a station-year.

    Adds derived columns:
        datetime  — UTC timestamp from date + UTCtime
        WSE       — water surface elevation (antenna_height - RH)
        WSE_dm    — demeaned WSE
        elev_avg  — midpoint elevation angle

    Args:
        station: station name
        year: data year
        results_dir: path to results_annual/
        station_config: pre-loaded config (avoids re-reading JSON)
        apply_corrections: if True, apply RHdot+IF corrections when available

    Returns:
        DataFrame with per-arc data.

    Raises:
        FileNotFoundError: if neither parquet nor CSV exists.
    """
    if station_config is None:
        station_config = load_station_config(station)

    parquet_path = results_dir / station / f"{station}_{year}_per_arc.parquet"
    csv_path = results_dir / station / f"{station}_{year}_combined_raw.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        log.info(f"Loaded per_arc parquet: {parquet_path.name} ({len(df)} rows)")
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info(f"Loaded combined_raw CSV: {csv_path.name} ({len(df)} rows)")
    else:
        raise FileNotFoundError(
            f"No per-arc data for {station}/{year}. "
            f"Checked: {parquet_path}, {csv_path}"
        )

    # Build datetime
    if "date" in df.columns and "UTCtime" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
            df["UTCtime"], unit="h"
        )
    elif "MJD" in df.columns:
        df["datetime"] = pd.to_datetime(df["MJD"] + 2400000.5, unit="D", origin="julian")

    # Ensure doy column exists
    if "doy" not in df.columns and "datetime" in df.columns:
        df["doy"] = df["datetime"].dt.dayofyear

    # Apply RHdot + IF bias corrections if available
    if apply_corrections and "MJD" in df.columns:
        df = _apply_rhdot_corrections(df, station, year)

    # Derived columns
    antenna_height = station_config["ellipsoidal_height_m"]
    if "wse" in df.columns:
        df["WSE"] = df["wse"]
    else:
        df["WSE"] = antenna_height - df["RH"]

    df["WSE_dm"] = df["WSE"] - df["WSE"].mean()
    df["elev_avg"] = (df["eminO"] + df["emaxO"]) / 2.0

    return df


def _apply_rhdot_corrections(df: pd.DataFrame, station: str, year: int) -> pd.DataFrame:
    """Apply RHdot + interfrequency bias corrections if the corrected file exists."""
    station_lower = station.lower()
    local_refl = PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code"
    corrected_path = (
        local_refl / "Files" / station_lower
        / f"{station_lower}_{year}_subdaily_edit.txt.withrhdotIF"
    )
    if not corrected_path.exists():
        return df

    try:
        from scripts.utils.subdaily_loader import load_corrected_retrievals

        corrected = load_corrected_retrievals(corrected_path)
        if len(corrected) == 0:
            return df

        merge_keys = ["MJD", "sat", "freq"]
        available_keys = [
            k for k in merge_keys if k in df.columns and k in corrected.columns
        ]
        if not available_keys:
            return df

        merged = df.merge(
            corrected[available_keys + ["rh_if_corrected"]],
            on=available_keys,
            how="left",
        )
        mask = merged["rh_if_corrected"].notna()
        n_matched = mask.sum()
        if n_matched > 0:
            df.loc[mask.values, "RH"] = merged.loc[mask, "rh_if_corrected"].values
            log.info(f"Applied RHdot+IF corrections to {n_matched}/{len(df)} retrievals")
    except ImportError:
        log.debug("subdaily_loader not available, skipping corrections")
    except Exception as e:
        log.warning(f"Failed to apply RHdot corrections: {e}")

    return df


# ---------------------------------------------------------------------------
# Reference timeseries (ERDDAP / USGS / CO-OPS)
# ---------------------------------------------------------------------------

def load_reference_timeseries(
    station: str,
    year: int,
    results_dir: Path,
    station_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load reference water-level timeseries for a station-year.

    Tries ERDDAP first (if configured), then USGS IV, then CO-OPS.
    Returns a DataFrame with columns [datetime, wl, wl_dm] or None if
    no reference data is available.

    All datetimes are returned as timezone-naive UTC (critical for merge_asof).
    """
    if station_config is None:
        station_config = load_station_config(station)

    ext = station_config.get("external_data_sources", {})
    erddap_config = ext.get("erddap", {})

    # Try ERDDAP
    if erddap_config.get("enabled"):
        ref = _load_erddap_reference(station, year, results_dir, erddap_config)
        if ref is not None:
            return ref

    # Try USGS
    usgs_path = results_dir / station / f"{station}_{year}_usgs_iv.csv"
    if usgs_path.exists():
        return _load_usgs_reference(usgs_path)

    # Try CO-OPS
    coops_path = results_dir / station / f"{station}_{year}_coops_6min.csv"
    if coops_path.exists():
        return _load_coops_reference(coops_path)

    log.info(f"No reference data found for {station}/{year}")
    return None


def _load_erddap_reference(
    station: str,
    year: int,
    results_dir: Path,
    erddap_config: dict,
) -> pd.DataFrame | None:
    """Load ERDDAP reference CSV."""
    station_name = erddap_config.get("station_name", "")
    if not station_name:
        return None

    # Convert "Bartlett Cove, AK" -> "bartlett_cove_ak"
    name_clean = station_name.lower().replace(" ", "_").replace(",", "").replace(".", "")
    erddap_path = results_dir / station / f"{name_clean}_{year}_raw.csv"

    # Also try without state suffix
    if not erddap_path.exists():
        parts = name_clean.split("_")
        if len(parts) > 1:
            erddap_path = results_dir / station / f"{'_'.join(parts[:-1])}_{year}_raw.csv"

    if not erddap_path.exists():
        log.debug(f"ERDDAP file not found: {erddap_path}")
        return None

    df = pd.read_csv(erddap_path, skiprows=[1])  # Skip units row
    df["datetime"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)

    # Find water level column
    wl_col = erddap_config.get("variables", {}).get("water_level")
    if wl_col is None or wl_col not in df.columns:
        for candidate in [
            "water_surface_above_navd88",
            "sea_surface_height_above_geopotential_datum",
            "sea_level",
            "water_level",
            "wl",
        ]:
            if candidate in df.columns:
                wl_col = candidate
                break

    if wl_col is None or wl_col not in df.columns:
        log.warning(f"No water level column found in {erddap_path}")
        return None

    units_scale = erddap_config.get("variables", {}).get("units_scale", 1.0)
    df["wl"] = df[wl_col] * units_scale
    df["wl_dm"] = df["wl"] - df["wl"].mean()
    log.info(f"Loaded ERDDAP reference: {erddap_path.name} ({len(df)} rows)")
    return df[["datetime", "wl", "wl_dm"]].copy()


def _load_usgs_reference(usgs_path: Path) -> pd.DataFrame | None:
    """Load USGS instantaneous-values CSV."""
    df = pd.read_csv(usgs_path)
    dt_cols = [c for c in df.columns if "datetime" in c.lower()]
    if not dt_cols:
        log.warning(f"No datetime column in {usgs_path}")
        return None

    df["datetime"] = pd.to_datetime(df[dt_cols[0]], utc=True).dt.tz_convert(None)

    if "value_m" in df.columns:
        df["wl"] = df["value_m"]
    elif "value" in df.columns:
        df["wl"] = df["value"]
    else:
        log.warning(f"No value column in {usgs_path}")
        return None

    df["wl_dm"] = df["wl"] - df["wl"].mean()
    log.info(f"Loaded USGS reference: {usgs_path.name} ({len(df)} rows)")
    return df[["datetime", "wl", "wl_dm"]].copy()


def _load_coops_reference(coops_path: Path) -> pd.DataFrame | None:
    """Load CO-OPS 6-minute water level CSV."""
    df = pd.read_csv(coops_path)

    # Filter to observations only
    if "is_observation" in df.columns:
        df = df[df["is_observation"]].copy()

    dt_cols = [c for c in df.columns if "datetime" in c.lower()]
    if not dt_cols:
        log.warning(f"No datetime column in {coops_path}")
        return None

    df["datetime"] = pd.to_datetime(df[dt_cols[0]], utc=True).dt.tz_convert(None)

    # Find water level column
    wl_col = None
    for col in df.columns:
        if "water_level" in col.lower() or col.lower() == "wl":
            wl_col = col
            break

    if wl_col is None:
        log.warning(f"No water level column in {coops_path}")
        return None

    df["wl"] = df[wl_col]
    df["wl_dm"] = df["wl"] - df["wl"].mean()
    log.info(f"Loaded CO-OPS reference: {coops_path.name} ({len(df)} rows)")
    return df[["datetime", "wl", "wl_dm"]].copy()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_gnssir_to_reference(
    gnss_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    max_time_diff_min: float = 30,
) -> pd.DataFrame:
    """Match GNSS-IR observations to nearest reference measurements using merge_asof.

    Args:
        gnss_df: per-arc DataFrame with 'datetime' and 'WSE_dm' columns.
        ref_df: reference DataFrame with 'datetime' and 'wl_dm' columns.
        max_time_diff_min: maximum allowed time difference in minutes.

    Returns:
        DataFrame with columns: gnss_datetime, gnss_wse, gnss_wse_dm,
        ref_datetime, ref_wl_dm, residual, time_diff_sec.
    """
    # Prepare sorted copies
    gnss = gnss_df[["datetime", "WSE", "WSE_dm"]].copy().sort_values("datetime")
    ref = ref_df[["datetime", "wl_dm"]].dropna(subset=["wl_dm"]).copy().sort_values("datetime")

    # Ensure timezone-naive
    if gnss["datetime"].dt.tz is not None:
        gnss["datetime"] = gnss["datetime"].dt.tz_convert(None)
    if ref["datetime"].dt.tz is not None:
        ref["datetime"] = ref["datetime"].dt.tz_convert(None)

    tolerance = pd.Timedelta(minutes=max_time_diff_min)
    merged = pd.merge_asof(
        gnss,
        ref.rename(columns={"datetime": "ref_datetime", "wl_dm": "ref_wl_dm"}),
        left_on="datetime",
        right_on="ref_datetime",
        tolerance=tolerance,
        direction="nearest",
    )

    # Drop unmatched
    merged = merged.dropna(subset=["ref_wl_dm"]).copy()

    merged["residual"] = merged["WSE_dm"] - merged["ref_wl_dm"]
    merged["time_diff_sec"] = (
        (merged["datetime"] - merged["ref_datetime"]).dt.total_seconds().abs()
    )

    result = merged.rename(columns={
        "datetime": "gnss_datetime",
        "WSE": "gnss_wse",
        "WSE_dm": "gnss_wse_dm",
    })

    return result


def compute_match_statistics(matched_df: pd.DataFrame) -> dict:
    """Compute RMSE, correlation, bias, and timing stats from matched data.

    Args:
        matched_df: output of match_gnssir_to_reference().

    Returns:
        dict with rmse, correlation, bias, n_matched, mean_time_diff_sec.
    """
    if len(matched_df) == 0:
        return {
            "rmse": np.nan,
            "correlation": np.nan,
            "bias": np.nan,
            "n_matched": 0,
            "mean_time_diff_sec": np.nan,
        }

    residuals = matched_df["residual"]
    rmse = np.sqrt((residuals ** 2).mean())
    bias = residuals.mean()

    corr = np.nan
    if len(matched_df) > 2:
        corr = matched_df["gnss_wse_dm"].corr(matched_df["ref_wl_dm"])

    return {
        "rmse": rmse,
        "correlation": corr,
        "bias": bias,
        "n_matched": len(matched_df),
        "mean_time_diff_sec": matched_df["time_diff_sec"].mean(),
    }
