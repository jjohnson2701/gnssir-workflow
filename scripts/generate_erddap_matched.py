#!/usr/bin/env python3
# ABOUTME: Creates subdaily_matched.csv using ERDDAP reference data from config.
# ABOUTME: Config-driven script for matching GNSS-IR to co-located ERDDAP water levels.

"""
Generate matched subdaily data using ERDDAP water level reference.

This script:
1. Reads station configuration from stations_config.json
2. Loads GNSS-IR reflector height data
3. Loads ERDDAP water level data (based on config)
4. Matches observations based on timestamps (nearest neighbor)
5. Computes water surface elevations and demeaned values
6. Saves matched dataset for validation analysis

Usage:
    python scripts/generate_erddap_matched.py --station GLBX --year 2024
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_station_config(station: str, config_path: Path) -> dict:
    """Load station configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)

    if station not in config:
        raise ValueError(f"Station {station} not found in config. Available: {list(config.keys())}")

    return config[station]


def load_gnssir_data(raw_file: Path, antenna_height: float) -> pd.DataFrame:
    """Load and process GNSS-IR reflector height data."""
    logger.info(f"Loading GNSS-IR data from {raw_file}")

    df = pd.read_csv(raw_file)
    logger.info(f"  Loaded {len(df):,} GNSS-IR observations")

    # Handle malformed column names (legacy fix)
    for col in df.columns:
        if col.startswith("PkNoise") and col != "PkNoise":
            df = df.rename(columns={col: "PkNoise"})
            logger.debug(f"  Renamed malformed column: {col} -> PkNoise")
            break

    # Create datetime column (UTC-aware)
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["UTCtime"], unit="h")
    df["datetime"] = df["datetime"].dt.tz_localize("UTC")

    # Compute water surface elevation (WSE) from reflector height
    # WSE = Antenna Height - Reflector Height
    df["wse_ellips"] = antenna_height - df["RH"]

    logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"  RH range: {df['RH'].min():.2f} to {df['RH'].max():.2f} m")

    return df


def load_erddap_data(erddap_file: Path, erddap_config: dict) -> pd.DataFrame:
    """Load ERDDAP water level data based on configuration."""
    logger.info(f"Loading ERDDAP data from {erddap_file}")

    # ERDDAP files typically have a units row after the header
    df = pd.read_csv(erddap_file, skiprows=[1])
    logger.info(f"  Loaded {len(df):,} ERDDAP observations")

    # Parse datetime
    time_col = erddap_config.get("variables", {}).get("time", "time")
    df["datetime"] = pd.to_datetime(df[time_col], utc=True)

    # Get water level column from config
    wl_col = erddap_config.get("variables", {}).get("water_level", None)

    # Try common column names if not specified in config
    if wl_col is None or wl_col not in df.columns:
        possible_cols = [
            "water_surface_above_navd88",
            "sea_surface_height_above_geopotential_datum",
            "water_level",
            "wl",
        ]
        for col in possible_cols:
            if col in df.columns:
                wl_col = col
                break

    if wl_col is None or wl_col not in df.columns:
        raise ValueError(f"Could not find water level column. Available: {df.columns.tolist()}")

    df["wl"] = df[wl_col]

    logger.info(f"  Using water level column: {wl_col}")
    logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"  Water level range: {df['wl'].min():.2f} to {df['wl'].max():.2f} m")

    return df


def match_observations(
    gnss_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    max_time_diff_min: int = 30,
    ref_name: str = "erddap",
) -> pd.DataFrame:
    """Match GNSS-IR observations to reference water level data."""
    logger.info(f"Matching observations (max {max_time_diff_min} min difference)...")

    matched_records = []
    max_time_diff = timedelta(minutes=max_time_diff_min)

    for idx, gnss_row in gnss_df.iterrows():
        gnss_time = gnss_row["datetime"]

        # Find nearest reference observation
        time_diffs = (ref_df["datetime"] - gnss_time).abs()
        nearest_idx = time_diffs.idxmin()
        min_diff = time_diffs.loc[nearest_idx]

        if min_diff <= max_time_diff:
            ref_row = ref_df.loc[nearest_idx]

            record = {
                "gnss_datetime": gnss_time,
                "gnss_wse": gnss_row["wse_ellips"],
                "gnss_rh": gnss_row["RH"],
                f"{ref_name}_datetime": ref_row["datetime"],
                f"{ref_name}_wl": ref_row["wl"],
                "time_diff_sec": min_diff.total_seconds(),
                "satellite": gnss_row["sat"],
                "azimuth": gnss_row["Azim"],
                "amplitude": gnss_row["Amp"],
                "freq": gnss_row["freq"],
            }

            # Add PkNoise if available
            if "PkNoise" in gnss_row.index:
                record["pknoise"] = gnss_row["PkNoise"]

            matched_records.append(record)

    matched_df = pd.DataFrame(matched_records)

    if len(matched_df) == 0:
        raise ValueError("No observations matched. Check date ranges overlap.")

    match_pct = len(matched_df) / len(gnss_df) * 100
    logger.info(f"  Matched {len(matched_df):,} observations ({match_pct:.1f}% of GNSS-IR data)")
    logger.info(f"  Mean time difference: {matched_df['time_diff_sec'].mean():.1f} seconds")

    return matched_df


def compute_statistics(matched_df: pd.DataFrame, ref_name: str = "erddap") -> dict:
    """Compute demeaned values, residuals, and statistics."""
    logger.info("Computing demeaned values and residuals...")

    # Column names
    ref_wl_col = f"{ref_name}_wl"
    ref_dm_col = f"{ref_name}_dm"

    # Demean both datasets
    gnss_mean = matched_df["gnss_wse"].mean()
    ref_mean = matched_df[ref_wl_col].mean()

    matched_df["gnss_dm"] = matched_df["gnss_wse"] - gnss_mean
    matched_df[ref_dm_col] = matched_df[ref_wl_col] - ref_mean

    # Compute residual (GNSS - Reference)
    matched_df["residual"] = matched_df["gnss_dm"] - matched_df[ref_dm_col]

    # Statistics
    correlation = matched_df["gnss_dm"].corr(matched_df[ref_dm_col])
    rmse = np.sqrt((matched_df["residual"] ** 2).mean())

    stats = {
        "gnss_mean_wse": gnss_mean,
        "ref_mean_wl": ref_mean,
        "residual_mean": matched_df["residual"].mean(),
        "residual_std": matched_df["residual"].std(),
        "rmse": rmse,
        "correlation": correlation,
        "n_matched": len(matched_df),
    }

    logger.info(f"  RMSE: {rmse:.3f} m")
    logger.info(f"  Correlation (r): {correlation:.3f}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate matched subdaily data using ERDDAP reference"
    )
    parser.add_argument("--station", type=str, required=True, help="Station ID (e.g., GLBX)")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument(
        "--max_time_diff",
        type=int,
        default=30,
        help="Maximum time difference for matching (minutes)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to stations_config.json (default: auto-detect)",
    )
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / "config" / "stations_config.json"

    results_dir = project_root / "results_annual" / args.station

    print("=" * 70)
    print(f"{args.station} Matched Data Generation using ERDDAP Reference")
    print("=" * 70)
    print()

    # Load station configuration
    station_config = load_station_config(args.station, config_path)

    # Get ERDDAP configuration
    erddap_config = station_config.get("external_data_sources", {}).get("erddap", {})
    if not erddap_config.get("enabled", False):
        # Also check top-level erddap key
        erddap_config = station_config.get("erddap", {})

    if not erddap_config:
        raise ValueError(f"No ERDDAP configuration found for station {args.station}")

    # Get antenna height
    antenna_height = station_config.get(
        "ellipsoidal_height_m", station_config.get("antenna_ellipsoidal_height_m")
    )
    if antenna_height is None:
        raise ValueError(f"No antenna height found in config for station {args.station}")

    # Determine ERDDAP data file path - try multiple naming patterns
    erddap_file = None
    tried_paths = []

    # Pattern 1: station_name based (e.g., "Bartlett Cove, AK" -> "bartlett_cove_ak")
    station_name_clean = erddap_config.get("station_name", "").lower()
    station_name_clean = station_name_clean.replace(" ", "_").replace(",", "").replace("__", "_")
    if station_name_clean:
        candidate = results_dir / f"{station_name_clean}_{args.year}_raw.csv"
        tried_paths.append(candidate)
        if candidate.exists():
            erddap_file = candidate

    # Pattern 2: Just the location name without state (e.g., "bartlett_cove")
    if erddap_file is None:
        parts = station_name_clean.split("_")
        if len(parts) > 1:
            location_name = "_".join(parts[:-1])  # Remove last part (often state code)
            candidate = results_dir / f"{location_name}_{args.year}_raw.csv"
            tried_paths.append(candidate)
            if candidate.exists():
                erddap_file = candidate

    # Pattern 3: dataset_id based
    if erddap_file is None:
        dataset_id = erddap_config.get("dataset_id", "")
        if dataset_id:
            parts = dataset_id.split("_")
            for i in range(len(parts), 0, -1):
                candidate = results_dir / f"{'_'.join(parts[-i:])}_{args.year}_raw.csv"
                tried_paths.append(candidate)
                if candidate.exists():
                    erddap_file = candidate
                    break

    # Pattern 4: Search for any *_raw.csv that's not the GNSS combined file
    if erddap_file is None:
        available = list(results_dir.glob("*_raw.csv"))
        gnss_raw = results_dir / f"{args.station}_{args.year}_combined_raw.csv"
        for f in available:
            if f != gnss_raw and "combined" not in f.name:
                erddap_file = f
                logger.info(f"Auto-detected ERDDAP file: {f.name}")
                break

    if erddap_file is None:
        available = list(results_dir.glob("*_raw.csv"))
        raise FileNotFoundError(
            "ERDDAP data file not found. Tried:\n"
            + "\n".join(f"  - {p}" for p in tried_paths[:5])
            + f"\nAvailable files: {[f.name for f in available]}"
        )

    # Reference name for column naming (from station name, simplified)
    ref_name = erddap_config.get("station_name", "erddap").split(",")[0].lower().replace(" ", "_")

    # Load data
    raw_file = results_dir / f"{args.station}_{args.year}_combined_raw.csv"
    gnss_df = load_gnssir_data(raw_file, antenna_height)
    ref_df = load_erddap_data(erddap_file, erddap_config)

    # Match observations
    matched_df = match_observations(gnss_df, ref_df, args.max_time_diff, ref_name)

    # Compute statistics
    stats = compute_statistics(matched_df, ref_name)

    # Save output
    output_file = results_dir / f"{args.station}_{args.year}_subdaily_matched.csv"
    matched_df.to_csv(output_file, index=False)
    logger.info(f"Saved to: {output_file}")

    # Print summary
    print()
    print("=" * 70)
    print(f"SUCCESS: {args.station} matched data generation complete")
    print("=" * 70)
    print()
    print("Summary Statistics:")
    print(f"  Total GNSS-IR obs: {len(gnss_df):,}")
    print(f"  Matched obs: {stats['n_matched']:,} ({stats['n_matched']/len(gnss_df)*100:.1f}%)")
    print(f"  Correlation: r = {stats['correlation']:.3f}")
    print(f"  RMSE: {stats['rmse']:.3f} m")
    print(f"  Reference: {erddap_config.get('station_name', 'ERDDAP')}")
    print(f"  Distance: {erddap_config.get('distance_km', 'unknown')} km")


if __name__ == "__main__":
    main()
