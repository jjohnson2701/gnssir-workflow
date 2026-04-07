# ABOUTME: Ingests gnssir results from the Canadian/Great Lakes archive into the
# ABOUTME: standalone workflow's results_annual/ directory with enriched parquets.

"""
Ingest gnssir results from an external archive into the workflow's results_annual/ layout.

The Canadian/Great Lakes archive at /net/tiampostorage/LabShare/forJoel/gnss/research/
stores gnssir outputs as YEAR/results/station/DDD.txt (bare day-of-year filenames).

This script reads those files and produces the standard output set:
  - STATION_YEAR_combined_rh.csv     (daily aggregated stats)
  - STATION_YEAR_combined_raw.csv    (all per-arc rows)
  - STATION_YEAR_per_arc.parquet     (per-arc with derived columns)
  - STATION_YEAR_combined_enriched.parquet  (daily stats by az_bin x freq_group)
  - STATION_YEAR_daily_enriched.parquet     (backward-compatible alias)
  - STATION_YEAR_combined_interfreq.parquet (interfrequency divergence)

Usage:
    # Single station, single year
    python scripts/ingest_gnssrefl_archive.py --station MCHN --year 2022

    # Single station, all available years
    python scripts/ingest_gnssrefl_archive.py --station MCHN

    # All 10 Canadian stations, all years
    python scripts/ingest_gnssrefl_archive.py --all

    # Dry run (show what would be processed)
    python scripts/ingest_gnssrefl_archive.py --all --dry_run

    # Force overwrite existing results
    python scripts/ingest_gnssrefl_archive.py --station ROSS --year 2020 --overwrite
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory so we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.results_handler import (
    GNSSIR_V3_COLUMNS,
    add_derived_columns,
    compute_enriched_daily,
    compute_interfreq_daily,
)

ARCHIVE_ROOT = Path("/net/tiampostorage/LabShare/forJoel/gnss/research")
RESULTS_ANNUAL_DIR = Path(__file__).resolve().parent.parent / "results_annual"
STATIONS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "stations_config.json"

# The 10 Canadian/Great Lakes stations in the archive
CANADIAN_STATIONS = [
    "ross", "mchn", "pary", "clwd", "toby",
    "god2", "cbrg", "pwel", "kngv", "psta",
]


def load_antenna_height(station_upper):
    """Load antenna height from the station's gnssir param file."""
    config_path = Path(__file__).resolve().parent.parent / "config" / f"{station_upper.lower()}.json"
    if config_path.exists():
        with open(config_path) as f:
            params = json.load(f)
        return params.get("ht")
    return None


def discover_archive_years(station_lower):
    """Find all years with results for a station in the archive."""
    years = []
    for year_dir in sorted(ARCHIVE_ROOT.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        results_dir = year_dir / "results" / station_lower
        if results_dir.is_dir():
            # Check it actually has .txt files (not just an empty dir)
            txt_files = list(results_dir.glob("*.txt"))
            if txt_files:
                years.append(int(year_dir.name))
    return years


def read_archive_day(txt_path):
    """Read a single archive DDD.txt file into a DataFrame.

    Returns DataFrame with GNSSIR_V3_COLUMNS, or None if empty/invalid.
    """
    with open(txt_path) as f:
        lines = f.readlines()

    if not lines:
        return None

    # Count header lines (starting with %)
    header_lines = 0
    for line in lines:
        if line.strip().startswith("%"):
            header_lines += 1
        else:
            break

    if header_lines >= len(lines):
        return None

    try:
        df = pd.read_csv(
            txt_path,
            skiprows=header_lines,
            sep=r"\s+",
            header=None,
        )
    except pd.errors.EmptyDataError:
        return None

    if df.empty:
        return None

    # Assign canonical column names
    num_cols = len(df.columns)
    if num_cols == len(GNSSIR_V3_COLUMNS):
        df.columns = GNSSIR_V3_COLUMNS
    elif num_cols < len(GNSSIR_V3_COLUMNS):
        df.columns = GNSSIR_V3_COLUMNS[:num_cols]
    else:
        df.columns = GNSSIR_V3_COLUMNS + [f"Col{i}" for i in range(num_cols - len(GNSSIR_V3_COLUMNS))]

    return df


def ingest_station_year(station_lower, year, antenna_height_m=None, overwrite=False):
    """Ingest one station-year from the archive into results_annual/.

    Returns:
        dict with keys: station, year, n_days, n_arcs, status
    """
    station_upper = station_lower.upper()
    output_dir = RESULTS_ANNUAL_DIR / station_upper
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    combined_rh_path = output_dir / f"{station_upper}_{year}_combined_rh.csv"
    if combined_rh_path.exists() and not overwrite:
        return {
            "station": station_upper, "year": year,
            "n_days": 0, "n_arcs": 0, "status": "skipped (exists)",
        }

    # Read all daily files
    archive_dir = ARCHIVE_ROOT / str(year) / "results" / station_lower
    if not archive_dir.is_dir():
        return {
            "station": station_upper, "year": year,
            "n_days": 0, "n_arcs": 0, "status": "no archive dir",
        }

    txt_files = sorted(archive_dir.glob("*.txt"))
    if not txt_files:
        return {
            "station": station_upper, "year": year,
            "n_days": 0, "n_arcs": 0, "status": "no files",
        }

    all_data = []
    for txt_file in txt_files:
        df = read_archive_day(txt_file)
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return {
            "station": station_upper, "year": year,
            "n_days": 0, "n_arcs": 0, "status": "all files empty",
        }

    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by DOY and time
    sort_cols = [c for c in ["doy", "UTCtime"] if c in combined_df.columns]
    if sort_cols:
        combined_df.sort_values(sort_cols, inplace=True)

    # Add date column
    if "year" in combined_df.columns and "doy" in combined_df.columns:
        combined_df["date"] = combined_df.apply(
            lambda row: (
                datetime(int(row["year"]), 1, 1) + timedelta(days=int(row["doy"]) - 1)
            ).strftime("%Y-%m-%d"),
            axis=1,
        )

    n_arcs = len(combined_df)
    n_days = combined_df["date"].nunique() if "date" in combined_df.columns else 0

    # Add derived columns
    add_derived_columns(combined_df, antenna_height_m=antenna_height_m)

    # --- Save per-arc parquet ---
    per_arc_path = output_dir / f"{station_upper}_{year}_per_arc.parquet"
    combined_df.to_parquet(per_arc_path, index=False, engine="pyarrow")

    # --- Save combined_raw.csv ---
    raw_csv_path = output_dir / f"{station_upper}_{year}_combined_raw.csv"
    combined_df.to_csv(raw_csv_path, index=False)

    # --- Compute and save enriched daily ---
    if "date" in combined_df.columns:
        enriched = compute_enriched_daily(combined_df)
        if enriched is not None:
            enriched_path = output_dir / f"{station_upper}_{year}_combined_enriched.parquet"
            enriched.to_parquet(enriched_path, index=False, engine="pyarrow")
            compat_path = output_dir / f"{station_upper}_{year}_daily_enriched.parquet"
            enriched.to_parquet(compat_path, index=False, engine="pyarrow")

        # --- Compute interfrequency divergence ---
        interfreq = compute_interfreq_daily(combined_df)
        if interfreq is not None:
            interfreq_path = output_dir / f"{station_upper}_{year}_combined_interfreq.parquet"
            interfreq.to_parquet(interfreq_path, index=False, engine="pyarrow")

    # --- Backward-compatible daily CSV ---
    if "date" in combined_df.columns and "RH" in combined_df.columns:
        daily_agg = combined_df.groupby("date").agg(
            {"RH": ["count", "mean", "median", "std", "min", "max"]}
        )
        daily_agg.columns = ["_".join(col).strip() for col in daily_agg.columns.values]
        daily_agg.rename(
            columns={
                "RH_count": "rh_count",
                "RH_mean": "rh_mean_m",
                "RH_median": "rh_median_m",
                "RH_std": "rh_std_m",
                "RH_min": "rh_min_m",
                "RH_max": "rh_max_m",
            },
            inplace=True,
        )
        daily_agg.reset_index(inplace=True)
        daily_agg["datetime"] = pd.to_datetime(daily_agg["date"])
        daily_agg["year"] = daily_agg["datetime"].dt.year
        daily_agg["doy"] = daily_agg["datetime"].dt.strftime("%j").astype(int)
        daily_agg.to_csv(combined_rh_path, index=False)

    return {
        "station": station_upper, "year": year,
        "n_days": n_days, "n_arcs": n_arcs, "status": "ingested",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest gnssir results from the Canadian/Great Lakes archive"
    )
    parser.add_argument("--station", type=str, help="Station ID (e.g., MCHN)")
    parser.add_argument("--year", type=int, help="Specific year to ingest")
    parser.add_argument("--all", action="store_true", help="Process all 10 Canadian stations")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not ARCHIVE_ROOT.exists():
        logging.error(f"Archive not found: {ARCHIVE_ROOT}")
        sys.exit(1)

    # Determine which stations to process
    if args.all:
        stations = CANADIAN_STATIONS
    elif args.station:
        stations = [args.station.lower()]
    else:
        parser.error("Specify --station XXXX or --all")

    # Build work list: (station, year) pairs
    work = []
    for station in stations:
        available_years = discover_archive_years(station)
        if args.year:
            if args.year in available_years:
                work.append((station, args.year))
            else:
                logging.warning(f"{station.upper()}: year {args.year} not in archive")
        else:
            for y in available_years:
                work.append((station, y))

    if not work:
        logging.error("No station-year combinations found to process")
        sys.exit(1)

    # Dry run: just show the plan
    if args.dry_run:
        print(f"\nDry run: {len(work)} station-year combinations to process\n")
        current_station = None
        for station, year in work:
            su = station.upper()
            if su != current_station:
                current_station = su
                print(f"\n{su}:")
            existing = (RESULTS_ANNUAL_DIR / su / f"{su}_{year}_combined_rh.csv").exists()
            flag = " [EXISTS]" if existing else ""
            print(f"  {year}{flag}")
        print(f"\nUse without --dry_run to execute. Add --overwrite to replace existing.")
        return

    # Process
    results = []
    for station, year in work:
        station_upper = station.upper()
        antenna_height = load_antenna_height(station_upper)
        logging.info(f"Processing {station_upper} {year} (antenna_ht={antenna_height})")

        result = ingest_station_year(
            station, year,
            antenna_height_m=antenna_height,
            overwrite=args.overwrite,
        )
        results.append(result)

        status = result["status"]
        if status == "ingested":
            logging.info(
                f"  -> {result['n_days']} days, {result['n_arcs']} arcs"
            )
        else:
            logging.info(f"  -> {status}")

    # Summary
    ingested = [r for r in results if r["status"] == "ingested"]
    skipped = [r for r in results if r["status"] == "skipped (exists)"]
    failed = [r for r in results if r["status"] not in ("ingested", "skipped (exists)")]

    print(f"\n{'='*60}")
    print(f"Ingestion complete: {len(ingested)} ingested, {len(skipped)} skipped, {len(failed)} failed")
    if ingested:
        total_arcs = sum(r["n_arcs"] for r in ingested)
        total_days = sum(r["n_days"] for r in ingested)
        print(f"Total: {total_arcs:,} arcs across {total_days:,} station-days")
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['station']} {r['year']}: {r['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
