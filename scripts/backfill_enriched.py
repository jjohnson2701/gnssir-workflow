# ABOUTME: Backfill script to produce enriched parquet files from existing raw CSVs
# ABOUTME: Re-aggregates per-arc data without re-running gnssir processing

"""
Backfill enriched per-arc and daily parquet files from existing combined_raw.csv.

Usage:
    # Single station
    python scripts/backfill_enriched.py --station UMNQ --year 2025

    # All stations, all years found in results_annual/
    python scripts/backfill_enriched.py --all

    # Dry run (show what would be processed)
    python scripts/backfill_enriched.py --all --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_antenna_height(station_id):
    """Look up antenna height from stations_config.json."""
    config_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    station_config = config.get(station_id, {})
    return station_config.get("ellipsoidal_height_m")


def find_raw_csvs(station=None, year=None):
    """Find all combined_raw.csv files, optionally filtered by station/year."""
    results_dir = PROJECT_ROOT / "results_annual"
    if not results_dir.exists():
        return []

    if station and year:
        pattern = f"{station}/{station}_{year}_combined_raw.csv"
        matches = list(results_dir.glob(pattern))
    elif station:
        pattern = f"{station}/{station}_*_combined_raw.csv"
        matches = list(results_dir.glob(pattern))
    else:
        pattern = "*/*_combined_raw.csv"
        matches = list(results_dir.glob(pattern))

    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(description="Backfill enriched parquet from raw CSVs")
    parser.add_argument("--station", type=str, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--year", type=int, help="Year (e.g., 2025)")
    parser.add_argument("--all", action="store_true", help="Process all stations and years")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.all and not args.station:
        parser.error("Specify --station STATION [--year YEAR] or --all")

    # Find raw CSVs to process
    raw_files = find_raw_csvs(
        station=args.station if not args.all else None,
        year=args.year,
    )

    if not raw_files:
        logging.error("No combined_raw.csv files found")
        sys.exit(1)

    logging.info(f"Found {len(raw_files)} raw CSV file(s) to backfill")

    if args.dry_run:
        for f in raw_files:
            station_id = f.parent.name
            h = get_antenna_height(station_id)
            logging.info(f"  Would process: {f.name} (antenna_height={h})")
        return

    # Import here to avoid import errors if pyarrow missing
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.results_handler import backfill_from_raw_csv

    results = []
    for raw_csv in raw_files:
        station_id = raw_csv.parent.name
        antenna_height = get_antenna_height(station_id)
        output_dir = raw_csv.parent

        logging.info(f"--- Processing {raw_csv.name} (antenna_height={antenna_height}) ---")

        per_arc_path, enriched_path = backfill_from_raw_csv(
            raw_csv_path=raw_csv,
            annual_results_dir=output_dir,
            antenna_height_m=antenna_height,
        )

        results.append({
            "station": station_id,
            "raw_csv": raw_csv.name,
            "per_arc": per_arc_path.name if per_arc_path else "FAILED",
            "enriched": enriched_path.name if enriched_path else "FAILED",
        })

    # Summary
    logging.info("=" * 60)
    logging.info("Backfill complete:")
    for r in results:
        status = "OK" if r["enriched"] != "FAILED" else "FAILED"
        logging.info(f"  {r['station']}: {status} -> {r['per_arc']}, {r['enriched']}")


if __name__ == "__main__":
    main()
