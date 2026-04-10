#!/usr/bin/env python3
# ABOUTME: Orchestrator for Layers 2-3 classification pipeline after gnssrefl processing.
# ABOUTME: Runs feature_aggregator → classifier_v2 → classifier_v3 with freshness checks.

"""
Classification Pipeline Orchestrator

Runs the full Layer 2-3 pipeline for a station:
  1. feature_aggregator: arc_table.parquet → daily_features.parquet
  2. classifier_v2: daily_features.parquet → ice_classification_v2.parquet
  3. classifier_v3: v2 + interfreq + era5 → ice_classification_v3.parquet

Prerequisites: arc_table.parquet with SNR features must already exist
  (produced by process_station.py + snr_feature_extractor.py).

Usage:
    python scripts/run_classification.py --station ROSS --year 2024
    python scripts/run_classification.py --station ROSS --year 2024 --force
    python scripts/run_classification.py --station ROSS  # all available years
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)


def _mtime(path: Path) -> float:
    """Return file modification time, or 0 if file doesn't exist."""
    return path.stat().st_mtime if path.exists() else 0.0


def _find_years(station: str, results_dir: Path) -> list[int]:
    """Discover available years for a station by scanning for arc_table files."""
    station_dir = results_dir / station
    if not station_dir.exists():
        return []
    years = []
    for f in sorted(station_dir.glob(f"{station}_*_arc_table.parquet")):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                years.append(int(parts[1]))
            except ValueError:
                continue
    return sorted(years)


def run_layer2(station: str, year: int, results_dir: Path, force: bool) -> bool:
    """Run feature_aggregator if needed. Returns True on success."""
    station_dir = results_dir / station
    arc_path = station_dir / f"{station}_{year}_arc_table.parquet"
    out_path = station_dir / f"{station}_{year}_daily_features.parquet"

    if not force and out_path.exists() and _mtime(out_path) > _mtime(arc_path):
        log.info(f"  Layer 2: daily_features.parquet is up-to-date, skipping")
        return True

    log.info(f"  Layer 2: Running feature_aggregator...")
    from scripts.feature_aggregator import aggregate_daily_features
    result = aggregate_daily_features(station, year, results_dir=station_dir)
    if result is None:
        log.error(f"  Layer 2: feature_aggregator failed for {station} {year}")
        return False
    log.info(f"  Layer 2: {result}")
    return True


def run_layer3a(station: str, years: list[int], results_dir: Path, force: bool) -> bool:
    """Run classifier_v2 if needed. Returns True on success.

    v2 builds a multi-year PCA model, so it processes all years in one call.
    We trigger a re-run if any year's v2 output is missing or stale relative
    to its daily_features input.
    """
    needs_run = force
    if not needs_run:
        for year in years:
            station_dir = results_dir / station
            feat_path = station_dir / f"{station}_{year}_daily_features.parquet"
            v2_path = station_dir / f"{station}_{year}_ice_classification_v2.parquet"
            if not v2_path.exists() or _mtime(v2_path) < _mtime(feat_path):
                needs_run = True
                break

    if not needs_run:
        log.info(f"  Layer 3a: ice_classification_v2 is up-to-date for all years, skipping")
        return True

    log.info(f"  Layer 3a: Running classifier_v2 for years {years}...")
    from scripts.classifier_v2 import run_station as run_v2
    try:
        run_v2(station, years)
    except Exception as e:
        log.error(f"  Layer 3a: classifier_v2 failed: {e}")
        return False
    return True


def run_layer3b(station: str, years: list[int], results_dir: Path, force: bool) -> bool:
    """Run classifier_v3 if needed. Returns True on success."""
    needs_run = force
    if not needs_run:
        for year in years:
            station_dir = results_dir / station
            v2_path = station_dir / f"{station}_{year}_ice_classification_v2.parquet"
            v3_path = station_dir / f"{station}_{year}_ice_classification_v3.parquet"
            if not v3_path.exists() or (v2_path.exists() and _mtime(v3_path) < _mtime(v2_path)):
                needs_run = True
                break

    if not needs_run:
        log.info(f"  Layer 3b: ice_classification_v3 is up-to-date for all years, skipping")
        return True

    log.info(f"  Layer 3b: Running classifier_v3 for years {years}...")
    from scripts.classifier_v3 import run_station as run_v3
    try:
        run_v3(station, years)
    except Exception as e:
        log.error(f"  Layer 3b: classifier_v3 failed: {e}")
        return False
    return True


def print_summary(station: str, years: list[int], results_dir: Path):
    """Print a summary of what was produced."""
    import pandas as pd

    print(f"\n{'=' * 60}")
    print(f"{station} classification complete")
    print(f"{'=' * 60}")

    for year in years:
        station_dir = results_dir / station
        feat_path = station_dir / f"{station}_{year}_daily_features.parquet"
        v2_path = station_dir / f"{station}_{year}_ice_classification_v2.parquet"
        v3_path = station_dir / f"{station}_{year}_ice_classification_v3.parquet"

        print(f"\n  {station} {year}:")

        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            print(f"    Layer 2: daily_features.parquet ({len(df)} rows, {len(df.columns)} columns)")
        else:
            print(f"    Layer 2: MISSING")

        if v2_path.exists():
            df = pd.read_parquet(v2_path)
            print(f"    Layer 3a: ice_classification_v2.parquet ({len(df)} days)")
        else:
            print(f"    Layer 3a: MISSING")

        if v3_path.exists():
            df = pd.read_parquet(v3_path)
            print(f"    Layer 3b: ice_classification_v3.parquet ({len(df)} days)")
            if "v3_state" in df.columns:
                counts = df["v3_state"].value_counts()
                states_str = ", ".join(f"{s}={counts.get(s, 0)}" for s in
                                       ["open_water", "freeze_up", "ice_surface",
                                        "ice_layered", "ice_decaying", "break_up"]
                                       if counts.get(s, 0) > 0)
                print(f"    States: {states_str}")
        else:
            print(f"    Layer 3b: MISSING")


def main():
    import warnings
    warnings.warn(
        "run_classification.py is deprecated. Use run_analysis.py instead:\n"
        "  python scripts/run_analysis.py --station ROSS --year 2024\n",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Run classification pipeline (Layers 2-3) for a station "
                    "[DEPRECATED: use run_analysis.py]"
    )
    parser.add_argument("--station", required=True, help="Station ID (e.g., ROSS)")
    parser.add_argument("--year", type=int, default=None,
                        help="Specific year. If omitted, processes all available years.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist and are up-to-date")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: results_annual/)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results_annual"

    # Determine years to process
    if args.year is not None:
        years = [args.year]
    else:
        years = _find_years(args.station, results_dir)
        if not years:
            print(f"ERROR: No arc_table.parquet files found for {args.station} in {results_dir / args.station}")
            sys.exit(1)
        log.info(f"Discovered years for {args.station}: {years}")

    # Check prerequisites
    missing = []
    for year in years:
        arc_path = results_dir / args.station / f"{args.station}_{year}_arc_table.parquet"
        if not arc_path.exists():
            missing.append(year)
    if missing:
        print(f"ERROR: arc_table.parquet missing for {args.station} years: {missing}")
        print(f"Run snr_feature_extractor.py first to produce arc_table with SNR features.")
        sys.exit(1)

    log.info(f"Processing {args.station}: years={years}, force={args.force}")

    # Layer 2: feature_aggregator (per-year)
    for year in years:
        log.info(f"{args.station} {year}:")
        if not run_layer2(args.station, year, results_dir, args.force):
            print(f"ERROR: Layer 2 failed for {args.station} {year}")
            sys.exit(1)

    # Layer 3a: classifier_v2 (all years at once — builds multi-year PCA model)
    if not run_layer3a(args.station, years, results_dir, args.force):
        print(f"ERROR: Layer 3a (v2) failed for {args.station}")
        sys.exit(1)

    # Layer 3b: classifier_v3 (all years at once)
    if not run_layer3b(args.station, years, results_dir, args.force):
        print(f"ERROR: Layer 3b (v3) failed for {args.station}")
        sys.exit(1)

    # Summary
    print_summary(args.station, years, results_dir)


if __name__ == "__main__":
    main()
