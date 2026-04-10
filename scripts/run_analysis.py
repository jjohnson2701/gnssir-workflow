#!/usr/bin/env python3
# ABOUTME: Single entry point for the GNSS-IR surface analysis pipeline
# ABOUTME: Chains L1 extraction → L2 aggregation → baseline detection → L3 anomaly detection → feature profiling

"""
GNSS-IR Surface Analysis Pipeline

Single command that runs the full analysis chain:
  L1: snr_feature_extractor → arc_features.csv
  L2: feature_aggregator → daily_features.csv
  Baseline: baseline_detection → baseline_definition.csv
  L3: anomaly_detector → anomaly_scores.csv
  Profiling: feature_profiler → feature_scorecard.csv

Usage:
    python scripts/run_analysis.py --station ROSS --year 2024
    python scripts/run_analysis.py --station ROSS --year 2024 --baseline 150-243
    python scripts/run_analysis.py --station ROSS --year 2024 --baseline-months 6,7,8,9
    python scripts/run_analysis.py --station ROSS --year 2024 --force
    python scripts/run_analysis.py --station ROSS --year 2024 --num_cores 8
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


def _check_snr_availability(station, year):
    """Check how many SNR files are available for a station-year.

    Returns (n_days, snr_dir_path) or (0, None).
    """
    station_lower = station.lower()
    snr_dir = (PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code"
               / str(year) / "snr" / station_lower)
    if not snr_dir.exists():
        return 0, None
    snr_files = list(snr_dir.glob(f"{station_lower}*0.*.snr66*"))
    return len(snr_files), snr_dir


def _check_per_arc(station, year, results_dir):
    """Check gnssrefl per-arc results availability.

    Returns (n_arcs, n_days, path) or (0, 0, None).
    """
    from scripts.results_handler import resolve_layer1
    path = resolve_layer1(station, year, results_dir=results_dir)
    if path is None:
        return 0, 0, None
    import pandas as pd
    per_arc = pd.read_parquet(path)
    n_arcs = len(per_arc)
    n_days = per_arc["doy"].nunique()
    return n_arcs, n_days, path


def print_startup_diagnostic(station, year, results_dir, baseline_info=None):
    """Print data availability diagnostic before processing."""
    n_arcs, n_days, per_arc_path = _check_per_arc(station, year, results_dir)
    n_snr, snr_dir = _check_snr_availability(station, year)

    print(f"\nStation: {station}  Year: {year}")
    print(f"Data found:")

    if n_arcs > 0:
        print(f"  [OK] gnssrefl per-arc results: {n_arcs:,} arcs across {n_days} days")
    else:
        print(f"  [!!] gnssrefl per-arc results: not found")

    if n_snr > 0:
        print(f"  [OK] SNR files: {n_snr} days at {snr_dir}")
        print(f"  -> Full feature extraction available (11 SNR-derived features)")
    else:
        print(f"  [!!] SNR files: not found")
        print(f"  -> Surface feature extraction requires SNR files")
        print(f"  -> Proceeding with basic features only (RH, Amp, PkNoise)")

    if baseline_info:
        prov = " [provisional]" if baseline_info.get("provisional") else ""
        print(f"\nBaseline: {baseline_info.get('method', 'auto')}-detected "
              f"DOY {baseline_info['start_doy']}-{baseline_info['end_doy']} "
              f"({baseline_info['n_days']} days, mean {baseline_info['mean_n_arcs']:.0f} arcs/day)"
              f"{prov}")

    print()
    return n_arcs > 0, n_snr > 0


def run_l1(station, year, results_dir, force, num_cores=1):
    """Run L1 feature extraction if needed."""
    arc_features_path = results_dir / f"{station}_{year}_arc_features.csv"
    arc_table_path = results_dir / f"{station}_{year}_arc_table.parquet"

    # Check freshness: arc_features.csv newer than per_arc input
    from scripts.results_handler import resolve_layer1
    per_arc_path = resolve_layer1(station, year, results_dir=results_dir)
    if per_arc_path is None:
        log.error(f"No per-arc data found for {station} {year}")
        return False

    if (not force and arc_features_path.exists()
            and _mtime(arc_features_path) > _mtime(per_arc_path)):
        log.info(f"  L1: arc_features.csv is up-to-date, skipping")
        return True

    log.info(f"  L1: Running SNR feature extraction...")
    from scripts.snr_feature_extractor import extract_features
    result = extract_features(station, year, num_cores=num_cores)
    if result is None:
        log.error(f"  L1: Feature extraction failed")
        return False

    log.info(f"  L1: Complete ({len(result)} arcs)")
    return True


def run_l2(station, year, results_dir, force):
    """Run L2 aggregation if needed."""
    # Check for input files
    arc_features_path = results_dir / f"{station}_{year}_arc_features.csv"
    arc_table_path = results_dir / f"{station}_{year}_arc_table.parquet"

    input_path = arc_features_path if arc_features_path.exists() else arc_table_path
    if not input_path.exists():
        log.error(f"No arc data found for L2 aggregation")
        return False

    out_csv = results_dir / f"{station}_{year}_daily_features.csv"
    out_pq = results_dir / f"{station}_{year}_daily_features.parquet"

    if (not force and (out_csv.exists() or out_pq.exists())
            and max(_mtime(out_csv), _mtime(out_pq)) > _mtime(input_path)):
        log.info(f"  L2: daily_features is up-to-date, skipping")
        return True

    log.info(f"  L2: Running feature aggregation...")
    from scripts.feature_aggregator import aggregate_daily_features
    result = aggregate_daily_features(station, year, results_dir=results_dir)
    if result is None:
        log.error(f"  L2: Feature aggregation failed")
        return False

    log.info(f"  L2: Complete → {result}")
    return True


def run_baseline(station, year, results_dir, force,
                 baseline_spec=None, baseline_months=None):
    """Run baseline detection if needed.

    Args:
        baseline_spec: DOY range string like "150-243"
        baseline_months: list of month ints like [6, 7, 8, 9]

    Returns:
        dict with baseline definition, or None on failure.
    """
    import pandas as pd
    from scripts.baseline_detection import (
        detect_baseline, save_baseline_definition, load_baseline_definition,
    )

    out_path = results_dir / f"{station}_{year}_baseline_definition.csv"

    # If user specified baseline, always recompute
    if baseline_spec or baseline_months:
        force = True

    if not force and out_path.exists():
        existing = load_baseline_definition(station, year, results_dir)
        if existing:
            log.info(f"  Baseline: using existing definition "
                     f"DOY {existing['start_doy']}-{existing['end_doy']}")
            return existing

    # Load daily features
    for ext in ["csv", "parquet"]:
        feat_path = results_dir / f"{station}_{year}_daily_features.{ext}"
        if feat_path.exists():
            break
    else:
        log.error("daily_features not found for baseline detection")
        return None

    daily = pd.read_csv(feat_path) if ext == "csv" else pd.read_parquet(feat_path)
    if "azimuth_bin" in daily.columns:
        daily = daily[daily["azimuth_bin"] == -1].copy()

    # Detect baseline
    if baseline_spec:
        parts = baseline_spec.split("-")
        doy_range = (int(parts[0]), int(parts[1]))
        result = detect_baseline(daily, method="user_doy", doy_range=doy_range)
    elif baseline_months:
        result = detect_baseline(daily, method="user_months", months=baseline_months)
    else:
        result = detect_baseline(daily)

    if result:
        save_baseline_definition(result, station, year, results_dir)
        return result
    else:
        log.error("Baseline detection failed")
        return None


def run_l3(station, year, results_dir, force, baseline_def=None):
    """Run L3 anomaly detection if needed.

    Uses anomaly_detector if available, falls back to legacy classifier pipeline.
    """
    out_path = results_dir / f"{station}_{year}_anomaly_scores.csv"

    # Check freshness against daily_features
    feat_csv = results_dir / f"{station}_{year}_daily_features.csv"
    feat_pq = results_dir / f"{station}_{year}_daily_features.parquet"
    feat_path = feat_csv if feat_csv.exists() else feat_pq

    if (not force and out_path.exists()
            and _mtime(out_path) > _mtime(feat_path)):
        log.info(f"  L3: anomaly_scores.csv is up-to-date, skipping")
        return True

    # Try new anomaly_detector (Phase 4)
    try:
        from scripts.anomaly_detector import run_anomaly_detection
        log.info(f"  L3: Running anomaly detection...")
        result = run_anomaly_detection(station, year, results_dir=results_dir,
                                       baseline_def=baseline_def)
        if result is not None:
            log.info(f"  L3: Complete → {result}")
            return True
    except ImportError:
        log.info(f"  L3: anomaly_detector not yet available, trying legacy pipeline")

    # Fallback: run legacy classifier pipeline
    try:
        from scripts.classifier_v2 import run_station as run_v2
        from scripts.classifier_v3 import run_station as run_v3
        log.info(f"  L3: Running legacy classifier pipeline...")
        run_v2(station, [year])
        run_v3(station, [year])
        log.info(f"  L3: Legacy pipeline complete")
        return True
    except Exception as e:
        log.error(f"  L3: Classification failed: {e}")
        return False


def run_profiling(station, year, results_dir, force):
    """Run feature profiling if needed (Phase 5)."""
    out_path = results_dir / f"{station}_{year}_feature_scorecard.csv"

    if not force and out_path.exists():
        log.info(f"  Profiling: feature_scorecard.csv exists, skipping")
        return True

    try:
        from scripts.feature_profiler import run_profiling as profile
        log.info(f"  Profiling: Running feature validation...")
        result = profile(station, year, results_dir=results_dir)
        if result is not None:
            log.info(f"  Profiling: Complete → {result}")
            return True
    except ImportError:
        log.info(f"  Profiling: feature_profiler not yet available, skipping")
        return True

    return False


def print_summary(station, year, results_dir):
    """Print summary of pipeline outputs."""
    import pandas as pd

    print(f"\n{'=' * 60}")
    print(f"{station} {year}: Analysis complete")
    print(f"{'=' * 60}")

    outputs = [
        ("L1", "arc_features.csv", None),
        ("L2", "daily_features.csv", None),
        ("Baseline", "baseline_definition.csv", None),
        ("L3", "anomaly_scores.csv", None),
        ("Profiling", "feature_scorecard.csv", None),
    ]

    for label, filename, _ in outputs:
        path = results_dir / f"{station}_{year}_{filename}"
        if path.exists():
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            size_kb = path.stat().st_size / 1024
            print(f"  [{label:>10s}] {filename}: {len(df)} rows, {size_kb:.0f} KB")
        else:
            print(f"  [{label:>10s}] {filename}: not produced")

    # Show baseline info
    baseline_path = results_dir / f"{station}_{year}_baseline_definition.csv"
    if baseline_path.exists():
        bl = pd.read_csv(baseline_path)
        if len(bl) > 0:
            r = bl.iloc[0]
            prov = " [provisional]" if r.get("provisional") else ""
            print(f"\n  Baseline: DOY {int(r['start_doy'])}-{int(r['end_doy'])} "
                  f"({int(r['n_days'])} days, {r['mean_n_arcs']:.0f} arcs/day) "
                  f"[{r['method']}]{prov}")

    # Show anomaly state summary if available
    scores_path = results_dir / f"{station}_{year}_anomaly_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        if "state" in scores.columns:
            counts = scores["state"].value_counts()
            print(f"\n  States: {dict(counts)}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="GNSS-IR Surface Analysis Pipeline"
    )
    parser.add_argument("--station", required=True, help="Station ID (e.g., ROSS)")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline DOY range (e.g., 150-243)")
    parser.add_argument("--baseline-months", type=str, default=None,
                        help="Baseline months (e.g., 6,7,8,9)")
    parser.add_argument("--num_cores", type=int, default=1,
                        help="Number of parallel workers for L1 extraction")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs are up-to-date")
    parser.add_argument("--skip-l1", action="store_true",
                        help="Skip L1 feature extraction")
    parser.add_argument("--skip-l3", action="store_true",
                        help="Skip L3 anomaly detection")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = PROJECT_ROOT / "results_annual" / args.station
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parse baseline months if provided
    baseline_months = None
    if args.baseline_months:
        baseline_months = [int(m) for m in args.baseline_months.split(",")]

    # Startup diagnostic
    has_per_arc, has_snr = print_startup_diagnostic(
        args.station, args.year, results_dir
    )

    if not has_per_arc:
        print("ERROR: gnssrefl per-arc results required. Run process_station.py first.")
        sys.exit(1)

    # L1: Feature extraction (requires SNR files)
    if not args.skip_l1 and has_snr:
        log.info(f"Step 1/5: L1 feature extraction")
        if not run_l1(args.station, args.year, results_dir, args.force,
                      num_cores=args.num_cores):
            print("ERROR: L1 feature extraction failed")
            sys.exit(1)
    elif not has_snr:
        log.info(f"Step 1/5: L1 skipped (no SNR files)")
    else:
        log.info(f"Step 1/5: L1 skipped (--skip-l1)")

    # L2: Feature aggregation
    log.info(f"Step 2/5: L2 feature aggregation")
    if not run_l2(args.station, args.year, results_dir, args.force):
        print("ERROR: L2 feature aggregation failed")
        sys.exit(1)

    # Baseline detection
    log.info(f"Step 3/5: Baseline detection")
    baseline_def = run_baseline(
        args.station, args.year, results_dir, args.force,
        baseline_spec=args.baseline,
        baseline_months=baseline_months,
    )
    if baseline_def is None:
        print("WARNING: Baseline detection failed, proceeding without baseline")

    # Update startup diagnostic with baseline info
    if baseline_def:
        prov = " [provisional]" if baseline_def.get("provisional") else ""
        print(f"Baseline: DOY {baseline_def['start_doy']}-{baseline_def['end_doy']} "
              f"({baseline_def['n_days']} days, mean {baseline_def['mean_n_arcs']:.0f} arcs/day) "
              f"[{baseline_def['method']}]{prov}")

    # L3: Anomaly detection
    if not args.skip_l3:
        log.info(f"Step 4/5: L3 anomaly detection")
        if not run_l3(args.station, args.year, results_dir, args.force,
                      baseline_def=baseline_def):
            print("WARNING: L3 anomaly detection failed")
    else:
        log.info(f"Step 4/5: L3 skipped (--skip-l3)")

    # Feature profiling
    log.info(f"Step 5/5: Feature profiling")
    run_profiling(args.station, args.year, results_dir, args.force)

    # Summary
    print_summary(args.station, args.year, results_dir)


if __name__ == "__main__":
    main()
