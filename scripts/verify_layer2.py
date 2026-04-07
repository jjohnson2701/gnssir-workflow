# ABOUTME: Verification script comparing daily_features.parquet against ice_classifier internals
# ABOUTME: Confirms that Layer 2 aggregation matches the classifier's transient computations

"""
Verify that daily_features.parquet (Layer 2) matches the values
that ice_classifier.py computes transiently in _extract_indicators().

Compares: CLR, PR, AF, gamma medians and amp_mean, amp_cv, rh_std
per (date, sector). Does NOT compare delta_rh because the methods
intentionally differ:
  - Classifier: band-median std (pools all arcs per band per sector)
  - Feature aggregator: matched-arc std (groups by sat, rise first)

Usage:
    python scripts/verify_layer2.py --station ROSS --year 2022
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def verify(station, year):
    """Compare daily_features against classifier's _extract_indicators output."""
    results_dir = PROJECT_ROOT / "results_annual" / station

    # Load daily_features (Layer 2 output)
    df_path = results_dir / f"{station}_{year}_daily_features.parquet"
    if not df_path.exists():
        print(f"ERROR: daily_features not found: {df_path}")
        return False
    daily_features = pd.read_parquet(df_path)
    # Filter to real sectors (not pooled)
    df_sectors = daily_features[daily_features["azimuth_bin"] >= 0].copy()

    # Reproduce classifier's computation for comparison
    arc_path = results_dir / f"{station}_{year}_arc_table.parquet"
    if not arc_path.exists():
        print(f"ERROR: arc_table not found: {arc_path}")
        return False
    arc_table = pd.read_parquet(arc_path)

    has_snr = "CLR" in arc_table.columns

    # Load ice_free_months from config
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    with open(cfg_path) as f:
        all_cfg = json.load(f)
    station_cfg = all_cfg.get(station, {})
    ice_free_months = station_cfg.get("ice_free_months")

    # Two comparison passes:
    # 1. Raw medians: compare un-normalized arc_table medians vs daily_features *_med columns
    # 2. Z-scored medians: normalize like the classifier, compare vs daily_features *_z columns

    # --- Pass 1: Raw feature medians + RH/Amp stats ---
    raw_compare = ["amp_mean", "amp_cv", "rh_std"]
    if has_snr:
        raw_compare += ["clr_med", "af_med", "pr_med", "gamma_med"]

    # --- Pass 2: Z-scored medians (classifier-style normalization) ---
    z_compare = []
    if has_snr:
        from scripts.ice_classifier import _normalize_features_per_satellite
        clf_arc = arc_table.copy()
        feat_cols = [c for c in ["CLR", "AF", "PR", "gamma"]
                     if c in clf_arc.columns]
        _normalize_features_per_satellite(clf_arc, feat_cols, ice_free_months)
        z_compare = ["clr_z", "af_z", "pr_z", "gamma_z"]
    else:
        clf_arc = arc_table

    all_compare = raw_compare + z_compare

    print(f"\n{'='*70}")
    print(f"Layer 2 Verification: {station} {year}")
    print(f"{'='*70}")
    print(f"daily_features: {len(df_sectors)} (date, sector) rows")
    print(f"arc_table: {len(arc_table)} arcs")
    print(f"SNR features: {'YES' if has_snr else 'NO'}")
    print(f"ice_free_months: {ice_free_months}")
    print(f"Raw comparisons: {raw_compare}")
    print(f"Z-score comparisons: {z_compare}")
    print(f"NOT comparing: delta_rh (intentionally different method)")
    print()

    mismatches = {col: [] for col in all_compare}
    total_comparisons = 0

    for (date, az_bin), group in arc_table.groupby(["date", "azimuth_bin"]):
        if len(group) < 3:
            continue

        mask = (df_sectors["date"] == date) & (df_sectors["azimuth_bin"] == az_bin)
        if mask.sum() == 0:
            continue

        df_row = df_sectors[mask].iloc[0]
        total_comparisons += 1

        # RH/Amp stats (from un-normalized arc_table)
        amp_mean = group["Amp"].mean()
        clf_vals = {
            "amp_mean": amp_mean,
            "amp_cv": group["Amp"].std() / amp_mean if amp_mean > 0 else np.nan,
            "rh_std": group["RH"].std(),
        }

        # Raw SNR medians (from un-normalized arc_table)
        if has_snr:
            for feat in ["CLR", "AF", "PR", "gamma"]:
                col_name = f"{feat.lower()}_med"
                if feat in group.columns:
                    clf_vals[col_name] = group[feat].median()

        for col, clf_val in clf_vals.items():
            df_val = df_row.get(col, np.nan)
            if pd.notna(clf_val) and pd.notna(df_val):
                if not np.isclose(clf_val, df_val, rtol=1e-4, atol=1e-6):
                    mismatches[col].append({
                        "date": date, "az_bin": az_bin,
                        "classifier": clf_val, "layer2": df_val,
                        "diff": abs(clf_val - df_val),
                    })

        # Z-scored medians: classifier normalizes in-place then takes median
        if has_snr and z_compare:
            clf_z_group = clf_arc[
                (clf_arc["date"] == date) & (clf_arc["azimuth_bin"] == az_bin)
            ]
            if len(clf_z_group) >= 3:
                for feat in ["CLR", "AF", "PR", "gamma"]:
                    z_col = f"{feat.lower()}_z"
                    if feat in clf_z_group.columns and z_col in df_row.index:
                        clf_val = clf_z_group[feat].median()
                        df_val = df_row[z_col]
                        if pd.notna(clf_val) and pd.notna(df_val):
                            if not np.isclose(clf_val, df_val, rtol=1e-4, atol=1e-6):
                                mismatches[z_col].append({
                                    "date": date, "az_bin": az_bin,
                                    "classifier": clf_val, "layer2": df_val,
                                    "diff": abs(clf_val - df_val),
                                })

    print(f"Compared {total_comparisons} (date, sector) groups\n")
    print("--- Raw feature medians ---")

    all_pass = True
    for col in raw_compare:
        n_bad = len(mismatches[col])
        status = "PASS" if n_bad == 0 else "MISMATCH"
        if n_bad > 0:
            all_pass = False
        print(f"  {col:>12s}: {status} ({n_bad}/{total_comparisons} mismatches)")
        if 0 < n_bad <= 5:
            for m in mismatches[col][:5]:
                print(f"    {m['date']} az={m['az_bin']}: "
                      f"clf={m['classifier']:.6f} vs l2={m['layer2']:.6f} "
                      f"(diff={m['diff']:.2e})")

    if z_compare:
        print("\n--- Z-scored feature medians ---")
        for col in z_compare:
            n_bad = len(mismatches[col])
            status = "PASS" if n_bad == 0 else "MISMATCH"
            if n_bad > 0:
                all_pass = False
            print(f"  {col:>12s}: {status} ({n_bad}/{total_comparisons} mismatches)")
            if 0 < n_bad <= 5:
                for m in mismatches[col][:5]:
                    print(f"    {m['date']} az={m['az_bin']}: "
                          f"clf={m['classifier']:.6f} vs l2={m['layer2']:.6f} "
                          f"(diff={m['diff']:.2e})")

    print(f"\n{'='*70}")
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: MISMATCHES FOUND (see above)")
    print(f"{'='*70}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Verify daily_features.parquet against classifier internals"
    )
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    success = verify(args.station, args.year)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
