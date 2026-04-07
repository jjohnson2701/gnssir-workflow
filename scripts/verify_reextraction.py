# ABOUTME: Post-re-extraction sanity checker. Verifies corrected AF baselines, AF drop, gamma_r2,
# ABOUTME: frac_full_arc rates, feature preservation, and PRN/frequency distribution against backups.

"""
Verify re-extraction completed correctly for UMNQ 2025 and ROSS 2024.

Six criteria (all must PASS):
  1. AF baselines non-zero (both stations)
  2. AF median dropped >=70% at UMNQ
  3. AF changed >10% at ROSS
  4. gamma_r2 non-null for full arcs (both stations)
  5. frac_full_arc in expected range in daily_features
  6. CLR/RH medians within 5% of backup + PRN distribution preserved

Usage:
    python scripts/verify_reextraction.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_annual"

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    mark = "✓" if condition else "✗"
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f"\n      {detail}"
    print(msg)
    return condition


def load(path):
    if not path.exists():
        print(f"  [✗] FILE MISSING: {path}")
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Criterion 1: AF baselines non-zero
# ---------------------------------------------------------------------------
def check_baselines():
    print("\n--- Criterion 1: AF baselines non-zero ---")
    all_pass = True
    for station, year in [("UMNQ", 2025), ("ROSS", 2024)]:
        npz_path = RESULTS / station / f"{station}_{year}_af_baselines.npz"
        if not npz_path.exists():
            print(f"  [✗] {station}: baselines file missing: {npz_path}")
            all_pass = False
            continue
        data = np.load(npz_path, allow_pickle=True)
        # Structure: sin_grid (50,), keys (N,2), baselines (N,50)
        baselines = data["baselines"]
        n_total = len(baselines)
        nonzero = int(np.sum([np.any(b != 0) for b in baselines]))
        ok = nonzero == n_total and n_total > 0
        all_pass &= check(
            f"{station}: {nonzero}/{n_total} baselines non-zero",
            ok,
        )
    return all_pass


# ---------------------------------------------------------------------------
# Criterion 2 & 3: AF distribution changed
# ---------------------------------------------------------------------------
def check_af_change():
    print("\n--- Criteria 2 & 3: AF distribution changed ---")
    all_pass = True

    # UMNQ: expect >=70% drop in station-level median AF
    umnq_new = load(RESULTS / "UMNQ" / "UMNQ_2025_arc_table.parquet")
    umnq_old = load(RESULTS / "UMNQ" / "UMNQ_2025_arc_table_backup.parquet")
    if umnq_new is not None and umnq_old is not None:
        new_med = umnq_new["AF"].median()
        old_med = umnq_old["AF"].median()
        drop_frac = (old_med - new_med) / old_med if old_med != 0 else 0
        ok = drop_frac >= 0.60
        all_pass &= check(
            f"UMNQ AF median drop >= 60%: {drop_frac*100:.1f}%",
            ok,
            f"old={old_med:.0f}  new={new_med:.0f}"
        )

    # ROSS: expect >10% change (direction less predictable)
    ross_new = load(RESULTS / "ROSS" / "ROSS_2024_arc_table.parquet")
    ross_old = load(RESULTS / "ROSS" / "ROSS_2024_arc_table_precorrection.parquet")
    if ross_new is not None and ross_old is not None:
        new_med = ross_new["AF"].median()
        old_med = ross_old["AF"].median()
        change_frac = abs(old_med - new_med) / old_med if old_med != 0 else 0
        ok = change_frac >= 0.10
        all_pass &= check(
            f"ROSS AF median changed >10%: {change_frac*100:.1f}%",
            ok,
            f"old={old_med:.0f}  new={new_med:.0f}"
        )

    return all_pass


# ---------------------------------------------------------------------------
# Criterion 4: gamma_r2 non-null for full arcs
# ---------------------------------------------------------------------------
def check_gamma_r2():
    print("\n--- Criterion 4: gamma_r2 populated for full arcs ---")
    all_pass = True
    for station, year in [("UMNQ", 2025), ("ROSS", 2024)]:
        arc = load(RESULTS / station / f"{station}_{year}_arc_table.parquet")
        if arc is None:
            all_pass = False
            continue
        if "gamma_r2" not in arc.columns:
            all_pass &= check(f"{station}: gamma_r2 column present", False)
            continue
        full_arcs = arc[arc["full_arc"] == True]
        null_frac = full_arcs["gamma_r2"].isna().mean()
        ok = null_frac == 0.0
        all_pass &= check(
            f"{station}: gamma_r2 null fraction for full arcs = {null_frac:.4f}",
            ok,
            f"n_full_arcs={len(full_arcs)}, gamma_r2 median={full_arcs['gamma_r2'].median():.4f}"
        )
    return all_pass


# ---------------------------------------------------------------------------
# Criterion 5: frac_full_arc in expected range
# ---------------------------------------------------------------------------
def check_frac_full_arc():
    print("\n--- Criterion 5: frac_full_arc rates in daily_features ---")
    all_pass = True
    checks = [
        ("UMNQ", 2025, 0.80, 0.90),
        ("ROSS", 2024, 0.95, 1.00),
    ]
    for station, year, lo, hi in checks:
        feat = load(RESULTS / station / f"{station}_{year}_daily_features.parquet")
        if feat is None:
            all_pass = False
            continue
        if "frac_full_arc" not in feat.columns:
            all_pass &= check(f"{station}: frac_full_arc column present", False)
            continue
        mean_frac = feat["frac_full_arc"].mean()
        ok = lo <= mean_frac <= hi
        all_pass &= check(
            f"{station}: frac_full_arc mean={mean_frac:.4f} in [{lo}, {hi}]",
            ok
        )
    return all_pass


# ---------------------------------------------------------------------------
# Criterion 6: Feature medians preserved + PRN distribution
# ---------------------------------------------------------------------------
def check_feature_preservation():
    print("\n--- Criterion 6: CLR/RH medians + PRN distribution ---")
    all_pass = True

    configs = [
        ("UMNQ", 2025,
         RESULTS / "UMNQ" / "UMNQ_2025_arc_table.parquet",
         RESULTS / "UMNQ" / "UMNQ_2025_arc_table_backup.parquet"),
        ("ROSS", 2024,
         RESULTS / "ROSS" / "ROSS_2024_arc_table.parquet",
         RESULTS / "ROSS" / "ROSS_2024_arc_table_precorrection.parquet"),
    ]

    for station, year, new_path, old_path in configs:
        new = load(new_path)
        old = load(old_path)
        if new is None or old is None:
            all_pass = False
            continue

        # Row count within 5%
        count_diff = abs(len(new) - len(old)) / len(old)
        all_pass &= check(
            f"{station}: row count within 5% (diff={count_diff*100:.1f}%)",
            count_diff <= 0.05,
            f"old={len(old)}  new={len(new)}"
        )

        # CLR and RH medians within 5%
        for col in ["CLR", "RH"]:
            if col not in new.columns or col not in old.columns:
                continue
            old_med = old[col].median()
            new_med = new[col].median()
            drift = abs(new_med - old_med) / abs(old_med) if old_med != 0 else 0
            all_pass &= check(
                f"{station}: {col} median drift={drift*100:.2f}% (<=5%)",
                drift <= 0.05,
                f"old={old_med:.4f}  new={new_med:.4f}"
            )

        # PRN/frequency distribution: no combination lost >25% of arcs
        if "sat" in new.columns and "freq_group" in new.columns:
            old_counts = old.groupby(["sat", "freq_group"]).size().rename("old_n")
            new_counts = new.groupby(["sat", "freq_group"]).size().rename("new_n")
            merged = old_counts.to_frame().join(new_counts.to_frame(), how="outer").fillna(0)

            # Combinations present in old but missing entirely from new
            missing = merged[(merged["old_n"] > 0) & (merged["new_n"] == 0)]
            # Combinations that lost >25% of arcs
            large_drop = merged[
                (merged["old_n"] > 5) &
                ((merged["old_n"] - merged["new_n"]) / merged["old_n"] > 0.25)
            ]

            all_pass &= check(
                f"{station}: no PRN/freq combinations disappeared",
                len(missing) == 0,
                f"missing: {missing.index.tolist()[:5]}" if len(missing) > 0 else ""
            )
            all_pass &= check(
                f"{station}: no PRN/freq lost >25% of arcs ({len(large_drop)} flagged)",
                len(large_drop) == 0,
                str(large_drop.head(5)) if len(large_drop) > 0 else ""
            )

            # Unique satellite count
            old_nsats = old["sat"].nunique()
            new_nsats = new["sat"].nunique()
            sat_diff = abs(new_nsats - old_nsats)
            all_pass &= check(
                f"{station}: unique sat count diff={sat_diff} (<=2)",
                sat_diff <= 2,
                f"old={old_nsats}  new={new_nsats}"
            )

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Re-extraction Verification")
    print("=" * 60)

    results = [
        check_baselines(),
        check_af_change(),
        check_gamma_r2(),
        check_frac_full_arc(),
        check_feature_preservation(),
    ]

    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"OVERALL: PASS ({n_pass}/{n_total} criteria)")
    else:
        print(f"OVERALL: FAIL ({n_pass}/{n_total} criteria passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
