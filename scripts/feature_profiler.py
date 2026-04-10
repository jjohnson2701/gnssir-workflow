# ABOUTME: 5-gate feature validation producing feature_scorecard.csv
# ABOUTME: Gates: data sufficiency, computation health, baseline stability, discriminability, redundancy

"""
Feature profiler for GNSS-IR surface analysis.

Validates anomaly detection results through a 5-gate sequential evaluation.
Each feature is assessed for trustworthiness; features passing all gates are
ranked by discriminability.

Gates:
  1. Data Sufficiency — enough arcs in baseline AND evaluation windows
  2. Computation Health — feature-specific prerequisites (gamma_r2, AF baseline)
  3. Baseline Stability — low CV during calm period
  4. Discriminability — Cohen's d between baseline and evaluation windows
  5. Redundancy — pairwise correlation among passing features

Output: feature_scorecard.csv (one row per feature per time window)

Usage:
    python scripts/feature_profiler.py --station ROSS --year 2024
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Gate thresholds
_MIN_ARCS_PER_DAY = 10        # Gate 1: minimum mean arcs/day in each window
_MIN_DAY_COVERAGE = 0.70      # Gate 1: minimum fraction of days with data
_GAMMA_R2_THRESHOLD = 0.7     # Gate 2: minimum gamma_r2 for gamma trust
_CV_THRESHOLD = 0.5           # Gate 3: maximum CV during baseline
_COHENS_D_THRESHOLD = 0.5     # Gate 4: minimum |d| for discriminability
_CORRELATION_THRESHOLD = 0.85  # Gate 5: |r| above this = redundant

# Sliding window sizes and stride
_WINDOW_SIZES = [14, 28]  # 2-week and 4-week
_WINDOW_STRIDE = 7        # 1-week stride

# Metadata columns not to profile
_SKIP_COLS = {
    "year", "doy", "date", "date_dt", "azimuth_bin", "n_arcs", "n_sats",
    "n_full_arcs", "frac_full_arc", "rh_count",
}


def _get_feature_columns(df):
    """Get profiling-eligible feature columns from daily features."""
    features = []
    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        if col.startswith("n_arcs_"):
            continue
        if col.endswith("_count"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def _cohens_d(group1, group2):
    """Compute Cohen's d (effect size) between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((m2 - m1) / pooled_std)


def gate1_data_sufficiency(daily, feature, baseline_doys, window_doys):
    """Gate 1: Check data sufficiency in baseline AND evaluation window.

    Returns (pass, n_arcs_baseline, n_arcs_window, detail).
    """
    bl = daily[daily["doy"].isin(baseline_doys)]
    win = daily[daily["doy"].isin(window_doys)]

    # Check baseline
    bl_valid = bl[feature].notna()
    bl_arcs = bl["n_arcs"].mean() if "n_arcs" in bl.columns and len(bl) > 0 else 0
    bl_coverage = bl_valid.mean() if len(bl) > 0 else 0

    # Check window
    win_valid = win[feature].notna()
    win_arcs = win["n_arcs"].mean() if "n_arcs" in win.columns and len(win) > 0 else 0
    win_coverage = win_valid.mean() if len(win) > 0 else 0

    passes = (bl_arcs >= _MIN_ARCS_PER_DAY and win_arcs >= _MIN_ARCS_PER_DAY
              and bl_coverage >= _MIN_DAY_COVERAGE and win_coverage >= _MIN_DAY_COVERAGE)

    return passes, float(bl_arcs), float(win_arcs), f"bl_cov={bl_coverage:.2f},win_cov={win_coverage:.2f}"


def gate2_computation_health(daily, feature, window_doys):
    """Gate 2: Check feature-specific computation health.

    Returns (pass, detail_string).
    """
    win = daily[daily["doy"].isin(window_doys)]

    # gamma: check gamma_r2
    if feature in ("gamma_med", "gamma_z", "gamma_wmed", "gamma_cv"):
        if "gamma_r2_med" in win.columns:
            r2 = win["gamma_r2_med"].dropna().median()
            if np.isfinite(r2):
                passes = r2 >= _GAMMA_R2_THRESHOLD
                return passes, f"gamma_r2={r2:.3f}"
        return False, "gamma_r2 unavailable"

    # AF: check non-zero values (zero = domain bug)
    if feature in ("af_med", "af_z", "af_wmed", "af_cv"):
        if feature in win.columns:
            af_vals = win[feature].dropna()
            if len(af_vals) > 0:
                zero_frac = (af_vals.abs() < 1e-6).mean()
                passes = zero_frac < 0.5
                return passes, f"zero_frac={zero_frac:.2f}"
        return False, "AF unavailable"

    # CLR: check component stability if available
    if feature in ("clr_med", "clr_z", "clr_wmed", "clr_cv"):
        # Will decompose in profiler output if clr_peak_power/total_power available
        # For gate purposes, check CLR is non-trivial
        if feature in win.columns:
            clr_vals = win[feature].dropna()
            if len(clr_vals) > 0 and clr_vals.std() > 0:
                return True, "CLR valid"
        return False, "CLR unavailable"

    # Frequency-split amplitude: check per-band arc counts
    if feature.startswith("amp_L") and feature.endswith("_mean"):
        band = feature.replace("amp_", "").replace("_mean", "")
        count_col = f"n_arcs_{band}"
        if count_col in win.columns:
            mean_count = win[count_col].dropna().mean()
            passes = mean_count >= 5
            return passes, f"n_arcs_{band}={mean_count:.0f}"
        return True, "no per-band count available"

    # Default: passes
    return True, "no specific health check"


def gate3_baseline_stability(daily, feature, baseline_doys):
    """Gate 3: Check coefficient of variation during baseline.

    Returns (pass, cv_value).
    """
    bl = daily[daily["doy"].isin(baseline_doys)]
    vals = bl[feature].dropna()

    if len(vals) < 5:
        return False, np.nan

    mean_val = vals.mean()
    if abs(mean_val) < 1e-12:
        return False, np.nan

    cv = float(vals.std() / abs(mean_val))
    return cv < _CV_THRESHOLD, cv


def gate4_discriminability(daily, feature, baseline_doys, window_doys):
    """Gate 4: Compute Cohen's d between baseline and evaluation window.

    Returns (pass, cohens_d_value).
    """
    bl = daily[daily["doy"].isin(baseline_doys)][feature].dropna()
    win = daily[daily["doy"].isin(window_doys)][feature].dropna()

    if len(bl) < 5 or len(win) < 5:
        return False, np.nan

    d = _cohens_d(bl, win)
    if not np.isfinite(d):
        return False, np.nan

    return abs(d) >= _COHENS_D_THRESHOLD, d


def gate5_redundancy(scorecard_rows, daily, baseline_doys):
    """Gate 5: Check pairwise correlation among features passing gates 1-4.

    Modifies scorecard_rows in place: marks redundant features.
    """
    # Get features that passed gates 1-4
    passing = [r for r in scorecard_rows if r.get("gate4_pass") and r.get("usable")]
    if len(passing) < 2:
        return

    # Compute pairwise correlations on baseline data
    bl = daily[daily["doy"].isin(baseline_doys)]
    features = [r["feature"] for r in passing if r["feature"] in bl.columns]

    if len(features) < 2:
        return

    corr_matrix = bl[features].corr()

    # For each pair with |r| > threshold, keep the one with higher |d|
    marked_redundant = set()
    for i, f1 in enumerate(features):
        if f1 in marked_redundant:
            continue
        for j, f2 in enumerate(features):
            if j <= i or f2 in marked_redundant:
                continue
            r = corr_matrix.loc[f1, f2]
            if abs(r) > _CORRELATION_THRESHOLD:
                # Find their scorecard rows and compare d
                r1 = next(row for row in passing if row["feature"] == f1)
                r2 = next(row for row in passing if row["feature"] == f2)
                d1 = abs(r1.get("gate4_cohens_d", 0) or 0)
                d2 = abs(r2.get("gate4_cohens_d", 0) or 0)
                loser = f2 if d1 >= d2 else f1
                winner = f1 if d1 >= d2 else f2
                marked_redundant.add(loser)

                # Update the scorecard row
                for row in scorecard_rows:
                    if row["feature"] == loser and row.get("usable"):
                        row["gate5_pass"] = False
                        row["gate5_correlated_with"] = winner
                        row["usable"] = False
                        row["failure_reason"] = "redundant"


def profile_features(daily, baseline_doys, station, year):
    """Run all 5 gates on all features at both window sizes.

    Args:
        daily: pooled daily features DataFrame
        baseline_doys: set/list of DOY values in the baseline period
        station: station ID
        year: processing year

    Returns:
        DataFrame (feature_scorecard)
    """
    baseline_doys = set(baseline_doys)
    features = _get_feature_columns(daily)
    all_doys = sorted(daily["doy"].unique())

    if not features:
        logger.error("No feature columns found for profiling")
        return pd.DataFrame()

    logger.info(f"Profiling {len(features)} features across "
                f"{len(all_doys)} days")

    scorecard_rows = []

    for window_size in _WINDOW_SIZES:
        window_label = f"{window_size // 7}w"

        # Generate sliding windows
        windows = []
        for start_idx in range(0, len(all_doys) - window_size + 1, _WINDOW_STRIDE):
            window_doys = set(all_doys[start_idx:start_idx + window_size])
            # Skip windows that are entirely within baseline
            if window_doys.issubset(baseline_doys):
                continue
            start_doy = min(window_doys)
            windows.append((start_doy, window_doys))

        for feature in features:
            if feature not in daily.columns:
                continue

            for start_doy, window_doys in windows:
                row = {
                    "station": station,
                    "year": year,
                    "feature": feature,
                    "window_start_doy": start_doy,
                    "window_size": window_label,
                }

                # Gate 1
                g1_pass, g1_bl_arcs, g1_win_arcs, g1_detail = gate1_data_sufficiency(
                    daily, feature, baseline_doys, window_doys
                )
                row["gate1_pass"] = g1_pass
                row["gate1_n_arcs_baseline"] = g1_bl_arcs
                row["gate1_n_arcs_window"] = g1_win_arcs

                if not g1_pass:
                    row.update({"gate2_pass": False, "gate2_detail": "",
                                "gate3_pass": False, "gate3_cv_baseline": np.nan,
                                "gate4_pass": False, "gate4_cohens_d": np.nan,
                                "gate5_pass": False, "gate5_correlated_with": "",
                                "usable": False, "rank": np.nan,
                                "failure_reason": "insufficient_data"})
                    scorecard_rows.append(row)
                    continue

                # Gate 2
                g2_pass, g2_detail = gate2_computation_health(daily, feature, window_doys)
                row["gate2_pass"] = g2_pass
                row["gate2_detail"] = g2_detail

                if not g2_pass:
                    row.update({"gate3_pass": False, "gate3_cv_baseline": np.nan,
                                "gate4_pass": False, "gate4_cohens_d": np.nan,
                                "gate5_pass": False, "gate5_correlated_with": "",
                                "usable": False, "rank": np.nan,
                                "failure_reason": "untrustworthy_computation"})
                    scorecard_rows.append(row)
                    continue

                # Gate 3
                g3_pass, g3_cv = gate3_baseline_stability(daily, feature, baseline_doys)
                row["gate3_pass"] = g3_pass
                row["gate3_cv_baseline"] = g3_cv

                if not g3_pass:
                    row.update({"gate4_pass": False, "gate4_cohens_d": np.nan,
                                "gate5_pass": False, "gate5_correlated_with": "",
                                "usable": False, "rank": np.nan,
                                "failure_reason": "unstable_baseline"})
                    scorecard_rows.append(row)
                    continue

                # Gate 4
                g4_pass, g4_d = gate4_discriminability(daily, feature,
                                                       baseline_doys, window_doys)
                row["gate4_pass"] = g4_pass
                row["gate4_cohens_d"] = g4_d

                if not g4_pass:
                    row.update({"gate5_pass": False, "gate5_correlated_with": "",
                                "usable": False, "rank": np.nan,
                                "failure_reason": "non_discriminant"})
                    scorecard_rows.append(row)
                    continue

                # Passes gates 1-4
                row["gate5_pass"] = True
                row["gate5_correlated_with"] = ""
                row["usable"] = True
                row["rank"] = np.nan
                row["failure_reason"] = ""

                scorecard_rows.append(row)

    # Gate 5: Redundancy (across features within each window)
    # Group by window and apply redundancy check
    by_window = {}
    for row in scorecard_rows:
        key = (row["window_start_doy"], row["window_size"])
        by_window.setdefault(key, []).append(row)

    for (start_doy, ws), rows in by_window.items():
        gate5_redundancy(rows, daily, baseline_doys)

    # Rank usable features by |Cohen's d| within each window
    for (start_doy, ws), rows in by_window.items():
        usable = [(r, abs(r.get("gate4_cohens_d", 0) or 0))
                   for r in rows if r.get("usable")]
        usable.sort(key=lambda x: -x[1])
        for rank, (r, _) in enumerate(usable, 1):
            r["rank"] = rank

    scorecard = pd.DataFrame(scorecard_rows)

    # Sort by window, then rank
    if not scorecard.empty:
        scorecard = scorecard.sort_values(
            ["window_size", "window_start_doy", "rank", "feature"]
        ).reset_index(drop=True)

    n_usable = scorecard["usable"].sum() if "usable" in scorecard.columns else 0
    logger.info(f"Scorecard: {len(scorecard)} entries, "
                f"{n_usable} usable feature-window combinations")

    return scorecard


def run_profiling(station, year, results_dir=None):
    """Run feature profiling and save scorecard.

    Args:
        station: Station ID
        year: Processing year
        results_dir: Path to results directory

    Returns:
        Path to feature_scorecard.csv, or None on failure.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results_annual" / station
    results_dir = Path(results_dir)

    # Load daily features
    for ext in ["csv", "parquet"]:
        feat_path = results_dir / f"{station}_{year}_daily_features.{ext}"
        if feat_path.exists():
            break
    else:
        logger.error(f"daily_features not found for {station} {year}")
        return None

    daily = pd.read_csv(feat_path) if ext == "csv" else pd.read_parquet(feat_path)

    # Filter to pooled rows
    if "azimuth_bin" in daily.columns:
        daily = daily[daily["azimuth_bin"] == -1].copy()

    if "doy" not in daily.columns and "date" in daily.columns:
        daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear

    # Load baseline definition
    from scripts.baseline_detection import load_baseline_definition
    baseline_def = load_baseline_definition(station, year, results_dir)
    if baseline_def is None:
        logger.error("No baseline definition found. Run baseline detection first.")
        return None

    start_doy = int(baseline_def["start_doy"])
    end_doy = int(baseline_def["end_doy"])
    baseline_doys = set(range(start_doy, end_doy + 1))

    # Profile
    scorecard = profile_features(daily, baseline_doys, station, year)
    if scorecard.empty:
        logger.error("No scorecard entries produced")
        return None

    # Save
    out_path = results_dir / f"{station}_{year}_feature_scorecard.csv"
    scorecard.to_csv(out_path, index=False, float_format="%.4f")
    logger.info(f"Feature scorecard saved: {out_path} ({len(scorecard)} entries)")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature profiling")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_profiling(args.station, args.year)
    if result:
        print(f"Output: {result}")
    else:
        print("ERROR: Feature profiling failed")
        sys.exit(1)
