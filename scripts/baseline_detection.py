# ABOUTME: Auto-detect baseline period from daily_features using variance and arc density
# ABOUTME: Outputs baseline_definition.csv with window boundaries and quality metrics

"""
Baseline period detection for GNSS-IR surface analysis.

Identifies the most stable contiguous window in a year of daily features,
suitable for anchoring anomaly detection. The baseline period is where the
surface state is most consistent and data quality is highest.

Three modes:
  1. User-provided months: --baseline-months 6,7,8,9
  2. User-provided DOY range: --baseline 150-243
  3. Auto-detected: longest contiguous stable + dense window (min 30 days)

Usage:
    python scripts/baseline_detection.py --station ROSS --year 2024
    python scripts/baseline_detection.py --station ROSS --year 2024 --baseline 150-243
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

# Auto-detection parameters
_MIN_WINDOW_DAYS = 30
_STABILITY_PERCENTILE = 25  # composite variance must be below this percentile
_DENSITY_PERCENTILE = 50    # arc density must be above this percentile
_ROLLING_WINDOW = 30        # days for rolling variance and density


def detect_baseline(daily_features, method="auto", months=None, doy_range=None):
    """Detect or apply a baseline period from daily feature data.

    Args:
        daily_features: DataFrame with columns including 'doy', 'n_arcs',
            and numeric feature columns. Should be pooled rows (azimuth_bin=-1).
        method: 'auto', 'user_months', or 'user_doy'
        months: list of month ints (for method='user_months')
        doy_range: tuple of (start_doy, end_doy) (for method='user_doy')

    Returns:
        dict with baseline definition:
            station, year, start_doy, end_doy, n_days, mean_n_arcs,
            method, provisional
    """
    df = daily_features.copy()

    # Ensure we have doy
    if "doy" not in df.columns and "date" in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear

    if method == "user_months" and months:
        if "date" in df.columns:
            month_col = pd.to_datetime(df["date"]).dt.month
        else:
            # Approximate month from DOY
            month_col = pd.Series(
                pd.Timestamp(year=2024, month=1, day=1) + pd.to_timedelta(df["doy"] - 1, unit="D")
            ).dt.month
        mask = month_col.isin(months)
        baseline_days = df[mask]
        if len(baseline_days) == 0:
            logger.warning(f"No data in months {months}")
            return None
        start_doy = int(baseline_days["doy"].min())
        end_doy = int(baseline_days["doy"].max())
        return {
            "start_doy": start_doy,
            "end_doy": end_doy,
            "n_days": len(baseline_days),
            "mean_n_arcs": float(baseline_days["n_arcs"].mean()),
            "method": "user_months",
            "provisional": False,
        }

    elif method == "user_doy" and doy_range:
        start_doy, end_doy = doy_range
        baseline_days = df[(df["doy"] >= start_doy) & (df["doy"] <= end_doy)]
        if len(baseline_days) == 0:
            logger.warning(f"No data in DOY range {start_doy}-{end_doy}")
            return None
        return {
            "start_doy": start_doy,
            "end_doy": end_doy,
            "n_days": len(baseline_days),
            "mean_n_arcs": float(baseline_days["n_arcs"].mean()),
            "method": "user_doy",
            "provisional": False,
        }

    # Auto-detection
    return _auto_detect_baseline(df)


def _auto_detect_baseline(df):
    """Auto-detect baseline period as the longest stable, dense window.

    Algorithm:
    1. Compute rolling variance (30-day) for each numeric feature
    2. Composite stability = mean of normalized rolling variances
    3. Compute rolling arc density (30-day mean of n_arcs)
    4. Find longest contiguous window (min 30 days) where stability is
       below 25th percentile AND density is above 50th percentile
    """
    df = df.sort_values("doy").reset_index(drop=True)

    # Get numeric feature columns (exclude metadata)
    skip_cols = {"year", "doy", "date", "date_dt", "azimuth_bin", "n_arcs",
                 "n_sats", "n_full_arcs"}
    feature_cols = [c for c in df.columns
                    if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]

    if not feature_cols:
        logger.error("No numeric feature columns found for baseline detection")
        return None

    # Rolling variance per feature (normalized by full-year variance)
    rolling_vars = pd.DataFrame(index=df.index)
    for col in feature_cols:
        vals = df[col].astype(float)
        full_var = vals.var()
        if full_var > 0 and vals.notna().sum() > _ROLLING_WINDOW:
            rv = vals.rolling(_ROLLING_WINDOW, center=True, min_periods=_ROLLING_WINDOW // 2).var()
            rolling_vars[col] = rv / full_var
        else:
            rolling_vars[col] = np.nan

    # Drop all-NaN columns
    rolling_vars = rolling_vars.dropna(axis=1, how="all")
    if rolling_vars.empty:
        logger.error("Insufficient data for rolling variance computation")
        return None

    # Composite stability score
    composite = rolling_vars.mean(axis=1)

    # Rolling arc density
    arc_density = df["n_arcs"].rolling(_ROLLING_WINDOW, center=True,
                                       min_periods=_ROLLING_WINDOW // 2).mean()

    # Thresholds
    stability_thresh = np.nanpercentile(composite.dropna(), _STABILITY_PERCENTILE)
    density_thresh = np.nanpercentile(arc_density.dropna(), _DENSITY_PERCENTILE)

    # Candidate mask: stable AND dense
    candidate = (composite <= stability_thresh) & (arc_density >= density_thresh)
    candidate = candidate.fillna(False)

    # Find longest contiguous True run
    best_start, best_len = _longest_run(candidate.values)

    if best_len < _MIN_WINDOW_DAYS:
        logger.warning(
            f"Auto-detection found only {best_len}-day window "
            f"(minimum {_MIN_WINDOW_DAYS}). Using best available."
        )
        if best_len == 0:
            # Fallback: use the densest 60-day window
            best_start, best_len = _fallback_densest_window(arc_density, df)
            if best_len == 0:
                logger.error("Cannot detect any baseline period")
                return None

    start_doy = int(df.iloc[best_start]["doy"])
    end_idx = min(best_start + best_len - 1, len(df) - 1)
    end_doy = int(df.iloc[end_idx]["doy"])

    baseline_days = df[(df["doy"] >= start_doy) & (df["doy"] <= end_doy)]

    logger.info(
        f"Auto-detected baseline: DOY {start_doy}-{end_doy} "
        f"({len(baseline_days)} days, mean {baseline_days['n_arcs'].mean():.0f} arcs/day)"
    )

    return {
        "start_doy": start_doy,
        "end_doy": end_doy,
        "n_days": len(baseline_days),
        "mean_n_arcs": float(baseline_days["n_arcs"].mean()),
        "method": "auto",
        "provisional": True,
    }


def _longest_run(arr):
    """Find the longest contiguous True run in a boolean array.

    Returns (start_index, length).
    """
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i, val in enumerate(arr):
        if val:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0

    return best_start, best_len


def _fallback_densest_window(arc_density, df, window=60):
    """Fallback: find the densest 60-day window by arc count."""
    if len(df) < window:
        return 0, len(df)

    best_start = 0
    best_density = -1
    for i in range(len(df) - window + 1):
        d = arc_density.iloc[i:i + window].mean()
        if np.isfinite(d) and d > best_density:
            best_density = d
            best_start = i

    return best_start, window


def save_baseline_definition(baseline_def, station, year, results_dir):
    """Save baseline definition to CSV.

    Args:
        baseline_def: dict from detect_baseline()
        station: station ID
        year: processing year
        results_dir: Path to results directory
    """
    results_dir = Path(results_dir)
    row = {
        "station": station,
        "year": year,
        **baseline_def,
    }
    df = pd.DataFrame([row])
    out_path = results_dir / f"{station}_{year}_baseline_definition.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Baseline definition saved: {out_path}")
    return out_path


def load_baseline_definition(station, year, results_dir=None):
    """Load baseline definition from CSV.

    Returns dict with baseline parameters, or None if not found.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results_annual" / station
    results_dir = Path(results_dir)
    path = results_dir / f"{station}_{year}_baseline_definition.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df.iloc[0].to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect baseline period")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--baseline", type=str, default=None,
                        help="DOY range (e.g., 150-243)")
    parser.add_argument("--baseline-months", type=str, default=None,
                        help="Comma-separated months (e.g., 6,7,8,9)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = PROJECT_ROOT / "results_annual" / args.station

    # Load daily features
    for ext in ["csv", "parquet"]:
        feat_path = results_dir / f"{args.station}_{args.year}_daily_features.{ext}"
        if feat_path.exists():
            break
    else:
        print(f"ERROR: daily_features not found for {args.station} {args.year}")
        sys.exit(1)

    if ext == "csv":
        daily = pd.read_csv(feat_path)
    else:
        daily = pd.read_parquet(feat_path)

    # Filter to pooled rows
    if "azimuth_bin" in daily.columns:
        daily = daily[daily["azimuth_bin"] == -1].copy()

    # Determine method
    if args.baseline:
        parts = args.baseline.split("-")
        doy_range = (int(parts[0]), int(parts[1]))
        result = detect_baseline(daily, method="user_doy", doy_range=doy_range)
    elif args.baseline_months:
        months = [int(m) for m in args.baseline_months.split(",")]
        result = detect_baseline(daily, method="user_months", months=months)
    else:
        result = detect_baseline(daily)

    if result:
        save_baseline_definition(result, args.station, args.year, results_dir)
        prov = " [provisional]" if result["provisional"] else ""
        print(f"Baseline: DOY {result['start_doy']}-{result['end_doy']} "
              f"({result['n_days']} days, mean {result['mean_n_arcs']:.0f} arcs/day) "
              f"[{result['method']}]{prov}")
    else:
        print("ERROR: Failed to detect baseline period")
        sys.exit(1)
