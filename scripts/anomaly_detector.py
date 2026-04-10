# ABOUTME: PCA + Mahalanobis anomaly detection with per-feature contributions
# ABOUTME: Observational state labels (baseline/anomalous/transition), n_arcs weighting

"""
Anomaly detection for GNSS-IR surface analysis.

Builds a PCA baseline model from the baseline period, scores every day by
Mahalanobis distance, and decomposes the distance back to per-feature
contributions. State labels are observational, not phenomenon-specific.

Pipeline:
  1. build_baseline_model() — PCA + covariance from baseline period
  2. score_anomalies()      — project all days, compute distances + contributions
  3. classify_states()      — temporal coherence state machine

Usage:
    Called by run_analysis.py, or standalone:
    python scripts/anomaly_detector.py --station ROSS --year 2024
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Metadata columns to exclude from feature selection
_METADATA_COLS = {
    "year", "doy", "date", "date_dt", "azimuth_bin", "n_arcs", "n_sats",
    "n_full_arcs", "frac_full_arc", "rh_count",
}

# Default threshold when no ROC optimization is possible
_DEFAULT_THRESHOLD = 3.0


def _select_features(daily_features, exclude=None):
    """Auto-select all numeric feature columns, excluding metadata.

    Args:
        daily_features: DataFrame
        exclude: additional columns to exclude

    Returns:
        list of feature column names
    """
    skip = set(_METADATA_COLS)
    if exclude:
        skip.update(exclude)

    features = []
    for col in daily_features.columns:
        if col in skip:
            continue
        if col.startswith("n_arcs_"):
            continue  # per-band arc counts are metadata
        if col.startswith("rh_") and col.endswith("_count"):
            continue
        if pd.api.types.is_numeric_dtype(daily_features[col]):
            # Must have enough non-NaN values
            if daily_features[col].notna().sum() > 20:
                features.append(col)

    return features


def build_baseline_model(baseline_features, features=None,
                         variance_threshold=0.95):
    """Build PCA baseline model from the baseline period.

    Args:
        baseline_features: DataFrame with feature columns. Should be pooled
            rows (azimuth_bin=-1) from the baseline period only.
        features: list of column names to use. If None, auto-selects all
            numeric columns (excluding metadata).
        variance_threshold: cumulative PCA variance to retain (default 0.95)

    Returns:
        dict with: scaler, pca, mean_pc, cov_pc_inv, n_components,
        feature_names, explained_variance, n_training_days, pca_components
    """
    if features is None:
        features = _select_features(baseline_features)

    if not features:
        raise ValueError("No valid feature columns found")

    data = baseline_features[features].dropna()
    if len(data) < 10:
        raise ValueError(f"Need >=10 baseline days, got {len(data)}")

    logger.info(f"Building baseline model with {len(features)} features, "
                f"{len(data)} days")

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA — retain components explaining >= threshold variance, min 2
    n_max = min(len(features), len(data) - 1)
    pca = PCA(n_components=n_max)
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_components = max(n_components, 2)
    n_components = min(n_components, n_max)

    logger.info(f"PCA: {n_components}/{len(features)} components, "
                f"{cumvar[n_components - 1]:.1%} variance retained")

    # Baseline centroid and covariance in reduced PC space
    pc_coords = pca.transform(scaled)[:, :n_components]
    mean_pc = pc_coords.mean(axis=0)
    cov_pc = np.cov(pc_coords, rowvar=False)

    # Regularize if ill-conditioned
    if cov_pc.ndim == 0:
        cov_pc = np.array([[cov_pc]])
    cond = np.linalg.cond(cov_pc)
    if cond > 1e6:
        logger.warning(f"Covariance condition number {cond:.1e}, regularizing")
        cov_pc += np.eye(n_components) * 1e-6

    cov_pc_inv = np.linalg.inv(cov_pc)

    return {
        "scaler": scaler,
        "pca": pca,
        "mean_pc": mean_pc,
        "cov_pc": cov_pc,
        "cov_pc_inv": cov_pc_inv,
        "n_components": n_components,
        "feature_names": features,
        "explained_variance": pca.explained_variance_ratio_[:n_components].tolist(),
        "n_training_days": len(data),
        "pca_components": pca.components_[:n_components],  # for feature contributions
    }


def score_anomalies(daily_features, model):
    """Compute Mahalanobis distance + per-feature contributions for each day.

    Per-feature contributions are computed by decomposing the Mahalanobis
    distance through the PCA back to original feature space. For each day,
    the contribution of feature j approximately sums to the total distance.

    Args:
        daily_features: DataFrame with columns matching model's feature_names.
        model: dict from build_baseline_model()

    Returns:
        DataFrame with: doy, n_arcs, mahal_distance, {feature}_contribution
        for each feature in the model.
    """
    features = model["feature_names"]
    n_comp = model["n_components"]

    # Get valid rows (non-NaN for all features)
    valid_mask = daily_features[features].notna().all(axis=1)
    valid = daily_features[valid_mask].copy()

    if valid.empty:
        logger.warning("No valid rows for anomaly scoring")
        return pd.DataFrame()

    # Project into PC space
    scaled = model["scaler"].transform(valid[features])
    pc_coords = model["pca"].transform(scaled)[:, :n_comp]

    # Mahalanobis distance
    diffs_pc = pc_coords - model["mean_pc"]
    left = diffs_pc @ model["cov_pc_inv"]
    distances = np.sqrt(np.sum(left * diffs_pc, axis=1))

    # Per-feature contributions via PCA decomposition
    # Map PC-space deviation back to standardized feature space
    # contribution_j = sum_k (W_kj * diff_pc_k * (Sigma^-1 @ diff_pc)_k)
    # where W_kj is the PCA loading for component k, feature j
    pca_components = model["pca_components"]  # shape (n_comp, n_features)

    # For each day, compute per-feature contribution
    n_features = len(features)
    contributions = np.zeros((len(valid), n_features))

    for i in range(len(valid)):
        # Weighted PC diff: (Sigma^-1 @ diff_pc) element-wise * diff_pc
        weighted_pc = left[i] * diffs_pc[i]  # shape (n_comp,)
        # Map back to feature space
        for j in range(n_features):
            contributions[i, j] = np.sum(weighted_pc * pca_components[:, j])

    # Build result DataFrame
    result = pd.DataFrame(index=valid.index)
    if "year" in valid.columns:
        result["year"] = valid["year"].values
    if "doy" in valid.columns:
        result["doy"] = valid["doy"].values
    if "n_arcs" in valid.columns:
        result["n_arcs"] = valid["n_arcs"].values
    if "date" in valid.columns:
        result["date"] = valid["date"].values

    result["mahal_distance"] = distances

    for j, feat in enumerate(features):
        result[f"{feat}_contribution"] = contributions[:, j]

    return result


def classify_states(scores_df, threshold=None, min_duration=3):
    """Classify each day into observational states with temporal coherence.

    States:
      - baseline: within expected variation
      - anomalous: significant deviation from baseline
      - transition_in: onset of anomalous period
      - transition_out: return toward baseline

    Args:
        scores_df: DataFrame with 'mahal_distance' column, sorted by doy
        threshold: distance threshold (default: auto from distribution)
        min_duration: minimum consecutive days for state change

    Returns:
        DataFrame with added 'state' column
    """
    if threshold is None:
        # Auto-threshold: use percentile-based approach
        # The baseline model is centered, so baseline days cluster near 0
        # Use the 90th percentile as a reasonable separator
        threshold = np.percentile(scores_df["mahal_distance"].dropna(),
                                  _percentile_for_threshold(scores_df))
        threshold = max(threshold, 2.0)  # floor at 2.0
        logger.info(f"Auto-threshold: {threshold:.2f}")

    df = scores_df.copy()
    if "doy" in df.columns:
        df = df.sort_values("doy").reset_index(drop=True)

    df["above_threshold"] = df["mahal_distance"] > threshold

    n = len(df)
    states = ["baseline"] * n
    current_state = "baseline"
    consec_above = 0
    consec_below = 0

    for i in range(n):
        if df.loc[i, "above_threshold"]:
            consec_above += 1
            consec_below = 0
        else:
            consec_below += 1
            consec_above = 0

        if current_state == "baseline":
            if consec_above >= min_duration:
                current_state = "transition_in"
                for j in range(max(0, i - min_duration + 1), i):
                    states[j] = "transition_in"
        elif current_state == "transition_in":
            if consec_above >= min_duration:
                current_state = "anomalous"
            elif consec_below >= min_duration:
                current_state = "baseline"
        elif current_state == "anomalous":
            if consec_below >= min_duration:
                current_state = "transition_out"
                for j in range(max(0, i - min_duration + 1), i):
                    states[j] = "transition_out"
        elif current_state == "transition_out":
            if consec_below >= min_duration:
                current_state = "baseline"
            elif consec_above >= min_duration:
                current_state = "anomalous"

        states[i] = current_state

    df["state"] = states
    return df


def _percentile_for_threshold(scores_df):
    """Estimate threshold percentile based on data distribution.

    If the station has a clear bimodal distribution (baseline vs anomalous),
    use a lower percentile. Otherwise use a higher one.
    """
    # Simple heuristic: if >25% of days have high distance, lower the percentile
    d = scores_df["mahal_distance"].dropna()
    if len(d) < 30:
        return 90
    high_frac = (d > d.median() * 2).mean()
    if high_frac > 0.25:
        return 75  # bimodal-ish, use 75th
    return 90


def run_anomaly_detection(station, year, results_dir=None, baseline_def=None):
    """Run full anomaly detection pipeline.

    Args:
        station: Station ID
        year: Processing year
        results_dir: Path to results directory
        baseline_def: dict with start_doy, end_doy (from baseline_detection)

    Returns:
        Path to anomaly_scores.csv, or None on failure.
    """
    import json

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

    # Load or detect baseline
    if baseline_def is None:
        from scripts.baseline_detection import load_baseline_definition
        baseline_def = load_baseline_definition(station, year, results_dir)

    if baseline_def is None:
        logger.error("No baseline definition found. Run baseline detection first.")
        return None

    start_doy = int(baseline_def["start_doy"])
    end_doy = int(baseline_def["end_doy"])
    logger.info(f"Using baseline DOY {start_doy}-{end_doy}")

    # Split into baseline and full dataset
    baseline_mask = (daily["doy"] >= start_doy) & (daily["doy"] <= end_doy)
    baseline_data = daily[baseline_mask]

    if len(baseline_data) < 10:
        logger.error(f"Only {len(baseline_data)} baseline days (need >=10)")
        return None

    # Build model using ALL available features
    try:
        model = build_baseline_model(baseline_data)
    except ValueError as e:
        logger.error(f"Failed to build baseline model: {e}")
        return None

    # Score all days
    scores = score_anomalies(daily, model)
    if scores.empty:
        logger.error("No valid anomaly scores computed")
        return None

    # Load threshold from config if available
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    threshold = _DEFAULT_THRESHOLD
    if cfg_path.exists():
        with open(cfg_path) as f:
            all_cfg = json.load(f)
        station_cfg = all_cfg.get(station, {})
        threshold = station_cfg.get("classification", {}).get("threshold", threshold)

    # Classify states
    scores = classify_states(scores, threshold=threshold)

    # Save
    out_path = results_dir / f"{station}_{year}_anomaly_scores.csv"
    scores.to_csv(out_path, index=False, float_format="%.6f")
    logger.info(f"Anomaly scores saved: {out_path} ({len(scores)} days)")

    # Log state counts
    if "state" in scores.columns:
        counts = scores["state"].value_counts()
        logger.info(f"States: {dict(counts)}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_anomaly_detection(args.station, args.year)
    if result:
        print(f"Output: {result}")
    else:
        print("ERROR: Anomaly detection failed")
        sys.exit(1)
