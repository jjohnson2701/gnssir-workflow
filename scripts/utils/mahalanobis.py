# ABOUTME: PCA + Mahalanobis distance classifier for surface state detection
# ABOUTME: Builds summer water-state baseline, computes anomaly distance, optimizes threshold via ROC

"""
Mahalanobis distance classifier for GNSS-IR surface state detection.

Builds a PCA-reduced baseline from ice-free (summer) months, then scores
every day by its Mahalanobis distance from that baseline. Days far from
the water state are likely ice/snow/frozen ground.

Core pipeline:
  1. build_water_model()  — PCA + covariance from summer daily_features
  2. score_mahalanobis()  — project new data, compute distances
  3. optimize_threshold() — ROC analysis against ground truth (GLERL)
  4. detect_states()      — temporal coherence state machine

Extracted from classifier_v2.py. Validated AUC=0.895 vs GLERL across
9 Great Lakes stations.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default feature set (highest discriminating power from Cohen's d analysis)
DEFAULT_FEATURES = [
    "amp_mean",    # Mean amplitude — elevated under rough ice
    "rh_std",      # RH scatter — elevated under ice
    "clr_med",     # Clarity ratio median
    "af_med",      # Area factor median — elevated under rough ice
    "gamma_med",   # Damping median — elevated under rough ice
    "pr_med",      # Peak ratio median
    "p2n_mean",    # Peak-to-noise ratio — highest Cohen's d (0.70)
]


# ---------------------------------------------------------------------------
# Step 1: Build water-state model from summer data
# ---------------------------------------------------------------------------

def build_water_model(summer_features, features=None, variance_threshold=0.95):
    """Build PCA water-state model from ice-free period data.

    Args:
        summer_features: DataFrame with columns matching `features` list.
            Should contain only ice-free (summer) days, pooled across
            azimuths (azimuth_bin == -1).
        features: list of column names to use (default: DEFAULT_FEATURES)
        variance_threshold: cumulative variance to retain (default 0.95)

    Returns:
        dict with keys: scaler, pca, mean_pc, cov_pc_inv, n_components,
        feature_names, explained_variance, n_training_days
    """
    if features is None:
        features = DEFAULT_FEATURES

    data = summer_features[features].dropna()
    if len(data) < 10:
        raise ValueError(f"Need >=10 summer days, got {len(data)}")

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA — retain components explaining >= threshold variance, min 2
    pca = PCA()
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_components = max(n_components, 2)

    logger.info(f"PCA: {n_components} components, "
                f"{cumvar[n_components - 1]:.1%} variance, "
                f"{len(data)} training days")

    # Water-state centroid and covariance in reduced PC space
    pc_coords = pca.transform(scaled)[:, :n_components]
    mean_pc = pc_coords.mean(axis=0)
    cov_pc = np.cov(pc_coords, rowvar=False)

    # Regularize if ill-conditioned
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
    }


# ---------------------------------------------------------------------------
# Step 2: Score new data
# ---------------------------------------------------------------------------

def score_mahalanobis(daily_features, model):
    """Compute Mahalanobis distance from water state for each day.

    Args:
        daily_features: DataFrame with columns matching model's feature_names.
            Should be pooled rows (azimuth_bin == -1).
        model: dict from build_water_model()

    Returns:
        DataFrame with columns: date (if present), mahal_dist, pc1, pc2, ...
    """
    features = model["feature_names"]
    n_comp = model["n_components"]

    feats = daily_features[features].dropna()
    if feats.empty:
        return pd.DataFrame()

    # Project into PC space
    scaled = model["scaler"].transform(feats[features])
    pc_coords = model["pca"].transform(scaled)[:, :n_comp]

    # Vectorized Mahalanobis: d_i = sqrt((x_i - mu) @ Sigma^-1 @ (x_i - mu)^T)
    diffs = pc_coords - model["mean_pc"]
    left = diffs @ model["cov_pc_inv"]
    distances = np.sqrt(np.sum(left * diffs, axis=1))

    result = pd.DataFrame({"mahal_dist": distances}, index=feats.index)
    for j in range(n_comp):
        result[f"pc{j + 1}"] = pc_coords[:, j]

    # Carry over date if available
    if "date" in daily_features.columns:
        result["date"] = daily_features.loc[feats.index, "date"].values

    return result


# ---------------------------------------------------------------------------
# Step 3: Threshold optimization via ROC (requires ground truth)
# ---------------------------------------------------------------------------

def optimize_threshold(mahal_df, ground_truth_df,
                       ice_col="ice_concentration",
                       ice_threshold=30, water_threshold=5):
    """Find optimal Mahalanobis threshold using external ground truth.

    Uses ROC analysis with Youden's J statistic.

    Args:
        mahal_df: DataFrame with 'date' and 'mahal_dist' columns
        ground_truth_df: DataFrame with 'date' and ice_col columns
        ice_col: column name for ice concentration (0-100)
        ice_threshold: concentration above which = ice
        water_threshold: concentration below which = water

    Returns:
        dict with threshold, auc, tpr, fpr, n_valid, etc.
    """
    from sklearn.metrics import roc_curve, auc

    merged = mahal_df.merge(ground_truth_df, on="date", how="inner")

    # Binary labels: clear ice vs clear water, drop ambiguous
    ice_mask = merged[ice_col] > ice_threshold
    water_mask = merged[ice_col] < water_threshold
    valid = merged[ice_mask | water_mask].copy()

    if len(valid) < 20:
        logger.warning(f"Only {len(valid)} clear days, threshold unreliable")
        return {"threshold": 3.0, "auc": np.nan, "n_valid": len(valid)}

    valid["is_ice"] = (valid[ice_col] > ice_threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(valid["is_ice"], valid["mahal_dist"])
    roc_auc = auc(fpr, tpr)

    # Youden's J — optimal operating point
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(f"ROC AUC: {roc_auc:.3f}, "
                f"threshold: {optimal_threshold:.2f} "
                f"(TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})")

    return {
        "threshold": float(optimal_threshold),
        "auc": float(roc_auc),
        "tpr": float(tpr[optimal_idx]),
        "fpr": float(fpr[optimal_idx]),
        "n_ice_days": int(valid["is_ice"].sum()),
        "n_water_days": int(len(valid) - valid["is_ice"].sum()),
        "n_valid": int(len(valid)),
    }


# ---------------------------------------------------------------------------
# Step 4: State machine with temporal coherence
# ---------------------------------------------------------------------------

def detect_states(mahal_df, threshold, min_duration=3):
    """Detect surface state transitions with temporal coherence.

    Enforces minimum consecutive days above/below threshold before
    transitioning. States: water → freeze_up → ice → break_up → water.

    Args:
        mahal_df: DataFrame with 'date' and 'mahal_dist'
        threshold: Mahalanobis distance threshold (from optimize_threshold)
        min_duration: minimum consecutive days for state change

    Returns:
        DataFrame with added columns: above_threshold, state, state_change
    """
    df = mahal_df.sort_values("date").reset_index(drop=True).copy()
    df["above_threshold"] = df["mahal_dist"] > threshold

    n = len(df)
    states = ["water"] * n
    current_state = "water"
    consec_above = 0
    consec_below = 0

    for i in range(n):
        if df.loc[i, "above_threshold"]:
            consec_above += 1
            consec_below = 0
        else:
            consec_below += 1
            consec_above = 0

        if current_state == "water":
            if consec_above >= min_duration:
                current_state = "freeze_up"
                for j in range(max(0, i - min_duration + 1), i):
                    states[j] = "freeze_up"
        elif current_state == "freeze_up":
            if consec_above >= min_duration:
                current_state = "ice"
            elif consec_below >= min_duration:
                current_state = "water"
        elif current_state == "ice":
            if consec_below >= min_duration:
                current_state = "break_up"
                for j in range(max(0, i - min_duration + 1), i):
                    states[j] = "break_up"
        elif current_state == "break_up":
            if consec_below >= min_duration:
                current_state = "water"
            elif consec_above >= min_duration:
                current_state = "ice"

        states[i] = current_state

    df["state"] = states
    df["state_change"] = df["state"] != df["state"].shift(1)
    return df
