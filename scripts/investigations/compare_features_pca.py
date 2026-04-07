#!/usr/bin/env python3
"""PCA-based analysis of SNR feature discriminatory power for ice detection.

Loads per-arc snr_features.parquet and daily ice_classification.parquet,
merges them, and produces:
  1. correlation_matrix.png  — inter-feature Pearson correlations
  2. pca_biplot.png          — PC1 vs PC2 with feature loading arrows, colored by label
  3. feature_separation.csv  — Cohen's d for each feature (ice vs water)
  4. pairwise_scatter.png    — key 2D scatter plots colored by label

Usage:
    python scripts/compare_features_pca.py --station ROSS --year 2024
    python scripts/compare_features_pca.py --station ROSS --years 2020-2024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Features to analyze
FEATURES = ["CLR", "AF", "PR", "gamma", "phase", "MS", "VS"]
# Core features used by the classifier
CORE_FEATURES = ["CLR", "AF", "PR", "gamma", "phase"]


def load_and_merge(station, years, results_dir="results_annual"):
    """Load snr_features + ice_classification, merge on date (doy)."""
    all_arcs = []

    for year in years:
        sf_path = Path(results_dir) / station / f"{station}_{year}_snr_features.parquet"
        ic_path = Path(results_dir) / station / f"{station}_{year}_ice_classification.parquet"

        if not sf_path.exists():
            print(f"  SKIP {sf_path} (not found)")
            continue
        if not ic_path.exists():
            print(f"  SKIP {ic_path} (no ice classification)")
            continue

        sf = pd.read_parquet(sf_path)
        ic = pd.read_parquet(ic_path)

        if "phase" not in sf.columns:
            print(f"  SKIP {sf_path} (no phase column)")
            continue

        # Ice classification has daily labels; use the station-level 'classification' column
        if "classification" not in ic.columns:
            print(f"  SKIP {ic_path} (no 'classification' column)")
            continue

        # Convert ice classification date to doy
        ic["date_dt"] = pd.to_datetime(ic["date"])
        ic["doy"] = ic["date_dt"].dt.dayofyear
        label_map = ic.set_index("doy")["classification"].to_dict()

        # Assign labels to arcs
        sf = sf.copy()
        sf["label"] = sf["doy"].map(label_map)
        sf["year"] = year
        all_arcs.append(sf)

    if not all_arcs:
        print("ERROR: No data loaded")
        sys.exit(1)

    df = pd.concat(all_arcs, ignore_index=True)
    print(f"Loaded {len(df):,} arcs across {len(years)} year(s)")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Unlabeled (no ice_cls for that day): {df['label'].isna().sum():,}")
    return df


def compute_cohens_d(group1, group2):
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (m1 - m2) / pooled_std


def plot_correlation_matrix(df, features, out_path):
    """Correlation matrix heatmap."""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return corr


def plot_pca_biplot(df, features, out_path):
    """PCA biplot: PC1 vs PC2 with loading arrows, colored by label."""
    labeled = df.dropna(subset=["label"])
    labeled = labeled[labeled["label"].isin(["ice", "water"])]
    valid = labeled[features].dropna()
    if len(valid) < 20:
        print(f"  SKIP PCA biplot: only {len(valid)} valid ice/water arcs")
        return None

    X = valid[features].values
    labels = labeled.loc[valid.index, "label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(features), X_scaled.shape[1]))
    scores = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"ice": "#1f77b4", "water": "#ff7f0e"}
    for label in ["water", "ice"]:
        mask = labels == label
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=colors[label], label=label, alpha=0.3, s=8, edgecolors="none")

    # Loading arrows
    loadings = pca.components_[:2, :].T  # (n_features, 2)
    scale = np.abs(scores[:, :2]).max() * 0.8
    for i, feat in enumerate(features):
        ax.annotate(
            feat,
            xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            fontsize=11, fontweight="bold", color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            xytext=(0, 0),
            textcoords="data",
        )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA Biplot: Ice vs Water Arcs")
    ax.legend(loc="upper right")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Print variance explained
    print(f"  Variance explained: " + ", ".join(
        f"PC{i+1}={v:.1%}" for i, v in enumerate(var_explained[:5])
    ))
    return pca, loadings, var_explained


def compute_feature_separation(df, features, out_path):
    """Cohen's d for each feature (ice vs water), saved to CSV."""
    labeled = df.dropna(subset=["label"])
    ice = labeled[labeled["label"] == "ice"]
    water = labeled[labeled["label"] == "water"]

    rows = []
    for feat in features:
        ice_vals = ice[feat].dropna()
        water_vals = water[feat].dropna()
        d = compute_cohens_d(ice_vals, water_vals)
        rows.append({
            "feature": feat,
            "cohens_d": d,
            "abs_d": abs(d) if not np.isnan(d) else np.nan,
            "ice_mean": ice_vals.mean() if len(ice_vals) > 0 else np.nan,
            "ice_std": ice_vals.std() if len(ice_vals) > 0 else np.nan,
            "water_mean": water_vals.mean() if len(water_vals) > 0 else np.nan,
            "water_std": water_vals.std() if len(water_vals) > 0 else np.nan,
            "ice_n": len(ice_vals),
            "water_n": len(water_vals),
        })

    result = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    result.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Saved {out_path}")
    print("\n  Feature separation (|Cohen's d|, ice vs water):")
    for _, r in result.iterrows():
        d_val = r["abs_d"]
        strength = "LARGE" if d_val > 0.8 else "MEDIUM" if d_val > 0.5 else "small"
        print(f"    {r['feature']:>8s}: d={r['cohens_d']:+.3f}  |d|={d_val:.3f}  ({strength})")
    return result


def plot_pairwise_scatter(df, out_path):
    """Key pairwise scatter plots: gamma vs phase, CLR vs phase, AF vs phase, CLR vs gamma."""
    labeled = df.dropna(subset=["label"])
    labeled = labeled[labeled["label"].isin(["ice", "water", "transition"])]

    pairs = [
        ("gamma", "phase"),
        ("CLR", "phase"),
        ("AF", "phase"),
        ("CLR", "gamma"),
        ("AF", "gamma"),
        ("CLR", "AF"),
    ]
    # Only plot pairs where both features exist
    pairs = [(a, b) for a, b in pairs if a in labeled.columns and b in labeled.columns]

    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}

    for idx, (fx, fy) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        for label in ["transition", "water", "ice"]:
            sub = labeled[labeled["label"] == label]
            sub_valid = sub[[fx, fy]].dropna()
            ax.scatter(sub_valid[fx], sub_valid[fy],
                       c=colors[label], label=label, alpha=0.2, s=5, edgecolors="none")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fx} vs {fy}")
        ax.legend(fontsize=8, markerscale=3)

    # Hide unused axes
    for idx in range(len(pairs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Pairwise Feature Scatter (ice / water / transition)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_phase_seasonal(df, out_path):
    """Phase vs DOY colored by label — does phase show seasonal structure?"""
    labeled = df.dropna(subset=["label"])
    labeled = labeled[labeled["label"].isin(["ice", "water", "transition"])]
    if "phase" not in labeled.columns or labeled.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}
    for label in ["transition", "water", "ice"]:
        sub = labeled[labeled["label"] == label]
        ax.scatter(sub["doy"], np.degrees(sub["phase"]),
                   c=colors[label], label=label, alpha=0.15, s=3, edgecolors="none")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("SNR Phase vs Season")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PCA feature analysis for ice detection")
    parser.add_argument("--station", required=True, help="Station name (e.g. ROSS)")
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--years", help="Year range, e.g. 2020-2024")
    parser.add_argument("--results_dir", default="results_annual")
    parser.add_argument("--output_dir", help="Output directory (default: results_annual/{station}/pca)")
    args = parser.parse_args()

    if args.years:
        start, end = map(int, args.years.split("-"))
        years = list(range(start, end + 1))
    elif args.year:
        years = [args.year]
    else:
        parser.error("Must specify --year or --years")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.results_dir) / args.station / "pca"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PCA Feature Analysis: {args.station} {years} ===")

    # Load data
    df = load_and_merge(args.station, years, args.results_dir)

    # Filter to arcs with valid features
    available = [f for f in FEATURES if f in df.columns]
    print(f"Available features: {available}")
    df_valid = df.dropna(subset=available)
    print(f"Arcs with all features valid: {len(df_valid):,} / {len(df):,}")

    # 1. Correlation matrix
    print("\n--- Correlation Matrix ---")
    corr = plot_correlation_matrix(df_valid, available, out_dir / "correlation_matrix.png")

    # Highlight phase correlations
    if "phase" in available:
        print("  Phase correlations with other features:")
        for feat in available:
            if feat != "phase":
                r = corr.loc["phase", feat]
                print(f"    phase ↔ {feat:>6s}: r={r:+.3f}  {'(redundant)' if abs(r) > 0.8 else '(independent)' if abs(r) < 0.3 else ''}")

    # 2. PCA biplot
    print("\n--- PCA Biplot ---")
    pca_result = plot_pca_biplot(df_valid, available, out_dir / "pca_biplot.png")

    # 3. Feature separation (Cohen's d)
    print("\n--- Feature Separation (Cohen's d) ---")
    sep = compute_feature_separation(df_valid, available, out_dir / "feature_separation.csv")

    # 4. Pairwise scatter
    print("\n--- Pairwise Scatter ---")
    plot_pairwise_scatter(df_valid, out_dir / "pairwise_scatter.png")

    # 5. Phase seasonal plot
    print("\n--- Phase Seasonal ---")
    plot_phase_seasonal(df_valid, out_dir / "phase_seasonal.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "phase" in available and sep is not None:
        phase_row = sep[sep["feature"] == "phase"]
        if not phase_row.empty:
            d = phase_row.iloc[0]["abs_d"]
            phase_corr_max = max(abs(corr.loc["phase", f]) for f in available if f != "phase")
            print(f"  Phase |Cohen's d|: {d:.3f}")
            print(f"  Max |correlation| with other features: {phase_corr_max:.3f}")
            if d > 0.5 and phase_corr_max < 0.5:
                print("  → RECOMMENDATION: Phase adds independent discriminatory power.")
                print("    Consider increasing weight from 0.5 to 1.0 in ice_classifier.")
            elif d > 0.5 and phase_corr_max >= 0.5:
                print("  → Phase is discriminatory but partially redundant with existing features.")
                print("    Keep current weight=0.5.")
            elif d <= 0.5:
                print("  → Phase has weak discriminatory power (d ≤ 0.5).")
                print("    Keep in parquet for research; current weight=0.5 is appropriate.")
    print(f"\nOutputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
