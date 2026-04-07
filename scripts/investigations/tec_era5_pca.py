#!/usr/bin/env python3
"""Combined PCA: SNR features + TEC + ERA5 context variables.

Determines whether interfrequency RH divergence is ionosphere-driven (TEC)
or surface-driven (ice), and whether ERA5 meteorology predicts SNR changes.

Usage:
    python scripts/tec_era5_pca.py --station ROSS --year 2024
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_all_data(station, year):
    """Load and merge all data sources on DOY."""
    results = PROJECT_ROOT / "results_annual" / station

    # SNR features (daily medians)
    sf_path = results / f"{station}_{year}_snr_features.parquet"
    if not sf_path.exists():
        logger.error(f"No SNR features: {sf_path}")
        return None
    sf = pd.read_parquet(sf_path)
    daily_snr = sf.groupby("doy")[["CLR", "AF", "PR", "gamma", "phase", "VS", "RH"]].median().reset_index()

    # Interfrequency divergence
    interfreq_path = results / f"{station}_{year}_combined_interfreq.parquet"
    if interfreq_path.exists():
        interfreq = pd.read_parquet(interfreq_path)
        # Daily median across azimuth bins
        daily_interfreq = interfreq.groupby(
            interfreq["date"].apply(lambda d: pd.Timestamp(d).timetuple().tm_yday)
        ).agg(
            rh_diff_median=("L1_L2C_rh_diff", "median"),
            rh_diff_std=("L1_L2C_rh_diff", "std"),
            amp_diff_median=("L1_L2C_amp_diff", "median"),
        ).reset_index().rename(columns={"date": "doy"})
    else:
        daily_interfreq = None

    # TEC
    tec_path = results / f"{station}_{year}_tec.parquet"
    if tec_path.exists():
        tec = pd.read_parquet(tec_path)
    else:
        tec = None

    # ERA5
    era5_path = results / f"{station}_{year}_era5.parquet"
    if era5_path.exists():
        era5 = pd.read_parquet(era5_path)
    else:
        era5 = None

    # Ice classification
    ice_path = results / f"{station}_{year}_ice_classification.parquet"
    if ice_path.exists():
        ice = pd.read_parquet(ice_path)
        ice["date_dt"] = pd.to_datetime(ice["date"])
        ice["doy"] = ice["date_dt"].dt.dayofyear
        daily_ice = ice[["doy", "classification"]].copy()
        if "ice_score" in ice.columns:
            daily_ice["ice_score"] = ice["ice_score"]
    else:
        daily_ice = None

    # Merge everything on DOY
    merged = daily_snr.copy()

    if daily_interfreq is not None:
        merged = merged.merge(daily_interfreq, on="doy", how="left")
        logger.info(f"Merged interfreq: {len(daily_interfreq)} days")

    if tec is not None:
        merged = merged.merge(tec, on="doy", how="left")
        logger.info(f"Merged TEC: {len(tec)} days")

    if era5 is not None:
        merged = merged.merge(era5, on="doy", how="left")
        logger.info(f"Merged ERA5: {len(era5)} days")

    if daily_ice is not None:
        merged = merged.merge(daily_ice, on="doy", how="left")
        logger.info(f"Merged ice classification: {len(daily_ice)} days")

    logger.info(f"Final merged: {len(merged)} days, {len(merged.columns)} columns")
    return merged


def analyze_tec_vs_interfreq(merged, station, year, out_dir):
    """Key question: does TEC explain interfrequency RH divergence?"""
    tec_cols = [c for c in merged.columns if c.startswith("tec_")]
    if not tec_cols or "rh_diff_median" not in merged.columns:
        logger.info("Skipping TEC vs interfreq (missing data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Time series: TEC and interfreq divergence
    ax = axes[0, 0]
    valid = merged[["doy", "tec_daily_mean", "rh_diff_median"]].dropna()
    ax.plot(valid["doy"], valid["tec_daily_mean"], "b-", lw=0.8, label="TEC (TECU)")
    ax.set_ylabel("TEC (TECU)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax_r = ax.twinx()
    ax_r.plot(valid["doy"], valid["rh_diff_median"] * 100, "r-", lw=0.8,
              label="L1-L2C ΔRH (cm)")
    ax_r.set_ylabel("L1-L2C ΔRH (cm)", color="red")
    ax_r.tick_params(axis="y", labelcolor="red")
    ax.set_xlabel("DOY")
    ax.set_title("TEC vs Interfrequency RH Divergence")

    # 2. Scatter: TEC vs interfreq divergence
    ax = axes[0, 1]
    valid = merged[["tec_daily_mean", "rh_diff_median", "classification"]].dropna()
    cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}
    for cls in ["transition", "water", "ice"]:
        mask = valid["classification"] == cls
        if mask.sum() > 0:
            ax.scatter(valid.loc[mask, "tec_daily_mean"],
                       valid.loc[mask, "rh_diff_median"] * 100,
                       c=cls_colors.get(cls, "gray"), label=cls, s=15, alpha=0.5)
    if len(valid) > 10:
        r = valid["tec_daily_mean"].corr(valid["rh_diff_median"])
        ax.set_title(f"TEC vs ΔRH: r={r:.3f}")
    ax.set_xlabel("TEC daily mean (TECU)")
    ax.set_ylabel("L1-L2C ΔRH (cm)")
    ax.legend(fontsize=8)

    # 3. TEC rate vs interfreq std
    ax = axes[1, 0]
    valid = merged[["tec_rate_max", "rh_diff_std"]].dropna()
    if len(valid) > 10:
        ax.scatter(valid["tec_rate_max"], valid["rh_diff_std"] * 100,
                   s=10, alpha=0.5, c="gray")
        r = valid["tec_rate_max"].corr(valid["rh_diff_std"])
        ax.set_title(f"TEC rate vs ΔRH variability: r={r:.3f}")
    ax.set_xlabel("Max TEC rate (TECU/hr)")
    ax.set_ylabel("ΔRH std across azimuths (cm)")

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis("off")
    valid = merged[["tec_daily_mean", "rh_diff_median"]].dropna()
    r_tec_drh = valid["tec_daily_mean"].corr(valid["rh_diff_median"]) if len(valid) > 10 else np.nan
    summary = (
        f"TEC vs ΔRH correlation: r={r_tec_drh:.3f}\n\n"
        f"Interpretation:\n"
        f"  |r| > 0.5: Ionosphere confounds interfreq signal\n"
        f"            → TEC correction needed\n"
        f"  |r| < 0.3: Interfreq divergence is surface-driven\n"
        f"            → Current ice indicator is valid\n\n"
        f"Result: {'IONOSPHERE CONFOUND' if abs(r_tec_drh) > 0.5 else 'SURFACE-DRIVEN (valid)' if abs(r_tec_drh) < 0.3 else 'INCONCLUSIVE'}"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(facecolor="lightyellow", edgecolor="gray", pad=10))

    fig.suptitle(f"{station} {year}: TEC vs Interfrequency RH Divergence", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_tec_vs_interfreq.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def analyze_era5_vs_features(merged, station, year, out_dir):
    """How do ERA5 meteorological variables relate to SNR features?"""
    era5_cols = ["t2m_mean", "t2m_min", "wind_speed_ms", "precip_mm",
                 "snow_depth_m", "soil_moisture"]
    era5_cols = [c for c in era5_cols if c in merged.columns]
    snr_cols = ["CLR", "AF", "gamma", "VS", "RH"]
    snr_cols = [c for c in snr_cols if c in merged.columns]

    if not era5_cols or not snr_cols:
        logger.info("Skipping ERA5 vs features (missing data)")
        return

    cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}

    nrows = len(era5_cols)
    ncols = len(snr_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, ec in enumerate(era5_cols):
        for ci, sc in enumerate(snr_cols):
            ax = axes[ri, ci]
            valid = merged[[ec, sc, "classification"]].dropna()
            if len(valid) < 10:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
                continue

            for cls in ["transition", "water", "ice"]:
                mask = valid["classification"] == cls
                if mask.sum() > 0:
                    ax.scatter(valid.loc[mask, ec], valid.loc[mask, sc],
                               c=cls_colors.get(cls, "gray"), s=8, alpha=0.4)

            r = valid[ec].corr(valid[sc])
            ax.set_title(f"r={r:+.2f}", fontsize=9,
                         fontweight="bold" if abs(r) > 0.3 else "normal",
                         color="darkred" if abs(r) > 0.5 else "black")
            if ri == nrows - 1:
                ax.set_xlabel(sc, fontsize=9)
            if ci == 0:
                ax.set_ylabel(ec, fontsize=9)

    fig.suptitle(f"{station} {year}: ERA5 Meteorology vs GNSS-IR Features",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_era5_vs_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def run_combined_pca(merged, station, year, out_dir):
    """PCA with all available features: SNR + TEC + ERA5."""
    # Select numeric columns for PCA
    snr_features = ["CLR", "AF", "PR", "gamma", "phase", "VS"]
    tec_features = ["tec_daily_mean", "tec_rate_max"]
    era5_features = ["t2m_mean", "wind_speed_ms", "snow_depth_m"]
    interfreq_features = ["rh_diff_median"]

    all_features = []
    for group in [snr_features, tec_features, era5_features, interfreq_features]:
        all_features.extend([f for f in group if f in merged.columns])

    if len(all_features) < 4:
        logger.info(f"Only {len(all_features)} features available, skipping PCA")
        return

    valid = merged[all_features + ["doy"]].dropna()
    if "classification" in merged.columns:
        valid = valid.join(merged[["classification"]], how="left")

    if len(valid) < 20:
        logger.info(f"Only {len(valid)} valid rows, skipping PCA")
        return

    logger.info(f"PCA with {len(all_features)} features, {len(valid)} days")

    X = valid[all_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(all_features), X_scaled.shape[1]))
    scores = pca.fit_transform(X_scaled)

    # --- Biplot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # Color by classification if available
    if "classification" in valid.columns:
        cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}
        for cls in ["transition", "water", "ice"]:
            mask = valid["classification"] == cls
            if mask.sum() > 0:
                ax.scatter(scores[mask.values, 0], scores[mask.values, 1],
                           c=cls_colors[cls], label=cls, s=20, alpha=0.5)
        ax.legend(fontsize=9)
    else:
        ax.scatter(scores[:, 0], scores[:, 1], c=valid["doy"], cmap="Spectral",
                   s=20, alpha=0.5)

    # Loading arrows with color coding by source
    loadings = pca.components_[:2, :].T
    scale = np.abs(scores[:, :2]).max() * 0.8

    source_colors = {}
    for f in all_features:
        if f in snr_features:
            source_colors[f] = "black"
        elif f in tec_features:
            source_colors[f] = "red"
        elif f in era5_features:
            source_colors[f] = "blue"
        elif f in interfreq_features:
            source_colors[f] = "purple"

    for i, feat in enumerate(all_features):
        color = source_colors.get(feat, "gray")
        ax.annotate(
            feat, xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            fontsize=10, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            xytext=(0, 0), textcoords="data",
        )

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
    ax.set_title(f"{station} {year}: Combined PCA\n"
                 f"Black=SNR, Red=TEC, Blue=ERA5, Purple=Interfreq")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)

    fig.tight_layout()
    path = out_dir / f"{station}_{year}_combined_pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")

    # Print variance explained
    print(f"\nVariance explained: " + ", ".join(
        f"PC{i+1}={v:.1%}" for i, v in enumerate(var_exp[:5])))

    # --- Correlation matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = valid[all_features].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(all_features)))
    ax.set_yticks(range(len(all_features)))
    ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_features, fontsize=9)
    for i in range(len(all_features)):
        for j in range(len(all_features)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"{station} {year}: Full Correlation Matrix\n"
                 f"SNR + TEC + ERA5 + Interfrequency")
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_combined_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Combined PCA: SNR + TEC + ERA5")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    out_dir = PROJECT_ROOT / "results_annual" / args.station / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_all_data(args.station, args.year)
    if merged is None:
        return

    print(f"\n{'=' * 60}")
    print(f"Combined PCA: {args.station} {args.year}")
    print(f"{'=' * 60}")
    print(f"Total days: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")

    # Analysis 1: TEC vs interfrequency
    analyze_tec_vs_interfreq(merged, args.station, args.year, out_dir)

    # Analysis 2: ERA5 vs SNR features
    analyze_era5_vs_features(merged, args.station, args.year, out_dir)

    # Analysis 3: Combined PCA
    run_combined_pca(merged, args.station, args.year, out_dir)

    print(f"\nOutputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
