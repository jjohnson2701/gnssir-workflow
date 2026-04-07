#!/usr/bin/env python3
# ABOUTME: Land vs water azimuth feature analysis for classifier redesign
# ABOUTME: Compares edge/core azimuth features, tests interfreq_spread, prototypes Mahalanobis distance

"""
Land vs Water Azimuth Feature Analysis

Empirically characterizes what land-facing vs water-facing azimuths look like
in the arc_table feature space.  Informs the classifier redesign toward a
water-state Mahalanobis model.

Stations: ROSS, MCHN, GOD2, PSTA  (mixed land/water Great Lakes environments)
           UMNQ (Greenland fjord — confirmed land at 50-60°)

Steps:
  1. Classify azimuths as edge/core/sparse from gnssrefl config + arc density
  2. Compare feature distributions (edge vs core, summer only)
  3. Test interfreq_spread as primary ice discriminator vs GLERL
  4. Assess subdaily temporal resolution feasibility
  5. Prototype Mahalanobis distance from summer water-state baseline
  6. Print consolidated summary report

Usage:
    python scripts/analyze_land_water_azimuths.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_annual"
CONFIG_DIR = PROJECT_ROOT / "config"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
PLOT_DIR = ANALYSIS_DIR / "plots"

STATIONS = ["ROSS", "MCHN", "GOD2", "PSTA", "UMNQ"]

# Ice-free months from stations_config.json (loaded dynamically below)
# Summer = ice_free_months; Winter = complement

# Arc-level features to compare
ARC_FEATURES = [
    "Amp", "RH", "PkNoise", "CLR", "PR", "AF", "gamma",
    "MS", "VS", "phase", "NumbOf", "DelT",
]

# Daily features for Mahalanobis prototype
MAHAL_FEATURES = [
    "amp_mean", "rh_std", "clr_med", "af_med", "gamma_med", "pr_med",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stations_config():
    """Load master station config."""
    with open(CONFIG_DIR / "stations_config.json") as f:
        return json.load(f)


def load_gnssrefl_params(station: str) -> dict:
    """Load gnssrefl param JSON for a station."""
    path = CONFIG_DIR / f"{station.lower()}.json"
    if not path.exists():
        path = (
            PROJECT_ROOT
            / "gnssrefl_data_workspace"
            / "refl_code"
            / "input"
            / f"{station.lower()}.json"
        )
    with open(path) as f:
        return json.load(f)


def load_arc_tables(station: str, years=None) -> pd.DataFrame:
    """Concatenate arc_table parquets for a station across years."""
    if years is None:
        years = range(2003, 2026)
    frames = []
    for year in years:
        path = RESULTS_DIR / station / f"{station}_{year}_arc_table.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df


def load_daily_features(station: str, years=None) -> pd.DataFrame:
    """Concatenate daily_features parquets for a station."""
    if years is None:
        years = range(2003, 2026)
    frames = []
    for year in years:
        path = RESULTS_DIR / station / f"{station}_{year}_daily_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["year"] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df


def fetch_glerl(station: str, years) -> pd.DataFrame:
    """Fetch GLERL ice concentration for multiple years, return merged daily DF."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from external_apis.glerl_ice import GLERLIceClient
    except ImportError:
        log.warning("GLERLIceClient not importable — skipping GLERL")
        return pd.DataFrame()

    client = GLERLIceClient()
    frames = []
    for year in years:
        try:
            df = client.get_ice_concentration(station, year)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log.debug(f"GLERL {station} {year}: {e}")
    if not frames:
        return pd.DataFrame()
    glerl = pd.concat(frames, ignore_index=True)
    glerl["datetime"] = pd.to_datetime(glerl["datetime"], utc=True)
    glerl["date"] = glerl["datetime"].dt.tz_localize(None).dt.normalize()
    # Deduplicate by date (overlapping winter seasons)
    glerl = glerl.sort_values("date").drop_duplicates(subset="date", keep="last")
    return glerl


# ===================================================================
# STEP 1: Identify azimuth sectors
# ===================================================================

def _parse_suspect_azimuths(suspect_cfg: dict) -> list:
    """Parse suspect_azimuths config into list of azimuth bin values.

    Config format: {"50-60": "land_contamination", "notes": "..."}
    Returns list of ints, e.g. [50, 60] for the range "50-60".
    """
    land_bins = []
    for key, val in suspect_cfg.items():
        if key == "notes":
            continue
        if "-" in key:
            lo, hi = key.split("-")
            # Generate bins in 10° steps within the range
            for b in range(int(lo), int(hi) + 1, 10):
                land_bins.append(b)
    return sorted(set(land_bins))


def step1_classify_azimuths(stations_config: dict) -> dict:
    """Classify each station's azimuth bins as land, core, edge, or sparse.

    Uses suspect_azimuths from config for true land labeling where available,
    otherwise falls back to edge/core classification from arc density.
    """
    log.info("=" * 60)
    log.info("STEP 1: Azimuth classification")
    log.info("=" * 60)

    az_labels = {}

    for station in STATIONS:
        params = load_gnssrefl_params(station)
        az_lo, az_hi = params["azval2"]

        # Load all arc_tables to get density
        at = load_arc_tables(station)
        if at.empty:
            log.warning(f"{station}: no arc_table data")
            continue

        bins = sorted(at["azimuth_bin"].unique())

        # Summer-only density per bin
        ice_free = stations_config[station].get("ice_free_months", [6, 7, 8, 9])
        summer = at[at["month"].isin(ice_free)]
        summer_counts = summer.groupby("azimuth_bin").size()

        # Parse suspect_azimuths for true land labeling
        suspect_cfg = stations_config[station].get("suspect_azimuths", {})
        land_bins = _parse_suspect_azimuths(suspect_cfg) if suspect_cfg else []

        # Classify bins
        median_count = summer_counts.median() if len(summer_counts) > 0 else 0
        threshold = median_count * 0.20

        land, core, edge, sparse = [], [], [], []
        for b in bins:
            count = summer_counts.get(b, 0)
            if b in land_bins:
                land.append(b)
            elif count < threshold:
                sparse.append(b)
            elif b == bins[0] or b == bins[-1]:
                edge.append(b)
            else:
                core.append(b)

        # "water" = core + edge (non-land, non-sparse)
        water = core + edge

        az_labels[station] = {
            "land": land,
            "core": core,
            "edge": edge,
            "sparse": sparse,
            "water": water,
            "all": bins,
            "az_range": (az_lo, az_hi),
            "ice_free_months": ice_free,
        }

        log.info(
            f"  {station}: az_range={az_lo:.0f}-{az_hi:.0f}°  bins={bins}"
        )
        if land:
            log.info(f"    LAND (from suspect_azimuths): {land}")
        log.info(f"    CORE: {core}  EDGE: {edge}  SPARSE: {sparse}")

        # Print summer arc density per bin
        for b in bins:
            c = summer_counts.get(b, 0)
            if b in land:
                tag = "LAND"
            elif b in core:
                tag = "CORE"
            elif b in edge:
                tag = "EDGE"
            else:
                tag = "SPARSE"
            log.info(f"    az={b:3d}: {c:6d} summer arcs  [{tag}]")

    return az_labels


# ===================================================================
# STEP 2: Feature distributions (edge vs core, summer)
# ===================================================================

def step2_feature_distributions(az_labels: dict) -> dict:
    """Compare arc-level feature distributions: land vs water (or edge vs core), summer only.

    For stations with known land sectors (from suspect_azimuths), compares
    land vs water azimuths. For others, compares edge vs core water azimuths.
    """
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Feature distributions (land vs water / edge vs core, summer)")
    log.info("=" * 60)

    results = {}

    for station in STATIONS:
        if station not in az_labels:
            continue
        labels = az_labels[station]

        at = load_arc_tables(station)
        if at.empty:
            continue

        ice_free = labels["ice_free_months"]
        summer = at[at["month"].isin(ice_free)]

        # Decide comparison groups
        has_land = bool(labels["land"])
        if has_land:
            group_a_name, group_b_name = "water", "land"
            group_a_bins = labels["water"]
            group_b_bins = labels["land"]
        else:
            group_a_name, group_b_name = "core", "edge"
            group_a_bins = labels["core"]
            group_b_bins = labels["edge"]

        if not group_a_bins or not group_b_bins:
            log.info(f"  {station}: need both {group_a_name} and {group_b_name} bins — skipping")
            continue

        a_arcs = summer[summer["azimuth_bin"].isin(group_a_bins)]
        b_arcs = summer[summer["azimuth_bin"].isin(group_b_bins)]

        log.info(
            f"\n  {station}: {len(a_arcs)} {group_a_name} arcs, "
            f"{len(b_arcs)} {group_b_name} arcs  "
            f"(summer, {len(at)} total)"
        )

        station_results = {"comparison": f"{group_a_name}_vs_{group_b_name}"}
        for feat in ARC_FEATURES:
            if feat not in summer.columns:
                continue
            a = a_arcs[feat].dropna()
            b = b_arcs[feat].dropna()
            if len(a) < 30 or len(b) < 30:
                continue

            ks_stat, ks_p = stats.ks_2samp(a, b)
            pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
            cohens_d = abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

            station_results[feat] = {
                f"{group_a_name}_mean": a.mean(),
                f"{group_a_name}_std": a.std(),
                f"{group_b_name}_mean": b.mean(),
                f"{group_b_name}_std": b.std(),
                f"{group_a_name}_n": len(a),
                f"{group_b_name}_n": len(b),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "cohens_d": cohens_d,
            }

            log.info(
                f"    {feat:>10s}: {group_a_name}={a.mean():10.4f}±{a.std():.4f}  "
                f"{group_b_name}={b.mean():10.4f}±{b.std():.4f}  "
                f"d={cohens_d:.3f}  KS={ks_stat:.3f} p={ks_p:.2e}"
            )

        results[station] = station_results

    # --- Plot: top discriminative features across stations ---
    _plot_feature_discrimination(results)

    return results


def _plot_feature_discrimination(results: dict):
    """Bar chart of Cohen's d for each feature across stations."""
    if not results:
        return

    all_feats = set()
    for sr in results.values():
        all_feats.update(k for k in sr.keys() if k != "comparison")
    all_feats = sorted(all_feats)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_feats))
    width = 0.8 / max(len(results), 1)

    for i, (station, sr) in enumerate(results.items()):
        vals = [sr.get(f, {}).get("cohens_d", 0) for f in all_feats]
        ax.bar(x + i * width, vals, width, label=station, alpha=0.8)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(all_feats, rotation=45, ha="right")
    ax.set_ylabel("Cohen's d (edge vs core)")
    ax.set_title("Feature discrimination: edge vs core azimuths (summer)")
    ax.legend()
    ax.axhline(0.2, ls="--", color="gray", alpha=0.5, label="small effect")
    ax.axhline(0.5, ls="--", color="orange", alpha=0.5, label="medium effect")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_discrimination_edge_vs_core.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {PLOT_DIR / 'feature_discrimination_edge_vs_core.png'}")


# ===================================================================
# STEP 3: Interfreq spread as ice discriminator
# ===================================================================

def step3_interfreq_spread(az_labels: dict) -> dict:
    """Test interfreq_spread against GLERL as a binary ice detector."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3: Interfreq spread as primary ice discriminator")
    log.info("=" * 60)

    results = {}

    for station in STATIONS:
        if station not in az_labels:
            continue
        labels = az_labels[station]

        daily = load_daily_features(station)
        if daily.empty or "interfreq_spread" not in daily.columns:
            log.info(f"  {station}: no daily_features or no interfreq_spread")
            continue

        # Use pooled rows (azimuth_bin == -1) for station-level signal
        pooled = daily[daily["azimuth_bin"] == -1].copy()
        if pooled.empty:
            log.info(f"  {station}: no pooled (azimuth_bin=-1) rows")
            continue

        # Also compute per-azimuth interfreq spread from arc_table
        at = load_arc_tables(station)
        per_az_spread = {}
        if not at.empty:
            for az in labels["all"]:
                az_data = at[at["azimuth_bin"] == az]
                # Group by date, compute std of RH across freq_groups
                daily_rh_spread = (
                    az_data.groupby("date")
                    .apply(lambda g: g.groupby("freq_group")["RH"].median().std()
                           if g["freq_group"].nunique() >= 2 else np.nan)
                )
                per_az_spread[az] = daily_rh_spread.dropna().median()
            log.info(f"  {station} per-azimuth interfreq spread (median):")
            for az, val in sorted(per_az_spread.items()):
                tag = "CORE" if az in labels["core"] else (
                    "EDGE" if az in labels["edge"] else "SPARSE"
                )
                log.info(f"    az={az}: {val:.4f} m  [{tag}]")

        # Fetch GLERL for validation
        available_years = sorted(pooled["year"].unique())
        glerl = fetch_glerl(station, available_years)

        if glerl.empty:
            log.info(f"  {station}: no GLERL data — reporting distributions only")
            # Report summer vs winter interfreq_spread without GLERL
            ice_free = labels["ice_free_months"]
            summer_ifs = pooled[pooled["month"].isin(ice_free)]["interfreq_spread"].dropna()
            winter_ifs = pooled[~pooled["month"].isin(ice_free)]["interfreq_spread"].dropna()
            log.info(
                f"    Summer interfreq_spread: {summer_ifs.mean():.4f} ± {summer_ifs.std():.4f}  (n={len(summer_ifs)})"
            )
            log.info(
                f"    Winter interfreq_spread: {winter_ifs.mean():.4f} ± {winter_ifs.std():.4f}  (n={len(winter_ifs)})"
            )
            if len(summer_ifs) > 0 and len(winter_ifs) > 0:
                pooled_std = np.sqrt((summer_ifs.std()**2 + winter_ifs.std()**2) / 2)
                d = abs(summer_ifs.mean() - winter_ifs.mean()) / pooled_std if pooled_std > 0 else 0
                log.info(f"    Cohen's d (winter vs summer): {d:.3f}")
            results[station] = {
                "summer_mean": summer_ifs.mean(),
                "winter_mean": winter_ifs.mean(),
                "glerl_available": False,
                "per_az_spread": per_az_spread,
            }
            continue

        # Merge interfreq_spread with GLERL
        merged = pooled[["date", "interfreq_spread"]].dropna().merge(
            glerl[["date", "ice_concentration"]],
            on="date",
            how="inner",
        )

        if len(merged) < 20:
            log.info(f"  {station}: only {len(merged)} merged rows — insufficient")
            results[station] = {"glerl_available": False, "n_merged": len(merged)}
            continue

        log.info(f"  {station}: {len(merged)} days with both interfreq_spread and GLERL")

        # Pearson correlation
        r, p = stats.pearsonr(merged["interfreq_spread"], merged["ice_concentration"])
        log.info(f"    Pearson r = {r:.3f}, p = {p:.2e}")

        # Binary classification: ice = GLERL > 30%, water = GLERL < 5%
        ice_mask = merged["ice_concentration"] > 30
        water_mask = merged["ice_concentration"] < 5
        binary = merged[ice_mask | water_mask].copy()
        binary["is_ice"] = (binary["ice_concentration"] > 30).astype(int)

        auc, threshold_opt = np.nan, np.nan
        if binary["is_ice"].nunique() == 2 and len(binary) >= 20:
            auc = roc_auc_score(binary["is_ice"], binary["interfreq_spread"])
            fpr, tpr, thresholds = roc_curve(binary["is_ice"], binary["interfreq_spread"])
            youdens_j = tpr - fpr
            best_idx = np.argmax(youdens_j)
            threshold_opt = thresholds[best_idx]
            log.info(f"    ROC AUC = {auc:.3f}")
            log.info(f"    Optimal threshold (Youden's J) = {threshold_opt:.4f} m")
            log.info(
                f"    At threshold: TPR={tpr[best_idx]:.2f}, FPR={fpr[best_idx]:.2f}"
            )

            # Distribution stats
            ice_ifs = binary[binary["is_ice"] == 1]["interfreq_spread"]
            water_ifs = binary[binary["is_ice"] == 0]["interfreq_spread"]
            log.info(
                f"    Ice  interfreq_spread: {ice_ifs.mean():.4f} ± {ice_ifs.std():.4f}  (n={len(ice_ifs)})"
            )
            log.info(
                f"    Water interfreq_spread: {water_ifs.mean():.4f} ± {water_ifs.std():.4f}  (n={len(water_ifs)})"
            )
        else:
            log.info(f"    Binary classification not feasible (need both ice and water, got n={len(binary)})")

        results[station] = {
            "pearson_r": r,
            "pearson_p": p,
            "auc": auc,
            "threshold_opt": threshold_opt,
            "n_merged": len(merged),
            "glerl_available": True,
            "per_az_spread": per_az_spread,
        }

    # --- Plot: interfreq_spread time series with GLERL overlay ---
    _plot_interfreq_vs_glerl(az_labels)

    return results


def _plot_interfreq_vs_glerl(az_labels: dict):
    """Time series of interfreq_spread colored by GLERL ice concentration."""
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(14, 4 * len(STATIONS)), sharex=False)
    if len(STATIONS) == 1:
        axes = [axes]

    for ax, station in zip(axes, STATIONS):
        if station not in az_labels:
            ax.set_title(f"{station}: no data")
            continue

        daily = load_daily_features(station)
        if daily.empty or "interfreq_spread" not in daily.columns:
            ax.set_title(f"{station}: no interfreq_spread")
            continue

        pooled = daily[daily["azimuth_bin"] == -1].copy()
        if pooled.empty:
            continue

        pooled = pooled.dropna(subset=["interfreq_spread"])
        ax.scatter(
            pooled["date"], pooled["interfreq_spread"],
            s=3, alpha=0.4, c="steelblue", label="interfreq_spread",
        )

        # Overlay GLERL on twin axis
        available_years = sorted(pooled["year"].unique())
        glerl = fetch_glerl(station, available_years)
        if not glerl.empty:
            ax2 = ax.twinx()
            ax2.fill_between(
                glerl["date"], 0, glerl["ice_concentration"],
                alpha=0.15, color="red", label="GLERL ice %",
            )
            ax2.set_ylabel("GLERL ice %", color="red")
            ax2.set_ylim(0, 105)

        ax.set_ylabel("interfreq_spread (m)")
        ax.set_title(f"{station} — interfreq spread vs GLERL")
        ax.legend(loc="upper left", markerscale=3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "interfreq_spread_vs_glerl.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {PLOT_DIR / 'interfreq_spread_vs_glerl.png'}")


# ===================================================================
# STEP 4: Temporal resolution assessment
# ===================================================================

def step4_temporal_resolution(az_labels: dict) -> dict:
    """Check arc density for subdaily classification feasibility."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4: Temporal resolution assessment")
    log.info("=" * 60)

    results = {}

    for station in STATIONS:
        if station not in az_labels:
            continue

        # Use a single recent year for this assessment
        for test_year in [2024, 2023, 2022]:
            path = RESULTS_DIR / station / f"{station}_{test_year}_arc_table.parquet"
            if path.exists():
                break
        else:
            log.info(f"  {station}: no recent arc_table")
            continue

        at = pd.read_parquet(path)
        at["hour_block"] = (at["UTCtime"] // 6).astype(int).clip(0, 3)

        log.info(f"\n  {station} ({test_year}): {len(at)} total arcs")

        block_stats = {}
        for block in range(4):
            block_data = at[at["hour_block"] == block]
            subdaily = (
                block_data.groupby(["doy", "azimuth_bin"])
                .agg(n_arcs=("sat", "count"), n_sats=("sat", "nunique"))
                .reset_index()
            )
            pct_ge5 = (subdaily["n_arcs"] >= 5).mean() * 100 if len(subdaily) > 0 else 0
            mean_arcs = subdaily["n_arcs"].mean() if len(subdaily) > 0 else 0
            n_days = block_data["doy"].nunique()

            block_stats[block] = {
                "n_days": n_days,
                "mean_arcs_per_sector": mean_arcs,
                "pct_sectors_ge5": pct_ge5,
            }

            log.info(
                f"    Block {block} ({block*6:02d}-{(block+1)*6:02d} UTC): "
                f"{n_days} days, {mean_arcs:.1f} arcs/sector, "
                f"{pct_ge5:.0f}% sectors ≥5 arcs"
            )

        # Per-azimuth temporal coverage
        az_temporal = (
            at.groupby("azimuth_bin")
            .agg(
                total_arcs=("sat", "count"),
                n_days=("doy", "nunique"),
                hour_span=("UTCtime", lambda x: x.max() - x.min()),
            )
            .sort_values("total_arcs", ascending=False)
        )
        log.info(f"\n    Azimuth sectors ranked by total arcs:")
        for az, row in az_temporal.iterrows():
            log.info(
                f"      az={az:3d}: {row['total_arcs']:5.0f} arcs, "
                f"{row['n_days']:3.0f} days, {row['hour_span']:.1f}h span"
            )

        # Feasibility assessment
        mean_pct = np.mean([v["pct_sectors_ge5"] for v in block_stats.values()])
        if mean_pct > 60:
            feasibility = "FEASIBLE"
        elif mean_pct > 30:
            feasibility = "MARGINAL"
        else:
            feasibility = "INSUFFICIENT"

        results[station] = {
            "year": test_year,
            "total_arcs": len(at),
            "block_stats": block_stats,
            "feasibility": feasibility,
            "mean_pct_ge5": mean_pct,
        }
        log.info(f"    → Subdaily feasibility: {feasibility} ({mean_pct:.0f}% mean)")

    return results


# ===================================================================
# STEP 5: Mahalanobis distance prototype
# ===================================================================

def step5_mahalanobis(az_labels: dict) -> dict:
    """Compute Mahalanobis distance from summer water-state baseline."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5: Mahalanobis distance prototype")
    log.info("=" * 60)

    results = {}

    for station in STATIONS:
        if station not in az_labels:
            continue
        labels = az_labels[station]
        ice_free = labels["ice_free_months"]

        daily = load_daily_features(station)
        if daily.empty:
            log.info(f"  {station}: no daily_features")
            continue

        # Use water-facing sectors (core + edge, excludes land and sparse)
        water_bins = labels["water"]
        water_sectors = daily[daily["azimuth_bin"].isin(water_bins)].copy()

        if water_sectors.empty:
            log.info(f"  {station}: no water sector data")
            continue

        # Check which features are available and have enough non-NaN
        available_feats = [f for f in MAHAL_FEATURES if f in water_sectors.columns]
        summer = water_sectors[water_sectors["month"].isin(ice_free)]
        summer_feats = summer[available_feats].dropna()

        min_rows = len(available_feats) * 10
        if len(summer_feats) < min_rows:
            log.info(
                f"  {station}: only {len(summer_feats)} clean summer rows "
                f"(need {min_rows}) — reducing feature set"
            )
            # Try with fewer features
            available_feats = [f for f in available_feats
                               if summer[f].notna().sum() > min_rows]
            summer_feats = summer[available_feats].dropna()
            if len(summer_feats) < len(available_feats) * 5:
                log.info(f"  {station}: still insufficient — skipping")
                continue

        log.info(
            f"\n  {station}: {len(summer_feats)} summer rows, "
            f"features={available_feats}"
        )

        mean_water = summer_feats.mean().values
        cov_water = summer_feats.cov().values

        # Condition number check
        cond = np.linalg.cond(cov_water)
        log.info(f"    Covariance condition number: {cond:.1f}")
        if cond > 1e8:
            log.info("    → Near-singular — adding regularization")
            cov_water += np.eye(len(available_feats)) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov_water)
        except np.linalg.LinAlgError:
            log.warning(f"  {station}: covariance singular — skipping")
            continue

        # Compute Mahalanobis distance for all sector-days
        all_data = water_sectors[available_feats + ["date", "azimuth_bin", "year", "month"]].dropna(
            subset=available_feats
        )

        feat_matrix = all_data[available_feats].values
        diffs = feat_matrix - mean_water
        # Vectorized Mahalanobis: sqrt(diag(diffs @ cov_inv @ diffs.T))
        left = diffs @ cov_inv
        mahal_dists = np.sqrt(np.sum(left * diffs, axis=1))

        all_data = all_data.copy()
        all_data["mahal_dist"] = mahal_dists

        # Monthly summary
        monthly = all_data.groupby("month")["mahal_dist"].agg(["mean", "median", "std", "count"])
        log.info(f"\n    Mahalanobis distance by month:")
        for m, row in monthly.iterrows():
            marker = " ← ice-free" if m in ice_free else ""
            log.info(
                f"      Month {m:2d}: mean={row['mean']:6.2f}  "
                f"median={row['median']:6.2f}  std={row['std']:5.2f}  "
                f"n={row['count']:5.0f}{marker}"
            )

        # Summer vs winter separation
        summer_d = all_data[all_data["month"].isin(ice_free)]["mahal_dist"]
        winter_d = all_data[~all_data["month"].isin(ice_free)]["mahal_dist"]

        if len(summer_d) > 0 and len(winter_d) > 0:
            sep = (winter_d.mean() - summer_d.mean()) / np.sqrt(
                (summer_d.std() ** 2 + winter_d.std() ** 2) / 2
            )
            log.info(
                f"\n    Summer Mahalanobis: {summer_d.mean():.2f} ± {summer_d.std():.2f}"
            )
            log.info(
                f"    Winter Mahalanobis: {winter_d.mean():.2f} ± {winter_d.std():.2f}"
            )
            log.info(f"    Separation (Cohen's d): {sep:.3f}")
        else:
            sep = np.nan

        # GLERL validation if available
        auc_mahal = np.nan
        available_years = sorted(all_data["year"].unique())
        glerl = fetch_glerl(station, available_years)
        if not glerl.empty:
            daily_mahal = all_data.groupby("date")["mahal_dist"].mean().reset_index()
            merged = daily_mahal.merge(glerl[["date", "ice_concentration"]], on="date")
            if len(merged) > 20:
                r_mahal, _ = stats.pearsonr(merged["mahal_dist"], merged["ice_concentration"])
                log.info(f"    Mahalanobis vs GLERL Pearson r = {r_mahal:.3f}")

                # Binary AUC
                ice_mask = merged["ice_concentration"] > 30
                water_mask = merged["ice_concentration"] < 5
                binary = merged[ice_mask | water_mask].copy()
                binary["is_ice"] = (binary["ice_concentration"] > 30).astype(int)
                if binary["is_ice"].nunique() == 2 and len(binary) >= 20:
                    auc_mahal = roc_auc_score(binary["is_ice"], binary["mahal_dist"])
                    log.info(f"    Mahalanobis ROC AUC = {auc_mahal:.3f}")

        # Save distances
        out_path = RESULTS_DIR / station / f"{station}_mahalanobis_distances.parquet"
        all_data[["date", "azimuth_bin", "year", "month", "mahal_dist"]].to_parquet(
            out_path, index=False
        )
        log.info(f"    Saved: {out_path}")

        results[station] = {
            "features_used": available_feats,
            "cov_condition": cond,
            "summer_mean": summer_d.mean() if len(summer_d) > 0 else np.nan,
            "winter_mean": winter_d.mean() if len(winter_d) > 0 else np.nan,
            "separation": sep,
            "auc_vs_glerl": auc_mahal,
            "n_summer": len(summer_feats),
            "n_total": len(all_data),
        }

    # --- Plot: Mahalanobis time series ---
    _plot_mahalanobis(az_labels, results)

    return results


def _plot_mahalanobis(az_labels: dict, mahal_results: dict):
    """Monthly boxplots of Mahalanobis distance per station."""
    stations_with_data = [s for s in STATIONS if s in mahal_results]
    if not stations_with_data:
        return

    fig, axes = plt.subplots(
        len(stations_with_data), 1,
        figsize=(12, 4 * len(stations_with_data)),
        sharex=True,
    )
    if len(stations_with_data) == 1:
        axes = [axes]

    for ax, station in zip(axes, stations_with_data):
        out_path = RESULTS_DIR / station / f"{station}_mahalanobis_distances.parquet"
        if not out_path.exists():
            continue
        df = pd.read_parquet(out_path)
        ice_free = az_labels[station]["ice_free_months"]

        # Boxplot by month
        months = sorted(df["month"].unique())
        data_by_month = [df[df["month"] == m]["mahal_dist"].values for m in months]
        bp = ax.boxplot(
            data_by_month, positions=months, widths=0.6,
            patch_artist=True, showfliers=False,
        )
        for i, m in enumerate(months):
            color = "#4ECDC4" if m in ice_free else "#FF6B6B"
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_alpha(0.6)

        ax.set_ylabel("Mahalanobis distance")
        ax.set_title(
            f"{station} — Mahalanobis from summer water baseline  "
            f"(sep={mahal_results[station]['separation']:.2f})"
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mahalanobis_monthly_boxplots.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {PLOT_DIR / 'mahalanobis_monthly_boxplots.png'}")


# ===================================================================
# STEP 6: Summary report
# ===================================================================

def step6_summary(
    az_labels: dict,
    feat_results: dict,
    interfreq_results: dict,
    temporal_results: dict,
    mahal_results: dict,
):
    """Print and save consolidated summary."""
    log.info("")
    log.info("=" * 60)
    log.info("STEP 6: Summary report")
    log.info("=" * 60)

    lines = []
    lines.append("=" * 70)
    lines.append("LAND vs WATER AZIMUTH ANALYSIS — SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # 1. Azimuth classification
    lines.append("1. AZIMUTH CLASSIFICATION")
    lines.append("-" * 40)
    for station in STATIONS:
        if station not in az_labels:
            continue
        labels = az_labels[station]
        land_str = f"  land={labels['land']}" if labels["land"] else ""
        lines.append(
            f"  {station}: range={labels['az_range'][0]:.0f}-{labels['az_range'][1]:.0f}°  "
            f"core={labels['core']}  edge={labels['edge']}  "
            f"sparse={labels['sparse']}{land_str}"
        )
    lines.append("")

    # 2. Feature discrimination
    lines.append("2. FEATURE DISCRIMINATION (Cohen's d, summer)")
    lines.append("-" * 40)
    for station, sr in feat_results.items():
        comp = sr.get("comparison", "edge_vs_core")
        lines.append(f"  {station} ({comp}):")
    lines.append("")
    # Collect all features and rank by mean Cohen's d
    feat_scores = {}
    for station, sr in feat_results.items():
        for feat, vals in sr.items():
            if feat == "comparison":
                continue
            feat_scores.setdefault(feat, []).append(vals["cohens_d"])
    feat_ranked = sorted(feat_scores.items(), key=lambda x: -np.mean(x[1]))
    lines.append(f"  {'Feature':>12s}  {'Mean d':>8s}  " + "  ".join(f"{s:>8s}" for s in feat_results))
    for feat, scores in feat_ranked:
        per_station = []
        for station in feat_results:
            d = feat_results[station].get(feat, {}).get("cohens_d", float("nan"))
            per_station.append(f"{d:8.3f}")
        lines.append(f"  {feat:>12s}  {np.mean(scores):8.3f}  " + "  ".join(per_station))
    lines.append("")

    # 3. Interfreq spread
    lines.append("3. INTERFREQ SPREAD AS ICE DETECTOR")
    lines.append("-" * 40)
    for station, ir in interfreq_results.items():
        if ir.get("glerl_available"):
            lines.append(
                f"  {station}: Pearson r={ir['pearson_r']:.3f}  "
                f"AUC={ir['auc']:.3f}  threshold={ir['threshold_opt']:.4f} m  "
                f"(n={ir['n_merged']})"
            )
        else:
            if "summer_mean" in ir:
                lines.append(
                    f"  {station}: summer={ir['summer_mean']:.4f}  "
                    f"winter={ir['winter_mean']:.4f}  (no GLERL)"
                )
            else:
                lines.append(f"  {station}: insufficient data")
    lines.append("")

    # 4. Subdaily feasibility
    lines.append("4. SUBDAILY FEASIBILITY")
    lines.append("-" * 40)
    for station, tr in temporal_results.items():
        lines.append(
            f"  {station}: {tr['feasibility']} — "
            f"{tr['total_arcs']} arcs/year, "
            f"{tr['mean_pct_ge5']:.0f}% 6hr blocks ≥5 arcs/sector"
        )
    lines.append("")

    # 5. Mahalanobis distance
    lines.append("5. MAHALANOBIS DISTANCE")
    lines.append("-" * 40)
    for station, mr in mahal_results.items():
        glerl_line = ""
        if not np.isnan(mr["auc_vs_glerl"]):
            glerl_line = f"  GLERL AUC={mr['auc_vs_glerl']:.3f}"
        lines.append(
            f"  {station}: summer={mr['summer_mean']:.2f}  "
            f"winter={mr['winter_mean']:.2f}  "
            f"separation={mr['separation']:.3f}  "
            f"cond={mr['cov_condition']:.0f}  "
            f"features={mr['features_used']}"
            f"{glerl_line}"
        )
    lines.append("")

    # 6. Key findings
    lines.append("6. KEY FINDINGS")
    lines.append("-" * 40)

    # Best discriminating features
    if feat_ranked:
        top3 = [f for f, _ in feat_ranked[:3]]
        # Check if any station had true land comparison
        has_land = any(sr.get("comparison") == "water_vs_land" for sr in feat_results.values())
        land_stations = [s for s, sr in feat_results.items() if sr.get("comparison") == "water_vs_land"]
        if has_land:
            lines.append(f"  - Stations with true land sectors: {', '.join(land_stations)}")
        lines.append(f"  - Top discriminating features (land/edge vs water/core): {', '.join(top3)}")

    # Interfreq spread verdict
    aucs = [ir["auc"] for ir in interfreq_results.values()
            if ir.get("glerl_available") and not np.isnan(ir.get("auc", np.nan))]
    if aucs:
        mean_auc = np.mean(aucs)
        verdict = "strong" if mean_auc > 0.85 else ("moderate" if mean_auc > 0.7 else "weak")
        lines.append(
            f"  - Interfreq spread as ice detector: {verdict} "
            f"(mean AUC={mean_auc:.3f} across {len(aucs)} stations)"
        )

    # Mahalanobis verdict
    seps = [mr["separation"] for mr in mahal_results.values()
            if not np.isnan(mr.get("separation", np.nan))]
    if seps:
        mean_sep = np.mean(seps)
        verdict = "clear" if mean_sep > 1.0 else ("moderate" if mean_sep > 0.5 else "weak")
        lines.append(
            f"  - Mahalanobis ice/water separation: {verdict} "
            f"(mean Cohen's d={mean_sep:.3f})"
        )

    # Mahalanobis vs single-feature comparison
    mahal_aucs = [mr["auc_vs_glerl"] for mr in mahal_results.values()
                  if not np.isnan(mr.get("auc_vs_glerl", np.nan))]
    if mahal_aucs and aucs:
        lines.append(
            f"  - Mahalanobis AUC vs GLERL: {np.mean(mahal_aucs):.3f}  "
            f"vs interfreq_spread AUC: {np.mean(aucs):.3f}"
        )

    # Subdaily verdict
    feasible = [s for s, tr in temporal_results.items() if tr["feasibility"] == "FEASIBLE"]
    marginal = [s for s, tr in temporal_results.items() if tr["feasibility"] == "MARGINAL"]
    if feasible:
        lines.append(f"  - Subdaily feasible: {', '.join(feasible)}")
    if marginal:
        lines.append(f"  - Subdaily marginal: {', '.join(marginal)}")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    report_path = ANALYSIS_DIR / "land_water_azimuth_report.txt"
    report_path.write_text(report)
    log.info(f"\nReport saved: {report_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    stations_config = load_stations_config()

    az_labels = step1_classify_azimuths(stations_config)
    feat_results = step2_feature_distributions(az_labels)
    interfreq_results = step3_interfreq_spread(az_labels)
    temporal_results = step4_temporal_resolution(az_labels)
    mahal_results = step5_mahalanobis(az_labels)

    step6_summary(
        az_labels, feat_results, interfreq_results,
        temporal_results, mahal_results,
    )


if __name__ == "__main__":
    main()
