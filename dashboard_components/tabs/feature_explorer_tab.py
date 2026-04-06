# ABOUTME: Feature explorer tab — interactive exploration of daily_features.parquet columns
# ABOUTME: Shows feature time series, correlations, per-sector comparison, multi-station overlay

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dashboard_components.data_loader import (
    load_daily_features,
    load_ice_classification,
    load_available_stations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_COLOR = {"ice": "#1565c0", "transition": "#f9a825", "water": "#2e7d32"}

# Feature columns to exclude from selection (keys / metadata, not plottable)
_META_COLS = {"date", "azimuth_bin"}

# Friendly labels for common features
_FEATURE_LABELS = {
    "clr_z": "CLR z-score",
    "af_z": "AF z-score",
    "pr_z": "PR z-score",
    "gamma_z": "Gamma z-score",
    "clr_med": "CLR (raw median)",
    "af_med": "AF (raw median)",
    "pr_med": "PR (raw median)",
    "gamma_med": "Gamma (raw median)",
    "amp_mean": "Amplitude mean",
    "amp_cv": "Amplitude CV",
    "rh_std": "RH std dev (m)",
    "rh_mean": "RH mean (m)",
    "delta_rh_mean": "ΔRH mean (m)",
    "delta_rh_std": "ΔRH std (m)",
    "phase_circ_mean": "Phase circ. mean (rad)",
    "phase_circ_std": "Phase circ. std (rad)",
    "diff_phase_L1_L2": "Diff. phase L1-L2 (rad)",
    "interfreq_spread": "Interfreq. spread (m)",
    "n_arcs": "Arc count",
    "n_sats": "Satellite count",
    "frac_full_arc": "Full-arc fraction",
    "p2n_mean": "Peak-to-noise mean",
    "wse_mean": "WSE mean (m)",
}


def _label(col):
    return _FEATURE_LABELS.get(col, col)


def _get_feature_cols(df):
    """Return plottable numeric feature columns from daily_features."""
    return sorted(
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    )


def _render_feature_timeseries(daily_features, clf, station, year):
    """Feature time series colored by classification."""
    feature_cols = _get_feature_cols(daily_features)
    if not feature_cols:
        st.warning("No numeric feature columns found.")
        return

    # Default to a useful feature if available
    default_idx = 0
    for pref in ["clr_z", "gamma_z", "delta_rh_mean", "amp_mean"]:
        if pref in feature_cols:
            default_idx = feature_cols.index(pref)
            break

    selected = st.selectbox(
        "Feature",
        feature_cols,
        index=default_idx,
        format_func=_label,
        key=f"feat_ts_{station}_{year}",
    )

    sector_choice = st.radio(
        "Sectors",
        ["Pooled (station-level)", "Per-sector overlay"],
        horizontal=True,
        key=f"feat_sector_{station}_{year}",
    )

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    if sector_choice == "Pooled (station-level)":
        pooled = daily_features[daily_features["azimuth_bin"] == -1].copy()
        if pooled.empty:
            st.info("No pooled rows (azimuth_bin=-1) in daily_features.")
            plt.close(fig)
            return
        pooled["date_dt"] = pd.to_datetime(pooled["date"])

        if clf is not None:
            merged = pooled.merge(
                clf[["date", "classification"]].drop_duplicates("date"),
                on="date", how="left",
            )
            for cls, color in CLASS_COLOR.items():
                mask = merged["classification"] == cls
                if mask.sum() > 0:
                    ax.scatter(merged.loc[mask, "date_dt"], merged.loc[mask, selected],
                               c=color, s=10, alpha=0.7, label=cls.capitalize())
            unclassed = merged["classification"].isna()
            if unclassed.sum() > 0:
                ax.scatter(merged.loc[unclassed, "date_dt"], merged.loc[unclassed, selected],
                           c="#888888", s=8, alpha=0.5, label="Unclassified")
        else:
            ax.scatter(pooled["date_dt"], pooled[selected], c="#333", s=10, alpha=0.6)

        ax.legend(fontsize=8, loc="upper right")
    else:
        sectors = daily_features[daily_features["azimuth_bin"] >= 0]
        if sectors.empty:
            st.info("No per-sector rows in daily_features.")
            plt.close(fig)
            return
        sectors = sectors.copy()
        sectors["date_dt"] = pd.to_datetime(sectors["date"])
        cmap = plt.cm.tab10
        az_bins = sorted(sectors["azimuth_bin"].unique())
        for i, az in enumerate(az_bins):
            sec = sectors[sectors["azimuth_bin"] == az]
            ax.scatter(sec["date_dt"], sec[selected], s=6, alpha=0.5,
                       color=cmap(i % 10), label=f"Az {az}")
        ax.legend(fontsize=7, loc="upper right", ncol=min(len(az_bins), 4))

    ax.set_ylabel(_label(selected))
    ax.set_title(f"{station} {year} — {_label(selected)}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_feature_correlation(daily_features, clf, station, year):
    """Scatter plot of two features, colored by classification or month."""
    feature_cols = _get_feature_cols(daily_features)
    if len(feature_cols) < 2:
        st.info("Need at least 2 features for correlation plot.")
        return

    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("X-axis", feature_cols, index=0,
                               format_func=_label, key=f"corr_x_{station}_{year}")
    with col2:
        default_y = min(1, len(feature_cols) - 1)
        feat_y = st.selectbox("Y-axis", feature_cols, index=default_y,
                               format_func=_label, key=f"corr_y_{station}_{year}")

    color_by = st.radio("Color by", ["Classification", "Month"], horizontal=True,
                        key=f"corr_color_{station}_{year}")

    pooled = daily_features[daily_features["azimuth_bin"] == -1].copy()
    if pooled.empty:
        pooled = daily_features.copy()
    pooled["date_dt"] = pd.to_datetime(pooled["date"])
    pooled["month"] = pooled["date_dt"].dt.month

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    valid = pooled.dropna(subset=[feat_x, feat_y])

    if color_by == "Classification" and clf is not None:
        merged = valid.merge(
            clf[["date", "classification"]].drop_duplicates("date"),
            on="date", how="left",
        )
        for cls, color in CLASS_COLOR.items():
            mask = merged["classification"] == cls
            if mask.sum() > 0:
                ax.scatter(merged.loc[mask, feat_x], merged.loc[mask, feat_y],
                           c=color, s=12, alpha=0.6, label=cls.capitalize())
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(valid[feat_x], valid[feat_y], c=valid["month"],
                        cmap="twilight", s=12, alpha=0.6, vmin=1, vmax=12)
        plt.colorbar(sc, ax=ax, label="Month", shrink=0.8)

    # Correlation
    r = valid[feat_x].corr(valid[feat_y])
    ax.set_xlabel(_label(feat_x))
    ax.set_ylabel(_label(feat_y))
    ax.set_title(f"{station} {year} — r = {r:+.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_sector_comparison(daily_features, station, year):
    """Same feature across azimuth sectors."""
    sectors = daily_features[daily_features["azimuth_bin"] >= 0]
    if sectors.empty:
        st.info("No per-sector data available.")
        return

    feature_cols = _get_feature_cols(sectors)
    selected = st.selectbox("Feature", feature_cols, index=0, format_func=_label,
                            key=f"sector_cmp_{station}_{year}")

    az_bins = sorted(sectors["azimuth_bin"].unique())
    n_sectors = len(az_bins)

    fig, ax = plt.subplots(figsize=(max(6, n_sectors * 1.2), 4), dpi=100)

    box_data = []
    box_labels = []
    for az in az_bins:
        vals = sectors[sectors["azimuth_bin"] == az][selected].dropna()
        if len(vals) > 0:
            box_data.append(vals.values)
            box_labels.append(f"{az}")

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
        cmap = plt.cm.tab10
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(i % 10))
            patch.set_alpha(0.6)

    ax.set_xlabel("Azimuth Bin")
    ax.set_ylabel(_label(selected))
    ax.set_title(f"{station} {year} — {_label(selected)} by Sector")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_multi_station(year):
    """Overlay same feature from multiple stations."""
    results_base = PROJECT_ROOT / "results_annual"
    if not results_base.exists():
        st.info("No results_annual directory found.")
        return

    # Find stations with daily_features for this year
    available = []
    for d in sorted(results_base.iterdir()):
        if d.is_dir():
            df_path = d / f"{d.name}_{year}_daily_features.parquet"
            if df_path.exists():
                available.append(d.name)

    if len(available) < 2:
        st.info(f"Need at least 2 stations with daily_features for {year}. Found: {available}")
        return

    selected_stations = st.multiselect("Stations", available, default=available[:3],
                                        key=f"multi_sta_{year}")
    if len(selected_stations) < 1:
        return

    # Load all and find common features
    dfs = {}
    common_cols = None
    for sta in selected_stations:
        df = load_daily_features(sta, year)
        if df is not None:
            pooled = df[df["azimuth_bin"] == -1]
            if not pooled.empty:
                dfs[sta] = pooled
                cols = set(_get_feature_cols(pooled))
                common_cols = cols if common_cols is None else common_cols & cols

    if not dfs or not common_cols:
        st.info("No common features found across selected stations.")
        return

    common_cols = sorted(common_cols)
    selected = st.selectbox("Feature", common_cols, index=0, format_func=_label,
                            key=f"multi_feat_{year}")

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    cmap = plt.cm.tab10
    for i, (sta, df) in enumerate(dfs.items()):
        df = df.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        vals = df.dropna(subset=[selected])
        rolling = vals.set_index("date_dt")[selected].rolling("7D", center=True).median()
        ax.plot(rolling.index, rolling.values, linewidth=1.5, alpha=0.8,
                color=cmap(i % 10), label=sta)
        ax.scatter(vals["date_dt"], vals[selected], s=4, alpha=0.2, color=cmap(i % 10))

    ax.set_ylabel(_label(selected))
    ax.set_title(f"{_label(selected)} — Multi-Station Overlay ({year})")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def has_feature_data(station, year):
    """Check if daily_features.parquet exists for a station/year."""
    path = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_daily_features.parquet"
    return path.exists()


def render_feature_explorer_tab(station_id, year):
    """Render the feature explorer tab."""
    st.header("Feature Explorer")

    daily_features = load_daily_features(station_id, year)
    if daily_features is None or daily_features.empty:
        st.info(
            f"No daily_features found for {station_id} {year}. Generate with:\n"
            f"```\npython scripts/feature_aggregator.py --station {station_id} --year {year}\n```"
        )
        return

    clf = load_ice_classification(station_id, year)

    st.caption(
        f"Exploring `{station_id}_{year}_daily_features.parquet` — "
        f"{len(daily_features)} rows, "
        f"{daily_features['azimuth_bin'].nunique()} sectors"
    )

    section = st.radio(
        "View",
        ["Time Series", "Feature Correlation", "Per-Sector Comparison", "Multi-Station Overlay"],
        horizontal=True,
        key=f"feat_view_{station_id}_{year}",
    )

    if section == "Time Series":
        _render_feature_timeseries(daily_features, clf, station_id, year)
    elif section == "Feature Correlation":
        _render_feature_correlation(daily_features, clf, station_id, year)
    elif section == "Per-Sector Comparison":
        _render_sector_comparison(daily_features, station_id, year)
    elif section == "Multi-Station Overlay":
        _render_multi_station(year)
