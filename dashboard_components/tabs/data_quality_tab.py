# ABOUTME: Data Quality tab — internal GNSS-IR signal health, independent of external reference.
# ABOUTME: Annual summary, monthly drill-down, and single-day quickLook diagnostics.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from dashboard_components.data_loader import (  # noqa: E402
    load_enriched,
    load_per_arc,
    get_available_diagnostic_days,
    get_quicklook_plots_for_day,
    doy_to_date,
)

# Import self-contained diagnostic rendering functions
from dashboard_components.tabs.diagnostics_tab import (  # noqa: E402
    _render_azimuth_quality,
    _render_frequency_performance,
    _render_arc_quality,
    _render_daily_quality_timeline,
    _render_annual_qc,
    _render_quicklook_section,
)

# Calendar heatmap from visualizer library
try:
    from scripts.visualizer.dashboard_plots import create_calendar_heatmap

    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False


def has_quality_data(station_id: str, year: int) -> bool:
    """Check whether any quality data exists for this station/year."""
    enriched_path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_daily_enriched.parquet"
    per_arc_path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_per_arc.parquet"
    return enriched_path.exists() or per_arc_path.exists()


def _render_key_metrics(enriched, per_arc):
    """Top strip: key quality metrics with traffic-light indicators."""
    # Get pooled daily stats
    pooled = enriched[
        (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
    ].copy() if enriched is not None else pd.DataFrame()

    if pooled.empty and per_arc is None:
        return

    cols = st.columns(4)

    if not pooled.empty:
        n_days = len(pooled)
        mean_arcs = pooled["rh_count"].mean()
        mean_amp = pooled["amp_mean"].mean() if "amp_mean" in pooled.columns else np.nan
        mean_p2n = pooled["p2n_mean"].mean() if "p2n_mean" in pooled.columns else np.nan

        # Traffic lights
        def _indicator(val, good, fair):
            if pd.isna(val):
                return ""
            if val >= good:
                return "🟢"
            if val >= fair:
                return "🟡"
            return "🔴"

        cols[0].metric(
            f"{_indicator(n_days, 200, 100)} Processing Days",
            f"{n_days}",
        )
        cols[1].metric(
            f"{_indicator(mean_arcs, 30, 15)} Mean Arcs/Day",
            f"{mean_arcs:.1f}",
        )
        cols[2].metric(
            f"{_indicator(mean_amp, 10, 5)} Mean Amplitude",
            f"{mean_amp:.1f}" if not np.isnan(mean_amp) else "N/A",
        )
        cols[3].metric(
            f"{_indicator(mean_p2n, 3, 2)} Mean Peak-to-Noise",
            f"{mean_p2n:.1f}" if not np.isnan(mean_p2n) else "N/A",
        )
    elif per_arc is not None:
        n_days = per_arc["date"].nunique() if "date" in per_arc.columns else 0
        n_arcs = len(per_arc)
        cols[0].metric("Processing Days", f"{n_days}")
        cols[1].metric("Total Arcs", f"{n_arcs:,}")


def _render_annual_summary(enriched, per_arc, station_id, year):
    """Annual Summary view: calendar heatmap + azimuth quality + frequency performance."""

    # --- Calendar heatmap ---
    if enriched is not None and HAS_CALENDAR:
        pooled = enriched[
            (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
        ].copy()

        if not pooled.empty:
            st.markdown("#### Daily Retrieval Count")
            cal_df = pooled[["date", "rh_count"]].copy()
            cal_df["date"] = pd.to_datetime(cal_df["date"])

            fig = create_calendar_heatmap(
                cal_df,
                metric_col="rh_count",
                year=year,
                station_name=station_id,
                cmap="YlGn",
            )
            st.pyplot(fig)
            plt.close(fig)
    elif enriched is not None:
        # Fallback: simple bar chart of daily arc count
        pooled = enriched[
            (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
        ].copy()
        if not pooled.empty:
            st.markdown("#### Daily Retrieval Count")
            pooled["date_dt"] = pd.to_datetime(pooled["date"])
            pooled = pooled.sort_values("date_dt")
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.bar(pooled["date_dt"], pooled["rh_count"], width=1, color="#2e7d32", alpha=0.7)
            ax.axhline(pooled["rh_count"].median(), color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Arc count")
            ax.set_xlabel("Date")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- Azimuth quality + Frequency performance side by side ---
    if per_arc is not None or enriched is not None:
        col1, col2 = st.columns(2)

        with col1:
            if per_arc is not None:
                st.markdown("#### Azimuth Quality")
                _render_azimuth_quality(per_arc)

        with col2:
            if enriched is not None:
                st.markdown("#### Frequency Band Performance")
                _render_frequency_performance(enriched)


def _render_monthly_detail(enriched, per_arc, station_id, year):
    """Monthly Detail view: box plots + monthly signal quality."""

    if enriched is None:
        st.warning("No enriched daily data available for monthly analysis.")
        return

    pooled = enriched[
        (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
    ].copy()

    if pooled.empty:
        st.warning("No pooled daily data available.")
        return

    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled["month"] = pooled["date"].dt.month
    pooled["month_name"] = pooled["date"].dt.strftime("%b")

    # Metric selector
    metric = st.selectbox(
        "Select metric",
        ["rh_count", "rh_std", "amp_mean", "amp_cv", "p2n_mean"],
        format_func=lambda x: {
            "rh_count": "Arc Count",
            "rh_std": "RH Standard Deviation (m)",
            "amp_mean": "Mean Amplitude",
            "amp_cv": "Amplitude CV",
            "p2n_mean": "Mean Peak-to-Noise",
        }.get(x, x),
    )

    if metric not in pooled.columns:
        st.warning(f"Metric '{metric}' not available in enriched data.")
        return

    # --- Monthly box plots ---
    st.markdown(f"#### Monthly Distribution")

    months_present = sorted(pooled["month"].unique())
    monthly_data = [pooled[pooled["month"] == m][metric].dropna().values for m in months_present]
    month_labels = [pd.Timestamp(year=year, month=m, day=1).strftime("%b") for m in months_present]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(
        monthly_data, positions=range(len(months_present)),
        patch_artist=True, widths=0.6,
    )

    # Color by season
    season_colors = {12: "#5c6bc0", 1: "#5c6bc0", 2: "#5c6bc0",   # winter
                     3: "#66bb6a", 4: "#66bb6a", 5: "#66bb6a",     # spring
                     6: "#ffa726", 7: "#ffa726", 8: "#ffa726",     # summer
                     9: "#8d6e63", 10: "#8d6e63", 11: "#8d6e63"}   # fall

    for i, m in enumerate(months_present):
        bp["boxes"][i].set_facecolor(season_colors.get(m, "#90a4ae"))
        bp["boxes"][i].set_alpha(0.6)

    # Overlay individual points
    for i, (m, data) in enumerate(zip(months_present, monthly_data)):
        if len(data) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
            ax.scatter(np.full_like(data, i) + jitter, data,
                       c=season_colors.get(m, "#90a4ae"), s=8, alpha=0.4, zorder=3)

    ax.set_xticks(range(len(months_present)))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel({
        "rh_count": "Arc Count",
        "rh_std": "RH Std (m)",
        "amp_mean": "Amplitude",
        "amp_cv": "Amplitude CV",
        "p2n_mean": "Peak-to-Noise",
    }.get(metric, metric), fontsize=11)

    # Add mean trend line
    monthly_means = [np.nanmean(d) if len(d) > 0 else np.nan for d in monthly_data]
    valid_means = [(i, v) for i, v in enumerate(monthly_means) if not np.isnan(v)]
    if valid_means:
        ax.plot([x[0] for x in valid_means], [x[1] for x in valid_means],
                "k--", linewidth=1, alpha=0.5, label="Monthly mean")

    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Monthly signal quality: amp CV + RH std ---
    st.markdown("#### Signal Quality by Month")

    # Per-azimuth monthly breakdown
    az_data = enriched[
        (enriched["azimuth_bin"] != -1) & (enriched["freq_group"] == "ALL")
    ].copy()

    if not az_data.empty:
        az_data["date"] = pd.to_datetime(az_data["date"])
        az_data["month"] = az_data["date"].dt.month

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Amp CV by month
        monthly_cv = az_data.groupby("month")["amp_cv"].agg(["mean", "std"]).reindex(months_present)
        ax1.bar(range(len(months_present)), monthly_cv["mean"],
                yerr=monthly_cv["std"], capsize=3, color="#42a5f5", alpha=0.7)
        ax1.set_xticks(range(len(months_present)))
        ax1.set_xticklabels(month_labels)
        ax1.set_ylabel("Amplitude CV")
        ax1.set_title("Monthly Amplitude Consistency", fontsize=11)
        ax1.grid(True, alpha=0.2, axis="y")

        # RH std by month
        if "rh_std" in az_data.columns:
            monthly_rh = az_data.groupby("month")["rh_std"].agg(["mean", "std"]).reindex(months_present)
            ax2.bar(range(len(months_present)), monthly_rh["mean"],
                    yerr=monthly_rh["std"], capsize=3, color="#ef5350", alpha=0.7)
            ax2.set_xticks(range(len(months_present)))
            ax2.set_xticklabels(month_labels)
            ax2.set_ylabel("RH Std (m)")
            ax2.set_title("Monthly RH Scatter", fontsize=11)
            ax2.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_single_day(enriched, per_arc, station_id, year):
    """Single-Day Diagnostic: quickLook images + arc quality for selected day."""

    # QuickLook section (date picker + LSP/summary plots)
    _render_quicklook_section(station_id, year)

    # Arc quality distributions below quickLook
    if per_arc is not None:
        with st.expander("Arc Quality Distributions", expanded=False):
            _render_arc_quality(per_arc)

    # Annual QC overview
    if per_arc is not None:
        with st.expander("Annual QC Overview", expanded=False):
            _render_annual_qc(per_arc, station_id)


def render_data_quality_tab(station_id: str, year: int):
    """Entry point for the Data Quality tab."""
    st.header("Data Quality")

    enriched = load_enriched(station_id, year)
    per_arc = load_per_arc(station_id, year)

    if enriched is None and per_arc is None:
        st.warning(
            f"No quality data found for {station_id} {year}. "
            f"Run the processing pipeline to generate enriched and per-arc data."
        )
        return

    # Key metrics strip (always visible)
    _render_key_metrics(enriched, per_arc)

    st.divider()

    # View selector
    view = st.radio(
        "View",
        ["Annual Summary", "Monthly Detail", "Single-Day Diagnostic"],
        horizontal=True,
    )

    if view == "Annual Summary":
        _render_annual_summary(enriched, per_arc, station_id, year)

    elif view == "Monthly Detail":
        _render_monthly_detail(enriched, per_arc, station_id, year)

    elif view == "Single-Day Diagnostic":
        _render_single_day(enriched, per_arc, station_id, year)


__all__ = ["render_data_quality_tab", "has_quality_data"]
