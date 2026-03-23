# ABOUTME: Validation tab — temporal zoom (Year/Month/Week) comparison of GNSS-IR vs reference.
# ABOUTME: Self-loading data with three view levels matched to data density.

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from dashboard_components.station_metadata import (  # noqa: E402
    get_reference_source_info,
    get_antenna_height,
)

# Frequency band colors for week view scatter
FREQ_COLORS = {
    1: "#1f77b4",    # L1 — blue
    2: "#2ca02c",    # L2 — green
    5: "#ff7f0e",    # L5 — orange
    20: "#d62728",   # L2C — red
    101: "#9467bd",  # GLONASS L1 — purple
    102: "#8c564b",  # GLONASS L2 — brown
    201: "#9467bd",  # Galileo E1
    205: "#ff7f0e",  # Galileo E5a
    206: "#e377c2",  # Galileo E5b
    207: "#bcbd22",  # Galileo E6
}


def has_validation_data(station_id: str, year: int) -> bool:
    """Check whether the Validation tab should be shown for this station/year."""
    ref_info = get_reference_source_info(station_id)
    if ref_info["primary_source"] == "Unknown":
        return False
    path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_subdaily_matched.csv"
    return path.exists()


@st.cache_data(ttl=3600)
def _load_subdaily_matched(station_id: str, year: int):
    """Load subdaily_matched.csv with datetime parsing."""
    path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_subdaily_matched.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["gnss_datetime"] = pd.to_datetime(df["gnss_datetime"], format="mixed", utc=True)
    return df.sort_values("gnss_datetime").reset_index(drop=True)


@st.cache_data(ttl=3600)
def _load_spline_matched(station_id: str, year: int):
    """Load spline_matched.csv if it exists."""
    path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_spline_matched.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["spline_datetime"] = pd.to_datetime(df["spline_datetime"], format="mixed", utc=True)
    return df.sort_values("spline_datetime").reset_index(drop=True)


def _detect_columns(df):
    """Detect column roles in a subdaily_matched DataFrame.

    Returns dict with standardized keys for GNSS-IR columns, reference columns,
    and optional metadata columns (freq, satellite, azimuth, amplitude).
    """
    cols = {"gnss_datetime": "gnss_datetime", "gnss_wse": "gnss_wse"}

    # GNSS demeaned column
    if "gnss_dm" in df.columns:
        cols["gnss_dm"] = "gnss_dm"
    elif "gnss_wse_dm" in df.columns:
        cols["gnss_dm"] = "gnss_wse_dm"
    else:
        cols["gnss_dm"] = None

    # Reference demeaned column
    if "usgs_wl_dm" in df.columns:
        cols["ref_dm"] = "usgs_wl_dm"
        cols["ref_source"] = "USGS"
    elif "coops_dm" in df.columns:
        cols["ref_dm"] = "coops_dm"
        cols["ref_source"] = "CO-OPS"
    else:
        # ERDDAP or other — find any *_dm not gnss/spline
        dm_cols = [c for c in df.columns if c.endswith("_dm")
                   and not c.startswith(("gnss", "spline"))]
        if dm_cols:
            cols["ref_dm"] = dm_cols[0]
            cols["ref_source"] = "ERDDAP"
        else:
            cols["ref_dm"] = None
            cols["ref_source"] = "Unknown"

    # Reference absolute water level column
    if "usgs_wl_m" in df.columns:
        cols["ref_wl"] = "usgs_wl_m"
    elif "coops_wl" in df.columns:
        cols["ref_wl"] = "coops_wl"
    else:
        # ERDDAP — find any *_wl not gnss
        wl_cols = [c for c in df.columns if c.endswith("_wl")
                   and not c.startswith("gnss")]
        if wl_cols:
            cols["ref_wl"] = wl_cols[0]
        else:
            cols["ref_wl"] = None

    # Reference datetime column
    if "usgs_datetime" in df.columns:
        cols["ref_datetime"] = "usgs_datetime"
    elif "coops_datetime" in df.columns:
        cols["ref_datetime"] = "coops_datetime"
    else:
        dt_cols = [c for c in df.columns if c.endswith("_datetime")
                   and not c.startswith(("gnss", "spline"))]
        if dt_cols:
            cols["ref_datetime"] = dt_cols[0]
        else:
            cols["ref_datetime"] = None

    # Optional metadata columns
    cols["freq"] = "freq" if "freq" in df.columns else None
    cols["satellite"] = "satellite" if "satellite" in df.columns else ("sat" if "sat" in df.columns else None)
    cols["azimuth"] = "azimuth" if "azimuth" in df.columns else ("Az" if "Az" in df.columns else None)
    cols["amplitude"] = "amplitude" if "amplitude" in df.columns else ("Amp" if "Amp" in df.columns else None)

    return cols


def _aggregate_to_daily(df, cols):
    """Aggregate subdaily retrievals to daily statistics."""
    work = df.copy()
    work["date"] = work["gnss_datetime"].dt.date

    agg_dict = {
        cols["gnss_wse"]: ["median", "std"],
        cols["gnss_dm"]: "median",
    }

    if cols["ref_wl"]:
        agg_dict[cols["ref_wl"]] = "mean"
    if cols["ref_dm"]:
        agg_dict[cols["ref_dm"]] = "mean"

    daily = work.groupby("date").agg(agg_dict)
    # Flatten multi-level column index
    daily.columns = ["gnss_wse_median", "gnss_wse_std", "gnss_dm_median",
                      *([f"ref_wl_mean"] if cols["ref_wl"] else []),
                      *([f"ref_dm_mean"] if cols["ref_dm"] else [])]
    daily["n_retrievals"] = work.groupby("date")[cols["gnss_wse"]].count()
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    if "ref_dm_mean" in daily.columns:
        daily["residual"] = daily["gnss_dm_median"] - daily["ref_dm_mean"]
        daily["daily_rmse"] = np.abs(daily["residual"])  # will compute rolling properly

    return daily


def _compute_stats(gnss_vals, ref_vals):
    """Compute correlation, RMSE, bias, and count from paired series."""
    valid = gnss_vals.notna() & ref_vals.notna()
    g, r = gnss_vals[valid], ref_vals[valid]
    if len(g) < 2:
        return {"correlation": np.nan, "rmse": np.nan, "bias": np.nan, "n": len(g)}

    return {
        "correlation": g.corr(r),
        "rmse": np.sqrt(((g - r) ** 2).mean()),
        "bias": (g - r).mean(),
        "n": len(g),
    }


def _render_stats_bar(stats, extra_metrics=None):
    """Render a row of st.metric widgets for key statistics."""
    ncols = 4 + (len(extra_metrics) if extra_metrics else 0)
    columns = st.columns(ncols)
    columns[0].metric("Correlation (r)", f"{stats['correlation']:.3f}" if not np.isnan(stats['correlation']) else "N/A")
    columns[1].metric("RMSE", f"{stats['rmse']:.3f} m" if not np.isnan(stats['rmse']) else "N/A")
    columns[2].metric("Bias", f"{stats['bias']:.3f} m" if not np.isnan(stats['bias']) else "N/A")
    columns[3].metric("N Points", f"{stats['n']:,}")
    if extra_metrics:
        for i, (label, value) in enumerate(extra_metrics.items()):
            columns[4 + i].metric(label, value)


def _render_year_view(df, daily, cols, ref_info):
    """Year view: rolling 30-day RMSE ribbon + monthly summary table."""
    if "ref_dm_mean" not in daily.columns:
        st.warning("No reference data available for year view.")
        return

    # --- Stats bar ---
    stats = _compute_stats(daily["gnss_dm_median"], daily["ref_dm_mean"])
    # Real bias from absolute values
    abs_bias = np.nan
    if cols["ref_wl"] and "ref_wl_mean" in daily.columns:
        abs_bias = daily["gnss_wse_median"].mean() - daily["ref_wl_mean"].mean()

    total_days_possible = (daily["date"].max() - daily["date"].min()).days + 1
    coverage = len(daily) / total_days_possible * 100 if total_days_possible > 0 else 0

    extra = {"Datum Offset": f"{abs_bias:.3f} m" if not np.isnan(abs_bias) else "N/A",
             "Coverage": f"{coverage:.0f}%"}
    _render_stats_bar(stats, extra_metrics=extra)

    # --- Rolling stats plot ---
    daily_sorted = daily.sort_values("date").copy()

    # Rolling RMSE: compute from daily residuals
    daily_sorted["rolling_rmse"] = np.sqrt(
        (daily_sorted["residual"] ** 2).rolling(30, min_periods=7, center=True).mean()
    )
    # Rolling correlation
    daily_sorted["rolling_corr"] = (
        daily_sorted["gnss_dm_median"]
        .rolling(30, min_periods=7, center=True)
        .corr(daily_sorted["ref_dm_mean"])
    )

    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor="white")
    ax1.set_facecolor("white")

    dates = daily_sorted["date"]

    # RMSE ribbon
    rmse_vals = daily_sorted["rolling_rmse"].values
    ax1.fill_between(dates, 0, rmse_vals, alpha=0.3, color="#e74c3c", label="30-day rolling RMSE")
    ax1.plot(dates, rmse_vals, color="#e74c3c", linewidth=1.5)
    ax1.set_ylabel("Rolling RMSE (m)", color="#e74c3c", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_ylim(bottom=0)

    # Correlation on twin axis
    ax2 = ax1.twinx()
    corr_vals = daily_sorted["rolling_corr"].values
    ax2.plot(dates, corr_vals, color="#2e86ab", linewidth=2, alpha=0.8, label="30-day rolling r")
    ax2.set_ylabel("Rolling Correlation (r)", color="#2e86ab", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2e86ab")
    ax2.set_ylim(-0.2, 1.05)

    # Gap shading: shade days with < 5 retrievals
    for _, row in daily_sorted[daily_sorted["n_retrievals"] < 5].iterrows():
        ax1.axvspan(row["date"] - pd.Timedelta(hours=12),
                     row["date"] + pd.Timedelta(hours=12),
                     alpha=0.05, color="gray", zorder=0)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(f"Year View — Rolling Agreement Statistics", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Monthly summary table ---
    st.markdown("#### Monthly Summary")
    daily_sorted["month"] = daily_sorted["date"].dt.month
    daily_sorted["month_name"] = daily_sorted["date"].dt.strftime("%b")

    monthly = daily_sorted.groupby(["month", "month_name"]).apply(
        lambda g: pd.Series({
            "N days": len(g),
            "RMSE (m)": np.sqrt((g["residual"] ** 2).mean()) if "residual" in g.columns else np.nan,
            "Correlation": g["gnss_dm_median"].corr(g["ref_dm_mean"]) if len(g) > 2 else np.nan,
            "Mean arcs": g["n_retrievals"].mean(),
        })
    ).reset_index()

    monthly = monthly.sort_values("month")
    display_df = monthly[["month_name", "N days", "RMSE (m)", "Correlation", "Mean arcs"]].copy()
    display_df = display_df.rename(columns={"month_name": "Month"})
    display_df["RMSE (m)"] = display_df["RMSE (m)"].round(3)
    display_df["Correlation"] = display_df["Correlation"].round(3)
    display_df["Mean arcs"] = display_df["Mean arcs"].round(1)

    def _color_rmse(val):
        if pd.isna(val):
            return ""
        if val < 0.1:
            return "background-color: #c8e6c9"  # green
        if val < 0.2:
            return "background-color: #fff9c4"  # yellow
        return "background-color: #ffcdd2"  # red

    def _color_corr(val):
        if pd.isna(val):
            return ""
        if val > 0.8:
            return "background-color: #c8e6c9"
        if val > 0.6:
            return "background-color: #fff9c4"
        return "background-color: #ffcdd2"

    styled = display_df.style.map(_color_rmse, subset=["RMSE (m)"]).map(
        _color_corr, subset=["Correlation"]
    ).format({"RMSE (m)": "{:.3f}", "Correlation": "{:.3f}", "Mean arcs": "{:.1f}"}, na_rep="—")

    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_month_view(df, daily, cols, ref_info, selected_month):
    """Month view: daily median WSE with error bars + reference line + residual."""
    if "ref_dm_mean" not in daily.columns:
        st.warning("No reference data available for month view.")
        return

    month_data = daily[daily["date"].dt.month == selected_month].copy()
    if month_data.empty:
        st.warning(f"No data for selected month.")
        return

    month_data = month_data.sort_values("date")

    # Stats bar
    stats = _compute_stats(month_data["gnss_dm_median"], month_data["ref_dm_mean"])
    _render_stats_bar(stats)

    fig, (ax_ts, ax_resid) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True, facecolor="white"
    )
    ax_ts.set_facecolor("white")
    ax_resid.set_facecolor("white")
    plt.subplots_adjust(hspace=0.05)

    dates = month_data["date"]

    # GNSS-IR daily median with error bars
    ax_ts.errorbar(
        dates, month_data["gnss_dm_median"],
        yerr=month_data["gnss_wse_std"].fillna(0),
        fmt="o", color="#2e86ab", markersize=6, capsize=3,
        label="GNSS-IR daily median", zorder=4,
    )

    # Reference daily mean as line
    ax_ts.plot(
        dates, month_data["ref_dm_mean"],
        "-", color="#c0392b", linewidth=2, alpha=0.9,
        label=f"{ref_info['primary_source']} daily mean", zorder=3,
    )

    ax_ts.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_ts.set_ylabel("Demeaned Water Level (m)", fontsize=11)
    ax_ts.legend(loc="upper left", fontsize=9)
    ax_ts.grid(True, alpha=0.2)
    ax_ts.set_title(f"Month View — {month_data['date'].dt.strftime('%B %Y').iloc[0]}", fontsize=13)

    # Residual scatter colored by arc count
    if "residual" in month_data.columns:
        sc = ax_resid.scatter(
            dates, month_data["residual"],
            c=month_data["n_retrievals"], cmap="YlOrRd", s=30,
            edgecolors="gray", linewidth=0.5, zorder=4,
        )
        plt.colorbar(sc, ax=ax_resid, label="Arc count", pad=0.02, shrink=0.8)

        # RMSE reference lines
        rmse = stats["rmse"]
        if not np.isnan(rmse):
            ax_resid.axhline(rmse, color="#e74c3c", linestyle=":", alpha=0.5, linewidth=1)
            ax_resid.axhline(-rmse, color="#e74c3c", linestyle=":", alpha=0.5, linewidth=1)

    ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_resid.set_ylabel("Residual (m)", fontsize=11)
    ax_resid.set_xlabel("Date", fontsize=11)
    ax_resid.grid(True, alpha=0.2)

    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax_resid.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(month_data) // 15)))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_week_view(df, cols, ref_info, week_start, spline_df=None):
    """Week view: raw retrieval scatter + reference line + per-retrieval residual."""
    week_end = week_start + pd.Timedelta(days=7)
    mask = (df["gnss_datetime"] >= week_start) & (df["gnss_datetime"] < week_end)
    week_data = df[mask].copy()

    if week_data.empty:
        st.warning("No retrievals in selected week.")
        return

    # Stats bar
    if cols["gnss_dm"] and cols["ref_dm"]:
        stats = _compute_stats(week_data[cols["gnss_dm"]], week_data[cols["ref_dm"]])
        n_days = week_data["gnss_datetime"].dt.date.nunique()
        extra = {"Days with data": f"{n_days}/7",
                 "Arcs/day": f"{len(week_data) / max(n_days, 1):.1f}"}
        _render_stats_bar(stats, extra_metrics=extra)

    fig, (ax_ts, ax_resid) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True, facecolor="white"
    )
    ax_ts.set_facecolor("white")
    ax_resid.set_facecolor("white")
    plt.subplots_adjust(hspace=0.05)

    gnss_dm_col = cols["gnss_dm"]
    ref_dm_col = cols["ref_dm"]

    # Color by frequency if available
    if cols["freq"]:
        freq_vals = week_data[cols["freq"]].values
        unique_freqs = sorted(week_data[cols["freq"]].unique())
        scatter_colors = [FREQ_COLORS.get(int(f), "#888888") for f in freq_vals]

        ax_ts.scatter(
            week_data["gnss_datetime"], week_data[gnss_dm_col],
            c=scatter_colors, s=15, alpha=0.7, zorder=4,
        )

        # Legend for frequency bands
        legend_handles = []
        for f in unique_freqs:
            color = FREQ_COLORS.get(int(f), "#888888")
            label = {1: "L1", 2: "L2", 5: "L5", 20: "L2C",
                     101: "G1", 102: "G2", 201: "E1", 205: "E5a",
                     206: "E5b", 207: "E6"}.get(int(f), f"F{int(f)}")
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=6, linestyle="None", label=label)
            )

        # Residual with same colors
        if ref_dm_col:
            residuals = week_data[gnss_dm_col].values - week_data[ref_dm_col].values
            ax_resid.scatter(
                week_data["gnss_datetime"], residuals,
                c=scatter_colors, s=10, alpha=0.6, zorder=4,
            )
    else:
        # Single color scatter
        ax_ts.scatter(
            week_data["gnss_datetime"], week_data[gnss_dm_col],
            c="#2e86ab", s=15, alpha=0.7, zorder=4, label="GNSS-IR retrievals",
        )
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e86ab",
                   markersize=6, linestyle="None", label="GNSS-IR retrievals")
        ]

        if ref_dm_col:
            residuals = week_data[gnss_dm_col].values - week_data[ref_dm_col].values
            ax_resid.scatter(
                week_data["gnss_datetime"], residuals,
                c="#8e44ad", s=10, alpha=0.5, zorder=4,
            )

    # Reference gauge as hourly-binned line
    if ref_dm_col and cols["ref_datetime"]:
        ref_dt = cols["ref_datetime"]
        if ref_dt in week_data.columns:
            ref_times = pd.to_datetime(week_data[ref_dt], format="mixed", utc=True)
            ref_df = pd.DataFrame({"dt": ref_times, "val": week_data[ref_dm_col].values})
            ref_df["hour"] = ref_df["dt"].dt.floor("h")
            hourly = ref_df.groupby("hour")["val"].mean().sort_index()
            ax_ts.plot(hourly.index, hourly.values, color="#c0392b", linewidth=1.5,
                       alpha=0.9, zorder=5, label=ref_info["primary_source"])
            legend_handles.append(
                Line2D([0], [0], color="#c0392b", linewidth=1.5,
                       label=ref_info["primary_source"])
            )

    # Optional spline overlay
    if spline_df is not None:
        spline_mask = (spline_df["spline_datetime"] >= week_start) & (spline_df["spline_datetime"] < week_end)
        spline_week = spline_df[spline_mask]
        if not spline_week.empty and "spline_dm" in spline_week.columns:
            ax_ts.plot(
                spline_week["spline_datetime"], spline_week["spline_dm"],
                color="#2e86ab", linewidth=1.0, linestyle="--", alpha=0.6,
                zorder=3, label="Spline fit",
            )
            legend_handles.append(
                Line2D([0], [0], color="#2e86ab", linewidth=1.0, linestyle="--",
                       alpha=0.6, label="Spline fit")
            )

    ax_ts.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_ts.set_ylabel("Demeaned Water Level (m)", fontsize=11)
    ax_ts.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax_ts.grid(True, alpha=0.2)
    ax_ts.set_title("Week View — Subdaily Retrievals", fontsize=13)

    ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_resid.set_ylabel("Residual (m)", fontsize=11)
    ax_resid.set_xlabel("Date", fontsize=11)
    ax_resid.grid(True, alpha=0.2)

    # Auto-scale residual
    if ref_dm_col:
        resid_vals = week_data[gnss_dm_col].values - week_data[ref_dm_col].values
        valid_resid = resid_vals[~np.isnan(resid_vals)]
        if len(valid_resid) > 0:
            lim = max(0.3, np.percentile(np.abs(valid_resid), 98) * 1.3)
            ax_resid.set_ylim(-lim, lim)

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_validation_tab(station_id: str, year: int):
    """Entry point for the Validation tab."""
    st.header("Validation")

    ref_info = get_reference_source_info(station_id)

    # Reference info header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.info(f"**Reference:** {ref_info['primary_source']} — {ref_info['station_name']}")
    with col2:
        if ref_info.get("distance_km"):
            st.metric("Distance", f"{ref_info['distance_km']:.1f} km")
    with col3:
        antenna_h = get_antenna_height(station_id)
        st.metric("Antenna Height", f"{antenna_h:.3f} m")

    # Load data
    df = _load_subdaily_matched(station_id, year)
    if df is None or df.empty:
        st.error("No subdaily matched data found. Run the subdaily matching pipeline first.")
        return

    cols = _detect_columns(df)
    if cols["gnss_dm"] is None or cols["ref_dm"] is None:
        st.error(f"Could not detect required columns. Available: {list(df.columns)}")
        return

    # Aggregate to daily for Year/Month views
    daily = _aggregate_to_daily(df, cols)

    # Temporal zoom selector
    zoom = st.radio(
        "Temporal zoom",
        ["Year", "Month", "Week"],
        horizontal=True,
        help="Year: seasonal patterns. Month: daily tracking. Week: individual retrievals.",
    )

    if zoom == "Year":
        _render_year_view(df, daily, cols, ref_info)

    elif zoom == "Month":
        available_months = sorted(daily["date"].dt.month.unique())
        month_labels = {m: pd.Timestamp(year=year, month=m, day=1).strftime("%B") for m in available_months}

        selected_month = st.selectbox(
            "Select month",
            available_months,
            format_func=lambda m: month_labels[m],
        )
        _render_month_view(df, daily, cols, ref_info, selected_month)

    elif zoom == "Week":
        min_dt = df["gnss_datetime"].min().normalize()
        max_dt = df["gnss_datetime"].max().normalize()
        min_d, max_d = min_dt.date(), max_dt.date()

        # Initialize session state for week start
        key = f"val_week_start_{station_id}_{year}"
        if key not in st.session_state:
            st.session_state[key] = min_d

        # Prev/next buttons update session state before the date_input renders
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("< Prev week", key="val_prev"):
                new_d = st.session_state[key] - datetime.timedelta(days=7)
                st.session_state[key] = max(new_d, min_d)
                st.rerun()
        with nav_col3:
            if st.button("Next week >", key="val_next"):
                new_d = st.session_state[key] + datetime.timedelta(days=7)
                st.session_state[key] = min(new_d, max_d)
                st.rerun()
        with nav_col2:
            week_start_date = st.date_input(
                "Week starting",
                value=st.session_state[key],
                min_value=min_d,
                max_value=max_d,
                key=f"val_week_input_{station_id}_{year}",
            )
            # Sync manual date_input changes back to session state
            if week_start_date != st.session_state[key]:
                st.session_state[key] = week_start_date

        week_start_ts = pd.Timestamp(week_start_date, tz="UTC")

        # Spline overlay toggle
        spline_df = None
        spline_path = PROJECT_ROOT / "results_annual" / station_id / f"{station_id}_{year}_spline_matched.csv"
        if spline_path.exists():
            show_spline = st.checkbox("Show spline overlay", value=False)
            if show_spline:
                spline_df = _load_spline_matched(station_id, year)

        _render_week_view(df, cols, ref_info, week_start_ts, spline_df=spline_df)


__all__ = ["render_validation_tab", "has_validation_data"]
