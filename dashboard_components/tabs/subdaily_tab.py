# ABOUTME: Implements subdaily comparison tab for GNSS-IR dashboard.
# ABOUTME: Shows spline-corrected WSE vs reference with confidence annotations and gap shading.

"""
Subdaily Comparison Tab Implementation

Displays spline-corrected GNSS-IR water surface elevation compared to
reference gauge data. Shows raw retrievals as context, spline fit with
confidence (observed vs interpolated), gap regions, and datum info.
"""

import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dashboard_components.station_metadata import (  # noqa: E402
    get_reference_source_info,
)


def load_spline_matched_data(station_id: str, year: int) -> tuple:
    """
    Load spline-matched data, falling back to raw-matched if unavailable.

    Returns:
        Tuple of (DataFrame or None, data_type str, error_message or None)
        data_type is "spline" or "raw"
    """
    results_dir = project_root / "results_annual" / station_id

    # Prefer spline-matched data
    spline_file = results_dir / f"{station_id}_{year}_spline_matched.csv"
    if spline_file.exists():
        try:
            df = pd.read_csv(spline_file)
            df["spline_datetime"] = pd.to_datetime(
                df["spline_datetime"], format="mixed", utc=True
            )
            df = df.sort_values("spline_datetime").reset_index(drop=True)
            return df, "spline", None
        except Exception as e:
            return None, "spline", f"Error loading spline data: {e}"

    # Fall back to raw-matched
    raw_file = results_dir / f"{station_id}_{year}_subdaily_matched.csv"
    if raw_file.exists():
        try:
            df = pd.read_csv(raw_file)
            df["gnss_datetime"] = pd.to_datetime(
                df["gnss_datetime"], format="mixed", utc=True
            )
            df = df.sort_values("gnss_datetime").reset_index(drop=True)
            return df, "raw", None
        except Exception as e:
            return None, "raw", f"Error loading raw-matched data: {e}"

    return None, "none", "No subdaily matched data found."


def load_raw_retrievals(station_id: str, year: int) -> pd.DataFrame:
    """Load raw individual retrieval data for scatter overlay."""
    results_dir = project_root / "results_annual" / station_id
    raw_file = results_dir / f"{station_id}_{year}_subdaily_matched.csv"

    if not raw_file.exists():
        return None

    try:
        df = pd.read_csv(raw_file)
        df["gnss_datetime"] = pd.to_datetime(
            df["gnss_datetime"], format="mixed", utc=True
        )
        return df
    except Exception:
        return None


def load_confidence_data(station_id: str, year: int) -> pd.DataFrame:
    """Load subdaily.csv with confidence annotations and gap info."""
    results_dir = project_root / "results_annual" / station_id
    subdaily_file = results_dir / f"{station_id.lower()}_{year}_subdaily.csv"

    if not subdaily_file.exists():
        return None

    try:
        df = pd.read_csv(subdaily_file)
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True)
        return df
    except Exception:
        return None


def detect_ref_column(df: pd.DataFrame) -> tuple:
    """
    Detect the reference demeaned column name and source type.

    Returns:
        Tuple of (ref_dm_col, ref_source_name)
    """
    if "usgs_wl_dm" in df.columns:
        return "usgs_wl_dm", "USGS"
    if "coops_dm" in df.columns:
        return "coops_dm", "CO-OPS"

    # Generic ERDDAP detection
    ref_dm_cols = [
        col for col in df.columns
        if col.endswith("_dm") and not col.startswith(("gnss", "spline"))
    ]
    if ref_dm_cols:
        return ref_dm_cols[0], "ERDDAP"

    return None, "Unknown"


def create_spline_comparison_plot(
    spline_df: pd.DataFrame,
    ref_col: str,
    ref_source: str,
    station_name: str,
    year: int,
    ref_site_name: str = "Reference",
    distance_km: float = 0.0,
    raw_df: pd.DataFrame = None,
    confidence_df: pd.DataFrame = None,
    show_raw_scatter: bool = True,
) -> plt.Figure:
    """Create subdaily comparison plot with spline overlay and confidence."""

    time_col = "spline_datetime"
    spline_col = "spline_dm"

    # Compute stats
    correlation = spline_df[spline_col].corr(spline_df[ref_col])
    rmse = np.sqrt(np.mean((spline_df[spline_col] - spline_df[ref_col]) ** 2))
    residuals = spline_df[spline_col] - spline_df[ref_col]
    n_points = len(spline_df)

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True, facecolor="white"
    )
    ax_main.set_facecolor("white")
    ax_resid.set_facecolor("white")

    for ax in [ax_main, ax_resid]:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.subplots_adjust(hspace=0.05)

    # Shade observation gaps from confidence data
    if confidence_df is not None and "is_interpolated" in confidence_df.columns:
        interp_mask = confidence_df["is_interpolated"].values
        times = confidence_df["datetime"].values

        # Find contiguous interpolated regions
        in_gap = False
        gap_start = None
        for i in range(len(interp_mask)):
            if interp_mask[i] and not in_gap:
                gap_start = times[i]
                in_gap = True
            elif not interp_mask[i] and in_gap:
                ax_main.axvspan(gap_start, times[i], alpha=0.08, color="#E74C3C", zorder=0)
                ax_resid.axvspan(gap_start, times[i], alpha=0.08, color="#E74C3C", zorder=0)
                in_gap = False
        if in_gap:
            ax_main.axvspan(gap_start, times[-1], alpha=0.08, color="#E74C3C", zorder=0)
            ax_resid.axvspan(gap_start, times[-1], alpha=0.08, color="#E74C3C", zorder=0)

    # Raw retrieval scatter (background context)
    if show_raw_scatter and raw_df is not None:
        gnss_dm_col = None
        if "gnss_dm" in raw_df.columns:
            gnss_dm_col = "gnss_dm"
        elif "gnss_wse_dm" in raw_df.columns:
            gnss_dm_col = "gnss_wse_dm"

        if gnss_dm_col:
            ax_main.scatter(
                raw_df["gnss_datetime"], raw_df[gnss_dm_col],
                c="#B0C4DE", s=4, alpha=0.3, zorder=1, rasterized=True,
            )

    # Reference line (hourly averaged)
    spline_df = spline_df.copy()
    spline_df["_hour"] = spline_df[time_col].dt.floor("h")
    hourly_ref = spline_df.groupby("_hour")[ref_col].mean()

    ax_main.plot(
        hourly_ref.index, hourly_ref.values,
        color="#C0392B", linewidth=2.5, alpha=0.9, zorder=5,
    )

    # Spline line — solid where observed, dashed where interpolated
    if confidence_df is not None and "is_interpolated" in confidence_df.columns:
        # Merge confidence flags onto spline matched data
        spline_with_conf = pd.merge_asof(
            spline_df.sort_values(time_col),
            confidence_df[["datetime", "is_interpolated"]].sort_values("datetime"),
            left_on=time_col, right_on="datetime",
            tolerance=pd.Timedelta(minutes=20),
            direction="nearest",
        )
        is_interp = spline_with_conf["is_interpolated"].fillna(False).values
        times = spline_df[time_col].values
        vals = spline_df[spline_col].values

        # Plot observed segments as solid, interpolated as dashed
        i = 0
        while i < len(times):
            j = i + 1
            while j < len(times) and is_interp[j] == is_interp[i]:
                j += 1
            # Extend one point for continuity
            end = min(j + 1, len(times))
            style = "--" if is_interp[i] else "-"
            alpha = 0.5 if is_interp[i] else 0.9
            ax_main.plot(
                times[i:end], vals[i:end],
                color="#2471A3", linewidth=2.0, linestyle=style,
                alpha=alpha, zorder=4,
            )
            i = j
    else:
        # No confidence data — solid line throughout
        ax_main.plot(
            spline_df[time_col], spline_df[spline_col],
            color="#2471A3", linewidth=2.0, alpha=0.9, zorder=4,
        )

    # Zero line
    ax_main.axhline(0, color="gray", linestyle="--", alpha=0.5, zorder=0)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="#C0392B", linewidth=2.5),
        Line2D([0], [0], color="#2471A3", linewidth=2.0),
        Line2D([0], [0], color="#2471A3", linewidth=2.0, linestyle="--", alpha=0.5),
    ]
    legend_labels = [
        f"{ref_source} ({ref_site_name[:25]})",
        "Spline fit (observed)",
        "Spline fit (interpolated)",
    ]
    if show_raw_scatter and raw_df is not None:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0C4DE",
                   markersize=5, linestyle="None")
        )
        legend_labels.append("Raw retrievals")
    if confidence_df is not None and "is_interpolated" in confidence_df.columns:
        legend_handles.append(Patch(facecolor="#E74C3C", alpha=0.08))
        legend_labels.append("Observation gap")

    ax_main.set_ylabel("Demeaned Water Level (m)", fontsize=12, color="black")
    ax_main.set_title(
        f"{station_name} Subdaily: Spline WSE vs {ref_source} ({year})",
        fontsize=14, color="black",
    )
    ax_main.legend(handles=legend_handles, labels=legend_labels, loc="upper left", fontsize=10)
    ax_main.tick_params(colors="black")
    ax_main.grid(True, alpha=0.3)

    # y-axis limits from reference range
    ref_range = spline_df[ref_col].max() - spline_df[ref_col].min()
    y_limit = max(1.0, ref_range * 2.0)
    ax_main.set_ylim(-y_limit, y_limit)

    # Info box with datum
    datum = spline_df["datum"].iloc[0] if "datum" in spline_df.columns else "unknown"
    info_lines = [
        f"r = {correlation:.3f}",
        f"RMSE = {rmse:.3f} m",
        f"N = {n_points:,}",
        f"Datum: {datum}",
    ]
    if distance_km > 0:
        info_lines.append(f"Dist: {distance_km:.1f} km")

    props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95, edgecolor="gray")
    ax_main.text(
        0.99, 0.98, "\n".join(info_lines),
        transform=ax_main.transAxes, fontsize=10, va="top", ha="right",
        bbox=props, color="black", family="monospace",
    )

    # Residual panel
    ax_resid.scatter(spline_df[time_col], residuals, c="#8E44AD", s=3, alpha=0.5)
    ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.5)

    resid_mean = residuals.mean()
    resid_std = residuals.std()
    ax_resid.text(
        0.02, 0.95,
        f"bias={resid_mean:.3f}m, std={resid_std:.3f}m",
        transform=ax_resid.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color="black",
    )

    ax_resid.set_ylabel("Residual (m)", fontsize=11, color="black")
    ax_resid.set_xlabel("Date", fontsize=12, color="black")
    ax_resid.tick_params(colors="black")
    ax_resid.grid(True, alpha=0.3)

    # Auto-scale residual axis
    resid_limit = max(0.5, np.percentile(np.abs(residuals), 99) * 1.3)
    ax_resid.set_ylim(-resid_limit, resid_limit)

    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator())
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    plt.tight_layout()
    return fig


def create_raw_comparison_plot(
    df: pd.DataFrame,
    gnss_col: str,
    ref_col: str,
    ref_source: str,
    station_name: str,
    year: int,
    ref_site_name: str = "Reference",
    distance_km: float = 0.0,
    show_ribbon: bool = True,
    ribbon_window: int = 50,
) -> plt.Figure:
    """Create comparison plot from raw retrievals (fallback when no spline)."""

    correlation = df[gnss_col].corr(df[ref_col])
    rmse = np.sqrt(np.mean((df[gnss_col] - df[ref_col]) ** 2))
    residuals = df[gnss_col] - df[ref_col]
    n_points = len(df)

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True, facecolor="white"
    )
    ax_main.set_facecolor("white")
    ax_resid.set_facecolor("white")

    for ax in [ax_main, ax_resid]:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.subplots_adjust(hspace=0.05)

    # Reference as hourly line
    df = df.copy()
    df["_hour"] = df["gnss_datetime"].dt.floor("h")
    hourly_ref = df.groupby("_hour")[ref_col].mean()

    ax_main.plot(
        hourly_ref.index, hourly_ref.values,
        color="#C0392B", linewidth=2.5, alpha=0.9, zorder=5,
    )

    # Rolling ribbon
    if show_ribbon:
        rolling_mean = df[gnss_col].rolling(ribbon_window, center=True, min_periods=5).mean()
        rolling_std = df[gnss_col].rolling(ribbon_window, center=True, min_periods=5).std()

        ax_main.fill_between(
            df["gnss_datetime"],
            rolling_mean - rolling_std, rolling_mean + rolling_std,
            alpha=0.25, color="#3498DB", zorder=1,
        )
        ax_main.plot(
            df["gnss_datetime"], rolling_mean,
            color="#2471A3", linewidth=2.0, alpha=0.8, zorder=3,
        )

    # Raw scatter
    ax_main.scatter(
        df["gnss_datetime"], df[gnss_col],
        c="#1A5276", s=8, alpha=0.6, zorder=4,
    )

    ax_main.axhline(0, color="gray", linestyle="--", alpha=0.5, zorder=0)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="#C0392B", linewidth=2.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1A5276",
               markersize=6, linestyle="None"),
    ]
    legend_labels = [
        f"{ref_source} ({ref_site_name[:25]})",
        "GNSS-IR retrievals",
    ]
    if show_ribbon:
        legend_handles.insert(1, Line2D([0], [0], color="#2471A3", linewidth=2))
        legend_handles.insert(2, Patch(facecolor="#3498DB", alpha=0.25))
        legend_labels.insert(1, "Rolling mean")
        legend_labels.insert(2, "Rolling +/-1 std")

    ax_main.set_ylabel("Demeaned Water Level (m)", fontsize=12, color="black")
    ax_main.set_title(
        f"{station_name} Subdaily: Raw Retrievals vs {ref_source} ({year})",
        fontsize=14, color="black",
    )
    ax_main.legend(handles=legend_handles, labels=legend_labels, loc="upper left", fontsize=10)
    ax_main.tick_params(colors="black")
    ax_main.grid(True, alpha=0.3)

    ref_range = df[ref_col].max() - df[ref_col].min()
    y_limit = max(1.0, ref_range * 2.5)
    ax_main.set_ylim(-y_limit, y_limit)

    # Info box
    info_lines = [
        f"r = {correlation:.3f}",
        f"RMSE = {rmse:.3f} m",
        f"N = {n_points:,}",
    ]
    if distance_km > 0:
        info_lines.append(f"Dist: {distance_km:.1f} km")

    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray")
    ax_main.text(
        0.99, 0.98, "\n".join(info_lines),
        transform=ax_main.transAxes, fontsize=10, va="top", ha="right",
        bbox=props, color="black",
    )

    # Residuals
    ax_resid.scatter(df["gnss_datetime"], residuals, c="#8E44AD", s=2, alpha=0.4)
    ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax_resid.text(
        0.02, 0.95,
        f"bias={residuals.mean():.3f}m, std={residuals.std():.3f}m",
        transform=ax_resid.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color="black",
    )

    ax_resid.set_ylabel("Residual (m)", fontsize=11, color="black")
    ax_resid.set_xlabel("Date", fontsize=12, color="black")
    ax_resid.tick_params(colors="black")
    ax_resid.grid(True, alpha=0.3)
    ax_resid.set_ylim(-1.5, 1.5)

    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator())
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    plt.tight_layout()
    return fig


def render_subdaily_tab(station_id: str, year: int, rh_data=None, comparison_data=None):
    """Render the subdaily comparison tab."""
    st.header("Subdaily Comparison")

    # Get station metadata
    ref_info = get_reference_source_info(station_id)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(
            f"**Reference:** {ref_info['primary_source']} - {ref_info['station_name']}"
        )
    with col2:
        if ref_info["distance_km"]:
            st.metric("Distance to Reference", f"{ref_info['distance_km']:.1f} km")

    # Load data
    df, data_type, error = load_spline_matched_data(station_id, year)

    if error:
        st.error(error)
        st.markdown(
            f"""
        **To generate subdaily data, run:**
        ```bash
        python scripts/process_station.py --station {station_id} --year {year} --skip_gnssir
        ```
        """
        )
        return

    if data_type == "spline":
        _render_spline_view(df, station_id, year, ref_info)
    else:
        _render_raw_view(df, station_id, year, ref_info)


def _render_spline_view(df, station_id, year, ref_info):
    """Render the spline-based comparison view."""
    ref_col, ref_source = detect_ref_column(df)

    if ref_col is None or "spline_dm" not in df.columns:
        st.error(f"Missing required columns. Available: {df.columns.tolist()}")
        return

    # Load supplementary data
    raw_df = load_raw_retrievals(station_id, year)
    confidence_df = load_confidence_data(station_id, year)

    # Status
    datum = df["datum"].iloc[0] if "datum" in df.columns else "unknown"
    st.success(
        f"Loaded **{len(df):,}** spline-matched points "
        f"(datum: {datum})"
    )

    if confidence_df is not None and "is_interpolated" in confidence_df.columns:
        n_interp = confidence_df["is_interpolated"].sum()
        pct_interp = n_interp / len(confidence_df) * 100
        st.caption(
            f"Confidence: {pct_interp:.1f}% of spline points are interpolated "
            f"(>{3.0:.0f}h from nearest observation)"
        )

    # Settings
    st.markdown("### Plot Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_raw = st.checkbox(
            "Show raw retrievals",
            value=True,
            help="Overlay individual GNSS-IR retrievals as background scatter",
        )

    with col2:
        # Placeholder for future settings
        pass

    with col3:
        min_date = df["spline_datetime"].min().date()
        max_date = df["spline_datetime"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["spline_datetime"].dt.date >= start_date) & (
            df["spline_datetime"].dt.date <= end_date
        )
        df_filtered = df[mask].copy()

        # Filter raw data to same range
        if raw_df is not None:
            raw_mask = (raw_df["gnss_datetime"].dt.date >= start_date) & (
                raw_df["gnss_datetime"].dt.date <= end_date
            )
            raw_filtered = raw_df[raw_mask].copy() if raw_mask.any() else None
        else:
            raw_filtered = None

        # Filter confidence data
        if confidence_df is not None:
            conf_mask = (confidence_df["datetime"].dt.date >= start_date) & (
                confidence_df["datetime"].dt.date <= end_date
            )
            conf_filtered = confidence_df[conf_mask].copy() if conf_mask.any() else None
        else:
            conf_filtered = None
    else:
        df_filtered = df.copy()
        raw_filtered = raw_df
        conf_filtered = confidence_df

    if len(df_filtered) == 0:
        st.warning("No data in selected date range")
        return

    # Generate plot
    with st.spinner("Generating comparison plot..."):
        fig = create_spline_comparison_plot(
            spline_df=df_filtered,
            ref_col=ref_col,
            ref_source=ref_source,
            station_name=station_id,
            year=year,
            ref_site_name=ref_info["station_name"],
            distance_km=ref_info["distance_km"] or 0.0,
            raw_df=raw_filtered,
            confidence_df=conf_filtered,
            show_raw_scatter=show_raw,
        )

    st.pyplot(fig)
    plt.close()

    # Statistics
    _render_stats(df_filtered, "spline_dm", ref_col)

    # Export
    _render_export(df_filtered, station_id, year, fig, ref_col, ref_source, ref_info)

    # Guide
    _render_interpretation_guide(has_spline=True)


def _render_raw_view(df, station_id, year, ref_info):
    """Render the raw retrieval comparison view (fallback)."""

    # Detect columns
    gnss_col = "gnss_dm" if "gnss_dm" in df.columns else "gnss_wse_dm"
    ref_dm_cols = [
        col for col in df.columns
        if col.endswith("_dm") and not col.startswith("gnss")
    ]
    if not ref_dm_cols or gnss_col not in df.columns:
        st.error(f"Missing required columns. Available: {df.columns.tolist()}")
        return

    ref_col = ref_dm_cols[0]
    ref_source = "ERDDAP" if "usgs" not in ref_col and "coops" not in ref_col else ref_col.split("_")[0].upper()

    st.warning(
        "Spline-matched data not available. Showing raw retrievals. "
        "Run `subdaily` + `generate_erddap_matched.py` to generate spline data."
    )

    st.success(f"Loaded **{len(df):,}** raw-matched points")

    # Settings
    st.markdown("### Plot Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_ribbon = st.checkbox("Show scatter ribbon", value=True)
    with col2:
        ribbon_window = st.slider("Window size", 20, 100, 50, disabled=not show_ribbon)
    with col3:
        min_date = df["gnss_datetime"].min().date()
        max_date = df["gnss_datetime"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["gnss_datetime"].dt.date >= start_date) & (
            df["gnss_datetime"].dt.date <= end_date
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    if len(df_filtered) == 0:
        st.warning("No data in selected date range")
        return

    with st.spinner("Generating comparison plot..."):
        fig = create_raw_comparison_plot(
            df=df_filtered,
            gnss_col=gnss_col,
            ref_col=ref_col,
            ref_source=ref_source,
            station_name=station_id,
            year=year,
            ref_site_name=ref_info["station_name"],
            distance_km=ref_info["distance_km"] or 0.0,
            show_ribbon=show_ribbon,
            ribbon_window=ribbon_window,
        )

    st.pyplot(fig)
    plt.close()

    _render_stats(df_filtered, gnss_col, ref_col)
    _render_interpretation_guide(has_spline=False)


def _render_stats(df, gnss_col, ref_col):
    """Render statistics summary."""
    st.markdown("### Statistics")

    correlation = df[gnss_col].corr(df[ref_col])
    rmse = np.sqrt(np.mean((df[gnss_col] - df[ref_col]) ** 2))
    residuals = df[gnss_col] - df[ref_col]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (r)", f"{correlation:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.3f} m")
    with col3:
        st.metric("Bias", f"{residuals.mean():.3f} m")
    with col4:
        st.metric("N Points", f"{len(df):,}")


def _render_export(df, station_id, year, fig, ref_col, ref_source, ref_info):
    """Render export buttons."""
    st.markdown("### Export")

    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Data (CSV)",
            data=csv_data,
            file_name=f"{station_id}_{year}_subdaily_filtered.csv",
            mime="text/csv",
        )

    with col2:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        buf.seek(0)

        st.download_button(
            label="Download Plot (PNG)",
            data=buf.getvalue(),
            file_name=f"{station_id}_{year}_subdaily_comparison.png",
            mime="image/png",
        )


def _render_interpretation_guide(has_spline: bool):
    """Render interpretation guide expander."""
    with st.expander("Interpretation Guide"):
        if has_spline:
            st.markdown(
                """
**Top Panel - Time Series:**
- **Blue solid line**: Spline fit to GNSS-IR WSE (RHdot + IF bias corrected)
- **Blue dashed line**: Spline in interpolated regions (far from observations)
- **Red line**: Reference gauge hourly averages
- **Light scatter**: Individual raw GNSS-IR retrievals (before corrections)
- **Pink shading**: Observation gaps (>3h without data)

**Bottom Panel - Residuals:**
- Spline minus reference (demeaned)
- **bias**: Mean residual offset
- **std**: Scatter around mean

**Info Box:**
- **Datum**: Vertical reference used for WSE (orthometric EGM96)
- Statistics are computed from the spline fit, not raw retrievals
            """
            )
        else:
            st.markdown(
                """
**Top Panel - Time Series:**
- **Blue scatter**: Individual GNSS-IR water surface elevation retrievals
- **Blue ribbon**: Rolling mean +/- 1 standard deviation
- **Red line**: Reference gauge hourly averages

**Bottom Panel - Residuals:**
- GNSS-IR minus reference (demeaned)

**Note:** This view shows raw retrievals without RHdot or IF bias correction.
Run `subdaily` to generate corrected spline output for better results.
            """
            )


__all__ = ["render_subdaily_tab"]
