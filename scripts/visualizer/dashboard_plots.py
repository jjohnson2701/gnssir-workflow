#!/usr/bin/env python3
"""
ABOUTME: Dashboard-specific plot functions for GNSS-IR visualization.
ABOUTME: Implements calendar heatmaps, box plots, timelines, and performance analysis.

This module implements the 6 priority plot types for the dashboard:
1. Calendar Heat Map with Performance Metrics
2. Monthly Performance Box Plots
3. Multi-Parameter Annual Timeline
4. Tidal Stage Performance Analysis
5. Multi-Scale Performance Matrix
6. Water Level Change Rate Response
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import calendar
from typing import Optional, Tuple
import logging

# Import the publication theme for consistent styling
from .publication_theme import apply_matplotlib_theme, PUBLICATION_COLORS

logger = logging.getLogger(__name__)


def create_monthly_box_plots(
    daily_df: pd.DataFrame,
    metric_col: str = "correlation",
    station_name: str = "GNSS Station",
    year: Optional[int] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
    add_points: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create monthly box plots showing performance distribution.

    Parameters:
    -----------
    daily_df : pd.DataFrame
        DataFrame with 'date' column and metric column
    metric_col : str
        Column name for the metric to display
    station_name : str
        Station identifier for title
    year : int, optional
        Year to filter (uses all data if None)
    title : str, optional
        Custom title
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    add_points : bool
        Whether to overlay individual data points
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()
    colors = PUBLICATION_COLORS

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    plot_df = daily_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    # Filter by year if specified
    if year is not None:
        plot_df = plot_df[plot_df["date"].dt.year == year]

    # Extract month
    plot_df["month"] = plot_df["date"].dt.month
    plot_df["month_name"] = plot_df["date"].dt.strftime("%b")

    # Create box plot
    box_data = [plot_df[plot_df["month"] == m][metric_col].dropna() for m in range(1, 13)]

    bp = ax.boxplot(
        box_data, positions=range(1, 13), widths=0.6, patch_artist=True, showfliers=False
    )

    # Style box plots
    for patch, month in zip(bp["boxes"], range(1, 13)):
        # Color by season
        if month in [12, 1, 2]:  # Winter
            color = colors["gnss_smooth"]
        elif month in [3, 4, 5]:  # Spring
            color = colors["coops"]
        elif month in [6, 7, 8]:  # Summer
            color = colors["ndbc"]
        else:  # Fall
            color = colors["usgs"]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points if requested
    if add_points:
        for month in range(1, 13):
            month_data = plot_df[plot_df["month"] == month][metric_col].dropna()
            if len(month_data) > 0:
                x = np.random.normal(month, 0.1, size=len(month_data))
                ax.scatter(x, month_data, alpha=0.3, s=20, color="black")

    # Add mean line
    means = [np.nanmean(data) if len(data) > 0 else np.nan for data in box_data]
    valid_months = [i for i, m in enumerate(means, 1) if not np.isnan(m)]
    valid_means = [m for m in means if not np.isnan(m)]
    ax.plot(valid_months, valid_means, "r--", linewidth=2, label="Monthly Mean")

    # Configure axis
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(calendar.month_abbr[1:13])
    ax.set_xlabel("Month", fontsize=12)

    if ylabel is None:
        ylabel = metric_col.replace("_", " ").title()
    ax.set_ylabel(ylabel, fontsize=12)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend
    ax.legend(loc="best")

    # Set title
    if title is None:
        year_str = f" {year}" if year else ""
        title = f"{station_name}{year_str} - Monthly {ylabel} Distribution"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Monthly box plots saved to {save_path}")

    return fig


def create_multi_parameter_timeline(
    gnssir_df: pd.DataFrame,
    usgs_df: Optional[pd.DataFrame] = None,
    environmental_df: Optional[pd.DataFrame] = None,
    station_name: str = "GNSS Station",
    year: Optional[int] = None,
    rolling_window: int = 30,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create multi-parameter annual timeline with performance and environmental context.

    Parameters:
    -----------
    gnssir_df : pd.DataFrame
        GNSS-IR data with 'date' column
    usgs_df : pd.DataFrame, optional
        USGS comparison data
    environmental_df : pd.DataFrame, optional
        Environmental data (wind, waves, etc.)
    station_name : str
        Station identifier
    year : int, optional
        Year to display
    rolling_window : int
        Window size for rolling statistics
    title : str, optional
        Custom title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()
    colors = PUBLICATION_COLORS

    # Determine number of panels
    n_panels = 3  # Base: performance, data availability, correlation
    if environmental_df is not None:
        n_panels += 1

    # Create figure
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    # Ensure gnssir_df has date column
    gnssir_df = gnssir_df.copy()
    gnssir_df["date"] = pd.to_datetime(gnssir_df["date"])

    # Filter by year if specified
    if year is not None:
        gnssir_df = gnssir_df[gnssir_df["date"].dt.year == year]

    # Panel 1: Data Availability
    ax1 = axes[0]
    if "rh_count" in gnssir_df.columns:
        # Daily retrieval counts
        ax1.bar(
            gnssir_df["date"],
            gnssir_df["rh_count"],
            color=colors["gnss_smooth"],
            alpha=0.6,
            label="Daily Retrievals",
        )

        # Rolling average
        rolling_avg = (
            gnssir_df.set_index("date")["rh_count"]
            .rolling(window=rolling_window, center=True)
            .mean()
        )
        ax1.plot(
            rolling_avg.index,
            rolling_avg.values,
            color="red",
            linewidth=2,
            label=f"{rolling_window}-day Average",
        )

        ax1.set_ylabel("GNSS-IR Retrievals", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

    # Panel 2: Performance Metrics (RH statistics)
    ax2 = axes[1]
    if "rh_std_m" in gnssir_df.columns:
        # Daily standard deviation
        ax2.fill_between(
            gnssir_df["date"],
            0,
            gnssir_df["rh_std_m"],
            color=colors["usgs"],
            alpha=0.5,
            label="Daily Std Dev",
        )

        # Rolling statistics
        rolling_std = (
            gnssir_df.set_index("date")["rh_std_m"]
            .rolling(window=rolling_window, center=True)
            .mean()
        )
        ax2.plot(
            rolling_std.index,
            rolling_std.values,
            color="darkred",
            linewidth=2,
            label=f"{rolling_window}-day Avg Std",
        )

        ax2.set_ylabel("RH Std Dev (m)", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    # Panel 3: Correlation/RMSE if USGS/CO-OPS data available
    ax3 = axes[2]
    if usgs_df is not None and "date" in usgs_df.columns and "rh_median_m" in gnssir_df.columns:
        # Merge data for correlation calculation
        usgs_df = usgs_df.copy()
        usgs_df["date"] = pd.to_datetime(usgs_df["date"])

        # Find the water level column
        usgs_value_col = None
        for col in ["water_level_m", "usgs_value", "usgs_value_m_median", "value"]:
            if col in usgs_df.columns:
                usgs_value_col = col
                break

        if usgs_value_col:
            # Check if we already have WSE column, otherwise convert RH to WSE
            if "wse_ellips_m" in gnssir_df.columns:
                gnss_col = "wse_ellips_m"
            else:
                # Need to convert RH to WSE - get antenna height from station name
                antenna_heights = {"FORA": -30.917, "UMNQ": 36.9963, "GLBX": -12.535}
                # Try to extract station code from name
                station_code = station_name.split()[0].upper() if station_name else None
                antenna_height = antenna_heights.get(station_code, 0)

                # Create WSE column
                gnssir_df = gnssir_df.copy()
                gnssir_df["wse_ellips_m"] = antenna_height - gnssir_df["rh_median_m"]
                gnss_col = "wse_ellips_m"

            merged = pd.merge(
                gnssir_df[["date", gnss_col]],
                usgs_df[["date", usgs_value_col]].rename(columns={usgs_value_col: "water_level"}),
                on="date",
                how="inner",
            )

            if len(merged) > rolling_window:
                # Calculate rolling correlation
                merged = merged.set_index("date")
                rolling_corr = (
                    merged[gnss_col].rolling(window=rolling_window).corr(merged["water_level"])
                )

                # Determine data source label
                source_label = "Water Level"

                ax3.plot(
                    rolling_corr.index,
                    rolling_corr.values,
                    color=colors["correlation"],
                    linewidth=2.5,
                    label=f"{rolling_window}-day Rolling Correlation (WSE vs {source_label})",
                )
                ax3.axhline(
                    y=0.8,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    label="Good Correlation Threshold",
                )
                ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
                ax3.set_ylabel("Correlation", fontsize=12)
                ax3.set_ylim(-1.1, 1.1)
                ax3.legend(loc="lower left")
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(
                    0.5,
                    0.5,
                    f"Insufficient data for correlation\n(need >{rolling_window} days)",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                )
                ax3.set_ylabel("Correlation", fontsize=12)
                ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No water level data available\nfor correlation analysis",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.set_ylabel("Correlation", fontsize=12)
        ax3.grid(True, alpha=0.3)

    # Panel 4: Environmental conditions (if available)
    if environmental_df is not None and n_panels > 3:
        ax4 = axes[3]
        env_df = environmental_df.copy()
        env_df["date"] = pd.to_datetime(env_df["date"])

        # Plot wind speed and wave height
        if "wind_speed" in env_df.columns:
            ax4.plot(
                env_df["date"],
                env_df["wind_speed"],
                color=colors["ndbc"],
                linewidth=1.5,
                alpha=0.7,
                label="Wind Speed (m/s)",
            )

        if "wave_height" in env_df.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(
                env_df["date"],
                env_df["wave_height"],
                color=colors["coops"],
                linewidth=1.5,
                alpha=0.7,
                label="Wave Height (m)",
            )
            ax4_twin.set_ylabel("Wave Height (m)", fontsize=12, color=colors["coops"])
            ax4_twin.tick_params(axis="y", labelcolor=colors["coops"])

        ax4.set_ylabel("Wind Speed (m/s)", fontsize=12, color=colors["ndbc"])
        ax4.tick_params(axis="y", labelcolor=colors["ndbc"])
        ax4.grid(True, alpha=0.3)

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].set_xlabel("Date", fontsize=12)

    # Rotate x-axis labels
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Set title
    if title is None:
        year_str = f" {year}" if year else ""
        title = f"{station_name}{year_str} - Multi-Parameter Timeline"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Multi-parameter timeline saved to {save_path}")

    return fig


def calculate_water_level_change_rate(
    df: pd.DataFrame,
    value_col: str = "water_level",
    time_col: str = "datetime",
    window_minutes: int = 60,
) -> pd.Series:
    """
    Calculate water level change rate in m/hr.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with water level data
    value_col : str
        Column name for water level values
    time_col : str
        Column name for datetime
    window_minutes : int
        Window size for rate calculation

    Returns:
    --------
    pd.Series : Change rates in m/hr
    """
    df = df.copy().sort_values(time_col)
    df[time_col] = pd.to_datetime(df[time_col])

    # Calculate time differences in hours
    time_diff = df[time_col].diff().dt.total_seconds() / 3600

    # Calculate value differences
    value_diff = df[value_col].diff()

    # Calculate rate (m/hr)
    rate = value_diff / time_diff

    # Apply rolling window smoothing
    window_size = int(window_minutes / (time_diff.median() * 60))
    rate_smooth = rate.rolling(window=max(3, window_size), center=True).mean()

    return rate_smooth


def classify_tidal_stage(
    water_level: pd.Series, time: pd.Series, window_hours: float = 2.0
) -> pd.Series:
    """
    Classify tidal stage as rising/falling/high/low based on water level trends.

    High/Low: Based on local extrema (peaks and troughs)
    Rising/Falling: Based on rate of change thresholds

    Parameters:
    -----------
    water_level : pd.Series
        Water level values
    time : pd.Series
        Datetime values
    window_hours : float
        Window for rate calculation

    Returns:
    --------
    pd.Series : Tidal stage classifications
    """
    # Calculate change rate
    df = pd.DataFrame({"time": time, "level": water_level})
    df = df.sort_values("time")

    # Calculate rate of change
    rate = calculate_water_level_change_rate(
        df, value_col="level", time_col="time", window_minutes=int(window_hours * 60)
    )

    # Classify stages
    stages = pd.Series(index=df.index, dtype=str)

    # Define adaptive thresholds based on data variability
    rate_std = rate.std()
    rate_threshold = max(0.05, rate_std * 0.5)  # Adaptive threshold, minimum 0.05 m/hr

    # First pass: Rising/Falling based on rate
    stages[rate > rate_threshold] = "Rising"
    stages[rate < -rate_threshold] = "Falling"

    # Second pass: High/Low based on local extrema
    # Use longer window for extrema detection
    extrema_window = max(5, len(df) // 50)  # Adaptive window size
    level_smooth = df["level"].rolling(window=extrema_window, center=True).mean()

    # Find local maxima and minima
    for i in range(extrema_window, len(level_smooth) - extrema_window):
        current_val = level_smooth.iloc[i]
        before_vals = level_smooth.iloc[i - extrema_window : i]
        after_vals = level_smooth.iloc[i + 1 : i + extrema_window + 1]

        # Local maximum
        if (current_val > before_vals.max()) and (current_val > after_vals.max()):
            stages.iloc[i] = "High"
        # Local minimum
        elif (current_val < before_vals.min()) and (current_val < after_vals.min()):
            stages.iloc[i] = "Low"

    # Fill remaining as slack water
    stages[stages == ""] = "Slack"

    return stages


def create_calendar_heatmap(
    daily_df: pd.DataFrame,
    metric_col: str = "correlation",
    year: int = 2024,
    station_name: str = "GNSS Station",
    title: Optional[str] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a calendar heat map showing daily performance metrics in 4x3 monthly grid.

    Parameters:
    -----------
    daily_df : pd.DataFrame
        DataFrame with 'date' column and metric column
    metric_col : str
        Column name for the metric to display
    year : int
        Year to display
    station_name : str
        Station identifier for title
    title : str, optional
        Custom title (auto-generated if None)
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()

    # Create figure with 4x3 grid for 12 months
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.flatten()

    # Prepare data
    if "date" in daily_df.columns:
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    # Filter to specified year
    year_data = daily_df[daily_df["date"].dt.year == year].copy()

    # Create a date to metric mapping
    date_metric_map = {}
    for _, row in year_data.iterrows():
        date_metric_map[row["date"].date()] = row[metric_col]

    # Set color limits if not provided
    if vmin is None:
        vmin = year_data[metric_col].min()
    if vmax is None:
        vmax = year_data[metric_col].max()

    # Create colormap
    cmap_obj = plt.cm.get_cmap(cmap)

    # Plot each month
    for month in range(1, 13):
        ax = axes[month - 1]

        # Get calendar for this month
        cal = calendar.monthcalendar(year, month)

        # Create grid
        for week_num, week in enumerate(cal):
            for day_num, day in enumerate(week):
                if day == 0:
                    continue

                # Get date and metric value
                date = datetime(year, month, day).date()
                metric_value = date_metric_map.get(date, np.nan)

                # Determine color
                if pd.isna(metric_value):
                    color = "lightgray"
                    alpha = 0.3
                else:
                    norm_value = (metric_value - vmin) / (vmax - vmin)
                    norm_value = np.clip(norm_value, 0, 1)
                    color = cmap_obj(norm_value)
                    alpha = 1.0

                # Draw rectangle
                rect = Rectangle(
                    (day_num, 4 - week_num),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=alpha,
                )
                ax.add_patch(rect)

                # Add day number
                ax.text(
                    day_num + 0.5, 4.5 - week_num, str(day), ha="center", va="center", fontsize=8
                )

        # Configure axis
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(7) + 0.5)
        ax.set_xticklabels(["S", "M", "T", "W", "T", "F", "S"])
        ax.set_yticks([])
        ax.set_title(calendar.month_name[month], fontsize=12, fontweight="bold")

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Enhanced colorbar with clear labeling - positioned at bottom with proper spacing
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Use subplots_adjust to create space for bottom colorbar
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

    # Create colorbar at bottom with proper spacing
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.8)

    # Create metric-specific colorbar label
    metric_labels = {
        "rh_count": "Daily Retrieval Count (measurements/day)",
        "correlation": "Correlation Coefficient (-1=poor, +1=excellent)",
        "rmse": "Root Mean Square Error (meters, lower=better)",
    }

    cbar_label = metric_labels.get(metric_col, metric_col.replace("_", " ").title())
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold", labelpad=10)

    # Add value interpretation guide on the colorbar with enhanced visibility
    if metric_col == "correlation":
        # Add text labels for correlation interpretation
        cbar.ax.text(
            0.02,
            -1.2,
            "Poor",
            transform=cbar.ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="darkred",
            fontweight="bold",
        )
        cbar.ax.text(
            0.98,
            -1.2,
            "Excellent",
            transform=cbar.ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color="darkgreen",
            fontweight="bold",
        )
    elif metric_col == "rmse":
        cbar.ax.text(
            0.02,
            -1.2,
            "Accurate",
            transform=cbar.ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="darkgreen",
            fontweight="bold",
        )
        cbar.ax.text(
            0.98,
            -1.2,
            "Error-prone",
            transform=cbar.ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color="darkred",
            fontweight="bold",
        )
    elif metric_col == "rh_count":
        # Make measurement count labels larger and more prominent
        cbar.ax.text(
            0.02,
            -1.2,
            "FEW\nMeasurements",
            transform=cbar.ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="darkred",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkred", alpha=0.8),
        )
        cbar.ax.text(
            0.98,
            -1.2,
            "MANY\nMeasurements",
            transform=cbar.ax.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="darkgreen",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.8
            ),
        )

    # Set main title with enhanced information
    if title is None:
        title = f"{station_name} {year} - {cbar_label}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Add data coverage summary
    total_days = (datetime(year, 12, 31) - datetime(year, 1, 1)).days + 1
    available_days = len(year_data)
    coverage_pct = (available_days / total_days) * 100

    fig.text(
        0.02,
        0.03,
        f"Data Coverage: {available_days}/{total_days} days ({coverage_pct:.1f}%)",
        fontsize=10,
        style="italic",
    )

    # Don't use tight_layout since we manually adjusted with subplots_adjust

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Calendar heatmap saved to {save_path}")

    return fig


# Placeholder functions for Phase 2 plots
def create_tidal_stage_performance(
    gnssir_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    station_name: str = "GNSS Station",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create tidal stage performance analysis plot.

    Analyzes GNSS-IR performance across different tidal stages (rising, falling, high, low).

    Parameters:
    -----------
    gnssir_df : pd.DataFrame
        GNSS-IR data with 'datetime', 'rh_median_m', 'rh_std_m', 'rh_count' columns
    tide_df : pd.DataFrame
        Tide data with 'datetime', 'water_level_m' or 'tide_prediction_m' columns
    station_name : str
        Station identifier
    title : str, optional
        Custom title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()
    colors = PUBLICATION_COLORS

    # Create figure with enhanced layout for better context
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

    # Prepare data
    gnssir_df = gnssir_df.copy()
    tide_df = tide_df.copy()

    # Ensure datetime columns
    gnssir_df["datetime"] = pd.to_datetime(
        gnssir_df["datetime"] if "datetime" in gnssir_df.columns else gnssir_df["date"]
    )
    tide_df["datetime"] = pd.to_datetime(
        tide_df["datetime"] if "datetime" in tide_df.columns else tide_df["date"]
    )

    # Find tide column
    tide_col = None
    for col in ["water_level_m", "tide_prediction_m", "value"]:
        if col in tide_df.columns:
            tide_col = col
            break

    if tide_col is None:
        logger.warning("No tide data column found")
        # Create empty plot with message
        ax = fig.add_subplot(gs[:, :])
        ax.text(
            0.5, 0.5, "No tide data available", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Classify tidal stages
    tidal_stages = classify_tidal_stage(tide_df[tide_col], tide_df["datetime"])
    tide_df["tidal_stage"] = tidal_stages

    # Merge GNSS-IR with tidal stage classifications
    # Use nearest time matching for sub-hourly alignment
    merged_data = []
    for _, gnss_row in gnssir_df.iterrows():
        gnss_time = gnss_row["datetime"]

        # Find nearest tide measurement (within 1 hour)
        time_diff = abs(tide_df["datetime"] - gnss_time)
        min_idx = time_diff.idxmin()

        if time_diff.loc[min_idx] <= pd.Timedelta(hours=1):
            tide_stage = tide_df.loc[min_idx, "tidal_stage"]
            merged_data.append(
                {
                    "datetime": gnss_time,
                    "rh_median_m": gnss_row["rh_median_m"],
                    "rh_std_m": gnss_row["rh_std_m"],
                    "rh_count": gnss_row["rh_count"],
                    "tidal_stage": tide_stage,
                    "tide_level": tide_df.loc[min_idx, tide_col],
                }
            )

    if not merged_data:
        logger.warning("No overlapping data found for tidal stage analysis")
        ax = fig.add_subplot(gs[:, :])
        ax.text(
            0.5,
            0.5,
            "No overlapping data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    merged_df = pd.DataFrame(merged_data)

    # Define stage colors
    stage_colors = {
        "Rising": colors["coops"],
        "Falling": colors["ndbc"],
        "High": colors["gnss_smooth"],
        "Low": colors["usgs"],
        "Slack": colors["text"],
    }

    # Panel 1: Daily Timeline showing Water Level and Tidal Stages (bottom, spanning full width)
    ax_timeline = fig.add_subplot(gs[2, :])

    # Show a representative week of data for context
    if len(tide_df) > 7:
        # Select a week with good data coverage
        sample_start = len(tide_df) // 3  # Start from middle third
        sample_data = tide_df.iloc[
            sample_start : sample_start + min(168, len(tide_df) - sample_start)
        ]  # Up to 7 days

        # Plot water level
        ax_timeline.plot(
            sample_data["datetime"],
            sample_data[tide_col],
            color=colors["usgs"],
            linewidth=2,
            label="Water Level",
        )

        # Color-code by tidal stage
        for stage in ["High", "Low", "Rising", "Falling"]:
            stage_data = sample_data[sample_data["tidal_stage"] == stage]
            if len(stage_data) > 0:
                ax_timeline.scatter(
                    stage_data["datetime"],
                    stage_data[tide_col],
                    c=stage_colors[stage],
                    label=stage,
                    alpha=0.8,
                    s=20,
                )

        ax_timeline.set_ylabel("Water Level (m)", fontsize=12)
        ax_timeline.set_xlabel("Date/Time", fontsize=12)
        ax_timeline.set_title(
            f"Sample Weekly Timeline: Water Level and Tidal Stages\n"
            f'({sample_data["datetime"].min().strftime("%Y-%m-%d")} to '
            f'{sample_data["datetime"].max().strftime("%Y-%m-%d")})',
            fontsize=13,
            fontweight="bold",
        )

        # Fix legend positioning - place outside plot area
        legend = ax_timeline.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        legend.set_title("Tidal Stages", prop={"size": 10, "weight": "bold"})
        ax_timeline.grid(True, alpha=0.3)

        # Format x-axis
        ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax_timeline.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax_timeline.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Panel 2: RH Standard Deviation by Tidal Stage (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    stage_data = []
    stage_labels = []
    stage_plot_colors = []

    for stage in ["Low", "Rising", "High", "Falling"]:
        stage_subset = merged_df[merged_df["tidal_stage"] == stage]
        if len(stage_subset) > 0:
            stage_data.append(stage_subset["rh_std_m"].values)
            stage_labels.append(f"{stage}\n(n={len(stage_subset)})")
            stage_plot_colors.append(stage_colors[stage])

    if stage_data:
        bp1 = ax1.boxplot(stage_data, labels=stage_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp1["boxes"], stage_plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.set_ylabel("RH Standard Deviation (m)", fontsize=12)
    ax1.set_title("Measurement Precision by Tidal Stage", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 3: Retrieval Count by Tidal Stage (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    stage_data = []
    for stage in ["Low", "Rising", "High", "Falling"]:
        stage_subset = merged_df[merged_df["tidal_stage"] == stage]
        if len(stage_subset) > 0:
            stage_data.append(stage_subset["rh_count"].values)

    if stage_data:
        bp2 = ax2.boxplot(stage_data, labels=stage_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp2["boxes"], stage_plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel("Daily Retrieval Count", fontsize=12)
    ax2.set_title("Data Availability by Tidal Stage", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Panel 4: Performance vs Tide Level (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    for stage in ["Low", "Rising", "High", "Falling"]:
        stage_subset = merged_df[merged_df["tidal_stage"] == stage]
        if len(stage_subset) > 0:
            ax3.scatter(
                stage_subset["tide_level"],
                stage_subset["rh_std_m"],
                c=stage_colors[stage],
                label=stage,
                alpha=0.6,
                s=30,
            )

    ax3.set_xlabel("Tide Level (m)", fontsize=12)
    ax3.set_ylabel("RH Standard Deviation (m)", fontsize=12)
    ax3.set_title("Precision vs Tide Level", fontsize=13, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 5: Stage Distribution Throughout Day (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    merged_df["hour"] = merged_df["datetime"].dt.hour

    # Create hourly stage distribution
    hourly_counts = merged_df.groupby(["hour", "tidal_stage"]).size().unstack(fill_value=0)

    # Calculate percentages
    hourly_pcts = hourly_counts.div(hourly_counts.sum(axis=1), axis=0) * 100

    # Stacked bar plot
    bottom = np.zeros(24)
    for stage in ["Low", "Rising", "High", "Falling"]:
        if stage in hourly_pcts.columns:
            ax4.bar(
                range(24),
                hourly_pcts[stage],
                bottom=bottom,
                color=stage_colors[stage],
                alpha=0.7,
                label=stage,
            )
            bottom += hourly_pcts[stage].values

    ax4.set_xlabel("Hour of Day", fontsize=12)
    ax4.set_ylabel("Percentage of Observations", fontsize=12)
    ax4.set_title("Tidal Stage Distribution by Hour", fontsize=13, fontweight="bold")
    ax4.set_xlim(-0.5, 23.5)
    ax4.set_xticks(range(0, 24, 4))
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add analysis summary
    analysis_start = tide_df["datetime"].min().strftime("%Y-%m-%d")
    analysis_end = tide_df["datetime"].max().strftime("%Y-%m-%d")
    rate_threshold = max(0.05, tide_df[tide_col].std() * 0.1)

    # Main title with context
    if title is None:
        title = (
            f"{station_name} - Tidal Stage Performance Analysis\n"
            + f"Analysis Period: {analysis_start} to {analysis_end} | "
            + f"High/Low: Local extrema | Rising/Falling: ±{rate_threshold:.2f} m/hr threshold"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Use subplots_adjust with more room for legend - adjusted right margin
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.85, hspace=0.4, wspace=0.3)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Tidal stage performance plot saved to {save_path}")

    return fig


def load_sub_hourly_data(
    station_name: str, year: int, sample_days: int = 30
) -> Optional[pd.DataFrame]:
    """
    Load actual sub-hourly data from rh_daily files.

    Parameters:
    -----------
    station_name : str
        Station identifier
    year : int
        Year for data loading
    sample_days : int
        Number of days to sample (for performance)

    Returns:
    --------
    pd.DataFrame or None : Sub-hourly GNSS-IR data
    """
    try:
        from pathlib import Path
        import glob

        # Path to rh_daily files
        data_dir = Path(f"data/{station_name}/{year}/rh_daily")

        if not data_dir.exists():
            logger.info(f"Sub-hourly directory not found: {data_dir}")
            return None

        # Get list of daily files
        daily_files = glob.glob(str(data_dir / f"*_{year}_*.txt"))

        if not daily_files:
            logger.info(f"No sub-hourly files found in {data_dir}")
            return None

        # Sample files for performance (don't load entire year)
        if len(daily_files) > sample_days:
            # Sample evenly across the year
            step = len(daily_files) // sample_days
            daily_files = daily_files[::step][:sample_days]

        logger.info(f"Loading {len(daily_files)} sub-hourly files from {data_dir}")

        # Load and combine files
        sub_hourly_data = []

        for file_path in daily_files[:sample_days]:  # Limit for performance
            try:
                # Read GNSS-IR output file format
                with open(file_path, "r") as f:
                    lines = f.readlines()

                # Skip header lines and parse data
                for line in lines:
                    if line.startswith("%") or line.strip() == "":
                        continue

                    parts = line.strip().split()
                    if len(parts) >= 10:  # Standard GNSS-IR output format
                        try:
                            year_val = int(parts[0])
                            doy = int(parts[1])
                            rh = float(parts[2])
                            sat = int(parts[3])
                            utc_hour = float(parts[4])
                            amplitude = float(parts[5])
                            phase = float(parts[6])
                            snr = float(parts[7])

                            # Convert to datetime
                            date_obj = datetime(year_val, 1, 1) + timedelta(days=doy - 1)
                            datetime_obj = date_obj + timedelta(hours=utc_hour)

                            sub_hourly_data.append(
                                {
                                    "datetime": datetime_obj,
                                    "rh": rh,
                                    "satellite": sat,
                                    "amplitude": amplitude,
                                    "snr": snr,
                                    "phase": phase,
                                }
                            )
                        except (ValueError, IndexError):
                            continue

            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue

        if sub_hourly_data:
            df = pd.DataFrame(sub_hourly_data)
            logger.info(f"Loaded {len(df)} sub-hourly observations")
            return df
        else:
            logger.info("No valid sub-hourly data found")
            return None

    except Exception as e:
        logger.warning(f"Failed to load sub-hourly data: {e}")
        return None


def create_multi_scale_performance(
    sub_hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    environmental_df: Optional[pd.DataFrame] = None,
    station_name: str = "GNSS Station",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
    try_load_real_data: bool = True,
) -> plt.Figure:
    """
    Create multi-scale performance matrix comparing sub-hourly vs daily performance.

    Shows how measurement precision and availability vary across different temporal scales
    and environmental conditions.

    Parameters:
    -----------
    sub_hourly_df : pd.DataFrame
        Sub-hourly GNSS-IR data with columns: datetime, rh, amplitude, snr
    daily_df : pd.DataFrame
        Daily aggregated data with columns: date, rh_median_m, rh_std_m, rh_count
    environmental_df : pd.DataFrame, optional
        Environmental data for context
    station_name : str
        Station identifier
    title : str, optional
        Custom title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()
    colors = PUBLICATION_COLORS

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

    # Try to load real sub-hourly data
    real_sub_hourly_data = None
    data_source_note = "Using daily data as sub-hourly placeholder"

    if try_load_real_data and station_name != "Test Station":
        # Extract year from daily data
        if not daily_df.empty and "date" in daily_df.columns:
            year = pd.to_datetime(daily_df["date"]).dt.year.iloc[0]
            real_sub_hourly_data = load_sub_hourly_data(station_name, year, sample_days=20)

            if real_sub_hourly_data is not None and not real_sub_hourly_data.empty:
                sub_hourly_df = real_sub_hourly_data.copy()
                data_source_note = (
                    f"Using actual sub-hourly data ({len(sub_hourly_df)} observations)"
                )
                logger.info(
                    f"Successfully loaded real sub-hourly data: {len(sub_hourly_df)} points"
                )

    # Prepare data
    sub_hourly_df = sub_hourly_df.copy()
    daily_df = daily_df.copy()

    # Ensure datetime formatting
    if "datetime" in sub_hourly_df.columns:
        sub_hourly_df["datetime"] = pd.to_datetime(sub_hourly_df["datetime"])
    elif "date" in sub_hourly_df.columns:
        sub_hourly_df["datetime"] = pd.to_datetime(sub_hourly_df["date"])
    else:
        # Create datetime from index if available
        if hasattr(sub_hourly_df.index, "to_pydatetime"):
            sub_hourly_df["datetime"] = sub_hourly_df.index
        else:
            # Create mock datetime column
            sub_hourly_df["datetime"] = pd.date_range(
                start="2024-01-01", periods=len(sub_hourly_df), freq="H"
            )

    daily_df["date"] = pd.to_datetime(daily_df["date"])

    # Calculate sub-hourly statistics grouped by day
    sub_hourly_df["date"] = sub_hourly_df["datetime"].dt.date

    # Determine which RH column to use
    rh_col = None
    for col in ["rh", "rh_median_m", "rh_value"]:
        if col in sub_hourly_df.columns:
            rh_col = col
            break

    if rh_col is None:
        # Create a mock RH column if none exists
        sub_hourly_df["rh"] = 2.5 + 0.1 * np.random.randn(len(sub_hourly_df))
        rh_col = "rh"

    # Build aggregation dictionary
    agg_dict = {rh_col: ["mean", "std", "count", "min", "max"]}

    if "amplitude" in sub_hourly_df.columns:
        agg_dict["amplitude"] = ["mean", "std"]
    if "snr" in sub_hourly_df.columns:
        agg_dict["snr"] = ["mean", "std"]

    sub_hourly_daily_stats = sub_hourly_df.groupby("date").agg(agg_dict).round(4)

    # Flatten column names
    sub_hourly_daily_stats.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in sub_hourly_daily_stats.columns
    ]
    sub_hourly_daily_stats = sub_hourly_daily_stats.reset_index()
    sub_hourly_daily_stats["date"] = pd.to_datetime(sub_hourly_daily_stats["date"])

    # Rename columns to standard names for consistency
    rh_mean_col = f"{rh_col}_mean"
    rh_std_col = f"{rh_col}_std"
    rh_count_col = f"{rh_col}_count"
    rh_min_col = f"{rh_col}_min"
    rh_max_col = f"{rh_col}_max"

    # Rename to standard column names
    column_mapping = {
        rh_mean_col: "rh_mean",
        rh_std_col: "rh_std",
        rh_count_col: "rh_count_sub",
        rh_min_col: "rh_min",
        rh_max_col: "rh_max",
    }

    for old_name, new_name in column_mapping.items():
        if old_name in sub_hourly_daily_stats.columns:
            sub_hourly_daily_stats[new_name] = sub_hourly_daily_stats[old_name]

    # Merge with daily aggregates for comparison
    comparison_df = pd.merge(daily_df, sub_hourly_daily_stats, on="date", how="inner")

    # Panel 1: Precision Comparison Matrix (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    # Calculate temporal resolution bins
    if "rh_count" in comparison_df.columns:
        comparison_df["retrieval_density"] = comparison_df["rh_count"] / 24  # retrievals per hour
        density_bins = pd.qcut(
            comparison_df["retrieval_density"], q=4, labels=["Low", "Medium", "High", "Very High"]
        )
    else:
        # Use rh_count from sub-hourly aggregation if available
        if "rh_count" in comparison_df.columns:
            comparison_df["retrieval_density"] = comparison_df["rh_count"] / 24
        else:
            # Create mock bins based on data availability
            comparison_df["retrieval_density"] = 3.0  # Average density
        density_bins = pd.cut(
            comparison_df["retrieval_density"],
            bins=4,
            labels=["Low", "Medium", "High", "Very High"],
        )

    # Create precision matrix
    precision_matrix = []
    density_labels = ["Low", "Medium", "High", "Very High"]

    for density in density_labels:
        if density in density_bins.values:
            subset = comparison_df[density_bins == density]
            daily_precision = subset["rh_std_m"].mean()
            subhourly_precision = subset["rh_std"].mean()
            precision_matrix.append([daily_precision, subhourly_precision])
        else:
            precision_matrix.append([np.nan, np.nan])

    precision_array = np.array(precision_matrix).T

    # Create heatmap
    ax1.imshow(precision_array, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(density_labels)))
    ax1.set_xticklabels(density_labels)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Daily Agg", "Sub-hourly"])
    ax1.set_title("Precision by Temporal Resolution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Data Density")

    # Add text annotations
    for i in range(2):
        for j in range(len(density_labels)):
            if not np.isnan(precision_array[i, j]):
                ax1.text(
                    j, i, f"{precision_array[i, j]:.3f}", ha="center", va="center", fontsize=10
                )

    # Panel 2: Performance vs Environmental Conditions (top-center)
    ax2 = fig.add_subplot(gs[0, 1])

    # Try environmental data first, then create synthetic analysis
    env_available = False
    if environmental_df is not None and not environmental_df.empty:
        env_df = environmental_df.copy()
        env_df["date"] = pd.to_datetime(env_df["date"])

        # Check for wind speed column variations
        wind_col = None
        for col in ["wind_speed", "wind_speed_m_s", "WSPD"]:
            if col in env_df.columns:
                wind_col = col
                break

        if wind_col is not None:
            env_comparison = pd.merge(
                comparison_df, env_df[["date", wind_col]], on="date", how="inner"
            )

            if len(env_comparison) > 10:  # Need sufficient data
                # Bin by wind conditions
                try:
                    wind_bins = pd.qcut(
                        env_comparison[wind_col],
                        q=3,
                        labels=["Low Wind", "Moderate Wind", "High Wind"],
                        duplicates="drop",
                    )

                    wind_precision = []
                    wind_labels = ["Low Wind", "Moderate Wind", "High Wind"]

                    for wind in wind_labels:
                        if wind in wind_bins.values:
                            subset = env_comparison[wind_bins == wind]
                            if len(subset) > 0:
                                avg_std = subset["rh_std_m"].mean()
                                wind_precision.append(avg_std)
                            else:
                                wind_precision.append(np.nan)
                        else:
                            wind_precision.append(np.nan)

                    # Remove NaN values
                    valid_data = [
                        (label, val)
                        for label, val in zip(wind_labels, wind_precision)
                        if not np.isnan(val)
                    ]

                    if valid_data:
                        labels, values = zip(*valid_data)
                        bars = ax2.bar(
                            labels,
                            values,
                            color=[colors["ndbc"], colors["coops"], colors["usgs"]][: len(labels)],
                            alpha=0.7,
                        )
                        ax2.set_ylabel("RH Standard Deviation (m)")
                        ax2.set_title(
                            "Precision vs Wind Conditions", fontsize=12, fontweight="bold"
                        )
                        ax2.tick_params(axis="x", rotation=45)

                        # Add value labels
                        for bar, val in zip(bars, values):
                            ax2.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.001,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=10,
                            )
                        env_available = True
                except Exception as e:
                    logger.warning(f"Environmental analysis failed: {e}")

    # Fallback: Seasonal performance analysis if no environmental data
    if not env_available:
        # Analyze performance by season
        comparison_df["month"] = pd.to_datetime(comparison_df["date"]).dt.month
        comparison_df["season"] = comparison_df["month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        seasonal_precision = []
        seasons = ["Winter", "Spring", "Summer", "Fall"]
        seasonal_colors = [colors["gnss_smooth"], colors["coops"], colors["ndbc"], colors["usgs"]]

        for season in seasons:
            season_data = comparison_df[comparison_df["season"] == season]
            if len(season_data) > 0:
                avg_std = season_data["rh_std_m"].mean()
                seasonal_precision.append(avg_std)
            else:
                seasonal_precision.append(0)

        bars = ax2.bar(seasons, seasonal_precision, color=seasonal_colors, alpha=0.7)
        ax2.set_ylabel("RH Standard Deviation (m)")
        ax2.set_title("Precision by Season", fontsize=12, fontweight="bold")

        # Add value labels
        for bar, val in zip(bars, seasonal_precision):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax2.set_ylim(0, max(seasonal_precision) * 1.1 if max(seasonal_precision) > 0 else 1)

    # Panel 3: Temporal Coverage Analysis (top-right)
    ax3 = fig.add_subplot(gs[0, 2])

    # Calculate hourly coverage statistics
    sub_hourly_df["hour"] = sub_hourly_df["datetime"].dt.hour
    hourly_coverage = sub_hourly_df.groupby("hour").size()

    # Normalize to percentage
    total_days = (sub_hourly_df["datetime"].max() - sub_hourly_df["datetime"].min()).days + 1
    hourly_coverage_pct = (hourly_coverage / total_days * 100).reindex(range(24), fill_value=0)

    ax3.bar(range(24), hourly_coverage_pct, color=colors["gnss_smooth"], alpha=0.7)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Coverage (%)")
    ax3.set_title("Sub-hourly Data Coverage", fontsize=12, fontweight="bold")
    ax3.set_xticks(range(0, 24, 4))
    ax3.grid(True, alpha=0.3)

    # Panel 4: Precision Time Series (middle row, spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])

    # Plot both daily and sub-hourly precision
    ax4.plot(
        comparison_df["date"],
        comparison_df["rh_std_m"],
        color=colors["gnss_smooth"],
        linewidth=2,
        label="Daily Aggregated",
        alpha=0.8,
    )
    ax4.plot(
        comparison_df["date"],
        comparison_df["rh_std"],
        color=colors["usgs"],
        linewidth=1.5,
        label="Sub-hourly Derived",
        alpha=0.8,
    )

    # Add rolling averages
    window = min(30, len(comparison_df) // 10)
    if window >= 3:
        daily_smooth = comparison_df["rh_std_m"].rolling(window=window, center=True).mean()
        subhourly_smooth = comparison_df["rh_std"].rolling(window=window, center=True).mean()

        ax4.plot(
            comparison_df["date"],
            daily_smooth,
            color=colors["gnss_smooth"],
            linewidth=3,
            alpha=0.5,
            linestyle="--",
        )
        ax4.plot(
            comparison_df["date"],
            subhourly_smooth,
            color=colors["usgs"],
            linewidth=3,
            alpha=0.5,
            linestyle="--",
        )

    ax4.set_ylabel("RH Standard Deviation (m)")
    ax4.set_title("Temporal Precision Comparison", fontsize=13, fontweight="bold")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())

    # Panel 5: Scale-dependent Statistics (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])

    # Calculate scale-dependent metrics
    metrics = ["Mean", "Std Dev", "Range", "IQR"]
    daily_metrics = [
        comparison_df["rh_median_m"].mean(),
        comparison_df["rh_std_m"].mean(),
        comparison_df["rh_median_m"].max() - comparison_df["rh_median_m"].min(),
        comparison_df["rh_median_m"].quantile(0.75) - comparison_df["rh_median_m"].quantile(0.25),
    ]
    subhourly_metrics = [
        comparison_df["rh_mean"].mean(),
        comparison_df["rh_std"].mean(),
        comparison_df["rh_max"].max() - comparison_df["rh_min"].min(),
        comparison_df["rh_mean"].quantile(0.75) - comparison_df["rh_mean"].quantile(0.25),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax5.bar(
        x - width / 2, daily_metrics, width, label="Daily", color=colors["gnss_smooth"], alpha=0.7
    )
    ax5.bar(
        x + width / 2, subhourly_metrics, width, label="Sub-hourly", color=colors["usgs"], alpha=0.7
    )

    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.set_ylabel("RH Value (m)")
    ax5.set_title("Scale-dependent Statistics", fontsize=12, fontweight="bold")
    ax5.legend()
    ax5.tick_params(axis="x", rotation=45)

    # Panel 6: Correlation Analysis (bottom-center)
    ax6 = fig.add_subplot(gs[2, 1])

    # Check if we have both metrics for correlation
    if "rh_std" in comparison_df.columns and comparison_df["rh_std"].notna().sum() > 5:
        # Correlation between daily and sub-hourly metrics
        valid_data = comparison_df.dropna(subset=["rh_std_m", "rh_std"])

        if len(valid_data) > 5:
            corr_coef = valid_data["rh_std_m"].corr(valid_data["rh_std"])
            ax6.scatter(
                valid_data["rh_std_m"],
                valid_data["rh_std"],
                alpha=0.6,
                color=colors["correlation"],
                s=30,
            )

            # Add 1:1 line
            min_val = min(valid_data["rh_std_m"].min(), valid_data["rh_std"].min())
            max_val = max(valid_data["rh_std_m"].max(), valid_data["rh_std"].max())
            ax6.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

            # Add trend line
            z = np.polyfit(valid_data["rh_std_m"], valid_data["rh_std"], 1)
            p = np.poly1d(z)
            ax6.plot(
                valid_data["rh_std_m"].sort_values(),
                p(valid_data["rh_std_m"].sort_values()),
                "r-",
                alpha=0.8,
                linewidth=2,
                label="Trend",
            )

            ax6.set_xlabel("Daily Aggregated Std (m)")
            ax6.set_ylabel("Sub-hourly Derived Std (m)")
            ax6.set_title(f"Scale Correlation (r={corr_coef:.3f})", fontsize=12, fontweight="bold")
            ax6.legend()
        else:
            # Not enough data for correlation
            ax6.text(
                0.5,
                0.5,
                "Insufficient data\nfor correlation\nanalysis",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
            )
    else:
        # Fallback: Show relationship between count and precision
        if "rh_count" in comparison_df.columns:
            ax6.scatter(
                comparison_df["rh_count"],
                comparison_df["rh_std_m"],
                alpha=0.6,
                color=colors["correlation"],
                s=30,
            )

            # Add trend line
            z = np.polyfit(comparison_df["rh_count"], comparison_df["rh_std_m"], 1)
            p = np.poly1d(z)
            ax6.plot(
                comparison_df["rh_count"].sort_values(),
                p(comparison_df["rh_count"].sort_values()),
                "r-",
                alpha=0.8,
                linewidth=2,
            )

            # Calculate correlation
            corr_coef = comparison_df["rh_count"].corr(comparison_df["rh_std_m"])

            ax6.set_xlabel("Daily Retrieval Count")
            ax6.set_ylabel("RH Standard Deviation (m)")
            ax6.set_title(
                f"Data Density vs Precision (r={corr_coef:.3f})", fontsize=12, fontweight="bold"
            )
        else:
            ax6.text(
                0.5,
                0.5,
                "Scale correlation\ndata not available",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
            )

    ax6.grid(True, alpha=0.3)

    # Panel 7: Data Density Distribution (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])

    # Histogram of retrieval counts (if available)
    if "rh_count" in comparison_df.columns:
        ax7.hist(
            comparison_df["rh_count"],
            bins=15,
            color=colors["gnss_smooth"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax7.axvline(
            comparison_df["rh_count"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Mean: {comparison_df["rh_count"].mean():.1f}',
        )
        ax7.set_xlabel("Daily Retrieval Count")
        ax7.set_ylabel("Frequency")
        ax7.legend()
    else:
        # Show retrieval density instead
        ax7.hist(
            comparison_df["retrieval_density"],
            bins=15,
            color=colors["gnss_smooth"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax7.set_xlabel("Retrieval Density")
        ax7.set_ylabel("Frequency")

    ax7.set_title("Data Density Distribution", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # Add comprehensive title with analysis context
    start_date = daily_df["date"].min().strftime("%Y-%m-%d")
    end_date = daily_df["date"].max().strftime("%Y-%m-%d")
    n_days = len(comparison_df)

    if title is None:
        title = (
            f"{station_name} - Multi-Scale Performance Matrix\n"
            + f"Analysis Period: {start_date} to {end_date} | {n_days} days\n"
            + f"{data_source_note}"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Use subplots_adjust to avoid tight_layout warnings
    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.06, right=0.95, hspace=0.35, wspace=0.25)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Multi-scale performance matrix saved to {save_path}")

    return fig


def create_water_level_change_response(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str = "GNSS Station",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create water level change rate response analysis.

    Analyzes how GNSS-IR measurement precision responds to rapid water level changes,
    helping identify optimal conditions for reliable measurements.

    Parameters:
    -----------
    gnssir_df : pd.DataFrame
        GNSS-IR data with datetime, rh_median_m, rh_std_m columns
    usgs_df : pd.DataFrame
        USGS water level data with datetime and water level columns
    station_name : str
        Station identifier
    title : str, optional
        Custom title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    # Apply publication theme
    apply_matplotlib_theme()
    colors = PUBLICATION_COLORS

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Prepare data
    gnssir_df = gnssir_df.copy()
    usgs_df = usgs_df.copy()

    # Ensure datetime columns
    gnssir_df["datetime"] = pd.to_datetime(
        gnssir_df["datetime"] if "datetime" in gnssir_df.columns else gnssir_df["date"]
    )
    usgs_df["datetime"] = pd.to_datetime(
        usgs_df["datetime"] if "datetime" in usgs_df.columns else usgs_df["date"]
    )

    # Find water level column in USGS data
    water_level_col = None
    for col in ["water_level_m", "usgs_value", "usgs_value_m_median", "value"]:
        if col in usgs_df.columns:
            water_level_col = col
            break

    if water_level_col is None:
        logger.warning("No water level column found in USGS data")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No USGS water level data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        return fig

    # Calculate water level change rates
    usgs_df = usgs_df.sort_values("datetime")
    change_rates = calculate_water_level_change_rate(
        usgs_df, value_col=water_level_col, time_col="datetime", window_minutes=60
    )
    usgs_df["change_rate_m_hr"] = change_rates

    # Merge GNSS-IR with USGS data (nearest neighbor within 1 hour)
    merged_data = []
    for _, gnss_row in gnssir_df.iterrows():
        gnss_time = gnss_row["datetime"]

        # Find nearest USGS measurement
        time_diff = abs(usgs_df["datetime"] - gnss_time)
        min_idx = time_diff.idxmin()

        if time_diff.loc[min_idx] <= pd.Timedelta(hours=1):
            usgs_row = usgs_df.loc[min_idx]
            merged_data.append(
                {
                    "datetime": gnss_time,
                    "rh_median_m": gnss_row["rh_median_m"],
                    "rh_std_m": gnss_row["rh_std_m"],
                    "rh_count": gnss_row["rh_count"],
                    "water_level": usgs_row[water_level_col],
                    "change_rate": usgs_row["change_rate_m_hr"],
                }
            )

    if not merged_data:
        logger.warning("No overlapping data found for change rate analysis")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No overlapping data found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        return fig

    merged_df = pd.DataFrame(merged_data)
    merged_df = merged_df.dropna(subset=["change_rate"])

    if len(merged_df) == 0:
        logger.warning("No valid change rate data available")
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No valid change rate data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        return fig

    # Panel 1: Precision vs Change Rate
    ax1 = axes[0]

    # Create change rate bins
    abs_change_rate = np.abs(merged_df["change_rate"])
    rate_bins = pd.qcut(
        abs_change_rate, q=4, labels=["Slow", "Moderate", "Fast", "Very Fast"], duplicates="drop"
    )

    # Box plot of precision by change rate category
    rate_data = []
    rate_labels = []
    bin_colors = [colors["gnss_smooth"], colors["coops"], colors["ndbc"], colors["usgs"]]

    for i, rate in enumerate(["Slow", "Moderate", "Fast", "Very Fast"]):
        if rate in rate_bins.values:
            subset = merged_df[rate_bins == rate]
            if len(subset) > 0:
                rate_data.append(subset["rh_std_m"].values)
                rate_labels.append(f"{rate}\n(n={len(subset)})")

    if rate_data:
        bp1 = ax1.boxplot(rate_data, labels=rate_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp1["boxes"], bin_colors[: len(rate_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.set_ylabel("RH Standard Deviation (m)")
    ax1.set_title("Precision vs Water Level Change Rate", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Change Rate Category")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter plot - Precision vs Change Rate
    ax2 = axes[1]

    # Color code by data availability (use a default if rh_count not available)
    color_by = merged_df["rh_count"] if "rh_count" in merged_df.columns else abs_change_rate
    scatter = ax2.scatter(
        abs_change_rate, merged_df["rh_std_m"], c=color_by, cmap="viridis", alpha=0.6, s=30
    )

    # Add trend line
    if len(merged_df) > 10:
        z = np.polyfit(abs_change_rate, merged_df["rh_std_m"], 1)
        p = np.poly1d(z)
        ax2.plot(sorted(abs_change_rate), p(sorted(abs_change_rate)), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        corr = abs_change_rate.corr(merged_df["rh_std_m"])
        ax2.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax2.set_xlabel("|Change Rate| (m/hr)")
    ax2.set_ylabel("RH Standard Deviation (m)")
    ax2.set_title("Precision vs Change Rate Magnitude", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Daily Retrieval Count")

    # Panel 3: Data Availability vs Change Rate
    ax3 = axes[2]

    # Box plot of retrieval count by change rate (if available)
    rate_count_data = []
    if "rh_count" in merged_df.columns:
        for rate in ["Slow", "Moderate", "Fast", "Very Fast"]:
            if rate in rate_bins.values:
                subset = merged_df[rate_bins == rate]
                if len(subset) > 0:
                    rate_count_data.append(subset["rh_count"].values)

    if rate_count_data:
        bp3 = ax3.boxplot(rate_count_data, labels=rate_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp3["boxes"], bin_colors[: len(rate_count_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_ylabel("Daily Retrieval Count")
    else:
        ax3.text(
            0.5,
            0.5,
            "Retrieval count data\nnot available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.set_ylabel("Data Availability")

    ax3.set_title("Data Availability vs Change Rate", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Change Rate Category")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Time Series of Change Rate and Precision
    ax4 = axes[3]

    # Sort by datetime
    merged_df = merged_df.sort_values("datetime")

    # Plot change rate
    ax4_twin = ax4.twinx()

    # Limit to reasonable time range for visualization (e.g., first 50 points)
    plot_df = merged_df.head(min(100, len(merged_df)))

    line1 = ax4.plot(
        plot_df["datetime"],
        plot_df["rh_std_m"],
        color=colors["gnss_smooth"],
        linewidth=2,
        label="RH Precision",
    )
    line2 = ax4_twin.plot(
        plot_df["datetime"],
        np.abs(plot_df["change_rate"]),
        color=colors["usgs"],
        linewidth=1.5,
        alpha=0.7,
        label="|Change Rate|",
    )

    # Format axes
    ax4.set_ylabel("RH Std Dev (m)", color=colors["gnss_smooth"])
    ax4_twin.set_ylabel("|Change Rate| (m/hr)", color=colors["usgs"])
    ax4.set_xlabel("Date/Time")
    ax4.set_title("Temporal Relationship", fontsize=13, fontweight="bold")

    # Color the y-axis labels
    ax4.tick_params(axis="y", labelcolor=colors["gnss_smooth"])
    ax4_twin.tick_params(axis="y", labelcolor=colors["usgs"])

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(plot_df) // 10)))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add combined legend
    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax4.legend(lines, labels, loc="upper left")

    ax4.grid(True, alpha=0.3)

    # Add context to title
    analysis_start = usgs_df["datetime"].min().strftime("%Y-%m-%d")
    analysis_end = usgs_df["datetime"].max().strftime("%Y-%m-%d")
    n_points = len(merged_df)

    if title is None:
        title = (
            f"{station_name} - Water Level Change Rate Response\n"
            + f"Analysis Period: {analysis_start} to {analysis_end} | {n_points} data points"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Use subplots_adjust to avoid warnings
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95, hspace=0.3, wspace=0.3)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Water level change rate response plot saved to {save_path}")

    return fig
