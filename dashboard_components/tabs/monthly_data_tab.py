# ABOUTME: Monthly data visualization tab with toggleable plot types
# ABOUTME: Shows time series, correlations, and comparison plots by month

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import station metadata helper
from dashboard_components.station_metadata import get_antenna_height  # noqa: E402

# Import visualization functions
try:
    from scripts.visualizer.dashboard_plots import (
        create_calendar_heatmap,
        create_monthly_box_plots,
        create_multi_scale_performance,
    )

    DASHBOARD_PLOTS_AVAILABLE = True
    PHASE2_PLOTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dashboard plots not available: {e}")
    DASHBOARD_PLOTS_AVAILABLE = False
    PHASE2_PLOTS_AVAILABLE = False


def render_monthly_data_tab(
    rh_data, usgs_data, coops_data, selected_station, selected_year, include_coops=True
):
    """
    Render the monthly data tab with all visualizations.

    Parameters:
    -----------
    rh_data : pd.DataFrame
        GNSS-IR reflector height data
    usgs_data : pd.DataFrame
        USGS water level data
    coops_data : pd.DataFrame
        NOAA CO-OPS tide data
    selected_station : str
        Station ID
    selected_year : int
        Year for analysis
    include_coops : bool
        Whether CO-OPS data is included
    """
    st.header("📊 Monthly Data Analysis")

    if not DASHBOARD_PLOTS_AVAILABLE:
        st.error("Dashboard plotting functions not available. Please check installation.")
        return

    if rh_data is None or rh_data.empty:
        st.warning("⚠️ No GNSS-IR data available for visualization")
        return

    # Prepare standardized USGS data for all plot types
    usgs_for_plot = None
    if usgs_data is not None and not usgs_data.empty:
        # Create a standardized USGS dataframe for plotting
        usgs_for_plot = usgs_data.copy()

        # Ensure it has the expected column names
        if "water_level_m" not in usgs_for_plot.columns:
            # Find the value column
            value_col = None
            for col in ["usgs_value", "usgs_value_m_median", "value"]:
                if col in usgs_for_plot.columns:
                    value_col = col
                    break

            if value_col:
                usgs_for_plot["water_level_m"] = usgs_for_plot[value_col]

        # Ensure date column exists
        if "date" not in usgs_for_plot.columns:
            if "merge_date" in usgs_for_plot.columns:
                usgs_for_plot["date"] = usgs_for_plot["merge_date"]
            elif "datetime" in usgs_for_plot.columns:
                usgs_for_plot["date"] = usgs_for_plot["datetime"]

    # Calculate performance metrics if USGS data is available
    perf_data = rh_data.copy()
    if usgs_for_plot is not None and not usgs_for_plot.empty:
        # Ensure date columns are datetime for proper merging
        perf_data["date"] = pd.to_datetime(perf_data["date"])
        usgs_for_plot["date"] = pd.to_datetime(usgs_for_plot["date"])

        # Merge GNSS-IR and USGS data for correlation calculation
        merged_perf = pd.merge(
            perf_data[["date", "rh_median_m", "rh_std_m", "rh_count"]],
            usgs_for_plot[["date", "water_level_m"]],
            on="date",
            how="inner",
        )

        # Calculate daily correlation and RMSE
        if len(merged_perf) > 0:
            # Get antenna height from config (no longer hardcoded)
            antenna_height = get_antenna_height(selected_station)

            # Convert Reflector Height to Water Surface Elevation
            # WSE = Antenna Height - RH (since RH is distance DOWN to water)
            merged_perf["wse_ellips_m"] = antenna_height - merged_perf["rh_median_m"]

            # Now calculate RMSE between WSE and water level
            merged_perf["rmse"] = np.sqrt(
                (merged_perf["wse_ellips_m"] - merged_perf["water_level_m"]) ** 2
            )

            # Calculate rolling correlation (30-day window) using WSE, not RH
            # WSE and water level should have POSITIVE correlation
            window_size = min(30, len(merged_perf) // 3)
            if window_size >= 3:
                merged_perf = merged_perf.sort_values("date")
                merged_perf["correlation"] = (
                    merged_perf["wse_ellips_m"]
                    .rolling(window=window_size)
                    .corr(merged_perf["water_level_m"])
                )
            else:
                # Global correlation if not enough data
                global_corr = merged_perf["wse_ellips_m"].corr(merged_perf["water_level_m"])
                merged_perf["correlation"] = global_corr

            perf_data = merged_perf
        else:
            st.info("💡 USGS data not available - showing GNSS-IR data availability metrics only")

    # Visualization Selection - single flat dropdown
    plot_options = ["Calendar Heat Map", "Monthly Box Plots"]
    if PHASE2_PLOTS_AVAILABLE:
        plot_options.append("Multi-Scale Performance Matrix")

    selected_plot = st.selectbox(
        "📊 Select Visualization", plot_options, key="monthly_plot_selection"
    )

    # Render the selected visualization
    if selected_plot == "Calendar Heat Map":
        st.markdown("### 📅 Calendar Heat Map")

        # Metric selection for heat map with clearer descriptions
        available_metrics = ["rh_count"]
        metric_labels = {
            "rh_count": "📊 Daily Retrieval Count (number of successful measurements per day)"
        }

        if "correlation" in perf_data.columns:
            available_metrics.extend(["correlation", "rmse"])
            metric_labels.update(
                {
                    "correlation": "Correlation with USGS (daily coefficient: -1 to +1)",
                    "rmse": "📏 RMSE: WSE vs Water Level (meters) - RH converted to elevation",
                }
            )

        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox(
                "Select Metric:",
                available_metrics,
                format_func=lambda x: metric_labels[x],
                key="heatmap_metric",
            )

        with col2:
            # Color map selection with better descriptions
            color_schemes = {
                "RdYlGn": "🟢 Red-Yellow-Green (Green = Good)",
                "RdYlGn_r": "🔴 Red-Yellow-Green Reversed (Red = Good)",
                "viridis": "🟣 Viridis (Purple-Blue-Green)",
                "Blues": "🔵 Blues (Light to Dark Blue)",
                "Reds": "🔴 Reds (Light to Dark Red)",
                "coolwarm": "❄️🔥 Cool-Warm (Blue-Red)",
            }

            # Auto-select appropriate color scheme based on metric
            if selected_metric == "rmse":
                default_cmap = "RdYlGn_r"  # Reversed - lower RMSE is better (green)
                st.info("💡 For RMSE: Green = Low Error (Good), Red = High Error (Bad)")
            elif selected_metric == "correlation":
                default_cmap = "RdYlGn"  # Higher correlation is better (green)
                st.info("For Correlation: Green = High (Good), Red = Low (Bad)")
            else:
                default_cmap = "RdYlGn"  # Higher count is generally better
                st.info("For Retrieval Count: Green = Many (Good), Red = Few (Poor)")

            cmap_options = list(color_schemes.keys())
            default_index = cmap_options.index(default_cmap) if default_cmap in cmap_options else 0

            selected_cmap = st.selectbox(
                "Color Scheme:",
                cmap_options,
                format_func=lambda x: color_schemes[x],
                index=default_index,
                key="heatmap_cmap",
            )

        # Show conversion note if RMSE is selected
        if selected_metric == "rmse":
            ant_height = get_antenna_height(selected_station)
            st.info(f"**RH to WSE Conversion**: WSE = Antenna Height ({ant_height:.3f} m) - RH")

        # Create calendar heat map
        fig = create_calendar_heatmap(
            perf_data,
            metric_col=selected_metric,
            year=selected_year,
            station_name=selected_station,
            cmap=selected_cmap,
            figsize=(16, 12),  # Made taller to accommodate colorbar properly
        )

        st.pyplot(fig)
        plt.close()

    elif selected_plot == "Monthly Box Plots":
        st.markdown("### 📊 Monthly Performance Distribution")

        # Metric selection for box plots
        available_metrics = ["rh_std_m", "rh_count"]
        metric_labels = {
            "rh_std_m": "RH Standard Deviation (m)",
            "rh_count": "Daily Retrieval Count",
        }

        if "correlation" in perf_data.columns:
            available_metrics.extend(["correlation", "rmse"])
            metric_labels.update({"correlation": "Correlation with USGS", "rmse": "RMSE (m)"})

        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox(
                "Select Metric:",
                available_metrics,
                format_func=lambda x: metric_labels[x],
                key="boxplot_metric",
            )

        with col2:
            show_points = st.checkbox(
                "Show individual data points", value=True, key="boxplot_points"
            )

        # Create monthly box plots
        fig = create_monthly_box_plots(
            perf_data,
            metric_col=selected_metric,
            station_name=selected_station,
            year=selected_year,
            ylabel=metric_labels[selected_metric],
            add_points=show_points,
            figsize=(14, 8),
        )

        st.pyplot(fig)
        plt.close()

    elif selected_plot == "Multi-Scale Performance Matrix" and PHASE2_PLOTS_AVAILABLE:
        st.markdown("### 📊 Multi-Scale Performance Matrix")

        # Check if we have sub-hourly data
        st.info("Note: Using daily data as placeholder for sub-hourly analysis")

        # For demonstration, use daily data as sub-hourly placeholder
        sub_hourly_data = perf_data.copy()

        # Create multi-scale performance matrix
        fig = create_multi_scale_performance(
            sub_hourly_data,
            perf_data,
            environmental_df=None,
            station_name=selected_station,
            figsize=(16, 12),
            try_load_real_data=True,  # Try to load actual sub-hourly data
        )

        st.pyplot(fig)
        plt.close()

        st.markdown(
            """
        **Analysis**: This matrix compares performance across different temporal scales:

        **Multi-Scale Comparison:**
        - **Precision Matrix (top-left)**: How uncertainty varies with data density
        - **Environmental/Seasonal Effects (top-center)**: Wind/seasonal impact on precision
        - **Temporal Coverage (top-right)**: Hourly data availability patterns
        - **Scale Correlation (bottom-center)**: Relationship between temporal scales
        - **Time Series (middle)**: Daily vs sub-hourly precision comparison

        **Key Insights:**
        - Shows whether higher temporal resolution improves measurement precision
        - Identifies optimal conditions for reliable measurements
        - Reveals temporal patterns in data quality and availability
        - Compares different aggregation methods for GNSS-IR data

        **Data Sources**: The plot automatically tries to load actual sub-hourly data from
        the `rh_daily/` directory. If unavailable, it uses daily data as comparison baseline.
        Check the plot title for data source information.
        """
        )

    # Summary statistics for selected visualization
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Processing Days", len(perf_data))

    with col2:
        avg_retrievals = perf_data["rh_count"].mean()
        st.metric("Avg Daily Retrievals", f"{avg_retrievals:.1f}")

    with col3:
        if "correlation" in perf_data.columns:
            avg_corr = perf_data["correlation"].mean()
            st.metric("Avg Correlation", f"{avg_corr:.3f}")
        else:
            rh_range = perf_data["rh_median_m"].max() - perf_data["rh_median_m"].min()
            st.metric("RH Range (m)", f"{rh_range:.2f}")

    with col4:
        data_coverage = len(perf_data) / 365 * 100
        st.metric("Data Coverage", f"{data_coverage:.1f}%")


# Export the render function
__all__ = ["render_monthly_data_tab"]
