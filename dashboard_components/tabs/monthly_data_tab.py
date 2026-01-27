"""
Monthly Data Tab Implementation

This module implements the consolidated monthly data tab that contains
all visualization types with toggleable selection.
"""

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
from dashboard_components.station_metadata import get_antenna_height

# Import visualization functions
try:
    from scripts.visualizer.dashboard_plots import (
        create_calendar_heatmap,
        create_monthly_box_plots,
        create_multi_parameter_timeline,
        create_tidal_stage_performance,
        create_multi_scale_performance,
        create_water_level_change_response
    )
    DASHBOARD_PLOTS_AVAILABLE = True
    PHASE2_PLOTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dashboard plots not available: {e}")
    DASHBOARD_PLOTS_AVAILABLE = False
    PHASE2_PLOTS_AVAILABLE = False


def render_monthly_data_tab(rh_data, usgs_data, coops_data, ndbc_data, 
                           selected_station, selected_year, include_coops=True, include_ndbc=True):
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
    ndbc_data : pd.DataFrame
        NDBC buoy data
    selected_station : str
        Station ID
    selected_year : int
        Year for analysis
    include_coops : bool
        Whether CO-OPS data is included
    include_ndbc : bool
        Whether NDBC data is included
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
        if 'water_level_m' not in usgs_for_plot.columns:
            # Find the value column
            value_col = None
            for col in ['usgs_value', 'usgs_value_m_median', 'value']:
                if col in usgs_for_plot.columns:
                    value_col = col
                    break
            
            if value_col:
                usgs_for_plot['water_level_m'] = usgs_for_plot[value_col]
        
        # Ensure date column exists
        if 'date' not in usgs_for_plot.columns:
            if 'merge_date' in usgs_for_plot.columns:
                usgs_for_plot['date'] = usgs_for_plot['merge_date']
            elif 'datetime' in usgs_for_plot.columns:
                usgs_for_plot['date'] = usgs_for_plot['datetime']

    # Calculate performance metrics if USGS data is available
    perf_data = rh_data.copy()
    if usgs_for_plot is not None and not usgs_for_plot.empty:
        # Ensure date columns are datetime for proper merging
        perf_data['date'] = pd.to_datetime(perf_data['date'])
        usgs_for_plot['date'] = pd.to_datetime(usgs_for_plot['date'])
        
        # Merge GNSS-IR and USGS data for correlation calculation
        merged_perf = pd.merge(
            perf_data[['date', 'rh_median_m', 'rh_std_m', 'rh_count']],
            usgs_for_plot[['date', 'water_level_m']],
            on='date',
            how='inner'
        )
        
        # Calculate daily correlation and RMSE
        if len(merged_perf) > 0:
            # Get antenna height from config (no longer hardcoded)
            antenna_height = get_antenna_height(selected_station)
            
            # Convert Reflector Height to Water Surface Elevation
            # WSE = Antenna Height - RH (since RH is distance DOWN to water)
            merged_perf['wse_ellips_m'] = antenna_height - merged_perf['rh_median_m']
            
            # Now calculate RMSE between WSE and water level
            merged_perf['rmse'] = np.sqrt((merged_perf['wse_ellips_m'] - merged_perf['water_level_m'])**2)
            
            # Calculate rolling correlation (30-day window) using WSE, not RH
            # WSE and water level should have POSITIVE correlation
            window_size = min(30, len(merged_perf) // 3)
            if window_size >= 3:
                merged_perf = merged_perf.sort_values('date')
                merged_perf['correlation'] = merged_perf['wse_ellips_m'].rolling(window=window_size).corr(
                    merged_perf['water_level_m']
                )
            else:
                # Global correlation if not enough data
                global_corr = merged_perf['wse_ellips_m'].corr(merged_perf['water_level_m'])
                merged_perf['correlation'] = global_corr
            
            perf_data = merged_perf
        else:
            st.info("💡 USGS data not available - showing GNSS-IR data availability metrics only")

    # Visualization Selection
    st.markdown("### 🎯 Select Visualization Type")
    
    # Create visualization categories
    basic_viz = ["Calendar Heat Map", "Monthly Box Plots"]
    advanced_viz = []
    if PHASE2_PLOTS_AVAILABLE:
        advanced_viz = ["Tidal Stage Performance", "Multi-Scale Performance Matrix", "Water Level Change Rate Response"]
    
    # Create selection interface
    viz_category = st.radio(
        "Visualization Category:",
        ["📊 Basic Analysis", "🔬 Advanced Analysis"] if advanced_viz else ["📊 Basic Analysis"],
        key="viz_category"
    )
    
    if viz_category == "📊 Basic Analysis":
        plot_options = basic_viz
    else:
        plot_options = advanced_viz
    
    selected_plot = st.selectbox(
        "Choose Visualization:",
        plot_options,
        key="monthly_plot_selection"
    )
    
    # Add plot-specific controls
    st.markdown("### ⚙️ Plot Settings")
    
    # Render the selected visualization
    if selected_plot == "Calendar Heat Map":
        st.markdown("### 📅 Calendar Heat Map")
        
        # Metric selection for heat map with clearer descriptions
        available_metrics = ['rh_count']
        metric_labels = {
            'rh_count': '📊 Daily Retrieval Count (number of successful measurements per day)'
        }
        
        if 'correlation' in perf_data.columns:
            available_metrics.extend(['correlation', 'rmse'])
            metric_labels.update({
                'correlation': '📈 Correlation with USGS (daily correlation coefficient: -1 to +1)',
                'rmse': '📏 RMSE: WSE vs Water Level (meters) - RH converted to elevation'
            })
        
        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox(
                "Select Metric:",
                available_metrics,
                format_func=lambda x: metric_labels[x],
                key="heatmap_metric"
            )
        
        with col2:
            # Color map selection with better descriptions
            color_schemes = {
                'RdYlGn': '🟢 Red-Yellow-Green (Green = Good)',
                'RdYlGn_r': '🔴 Red-Yellow-Green Reversed (Red = Good)', 
                'viridis': '🟣 Viridis (Purple-Blue-Green)',
                'Blues': '🔵 Blues (Light to Dark Blue)',
                'Reds': '🔴 Reds (Light to Dark Red)',
                'coolwarm': '❄️🔥 Cool-Warm (Blue-Red)'
            }
            
            # Auto-select appropriate color scheme based on metric
            if selected_metric == 'rmse':
                default_cmap = 'RdYlGn_r'  # Reversed - lower RMSE is better (green)
                st.info("💡 For RMSE: Green = Low Error (Good), Red = High Error (Bad)")
            elif selected_metric == 'correlation':
                default_cmap = 'RdYlGn'  # Higher correlation is better (green)
                st.info("💡 For Correlation: Green = High Correlation (Good), Red = Low Correlation (Bad)")
            else:
                default_cmap = 'RdYlGn'  # Higher count is generally better
                st.info("💡 For Retrieval Count: Green = Many Measurements (Good), Red = Few Measurements (Poor)")
                
            cmap_options = list(color_schemes.keys())
            default_index = cmap_options.index(default_cmap) if default_cmap in cmap_options else 0
            
            selected_cmap = st.selectbox(
                "Color Scheme:",
                cmap_options,
                format_func=lambda x: color_schemes[x],
                index=default_index,
                key="heatmap_cmap"
            )
        
        # Show conversion note if RMSE is selected
        if selected_metric == 'rmse':
            st.info(f"📐 **RH to WSE Conversion**: Water Surface Elevation = Antenna Height ({get_antenna_height(selected_station):.3f} m) - Reflector Height")
        
        # Create calendar heat map
        fig = create_calendar_heatmap(
            perf_data,
            metric_col=selected_metric,
            year=selected_year,
            station_name=selected_station,
            cmap=selected_cmap,
            figsize=(16, 12)  # Made taller to accommodate colorbar properly
        )
        
        st.pyplot(fig)
        plt.close()
        
    elif selected_plot == "Monthly Box Plots":
        st.markdown("### 📊 Monthly Performance Distribution")
        
        # Metric selection for box plots
        available_metrics = ['rh_std_m', 'rh_count']
        metric_labels = {
            'rh_std_m': 'RH Standard Deviation (m)',
            'rh_count': 'Daily Retrieval Count'
        }
        
        if 'correlation' in perf_data.columns:
            available_metrics.extend(['correlation', 'rmse'])
            metric_labels.update({
                'correlation': 'Correlation with USGS',
                'rmse': 'RMSE (m)'
            })
        
        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox(
                "Select Metric:",
                available_metrics,
                format_func=lambda x: metric_labels[x],
                key="boxplot_metric"
            )
        
        with col2:
            show_points = st.checkbox("Show individual data points", value=True, key="boxplot_points")
        
        # Create monthly box plots
        fig = create_monthly_box_plots(
            perf_data,
            metric_col=selected_metric,
            station_name=selected_station,
            year=selected_year,
            ylabel=metric_labels[selected_metric],
            add_points=show_points,
            figsize=(14, 8)
        )
        
        st.pyplot(fig)
        plt.close()
        
    elif selected_plot == "Multi-Parameter Timeline":
        st.markdown("### 📈 Multi-Parameter Annual Timeline")
        
        # Prepare environmental data if available
        env_data = None
        if include_ndbc and ndbc_data is not None and not ndbc_data.empty:
            # Use NDBC data for environmental context
            env_data = ndbc_data[['date', 'wind_speed_m_s', 'wave_height_m']].copy()
            env_data.columns = ['date', 'wind_speed', 'wave_height']
        
        # Use CO-OPS data if available and no USGS data
        water_level_data = usgs_for_plot
        if (usgs_for_plot is None or usgs_for_plot.empty) and coops_data is not None and not coops_data.empty:
            water_level_data = coops_data
            st.info("Using CO-OPS tide data for water level comparison")
        elif coops_data is not None and not coops_data.empty:
            # Show option to use CO-OPS instead of USGS
            use_coops = st.checkbox("Use CO-OPS data instead of USGS", value=False, key="timeline_coops")
            if use_coops:
                water_level_data = coops_data
        
        # Rolling window control
        rolling_window = st.slider("Rolling window (days):", 7, 60, 30, key="timeline_window")
        
        # Create multi-parameter timeline
        fig = create_multi_parameter_timeline(
            perf_data,
            usgs_df=water_level_data,
            environmental_df=env_data,
            station_name=selected_station,
            year=selected_year,
            rolling_window=rolling_window,
            figsize=(16, 12)
        )
        
        st.pyplot(fig)
        plt.close()
        
    elif selected_plot == "Tidal Stage Performance" and PHASE2_PLOTS_AVAILABLE:
        st.markdown("### 🌊 Tidal Stage Performance Analysis")
        
        # Use CO-OPS data if available, otherwise USGS
        tide_data = coops_data if coops_data is not None and not coops_data.empty else usgs_for_plot
        
        if tide_data is not None and not tide_data.empty:
            # Create tidal stage performance plot
            fig = create_tidal_stage_performance(
                perf_data,
                tide_data,
                station_name=selected_station,
                figsize=(14, 10)
            )
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Analysis**: This plot shows how GNSS-IR measurement precision varies across different tidal stages:
            
            **🌊 Tidal Stage Definitions:**
            - **High/Low**: Local water level extrema (peaks and troughs in the time series)
            - **Rising/Falling**: Based on **water level change rate thresholds** measured in **meters per hour (m/hr)**
            - **Slack**: Transition periods with minimal water level change (rate ≈ 0 m/hr)
            
            **📊 Key Insights:**
            - **Bottom Timeline**: Shows a sample week of water level data with color-coded tidal stages
            - **Precision Analysis**: How measurement uncertainty varies with tidal conditions
            - **Data Availability**: Whether tidal stage affects successful data retrieval
            - **Temporal Patterns**: When different tidal stages occur throughout the day
            
            **⚙️ Method**: Tidal stages are determined from water level time series using adaptive rate thresholds 
            (typically ±0.05 m/hr minimum) based on local data characteristics, not external tide predictions.
            The **legend is positioned outside the plot area** for better readability.
            """)
        else:
            st.warning("⚠️ Tide/water level data required for tidal stage analysis")
    
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
            environmental_df=env_data if include_ndbc and ndbc_data is not None else None,
            station_name=selected_station,
            figsize=(16, 12),
            try_load_real_data=True  # Try to load actual sub-hourly data
        )
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Analysis**: This matrix compares performance across different temporal scales:
        
        **📊 Multi-Scale Comparison:**
        - **Precision Matrix (top-left)**: How measurement uncertainty varies with data density
        - **Environmental/Seasonal Effects (top-center)**: Wind conditions or seasonal patterns impact on precision  
        - **Temporal Coverage (top-right)**: Hourly data availability patterns
        - **Scale Correlation (bottom-center)**: Relationship between different temporal scales
        - **Time Series (middle)**: Direct comparison of daily vs sub-hourly precision
        
        **🔍 Key Insights:**
        - Shows whether higher temporal resolution improves measurement precision
        - Identifies optimal conditions for reliable measurements
        - Reveals temporal patterns in data quality and availability
        - Compares different aggregation methods for GNSS-IR data
        
        **📁 Data Sources**: The plot automatically tries to load actual sub-hourly data from 
        the `rh_daily/` directory. If unavailable, it uses daily data as a comparison baseline.
        Check the plot title for data source information.
        """)
    
    elif selected_plot == "Water Level Change Rate Response" and PHASE2_PLOTS_AVAILABLE:
        st.markdown("### 🌊 Water Level Change Rate Response")
        
        if usgs_for_plot is not None and not usgs_for_plot.empty:
            # Create water level change rate response plot
            fig = create_water_level_change_response(
                perf_data,
                usgs_for_plot,
                station_name=selected_station,
                figsize=(14, 10)
            )
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Analysis**: This analysis shows how GNSS-IR responds to rapid water level changes:
            - **Precision vs Change Rate**: How measurement quality varies with water level dynamics
            - **Data Availability**: Whether rapid changes affect data collection success
            - **Temporal Patterns**: When rapid changes occur and their impact
            - **Operational Guidance**: Optimal conditions for reliable measurements
            """)
        else:
            st.warning("⚠️ USGS water level data required for change rate analysis")
    
    # Summary statistics for selected visualization
    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Days", len(perf_data))
        
    with col2:
        avg_retrievals = perf_data['rh_count'].mean()
        st.metric("Avg Daily Retrievals", f"{avg_retrievals:.1f}")
        
    with col3:
        if 'correlation' in perf_data.columns:
            avg_corr = perf_data['correlation'].mean()
            st.metric("Avg Correlation", f"{avg_corr:.3f}")
        else:
            rh_range = perf_data['rh_median_m'].max() - perf_data['rh_median_m'].min()
            st.metric("RH Range (m)", f"{rh_range:.2f}")
            
    with col4:
        data_coverage = len(perf_data) / 365 * 100
        st.metric("Data Coverage", f"{data_coverage:.1f}%")


# Export the render function
__all__ = ['render_monthly_data_tab']