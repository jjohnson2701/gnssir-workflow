# ABOUTME: Yearly residual analysis tab for GNSS-IR vs reference comparison
# ABOUTME: Supports USGS, CO-OPS, and ERDDAP reference sources

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from datetime import datetime, timedelta
from scipy import interpolate
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import station metadata helper
from dashboard_components.station_metadata import get_antenna_height

# Import visualization functions
try:
    from scripts.visualizer.publication_theme import apply_matplotlib_theme, PUBLICATION_COLORS
    PUBLICATION_THEME_AVAILABLE = True
except ImportError:
    PUBLICATION_THEME_AVAILABLE = False

try:
    from scripts.visualizer.dashboard_plots import create_multi_parameter_timeline
    DASHBOARD_PLOTS_AVAILABLE = True
except ImportError:
    DASHBOARD_PLOTS_AVAILABLE = False


def preprocess_for_residual_analysis(
    gnss_data: pd.DataFrame, 
    usgs_data: pd.DataFrame,
    target_resolution: str = 'D',
    station_id: str = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocess and align GNSS-IR and USGS data for residual analysis.
    
    Parameters:
    -----------
    gnss_data : pd.DataFrame
        GNSS-IR data with date, rh_median_m columns
    usgs_data : pd.DataFrame  
        USGS data with date/datetime, water_level_m columns
    target_resolution : str
        Target temporal resolution ('D' for daily, 'H' for hourly)
        
    Returns:
    --------
    aligned_data : pd.DataFrame
        Temporally aligned data with columns: timestamp, gnss_water_level, 
        ref_water_level, difference, uncertainty
    metadata : dict
        Processing metadata and statistics
    """
    # Standardize date columns
    gnss_df = gnss_data.copy()
    usgs_df = usgs_data.copy()
    
    # Ensure datetime columns
    if 'date' in gnss_df.columns:
        gnss_df['timestamp'] = pd.to_datetime(gnss_df['date'])
    elif 'datetime' in gnss_df.columns:
        gnss_df['timestamp'] = pd.to_datetime(gnss_df['datetime'])
    
    if 'date' in usgs_df.columns:
        usgs_df['timestamp'] = pd.to_datetime(usgs_df['date'])
    elif 'datetime' in usgs_df.columns:
        usgs_df['timestamp'] = pd.to_datetime(usgs_df['datetime'])
    
    # Identify water level columns
    gnss_wl_col = None
    for col in ['rh_median_m', 'water_level_m', 'wse_ellips']:
        if col in gnss_df.columns:
            gnss_wl_col = col
            break
    
    # Reference water level column - check known names first, then generic
    ref_wl_col = None
    for col in ['water_level_m', 'usgs_value', 'value', 'coops_value']:
        if col in usgs_df.columns:
            ref_wl_col = col
            break
    # If not found, look for ERDDAP-style columns (*_wl that isn't gnss_*)
    if ref_wl_col is None:
        wl_cols = [col for col in usgs_df.columns
                   if col.endswith('_wl') and not col.startswith('gnss')]
        if wl_cols:
            ref_wl_col = wl_cols[0]
    
    if gnss_wl_col is None or ref_wl_col is None:
        st.error("Could not identify water level columns in data")
        return pd.DataFrame(), {}
    
    # Convert GNSS reflector heights to water levels if needed
    # Note: This assumes RH values need to be inverted to water levels
    if gnss_wl_col == 'rh_median_m':
        # Get antenna height from config (no longer hardcoded)
        antenna_height = get_antenna_height(station_id) if station_id else 0.0
        
        # Convert Reflector Height to Water Surface Elevation
        # WSE = Antenna Height - RH (since RH is distance DOWN to water)
        gnss_df['gnss_water_level'] = antenna_height - gnss_df[gnss_wl_col]
        conversion_note = f"RH converted to WSE using antenna height: {antenna_height:.3f} m"
    else:
        gnss_df['gnss_water_level'] = gnss_df[gnss_wl_col]
        conversion_note = "Using pre-computed water surface elevation"
    
    usgs_df['ref_water_level'] = usgs_df[ref_wl_col]
    
    # Create common time grid
    start_date = max(gnss_df['timestamp'].min(), usgs_df['timestamp'].min())
    end_date = min(gnss_df['timestamp'].max(), usgs_df['timestamp'].max())
    
    if target_resolution == 'D':
        time_grid = pd.date_range(start_date.date(), end_date.date(), freq='D')
    else:
        time_grid = pd.date_range(start_date, end_date, freq=target_resolution)
    
    # Interpolate both datasets to common grid
    aligned_data = pd.DataFrame({'timestamp': time_grid})
    
    # GNSS interpolation
    gnss_clean = gnss_df.dropna(subset=['gnss_water_level']).sort_values('timestamp')
    if len(gnss_clean) > 1:
        gnss_interp = interpolate.interp1d(
            gnss_clean['timestamp'].astype(np.int64),
            gnss_clean['gnss_water_level'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        aligned_data['gnss_water_level'] = gnss_interp(time_grid.astype(np.int64))
    else:
        aligned_data['gnss_water_level'] = np.nan
    
    # USGS interpolation
    usgs_clean = usgs_df.dropna(subset=['ref_water_level']).sort_values('timestamp')
    if len(usgs_clean) > 1:
        usgs_interp = interpolate.interp1d(
            usgs_clean['timestamp'].astype(np.int64),
            usgs_clean['ref_water_level'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        aligned_data['ref_water_level'] = usgs_interp(time_grid.astype(np.int64))
    else:
        aligned_data['ref_water_level'] = np.nan
    
    # Calculate differences and statistics
    aligned_data['difference'] = aligned_data['gnss_water_level'] - aligned_data['ref_water_level']
    
    # Estimate uncertainty (simple approach using local variability)
    if 'rh_std_m' in gnss_df.columns:
        # Use GNSS measurement uncertainty if available
        gnss_unc_clean = gnss_df.dropna(subset=['rh_std_m']).sort_values('timestamp')
        if len(gnss_unc_clean) > 1:
            unc_interp = interpolate.interp1d(
                gnss_unc_clean['timestamp'].astype(np.int64),
                gnss_unc_clean['rh_std_m'],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            aligned_data['uncertainty'] = unc_interp(time_grid.astype(np.int64))
        else:
            aligned_data['uncertainty'] = np.nan
    else:
        # Estimate from local variability
        window = min(30, len(aligned_data) // 10)
        if window >= 3:
            aligned_data['uncertainty'] = aligned_data['difference'].rolling(
                window=window, center=True
            ).std()
        else:
            aligned_data['uncertainty'] = np.nan
    
    # Calculate metadata and statistics
    valid_data = aligned_data.dropna(subset=['difference'])
    
    metadata = {
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'temporal_resolution': target_resolution,
        'total_points': len(aligned_data),
        'valid_points': len(valid_data),
        'data_coverage': len(valid_data) / len(aligned_data) * 100 if len(aligned_data) > 0 else 0,
        'time_range': (start_date, end_date),
        'gnss_source_col': gnss_wl_col,
        'conversion_note': conversion_note,
        'ref_source_col': ref_wl_col,
    }
    
    if len(valid_data) > 0:
        metadata.update({
            'mean_difference': valid_data['difference'].mean(),
            'rmse': np.sqrt((valid_data['difference']**2).mean()),
            'std_difference': valid_data['difference'].std(),
            'bias': valid_data['difference'].mean(),
            'correlation': valid_data['gnss_water_level'].corr(valid_data['ref_water_level']),
            'mad': valid_data['difference'].abs().median(),  # Median Absolute Deviation
        })
    
    return aligned_data, metadata


def create_residual_analysis_plot(
    aligned_data: pd.DataFrame,
    metadata: dict,
    station_name: str = "GNSS Station",
    figsize: Tuple[float, float] = (16, 12),
    plot_mode: str = "dual_axis",
    reference_source: str = "Reference"
) -> plt.Figure:
    """
    Create comprehensive residual analysis plot.

    Parameters:
    -----------
    aligned_data : pd.DataFrame
        Aligned data with timestamp, gnss_water_level, ref_water_level, difference, uncertainty
    metadata : dict
        Processing metadata and statistics
    station_name : str
        Station identifier
    figsize : tuple
        Figure size
    reference_source : str
        Reference data source name (USGS, CO-OPS, ERDDAP)

    Returns:
    --------
    plt.Figure : The created figure
    """
    if PUBLICATION_THEME_AVAILABLE:
        apply_matplotlib_theme()
        colors = PUBLICATION_COLORS
    else:
        colors = {
            'gnss': '#2E86AB', 'usgs': '#A23B72', 'difference': '#F18F01',
            'background': '#FAFAFA', 'grid': '#E0E0E0', 'gnss_smooth': '#2E86AB'
        }
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1.5, 1, 1], hspace=0.3, wspace=0.3)
    
    # Filter valid data
    valid_data = aligned_data.dropna(subset=['difference'])
    
    # Main time series plot (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot based on selected mode
    if plot_mode == "dual_axis":
        # Dual y-axis plot
        ax1.plot(aligned_data['timestamp'], aligned_data['gnss_water_level'], 
                 color=colors.get('gnss', '#2E86AB'), linewidth=1.5, label='GNSS-IR WSE', alpha=0.8)
        ax1.set_ylabel('GNSS-IR Water Surface Elevation (m)', fontsize=12, fontweight='bold', 
                      color=colors.get('gnss', '#2E86AB'))
        ax1.tick_params(axis='y', labelcolor=colors.get('gnss', '#2E86AB'))
        
        # Create second y-axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(aligned_data['timestamp'], aligned_data['ref_water_level'],
                     color=colors.get('usgs', '#A23B72'), linewidth=1.5, label=f'{reference_source} Water Level', alpha=0.8)
        ax1_twin.set_ylabel(f'{reference_source} Water Level (m)', fontsize=12, fontweight='bold',
                           color=colors.get('usgs', '#A23B72'))
        ax1_twin.tick_params(axis='y', labelcolor=colors.get('usgs', '#A23B72'))
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        title_suffix = " (Dual Y-Axis)"
        
    elif plot_mode == "detrended":
        # Remove mean from each series
        gnss_mean = aligned_data['gnss_water_level'].mean()
        usgs_mean = aligned_data['ref_water_level'].mean()
        
        ax1.plot(aligned_data['timestamp'], aligned_data['gnss_water_level'] - gnss_mean, 
                 color=colors.get('gnss', '#2E86AB'), linewidth=1.5, 
                 label=f'GNSS-IR WSE (mean removed: {gnss_mean:.2f} m)', alpha=0.8)
        ax1.plot(aligned_data['timestamp'], aligned_data['ref_water_level'] - usgs_mean,
                 color=colors.get('usgs', '#A23B72'), linewidth=1.5,
                 label=f'{reference_source} Water Level (mean removed: {usgs_mean:.2f} m)', alpha=0.8)
        
        ax1.set_ylabel('Detrended Water Level (m)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        title_suffix = " (Detrended)"
        
    elif plot_mode == "normalized":
        # Normalize to standard deviations
        gnss_mean = aligned_data['gnss_water_level'].mean()
        gnss_std = aligned_data['gnss_water_level'].std()
        usgs_mean = aligned_data['ref_water_level'].mean()
        usgs_std = aligned_data['ref_water_level'].std()
        
        ax1.plot(aligned_data['timestamp'], 
                 (aligned_data['gnss_water_level'] - gnss_mean) / gnss_std, 
                 color=colors.get('gnss', '#2E86AB'), linewidth=1.5, 
                 label='GNSS-IR WSE (normalized)', alpha=0.8)
        ax1.plot(aligned_data['timestamp'],
                 (aligned_data['ref_water_level'] - usgs_mean) / usgs_std,
                 color=colors.get('usgs', '#A23B72'), linewidth=1.5,
                 label=f'{reference_source} Water Level (normalized)', alpha=0.8)
        
        ax1.set_ylabel('Normalized Water Level (σ)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        title_suffix = " (Normalized)"
        
    else:  # original mode
        ax1.plot(aligned_data['timestamp'], aligned_data['gnss_water_level'], 
                 color=colors.get('gnss', '#2E86AB'), linewidth=1.5, label='GNSS-IR WSE', alpha=0.8)
        ax1.plot(aligned_data['timestamp'], aligned_data['ref_water_level'],
                 color=colors.get('usgs', '#A23B72'), linewidth=1.5, label=f'{reference_source} Water Level', alpha=0.8)
        ax1.set_ylabel('Water Level (m)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        title_suffix = ""
    
    ax1.set_title(f'{station_name} - Water Level Comparison{title_suffix}\n'
                  f'({metadata["time_range"][0].strftime("%Y-%m-%d")} to '
                  f'{metadata["time_range"][1].strftime("%Y-%m-%d")})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Residual/difference plot (second row, full width)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Plot differences with uncertainty bands if available
    ax2.plot(aligned_data['timestamp'], aligned_data['difference'],
             color=colors.get('difference', '#F18F01'), linewidth=1, label=f'GNSS - {reference_source} Difference', alpha=0.8)
    
    # Add uncertainty bands if available
    if 'uncertainty' in aligned_data.columns and not aligned_data['uncertainty'].isna().all():
        unc_data = aligned_data.dropna(subset=['difference', 'uncertainty'])
        ax2.fill_between(unc_data['timestamp'], 
                        unc_data['difference'] - unc_data['uncertainty'],
                        unc_data['difference'] + unc_data['uncertainty'],
                        alpha=0.2, color=colors.get('difference', '#F18F01'), label='Uncertainty Band')
    
    # Add zero reference line
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add mean bias line
    if len(valid_data) > 0:
        mean_diff = metadata.get('mean_difference', 0)
        ax2.axhline(y=mean_diff, color='red', linestyle=':', alpha=0.7, linewidth=2, 
                   label=f'Mean Bias: {mean_diff:.3f} m')
    
    ax2.set_ylabel('Difference (m)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Statistics panel (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    # Create statistics text
    if len(valid_data) > 0:
        stats_text = [
            f"📊 RESIDUAL STATISTICS",
            f"Valid Points: {metadata.get('valid_points', 0):,} / {metadata.get('total_points', 0):,}",
            f"Coverage: {metadata.get('data_coverage', 0):.1f}%",
            f"",
            f"Mean Difference: {metadata.get('mean_difference', 0):.4f} m",
            f"RMSE: {metadata.get('rmse', 0):.4f} m", 
            f"Std Deviation: {metadata.get('std_difference', 0):.4f} m",
            f"Bias: {metadata.get('bias', 0):.4f} m",
            f"Correlation: {metadata.get('correlation', 0):.4f}",
            f"MAD: {metadata.get('mad', 0):.4f} m",
        ]
    else:
        stats_text = ["No valid overlapping data"]
    
    stats_str = '\n'.join(stats_text)
    ax3.text(0.05, 0.95, stats_str, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # Seasonal breakdown (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    
    if len(valid_data) > 0:
        # Group by season
        valid_data_copy = valid_data.copy()
        valid_data_copy['month'] = valid_data_copy['timestamp'].dt.month
        valid_data_copy['season'] = valid_data_copy['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring', 
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        seasonal_stats = valid_data_copy.groupby('season')['difference'].agg(['mean', 'std', 'count'])
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        season_colors = [colors.get('gnss', '#2E86AB'), colors.get('usgs', '#A23B72'), 
                        colors.get('difference', '#F18F01'), '#8B4513']
        
        for i, season in enumerate(seasons):
            if season in seasonal_stats.index:
                mean_val = seasonal_stats.loc[season, 'mean']
                std_val = seasonal_stats.loc[season, 'std']
                count = seasonal_stats.loc[season, 'count']
                
                ax4.bar(i, mean_val, yerr=std_val, color=season_colors[i], alpha=0.7,
                       capsize=5, label=f'{season}\n(n={count})')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Mean Difference (m)', fontsize=10)
        ax4.set_title('Seasonal Breakdown', fontsize=11, fontweight='bold')
        ax4.set_xticks(range(len(seasons)))
        ax4.set_xticklabels(seasons, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No seasonal data', ha='center', va='center', 
                transform=ax4.transAxes)
    
    # Rolling statistics (bottom, full width)
    ax5 = fig.add_subplot(gs[3, :])
    
    if len(valid_data) > 30:  # Need sufficient data for rolling stats
        window = min(30, len(valid_data) // 10)
        rolling_rmse = np.sqrt((valid_data['difference']**2).rolling(window=window, center=True).mean())
        rolling_mean = valid_data['difference'].rolling(window=window, center=True).mean()
        
        ax5.plot(valid_data['timestamp'], rolling_rmse, color='red', linewidth=2, 
                label=f'{window}-day Rolling RMSE (WSE - Water Level)')
        ax5.plot(valid_data['timestamp'], rolling_mean.abs(), color='blue', linewidth=2,
                label=f'{window}-day Rolling |Mean Difference|')
        
        ax5.set_ylabel('Rolling Stats (m)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax5.set_title(f'{station_name} - Rolling Statistics of Residuals', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Format x-axis
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for rolling statistics', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # Add processing info
    fig.text(0.02, 0.02, f'Processed: {metadata.get("processing_date", "Unknown")} | '
                        f'Resolution: {metadata.get("temporal_resolution", "Unknown")}',
             fontsize=8, style='italic')
    
    return fig


def render_yearly_residual_tab(rh_data, usgs_data, coops_data,
                              selected_station, selected_year, erddap_data=None):
    """
    Render the yearly residual analysis tab.

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
    erddap_data : pd.DataFrame, optional
        ERDDAP water level data (for co-located sensor stations)
    """
    st.header("📈 Yearly Time Series & Residual Analysis")

    st.markdown("""
    **Comprehensive yearly view showing the complete time series comparison between GNSS-IR and reference measurements,
    with detailed residual analysis to quantify measurement differences and uncertainty.**
    """)

    if rh_data is None or rh_data.empty:
        st.warning("⚠️ No GNSS-IR data available for residual analysis")
        return

    # Determine reference source: ERDDAP > USGS > CO-OPS
    reference_data = None
    reference_source = None

    if erddap_data is not None and not erddap_data.empty:
        reference_data = erddap_data
        reference_source = 'ERDDAP'
    elif usgs_data is not None and not usgs_data.empty:
        reference_data = usgs_data
        reference_source = 'USGS'
    elif coops_data is not None and not coops_data.empty:
        reference_data = coops_data
        reference_source = 'CO-OPS'

    if reference_data is None:
        st.warning("⚠️ No reference data (ERDDAP, USGS, or CO-OPS) available for residual analysis")
        return

    st.info(f"📊 Using **{reference_source}** as reference source for comparison")
    
    # Analysis settings
    st.markdown("### ⚙️ Analysis Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        resolution = st.selectbox(
            "Temporal Resolution:",
            ["D", "6H", "3H", "H"],
            format_func=lambda x: {"D": "Daily", "6H": "6-Hourly", "3H": "3-Hourly", "H": "Hourly"}[x],
            index=0,
            help="Target resolution for data alignment"
        )
    
    with col2:
        uncertainty_method = st.selectbox(
            "Uncertainty Estimation:",
            ["measurement_std", "rolling_variability", "none"],
            format_func=lambda x: {
                "measurement_std": "GNSS Measurement Std",
                "rolling_variability": "Local Rolling Variability", 
                "none": "No Uncertainty Bands"
            }[x],
            help="Method for estimating measurement uncertainty"
        )
    
    with col3:
        outlier_filter = st.checkbox(
            "Apply Outlier Filter",
            value=True,
            help="Remove statistical outliers before analysis"
        )
    
    # Add visualization type selector
    st.markdown("### 📊 Visualization Type")
    visualization_type = st.selectbox(
        "Select visualization:",
        ["Residual Analysis", "Multi-Parameter Timeline"],
        help="Choose between detailed residual analysis or multi-parameter timeline"
    )
    
    # Initialize default values
    plot_mode = "dual_axis"  # Default for residual analysis
    rolling_window = 30      # Default for timeline
    
    if visualization_type == "Residual Analysis":
        # Add plot mode selector for residual analysis
        plot_mode = st.radio(
            "Select how to display the time series comparison:",
            ["dual_axis", "detrended", "normalized", "original"],
            format_func=lambda x: {
                "dual_axis": "🔀 Dual Y-Axis (each dataset on its own scale)", 
                "detrended": "📉 Detrended (mean removed from each)",
                "normalized": "📏 Normalized (standardized to σ units)",
                "original": "📈 Original Values (same scale)"
            }[x],
            index=0,
            help="Choose how to visualize the water level comparison"
        )
    else:
        # Multi-Parameter Timeline settings
        rolling_window = st.slider("Rolling window (days):", 7, 60, 30, key="yearly_timeline_window")
    
    # Process data button
    button_label = "🔄 Process Analysis" if visualization_type == "Multi-Parameter Timeline" else "🔄 Process Residual Analysis"
    if st.button(button_label, type="primary"):
        if visualization_type == "Multi-Parameter Timeline":
            # Multi-Parameter Timeline Processing
            if not DASHBOARD_PLOTS_AVAILABLE:
                st.error("Dashboard plotting functions not available. Please check installation.")
                return
                
            with st.spinner("Creating multi-parameter timeline..."):
                # Prepare standardized USGS data for plotting
                usgs_for_plot = None
                if usgs_data is not None and not usgs_data.empty:
                    usgs_for_plot = usgs_data.copy()
                    
                    # Ensure it has the expected column names
                    if 'water_level_m' not in usgs_for_plot.columns:
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

                # Use CO-OPS data if available and no USGS data
                water_level_data = usgs_for_plot
                if (usgs_for_plot is None or usgs_for_plot.empty) and coops_data is not None and not coops_data.empty:
                    water_level_data = coops_data
                    st.info("Using CO-OPS tide data for water level comparison")
                elif coops_data is not None and not coops_data.empty:
                    # Show option to use CO-OPS instead of USGS
                    use_coops = st.checkbox("Use CO-OPS data instead of USGS", value=False, key="yearly_timeline_coops")
                    if use_coops:
                        water_level_data = coops_data

                # Calculate performance metrics for timeline
                perf_data = rh_data.copy()
                if water_level_data is not None and not water_level_data.empty:
                    # Ensure date columns are datetime for proper merging
                    perf_data['date'] = pd.to_datetime(perf_data['date'])
                    water_level_data['date'] = pd.to_datetime(water_level_data['date'])
                    
                    # Merge GNSS-IR and water level data for correlation calculation
                    merged_perf = pd.merge(
                        perf_data[['date', 'rh_median_m', 'rh_std_m', 'rh_count']],
                        water_level_data[['date', 'water_level_m']],
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
                        
                        # Calculate rolling correlation (30-day window) using WSE instead of RH
                        window_size = min(30, len(merged_perf) // 3)
                        if window_size >= 3:
                            merged_perf = merged_perf.sort_values('date')
                            merged_perf['correlation'] = merged_perf['wse_ellips_m'].rolling(window=window_size).corr(
                                merged_perf['water_level_m']
                            )
                        else:
                            # Global correlation if not enough data using WSE
                            global_corr = merged_perf['wse_ellips_m'].corr(merged_perf['water_level_m'])
                            merged_perf['correlation'] = global_corr
                        
                        perf_data = merged_perf

                # Create multi-parameter timeline
                fig = create_multi_parameter_timeline(
                    perf_data,
                    usgs_df=water_level_data,
                    environmental_df=None,
                    station_name=selected_station,
                    year=selected_year,
                    rolling_window=rolling_window,
                    figsize=(16, 12)
                )
                
                st.pyplot(fig)
                plt.close()
                
                st.markdown("""
                ### 📊 Multi-Parameter Timeline Analysis

                This comprehensive timeline shows:
                - **Data Availability**: Daily retrieval counts with rolling averages
                - **Performance Metrics**: Measurement precision over time
                - **Correlation Analysis**: Rolling correlation with water level data (if available)

                The timeline helps identify seasonal patterns and optimal measurement periods.
                """)
        
        else:
            # Residual Analysis Processing
            with st.spinner("Processing temporal alignment and residual analysis..."):
                # Preprocess and align data
                aligned_data, metadata = preprocess_for_residual_analysis(
                    rh_data, reference_data, target_resolution=resolution, station_id=selected_station
                )
                # Store reference source in metadata for display
                metadata['reference_source'] = reference_source
                
                if aligned_data.empty:
                    st.error("❌ Failed to align data - no overlapping time periods found")
                    return
                
                # Apply outlier filtering if requested
                if outlier_filter and len(aligned_data) > 10:
                    # Remove outliers using IQR method
                    Q1 = aligned_data['difference'].quantile(0.25)
                    Q3 = aligned_data['difference'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (aligned_data['difference'] >= lower_bound) & (aligned_data['difference'] <= upper_bound)
                    n_outliers = (~outlier_mask).sum()
                    aligned_data.loc[~outlier_mask, 'difference'] = np.nan
                    
                    st.info(f"🔍 Outlier filtering removed {n_outliers} data points")
                
                # Create comprehensive plot
                fig = create_residual_analysis_plot(
                    aligned_data, metadata, selected_station, figsize=(16, 14),
                    plot_mode=plot_mode, reference_source=reference_source
                )
            
            # Display plot
            st.pyplot(fig)
            plt.close()
            
            # Display conversion information
            if metadata.get('conversion_note'):
                st.info(f"📐 **Data Conversion**: {metadata['conversion_note']}")
            
            # Display detailed statistics
            st.markdown("### 📊 Detailed Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Coverage", f"{metadata.get('data_coverage', 0):.1f}%")
                st.metric("Valid Points", f"{metadata.get('valid_points', 0):,}")
            
            with col2:
                st.metric("RMSE", f"{metadata.get('rmse', 0):.4f} m")
                st.metric("Mean Bias", f"{metadata.get('bias', 0):.4f} m")
            
            with col3:
                st.metric("Correlation", f"{metadata.get('correlation', 0):.4f}")
                st.metric("Std Deviation", f"{metadata.get('std_difference', 0):.4f} m")
            
            with col4:
                st.metric("MAD", f"{metadata.get('mad', 0):.4f} m")
                st.metric("Total Points", f"{metadata.get('total_points', 0):,}")
            
            # RMSE Explanation
            st.markdown("### 🧮 RMSE Calculation Details")
            st.markdown(f"""
            **Root Mean Square Error (RMSE)** quantifies the magnitude of differences between GNSS-IR water surface elevation (WSE) and {reference_source} measurements:

            ```
            RMSE = √(Σ(GNSS_water_level - {reference_source}_water_level)² / N)
            ```
            
            **Current calculation method:**
            - **Data Source**: {metadata.get('gnss_source_col', 'Unknown')} vs {metadata.get('ref_source_col', 'Unknown')}
            - **Temporal Alignment**: {resolution} resolution with interpolation
            - **Valid Pairs**: {metadata.get('valid_points', 0):,} synchronized measurements
            - **RMSE Value**: {metadata.get('rmse', 0):.4f} meters
            
            **Interpretation:**
            - Lower RMSE indicates better agreement between measurement systems
            - Typical values for well-calibrated GNSS-IR: 0.05-0.15 m
            - Values > 0.20 m may indicate calibration or processing issues
            """)
            
            # Export options
            if st.button("💾 Export Analysis Data"):
                # Prepare export data
                export_data = aligned_data.copy()
                export_data['station'] = selected_station
                export_data['year'] = selected_year
                
                # Convert to CSV
                csv_data = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{selected_station}_{selected_year}_residual_analysis.csv",
                    mime="text/csv"
                )


# Export the render function
__all__ = ['render_yearly_residual_tab']