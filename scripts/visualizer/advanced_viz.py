"""
Module for advanced visualizations combining GNSS-IR data with auxiliary measurements.

This module provides functionality for creating complex, multi-panel visualizations 
that combine GNSS-IR water level data with other environmental measurements like 
wind, pressure, or tide predictions.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .base import ensure_output_dir, add_summary_textbox, PLOT_COLORS, PLOT_STYLES

def plot_wse_wind_comparison(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame, 
    wind_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    wind_station_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_col: str = 'rh_median_m',
    usgs_col: str = 'usgs_value_m_median',
    wind_speed_col: str = 'wind_speed_mps',
    wind_dir_col: str = 'wind_dir_deg',
    wind_forcing_col: Optional[str] = 'wind_forcing',
    plot_wind_forcing: bool = True,
    highlight_dates: Optional[List[str]] = None,
    fetch_directions: Optional[List[float]] = None,
    style: str = 'default'
) -> Optional[Path]:
    """
    Create a multi-panel plot showing GNSS-IR WSE, USGS gauge water level, and wind data.
    
    Args:
        gnssir_df: DataFrame with GNSS-IR data
        usgs_df: DataFrame with USGS gauge data
        wind_df: DataFrame with wind data
        station_name: GNSS-IR station name
        usgs_gauge_info: Dictionary with USGS gauge information
        wind_station_info: Dictionary with wind station information
        output_plot_path: Path to save the plot
        gnssir_col: Column name for GNSS-IR data
        usgs_col: Column name for USGS gauge data
        wind_speed_col: Column name for wind speed
        wind_dir_col: Column name for wind direction
        wind_forcing_col: Column name for wind forcing (if available)
        plot_wind_forcing: Whether to plot wind forcing (if available)
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        fetch_directions: List of fetch directions to show on the wind rose
        style: Plot style to use
    
    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)
    
    try:
        # Check if DataFrames are valid
        if gnssir_df is None or gnssir_df.empty:
            logging.error("GNSS-IR data DataFrame is empty")
            return None
        
        if usgs_df is None or usgs_df.empty:
            logging.error("USGS gauge data DataFrame is empty")
            return None
            
        if wind_df is None or wind_df.empty:
            logging.error("Wind data DataFrame is empty")
            return None
        
        # Check if required columns exist
        if gnssir_col not in gnssir_df.columns:
            logging.error(f"Column '{gnssir_col}' not found in GNSS-IR data")
            return None
        
        if usgs_col not in usgs_df.columns:
            logging.error(f"Column '{usgs_col}' not found in USGS gauge data")
            return None
            
        if wind_speed_col not in wind_df.columns:
            logging.error(f"Column '{wind_speed_col}' not found in wind data")
            return None
            
        if wind_dir_col not in wind_df.columns:
            logging.error(f"Column '{wind_dir_col}' not found in wind data")
            return None
        
        # Determine if we should plot wind forcing
        plot_forcing = plot_wind_forcing and (wind_forcing_col in wind_df.columns)
        
        # Extract metadata
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', f'USGS {usgs_site_code}')
        usgs_distance_km = usgs_gauge_info.get('distance_km', 0)
        usgs_vertical_datum = usgs_gauge_info.get('vertical_datum', 'Unknown')
        
        wind_station_id = wind_station_info.get('id', 'Unknown')
        wind_station_name = wind_station_info.get('name', f'Station {wind_station_id}')
        wind_station_type = wind_station_info.get('type', 'Unknown')
        
        # Create datetime for plotting if not already present
        if 'datetime' not in gnssir_df.columns and 'date' in gnssir_df.columns:
            gnssir_df['datetime'] = pd.to_datetime(gnssir_df['date'])
        
        if 'datetime' not in usgs_df.columns and 'date' in usgs_df.columns:
            usgs_df['datetime'] = pd.to_datetime(usgs_df['date'])
            
        if 'datetime' not in wind_df.columns and 'date' in wind_df.columns:
            wind_df['datetime'] = pd.to_datetime(wind_df['date'])
        
        # Set up plotting data
        gnssir_x = gnssir_df['datetime'] if 'datetime' in gnssir_df.columns else gnssir_df['date']
        usgs_x = usgs_df['datetime'] if 'datetime' in usgs_df.columns else usgs_df['date']
        wind_x = wind_df['datetime'] if 'datetime' in wind_df.columns else wind_df['date']
        
        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value
        
        # Create figure with GridSpec for multiple panels
        if plot_forcing:
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
        else:
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
        
        # Panel 1: Water level comparison
        ax1 = fig.add_subplot(gs[0])
        
        # Plot GNSS-IR data
        ax1.plot(gnssir_x, gnssir_df[gnssir_col], 'o-', 
               label=f"{station_name} GNSS-IR WSE", 
               color=PLOT_COLORS['gnssir'], 
               linewidth=2, markersize=6, alpha=0.7)
        
        # Plot USGS data
        ax1.plot(usgs_x, usgs_df[usgs_col], 's-', 
               label=f"{usgs_site_code} USGS WL", 
               color=PLOT_COLORS['usgs'], 
               linewidth=2, markersize=6, alpha=0.7)
        
        # Highlight specific dates if requested
        if highlight_dates:
            try:
                highlight_dates_dt = pd.to_datetime(highlight_dates)
                for date in highlight_dates_dt:
                    ax1.axvline(x=date, color=PLOT_COLORS['highlight'], linestyle='--', alpha=0.5,
                              label="Highlighted Date" if date == highlight_dates_dt[0] else "")
            except Exception as e:
                logging.warning(f"Could not highlight dates: {e}")
        
        # Set labels and title
        ax1.set_title(f"{station_name} Water Levels with Meteorological Data", fontsize=16)
        ax1.set_ylabel("Water Level (m)", fontsize=14)
        ax1.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
        ax1.legend(fontsize=12)
        
        # Format x-axis for dates - shared across panels
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Panel 2: Wind speed
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot wind speed
        ax2.plot(wind_x, wind_df[wind_speed_col], '-', 
                color=PLOT_COLORS['secondary'], 
                linewidth=2, 
                label=f"Wind Speed ({wind_station_id})")
        
        # Add markers to highlight wind direction
        dir_markers = []
        for i, (idx, row) in enumerate(wind_df.iterrows()):
            if i % 3 == 0:  # Plot every 3rd point to avoid crowding
                direction = row[wind_dir_col]
                # Map direction (0-360) to arrow directions
                dx = np.sin(np.radians(direction))
                dy = np.cos(np.radians(direction))
                ax2.quiver(row['datetime'], row[wind_speed_col], 
                          dx, dy, 
                          pivot='mid', 
                          color=PLOT_COLORS['secondary'],
                          scale=12, 
                          width=0.005)
        
        # Set labels
        ax2.set_ylabel("Wind Speed (m/s)", fontsize=14)
        ax2.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
        ax2.legend(fontsize=12)
        
        # Add wind station info
        wind_info_text = {
            "Wind Station": wind_station_id,
            "Name": wind_station_name,
            "Type": wind_station_type
        }
        add_summary_textbox(plt, None, wind_info_text, position=(0.02, 0.02), fontsize=10)
        
        # Panel 3: Wind forcing (if available)
        if plot_forcing:
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            # Plot wind forcing
            ax3.plot(wind_x, wind_df[wind_forcing_col], '-', 
                    color='purple', 
                    linewidth=2, 
                    label="Wind Forcing Index")
            
            # Set labels
            ax3.set_ylabel("Wind Forcing", fontsize=14)
            ax3.set_xlabel("Date", fontsize=14)
            ax3.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
            ax3.legend(fontsize=12)
        else:
            # If not plotting wind forcing, add x-label to wind speed panel
            ax2.set_xlabel("Date", fontsize=14)
        
        # Calculate correlation between water levels and wind
        try:
            # Merge DataFrames for correlation calculation
            merged_wl = pd.merge(
                gnssir_df[['datetime', gnssir_col]], 
                usgs_df[['datetime', usgs_col]],
                on='datetime',
                how='inner'
            )
            
            merged_wind = pd.merge(
                merged_wl,
                wind_df[['datetime', wind_speed_col]],
                on='datetime',
                how='inner'
            )
            
            if not merged_wind.empty and len(merged_wind) >= 3:
                corr_gnssir_wind = merged_wind[gnssir_col].corr(merged_wind[wind_speed_col])
                corr_usgs_wind = merged_wind[usgs_col].corr(merged_wind[wind_speed_col])
                
                # Add correlation info
                corr_text = {
                    f"Corr({station_name}, Wind)": f"{corr_gnssir_wind:.3f}",
                    f"Corr({usgs_site_code}, Wind)": f"{corr_usgs_wind:.3f}"
                }
                add_summary_textbox(plt, None, corr_text, position=(0.75, 0.02), fontsize=12)
        except Exception as e:
            logging.warning(f"Could not calculate correlations: {e}")
        
        # Add USGS gauge info
        gauge_info_text = {
            "USGS Site": usgs_site_code,
            "Name": usgs_site_name,
            "Distance": f"{usgs_distance_km:.2f} km",
            "Datum": usgs_vertical_datum
        }
        add_summary_textbox(plt, None, gauge_info_text, position=(0.02, 0.85), fontsize=10)
        
        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        
        logging.info(f"Multi-panel wind comparison plot saved to {output_plot_path}")
        return output_plot_path
    
    except Exception as e:
        logging.error(f"Error creating wind comparison plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def plot_quality_metrics_vs_residuals(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_wse_col: str = 'rh_median_m',
    usgs_wl_col: str = 'usgs_value_m_median',
    count_col: str = 'rh_count',
    std_col: str = 'rh_std_m',
    amplitude_col: Optional[str] = None,
    peak_noise_col: Optional[str] = None,
    style: str = 'default'
) -> Optional[Path]:
    """
    Create a multi-panel plot showing GNSS-IR data quality metrics vs. water level residuals.
    
    Args:
        gnssir_df: DataFrame with GNSS-IR data (must include quality metrics)
        usgs_df: DataFrame with USGS gauge data
        station_name: GNSS-IR station name
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Path to save the plot
        gnssir_wse_col: Column name for GNSS-IR WSE
        usgs_wl_col: Column name for USGS water level
        count_col: Column name for RH retrieval count
        std_col: Column name for RH standard deviation
        amplitude_col: Column name for LSP amplitude (if available)
        peak_noise_col: Column name for LSP peak-to-noise ratio (if available)
        style: Plot style to use
    
    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)
    
    try:
        # Check if DataFrames are valid
        if gnssir_df is None or gnssir_df.empty:
            logging.error("GNSS-IR data DataFrame is empty")
            return None
        
        if usgs_df is None or usgs_df.empty:
            logging.error("USGS gauge data DataFrame is empty")
            return None
        
        # Check if required columns exist
        for col in [gnssir_wse_col, count_col, std_col]:
            if col not in gnssir_df.columns:
                logging.error(f"Required column '{col}' not found in GNSS-IR data")
                return None
        
        if usgs_wl_col not in usgs_df.columns:
            logging.error(f"Column '{usgs_wl_col}' not found in USGS gauge data")
            return None
        
        # Extract USGS gauge info
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', f'USGS {usgs_site_code}')
        
        # Create datetime for plotting if not already present
        if 'datetime' not in gnssir_df.columns and 'date' in gnssir_df.columns:
            gnssir_df['datetime'] = pd.to_datetime(gnssir_df['date'])
        
        if 'datetime' not in usgs_df.columns and 'date' in usgs_df.columns:
            usgs_df['datetime'] = pd.to_datetime(usgs_df['date'])
        
        # Merge DataFrames on datetime for residuals calculation
        merged_df = pd.merge(
            gnssir_df,
            usgs_df[['datetime', usgs_wl_col]],
            on='datetime',
            how='inner'
        )
        
        if merged_df.empty:
            logging.error("No overlapping data points between GNSS-IR and USGS")
            return None
        
        # Calculate residuals (difference between demeaned series)
        gnssir_mean = merged_df[gnssir_wse_col].mean()
        usgs_mean = merged_df[usgs_wl_col].mean()
        
        merged_df['gnssir_demeaned'] = merged_df[gnssir_wse_col] - gnssir_mean
        merged_df['usgs_demeaned'] = merged_df[usgs_wl_col] - usgs_mean
        merged_df['residuals'] = merged_df['gnssir_demeaned'] - merged_df['usgs_demeaned']
        
        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value
        
        # Determine number of panels based on available metrics
        num_quality_metrics = 2  # Count and StdDev always present
        if amplitude_col and amplitude_col in gnssir_df.columns:
            num_quality_metrics += 1
        if peak_noise_col and peak_noise_col in gnssir_df.columns:
            num_quality_metrics += 1
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(num_quality_metrics + 1, 1, hspace=0.2)
        
        # Panel 1: Residuals
        ax1 = fig.add_subplot(gs[0])
        
        # Plot residuals
        ax1.plot(merged_df['datetime'], merged_df['residuals'], 'o-',
               color=PLOT_COLORS['gnssir'],
               linewidth=2, markersize=6, alpha=0.7)
        
        # Add horizontal line at zero
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels
        ax1.set_title(f"{station_name} GNSS-IR Data Quality vs Residuals", fontsize=16)
        ax1.set_ylabel("WSE Residuals (m)\nGNSS-IR - USGS", fontsize=14)
        ax1.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
        
        # Format x-axis for dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Panel 2: RH Count
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot RH count
        ax2.plot(merged_df['datetime'], merged_df[count_col], 'o-',
               color=PLOT_COLORS['secondary'],
               linewidth=2, markersize=6, alpha=0.7)
        
        # Set labels
        ax2.set_ylabel("RH Count", fontsize=14)
        ax2.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
        
        # Panel 3: RH Standard Deviation
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Plot RH standard deviation
        ax3.plot(merged_df['datetime'], merged_df[std_col], 'o-',
               color='purple',
               linewidth=2, markersize=6, alpha=0.7)
        
        # Set labels
        ax3.set_ylabel("RH Std Dev (m)", fontsize=14)
        ax3.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
        
        # Current panel index
        panel_idx = 3
        
        # Panel 4: LSP Amplitude (if available)
        if amplitude_col and amplitude_col in gnssir_df.columns:
            ax4 = fig.add_subplot(gs[panel_idx], sharex=ax1)
            
            # Plot LSP amplitude
            ax4.plot(merged_df['datetime'], merged_df[amplitude_col], 'o-',
                   color='green',
                   linewidth=2, markersize=6, alpha=0.7)
            
            # Set labels
            ax4.set_ylabel("LSP Amplitude", fontsize=14)
            ax4.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
            
            panel_idx += 1
        
        # Panel 5: LSP Peak-to-Noise Ratio (if available)
        if peak_noise_col and peak_noise_col in gnssir_df.columns:
            ax5 = fig.add_subplot(gs[panel_idx], sharex=ax1)
            
            # Plot LSP peak-to-noise ratio
            ax5.plot(merged_df['datetime'], merged_df[peak_noise_col], 'o-',
                   color='orange',
                   linewidth=2, markersize=6, alpha=0.7)
            
            # Set labels
            ax5.set_ylabel("LSP Peak/Noise", fontsize=14)
            ax5.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])
            
            panel_idx += 1
        
        # Add x-label to the bottom panel
        fig.axes[-1].set_xlabel("Date", fontsize=14)
        
        # Calculate correlations between residuals and quality metrics
        corr_count = merged_df['residuals'].corr(merged_df[count_col])
        corr_std = merged_df['residuals'].corr(merged_df[std_col])
        
        # Add correlation info
        corr_text = {
            "Corr(Residuals, Count)": f"{corr_count:.3f}",
            "Corr(Residuals, StdDev)": f"{corr_std:.3f}"
        }
        
        if amplitude_col and amplitude_col in gnssir_df.columns:
            corr_amp = merged_df['residuals'].corr(merged_df[amplitude_col])
            corr_text[f"Corr(Residuals, Amplitude)"] = f"{corr_amp:.3f}"
            
        if peak_noise_col and peak_noise_col in gnssir_df.columns:
            corr_pknoise = merged_df['residuals'].corr(merged_df[peak_noise_col])
            corr_text[f"Corr(Residuals, Peak/Noise)"] = f"{corr_pknoise:.3f}"
        
        add_summary_textbox(plt, None, corr_text, position=(0.70, 0.02), fontsize=12)
        
        # Add residual statistics
        res_stats = {
            "Mean Residual": f"{merged_df['residuals'].mean():.3f} m",
            "Std Dev": f"{merged_df['residuals'].std():.3f} m",
            "RMSE": f"{np.sqrt((merged_df['residuals']**2).mean()):.3f} m",
            "Data Points": f"{len(merged_df)}"
        }
        add_summary_textbox(plt, None, res_stats, position=(0.02, 0.02), fontsize=12)
        
        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        
        logging.info(f"Quality metrics vs residuals plot saved to {output_plot_path}")
        return output_plot_path
    
    except Exception as e:
        logging.error(f"Error creating quality metrics plot: {e}")
        return None

def plot_daily_wl_change_correlation(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_wl_col: str = 'wse_ellips_m',
    usgs_wl_col: str = 'usgs_value_m_median',
    datetime_col: str = 'datetime',
    apply_time_lag: Optional[int] = None,
    max_lag_days: int = 5,
    calculate_optimal_lag: bool = True,
    highlight_dates: Optional[List[str]] = None,
    style: str = 'default'
) -> Optional[Path]:
    """
    Create a visualization showing correlation between daily water level changes
    in GNSS-IR and USGS datasets.
    
    Args:
        gnssir_df: DataFrame with GNSS-IR daily water level data
        usgs_df: DataFrame with USGS gauge daily water level data
        station_name: GNSS-IR station name
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Path to save the plot
        gnssir_wl_col: Column name for GNSS-IR water level
        usgs_wl_col: Column name for USGS water level
        datetime_col: Column name for date/datetime
        apply_time_lag: Time lag to apply in days (positive means USGS lags GNSS-IR)
        max_lag_days: Maximum time lag to consider when calculating optimal lag
        calculate_optimal_lag: Whether to calculate optimal time lag
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        style: Plot style to use
    
    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)
    
    try:
        # Check if DataFrames are valid
        if gnssir_df is None or gnssir_df.empty:
            logging.error("GNSS-IR data DataFrame is empty")
            return None
        
        if usgs_df is None or usgs_df.empty:
            logging.error("USGS gauge data DataFrame is empty")
            return None
        
        # Check if required columns exist
        for col in [gnssir_wl_col, datetime_col]:
            if col not in gnssir_df.columns:
                logging.error(f"Required column '{col}' not found in GNSS-IR data")
                return None
        
        for col in [usgs_wl_col, datetime_col]:
            if col not in usgs_df.columns:
                logging.error(f"Required column '{col}' not found in USGS data")
                return None
        
        # Extract USGS gauge info
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', f'USGS {usgs_site_code}')
        usgs_distance_km = usgs_gauge_info.get('distance_km', 0)
        usgs_vertical_datum = usgs_gauge_info.get('vertical_datum', 'Unknown')
        
        # Ensure datetime columns are in datetime format
        gnssir_df = gnssir_df.copy()
        usgs_df = usgs_df.copy()
        
        gnssir_df[datetime_col] = pd.to_datetime(gnssir_df[datetime_col])
        usgs_df[datetime_col] = pd.to_datetime(usgs_df[datetime_col])
        
        # Sort by datetime
        gnssir_df = gnssir_df.sort_values(by=datetime_col)
        usgs_df = usgs_df.sort_values(by=datetime_col)
        
        # Calculate day-to-day changes
        gnssir_df['wl_change'] = gnssir_df[gnssir_wl_col].diff()
        usgs_df['wl_change'] = usgs_df[usgs_wl_col].diff()
        
        # Drop NaN values (first row will have NaN for change)
        gnssir_df = gnssir_df.dropna(subset=['wl_change'])
        usgs_df = usgs_df.dropna(subset=['wl_change'])
        
        # Calculate optimal time lag if requested
        optimal_lag = 0
        lag_confidence = 0.0
        
        if calculate_optimal_lag:
            # Merge datasets to get common dates
            merged_df = pd.merge(
                gnssir_df[[datetime_col, 'wl_change']],
                usgs_df[[datetime_col, 'wl_change']],
                on=datetime_col,
                how='inner',
                suffixes=('_gnssir', '_usgs')
            )
            
            if len(merged_df) >= 5:  # Need at least 5 points for meaningful lag
                # Create time series for lag analysis
                gnssir_series = merged_df['wl_change_gnssir'].values
                usgs_series = merged_df['wl_change_usgs'].values
                
                # Calculate cross-correlation
                max_lag = min(max_lag_days, len(gnssir_series) // 3)  # Limit max lag to 1/3 of series length
                
                # Use scipy.signal.correlate for cross-correlation
                corr = np.correlate(gnssir_series, usgs_series, mode='full')
                corr_center = len(corr) // 2
                
                # Find lag with maximum correlation within max_lag range
                lag_range = np.arange(corr_center - max_lag, corr_center + max_lag + 1)
                valid_indices = np.logical_and(lag_range >= 0, lag_range < len(corr))
                valid_lags = lag_range[valid_indices]
                valid_corrs = corr[valid_indices]
                
                if len(valid_corrs) > 0:
                    max_corr_idx = np.argmax(valid_corrs)
                    optimal_lag = valid_lags[max_corr_idx] - corr_center
                    
                    # Calculate confidence in the lag based on correlation value and data length
                    max_corr = valid_corrs[max_corr_idx]
                    second_max = np.sort(valid_corrs)[-2] if len(valid_corrs) > 1 else 0
                    lag_confidence = min(1.0, max(0.0, max_corr / (second_max + 1e-6) * (len(gnssir_series) / 20.0)))
                    
                    logging.info(f"Optimal lag detected: {optimal_lag} days (confidence: {lag_confidence:.2f})")
        
        # Use provided lag if specified, otherwise use optimal lag if found with good confidence
        time_lag = apply_time_lag if apply_time_lag is not None else (optimal_lag if lag_confidence > 0.5 else 0)
        
        # Apply time lag if needed
        if time_lag != 0:
            logging.info(f"Applying time lag: {time_lag} days to USGS data")
            if time_lag > 0:
                # Positive lag: USGS lags behind GNSS-IR
                usgs_shifted = usgs_df.copy()
                usgs_shifted[datetime_col] = usgs_shifted[datetime_col] - pd.Timedelta(days=time_lag)
            else:
                # Negative lag: GNSS-IR lags behind USGS
                usgs_shifted = usgs_df.copy()
                usgs_shifted[datetime_col] = usgs_shifted[datetime_col] + pd.Timedelta(days=abs(time_lag))
        else:
            # No lag
            usgs_shifted = usgs_df.copy()
        
        # Merge datasets for correlation analysis
        merged_df = pd.merge(
            gnssir_df[[datetime_col, gnssir_wl_col, 'wl_change']],
            usgs_shifted[[datetime_col, usgs_wl_col, 'wl_change']],
            on=datetime_col,
            how='inner',
            suffixes=('_gnssir', '')
        )
        
        # Check if there are enough data points for analysis
        if len(merged_df) < 3:
            logging.error("Insufficient data points for correlation analysis after time alignment")
            return None
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'wl_change_gnssir': 'gnssir_change',
            'wl_change': 'usgs_change'
        })
        
        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
        
        # Panel 1: Time series of daily changes
        ax1 = fig.add_subplot(gs[0])
        
        # Plot GNSS-IR water level changes
        ax1.plot(merged_df[datetime_col], merged_df['gnssir_change'], 'o-',
               label=f"{station_name} Daily Change",
               color=PLOT_COLORS.get('gnssir', 'red'),
               linewidth=2, markersize=6, alpha=0.7)
        
        # Plot USGS water level changes
        ax1.plot(merged_df[datetime_col], merged_df['usgs_change'], 's-',
               label=f"{usgs_site_code} Daily Change",
               color=PLOT_COLORS.get('usgs', 'blue'),
               linewidth=2, markersize=6, alpha=0.7)
        
        # Highlight specific dates if requested
        if highlight_dates:
            try:
                highlight_dates_dt = pd.to_datetime(highlight_dates)
                for date in highlight_dates_dt:
                    ax1.axvline(x=date, color=PLOT_COLORS.get('highlight', 'orange'), linestyle='--', alpha=0.5,
                              label="Highlighted Date" if date == highlight_dates_dt[0] else "")
            except Exception as e:
                logging.warning(f"Could not highlight dates: {e}")
        
        # Add horizontal line at zero
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels
        time_lag_info = f" (Time lag: {time_lag} days)" if time_lag != 0 else ""
        ax1.set_title(f"Daily Water Level Changes: {station_name} vs {usgs_site_code}{time_lag_info}", fontsize=16)
        ax1.set_ylabel("Daily Change (m/day)", fontsize=14)
        ax1.grid(True, alpha=0.3, color=PLOT_COLORS.get('grid', 'lightgray'))
        ax1.legend(fontsize=12)
        
        # Format x-axis for dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Panel 2: Scatter plot with regression
        ax2 = fig.add_subplot(gs[1])
        
        # Create scatter plot
        ax2.scatter(merged_df['gnssir_change'], merged_df['usgs_change'],
                  color=PLOT_COLORS.get('scatter', 'purple'),
                  s=60, alpha=0.7)
        
        # Add text labels for specific points if needed
        # for i, row in merged_df.iterrows():
        #     if abs(row['gnssir_change']) > some_threshold or abs(row['usgs_change']) > some_threshold:
        #         ax2.annotate(
        #             row[datetime_col].strftime('%Y-%m-%d'),
        #             (row['gnssir_change'], row['usgs_change']),
        #             xytext=(5, 5),
        #             textcoords='offset points',
        #             fontsize=8,
        #             alpha=0.7
        #         )
        
        # Calculate correlation and regression
        correlation = merged_df['gnssir_change'].corr(merged_df['usgs_change'])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df['gnssir_change'],
            merged_df['usgs_change']
        )
        
        # Calculate RMSE
        predictions = slope * merged_df['gnssir_change'] + intercept
        rmse = np.sqrt(np.mean((merged_df['usgs_change'] - predictions) ** 2))
        
        # Plot regression line
        x_line = np.array([merged_df['gnssir_change'].min(), merged_df['gnssir_change'].max()])
        y_line = slope * x_line + intercept
        ax2.plot(x_line, y_line, '-', color='darkred', linewidth=2,
               label=f'y = {slope:.3f}x + {intercept:.3f}')
        
        # Add identity line (y=x)
        min_val = min(merged_df['gnssir_change'].min(), merged_df['usgs_change'].min())
        max_val = max(merged_df['gnssir_change'].max(), merged_df['usgs_change'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5,
               label='y = x (perfect correlation)')
        
        # Set labels
        ax2.set_title("Correlation of Daily Water Level Changes", fontsize=16)
        ax2.set_xlabel(f"{station_name} Daily Change (m/day)", fontsize=14)
        ax2.set_ylabel(f"{usgs_site_code} Daily Change (m/day)", fontsize=14)
        ax2.grid(True, alpha=0.3, color=PLOT_COLORS.get('grid', 'lightgray'))
        ax2.legend(fontsize=10)
        
        # Ensure equal scaling
        ax2.set_aspect('equal')
        
        # Add correlation info
        corr_text = {
            "Correlation (r)": f"{correlation:.3f}",
            "R²": f"{r_value**2:.3f}",
            "Slope": f"{slope:.3f}",
            "RMSE": f"{rmse:.3f} m",
            "Data Points": f"{len(merged_df)}"
        }
        add_summary_textbox(plt, ax2, corr_text, position=(0.02, 0.02), fontsize=12)
        
        # Add USGS gauge info to the first panel
        gauge_info_text = {
            "USGS Site": usgs_site_code,
            "Distance": f"{usgs_distance_km:.2f} km",
            "Datum": usgs_vertical_datum
        }
        add_summary_textbox(plt, ax1, gauge_info_text, position=(0.02, 0.02), fontsize=10)
        
        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        
        logging.info(f"Daily water level change correlation plot saved to {output_plot_path}")
        return output_plot_path
    
    except Exception as e:
        logging.error(f"Error creating water level change correlation plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
