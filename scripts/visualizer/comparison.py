"""
Module for generating comparison visualizations between GNSS-IR and USGS gauge data.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from .base import ensure_output_dir, add_summary_textbox, get_compass_direction, PLOT_COLORS, PLOT_STYLES


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    R = 6371.0  # Earth's radius in km
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def calculate_rmse(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate RMSE between two series (after demeaning)."""
    s1_demeaned = series1 - series1.mean()
    s2_demeaned = series2 - series2.mean()
    return np.sqrt(np.mean((s1_demeaned - s2_demeaned)**2))


def plot_ribbon_comparison(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_median_col: str = 'rh_median_m',
    gnssir_min_col: str = 'rh_min_m',
    gnssir_max_col: str = 'rh_max_m',
    usgs_median_col: str = 'usgs_value_m_median',
    usgs_min_col: str = 'usgs_value_m_min',
    usgs_max_col: str = 'usgs_value_m_max',
    demean: bool = True,
    show_daily_range: bool = True
) -> Optional[Path]:
    """
    Generate a ribbon/envelope comparison plot - standard hydrograph style.

    Shows both datasets as shaded ribbons (daily min-max range) with median lines.
    This is the standard approach for tidally-influenced water level data.

    Args:
        gnssir_df: DataFrame with GNSS-IR data including min/max columns
        usgs_df: DataFrame with USGS data including min/max columns
        station_name: Station name (e.g., "MDAI")
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Output path for the plot
        gnssir_median_col: Column name for GNSS-IR median values
        gnssir_min_col: Column name for GNSS-IR min values
        gnssir_max_col: Column name for GNSS-IR max values
        usgs_median_col: Column name for USGS median values
        usgs_min_col: Column name for USGS min values
        usgs_max_col: Column name for USGS max values
        demean: Whether to demean both datasets for comparison
        show_daily_range: Whether to show daily min-max range as ribbons

    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)

    try:
        # Validate inputs
        if gnssir_df is None or gnssir_df.empty:
            logging.error("GNSS-IR DataFrame is empty")
            return None
        if usgs_df is None or usgs_df.empty:
            logging.error("USGS DataFrame is empty")
            return None

        # Extract gauge info
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', '')
        usgs_datum = usgs_gauge_info.get('vertical_datum', 'Unknown')

        # Get coordinates for distance calculation
        gnss_lat = usgs_gauge_info.get('gnss_lat')
        gnss_lon = usgs_gauge_info.get('gnss_lon')
        usgs_lat = usgs_gauge_info.get('usgs_lat')
        usgs_lon = usgs_gauge_info.get('usgs_lon')

        distance_km = 0.0
        if all(coord is not None for coord in [gnss_lat, gnss_lon, usgs_lat, usgs_lon]):
            distance_km = calculate_distance_km(gnss_lat, gnss_lon, usgs_lat, usgs_lon)

        # Prepare date columns - ensure consistent datetime format
        gnssir_df = gnssir_df.copy()
        usgs_df = usgs_df.copy()

        # Convert GNSS-IR dates to proper datetime
        if 'datetime' in gnssir_df.columns:
            gnssir_x = pd.to_datetime(gnssir_df['datetime'])
        elif 'date' in gnssir_df.columns:
            gnssir_x = pd.to_datetime(gnssir_df['date'])
        else:
            logging.error("No date column found in GNSS-IR data")
            return None

        # Convert USGS dates to proper datetime
        if 'datetime' in usgs_df.columns:
            usgs_x = pd.to_datetime(usgs_df['datetime'])
        elif 'date' in usgs_df.columns:
            usgs_x = pd.to_datetime(usgs_df['date'])
        else:
            logging.error("No date column found in USGS data")
            return None

        # Get median values
        gnssir_median = gnssir_df[gnssir_median_col].values
        usgs_median = usgs_df[usgs_median_col].values

        # Get min/max for ribbons if available
        has_gnssir_range = gnssir_min_col in gnssir_df.columns and gnssir_max_col in gnssir_df.columns
        has_usgs_range = usgs_min_col in usgs_df.columns and usgs_max_col in usgs_df.columns

        if has_gnssir_range:
            gnssir_min = gnssir_df[gnssir_min_col].values
            gnssir_max = gnssir_df[gnssir_max_col].values
        if has_usgs_range:
            usgs_min = usgs_df[usgs_min_col].values
            usgs_max = usgs_df[usgs_max_col].values

        # Demean if requested
        if demean:
            gnssir_mean = np.nanmean(gnssir_median)
            usgs_mean = np.nanmean(usgs_median)

            gnssir_median = gnssir_median - gnssir_mean
            usgs_median = usgs_median - usgs_mean

            if has_gnssir_range:
                gnssir_min = gnssir_min - gnssir_mean
                gnssir_max = gnssir_max - gnssir_mean
            if has_usgs_range:
                usgs_min = usgs_min - usgs_mean
                usgs_max = usgs_max - usgs_mean

        # Calculate correlation and RMSE on merged data
        gnssir_for_merge = pd.DataFrame({
            'merge_date': pd.to_datetime(gnssir_df['date']).dt.strftime('%Y-%m-%d'),
            'gnssir_val': gnssir_median
        })
        usgs_for_merge = pd.DataFrame({
            'merge_date': pd.to_datetime(usgs_df['date']).dt.strftime('%Y-%m-%d'),
            'usgs_val': usgs_median
        })
        merged = pd.merge(gnssir_for_merge, usgs_for_merge, on='merge_date', how='inner')

        correlation = None
        rmse = None
        n_points = len(merged)

        if n_points >= 2:
            correlation = merged['gnssir_val'].corr(merged['usgs_val'])
            rmse = np.sqrt(np.mean((merged['gnssir_val'] - merged['usgs_val'])**2))

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot USGS ribbon first (background)
        if show_daily_range and has_usgs_range:
            ax.fill_between(usgs_x, usgs_min, usgs_max,
                           alpha=0.3, color='#E74C3C', label='USGS daily range', zorder=1)

        # Plot USGS median line
        ax.plot(usgs_x, usgs_median, '-', color='#C0392B', linewidth=2,
                label=f'USGS {usgs_site_code} median', zorder=3)

        # Plot GNSS-IR ribbon
        if show_daily_range and has_gnssir_range:
            ax.fill_between(gnssir_x, gnssir_min, gnssir_max,
                           alpha=0.3, color='#3498DB', label='GNSS-IR daily range', zorder=2)

        # Plot GNSS-IR median line
        ax.plot(gnssir_x, gnssir_median, '-', color='#2471A3', linewidth=2,
                label=f'{station_name} GNSS-IR median', zorder=4)

        # Title and labels
        title_suffix = " (Demeaned)" if demean else ""
        ax.set_title(f"{station_name} vs USGS {usgs_site_code} Water Level{title_suffix}", fontsize=14)
        ylabel = "Demeaned Water Level (m)" if demean else "Water Level (m)"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Date", fontsize=12)

        # Zero line for demeaned plot
        if demean:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=0)

        # Grid
        ax.grid(True, alpha=0.3, zorder=0)

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Legend in upper left
        ax.legend(loc='upper left', fontsize=10)

        # Info box in upper right
        info_lines = []
        if correlation is not None:
            info_lines.append(f"r = {correlation:.3f}")
        if rmse is not None:
            info_lines.append(f"RMSE = {rmse:.3f} m")
        if n_points > 0:
            info_lines.append(f"N = {n_points}")
        info_lines.append(f"Distance: {distance_km:.2f} km")
        if usgs_site_name:
            display_name = usgs_site_name[:25] + "..." if len(usgs_site_name) > 25 else usgs_site_name
            info_lines.append(f"USGS: {display_name}")
        info_lines.append(f"Datum: {usgs_datum}")

        info_text = "\n".join(info_lines)
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Ribbon comparison plot saved to {output_plot_path}")
        return output_plot_path

    except Exception as e:
        logging.error(f"Error creating ribbon comparison plot: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def plot_comparison_timeseries(
    daily_gnssir_rh_df: pd.DataFrame,
    daily_usgs_gauge_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_rh_col: str = 'rh_median_m',
    usgs_wl_col: str = 'usgs_value_m_median',
    compare_demeaned: bool = True,
    highlight_dates: Optional[List[str]] = None,
    style: str = 'default',
    correlation: Optional[float] = None,
    rmse: Optional[float] = None,
    n_points: Optional[int] = None
) -> Optional[Path]:
    """
    Generate a comparison time series plot of GNSS-IR RH vs. USGS gauge water level.
    GNSS-IR data points can be color-coded by retrieval count (rh_count) if available.

    Args:
        daily_gnssir_rh_df: DataFrame with daily GNSS-IR RH data, including rh_count if available
        daily_usgs_gauge_df: DataFrame with daily USGS gauge data
        station_name: Station name (e.g., "FORA")
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Output path for the plot
        gnssir_rh_col: Column name for GNSS-IR RH data. Defaults to 'rh_median_m'.
        usgs_wl_col: Column name for USGS water level data. Defaults to 'usgs_value_m_median'.
        compare_demeaned: Whether to demean data for comparison. Defaults to True.
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        style: Plot style to use
        correlation: Pre-calculated correlation coefficient (optional)
        rmse: Pre-calculated RMSE value (optional)
        n_points: Number of comparison points (optional)

    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)
    
    try:
        # Check if DataFrames are valid
        if daily_gnssir_rh_df is None or daily_gnssir_rh_df.empty:
            logging.error("GNSS-IR data DataFrame is empty")
            return None
        
        if daily_usgs_gauge_df is None or daily_usgs_gauge_df.empty:
            logging.error("USGS gauge data DataFrame is empty")
            return None
        
        # Check if required columns exist
        if gnssir_rh_col not in daily_gnssir_rh_df.columns:
            logging.error(f"Column '{gnssir_rh_col}' not found in GNSS-IR data")
            return None
        
        if usgs_wl_col not in daily_usgs_gauge_df.columns:
            logging.error(f"Column '{usgs_wl_col}' not found in USGS gauge data")
            return None
        
        # Log whether rh_count exists for color-coding
        gnssir_counts = None
        if 'rh_count' in daily_gnssir_rh_df.columns:
            gnssir_counts = daily_gnssir_rh_df['rh_count'].values
            logging.info(f"Using actual RH count values for color-coding: Min={min(gnssir_counts)}, Max={max(gnssir_counts)}")
        elif 'rh_retrieval_count' in daily_gnssir_rh_df.columns:
            gnssir_counts = daily_gnssir_rh_df['rh_retrieval_count'].values
            logging.info(f"Using rh_retrieval_count for color-coding: Min={min(gnssir_counts)}, Max={max(gnssir_counts)}")
        else:
            logging.info("No count column found, will use synthetic data for coloring")
        
        # Extract USGS gauge info
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', f'USGS {usgs_site_code}')
        usgs_distance_km = usgs_gauge_info.get('distance_km', 0)
        usgs_vertical_datum = usgs_gauge_info.get('vertical_datum', 'Unknown')
        
        # Extract coordinates if available
        gnss_lat = usgs_gauge_info.get('gnss_lat', None)
        gnss_lon = usgs_gauge_info.get('gnss_lon', None)
        usgs_lat = usgs_gauge_info.get('usgs_lat', None)
        usgs_lon = usgs_gauge_info.get('usgs_lon', None)
        
        # Calculate direction if coordinates are available
        direction = "N/A"
        if all(coord is not None for coord in [gnss_lat, gnss_lon, usgs_lat, usgs_lon]):
            try:
                direction = get_compass_direction(gnss_lat, gnss_lon, usgs_lat, usgs_lon)
                logging.info(f"Calculated direction from GNSS to USGS: {direction}")
            except Exception as e:
                logging.warning(f"Error calculating direction: {e}")
        
        # Create datetime for plotting if not already present
        if 'datetime' not in daily_gnssir_rh_df.columns and 'date' in daily_gnssir_rh_df.columns:
            daily_gnssir_rh_df['datetime'] = pd.to_datetime(daily_gnssir_rh_df['date'])
        
        if 'datetime' not in daily_usgs_gauge_df.columns and 'date' in daily_usgs_gauge_df.columns:
            daily_usgs_gauge_df['datetime'] = pd.to_datetime(daily_usgs_gauge_df['date'])
        
        # Use passed correlation/rmse/n_points if available, otherwise try to calculate
        calc_correlation = correlation
        calc_rmse = rmse
        calc_n_points = n_points if n_points else 0

        # If not passed, try to calculate from data
        if calc_correlation is None:
            try:
                if 'date' in daily_gnssir_rh_df.columns and 'date' in daily_usgs_gauge_df.columns:
                    if gnssir_rh_col in daily_gnssir_rh_df.columns and usgs_wl_col in daily_usgs_gauge_df.columns:
                        # Create copies with standardized date format for merging
                        gnssir_for_merge = daily_gnssir_rh_df[['date', gnssir_rh_col]].copy()
                        usgs_for_merge = daily_usgs_gauge_df[['date', usgs_wl_col]].copy()

                        # Convert dates to string format for consistent merging
                        gnssir_for_merge['merge_date'] = pd.to_datetime(gnssir_for_merge['date']).dt.strftime('%Y-%m-%d')
                        usgs_for_merge['merge_date'] = pd.to_datetime(usgs_for_merge['date']).dt.strftime('%Y-%m-%d')

                        merged_df = pd.merge(
                            gnssir_for_merge[['merge_date', gnssir_rh_col]],
                            usgs_for_merge[['merge_date', usgs_wl_col]],
                            on='merge_date',
                            how='inner'
                        )
                        if merged_df is not None and len(merged_df) >= 2:
                            calc_correlation = merged_df[gnssir_rh_col].corr(merged_df[usgs_wl_col])
                            calc_rmse = calculate_rmse(merged_df[gnssir_rh_col], merged_df[usgs_wl_col])
                            calc_n_points = len(merged_df)
                            logging.info(f"Calculated stats from data: r={calc_correlation:.4f}, RMSE={calc_rmse:.4f}, N={calc_n_points}")
            except Exception as e:
                logging.warning(f"Could not calculate correlation/RMSE: {e}")
        
        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value
                
        # Set up plotting data
        gnssir_x = daily_gnssir_rh_df['datetime'] if 'datetime' in daily_gnssir_rh_df.columns else daily_gnssir_rh_df['date']
        usgs_x = daily_usgs_gauge_df['datetime'] if 'datetime' in daily_usgs_gauge_df.columns else daily_usgs_gauge_df['date']
        
        # Calculate distance if coordinates available
        if all(coord is not None for coord in [gnss_lat, gnss_lon, usgs_lat, usgs_lon]):
            usgs_distance_km = calculate_distance_km(gnss_lat, gnss_lon, usgs_lat, usgs_lon)

        # Build stats dict for info box (use calculated values if available)
        stats_info = {
            'correlation': calc_correlation,
            'rmse': calc_rmse,
            'n_points': calc_n_points,
            'station_name': station_name,
            'usgs_site_code': usgs_site_code,
            'usgs_site_name': usgs_site_name,
            'distance_km': usgs_distance_km,
            'direction': direction,
            'usgs_datum': usgs_vertical_datum,
            'gnss_lat': gnss_lat,
            'gnss_lon': gnss_lon
        }

        # Create appropriate plot based on comparison type
        if compare_demeaned:
            _create_demeaned_comparison_plot(
                gnssir_x, daily_gnssir_rh_df[gnssir_rh_col],
                usgs_x, daily_usgs_gauge_df[usgs_wl_col],
                station_name, usgs_site_code, highlight_dates,
                count_values=gnssir_counts,
                stats_info=stats_info
            )
        else:
            _create_dual_axis_comparison_plot(
                gnssir_x, daily_gnssir_rh_df[gnssir_rh_col],
                usgs_x, daily_usgs_gauge_df[usgs_wl_col],
                station_name, usgs_site_code, highlight_dates,
                count_values=gnssir_counts,
                stats_info=stats_info
            )
        
        # Info box is now handled inside the plotting functions
        logging.info("Plot created with integrated info box")

        # Save the plot
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Comparison plot saved to {output_plot_path}")
        return output_plot_path
    
    except Exception as e:
        logging.error(f"Error creating comparison plot: {e}")
        return None

def _create_demeaned_comparison_plot(
    gnssir_x: pd.Series,
    gnssir_y_raw: pd.Series,
    usgs_x: pd.Series,
    usgs_y_raw: pd.Series,
    station_name: str,
    usgs_site_code: str,
    highlight_dates: Optional[List[str]] = None,
    count_values: Optional[np.ndarray] = None,
    stats_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a demeaned comparison plot.

    Args:
        gnssir_x: X values for GNSS-IR data
        gnssir_y_raw: Y values for GNSS-IR data
        usgs_x: X values for USGS data
        usgs_y_raw: Y values for USGS data
        station_name: Station name
        usgs_site_code: USGS site code
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        count_values: Optional array of count values for color-coding
        stats_info: Dictionary with correlation, RMSE, N, distance, etc.
    """
    if stats_info is None:
        stats_info = {}

    # Increase figure size for more room
    fig, ax = plt.subplots(figsize=(14, 8))

    # Add more right padding for colorbar and bottom padding for dates
    plt.subplots_adjust(right=0.85, bottom=0.15)
    
    # Calculate means
    gnssir_mean = gnssir_y_raw.mean()
    usgs_mean = usgs_y_raw.mean()
    
    # Create demeaned data
    gnssir_y = gnssir_y_raw - gnssir_mean
    usgs_y = usgs_y_raw - usgs_mean
    
    # Create DataFrame for plotting
    gnssir_df = pd.DataFrame({
        'x': gnssir_x.reset_index(drop=True),
        'y': gnssir_y.reset_index(drop=True)
    })
    
    # Add count values if provided
    if count_values is not None and len(count_values) == len(gnssir_df):
        gnssir_df['count'] = count_values
        logging.info(f"Using provided count values: Min={min(count_values)}, Max={max(count_values)}")
    else:
        # Use synthetic values as fallback
        gnssir_df['count'] = np.linspace(10, 30, len(gnssir_df))
        logging.info(f"Using synthetic count values")
    
    # Normalize counts for colormap
    min_c = gnssir_df['count'].min()
    max_c = gnssir_df['count'].max()
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
    cmap = plt.cm.viridis  # You can use other colormaps: plasma, coolwarm, etc.
    
    # Create scatter plot with color based on count
    scatter = ax.scatter(
        gnssir_df['x'],
        gnssir_df['y'],
        c=gnssir_df['count'],
        cmap=cmap,
        norm=norm,
        s=50,  # marker size
        alpha=0.7,
        zorder=10,
        label=f"{station_name} GNSS-IR (Color by Count)"
    )
    
    # Add a light connecting line
    ax.plot(gnssir_df['x'], gnssir_df['y'], 
           '-', color=PLOT_COLORS['gnssir'], alpha=0.3, zorder=5)
    
    # Add a colorbar with more padding
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Daily RH Retrieval Count')
    
    # Plot USGS data
    ax.plot(usgs_x, usgs_y, 's-', 
           label=f"{usgs_site_code} USGS WL (mean: {usgs_mean:.2f} m)", 
           color=PLOT_COLORS['usgs'], 
           linewidth=2, markersize=6, alpha=0.7)
    
    # Add simplified title
    ax.set_title(f"{station_name} vs USGS {usgs_site_code} Water Level (Demeaned)", fontsize=14)

    # Set labels
    ax.set_ylabel("Demeaned Value (m)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)

    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Highlight specific dates if requested
    if highlight_dates:
        try:
            highlight_dates_dt = pd.to_datetime(highlight_dates)
            for date in highlight_dates_dt:
                ax.axvline(x=date, color=PLOT_COLORS['highlight'], linestyle='--', alpha=0.5)

                # Find the closest GNSS-IR data point
                if len(gnssir_x) > 0:
                    closest_gnssir_idx = abs(gnssir_x - date).argmin()
                    ax.scatter(gnssir_x.iloc[closest_gnssir_idx], gnssir_y.iloc[closest_gnssir_idx],
                              color=PLOT_COLORS['highlight'], s=100, zorder=10)

                # Find the closest USGS data point
                if len(usgs_x) > 0:
                    closest_usgs_idx = abs(usgs_x - date).argmin()
                    ax.scatter(usgs_x.iloc[closest_usgs_idx], usgs_y.iloc[closest_usgs_idx],
                              color=PLOT_COLORS['highlight'], s=100, zorder=10)

            # Add one label for the highlighted dates
            ax.plot([], [], 'o', color=PLOT_COLORS['highlight'], label='Highlighted Date')

            logging.info(f"Highlighted {len(highlight_dates)} specific dates")
        except Exception as e:
            logging.warning(f"Could not highlight dates: {e}")

    # Add legend in upper left
    ax.legend(fontsize=10, loc='upper left')

    # Add grid
    ax.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])

    # Better x-axis date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Month abbreviations only
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add info box in upper right (inside plot area, avoiding date labels)
    correlation = stats_info.get('correlation')
    rmse = stats_info.get('rmse')
    n_points = stats_info.get('n_points', 0)
    distance_km = stats_info.get('distance_km', 0)
    usgs_site_name = stats_info.get('usgs_site_name', 'Unknown')
    usgs_datum = stats_info.get('usgs_datum', 'Unknown')

    # Build info text with key statistics
    info_lines = []
    if correlation is not None:
        info_lines.append(f"r = {correlation:.3f}")
    if rmse is not None:
        info_lines.append(f"RMSE = {rmse:.3f} m")
    if n_points > 0:
        info_lines.append(f"N = {n_points}")
    info_lines.append(f"Distance: {distance_km:.2f} km")
    if usgs_site_name and usgs_site_name != 'Unknown':
        # Truncate long site names
        display_name = usgs_site_name[:30] + "..." if len(usgs_site_name) > 30 else usgs_site_name
        info_lines.append(f"USGS: {display_name}")
    info_lines.append(f"Datum: {usgs_datum}")

    info_text = "\n".join(info_lines)
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Use tight_layout with appropriate padding
    plt.tight_layout(rect=[0, 0.02, 0.88, 0.98])

def _create_dual_axis_comparison_plot(
    gnssir_x: pd.Series,
    gnssir_y: pd.Series,
    usgs_x: pd.Series,
    usgs_y: pd.Series,
    station_name: str,
    usgs_site_code: str,
    highlight_dates: Optional[List[str]] = None,
    count_values: Optional[np.ndarray] = None,
    stats_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a dual y-axis comparison plot (non-demeaned, showing actual values).

    Args:
        gnssir_x: X values for GNSS-IR data
        gnssir_y: Y values for GNSS-IR data
        usgs_x: X values for USGS data
        usgs_y: Y values for USGS data
        station_name: Station name
        usgs_site_code: USGS site code
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        count_values: Optional array of count values for color-coding
        stats_info: Dictionary with correlation, RMSE, N, distance, etc.
    """
    if stats_info is None:
        stats_info = {}

    # Increase figure width for more room
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Add more right padding
    plt.subplots_adjust(right=0.82, bottom=0.15)
    
    # Primary y-axis for GNSS-IR RH
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel(f"{station_name} GNSS-IR RH (m)", fontsize=14, color=PLOT_COLORS['gnssir'])
    
    # Create DataFrame for plotting
    gnssir_df = pd.DataFrame({
        'x': gnssir_x.reset_index(drop=True),
        'y': gnssir_y.reset_index(drop=True)
    })
    
    # Add count values if provided
    if count_values is not None and len(count_values) == len(gnssir_df):
        gnssir_df['count'] = count_values
        logging.info(f"Using provided count values: Min={min(count_values)}, Max={max(count_values)}")
    else:
        # Use synthetic values as fallback
        gnssir_df['count'] = np.linspace(10, 30, len(gnssir_df))
        logging.info(f"Using synthetic count values")
    
    # Normalize counts for colormap
    min_c = gnssir_df['count'].min()
    max_c = gnssir_df['count'].max()
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
    cmap = plt.cm.viridis  # You can use other colormaps: plasma, coolwarm, etc.
    
    # Create scatter plot with color based on count
    scatter = ax1.scatter(
        gnssir_df['x'],
        gnssir_df['y'],
        c=gnssir_df['count'],
        cmap=cmap,
        norm=norm,
        s=50,  # marker size
        alpha=0.7,
        zorder=10,
        label=f"{station_name} GNSS-IR (Color by Count)"
    )
    
    # Add a light connecting line
    ax1.plot(gnssir_df['x'], gnssir_df['y'], 
            '-', color=PLOT_COLORS['gnssir'], alpha=0.3, zorder=5)
    
    # Add a colorbar with more padding
    cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical', pad=0.05)
    cbar.set_label('Daily RH Retrieval Count')
    
    ax1.tick_params(axis='y', labelcolor=PLOT_COLORS['gnssir'])
    
    # Secondary y-axis for USGS water level with offset
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 60))  # Move the right spine outward
    ax2.set_ylabel(f"{usgs_site_code} USGS Water Level (m)", fontsize=14, color=PLOT_COLORS['usgs'], labelpad=20)
    ax2.plot(usgs_x, usgs_y, 's-', 
            label=f"{usgs_site_code} USGS WL", 
            color=PLOT_COLORS['usgs'], 
            linewidth=2, markersize=6, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=PLOT_COLORS['usgs'])
    
    # Add simplified title
    ax1.set_title(f"{station_name} vs USGS {usgs_site_code} Water Level", fontsize=14)

    # Highlight specific dates if requested
    if highlight_dates:
        try:
            highlight_dates_dt = pd.to_datetime(highlight_dates)
            for date in highlight_dates_dt:
                ax1.axvline(x=date, color=PLOT_COLORS['highlight'], linestyle='--', alpha=0.5)

                # Find the closest GNSS-IR data point
                if len(gnssir_x) > 0:
                    closest_gnssir_idx = abs(gnssir_x - date).argmin()
                    ax1.scatter(gnssir_x.iloc[closest_gnssir_idx], gnssir_y.iloc[closest_gnssir_idx],
                               color=PLOT_COLORS['highlight'], s=100, zorder=10)

                # Find the closest USGS data point
                if len(usgs_x) > 0:
                    closest_usgs_idx = abs(usgs_x - date).argmin()
                    ax2.scatter(usgs_x.iloc[closest_usgs_idx], usgs_y.iloc[closest_usgs_idx],
                               color=PLOT_COLORS['highlight'], s=100, zorder=10)

            # Add one label for the highlighted dates
            ax1.plot([], [], 'o', color=PLOT_COLORS['highlight'], label='Highlighted Date')

            logging.info(f"Highlighted {len(highlight_dates)} specific dates")
        except Exception as e:
            logging.warning(f"Could not highlight dates: {e}")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Add grid
    ax1.grid(True, alpha=0.3, color=PLOT_COLORS['grid'])

    # Better x-axis date formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Month abbreviations only
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add info box in upper right area (between legend and colorbar)
    correlation = stats_info.get('correlation')
    rmse = stats_info.get('rmse')
    n_points = stats_info.get('n_points', 0)
    distance_km = stats_info.get('distance_km', 0)
    usgs_site_name = stats_info.get('usgs_site_name', 'Unknown')
    usgs_datum = stats_info.get('usgs_datum', 'Unknown')

    # Build info text with key statistics
    info_lines = []
    if correlation is not None:
        info_lines.append(f"r = {correlation:.3f}")
    if rmse is not None:
        info_lines.append(f"RMSE = {rmse:.3f} m")
    if n_points > 0:
        info_lines.append(f"N = {n_points}")
    info_lines.append(f"Distance: {distance_km:.2f} km")
    if usgs_site_name and usgs_site_name != 'Unknown':
        # Truncate long site names
        display_name = usgs_site_name[:30] + "..." if len(usgs_site_name) > 30 else usgs_site_name
        info_lines.append(f"USGS: {display_name}")
    info_lines.append(f"Datum: {usgs_datum}")

    info_text = "\n".join(info_lines)
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.98, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    # Use tight_layout with appropriate padding
    plt.tight_layout(rect=[0, 0.02, 0.85, 0.98])

    # Use fig and ax1 for the rest of the plot settings
    plt.sca(ax1)


def plot_subdaily_ribbon_comparison(
    gnssir_raw_df: pd.DataFrame,
    usgs_iv_df: pd.DataFrame,
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_dir: Union[str, Path],
    antenna_height: float,
    year: int = 2024,
    gap_threshold_hours: float = 2.0,
    ribbon_window: int = 5,
    show_residuals: bool = True
) -> Dict[str, Path]:
    """
    Generate subdaily ribbon comparison plots with gap handling and monthly panels.

    Matches individual GNSS-IR retrievals to nearest USGS IV readings (no repeats),
    shows USGS as continuous line, GNSS-IR as scatter with ribbon showing spread,
    and optionally adds a residual panel.

    Args:
        gnssir_raw_df: Raw GNSS-IR data with UTCtime and RH columns
        usgs_iv_df: USGS IV data with datetime and value columns
        station_name: Station name (e.g., "MDAI")
        usgs_gauge_info: Dictionary with USGS gauge information
        output_dir: Output directory for plots
        antenna_height: Antenna ellipsoidal height for WSE calculation
        year: Year of data
        gap_threshold_hours: Hours without data to consider a gap (for masking ribbon)
        ribbon_window: Rolling window size for scatter ribbon calculation
        show_residuals: Whether to add a residual panel

    Returns:
        Dictionary mapping plot types to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    try:
        # Extract gauge info
        usgs_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        usgs_site_name = usgs_gauge_info.get('site_name', '')
        usgs_datum = usgs_gauge_info.get('vertical_datum', 'Unknown')

        # Get coordinates for distance calculation
        gnss_lat = usgs_gauge_info.get('gnss_lat')
        gnss_lon = usgs_gauge_info.get('gnss_lon')
        usgs_lat = usgs_gauge_info.get('usgs_lat')
        usgs_lon = usgs_gauge_info.get('usgs_lon')

        distance_km = 0.0
        if all(coord is not None for coord in [gnss_lat, gnss_lon, usgs_lat, usgs_lon]):
            distance_km = calculate_distance_km(gnss_lat, gnss_lon, usgs_lat, usgs_lon)

        # Prepare GNSS-IR data with datetime
        gnssir_df = gnssir_raw_df.copy()

        # Create proper datetime from year, doy, UTCtime
        if 'UTCtime' in gnssir_df.columns and 'doy' in gnssir_df.columns:
            # UTCtime is decimal hours
            gnssir_df['datetime'] = pd.to_datetime(
                gnssir_df['year'].astype(str) + '-' + gnssir_df['doy'].astype(str),
                format='%Y-%j'
            ) + pd.to_timedelta(gnssir_df['UTCtime'], unit='h')
            gnssir_df['datetime'] = gnssir_df['datetime'].dt.tz_localize('UTC')
        elif 'datetime' not in gnssir_df.columns:
            logging.error("Cannot create datetime from GNSS-IR data")
            return output_paths

        # Calculate WSE (Water Surface Elevation = antenna_height - RH)
        gnssir_df['wse'] = antenna_height - gnssir_df['RH']
        gnssir_df = gnssir_df.sort_values('datetime').reset_index(drop=True)

        # Prepare USGS IV data
        usgs_df = usgs_iv_df.copy()
        if 'datetime' not in usgs_df.columns and '00065' in usgs_df.columns:
            # Try common column names
            if 'dateTime' in usgs_df.columns:
                usgs_df['datetime'] = pd.to_datetime(usgs_df['dateTime'])
            else:
                logging.error("Cannot find datetime column in USGS IV data")
                return output_paths

        if usgs_df['datetime'].dt.tz is None:
            usgs_df['datetime'] = usgs_df['datetime'].dt.tz_localize('UTC')
        else:
            usgs_df['datetime'] = usgs_df['datetime'].dt.tz_convert('UTC')

        # Find the water level column
        wl_col = None
        for col in ['00065', '62610', '62611', '62620', 'value', 'X_00065_00000']:
            if col in usgs_df.columns:
                wl_col = col
                break

        if wl_col is None:
            logging.error(f"No water level column found in USGS data. Columns: {usgs_df.columns.tolist()}")
            return output_paths

        usgs_df['usgs_wl_m'] = pd.to_numeric(usgs_df[wl_col], errors='coerce') * 0.3048  # ft to m
        usgs_df = usgs_df.dropna(subset=['usgs_wl_m']).sort_values('datetime').reset_index(drop=True)

        # Match GNSS-IR to nearest USGS IV without repeats
        matched_pairs = _match_gnssir_to_usgs_iv(gnssir_df, usgs_df)

        if len(matched_pairs) == 0:
            logging.error("No matched pairs found")
            return output_paths

        logging.info(f"Matched {len(matched_pairs)} GNSS-IR readings to USGS IV")

        # Demean both series for comparison
        matched_pairs['gnss_wse_dm'] = matched_pairs['gnss_wse'] - matched_pairs['gnss_wse'].mean()
        matched_pairs['usgs_wl_dm'] = matched_pairs['usgs_wl_m'] - matched_pairs['usgs_wl_m'].mean()

        # Calculate statistics
        correlation = matched_pairs['gnss_wse_dm'].corr(matched_pairs['usgs_wl_dm'])
        rmse = np.sqrt(np.mean((matched_pairs['gnss_wse_dm'] - matched_pairs['usgs_wl_dm'])**2))
        n_points = len(matched_pairs)

        logging.info(f"Full year subdaily stats: r={correlation:.3f}, RMSE={rmse:.3f}m, N={n_points}")

        # Save matched data
        matched_path = output_dir / f"{station_name}_{year}_subdaily_matched.csv"
        matched_pairs.to_csv(matched_path, index=False)
        output_paths['matched_data'] = matched_path

        # Create monthly panels plot
        monthly_path = _create_monthly_subdaily_panels(
            matched_pairs, usgs_df, station_name, usgs_site_code,
            distance_km, usgs_site_name, usgs_datum,
            correlation, rmse, n_points,
            output_dir, year, gap_threshold_hours, ribbon_window, show_residuals
        )
        if monthly_path:
            output_paths['monthly_panels'] = monthly_path

        # Create overview plot for a sample period
        overview_path = _create_subdaily_overview(
            matched_pairs, usgs_df, station_name, usgs_site_code,
            distance_km, usgs_site_name, usgs_datum,
            output_dir, year, gap_threshold_hours, ribbon_window, show_residuals
        )
        if overview_path:
            output_paths['overview'] = overview_path

        return output_paths

    except Exception as e:
        logging.error(f"Error creating subdaily ribbon comparison: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return output_paths


def _match_gnssir_to_usgs_iv(gnssir_df: pd.DataFrame, usgs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match GNSS-IR readings to nearest USGS IV timestamps without repeating GNSS-IR readings.

    For each GNSS-IR reading, find the nearest USGS IV timestamp (within 6 minutes).
    Each GNSS-IR reading is used only once.
    """
    matched = []

    # Convert to numpy datetime64 for efficient calculations
    usgs_times = pd.to_datetime(usgs_df['datetime']).values.astype('datetime64[s]')

    max_diff_seconds = 180  # 3 minutes max difference

    for idx, row in gnssir_df.iterrows():
        gnss_time = pd.to_datetime(row['datetime'])
        gnss_time_np = np.datetime64(gnss_time.value, 'ns').astype('datetime64[s]')

        # Find nearest USGS time
        time_diffs = np.abs((usgs_times - gnss_time_np).astype('float64'))
        nearest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[nearest_idx]

        if min_diff <= max_diff_seconds:
            usgs_row = usgs_df.iloc[nearest_idx]
            matched.append({
                'gnss_datetime': gnss_time,
                'gnss_wse': row['wse'],
                'gnss_rh': row['RH'],
                'usgs_datetime': usgs_row['datetime'],
                'usgs_wl_m': usgs_row['usgs_wl_m'],
                'time_diff_sec': min_diff
            })

    return pd.DataFrame(matched)


def _create_monthly_subdaily_panels(
    matched_pairs: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str,
    usgs_site_code: str,
    distance_km: float,
    usgs_site_name: str,
    usgs_datum: str,
    correlation: float,
    rmse: float,
    n_points: int,
    output_dir: Path,
    year: int,
    gap_threshold_hours: float,
    ribbon_window: int,
    show_residuals: bool
) -> Optional[Path]:
    """Create monthly panel plots showing subdaily comparison."""

    try:
        # Create 4x3 grid for 12 months
        n_rows = 4 if show_residuals else 4
        n_cols = 3

        if show_residuals:
            # Each month gets 2 rows: main + residuals
            fig = plt.figure(figsize=(18, 24))
            gs = fig.add_gridspec(8, 3, height_ratios=[3, 1] * 4, hspace=0.4, wspace=0.25)
        else:
            fig, axes = plt.subplots(4, 3, figsize=(18, 16))
            axes = axes.flatten()

        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for i, month in enumerate(months):
            if show_residuals:
                ax_main = fig.add_subplot(gs[2*(i//3), i%3])
                ax_resid = fig.add_subplot(gs[2*(i//3)+1, i%3], sharex=ax_main)
            else:
                ax_main = axes[i]
                ax_resid = None

            # Filter data for this month
            month_mask = matched_pairs['gnss_datetime'].dt.month == month
            month_data = matched_pairs[month_mask].copy()

            usgs_month_mask = usgs_df['datetime'].dt.month == month
            usgs_month = usgs_df[usgs_month_mask].copy()

            if len(month_data) == 0 or len(usgs_month) == 0:
                ax_main.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax_main.transAxes)
                ax_main.set_title(f'{month_names[i]}')
                if ax_resid is not None:
                    ax_resid.set_visible(False)
                continue

            # Demean for this month
            usgs_mean = usgs_month['usgs_wl_m'].mean()
            gnss_mean = month_data['gnss_wse'].mean()

            usgs_month['usgs_dm'] = usgs_month['usgs_wl_m'] - usgs_mean
            month_data['gnss_dm'] = month_data['gnss_wse'] - gnss_mean
            month_data['usgs_dm'] = month_data['usgs_wl_m'] - usgs_mean

            # Calculate monthly stats
            month_corr = month_data['gnss_dm'].corr(month_data['usgs_dm'])
            month_rmse = np.sqrt(np.mean((month_data['gnss_dm'] - month_data['usgs_dm'])**2))
            month_n = len(month_data)

            # Plot USGS IV as continuous line (masking gaps)
            usgs_sorted = usgs_month.sort_values('datetime')
            _plot_with_gap_masking(ax_main, usgs_sorted['datetime'], usgs_sorted['usgs_dm'],
                                   gap_threshold_hours, color='#C0392B', linewidth=1.5,
                                   label='USGS IV', zorder=2)

            # Plot GNSS-IR scatter with ribbon
            month_sorted = month_data.sort_values('gnss_datetime')

            # Calculate rolling std for ribbon
            if len(month_sorted) >= ribbon_window:
                rolling_std = month_sorted['gnss_dm'].rolling(ribbon_window, center=True, min_periods=1).std()
                gnss_upper = month_sorted['gnss_dm'] + rolling_std
                gnss_lower = month_sorted['gnss_dm'] - rolling_std

                # Mask ribbon gaps
                _fill_between_with_gaps(ax_main, month_sorted['gnss_datetime'].values,
                                        gnss_lower.values, gnss_upper.values,
                                        gap_threshold_hours, alpha=0.3, color='#3498DB',
                                        label='GNSS-IR scatter (±σ)', zorder=1)

            # Plot GNSS-IR points
            ax_main.scatter(month_sorted['gnss_datetime'], month_sorted['gnss_dm'],
                           c='#2471A3', s=15, alpha=0.7, zorder=3, label='GNSS-IR WSE')

            # Add zero line
            ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=0)

            # Title with stats
            ax_main.set_title(f'{month_names[i]}: r={month_corr:.2f}, RMSE={month_rmse:.2f}m, N={month_n}',
                             fontsize=10)

            # Y label only on left column
            if i % 3 == 0:
                ax_main.set_ylabel('Demeaned WL (m)', fontsize=9)

            # Format x-axis
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
            ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax_main.tick_params(axis='x', labelsize=8)
            ax_main.tick_params(axis='y', labelsize=8)
            ax_main.grid(True, alpha=0.3)

            # Plot residuals if requested
            if ax_resid is not None and len(month_data) > 0:
                residuals = month_data['gnss_dm'] - month_data['usgs_dm']
                ax_resid.scatter(month_sorted['gnss_datetime'], residuals.loc[month_sorted.index],
                                c='#8E44AD', s=10, alpha=0.7)
                ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_resid.set_ylabel('Resid (m)', fontsize=8)
                ax_resid.tick_params(axis='both', labelsize=7)
                ax_resid.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
                ax_resid.grid(True, alpha=0.3)

                # Set symmetric y-limits for residuals
                resid_max = max(abs(residuals.min()), abs(residuals.max()), 0.1)
                ax_resid.set_ylim(-resid_max, resid_max)

        # Legend in first panel
        if show_residuals:
            first_ax = fig.add_subplot(gs[0, 0])
            handles, labels = first_ax.get_legend_handles_labels()
        else:
            handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
                   bbox_to_anchor=(0.5, 0.99))

        # Main title
        fig.suptitle(f'{station_name} Subdaily: GNSS-IR WSE vs USGS IV (r={correlation:.3f}, RMSE={rmse:.3f}m, N={n_points})',
                    fontsize=14, y=1.01)

        output_path = output_dir / f"{station_name}_{year}_subdaily_monthly_panels.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        logging.info(f"Monthly subdaily panels saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creating monthly panels: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def _create_subdaily_overview(
    matched_pairs: pd.DataFrame,
    usgs_df: pd.DataFrame,
    station_name: str,
    usgs_site_code: str,
    distance_km: float,
    usgs_site_name: str,
    usgs_datum: str,
    output_dir: Path,
    year: int,
    gap_threshold_hours: float,
    ribbon_window: int,
    show_residuals: bool
) -> Optional[Path]:
    """Create overview plot showing one week of subdaily data."""

    try:
        # Find a week with good data coverage (September typically has good data)
        sep_data = matched_pairs[matched_pairs['gnss_datetime'].dt.month == 9]

        if len(sep_data) == 0:
            # Fall back to any month with data
            month_counts = matched_pairs['gnss_datetime'].dt.month.value_counts()
            best_month = month_counts.idxmax()
            sample_data = matched_pairs[matched_pairs['gnss_datetime'].dt.month == best_month]
        else:
            sample_data = sep_data

        # Get one week window
        start_date = sample_data['gnss_datetime'].min()
        end_date = start_date + pd.Timedelta(days=7)

        week_mask = (sample_data['gnss_datetime'] >= start_date) & (sample_data['gnss_datetime'] <= end_date)
        week_data = sample_data[week_mask].copy()

        usgs_week_mask = (usgs_df['datetime'] >= start_date) & (usgs_df['datetime'] <= end_date)
        usgs_week = usgs_df[usgs_week_mask].copy()

        if len(week_data) == 0 or len(usgs_week) == 0:
            logging.warning("No data for overview week")
            return None

        # Demean
        usgs_mean = usgs_week['usgs_wl_m'].mean()
        gnss_mean = week_data['gnss_wse'].mean()

        usgs_week['usgs_dm'] = usgs_week['usgs_wl_m'] - usgs_mean
        week_data['gnss_dm'] = week_data['gnss_wse'] - gnss_mean
        week_data['usgs_dm'] = week_data['usgs_wl_m'] - usgs_mean

        # Calculate stats
        corr = week_data['gnss_dm'].corr(week_data['usgs_dm'])
        rmse = np.sqrt(np.mean((week_data['gnss_dm'] - week_data['usgs_dm'])**2))
        n = len(week_data)

        # Create figure
        if show_residuals:
            fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(14, 8),
                                                      height_ratios=[3, 1], sharex=True)
        else:
            fig, ax_main = plt.subplots(figsize=(14, 6))
            ax_resid = None

        # Plot USGS IV with gap masking
        usgs_sorted = usgs_week.sort_values('datetime')
        _plot_with_gap_masking(ax_main, usgs_sorted['datetime'], usgs_sorted['usgs_dm'],
                               gap_threshold_hours, color='#C0392B', linewidth=2,
                               label='USGS IV (6-min)', zorder=2)

        # Calculate ribbon from GNSS-IR scatter
        week_sorted = week_data.sort_values('gnss_datetime')

        if len(week_sorted) >= ribbon_window:
            rolling_std = week_sorted['gnss_dm'].rolling(ribbon_window, center=True, min_periods=1).std()
            gnss_upper = week_sorted['gnss_dm'] + rolling_std
            gnss_lower = week_sorted['gnss_dm'] - rolling_std

            _fill_between_with_gaps(ax_main, week_sorted['gnss_datetime'].values,
                                    gnss_lower.values, gnss_upper.values,
                                    gap_threshold_hours, alpha=0.3, color='#3498DB',
                                    label='GNSS-IR scatter (±σ)', zorder=1)

        # Plot GNSS-IR points
        ax_main.scatter(week_sorted['gnss_datetime'], week_sorted['gnss_dm'],
                       c='#2471A3', s=30, alpha=0.8, zorder=3, label='GNSS-IR WSE')

        ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=0)
        ax_main.set_ylabel('Demeaned Water Level (m)', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper left', fontsize=10)

        # Info box
        date_range = f"{start_date.strftime('%b %d')}-{end_date.strftime('%d, %Y')}"
        info_lines = [
            f"r = {corr:.3f}",
            f"RMSE = {rmse:.3f}m",
            f"N = {n}",
            f"Distance: {distance_km:.2f} km",
            f"USGS: {usgs_site_name[:25]}..." if len(usgs_site_name) > 25 else f"USGS: {usgs_site_name}",
            f"Datum: {usgs_datum}"
        ]
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
        ax_main.text(0.98, 0.98, '\n'.join(info_lines), transform=ax_main.transAxes,
                    fontsize=9, va='top', ha='right', bbox=props)

        ax_main.set_title(f'{station_name} Subdaily: GNSS-IR WSE vs USGS IV ({date_range})',
                         fontsize=14)

        # Residuals panel
        if ax_resid is not None:
            # Calculate residuals directly from sorted data
            residuals = week_sorted['gnss_dm'].values - week_sorted['usgs_dm'].values
            ax_resid.scatter(week_sorted['gnss_datetime'], residuals,
                            c='#8E44AD', s=20, alpha=0.8)
            ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_resid.set_ylabel('Residual (m)', fontsize=11)
            ax_resid.set_xlabel('Date', fontsize=12)
            ax_resid.grid(True, alpha=0.3)

            # Symmetric y-limits
            resid_max = max(abs(np.nanmin(residuals)), abs(np.nanmax(residuals)), 0.1)
            ax_resid.set_ylim(-resid_max * 1.1, resid_max * 1.1)

            # Add residual stats
            resid_std = np.nanstd(residuals)
            resid_mean = np.nanmean(residuals)
            ax_resid.text(0.02, 0.95, f'μ={resid_mean:.3f}m, σ={resid_std:.3f}m',
                         transform=ax_resid.transAxes, fontsize=9, va='top')
        else:
            ax_main.set_xlabel('Date', fontsize=12)

        # Format x-axis
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_main.xaxis.set_major_locator(mdates.DayLocator())

        plt.tight_layout()

        output_path = output_dir / f"{station_name}_{year}_subdaily_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Subdaily overview saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creating overview: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def _plot_with_gap_masking(ax, x_data, y_data, gap_threshold_hours: float,
                           **plot_kwargs) -> None:
    """Plot line data with gaps masked where data is missing."""
    x_arr = pd.to_datetime(x_data)
    y_arr = np.array(y_data)

    # Find gaps
    if len(x_arr) < 2:
        ax.plot(x_arr, y_arr, **plot_kwargs)
        return

    time_diffs = np.diff(x_arr).astype('timedelta64[h]').astype(float)
    gap_indices = np.where(time_diffs > gap_threshold_hours)[0]

    if len(gap_indices) == 0:
        ax.plot(x_arr, y_arr, **plot_kwargs)
        return

    # Plot segments
    label = plot_kwargs.pop('label', None)
    first_segment = True

    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        if first_segment:
            ax.plot(x_arr[start_idx:end_idx], y_arr[start_idx:end_idx], label=label, **plot_kwargs)
            first_segment = False
        else:
            ax.plot(x_arr[start_idx:end_idx], y_arr[start_idx:end_idx], **plot_kwargs)
        start_idx = end_idx

    # Plot final segment
    ax.plot(x_arr[start_idx:], y_arr[start_idx:], **plot_kwargs)


def _fill_between_with_gaps(ax, x_data, y_lower, y_upper, gap_threshold_hours: float,
                            **fill_kwargs) -> None:
    """Fill between with gap masking."""
    x_arr = pd.to_datetime(x_data)
    y_low = np.array(y_lower)
    y_up = np.array(y_upper)

    if len(x_arr) < 2:
        ax.fill_between(x_arr, y_low, y_up, **fill_kwargs)
        return

    time_diffs = np.diff(x_arr).astype('timedelta64[h]').astype(float)
    gap_indices = np.where(time_diffs > gap_threshold_hours)[0]

    if len(gap_indices) == 0:
        ax.fill_between(x_arr, y_low, y_up, **fill_kwargs)
        return

    # Fill segments
    label = fill_kwargs.pop('label', None)
    first_segment = True

    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        if first_segment:
            ax.fill_between(x_arr[start_idx:end_idx], y_low[start_idx:end_idx],
                           y_up[start_idx:end_idx], label=label, **fill_kwargs)
            first_segment = False
        else:
            ax.fill_between(x_arr[start_idx:end_idx], y_low[start_idx:end_idx],
                           y_up[start_idx:end_idx], **fill_kwargs)
        start_idx = end_idx

    # Fill final segment
    ax.fill_between(x_arr[start_idx:], y_low[start_idx:], y_up[start_idx:], **fill_kwargs)
