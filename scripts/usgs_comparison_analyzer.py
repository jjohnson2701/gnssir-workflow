"""
USGS Comparison Analyzer module for GNSS-IR processing.
This module provides tools to compare GNSS-IR reflector height data with USGS gauge data.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy import signal

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).resolve().parent))

# Import project modules
import usgs_data_handler
import visualizer

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration files
STATIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        return None

def load_gnssir_data(station_name, year, doy_range=None):
    """
    Load GNSS-IR reflector height data for a given station, year, and DOY range.
    
    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
        year (int or str): Year to load data for
        doy_range (tuple, optional): Range of DOYs to include (start, end). 
                                   Defaults to None (all available).
    
    Returns:
        pd.DataFrame: DataFrame containing the reflector height data
    """
    # Convert year to string if needed
    year_str = str(year)
    
    # Define path to combined RH CSV
    rh_csv_path = PROJECT_ROOT / "results_annual" / station_name / f"{station_name}_{year_str}_combined_rh.csv"
    
    if not rh_csv_path.exists():
        logging.error(f"Combined RH CSV file not found at {rh_csv_path}")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(rh_csv_path)
        logging.info(f"Loaded GNSS-IR data from {rh_csv_path}: {len(df)} rows")
        
        # Check if we should filter by DOY range
        if doy_range is not None:
            # Filter by DOY
            doy_min, doy_max = doy_range
            
            # Check if we have 'doy' or 'DOY' in the columns
            doy_col = None
            if 'doy' in df.columns:
                doy_col = 'doy'
            elif 'DOY' in df.columns:
                doy_col = 'DOY'
            else:
                logging.error(f"Could not find 'doy' or 'DOY' column in the data")
                return None
                
            df = df[(df[doy_col] >= doy_min) & (df[doy_col] <= doy_max)]
            logging.info(f"Filtered to DOY range {doy_min}-{doy_max}: {len(df)} rows")
        
        # Generate datetime for easier comparison
        # If we have a date column, use it
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            # Generate datetime from year and DOY
            year_col = 'year' if 'year' in df.columns else 'Year'
            doy_col = 'doy' if 'doy' in df.columns else 'DOY'
            
            if year_col in df.columns and doy_col in df.columns:
                df['datetime'] = pd.to_datetime(df[year_col].astype(str) + 
                                              df[doy_col].astype(str).str.zfill(3), 
                                              format='%Y%j')
            else:
                logging.error(f"Could not find year or DOY columns for datetime generation")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading GNSS-IR data: {e}")
        return None

def aggregate_gnssir_daily(gnssir_df, station_config=None):
    """
    Aggregate GNSS-IR reflector height data to daily statistics.
    Optionally calculates Water Surface Ellipsoidal Height (WSE_ellips) if station_config is provided.
    
    Args:
        gnssir_df (pd.DataFrame): DataFrame containing raw GNSS-IR data
        station_config (dict, optional): Station configuration dictionary with ellipsoidal_height_m
    
    Returns:
        pd.DataFrame: DataFrame with daily aggregated statistics
    """
    if gnssir_df is None or gnssir_df.empty:
        logging.warning("No GNSS-IR data to aggregate")
        return pd.DataFrame()
    
    try:
        # Create a date column (just the date part of the datetime)
        gnssir_df['date'] = gnssir_df['datetime'].dt.date
        
        # Find the reflector height column
        rh_col = None
        for col_name in ['RH', 'rh', 'reflector_height', 'height']:
            if col_name in gnssir_df.columns:
                rh_col = col_name
                break
        
        if rh_col is None:
            logging.warning("Could not identify reflector height column, using column at index 2")
            rh_col = gnssir_df.columns[2]  # Default to third column
        
        # Group by date and calculate statistics
        daily_stats = gnssir_df.groupby('date').agg({
            rh_col: ['count', 'mean', 'median', 'std', 'min', 'max']
        })
        
        # Flatten multi-level columns
        daily_stats.columns = [f"rh_{col[1]}" if col[1] != 'mean' else 'rh_mean_m' 
                             for col in daily_stats.columns.values]
        
        # Ensure we have an rh_median_m column for plotting
        if 'rh_median' in daily_stats.columns:
            daily_stats = daily_stats.rename(columns={'rh_median': 'rh_median_m'})
        
        # Reset index to make date a column
        daily_stats = daily_stats.reset_index()
        
        # Convert date back to datetime for easier plotting
        daily_stats['datetime'] = pd.to_datetime(daily_stats['date'])
        
        # Calculate Water Surface Ellipsoidal Height if station_config provided
        if station_config and 'ellipsoidal_height_m' in station_config:
            antenna_ellipsoidal_height = station_config['ellipsoidal_height_m']
            
            # WSE_ellips = Antenna_Ellipsoidal_Height - RH_median_m
            daily_stats['wse_ellips_m'] = antenna_ellipsoidal_height - daily_stats['rh_median_m']
            
            # Create demeaned versions
            daily_stats['rh_median_m_demeaned'] = daily_stats['rh_median_m'] - daily_stats['rh_median_m'].mean()
            daily_stats['wse_ellips_m_demeaned'] = daily_stats['wse_ellips_m'] - daily_stats['wse_ellips_m'].mean()
            
            logging.info(f"Calculated WSE_ellips using antenna height {antenna_ellipsoidal_height} m")
        
        logging.info(f"Created daily aggregated statistics for GNSS-IR data ({len(daily_stats)} days)")
        return daily_stats
    
    except Exception as e:
        logging.error(f"Error creating daily statistics for GNSS-IR data: {e}")
        return pd.DataFrame()

def calculate_time_lag_correlation(series1, series2, max_lag_days=10):
    """
    Calculate the time lag with maximum correlation between two time series.
    
    Args:
        series1 (pd.Series): First time series
        series2 (pd.Series): Second time series
        max_lag_days (int): Maximum lag to consider in days (both directions)
    
    Returns:
        tuple: (optimal_lag_days, max_correlation, confidence_level, all_correlations)
    """
    try:
        # For shorter time series, limit the maximum lag to a reasonable value
        series_length = min(len(series1), len(series2))
        
        # If we have a very short time series, limit the maximum lag
        # to ensure we have at least 3 overlapping points
        effective_max_lag = min(max_lag_days, series_length - 3)
        if effective_max_lag < max_lag_days:
            logging.info(f"Limited maximum lag to {effective_max_lag} days to ensure at least 3 overlapping points")
        
        if series_length < 4:  # Need at least 4 points to have 2+ points overlap with a lag
            logging.warning("Time series too short for meaningful lag analysis (less than 4 points)")
            return None, None, "Low", {}
        
        # First, ensure both series are demeaned (zero mean) and normalized
        series1_mean = np.mean(series1)
        series1_std = np.std(series1) if np.std(series1) != 0 else 1
        series1_norm = (series1 - series1_mean) / series1_std
        
        series2_mean = np.mean(series2)
        series2_std = np.std(series2) if np.std(series2) != 0 else 1
        series2_norm = (series2 - series2_mean) / series2_std
        
        # Convert to numpy arrays
        s1 = np.array(series1_norm)
        s2 = np.array(series2_norm)
        
        # Initialize variables to track best lag
        best_lag = 0
        best_corr = -np.inf
        all_correlations = {}
        num_overlap_points = {}
        
        # Calculate correlation for different lags up to effective_max_lag in both directions
        for lag in range(-effective_max_lag, effective_max_lag + 1):
            # Calculate the overlap size at this lag
            if lag < 0:
                overlap_size = min(len(s1) + lag, len(s2))
            elif lag > 0:
                overlap_size = min(len(s1) - lag, len(s2))
            else:  # lag == 0
                overlap_size = min(len(s1), len(s2))
            
            # Skip if not enough overlap
            if overlap_size < 3:  # Require at least 3 overlapping points for correlation
                logging.debug(f"Skipping lag {lag} - only {overlap_size} overlapping points")
                all_correlations[lag] = None
                num_overlap_points[lag] = overlap_size
                continue
            
            # Shift the arrays and calculate correlation
            if lag < 0:
                # s2 is ahead of s1 by |lag| steps
                x1 = s1[:lag]
                x2 = s2[-lag:]
            elif lag > 0:
                # s1 is ahead of s2 by lag steps
                x1 = s1[lag:]
                x2 = s2[:-lag]
            else:
                # No lag
                x1 = s1
                x2 = s2
            
            # Calculate correlation
            try:
                corr = np.corrcoef(x1, x2)[0, 1]
                all_correlations[lag] = corr
                num_overlap_points[lag] = overlap_size
                
                # Update best lag if correlation is higher
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
                    
                logging.debug(f"Lag {lag}: corr = {corr:.4f}, overlap = {overlap_size} points")
            except Exception as e:
                logging.error(f"Error calculating correlation for lag {lag}: {e}")
                all_correlations[lag] = None
                num_overlap_points[lag] = overlap_size
        
        # Determine confidence level based on number of overlapping points
        # and correlation strength
        overlap_at_best_lag = num_overlap_points.get(best_lag, 0)
        
        if overlap_at_best_lag >= 7 and abs(best_corr) > 0.8:
            confidence = "High"
        elif overlap_at_best_lag >= 5 and abs(best_corr) > 0.7:
            confidence = "Medium"
        elif overlap_at_best_lag >= 3 and abs(best_corr) > 0.5:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        logging.info(f"Optimal time lag: {best_lag} days with correlation {best_corr:.4f} " +
                   f"(confidence: {confidence}, {overlap_at_best_lag} overlapping points)")
        
        return best_lag, best_corr, confidence, all_correlations
    
    except Exception as e:
        logging.error(f"Error calculating time lag correlation: {e}")
        return None, None, "Error", {}

def create_lag_adjusted_series(series1, series2, lag_days):
    """
    Create lag-adjusted versions of two time series.
    
    Args:
        series1 (pd.Series): First time series
        series2 (pd.Series): Second time series
        lag_days (int): Lag in days (positive means series1 leads series2)
    
    Returns:
        tuple: (adjusted_series1, adjusted_series2)
    """
    try:
        if lag_days == 0:
            return series1, series2
        
        # Convert to numpy arrays
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        # Adjust for lag
        if lag_days > 0:
            # series1 leads series2 by lag_days
            s1_adj = s1[lag_days:]
            s2_adj = s2[:-lag_days]
        else:
            # series2 leads series1 by abs(lag_days)
            lag_days = abs(lag_days)
            s1_adj = s1[:-lag_days]
            s2_adj = s2[lag_days:]
        
        # Convert back to series
        s1_adj_series = pd.Series(s1_adj)
        s2_adj_series = pd.Series(s2_adj)
        
        logging.info(f"Created lag-adjusted series with lag {lag_days} days")
        return s1_adj_series, s2_adj_series
    
    except Exception as e:
        logging.error(f"Error creating lag-adjusted series: {e}")
        return series1, series2

def find_usgs_gauge_for_station(station_config):
    """
    Find the best USGS gauge for a given station based on configuration.
    
    Args:
        station_config (dict): Station configuration dictionary
    
    Returns:
        tuple: (usgs_site_code, parameter_code, gauge_info_dict) or (None, None, None) if no gauge found
    """
    # Check if station config has USGS comparison settings
    if 'usgs_comparison' not in station_config:
        logging.error(f"Station configuration does not have usgs_comparison section")
        return None, None, None
    
    usgs_config = station_config.get('usgs_comparison', {})
    
    # Check if a specific gauge is already configured
    target_site = usgs_config.get('target_usgs_site')
    target_param = usgs_config.get('usgs_parameter_code_to_use')
    
    if target_site and target_param:
        logging.info(f"Using pre-configured USGS gauge: {target_site} with parameter {target_param}")
        gauge_info = {
            'site_code': target_site,
            'parameter_code': target_param,
            'site_name': f"USGS {target_site}",
            'configured': True
        }
        return target_site, target_param, gauge_info
    
    # If not pre-configured, we need to find a gauge
    logging.info(f"No pre-configured USGS gauge. Finding the best gauge.")
    
    # Get station coordinates
    station_lat = station_config.get('latitude_deg')
    station_lon = station_config.get('longitude_deg')
    
    if station_lat is None or station_lon is None:
        logging.error(f"Station configuration does not have latitude_deg or longitude_deg")
        return None, None, None
    
    # Get search parameters from config
    search_radius_km = usgs_config.get('search_radius_km', 50.0)
    priority_param_codes = usgs_config.get('priority_param_codes', ["62610", "62611", "62620", "00065"])
    
    # Find nearby gauges
    nearby_gauges = usgs_data_handler.find_nearby_usgs_gauges(
        gnss_station_lat=station_lat,
        gnss_station_lon=station_lon,
        radius_km=search_radius_km,
        desired_parameter_codes=priority_param_codes
    )
    
    if nearby_gauges is None or nearby_gauges.empty:
        logging.warning(f"No USGS gauges found within {search_radius_km} km")
        return None, None, None
    
    # Find the best gauge and parameter based on priority codes
    for param_code in priority_param_codes:
        # Find gauges that have this parameter
        gauges_with_param = [
            g for _, g in nearby_gauges.iterrows() 
            if param_code in g.get('matched_parameters', [])
        ]
        
        if gauges_with_param:
            # Pick the closest one
            best_gauge = min(gauges_with_param, key=lambda g: g.get('distance_km', float('inf')))
            
            site_code = best_gauge.get('site_code')
            
            logging.info(f"Selected USGS gauge {site_code} ({best_gauge.get('site_name')}) "
                       f"with parameter {param_code} at distance {best_gauge.get('distance_km'):.2f} km")
            
            gauge_info = {
                'site_code': site_code,
                'parameter_code': param_code,
                'site_name': best_gauge.get('site_name'),
                'distance_km': best_gauge.get('distance_km'),
                'vertical_datum': best_gauge.get('vertical_datum'),
                'site_type': best_gauge.get('site_type'),
                'configured': False
            }
            
            return site_code, param_code, gauge_info
    
    logging.warning(f"No USGS gauges found with any of the priority parameters")
    return None, None, None

def run_usgs_comparison(station_name, year, doy_range=None, time_lag_analysis=True, max_lag_days=10):
    """
    Run the USGS comparison analysis for a specific station and time period.
    
    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
        year (int or str): Year to analyze
        doy_range (tuple, optional): Range of DOYs to include (start, end). 
                                   Defaults to None (all available).
        time_lag_analysis (bool, optional): Whether to perform time lag analysis.
                                        Defaults to True.
        max_lag_days (int, optional): Maximum lag to consider in days.
                                    Defaults to 10.
    
    Returns:
        dict: Analysis results and paths to output files
    """
    # Load station configuration
    stations_config = load_config(STATIONS_CONFIG_PATH)
    if stations_config is None:
        logging.error(f"Failed to load stations configuration")
        return {'success': False, 'error': 'Failed to load stations configuration'}
    
    station_config = stations_config.get(station_name)
    if station_config is None:
        logging.error(f"Station {station_name} not found in configuration")
        return {'success': False, 'error': f'Station {station_name} not found in configuration'}
    
    # Load GNSS-IR data
    gnssir_df = load_gnssir_data(station_name, year, doy_range)
    if gnssir_df is None or gnssir_df.empty:
        logging.error(f"Failed to load GNSS-IR data for {station_name} {year}")
        return {'success': False, 'error': 'Failed to load GNSS-IR data'}
    
    # Aggregate GNSS-IR data to daily statistics
    gnssir_daily_df = aggregate_gnssir_daily(gnssir_df)
    if gnssir_daily_df.empty:
        logging.error(f"Failed to aggregate GNSS-IR data for {station_name} {year}")
        return {'success': False, 'error': 'Failed to aggregate GNSS-IR data'}
    
    # Find the best USGS gauge for this station
    usgs_site_code, parameter_code, gauge_info = find_usgs_gauge_for_station(station_config)
    if usgs_site_code is None or parameter_code is None:
        logging.error(f"Failed to find suitable USGS gauge for {station_name}")
        return {
            'success': False, 
            'error': 'Failed to find suitable USGS gauge',
            'gnssir_daily_df': gnssir_daily_df  # Still return the GNSS-IR data
        }
    
    # Determine date range for USGS data request
    # Use the min/max dates from the GNSS-IR data with a buffer
    min_date = gnssir_daily_df['datetime'].min().date() - timedelta(days=1)
    max_date = gnssir_daily_df['datetime'].max().date() + timedelta(days=1)
    
    start_date_str = min_date.strftime('%Y-%m-%d')
    end_date_str = max_date.strftime('%Y-%m-%d')
    
    # Get the priority parameter codes from config
    usgs_config = station_config.get('usgs_comparison', {})
    priority_param_codes = usgs_config.get('priority_param_codes', ["62610", "62611", "62620", "00065"])
    
    # Fetch USGS gauge data with enhanced function that tries multiple parameter codes and services
    usgs_df, usgs_metadata, used_param_code = usgs_data_handler.fetch_usgs_gauge_data(
        usgs_site_code=usgs_site_code,
        parameter_code=parameter_code,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        service="iv",  # Use instantaneous values for better resolution
        priority_parameter_codes=priority_param_codes
    )
    
    if usgs_df.empty:
        logging.warning(f"No USGS data found for site {usgs_site_code} with any parameter code")
        return {
            'success': False,
            'error': 'No USGS data found',
            'gnssir_daily_df': gnssir_daily_df,
            'gauge_info': gauge_info
        }
    
    # Log which parameter code was successfully used
    if used_param_code and used_param_code != parameter_code:
        logging.info(f"Using alternative parameter code {used_param_code} instead of {parameter_code}")
        # Update gauge_info with the used parameter code
        gauge_info['parameter_code'] = used_param_code
    
    # Process USGS data (convert units, aggregate to daily)
    usgs_daily_df = usgs_data_handler.process_usgs_data(
        usgs_df=usgs_df,
        usgs_metadata=usgs_metadata,
        convert_to_meters=True
    )
    
    if usgs_daily_df.empty:
        logging.error(f"Failed to process USGS data")
        return {
            'success': False,
            'error': 'Failed to process USGS data',
            'gnssir_daily_df': gnssir_daily_df,
            'gauge_info': gauge_info
        }
    
    # Generate comparison plots
    output_dir = PROJECT_ROOT / "results_annual" / station_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot with raw values (dual y-axis)
    raw_plot_path = output_dir / f"{station_name}_{year}_usgs_comparison_raw.png"
    try:
        raw_plot_path = visualizer.plot_comparison_timeseries(
            daily_gnssir_rh_df=gnssir_daily_df,
            daily_usgs_gauge_df=usgs_daily_df,
            station_name=station_name,
            usgs_gauge_info=gauge_info,
            output_plot_path=raw_plot_path,
            gnssir_rh_col='rh_median_m',
            usgs_wl_col='usgs_value_m_median',
            compare_demeaned=False
        )
        logging.info(f"Generated comparison plot with raw values: {raw_plot_path}")
    except Exception as e:
        logging.error(f"Error generating raw comparison plot: {e}")
        raw_plot_path = None
    
    # Plot with demeaned values (same y-axis)
    demeaned_plot_path = output_dir / f"{station_name}_{year}_usgs_comparison_demeaned.png"
    try:
        demeaned_plot_path = visualizer.plot_comparison_timeseries(
            daily_gnssir_rh_df=gnssir_daily_df,
            daily_usgs_gauge_df=usgs_daily_df,
            station_name=station_name,
            usgs_gauge_info=gauge_info,
            output_plot_path=demeaned_plot_path,
            gnssir_rh_col='rh_median_m',
            usgs_wl_col='usgs_value_m_median',
            compare_demeaned=True
        )
        logging.info(f"Generated comparison plot with demeaned values: {demeaned_plot_path}")
    except Exception as e:
        logging.error(f"Error generating demeaned comparison plot: {e}")
        demeaned_plot_path = None
    
    # Calculate correlation statistics
    try:
        # Merge dataframes on date for correlation analysis
        merged_df = pd.merge(
            gnssir_daily_df[['date', 'rh_median_m']],
            usgs_daily_df[['date', 'usgs_value_m_median']],
            on='date',
            how='inner'
        )
        
        correlation = merged_df['rh_median_m'].corr(merged_df['usgs_value_m_median'])
        logging.info(f"Correlation between GNSS-IR RH and USGS water level: {correlation:.4f}")
        
        # Check if correlation is negative
        if correlation < 0:
            logging.info("Negative correlation: GNSS-IR RH increases as water level decreases")
        else:
            logging.info("Positive correlation: GNSS-IR RH increases as water level increases")
    except Exception as e:
        logging.error(f"Error calculating correlation statistics: {e}")
        correlation = None
    
    # Save results to CSV
    comparison_csv_path = output_dir / f"{station_name}_{year}_usgs_comparison.csv"
    try:
        # Merge dataframes on date
        merged_df = pd.merge(
            gnssir_daily_df, 
            usgs_daily_df, 
            on='date', 
            how='outer',
            suffixes=('_gnssir', '_usgs')
        )
        
        # Add gauge info to the dataframe
        merged_df['usgs_site_code'] = usgs_site_code
        merged_df['usgs_parameter_code'] = parameter_code
        merged_df['usgs_site_name'] = gauge_info.get('site_name', '')
        merged_df['usgs_distance_km'] = gauge_info.get('distance_km', 0)
        merged_df['usgs_vertical_datum'] = gauge_info.get('vertical_datum', 'Unknown')
        
        # Save to CSV
        merged_df.to_csv(comparison_csv_path, index=False)
        logging.info(f"Saved comparison data to {comparison_csv_path}")
    except Exception as e:
        logging.error(f"Error saving comparison data to CSV: {e}")
        comparison_csv_path = None
    
    # Return results
    return {
        'success': True,
        'gnssir_daily_df': gnssir_daily_df,
        'usgs_daily_df': usgs_daily_df,
        'gauge_info': gauge_info,
        'correlation': correlation,
        'raw_plot_path': raw_plot_path,
        'demeaned_plot_path': demeaned_plot_path,
        'comparison_csv_path': comparison_csv_path,
        'usgs_metadata': usgs_metadata
    }

def main():
    """Main function to run USGS comparison analysis"""
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(description='USGS Comparison for GNSS-IR')
    parser.add_argument('--station', type=str, required=True, help='Station ID (4-char uppercase)')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--doy_start', type=int, help='Starting day of year (optional)')
    parser.add_argument('--doy_end', type=int, help='Ending day of year (optional)')
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Logging level')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.station}_{args.year}_usgs_comparison.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set DOY range if provided
    doy_range = None
    if args.doy_start is not None and args.doy_end is not None:
        doy_range = (args.doy_start, args.doy_end)
        logging.info(f"Processing DOY range: {doy_range[0]}-{doy_range[1]}")
    
    # Run analysis
    results = run_usgs_comparison(args.station, args.year, doy_range)
    
    # Check if analysis was successful
    if results.get('success', False):
        logging.info("USGS comparison analysis completed successfully")
        
        # Log correlation if available
        if 'correlation' in results and results['correlation'] is not None:
            logging.info(f"Correlation between GNSS-IR RH and USGS water level: {results['correlation']:.4f}")
        
        # Log output file paths
        if 'raw_plot_path' in results and results['raw_plot_path'] is not None:
            logging.info(f"Raw comparison plot: {results['raw_plot_path']}")
        
        if 'demeaned_plot_path' in results and results['demeaned_plot_path'] is not None:
            logging.info(f"Demeaned comparison plot: {results['demeaned_plot_path']}")
        
        if 'comparison_csv_path' in results and results['comparison_csv_path'] is not None:
            logging.info(f"Comparison data CSV: {results['comparison_csv_path']}")
    else:
        logging.error(f"USGS comparison analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
