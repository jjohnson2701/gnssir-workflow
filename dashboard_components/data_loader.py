# ABOUTME: Data loading functions for Streamlit GNSS-IR dashboard
# ABOUTME: Handles CSV loading, API fetching, and data caching

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import cache manager
from dashboard_components.cache_manager import (
    disk_cache, 
    load_and_aggregate_subhourly_data,
    monitor_performance
)

# Import required modules
from dashboard_components.station_metadata import get_station_config

try:
    from scripts.external_apis.noaa_coops import NOAACOOPSClient
    from scripts.external_apis.ndbc_client import NDBCClient
except ImportError as e:
    print(f"Warning: External API modules not available: {e}")
    NOAACOOPSClient = None
    NDBCClient = None


def get_preferred_coops_stations(station_id):
    """Get preferred CO-OPS stations from station config."""
    config = get_station_config(station_id)
    if config:
        coops_config = config.get('external_data_sources', {}).get('noaa_coops', {})
        return coops_config.get('preferred_stations', [])
    return []


def get_nearest_coops_station(lat, lon):
    """Find nearest CO-OPS station (placeholder - returns None if no preferred stations)."""
    return None


def get_preferred_ndbc_buoys(station_id):
    """Get preferred NDBC buoys from station config."""
    config = get_station_config(station_id)
    if config:
        ndbc_config = config.get('external_data_sources', {}).get('ndbc_buoys', {})
        return ndbc_config.get('preferred_buoys', [])
    return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
@monitor_performance
def load_station_data(station_id="FORA", year=2024):
    """Load comprehensive GNSS-IR and comparison data for a station."""
    results_dir = project_root / "results_annual" / station_id
    
    # Load combined RH data
    rh_file = results_dir / f"{station_id}_{year}_combined_rh.csv"
    rh_data = None
    if rh_file.exists():
        rh_data = pd.read_csv(rh_file)
        rh_data['date'] = pd.to_datetime(rh_data['date'])
    
    # Load comparison data (includes lag analysis)
    comparison_file = results_dir / f"{station_id}_{year}_comparison.csv"
    comparison_data = None
    if comparison_file.exists():
        comparison_data = pd.read_csv(comparison_file)
        comparison_data['merge_date'] = pd.to_datetime(comparison_data['merge_date'])
    
    # Load USGS gauge data
    usgs_file = results_dir / f"{station_id}_{year}_usgs_gauge_data.csv"
    usgs_data = None
    if usgs_file.exists():
        usgs_data = pd.read_csv(usgs_file)
        # Check which date column exists and standardize
        if 'datetime' in usgs_data.columns:
            usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime'])
        elif 'date' in usgs_data.columns:
            usgs_data['date'] = pd.to_datetime(usgs_data['date'])
            usgs_data['datetime'] = usgs_data['date']  # Create datetime column for consistency
    
    # If we have comparison data with USGS values, use that instead
    if comparison_data is not None and not comparison_data.empty and 'usgs_value' in comparison_data.columns:
        # Use comparison data as primary USGS source since it has better alignment
        usgs_aligned = comparison_data[['merge_date', 'usgs_value']].copy()
        usgs_aligned['datetime'] = usgs_aligned['merge_date']
        usgs_aligned = usgs_aligned.dropna(subset=['usgs_value'])
        
        if not usgs_aligned.empty:
            usgs_data = usgs_aligned  # Use the aligned data
            # Ensure we have a 'date' column for compatibility
            if 'date' not in usgs_data.columns and 'datetime' in usgs_data.columns:
                usgs_data['date'] = usgs_data['datetime']
            elif 'date' not in usgs_data.columns and 'merge_date' in usgs_data.columns:
                usgs_data['date'] = usgs_data['merge_date']
    
    # Load CO-OPS data if available (check multiple filename patterns)
    coops_data = None
    coops_file_patterns = [
        results_dir / f"{station_id}_{year}_coops_daily.csv",
        results_dir / f"{station_id}_{year}_coops_6min.csv",
        results_dir / f"{station_id}_{year}_coops_hourly.csv",
    ]

    coops_file = None
    for pattern in coops_file_patterns:
        if pattern.exists():
            coops_file = pattern
            break

    if coops_file:
        coops_data = pd.read_csv(coops_file)
        # Handle datetime column
        if 'date' in coops_data.columns:
            coops_data['date'] = pd.to_datetime(coops_data['date'])
            coops_data['datetime'] = coops_data['date']
        elif 'datetime' in coops_data.columns:
            coops_data['datetime'] = pd.to_datetime(coops_data['datetime'])
            coops_data['date'] = coops_data['datetime'].dt.date

        # Rename water level columns for consistency
        for col in ['water_level_mean', 'water_level', 'water_level_m', 'v']:
            if col in coops_data.columns and 'water_level_m' not in coops_data.columns:
                coops_data['water_level_m'] = coops_data[col]
                break

        # Get station ID from config if available
        from dashboard_components.station_metadata import get_reference_source_info
        ref_info = get_reference_source_info(station_id)
        coops_station_id = ref_info.get('station_id', 'Unknown') if ref_info['primary_source'] == 'CO-OPS' else 'Unknown'

        coops_data['source'] = 'NOAA CO-OPS'
        coops_data['station_id'] = coops_station_id
    
    return rh_data, comparison_data, usgs_data, coops_data


@st.cache_data
def load_available_stations():
    """Load list of available stations from configuration."""
    config_file = project_root / "config" / "stations_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            return list(config.keys())
    return ["FORA"]


def get_station_coordinates(station_id):
    """Get station coordinates from configuration."""
    config = get_station_config(station_id)
    if config:
        lat = config.get('latitude', config.get('latitude_deg'))
        lon = config.get('longitude', config.get('longitude_deg'))
        return lat, lon
    return None, None


@disk_cache('external_api')
@monitor_performance
def fetch_coops_data(station_id, year, doy_range=None, rh_data=None):
    """Fetch NOAA CO-OPS data for the specified station and time range.
    
    This function is cached to disk to avoid repeated API calls.
    Cache expires after 7 days.
    """
    try:
        client = NOAACOOPSClient()
        
        # Get station coordinates
        lat, lon = get_station_coordinates(station_id)
        if lat is None or lon is None:
            return None, None
        
        # Check for preferred stations first
        preferred_stations = get_preferred_coops_stations(station_id)
        
        if preferred_stations:
            # Use the first preferred station
            coops_station_id = preferred_stations[0]
            st.info(f"Using preferred CO-OPS station: {coops_station_id}")
        else:
            # Find nearest station
            coops_station_id = get_nearest_coops_station(lat, lon)
            if not coops_station_id:
                st.warning("No nearby CO-OPS stations found")
                return None, None
        
        # Determine date range
        if rh_data is not None and not rh_data.empty:
            # Use actual GNSS-IR data range
            start_date = rh_data['date'].min()
            end_date = rh_data['date'].max()
        elif doy_range:
            # Use DOY range
            start_doy, end_doy = doy_range
            start_date = datetime(year, 1, 1) + timedelta(days=start_doy - 1)
            end_date = datetime(year, 1, 1) + timedelta(days=end_doy - 1)
        else:
            # Default to full year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
        
        # Format dates for API
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Fetch water level observations
        water_levels = client.get_water_levels(
            station_id=coops_station_id,
            start_date=start_str,
            end_date=end_str,
            datum='NAVD',  # North American Vertical Datum
            units='metric',
            time_zone='gmt'
        )
        
        if water_levels is not None and not water_levels.empty:
            # Convert time column to datetime and rename for consistency
            water_levels['datetime'] = pd.to_datetime(water_levels['time'])
            water_levels['date'] = water_levels['datetime'].dt.date
            water_levels['water_level_m'] = water_levels['value']
            
            # Also fetch tide predictions for the same period
            predictions = client.get_tide_predictions(
                station_id=coops_station_id,
                start_date=start_str,
                end_date=end_str,
                datum='NAVD',
                units='metric',
                time_zone='gmt'
            )
            
            if predictions is not None and not predictions.empty:
                predictions['datetime'] = pd.to_datetime(predictions['time'])
                predictions['tide_prediction_m'] = predictions['value']
                
                # Merge observations with predictions
                water_levels = pd.merge(
                    water_levels,
                    predictions[['datetime', 'tide_prediction_m']],
                    on='datetime',
                    how='left'
                )
                
                # Calculate residuals (observed - predicted)
                water_levels['residual_m'] = water_levels['water_level_m'] - water_levels['tide_prediction_m']
            
            return water_levels, coops_station_id
        
    except Exception as e:
        st.error(f"Error fetching CO-OPS data: {str(e)}")
    
    return None, None


@disk_cache('external_api')
@monitor_performance
def fetch_ndbc_data(station_id, year, doy_range=None, rh_data=None):
    """Fetch NDBC buoy data for the specified station and time range.
    
    This function is cached to disk to avoid repeated API calls.
    Cache expires after 7 days.
    """
    try:
        client = NDBCClient()
        
        # Get station coordinates
        lat, lon = get_station_coordinates(station_id)
        if lat is None or lon is None:
            return None, None
        
        # Check for preferred buoys first
        preferred_buoys = get_preferred_ndbc_buoys(station_id)
        
        if preferred_buoys:
            # Use the first preferred buoy
            buoy_id = preferred_buoys[0]
            st.info(f"Using preferred NDBC buoy: {buoy_id}")
        else:
            # Find nearest buoy
            nearby_buoys = client.find_nearest_stations(lat, lon, n_stations=5)
            if nearby_buoys.empty:
                st.warning("No nearby NDBC buoys found")
                return None, None
            
            # Use the nearest buoy
            buoy_id = nearby_buoys.iloc[0]['station_id']
            distance = nearby_buoys.iloc[0]['distance_km']
            st.info(f"Using nearest NDBC buoy: {buoy_id} ({distance:.1f} km away)")
        
        # Determine date range
        if rh_data is not None and not rh_data.empty:
            # Use actual GNSS-IR data range
            start_date = rh_data['date'].min()
            end_date = rh_data['date'].max()
            days_back = (datetime.now() - start_date).days + 1
        else:
            # Default to last 365 days
            days_back = 365
        
        # Fetch meteorological data
        met_data = client.get_meteorological_data(buoy_id, days_back=days_back)
        
        if met_data is not None and not met_data.empty:
            # Add date column for consistency
            met_data['date'] = met_data['datetime'].dt.date
            
            # Calculate derived quantities
            if 'WSPD' in met_data.columns:
                met_data['wind_speed_m_s'] = met_data['WSPD']
            
            if 'WVHT' in met_data.columns:
                met_data['wave_height_m'] = met_data['WVHT']
            
            if 'WSPD' in met_data.columns and 'WDIR' in met_data.columns:
                # Calculate wind forcing components
                wind_dir_rad = np.deg2rad(met_data['WDIR'])
                met_data['wind_u'] = -met_data['WSPD'] * np.sin(wind_dir_rad)
                met_data['wind_v'] = -met_data['WSPD'] * np.cos(wind_dir_rad)
            
            # Filter to requested date range if specified
            if rh_data is not None and not rh_data.empty:
                met_data = met_data[
                    (met_data['datetime'] >= rh_data['date'].min()) &
                    (met_data['datetime'] <= rh_data['date'].max() + pd.Timedelta(days=1))
                ]
            
            return met_data, buoy_id
        
    except Exception as e:
        st.error(f"Error fetching NDBC data: {str(e)}")
    
    return None, None


@st.cache_data(ttl=3600)
def load_subhourly_data_progressive(station_id, year, max_points=10000):
    """Load sub-hourly GNSS-IR data with intelligent decimation.
    
    For visualization, we don't always need all 36,000+ points.
    This function provides smart data reduction while preserving patterns.
    """
    data_dir = project_root / "data" / station_id / str(year) / "rh_daily"
    
    if not data_dir.exists():
        return None
    
    all_data = []
    
    # Read all daily files
    for file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                all_data.append(df)
        except Exception:
            continue
    
    if not all_data:
        return None
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # If data is larger than max_points, use intelligent decimation
    if len(combined) > max_points:
        # Option 1: Time-based decimation (keep every Nth point)
        decimation_factor = len(combined) // max_points
        combined = combined.iloc[::decimation_factor]
        
        # Option 2: Could also use LTTB algorithm for better visual preservation
        # combined = lttb_downsample(combined, max_points)
    
    return combined


@st.cache_data(ttl=3600)
def create_performance_summary(rh_data, usgs_data=None, coops_data=None):
    """Create quick performance metrics without loading full datasets."""
    summary = {
        'gnss_ir': {
            'total_days': rh_data['date'].nunique() if rh_data is not None else 0,
            'total_measurements': len(rh_data) if rh_data is not None else 0,
            'avg_daily_count': rh_data.groupby('date').size().mean() if rh_data is not None else 0,
            'data_completeness': (rh_data['date'].nunique() / 366) * 100 if rh_data is not None else 0
        }
    }
    
    if usgs_data is not None:
        summary['usgs'] = {
            'total_measurements': len(usgs_data),
            'data_availability': True
        }
    
    if coops_data is not None:
        summary['coops'] = {
            'total_measurements': len(coops_data),
            'has_predictions': 'tide_prediction_m' in coops_data.columns if coops_data is not None else False
        }
    
    return summary


def load_data_with_progress(station_id, year, include_external=True):
    """Load all data with progress indicators."""
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Load local data (fast)
        progress_text.text("Loading GNSS-IR data...")
        progress_bar.progress(0.2)
        rh_data, comparison_data, usgs_data = load_station_data(station_id, year)
        
        # Step 2: Load external data if requested (potentially slow)
        coops_data = None
        ndbc_data = None
        
        if include_external:
            # Check if we have cached data first
            progress_text.text("Checking for cached external data...")
            progress_bar.progress(0.4)
            
            # Load CO-OPS data
            progress_text.text("Loading NOAA CO-OPS data...")
            progress_bar.progress(0.6)
            coops_data, coops_station = fetch_coops_data(station_id, year, rh_data=rh_data)
            
            # Load NDBC data
            progress_text.text("Loading NDBC buoy data...")
            progress_bar.progress(0.8)
            ndbc_data, buoy_id = fetch_ndbc_data(station_id, year, rh_data=rh_data)
        
        progress_text.text("Data loading complete!")
        progress_bar.progress(1.0)
        
        # Clear progress indicators after a short delay
        time.sleep(0.5)
        progress_text.empty()
        progress_bar.empty()
        
        return {
            'rh_data': rh_data,
            'comparison_data': comparison_data,
            'usgs_data': usgs_data,
            'coops_data': coops_data,
            'ndbc_data': ndbc_data
        }
        
    except Exception as e:
        progress_text.text(f"Error loading data: {str(e)}")
        progress_bar.empty()
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes 
@monitor_performance
def discover_quicklook_plots(station_id="FORA", year=2024):
    """Discover available QuickLook diagnostic plots for a station/year."""
    plots_dir = project_root / "data" / station_id / str(year) / "quicklook_plots_daily"
    
    if not plots_dir.exists():
        return {}
    
    # Scan for available plot files
    plot_files = {}
    for plot_file in plots_dir.glob("*.png"):
        # Parse filename: valr_2024_247_lsp.png or valr_2024_247_summary.png
        parts = plot_file.stem.split('_')
        if len(parts) >= 4:
            try:
                station = parts[0].upper()
                file_year = int(parts[1])
                doy = int(parts[2])
                plot_type = parts[3]  # 'lsp' or 'summary'
                
                if station == station_id.upper() and file_year == year:
                    if doy not in plot_files:
                        plot_files[doy] = {}
                    plot_files[doy][plot_type] = plot_file
            except (ValueError, IndexError):
                continue
    
    return plot_files


@st.cache_data(ttl=300)
@monitor_performance  
def get_quicklook_plots_for_day(station_id="FORA", year=2024, doy=1):
    """Get QuickLook plot file paths for a specific day."""
    plot_files = discover_quicklook_plots(station_id, year)
    
    if doy not in plot_files:
        return None
    
    day_plots = plot_files[doy]
    result = {}
    
    # Check for both required plot types
    if 'lsp' in day_plots and day_plots['lsp'].exists():
        result['lsp'] = day_plots['lsp']
    
    if 'summary' in day_plots and day_plots['summary'].exists():
        result['summary'] = day_plots['summary']
    
    return result if result else None


@st.cache_data(ttl=3600)
@monitor_performance
def get_available_diagnostic_days(station_id="FORA", year=2024):
    """Get list of days with available diagnostic plots."""
    plot_files = discover_quicklook_plots(station_id, year)
    
    # Only include days that have both LSP and summary plots
    complete_days = []
    for doy, plots in plot_files.items():
        if ('lsp' in plots and plots['lsp'].exists() and 
            'summary' in plots and plots['summary'].exists()):
            complete_days.append(doy)
    
    return sorted(complete_days)


def doy_to_date(year, doy):
    """Convert day of year to datetime date."""
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def date_to_doy(date):
    """Convert datetime date to day of year."""
    return date.timetuple().tm_yday


# Export all functions
__all__ = [
    'load_station_data',
    'load_available_stations',
    'get_station_coordinates',
    'fetch_coops_data',
    'fetch_ndbc_data',
    'load_subhourly_data_progressive',
    'create_performance_summary',
    'load_data_with_progress',
    'discover_quicklook_plots',
    'get_quicklook_plots_for_day',
    'get_available_diagnostic_days',
    'doy_to_date',
    'date_to_doy'
]