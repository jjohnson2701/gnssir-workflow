"""
Module for integrating meteorological data with GNSS-IR analysis.

This module provides functionality to:
1. Fetch wind data from NOAA NDBC buoys or NWS stations
2. Process and align the data with GNSS-IR and USGS time series
3. Provide utilities for wind analysis and visualization
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import datetime
import io
import re

# Constants for API endpoints and data sources
NDBC_BASE_URL = "https://www.ndbc.noaa.gov/data/realtime2/"
NDBC_HISTORICAL_URL = "https://www.ndbc.noaa.gov/view_text_file.php?filename="
NWS_METAR_URL = "https://aviationweather.gov/adds/dataserver_current/httpparam"

# Wind speed conversion factors
KNOTS_TO_MPS = 0.51444444  # Knots to meters per second
MPH_TO_MPS = 0.44704      # Miles per hour to meters per second

def find_nearest_ndbc_stations(
    latitude: float, 
    longitude: float, 
    radius_km: float = 50,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the nearest NDBC stations to a given location.
    
    Args:
        latitude: Location latitude in decimal degrees
        longitude: Location longitude in decimal degrees
        radius_km: Search radius in kilometers
        limit: Maximum number of stations to return
        
    Returns:
        List of dictionaries containing station information
    """
    # In a real implementation, this would query the NDBC API
    # For this example, we'll use a simulated response
    logging.info(f"Finding NDBC stations near ({latitude}, {longitude}) within {radius_km} km")
    
    try:
        # Use the NDBC API to get station data
        # This would be replaced with an actual API call
        stations = [
            {
                'id': '41013',
                'name': 'Frying Pan Shoals',
                'lat': 33.436,
                'lon': -77.743,
                'distance_km': 35.2,
                'type': 'buoy',
                'data_types': ['wind', 'wave', 'water_temp']
            },
            {
                'id': '41025',
                'name': 'Diamond Shoals',
                'lat': 35.006,
                'lon': -75.402,
                'distance_km': 42.8,
                'type': 'buoy',
                'data_types': ['wind', 'wave', 'water_temp']
            }
        ]
        
        # Sort by distance and limit results
        return stations[:limit]
    
    except Exception as e:
        logging.error(f"Error finding NDBC stations: {e}")
        return []

def find_nearest_metar_stations(
    latitude: float, 
    longitude: float, 
    radius_km: float = 50,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the nearest METAR stations (airports) to a given location.
    
    Args:
        latitude: Location latitude in decimal degrees
        longitude: Location longitude in decimal degrees
        radius_km: Search radius in kilometers
        limit: Maximum number of stations to return
        
    Returns:
        List of dictionaries containing station information
    """
    logging.info(f"Finding METAR stations near ({latitude}, {longitude}) within {radius_km} km")
    
    try:
        # In a real implementation, this would query the NWS API
        # For this example, we'll use a simulated response
        stations = [
            {
                'id': 'KMQI',
                'name': 'Manteo / Dare County Regional',
                'lat': 35.920,
                'lon': -75.701,
                'distance_km': 15.3,
                'type': 'airport',
                'data_types': ['wind', 'weather', 'temperature']
            },
            {
                'id': 'KECG',
                'name': 'Elizabeth City',
                'lat': 36.258,
                'lon': -76.172,
                'distance_km': 28.6,
                'type': 'airport',
                'data_types': ['wind', 'weather', 'temperature']
            }
        ]
        
        # Sort by distance and limit results
        return stations[:limit]
    
    except Exception as e:
        logging.error(f"Error finding METAR stations: {e}")
        return []

def get_ndbc_wind_data(
    station_id: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    return_as_df: bool = True
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Get wind data from an NDBC buoy or station.
    
    Args:
        station_id: NDBC station ID (e.g., '41013')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        return_as_df: Whether to return data as a DataFrame (True) or dict (False)
        
    Returns:
        DataFrame or dictionary with wind data
    """
    logging.info(f"Fetching NDBC wind data for station {station_id} from {start_date} to {end_date}")
    
    try:
        # In a real implementation, this would fetch data from the NDBC API
        # For this example, we'll generate synthetic data
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate synthetic wind data
        np.random.seed(42)  # For reproducible results
        
        # Base wind pattern with diurnal and synoptic variations
        t = np.arange(len(date_range))
        
        # Wind speed (m/s) - combination of different cycles plus noise
        wind_speed = (
            5.0 +                                    # Base wind speed
            2.0 * np.sin(2 * np.pi * t / 24) +       # Diurnal cycle
            3.0 * np.sin(2 * np.pi * t / (24*7)) +   # Weekly cycle
            1.5 * np.random.randn(len(t))            # Random variations
        )
        wind_speed = np.maximum(0, wind_speed)  # No negative wind speeds
        
        # Wind direction (degrees) - slowly varying direction plus noise
        wind_dir = (
            180 +                                    # Mean direction
            90 * np.sin(2 * np.pi * t / (24*3)) +    # 3-day cycle
            20 * np.random.randn(len(t))             # Random variations
        ) % 360  # Convert to 0-360 range
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': date_range,
            'wind_speed_mps': wind_speed,
            'wind_dir_deg': wind_dir,
            'station_id': station_id
        })
        
        # Add date column
        df['date'] = df['datetime'].dt.date
        
        # Return as requested format
        if return_as_df:
            return df
        else:
            # Convert to dictionary format
            return {
                'station_id': station_id,
                'data': df.to_dict(orient='records'),
                'start_date': start_date,
                'end_date': end_date
            }
    
    except Exception as e:
        logging.error(f"Error fetching NDBC wind data: {e}")
        if return_as_df:
            return pd.DataFrame()
        else:
            return {'error': str(e), 'station_id': station_id}

def get_metar_wind_data(
    station_id: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    return_as_df: bool = True
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Get wind data from a METAR station (airport).
    
    Args:
        station_id: METAR station ID (e.g., 'KMQI')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        return_as_df: Whether to return data as a DataFrame (True) or dict (False)
        
    Returns:
        DataFrame or dictionary with wind data
    """
    logging.info(f"Fetching METAR wind data for station {station_id} from {start_date} to {end_date}")
    
    try:
        # In a real implementation, this would fetch data from the NWS API
        # For this example, we'll generate synthetic data similar to NDBC but with gaps
        
        # Create hourly date range with some gaps
        all_hours = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Add gaps (remove ~10% of hours randomly)
        np.random.seed(42)
        mask = np.random.random(len(all_hours)) > 0.1
        date_range = all_hours[mask]
        
        # Generate synthetic wind data
        t = np.arange(len(date_range))
        
        # Wind speed (m/s) - combination of different cycles plus noise
        wind_speed = (
            4.0 +                                    # Base wind speed
            1.5 * np.sin(2 * np.pi * t / 24) +       # Diurnal cycle
            2.5 * np.sin(2 * np.pi * t / (24*5)) +   # 5-day cycle
            1.2 * np.random.randn(len(t))            # Random variations
        )
        wind_speed = np.maximum(0, wind_speed)  # No negative wind speeds
        
        # Wind direction (degrees) - slowly varying direction plus noise
        wind_dir = (
            220 +                                    # Mean direction
            70 * np.sin(2 * np.pi * t / (24*4)) +    # 4-day cycle
            25 * np.random.randn(len(t))             # Random variations
        ) % 360  # Convert to 0-360 range
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': date_range,
            'wind_speed_mps': wind_speed,
            'wind_dir_deg': wind_dir,
            'station_id': station_id
        })
        
        # Add date column
        df['date'] = df['datetime'].dt.date
        
        # Return as requested format
        if return_as_df:
            return df
        else:
            # Convert to dictionary format
            return {
                'station_id': station_id,
                'data': df.to_dict(orient='records'),
                'start_date': start_date,
                'end_date': end_date
            }
    
    except Exception as e:
        logging.error(f"Error fetching METAR wind data: {e}")
        if return_as_df:
            return pd.DataFrame()
        else:
            return {'error': str(e), 'station_id': station_id}

def process_wind_data(
    wind_df: pd.DataFrame,
    resample_freq: str = 'D',
    agg_method: str = 'mean'
) -> pd.DataFrame:
    """
    Process wind data for analysis with GNSS-IR data.
    
    Args:
        wind_df: DataFrame with wind data (must contain 'datetime', 'wind_speed_mps', 'wind_dir_deg')
        resample_freq: Frequency for resampling ('D' for daily, 'H' for hourly, etc.)
        agg_method: Aggregation method ('mean', 'max', 'min', 'median', etc.)
        
    Returns:
        Processed DataFrame with resampled wind data
    """
    logging.info(f"Processing wind data with {agg_method} aggregation at {resample_freq} frequency")
    
    try:
        # Check if DataFrame is empty
        if wind_df.empty:
            logging.warning("Empty wind DataFrame provided")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_cols = ['datetime', 'wind_speed_mps', 'wind_dir_deg']
        for col in required_cols:
            if col not in wind_df.columns:
                logging.error(f"Required column '{col}' not found in wind DataFrame")
                return pd.DataFrame()
        
        # Ensure datetime is in datetime format
        wind_df['datetime'] = pd.to_datetime(wind_df['datetime'])
        
        # Convert wind directions to u, v components for proper averaging
        wind_df['u'] = -wind_df['wind_speed_mps'] * np.sin(np.radians(wind_df['wind_dir_deg']))
        wind_df['v'] = -wind_df['wind_speed_mps'] * np.cos(np.radians(wind_df['wind_dir_deg']))
        
        # Set datetime as index for resampling
        wind_df = wind_df.set_index('datetime')
        
        # Determine aggregation functions
        if agg_method == 'mean':
            agg_funcs = {
                'wind_speed_mps': 'mean',
                'u': 'mean',
                'v': 'mean',
                'station_id': 'first'
            }
        elif agg_method == 'max':
            agg_funcs = {
                'wind_speed_mps': 'max',
                'u': lambda x: x.iloc[x.index.get_indexer([x['wind_speed_mps'].idxmax()])[0]],
                'v': lambda x: x.iloc[x.index.get_indexer([x['wind_speed_mps'].idxmax()])[0]],
                'wind_dir_deg': lambda x: x.iloc[x.index.get_indexer([x['wind_speed_mps'].idxmax()])[0]],
                'station_id': 'first'
            }
        else:
            # Default to mean
            agg_funcs = {
                'wind_speed_mps': agg_method,
                'u': agg_method,
                'v': agg_method,
                'station_id': 'first'
            }
        
        # Resample
        resampled = wind_df.resample(resample_freq).agg(agg_funcs)
        
        # Convert u, v back to direction (radians to degrees, then correct quadrant)
        resampled['wind_dir_deg'] = (270 - np.degrees(np.arctan2(resampled['v'], resampled['u']))) % 360
        
        # Reset index to get datetime as column
        resampled = resampled.reset_index()
        
        # Add date column if daily or longer frequency
        if resample_freq in ['D', 'W', 'M', 'Q', 'Y', 'A']:
            resampled['date'] = resampled['datetime'].dt.date
        
        return resampled
    
    except Exception as e:
        logging.error(f"Error processing wind data: {e}")
        return pd.DataFrame()

def calculate_wind_forcing(
    wind_df: pd.DataFrame,
    fetch_direction: Union[float, List[float]],
    fetch_width: float = 45.0
) -> pd.DataFrame:
    """
    Calculate wind forcing index based on wind speed, direction, and fetch.
    
    Wind forcing is strongest when wind direction aligns with fetch direction
    and wind speed is high.
    
    Args:
        wind_df: DataFrame with wind data (must contain 'wind_speed_mps', 'wind_dir_deg')
        fetch_direction: Direction(s) of fetch in degrees (0-360, where 0 is from North)
            If a list is provided, the maximum forcing from any direction is used
        fetch_width: Angular width of fetch in degrees
        
    Returns:
        DataFrame with added wind forcing column
    """
    logging.info(f"Calculating wind forcing for fetch direction(s): {fetch_direction}")
    
    try:
        # Check if DataFrame is empty
        if wind_df.empty:
            logging.warning("Empty wind DataFrame provided")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_cols = ['wind_speed_mps', 'wind_dir_deg']
        for col in required_cols:
            if col not in wind_df.columns:
                logging.error(f"Required column '{col}' not found in wind DataFrame")
                return pd.DataFrame()
        
        # Create output DataFrame
        result_df = wind_df.copy()
        
        # Handle single direction or list of directions
        if isinstance(fetch_direction, (int, float)):
            fetch_directions = [fetch_direction]
        else:
            fetch_directions = fetch_direction
        
        # Calculate forcing for each fetch direction
        forcing_values = []
        
        for fetch_dir in fetch_directions:
            # Calculate angular difference (0-180)
            diff = np.minimum(
                np.abs(result_df['wind_dir_deg'] - fetch_dir),
                360 - np.abs(result_df['wind_dir_deg'] - fetch_dir)
            )
            
            # Calculate directional factor (1.0 for perfect alignment, decreasing to 0.0 at fetch_width/2)
            dir_factor = np.maximum(0, 1.0 - (diff / (fetch_width / 2)))
            
            # Calculate forcing (wind speed * directional factor)
            # Squared relation to wind speed (drag proportional to velocity squared)
            current_forcing = result_df['wind_speed_mps']**2 * dir_factor
            
            forcing_values.append(current_forcing)
        
        # Use maximum forcing from any direction
        result_df['wind_forcing'] = np.max(np.column_stack(forcing_values), axis=1)
        
        return result_df
    
    except Exception as e:
        logging.error(f"Error calculating wind forcing: {e}")
        return wind_df.copy()  # Return original DataFrame on error
