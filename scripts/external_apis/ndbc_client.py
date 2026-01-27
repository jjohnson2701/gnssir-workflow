#!/usr/bin/env python3
"""
NDBC Buoy Data Client for Wave Height and Wind Data
==================================================

Client for integrating NOAA National Data Buoy Center (NDBC) data 
into GNSS-IR analysis workflows.

Features:
- Wave height measurements from buoys and coastal stations
- Wind speed and direction data
- Atmospheric pressure and temperature
- Spectral wave analysis data
- Station discovery and metadata retrieval

Data Access: https://www.ndbc.noaa.gov/data/realtime2/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import time
import re
from io import StringIO

class NDBCClient:
    """
    Client for NDBC buoy data integration.
    
    Provides methods to retrieve wave height, wind data, and meteorological
    observations from NOAA buoys and coastal stations.
    """
    
    def __init__(self, base_url: str = "https://www.ndbc.noaa.gov/data/realtime2/"):
        """
        Initialize the NDBC client.
        
        Args:
            base_url: Base URL for NDBC real-time data
        """
        self.base_url = base_url
        self.metadata_url = "https://www.ndbc.noaa.gov/data/stations/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNSS-IR-Processing/1.0 (Research Application)'
        })
        
        # Rate limiting (be respectful)
        self.min_request_interval = 0.2  # 200ms between requests
        self.last_request_time = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Column mappings for different file types
        self.column_mappings = {
            'standard_met': {
                'YY': 'year',
                'MM': 'month', 
                'DD': 'day',
                'hh': 'hour',
                'mm': 'minute',
                'WDIR': 'wind_direction_deg',
                'WSPD': 'wind_speed_ms',
                'GST': 'wind_gust_ms',
                'WVHT': 'wave_height_m',
                'DPD': 'dominant_wave_period_s',
                'APD': 'average_wave_period_s',
                'MWD': 'mean_wave_direction_deg',
                'PRES': 'pressure_hpa',
                'ATMP': 'air_temp_c',
                'WTMP': 'water_temp_c',
                'DEWP': 'dewpoint_c',
                'VIS': 'visibility_km',
                'TIDE': 'tide_m'
            },
            'wave_spec': {
                'YY': 'year',
                'MM': 'month',
                'DD': 'day', 
                'hh': 'hour',
                'mm': 'minute',
                'WVHT': 'significant_wave_height_m',
                'SwH': 'swell_height_m',
                'SwP': 'swell_period_s',
                'SwD': 'swell_direction_deg',
                'WWH': 'wind_wave_height_m',
                'WWP': 'wind_wave_period_s',
                'WWD': 'wind_wave_direction_deg'
            }
        }
    
    def _rate_limit(self):
        """Implement respectful rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> requests.Response:
        """
        Make a rate-limited request with error handling.
        
        Args:
            url: Full URL to request
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails
        """
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"NDBC request failed: {e}")
            raise
    
    def find_nearby_buoys(
        self, 
        latitude: float, 
        longitude: float, 
        radius_km: float = 200
    ) -> List[Dict]:
        """
        Find NDBC buoys within a specified radius of coordinates.
        
        Note: This is a simplified implementation. For production use,
        consider using the NDBC station listings or a comprehensive database.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees  
            radius_km: Search radius in kilometers
            
        Returns:
            List of buoy dictionaries with metadata
        """
        self.logger.info(f"Searching for NDBC buoys near ({latitude}, {longitude}) within {radius_km} km")
        
        # Hardcoded list of common East Coast buoys for demonstration
        # In production, this would query the NDBC station database
        known_buoys = [
            {'id': '44025', 'name': 'Long Island 30 NM South of Islip, NY', 'lat': 40.251, 'lon': -73.164},
            {'id': '44014', 'name': 'Virginia Beach 64 NM East of Virginia Beach, VA', 'lat': 36.611, 'lon': -74.845},
            {'id': '44009', 'name': 'Delaware Bay 26 NM Southeast of Cape Henlopen, DE', 'lat': 38.461, 'lon': -74.703},
            {'id': '41025', 'name': 'Diamond Shoals 40 NM Southeast of Cape Hatteras, NC', 'lat': 35.006, 'lon': -75.402},
            {'id': '41013', 'name': 'Frying Pan Shoals 20 NM Southeast of Cape Fear, NC', 'lat': 33.436, 'lon': -77.743},
            {'id': '44007', 'name': 'Portland 12 NM East of Portsmouth, NH', 'lat': 43.525, 'lon': -70.141},
            {'id': '44008', 'name': 'Nantucket 54 NM Southeast of Nantucket, MA', 'lat': 40.504, 'lon': -69.247},
            {'id': '44011', 'name': 'Georges Bank 170 NM East of Hyannis, MA', 'lat': 41.089, 'lon': -66.618},
            {'id': '44017', 'name': 'Montauk Point 19 NM South of Montauk Point, NY', 'lat': 40.691, 'lon': -72.048},
            {'id': '44020', 'name': 'Nantucket Sound 24 NM Southeast of Nantucket, MA', 'lat': 41.443, 'lon': -70.279}
        ]
        
        nearby_buoys = []
        
        for buoy in known_buoys:
            try:
                distance_km = self._calculate_distance(latitude, longitude, buoy['lat'], buoy['lon'])
                
                if distance_km <= radius_km:
                    buoy_info = {
                        'id': buoy['id'],
                        'name': buoy['name'],
                        'latitude': buoy['lat'],
                        'longitude': buoy['lon'],
                        'distance_km': round(distance_km, 2),
                        'type': 'buoy'
                    }
                    nearby_buoys.append(buoy_info)
            
            except (ValueError, TypeError, KeyError):
                continue
        
        # Sort by distance
        nearby_buoys.sort(key=lambda x: x['distance_km'])
        
        self.logger.info(f"Found {len(nearby_buoys)} NDBC buoys within {radius_km} km")
        return nearby_buoys
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_meteorological_data(
        self, 
        station_id: str, 
        days_back: int = 45
    ) -> pd.DataFrame:
        """
        Get standard meteorological data from a buoy or coastal station.
        
        Args:
            station_id: NDBC station ID (e.g., '44025')
            days_back: Number of days of data to retrieve (max 45)
            
        Returns:
            DataFrame with meteorological observations
        """
        self.logger.info(f"Fetching meteorological data for station {station_id} ({days_back} days)")
        
        url = f"{self.base_url}{station_id}.txt"
        
        try:
            response = self._make_request(url)
            content = response.text
            
            # Parse the NDBC standard meteorological format
            df = self._parse_ndbc_standard_met(content, station_id)
            
            # Filter to requested time range
            if not df.empty and days_back < 45:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df['datetime'] >= cutoff_date]
            
            self.logger.info(f"Retrieved {len(df)} meteorological observations for station {station_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching meteorological data: {e}")
            return pd.DataFrame()
    
    def get_wave_data(
        self, 
        station_id: str, 
        days_back: int = 45
    ) -> pd.DataFrame:
        """
        Get spectral wave data from a buoy.
        
        Args:
            station_id: NDBC station ID (e.g., '44025')
            days_back: Number of days of data to retrieve (max 45)
            
        Returns:
            DataFrame with wave measurements
        """
        self.logger.info(f"Fetching wave data for station {station_id} ({days_back} days)")
        
        url = f"{self.base_url}{station_id}.spec"
        
        try:
            response = self._make_request(url)
            content = response.text
            
            # Parse the NDBC spectral wave format
            df = self._parse_ndbc_wave_spec(content, station_id)
            
            # Filter to requested time range
            if not df.empty and days_back < 45:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df['datetime'] >= cutoff_date]
            
            self.logger.info(f"Retrieved {len(df)} wave observations for station {station_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching wave data: {e}")
            return pd.DataFrame()
    
    def get_continuous_winds(
        self, 
        station_id: str, 
        days_back: int = 45
    ) -> pd.DataFrame:
        """
        Get continuous wind data from a buoy or coastal station.
        
        Args:
            station_id: NDBC station ID (e.g., '44025')
            days_back: Number of days of data to retrieve (max 45)
            
        Returns:
            DataFrame with continuous wind measurements
        """
        self.logger.info(f"Fetching continuous wind data for station {station_id} ({days_back} days)")
        
        url = f"{self.base_url}{station_id}.cwind"
        
        try:
            response = self._make_request(url)
            content = response.text
            
            # Parse the NDBC continuous wind format
            df = self._parse_ndbc_continuous_winds(content, station_id)
            
            # Filter to requested time range
            if not df.empty and days_back < 45:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df['datetime'] >= cutoff_date]
            
            self.logger.info(f"Retrieved {len(df)} continuous wind observations for station {station_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching continuous wind data: {e}")
            return pd.DataFrame()
    
    def _parse_ndbc_standard_met(self, content: str, station_id: str) -> pd.DataFrame:
        """
        Parse NDBC standard meteorological data format.
        
        Format: YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS TIDE
        """
        try:
            lines = content.strip().split('\n')
            
            # Skip header lines (usually first 2 lines are headers)
            data_lines = [line for line in lines[2:] if line.strip() and not line.startswith('#')]
            
            if not data_lines:
                return pd.DataFrame()
            
            # Parse each line
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 13:  # Minimum required columns
                    try:
                        # Handle 2-digit vs 4-digit years
                        year = int(parts[0])
                        if year < 50:
                            year += 2000
                        elif year < 100:
                            year += 1900
                        
                        record = {
                            'year': year,
                            'month': int(parts[1]),
                            'day': int(parts[2]),
                            'hour': int(parts[3]),
                            'minute': int(parts[4]),
                            'wind_direction_deg': self._safe_float(parts[5]),
                            'wind_speed_ms': self._safe_float(parts[6]),
                            'wind_gust_ms': self._safe_float(parts[7]),
                            'wave_height_m': self._safe_float(parts[8]),
                            'dominant_wave_period_s': self._safe_float(parts[9]),
                            'average_wave_period_s': self._safe_float(parts[10]),
                            'mean_wave_direction_deg': self._safe_float(parts[11]),
                            'pressure_hpa': self._safe_float(parts[12])
                        }
                        
                        # Additional columns if available
                        if len(parts) > 13:
                            record['air_temp_c'] = self._safe_float(parts[13])
                        if len(parts) > 14:
                            record['water_temp_c'] = self._safe_float(parts[14])
                        if len(parts) > 15:
                            record['dewpoint_c'] = self._safe_float(parts[15])
                        if len(parts) > 16:
                            record['visibility_km'] = self._safe_float(parts[16])
                        if len(parts) > 17:
                            record['tide_m'] = self._safe_float(parts[17])
                        
                        data.append(record)
                    
                    except (ValueError, IndexError):
                        continue
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Create datetime column
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
            
            # Add metadata
            df['station_id'] = station_id
            df['data_type'] = 'meteorological'
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['datetime'])
            
            return df.sort_values('datetime')
            
        except Exception as e:
            self.logger.error(f"Error parsing standard meteorological data: {e}")
            return pd.DataFrame()
    
    def _parse_ndbc_wave_spec(self, content: str, station_id: str) -> pd.DataFrame:
        """
        Parse NDBC spectral wave data format.
        """
        try:
            lines = content.strip().split('\n')
            
            # Skip header lines
            data_lines = [line for line in lines[2:] if line.strip() and not line.startswith('#')]
            
            if not data_lines:
                return pd.DataFrame()
            
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 5:  # Minimum required columns
                    try:
                        # Handle 2-digit vs 4-digit years
                        year = int(parts[0])
                        if year < 50:
                            year += 2000
                        elif year < 100:
                            year += 1900
                        
                        record = {
                            'year': year,
                            'month': int(parts[1]),
                            'day': int(parts[2]),
                            'hour': int(parts[3]),
                            'minute': int(parts[4]),
                        }
                        
                        # Variable number of wave parameters depending on format
                        if len(parts) > 5:
                            record['significant_wave_height_m'] = self._safe_float(parts[5])
                        if len(parts) > 6:
                            record['swell_height_m'] = self._safe_float(parts[6])
                        if len(parts) > 7:
                            record['swell_period_s'] = self._safe_float(parts[7])
                        if len(parts) > 8:
                            record['swell_direction_deg'] = self._safe_float(parts[8])
                        if len(parts) > 9:
                            record['wind_wave_height_m'] = self._safe_float(parts[9])
                        if len(parts) > 10:
                            record['wind_wave_period_s'] = self._safe_float(parts[10])
                        if len(parts) > 11:
                            record['wind_wave_direction_deg'] = self._safe_float(parts[11])
                        
                        data.append(record)
                    
                    except (ValueError, IndexError):
                        continue
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Create datetime column
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
            
            # Add metadata
            df['station_id'] = station_id
            df['data_type'] = 'wave_spectral'
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['datetime'])
            
            return df.sort_values('datetime')
            
        except Exception as e:
            self.logger.error(f"Error parsing wave spectral data: {e}")
            return pd.DataFrame()
    
    def _parse_ndbc_continuous_winds(self, content: str, station_id: str) -> pd.DataFrame:
        """
        Parse NDBC continuous wind data format.
        """
        try:
            lines = content.strip().split('\n')
            
            # Skip header lines
            data_lines = [line for line in lines[2:] if line.strip() and not line.startswith('#')]
            
            if not data_lines:
                return pd.DataFrame()
            
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 7:  # Minimum required columns for continuous winds
                    try:
                        # Handle 2-digit vs 4-digit years
                        year = int(parts[0])
                        if year < 50:
                            year += 2000
                        elif year < 100:
                            year += 1900
                        
                        record = {
                            'year': year,
                            'month': int(parts[1]),
                            'day': int(parts[2]),
                            'hour': int(parts[3]),
                            'minute': int(parts[4]),
                            'wind_direction_deg': self._safe_float(parts[5]),
                            'wind_speed_ms': self._safe_float(parts[6])
                        }
                        
                        # Additional wind parameters if available
                        if len(parts) > 7:
                            record['wind_gust_ms'] = self._safe_float(parts[7])
                        
                        data.append(record)
                    
                    except (ValueError, IndexError):
                        continue
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Create datetime column
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
            
            # Add metadata
            df['station_id'] = station_id
            df['data_type'] = 'continuous_winds'
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['datetime'])
            
            return df.sort_values('datetime')
            
        except Exception as e:
            self.logger.error(f"Error parsing continuous wind data: {e}")
            return pd.DataFrame()
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float, handling NDBC missing data codes."""
        try:
            if value in ['MM', '999', '999.0', '9999', '99.0', '999.00']:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def interpolate_to_timestamps(
        self,
        ndbc_data: pd.DataFrame,
        target_timestamps: pd.Series,
        value_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Interpolate NDBC data to match target timestamps (e.g., GNSS-IR measurements).
        
        Args:
            ndbc_data: DataFrame with NDBC data
            target_timestamps: Series of target datetime stamps
            value_columns: List of columns to interpolate
            
        Returns:
            DataFrame with interpolated values at target timestamps
        """
        if ndbc_data.empty or target_timestamps.empty:
            return pd.DataFrame()
        
        if value_columns is None:
            # Default columns to interpolate
            value_columns = [col for col in ndbc_data.columns 
                           if col.endswith(('_m', '_ms', '_deg', '_s', '_hpa', '_c', '_km'))]
        
        try:
            # Ensure datetime is the index for interpolation
            ndbc_indexed = ndbc_data.set_index('datetime').sort_index()
            
            # Create a continuous time series with 10-minute resolution
            start_time = min(target_timestamps.min(), ndbc_indexed.index.min())
            end_time = max(target_timestamps.max(), ndbc_indexed.index.max())
            
            continuous_index = pd.date_range(start=start_time, end=end_time, freq='10min')
            
            # Interpolate to continuous time series
            continuous_data = ndbc_indexed.reindex(continuous_index)
            
            for col in value_columns:
                if col in continuous_data.columns:
                    continuous_data[col] = continuous_data[col].interpolate(method='linear')
            
            # Extract values at target timestamps
            result_data = {'datetime': target_timestamps}
            
            for col in value_columns:
                if col in continuous_data.columns:
                    interpolated_values = []
                    for timestamp in target_timestamps:
                        closest_idx = continuous_data.index.get_indexer([timestamp], method='nearest')[0]
                        closest_time = continuous_data.index[closest_idx]
                        interpolated_values.append(continuous_data.loc[closest_time, col])
                    
                    result_data[f'{col}_interpolated'] = interpolated_values
            
            # Create result DataFrame
            result_df = pd.DataFrame(result_data)
            
            # Add metadata from original data
            if not ndbc_data.empty:
                for col in ['station_id', 'data_type']:
                    if col in ndbc_data.columns:
                        result_df[col] = ndbc_data[col].iloc[0]
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error interpolating NDBC data: {e}")
            return pd.DataFrame()

def test_ndbc_client():
    """Basic test function for the NDBC client."""
    client = NDBCClient()
    
    # Test with Mid-Atlantic location (near FORA station)
    test_lat, test_lon = 36.0, -75.0
    
    print("Testing NDBC Client...")
    
    # Find nearby buoys
    buoys = client.find_nearby_buoys(test_lat, test_lon, radius_km=200)
    print(f"Found {len(buoys)} buoys")
    
    if buoys:
        buoy_id = buoys[0]['id']
        print(f"Testing with buoy: {buoy_id} - {buoys[0]['name']}")
        
        # Test meteorological data
        met_data = client.get_meteorological_data(buoy_id, days_back=2)
        print(f"Retrieved {len(met_data)} meteorological observations")
        
        if not met_data.empty:
            print("Sample meteorological data:")
            print(met_data[['datetime', 'wind_speed_ms', 'wave_height_m', 'pressure_hpa']].head())
        
        # Test wave data
        wave_data = client.get_wave_data(buoy_id, days_back=2)
        print(f"Retrieved {len(wave_data)} wave observations")
        
        if not wave_data.empty:
            print("Sample wave data:")
            print(wave_data[['datetime', 'significant_wave_height_m']].head())

if __name__ == "__main__":
    test_ndbc_client()