# ABOUTME: NOAA CO-OPS API client for tide predictions and water level observations
# ABOUTME: Retrieves verified/preliminary water levels, predictions, and station metadata

import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import time


class NOAACOOPSClient:
    """
    Client for NOAA CO-OPS API integration.

    Provides methods to retrieve tide predictions, water level observations,
    and station metadata for integration with GNSS-IR analysis.
    """

    def __init__(
        self,
        base_url: str = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the NOAA CO-OPS API client.

        Args:
            base_url: Base URL for the CO-OPS API
            cache_dir: Directory for caching API responses (optional)
        """
        self.base_url = base_url
        self.metadata_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GNSS-IR-Processing/1.0 (Research Application)"})

        # API rate limiting (be respectful)
        self.min_request_interval = 0.1  # 100ms between requests
        self.last_request_time = 0

        self.logger = logging.getLogger(__name__)

        # Set up cache directory
        if cache_dir is None:
            # Default cache directory in project data folder
            self.cache_dir = Path("data") / ".cache" / "noaa_coops"
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"NOAA CO-OPS cache directory: {self.cache_dir}")

    def _generate_cache_key(
        self, product: str, station_id: str, start_date: datetime, end_date: datetime, **kwargs
    ) -> str:
        """Generate a unique cache key for the request."""
        # Create a unique identifier based on request parameters
        key_parts = [
            product,
            station_id,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
        ]

        # Add optional parameters
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")

        return "_".join(key_parts) + ".json"

    def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached data if available."""
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                self.logger.info(f"Using cached data from {cache_file}")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        cache_file = self.cache_dir / cache_key

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
            self.logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def clear_cache(self, station_id: Optional[str] = None, product: Optional[str] = None):
        """
        Clear cache files.

        Args:
            station_id: Clear only cache for specific station (optional)
            product: Clear only specific product type ('predictions' or 'water_level') (optional)
        """
        import os

        if station_id and product:
            # Clear specific station and product
            pattern = f"{product}_{station_id}_*.json"
        elif station_id:
            # Clear all products for specific station
            pattern = f"*_{station_id}_*.json"
        elif product:
            # Clear specific product for all stations
            pattern = f"{product}_*.json"
        else:
            # Clear all cache
            pattern = "*.json"

        cache_files = list(self.cache_dir.glob(pattern))
        removed_count = 0

        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        self.logger.info(f"Cleared {removed_count} cache files matching pattern: {pattern}")

    def _rate_limit(self):
        """Implement respectful rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict) -> requests.Response:
        """
        Make a rate-limited API request with error handling.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def find_nearby_stations(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100,
        product_types: List[str] = None,
    ) -> List[Dict]:
        """
        Find CO-OPS stations within a specified radius of coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            radius_km: Search radius in kilometers
            product_types: Filter by product types (e.g., ['water_level', 'predictions'])

        Returns:
            List of station dictionaries with metadata
        """
        self.logger.info(
            f"Searching for CO-OPS stations near ({latitude}, {longitude}) within {radius_km} km"
        )

        try:
            # Use metadata API to get stations
            url = f"{self.metadata_url}/stations.json"
            params = {"type": "waterlevels", "expand": "details"}  # Focus on water level stations

            response = self._make_request(url, params)
            stations_data = response.json()

            nearby_stations = []

            for station in stations_data.get("stations", []):
                try:
                    station_lat = float(station.get("lat", 0))
                    station_lon = float(station.get("lng", 0))

                    # Calculate approximate distance using haversine formula
                    distance_km = self._calculate_distance(
                        latitude, longitude, station_lat, station_lon
                    )

                    if distance_km <= radius_km:
                        station_info = {
                            "id": station.get("id"),
                            "name": station.get("name"),
                            "latitude": station_lat,
                            "longitude": station_lon,
                            "distance_km": round(distance_km, 2),
                            "state": station.get("state"),
                            "timezone": station.get("timezone"),
                            "type": station.get("type"),
                            "details": station.get("details", {}),
                        }
                        nearby_stations.append(station_info)

                except (ValueError, TypeError, KeyError):
                    continue

            # Sort by distance
            nearby_stations.sort(key=lambda x: x["distance_km"])

            self.logger.info(f"Found {len(nearby_stations)} CO-OPS stations within {radius_km} km")
            return nearby_stations

        except Exception as e:
            self.logger.error(f"Error finding nearby stations: {e}")
            return []

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using haversine formula."""
        R = 6371  # Earth's radius in kilometers

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_tide_predictions(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        datum: str = "MLLW",
        interval: str = "hilo",
        time_zone: str = "lst_ldt",
        units: str = "metric",
    ) -> pd.DataFrame:
        """
        Get tide predictions for a station and date range.

        Args:
            station_id: 7-character CO-OPS station ID
            start_date: Start date for predictions
            end_date: End date for predictions
            datum: Vertical datum (MLLW, MSL, NAVD88, etc.)
            interval: Data interval ('hilo' for high/low, 'h' for hourly, '6' for 6-minute)
            time_zone: Time zone ('gmt', 'lst_ldt')
            units: Units ('metric' or 'english')

        Returns:
            DataFrame with tide predictions
        """
        self.logger.info(
            f"Fetching tide predictions for station {station_id} from {start_date} to {end_date}"
        )

        # Check cache first
        cache_key = self._generate_cache_key(
            "predictions",
            station_id,
            start_date,
            end_date,
            datum=datum,
            interval=interval,
            time_zone=time_zone,
            units=units,
        )

        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            data = cached_data
        else:
            # Fetch from API
            params = {
                "product": "predictions",
                "application": "GNSS_IR_Analysis",
                "format": "json",
                "station": station_id,
                "begin_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
                "datum": datum,
                "interval": interval,
                "time_zone": time_zone,
                "units": units,
            }

            try:
                response = self._make_request(self.base_url, params)
                data = response.json()

                # Save to cache for future use
                self._save_to_cache(cache_key, data)
            except Exception as e:
                self.logger.error(f"API request failed: {e}")
                raise

        if "predictions" not in data:
            self.logger.warning(f"No predictions data returned for station {station_id}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data["predictions"])

        if df.empty:
            return df

        # Parse datetime
        df["datetime"] = pd.to_datetime(df["t"])
        df["prediction_m"] = pd.to_numeric(df["v"], errors="coerce")

        # Add metadata
        df["station_id"] = station_id
        df["datum"] = datum
        df["units"] = units
        df["data_type"] = "predictions"

        # Clean up columns
        df = df[["datetime", "prediction_m", "station_id", "datum", "units", "data_type"]]
        df = df.dropna(subset=["prediction_m"])

        self.logger.info(f"Retrieved {len(df)} tide predictions for station {station_id}")
        return df

    def get_water_level_observations(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        datum: str = "MLLW",
        interval: str = "h",
        time_zone: str = "lst_ldt",
        units: str = "metric",
    ) -> pd.DataFrame:
        """
        Get observed water level data for a station and date range.

        Args:
            station_id: 7-character CO-OPS station ID
            start_date: Start date for observations
            end_date: End date for observations
            datum: Vertical datum (MLLW, MSL, NAVD88, etc.)
            interval: Data interval ('h' for hourly, '6' for 6-minute)
            time_zone: Time zone ('gmt', 'lst_ldt')
            units: Units ('metric' or 'english')

        Returns:
            DataFrame with water level observations
        """
        self.logger.info(
            f"Fetching water level observations for station {station_id} "
            f"from {start_date} to {end_date}"
        )

        # Check if we can get all data from cache
        full_cache_key = self._generate_cache_key(
            "water_level",
            station_id,
            start_date,
            end_date,
            datum=datum,
            interval=interval,
            time_zone=time_zone,
            units=units,
        )

        cached_data = self._get_cached_data(full_cache_key)
        if cached_data is not None:
            if "data" not in cached_data:
                self.logger.warning(f"No water level data in cache for station {station_id}")
                return pd.DataFrame()

            # Create DataFrame from cached data
            df = pd.DataFrame(cached_data["data"])
        else:
            # NOAA CO-OPS has a limit of 31 days for water level data
            date_diff = (end_date - start_date).days
            if date_diff > 31:
                self.logger.warning(
                    f"Date range ({date_diff} days) exceeds CO-OPS limit of 31 days "
                    "for water level data. Fetching in chunks..."
                )

            # Fetch data in 30-day chunks
            all_data = []
            current_date = start_date

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=30), end_date)

                params = {
                    "product": "water_level",
                    "application": "GNSS_IR_Analysis",
                    "format": "json",
                    "station": station_id,
                    "begin_date": current_date.strftime("%Y%m%d"),
                    "end_date": chunk_end.strftime("%Y%m%d"),
                    "datum": datum,
                    "interval": interval,
                    "time_zone": time_zone,
                    "units": units,
                }

                # Check cache for this chunk first
                chunk_cache_key = self._generate_cache_key(
                    "water_level",
                    station_id,
                    current_date,
                    chunk_end,
                    datum=datum,
                    interval=interval,
                    time_zone=time_zone,
                    units=units,
                )

                chunk_cached_data = self._get_cached_data(chunk_cache_key)
                if chunk_cached_data is not None and "data" in chunk_cached_data:
                    all_data.extend(chunk_cached_data["data"])
                else:
                    try:
                        response = self._make_request(self.base_url, params)
                        data = response.json()

                        if "data" in data:
                            all_data.extend(data["data"])
                            # Cache this chunk
                            self._save_to_cache(chunk_cache_key, data)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to fetch chunk from {current_date} to {chunk_end}: {e}"
                        )

                current_date = chunk_end + timedelta(days=1)

            if not all_data:
                return pd.DataFrame()

            # Create DataFrame from all chunks
            df = pd.DataFrame(all_data)

            # Save the combined data to cache for next time
            combined_data = {"data": all_data}
            self._save_to_cache(full_cache_key, combined_data)

        if df.empty:
            return df

        # Parse datetime and values
        df["datetime"] = pd.to_datetime(df["t"])
        df["water_level_m"] = pd.to_numeric(df["v"], errors="coerce")

        # Add quality flags if available
        if "f" in df.columns:
            df["quality_flag"] = df["f"]

        # Add metadata
        df["station_id"] = station_id
        df["datum"] = datum
        df["units"] = units
        df["data_type"] = "observations"

        # Clean up columns
        base_cols = ["datetime", "water_level_m", "station_id", "datum", "units", "data_type"]
        if "quality_flag" in df.columns:
            base_cols.append("quality_flag")

        df = df[base_cols]
        df = df.dropna(subset=["water_level_m"])

        self.logger.info(f"Retrieved {len(df)} water level observations for station {station_id}")
        return df

    def get_high_low_tides(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        datum: str = "MLLW",
        time_zone: str = "lst_ldt",
        units: str = "metric",
    ) -> pd.DataFrame:
        """
        Get high and low tide times and heights.

        Args:
            station_id: 7-character CO-OPS station ID
            start_date: Start date for high/low tides
            end_date: End date for high/low tides
            datum: Vertical datum (MLLW, MSL, NAVD88, etc.)
            time_zone: Time zone ('gmt', 'lst_ldt')
            units: Units ('metric' or 'english')

        Returns:
            DataFrame with high/low tide information
        """
        self.logger.info(
            f"Fetching high/low tides for station {station_id} from {start_date} to {end_date}"
        )

        params = {
            "product": "predictions",
            "application": "GNSS_IR_Analysis",
            "format": "json",
            "station": station_id,
            "begin_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "datum": datum,
            "interval": "hilo",
            "time_zone": time_zone,
            "units": units,
        }

        try:
            response = self._make_request(self.base_url, params)
            data = response.json()

            if "predictions" not in data:
                self.logger.warning(f"No high/low tide data returned for station {station_id}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data["predictions"])

            if df.empty:
                return df

            # Parse datetime and values
            df["datetime"] = pd.to_datetime(df["t"])
            df["height_m"] = pd.to_numeric(df["v"], errors="coerce")

            # Add tide type (High/Low) if available
            if "type" in df.columns:
                df["tide_type"] = df["type"]
            else:
                # Infer tide type from height pattern
                df = df.sort_values("datetime")
                df["tide_type"] = "Unknown"

                # Simple peak/trough detection
                heights = df["height_m"].values
                for i in range(1, len(heights) - 1):
                    if heights[i] > heights[i - 1] and heights[i] > heights[i + 1]:
                        df.iloc[i, df.columns.get_loc("tide_type")] = "High"
                    elif heights[i] < heights[i - 1] and heights[i] < heights[i + 1]:
                        df.iloc[i, df.columns.get_loc("tide_type")] = "Low"

            # Add metadata
            df["station_id"] = station_id
            df["datum"] = datum
            df["units"] = units
            df["data_type"] = "high_low_tides"

            # Clean up columns
            df = df[
                ["datetime", "height_m", "tide_type", "station_id", "datum", "units", "data_type"]
            ]
            df = df.dropna(subset=["height_m"])

            self.logger.info(f"Retrieved {len(df)} high/low tides for station {station_id}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching high/low tides: {e}")
            return pd.DataFrame()

    def get_station_metadata(self, station_id: str) -> Dict:
        """
        Get detailed metadata for a specific station.

        Args:
            station_id: 7-character CO-OPS station ID

        Returns:
            Dictionary with station metadata
        """
        try:
            url = f"{self.metadata_url}/stations/{station_id}.json"
            response = self._make_request(url, {})

            station_data = response.json()

            if "stations" in station_data and station_data["stations"]:
                station = station_data["stations"][0]

                metadata = {
                    "id": station.get("id"),
                    "name": station.get("name"),
                    "latitude": float(station.get("lat", 0)),
                    "longitude": float(station.get("lng", 0)),
                    "state": station.get("state"),
                    "timezone": station.get("timezone"),
                    "type": station.get("type"),
                    "datum_info": station.get("datums", {}),
                    "products": station.get("products", []),
                    "sensors": station.get("sensors", []),
                }

                return metadata

            return {}

        except Exception as e:
            self.logger.error(f"Error fetching station metadata: {e}")
            return {}

    def interpolate_to_timestamps(
        self,
        coops_data: pd.DataFrame,
        target_timestamps: pd.Series,
        value_column: str = "prediction_m",
    ) -> pd.DataFrame:
        """
        Interpolate CO-OPS data to match target timestamps (e.g., GNSS-IR measurements).

        Args:
            coops_data: DataFrame with CO-OPS data
            target_timestamps: Series of target datetime stamps
            value_column: Column name containing values to interpolate

        Returns:
            DataFrame with interpolated values at target timestamps
        """
        if coops_data.empty or target_timestamps.empty:
            return pd.DataFrame()

        try:
            # Ensure datetime is the index for interpolation
            coops_indexed = coops_data.set_index("datetime").sort_index()

            # Create a continuous time series with 1-minute resolution for better interpolation
            start_time = min(target_timestamps.min(), coops_indexed.index.min())
            end_time = max(target_timestamps.max(), coops_indexed.index.max())

            continuous_index = pd.date_range(start=start_time, end=end_time, freq="1min")

            # Interpolate to continuous time series
            continuous_data = coops_indexed.reindex(continuous_index)
            continuous_data[value_column] = continuous_data[value_column].interpolate(
                method="cubic"
            )

            # Extract values at target timestamps
            interpolated_values = []
            for timestamp in target_timestamps:
                closest_idx = continuous_data.index.get_indexer([timestamp], method="nearest")[0]
                closest_time = continuous_data.index[closest_idx]
                interpolated_values.append(continuous_data.loc[closest_time, value_column])

            # Create result DataFrame
            result_df = pd.DataFrame(
                {"datetime": target_timestamps, f"{value_column}_interpolated": interpolated_values}
            )

            # Add metadata from original data
            if not coops_data.empty:
                for col in ["station_id", "datum", "units", "data_type"]:
                    if col in coops_data.columns:
                        result_df[col] = coops_data[col].iloc[0]

            return result_df

        except Exception as e:
            self.logger.error(f"Error interpolating CO-OPS data: {e}")
            return pd.DataFrame()


def test_coops_client():
    """Basic test function for the CO-OPS client."""
    client = NOAACOOPSClient()

    # Test with Duck, NC area (near FORA station)
    test_lat, test_lon = 36.1833, -75.7467

    print("Testing NOAA CO-OPS Client...")

    # Find nearby stations
    stations = client.find_nearby_stations(test_lat, test_lon, radius_km=100)
    print(f"Found {len(stations)} stations")

    if stations:
        station_id = stations[0]["id"]
        print(f"Testing with station: {station_id} - {stations[0]['name']}")

        # Test tide predictions for recent date
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        predictions = client.get_tide_predictions(station_id, start_date, end_date)
        print(f"Retrieved {len(predictions)} tide predictions")

        if not predictions.empty:
            print("Sample predictions:")
            print(predictions.head())


if __name__ == "__main__":
    test_coops_client()
