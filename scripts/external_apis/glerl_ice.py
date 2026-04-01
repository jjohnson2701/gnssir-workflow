# ABOUTME: GLERL gridded ice concentration client for Great Lakes stations
# ABOUTME: Fetches station-local ice concentration from ERDDAP griddap endpoint

"""
GLERL Gridded Ice Concentration Client.

Fetches ice concentration data from the NOAA GLERL ERDDAP griddap endpoint
for a small bounding box around each GNSS-IR station. Returns a daily time
series of local ice concentration (percent, 0-100).

Dataset: GL_Ice_Concentration_GCS
    - Source: GLSEA + National Ice Center
    - Resolution: ~0.014° (~1.3 km)
    - Coverage: Great Lakes, 1995-present
    - Variable: ice_concentration (percent)

Usage:
    from scripts.external_apis.glerl_ice import GLERLIceClient

    client = GLERLIceClient()
    df = client.get_ice_concentration("ROSS", 2023)
    # Returns DataFrame with columns: [datetime, ice_concentration, n_cells]
"""

import json
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# GLERL ERDDAP griddap endpoint (apps.glerl, NOT coastwatch — that redirects)
GRIDDAP_BASE = "https://apps.glerl.noaa.gov/erddap/griddap"
DATASET_ID = "GL_Ice_Concentration_GCS"
VARIABLE = "ice_concentration"

# Spatial bounds of the dataset
LAT_MIN, LAT_MAX = 38.875, 50.606
LON_MIN, LON_MAX = -92.420, -75.882

# Default bounding box half-width in degrees (~5 km radius at GL latitudes)
DEFAULT_RADIUS_DEG = 0.05

# Missing value sentinel in the dataset
MISSING_VALUE = -99999.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class GLERLIceClient:
    """Client for GLERL gridded ice concentration data via ERDDAP."""

    def __init__(self, cache_dir: Optional[Path] = None, radius_deg: float = DEFAULT_RADIUS_DEG):
        """
        Args:
            cache_dir: Directory for caching downloaded CSVs.
            radius_deg: Half-width of bounding box around station (degrees).
                        0.05° ≈ 5 km, covers ~50 grid cells.
        """
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "GNSS-IR-Processing/1.0 (Research)"}
        )
        self.radius_deg = radius_deg

        if cache_dir is None:
            self.cache_dir = PROJECT_ROOT / "data" / ".cache" / "glerl_ice"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ice_concentration(
        self,
        station: str,
        year: int,
        month_start: int = 11,
        month_end: int = 6,
        azimuth_filter: bool = False,
        max_distance_km: float = 10.0,
    ) -> pd.DataFrame:
        """Fetch local ice concentration time series for a station's ice season.

        The ice season spans November of the previous year through June.
        For year=2023, this fetches 2022-11-01 to 2023-06-30.

        Args:
            station: Station name (must be in stations_config.json with lat/lon).
            year: The winter year (e.g., 2023 = winter 2022-2023).
            month_start: Start month of ice season (in prior year). Default 11.
            month_end: End month of ice season. Default 6.
            azimuth_filter: If True, only average cells within the station's
                            GNSS-IR azimuth window (from config azval2).
            max_distance_km: Max distance from station for azimuth filtering.

        Returns:
            DataFrame with columns [datetime, ice_concentration, n_cells].
            ice_concentration is the spatial mean (%) over the selected cells.
            n_cells is the number of valid grid cells averaged.
        """
        lat, lon = self._get_station_coords(station)
        self._validate_coverage(station, lat, lon)

        t_start = datetime(year - 1, month_start, 1)
        t_end = datetime(year, month_end, 30)

        # Determine cache filename (different for azimuth-filtered)
        if azimuth_filter:
            cache_file = self._cache_path_az(station, year)
        else:
            cache_file = self._cache_path(station, year)

        if cache_file.exists():
            logger.info(f"Loading cached ice data: {cache_file.name}")
            return pd.read_csv(cache_file, parse_dates=["datetime"])

        # Load azimuth window from station gnssir config
        az_min, az_max = None, None
        if azimuth_filter:
            az_min, az_max = self._get_station_azimuths(station)
            logger.info(
                f"Azimuth-filtered mode: {az_min}°–{az_max}°, "
                f"max distance {max_distance_km} km"
            )
            # Expand fetch radius to cover the distance + azimuth wedge
            fetch_radius = max(self.radius_deg, max_distance_km / 111.0)
        else:
            fetch_radius = self.radius_deg

        logger.info(
            f"Fetching GLERL ice for {station} ({lat:.3f}°N, {lon:.3f}°E), "
            f"{t_start:%Y-%m-%d} to {t_end:%Y-%m-%d}"
        )

        # Use wider radius for azimuth-filtered fetches
        orig_radius = self.radius_deg
        self.radius_deg = fetch_radius
        raw_df = self._fetch_griddap(lat, lon, t_start, t_end)
        self.radius_deg = orig_radius

        if raw_df.empty:
            logger.warning(f"No ice data returned for {station} {year}")
            return pd.DataFrame(columns=["datetime", "ice_concentration", "n_cells"])

        daily = self._aggregate_daily(
            raw_df,
            station_lat=lat if azimuth_filter else None,
            station_lon=lon if azimuth_filter else None,
            az_min=az_min,
            az_max=az_max,
            max_distance_km=max_distance_km if azimuth_filter else None,
        )
        daily.to_csv(cache_file, index=False)
        logger.info(f"Cached {len(daily)} days to {cache_file.name}")
        return daily

    def get_ice_for_dates(
        self,
        station: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch ice concentration for an arbitrary date range.

        Unlike get_ice_concentration(), this doesn't assume an ice season.
        No caching — use for ad-hoc queries.
        """
        lat, lon = self._get_station_coords(station)
        self._validate_coverage(station, lat, lon)

        raw_df = self._fetch_griddap(lat, lon, start_date, end_date)
        if raw_df.empty:
            return pd.DataFrame(columns=["datetime", "ice_concentration", "n_cells"])
        return self._aggregate_daily(raw_df)

    def station_in_coverage(self, station: str) -> bool:
        """Check whether a station falls within the GLERL grid bounds."""
        try:
            lat, lon = self._get_station_coords(station)
            return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX
        except (ValueError, KeyError):
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_station_coords(self, station: str) -> tuple:
        """Look up station lat/lon from stations_config.json."""
        config_path = PROJECT_ROOT / "config" / "stations_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        if station not in cfg:
            raise ValueError(f"Station {station} not in stations_config.json")
        sc = cfg[station]
        return sc["latitude_deg"], sc["longitude_deg"]

    def _validate_coverage(self, station: str, lat: float, lon: float):
        """Raise if station is outside the GLERL grid."""
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            raise ValueError(
                f"{station} at ({lat:.2f}, {lon:.2f}) is outside GLERL Great Lakes "
                f"coverage ({LAT_MIN}-{LAT_MAX}°N, {LON_MIN}-{LON_MAX}°W)"
            )

    def _cache_path(self, station: str, year: int) -> Path:
        return self.cache_dir / f"{station.lower()}_ice_{year}.csv"

    def _cache_path_az(self, station: str, year: int) -> Path:
        return self.cache_dir / f"{station.lower()}_ice_{year}_azfilt.csv"

    def _get_station_azimuths(self, station: str) -> tuple:
        """Read azimuth window from the station's gnssir JSON config."""
        config_path = PROJECT_ROOT / "config" / "stations_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        sc = cfg.get(station, {})
        gnssir_path = sc.get("gnssir_json_params_path")
        if gnssir_path:
            full_path = PROJECT_ROOT / gnssir_path
            if full_path.exists():
                with open(full_path) as f:
                    params = json.load(f)
                azval2 = params.get("azval2", [0, 360])
                return float(azval2[0]), float(azval2[1])
        return 0.0, 360.0

    def _build_url(
        self, lat: float, lon: float, t_start: datetime, t_end: datetime
    ) -> str:
        """Construct ERDDAP griddap CSV request URL."""
        lat_lo = lat - self.radius_deg
        lat_hi = lat + self.radius_deg
        lon_lo = lon - self.radius_deg
        lon_hi = lon + self.radius_deg

        # Clamp to dataset bounds
        lat_lo = max(lat_lo, LAT_MIN)
        lat_hi = min(lat_hi, LAT_MAX)
        lon_lo = max(lon_lo, LON_MIN)
        lon_hi = min(lon_hi, LON_MAX)

        url = (
            f"{GRIDDAP_BASE}/{DATASET_ID}.csv?"
            f"{VARIABLE}"
            f"[({t_start:%Y-%m-%dT%H:%M:%SZ}):1:({t_end:%Y-%m-%dT%H:%M:%SZ})]"
            f"[({lat_lo}):1:({lat_hi})]"
            f"[({lon_lo}):1:({lon_hi})]"
        )
        return url

    def _fetch_griddap(
        self, lat: float, lon: float, t_start: datetime, t_end: datetime
    ) -> pd.DataFrame:
        """Download CSV from ERDDAP griddap and parse into DataFrame."""
        url = self._build_url(lat, lon, t_start, t_end)
        logger.debug(f"ERDDAP URL: {url}")

        try:
            resp = self.session.get(url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"ERDDAP request failed: {e}")
            return pd.DataFrame()

        # ERDDAP CSV has a units row after the header
        lines = resp.text.strip().split("\n")
        if len(lines) < 3:
            logger.warning("ERDDAP returned fewer than 3 lines (header + units + data)")
            return pd.DataFrame()

        # Remove the units row (second line)
        csv_text = lines[0] + "\n" + "\n".join(lines[2:])

        df = pd.read_csv(StringIO(csv_text))

        # Parse time
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        # Replace missing values
        if VARIABLE in df.columns:
            df.loc[df[VARIABLE] <= MISSING_VALUE + 1, VARIABLE] = np.nan

        return df

    def _filter_by_azimuth(self, df: pd.DataFrame, lat: float, lon: float,
                           az_min: float, az_max: float,
                           max_distance_km: float) -> pd.DataFrame:
        """Filter grid cells to those within an azimuth wedge from the station.

        Computes azimuth and distance from the station to each grid cell,
        keeps only cells within [az_min, az_max] and within max_distance_km.
        """
        if df.empty or "latitude" not in df.columns:
            return df

        dlat = df["latitude"] - lat
        dlon = (df["longitude"] - lon) * np.cos(np.radians(lat))
        az = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360
        dist_km = np.sqrt((dlat * 111.0)**2 + (dlon * 111.0)**2)

        if az_min <= az_max:
            mask = (az >= az_min) & (az <= az_max)
        else:
            # Wraps around 360, e.g., 350-10
            mask = (az >= az_min) | (az <= az_max)

        mask &= (dist_km <= max_distance_km)
        return df[mask]

    def _aggregate_daily(self, df: pd.DataFrame,
                         station_lat: float = None, station_lon: float = None,
                         az_min: float = None, az_max: float = None,
                         max_distance_km: float = None) -> pd.DataFrame:
        """Average ice concentration over the spatial box for each timestamp.

        Each ERDDAP time step has multiple rows (one per grid cell).
        We take the spatial mean, excluding NaN/missing cells.

        If az_min/az_max/max_distance_km are provided, cells are filtered
        to the azimuth wedge before averaging.
        """
        if df.empty:
            return pd.DataFrame(columns=["datetime", "ice_concentration", "n_cells"])

        do_az_filter = (az_min is not None and az_max is not None
                        and max_distance_km is not None
                        and station_lat is not None)

        if do_az_filter:
            # Filter per timestep (same cell set each day, but be safe)
            results = []
            for t, grp in df.groupby("time"):
                filtered = self._filter_by_azimuth(
                    grp, station_lat, station_lon,
                    az_min, az_max, max_distance_km)
                valid = filtered[VARIABLE].dropna()
                if len(valid) > 0:
                    results.append({
                        "datetime": t,
                        "ice_concentration": round(valid.mean(), 1),
                        "n_cells": len(valid),
                    })
            if not results:
                return pd.DataFrame(columns=["datetime", "ice_concentration", "n_cells"])
            grouped = pd.DataFrame(results)
        else:
            grouped = df.groupby("time")[VARIABLE].agg(["mean", "count"]).reset_index()
            grouped.columns = ["datetime", "ice_concentration", "n_cells"]
            grouped["ice_concentration"] = grouped["ice_concentration"].round(1)

        grouped = grouped.sort_values("datetime").reset_index(drop=True)
        return grouped


def get_great_lakes_stations() -> dict:
    """Return dict of stations that fall within GLERL coverage, with their lake."""
    config_path = PROJECT_ROOT / "config" / "stations_config.json"
    with open(config_path) as f:
        all_stations = json.load(f)

    # Approximate lake bounding boxes (lat_min, lat_max, lon_min, lon_max)
    lakes = {
        "Superior": (46.4, 49.1, -92.2, -84.3),
        "Michigan": (41.6, 46.1, -87.8, -84.7),
        "Huron": (43.0, 46.3, -84.8, -79.7),
        "Georgian Bay": (44.5, 45.9, -81.8, -79.7),
        "Erie": (41.3, 42.9, -83.5, -78.8),
        "Ontario": (43.2, 44.3, -79.9, -75.9),
    }

    result = {}
    client = GLERLIceClient()

    for station, cfg in all_stations.items():
        lat = cfg.get("latitude_deg", 0)
        lon = cfg.get("longitude_deg", 0)

        if not client.station_in_coverage(station):
            continue

        # Identify which lake
        lake = "Unknown"
        for lake_name, (la_min, la_max, lo_min, lo_max) in lakes.items():
            if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
                lake = lake_name
                break

        result[station] = {
            "latitude": lat,
            "longitude": lon,
            "lake": lake,
            "data_availability": cfg.get("data_availability", "unknown"),
        }

    return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch GLERL ice concentration for a station")
    parser.add_argument("--station", help="Station name (e.g., ROSS)")
    parser.add_argument("--year", type=int, help="Winter year (e.g., 2023 = winter 2022-2023)")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_DEG,
                        help=f"Bounding box half-width in degrees (default: {DEFAULT_RADIUS_DEG})")
    parser.add_argument("--list-stations", action="store_true",
                        help="List all Great Lakes stations and exit")
    args = parser.parse_args()

    if args.list_stations or (not args.station and not args.year):
        stations = get_great_lakes_stations()
        print(f"\nGreat Lakes stations in config ({len(stations)}):\n")
        for name, info in sorted(stations.items()):
            print(f"  {name:6s}  {info['latitude']:7.3f}°N  {info['longitude']:8.3f}°W  "
                  f"{info['lake']:14s}  data: {info['data_availability']}")
        print()
    elif args.station and args.year:
        client = GLERLIceClient(radius_deg=args.radius)
        df = client.get_ice_concentration(args.station, args.year)

        if df.empty:
            print(f"No ice data for {args.station} winter {args.year}")
        else:
            print(f"\n{args.station} ice concentration, winter {args.year}:")
            print(f"  Days with data: {len(df)}")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"  Max ice: {df['ice_concentration'].max():.1f}%")
            print(f"  Mean ice (when >0): {df.loc[df['ice_concentration'] > 0, 'ice_concentration'].mean():.1f}%")
            ice_days = (df["ice_concentration"] > 5).sum()
            print(f"  Days with >5% ice: {ice_days}")
            print(f"\nSample rows:")
            print(df.to_string(index=False, max_rows=20))
