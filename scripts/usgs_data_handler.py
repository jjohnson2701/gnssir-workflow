# ABOUTME: USGS gauge discovery, configuration lookup, and water level data retrieval
# ABOUTME: Consolidates gauge finding, progressive search, and data fetching into one module

import json
import logging
import pandas as pd
from pathlib import Path
from dataretrieval import nwis
from datetime import datetime, timedelta

from utils.geo_utils import haversine_distance, get_bounding_box

# Define project root and config path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"


# =============================================================================
# Configuration Loading Functions (from usgs_gauge_finder.py)
# =============================================================================


def load_config(config_path=None):
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = STATIONS_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        return None


def get_station_config(station_name):
    """Get the full station configuration."""
    stations_config = load_config()
    if stations_config is None:
        return None

    station_config = stations_config.get(station_name)
    if station_config is None:
        logging.error(f"Station {station_name} not found in configuration")
        return None

    return station_config


def find_usgs_gauge_for_station(station_name):
    """Find the configured USGS gauge for a station."""
    station_config = get_station_config(station_name)
    if station_config is None:
        return None

    usgs_config = station_config.get("usgs_comparison", {})
    target_site = usgs_config.get("target_usgs_site")

    if target_site:
        logging.info(f"Using pre-configured USGS gauge: {target_site}")
        return target_site

    logging.warning(f"No pre-configured USGS gauge found for {station_name}")
    return None


def get_usgs_parameter_code(station_name):
    """Get the USGS parameter code for a station."""
    station_config = get_station_config(station_name)
    if station_config is None:
        return "00065"

    usgs_config = station_config.get("usgs_comparison", {})
    parameter_code = usgs_config.get("usgs_parameter_code_to_use", "00065")
    return parameter_code


# =============================================================================
# Progressive Search Functions (from usgs_progressive_search.py)
# =============================================================================


def get_state_from_coordinates(lat, lon):
    """Determine the most likely US state based on coordinates."""
    state_bounds = {
        "NC": (33.5, 36.6, -84.4, -75.2),
        "SC": (32.0, 35.3, -83.4, -78.5),
        "FL": (24.5, 31.0, -87.6, -80.0),
        "CA": (32.5, 42.0, -124.5, -114.0),
        "WA": (45.5, 49.0, -124.8, -116.9),
        "OR": (42.0, 46.3, -124.7, -116.4),
        "TX": (25.8, 36.5, -106.7, -93.5),
        "LA": (29.0, 33.0, -94.1, -88.8),
        "MS": (30.0, 35.0, -92.0, -88.0),
        "AL": (30.1, 35.0, -88.5, -84.9),
        "GA": (30.4, 35.0, -85.6, -80.8),
        "VA": (36.5, 39.5, -83.7, -75.2),
        "MD": (37.9, 39.8, -79.5, -75.0),
        "DE": (38.4, 39.8, -75.8, -74.9),
        "NJ": (38.9, 41.4, -75.6, -73.9),
        "NY": (40.5, 45.0, -79.8, -71.8),
        "CT": (40.9, 42.1, -73.7, -71.8),
        "RI": (41.1, 42.0, -71.9, -71.1),
        "MA": (41.2, 42.9, -73.5, -69.9),
        "NH": (42.7, 45.3, -72.6, -70.7),
        "ME": (43.1, 47.5, -71.1, -66.9),
        "HI": (18.5, 22.5, -160.5, -154.5),
        "AK": (51.0, 71.5, -180.0, -130.0),
    }

    for state, (min_lat, max_lat, min_lon, max_lon) in state_bounds.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return state

    if lon < -100:
        return "WA" if lat > 40 else "CA"
    else:
        return "NY" if lat > 40 else "FL"


def construct_bbox_string(min_lat, max_lat, min_lon, max_lon):
    """Construct bounding box string for USGS API."""
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def progressive_gauge_search(
    gnss_station_lat,
    gnss_station_lon,
    initial_radius_km,
    radius_increment_km,
    max_radius_km,
    desired_parameter_codes,
    min_gauges_found=3,
    data_start_date=None,
    data_end_date=None,
):
    """
    Perform progressive search for USGS gauges, expanding radius until
    minimum gauges found or maximum radius reached.
    """
    logging.info(f"Starting progressive gauge search from ({gnss_station_lat}, {gnss_station_lon})")

    current_radius = initial_radius_km
    all_gauges = []
    matching_gauges = []

    while current_radius <= max_radius_km:
        logging.info(f"Searching with radius {current_radius} km")

        try:
            min_lat, max_lat, min_lon, max_lon = get_bounding_box(
                gnss_station_lat, gnss_station_lon, current_radius
            )
            bbox_str = construct_bbox_string(min_lat, max_lat, min_lon, max_lon)

            site_types = ["ST", "LK", "ES", "OC"]
            parameter_codes_str = ",".join(desired_parameter_codes)
            gauges_this_radius = []

            try:
                sites_df = nwis.get_record(
                    service="site",
                    bBox=bbox_str,
                    parameterCd=parameter_codes_str,
                    siteType=",".join(site_types),
                )

                if sites_df is not None and not sites_df.empty:
                    for site_no, site in sites_df.iterrows():
                        try:
                            site_code = site.get("site_no", site_no)
                            site_lat = float(site.get("dec_lat_va", 0))
                            site_lon = float(site.get("dec_long_va", 0))

                            distance = haversine_distance(
                                gnss_station_lat, gnss_station_lon, site_lat, site_lon
                            )

                            if distance <= current_radius:
                                gauges_this_radius.append(
                                    {
                                        "site_no": site_code,
                                        "station_nm": site.get("station_nm", ""),
                                        "dec_lat_va": site_lat,
                                        "dec_long_va": site_lon,
                                        "distance_km": distance,
                                        "site_tp_cd": site.get("site_tp_cd", ""),
                                        "alt_datum_cd": site.get("alt_datum_cd", "Unknown"),
                                        "matched_parameters": [],
                                    }
                                )
                        except Exception as e:
                            logging.warning(f"Error processing site: {e}")

            except Exception as e:
                logging.warning(f"Error querying sites: {e}")

            all_gauges.extend(gauges_this_radius)

            site_codes_seen = set()
            matching_gauges = []
            for gauge in sorted(all_gauges, key=lambda g: g["distance_km"]):
                if gauge["site_no"] not in site_codes_seen:
                    matching_gauges.append(gauge)
                    site_codes_seen.add(gauge["site_no"])

            if len(matching_gauges) >= min_gauges_found:
                break

            current_radius += radius_increment_km

        except Exception as e:
            logging.error(f"Error in search iteration: {e}")
            current_radius += radius_increment_km

    if matching_gauges:
        return pd.DataFrame(matching_gauges)
    return pd.DataFrame()


# =============================================================================
# Gauge Discovery Functions
# =============================================================================


def find_nearby_usgs_gauges(
    gnss_station_lat,
    gnss_station_lon,
    radius_km=50.0,
    desired_parameter_codes=None,
    state_code=None,
    huc=None,
):
    """Find nearby USGS gauges within a specified radius of a GNSS station."""
    logging.info(f"Finding nearby USGS gauges for ({gnss_station_lat:.6f}, {gnss_station_lon:.6f})")

    if desired_parameter_codes is None:
        desired_parameter_codes = ["62610", "62611", "62620", "00065"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    gauges_df = progressive_gauge_search(
        gnss_station_lat=gnss_station_lat,
        gnss_station_lon=gnss_station_lon,
        initial_radius_km=radius_km,
        radius_increment_km=radius_km,
        max_radius_km=radius_km,
        desired_parameter_codes=desired_parameter_codes,
        min_gauges_found=1,
        data_start_date=start_date,
        data_end_date=end_date,
    )

    if gauges_df is not None and not gauges_df.empty:
        column_mapping = {
            "site_no": "site_code",
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "site_tp_cd": "site_type",
            "alt_datum_cd": "vertical_datum",
        }
        for old_col, new_col in column_mapping.items():
            if old_col in gauges_df.columns:
                gauges_df[new_col] = gauges_df[old_col]

        logging.info(f"Found {len(gauges_df)} USGS gauges within {radius_km} km")
        return gauges_df

    logging.warning(f"No USGS gauges found within {radius_km} km")
    return pd.DataFrame()


# =============================================================================
# Data Fetching Functions
# =============================================================================


def fetch_usgs_gauge_data(
    usgs_site_code,
    parameter_code=None,
    start_date_str=None,
    end_date_str=None,
    service="iv",
    priority_parameter_codes=None,
):
    """Fetch water level data for a specific USGS gauge."""
    logging.info(f"Fetching USGS gauge data for site {usgs_site_code}")

    if priority_parameter_codes is None:
        if parameter_code is not None:
            priority_parameter_codes = [parameter_code, "00065", "62610", "62611", "62620"]
        else:
            priority_parameter_codes = ["00065", "62610", "62611", "62620"]
    elif parameter_code is not None and parameter_code not in priority_parameter_codes:
        priority_parameter_codes = [parameter_code] + priority_parameter_codes

    for param_code in priority_parameter_codes:
        # Try instantaneous values first
        try:
            site_data = nwis.get_record(
                sites=usgs_site_code,
                service="iv",
                parameterCd=param_code,
                start=start_date_str,
                end=end_date_str,
            )

            if site_data is not None and not site_data.empty:
                metadata = _create_metadata(usgs_site_code, param_code, "iv")
                metadata = enhance_gauge_metadata(usgs_site_code, param_code, site_data, metadata)
                return site_data, metadata, param_code
        except Exception as e:
            logging.warning(f"Error fetching iv data for {param_code}: {e}")

        # Try daily values
        try:
            site_data = nwis.get_record(
                sites=usgs_site_code,
                service="dv",
                parameterCd=param_code,
                start=start_date_str,
                end=end_date_str,
                statCd="00003",
            )

            if site_data is not None and not site_data.empty:
                metadata = _create_metadata(usgs_site_code, param_code, "dv")
                metadata = enhance_gauge_metadata(usgs_site_code, param_code, site_data, metadata)
                return site_data, metadata, param_code
        except Exception as e:
            logging.warning(f"Error fetching dv data for {param_code}: {e}")

    logging.error(f"No data found for site {usgs_site_code}")
    return pd.DataFrame(), {}, None


def _create_metadata(site_code, param_code, service):
    """Create initial metadata dictionary."""
    return {
        "site_code": site_code,
        "parameter_code": param_code,
        "site_name": "Unknown",
        "parameter_desc": "Unknown",
        "units": "Unknown",
        "datum": "Unknown",
        "service": service,
    }


def enhance_gauge_metadata(usgs_site_code, parameter_code, site_data, metadata):
    """Enhance gauge metadata with site information."""
    try:
        site_info = nwis.get_record(sites=usgs_site_code, service="site")

        if site_info is not None and not site_info.empty:
            row = site_info.iloc[0]
            if "station_nm" in site_info.columns:
                metadata["site_name"] = row["station_nm"]
            if "alt_datum_cd" in site_info.columns:
                metadata["datum"] = row["alt_datum_cd"]
            if "dec_lat_va" in site_info.columns:
                metadata["latitude"] = row["dec_lat_va"]
            if "dec_long_va" in site_info.columns:
                metadata["longitude"] = row["dec_long_va"]
    except Exception as e:
        logging.warning(f"Error retrieving site metadata: {e}")

    # Apply parameter-specific defaults
    param_descriptions = {
        "00065": "Gage height",
        "62610": "Water level, elevation, NAVD88",
        "62611": "Water level, elevation, NGVD29",
        "62620": "Water level, elevation, MSL",
    }

    if parameter_code in param_descriptions:
        metadata["parameter_desc"] = param_descriptions[parameter_code]
        metadata["units"] = "ft"

        if parameter_code == "62610":
            metadata["datum"] = "NAVD88"
        elif parameter_code == "62611":
            metadata["datum"] = "NGVD29"
        elif parameter_code == "62620":
            metadata["datum"] = "MSL"

    if not metadata.get("site_name"):
        metadata["site_name"] = f"USGS Site {usgs_site_code}"

    return metadata


def process_usgs_data(usgs_df, usgs_metadata, convert_to_meters=True):
    """Process USGS gauge data for comparison with GNSS-IR data."""
    if usgs_df.empty:
        return pd.DataFrame()

    df = usgs_df.copy()
    param_code = usgs_metadata.get("parameter_code", None)

    # Find value column
    if param_code and param_code in df.columns:
        df["usgs_value"] = df[param_code].astype(float)
    elif "usgs_value" not in df.columns:
        value_cols = [c for c in df.columns if "value" in str(c).lower() and not c.endswith("_cd")]
        if value_cols:
            df["usgs_value"] = df[value_cols[0]].astype(float)
        else:
            logging.warning("Could not identify data column in USGS data")
            return pd.DataFrame()

    # Convert to meters
    if convert_to_meters and "usgs_value" in df.columns:
        df["usgs_value_m"] = df["usgs_value"] * 0.3048

    # Aggregate to daily
    if "usgs_value_m" in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df["date"] = df.index.date
            elif "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date

            daily_stats = df.groupby("date").agg(
                {"usgs_value_m": ["count", "mean", "median", "std", "min", "max"]}
            )
            daily_stats.columns = ["_".join(col).strip() for col in daily_stats.columns.values]
            daily_stats = daily_stats.reset_index()
            daily_stats["datetime"] = pd.to_datetime(daily_stats["date"])

            return daily_stats
        except Exception as e:
            logging.error(f"Error creating daily statistics: {e}")

    return df
