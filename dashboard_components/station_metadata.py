# ABOUTME: Provides station metadata loading from config for dashboard components.
# ABOUTME: Single source of truth for antenna heights, reference sources, and station info.

"""
Station Metadata Helper for Enhanced GNSS-IR Dashboard

Loads station configuration from stations_config.json and provides
helper functions for accessing antenna heights, reference sources,
and other station-specific metadata.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

# Project root for config loading
PROJECT_ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def _load_stations_config() -> Dict[str, Any]:
    """Load and cache the stations configuration file."""
    config_path = PROJECT_ROOT / "config" / "stations_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_station_config(station_id: str) -> Optional[Dict[str, Any]]:
    """Get full configuration for a station."""
    config = _load_stations_config()
    return config.get(station_id.upper())


def get_antenna_height(station_id: str) -> float:
    """
    Get antenna ellipsoidal height for a station.

    Args:
        station_id: Station identifier (e.g., 'FORA', 'VALR')

    Returns:
        Antenna height in meters, or 0.0 if not found
    """
    config = get_station_config(station_id)
    if config:
        return config.get('ellipsoidal_height_m', 0.0)
    return 0.0


def get_reference_source_info(station_id: str) -> Dict[str, Any]:
    """
    Determine the primary reference source for a station.

    Returns info about whether to use ERDDAP, USGS, or CO-OPS as primary reference,
    based on configuration. Priority order: ERDDAP > USGS > CO-OPS.

    Args:
        station_id: Station identifier

    Returns:
        Dict with keys:
            - primary_source: 'ERDDAP', 'USGS', or 'CO-OPS'
            - station_id: Reference station ID
            - station_name: Human-readable name
            - distance_km: Distance to reference (if known)
            - notes: Any relevant notes
    """
    config = get_station_config(station_id)
    if not config:
        return {
            'primary_source': 'USGS',
            'station_id': None,
            'station_name': 'Unknown',
            'distance_km': None,
            'notes': 'Station not found in config'
        }

    # Check for ERDDAP configuration first (highest priority)
    erddap_config = config.get('external_data_sources', {}).get('erddap', {})
    if erddap_config.get('enabled') and erddap_config.get('primary_reference'):
        return {
            'primary_source': 'ERDDAP',
            'station_id': erddap_config.get('dataset_id', 'unknown'),
            'station_name': erddap_config.get('station_name', 'ERDDAP Station'),
            'distance_km': erddap_config.get('distance_km'),
            'notes': erddap_config.get('notes', 'ERDDAP water level station')
        }

    # Check for CO-OPS configuration (second priority after ERDDAP)
    coops_config = config.get('external_data_sources', {}).get('noaa_coops', {})
    if coops_config.get('enabled') and coops_config.get('primary_reference'):
        coops_stations = coops_config.get('preferred_stations', [])
        coops_nearest = coops_config.get('nearest_station', {})
        coops_id = coops_stations[0] if coops_stations else coops_nearest.get('id')
        # Try to get distance and name from config
        distance = coops_config.get('distance_km') or coops_nearest.get('distance_km')
        station_name = coops_config.get('station_name') or _get_coops_station_name(coops_id)
        if not distance:
            distance = _calculate_coops_distance(station_id, coops_id)
        return {
            'primary_source': 'CO-OPS',
            'station_id': coops_id,
            'station_name': station_name,
            'distance_km': distance,
            'notes': coops_config.get('notes', 'CO-OPS tide gauge reference')
        }

    # Check for USGS configuration
    usgs_config = config.get('usgs_comparison', {})
    usgs_site = usgs_config.get('target_usgs_site')
    usgs_notes = usgs_config.get('notes', '')

    # Default to USGS
    if usgs_site:
        return {
            'primary_source': 'USGS',
            'station_id': usgs_site,
            'station_name': _extract_usgs_name_from_notes(usgs_notes),
            'distance_km': _extract_distance_from_notes(usgs_notes),
            'notes': usgs_notes
        }

    # Fallback
    return {
        'primary_source': 'Unknown',
        'station_id': None,
        'station_name': 'No reference configured',
        'distance_km': None,
        'notes': 'No USGS or CO-OPS reference configured'
    }


def _get_coops_station_name(station_id: str) -> str:
    """Get human-readable name for CO-OPS station."""
    # Common CO-OPS stations used in this project
    coops_names = {
        '1612340': 'Honolulu Harbor, HI',
        '1612401': 'Pearl Harbor, HI',
        '1611400': 'Nawiliwili, HI',
        '9452634': 'Elfin Cove, AK',
        '9452210': 'Juneau, AK',
        '8651370': 'Duck, NC',
        '8652587': 'Oregon Inlet Marina, NC',
    }
    return coops_names.get(station_id, f'CO-OPS {station_id}')


def _extract_usgs_name_from_notes(notes: str) -> str:
    """Extract USGS station name from notes field."""
    if not notes:
        return 'USGS Gauge'
    # Notes format: "Using STATION NAME (ID) at Xkm..."
    if 'Using ' in notes and '(' in notes:
        start = notes.find('Using ') + 6
        end = notes.find('(')
        if end > start:
            return notes[start:end].strip()
    return 'USGS Gauge'


def _extract_distance_from_notes(notes: str) -> Optional[float]:
    """Extract distance in km from notes field."""
    if not notes:
        return None
    # Notes format: "...at X.Xkm..."
    import re
    match = re.search(r'at\s+(\d+\.?\d*)\s*km', notes, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _calculate_coops_distance(gnss_station_id: str, coops_station_id: str) -> Optional[float]:
    """Calculate distance between GNSS station and CO-OPS station.

    Uses Haversine formula to calculate great-circle distance.
    """
    import math

    # GNSS station coordinates
    config = get_station_config(gnss_station_id)
    if not config:
        return None

    gnss_lat = config.get('latitude_deg')
    gnss_lon = config.get('longitude_deg')
    if gnss_lat is None or gnss_lon is None:
        return None

    # CO-OPS station coordinates (approximate)
    coops_coords = {
        '1612340': (21.3067, -157.867),   # Honolulu Harbor, HI
        '1612401': (21.3642, -157.9589),  # Pearl Harbor, HI
        '1611400': (21.9544, -159.356),   # Nawiliwili, HI
        '9452634': (58.195, -136.347),    # Elfin Cove, AK
        '9452210': (58.298, -134.412),    # Juneau, AK
        '8651370': (36.183, -75.747),     # Duck, NC
        '8652587': (35.795, -75.548),     # Oregon Inlet Marina, NC
        '8658163': (33.908, -78.016),     # Wrightsville Beach, NC
    }

    if coops_station_id not in coops_coords:
        return None

    coops_lat, coops_lon = coops_coords[coops_station_id]

    # Haversine formula
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(gnss_lat), math.radians(gnss_lon)
    lat2, lon2 = math.radians(coops_lat), math.radians(coops_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return round(R * c, 1)


def get_station_coordinates(station_id: str) -> Tuple[Optional[float], Optional[float]]:
    """Get station latitude and longitude."""
    config = get_station_config(station_id)
    if config:
        return (
            config.get('latitude_deg'),
            config.get('longitude_deg')
        )
    return None, None


def get_all_station_ids() -> list:
    """Get list of all configured station IDs."""
    config = _load_stations_config()
    return list(config.keys())


def get_station_display_info(station_id: str) -> Dict[str, Any]:
    """
    Get comprehensive display information for a station.

    Useful for dashboard overview panels.
    """
    config = get_station_config(station_id)
    if not config:
        return {'error': f'Station {station_id} not found'}

    ref_info = get_reference_source_info(station_id)
    lat, lon = get_station_coordinates(station_id)

    return {
        'station_id': station_id,
        'latitude': lat,
        'longitude': lon,
        'antenna_height_m': get_antenna_height(station_id),
        'primary_reference': ref_info['primary_source'],
        'reference_station_id': ref_info['station_id'],
        'reference_station_name': ref_info['station_name'],
        'reference_distance_km': ref_info['distance_km'],
        'gnssir_params_path': config.get('gnssir_json_params_path'),
        'has_coops': config.get('external_data_sources', {}).get('noaa_coops', {}).get('enabled', False),
    }


# Export all functions
__all__ = [
    'get_station_config',
    'get_antenna_height',
    'get_reference_source_info',
    'get_station_coordinates',
    'get_all_station_ids',
    'get_station_display_info',
]
