"""
USGS gauge finder module for GNSS-IR project.
This module handles finding USGS gauges based on station configuration.
"""

import logging
import json
from pathlib import Path

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

def find_usgs_gauge_for_station(station_name):
    """
    Find the best USGS gauge for a given station based on configuration.
    
    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
    
    Returns:
        str or None: USGS site code if found, None otherwise
    """
    # Load station configuration
    stations_config = load_config(STATIONS_CONFIG_PATH)
    if stations_config is None:
        logging.error(f"Failed to load stations configuration")
        return None
    
    station_config = stations_config.get(station_name)
    if station_config is None:
        logging.error(f"Station {station_name} not found in configuration")
        return None
    
    # Check if station config has USGS comparison settings
    if 'usgs_comparison' not in station_config:
        logging.error(f"Station configuration does not have usgs_comparison section")
        logging.error(f"Available sections: {list(station_config.keys())}")
        return None
    
    usgs_config = station_config.get('usgs_comparison', {})
    
    # Check if a specific gauge is already configured
    target_site = usgs_config.get('target_usgs_site')
    
    if target_site:
        logging.info(f"Using pre-configured USGS gauge: {target_site}")
        return target_site
    
    logging.warning(f"No pre-configured USGS gauge found for {station_name}")
    return None

def get_usgs_parameter_code(station_name):
    """
    Get the USGS parameter code for a given station.
    
    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
    
    Returns:
        str or None: USGS parameter code if found, None otherwise
    """
    # Load station configuration
    stations_config = load_config(STATIONS_CONFIG_PATH)
    if stations_config is None:
        logging.error(f"Failed to load stations configuration")
        return None
    
    station_config = stations_config.get(station_name)
    if station_config is None:
        logging.error(f"Station {station_name} not found in configuration")
        return None
    
    # Check if station config has USGS comparison settings
    if 'usgs_comparison' not in station_config:
        logging.error(f"Station configuration does not have usgs_comparison section")
        return None
    
    usgs_config = station_config.get('usgs_comparison', {})
    
    # Get USGS parameter code from config
    parameter_code = usgs_config.get('usgs_parameter_code_to_use')
    if parameter_code is None:
        logging.warning(f"No parameter code specified for {station_name}, defaulting to 00065 (gage height)")
        parameter_code = "00065"
    
    return parameter_code

def get_station_config(station_name):
    """
    Get the full station configuration.
    
    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
    
    Returns:
        dict or None: Station configuration if found, None otherwise
    """
    # Load station configuration
    stations_config = load_config(STATIONS_CONFIG_PATH)
    if stations_config is None:
        logging.error(f"Failed to load stations configuration")
        return None
    
    station_config = stations_config.get(station_name)
    if station_config is None:
        logging.error(f"Station {station_name} not found in configuration")
        return None
    
    return station_config
