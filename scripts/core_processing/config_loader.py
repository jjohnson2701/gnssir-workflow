# ABOUTME: Configuration loading module for GNSS-IR processing
# ABOUTME: Provides functions to load station configs and tool paths from JSON files

import json
import logging
import sys
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str or Path): Path to the JSON configuration file

    Returns:
        dict or None: Configuration dictionary, or None if loading failed
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        return None


def load_tool_paths(tool_paths_config_path, project_root):
    """
    Load tool paths configuration from JSON file with defaults for missing entries.

    Args:
        tool_paths_config_path (str or Path): Path to the tool paths JSON configuration file
        project_root (Path): Project root directory path for resolving relative paths

    Returns:
        dict: Tool paths dictionary with defaults for missing entries
    """
    # Define default tool paths
    default_tool_paths = {
        "gfzrnx_path": "gfzrnx",
        "rinex2snr_path": "rinex2snr",
        "gnssir_path": "gnssir",
        "quicklook_path": "quickLook",
    }

    # Convert to Path object if string is provided
    if isinstance(tool_paths_config_path, str):
        tool_paths_config_path = Path(tool_paths_config_path)

    # Resolve relative path if needed
    if not tool_paths_config_path.is_absolute():
        tool_paths_config_path = project_root / tool_paths_config_path

    # Load the configuration
    logging.info(f"Loading tool paths from {tool_paths_config_path}")
    tool_paths = load_config(tool_paths_config_path)

    # If loading failed, use defaults and log warning
    if tool_paths is None:
        logging.warning(f"Failed to load tool paths from {tool_paths_config_path}. Using defaults.")
        tool_paths = {}

    # Fill in missing entries with defaults
    for tool, default_path in default_tool_paths.items():
        if tool not in tool_paths:
            tool_paths[tool] = default_path
            logging.warning(
                f"Tool path for {tool} not found in config. Using default: {default_path}"
            )

    # Log all the tool paths being used
    for tool, path in tool_paths.items():
        logging.info(f"Tool path: {tool} = {path}")

    return tool_paths


def load_station_config(stations_config_path, station_id_uppercase, project_root):
    """
    Load station-specific configuration from stations_config.json.

    Args:
        stations_config_path (str or Path): Path to the stations configuration file
        station_id_uppercase (str): Station ID in uppercase (e.g., "FORA")
        project_root (Path): Project root directory path for resolving relative paths

    Returns:
        dict: Station-specific configuration dictionary

    Raises:
        SystemExit: If the station configuration is not found
    """
    # Convert to Path object if string is provided
    if isinstance(stations_config_path, str):
        stations_config_path = Path(stations_config_path)

    # Resolve relative path if needed
    if not stations_config_path.is_absolute():
        stations_config_path = project_root / stations_config_path

    # Load the configuration
    logging.info(f"Loading stations configuration from {stations_config_path}")
    stations_config = load_config(stations_config_path)

    # Check if configuration was loaded successfully
    if stations_config is None:
        logging.error(f"Failed to load stations configuration from {stations_config_path}")
        sys.exit(1)

    # Get station-specific configuration
    station_config = stations_config.get(station_id_uppercase)
    if station_config is None:
        logging.error(
            f"Station configuration for {station_id_uppercase} not found in {stations_config_path}"
        )
        sys.exit(1)

    logging.info(f"Successfully loaded configuration for station {station_id_uppercase}")
    return station_config
