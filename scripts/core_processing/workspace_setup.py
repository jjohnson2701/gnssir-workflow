"""
Workspace setup module for GNSS-IR processing.
Provides functions to setup the gnssrefl workspace directories.
"""

import logging
import shutil
from pathlib import Path

def setup_gnssrefl_workspace(station_id, year, refl_code_base, orbits_base, doy=None):
    """
    Setup the gnssrefl workspace with the necessary directory structure.
    
    Args:
        station_id (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        refl_code_base (Path): Base directory for REFL_CODE
        orbits_base (Path): Base directory for ORBITS
        doy (int, optional): Day of year. If provided, creates day-specific directories.
    
    Returns:
        dict: Dictionary of paths created for the station and year
    """
    # Ensure paths are Path objects
    refl_code_base = Path(refl_code_base)
    orbits_base = Path(orbits_base)
    
    # Create base directories
    refl_code_base.mkdir(parents=True, exist_ok=True)
    orbits_base.mkdir(parents=True, exist_ok=True)
    
    # Create year directories
    year_str = str(year)
    refl_code_year = refl_code_base / year_str
    orbits_year = orbits_base / year_str
    
    # Create required subdirectories
    paths = {
        'refl_code_base': refl_code_base,
        'orbits_base': orbits_base,
        'refl_code_input': refl_code_base / "input",
        'refl_code_year': refl_code_year,
        'refl_code_rinex': refl_code_year / "rinex" / station_id,
        'refl_code_snr': refl_code_year / "snr" / station_id,
        'refl_code_results': refl_code_year / "results" / station_id,
        'refl_code_plots': refl_code_year / "plots" / station_id,
        'orbits_sp3': orbits_year / "sp3",
        'orbits_nav': orbits_year / "nav"
    }
    
    # Create all directories
    for path_name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Created directory: {path}")
    
    # Create the Files directory for quickLook plots
    files_dir = refl_code_base / "Files"
    files_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the station-specific directory in Files for quickLook
    station_files_dir = files_dir / station_id
    station_files_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    logs_dir = refl_code_base / "logs" / station_id / year_str
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Add logs directory to the paths dictionary
    paths['refl_code_logs'] = logs_dir
    
    logging.info(f"GNSS-IR workspace setup complete for station {station_id}, year {year}")
    return paths

def copy_json_params(json_source_path, station_id, refl_code_base):
    """
    Copy the station's JSON parameter file to the gnssrefl workspace.
    
    Args:
        json_source_path (str or Path): Source path to the JSON parameter file
        station_id (str): Station ID in 4-character lowercase
        refl_code_base (Path): Base directory for REFL_CODE
    
    Returns:
        Path: Path to the copied JSON file, or None if copying failed
    """
    # Ensure paths are Path objects
    json_source_path = Path(json_source_path)
    refl_code_base = Path(refl_code_base)
    
    # Ensure source file exists
    if not json_source_path.exists():
        logging.error(f"JSON parameter file not found at {json_source_path}")
        return None
    
    # Define target directory and path
    target_dir = refl_code_base / "input"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / f"{station_id}.json"
    
    try:
        shutil.copy2(json_source_path, target_path)
        logging.info(f"Copied JSON parameter file from {json_source_path} to {target_path}")
        return target_path
    except Exception as e:
        logging.error(f"Failed to copy JSON parameter file: {e}")
        return None
