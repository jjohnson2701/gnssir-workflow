# ABOUTME: Workspace setup module for gnssrefl directory structure
# ABOUTME: Creates REFL_CODE and ORBITS directories required by gnssrefl tools

import logging
import shutil
import sqlite3
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
        "refl_code_base": refl_code_base,
        "orbits_base": orbits_base,
        "refl_code_input": refl_code_base / "input",
        "refl_code_year": refl_code_year,
        "refl_code_rinex": refl_code_year / "rinex" / station_id,
        "refl_code_snr": refl_code_year / "snr" / station_id,
        "refl_code_results": refl_code_year / "results" / station_id,
        "refl_code_plots": refl_code_year / "plots" / station_id,
        "orbits_sp3": orbits_year / "sp3",
        "orbits_nav": orbits_year / "nav",
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
    paths["refl_code_logs"] = logs_dir

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


def register_station_coordinates(
    station_id: str, lat: float, lon: float, ht: float, refl_code_base: Path
) -> bool:
    """
    Register station coordinates in the gnssrefl station database.

    This adds the station to the station_pos_2024.db database so that gnssrefl's
    quickLook tool can find the coordinates without showing a warning.

    Args:
        station_id: Station ID in 4-character lowercase
        lat: Latitude in degrees
        lon: Longitude in degrees
        ht: Ellipsoidal height in meters
        refl_code_base: Base directory for REFL_CODE

    Returns:
        True if successful, False otherwise
    """
    db_path = Path(refl_code_base) / "Files" / "station_pos_2024.db"

    if not db_path.exists():
        logging.warning(f"Station database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if station already exists
        cursor.execute("SELECT lat, lon, ht FROM stations WHERE station = ?", (station_id.lower(),))
        existing = cursor.fetchone()

        if existing:
            logging.debug(
                f"Station {station_id} already in database: "
                f"lat={existing[0]}, lon={existing[1]}, ht={existing[2]}"
            )
        else:
            # Insert the station
            cursor.execute(
                "INSERT INTO stations (station, lat, lon, ht) VALUES (?, ?, ?, ?)",
                (station_id.lower(), lat, lon, ht),
            )
            conn.commit()
            logging.info(
                f"Registered station {station_id} in gnssrefl database: "
                f"lat={lat}, lon={lon}, ht={ht}"
            )

        conn.close()
        return True

    except sqlite3.Error as e:
        logging.error(f"Database error registering station {station_id}: {e}")
        return False
