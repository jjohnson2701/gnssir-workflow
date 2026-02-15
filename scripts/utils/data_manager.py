# ABOUTME: Data manager for RINEX file acquisition from NPS and EarthScope archives
# ABOUTME: Handles downloads and file validation for multiple data sources

import logging
import os
import requests
from pathlib import Path

GNSS_BASE_URL = "https://gnss.nps.gov/doi-gnss"
RINEX_PATH_PATTERN = "Rinex/{year}/{doy:03d}/{station}/{station}{doy:03d}0.{yy}o"

EARTHSCOPE_BASE_URL = "https://gage-data.earthscope.org/archive/gnss/rinex/obs"
EARTHSCOPE_PATH_PATTERN = "{year}/{doy:03d}/{station_lower}{doy:03d}0.{yy}o.Z"


def download_rinex(station: str, year: int, doy: int, target_path: Path) -> bool:
    """
    Download RINEX file for station/year/doy from NPS GNSS archive.

    Args:
        station: 4-character station ID (e.g., "GLBX")
        year: 4-digit year
        doy: Day of year (1-366)
        target_path: Local path to save the file

    Returns:
        bool: True if download successful, False otherwise
    """
    station_upper = station.upper()
    yy = str(year)[-2:]

    path = RINEX_PATH_PATTERN.format(year=year, doy=doy, station=station_upper, yy=yy)
    url = f"{GNSS_BASE_URL}/{path}"

    return download_from_url(url, target_path)


def download_rinex_earthscope(station: str, year: int, doy: int, target_path: Path) -> bool:
    """
    Download RINEX 2.11 observation file from EarthScope GAGE archive.

    Requires EARTHSCOPE_TOKEN environment variable to be set.
    Downloads .o.Z file (RINEX 2.11 + Unix compress).

    Args:
        station: 4-character station ID (e.g., "UMNQ")
        year: 4-digit year
        doy: Day of year (1-366)
        target_path: Local path to save the .o.Z file

    Returns:
        bool: True if download successful, False otherwise
    """
    token = os.environ.get("EARTHSCOPE_TOKEN")
    if not token:
        logging.error("EARTHSCOPE_TOKEN environment variable not set")
        return False

    station_lower = station.lower()
    yy = str(year)[-2:]
    path = EARTHSCOPE_PATH_PATTERN.format(
        year=year, doy=doy, station_lower=station_lower, yy=yy
    )
    url = f"{EARTHSCOPE_BASE_URL}/{path}"

    headers = {"Authorization": f"Bearer {token}"}
    return download_from_url(url, target_path, headers=headers)


def download_from_url(url: str, target_path: Path, headers: dict = None) -> bool:
    """
    Download a file from a URL to a local path.

    Args:
        url: Full URL to download from
        target_path: Local path to save the file
        headers: Optional HTTP headers (e.g., for bearer token auth)

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        if isinstance(target_path, str):
            target_path = Path(target_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading from URL: {url}")

        response = requests.get(url, stream=True, timeout=120, headers=headers)

        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if target_path.exists() and target_path.stat().st_size > 0:
                logging.info(f"Successfully downloaded file ({target_path.stat().st_size} bytes)")
                return True
            else:
                logging.error(f"File downloaded but appears to be empty: {target_path}")
                return False
        else:
            logging.error(f"HTTP download failed with status code {response.status_code}: {url}")
            return False

    except Exception as e:
        logging.error(f"Error during URL download: {e}")
        return False


def check_file_exists(file_path, min_size_bytes=0):
    """
    Check if a file exists and optionally check its size.

    Args:
        file_path: Path to the file to check
        min_size_bytes: Minimum size in bytes the file should have (default 0)

    Returns:
        bool: True if the file exists and meets size criteria, False otherwise
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        logging.debug(f"File does not exist: {file_path}")
        return False

    if not file_path.is_file():
        logging.debug(f"Path exists but is not a file: {file_path}")
        return False

    file_size = file_path.stat().st_size
    if file_size < min_size_bytes:
        logging.debug(
            f"File is too small: {file_path} ({file_size} bytes < {min_size_bytes} bytes required)"
        )
        return False

    return True
