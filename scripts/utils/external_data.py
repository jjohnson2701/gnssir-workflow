# ABOUTME: Shared utilities for external/secondary data sources (ERA5, TEC, SMAP, NDBC)
# ABOUTME: Standardized config loading, cache/output path helpers, and NDBC wind data fetcher

"""
Shared utilities for external data integration.

Provides:
  - Station config loading (lat/lon/height from stations_config.json)
  - Cache and output path helpers (consistent naming across all extractors)
  - NDBC historical wind data fetcher

All external data scripts should import from here instead of
reimplementing config loading and path construction.

Pattern follows s1_fresnel_utils.py: centralized helpers, no global state.
"""

import gzip
import json
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Station config
# ---------------------------------------------------------------------------

def load_station_config(station, project_root=None):
    """Load full station config from stations_config.json.

    Returns the station's config dict, or raises ValueError if not found.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    config_path = root / "config" / "stations_config.json"
    with open(config_path) as f:
        all_cfg = json.load(f)
    if station not in all_cfg:
        raise ValueError(f"Station {station} not in {config_path}")
    return all_cfg[station]


def load_station_coords(station, project_root=None):
    """Load station lat/lon/height.

    Returns dict with keys: lat, lon, height_m
    """
    cfg = load_station_config(station, project_root)
    return {
        "lat": cfg["latitude_deg"],
        "lon": cfg["longitude_deg"],
        "height_m": cfg.get("ellipsoidal_height_m", 0),
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_cache_dir(source, project_root=None):
    """Return cache directory for an external data source.

    Args:
        source: one of "era5", "tec", "smap", "cygnss", "ndbc"

    Returns Path (created if it doesn't exist).
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    cache = root / "data" / ".cache" / source
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def get_results_dir(station, project_root=None):
    """Return results_annual directory for a station.

    Returns Path (created if it doesn't exist).
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    d = root / "results_annual" / station
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_path(station, year, suffix, project_root=None):
    """Construct canonical output path for a station-year data product.

    Args:
        station: station ID (e.g., "ROSS")
        year: processing year
        suffix: file suffix (e.g., "era5.parquet", "tec.parquet",
                "smap_comparison.parquet")

    Returns Path like results_annual/ROSS/ROSS_2024_era5.parquet
    """
    d = get_results_dir(station, project_root)
    return d / f"{station}_{year}_{suffix}"


# ---------------------------------------------------------------------------
# NDBC wind data
# ---------------------------------------------------------------------------

NDBC_BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet"

# Known buoys near GNSS-IR stations (can be extended via station config)
STATION_BUOY_MAP = {
    "ROSS": "pilm4",   # Passage Island, MI — Lake Superior
    "MCHN": "pilm4",   # Passage Island, MI — Lake Superior
    "CLWD": "45003",   # N Lake Huron / Georgian Bay
    "KNGV": "45005",   # W Lake Erie
    "GOD2": "lscm4",   # Lake St. Clair
}


def get_buoy_id(station):
    """Look up the nearest NDBC buoy for a station.

    Checks station config for an ndbc_buoy field first,
    falls back to the hardcoded STATION_BUOY_MAP.

    Returns buoy ID string or None.
    """
    try:
        cfg = load_station_config(station)
        buoy = cfg.get("ndbc_buoy")
        if buoy:
            return buoy
    except (ValueError, FileNotFoundError):
        pass
    return STATION_BUOY_MAP.get(station)


def fetch_ndbc_wind(buoy_id, year, cache_dir=None):
    """Fetch NDBC historical standard meteorological data for one year.

    Downloads gzip-compressed text from NDBC archives, parses the
    variable-width ASCII format, and returns a DataFrame with columns:
        datetime, WDIR (deg), WSPD (m/s), GST (m/s), ATMP (°C), WTMP (°C), ...

    Args:
        buoy_id: NDBC buoy/station ID (e.g., "pilm4", "45003")
        year: calendar year
        cache_dir: optional cache directory (default: data/.cache/ndbc/)

    Returns DataFrame or None on failure.
    """
    import requests

    if cache_dir is None:
        cache_dir = get_cache_dir("ndbc")
    cache_dir = Path(cache_dir)

    # Check cache
    cache_path = cache_dir / f"{buoy_id}_{year}_stdmet.parquet"
    if cache_path.exists():
        logger.debug(f"NDBC cached: {cache_path}")
        return pd.read_parquet(cache_path)

    url = f"{NDBC_BASE_URL}/{buoy_id}h{year}.txt.gz"
    logger.info(f"Fetching NDBC {buoy_id} {year}: {url}")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"NDBC fetch failed: HTTP {resp.status_code} for {url}")
            return None

        data = gzip.decompress(resp.content).decode("ascii", errors="replace")
        df = pd.read_csv(
            StringIO(data),
            sep=r"\s+",
            skiprows=[1],
            na_values=["999", "999.0", "99.0", "9999", "9999.0"],
        )

        # Clean up column names (strip leading #)
        df.columns = [c.lstrip("#") for c in df.columns]

        # Build datetime from year/month/day/hour/minute columns
        dt_cols = {}
        for src, dst in [("YY", "year"), ("MM", "month"), ("DD", "day"),
                         ("hh", "hour"), ("mm", "minute")]:
            if src in df.columns:
                dt_cols[src] = dst
        if dt_cols:
            df["datetime"] = pd.to_datetime(
                df[list(dt_cols.keys())].rename(columns=dt_cols)
            )

        # Cache as parquet for fast reload
        df.to_parquet(cache_path, index=False)
        logger.info(f"NDBC {buoy_id} {year}: {len(df)} records → {cache_path}")

        return df

    except Exception as e:
        logger.warning(f"NDBC fetch error for {buoy_id} {year}: {e}")
        return None


def fetch_ndbc_ice_season(station, year, months_before=(11, 12),
                          months_after=(1, 2, 3, 4, 5)):
    """Fetch NDBC wind data spanning an ice season (Nov prior → May target year).

    Combines Dec of year-1 + Jan-May of year into one DataFrame.
    Drops rows with missing WDIR or WSPD.

    Args:
        station: GNSS-IR station ID
        year: the ice-season year (e.g., 2024 means Nov 2023 – May 2024)
        months_before: months from prior year to include
        months_after: months from target year to include

    Returns DataFrame with datetime, WDIR, WSPD, etc. or None.
    """
    buoy_id = get_buoy_id(station)
    if buoy_id is None:
        logger.warning(f"No NDBC buoy mapped for {station}")
        return None

    parts = []

    # Prior year (Nov–Dec)
    w1 = fetch_ndbc_wind(buoy_id, year - 1)
    if w1 is not None and "datetime" in w1.columns:
        mask = w1["datetime"].dt.month.isin(months_before)
        if mask.any():
            parts.append(w1[mask])

    # Target year (Jan–May)
    w2 = fetch_ndbc_wind(buoy_id, year)
    if w2 is not None and "datetime" in w2.columns:
        mask = w2["datetime"].dt.month.isin(months_after)
        if mask.any():
            parts.append(w2[mask])

    if not parts:
        logger.warning(f"No NDBC data for {station} ice season {year}")
        return None

    wind = pd.concat(parts, ignore_index=True)
    wind = wind.dropna(subset=["WDIR", "WSPD"])
    logger.info(f"NDBC ice season {station} {year}: {len(wind)} records "
                f"(buoy {buoy_id})")
    return wind
