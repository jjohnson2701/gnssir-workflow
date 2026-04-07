# ABOUTME: Shared utilities for external/secondary data sources (HYDAT, EC Climate, ERA5, TEC, SMAP, NDBC)
# ABOUTME: Standardized config loading, cache/output path helpers, and data fetchers for Canadian/US reference stations

"""
Shared utilities for external data integration.

Provides:
  - Station config loading (lat/lon/height from stations_config.json)
  - Cache and output path helpers (consistent naming across all extractors)
  - HYDAT daily water level fetcher (Canadian Great Lakes gauges)
  - EC Climate daily temperature fetcher (Environment Canada stations)
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

# Map logical source names to on-disk cache directory names.
# Matches existing directories in data/.cache/ to avoid orphaning cached data.
_CACHE_DIR_NAMES = {
    "era5": "era5",
    "tec": "ionex",
    "smap": "smap_ft",
    "cygnss": "cygnss",
    "ndbc": "ndbc",
    "glerl": "glerl_ice",
    "hydat": "hydat",
    "ec_climate": "ec_climate",
}


def get_cache_dir(source, project_root=None):
    """Return cache directory for an external data source.

    Args:
        source: one of "era5", "tec", "smap", "cygnss", "ndbc", "glerl"

    Returns Path (created if it doesn't exist).
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    dirname = _CACHE_DIR_NAMES.get(source, source)
    cache = root / "data" / ".cache" / dirname
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


# ---------------------------------------------------------------------------
# HYDAT daily water levels (Canadian Great Lakes gauges)
# ---------------------------------------------------------------------------

HYDAT_API_BASE = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"


def get_hydat_station_id(station):
    """Look up the HYDAT station ID from station config.

    Returns station ID string (e.g., "02BA004") or None.
    """
    try:
        cfg = load_station_config(station)
        ext = cfg.get("external_data_sources", {})
        hydat = ext.get("hydat", {})
        if hydat.get("enabled") and hydat.get("station_id"):
            return hydat["station_id"]
    except (ValueError, FileNotFoundError):
        pass
    return None


def fetch_hydat_daily_level(hydat_station_id, year, cache_dir=None):
    """Fetch HYDAT daily mean water level for one year.

    Uses the MSC GeoMet OGC API (api.weather.gc.ca) to download daily
    mean water levels from the Water Survey of Canada HYDAT database.

    Args:
        hydat_station_id: HYDAT station number (e.g., "02BA004")
        year: calendar year
        cache_dir: optional cache directory (default: data/.cache/hydat/)

    Returns DataFrame with columns [date, level_m] or None on failure.
        level_m is daily mean water level in metres (IGLD85 datum).
    """
    import requests

    if cache_dir is None:
        cache_dir = get_cache_dir("hydat")
    cache_dir = Path(cache_dir)

    cache_path = cache_dir / f"{hydat_station_id}_{year}_daily_level.parquet"
    if cache_path.exists():
        logger.debug(f"HYDAT cached: {cache_path}")
        return pd.read_parquet(cache_path)

    url = (
        f"{HYDAT_API_BASE}"
        f"?STATION_NUMBER={hydat_station_id}"
        f"&datetime={year}-01-01/{year}-12-31"
        f"&f=json&limit=400"
    )
    logger.info(f"Fetching HYDAT {hydat_station_id} {year}")

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "GNSS-IR/1.0"})
        if resp.status_code != 200:
            logger.warning(f"HYDAT fetch failed: HTTP {resp.status_code}")
            return None

        data = resp.json()
        features = data.get("features", [])
        if not features:
            logger.warning(f"HYDAT {hydat_station_id} {year}: no data returned")
            return None

        records = []
        for f in features:
            props = f["properties"]
            level = props.get("LEVEL")
            if level is not None:
                records.append({
                    "date": props["DATE"],
                    "level_m": float(level),
                })

        if not records:
            logger.warning(f"HYDAT {hydat_station_id} {year}: all LEVEL values null")
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df.to_parquet(cache_path, index=False)
        logger.info(
            f"HYDAT {hydat_station_id} {year}: {len(df)} days "
            f"({df['level_m'].min():.2f}–{df['level_m'].max():.2f} m) → {cache_path.name}"
        )
        return df

    except Exception as e:
        logger.warning(f"HYDAT fetch error for {hydat_station_id} {year}: {e}")
        return None


def fetch_hydat_for_station(station, year, cache_dir=None):
    """Convenience: fetch HYDAT water level using a GNSS station name.

    Looks up the HYDAT station ID from config, then fetches.
    """
    hydat_id = get_hydat_station_id(station)
    if hydat_id is None:
        logger.warning(f"No HYDAT station configured for {station}")
        return None
    return fetch_hydat_daily_level(hydat_id, year, cache_dir)


# ---------------------------------------------------------------------------
# Environment Canada daily temperature
# ---------------------------------------------------------------------------

EC_CLIMATE_BASE = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"


def get_ec_climate_station_id(station):
    """Look up the EC Climate station ID from station config.

    Returns integer station_id or None.
    """
    try:
        cfg = load_station_config(station)
        ext = cfg.get("external_data_sources", {})
        ec = ext.get("ec_climate", {})
        if ec.get("enabled") and ec.get("station_id"):
            return int(ec["station_id"])
    except (ValueError, FileNotFoundError):
        pass
    return None


def fetch_ec_daily_temperature(ec_station_id, year, cache_dir=None):
    """Fetch Environment Canada daily temperature data for one year.

    Downloads bulk CSV from climate.weather.gc.ca with daily max, min,
    and mean temperature.

    Args:
        ec_station_id: EC Climate station ID (integer, e.g., 7747)
        year: calendar year
        cache_dir: optional cache directory (default: data/.cache/ec_climate/)

    Returns DataFrame with columns [date, max_temp_c, min_temp_c, mean_temp_c]
        or None on failure.
    """
    import requests

    if cache_dir is None:
        cache_dir = get_cache_dir("ec_climate")
    cache_dir = Path(cache_dir)

    cache_path = cache_dir / f"ec_{ec_station_id}_{year}_daily_temp.parquet"
    if cache_path.exists():
        logger.debug(f"EC Climate cached: {cache_path}")
        return pd.read_parquet(cache_path)

    url = (
        f"{EC_CLIMATE_BASE}?format=csv"
        f"&stationID={ec_station_id}"
        f"&Year={year}&Month=1&Day=1&timeframe=2"
    )
    logger.info(f"Fetching EC Climate station {ec_station_id} {year}")

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "GNSS-IR/1.0"})
        if resp.status_code != 200:
            logger.warning(f"EC Climate fetch failed: HTTP {resp.status_code}")
            return None

        from io import StringIO
        raw_df = pd.read_csv(StringIO(resp.text))

        # Standardize column names — EC CSV has verbose names with degree symbols
        col_map = {}
        for col in raw_df.columns:
            cl = col.lower()
            if "date/time" in cl:
                col_map[col] = "date"
            elif "max temp" in cl and "flag" not in cl:
                col_map[col] = "max_temp_c"
            elif "min temp" in cl and "flag" not in cl:
                col_map[col] = "min_temp_c"
            elif "mean temp" in cl and "flag" not in cl:
                col_map[col] = "mean_temp_c"
            elif "total precip" in cl and "flag" not in cl:
                col_map[col] = "total_precip_mm"

        df = raw_df.rename(columns=col_map)
        keep = [c for c in ["date", "max_temp_c", "min_temp_c", "mean_temp_c",
                             "total_precip_mm"] if c in df.columns]
        df = df[keep].copy()

        if "date" not in df.columns:
            logger.warning(f"EC Climate {ec_station_id} {year}: no date column found")
            return None

        df["date"] = pd.to_datetime(df["date"])
        # Coerce to numeric (EC puts empty strings for missing)
        for col in ["max_temp_c", "min_temp_c", "mean_temp_c", "total_precip_mm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        df.to_parquet(cache_path, index=False)
        valid_temp = df["mean_temp_c"].notna().sum() if "mean_temp_c" in df.columns else 0
        logger.info(
            f"EC Climate {ec_station_id} {year}: {len(df)} days, "
            f"{valid_temp} with temperature → {cache_path.name}"
        )
        return df

    except Exception as e:
        logger.warning(f"EC Climate fetch error for {ec_station_id} {year}: {e}")
        return None


def fetch_ec_temperature_for_station(station, year, cache_dir=None):
    """Convenience: fetch EC temperature using a GNSS station name.

    Looks up the EC Climate station ID from config, then fetches.
    """
    ec_id = get_ec_climate_station_id(station)
    if ec_id is None:
        logger.warning(f"No EC Climate station configured for {station}")
        return None
    return fetch_ec_daily_temperature(ec_id, year, cache_dir)


def fetch_ec_ice_season(station, year, months_before=(11, 12),
                        months_after=(1, 2, 3, 4, 5)):
    """Fetch EC temperature spanning an ice season (Nov prior → May target year).

    Args:
        station: GNSS-IR station ID
        year: the ice-season year (e.g., 2024 means Nov 2023 – May 2024)

    Returns DataFrame with date, mean_temp_c, etc. or None.
    """
    ec_id = get_ec_climate_station_id(station)
    if ec_id is None:
        logger.warning(f"No EC Climate station configured for {station}")
        return None

    parts = []
    t1 = fetch_ec_daily_temperature(ec_id, year - 1)
    if t1 is not None:
        mask = t1["date"].dt.month.isin(months_before)
        if mask.any():
            parts.append(t1[mask])

    t2 = fetch_ec_daily_temperature(ec_id, year)
    if t2 is not None:
        mask = t2["date"].dt.month.isin(months_after)
        if mask.any():
            parts.append(t2[mask])

    if not parts:
        logger.warning(f"No EC temperature data for {station} ice season {year}")
        return None

    df = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    logger.info(f"EC ice season {station} {year}: {len(df)} days")
    return df


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
