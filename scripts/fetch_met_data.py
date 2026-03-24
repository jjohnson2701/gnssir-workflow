#!/usr/bin/env python3
# ABOUTME: Fetch daily temperature data from Open-Meteo for GNSS-IR stations.
# ABOUTME: Saves CSV to results_annual/{STATION}/{STATION}_{year}_met_daily.csv.

"""
Fetch daily temperature data from Open-Meteo Archive API.

Reads lat/lon from stations_config.json. Saves CSV alongside other results.
Run once per station-year as a preprocessing step.

Usage:
    python scripts/fetch_met_data.py --station UMNQ --year 2025
    python scripts/fetch_met_data.py --all    # all stations with results
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"
RESULTS_DIR = PROJECT_ROOT / "results_annual"

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


def get_station_coords(station_id, config_path=None):
    """Read lat/lon for a station from stations_config.json.

    Returns (latitude, longitude).
    Raises KeyError if station not found.
    """
    cfg_path = config_path or CONFIG_PATH
    with open(cfg_path) as f:
        all_cfg = json.load(f)

    if station_id not in all_cfg:
        raise KeyError(f"Station '{station_id}' not found in {cfg_path}")

    cfg = all_cfg[station_id]
    return cfg["latitude_deg"], cfg["longitude_deg"]


def build_open_meteo_url(latitude, longitude, year):
    """Construct the Open-Meteo Archive API URL for daily temperature."""
    params = (
        f"latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={year}-01-01"
        f"&end_date={year}-12-31"
        f"&daily=temperature_2m_mean,temperature_2m_min,temperature_2m_max"
        f"&timezone=UTC"
    )
    return f"{OPEN_METEO_BASE}?{params}"


def parse_open_meteo_response(data):
    """Parse Open-Meteo JSON response into a DataFrame.

    Returns DataFrame with columns: date, temp_mean_c, temp_min_c, temp_max_c.
    """
    daily = data["daily"]
    return pd.DataFrame({
        "date": daily["time"],
        "temp_mean_c": daily["temperature_2m_mean"],
        "temp_min_c": daily["temperature_2m_min"],
        "temp_max_c": daily["temperature_2m_max"],
    })


def fetch_met_data(station_id, year, config_path=None, overwrite=False):
    """Fetch and save daily temperature data for one station-year.

    Returns the output path on success, None on failure.
    """
    out_dir = RESULTS_DIR / station_id
    out_path = out_dir / f"{station_id}_{year}_met_daily.csv"

    if out_path.exists() and not overwrite:
        print(f"  Skipping {station_id} {year} — already exists. Use --overwrite to re-fetch.")
        return out_path

    try:
        lat, lon = get_station_coords(station_id, config_path)
    except (KeyError, FileNotFoundError) as e:
        print(f"  Error reading config for {station_id}: {e}")
        return None

    url = build_open_meteo_url(lat, lon, year)
    print(f"  Fetching {station_id} {year} from Open-Meteo...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gnssir-workflow/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  Network error for {station_id} {year}: {e}")
        return None

    df = parse_open_meteo_response(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} days to {out_path}")
    return out_path


def find_all_station_years():
    """Scan results_annual/ to find all station-year combinations."""
    station_years = []
    if not RESULTS_DIR.exists():
        return station_years

    for station_dir in sorted(RESULTS_DIR.iterdir()):
        if not station_dir.is_dir():
            continue
        station_id = station_dir.name
        # Find years from any result file
        years = set()
        for f in station_dir.glob(f"{station_id}_*_*"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    years.add(int(parts[1]))
                except ValueError:
                    continue
        for year in sorted(years):
            station_years.append((station_id, year))

    return station_years


def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily temperature data from Open-Meteo Archive API"
    )
    parser.add_argument("--station", type=str, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--year", type=int, help="Year to fetch")
    parser.add_argument("--all", action="store_true",
                        help="Fetch for all stations with results")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch even if CSV already exists")
    args = parser.parse_args()

    if args.all:
        station_years = find_all_station_years()
        if not station_years:
            print("No station-year combinations found in results_annual/")
            sys.exit(1)

        print(f"Found {len(station_years)} station-year combinations")
        for station_id, year in station_years:
            fetch_met_data(station_id, year, overwrite=args.overwrite)
            time.sleep(1)  # Rate limit

    elif args.station and args.year:
        result = fetch_met_data(args.station, args.year, overwrite=args.overwrite)
        if result is None:
            sys.exit(1)

    else:
        parser.error("Provide --station and --year, or use --all")


if __name__ == "__main__":
    main()
