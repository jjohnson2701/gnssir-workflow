#!/usr/bin/env python3
"""
ABOUTME: Search regional ERDDAP servers for water level stations near GNSS-IR sites
ABOUTME: Identifies potential co-located or nearby reference gauges for validation
"""

import requests
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dashboard_components.station_metadata import get_station_config, get_all_station_ids
from scripts.utils.geo_utils import haversine_distance

# Regional ERDDAP servers
ERDDAP_SERVERS = {
    "AOOS": {
        "name": "Alaska Ocean Observing System",
        "base_url": "https://erddap.aoos.org/erddap",
        "region": "Alaska",
        "coverage": {"lat": (50, 72), "lon": (-180, -125)},
    },
    "PacIOOS": {
        "name": "Pacific Islands Ocean Observing System",
        "base_url": "https://pae-paha.pacioos.hawaii.edu/erddap",
        "region": "Pacific Islands (Hawaii)",
        "coverage": {"lat": (15, 30), "lon": (-165, -150)},
    },
    "SECOORA": {
        "name": "Southeast Coastal Ocean Observing",
        "base_url": "https://erddap.secoora.org/erddap",
        "region": "Southeast US (NC, SC, GA, FL)",
        "coverage": {"lat": (24, 37), "lon": (-90, -74)},
    },
    "MARACOOS": {
        "name": "Mid-Atlantic Regional Coastal Ocean Observing",
        "base_url": "https://erddap.maracoos.org/erddap",
        "region": "Mid-Atlantic (NY, NJ, DE, MD, VA)",
        "coverage": {"lat": (36, 42), "lon": (-77, -70)},
    },
    "GCOOS": {
        "name": "Gulf of Mexico Coastal Ocean Observing System",
        "base_url": "https://erddap.gcoos.org/erddap",
        "region": "Gulf of Mexico",
        "coverage": {"lat": (24, 31), "lon": (-98, -80)},
    },
    "NOAA_COOPS": {
        "name": "NOAA CO-OPS ERDDAP",
        "base_url": "https://coastwatch.pfeg.noaa.gov/erddap",
        "region": "National (NOAA)",
        "coverage": {"lat": (15, 72), "lon": (-180, -65)},
    },
}


def get_relevant_erddap_servers(lat: float, lon: float) -> List[Tuple[str, Dict]]:
    """Identify ERDDAP servers that cover the given coordinates."""
    relevant = []

    for server_id, server_info in ERDDAP_SERVERS.items():
        coverage = server_info["coverage"]

        lat_in_range = coverage["lat"][0] <= lat <= coverage["lat"][1]
        lon_in_range = coverage["lon"][0] <= lon <= coverage["lon"][1]

        if lat_in_range and lon_in_range:
            relevant.append((server_id, server_info))

    return relevant


def search_erddap_datasets(base_url: str, search_term: str = "water level") -> List[Dict]:
    """
    Search ERDDAP server for datasets matching search term.

    Returns list of dataset IDs and metadata.
    """
    search_url = f"{base_url}/search/index.csv?searchFor={search_term.replace(' ', '+')}"

    try:
        response = requests.get(search_url, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        from io import StringIO

        df = pd.read_csv(StringIO(response.text))

        datasets = []
        for _, row in df.iterrows():
            if "Dataset ID" in row and pd.notna(row["Dataset ID"]):
                datasets.append(
                    {
                        "dataset_id": row["Dataset ID"],
                        "title": row.get("Title", "Unknown"),
                        "summary": row.get("Summary", ""),
                    }
                )

        return datasets

    except Exception as e:
        print(f"  Warning: Search failed for {base_url}: {e}")
        return []


def get_dataset_coordinates(base_url: str, dataset_id: str) -> Tuple[float, float]:
    """
    Get coordinates for an ERDDAP dataset.

    Returns (lat, lon) or (None, None) if not found.
    """
    info_url = f"{base_url}/info/{dataset_id}/index.csv"

    try:
        response = requests.get(info_url, timeout=30)
        response.raise_for_status()

        # Parse metadata CSV
        from io import StringIO

        df = pd.read_csv(StringIO(response.text))

        # Look for latitude/longitude in metadata
        lat = None
        lon = None

        for _, row in df.iterrows():
            var_name = str(row.get("Variable Name", "")).lower()
            value = row.get("Value", None)

            if "latitude" in var_name and "actual_range" not in var_name:
                try:
                    lat = float(value)
                except (ValueError, TypeError):
                    pass

            if "longitude" in var_name and "actual_range" not in var_name:
                try:
                    lon = float(value)
                except (ValueError, TypeError):
                    pass

        return lat, lon

    except Exception:
        return None, None


def search_nearby_erddap_stations(station_id: str, max_distance_km: float = 50) -> List[Dict]:
    """
    Search for ERDDAP water level stations near a GNSS-IR station.

    Args:
        station_id: GNSS-IR station identifier
        max_distance_km: Maximum search radius

    Returns:
        List of nearby ERDDAP datasets with distance information
    """
    # Get station configuration
    config = get_station_config(station_id)
    if not config:
        print(f"ERROR: Station {station_id} not found in configuration")
        return []

    gnss_lat = config.get("latitude_deg")
    gnss_lon = config.get("longitude_deg")

    if gnss_lat is None or gnss_lon is None:
        print(f"ERROR: No coordinates found for station {station_id}")
        return []

    print(f"\n{'='*100}")
    print(f"Searching for ERDDAP stations near {station_id}")
    print(f"  Location: {gnss_lat:.4f}°N, {gnss_lon:.4f}°W")
    print(f"  Search Radius: {max_distance_km} km")
    print(f"{'='*100}\n")

    # Find relevant ERDDAP servers
    relevant_servers = get_relevant_erddap_servers(gnss_lat, gnss_lon)

    if not relevant_servers:
        print(f"No regional ERDDAP servers cover {station_id} location")
        return []

    print(f"Found {len(relevant_servers)} relevant ERDDAP server(s):\n")
    for server_id, server_info in relevant_servers:
        print(f"  • {server_info['name']} ({server_id})")
        print(f"    {server_info['base_url']}")
        print(f"    Region: {server_info['region']}")

    print()

    # Search each server for water level datasets
    nearby_stations = []

    for server_id, server_info in relevant_servers:
        print(f"\n{'-'*100}")
        print(f"Searching {server_info['name']}...")
        print(f"{'-'*100}")

        base_url = server_info["base_url"]

        # Search for water level datasets
        datasets = search_erddap_datasets(base_url, "water level")

        print(f"  Found {len(datasets)} dataset(s) matching 'water level'")

        if not datasets:
            continue

        # Check each dataset for proximity
        print(f"\n  Checking proximity to {station_id}...")

        for dataset in datasets:
            dataset_id = dataset["dataset_id"]

            # Get dataset coordinates
            ds_lat, ds_lon = get_dataset_coordinates(base_url, dataset_id)

            if ds_lat is None or ds_lon is None:
                continue

            # Calculate distance
            distance = haversine_distance(gnss_lat, gnss_lon, ds_lat, ds_lon)

            if distance <= max_distance_km:
                nearby_stations.append(
                    {
                        "erddap_server": server_id,
                        "erddap_name": server_info["name"],
                        "base_url": base_url,
                        "dataset_id": dataset_id,
                        "title": dataset["title"],
                        "latitude": ds_lat,
                        "longitude": ds_lon,
                        "distance_km": round(distance, 2),
                        "gnss_station": station_id,
                    }
                )

                print(f"\n    ✓ {dataset_id}")
                print(f"      Title: {dataset['title']}")
                print(f"      Location: {ds_lat:.4f}°N, {ds_lon:.4f}°W")
                print(f"      Distance: {distance:.2f} km")

    return nearby_stations


def main():
    """Search all configured stations for nearby ERDDAP water level sources."""

    # Get all configured stations
    station_ids = get_all_station_ids()

    print(f"\n{'='*100}")
    print(f"ERDDAP WATER LEVEL STATION SEARCH")
    print(f"{'='*100}")
    print(f"\nConfigured GNSS-IR Stations: {', '.join(station_ids)}\n")

    all_results = {}

    for station_id in station_ids:
        results = search_nearby_erddap_stations(station_id, max_distance_km=50)

        if results:
            all_results[station_id] = results
            print(f"\n{'='*100}")
            print(f"SUMMARY: Found {len(results)} nearby ERDDAP station(s) for {station_id}")
            print(f"{'='*100}\n")

    # Final summary
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY")
    print(f"{'='*100}\n")

    if all_results:
        for station_id, results in all_results.items():
            print(f"{station_id}: {len(results)} ERDDAP station(s) found")
            for result in results:
                print(
                    f"  • {result['dataset_id']} ({result['erddap_server']}) - {result['distance_km']} km"
                )

        # Save results to JSON
        output_file = project_root / "results_annual" / "erddap_search_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    else:
        print("No nearby ERDDAP stations found for any configured GNSS-IR stations")
        print("\nNote: GLBX already uses Bartlett Cove ERDDAP (configured manually)")

    print()


if __name__ == "__main__":
    main()
