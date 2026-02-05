#!/usr/bin/env python3
# ABOUTME: Unified reference station finder - searches USGS, CO-OPS, and ERDDAP.
# ABOUTME: Identifies nearest water level gauges for GNSS-IR validation.

"""
Find Reference Stations for GNSS-IR Validation

This script searches multiple data sources to find the nearest reference
water level stations for a GNSS station:

1. USGS Gauges - River and tide gauges from USGS NWIS
2. NOAA CO-OPS - Coastal tide stations
3. ERDDAP - Regional ocean observing system stations

Usage:
    # Search for reference stations near a GNSS station (from config)
    python scripts/find_reference_stations.py --station GLBX

    # Search using explicit coordinates
    python scripts/find_reference_stations.py --lat 58.4551 --lon -135.8885

    # Limit search radius
    python scripts/find_reference_stations.py --station GLBX --radius 50

    # Update station config with best reference
    python scripts/find_reference_stations.py --station GLBX --update-config
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.geo_utils import haversine_distance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ReferenceStation:
    """Represents a reference water level station."""
    source: str  # 'USGS', 'CO-OPS', 'ERDDAP'
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    distance_km: float
    datum: str = 'Unknown'
    notes: str = ''
    data_available: bool = True


def load_station_config(station: str, config_path: Path) -> Optional[dict]:
    """Load station configuration."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    return config.get(station)


def search_usgs_gauges(lat: float, lon: float, radius_km: float = 100) -> List[ReferenceStation]:
    """Search for nearby USGS water level gauges.

    Note: USGS gauges are primarily for inland rivers/streams. For coastal
    GNSS-IR stations, CO-OPS tide gauges and ERDDAP datasets are typically
    more relevant reference sources.
    """
    logger.info(f"Searching USGS gauges within {radius_km} km...")

    results = []

    try:
        # Try using dataretrieval if available
        import dataretrieval.nwis as nwis

        # Use bounding box search
        lat_delta = radius_km / 111  # ~111 km per degree latitude
        lon_delta = radius_km / (111 * math.cos(math.radians(lat)))

        bbox = f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}"

        try:
            # Get site info for gauges with gage height data
            # Note: USGS primarily covers inland rivers/streams, not coastal areas
            site_info = nwis.get_info(bBox=bbox, parameterCd='00065')

            if site_info is not None and len(site_info) > 0 and len(site_info[0]) > 0:
                df = site_info[0]

                for _, row in df.iterrows():
                    site_lat = float(row.get('dec_lat_va', 0))
                    site_lon = float(row.get('dec_long_va', 0))

                    if site_lat == 0 or site_lon == 0:
                        continue

                    distance = haversine_distance(lat, lon, site_lat, site_lon)

                    if distance <= radius_km:
                        results.append(ReferenceStation(
                            source='USGS',
                            station_id=row.get('site_no', 'Unknown'),
                            station_name=row.get('station_nm', 'Unknown'),
                            latitude=site_lat,
                            longitude=site_lon,
                            distance_km=round(distance, 2),
                            datum=row.get('alt_datum_cd', 'Unknown'),
                            notes=f"Type: {row.get('site_tp_cd', 'Unknown')}"
                        ))

                logger.info(f"  Found {len(results)} USGS gauges")
            else:
                logger.info("  No USGS gauges found in area")

        except Exception as e:
            # USGS API often fails for coastal areas with no inland gauges
            logger.debug(f"  USGS search failed: {e}")
            logger.info("  No USGS gauges found (USGS focuses on inland rivers/streams)")

    except ImportError:
        logger.info("  dataretrieval not installed - skipping USGS search")
        logger.info("  Install with: pip install dataretrieval")

    return results


def search_coops_stations(lat: float, lon: float, radius_km: float = 100) -> List[ReferenceStation]:
    """Search for nearby NOAA CO-OPS tide stations."""
    logger.info(f"Searching CO-OPS tide stations within {radius_km} km...")

    results = []

    try:
        from external_apis.noaa_coops import NOAACOOPSClient

        client = NOAACOOPSClient()
        stations = client.find_nearby_stations(lat, lon, radius_km=radius_km)

        for station in stations:
            results.append(ReferenceStation(
                source='CO-OPS',
                station_id=station.get('id', 'Unknown'),
                station_name=station.get('name', 'Unknown'),
                latitude=station.get('latitude', 0),
                longitude=station.get('longitude', 0),
                distance_km=round(station.get('distance_km', 0), 2),
                datum='NAVD88',
                notes=f"State: {station.get('state', 'Unknown')}"
            ))

        logger.info(f"  Found {len(results)} CO-OPS stations")

    except ImportError:
        logger.warning("  noaa_coops module not available")
    except Exception as e:
        logger.warning(f"  CO-OPS search failed: {e}")

    return results


def get_erddap_dataset_location(base_url: str, dataset_id: str, timeout: int = 10) -> Optional[Tuple[float, float]]:
    """Fetch actual coordinates from ERDDAP dataset metadata."""
    try:
        import requests

        # Get dataset info (metadata)
        info_url = f"{base_url}/info/{dataset_id}/index.csv"
        response = requests.get(info_url, timeout=timeout)

        if response.status_code != 200:
            return None

        # Parse metadata looking for latitude/longitude
        lat, lon = None, None

        for line in response.text.split('\n'):
            parts = line.split(',')
            if len(parts) >= 5:
                row_type = parts[0].strip()
                var_name = parts[1].strip().lower() if len(parts) > 1 else ''
                attr_name = parts[2].strip().lower() if len(parts) > 2 else ''
                value = parts[4].strip() if len(parts) > 4 else ''

                # Look for geospatial attributes
                if row_type == 'attribute':
                    if 'geospatial_lat' in attr_name and 'min' in attr_name:
                        try:
                            lat = float(value)
                        except ValueError:
                            pass
                    elif 'geospatial_lon' in attr_name and 'min' in attr_name:
                        try:
                            lon = float(value)
                        except ValueError:
                            pass
                    # Also check for actual_range on latitude/longitude variables
                    elif var_name in ('latitude', 'lat') and attr_name == 'actual_range':
                        try:
                            lat = float(value.split()[0])  # Take first value (min)
                        except (ValueError, IndexError):
                            pass
                    elif var_name in ('longitude', 'lon') and attr_name == 'actual_range':
                        try:
                            lon = float(value.split()[0])  # Take first value (min)
                        except (ValueError, IndexError):
                            pass

        if lat is not None and lon is not None:
            return (lat, lon)

    except Exception:
        pass

    return None


def search_erddap_stations(lat: float, lon: float, radius_km: float = 100) -> List[ReferenceStation]:
    """Search regional ERDDAP servers for nearby water level stations."""
    logger.info(f"Searching ERDDAP servers within {radius_km} km...")

    results = []

    # Regional ERDDAP servers
    erddap_servers = {
        'AOOS': {
            'name': 'Alaska Ocean Observing System',
            'base_url': 'https://erddap.aoos.org/erddap',
            'coverage': {'lat': (50, 72), 'lon': (-180, -125)}
        },
        'PacIOOS': {
            'name': 'Pacific Islands Ocean Observing System',
            'base_url': 'https://pae-paha.pacioos.hawaii.edu/erddap',
            'coverage': {'lat': (15, 30), 'lon': (-165, -150)}
        },
        'SECOORA': {
            'name': 'Southeast Coastal Ocean Observing',
            'base_url': 'https://erddap.secoora.org/erddap',
            'coverage': {'lat': (24, 37), 'lon': (-90, -74)}
        }
    }

    try:
        import requests
        import pandas as pd
        from io import StringIO

        # Find relevant servers
        for server_id, server_info in erddap_servers.items():
            coverage = server_info['coverage']

            # Check if coordinates are in coverage area
            lat_in_range = coverage['lat'][0] <= lat <= coverage['lat'][1]
            lon_in_range = coverage['lon'][0] <= lon <= coverage['lon'][1]

            if not (lat_in_range and lon_in_range):
                continue

            logger.info(f"  Searching {server_id} ({server_info['name']})...")

            try:
                # Use allDatasets endpoint which includes spatial metadata
                # This is more efficient than querying each dataset individually
                all_url = f"{server_info['base_url']}/tabledap/allDatasets.csv?datasetID,title,minLongitude,maxLongitude,minLatitude,maxLatitude"
                response = requests.get(all_url, timeout=30)

                if response.status_code != 200:
                    # Fallback to search if allDatasets not available
                    logger.debug(f"    allDatasets not available, using search...")
                    all_url = f"{server_info['base_url']}/search/index.csv?searchFor=water+level"
                    response = requests.get(all_url, timeout=30)

                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text), skiprows=[1])  # Skip units row

                    # Pre-filter by bounding box (rough filter before distance calc)
                    lat_delta = radius_km / 111
                    lon_delta = radius_km / (111 * max(0.1, math.cos(math.radians(lat))))

                    # Filter datasets that might be in range based on spatial bounds
                    nearby = []
                    for _, row in df.iterrows():
                        dataset_id = row.get('datasetID', row.get('Dataset ID', ''))
                        title = str(row.get('title', row.get('Title', '')))

                        if not dataset_id or pd.isna(dataset_id):
                            continue

                        # Skip non-water-level datasets based on title or dataset ID
                        title_lower = title.lower()
                        id_lower = str(dataset_id).lower()

                        # Water level indicators in title
                        water_terms = ['water', 'level', 'tide', 'sea surface', 'navd', 'mllw', 'cove', 'bay', 'harbor', 'port']
                        # Water level indicators in dataset ID (co-ops stations, ndbc buoys)
                        id_patterns = ['co_ops', 'nos_', 'ndbc', 'tide', 'wl_']

                        title_match = any(term in title_lower for term in water_terms)
                        id_match = any(pattern in id_lower for pattern in id_patterns)

                        if not (title_match or id_match):
                            continue

                        # Check if spatial bounds overlap with search area
                        min_lat = row.get('minLatitude', None)
                        max_lat = row.get('maxLatitude', None)
                        min_lon = row.get('minLongitude', None)
                        max_lon = row.get('maxLongitude', None)

                        # If we have spatial metadata, use it for pre-filtering
                        if all(v is not None and not pd.isna(v) for v in [min_lat, max_lat, min_lon, max_lon]):
                            try:
                                min_lat, max_lat = float(min_lat), float(max_lat)
                                min_lon, max_lon = float(min_lon), float(max_lon)

                                # Check bounding box overlap
                                if (max_lat < lat - lat_delta or min_lat > lat + lat_delta or
                                    max_lon < lon - lon_delta or min_lon > lon + lon_delta):
                                    continue

                                # Use center of dataset bounds for distance
                                station_lat = (min_lat + max_lat) / 2
                                station_lon = (min_lon + max_lon) / 2
                            except (ValueError, TypeError):
                                continue
                        else:
                            # No spatial metadata - would need to fetch individually
                            # Skip these for efficiency
                            continue

                        distance = haversine_distance(lat, lon, station_lat, station_lon)

                        if distance <= radius_km:
                            nearby.append((dataset_id, title, station_lat, station_lon, distance))

                    logger.info(f"    Found {len(nearby)} water level datasets within {radius_km} km")

                    for dataset_id, title, station_lat, station_lon, distance in nearby:
                        # Extract station name from title
                        if 'at ' in title.lower():
                            station_name = title.split('at ')[-1].strip()
                        else:
                            station_name = title

                        results.append(ReferenceStation(
                            source=f'ERDDAP-{server_id}',
                            station_id=dataset_id,
                            station_name=station_name,
                            latitude=round(station_lat, 4),
                            longitude=round(station_lon, 4),
                            distance_km=round(distance, 2),
                            datum='NAVD88',
                            notes=f"Server: {server_info['name']}"
                        ))

            except Exception as e:
                logger.warning(f"    Error searching {server_id}: {e}")

        logger.info(f"  Found {len(results)} ERDDAP stations total")

    except ImportError as e:
        logger.warning(f"  Missing required packages for ERDDAP search: {e}")
    except Exception as e:
        logger.warning(f"  ERDDAP search failed: {e}")

    return results


def find_all_reference_stations(lat: float, lon: float, radius_km: float = 100) -> List[ReferenceStation]:
    """Search all sources for reference stations."""
    all_results = []

    # Search each source
    all_results.extend(search_usgs_gauges(lat, lon, radius_km))
    all_results.extend(search_coops_stations(lat, lon, radius_km))
    all_results.extend(search_erddap_stations(lat, lon, radius_km))

    # Sort by distance
    all_results.sort(key=lambda x: x.distance_km)

    return all_results


def print_results(results: List[ReferenceStation], gnss_station: str = None):
    """Print formatted results."""
    print()
    print("=" * 80)
    if gnss_station:
        print(f"REFERENCE STATIONS FOR {gnss_station}")
    else:
        print("REFERENCE STATIONS FOUND")
    print("=" * 80)
    print()

    if not results:
        print("No reference stations found in search area.")
        return

    # Group by source
    sources = {}
    for r in results:
        source = r.source.split('-')[0]  # Handle ERDDAP-AOOS etc
        if source not in sources:
            sources[source] = []
        sources[source].append(r)

    for source, stations in sources.items():
        print(f"\n{source} STATIONS ({len(stations)} found)")
        print("-" * 60)

        for i, s in enumerate(stations[:5], 1):  # Top 5 per source
            print(f"  {i}. {s.station_name}")
            print(f"     ID: {s.station_id}")
            print(f"     Distance: {s.distance_km} km")
            print(f"     Location: ({s.latitude:.4f}, {s.longitude:.4f})")
            print(f"     Datum: {s.datum}")
            if s.notes:
                print(f"     Notes: {s.notes}")
            print()

    # Summary
    print("=" * 80)
    print("RECOMMENDATION")
    print("-" * 60)

    if results:
        best = results[0]
        print(f"  Closest station: {best.station_name}")
        print(f"  Source: {best.source}")
        print(f"  Distance: {best.distance_km} km")
        print(f"  Station ID: {best.station_id}")

        if best.distance_km < 1:
            print("\n  ** CO-LOCATED ** This station is effectively at the same location!")
        elif best.distance_km < 10:
            print("\n  ** EXCELLENT ** Very close reference for validation.")
        elif best.distance_km < 50:
            print("\n  ** GOOD ** Reasonable reference, may need time lag analysis.")
        else:
            print("\n  ** FAIR ** Distant reference, correlation may be affected.")

        # Print configuration instructions
        print()
        print("=" * 80)
        print("CONFIGURATION INSTRUCTIONS")
        print("-" * 60)
        print("Add the following to your station in config/stations_config.json:")
        print()

        if best.source == 'USGS':
            print(f'''  "usgs_comparison": {{
    "target_usgs_site": "{best.station_id}",
    "usgs_gauge_stated_datum": "{best.datum}",
    "distance_km": {best.distance_km}
  }}''')
        elif best.source == 'CO-OPS':
            print(f'''  "external_data_sources": {{
    "noaa_coops": {{
      "enabled": true,
      "target_station": "{best.station_id}",
      "station_name": "{best.station_name}",
      "distance_km": {best.distance_km}
    }}
  }}''')
        elif 'ERDDAP' in best.source:
            server_id = best.source.split('-')[-1] if '-' in best.source else 'AOOS'
            is_colocated = best.distance_km < 1
            print(f'''  "erddap": {{
    "enabled": true,
    "dataset_id": "{best.station_id}",
    "station_name": "{best.station_name}",
    "distance_km": {best.distance_km},
    "server": "{server_id}"{', ' + chr(10) + '    "primary_reference": true' if is_colocated else ''}
  }}''')

        print()
        print("Or run with --update-config to auto-update the config file:")
        print(f"  python scripts/find_reference_stations.py --station {gnss_station or 'STATION'} --update-config")
        print()


def update_station_config(station: str, best_ref: ReferenceStation, config_path: Path) -> bool:
    """Update station configuration with best reference."""
    logger.info(f"Updating configuration for {station}...")

    with open(config_path) as f:
        config = json.load(f)

    if station not in config:
        logger.error(f"Station {station} not found in config")
        return False

    station_config = config[station]

    # Update based on source type
    if best_ref.source == 'USGS':
        if 'usgs_comparison' not in station_config:
            station_config['usgs_comparison'] = {}
        station_config['usgs_comparison']['target_usgs_site'] = best_ref.station_id
        station_config['usgs_comparison']['usgs_gauge_stated_datum'] = best_ref.datum
        station_config['usgs_comparison']['notes'] = f"Using {best_ref.station_name} at {best_ref.distance_km}km"

    elif best_ref.source == 'CO-OPS':
        if 'external_data_sources' not in station_config:
            station_config['external_data_sources'] = {}
        if 'noaa_coops' not in station_config['external_data_sources']:
            station_config['external_data_sources']['noaa_coops'] = {}

        coops = station_config['external_data_sources']['noaa_coops']
        coops['enabled'] = True
        coops['preferred_stations'] = [best_ref.station_id]
        coops['nearest_station'] = {
            'id': best_ref.station_id,
            'name': best_ref.station_name,
            'distance_km': best_ref.distance_km
        }

    elif 'ERDDAP' in best_ref.source:
        if 'erddap' not in station_config:
            station_config['erddap'] = {}

        server_id = best_ref.source.split('-')[-1] if '-' in best_ref.source else 'Unknown'
        station_config['erddap']['enabled'] = True
        station_config['erddap']['dataset_id'] = best_ref.station_id
        station_config['erddap']['station_name'] = best_ref.station_name
        station_config['erddap']['distance_km'] = best_ref.distance_km
        station_config['erddap']['server'] = server_id

        # If very close, mark as primary
        if best_ref.distance_km < 1:
            station_config['erddap']['primary_reference'] = True

    # Save config
    backup_path = config_path.with_suffix('.json.bak')
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created backup at {backup_path}")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Updated {config_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Find reference water level stations for GNSS-IR validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--station', type=str, help='GNSS station ID (e.g., GLBX)')
    parser.add_argument('--lat', type=float, help='Latitude (if not using --station)')
    parser.add_argument('--lon', type=float, help='Longitude (if not using --station)')
    parser.add_argument('--radius', type=float, default=100, help='Search radius in km (default: 100)')
    parser.add_argument('--update-config', action='store_true', help='Update station config with best reference')
    parser.add_argument('--config', type=str, help='Path to stations_config.json')
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / 'config' / 'stations_config.json'

    # Get coordinates
    if args.station:
        station_config = load_station_config(args.station, config_path)
        if not station_config:
            logger.error(f"Station {args.station} not found in config")
            sys.exit(1)

        lat = station_config.get('latitude_deg', station_config.get('latitude'))
        lon = station_config.get('longitude_deg', station_config.get('longitude'))

        if lat is None or lon is None:
            logger.error(f"No coordinates found for station {args.station}")
            sys.exit(1)

        logger.info(f"Station {args.station} at ({lat}, {lon})")

    elif args.lat is not None and args.lon is not None:
        lat = args.lat
        lon = args.lon
        logger.info(f"Searching near ({lat}, {lon})")
    else:
        parser.error("Either --station or both --lat and --lon are required")

    # Search for reference stations
    results = find_all_reference_stations(lat, lon, args.radius)

    # Print results
    print_results(results, args.station)

    # Update config if requested
    if args.update_config and results:
        best = results[0]
        if update_station_config(args.station, best, config_path):
            print(f"\nConfiguration updated with {best.source} station {best.station_id}")
        else:
            print("\nFailed to update configuration")
            sys.exit(1)


if __name__ == '__main__':
    main()
