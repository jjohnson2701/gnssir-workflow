# ABOUTME: NOAA CO-OPS comparison for GNSS-IR processing
# ABOUTME: Compares reflector height data with CO-OPS tide gauge water levels

"""
NOAA CO-OPS Comparison for GNSS-IR Processing

Compares GNSS-IR reflector height data with NOAA CO-OPS tide gauge observations.
Similar to usgs_comparison.py but uses CO-OPS API for coastal tide stations.

Usage:
    python scripts/coops_comparison.py --station FORA --year 2024
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json

# Add project modules
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import project modules
from utils.gnssir_loader import load_gnssir_data
from utils.geo_utils import haversine_distance
import reflector_height_utils
from scripts.external_apis.noaa_coops import NOAACOOPSClient

try:
    from scripts.utils.segmented_analysis import (
        generate_monthly_segments,
        generate_seasonal_segments,
        perform_segmented_correlation,
        filter_by_segment
    )
except ImportError:
    from utils.segmented_analysis import (
        generate_monthly_segments,
        generate_seasonal_segments,
        perform_segmented_correlation,
        filter_by_segment
    )

try:
    import scripts.visualizer as visualizer
except ImportError:
    import visualizer

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_station_config(station_name: str) -> dict:
    """Load station configuration from stations_config.json."""
    config_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    return config.get(station_name.upper())


def find_nearest_coops_station(station_config: dict, max_distance_km: float = 50) -> dict:
    """
    Find the nearest CO-OPS tide gauge station for a GNSS station.

    Args:
        station_config: Station configuration dictionary with latitude_deg and longitude_deg
        max_distance_km: Maximum search radius in kilometers

    Returns:
        dict with station info: {'id', 'name', 'latitude', 'longitude', 'distance_km'}
        or None if no station found
    """
    lat = station_config.get('latitude_deg')
    lon = station_config.get('longitude_deg')

    if lat is None or lon is None:
        logging.error("Station config missing latitude_deg or longitude_deg")
        return None

    logging.info(f"Searching for CO-OPS stations within {max_distance_km} km of ({lat}, {lon})")

    client = NOAACOOPSClient()
    nearby_stations = client.find_nearby_stations(lat, lon, radius_km=max_distance_km)

    if not nearby_stations:
        logging.warning(f"No CO-OPS stations found within {max_distance_km} km")
        return None

    # Return the nearest station
    nearest = nearby_stations[0]
    logging.info(f"Found nearest CO-OPS station: {nearest['id']} - {nearest['name']} "
                 f"({nearest['distance_km']:.1f} km away)")

    return nearest


def get_coops_station_info(station_config: dict) -> dict:
    """
    Get CO-OPS station info from config or by auto-discovery.

    Args:
        station_config: Station configuration dictionary

    Returns:
        dict with station info: {'id', 'name', 'latitude', 'longitude', 'distance_km'}
        or None if no station found
    """
    coops_config = station_config.get('external_data_sources', {}).get('noaa_coops', {})

    # Get GNSS station coordinates for distance calculation
    gnss_lat = station_config.get('latitude_deg')
    gnss_lon = station_config.get('longitude_deg')

    # Check preferred stations first
    preferred = coops_config.get('preferred_stations', [])
    if preferred:
        station_id = preferred[0]
        # Get full metadata for the station
        client = NOAACOOPSClient()
        metadata = client.get_station_metadata(station_id)
        if metadata:
            coops_lat = metadata.get('latitude')
            coops_lon = metadata.get('longitude')
            # Calculate distance if we have both coordinates
            distance_km = None
            if gnss_lat and gnss_lon and coops_lat and coops_lon:
                distance_km = haversine_distance(gnss_lat, gnss_lon, coops_lat, coops_lon)
            return {
                'id': station_id,
                'name': metadata.get('name', f'CO-OPS {station_id}'),
                'latitude': coops_lat,
                'longitude': coops_lon,
                'distance_km': distance_km,
                'source': 'config_preferred'
            }
        else:
            return {
                'id': station_id,
                'name': f'CO-OPS {station_id}',
                'latitude': None,
                'longitude': None,
                'distance_km': None,
                'source': 'config_preferred'
            }

    # Check nearest station in config
    nearest = coops_config.get('nearest_station', {})
    if nearest.get('id'):
        return {
            'id': nearest['id'],
            'name': nearest.get('name', f'CO-OPS {nearest["id"]}'),
            'latitude': nearest.get('latitude'),
            'longitude': nearest.get('longitude'),
            'distance_km': nearest.get('distance_km'),
            'source': 'config_nearest'
        }

    # Auto-discover nearest station
    logging.info("No CO-OPS station in config, searching for nearest station...")
    discovered = find_nearest_coops_station(station_config)
    if discovered:
        discovered['source'] = 'auto_discovered'
        return discovered

    return None


def coops_comparison(station_name: str, year: int, doy_range: tuple = None,
                     max_lag_days: int = 10, output_dir: Path = None,
                     perform_segmented_analysis: bool = True) -> dict:
    """
    Run CO-OPS comparison analysis with time lag analysis.

    Args:
        station_name: Station name in uppercase (e.g., "FORA")
        year: Year to analyze
        doy_range: Range of DOYs to include (start, end)
        max_lag_days: Maximum lag to consider in days
        output_dir: Directory to save output files
        perform_segmented_analysis: Whether to perform monthly/seasonal analysis

    Returns:
        dict: Analysis results and paths to output files
    """
    year = str(year)

    # Get station configuration
    station_config = load_station_config(station_name)
    if station_config is None:
        return {'success': False, 'error': f'Station {station_name} not found in config'}

    # Check for ellipsoidal height
    if 'ellipsoidal_height_m' not in station_config:
        logging.error(f"Station {station_name} missing ellipsoidal_height_m")
        return {'success': False, 'error': 'Missing ellipsoidal_height_m in config'}

    antenna_height = station_config['ellipsoidal_height_m']

    # Get CO-OPS station info (from config or auto-discovery)
    coops_station_info = get_coops_station_info(station_config)
    if coops_station_info is None:
        logging.error(f"No CO-OPS station found for {station_name}")
        return {'success': False, 'error': 'No CO-OPS station found (checked config and auto-discovery)'}

    coops_station_id = coops_station_info['id']
    coops_station_name = coops_station_info.get('name', f'CO-OPS {coops_station_id}')
    coops_distance_km = coops_station_info.get('distance_km')
    coops_source = coops_station_info.get('source', 'unknown')

    # Get CO-OPS config settings
    coops_config = station_config.get('external_data_sources', {}).get('noaa_coops', {})
    datum = coops_config.get('datum', 'NAVD88')

    logging.info(f"Using CO-OPS station {coops_station_id} ({coops_station_name})")
    logging.info(f"  Source: {coops_source}")
    if coops_distance_km is not None:
        logging.info(f"  Distance from GNSS station: {coops_distance_km:.1f} km")
    logging.info(f"  Datum: {datum}")
    logging.info(f"  Antenna ellipsoidal height: {antenna_height} m")

    # Set up output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results_annual" / station_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert DOY range to dates
    if doy_range:
        doy_start, doy_end = doy_range
        start_date = datetime.strptime(f"{year}-{doy_start}", "%Y-%j")
        end_date = datetime.strptime(f"{year}-{doy_end}", "%Y-%j")
    else:
        start_date = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
        end_date = datetime.strptime(f"{year}-12-31", "%Y-%m-%d")

    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Load GNSS-IR data
    gnssir_df = load_gnssir_data(station_name, year, doy_range)
    if gnssir_df is None or gnssir_df.empty:
        logging.error(f"Failed to load GNSS-IR data for {station_name}")
        return {'success': False, 'error': 'Failed to load GNSS-IR data'}

    logging.info(f"Loaded {len(gnssir_df)} GNSS-IR daily records")

    # Calculate water surface elevation from reflector heights
    gnssir_df = reflector_height_utils.calculate_wse_from_rh(gnssir_df, antenna_height)

    # Fetch CO-OPS water level data
    logging.info(f"Fetching CO-OPS data for station {coops_station_id}")
    client = NOAACOOPSClient()

    try:
        coops_df = client.get_water_level_observations(
            station_id=coops_station_id,
            start_date=start_date,
            end_date=end_date,
            datum=datum,
            interval='h',  # Hourly data
            units='metric'
        )
    except Exception as e:
        logging.error(f"Failed to fetch CO-OPS data: {e}")
        return {'success': False, 'error': f'CO-OPS API error: {e}'}

    if coops_df is None or coops_df.empty:
        logging.error(f"No CO-OPS data retrieved for station {coops_station_id}")
        return {'success': False, 'error': 'No CO-OPS data retrieved'}

    logging.info(f"Retrieved {len(coops_df)} CO-OPS observations")

    # Process CO-OPS data - ensure datetime column and water level
    if 't' in coops_df.columns:
        coops_df['datetime'] = pd.to_datetime(coops_df['t'])
    elif 'time' in coops_df.columns:
        coops_df['datetime'] = pd.to_datetime(coops_df['time'])

    if 'v' in coops_df.columns:
        coops_df['water_level_m'] = pd.to_numeric(coops_df['v'], errors='coerce')
    elif 'water_level' in coops_df.columns:
        coops_df['water_level_m'] = pd.to_numeric(coops_df['water_level'], errors='coerce')

    # Aggregate CO-OPS to daily
    coops_df['date'] = coops_df['datetime'].dt.date
    coops_daily = coops_df.groupby('date').agg({
        'water_level_m': ['mean', 'std', 'count']
    }).reset_index()
    coops_daily.columns = ['date', 'coops_wl_mean', 'coops_wl_std', 'coops_count']
    coops_daily['date'] = pd.to_datetime(coops_daily['date'])

    logging.info(f"Aggregated to {len(coops_daily)} daily CO-OPS records")

    # Ensure GNSS-IR date column is datetime
    if 'date' in gnssir_df.columns:
        gnssir_df['date'] = pd.to_datetime(gnssir_df['date'])

    # Merge datasets
    merged_df = pd.merge(
        gnssir_df,
        coops_daily,
        on='date',
        how='inner'
    )

    if merged_df.empty:
        logging.error("No overlapping data between GNSS-IR and CO-OPS")
        return {'success': False, 'error': 'No overlapping data'}

    logging.info(f"Merged dataset: {len(merged_df)} matching days")

    # Demean both series for comparison
    gnssir_col = 'wse_ellips_m' if 'wse_ellips_m' in merged_df.columns else 'rh_median_m'
    merged_df['gnssir_dm'] = merged_df[gnssir_col] - merged_df[gnssir_col].mean()
    merged_df['coops_dm'] = merged_df['coops_wl_mean'] - merged_df['coops_wl_mean'].mean()

    # Calculate correlation
    correlation = merged_df['gnssir_dm'].corr(merged_df['coops_dm'])

    # Calculate RMSE
    residuals = merged_df['gnssir_dm'] - merged_df['coops_dm']
    rmse = np.sqrt((residuals ** 2).mean())

    logging.info(f"Correlation: {correlation:.4f}")
    logging.info(f"RMSE: {rmse:.3f} m")

    # Save comparison data
    comparison_csv = output_dir / f"{station_name}_{year}_coops_comparison.csv"
    merged_df.to_csv(comparison_csv, index=False)
    logging.info(f"Saved comparison data: {comparison_csv}")

    # Create comparison plot
    try:
        gauge_info = {
            'site_code': coops_station_id,
            'site_name': coops_station_name,
            'datum': datum,
            'latitude': coops_station_info.get('latitude'),
            'longitude': coops_station_info.get('longitude'),
            'distance_km': coops_distance_km,
            'source': coops_source,
            'gnss_lat': station_config.get('latitude_deg'),
            'gnss_lon': station_config.get('longitude_deg')
        }

        plot_path = output_dir / f"{station_name}_{year}_coops_comparison.png"
        plot_result = visualizer.plot_comparison_timeseries(
            merged_df, merged_df,
            station_name, gauge_info, plot_path,
            gnssir_rh_col='gnssir_dm',
            usgs_wl_col='coops_dm',
            compare_demeaned=True
        )
        if plot_result is not None:
            logging.info(f"Saved comparison plot: {plot_path}")
        else:
            plot_path = None
    except Exception as e:
        logging.warning(f"Could not create comparison plot: {e}")
        plot_path = None

    # Build results
    results = {
        'success': True,
        'station': station_name,
        'year': year,
        'coops_station': coops_station_id,
        'coops_station_name': coops_station_name,
        'coops_distance_km': coops_distance_km,
        'coops_source': coops_source,
        'datum': datum,
        'correlation': correlation,
        'rmse': rmse,
        'n_days': len(merged_df),
        'comparison_csv_path': str(comparison_csv),
        'plot_path': str(plot_path) if plot_path else None,
        'gauge_info': gauge_info
    }

    # Segmented analysis
    if perform_segmented_analysis and len(merged_df) >= 30:
        logging.info("Performing segmented correlation analysis...")
        try:
            # Monthly segments
            monthly_segments = generate_monthly_segments(int(year))
            monthly_correlations = perform_segmented_correlation(
                merged_df, 'gnssir_dm', 'coops_dm', monthly_segments
            )

            # Seasonal segments
            seasonal_segments = generate_seasonal_segments(int(year))
            seasonal_correlations = perform_segmented_correlation(
                merged_df, 'gnssir_dm', 'coops_dm', seasonal_segments
            )

            results['segmented_results'] = {
                'monthly_correlations': monthly_correlations,
                'seasonal_correlations': seasonal_correlations
            }

            logging.info("Segmented analysis complete")
        except Exception as e:
            logging.warning(f"Segmented analysis failed: {e}")

    return results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='CO-OPS Comparison for GNSS-IR')
    parser.add_argument('--station', type=str, required=True, help='Station ID (4-char uppercase)')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--doy_start', type=int, help='Starting day of year')
    parser.add_argument('--doy_end', type=int, help='Ending day of year')
    parser.add_argument('--max_lag_days', type=int, default=10, help='Maximum lag days')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--skip_segmented', action='store_true', help='Skip segmented analysis')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Build DOY range
    doy_range = None
    if args.doy_start and args.doy_end:
        doy_range = (args.doy_start, args.doy_end)

    # Run comparison
    results = coops_comparison(
        station_name=args.station,
        year=args.year,
        doy_range=doy_range,
        max_lag_days=args.max_lag_days,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        perform_segmented_analysis=not args.skip_segmented
    )

    # Report results
    if results.get('success'):
        print()
        print("=" * 70)
        print(f"CO-OPS Comparison Complete: {args.station} {args.year}")
        print("=" * 70)
        print(f"  CO-OPS Station: {results['coops_station']} ({results.get('coops_station_name', 'Unknown')})")
        if results.get('coops_distance_km') is not None:
            print(f"  Distance from GNSS: {results['coops_distance_km']:.1f} km")
        print(f"  Station Source: {results.get('coops_source', 'unknown')}")
        print(f"  Datum: {results['datum']}")
        print(f"  Matching Days: {results['n_days']}")
        print(f"  Correlation: {results['correlation']:.4f}")
        print(f"  RMSE: {results['rmse']:.3f} m")
        print()
        print(f"  Output: {results['comparison_csv_path']}")
        if results.get('plot_path'):
            print(f"  Plot: {results['plot_path']}")
        print("=" * 70)
    else:
        logging.error(f"Comparison failed: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
