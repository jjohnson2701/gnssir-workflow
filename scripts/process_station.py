#!/usr/bin/env python3
# ABOUTME: Unified station processing script - runs full workflow from config.
# ABOUTME: Handles all steps: GNSS-IR processing, reference matching, and visualization.

"""
Unified GNSS-IR Station Processing Script

This script runs the complete processing workflow for a station based on
configuration in stations_config.json:

1. Core GNSS-IR Processing (run_gnssir_processing.py)
2. Reference Data Matching (auto-selects USGS or ERDDAP based on config)
3. Visualization (resolution comparison, polar animation)

Usage:
    python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

The script determines the appropriate reference source from config:
- If erddap.primary_reference=true -> uses generate_erddap_matched.py
- Otherwise -> uses usgs_comparison.py
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(station: str, config_path: Path) -> dict:
    """Load station configuration."""
    with open(config_path) as f:
        config = json.load(f)

    if station not in config:
        available = list(config.keys())
        raise ValueError(f"Station {station} not found. Available: {available}")

    return config[station]


def get_reference_source(station_config: dict) -> str:
    """Determine primary reference source from config."""
    # Check ERDDAP first
    erddap = station_config.get('erddap', {})
    if not erddap:
        erddap = station_config.get('external_data_sources', {}).get('erddap', {})

    if erddap.get('primary_reference', False) or erddap.get('enabled', False):
        return 'erddap'

    # Check USGS
    usgs = station_config.get('usgs_comparison', {})
    if usgs.get('target_usgs_site'):
        return 'usgs'

    # Check CO-OPS
    coops = station_config.get('external_data_sources', {}).get('noaa_coops', {})
    if coops.get('enabled', False):
        return 'coops'

    return 'none'


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"  ✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Process GNSS-IR station through complete workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full year
  python scripts/process_station.py --station GLBX --year 2024

  # Process specific date range
  python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

  # Skip GNSS-IR processing (only run comparison/visualization)
  python scripts/process_station.py --station GLBX --year 2024 --skip_gnssir

  # Skip visualization
  python scripts/process_station.py --station GLBX --year 2024 --skip_viz
        """
    )
    parser.add_argument('--station', type=str, required=True, help='Station ID (e.g., GLBX, FORA)')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--doy_start', type=int, default=1, help='Start day of year (default: 1)')
    parser.add_argument('--doy_end', type=int, default=366, help='End day of year (default: 366)')
    parser.add_argument('--num_cores', type=int, default=8, help='Number of parallel cores (default: 8)')
    parser.add_argument('--skip_download', action='store_true', help='Skip RINEX download (data already present)')
    parser.add_argument('--skip_gnssir', action='store_true', help='Skip GNSS-IR processing (use existing results)')
    parser.add_argument('--skip_comparison', action='store_true', help='Skip reference comparison')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualization scripts')
    parser.add_argument('--config', type=str, default=None, help='Path to stations_config.json')
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / 'config' / 'stations_config.json'

    python_exe = sys.executable

    print("="*70)
    print(f"GNSS-IR Processing Workflow: {args.station} {args.year}")
    print("="*70)
    print()

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    station_config = load_config(args.station, config_path)

    # Determine reference source
    ref_source = get_reference_source(station_config)
    logger.info(f"Primary reference source: {ref_source.upper()}")

    if ref_source == 'none':
        print()
        print("="*70)
        print("WARNING: No reference station configured for this station!")
        print("="*70)
        print()
        print("To find and configure a reference station, run:")
        print(f"  python scripts/find_reference_stations.py --station {args.station}")
        print()
        print("Then add the reference to config/stations_config.json under:")
        print("  - 'usgs_comparison.target_usgs_site' for USGS gauges")
        print("  - 'external_data_sources.noaa_coops' for CO-OPS tide gauges")
        print("  - 'erddap' for ERDDAP data sources")
        print()
        print("Processing will continue, but Phase 2 (Reference Matching) will be skipped.")
        print("="*70)

    print()

    # Track results
    results = {}

    # =========================================================================
    # Step 1-7: Core GNSS-IR Processing
    # =========================================================================
    if not args.skip_gnssir:
        print("-"*70)
        print("PHASE 1: Core GNSS-IR Processing")
        print("-"*70)

        cmd = [
            python_exe, str(script_dir / 'run_gnssir_processing.py'),
            '--station', args.station,
            '--year', str(args.year),
            '--doy_start', str(args.doy_start),
            '--doy_end', str(args.doy_end),
            '--num_cores', str(args.num_cores)
        ]
        if args.skip_download:
            cmd.append('--skip_download')

        results['gnssir'] = run_command(cmd, "GNSS-IR Processing")
        print()
    else:
        logger.info("Skipping GNSS-IR processing (--skip_gnssir)")
        results['gnssir'] = True

    # =========================================================================
    # Step 8: Reference Data Matching
    # =========================================================================
    if not args.skip_comparison:
        print("-"*70)
        print("PHASE 2: Reference Data Matching")
        print("-"*70)

        if ref_source == 'erddap':
            cmd = [
                python_exe, str(script_dir / 'generate_erddap_matched.py'),
                '--station', args.station,
                '--year', str(args.year)
            ]
            results['comparison'] = run_command(cmd, "ERDDAP Matching")
        elif ref_source == 'usgs':
            cmd = [
                python_exe, str(script_dir / 'usgs_comparison.py'),
                '--station', args.station,
                '--year', str(args.year)
            ]
            results['comparison'] = run_command(cmd, "USGS Comparison")
        elif ref_source == 'coops':
            cmd = [
                python_exe, str(script_dir / 'coops_comparison.py'),
                '--station', args.station,
                '--year', str(args.year)
            ]
            results['comparison'] = run_command(cmd, "CO-OPS Comparison")
        else:
            logger.warning(f"No reference source configured for {args.station}")
            results['comparison'] = False
        print()
    else:
        logger.info("Skipping comparison (--skip_comparison)")
        results['comparison'] = True

    # =========================================================================
    # Step 9: Visualization
    # =========================================================================
    if not args.skip_viz:
        print("-"*70)
        print("PHASE 3: Visualization")
        print("-"*70)

        # Resolution comparison
        cmd = [
            python_exe, str(script_dir / 'plot_resolution_comparison.py'),
            '--station', args.station,
            '--year', str(args.year)
        ]
        results['resolution_plot'] = run_command(cmd, "Resolution Comparison Plot")

        # Polar animation
        cmd = [
            python_exe, str(script_dir / 'create_polar_animation.py'),
            '--station', args.station,
            '--year', str(args.year),
            '--doy_start', str(args.doy_start),
            '--doy_end', str(args.doy_end)
        ]
        results['polar_animation'] = run_command(cmd, "Polar Animation")
        print()
    else:
        logger.info("Skipping visualization (--skip_viz)")
        results['resolution_plot'] = True
        results['polar_animation'] = True

    # =========================================================================
    # Summary
    # =========================================================================
    print("="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print()
    print("Results:")
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step}")

    print()
    print("Output files:")
    results_dir = project_root / 'results_annual' / args.station
    print(f"  {results_dir}/")
    for f in sorted(results_dir.glob(f"{args.station}_{args.year}*")):
        print(f"    - {f.name}")

    print()
    print("Next step: Run the dashboard")
    print(f"  streamlit run dashboard.py")

    # Exit with error if any step failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
