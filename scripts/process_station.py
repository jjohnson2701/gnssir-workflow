#!/usr/bin/env python3
# ABOUTME: Single entry point for GNSS-IR station processing workflow.
# ABOUTME: Handles GNSS-IR processing, reference matching, and visualization.

"""
GNSS-IR Station Processing Script

This script runs the complete processing workflow for a station based on
configuration in stations_config.json:

1. Core GNSS-IR Processing (RINEX download, conversion, SNR extraction, RH retrieval)
2. Reference Data Matching (auto-selects USGS, ERDDAP, or CO-OPS based on config)
3. Visualization (resolution comparison, polar animation)

Usage:
    python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

The script determines the appropriate reference source from config:
- If erddap.primary_reference=true -> uses generate_erddap_matched.py
- Otherwise -> uses usgs_comparison.py or coops_comparison.py
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to Python path to enable imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from scripts.core_processing.config_loader import load_tool_paths, load_station_config
from scripts.utils.logging_config import setup_main_logger
from scripts.core_processing.parallel_orchestrator import process_station_parallel
import scripts.results_handler as results_handler

# Default paths
DEFAULT_GNSSREFL_WORKSPACE = project_root / "gnssrefl_data_workspace"
DEFAULT_REFL_CODE_BASE = DEFAULT_GNSSREFL_WORKSPACE / "refl_code"
DEFAULT_ORBITS_BASE = DEFAULT_GNSSREFL_WORKSPACE / "orbits"
DEFAULT_STATIONS_CONFIG_PATH = project_root / "config" / "stations_config.json"
DEFAULT_TOOL_PATHS_CONFIG_PATH = project_root / "config" / "tool_paths.json"
DEFAULT_MAIN_LOG_FILE = project_root / "run_log.txt"

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


def get_reference_source(station_config: dict) -> tuple:
    """
    Determine primary reference source from config.

    Returns:
        tuple: (source_name, reason) - e.g., ('erddap', 'primary_reference=true')
    """
    # Check ERDDAP first (highest priority)
    erddap = station_config.get('erddap', {})
    if not erddap:
        erddap = station_config.get('external_data_sources', {}).get('erddap', {})

    if erddap.get('primary_reference', False):
        return ('erddap', 'erddap.primary_reference=true')
    if erddap.get('enabled', False):
        return ('erddap', 'erddap.enabled=true')

    # Check USGS (second priority)
    usgs = station_config.get('usgs_comparison', {})
    if usgs.get('target_usgs_site'):
        site_id = usgs.get('target_usgs_site')
        return ('usgs', f'usgs_comparison.target_usgs_site={site_id}')

    # Check CO-OPS (third priority)
    coops = station_config.get('external_data_sources', {}).get('noaa_coops', {})
    if coops.get('enabled', False):
        preferred = coops.get('preferred_stations', [])
        if preferred:
            return ('coops', f'noaa_coops.enabled with preferred_stations={preferred[0]}')
        return ('coops', 'noaa_coops.enabled=true (will auto-discover)')

    return ('none', 'no reference source configured')


def validate_configuration(station: str, station_config: dict, project_root: Path) -> dict:
    """
    Validate station configuration before processing.

    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []
    warnings = []

    # Check gnssir params file exists
    params_path = station_config.get('gnssir_json_params_path')
    if params_path:
        full_path = project_root / params_path
        if not full_path.exists():
            issues.append(f"GNSS-IR params file not found: {params_path}")
        else:
            # Validate params file content
            try:
                with open(full_path) as f:
                    params = json.load(f)
                required = ['lat', 'lon', 'ht', 'azval2', 'freqs']
                for key in required:
                    if key not in params:
                        issues.append(f"Missing required key '{key}' in {params_path}")
                # Validate parameter ranges
                if 'minH' in params and 'maxH' in params:
                    if params['minH'] >= params['maxH']:
                        issues.append(f"Invalid reflector height range: minH ({params['minH']}) >= maxH ({params['maxH']})")
                if 'e1' in params and 'e2' in params:
                    if params['e1'] >= params['e2']:
                        issues.append(f"Invalid elevation angle range: e1 ({params['e1']}) >= e2 ({params['e2']})")
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {params_path}: {e}")
    else:
        issues.append("Missing 'gnssir_json_params_path' in station config")

    # Check coordinates
    lat = station_config.get('latitude_deg')
    lon = station_config.get('longitude_deg')
    if lat is None or lon is None:
        issues.append("Missing latitude_deg or longitude_deg in station config")

    # Check reference source
    ref_source, ref_reason = get_reference_source(station_config)
    if ref_source == 'none':
        warnings.append("No reference source configured - run find_reference_stations.py")

    # Check tool paths and executability
    tool_paths_file = project_root / 'config' / 'tool_paths.json'
    tool_paths = {}
    if tool_paths_file.exists():
        with open(tool_paths_file) as f:
            tool_paths = json.load(f)

    required_tools = {
        'gfzrnx_path': 'gfzrnx',
        'rinex2snr_path': 'rinex2snr',
        'gnssir_path': 'gnssir'
    }

    for config_key, tool_name in required_tools.items():
        tool_path = tool_paths.get(config_key, tool_name)
        # Check if it's an absolute path that exists, or findable in PATH
        if os.path.isabs(tool_path):
            if not os.path.isfile(tool_path):
                issues.append(f"Tool not found: {tool_path}")
            elif not os.access(tool_path, os.X_OK):
                issues.append(f"Tool not executable: {tool_path}")
        else:
            found = shutil.which(tool_path)
            if not found:
                issues.append(f"Tool '{tool_path}' not found in PATH")

    # Check REFL_CODE workspace is writable
    refl_code = os.environ.get('REFL_CODE', str(project_root / 'gnssrefl_data_workspace' / 'refl_code'))
    refl_code_path = Path(refl_code)
    if refl_code_path.exists():
        if not os.access(refl_code_path, os.W_OK):
            issues.append(f"REFL_CODE directory not writable: {refl_code_path}")
    else:
        # Will be created, check parent is writable
        parent = refl_code_path.parent
        if parent.exists() and not os.access(parent, os.W_OK):
            warnings.append(f"Cannot create REFL_CODE directory (parent not writable): {parent}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'ref_source': ref_source,
        'ref_reason': ref_reason
    }


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
  # Process today's data (for cron jobs)
  python scripts/process_station.py --station GLBX --today

  # Process last 7 days (catches up after gaps)
  python scripts/process_station.py --station GLBX --days 7

  # Process full year
  python scripts/process_station.py --station GLBX --year 2024

  # Process specific date range
  python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

  # Validate configuration before processing
  python scripts/process_station.py --station GLBX --year 2024 --validate
        """
    )
    parser.add_argument('--station', type=str, required=True, help='Station ID (e.g., GLBX, FORA)')
    parser.add_argument('--year', type=int, default=None, help='Year to process (default: current year)')
    parser.add_argument('--doy_start', type=int, default=None, help='Start day of year')
    parser.add_argument('--doy_end', type=int, default=None, help='End day of year')
    parser.add_argument('--today', action='store_true', help='Process today only (for cron jobs)')
    parser.add_argument('--days', type=int, default=None, help='Process last N days (handles year boundaries)')
    parser.add_argument('--num_cores', type=int, default=8, help='Number of parallel cores (default: 8)')
    parser.add_argument('--skip_download', action='store_true', help='Skip RINEX download (data already present)')
    parser.add_argument('--skip_gnssir', action='store_true', help='Skip GNSS-IR processing (use existing results)')
    parser.add_argument('--skip_comparison', action='store_true', help='Skip reference comparison')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualization scripts')
    parser.add_argument('--config', type=str, default=None, help='Path to stations_config.json')
    parser.add_argument('--validate', action='store_true', help='Validate configuration and exit without processing')
    args = parser.parse_args()

    # Handle date convenience flags
    today = datetime.now()
    if args.today:
        args.year = today.year
        args.doy_start = today.timetuple().tm_yday
        args.doy_end = args.doy_start
    elif args.days:
        from datetime import timedelta
        end_date = today
        start_date = today - timedelta(days=args.days - 1)
        # Handle year boundary - process each year separately if needed
        if start_date.year != end_date.year:
            # For simplicity, just process from Jan 1 of current year
            # A more complete solution would loop over years
            logger.warning(f"Date range spans year boundary. Processing {end_date.year} only.")
            args.year = end_date.year
            args.doy_start = 1
            args.doy_end = end_date.timetuple().tm_yday
        else:
            args.year = end_date.year
            args.doy_start = start_date.timetuple().tm_yday
            args.doy_end = end_date.timetuple().tm_yday
    elif args.year is None:
        args.year = today.year

    # Set defaults for doy if not specified
    if args.doy_start is None:
        args.doy_start = 1
    if args.doy_end is None:
        args.doy_end = 366

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

    # Validate configuration
    validation = validate_configuration(args.station, station_config, project_root)

    if args.validate:
        print("Configuration Validation Results")
        print("-" * 40)
        if validation['valid']:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has errors:")
            for issue in validation['issues']:
                print(f"  - {issue}")

        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        print(f"\nReference source: {validation['ref_source'].upper()}")
        print(f"  Reason: {validation['ref_reason']}")
        sys.exit(0 if validation['valid'] else 1)

    # Exit early if config is invalid
    if not validation['valid']:
        logger.error("Configuration validation failed:")
        for issue in validation['issues']:
            logger.error(f"  - {issue}")
        sys.exit(1)

    # Determine reference source
    ref_source = validation['ref_source']
    ref_reason = validation['ref_reason']
    logger.info(f"Primary reference source: {ref_source.upper()} ({ref_reason})")

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
    # Phase 1: Core GNSS-IR Processing
    # =========================================================================
    if not args.skip_gnssir:
        print("-"*70)
        print("PHASE 1: Core GNSS-IR Processing")
        print("-"*70)

        try:
            # Setup logging for GNSS-IR processing
            setup_main_logger(DEFAULT_MAIN_LOG_FILE, logging.INFO)

            logging.info("=" * 80)
            logging.info(f"Starting GNSS-IR Processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Station: {args.station}, Year: {args.year}, DOY range: {args.doy_start}-{args.doy_end}")
            logging.info(f"Using {args.num_cores} cores for parallel processing")
            logging.info("=" * 80)

            # Load tool paths configuration
            tool_paths = load_tool_paths(DEFAULT_TOOL_PATHS_CONFIG_PATH, project_root)

            # Load station-specific configuration for processing
            processing_config = load_station_config(DEFAULT_STATIONS_CONFIG_PATH, args.station.upper(), project_root)

            # Process station in parallel
            gnssir_results = process_station_parallel(
                station_config=processing_config,
                year=args.year,
                doy_range=(args.doy_start, args.doy_end),
                tool_paths=tool_paths,
                project_root=project_root,
                refl_code_base=DEFAULT_REFL_CODE_BASE,
                orbits_base=DEFAULT_ORBITS_BASE,
                num_cores=args.num_cores,
                results_handler=results_handler,
                skip_options={
                    'skip_download': args.skip_download,
                    'skip_rinex_conversion': False,
                    'skip_snr': False,
                    'skip_rh': False,
                    'skip_quicklook': False
                }
            )

            logging.info("=" * 80)
            logging.info(f"GNSS-IR Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Total DOYs attempted: {gnssir_results['attempted']}")
            logging.info(f"Total DOYs successful: {len(gnssir_results['successful'])}")
            logging.info(f"Total DOYs failed: {len(gnssir_results['failed'])}")
            logging.info("=" * 80)

            results['gnssir'] = len(gnssir_results['successful']) > 0
            logger.info(f"  ✓ GNSS-IR Processing completed: {len(gnssir_results['successful'])}/{gnssir_results['attempted']} DOYs successful")
        except Exception as e:
            logger.error(f"  ✗ GNSS-IR Processing failed: {e}")
            results['gnssir'] = False
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
