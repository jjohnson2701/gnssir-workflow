#!/usr/bin/env python3
"""
Run GNSS-IR Processing

This script is a simplified entry point to run the GNSS-IR processing pipeline.
It provides a clean interface for running the processing workflow with default
or custom command-line arguments.
"""

import sys
import os
import argparse
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

# Add scripts directory to Python path to enable imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import required modules
from scripts.core_processing.config_loader import load_tool_paths, load_station_config
from scripts.utils.logging_config import setup_main_logger
from scripts.core_processing.parallel_orchestrator import process_station_parallel
import scripts.results_handler as results_handler

# Define default paths
DEFAULT_GNSSREFL_WORKSPACE = project_root / "gnssrefl_data_workspace"
DEFAULT_REFL_CODE_BASE = DEFAULT_GNSSREFL_WORKSPACE / "refl_code"
DEFAULT_ORBITS_BASE = DEFAULT_GNSSREFL_WORKSPACE / "orbits"
DEFAULT_STATIONS_CONFIG_PATH = project_root / "config" / "stations_config.json"
DEFAULT_TOOL_PATHS_CONFIG_PATH = project_root / "config" / "tool_paths.json"
DEFAULT_MAIN_LOG_FILE = project_root / "run_log.txt"

def main():
    """Main function to process GNSS-IR data"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process GNSS-IR data")
    
    # Basic processing parameters
    parser.add_argument("--station", default="FORA", help="Station name (default: FORA)")
    parser.add_argument("--year", type=int, default=2024, help="Year to process (default: 2024)")
    parser.add_argument("--doy_start", type=int, default=260, help="Starting day of year (default: 260)")
    parser.add_argument("--doy_end", type=int, default=266, help="Ending day of year (default: 266)")
    parser.add_argument("--num_cores", type=int, default=max(1, multiprocessing.cpu_count() - 2),
                        help=f"Number of cores to use (default: CPU count - 2 = {max(1, multiprocessing.cpu_count() - 2)})")
    
    # Skip flags for various processing stages
    parser.add_argument("--skip_download", action="store_true", 
                        help="Skip S3 download if RINEX 3 file already exists")
    parser.add_argument("--skip_rinex_conversion", action="store_true", 
                        help="Skip RINEX 3 to RINEX 2.11 conversion if RINEX 2.11 file already exists")
    parser.add_argument("--skip_snr", action="store_true", 
                        help="Skip rinex2snr step if SNR file already exists")
    parser.add_argument("--skip_rh", action="store_true", 
                        help="Skip gnssir step if reflector height file already exists")
    parser.add_argument("--skip_quicklook", action="store_true", 
                        help="Skip quickLook step if quickLook plots already exist")
    parser.add_argument("--skip_all", action="store_true", 
                        help="Skip all steps if their output files exist (equivalent to setting all individual skip flags)")
    
    # Configuration and logging options
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Logging level (default: INFO)")
    parser.add_argument("--workspace_dir", type=str, default=None,
                        help=f"GNSS-IR workspace directory (default: {DEFAULT_GNSSREFL_WORKSPACE})")
    parser.add_argument("--stations_config", type=str, default=None,
                        help=f"Stations configuration file (default: {DEFAULT_STATIONS_CONFIG_PATH})")
    parser.add_argument("--tool_paths_config", type=str, default=None,
                        help=f"Tool paths configuration file (default: {DEFAULT_TOOL_PATHS_CONFIG_PATH})")
    parser.add_argument("--log_file", type=str, default=None,
                        help=f"Main log file (default: {DEFAULT_MAIN_LOG_FILE})")
    
    args = parser.parse_args()
    
    # If skip_all is set, set all individual skip flags
    if args.skip_all:
        args.skip_download = True
        args.skip_rinex_conversion = True
        args.skip_snr = True
        args.skip_rh = True
        args.skip_quicklook = True
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)
    
    # Set up paths
    main_log_file = DEFAULT_MAIN_LOG_FILE if args.log_file is None else Path(args.log_file)
    stations_config_path = DEFAULT_STATIONS_CONFIG_PATH if args.stations_config is None else Path(args.stations_config)
    tool_paths_config_path = DEFAULT_TOOL_PATHS_CONFIG_PATH if args.tool_paths_config is None else Path(args.tool_paths_config)
    
    # Set up workspace directories
    if args.workspace_dir is None:
        gnssrefl_workspace = DEFAULT_GNSSREFL_WORKSPACE
        refl_code_base = DEFAULT_REFL_CODE_BASE
        orbits_base = DEFAULT_ORBITS_BASE
    else:
        gnssrefl_workspace = Path(args.workspace_dir)
        refl_code_base = gnssrefl_workspace / "refl_code"
        orbits_base = gnssrefl_workspace / "orbits"
    
    # Setup main logger
    setup_main_logger(main_log_file, log_level)
    
    logging.info("=" * 80)
    logging.info(f"Starting GNSS-IR Processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Station: {args.station}, Year: {args.year}, DOY range: {args.doy_start}-{args.doy_end}")
    logging.info(f"Using {args.num_cores} cores for parallel processing")
    logging.info(f"GNSS-IR workspace: {gnssrefl_workspace}")
    logging.info(f"Log level: {args.log_level}")
    
    # Log which steps will be skipped if files exist
    skip_msgs = []
    if args.skip_download:
        skip_msgs.append("S3 download")
    if args.skip_rinex_conversion:
        skip_msgs.append("RINEX conversion")
    if args.skip_snr:
        skip_msgs.append("rinex2snr")
    if args.skip_rh:
        skip_msgs.append("gnssir")
    if args.skip_quicklook:
        skip_msgs.append("quickLook")
    
    if skip_msgs:
        logging.info(f"The following steps will be skipped if output files exist: {', '.join(skip_msgs)}")
    logging.info("=" * 80)
    
    # Load tool paths configuration
    tool_paths = load_tool_paths(tool_paths_config_path, project_root)
    
    # Load station-specific configuration
    station_config = load_station_config(stations_config_path, args.station.upper(), project_root)
    
    # Process station in parallel
    results = process_station_parallel(
        station_config=station_config,
        year=args.year,
        doy_range=(args.doy_start, args.doy_end),
        tool_paths=tool_paths,
        project_root=project_root,
        refl_code_base=refl_code_base,
        orbits_base=orbits_base,
        num_cores=args.num_cores,
        results_handler=results_handler,
        skip_options={
            'skip_download': args.skip_download,
            'skip_rinex_conversion': args.skip_rinex_conversion,
            'skip_snr': args.skip_snr,
            'skip_rh': args.skip_rh,
            'skip_quicklook': args.skip_quicklook
        }
    )
    
    # Log final summary
    logging.info("=" * 80)
    logging.info(f"GNSS-IR Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total DOYs attempted: {results['attempted']}")
    logging.info(f"Total DOYs successful: {len(results['successful'])}")
    logging.info(f"Total DOYs failed: {len(results['failed'])}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
