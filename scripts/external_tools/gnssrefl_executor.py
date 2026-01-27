"""
GNSSREFL Executor module for GNSS-IR processing.
This module wraps gnssrefl command-line tools execution.
"""

import os
import logging
import subprocess
import glob
import shutil
import threading
import time
from pathlib import Path

def execute_rinex2snr(rinex2snr_exe_path, station_4char_lower, year, doy_padded, 
                    refl_code_base, orbits_base, logs_dir, 
                    snr_code="66", rinex2_filename_override=None):
    """
    Execute rinex2snr command to extract SNR data from RINEX 2 file.
    
    Args:
        rinex2snr_exe_path (str): Path to rinex2snr executable
        station_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        doy_padded (str): Day of year, zero-padded to 3 digits
        refl_code_base (str or Path): Base directory for REFL_CODE
        orbits_base (str or Path): Base directory for ORBITS
        logs_dir (str or Path): Directory for logs
        snr_code (str, optional): SNR code. Defaults to "66".
        rinex2_filename_override (str, optional): Override the default RINEX 2 filename.
    
    Returns:
        bool: True on success, False on failure
    """
    # Convert paths to Path objects
    refl_code_base = Path(refl_code_base)
    orbits_base = Path(orbits_base)
    logs_dir = Path(logs_dir)
    
    # Ensure directories exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    log_file = logs_dir / f"{station_4char_lower}_{year}_{doy_padded}.log"
    day_logger = logging.getLogger(str(log_file))
    
    try:
        # Construct the rinex2snr command
        cmd = [
            rinex2snr_exe_path,
            station_4char_lower,
            str(year),
            doy_padded,
            "-snr", snr_code,
            "-nolook", "T"
        ]
        
        # Add optional overrides
        if rinex2_filename_override:
            cmd.extend(["-rinex2", rinex2_filename_override])
        
        # Prepare environment variables for gnssrefl
        my_env = os.environ.copy()
        my_env["REFL_CODE"] = str(refl_code_base)
        my_env["ORBITS"] = str(orbits_base)
        
        # Working directory
        cwd_for_rinex2snr = str(refl_code_base)
        
        # Log the command, environment, and working directory
        day_logger.info(f"Running command: {' '.join(cmd)}")
        day_logger.info(f"Working directory: {cwd_for_rinex2snr}")
        day_logger.info(f"Environment variables:")
        day_logger.info(f"  REFL_CODE: {my_env['REFL_CODE']}")
        day_logger.info(f"  ORBITS: {my_env['ORBITS']}")
        
        # Execute rinex2snr with capture_output=True to capture stdout/stderr
        process_rinex2snr = subprocess.run(
            cmd,
            capture_output=True,
            text=True,          # Decodes stdout/stderr as text
            shell=False,
            cwd=cwd_for_rinex2snr,
            env=my_env
        )
        
        # Log the direct output from the command
        day_logger.info(f"rinex2snr stdout for DOY {doy_padded}:\n{process_rinex2snr.stdout}")
        if process_rinex2snr.stderr:
            day_logger.warning(f"rinex2snr stderr for DOY {doy_padded}:\n{process_rinex2snr.stderr}")
        
        # Check rinex2snr's own return code first
        if process_rinex2snr.returncode != 0:
            day_logger.error(f"rinex2snr failed for {station_4char_lower} {year} {doy_padded} with explicit return code {process_rinex2snr.returncode}")
            return False
        
        # Expected SNR file paths (both compressed and uncompressed)
        expected_snr_file = refl_code_base / str(year) / "snr" / station_4char_lower / f"{station_4char_lower}{doy_padded}0.{str(year)[-2:]}.snr{snr_code}"
        expected_snr_file_gz = Path(f"{expected_snr_file}.gz")
        
        # Check if either the compressed or uncompressed file exists
        if expected_snr_file.exists():
            day_logger.info(f"SNR file created at {expected_snr_file}")
            return True
        elif expected_snr_file_gz.exists():
            day_logger.info(f"Compressed SNR file created at {expected_snr_file_gz}")
            return True
        else:
            day_logger.error(f"Expected SNR file not found at {expected_snr_file} or {expected_snr_file_gz}")
            day_logger.error(f"rinex2snr returned success (code 0) but no SNR file was created")
            return False
        
    except Exception as e:
        day_logger.error(f"Error executing rinex2snr: {e}")
        return False

def execute_gnssir(gnssir_exe_path, station_4char_lower, year, doy_padded, 
                 refl_code_base, orbits_base, logs_dir):
    """
    Execute gnssir command to process SNR data and calculate reflector heights.
    
    Args:
        gnssir_exe_path (str): Path to gnssir executable
        station_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        doy_padded (str): Day of year, zero-padded to 3 digits
        refl_code_base (str or Path): Base directory for REFL_CODE
        orbits_base (str or Path): Base directory for ORBITS
        logs_dir (str or Path): Directory for logs
    
    Returns:
        bool: True on success, False on failure
    """
    # Convert paths to Path objects
    refl_code_base = Path(refl_code_base)
    orbits_base = Path(orbits_base)
    logs_dir = Path(logs_dir)
    
    # Ensure directories exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    log_file = logs_dir / f"{station_4char_lower}_{year}_{doy_padded}.log"
    day_logger = logging.getLogger(str(log_file))
    
    try:
        # Construct the gnssir command
        cmd = [
            gnssir_exe_path,
            station_4char_lower,
            str(year),
            doy_padded
        ]
        
        # Prepare environment variables for gnssrefl
        my_env = os.environ.copy()
        my_env["REFL_CODE"] = str(refl_code_base)
        my_env["ORBITS"] = str(orbits_base)
        
        # Working directory
        cwd_for_gnssir = str(refl_code_base)
        
        # Log the command, environment, and working directory
        day_logger.info(f"Running command: {' '.join(cmd)}")
        day_logger.info(f"Working directory: {cwd_for_gnssir}")
        day_logger.info(f"Environment variables:")
        day_logger.info(f"  REFL_CODE: {my_env['REFL_CODE']}")
        day_logger.info(f"  ORBITS: {my_env['ORBITS']}")
        
        # Execute gnssir with capture_output=True to capture stdout/stderr
        process_gnssir = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            cwd=cwd_for_gnssir,
            env=my_env
        )
        
        # Log the direct output from the command
        day_logger.info(f"gnssir stdout for DOY {doy_padded}:\n{process_gnssir.stdout}")
        if process_gnssir.stderr:
            day_logger.warning(f"gnssir stderr for DOY {doy_padded}:\n{process_gnssir.stderr}")
        
        # Check gnssir's own return code first
        if process_gnssir.returncode != 0:
            day_logger.error(f"gnssir failed for {station_4char_lower} {year} {doy_padded} with explicit return code {process_gnssir.returncode}")
            return False
        
        # Verify results file exists (check multiple potential locations)
        result_file_patterns = [
            refl_code_base / str(year) / "results" / station_4char_lower / f"{doy_padded}.txt",
            refl_code_base / str(year) / "results" / station_4char_lower / f"{station_4char_lower}_{doy_padded}.txt",
            refl_code_base / "Files" / f"{station_4char_lower}_{year}_{doy_padded}.txt"
        ]
        
        result_file_found = False
        for file_path in result_file_patterns:
            if file_path.exists():
                day_logger.info(f"Result file found at {file_path}")
                result_file_found = True
                break
        
        if not result_file_found:
            # Search more broadly if exact patterns don't match
            possible_results_dirs = [
                refl_code_base / str(year) / "results" / station_4char_lower,
                refl_code_base / "Files"
            ]
            
            for results_dir in possible_results_dirs:
                if results_dir.exists():
                    txt_files = list(results_dir.glob("*.txt"))
                    if txt_files:
                        day_logger.info(f"Found {len(txt_files)} text files in {results_dir}, assuming success")
                        day_logger.info(f"Files: {[f.name for f in txt_files]}")
                        result_file_found = True
                        break
        
        if not result_file_found:
            day_logger.error(f"No result files found in expected locations")
            day_logger.error(f"gnssir returned success (code 0) but no result file was created")
            return False
        
        day_logger.info(f"Successfully executed gnssir for {station_4char_lower} {year} {doy_padded}")
        return True
    
    except Exception as e:
        day_logger.error(f"Error executing gnssir: {e}")
        return False

def execute_quicklook_threaded(quicklook_exe_path, station_4char_lower, year, doy_padded, 
                          refl_code_base, orbits_base, quicklook_plots_daily_dir, logs_dir):
    """
    Non-blocking version of execute_quicklook that runs in a separate thread.
    
    Args:
        quicklook_exe_path (str): Path to quickLook executable
        station_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        doy_padded (str): Day of year, zero-padded to 3 digits
        refl_code_base (str or Path): Base directory for REFL_CODE
        orbits_base (str or Path): Base directory for ORBITS
        quicklook_plots_daily_dir (str or Path): Directory for daily quickLook plots
        logs_dir (str or Path): Directory for logs
    
    Returns:
        threading.Thread: Thread object running the quickLook process
    """
    # Setup log file
    log_file = Path(logs_dir) / f"{station_4char_lower}_{year}_{doy_padded}.log"
    day_logger = logging.getLogger(str(log_file))
    day_logger.info(f"Starting quickLook in background thread for {station_4char_lower} {year} {doy_padded}")
    
    # Create and start a thread for quickLook execution
    quicklook_thread = threading.Thread(
        target=execute_quicklook,
        args=(quicklook_exe_path, station_4char_lower, year, doy_padded, 
              refl_code_base, orbits_base, quicklook_plots_daily_dir, logs_dir),
        daemon=True  # Set as daemon so it doesn't block program exit
    )
    quicklook_thread.start()
    
    day_logger.info(f"quickLook thread started for {station_4char_lower} {year} {doy_padded}")
    
    return quicklook_thread

def execute_quicklook(quicklook_exe_path, station_4char_lower, year, doy_padded, 
                    refl_code_base, orbits_base, quicklook_plots_daily_dir, logs_dir):
    """
    Execute quickLook command to generate daily plots.
    
    Args:
        quicklook_exe_path (str): Path to quickLook executable
        station_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        doy_padded (str): Day of year, zero-padded to 3 digits
        refl_code_base (str or Path): Base directory for REFL_CODE
        orbits_base (str or Path): Base directory for ORBITS
        quicklook_plots_daily_dir (str or Path): Directory for daily quickLook plots
        logs_dir (str or Path): Directory for logs
    
    Returns:
        bool: True on success, False on failure
    """
    # Convert paths to Path objects
    refl_code_base = Path(refl_code_base)
    orbits_base = Path(orbits_base)
    quicklook_plots_daily_dir = Path(quicklook_plots_daily_dir)
    logs_dir = Path(logs_dir)
    
    # Ensure directories exist
    quicklook_plots_daily_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    log_file = logs_dir / f"{station_4char_lower}_{year}_{doy_padded}.log"
    day_logger = logging.getLogger(str(log_file))
    
    try:
        # Construct the quickLook command
        cmd = [
            quicklook_exe_path,
            station_4char_lower,
            str(year),
            doy_padded
        ]
        
        # Prepare environment variables for gnssrefl
        my_env = os.environ.copy()
        my_env["REFL_CODE"] = str(refl_code_base)
        my_env["ORBITS"] = str(orbits_base)
        
        # Working directory
        cwd_for_quicklook = str(refl_code_base)
        
        # Log the command, environment, and working directory
        day_logger.info(f"Running command: {' '.join(cmd)}")
        day_logger.info(f"Working directory: {cwd_for_quicklook}")
        day_logger.info(f"Environment variables:")
        day_logger.info(f"  REFL_CODE: {my_env['REFL_CODE']}")
        day_logger.info(f"  ORBITS: {my_env['ORBITS']}")
        
        # Execute quickLook with capture_output=True to capture stdout/stderr
        process_quicklook = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            cwd=cwd_for_quicklook,
            env=my_env
        )
        
        # Log the direct output from the command
        day_logger.info(f"quickLook stdout for DOY {doy_padded}:\n{process_quicklook.stdout}")
        if process_quicklook.stderr:
            day_logger.warning(f"quickLook stderr for DOY {doy_padded}:\n{process_quicklook.stderr}")
        
        # Check for station coordinates in the output
        if "Did not find station coordinates" in process_quicklook.stdout:
            day_logger.warning("Station coordinates warning detected in quickLook output")
        
        # Check quickLook's own return code
        if process_quicklook.returncode != 0:
            day_logger.error(f"quickLook failed for {station_4char_lower} {year} {doy_padded} with explicit return code {process_quicklook.returncode}")
            return False
        
        # Define multiple possible station directory names
        station_variants = [
            station_4char_lower,            # lowercase (e.g., 'fora')
            station_4char_lower.upper(),    # uppercase (e.g., 'FORA')
            station_4char_lower.capitalize() # capitalized (e.g., 'Fora')
        ]
        
        # Define potential plot locations to search, prioritizing primary locations first
        plot_locations = []
        
        # Primary: $REFL_CODE_BASE/Files/station_4char_variants/
        files_dir = refl_code_base / "Files"
        if files_dir.exists():
            for station_var in station_variants:
                station_files_dir = files_dir / station_var
                if station_files_dir.exists():
                    plot_locations.append(station_files_dir)
                    day_logger.info(f"Found primary station directory: {station_files_dir}")
        
        # Add other potential locations
        plot_locations.extend([
            files_dir,                      # Secondary: $REFL_CODE_BASE/Files/
            refl_code_base / str(year) / "plots" / station_4char_lower, # Year/plots/station
            refl_code_base                  # Root directory
        ])
        
        # Log all search directories
        day_logger.info(f"Searching for plot files in the following directories:")
        for idx, loc in enumerate(plot_locations):
            if loc.exists():
                day_logger.info(f"  {idx+1}. {loc}")
        
        # Define search patterns in order of specificity
        # First look for day-specific plots
        day_specific_patterns = [
            f"{station_4char_lower}_{doy_padded}*.png",            # station_doy.png  
            f"{station_4char_lower.upper()}_{doy_padded}*.png",    # STATION_doy.png
            f"{doy_padded}*.png"                                   # doy.png
        ]
        
        # Then look for standard plot names
        standard_patterns = [
            f"{station_4char_lower}_lsp.png",        # LSP plot
            f"{station_4char_lower}_summary.png",    # Summary plot
            f"quickLook_lsp.png",                    # Alternative LSP plot name
            f"quickLook_summary.png",                # Alternative summary plot name
        ]
        
        # Last resort, look for any PNG files
        fallback_patterns = ["*.png"]
        
        # Search patterns in order of priority
        search_patterns = day_specific_patterns + standard_patterns + fallback_patterns
        
        # Collect all plot files
        all_plot_files = []
        found_pattern_info = {}  # To keep track of what patterns were successful
        
        # Search each location with each pattern until files are found
        for plot_dir in plot_locations:
            if not plot_dir.exists():
                continue
                
            # Try each pattern
            for pattern in search_patterns:
                matching_files = list(plot_dir.glob(pattern))
                if matching_files:
                    all_plot_files.extend(matching_files)
                    found_pattern_info[pattern] = len(matching_files)
                    day_logger.info(f"Found {len(matching_files)} files matching '{pattern}' in {plot_dir}")
            
            # If we found files in this directory, we can optionally break here
            # But we'll continue to check all directories to be thorough
        
        # Log what patterns were successful
        if found_pattern_info:
            day_logger.info(f"Successful search patterns: {found_pattern_info}")
        else:
            day_logger.warning(f"No files found with any search pattern")
        
        # Copy plot files to quicklook_plots_daily_dir with consistent naming
        copied_files = []
        
        for plot_file in all_plot_files:
            plot_base_name = plot_file.stem  # Get filename without extension
            
            # Determine plot type from filename
            plot_type = "plot"  # Default
            if "lsp" in plot_base_name.lower():
                plot_type = "lsp"
            elif "summary" in plot_base_name.lower():
                plot_type = "summary"
            elif "amp" in plot_base_name.lower():
                plot_type = "amplitude"
            elif "az" in plot_base_name.lower() or "azim" in plot_base_name.lower():
                plot_type = "azimuth"
                
            # Create consistent target name with station, year, doy, and plot type
            target_filename = f"{station_4char_lower}_{year}_{doy_padded}_{plot_type}.png"
            target_path = quicklook_plots_daily_dir / target_filename
            
            # Copy file and track it
            try:
                shutil.copy2(plot_file, target_path)
                copied_files.append(target_path)
                day_logger.info(f"Copied {plot_file.name} → {target_filename}")
            except Exception as copy_error:
                day_logger.error(f"Error copying {plot_file} to {target_path}: {copy_error}")
        
        # Log summary of copy operation
        if copied_files:
            day_logger.info(f"Successfully copied {len(copied_files)} plot files to {quicklook_plots_daily_dir}")
        else:
            day_logger.warning(f"No plot files were copied for {station_4char_lower} {year} {doy_padded}")
            # Not treating this as a failure - the process can continue even without plots
        
        day_logger.info(f"Successfully executed quickLook for {station_4char_lower} {year} {doy_padded}")
        return True
    
    except Exception as e:
        day_logger.error(f"Error executing quickLook: {e}")
        return False
