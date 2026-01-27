"""
Daily GNSS-IR Worker module.
Provides functionality for processing a single day of GNSS-IR data.
"""

import os
import logging
import shutil
import glob
from pathlib import Path

# Import utilities from project modules using relative imports
from ..utils.data_manager import download_s3_file, check_file_exists
from ..external_tools.preprocessor import convert_rinex3_to_rinex2
from ..external_tools.gnssrefl_executor import execute_rinex2snr, execute_gnssir, execute_quicklook_threaded
from ..utils.logging_config import setup_day_logger
from .workspace_setup import setup_gnssrefl_workspace, copy_json_params

# Define minimum file sizes for validating output files
MIN_RINEX3_SIZE_BYTES = 500 * 1024  # 500 KB
MIN_RINEX2_SIZE_BYTES = 500 * 1024  # 500 KB
MIN_SNR_SIZE_BYTES = 50 * 1024      # 50 KB
MIN_RH_RESULT_SIZE_BYTES = 500      # 500 bytes (small text file)

def process_single_day(station_config, year, doy, tool_paths, project_root, refl_code_base, orbits_base, skip_options=None):
    """
    Process a single DOY for a given station.
    
    Args:
        station_config (dict): Station configuration
        year (int): Year (4 digits)
        doy (int): Day of year
        tool_paths (dict): Paths to command-line tools
        project_root (Path): Path to the project root directory
        refl_code_base (Path): Path to the REFL_CODE base directory
        orbits_base (Path): Path to the ORBITS base directory
        skip_options (dict, optional): Dictionary of boolean flags for skipping steps if output exists
                                     Keys: skip_download, skip_rinex_conversion, skip_snr, skip_rh, skip_quicklook
    
    Returns:
        dict: Dictionary with processing results
    """
    # Set default skip options if not provided
    if skip_options is None:
        skip_options = {
            'skip_download': False,
            'skip_rinex_conversion': False,
            'skip_snr': False,
            'skip_rh': False,
            'skip_quicklook': False
        }
    
    station_id = station_config.get("station_id_4char_lower", "").lower()
    
    # Format DOY with zero-padding
    doy_padded = f"{doy:03d}"
    yy = str(year)[-2:]  # Last 2 digits of year
    
    # Define paths
    s3_bucket = station_config.get("s3_bucket_name")
    s3_key_template = station_config.get("s3_rinex_obs_path_template")
    s3_key = s3_key_template.format(
        YEAR=year,
        DOY_PADDED=doy_padded,
        YY=yy
    )
    
    # Setup gnssrefl workspace
    gnssrefl_paths = setup_gnssrefl_workspace(station_id, year, refl_code_base, orbits_base, doy)
    
    # Define local paths (for our project structure)
    station_data_dir = project_root / "data" / station_id.upper() / str(year)
    rinex3_dir = station_data_dir / "rinex3"
    logs_daily_dir = station_data_dir / "logs_daily"
    quicklook_plots_daily_dir = station_data_dir / "quicklook_plots_daily"
    
    # Define paths for gnssrefl workspace
    refl_code_rinex_dir = gnssrefl_paths['refl_code_rinex']
    refl_code_results_dir = gnssrefl_paths['refl_code_results']
    refl_code_snr_dir = gnssrefl_paths['refl_code_snr']
    
    # Ensure directories exist
    for dir_path in [rinex3_dir, logs_daily_dir, quicklook_plots_daily_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define log file for this DOY
    log_file = logs_daily_dir / f"{station_id}_{year}_{doy_padded}.log"
    day_logger = setup_day_logger(log_file)
    
    result = {
        "status": "failed",
        "station": station_id.upper(),
        "year": year,
        "doy": doy,
        "doy_padded": doy_padded,
        "rh_file_path": None,
        "errors": [],
        "skipped_steps": []
    }
    
    day_logger.info(f"Processing {station_id.upper()} for {year} DOY {doy_padded}")
    
    # Define file paths
    rinex3_filename = f"{station_id.upper()}{doy_padded}0.{yy}o"
    rinex3_file_path = rinex3_dir / rinex3_filename
    
    # Define RINEX 2.11 output path
    rinex2_filename = f"{station_id}{doy_padded}0.{yy}o"
    rinex2_file_path = refl_code_rinex_dir / rinex2_filename
    
    # Define SNR file path
    snr_file_path = refl_code_snr_dir / f"{station_id}{doy_padded}0.{yy}.snr66"
    snr_file_path_gz = Path(f"{snr_file_path}.gz")
    
    # Define reflector height result file path
    rh_file_path = refl_code_results_dir / f"{doy_padded}.txt"
    
    # Define result file path for our project structure
    rh_local_path = station_data_dir / "rh_daily" / f"{station_id}_{year}_{doy_padded}.txt"
    rh_local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define plot path pattern (we'll need to check for common patterns later)
    plot_patterns = [
        f"{station_id}_{doy_padded}*.png",
        f"{station_id.upper()}_{doy_padded}*.png",
        f"{doy_padded}*.png"
    ]
    
    # Copy JSON parameter file to gnssrefl workspace
    json_source_path = project_root / station_config.get("gnssir_json_params_path")
    json_target_path = copy_json_params(json_source_path, station_id, refl_code_base)
    
    if json_target_path is None:
        day_logger.error(f"Failed to copy JSON parameter file for {station_id.upper()} {year} {doy_padded}")
        result["errors"].append("Failed to copy JSON parameter file")
        return result
    
    # Get tool paths
    gfzrnx_exe_path = tool_paths.get("gfzrnx_path", "gfzrnx")
    rinex2snr_exe_path = tool_paths.get("rinex2snr_path", "rinex2snr")
    gnssir_exe_path = tool_paths.get("gnssir_path", "gnssir")
    quicklook_exe_path = tool_paths.get("quicklook_path", "quickLook")
    
    # Log the actual paths being used
    day_logger.info(f"Tool paths being used:")
    day_logger.info(f"gfzrnx: {gfzrnx_exe_path}")
    day_logger.info(f"rinex2snr: {rinex2snr_exe_path}")
    day_logger.info(f"gnssir: {gnssir_exe_path}")
    day_logger.info(f"quickLook: {quicklook_exe_path}")
    day_logger.info(f"REFL_CODE_BASE: {refl_code_base}")
    day_logger.info(f"ORBITS_BASE: {orbits_base}")
    
    # Step 1: Download RINEX 3 file from S3
    download_success = True  # Default to True for skipping case
    if skip_options.get('skip_download', False):
        day_logger.info("Step 1: S3 Download - SKIPPING as requested.")
        if not check_file_exists(rinex3_file_path, min_size_bytes=MIN_RINEX3_SIZE_BYTES):
            day_logger.error(f"S3 Download skipped, but expected file {rinex3_file_path} is missing or too small.")
            day_logger.warning("Attempting download despite skip flag...")
            try:
                download_success = download_s3_file(
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    local_target_path=rinex3_file_path
                )
                
                if not download_success:
                    day_logger.error(f"Failed to download RINEX 3 file for {station_id.upper()} {year} {doy_padded}")
                    result["errors"].append("Failed to download RINEX 3 file")
                    return result
            except Exception as e:
                day_logger.error(f"Exception during download step: {e}")
                result["errors"].append(f"Exception during download step: {e}")
                return result
        else:
            day_logger.info(f"RINEX 3 file exists at {rinex3_file_path} with sufficient size")
            result["skipped_steps"].append("download")
    else:
        try:
            day_logger.info("Step 1: Downloading RINEX 3 file from S3")
            download_success = download_s3_file(
                s3_bucket=s3_bucket,
                s3_key=s3_key,
                local_target_path=rinex3_file_path
            )
            
            if not download_success:
                day_logger.error(f"Failed to download RINEX 3 file for {station_id.upper()} {year} {doy_padded}")
                result["errors"].append("Failed to download RINEX 3 file")
                return result
        except Exception as e:
            day_logger.error(f"Exception during download step: {e}")
            result["errors"].append(f"Exception during download step: {e}")
            return result
    
    # Step 2: Convert RINEX 3 to RINEX 2.11 and place in gnssrefl workspace
    rinex2_path = None  # Default for skipping case
    if skip_options.get('skip_rinex_conversion', False):
        day_logger.info("Step 2: RINEX 3 to RINEX 2.11 conversion - SKIPPING as requested.")
        if not check_file_exists(rinex2_file_path, min_size_bytes=MIN_RINEX2_SIZE_BYTES):
            day_logger.error(f"RINEX conversion skipped, but expected file {rinex2_file_path} is missing or too small.")
            day_logger.warning("Attempting RINEX conversion despite skip flag...")
            try:
                rinex2_path = convert_rinex3_to_rinex2(
                    gfzrnx_exe_path=gfzrnx_exe_path,
                    rinex3_file_path=rinex3_file_path,
                    rinex2_output_dir=refl_code_rinex_dir,
                    station_4char_lower=station_id.lower(),
                    year=year,
                    doy=doy
                )
                
                if rinex2_path is None or not rinex2_path.exists():
                    day_logger.error(f"Failed to convert RINEX 3 to RINEX 2.11 for {station_id.upper()} {year} {doy_padded}")
                    result["errors"].append("Failed to convert RINEX 3 to RINEX 2.11")
                    return result
                
                day_logger.info(f"RINEX 2.11 file created at {rinex2_path}")
            except Exception as e:
                day_logger.error(f"Exception during RINEX conversion step: {e}")
                result["errors"].append(f"Exception during RINEX conversion step: {e}")
                return result
        else:
            day_logger.info(f"RINEX 2.11 file exists at {rinex2_file_path} with sufficient size")
            rinex2_path = rinex2_file_path
            result["skipped_steps"].append("rinex_conversion")
    else:
        try:
            day_logger.info("Step 2: Converting RINEX 3 to RINEX 2.11")
            
            rinex2_path = convert_rinex3_to_rinex2(
                gfzrnx_exe_path=gfzrnx_exe_path,
                rinex3_file_path=rinex3_file_path,
                rinex2_output_dir=refl_code_rinex_dir,
                station_4char_lower=station_id.lower(),
                year=year,
                doy=doy
            )
            
            if rinex2_path is None or not rinex2_path.exists():
                day_logger.error(f"Failed to convert RINEX 3 to RINEX 2.11 for {station_id.upper()} {year} {doy_padded}")
                result["errors"].append("Failed to convert RINEX 3 to RINEX 2.11")
                return result
            
            day_logger.info(f"RINEX 2.11 file created at {rinex2_path}")
            
        except Exception as e:
            day_logger.error(f"Exception during RINEX conversion step: {e}")
            result["errors"].append(f"Exception during RINEX conversion step: {e}")
            return result
    
    # Step 3: Run rinex2snr
    rinex2snr_success = True  # Default for skipping case
    if skip_options.get('skip_snr', False):
        day_logger.info("Step 3: rinex2snr - SKIPPING as requested.")
        # Check if either the compressed or uncompressed SNR file exists
        if (check_file_exists(snr_file_path, min_size_bytes=MIN_SNR_SIZE_BYTES) or 
            check_file_exists(snr_file_path_gz, min_size_bytes=MIN_SNR_SIZE_BYTES)):
            day_logger.info(f"SNR file exists at {snr_file_path} or {snr_file_path_gz} with sufficient size")
            result["skipped_steps"].append("rinex2snr")
        else:
            day_logger.error(f"rinex2snr skipped, but expected SNR file is missing or too small.")
            day_logger.warning("Attempting rinex2snr despite skip flag...")
            try:
                rinex2snr_success = execute_rinex2snr(
                    rinex2snr_exe_path=rinex2snr_exe_path,
                    station_4char_lower=station_id.lower(),
                    year=year,
                    doy_padded=doy_padded,
                    refl_code_base=refl_code_base,
                    orbits_base=orbits_base,
                    logs_dir=logs_daily_dir,
                    snr_code="66"
                )
                
                if not rinex2snr_success:
                    day_logger.error(f"Failed to run rinex2snr for {station_id.upper()} {year} {doy_padded}")
                    result["errors"].append("Failed to run rinex2snr")
                    return result
            except Exception as e:
                day_logger.error(f"Exception during rinex2snr step: {e}")
                result["errors"].append(f"Exception during rinex2snr step: {e}")
                return result
    else:
        try:
            day_logger.info("Step 3: Running rinex2snr")
            
            rinex2snr_success = execute_rinex2snr(
                rinex2snr_exe_path=rinex2snr_exe_path,
                station_4char_lower=station_id.lower(),
                year=year,
                doy_padded=doy_padded,
                refl_code_base=refl_code_base,
                orbits_base=orbits_base,
                logs_dir=logs_daily_dir,
                snr_code="66"
            )
            
            if not rinex2snr_success:
                day_logger.error(f"Failed to run rinex2snr for {station_id.upper()} {year} {doy_padded}")
                result["errors"].append("Failed to run rinex2snr")
                return result
        except Exception as e:
            day_logger.error(f"Exception during rinex2snr step: {e}")
            result["errors"].append(f"Exception during rinex2snr step: {e}")
            return result
    
    # Step 4: Run gnssir
    gnssir_success = True  # Default for skipping case
    if skip_options.get('skip_rh', False):
        day_logger.info("Step 4: gnssir - SKIPPING as requested.")
        if check_file_exists(rh_file_path, min_size_bytes=MIN_RH_RESULT_SIZE_BYTES):
            day_logger.info(f"Reflector height file exists at {rh_file_path} with sufficient size")
            result["skipped_steps"].append("gnssir")
            
            # If we're skipping gnssir but the result file exists in gnssrefl workspace,
            # make sure we copy it to our project structure for results aggregation
            if not check_file_exists(rh_local_path):
                try:
                    shutil.copy2(rh_file_path, rh_local_path)
                    day_logger.info(f"Copied result file from {rh_file_path} to {rh_local_path}")
                    result["rh_file_path"] = str(rh_local_path)
                except Exception as e:
                    day_logger.error(f"Error copying result file: {e}")
                    result["errors"].append(f"Error copying result file: {e}")
            else:
                result["rh_file_path"] = str(rh_local_path)
        else:
            day_logger.error(f"gnssir skipped, but expected result file {rh_file_path} is missing or too small.")
            day_logger.warning("Attempting gnssir despite skip flag...")
            try:
                gnssir_success = execute_gnssir(
                    gnssir_exe_path=gnssir_exe_path,
                    station_4char_lower=station_id.lower(),
                    year=year,
                    doy_padded=doy_padded,
                    refl_code_base=refl_code_base,
                    orbits_base=orbits_base,
                    logs_dir=logs_daily_dir
                )
                
                if not gnssir_success:
                    day_logger.error(f"Failed to run gnssir for {station_id.upper()} {year} {doy_padded}")
                    result["errors"].append("Failed to run gnssir")
                    return result
            except Exception as e:
                day_logger.error(f"Exception during gnssir step: {e}")
                result["errors"].append(f"Exception during gnssir step: {e}")
                return result
    else:
        try:
            day_logger.info("Step 4: Running gnssir")
            
            gnssir_success = execute_gnssir(
                gnssir_exe_path=gnssir_exe_path,
                station_4char_lower=station_id.lower(),
                year=year,
                doy_padded=doy_padded,
                refl_code_base=refl_code_base,
                orbits_base=orbits_base,
                logs_dir=logs_daily_dir
            )
            
            if not gnssir_success:
                day_logger.error(f"Failed to run gnssir for {station_id.upper()} {year} {doy_padded}")
                result["errors"].append("Failed to run gnssir")
                return result
        except Exception as e:
            day_logger.error(f"Exception during gnssir step: {e}")
            result["errors"].append(f"Exception during gnssir step: {e}")
            return result
    
    # Step 5: Run quickLook in a background thread
    if skip_options.get('skip_quicklook', False):
        day_logger.info("Step 5: quickLook - SKIPPING as requested.")
        
        # Check if any quickLook plots already exist
        any_plots_exist = False
        for pattern in plot_patterns:
            if list(quicklook_plots_daily_dir.glob(pattern)):
                any_plots_exist = True
                break
        
        if any_plots_exist:
            day_logger.info(f"quickLook plots already exist in {quicklook_plots_daily_dir}")
            result["skipped_steps"].append("quicklook")
        else:
            # If we're skipping but no plots exist, see if any are in the workspace
            files_dir = refl_code_base / "Files"
            station_files_dir = files_dir / station_id
            
            # Define potential plot locations to search
            plot_locations = [
                station_files_dir,
                files_dir,
                refl_code_base / str(year) / "plots" / station_id
            ]
            
            # Check if any plots exist in these locations
            plots_in_workspace = []
            for plot_dir in plot_locations:
                if plot_dir.exists():
                    for pattern in plot_patterns:
                        plots_in_workspace.extend(list(plot_dir.glob(pattern)))
            
            if plots_in_workspace:
                day_logger.info(f"Found {len(plots_in_workspace)} existing quickLook plots in workspace directories.")
                
                # Copy plots to our project structure
                for plot_file in plots_in_workspace:
                    plot_type = "plot"  # Default
                    if "lsp" in plot_file.stem.lower():
                        plot_type = "lsp"
                    elif "summary" in plot_file.stem.lower():
                        plot_type = "summary"
                    elif "amp" in plot_file.stem.lower():
                        plot_type = "amplitude"
                    elif "az" in plot_file.stem.lower() or "azim" in plot_file.stem.lower():
                        plot_type = "azimuth"
                        
                    target_filename = f"{station_id}_{year}_{doy_padded}_{plot_type}.png"
                    target_path = quicklook_plots_daily_dir / target_filename
                    
                    try:
                        shutil.copy2(plot_file, target_path)
                        day_logger.info(f"Copied {plot_file.name} → {target_filename}")
                    except Exception as copy_error:
                        day_logger.error(f"Error copying {plot_file} to {target_path}: {copy_error}")
                
                result["skipped_steps"].append("quicklook")
            else:
                day_logger.warning("quickLook skipped, but no plot files found in workspace. Running quickLook...")
                
                try:
                    quicklook_thread = execute_quicklook_threaded(
                        quicklook_exe_path=quicklook_exe_path,
                        station_4char_lower=station_id.lower(),
                        year=year,
                        doy_padded=doy_padded,
                        refl_code_base=refl_code_base,
                        orbits_base=orbits_base,
                        quicklook_plots_daily_dir=quicklook_plots_daily_dir,
                        logs_dir=logs_daily_dir
                    )
                    
                    day_logger.info(f"quickLook started in background thread, continuing processing")
                except Exception as e:
                    day_logger.error(f"Exception starting quickLook in background thread: {e}")
                    result["errors"].append(f"Exception starting quickLook in background thread: {e}")
                    # Continue anyway even if quickLook fails, as it's not critical
    else:
        # Run quickLook in a background thread
        try:
            day_logger.info("Step 5: Running quickLook in background thread")
            
            quicklook_thread = execute_quicklook_threaded(
                quicklook_exe_path=quicklook_exe_path,
                station_4char_lower=station_id.lower(),
                year=year,
                doy_padded=doy_padded,
                refl_code_base=refl_code_base,
                orbits_base=orbits_base,
                quicklook_plots_daily_dir=quicklook_plots_daily_dir,
                logs_dir=logs_daily_dir
            )
            
            day_logger.info(f"quickLook started in background thread, continuing processing")
            # Don't wait for quickLook to finish - it will run in the background
            
        except Exception as e:
            day_logger.error(f"Exception starting quickLook in background thread: {e}")
            result["errors"].append(f"Exception starting quickLook in background thread: {e}")
            # Continue anyway even if quickLook fails, as it's not critical
    
    # Copy the result file from gnssrefl workspace to our project structure (if not already done)
    try:
        if not result["rh_file_path"]:  # Only copy if we haven't already done so
            if rh_file_path.exists():
                # Copy the file
                shutil.copy2(rh_file_path, rh_local_path)
                day_logger.info(f"Copied result file from {rh_file_path} to {rh_local_path}")
                result["rh_file_path"] = str(rh_local_path)
            else:
                day_logger.warning(f"Result file not found at {rh_file_path}")
                result["errors"].append(f"Result file not found at {rh_file_path}")
    except Exception as e:
        day_logger.error(f"Exception copying result file: {e}")
        result["errors"].append(f"Exception copying result file: {e}")
    
    # Check for gnssrefl log files and copy them if needed
    try:
        gnssrefl_logs_dir = refl_code_base / "logs" / station_id / str(year)
        if gnssrefl_logs_dir.exists():
            for log_pattern in [
                f"{doy_padded}_translation.txt",
                f"{doy_padded}_translation.txt.gen",
                f"{doy_padded}_gnssir.txt"
            ]:
                log_file_src = gnssrefl_logs_dir / log_pattern
                if log_file_src.exists():
                    log_file_dst = logs_daily_dir / f"gnssrefl_{log_pattern}"
                    shutil.copy2(log_file_src, log_file_dst)
                    day_logger.info(f"Copied gnssrefl log file from {log_file_src} to {log_file_dst}")
    except Exception as e:
        day_logger.error(f"Exception copying gnssrefl log files: {e}")
        result["errors"].append(f"Exception copying gnssrefl log files: {e}")
    
    # If we got here, the processing was successful
    day_logger.info(f"Successfully processed {station_id.upper()} for {year} DOY {doy_padded}")
    if result["skipped_steps"]:
        day_logger.info(f"Skipped steps: {', '.join(result['skipped_steps'])}")
    result["status"] = "success"
    
    return result

def process_single_day_wrapper(args):
    """
    Wrapper function to unpack arguments for process_single_day.
    
    Args:
        args (tuple): Tuple of arguments to pass to process_single_day
        
    Returns:
        dict: Result dictionary from process_single_day
    """
    return process_single_day(*args)
