"""
Parallel Orchestrator module for GNSS-IR processing.
Provides functionality for parallel processing of multiple days of GNSS-IR data.
"""

import logging
import multiprocessing
from pathlib import Path

# Import project modules using relative imports
from .daily_gnssir_worker import process_single_day_wrapper
from .workspace_setup import setup_gnssrefl_workspace
from ..utils.visualizer import plot_annual_rh_timeseries

def process_station_parallel(station_config, year, doy_range, tool_paths, project_root, refl_code_base, orbits_base, 
                           num_cores, results_handler=None, skip_options=None):
    """
    Process a station for a given year in parallel, using multiple cores.
    
    Args:
        station_config (dict): Station configuration
        year (int): Year (4 digits)
        doy_range (tuple): Range of DOYs to process (start, end)
        tool_paths (dict): Paths to command-line tools
        project_root (Path): Path to the project root directory
        refl_code_base (Path): Path to the REFL_CODE base directory
        orbits_base (Path): Path to the ORBITS base directory
        num_cores (int): Number of cores to use for parallel processing
        results_handler (module, optional): Module containing the combine_daily_rh_results function.
        skip_options (dict, optional): Dictionary of boolean flags for skipping steps if output exists
                                      Keys: skip_download, skip_rinex_conversion, skip_snr, skip_rh, skip_quicklook
        
    Returns:
        dict: Summary of processing results
    """
    station_id = station_config.get("station_id_4char_lower", "").lower()
    station_name = station_id.upper()
    
    # Setup gnssrefl workspace for the year
    setup_gnssrefl_workspace(station_id, year, refl_code_base, orbits_base)
    
    # Set default skip options if not provided
    if skip_options is None:
        skip_options = {
            'skip_download': False,
            'skip_rinex_conversion': False,
            'skip_snr': False,
            'skip_rh': False,
            'skip_quicklook': False
        }
    
    # Generate tasks list for parallel processing
    task_args_list = []
    for doy in range(doy_range[0], doy_range[1] + 1):
        task_args = (
            station_config,
            year,
            doy,
            tool_paths,
            project_root,
            refl_code_base,
            orbits_base,
            skip_options
        )
        task_args_list.append(task_args)
    
    # Create multiprocessing pool
    logging.info(f"Starting parallel processing using {num_cores} cores")
    
    # Process in parallel
    task_results = []
    try:
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Map tasks to pool
            task_results = pool.map(process_single_day_wrapper, task_args_list)
    except Exception as e:
        logging.error(f"Exception during parallel processing: {e}")
    
    # Process results, aggregate data, and generate plots
    results = finalize_station_processing(
        task_results=task_results,
        station_id=station_id,
        year=year,
        project_root=project_root,
        refl_code_base=refl_code_base,
        results_handler=results_handler
    )
    
    return results

def finalize_station_processing(task_results, station_id, year, project_root, refl_code_base, results_handler=None):
    """
    Process the results of parallel day processing and generate aggregated results and plots.
    
    Args:
        task_results (list): List of result dictionaries from process_single_day
        station_id (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        project_root (Path): Path to the project root directory
        refl_code_base (Path): Path to the REFL_CODE base directory
        results_handler (module, optional): Module containing the combine_daily_rh_results function.
        
    Returns:
        dict: Summary of processing results
    """
    station_name = station_id.upper()
    
    # Initialize results dictionary
    results = {
        'attempted': len(task_results),
        'successful': [],
        'failed': [],
        'details': task_results
    }
    
    # Process results
    for result in task_results:
        if result['status'] == 'success':
            results['successful'].append(result['doy'])
        else:
            results['failed'].append(result['doy'])
    
    # Only continue with aggregation if we have successful days
    if not results['successful']:
        logging.error(f"No successful days processed for {station_name} {year}")
        return results
    
    # Combine daily results
    try:
        logging.info(f"Combining daily results for {station_name} {year}")
        
        station_data_dir = project_root / "data" / station_name / str(year)
        rh_daily_dir = station_data_dir / "rh_daily"
        annual_results_dir = project_root / "results_annual" / station_name
        
        # Ensure the imported results_handler module has the required function
        if results_handler and hasattr(results_handler, 'combine_daily_rh_results'):
            combined_csv_path = results_handler.combine_daily_rh_results(
                station_id_4char_lower=station_id,
                year=year,
                daily_rh_base_dir=rh_daily_dir,
                annual_results_dir=annual_results_dir
            )
        else:
            logging.error("Results handler module not provided or missing combine_daily_rh_results function")
            combined_csv_path = None
        
        if combined_csv_path is None:
            logging.error(f"Failed to combine daily results for {station_name} {year}")
        else:
            # Generate annual plot
            try:
                logging.info(f"Generating annual plot for {station_name} {year}")
                
                plot_path = plot_annual_rh_timeseries(
                    combined_rh_csv_path=combined_csv_path,
                    station_name=station_name,
                    year=year,
                    annual_results_dir=annual_results_dir
                )
                
                if plot_path is None:
                    logging.error(f"Failed to generate annual plot for {station_name} {year}")
                else:
                    logging.info(f"Annual plot saved to {plot_path}")
            except Exception as e:
                logging.error(f"Exception during plot generation: {e}")
    except Exception as e:
        logging.error(f"Exception during results combination: {e}")
    
    # Summarize results
    logging.info(f"Processing summary for {station_name} {year}:")
    logging.info(f"  Attempted: {results['attempted']} DOYs")
    logging.info(f"  Successful: {len(results['successful'])} DOYs")
    logging.info(f"  Failed: {len(results['failed'])} DOYs")
    
    if results['failed']:
        logging.info(f"  Failed DOYs: {results['failed']}")
    
    return results
