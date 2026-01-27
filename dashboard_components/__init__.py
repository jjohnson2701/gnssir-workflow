"""
Dashboard Components Package

Modular components for the enhanced GNSS-IR dashboard v3.
"""

from .data_loader import (
    load_station_data,
    load_available_stations,
    get_station_coordinates,
    fetch_coops_data,
    fetch_ndbc_data,
    load_subhourly_data_progressive,
    create_performance_summary,
    load_data_with_progress
)

from .analysis_runner import run_multi_source_analysis

from .cache_manager import (
    get_preloader,
    CACHE_DIR
)

from .station_metadata import (
    get_station_config,
    get_antenna_height,
    get_reference_source_info,
    get_all_station_ids,
    get_station_display_info
)

__all__ = [
    # Data loading
    'load_station_data',
    'load_available_stations',
    'get_station_coordinates',
    'fetch_coops_data',
    'fetch_ndbc_data',
    'load_subhourly_data_progressive',
    'create_performance_summary',
    'load_data_with_progress',

    # Analysis
    'run_multi_source_analysis',

    # Cache management
    'get_preloader',
    'CACHE_DIR',

    # Station metadata (v4)
    'get_station_config',
    'get_antenna_height',
    'get_reference_source_info',
    'get_all_station_ids',
    'get_station_display_info'
]