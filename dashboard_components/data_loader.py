# ABOUTME: Data loading functions for Streamlit GNSS-IR dashboard
# ABOUTME: Handles CSV loading, API fetching, and data caching

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import cache manager
from dashboard_components.cache_manager import (  # noqa: E402
    disk_cache,
    monitor_performance,
)

# Import required modules
from dashboard_components.station_metadata import get_station_config  # noqa: E402

try:
    from scripts.external_apis.noaa_coops import NOAACOOPSClient
except ImportError as e:
    print(f"Warning: External API modules not available: {e}")
    NOAACOOPSClient = None


def get_preferred_coops_stations(station_id):
    """Get preferred CO-OPS stations from station config."""
    config = get_station_config(station_id)
    if config:
        coops_config = config.get("external_data_sources", {}).get("noaa_coops", {})
        return coops_config.get("preferred_stations", [])
    return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
@monitor_performance
def load_station_data(station_id="FORA", year=2024):
    """Load comprehensive GNSS-IR and comparison data for a station.

    Returns:
        tuple: (rh_data, comparison_data, usgs_data, coops_data, erddap_data)
    """
    results_dir = project_root / "results_annual" / station_id

    # Load combined RH data
    rh_file = results_dir / f"{station_id}_{year}_combined_rh.csv"
    rh_data = None
    if rh_file.exists():
        rh_data = pd.read_csv(rh_file)
        rh_data["date"] = pd.to_datetime(rh_data["date"])

    # Load comparison data (includes lag analysis)
    comparison_file = results_dir / f"{station_id}_{year}_comparison.csv"
    comparison_data = None
    if comparison_file.exists():
        comparison_data = pd.read_csv(comparison_file)
        comparison_data["merge_date"] = pd.to_datetime(comparison_data["merge_date"])

    # Load USGS gauge data
    usgs_file = results_dir / f"{station_id}_{year}_usgs_gauge_data.csv"
    usgs_data = None
    if usgs_file.exists():
        usgs_data = pd.read_csv(usgs_file)
        # Check which date column exists and standardize
        if "datetime" in usgs_data.columns:
            usgs_data["datetime"] = pd.to_datetime(usgs_data["datetime"])
        elif "date" in usgs_data.columns:
            usgs_data["date"] = pd.to_datetime(usgs_data["date"])
            usgs_data["datetime"] = usgs_data["date"]  # Create datetime column for consistency

    # If we have comparison data with USGS values, use that instead
    if (
        comparison_data is not None
        and not comparison_data.empty
        and "usgs_value" in comparison_data.columns
    ):
        # Use comparison data as primary USGS source since it has better alignment
        usgs_aligned = comparison_data[["merge_date", "usgs_value"]].copy()
        usgs_aligned["datetime"] = usgs_aligned["merge_date"]
        usgs_aligned = usgs_aligned.dropna(subset=["usgs_value"])

        if not usgs_aligned.empty:
            usgs_data = usgs_aligned  # Use the aligned data
            # Ensure we have a 'date' column for compatibility
            if "date" not in usgs_data.columns and "datetime" in usgs_data.columns:
                usgs_data["date"] = usgs_data["datetime"]
            elif "date" not in usgs_data.columns and "merge_date" in usgs_data.columns:
                usgs_data["date"] = usgs_data["merge_date"]

    # Load CO-OPS data if available (check multiple filename patterns)
    coops_data = None
    coops_file_patterns = [
        results_dir / f"{station_id}_{year}_coops_daily.csv",
        results_dir / f"{station_id}_{year}_coops_6min.csv",
        results_dir / f"{station_id}_{year}_coops_hourly.csv",
    ]

    coops_file = None
    for pattern in coops_file_patterns:
        if pattern.exists():
            coops_file = pattern
            break

    if coops_file:
        coops_data = pd.read_csv(coops_file)
        # Handle datetime column
        if "date" in coops_data.columns:
            coops_data["date"] = pd.to_datetime(coops_data["date"])
            coops_data["datetime"] = coops_data["date"]
        elif "datetime" in coops_data.columns:
            coops_data["datetime"] = pd.to_datetime(coops_data["datetime"])
            coops_data["date"] = coops_data["datetime"].dt.date

        # Rename water level columns for consistency
        for col in ["water_level_mean", "water_level", "water_level_m", "v"]:
            if col in coops_data.columns and "water_level_m" not in coops_data.columns:
                coops_data["water_level_m"] = coops_data[col]
                break

        # Get station ID from config if available
        from dashboard_components.station_metadata import get_reference_source_info

        ref_info = get_reference_source_info(station_id)
        coops_station_id = (
            ref_info.get("station_id", "Unknown")
            if ref_info["primary_source"] == "CO-OPS"
            else "Unknown"
        )

        coops_data["source"] = "NOAA CO-OPS"
        coops_data["station_id"] = coops_station_id

    # Load ERDDAP data if available (from subdaily_matched.csv)
    # NOTE: ERDDAP column naming varies by station. Current known patterns:
    #   - GLBX: bartlett_cove_datetime, bartlett_cove_wl, bartlett_cove_dm
    #   - Other stations may use different prefixes based on their ERDDAP source
    # The code attempts to auto-detect columns ending in _datetime, _wl, _dm that aren't gnss_*
    erddap_data = None
    subdaily_file = results_dir / f"{station_id}_{year}_subdaily_matched.csv"
    if subdaily_file.exists():
        subdaily_df = pd.read_csv(subdaily_file)

        # Auto-detect ERDDAP reference columns (not gnss_* columns)
        # Look for datetime columns that aren't gnss_datetime
        ref_datetime_cols = [
            col
            for col in subdaily_df.columns
            if col.endswith("_datetime") and not col.startswith("gnss")
        ]
        # Look for water level columns that aren't gnss
        ref_wl_cols = [
            col for col in subdaily_df.columns if col.endswith("_wl") and not col.startswith("gnss")
        ]

        if ref_datetime_cols or ref_wl_cols:
            # Found ERDDAP reference data
            erddap_data = subdaily_df.copy()

            # Use first found reference datetime column
            if ref_datetime_cols:
                ref_dt_col = ref_datetime_cols[0]
                erddap_data["datetime"] = pd.to_datetime(erddap_data[ref_dt_col])
                erddap_data["date"] = erddap_data["datetime"].dt.date
                # Store the column prefix for reference
                erddap_data["_erddap_prefix"] = ref_dt_col.replace("_datetime", "")

            # Use first found reference water level column
            if ref_wl_cols:
                ref_wl_col = ref_wl_cols[0]
                erddap_data["water_level_m"] = erddap_data[ref_wl_col]

            # Add source info
            from dashboard_components.station_metadata import get_reference_source_info

            ref_info = get_reference_source_info(station_id)
            erddap_data["source"] = "ERDDAP"
            erddap_data["station_name"] = ref_info.get("station_name", "ERDDAP Station")

    return rh_data, comparison_data, usgs_data, coops_data, erddap_data


@st.cache_data
def load_available_stations():
    """Load list of available stations from configuration."""
    config_file = project_root / "config" / "stations_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
            return list(config.keys())
    return ["FORA"]


def get_station_coordinates(station_id):
    """Get station coordinates from configuration."""
    config = get_station_config(station_id)
    if config:
        lat = config.get("latitude", config.get("latitude_deg"))
        lon = config.get("longitude", config.get("longitude_deg"))
        return lat, lon
    return None, None


@disk_cache("external_api")
@monitor_performance
def fetch_coops_data(station_id, year, doy_range=None, rh_data=None):
    """Fetch NOAA CO-OPS data for the specified station and time range.

    This function is cached to disk to avoid repeated API calls.
    Cache expires after 7 days.
    """
    try:
        client = NOAACOOPSClient()

        # Get station coordinates
        lat, lon = get_station_coordinates(station_id)
        if lat is None or lon is None:
            return None, None

        # Check for preferred stations first
        preferred_stations = get_preferred_coops_stations(station_id)

        if not preferred_stations:
            st.warning("No CO-OPS stations configured for this station")
            return None, None

        # Use the first preferred station
        coops_station_id = preferred_stations[0]
        st.info(f"Using preferred CO-OPS station: {coops_station_id}")

        # Determine date range
        if rh_data is not None and not rh_data.empty:
            # Use actual GNSS-IR data range
            start_date = rh_data["date"].min()
            end_date = rh_data["date"].max()
        elif doy_range:
            # Use DOY range
            start_doy, end_doy = doy_range
            start_date = datetime(year, 1, 1) + timedelta(days=start_doy - 1)
            end_date = datetime(year, 1, 1) + timedelta(days=end_doy - 1)
        else:
            # Default to full year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

        # Format dates for API
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # Fetch water level observations
        water_levels = client.get_water_levels(
            station_id=coops_station_id,
            start_date=start_str,
            end_date=end_str,
            datum="NAVD",  # North American Vertical Datum
            units="metric",
            time_zone="gmt",
        )

        if water_levels is not None and not water_levels.empty:
            # Convert time column to datetime and rename for consistency
            water_levels["datetime"] = pd.to_datetime(water_levels["time"])
            water_levels["date"] = water_levels["datetime"].dt.date
            water_levels["water_level_m"] = water_levels["value"]

            # Also fetch tide predictions for the same period
            predictions = client.get_tide_predictions(
                station_id=coops_station_id,
                start_date=start_str,
                end_date=end_str,
                datum="NAVD",
                units="metric",
                time_zone="gmt",
            )

            if predictions is not None and not predictions.empty:
                predictions["datetime"] = pd.to_datetime(predictions["time"])
                predictions["tide_prediction_m"] = predictions["value"]

                # Merge observations with predictions
                water_levels = pd.merge(
                    water_levels,
                    predictions[["datetime", "tide_prediction_m"]],
                    on="datetime",
                    how="left",
                )

                # Calculate residuals (observed - predicted)
                water_levels["residual_m"] = (
                    water_levels["water_level_m"] - water_levels["tide_prediction_m"]
                )

            return water_levels, coops_station_id

    except Exception as e:
        st.error(f"Error fetching CO-OPS data: {str(e)}")

    return None, None


@st.cache_data(ttl=300)  # Cache for 5 minutes
@monitor_performance
def discover_quicklook_plots(station_id="FORA", year=2024):
    """Discover available QuickLook diagnostic plots for a station/year."""
    plots_dir = project_root / "data" / station_id / str(year) / "quicklook_plots_daily"

    if not plots_dir.exists():
        return {}

    # Scan for available plot files
    plot_files = {}
    for plot_file in plots_dir.glob("*.png"):
        # Parse filename: valr_2024_247_lsp.png or valr_2024_247_summary.png
        parts = plot_file.stem.split("_")
        if len(parts) >= 4:
            try:
                station = parts[0].upper()
                file_year = int(parts[1])
                doy = int(parts[2])
                plot_type = parts[3]  # 'lsp' or 'summary'

                if station == station_id.upper() and file_year == year:
                    if doy not in plot_files:
                        plot_files[doy] = {}
                    plot_files[doy][plot_type] = plot_file
            except (ValueError, IndexError):
                continue

    return plot_files


@st.cache_data(ttl=300)
@monitor_performance
def get_quicklook_plots_for_day(station_id="FORA", year=2024, doy=1):
    """Get QuickLook plot file paths for a specific day."""
    plot_files = discover_quicklook_plots(station_id, year)

    if doy not in plot_files:
        return None

    day_plots = plot_files[doy]
    result = {}

    # Check for both required plot types
    if "lsp" in day_plots and day_plots["lsp"].exists():
        result["lsp"] = day_plots["lsp"]

    if "summary" in day_plots and day_plots["summary"].exists():
        result["summary"] = day_plots["summary"]

    return result if result else None


@st.cache_data(ttl=3600)
@monitor_performance
def get_available_diagnostic_days(station_id="FORA", year=2024):
    """Get list of days with available diagnostic plots."""
    plot_files = discover_quicklook_plots(station_id, year)

    # Only include days that have both LSP and summary plots
    complete_days = []
    for doy, plots in plot_files.items():
        if (
            "lsp" in plots
            and plots["lsp"].exists()
            and "summary" in plots
            and plots["summary"].exists()
        ):
            complete_days.append(doy)

    return sorted(complete_days)


def doy_to_date(year, doy):
    """Convert day of year to datetime date."""
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def date_to_doy(date):
    """Convert datetime date to day of year."""
    return date.timetuple().tm_yday


# Export all functions
__all__ = [
    "load_station_data",
    "load_available_stations",
    "get_station_coordinates",
    "fetch_coops_data",
    "discover_quicklook_plots",
    "get_quicklook_plots_for_day",
    "get_available_diagnostic_days",
    "doy_to_date",
    "date_to_doy",
]
