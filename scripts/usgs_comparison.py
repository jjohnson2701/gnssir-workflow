"""
ABOUTME: USGS comparison for GNSS-IR processing.
ABOUTME: Compares reflector height data with USGS gauge data, with WSE_ellips and time lag analysis.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Force reload of visualizer modules to pick up changes
import importlib

if "visualizer.comparison" in sys.modules:
    importlib.reload(sys.modules["visualizer.comparison"])
if "visualizer.segmented_viz" in sys.modules:
    importlib.reload(sys.modules["visualizer.segmented_viz"])

# Add project modules
sys.path.append(str(Path(__file__).resolve().parent))

# Import project modules
from utils.gnssir_loader import load_gnssir_data
import usgs_data_handler
import visualizer
import time_lag_analyzer
import reflector_height_utils

try:
    from utils.segmented_analysis import (
        generate_monthly_segments,
        generate_seasonal_segments,
        perform_segmented_correlation,
        filter_by_segment,
    )
except ImportError:
    # Import path when running from different location
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.utils.segmented_analysis import (
        generate_monthly_segments,
        generate_seasonal_segments,
        perform_segmented_correlation,
        filter_by_segment,
    )
import matplotlib.pyplot as plt

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def usgs_comparison(
    station_name,
    year,
    doy_range=None,
    max_lag_days=10,
    output_dir=None,
    perform_segmented_analysis=True,
):
    """
    Run USGS comparison analysis with vertical datum alignment and time lag analysis.

    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
        year (int or str): Year to analyze
        doy_range (tuple, optional): Range of DOYs to include (start, end)
        max_lag_days (int, optional): Maximum lag to consider in days
        output_dir (str or Path, optional): Directory to save output files

    Returns:
        dict: Analysis results and paths to output files
    """
    # Convert year to string if needed
    year = str(year)

    # Get station configuration
    station_config = usgs_data_handler.get_station_config(station_name)
    if station_config is None:
        return {"success": False, "error": f"Station {station_name} not found or config error"}

    # Check for ellipsoidal height in config
    if "ellipsoidal_height_m" not in station_config:
        logging.error(f"Station {station_name} does not have ellipsoidal_height_m in configuration")
        return {"success": False, "error": "Missing ellipsoidal_height_m in station configuration"}

    antenna_ellipsoidal_height = station_config["ellipsoidal_height_m"]

    # Check for USGS gauge datum information
    usgs_config = station_config.get("usgs_comparison", {})
    usgs_gauge_stated_datum = usgs_config.get("usgs_gauge_stated_datum", "Unknown")

    logging.info(
        f"Datum information: GNSS antenna ellipsoidal height = {antenna_ellipsoidal_height} m, "
        f"USGS gauge stated datum = {usgs_gauge_stated_datum}"
    )

    # Set up output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results_annual" / station_name
    else:
        # Convert string to Path if needed
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output files will be saved to: {output_dir}")

    # Convert doy_range to dates for USGS data fetch
    if doy_range:
        doy_start, doy_end = doy_range
        start_date = datetime.strptime(f"{year}-{doy_start}", "%Y-%j").strftime("%Y-%m-%d")
        end_date = datetime.strptime(f"{year}-{doy_end}", "%Y-%j").strftime("%Y-%m-%d")
        logging.info(
            f"Analyzing data from DOY {doy_start} to {doy_end} ({start_date} to {end_date})"
        )
    else:
        # Default to full year if no range specified
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        logging.info(f"Analyzing data for full year {year} ({start_date} to {end_date})")

    # Load GNSS-IR data
    gnssir_daily_df = load_gnssir_data(station_name, year, doy_range)
    if gnssir_daily_df is None or gnssir_daily_df.empty:
        logging.error(
            f"Failed to load GNSS-IR data for {station_name}, {year}, DOY range {doy_range}"
        )
        return {"success": False, "error": "Failed to load GNSS-IR data"}

    # Log loaded data information
    logging.info(f"Loaded GNSS-IR data with columns: {gnssir_daily_df.columns.tolist()}")
    logging.info(f"Total rows in GNSS-IR data: {len(gnssir_daily_df)}")

    # Find appropriate USGS gauge
    usgs_site_code = usgs_data_handler.find_usgs_gauge_for_station(station_name)
    if usgs_site_code is None:
        logging.error(f"No USGS gauge found for station {station_name}")
        return {"success": False, "error": "No USGS gauge found"}

    # Get USGS parameter code from config
    parameter_code = usgs_data_handler.get_usgs_parameter_code(station_name)
    if parameter_code is None:
        parameter_code = "00065"  # Default to gage height

    # Fetch USGS gauge data using the full date range
    logging.info(
        f"Fetching USGS data for site {usgs_site_code}, parameter {parameter_code}, dates {start_date} to {end_date}"
    )
    usgs_df, gauge_info, param_code_used = usgs_data_handler.fetch_usgs_gauge_data(
        usgs_site_code,
        parameter_code=parameter_code,
        start_date_str=start_date,
        end_date_str=end_date,
        service="iv",  # Use instantaneous values
    )

    if usgs_df is None or usgs_df.empty:
        logging.error(f"Failed to fetch USGS data for site {usgs_site_code}")
        return {"success": False, "error": f"Failed to fetch USGS data for site {usgs_site_code}"}

    # Update gauge info with stated datum if not already set
    if (
        "datum" not in gauge_info
        or not gauge_info["datum"]
        or gauge_info["datum"] == "Unknown datum"
    ):
        gauge_info["datum"] = usgs_gauge_stated_datum
        logging.info(f"Using stated datum from config: {usgs_gauge_stated_datum}")

    # === Create subdaily_matched.csv by matching raw GNSS-IR to USGS IV ===
    # Load raw GNSS-IR data for subdaily matching
    raw_csv_path = (
        PROJECT_ROOT / "results_annual" / station_name / f"{station_name}_{year}_combined_raw.csv"
    )
    if raw_csv_path.exists():
        logging.info(f"Loading raw GNSS-IR data for subdaily matching from {raw_csv_path}")
        gnssir_raw_df = pd.read_csv(raw_csv_path)

        # Prepare gauge_info dict for subdaily function
        subdaily_gauge_info = {
            "site_code": usgs_site_code,
            "site_name": gauge_info.get("site_name", ""),
            "vertical_datum": usgs_gauge_stated_datum,
            "gnss_lat": station_config.get("latitude_deg"),
            "gnss_lon": station_config.get("longitude_deg"),
            "usgs_lat": gauge_info.get("latitude"),
            "usgs_lon": gauge_info.get("longitude"),
        }

        try:
            from visualizer.comparison import plot_subdaily_ribbon_comparison

            # Reset index to make datetime a column (USGS IV data has datetime as index)
            usgs_iv_for_subdaily = usgs_df.reset_index()
            subdaily_results = plot_subdaily_ribbon_comparison(
                gnssir_raw_df=gnssir_raw_df,
                usgs_iv_df=usgs_iv_for_subdaily,
                station_name=station_name,
                usgs_gauge_info=subdaily_gauge_info,
                output_dir=output_dir,
                antenna_height=antenna_ellipsoidal_height,
                year=int(year),
                gap_threshold_hours=2.0,
                ribbon_window=5,
                show_residuals=True,
            )
            if subdaily_results.get("matched_data"):
                logging.info(f"Created subdaily_matched.csv: {subdaily_results['matched_data']}")
            else:
                logging.warning("subdaily_matched.csv was not created")
        except Exception as e:
            logging.warning(f"Could not create subdaily_matched.csv: {e}")
    else:
        logging.warning(f"Raw GNSS-IR data not found at {raw_csv_path}, skipping subdaily matching")

    # Add station coordinates to gauge_info
    if "latitude_deg" in station_config and "longitude_deg" in station_config:
        gauge_info["gnss_lat"] = station_config["latitude_deg"]
        gauge_info["gnss_lon"] = station_config["longitude_deg"]
        logging.info(
            f"Added GNSS coordinates to gauge_info: ({gauge_info['gnss_lat']}, {gauge_info['gnss_lon']})"
        )

    # Try to get USGS gauge coordinates
    if "latitude" in gauge_info and "longitude" in gauge_info:
        gauge_info["usgs_lat"] = gauge_info["latitude"]
        gauge_info["usgs_lon"] = gauge_info["longitude"]
        logging.info(
            f"Using USGS coordinates from gauge_info: ({gauge_info['usgs_lat']}, {gauge_info['usgs_lon']})"
        )
    elif (
        "site_info" in gauge_info
        and "dec_lat_va" in gauge_info["site_info"]
        and "dec_long_va" in gauge_info["site_info"]
    ):
        gauge_info["usgs_lat"] = gauge_info["site_info"]["dec_lat_va"]
        gauge_info["usgs_lon"] = gauge_info["site_info"]["dec_long_va"]
        logging.info(
            f"Using USGS coordinates from site_info: ({gauge_info['usgs_lat']}, {gauge_info['usgs_lon']})"
        )

    # Process USGS data (convert units, aggregate, etc.)
    usgs_daily_df = usgs_data_handler.process_usgs_data(usgs_df, gauge_info)
    if usgs_daily_df is None or usgs_daily_df.empty:
        logging.error(f"Failed to process USGS data for site {usgs_site_code}")
        return {"success": False, "error": "Failed to process USGS data"}

    # Calculate WSE_ellips from RH
    gnssir_daily_df = reflector_height_utils.calculate_wse_from_rh(
        gnssir_daily_df, antenna_ellipsoidal_height
    )

    # Ensure rh_count is preserved (add additional safeguard)
    if "rh_count" in gnssir_daily_df.columns:
        logging.info(
            f"rh_count column found with values: Min={gnssir_daily_df['rh_count'].min()}, "
            f"Max={gnssir_daily_df['rh_count'].max()}, "
            f"Mean={gnssir_daily_df['rh_count'].mean():.2f}"
        )
    else:
        logging.warning(
            "rh_count column not found after WSE calculation. Checking original dataframe..."
        )
        if "rh_count" in gnssir_daily_df.columns:
            logging.info("Found rh_count in original dataframe, restoring it")
            gnssir_daily_df["rh_count"] = gnssir_daily_df["rh_count"]
        else:
            logging.warning("rh_count not found in original dataframe either")

    # Add gauge datum info to gauge_info
    gauge_info["vertical_datum"] = usgs_gauge_stated_datum

    # Create demeaned versions of USGS data
    if "usgs_value_m_median" in usgs_daily_df.columns:
        usgs_daily_df["usgs_value_m_median_demeaned"] = (
            usgs_daily_df["usgs_value_m_median"] - usgs_daily_df["usgs_value_m_median"].mean()
        )
    elif "usgs_value_m_mean" in usgs_daily_df.columns:
        usgs_daily_df["usgs_value_m_mean_demeaned"] = (
            usgs_daily_df["usgs_value_m_mean"] - usgs_daily_df["usgs_value_m_mean"].mean()
        )
    else:
        # Try to find any usgs value column that might exist
        usgs_cols = [col for col in usgs_daily_df.columns if "usgs_value" in col]
        if usgs_cols:
            use_col = usgs_cols[0]
            logging.info(f"Using column '{use_col}' for USGS data demeaning")
            usgs_daily_df[f"{use_col}_demeaned"] = (
                usgs_daily_df[use_col] - usgs_daily_df[use_col].mean()
            )
        else:
            logging.error("No USGS value column found for demeaning")

    # Merge data for correlation analysis
    merge_columns = ["date"]
    gnssir_columns = ["rh_median_m", "wse_ellips_m"]
    usgs_columns = []

    # Determine which USGS column to use for merging
    if "usgs_value_m_median" in usgs_daily_df.columns:
        usgs_columns.append("usgs_value_m_median")
        usgs_value_col = "usgs_value_m_median"
    elif "usgs_value_m_mean" in usgs_daily_df.columns:
        usgs_columns.append("usgs_value_m_mean")
        usgs_value_col = "usgs_value_m_mean"
    else:
        # Try to find any usgs value column that might exist
        usgs_cols = [col for col in usgs_daily_df.columns if "usgs_value" in col]
        if usgs_cols:
            usgs_value_col = usgs_cols[0]
            usgs_columns.append(usgs_value_col)
            logging.info(f"Using column '{usgs_value_col}' for USGS data in correlation analysis")
        else:
            logging.error("No USGS value column found for correlation analysis")
            return {
                "success": False,
                "error": "No USGS value column found for correlation analysis",
            }

    logging.info(f"GNSS-IR columns: {gnssir_columns}")
    logging.info(f"USGS columns: {usgs_columns}")

    # Add detailed logging about date columns before merging
    logging.info(f"GNSS-IR date column dtype: {gnssir_daily_df['date'].dtype}")
    logging.info(f"USGS date column dtype: {usgs_daily_df['date'].dtype}")
    logging.info(f"GNSS-IR first few dates: {gnssir_daily_df['date'].head().tolist()}")
    logging.info(f"USGS first few dates: {usgs_daily_df['date'].head().tolist()}")
    logging.info(
        f"GNSS-IR date range: {gnssir_daily_df['date'].min()} to {gnssir_daily_df['date'].max()}"
    )
    logging.info(f"USGS date range: {usgs_daily_df['date'].min()} to {usgs_daily_df['date'].max()}")

    # Add additional logging to inspect the dataframes before merging
    logging.info(f"GNSS-IR daily dataframe shape before merge: {gnssir_daily_df.shape}")
    logging.info(f"USGS daily dataframe shape before merge: {usgs_daily_df.shape}")

    # Ensure both dataframes have a standardized date format
    # Create new columns for merging to ensure date format compatibility
    gnssir_daily_df["merge_date"] = pd.to_datetime(gnssir_daily_df["date"]).dt.date
    usgs_daily_df["merge_date"] = pd.to_datetime(usgs_daily_df["date"]).dt.date

    logging.info(
        f"After conversion - GNSS-IR merge_date first few: {gnssir_daily_df['merge_date'].head().tolist()}"
    )
    logging.info(
        f"After conversion - USGS merge_date first few: {usgs_daily_df['merge_date'].head().tolist()}"
    )

    # Ensure we're using only the daily aggregated data from GNSS-IR
    # This is critical for proper daily-to-daily comparison
    if "doy" in gnssir_daily_df.columns:
        # Count unique days to verify we have daily data
        unique_days = gnssir_daily_df["merge_date"].nunique()
        logging.info(f"Number of unique days in GNSS-IR data: {unique_days}")

        # If we have multiple records per day, aggregate to daily
        if len(gnssir_daily_df) > unique_days:
            logging.info(f"Found multiple GNSS-IR records per day. Aggregating to daily...")

            # Group by date and calculate daily statistics
            gnssir_daily_agg = (
                gnssir_daily_df.groupby("merge_date")
                .agg(
                    {
                        "rh_median_m": "median",
                        "wse_ellips_m": "median",
                        "rh_median_m_demeaned": "median",
                        "wse_ellips_m_demeaned": "median",
                    }
                )
                .reset_index()
            )

            # Use the aggregated dataframe for merging
            logging.info(f"After aggregation - GNSS-IR shape: {gnssir_daily_agg.shape}")
            gnssir_merge_df = gnssir_daily_agg
        else:
            # Already have one record per day
            gnssir_merge_df = gnssir_daily_df[["merge_date"] + gnssir_columns]
    else:
        # Fallback if no doy column
        gnssir_merge_df = gnssir_daily_df[["merge_date"] + gnssir_columns]

    # Ensure USGS data is also daily
    usgs_merge_df = usgs_daily_df[["merge_date"] + usgs_columns]

    # Merge on the converted date column
    merged_df = pd.merge(gnssir_merge_df, usgs_merge_df, on="merge_date", how="inner")

    # If the merge was successful, rename merge_date back to 'date'
    if len(merged_df) > 0:
        merged_df["date"] = merged_df["merge_date"]

    logging.info(f"Merged data has {len(merged_df)} rows with overlapping dates")

    # Calculate correlations
    if len(merged_df) >= 2:  # Need at least 2 points for correlation
        rh_correlation = merged_df["rh_median_m"].corr(merged_df[usgs_value_col])
        wse_correlation = merged_df["wse_ellips_m"].corr(merged_df[usgs_value_col])

        rh_corr_str = (
            f"{rh_correlation:.4f}"
            if rh_correlation is not None and not pd.isna(rh_correlation)
            else "N/A"
        )
        wse_corr_str = (
            f"{wse_correlation:.4f}"
            if wse_correlation is not None and not pd.isna(wse_correlation)
            else "N/A"
        )

        logging.info(f"Correlation between GNSS-IR RH and USGS water level: {rh_corr_str}")
        logging.info(f"Correlation between WSE_ellips and USGS water level: {wse_corr_str}")
    else:
        logging.warning("Not enough overlapping data points to calculate correlation")
        rh_correlation = None
        wse_correlation = None

    # Perform time lag analysis
    time_lag_results = {
        "rh_lag_days": None,
        "rh_lag_correlation": None,
        "rh_lag_confidence": None,
        "wse_lag_days": None,
        "wse_lag_correlation": None,
        "wse_lag_confidence": None,
    }

    if len(merged_df) >= 3:  # Need at least 3 points for lag analysis
        try:
            rh_lag, rh_lag_corr, rh_lag_conf, rh_lag_all = (
                time_lag_analyzer.calculate_time_lag_correlation(
                    merged_df["rh_median_m"], merged_df[usgs_value_col], max_lag_days=max_lag_days
                )
            )

            time_lag_results["rh_lag_days"] = rh_lag
            time_lag_results["rh_lag_correlation"] = rh_lag_corr
            time_lag_results["rh_lag_confidence"] = rh_lag_conf

            logging.info(
                f"RH time lag analysis: {rh_lag} days, correlation {rh_lag_corr if rh_lag_corr is not None and not pd.isna(rh_lag_corr) else 'N/A'}, confidence {rh_lag_conf}"
            )
        except Exception as e:
            logging.error(f"Error calculating RH time lag correlation: {e}")

        try:
            wse_lag, wse_lag_corr, wse_lag_conf, wse_lag_all = (
                time_lag_analyzer.calculate_time_lag_correlation(
                    merged_df["wse_ellips_m"], merged_df[usgs_value_col], max_lag_days=max_lag_days
                )
            )

            time_lag_results["wse_lag_days"] = wse_lag
            time_lag_results["wse_lag_correlation"] = wse_lag_corr
            time_lag_results["wse_lag_confidence"] = wse_lag_conf

            logging.info(
                f"WSE time lag analysis: {wse_lag} days, correlation {wse_lag_corr if wse_lag_corr is not None and not pd.isna(wse_lag_corr) else 'N/A'}, confidence {wse_lag_conf}"
            )
        except Exception as e:
            logging.error(f"Error calculating WSE time lag correlation: {e}")
    else:
        logging.warning(
            "Not enough overlapping data points for time lag analysis (need at least 3 points)"
        )

    # Generate comparison plots
    logging.info(f"Generating comparison plots in {output_dir}")

    # Generate WSE vs USGS plot (dual y-axis)
    wse_plot_path = output_dir / f"{station_name}_{year}_wse_usgs_comparison.png"
    try:
        # Debug: verify rh_count exists and DataFrame structure
        if "rh_count" in gnssir_daily_df.columns:
            logging.info(f"Just before plotting: rh_count exists with {len(gnssir_daily_df)} rows")
            logging.info(f"Sample rh_count values: {gnssir_daily_df['rh_count'].head(3).tolist()}")

            # Create a new column in the DataFrame with a special name that our plotting function will look for
            gnssir_daily_df["rh_retrieval_count"] = gnssir_daily_df["rh_count"]
        else:
            logging.error(
                f"Just before plotting: rh_count missing from columns: {gnssir_daily_df.columns.tolist()}"
            )

        wse_plot_path = visualizer.plot_comparison_timeseries(
            daily_gnssir_rh_df=gnssir_daily_df,
            daily_usgs_gauge_df=usgs_daily_df,
            station_name=station_name,
            usgs_gauge_info=gauge_info,
            output_plot_path=wse_plot_path,
            gnssir_rh_col="wse_ellips_m",
            usgs_wl_col=usgs_value_col,
            compare_demeaned=False,
        )
        logging.info(f"Generated WSE vs USGS comparison plot: {wse_plot_path}")
    except Exception as e:
        logging.error(f"Error generating WSE comparison plot: {e}")
        wse_plot_path = None

    # Find the demeaned usgs column name
    if "usgs_value_m_median_demeaned" in usgs_daily_df.columns:
        usgs_demeaned_col = "usgs_value_m_median_demeaned"
    elif "usgs_value_m_mean_demeaned" in usgs_daily_df.columns:
        usgs_demeaned_col = "usgs_value_m_mean_demeaned"
    else:
        # Look for any demeaned column
        demeaned_cols = [col for col in usgs_daily_df.columns if "demeaned" in col]
        if demeaned_cols:
            usgs_demeaned_col = demeaned_cols[0]
        else:
            # If no demeaned columns exist, use the regular column
            usgs_demeaned_col = usgs_value_col
            logging.warning(f"No demeaned USGS column found, using {usgs_value_col} instead")

    logging.info(f"Using {usgs_demeaned_col} for demeaned comparison plot")

    # Generate demeaned comparison plot
    demeaned_plot_path = output_dir / f"{station_name}_{year}_demeaned_comparison.png"
    try:
        # Debug: verify rh_count again before demeaned plotting
        if "rh_count" in gnssir_daily_df.columns:
            logging.info(
                f"Before demeaned plotting: rh_count column confirmed present in gnssir_daily_df"
            )
        else:
            logging.error(
                f"Before demeaned plotting: rh_count missing! Creating a dummy column for testing"
            )
            # Create a dummy rh_count column to test if that's the issue
            gnssir_daily_df["rh_count"] = pd.Series([20] * len(gnssir_daily_df))

        demeaned_plot_path = visualizer.plot_comparison_timeseries(
            daily_gnssir_rh_df=gnssir_daily_df,
            daily_usgs_gauge_df=usgs_daily_df,
            station_name=station_name,
            usgs_gauge_info=gauge_info,
            output_plot_path=demeaned_plot_path,
            gnssir_rh_col="wse_ellips_m_demeaned",
            usgs_wl_col=usgs_demeaned_col,
            compare_demeaned=True,
        )
        logging.info(f"Generated demeaned comparison plot: {demeaned_plot_path}")
    except Exception as e:
        logging.error(f"Error generating demeaned comparison plot: {e}")
        demeaned_plot_path = None

    # If time lag is significant and confidence is medium or high, create a lag-adjusted plot
    lag_plot_path = None
    if wse_lag is not None and abs(wse_lag) > 0 and wse_lag_conf in ["Medium", "High"]:
        lag_plot_path = output_dir / f"{station_name}_{year}_lag_adjusted_comparison.png"
        try:
            # Get lag-adjusted series
            wse_series, usgs_series = time_lag_analyzer.create_lag_adjusted_series(
                merged_df["wse_ellips_m"], merged_df[usgs_value_col], wse_lag
            )

            # Calculate the number of overlapping points
            num_points = len(wse_series)

            # Only proceed if we have enough points for a meaningful plot
            if num_points >= 3:
                # Create temporary dataframes for plotting
                if wse_lag > 0:
                    # GNSS-IR leads USGS by lag_days
                    dates = merged_df["date"].iloc[wse_lag:]
                else:
                    # USGS leads GNSS-IR by abs(lag_days)
                    dates = merged_df["date"].iloc[:wse_lag]

                lag_gnssir_df = pd.DataFrame({"date": dates, "wse_ellips_m": wse_series.values})

                lag_usgs_df = pd.DataFrame({"date": dates, usgs_value_col: usgs_series.values})

                # Add lag information to gauge_info
                lag_gauge_info = gauge_info.copy()
                lag_gauge_info["time_lag_days"] = wse_lag
                lag_gauge_info["lag_correlation"] = wse_lag_corr
                lag_gauge_info["lag_confidence"] = wse_lag_conf
                lag_gauge_info["num_points"] = num_points

                # Create lag-adjusted plot
                lag_plot_path = visualizer.plot_comparison_timeseries(
                    daily_gnssir_rh_df=lag_gnssir_df,
                    daily_usgs_gauge_df=lag_usgs_df,
                    station_name=f"{station_name} (Lag-adjusted {wse_lag} days, {wse_lag_conf} confidence)",
                    usgs_gauge_info=lag_gauge_info,
                    output_plot_path=lag_plot_path,
                    gnssir_rh_col="wse_ellips_m",
                    usgs_wl_col=usgs_value_col,
                    compare_demeaned=False,
                )
                logging.info(
                    f"Generated lag-adjusted comparison plot with {num_points} points: {lag_plot_path}"
                )
            else:
                logging.warning(
                    f"Not enough points for lag-adjusted plot (only {num_points} points)"
                )
                lag_plot_path = None
        except Exception as e:
            logging.error(f"Error generating lag-adjusted plot: {e}")
            lag_plot_path = None
    else:
        reason = (
            "lag not significant"
            if wse_lag is None or abs(wse_lag) == 0
            else f"insufficient confidence ({wse_lag_conf})"
        )
        logging.info(f"No lag-adjusted plot generated: {reason}")

    # Save comparison results to CSV
    comparison_csv_path = output_dir / f"{station_name}_{year}_comparison.csv"
    try:
        # Add WSE ellipsoidal info to merged_df
        merged_df["antenna_ellipsoidal_height_m"] = antenna_ellipsoidal_height
        merged_df["usgs_gauge_stated_datum"] = usgs_gauge_stated_datum
        merged_df["usgs_site_code"] = gauge_info.get("site_code", "")
        merged_df["usgs_site_name"] = gauge_info.get("site_name", "")
        merged_df["optimal_time_lag_days"] = wse_lag

        # Make a copy of usgs_value_col for the CSV file
        merged_df["usgs_value"] = merged_df[usgs_value_col]

        # Save to CSV
        merged_df.to_csv(comparison_csv_path, index=False)
        logging.info(f"Saved comparison data to {comparison_csv_path}")
    except Exception as e:
        logging.error(f"Error saving comparison data: {e}")
        comparison_csv_path = None

    # Perform segmented correlation analysis if requested
    segmented_results = {}
    if perform_segmented_analysis and len(merged_df) >= 30:  # Only if enough data points
        try:
            logging.info("Performing segmented correlation analysis...")

            # Create demeaned columns in merged dataframe for segmented analysis
            if "wse_ellips_m" in merged_df.columns:
                merged_df["wse_ellips_m_demeaned"] = (
                    merged_df["wse_ellips_m"] - merged_df["wse_ellips_m"].mean()
                )
            if "usgs_value_m_median" in merged_df.columns:
                merged_df["usgs_value_m_median_demeaned"] = (
                    merged_df["usgs_value_m_median"] - merged_df["usgs_value_m_median"].mean()
                )

            segmented_results = perform_segmented_correlation_analysis(
                merged_df,
                int(year),
                station_name,
                output_dir,
                gnss_col="wse_ellips_m_demeaned",
                usgs_col="usgs_value_m_median_demeaned",
                min_points=10,
                logger=logging,
            )
            logging.info("Segmented correlation analysis completed successfully")
        except Exception as e:
            logging.error(f"Error performing segmented correlation analysis: {e}")
            segmented_results = {}
    elif perform_segmented_analysis:
        logging.warning(
            f"Not enough data points for meaningful segmented analysis (found {len(merged_df)})"
        )

    # Return results
    return {
        "success": True,
        "gnssir_daily_df": gnssir_daily_df,
        "usgs_daily_df": usgs_daily_df,
        "gauge_info": gauge_info,
        "correlations": {
            "rh_correlation": rh_correlation,
            "wse_correlation": wse_correlation,
        },
        "time_lag": time_lag_results,
        "plot_paths": {
            "wse_plot_path": wse_plot_path,
            "demeaned_plot_path": demeaned_plot_path,
            "lag_plot_path": lag_plot_path,
        },
        "comparison_csv_path": comparison_csv_path,
        "segmented_results": segmented_results,
    }


def perform_segmented_correlation_analysis(
    merged_df: pd.DataFrame,
    year: int,
    station: str,
    output_dir: Path,
    gnss_col: str = "wse_ellips_m_demeaned",
    usgs_col: str = "usgs_value_m_median_demeaned",
    min_points: int = 10,
    logger: logging.Logger = None,
) -> dict:
    """
    Perform segmented correlation analysis between GNSS-IR and USGS data.

    Parameters:
    -----------
    merged_df : DataFrame with both GNSS-IR and USGS data (date column required)
    year : Year for analysis
    station : Station ID
    output_dir : Output directory for saving results
    gnss_col : Column name for GNSS-IR data
    usgs_col : Column name for USGS data
    min_points : Minimum points for valid correlation
    logger : Logger

    Returns: Dictionary of analysis results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate segment definitions
    monthly_segments = generate_monthly_segments(year)
    seasonal_segments = generate_seasonal_segments(year)

    # Calculate correlations
    logger.info("Performing monthly segmented correlation analysis...")
    monthly_correlations, monthly_data = perform_segmented_correlation(
        merged_df, monthly_segments, gnss_col, usgs_col, min_points, logger
    )

    logger.info("Performing seasonal segmented correlation analysis...")
    seasonal_correlations, seasonal_data = perform_segmented_correlation(
        merged_df, seasonal_segments, gnss_col, usgs_col, min_points, logger
    )

    # Determine if demeaned or raw data was used
    data_type = "demeaned" if "demeaned" in gnss_col else "raw"

    # Create visualizations
    logger.info("Creating correlation bar plots...")
    monthly_fig = visualizer.plot_segment_correlations(
        monthly_correlations,
        title=f"{station} {year} - Monthly Correlation: GNSS-IR vs USGS ({data_type})",
        highlight_threshold=0.7,
        figsize=(12, 6),
        save_path=output_dir / f"{station}_{year}_monthly_correlation_{data_type}.png",
    )

    seasonal_fig = visualizer.plot_segment_correlations(
        seasonal_correlations,
        title=f"{station} {year} - Seasonal Correlation: GNSS-IR vs USGS ({data_type})",
        highlight_threshold=0.7,
        figsize=(10, 6),
        save_path=output_dir / f"{station}_{year}_seasonal_correlation_{data_type}.png",
    )

    # Create comparison grid plots
    logger.info("Creating segment comparison grid plots...")
    monthly_grid_fig = visualizer.plot_segment_comparison_grid(
        monthly_data,
        gnss_col,
        usgs_col,
        monthly_correlations,
        max_cols=4,
        figsize=(18, 12),
        save_path=output_dir / f"{station}_{year}_monthly_comparison_grid_{data_type}.png",
    )

    seasonal_grid_fig = visualizer.plot_segment_comparison_grid(
        seasonal_data,
        gnss_col,
        usgs_col,
        seasonal_correlations,
        max_cols=2,
        figsize=(14, 10),
        save_path=output_dir / f"{station}_{year}_seasonal_comparison_grid_{data_type}.png",
    )

    # Create time series by segment plot
    logger.info("Creating time series by segment plot...")
    time_series_fig = visualizer.plot_time_series_by_segment(
        merged_df,
        gnss_col,
        usgs_col,
        seasonal_segments,
        seasonal_correlations,
        station_name=station,
        demeaned="demeaned" in gnss_col,
        save_path=output_dir / f"{station}_{year}_seasonal_timeseries_{data_type}.png",
    )

    # Create correlation heatmap
    logger.info("Creating correlation heatmap...")
    heatmap_fig = visualizer.plot_heatmap_correlation_matrix(
        {**monthly_correlations, **seasonal_correlations},
        title=f"{station} {year} - Correlation Heatmap",
        save_path=output_dir / f"{station}_{year}_correlation_heatmap_{data_type}.png",
    )

    # Close figures to free memory
    plt.close(monthly_fig)
    plt.close(seasonal_fig)
    plt.close(monthly_grid_fig)
    plt.close(seasonal_grid_fig)
    plt.close(time_series_fig)
    plt.close(heatmap_fig)

    # Save results to a summary text file
    summary_path = output_dir / f"{station}_{year}_segmented_correlation_summary_{data_type}.txt"
    with open(summary_path, "w") as f:
        f.write(f"{station} {year} Segmented Correlation Analysis ({data_type})\n")
        f.write("=" * 60 + "\n\n")

        f.write("Monthly Correlations:\n")
        f.write("-" * 30 + "\n")
        for month, corr in monthly_correlations.items():
            if corr is not None:
                point_count = len(monthly_data[month])
                valid_count = (
                    monthly_data[month][gnss_col].notna() & monthly_data[month][usgs_col].notna()
                )
                valid_count = valid_count.sum()
                f.write(
                    f"{month}: {corr:.4f} ({valid_count} valid points of {point_count} total)\n"
                )
            else:
                f.write(f"{month}: Insufficient data\n")

        f.write("\nSeasonal Correlations:\n")
        f.write("-" * 30 + "\n")
        for season, corr in seasonal_correlations.items():
            if corr is not None:
                point_count = len(seasonal_data[season])
                valid_count = (
                    seasonal_data[season][gnss_col].notna()
                    & seasonal_data[season][usgs_col].notna()
                )
                valid_count = valid_count.sum()
                f.write(
                    f"{season}: {corr:.4f} ({valid_count} valid points of {point_count} total)\n"
                )
            else:
                f.write(f"{season}: Insufficient data\n")

        f.write("\nOverall Correlation:\n")
        f.write("-" * 30 + "\n")
        valid_mask = merged_df[gnss_col].notna() & merged_df[usgs_col].notna()
        overall_corr = merged_df.loc[valid_mask, gnss_col].corr(merged_df.loc[valid_mask, usgs_col])
        f.write(
            f"Full Period: {overall_corr:.4f} ({valid_mask.sum()} valid points of {len(merged_df)} total)\n"
        )

    logger.info(f"Segmented correlation analysis saved to {summary_path}")

    # Return results dictionary
    return {
        "monthly_correlations": monthly_correlations,
        "seasonal_correlations": seasonal_correlations,
        "plots": {
            "monthly_correlation": str(
                output_dir / f"{station}_{year}_monthly_correlation_{data_type}.png"
            ),
            "seasonal_correlation": str(
                output_dir / f"{station}_{year}_seasonal_correlation_{data_type}.png"
            ),
            "monthly_grid": str(
                output_dir / f"{station}_{year}_monthly_comparison_grid_{data_type}.png"
            ),
            "seasonal_grid": str(
                output_dir / f"{station}_{year}_seasonal_comparison_grid_{data_type}.png"
            ),
            "seasonal_timeseries": str(
                output_dir / f"{station}_{year}_seasonal_timeseries_{data_type}.png"
            ),
            "correlation_heatmap": str(
                output_dir / f"{station}_{year}_correlation_heatmap_{data_type}.png"
            ),
        },
        "summary_file": str(summary_path),
    }


def main():
    """Main function to run USGS comparison analysis"""
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Enhanced USGS Comparison for GNSS-IR")
    parser.add_argument("--station", type=str, required=True, help="Station ID (4-char uppercase)")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--doy_start", type=int, help="Starting day of year (optional)")
    parser.add_argument("--doy_end", type=int, help="Ending day of year (optional)")
    parser.add_argument("--max_lag_days", type=int, default=10, help="Maximum lag days to consider")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for plots and data files")
    parser.add_argument(
        "--skip_segmented", action="store_true", help="Skip segmented correlation analysis"
    )
    parser.add_argument(
        "--generate_quality_report",
        action="store_true",
        help="Generate automated data quality report",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{args.station}_{args.year}_comparison.log"),
            logging.StreamHandler(),
        ],
    )

    # Set DOY range if provided
    doy_range = None
    if args.doy_start is not None and args.doy_end is not None:
        doy_range = (args.doy_start, args.doy_end)
        logging.info(f"Processing DOY range: {doy_range[0]}-{doy_range[1]}")

    # Set output directory if provided
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        logging.info(f"Using custom output directory: {output_dir}")

    # Run analysis
    results = usgs_comparison(
        args.station,
        args.year,
        doy_range=doy_range,
        max_lag_days=args.max_lag_days,
        output_dir=output_dir,
        perform_segmented_analysis=not args.skip_segmented,
    )

    # Check if analysis was successful
    if results.get("success", False):
        logging.info("Enhanced USGS comparison analysis completed successfully")

        # Log correlation results
        correlations = results.get("correlations", {})
        if correlations:
            rh_corr = correlations.get("rh_correlation")
            wse_corr = correlations.get("wse_correlation")
            rh_corr_str = (
                f"{rh_corr:.4f}" if rh_corr is not None and not pd.isna(rh_corr) else "N/A"
            )
            wse_corr_str = (
                f"{wse_corr:.4f}" if wse_corr is not None and not pd.isna(wse_corr) else "N/A"
            )
            logging.info(f"RH-USGS correlation: {rh_corr_str}")
            logging.info(f"WSE-USGS correlation: {wse_corr_str}")

        # Log time lag results
        time_lag = results.get("time_lag", {})
        if time_lag:
            wse_lag_days = time_lag.get("wse_lag_days")
            wse_lag_conf = time_lag.get("wse_lag_confidence", "N/A")
            wse_lag_corr = time_lag.get("wse_lag_correlation")
            wse_lag_corr_str = (
                f"{wse_lag_corr:.4f}"
                if wse_lag_corr is not None and not pd.isna(wse_lag_corr)
                else "N/A"
            )

            logging.info(
                f"Optimal WSE time lag: {wse_lag_days if wse_lag_days is not None else 'N/A'} days (confidence: {wse_lag_conf})"
            )
            logging.info(f"Lag-adjusted correlation: {wse_lag_corr_str}")

        # Log output paths
        plot_paths = results.get("plot_paths", {})
        if plot_paths:
            for name, path in plot_paths.items():
                if path:
                    logging.info(f"{name}: {path}")

        if results.get("comparison_csv_path"):
            logging.info(f"Enhanced comparison data: {results['comparison_csv_path']}")

        # Log segmented correlation results
        if "segmented_results" in results and results["segmented_results"]:
            seg_results = results["segmented_results"]
            if "summary_file" in seg_results:
                logging.info(
                    f"Segmented correlation analysis summary: {seg_results['summary_file']}"
                )

            # Log sample of correlations
            if "seasonal_correlations" in seg_results:
                seasons = list(seg_results["seasonal_correlations"].keys())
                if seasons:
                    logging.info("Seasonal correlation examples:")
                    for season in seasons:
                        corr = seg_results["seasonal_correlations"].get(season)
                        if corr is not None:
                            logging.info(f"  {season}: {corr:.4f}")

            # Log plot paths
            if "plots" in seg_results:
                logging.info("Segmented correlation plots:")
                for name, path in seg_results["plots"].items():
                    logging.info(f"  {name}: {path}")

        # Generate quality report if requested
        if args.generate_quality_report:
            try:
                logging.info("Generating automated data quality report...")
                from data_quality_reporter import DataQualityReporter

                reporter = DataQualityReporter(args.station, args.year)
                report_results = reporter.generate_report(
                    doy_start=args.doy_start, doy_end=args.doy_end
                )

                logging.info(f"✅ Quality report generated: {report_results['report_path']}")

            except Exception as e:
                logging.error(f"Error generating quality report: {e}")
    else:
        logging.error(
            f"Enhanced USGS comparison analysis failed: {results.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()
