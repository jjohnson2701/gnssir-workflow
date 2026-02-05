# ABOUTME: Tide prediction integration for GNSS-IR validation
# ABOUTME: Fetches NOAA CO-OPS predictions and calculates tide residuals

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from .base import ensure_output_dir, add_summary_textbox, PLOT_COLORS, PLOT_STYLES

# Constants for API endpoints and data sources
NOAA_COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"


def find_nearest_tide_stations(
    latitude: float, longitude: float, radius_km: float = 50, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the nearest NOAA tide stations to a given location.

    Args:
        latitude: Location latitude in decimal degrees
        longitude: Location longitude in decimal degrees
        radius_km: Search radius in kilometers
        limit: Maximum number of stations to return

    Returns:
        List of dictionaries containing station information
    """
    # In a real implementation, this would query the NOAA CO-OPS API
    # For this example, we'll use a simulated response
    logging.info(f"Finding tide stations near ({latitude}, {longitude}) within {radius_km} km")

    try:
        # Use the NOAA API to get station data
        # This would be replaced with an actual API call
        stations = [
            {
                "id": "8651370",
                "name": "Duck, NC",
                "lat": 36.1833,
                "lon": -75.7467,
                "distance_km": 32.1,
                "type": "tide",
                "reference_datum": "NAVD88",
                "data_types": ["predictions", "water_level", "datums"],
            },
            {
                "id": "8652587",
                "name": "Oregon Inlet Marina, NC",
                "lat": 35.7950,
                "lon": -75.5483,
                "distance_km": 15.3,
                "type": "tide",
                "reference_datum": "NAVD88",
                "data_types": ["predictions", "water_level", "datums"],
            },
        ]

        # Sort by distance and limit results
        return stations[:limit]

    except Exception as e:
        logging.error(f"Error finding tide stations: {e}")
        return []


def get_noaa_tide_predictions(
    station_id: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    datum: str = "NAVD",
    interval: str = "6",  # 6-minute interval (highest resolution)
    units: str = "metric",
    return_as_df: bool = True,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Get tide predictions from NOAA CO-OPS API.

    Args:
        station_id: NOAA station ID (e.g., '8651370')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        datum: Vertical datum for predictions (NAVD, MSL, MLLW, etc.)
        interval: Prediction interval ('6', '60', 'hilo' for high/low tide only)
        units: 'metric' or 'english'
        return_as_df: Whether to return data as a DataFrame (True) or dict (False)

    Returns:
        DataFrame or dictionary with tide predictions
    """
    logging.info(
        f"Fetching NOAA tide predictions for station {station_id} from {start_date} to {end_date}"
    )

    try:
        # In a real implementation, this would fetch data from the NOAA CO-OPS API
        # For this example, we'll generate synthetic data

        # Create date range with 6-minute interval
        if interval == "6":
            freq = "6min"
        elif interval == "60":
            freq = "60min"
        else:  # 'hilo'
            # For high/low tides only, we'll generate 4 data points per day
            freq = "6h"

        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Generate synthetic tide predictions
        np.random.seed(42)  # For reproducible results

        # Base tide pattern with semidiurnal and monthly variations
        t = np.arange(len(date_range)) / (24 * 60 / 6)  # Convert to days for 6-minute interval

        # Semidiurnal tide (two high/low cycles per day) plus monthly modulation
        tide_level = (
            0.0  # Mean water level
            + 0.5 * np.sin(2 * np.pi * t / 0.517)  # M2 semidiurnal tide (~12.42 hours)
            + 0.2 * np.sin(2 * np.pi * t / 1.0)  # K1 diurnal tide (~24 hours)
            + 0.3 * np.sin(2 * np.pi * t / 14.77)  # Spring/neap cycle (~14.77 days)
            + 0.1 * np.sin(2 * np.pi * t / 29.53)  # Monthly cycle (~29.53 days)
            + 0.03 * np.random.randn(len(t))  # Small random variations
        )

        # Create DataFrame
        df = pd.DataFrame(
            {"datetime": date_range, "predicted_wl_m": tide_level, "station_id": station_id}
        )

        # Add date column
        df["date"] = df["datetime"].dt.date

        # Return as requested format
        if return_as_df:
            return df
        else:
            # Convert to dictionary format
            return {
                "station_id": station_id,
                "data": df.to_dict(orient="records"),
                "start_date": start_date,
                "end_date": end_date,
                "datum": datum,
                "units": units,
            }

    except Exception as e:
        logging.error(f"Error fetching NOAA tide predictions: {e}")
        if return_as_df:
            return pd.DataFrame()
        else:
            return {"error": str(e), "station_id": station_id}


def generate_synthetic_tide_predictions(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    latitude: float = 35.88,
    longitude: float = -75.65,
    interval: str = "6",
    mean_level: float = 0.0,
    amplitude: float = 0.8,
    phase_shift: float = 0.0,
    noise_level: float = 0.03,
) -> pd.DataFrame:
    """
    Generate synthetic tide predictions for a location.
    Used when NOAA data is unavailable.

    Args:
        start_date: Start date for predictions
        end_date: End date for predictions
        latitude: Location latitude (affects diurnal inequality)
        longitude: Location longitude (affects phase)
        interval: Prediction interval ('6', '60', 'hilo')
        mean_level: Mean water level in meters
        amplitude: Tide amplitude in meters
        phase_shift: Phase shift in radians
        noise_level: Standard deviation of random noise

    Returns:
        DataFrame with synthetic tide predictions
    """
    logging.info(f"Generating synthetic tide predictions from {start_date} to {end_date}")

    try:
        # Determine frequency based on interval
        if interval == "6":
            freq = "6min"
        elif interval == "60":
            freq = "60min"
        else:  # 'hilo'
            freq = "6h"

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Create time array in days since start
        t = np.array([(d - start_date).total_seconds() / 86400 for d in date_range])

        # Calculate tide components
        # M2 (principal lunar semidiurnal) - 12.42 hour period
        m2_period = 12.42 / 24  # in days
        m2_amp = amplitude * 0.7  # 70% of total amplitude
        m2 = m2_amp * np.sin(2 * np.pi * t / m2_period + phase_shift)

        # S2 (principal solar semidiurnal) - 12.00 hour period
        s2_period = 12.0 / 24  # in days
        s2_amp = amplitude * 0.3  # 30% of total amplitude
        s2 = s2_amp * np.sin(2 * np.pi * t / s2_period + phase_shift + 0.2)

        # Spring/neap cycle - 14.77 day period
        spring_neap_period = 14.77  # in days
        spring_neap = 0.3 * amplitude * np.sin(2 * np.pi * t / spring_neap_period)

        # Annual cycle - 365.25 day period
        annual_period = 365.25  # in days
        annual = 0.2 * amplitude * np.sin(2 * np.pi * t / annual_period)

        # Combine components
        tide_level = (
            mean_level
            + m2
            + s2  # Main semidiurnal constituents
            + spring_neap  # Spring/neap modulation
            + annual  # Annual cycle
            + noise_level * np.random.randn(len(t))  # Random noise
        )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "datetime": date_range,
                "predicted_wl_m": tide_level,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

        # Add date column
        df["date"] = df["datetime"].dt.date

        return df

    except Exception as e:
        logging.error(f"Error generating synthetic tide predictions: {e}")
        return pd.DataFrame()


def get_high_low_tide_times(
    tide_df: pd.DataFrame,
    value_col: str = "predicted_wl_m",
    datetime_col: str = "datetime",
    min_separation_hours: float = 4.0,
) -> pd.DataFrame:
    """
    Extract high and low tide times from tide predictions.

    Args:
        tide_df: DataFrame with tide predictions
        value_col: Column name for water level values
        datetime_col: Column name for datetime
        min_separation_hours: Minimum time between successive high or low tides

    Returns:
        DataFrame with high and low tide times and values
    """
    logging.info("Extracting high and low tide times")

    try:
        # Check if DataFrame is empty
        if tide_df.empty:
            logging.warning("Empty tide DataFrame provided")
            return pd.DataFrame()

        # Check if required columns exist
        required_cols = [value_col, datetime_col]
        for col in required_cols:
            if col not in tide_df.columns:
                logging.error(f"Required column '{col}' not found in tide DataFrame")
                return pd.DataFrame()

        # Make a copy to avoid modifying the original
        df = tide_df.copy()

        # Ensure datetime is in datetime format
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Sort by datetime
        df = df.sort_values(by=datetime_col)

        # Calculate forward and backward differences
        df["forward_diff"] = df[value_col].diff().shift(-1)
        df["backward_diff"] = df[value_col].diff()

        # Identify local maxima (high tides) and minima (low tides)
        high_tides = df[(df["backward_diff"] > 0) & (df["forward_diff"] < 0)].copy()
        low_tides = df[(df["backward_diff"] < 0) & (df["forward_diff"] > 0)].copy()

        # Add tide type column
        high_tides["tide_type"] = "High"
        low_tides["tide_type"] = "Low"

        # Combine high and low tides
        hilo_tides = pd.concat([high_tides, low_tides]).sort_values(by=datetime_col)

        # Filter out closely spaced tides of the same type
        filtered_hilo = []
        current_type = None
        last_time = None

        for _, row in hilo_tides.iterrows():
            this_type = row["tide_type"]
            this_time = row[datetime_col]

            if current_type != this_type or last_time is None:
                # Different tide type or first entry
                filtered_hilo.append(row)
                current_type = this_type
                last_time = this_time
            elif (this_time - last_time).total_seconds() / 3600 >= min_separation_hours:
                # Same tide type but sufficiently separated
                filtered_hilo.append(row)
                last_time = this_time

        # Create final DataFrame
        result_df = pd.DataFrame(filtered_hilo)

        # Drop intermediate columns
        if "forward_diff" in result_df.columns:
            result_df = result_df.drop(columns=["forward_diff", "backward_diff"])

        return result_df

    except Exception as e:
        logging.error(f"Error extracting high/low tide times: {e}")
        return pd.DataFrame()


def calculate_tide_residuals(
    observed_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    observed_datetime_col: str = "datetime",
    observed_wl_col: str = "wse_ellips_m",
    tide_datetime_col: str = "datetime",
    tide_wl_col: str = "predicted_wl_m",
    mean_offset: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate residuals between observed water levels and tide predictions.

    Args:
        observed_df: DataFrame with observed water levels
        tide_df: DataFrame with tide predictions
        observed_datetime_col: Column name for datetime in observed_df
        observed_wl_col: Column name for water level in observed_df
        tide_datetime_col: Column name for datetime in tide_df
        tide_wl_col: Column name for water level in tide_df
        mean_offset: Optional mean offset to apply to tide predictions
            If None, the mean difference between observed and predicted will be used

    Returns:
        DataFrame with observed water levels, tide predictions, and residuals
    """
    logging.info("Calculating tide residuals")

    try:
        # Check if DataFrames are empty
        if observed_df.empty:
            logging.warning("Empty observed DataFrame provided")
            return pd.DataFrame()

        if tide_df.empty:
            logging.warning("Empty tide DataFrame provided")
            return pd.DataFrame()

        # Make copies to avoid modifying originals
        obs_df = observed_df.copy()
        tide_df = tide_df.copy()

        # Ensure datetime columns are in datetime format
        obs_df[observed_datetime_col] = pd.to_datetime(obs_df[observed_datetime_col])
        tide_df[tide_datetime_col] = pd.to_datetime(tide_df[tide_datetime_col])

        # Interpolate tide predictions to observed times
        # Set datetime as index for interpolation
        tide_df = tide_df.set_index(tide_datetime_col)

        # Resample to 1-minute interval for better interpolation
        resampled_tide = tide_df.resample("1min").interpolate(method="cubic")

        # Get tide predictions at observed times
        observed_times = obs_df[observed_datetime_col]

        # Initialize tide predictions array
        tide_predictions = np.full(len(observed_times), np.nan)

        # Interpolate tide predictions for each observed time
        for i, obs_time in enumerate(observed_times):
            try:
                # Find closest time in resampled tide data
                closest_time = resampled_tide.index[
                    resampled_tide.index.get_indexer([obs_time], method="nearest")[0]
                ]

                # Get interpolated tide prediction
                tide_predictions[i] = resampled_tide.loc[closest_time, tide_wl_col]
            except (KeyError, IndexError, ValueError):
                # If interpolation fails, leave as NaN
                pass

        # Add tide predictions to observed DataFrame
        obs_df["tide_prediction_m"] = tide_predictions

        # Calculate mean offset if not provided
        if mean_offset is None:
            # Use only finite values for mean calculation
            valid_mask = np.isfinite(obs_df[observed_wl_col]) & np.isfinite(
                obs_df["tide_prediction_m"]
            )
            if valid_mask.sum() > 0:
                mean_offset = np.mean(
                    obs_df.loc[valid_mask, observed_wl_col]
                    - obs_df.loc[valid_mask, "tide_prediction_m"]
                )
            else:
                mean_offset = 0.0

        # Apply offset to tide predictions
        obs_df["tide_prediction_adjusted_m"] = obs_df["tide_prediction_m"] + mean_offset

        # Calculate residuals
        obs_df["tide_residual_m"] = obs_df[observed_wl_col] - obs_df["tide_prediction_adjusted_m"]

        # Add offset value as metadata
        obs_df["tide_mean_offset_m"] = mean_offset

        return obs_df

    except Exception as e:
        logging.error(f"Error calculating tide residuals: {e}")
        return pd.DataFrame()


def plot_subdaily_rh_vs_tide(
    gnssir_raw_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    usgs_iv_df: Optional[pd.DataFrame] = None,
    station_name: str = "GNSS",
    tide_station_info: Optional[Dict[str, Any]] = None,
    usgs_gauge_info: Optional[Dict[str, Any]] = None,
    output_plot_path: Union[str, Path] = "subdaily_rh_vs_tide.png",
    gnssir_datetime_col: str = "timeUTC",
    gnssir_rh_col: str = "reflHeight",
    gnssir_amp_col: Optional[str] = "ampDirect",
    gnssir_azimuth_col: Optional[str] = "azimuth",
    gnssir_satellite_col: Optional[str] = "sat",
    tide_datetime_col: str = "datetime",
    tide_wl_col: str = "predicted_wl_m",
    usgs_datetime_col: Optional[str] = "datetime",
    usgs_wl_col: Optional[str] = "value",
    antenna_height: Optional[float] = None,
    mean_offset: Optional[float] = None,
    plot_residuals: bool = True,
    color_by_azimuth: bool = True,
    style: str = "default",
) -> Optional[Path]:
    """
    Create a multi-panel visualization of sub-daily GNSS-IR reflector height retrievals
    compared with tide predictions and optional USGS instantaneous water level data.

    Args:
        gnssir_raw_df: DataFrame with raw (sub-daily) GNSS-IR reflector height retrievals
        tide_df: DataFrame with tide predictions
        usgs_iv_df: Optional DataFrame with USGS instantaneous water level data
        station_name: GNSS-IR station name
        tide_station_info: Dictionary with tide station information
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Path to save the plot
        gnssir_datetime_col: Column name for timestamp in GNSS-IR data
        gnssir_rh_col: Column name for reflector height in GNSS-IR data
        gnssir_amp_col: Column name for amplitude in GNSS-IR data
        gnssir_azimuth_col: Column name for azimuth in GNSS-IR data
        gnssir_satellite_col: Column name for satellite ID in GNSS-IR data
        tide_datetime_col: Column name for timestamp in tide data
        tide_wl_col: Column name for water level in tide data
        usgs_datetime_col: Column name for timestamp in USGS data
        usgs_wl_col: Column name for water level in USGS data
        antenna_height: Antenna height above the ellipsoid for WSE calculation
        mean_offset: Mean offset to apply between data series (if None, computed automatically)
        plot_residuals: Whether to include residuals panel
        color_by_azimuth: Whether to color RH retrievals by azimuth
        style: Plot style to use

    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)

    try:
        # Check if DataFrames are valid
        if gnssir_raw_df is None or gnssir_raw_df.empty:
            logging.error("Raw GNSS-IR data DataFrame is empty")
            return None

        if tide_df is None or tide_df.empty:
            logging.error("Tide predictions DataFrame is empty")
            return None

        # Check if required columns exist
        for col in [gnssir_datetime_col, gnssir_rh_col]:
            if col not in gnssir_raw_df.columns:
                logging.error(f"Required column '{col}' not found in GNSS-IR data")
                return None

        for col in [tide_datetime_col, tide_wl_col]:
            if col not in tide_df.columns:
                logging.error(f"Required column '{col}' not found in tide data")
                return None

        # Check if USGS data is provided and valid
        has_usgs_data = False
        if usgs_iv_df is not None and not usgs_iv_df.empty:
            if usgs_datetime_col in usgs_iv_df.columns and usgs_wl_col in usgs_iv_df.columns:
                has_usgs_data = True
            else:
                logging.warning("USGS data provided but required columns not found")

        # Extract metadata
        tide_station_id = tide_station_info.get("id", "Unknown") if tide_station_info else "Unknown"
        tide_station_name = (
            tide_station_info.get("name", f"Station {tide_station_id}")
            if tide_station_info
            else "Tide Station"
        )
        tide_datum = (
            tide_station_info.get("reference_datum", "Unknown") if tide_station_info else "Unknown"
        )

        usgs_site_code = (
            usgs_gauge_info.get("site_code", "Unknown") if usgs_gauge_info else "Unknown"
        )
        usgs_vertical_datum = (
            usgs_gauge_info.get("vertical_datum", "Unknown") if usgs_gauge_info else "Unknown"
        )

        # Convert datetimes to pandas datetime if not already
        gnssir_raw_df[gnssir_datetime_col] = pd.to_datetime(gnssir_raw_df[gnssir_datetime_col])
        tide_df[tide_datetime_col] = pd.to_datetime(tide_df[tide_datetime_col])

        if has_usgs_data:
            usgs_iv_df[usgs_datetime_col] = pd.to_datetime(usgs_iv_df[usgs_datetime_col])

        # Convert GNSS-IR RH to WSE if antenna height is provided
        if antenna_height is not None:
            gnssir_raw_df["wse_ellips_m"] = antenna_height - gnssir_raw_df[gnssir_rh_col]
            gnssir_value_col = "wse_ellips_m"
            gnssir_label = f"{station_name} WSE (m)"
        else:
            gnssir_value_col = gnssir_rh_col
            gnssir_label = f"{station_name} RH (m)"

        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value

        # Determine number of panels
        if plot_residuals:
            n_panels = 2
        else:
            n_panels = 1

        # Create figure with GridSpec
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(
            n_panels, 1, height_ratios=[3, 1] if n_panels == 2 else [1], hspace=0.15
        )

        # Panel 1: Sub-daily RH and Tide
        ax1 = fig.add_subplot(gs[0])

        # Plot tide predictions as continuous line
        ax1.plot(
            tide_df[tide_datetime_col],
            tide_df[tide_wl_col],
            "-",
            label=f"Tide Prediction ({tide_station_id})",
            color=PLOT_COLORS.get("tide", "dodgerblue"),
            linewidth=2,
            alpha=0.7,
        )

        # Plot USGS instantaneous data if available
        if has_usgs_data:
            ax1.plot(
                usgs_iv_df[usgs_datetime_col],
                usgs_iv_df[usgs_wl_col],
                "s-",
                label=f"USGS {usgs_site_code} ({usgs_vertical_datum})",
                color=PLOT_COLORS.get("usgs", "green"),
                linewidth=1,
                markersize=4,
                alpha=0.7,
            )

        # Plot GNSS-IR RH retrievals as scatter points
        if color_by_azimuth and gnssir_azimuth_col in gnssir_raw_df.columns:
            # Color points by azimuth
            scatter = ax1.scatter(
                gnssir_raw_df[gnssir_datetime_col],
                gnssir_raw_df[gnssir_value_col],
                c=gnssir_raw_df[gnssir_azimuth_col],
                cmap="hsv",
                s=30,
                alpha=0.7,
                label=gnssir_label,
            )

            # Add colorbar for azimuth
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label("Azimuth (°)", rotation=270, labelpad=15)
        else:
            # Simple scatter without coloring
            ax1.scatter(
                gnssir_raw_df[gnssir_datetime_col],
                gnssir_raw_df[gnssir_value_col],
                color=PLOT_COLORS.get("gnssir", "red"),
                s=30,
                alpha=0.7,
                label=gnssir_label,
            )

        # Add satellite labels if requested and available
        if gnssir_satellite_col in gnssir_raw_df.columns:
            # Add satellite ID to a subset of points to avoid overcrowding
            unique_sats = gnssir_raw_df[gnssir_satellite_col].unique()
            for sat in unique_sats:
                # Get first occurrence of each satellite
                sat_df = gnssir_raw_df[gnssir_raw_df[gnssir_satellite_col] == sat].iloc[0:1]

                # Add text label slightly offset from the point
                for idx, row in sat_df.iterrows():
                    ax1.annotate(
                        f"{row[gnssir_satellite_col]}",
                        (row[gnssir_datetime_col], row[gnssir_value_col]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

        # Calculate and apply offset if needed
        if mean_offset is None and antenna_height is not None:
            # Find times where both tide and GNSS-IR data are available
            gnssir_times = gnssir_raw_df[gnssir_datetime_col].values
            tide_times = tide_df[tide_datetime_col].values

            # Interpolate tide data to GNSS-IR times
            tide_interp = np.interp(
                mdates.date2num(gnssir_times),
                mdates.date2num(tide_times),
                tide_df[tide_wl_col].values,
            )

            # Calculate mean offset
            mean_offset = np.nanmean(gnssir_raw_df[gnssir_value_col].values - tide_interp)
            logging.info(f"Calculated mean offset: {mean_offset:.3f} m")

        # Set title and labels
        title_parts = [f"{station_name} Sub-daily RH vs Tide Predictions"]
        if antenna_height is not None:
            title_parts.append(f"(Antenna Height: {antenna_height:.2f} m)")
        if mean_offset is not None:
            title_parts.append(f"(Mean Offset: {mean_offset:.2f} m)")

        ax1.set_title(" ".join(title_parts), fontsize=16)
        ax1.set_ylabel("Water Level (m)", fontsize=14)
        ax1.grid(True, alpha=0.3, color=PLOT_COLORS.get("grid", "lightgray"))
        ax1.legend(fontsize=12)

        # Format x-axis for dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Add tide station info
        tide_info_text = {
            "Tide Station": tide_station_id,
            "Name": tide_station_name,
            "Datum": tide_datum,
        }
        add_summary_textbox(plt, ax1, tide_info_text, position=(0.02, 0.02), fontsize=10)

        # Panel 2: Residuals (if requested)
        if plot_residuals:
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            # Calculate residuals between GNSS-IR and interpolated tide
            gnssir_times_num = mdates.date2num(gnssir_raw_df[gnssir_datetime_col].values)
            tide_times_num = mdates.date2num(tide_df[tide_datetime_col].values)

            # Interpolate tide data to GNSS-IR times
            tide_interp = np.interp(gnssir_times_num, tide_times_num, tide_df[tide_wl_col].values)

            # Apply offset if provided
            if mean_offset is not None:
                tide_interp += mean_offset

            # Calculate residuals
            residuals = gnssir_raw_df[gnssir_value_col].values - tide_interp

            # Plot residuals
            if color_by_azimuth and gnssir_azimuth_col in gnssir_raw_df.columns:
                ax2.scatter(
                    gnssir_raw_df[gnssir_datetime_col],
                    residuals,
                    c=gnssir_raw_df[gnssir_azimuth_col],
                    cmap="hsv",
                    s=30,
                    alpha=0.7,
                )
            else:
                ax2.scatter(
                    gnssir_raw_df[gnssir_datetime_col],
                    residuals,
                    color=PLOT_COLORS.get("gnssir", "red"),
                    s=30,
                    alpha=0.7,
                )

            # Add horizontal line at zero
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            # Calculate and display statistics for residuals
            residual_stats = {
                "Mean Residual": f"{np.nanmean(residuals):.3f} m",
                "Std Dev": f"{np.nanstd(residuals):.3f} m",
                "RMSE": f"{np.sqrt(np.nanmean(residuals**2)):.3f} m",
                "Data Points": f"{len(residuals)}",
            }
            add_summary_textbox(plt, ax2, residual_stats, position=(0.70, 0.02), fontsize=10)

            # Set labels
            ax2.set_ylabel("Residuals (m)\nGNSS-IR - Tide", fontsize=14)
            ax2.set_xlabel("Date and Time (UTC)", fontsize=14)
            ax2.grid(True, alpha=0.3, color=PLOT_COLORS.get("grid", "lightgray"))

            # Format x-axis for dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.xticks(rotation=45)
        else:
            # If not plotting residuals, add x-label to main panel
            ax1.set_xlabel("Date and Time (UTC)", fontsize=14)
            plt.xticks(rotation=45)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()

        logging.info(f"Sub-daily RH vs tide plot saved to {output_plot_path}")
        return output_plot_path

    except Exception as e:
        logging.error(f"Error creating sub-daily RH vs tide plot: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return None
