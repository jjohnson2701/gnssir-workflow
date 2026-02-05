# ABOUTME: Time series plotting for GNSS-IR reflector height data
# ABOUTME: Creates annual RH plots with optional reference gauge overlay

import logging
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from .base import (
    find_column_by_name,
    add_summary_textbox,
    PLOT_COLORS,
    PLOT_STYLES,
)


def plot_annual_rh_timeseries(
    combined_rh_csv_path: Union[str, Path],
    station_name: str,
    year: int,
    annual_results_dir: Union[str, Path],
    reference_gauge_data_path: Optional[Union[str, Path]] = None,
    highlight_dates: Optional[List[str]] = None,
    style: str = "default",
) -> Optional[Path]:
    """
    Generate a time series plot of Reflector Height vs. Time.

    Args:
        combined_rh_csv_path: Path to combined RH CSV
        station_name: Station name (e.g., "FORA")
        year: Year (4 digits)
        annual_results_dir: Output directory for plot
        reference_gauge_data_path: Path to reference gauge data
        highlight_dates: List of dates to highlight on the plot (YYYY-MM-DD format)
        style: Plot style to use

    Returns:
        Path to the generated plot file on success, None on failure
    """
    combined_rh_csv_path = Path(combined_rh_csv_path)
    annual_results_dir = Path(annual_results_dir)

    # Ensure the output directory exists
    annual_results_dir.mkdir(parents=True, exist_ok=True)

    # Define output plot path
    output_plot_path = annual_results_dir / f"{station_name}_{year}_annual_waterlevel.png"

    try:
        # Check if the CSV file exists
        if not combined_rh_csv_path.exists():
            logging.error(f"Combined RH CSV file not found at {combined_rh_csv_path}")
            return None

        # Read the combined CSV
        try:
            df = pd.read_csv(combined_rh_csv_path)
        except pd.errors.EmptyDataError:
            logging.error(f"CSV file is empty: {combined_rh_csv_path}")
            return None

        # Check if DataFrame is empty
        if df.empty:
            logging.error(f"DataFrame is empty after reading {combined_rh_csv_path}")
            return None

        # Log the dataframe structure
        logging.debug(f"DataFrame columns: {df.columns.tolist()}")
        logging.debug(f"DataFrame shape: {df.shape}")
        logging.debug(f"DataFrame sample: {df.head()}")

        # Try to find the reflector height column
        rh_column = find_column_by_name(
            df,
            ["RH", "rh", "reflector_height", "height", "Col3"],
            column_position=2,
            column_type="reflector height",
        )

        if rh_column is None:
            logging.error("Could not identify a suitable RH column")
            return None

        # Find a suitable x-axis (time) column
        time_column = None
        date_present = False

        # First check if we have a date column that we can use directly
        if "date" in df.columns:
            # Verify it's actually a date
            try:
                pd.to_datetime(df["date"])
                time_column = "date"
                date_present = True
                logging.info("Using 'date' column for x-axis")
            except (ValueError, TypeError):
                logging.warning("'date' column exists but couldn't be parsed as datetime")

        # If no date column, try other common names or positions
        if not time_column:
            time_column = find_column_by_name(
                df,
                ["Datetime", "Date", "DOY", "doy", "UTCtime", "MJD"],
                column_position=None,
                column_type="time",
            )

        # If not found and we have at least 5 columns, use the 5th column (UTCtime, typically)
        if time_column is None and len(df.columns) >= 5:
            time_column = df.columns[4]  # UTCtime is typically the 5th column (index 4)
            logging.info(f"Using column at position 5 as time column: {time_column}")

        # If still not found, use DOY (which should be the 2nd column) as fallback
        if time_column is None and len(df.columns) >= 2:
            time_column = df.columns[1]  # DOY is typically the 2nd column (index 1)
            logging.info(f"Using column at position 2 as DOY column: {time_column}")

        if time_column is None:
            logging.error("Could not identify a suitable time/date column")
            return None

        # Apply the selected style
        if style in PLOT_STYLES:
            for key, value in PLOT_STYLES[style].items():
                plt.rcParams[key] = value

        # Create the plot
        plt.figure(figsize=(12, 8))

        # If using a datetime column, convert it first
        x_vals = df[time_column]
        if date_present:
            try:
                x_vals = pd.to_datetime(x_vals)
                # Configure date formatting on x-axis
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            except Exception as e:
                logging.warning(f"Error converting date column: {e}")

        # Plot RH data
        plt.scatter(
            x_vals,
            df[rh_column],
            label=f"{station_name} Reflector Height",
            alpha=0.7,
            s=15,
            color=PLOT_COLORS["gnssir"],
        )

        # Add trendline
        try:
            if len(df) > 1:  # Need at least 2 points for a line
                # If date_present, convert to ordinal for trendline calc
                if date_present:
                    x_numeric = mdates.date2num(x_vals)
                else:
                    x_numeric = x_vals

                # Simple linear regression
                z = np.polyfit(x_numeric, df[rh_column], 1)
                p = np.poly1d(z)

                # Add trendline to plot
                plt.plot(
                    x_vals,
                    p(x_numeric),
                    "--",
                    alpha=0.8,
                    color=PLOT_COLORS["trend"],
                    label=f"Trend (slope: {z[0]:.5f})",
                )
                logging.info(f"Added trendline with slope: {z[0]:.5f}")
        except Exception as e:
            logging.warning(f"Could not add trendline: {e}")

        # Plot reference gauge data if provided
        if reference_gauge_data_path:
            try:
                ref_path = Path(reference_gauge_data_path)
                ref_df = pd.read_csv(ref_path)

                # Identify date/time and water level columns
                # This would need customization based on the reference data format
                ref_time_col = [
                    col
                    for col in ref_df.columns
                    if any(time_str in col.lower() for time_str in ["time", "date"])
                ][0]

                ref_level_col = [
                    col
                    for col in ref_df.columns
                    if any(
                        level_str in col.lower() for level_str in ["level", "height", "elevation"]
                    )
                ][0]

                # Ensure datetime format
                ref_df[ref_time_col] = pd.to_datetime(ref_df[ref_time_col])

                # Plot reference data
                plt.plot(
                    ref_df[ref_time_col],
                    ref_df[ref_level_col],
                    label="Reference Gauge",
                    color=PLOT_COLORS["reference"],
                    alpha=0.7,
                )

                logging.info("Added reference gauge data to the plot")
            except Exception as e:
                logging.error(f"Error adding reference gauge data: {e}")

        # Highlight specific dates if requested
        if highlight_dates and date_present:
            try:
                highlight_dates_dt = pd.to_datetime(highlight_dates)
                for date in highlight_dates_dt:
                    # Find the closest data point
                    closest_idx = abs(x_vals - date).argmin()
                    plt.scatter(
                        x_vals.iloc[closest_idx],
                        df[rh_column].iloc[closest_idx],
                        color=PLOT_COLORS["highlight"],
                        s=100,
                        zorder=10,
                        label=(
                            f"Highlight: {date.strftime('%Y-%m-%d')}"
                            if date == highlight_dates_dt[0]
                            else ""
                        ),
                    )

                    # Add vertical line
                    plt.axvline(x=date, color=PLOT_COLORS["highlight"], linestyle="--", alpha=0.5)

                logging.info(f"Highlighted {len(highlight_dates)} specific dates")
            except Exception as e:
                logging.warning(f"Could not highlight dates: {e}")

        # Add title and labels
        plt.title(f"{station_name} Reflector Height Time Series - {year}", fontsize=16)
        x_label = "Date" if date_present else time_column
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Reflector Height (m)", fontsize=14)

        # Add grid and legend
        plt.grid(True, alpha=0.3, color=PLOT_COLORS["grid"])
        plt.legend(fontsize=12)

        # Add data summary in text box
        summary_stats = {
            "Total Points": len(df),
            "Min RH": f"{df[rh_column].min():.2f} m",
            "Max RH": f"{df[rh_column].max():.2f} m",
            "Mean RH": f"{df[rh_column].mean():.2f} m",
        }
        add_summary_textbox(plt, df, summary_stats)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()

        logging.info(f"Time series plot saved to {output_plot_path}")
        return output_plot_path

    except Exception as e:
        logging.error(f"Error creating RH time series plot: {e}")
        return None
