# ABOUTME: Time lag analysis between GNSS-IR and reference gauge data
# ABOUTME: Computes cross-correlation and optimal lag for tidal propagation delays

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .base import ensure_output_dir, PLOT_COLORS


def calculate_optimal_time_lag(
    gnssir_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    gnssir_col: str = "rh_median_m",
    usgs_col: str = "usgs_value_m_median",
    max_lag_days: int = 5,
    min_points_for_confidence: int = 7,
) -> Dict[str, Any]:
    """
    Calculate the optimal time lag between GNSS-IR and USGS gauge data.

    Args:
        gnssir_df: DataFrame with daily GNSS-IR data
        usgs_df: DataFrame with daily USGS gauge data
        gnssir_col: Column name for GNSS-IR data
        usgs_col: Column name for USGS gauge data
        max_lag_days: Maximum lag in days to consider
        min_points_for_confidence: Minimum number of overlapping points needed for confidence

    Returns:
        Dictionary containing optimal lag info:
            - optimal_lag_days: Optimal lag in days (negative=GNSS-IR leads, positive=USGS leads)
            - max_correlation: Maximum correlation coefficient achieved
            - lag_confidence: Confidence in the lag (high, medium, low)
            - lag_significant: Whether lag is statistically significant
            - overlapping_points: Number of overlapping points used in analysis
            - lag_adjusted_gnssir: Time-shifted GNSS-IR data (if lag is significant)
            - all_correlations: Dictionary of all tested lags and their correlations
    """
    result = {
        "optimal_lag_days": 0,
        "max_correlation": 0.0,
        "lag_confidence": "low",
        "lag_significant": False,
        "overlapping_points": 0,
        "lag_adjusted_gnssir": None,
        "all_correlations": {},
    }

    try:
        # Ensure both DataFrames have datetime or date index
        gnss_dates = None
        usgs_dates = None
        gnss_values = None
        usgs_values = None

        # Extract dates and values
        if "date" in gnssir_df.columns:
            gnss_dates = pd.to_datetime(gnssir_df["date"])
            gnss_values = gnssir_df[gnssir_col].values
        elif "datetime" in gnssir_df.columns:
            gnss_dates = pd.to_datetime(gnssir_df["datetime"])
            gnss_values = gnssir_df[gnssir_col].values
        else:
            logging.error(
                "Cannot calculate time lag: date/datetime column not found in GNSS-IR DataFrame"
            )
            return result

        if "date" in usgs_df.columns:
            usgs_dates = pd.to_datetime(usgs_df["date"])
            usgs_values = usgs_df[usgs_col].values
        elif "datetime" in usgs_df.columns:
            usgs_dates = pd.to_datetime(usgs_df["datetime"])
            usgs_values = usgs_df[usgs_col].values
        else:
            logging.error(
                "Cannot calculate time lag: date/datetime column not found in USGS DataFrame"
            )
            return result

        # Create pandas Series with dates as index and values as values
        gnss_series = pd.Series(gnss_values, index=gnss_dates)
        usgs_series = pd.Series(usgs_values, index=usgs_dates)

        # Resample to daily frequency if not already
        gnss_series = gnss_series.resample("D").mean()
        usgs_series = usgs_series.resample("D").mean()

        # Create a common date range
        all_dates = gnss_series.index.union(usgs_series.index)
        all_dates = all_dates.sort_values()

        # Reindex both series to the common date range
        gnss_reindexed = gnss_series.reindex(all_dates)
        usgs_reindexed = usgs_series.reindex(all_dates)

        # Calculate correlations at different lags
        correlations = {}
        best_lag = 0
        best_corr = -np.inf

        # Try different lags
        for lag in range(-max_lag_days, max_lag_days + 1):
            # Positive lag means GNSS-IR is BEHIND USGS (happens later)
            # Negative lag means GNSS-IR is AHEAD OF USGS (happens earlier)

            # Shift the GNSS data
            gnss_shifted = gnss_reindexed.shift(lag)

            # Find valid data (not NaN) in both series after shifting
            valid_mask = (~gnss_shifted.isna()) & (~usgs_reindexed.isna())

            # Skip if we don't have enough overlapping points
            if valid_mask.sum() <= 1:
                continue

            # Calculate correlation for this lag
            gnss_valid = gnss_shifted[valid_mask].values
            usgs_valid = usgs_reindexed[valid_mask].values

            if len(gnss_valid) > 1:
                corr = np.corrcoef(gnss_valid, usgs_valid)[0, 1]
                if not np.isnan(corr):
                    correlations[lag] = corr

                    # Update best correlation
                    if corr > best_corr:
                        best_corr = corr
                        best_lag = lag

        # If no valid correlations found, return default result
        if not correlations:
            logging.warning("No valid correlations found in lag analysis")
            return result

        # Determine confidence level based on number of overlapping points
        max_overlap = max(
            [
                sum((~gnss_reindexed.shift(lag).isna()) & (~usgs_reindexed.isna()))
                for lag in range(-max_lag_days, max_lag_days + 1)
            ]
        )
        result["overlapping_points"] = max_overlap

        if max_overlap >= 2 * min_points_for_confidence:
            confidence = "high"
        elif max_overlap >= min_points_for_confidence:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine if lag is significant
        # We consider a lag significant if it improves correlation by at least 0.05
        # compared to zero lag, and if the best correlation is greater than 0.5
        zero_lag_corr = correlations.get(0, 0)
        lag_is_significant = (
            best_corr > 0.5 and (best_lag != 0) and (best_corr - zero_lag_corr > 0.05)
        )

        # Create lag-adjusted GNSS-IR data if lag is significant
        lag_adjusted_gnssir = None
        if lag_is_significant or best_lag != 0:
            # Create a proper DataFrame with date column
            lag_adjusted_dates = gnss_series.shift(best_lag).index
            lag_adjusted_values = gnss_series.shift(best_lag).values

            lag_adjusted_gnssir = pd.DataFrame(
                {"date": lag_adjusted_dates, gnssir_col: lag_adjusted_values}
            )

        # Populate the result dictionary
        result["optimal_lag_days"] = best_lag
        result["max_correlation"] = best_corr
        result["lag_confidence"] = confidence
        result["lag_significant"] = lag_is_significant
        result["lag_adjusted_gnssir"] = lag_adjusted_gnssir
        result["all_correlations"] = correlations

        return result

    except Exception as e:
        logging.error(f"Error calculating optimal time lag: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return result


def plot_lag_correlation(
    lag_results: Dict[str, Any],
    output_plot_path: Union[str, Path],
    station_name: str,
    usgs_site_code: str,
) -> Optional[Path]:
    """
    Plot the correlation vs. lag analysis results.

    Args:
        lag_results: Dictionary from calculate_optimal_time_lag
        output_plot_path: Path to save the plot
        station_name: GNSS-IR station name
        usgs_site_code: USGS gauge site code

    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)

    try:
        # Extract data from lag_results
        correlations = lag_results.get("all_correlations", {})
        if not correlations:
            logging.error("No correlation data available to plot")
            return None

        best_lag = lag_results.get("optimal_lag_days", 0)
        best_corr = lag_results.get("max_correlation", 0)
        lag_confidence = lag_results.get("lag_confidence", "low")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Sort lag values and create x and y arrays
        lags = sorted(correlations.keys())
        corrs = [correlations[lag] for lag in lags]

        # Create bar plot
        plt.bar(lags, corrs, color=PLOT_COLORS["gnssir"], alpha=0.7)

        # Highlight optimal lag
        plt.bar([best_lag], [best_corr], color=PLOT_COLORS["highlight"], alpha=0.9)

        # Add horizontal line at zero correlation
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Add vertical line at zero lag
        plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Add title and labels
        plt.title(
            f"Time Lag Analysis: {station_name} GNSS-IR vs. USGS {usgs_site_code}", fontsize=16
        )
        plt.xlabel("Lag (days) [negative = GNSS-IR leads, positive = USGS leads]", fontsize=14)
        plt.ylabel("Correlation Coefficient", fontsize=14)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add text annotation for best lag
        plt.annotate(
            f"Optimal Lag: {best_lag} days\nCorr: {best_corr:.4f}\nConf: {lag_confidence}",
            xy=(best_lag, best_corr),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

        # Set y-axis limits with padding
        plt.ylim(min(min(corrs) - 0.1, -0.2), max(max(corrs) + 0.1, 1.0))

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()

        logging.info(f"Lag correlation plot saved to {output_plot_path}")
        return output_plot_path

    except Exception as e:
        logging.error(f"Error creating lag correlation plot: {e}")
        return None


def plot_lag_adjusted_comparison(
    daily_gnssir_rh_df: pd.DataFrame,
    daily_usgs_gauge_df: pd.DataFrame,
    lag_results: Dict[str, Any],
    station_name: str,
    usgs_gauge_info: Dict[str, Any],
    output_plot_path: Union[str, Path],
    gnssir_rh_col: str = "rh_median_m",
    usgs_wl_col: str = "usgs_value_m_median",
    compare_demeaned: bool = True,
    highlight_dates: Optional[List[str]] = None,
    style: str = "default",
) -> Optional[Path]:
    """
    Plot a comparison of GNSS-IR and USGS data with optimal time lag adjustment.

    Args:
        daily_gnssir_rh_df: DataFrame with daily GNSS-IR data
        daily_usgs_gauge_df: DataFrame with daily USGS data
        lag_results: Dictionary from calculate_optimal_time_lag
        station_name: GNSS-IR station name
        usgs_gauge_info: Dictionary with USGS gauge information
        output_plot_path: Path to save the plot
        gnssir_rh_col: Column name for GNSS-IR data
        usgs_wl_col: Column name for USGS data
        compare_demeaned: Whether to demean data for comparison
        highlight_dates: List of dates to highlight
        style: Plot style to use

    Returns:
        Path to the generated plot file on success, None on failure
    """
    output_plot_path = ensure_output_dir(output_plot_path)

    try:
        # Check if lag is significant
        if (
            not lag_results.get("lag_significant", False)
            and lag_results.get("optimal_lag_days", 0) == 0
        ):
            logging.warning("Lag is not significant and zero, no lag-adjusted plot will be created")
            return None

        # Get lag-adjusted GNSS-IR data
        lag_adjusted_gnssir = lag_results.get("lag_adjusted_gnssir")
        if lag_adjusted_gnssir is None:
            logging.error("Lag-adjusted GNSS-IR data not available")
            return None

        # Create a copy of the GNSS-IR data with the adjusted values
        adjusted_gnssir_df = daily_gnssir_rh_df.copy()

        # Convert date columns to datetime for merging
        if "date" in adjusted_gnssir_df.columns:
            adjusted_gnssir_df["date"] = pd.to_datetime(adjusted_gnssir_df["date"])
        if "date" in lag_adjusted_gnssir.columns:
            lag_adjusted_gnssir["date"] = pd.to_datetime(lag_adjusted_gnssir["date"])

        # Use merge to update the values while keeping other columns
        adjusted_gnssir_df = adjusted_gnssir_df.drop(columns=[gnssir_rh_col], errors="ignore")
        adjusted_gnssir_df = pd.merge(
            adjusted_gnssir_df, lag_adjusted_gnssir, on="date", how="right"
        )

        # If datetime column exists in original, add it to adjusted DataFrame
        if (
            "datetime" in daily_gnssir_rh_df.columns
            and "datetime" not in adjusted_gnssir_df.columns
        ):
            adjusted_gnssir_df["datetime"] = adjusted_gnssir_df["date"]

        # Get the optimal lag
        optimal_lag_days = lag_results.get("optimal_lag_days", 0)

        # Update the station name to indicate lag adjustment
        adjusted_station_name = f"{station_name} (lag-adjusted by {optimal_lag_days} days)"

        # Create the comparison plot
        from .comparison import plot_comparison_timeseries

        return plot_comparison_timeseries(
            adjusted_gnssir_df,
            daily_usgs_gauge_df,
            adjusted_station_name,
            usgs_gauge_info,
            output_plot_path,
            gnssir_rh_col,
            usgs_wl_col,
            compare_demeaned,
            highlight_dates,
            style,
        )

    except Exception as e:
        logging.error(f"Error creating lag-adjusted comparison plot: {e}")
        return None
