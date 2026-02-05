# ABOUTME: Reflector height to water surface elevation conversion utilities
# ABOUTME: Transforms GNSS-IR RH measurements to ellipsoidal and demeaned heights

import logging
import numpy as np


def calculate_wse_from_rh(gnssir_daily_df, antenna_ellipsoidal_height):
    """
    Calculate Water Surface Ellipsoidal Height (WSE_ellips) from Reflector Height (RH).

    Args:
        gnssir_daily_df (pd.DataFrame): DataFrame with GNSS-IR daily statistics
        antenna_ellipsoidal_height (float): Antenna ellipsoidal height in meters

    Returns:
        pd.DataFrame: Updated DataFrame with WSE_ellips columns
    """
    # Check for RH column with more detailed logging
    logging.info(
        f"Checking for RH median column in dataset with columns: {gnssir_daily_df.columns.tolist()}"
    )

    # Log if rh_count exists
    if "rh_count" in gnssir_daily_df.columns:
        logging.info("rh_count found in input DataFrame before WSE calculation")
    else:
        logging.warning("rh_count NOT found in input DataFrame before WSE calculation")

    # Try to find the RH column with different possible names
    rh_col = None
    for possible_col in [
        "rh_median_m",
        "RH_median_m",
        "RH_median",
        "rh_median",
        "reflector_height_median",
        "median",
    ]:
        if possible_col in gnssir_daily_df.columns:
            rh_col = possible_col
            logging.info(f"Found RH column: {rh_col}")
            break

    if rh_col is None:
        logging.warning("Standard RH median column not found. Looking for any RH-related column...")
        # Try to find any column that might contain RH data
        rh_related_cols = [
            col for col in gnssir_daily_df.columns if "rh" in col.lower() or "median" in col.lower()
        ]

        if rh_related_cols:
            rh_col = rh_related_cols[0]
            logging.info(f"Using column '{rh_col}' as RH data source")
        elif "RH" in gnssir_daily_df.columns:
            rh_col = "RH"
            logging.info("Using 'RH' column")
        else:
            # Get the numeric columns in case RH is there but not properly named
            numeric_cols = gnssir_daily_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:  # Assuming date, doy are first two
                rh_col = numeric_cols[2]  # Try the third numeric column
                logging.warning(f"No clear RH column found. Using numeric column: {rh_col}")
            else:
                logging.error("Could not identify any suitable RH column")
                # Return the dataframe unchanged
                return gnssir_daily_df

    # Create a copy to avoid modifying the original
    result_df = gnssir_daily_df.copy()

    # Create standardized rh_median_m column if it doesn't exist
    if "rh_median_m" not in result_df.columns:
        result_df["rh_median_m"] = result_df[rh_col]
        logging.info(f"Created 'rh_median_m' from '{rh_col}'")

    # WSE_ellips = Antenna_Ellipsoidal_Height - RH_median_m
    result_df["wse_ellips_m"] = antenna_ellipsoidal_height - result_df["rh_median_m"]

    # Create demeaned versions for comparison
    result_df["rh_median_m_demeaned"] = result_df["rh_median_m"] - result_df["rh_median_m"].mean()
    result_df["wse_ellips_m_demeaned"] = (
        result_df["wse_ellips_m"] - result_df["wse_ellips_m"].mean()
    )

    # Check if rh_count is preserved
    if "rh_count" in result_df.columns:
        logging.info("rh_count column preserved after WSE calculation")
    else:
        logging.warning("rh_count column LOST after WSE calculation")

    logging.info(f"Calculated WSE_ellips using antenna height {antenna_ellipsoidal_height} m")
    return result_df
