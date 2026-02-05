# ABOUTME: GNSS-IR data loading utilities for comparison scripts
# ABOUTME: Loads combined RH CSV files and prepares data for analysis

import logging
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_gnssir_data(station_name, year, doy_range=None):
    """
    Load GNSS-IR reflector height data for a given station, year, and DOY range.

    Args:
        station_name (str): Station name in uppercase (e.g., "FORA")
        year (int or str): Year to load data for
        doy_range (tuple, optional): Range of DOYs to include (start, end).
                                   Defaults to None (all available).

    Returns:
        pd.DataFrame: DataFrame containing the reflector height data with datetime column
    """
    year_str = str(year)

    rh_csv_path = PROJECT_ROOT / "results_annual" / station_name / f"{station_name}_{year_str}_combined_rh.csv"

    if not rh_csv_path.exists():
        logging.error(f"Combined RH CSV file not found at {rh_csv_path}")
        return None

    try:
        df = pd.read_csv(rh_csv_path)
        logging.info(f"Loaded GNSS-IR data from {rh_csv_path}: {len(df)} rows")

        # Filter by DOY range if specified
        if doy_range is not None:
            doy_min, doy_max = doy_range
            doy_col = 'doy' if 'doy' in df.columns else 'DOY' if 'DOY' in df.columns else None

            if doy_col is None:
                logging.error("Could not find 'doy' or 'DOY' column in the data")
                return None

            df = df[(df[doy_col] >= doy_min) & (df[doy_col] <= doy_max)]
            logging.info(f"Filtered to DOY range {doy_min}-{doy_max}: {len(df)} rows")

        # Generate datetime column
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            year_col = 'year' if 'year' in df.columns else 'Year'
            doy_col = 'doy' if 'doy' in df.columns else 'DOY'

            if year_col in df.columns and doy_col in df.columns:
                df['datetime'] = pd.to_datetime(
                    df[year_col].astype(str) + df[doy_col].astype(str).str.zfill(3),
                    format='%Y%j'
                )
            else:
                logging.error("Could not find year or DOY columns for datetime generation")

        return df

    except Exception as e:
        logging.error(f"Error loading GNSS-IR data: {e}")
        return None
