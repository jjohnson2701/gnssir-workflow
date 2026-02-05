# ABOUTME: Time-segmented correlation analysis for GNSS-IR vs reference data
# ABOUTME: Computes monthly and seasonal statistics with filtering utilities

import pandas as pd
import logging
import calendar
from typing import Dict, Tuple, Optional, Any


def filter_by_segment(df: pd.DataFrame, date_criteria: Any) -> pd.DataFrame:
    """
    Filter DataFrame by different types of date criteria.

    Parameters:
    -----------
    df : DataFrame with datetime index
    date_criteria : Date criteria in one of these formats:
        - Tuple of (start_date, end_date) as strings or datetime objects
        - List of month numbers [1, 2, 3] (Jan, Feb, Mar)
        - Tuple of (start_doy, end_doy) as integers

    Returns: Filtered DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            df.index = pd.to_datetime(df["datetime"])
        else:
            # Try to convert the existing index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except (ValueError, TypeError):
                logging.error("Could not convert DataFrame index to datetime")
                return df.iloc[0:0]  # Return empty DataFrame

    if isinstance(date_criteria, tuple) and len(date_criteria) == 2:
        if isinstance(date_criteria[0], (str, pd.Timestamp)) and isinstance(
            date_criteria[1], (str, pd.Timestamp)
        ):
            # Date range: ("2024-01-01", "2024-03-31")
            start_date, end_date = pd.to_datetime(date_criteria[0]), pd.to_datetime(
                date_criteria[1]
            )
            return df.loc[start_date:end_date]
        elif isinstance(date_criteria[0], int) and isinstance(date_criteria[1], int):
            # DOY range: (1, 90)
            start_doy, end_doy = date_criteria
            return df[df.index.dayofyear.between(start_doy, end_doy)]

    elif isinstance(date_criteria, list) and all(isinstance(m, int) for m in date_criteria):
        # Month numbers: [12, 1, 2]
        return df[df.index.month.isin(date_criteria)]

    # Default: return empty DataFrame if criteria format is not recognized
    return df.iloc[0:0]


def generate_monthly_segments(year: Optional[int] = None) -> Dict:
    """
    Generate dictionary of monthly segments.

    Parameters:
    -----------
    year : Year for date ranges. If None, only month numbers are used.

    Returns: Dictionary with month names as keys and month numbers or date ranges as values
    """
    months = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    if year is None:
        return {month: [num] for month, num in months.items()}

    # Create date ranges for each month
    segments = {}
    for month, num in months.items():
        _, last_day = calendar.monthrange(year, num)
        start_date = f"{year}-{num:02d}-01"
        end_date = f"{year}-{num:02d}-{last_day}"
        segments[month] = (start_date, end_date)

    return segments


def generate_seasonal_segments(year: Optional[int] = None) -> Dict:
    """
    Generate dictionary of seasonal segments.

    Parameters:
    -----------
    year : Year for date ranges. If None, only month numbers are used.

    Returns: Dictionary with season names as keys and month lists or date ranges as values
    """
    seasons = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}

    if year is None:
        return seasons

    # Create date ranges for each season
    segments = {}
    for season, months in seasons.items():
        if season == "Winter":
            # Winter spans across years
            prev_year = year - 1 if year > 1 else year
            start_date = f"{prev_year}-12-01"
            end_date = f"{year}-02-{29 if calendar.isleap(year) else 28}"
        elif season == "Spring":
            start_date = f"{year}-03-01"
            end_date = f"{year}-05-31"
        elif season == "Summer":
            start_date = f"{year}-06-01"
            end_date = f"{year}-08-31"
        elif season == "Fall":
            start_date = f"{year}-09-01"
            end_date = f"{year}-11-30"

        segments[season] = (start_date, end_date)

    return segments


def generate_custom_segments(year: int, segment_definitions: Dict[str, Tuple[int, int]]) -> Dict:
    """
    Generate dictionary of custom segments based on month ranges.

    Parameters:
    -----------
    year : Year for date ranges
    segment_definitions : Dict with segment names and (start_month, end_month) tuples

    Returns: Dictionary with segment names as keys and date ranges as values
    """
    segments = {}
    for segment_name, (start_month, end_month) in segment_definitions.items():
        start_date = f"{year}-{start_month:02d}-01"

        # Get the last day of the end month
        _, last_day = calendar.monthrange(year, end_month)
        end_date = f"{year}-{end_month:02d}-{last_day}"

        segments[segment_name] = (start_date, end_date)

    return segments


def perform_segmented_correlation(
    df: pd.DataFrame,
    segments_dict: Dict,
    gnss_col: str = "wse_ellips_m_demeaned",
    usgs_col: str = "usgs_value_m_median_demeaned",
    min_points: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict, Dict]:
    """
    Calculate correlation between GNSS-IR WSE and USGS water level for different time segments.

    Parameters:
    -----------
    df : DataFrame containing both GNSS-IR and USGS data with datetime index or date column
    segments_dict : Dictionary with segment names as keys and date criteria as values
    gnss_col : Column name for GNSS-IR data
    usgs_col : Column name for USGS data
    min_points : Minimum number of data points required for correlation
    logger : Logger for output messages

    Returns:
    --------
    Tuple containing:
    - Dictionary with segment names as keys and correlation values as values
    - Dictionary with segment names as keys and segment DataFrames as values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Ensure df has a datetime index for filtering
    df_with_dt_index = df.copy()
    if not isinstance(df_with_dt_index.index, pd.DatetimeIndex):
        if "date" in df_with_dt_index.columns:
            df_with_dt_index.index = pd.to_datetime(df_with_dt_index["date"])
        elif "datetime" in df_with_dt_index.columns:
            df_with_dt_index.index = pd.to_datetime(df_with_dt_index["datetime"])
        else:
            logger.error("DataFrame has no date or datetime column for filtering")
            return {}, {}

    segment_correlations = {}
    segment_data = {}

    for segment_name, date_criteria in segments_dict.items():
        # Filter df for dates in this segment
        segment_df = filter_by_segment(df_with_dt_index, date_criteria)

        # Check if both columns exist and have sufficient non-NaN values
        if gnss_col not in segment_df.columns or usgs_col not in segment_df.columns:
            logger.warning(f"Segment {segment_name}: One or both columns not found in dataframe")
            segment_correlations[segment_name] = None
            continue

        valid_gnss = segment_df[gnss_col].notna()
        valid_usgs = segment_df[usgs_col].notna()
        valid_both = valid_gnss & valid_usgs
        valid_count = valid_both.sum()

        if valid_count >= min_points:
            # Calculate correlation
            corr = segment_df[gnss_col].corr(segment_df[usgs_col])
            segment_correlations[segment_name] = corr
            segment_data[segment_name] = segment_df
            logger.info(f"Correlation for {segment_name}: {corr:.4f} ({valid_count} valid points)")
        else:
            logger.warning(
                f"Not enough valid data points for correlation in segment: "
                f"{segment_name} ({valid_count} points)"
            )
            segment_correlations[segment_name] = None
            # Still keep the segment data for reference
            segment_data[segment_name] = segment_df

    return segment_correlations, segment_data
