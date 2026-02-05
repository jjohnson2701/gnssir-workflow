# ABOUTME: Base visualization module with shared utilities and constants
# ABOUTME: Provides column finding, output directory management, and plot styling

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import math

# Configure matplotlib to avoid debug logging
plt.set_loglevel("WARNING")


def ensure_output_dir(output_path: Union[str, Path]) -> Path:
    """
    Ensure the output directory exists.

    Args:
        output_path: Output path

    Returns:
        Path object representing the output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def find_column_by_name(
    df: pd.DataFrame,
    possible_names: List[str],
    column_position: Optional[int] = None,
    column_type: str = "unknown",
) -> Optional[str]:
    """
    Find a column by name from a list of possible names.

    Args:
        df: DataFrame to search in
        possible_names: List of possible column names
        column_position: Position to use if no name match. Defaults to None.
        column_type: Type of column for logging. Defaults to 'unknown'.

    Returns:
        Name of the found column, or None if not found
    """
    # Try to find by name first
    for col_name in possible_names:
        if col_name in df.columns:
            logging.info(f"Found {column_type} column by name: {col_name}")
            return col_name

    # If column position is provided, try that
    if column_position is not None and len(df.columns) > column_position:
        col_name = df.columns[column_position]
        logging.info(
            f"Using column at position {column_position+1} as {column_type} column: {col_name}"
        )
        return col_name

    return None


def get_compass_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Calculate the compass direction from point 1 to point 2.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        A string with the compass direction (N, NE, E, SE, S, SW, W, NW)
    """
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the bearing
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(lon2_rad - lon1_rad)
    bearing = math.atan2(y, x)

    # Convert bearing to degrees
    bearing_deg = math.degrees(bearing)
    if bearing_deg < 0:
        bearing_deg += 360

    # Convert bearing to compass direction
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = round(bearing_deg / 45)

    return directions[index]


def add_summary_textbox(
    plt: Any,
    data: Optional[pd.DataFrame],
    stats_dict: Dict[str, Any],
    position: Tuple[float, float] = (0.02, 0.02),
    fontsize: int = 10,
    title: Optional[str] = None,
    box_style: Optional[Dict] = None,
) -> None:
    """
    Add a text box with summary statistics to a plot.

    Args:
        plt: Matplotlib pyplot object
        data: Data to summarize (can be None if stats_dict is pre-computed)
        stats_dict: Dictionary of statistic names and values
        position: Position of the text box. Defaults to (0.02, 0.02).
        fontsize: Font size. Defaults to 10.
        title: Optional title for the text box
        box_style: Optional dictionary with box style parameters
    """
    try:
        # Default box style
        default_style = dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.5", edgecolor="gray"
        )

        # Use provided style or default
        if box_style is not None:
            default_style.update(box_style)

        # Create the summary text
        if title:
            summary_text = f"{title}\n" + "\n".join(
                [f"{key}: {value}" for key, value in stats_dict.items()]
            )
        else:
            summary_text = "\n".join([f"{key}: {value}" for key, value in stats_dict.items()])

        plt.figtext(
            position[0],
            position[1],
            summary_text,
            fontsize=fontsize,
            bbox=default_style,
            horizontalalignment="left",
            verticalalignment="bottom",
        )
        logging.info("Added data summary to plot")
    except Exception as e:
        logging.warning(f"Could not add summary text: {e}")


# Color scheme definitions - professional and accessible hex colors
PLOT_COLORS = {
    "gnssir": "#2E86AB",  # Professional blue
    "gnss": "#2E86AB",  # Alias for gnssir
    "usgs": "#A23B72",  # Deep magenta (more readable than red)
    "coops": "#1B998B",  # Teal for NOAA CO-OPS
    "ndbc": "#F18F01",  # Orange for NDBC buoys
    "trend": "#A23B72",  # Same as usgs
    "reference": "#A23B72",  # Same as usgs
    "secondary": "#1B998B",  # Teal
    "highlight": "#F18F01",  # Orange for highlights
    "correlation": "#F18F01",  # Orange for correlation highlights
    "scatter": "#7B2CBF",  # Purple for scatter plots
    "grid": "#E5E5E5",  # Light gray for grid
    "text": "#333333",  # Dark gray for text
    "background": "#FFFFFF",  # White background
    "quality_good": "#2D9A6B",  # Green for good quality
    "quality_poor": "#E63946",  # Red for poor quality
    "quality_medium": "#F77F00",  # Orange for medium quality
    "tide": "#4ECDC4",  # Aqua for tide data
}

# Plot style configurations
PLOT_STYLES = {
    "default": {
        "figure.figsize": (12, 8),
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "grid.alpha": 0.3,
    }
}
