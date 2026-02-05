# ABOUTME: Publication-quality dark theme styling for GNSS-IR plots
# ABOUTME: Defines consistent color palette and matplotlib/plotly configurations

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Publication-quality dark theme color palette
PUBLICATION_COLORS = {
    # Data source colors
    "gnss_points": "#64B5F6",  # Light blue for individual points
    "gnss_smooth": "#1976D2",  # Blue for smoothed line
    "gnss_uncertainty": "#1976D2",  # Blue for uncertainty band
    "usgs": "#66BB6A",  # Green for USGS
    "coops": "#4FC3F7",  # Light cyan for NOAA CO-OPS
    "ndbc": "#FFB74D",  # Amber for NDBC buoy data
    # UI and layout colors
    "text": "#E8EAF6",  # Light blue-gray text
    "background": "#263238",  # Dark blue-gray background
    "grid": "#37474F",  # Dark blue-gray grid
    "accent": "#FFC107",  # Amber accent
    # Quality indicators
    "quality_excellent": "#4CAF50",  # Green
    "quality_good": "#FF9800",  # Orange
    "quality_poor": "#F44336",  # Red
    # Correlation and analysis
    "correlation": "#E91E63",  # Pink for correlation lines
    "highlight": "#FFEB3B",  # Yellow for highlighting
}

# Professional matplotlib style configuration
MATPLOTLIB_STYLE = {
    "figure.facecolor": PUBLICATION_COLORS["background"],
    "axes.facecolor": PUBLICATION_COLORS["background"],
    "axes.edgecolor": PUBLICATION_COLORS["text"],
    "axes.labelcolor": PUBLICATION_COLORS["text"],
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": PUBLICATION_COLORS["grid"],
    "grid.alpha": 0.3,
    "xtick.color": PUBLICATION_COLORS["text"],
    "ytick.color": PUBLICATION_COLORS["text"],
    "text.color": PUBLICATION_COLORS["text"],
    "font.size": 12,
    "font.weight": "normal",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.facecolor": PUBLICATION_COLORS["background"],
    "legend.edgecolor": PUBLICATION_COLORS["text"],
    "legend.framealpha": 0.9,
    "savefig.facecolor": PUBLICATION_COLORS["background"],
    "savefig.edgecolor": "none",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Plotly theme configuration
PLOTLY_THEME = {
    "layout": {
        "plot_bgcolor": PUBLICATION_COLORS["background"],
        "paper_bgcolor": PUBLICATION_COLORS["background"],
        "font": {"family": "Arial, sans-serif", "size": 12, "color": PUBLICATION_COLORS["text"]},
        "colorway": [
            PUBLICATION_COLORS["gnss_smooth"],
            PUBLICATION_COLORS["usgs"],
            PUBLICATION_COLORS["coops"],
            PUBLICATION_COLORS["ndbc"],
            PUBLICATION_COLORS["correlation"],
            PUBLICATION_COLORS["accent"],
        ],
        "title": {
            "font": {"size": 18, "color": PUBLICATION_COLORS["text"]},
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "showgrid": True,
            "gridcolor": PUBLICATION_COLORS["grid"],
            "gridwidth": 1,
            "showline": True,
            "linecolor": PUBLICATION_COLORS["text"],
            "linewidth": 1,
            "tickfont": {"color": PUBLICATION_COLORS["text"], "size": 11},
            "titlefont": {"color": PUBLICATION_COLORS["text"], "size": 14},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": PUBLICATION_COLORS["grid"],
            "gridwidth": 1,
            "showline": True,
            "linecolor": PUBLICATION_COLORS["text"],
            "linewidth": 1,
            "tickfont": {"color": PUBLICATION_COLORS["text"], "size": 11},
            "titlefont": {"color": PUBLICATION_COLORS["text"], "size": 14},
        },
        "legend": {
            "font": {"color": PUBLICATION_COLORS["text"], "size": 11},
            "bgcolor": f"rgba(38, 50, 56, 0.9)",  # Semi-transparent background
            "bordercolor": PUBLICATION_COLORS["grid"],
            "borderwidth": 1,
        },
    }
}


def apply_matplotlib_theme():
    """Apply publication-quality dark theme to matplotlib."""
    plt.style.use("dark_background")
    for param, value in MATPLOTLIB_STYLE.items():
        plt.rcParams[param] = value


def create_matplotlib_figure(figsize=(16, 10), title="", inverted_rh_axis=False):
    """
    Create a matplotlib figure with publication theme applied.

    Parameters:
    -----------
    figsize : tuple
        Figure size (width, height) in inches
    title : str
        Figure title
    inverted_rh_axis : bool
        Whether to invert the y-axis for GNSS-IR RH data

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    apply_matplotlib_theme()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(PUBLICATION_COLORS["background"])
    ax.set_facecolor(PUBLICATION_COLORS["background"])

    if title:
        ax.set_title(
            title, fontsize=18, fontweight="bold", color=PUBLICATION_COLORS["text"], pad=20
        )

    if inverted_rh_axis:
        ax.invert_yaxis()

    # Enhanced grid styling
    ax.grid(True, alpha=0.3, color=PUBLICATION_COLORS["grid"])

    return fig, ax


def create_plotly_figure(title="", inverted_rh_axis=False, secondary_y=False, height=600):
    """
    Create a Plotly figure with publication theme applied.

    Parameters:
    -----------
    title : str
        Figure title
    inverted_rh_axis : bool
        Whether to invert the y-axis for GNSS-IR RH data
    secondary_y : bool
        Whether to create a secondary y-axis
    height : int
        Figure height in pixels

    Returns:
    --------
    fig : plotly figure object
    """
    if secondary_y:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Apply publication theme elements individually to avoid issues with subplots
    theme_layout = PLOTLY_THEME["layout"].copy()

    # Apply basic layout properties
    fig.update_layout(
        plot_bgcolor=theme_layout["plot_bgcolor"],
        paper_bgcolor=theme_layout["paper_bgcolor"],
        font=theme_layout["font"],
        colorway=theme_layout["colorway"],
        height=height,
    )

    # Update axes with theme settings
    fig.update_xaxes(
        showgrid=theme_layout["xaxis"]["showgrid"],
        gridcolor=theme_layout["xaxis"]["gridcolor"],
        gridwidth=theme_layout["xaxis"]["gridwidth"],
        showline=theme_layout["xaxis"]["showline"],
        linecolor=theme_layout["xaxis"]["linecolor"],
        linewidth=theme_layout["xaxis"]["linewidth"],
        tickfont=theme_layout["xaxis"]["tickfont"],
        title_font=theme_layout["xaxis"]["titlefont"],  # Changed from titlefont to title_font
    )

    # Update primary y-axis - only specify secondary_y if we have subplots
    yaxis_update_params = {
        "showgrid": theme_layout["yaxis"]["showgrid"],
        "gridcolor": theme_layout["yaxis"]["gridcolor"],
        "gridwidth": theme_layout["yaxis"]["gridwidth"],
        "showline": theme_layout["yaxis"]["showline"],
        "linecolor": theme_layout["yaxis"]["linecolor"],
        "linewidth": theme_layout["yaxis"]["linewidth"],
        "tickfont": theme_layout["yaxis"]["tickfont"],
        "title_font": theme_layout["yaxis"]["titlefont"],  # Changed from titlefont to title_font
    }

    if secondary_y:
        # For subplots, we need to specify which y-axis
        fig.update_yaxes(**yaxis_update_params, secondary_y=False)
    else:
        # For regular figures, don't specify secondary_y
        fig.update_yaxes(**yaxis_update_params)

    # Update legend
    fig.update_layout(
        legend=dict(
            font=theme_layout["legend"]["font"],
            bgcolor=theme_layout["legend"]["bgcolor"],
            bordercolor=theme_layout["legend"]["bordercolor"],
            borderwidth=theme_layout["legend"]["borderwidth"],
        )
    )

    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color=PUBLICATION_COLORS["text"]),
            )
        )

    # Apply inverted axis if requested
    if inverted_rh_axis:
        if secondary_y:
            fig.update_yaxes(autorange="reversed", secondary_y=False)
        else:
            fig.update_yaxes(autorange="reversed")

    return fig


def style_gnss_scatter(scatter_params=None):
    """
    Get styling parameters for GNSS-IR scatter plots.

    Parameters:
    -----------
    scatter_params : dict, optional
        Override default parameters

    Returns:
    --------
    dict : Styling parameters for GNSS-IR scatter plots
    """
    default_params = {
        "color": PUBLICATION_COLORS["gnss_points"],
        "alpha": 0.6,
        "s": 8,  # matplotlib size
        "size": 8,  # plotly size
        "marker": dict(
            color=PUBLICATION_COLORS["gnss_points"],
            size=8,
            opacity=0.6,
            line=dict(width=1, color="white"),
        ),
    }

    if scatter_params:
        default_params.update(scatter_params)

    return default_params


def style_gnss_smooth_line(line_params=None):
    """
    Get styling parameters for GNSS-IR smoothed line plots.

    Parameters:
    -----------
    line_params : dict, optional
        Override default parameters

    Returns:
    --------
    dict : Styling parameters for GNSS-IR smoothed lines
    """
    default_params = {
        "color": PUBLICATION_COLORS["gnss_smooth"],
        "linewidth": 3,
        "alpha": 0.9,
        "line": dict(color=PUBLICATION_COLORS["gnss_smooth"], width=3),
    }

    if line_params:
        default_params.update(line_params)

    return default_params


def style_usgs_line(line_params=None):
    """
    Get styling parameters for USGS data line plots.

    Parameters:
    -----------
    line_params : dict, optional
        Override default parameters

    Returns:
    --------
    dict : Styling parameters for USGS lines
    """
    default_params = {
        "color": PUBLICATION_COLORS["usgs"],
        "linewidth": 2.5,
        "alpha": 0.9,
        "line": dict(color=PUBLICATION_COLORS["usgs"], width=2.5, dash="dash"),
    }

    if line_params:
        default_params.update(line_params)

    return default_params


def create_info_box_style():
    """
    Get styling parameters for information boxes.

    Returns:
    --------
    dict : Matplotlib bbox styling parameters
    """
    return dict(
        boxstyle="round,pad=0.6",
        facecolor="#37474F",
        alpha=0.9,
        edgecolor=PUBLICATION_COLORS["gnss_smooth"],
        linewidth=1.5,
    )


def create_directional_arrow_style():
    """
    Get styling parameters for directional arrows.

    Returns:
    --------
    dict : Matplotlib bbox styling parameters for directional info
    """
    return dict(
        boxstyle="round,pad=0.6",
        facecolor="#37474F",
        alpha=0.9,
        edgecolor=PUBLICATION_COLORS["accent"],
        linewidth=1.5,
    )


def get_quality_color_scale():
    """
    Get color scale for data quality indicators.

    Returns:
    --------
    dict : Quality-based color mapping
    """
    return {
        "excellent": PUBLICATION_COLORS["quality_excellent"],  # >= 40 retrievals
        "good": PUBLICATION_COLORS["quality_good"],  # 20-39 retrievals
        "poor": PUBLICATION_COLORS["quality_poor"],  # < 20 retrievals
    }


def apply_publication_styling_to_existing_plot(fig, title="", inverted_rh_axis=False):
    """
    Apply publication styling to an existing Plotly figure.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Existing Plotly figure
    title : str
        Figure title to set
    inverted_rh_axis : bool
        Whether to invert the y-axis

    Returns:
    --------
    fig : Modified Plotly figure
    """
    # Apply theme
    fig.update_layout(PLOTLY_THEME["layout"])

    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>", x=0.5, font=dict(size=18, color=PUBLICATION_COLORS["text"])
            )
        )

    if inverted_rh_axis:
        fig.update_yaxes(autorange="reversed")

    return fig


# Direction arrows for station metadata
DIRECTION_ARROWS = {
    "N": "↑",
    "NNE": "↗",
    "NE": "↗",
    "ENE": "↗",
    "E": "→",
    "ESE": "↘",
    "SE": "↘",
    "SSE": "↘",
    "S": "↓",
    "SSW": "↙",
    "SW": "↙",
    "WSW": "↙",
    "W": "←",
    "WNW": "↖",
    "NW": "↖",
    "NNW": "↖",
}


def get_direction_arrow(direction):
    """Get directional arrow symbol for compass direction."""
    return DIRECTION_ARROWS.get(direction, "•")


# Export all theme components
__all__ = [
    "PUBLICATION_COLORS",
    "MATPLOTLIB_STYLE",
    "PLOTLY_THEME",
    "apply_matplotlib_theme",
    "create_matplotlib_figure",
    "create_plotly_figure",
    "style_gnss_scatter",
    "style_gnss_smooth_line",
    "style_usgs_line",
    "create_info_box_style",
    "create_directional_arrow_style",
    "get_quality_color_scale",
    "apply_publication_styling_to_existing_plot",
    "get_direction_arrow",
    "DIRECTION_ARROWS",
]
