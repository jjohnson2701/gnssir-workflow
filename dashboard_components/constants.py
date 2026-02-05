# ABOUTME: Constants and configuration for GNSS-IR dashboard styling
# ABOUTME: Defines color schemes, plot settings, and theme configurations

# Try to import publication theme colors
try:
    from scripts.visualizer.publication_theme import PUBLICATION_COLORS

    PUBLICATION_THEME_AVAILABLE = True
except ImportError:
    PUBLICATION_THEME_AVAILABLE = False
    PUBLICATION_COLORS = None

# Define color scheme
if PUBLICATION_THEME_AVAILABLE and PUBLICATION_COLORS:
    ENHANCED_COLORS = {
        "gnss": PUBLICATION_COLORS["gnss_smooth"],
        "usgs": PUBLICATION_COLORS["usgs"],
        "coops": PUBLICATION_COLORS["coops"],
        "erddap": PUBLICATION_COLORS.get("erddap", "#48A9A6"),
        "correlation": PUBLICATION_COLORS["correlation"],
        "grid": PUBLICATION_COLORS["grid"],
        "text": PUBLICATION_COLORS["text"],
        "background": PUBLICATION_COLORS["background"],
        "highlight": PUBLICATION_COLORS["highlight"],
        "accent": PUBLICATION_COLORS["accent"],
        "quality_good": PUBLICATION_COLORS["quality_excellent"],
        "quality_poor": PUBLICATION_COLORS["quality_poor"],
        "quality_medium": PUBLICATION_COLORS["quality_good"],
    }
else:
    # Fallback color scheme
    ENHANCED_COLORS = {
        "gnss": "#2E86AB",
        "usgs": "#A23B72",
        "coops": "#1B998B",
        "erddap": "#48A9A6",
        "correlation": "#F18F01",
        "grid": "#E5E5E5",
        "text": "#333333",
        "background": "#FFFFFF",
        "highlight": "#E63946",
        "accent": "#FFC107",
        "quality_good": "#2D9A6B",
        "quality_poor": "#E63946",
        "quality_medium": "#F77F00",
    }

# Page configuration
PAGE_CONFIG = {
    "page_title": "GNSS-IR Dashboard",
    "page_icon": "🛰️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Tab configuration with subdaily comparison
TABS = [
    "🏠 Overview",
    "📊 Monthly Data",
    "🌊 Subdaily Comparison",
    "📈 Yearly Analysis",
    "🔍 Daily Diagnostics",
]

# Default values
DEFAULT_STATION = "GLBX"
DEFAULT_YEAR = 2024
DEFAULT_DOY_RANGE = (1, 365)

# Plotting parameters
PLOT_PARAMS = {
    "figure_width": 16,
    "figure_height": 10,
    "dpi": 300,
    "line_width": 2,
    "marker_size": 8,
    "font_size": 12,
    "title_font_size": 16,
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "min_daily_retrievals": 10,
    "good_correlation": 0.8,
    "acceptable_correlation": 0.6,
    "max_rmse_good": 0.1,
    "max_rmse_acceptable": 0.2,
}

# Time lag analysis parameters
TIME_LAG_PARAMS = {
    "max_lag_days": 10,
    "default_lag_days": 5,
    "min_data_points": 30,
    "correlation_window": 30,
}

# Data source indicators
DATA_SOURCE_EMOJI = {"gnss": "🛰️", "usgs": "📊", "coops": "🌊", "erddap": "🌐"}

# Export all constants
__all__ = [
    "ENHANCED_COLORS",
    "PUBLICATION_THEME_AVAILABLE",
    "PAGE_CONFIG",
    "TABS",
    "DEFAULT_STATION",
    "DEFAULT_YEAR",
    "DEFAULT_DOY_RANGE",
    "PLOT_PARAMS",
    "QUALITY_THRESHOLDS",
    "TIME_LAG_PARAMS",
    "DATA_SOURCE_EMOJI",
]
