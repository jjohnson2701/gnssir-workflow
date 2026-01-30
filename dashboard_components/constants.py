"""
Constants and Configuration for Enhanced GNSS-IR Dashboard

This module contains all constants, color schemes, and configuration
settings used throughout the dashboard.
"""

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
        'gnss': PUBLICATION_COLORS['gnss_smooth'],
        'usgs': PUBLICATION_COLORS['usgs'], 
        'coops': PUBLICATION_COLORS['coops'],
        'ndbc': PUBLICATION_COLORS['ndbc'],
        'correlation': PUBLICATION_COLORS['correlation'],
        'grid': PUBLICATION_COLORS['grid'],
        'text': PUBLICATION_COLORS['text'],
        'background': PUBLICATION_COLORS['background'],
        'highlight': PUBLICATION_COLORS['highlight'],
        'accent': PUBLICATION_COLORS['accent'],
        'quality_good': PUBLICATION_COLORS['quality_excellent'],
        'quality_poor': PUBLICATION_COLORS['quality_poor'],
        'quality_medium': PUBLICATION_COLORS['quality_good']
    }
else:
    # Fallback color scheme
    ENHANCED_COLORS = {
        'gnss': '#2E86AB',
        'usgs': '#A23B72', 
        'coops': '#1B998B',
        'ndbc': '#F18F01',
        'correlation': '#F18F01',
        'grid': '#E5E5E5',
        'text': '#333333',
        'background': '#FFFFFF',
        'highlight': '#E63946',
        'accent': '#FFC107',
        'quality_good': '#2D9A6B',
        'quality_poor': '#E63946',
        'quality_medium': '#F77F00'
    }

# Page configuration (v3 - kept for backwards compatibility)
PAGE_CONFIG = {
    "page_title": "Enhanced GNSS-IR Dashboard v3",
    "page_icon": "🛰️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Page configuration for v4
PAGE_CONFIG_V4 = {
    "page_title": "GNSS-IR Dashboard v4",
    "page_icon": "🛰️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Tab configurations for new dashboard structure
TABS_WITH_MULTI_SOURCE = [
    "🏠 Overview", 
    "📊 Monthly Data",
    "📈 Yearly Analysis",
    "🔍 Daily Diagnostics"
]

TABS_WITHOUT_MULTI_SOURCE = [
    "🏠 Overview",
    "📊 Monthly Data",
    "📈 Yearly Analysis",
    "🔍 Daily Diagnostics"
]

# v4 Tab configuration with subdaily comparison
TABS_V4 = [
    "🏠 Overview",
    "📊 Monthly Data",
    "🌊 Subdaily Comparison",
    "📈 Yearly Analysis",
    "🔍 Daily Diagnostics"
]

# Default values
DEFAULT_STATION = "FORA"
DEFAULT_YEAR = 2024
DEFAULT_DOY_RANGE = (1, 365)

# Plotting parameters
PLOT_PARAMS = {
    'figure_width': 16,
    'figure_height': 10,
    'dpi': 300,
    'line_width': 2,
    'marker_size': 8,
    'font_size': 12,
    'title_font_size': 16
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'min_daily_retrievals': 10,
    'good_correlation': 0.8,
    'acceptable_correlation': 0.6,
    'max_rmse_good': 0.1,
    'max_rmse_acceptable': 0.2
}

# Time lag analysis parameters
TIME_LAG_PARAMS = {
    'max_lag_days': 10,
    'default_lag_days': 5,
    'min_data_points': 30,
    'correlation_window': 30
}

# Environmental thresholds
ENVIRONMENTAL_THRESHOLDS = {
    'high_wind_speed': 15.0,  # m/s
    'moderate_wind_speed': 8.0,  # m/s
    'high_wave_height': 2.0,  # m
    'moderate_wave_height': 1.0,  # m
    'storm_pressure_drop': 5.0  # mb
}

# Data source indicators
DATA_SOURCE_EMOJI = {
    'gnss': '🛰️',
    'usgs': '📊',
    'coops': '🌊',
    'ndbc': '🌪️'
}

# Export all constants
__all__ = [
    'ENHANCED_COLORS',
    'PUBLICATION_THEME_AVAILABLE',
    'PAGE_CONFIG',
    'PAGE_CONFIG_V4',
    'TABS_WITH_MULTI_SOURCE',
    'TABS_WITHOUT_MULTI_SOURCE',
    'TABS_V4',
    'DEFAULT_STATION',
    'DEFAULT_YEAR',
    'DEFAULT_DOY_RANGE',
    'PLOT_PARAMS',
    'QUALITY_THRESHOLDS',
    'TIME_LAG_PARAMS',
    'ENVIRONMENTAL_THRESHOLDS',
    'DATA_SOURCE_EMOJI'
]