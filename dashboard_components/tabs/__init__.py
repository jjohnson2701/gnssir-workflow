"""
ABOUTME: Dashboard Tabs Package
ABOUTME: Individual tab implementations for the GNSS-IR dashboard.
"""

from .overview_tab import render_overview_tab
from .monthly_data_tab import render_monthly_data_tab
from .yearly_residual_tab import render_yearly_residual_tab
from .diagnostics_tab import render_diagnostics_tab

# Import other tabs only if their dependencies are available
try:
    from .multi_source_tab import render_multi_source_tab
    from .environmental_tab import render_environmental_tab
    ADVANCED_TABS_AVAILABLE = True
except ImportError:
    ADVANCED_TABS_AVAILABLE = False
    render_multi_source_tab = None
    render_environmental_tab = None

try:
    from .traditional_tab import render_traditional_tab
except ImportError:
    render_traditional_tab = None

# Subdaily tab is now available in v4
from .subdaily_tab import render_subdaily_tab

try:
    from .seasonal_tab import render_seasonal_tab
except ImportError:
    render_seasonal_tab = None

__all__ = [
    'render_overview_tab',
    'render_monthly_data_tab',
    'render_subdaily_tab',
    'render_yearly_residual_tab',
    'render_diagnostics_tab',
    'ADVANCED_TABS_AVAILABLE'
]