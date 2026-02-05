# ABOUTME: Dashboard tabs package with individual tab implementations
# ABOUTME: Contains overview, monthly, subdaily, residual, and diagnostic tabs

from .overview_tab import render_overview_tab
from .monthly_data_tab import render_monthly_data_tab
from .yearly_residual_tab import render_yearly_residual_tab
from .diagnostics_tab import render_diagnostics_tab
from .subdaily_tab import render_subdaily_tab

__all__ = [
    "render_overview_tab",
    "render_monthly_data_tab",
    "render_subdaily_tab",
    "render_yearly_residual_tab",
    "render_diagnostics_tab",
]
