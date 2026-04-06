# ABOUTME: Dashboard tabs package with individual tab implementations
# ABOUTME: Overview, Data Quality, Validation, Per-Arc, Ice, plus legacy tabs pending removal

from .overview_tab import render_overview_tab
from .monthly_data_tab import render_monthly_data_tab
from .yearly_residual_tab import render_yearly_residual_tab
from .diagnostics_tab import render_diagnostics_tab
from .subdaily_tab import render_subdaily_tab
from .per_arc_tab import render_per_arc_tab
from .ice_comparison_tab import render_ice_comparison_tab
from .feature_explorer_tab import render_feature_explorer_tab, has_feature_data
from .validation_tab import render_validation_tab, has_validation_data
from .data_quality_tab import render_data_quality_tab, has_quality_data

__all__ = [
    "render_overview_tab",
    "render_monthly_data_tab",
    "render_subdaily_tab",
    "render_yearly_residual_tab",
    "render_diagnostics_tab",
    "render_per_arc_tab",
    "render_ice_comparison_tab",
    "render_feature_explorer_tab",
    "has_feature_data",
    "render_validation_tab",
    "has_validation_data",
    "render_data_quality_tab",
    "has_quality_data",
]
