# ABOUTME: Visualization package for GNSS-IR analysis and validation
# ABOUTME: Provides time series, comparison, tide, and dashboard plot functions

from .timeseries import plot_annual_rh_timeseries
from .comparison import plot_comparison_timeseries, plot_ribbon_comparison, plot_subdaily_ribbon_comparison
from .comparison_plots import (
    create_comparison_plot,
    create_quality_diagnostic_plot,
    investigate_seasonal_correlation_issues,
    detect_outliers_and_anomalies,
    create_spring_investigation_plot,
    run_comprehensive_correlation_investigation
)
from .lag_analyzer import calculate_optimal_time_lag, plot_lag_correlation, plot_lag_adjusted_comparison
from .tide_integration import (
    find_nearest_tide_stations,
    get_noaa_tide_predictions,
    generate_synthetic_tide_predictions,
    get_high_low_tide_times,
    calculate_tide_residuals,
    plot_subdaily_rh_vs_tide
)
from .segmented_viz import (
    plot_segment_correlations,
    plot_segment_comparison_grid,
    plot_time_series_by_segment,
    plot_heatmap_correlation_matrix
)
from .dashboard_plots import (
    create_calendar_heatmap,
    create_monthly_box_plots,
    create_multi_parameter_timeline,
    create_tidal_stage_performance,
    create_multi_scale_performance,
    create_water_level_change_response,
    calculate_water_level_change_rate,
    classify_tidal_stage
)

__all__ = [
    # Basic visualization
    'plot_annual_rh_timeseries',
    'plot_comparison_timeseries',
    'plot_ribbon_comparison',
    'plot_subdaily_ribbon_comparison',

    # Enhanced comparison and diagnostics
    'create_comparison_plot',
    'create_quality_diagnostic_plot',
    'investigate_seasonal_correlation_issues',
    'detect_outliers_and_anomalies',
    'create_spring_investigation_plot',
    'run_comprehensive_correlation_investigation',

    # Time lag analysis
    'calculate_optimal_time_lag',
    'plot_lag_correlation',
    'plot_lag_adjusted_comparison',

    # Tide integration
    'find_nearest_tide_stations',
    'get_noaa_tide_predictions',
    'generate_synthetic_tide_predictions',
    'get_high_low_tide_times',
    'calculate_tide_residuals',
    'plot_subdaily_rh_vs_tide',

    # Segmented correlation analysis
    'plot_segment_correlations',
    'plot_segment_comparison_grid',
    'plot_time_series_by_segment',
    'plot_heatmap_correlation_matrix',

    # Dashboard plots
    'create_calendar_heatmap',
    'create_monthly_box_plots',
    'create_multi_parameter_timeline',
    'create_tidal_stage_performance',
    'create_multi_scale_performance',
    'create_water_level_change_response',
    'calculate_water_level_change_rate',
    'classify_tidal_stage'
]
