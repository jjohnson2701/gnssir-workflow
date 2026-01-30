"""
Analysis Runner for Enhanced GNSS-IR Dashboard

This module contains functions for running various analyses including
multi-source comparison and environmental analysis.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import analysis modules
try:
    from scripts.multi_source_comparison import MultiSourceComparison
    from scripts.environmental_analysis import EnvironmentalAnalyzer
    MULTI_SOURCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multi-source analysis not available: {e}")
    MULTI_SOURCE_AVAILABLE = False


@st.cache_data
def run_multi_source_analysis(station_id, year, doy_range=None):
    """Run multi-source analysis and cache results."""
    if not MULTI_SOURCE_AVAILABLE:
        return None
    
    try:
        comparison = MultiSourceComparison()
        results = comparison.run_comprehensive_analysis(
            station_name=station_id,
            year=year,
            doy_range=doy_range,
            include_external_sources=True
        )
        return results
    except Exception as e:
        st.error(f"Multi-source analysis failed: {e}")
        return None


def run_environmental_analysis(gnssir_data, environmental_data, station_id):
    """Run environmental impact analysis on GNSS-IR data."""
    if not MULTI_SOURCE_AVAILABLE:
        return None
    
    try:
        analyzer = EnvironmentalAnalyzer()
        results = analyzer.analyze_environmental_effects(
            gnssir_data=gnssir_data,
            environmental_data=environmental_data,
            station_name=station_id
        )
        return results
    except Exception as e:
        st.error(f"Environmental analysis failed: {e}")
        return None


def calculate_performance_metrics(gnssir_data, reference_data, metric_type='correlation', antenna_height=None):
    """Calculate performance metrics between GNSS-IR and reference data.

    Uses WSE (Water Surface Elevation) for correlation, not raw RH.
    RH decreases as water rises, so correlating RH directly gives negative values.
    WSE = antenna_height - RH gives the correct positive correlation.
    """
    import numpy as np

    try:
        # Merge data on date
        merged = gnssir_data.merge(reference_data, on='date', how='inner')

        if len(merged) < 2:
            return None

        # Determine which GNSS-IR column to use for comparison
        # Prefer wse_ellips_m if available, otherwise compute from RH
        if 'wse_ellips_m' in merged.columns:
            gnss_values = merged['wse_ellips_m']
        elif antenna_height is not None and 'rh_median_m' in merged.columns:
            gnss_values = antenna_height - merged['rh_median_m']
        else:
            # Fallback: use RH directly (will give negative correlation)
            gnss_values = merged['rh_median_m']

        ref_values = merged['water_level_m']

        if metric_type == 'correlation':
            return gnss_values.corr(ref_values)
        elif metric_type == 'rmse':
            # For RMSE, use demeaned values to focus on relative variations
            gnss_dm = gnss_values - gnss_values.mean()
            ref_dm = ref_values - ref_values.mean()
            return np.sqrt(np.mean((gnss_dm - ref_dm)**2))
        elif metric_type == 'bias':
            # Bias between demeaned signals
            gnss_dm = gnss_values - gnss_values.mean()
            ref_dm = ref_values - ref_values.mean()
            return np.mean(gnss_dm - ref_dm)
        else:
            return None

    except Exception as e:
        print(f"Error calculating {metric_type}: {e}")
        return None


# Export functions
__all__ = [
    'run_multi_source_analysis',
    'run_environmental_analysis',
    'calculate_performance_metrics',
    'MULTI_SOURCE_AVAILABLE'
]