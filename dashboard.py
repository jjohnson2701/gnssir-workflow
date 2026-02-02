#!/usr/bin/env python3
# ABOUTME: Main entry point for GNSS-IR Dashboard with subdaily comparison.
# ABOUTME: Provides station-aware reference sources (ERDDAP, USGS, CO-OPS).

"""
GNSS-IR Dashboard
=================

Features:
- Subdaily comparison tab showing individual retrievals vs reference
- Station-aware reference source detection (ERDDAP > USGS > CO-OPS)
- Antenna heights loaded from config
- Diagnostics tab for quality analysis

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import dashboard components
from dashboard_components import (
    load_station_data,
    load_available_stations,
    fetch_coops_data,
)

from dashboard_components.constants import (
    ENHANCED_COLORS,
    PAGE_CONFIG,
    TABS,
    DEFAULT_STATION,
    DEFAULT_YEAR,
    DEFAULT_DOY_RANGE,
    PUBLICATION_THEME_AVAILABLE
)

from dashboard_components.tabs import (
    render_overview_tab,
    render_monthly_data_tab,
    render_subdaily_tab,
    render_yearly_residual_tab,
    render_diagnostics_tab,
)

# Import station metadata helper
from dashboard_components.station_metadata import (
    get_station_display_info,
    get_reference_source_info,
    get_antenna_height
)

# Configure Streamlit page
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
publication_header_color = ENHANCED_COLORS['gnss'] if PUBLICATION_THEME_AVAILABLE else '#2E86AB'
publication_bg_color = ENHANCED_COLORS['background'] if PUBLICATION_THEME_AVAILABLE else '#fafafa'

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {publication_header_color};
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {ENHANCED_COLORS['text'] if PUBLICATION_THEME_AVAILABLE else '#555'};
        margin-bottom: 1.5rem;
        font-style: italic;
    }}
    .metric-container {{
        background-color: {publication_bg_color};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {publication_header_color};
        margin: 0.5rem 0;
    }}
    .analysis-section {{
        background-color: {publication_bg_color};
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid {ENHANCED_COLORS['grid']};
        margin: 1rem 0;
    }}
    .data-source-indicator {{
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        color: white;
        font-size: 0.8rem;
        margin: 0.1rem;
    }}
    .source-gnss {{ background-color: {ENHANCED_COLORS['gnss']}; }}
    .source-usgs {{ background-color: {ENHANCED_COLORS['usgs']}; }}
    .source-coops {{ background-color: {ENHANCED_COLORS['coops']}; }}
    .source-erddap {{ background-color: {ENHANCED_COLORS.get('erddap', '#48A9A6')}; }}
    .reference-info {{
        background-color: #e8f4f8 !important;
        border-left: 4px solid {ENHANCED_COLORS['coops']};
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
        color: #333333 !important;
    }}
    .reference-info strong {{
        color: #1a1a1a !important;
    }}
</style>
""", unsafe_allow_html=True)


def main():
    """Main application logic."""
    # Header
    st.markdown('<h1 class="main-header">🛰️ GNSS-IR Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Water Level Validation with Subdaily Comparison</p>',
                unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.title("Configuration")

    # Station selection
    available_stations = load_available_stations()
    selected_station = st.sidebar.selectbox(
        "Select Station",
        available_stations,
        index=available_stations.index(DEFAULT_STATION) if DEFAULT_STATION in available_stations else 0
    )

    # Display station reference info
    ref_info = get_reference_source_info(selected_station)
    antenna_height = get_antenna_height(selected_station)

    st.sidebar.markdown("### 📍 Station Info")
    distance_text = f"<strong>Distance:</strong> {ref_info['distance_km']:.1f} km<br>" if ref_info.get('distance_km') else ""
    st.sidebar.markdown(f"""
    <div class="reference-info">
    <strong>Reference:</strong> {ref_info['primary_source']}<br>
    <strong>Station:</strong> {ref_info['station_name']}<br>
    {distance_text}<strong>Antenna Height:</strong> {antenna_height:.3f} m
    </div>
    """, unsafe_allow_html=True)

    # Year selection
    current_year = pd.Timestamp.now().year
    selected_year = st.sidebar.number_input(
        "Select Year",
        min_value=2015,
        max_value=current_year,
        value=DEFAULT_YEAR
    )

    # DOY range selection
    st.sidebar.markdown("### Date Range (Day of Year)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        doy_start = st.number_input("Start DOY", min_value=1, max_value=366, value=DEFAULT_DOY_RANGE[0])
    with col2:
        doy_end = st.number_input("End DOY", min_value=1, max_value=366, value=DEFAULT_DOY_RANGE[1])

    doy_range = (doy_start, doy_end) if doy_start <= doy_end else DEFAULT_DOY_RANGE

    # Data source display
    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.markdown("🛰️ **GNSS-IR** - Satellite reflectometry")

    # Show primary reference source based on station config
    primary_source = ref_info['primary_source']
    if primary_source == 'ERDDAP':
        st.sidebar.markdown(f"🌐 **ERDDAP** - {ref_info['station_name']} (primary)")
    elif primary_source == 'CO-OPS':
        st.sidebar.markdown(f"🌀 **CO-OPS** - {ref_info['station_name']} (primary)")
    else:
        st.sidebar.markdown(f"🌊 **USGS** - {ref_info['station_name']} (primary)")

    # Load data button
    if st.sidebar.button("Load/Refresh Data", type="primary"):
        st.session_state.data_loaded = True
        # Clear any cached station data when reloading
        st.session_state.current_station = selected_station
        st.session_state.current_year = selected_year

    # Main content area
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        with st.spinner("Loading GNSS-IR and comparison data..."):
            # Load station data
            try:
                rh_data, comparison_data, usgs_data, coops_file_data, erddap_data = load_station_data(selected_station, selected_year)
            except ValueError:
                # Fallback for older version (4 return values)
                try:
                    rh_data, comparison_data, usgs_data, coops_file_data = load_station_data(selected_station, selected_year)
                    erddap_data = None
                except ValueError:
                    # Fallback for even older version (3 return values)
                    rh_data, comparison_data, usgs_data = load_station_data(selected_station, selected_year)
                    coops_file_data = None
                    erddap_data = None

        # Check for missing data and provide helpful messages
        if rh_data is None or rh_data.empty:
            results_dir = project_root / "results_annual" / selected_station
            expected_file = results_dir / f"{selected_station}_{selected_year}_combined_rh.csv"
            st.error(f"❌ No GNSS-IR data found for {selected_station} {selected_year}")
            st.markdown(f"""
            **Expected file:** `{expected_file}`

            To generate this data, run:
            ```bash
            python scripts/process_station.py --station {selected_station} --year {selected_year}
            ```
            """)
            return

        # Filter by DOY range if data exists
        if rh_data is not None and not rh_data.empty:
            rh_data = rh_data[(rh_data['doy'] >= doy_range[0]) & (rh_data['doy'] <= doy_range[1])]

        # Also filter comparison_data by DOY range if it exists
        if comparison_data is not None and not comparison_data.empty:
            comparison_data['doy'] = comparison_data['merge_date'].dt.dayofyear
            comparison_data = comparison_data[(comparison_data['doy'] >= doy_range[0]) & (comparison_data['doy'] <= doy_range[1])]

        # Initialize external data variables
        coops_data = None
        coops_station_id = None

        # Check if ERDDAP data was loaded (highest priority for ERDDAP-enabled stations)
        if erddap_data is not None and not erddap_data.empty and ref_info['primary_source'] == 'ERDDAP':
            st.info(f"🌐 Using ERDDAP data from {ref_info['station_name']} (co-located sensor)")

        # Check if CO-OPS data was loaded from file
        if coops_file_data is not None and not coops_file_data.empty:
            coops_data = coops_file_data
            # Extract station ID from config instead of hardcoding
            station_config = get_station_display_info(selected_station)
            if station_config.get('reference_station_id') and ref_info['primary_source'] == 'CO-OPS':
                coops_station_id = station_config['reference_station_id']
            elif 'station_id' in coops_data.columns:
                coops_station_id = str(coops_data['station_id'].iloc[0])
            else:
                coops_station_id = ref_info.get('station_id', 'Unknown')
            if ref_info['primary_source'] == 'CO-OPS':
                st.info(f"🌀 Using CO-OPS data from station {coops_station_id} (primary reference)")

        # Determine if CO-OPS data is available for tabs
        include_coops = coops_data is not None and not coops_data.empty

        # Create five-tab structure
        tabs = st.tabs(TABS)

        # Tab 1: Overview
        with tabs[0]:
            render_overview_tab(
                rh_data, usgs_data, coops_data, erddap_data,
                selected_station, selected_year,
                coops_station_id
            )

        # Tab 2: Monthly Data (all visualizations)
        with tabs[1]:
            render_monthly_data_tab(
                rh_data, usgs_data, coops_data,
                selected_station, selected_year, include_coops
            )

        # Tab 3: Subdaily Comparison
        with tabs[2]:
            render_subdaily_tab(
                station_id=selected_station,
                year=selected_year,
                rh_data=rh_data,
                comparison_data=comparison_data
            )

        # Tab 4: Yearly Analysis (residual analysis)
        with tabs[3]:
            render_yearly_residual_tab(
                rh_data, usgs_data, coops_data,
                selected_station, selected_year,
                erddap_data=erddap_data
            )

        # Tab 5: Daily Diagnostics (QuickLook plots)
        with tabs[4]:
            # Create data dictionary for consistency with new API
            data_dict = {
                'rh_data': rh_data,
                'comparison_data': comparison_data,
                'usgs_data': usgs_data,
                'coops_data': coops_data
            }
            render_diagnostics_tab(
                selected_station, selected_year, data_dict
            )

    else:
        st.info("👈 Please configure settings and click 'Load/Refresh Data' to begin analysis")

        # Show quick start guide
        st.markdown("### 🚀 Quick Start")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Available Stations:**
            - **GLBX** - Alaska (ERDDAP co-located sensor)
            - **FORA** - North Carolina coast (USGS reference)
            - **MDAI** - Maryland coast (USGS reference)
            - **VALR** - Hawaii (CO-OPS tide gauge reference)
            """)

        with col2:
            st.markdown("""
            **Features:**
            - 🌊 **Subdaily Comparison** tab with full-resolution data
            - 📍 Station-aware reference source detection
            - 📊 Improved statistics and export options
            - 🔧 Better error messages for missing data
            """)

        # Show example visualizations
        st.markdown("### Example Visualizations")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📅 Calendar Heat Map")
            st.caption("Year-at-a-glance performance metrics")

        with col2:
            st.markdown("#### 🌊 Subdaily Comparison")
            st.caption("Individual retrievals vs reference gauge")

        with col3:
            st.markdown("#### 📈 Yearly Residual Analysis")
            st.caption("Comprehensive error statistics")


if __name__ == "__main__":
    main()
