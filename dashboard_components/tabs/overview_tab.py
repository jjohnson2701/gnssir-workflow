# ABOUTME: Overview tab showing data availability and summary statistics
# ABOUTME: Displays station info, data counts, and reference source details

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import constants
from dashboard_components.constants import ENHANCED_COLORS, DATA_SOURCE_EMOJI

# Import station metadata helper
from dashboard_components.station_metadata import get_reference_source_info


def render_overview_tab(rh_data, usgs_data, coops_data, erddap_data,
                       selected_station, selected_year,
                       coops_station_id=None):
    """
    Render the redesigned overview tab with data availability indicator and yearly summary.

    Parameters:
    -----------
    rh_data : pd.DataFrame
        GNSS-IR reflector height data
    usgs_data : pd.DataFrame
        USGS water level data
    coops_data : pd.DataFrame
        NOAA CO-OPS tide data
    erddap_data : pd.DataFrame
        ERDDAP water level data (co-located sensors)
    selected_station : str
        Station ID
    selected_year : int
        Year for analysis
    coops_station_id : str
        CO-OPS station ID used
    """
    st.header("📊 Station Overview")

    # Get reference source info for this station
    ref_info = get_reference_source_info(selected_station)
    primary_source = ref_info['primary_source']  # 'ERDDAP', 'USGS', or 'CO-OPS'

    # Data availability status bar (horizontal red/green indicator)
    st.markdown("### 🚦 Data Source Availability")

    # Determine primary reference data based on station config
    if primary_source == 'ERDDAP':
        primary_ref_data = erddap_data
        primary_ref_name = 'ERDDAP'
        primary_ref_emoji = '🌐'
        primary_ref_details = ref_info['station_name'] if erddap_data is not None and not erddap_data.empty else "No data"
        if erddap_data is not None and not erddap_data.empty:
            primary_ref_details = f"{ref_info['station_name']} ({len(erddap_data)} records)"
    elif primary_source == 'CO-OPS':
        primary_ref_data = coops_data
        primary_ref_name = 'CO-OPS'
        primary_ref_emoji = '🌀'
        primary_ref_details = f"Station {coops_station_id}" if coops_station_id and coops_data is not None and not coops_data.empty else ref_info['station_name']
    else:
        primary_ref_data = usgs_data
        primary_ref_name = 'USGS'
        primary_ref_emoji = '🌊'
        primary_ref_details = f"{len(usgs_data)} records" if usgs_data is not None and not usgs_data.empty else "No data"

    # Create the horizontal status bar - show GNSS-IR and primary reference first
    data_sources = [
        {
            'name': 'GNSS-IR',
            'emoji': '🛰️',
            'available': rh_data is not None and not rh_data.empty,
            'count': len(rh_data) if rh_data is not None and not rh_data.empty else 0,
            'details': f"{len(rh_data)} days" if rh_data is not None and not rh_data.empty else "No data",
            'is_primary': False
        },
        {
            'name': f'{primary_ref_name} (Primary)',
            'emoji': primary_ref_emoji,
            'available': primary_ref_data is not None and not primary_ref_data.empty,
            'count': len(primary_ref_data) if primary_ref_data is not None and not primary_ref_data.empty else 0,
            'details': primary_ref_details if primary_ref_data is not None and not primary_ref_data.empty else "No data",
            'is_primary': True
        }
    ]

    # Add secondary reference sources if data is available
    if primary_source == 'ERDDAP':
        # For ERDDAP primary, show USGS and CO-OPS as secondary if available
        if usgs_data is not None and not usgs_data.empty:
            data_sources.append({
                'name': 'USGS',
                'emoji': '🌊',
                'available': True,
                'count': len(usgs_data),
                'details': f"{len(usgs_data)} records",
                'is_primary': False
            })
        if coops_data is not None and not coops_data.empty:
            data_sources.append({
                'name': 'CO-OPS',
                'emoji': '🌀',
                'available': True,
                'count': len(coops_data),
                'details': f"Station {coops_station_id}" if coops_station_id else "Available",
                'is_primary': False
            })
    elif primary_source == 'CO-OPS' and usgs_data is not None and not usgs_data.empty:
        data_sources.insert(2, {
            'name': 'USGS',
            'emoji': '🌊',
            'available': True,
            'count': len(usgs_data),
            'details': f"{len(usgs_data)} records",
            'is_primary': False
        })
    elif primary_source == 'USGS' and coops_data is not None and not coops_data.empty:
        data_sources.insert(2, {
            'name': 'CO-OPS',
            'emoji': '🌀',
            'available': True,
            'count': len(coops_data),
            'details': f"Station {coops_station_id}" if coops_station_id else "Available",
            'is_primary': False
        })
    
    # Create horizontal status bar using Streamlit columns (more reliable than HTML)
    cols = st.columns(len(data_sources))
    
    for i, source in enumerate(data_sources):
        with cols[i]:
            # Determine styling based on availability
            if source['available']:
                st.success(f"{source['emoji']} **{source['name']}**\n\n{source['details']}")
            else:
                st.error(f"{source['emoji']} **{source['name']}**\n\n{source['details']}")
    
    # Data source details in expandable sections
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🛰️ GNSS-IR Details", expanded=True):
            if rh_data is not None and not rh_data.empty:
                gnss_start = rh_data['date'].min().strftime('%Y-%m-%d')
                gnss_end = rh_data['date'].max().strftime('%Y-%m-%d')
                avg_retrievals = rh_data['rh_count'].mean()
                
                st.success(f"✅ **Active** ({len(rh_data)} processing days)")
                st.markdown(f"""
                - **Date Range:** {gnss_start} to {gnss_end}
                - **Daily Retrievals:** {avg_retrievals:.1f} average
                - **Data Quality:** {'Excellent' if avg_retrievals >= 50 else 'Good' if avg_retrievals >= 20 else 'Fair'}
                """)
            else:
                st.error("❌ **No GNSS-IR data available**")

    with col2:
        # Show primary reference source expander
        if primary_source == 'ERDDAP':
            with st.expander(f"🌐 {ref_info['station_name']} (Primary Reference)", expanded=True):
                if erddap_data is not None and not erddap_data.empty:
                    st.success(f"✅ **Active** ({len(erddap_data)} records)")

                    # Find water level column (auto-detect from various naming conventions)
                    # ERDDAP column naming varies by station (e.g., bartlett_cove_wl for GLBX)
                    water_col = None
                    if 'water_level_m' in erddap_data.columns:
                        water_col = 'water_level_m'
                    else:
                        # Look for any column ending in _wl that isn't gnss
                        wl_cols = [col for col in erddap_data.columns
                                   if col.endswith('_wl') and not col.startswith('gnss')]
                        if wl_cols:
                            water_col = wl_cols[0]

                    if water_col:
                        mean_level = erddap_data[water_col].mean()
                        std_level = erddap_data[water_col].std()
                        st.markdown(f"""
                        - **Water Level:** {mean_level:.2f} ± {std_level:.2f} m
                        - **Range:** {erddap_data[water_col].min():.2f} to {erddap_data[water_col].max():.2f} m
                        - **Dataset:** {ref_info['station_id']}
                        """)
                        if ref_info['distance_km']:
                            st.markdown(f"- **Distance:** {ref_info['distance_km']*1000:.0f} m (co-located)")
                else:
                    st.warning(f"⚠️ ERDDAP data not loaded. Station: {ref_info['station_name']}")

            # Show USGS as secondary if available
            if usgs_data is not None and not usgs_data.empty:
                with st.expander("🌊 USGS Water Levels (Secondary)"):
                    st.info(f"✅ **Available** ({len(usgs_data)} records)")

            # Show CO-OPS as secondary if available
            if coops_data is not None and not coops_data.empty:
                with st.expander("🌀 NOAA CO-OPS Tides (Secondary)"):
                    st.info(f"✅ **Available** (Station {coops_station_id})")

        elif primary_source == 'CO-OPS':
            with st.expander(f"🌀 {ref_info['station_name']} (Primary Reference)", expanded=True):
                if coops_data is not None and not coops_data.empty:
                    st.success(f"✅ **Active** ({len(coops_data)} records)")

                    # Find water level column
                    water_col = None
                    for col in ['water_level_m', 'water_level', 'v']:
                        if col in coops_data.columns:
                            water_col = col
                            break

                    if water_col:
                        mean_level = coops_data[water_col].mean()
                        std_level = coops_data[water_col].std()
                        st.markdown(f"""
                        - **Water Level:** {mean_level:.2f} ± {std_level:.2f} m
                        - **Range:** {coops_data[water_col].min():.2f} to {coops_data[water_col].max():.2f} m
                        - **Station ID:** {ref_info['station_id']}
                        """)
                        if ref_info['distance_km']:
                            st.markdown(f"- **Distance:** {ref_info['distance_km']:.1f} km")
                else:
                    st.warning(f"⚠️ CO-OPS data not loaded. Station: {ref_info['station_name']}")

            # Show USGS as secondary if available
            if usgs_data is not None and not usgs_data.empty:
                with st.expander("🌊 USGS Water Levels (Secondary)"):
                    st.info(f"✅ **Available** ({len(usgs_data)} records)")

        else:
            # USGS is primary
            with st.expander(f"🌊 USGS Water Levels (Primary Reference)", expanded=True):
                if usgs_data is not None and not usgs_data.empty:
                    st.success(f"✅ **Active** ({len(usgs_data)} records)")

                    # Find water level column
                    water_col = None
                    for col in ['water_level_m', 'usgs_value', 'usgs_value_m_median', 'value']:
                        if col in usgs_data.columns:
                            water_col = col
                            break

                    if water_col:
                        mean_level = usgs_data[water_col].mean()
                        std_level = usgs_data[water_col].std()
                        st.markdown(f"""
                        - **Water Level:** {mean_level:.2f} ± {std_level:.2f} m
                        - **Range:** {usgs_data[water_col].min():.2f} to {usgs_data[water_col].max():.2f} m
                        """)

                    if 'site_name' in usgs_data.columns:
                        st.markdown(f"- **Site:** {usgs_data['site_name'].iloc[0]}")
                    if ref_info['distance_km']:
                        st.markdown(f"- **Distance:** {ref_info['distance_km']:.1f} km")
                else:
                    st.error("❌ **No USGS data available**")

            # Show CO-OPS as secondary if available
            if coops_data is not None and not coops_data.empty:
                with st.expander("🌀 NOAA CO-OPS Tides (Secondary)"):
                    st.info(f"✅ **Available** (Station {coops_station_id})")
                    st.markdown(f"- **Records:** {len(coops_data)}")
                    if 'water_level_m' in coops_data.columns:
                        st.markdown(f"- **Tide Range:** {coops_data['water_level_m'].min():.2f} to {coops_data['water_level_m'].max():.2f} m")
    
    # Yearly Summary Section
    st.markdown("---")
    st.markdown("### 📊 Yearly Summary")
    
    if rh_data is not None and not rh_data.empty:
        # Calculate key yearly statistics
        total_days = len(rh_data)
        total_retrievals = rh_data['rh_count'].sum()
        avg_daily_retrievals = rh_data['rh_count'].mean()
        coverage_percent = (total_days / 365) * 100
        
        # RH statistics
        mean_rh = rh_data['rh_median_m'].mean()
        std_rh = rh_data['rh_median_m'].std()
        range_rh = rh_data['rh_median_m'].max() - rh_data['rh_median_m'].min()
        
        # Create summary metrics (using help text instead of delta to avoid misleading arrows)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📅 Data Coverage",
                value=f"{coverage_percent:.1f}%",
                help=f"{total_days} days with data out of 365"
            )
            st.caption(f"{total_days} days")

        with col2:
            st.metric(
                label="📡 Total Retrievals",
                value=f"{total_retrievals:,}",
                help="Total number of individual GNSS-IR water level retrievals"
            )
            st.caption(f"{avg_daily_retrievals:.1f}/day avg")

        with col3:
            st.metric(
                label="📏 Mean RH",
                value=f"{mean_rh:.3f} m",
                help="Average reflector height (antenna to water surface)"
            )
            st.caption(f"±{std_rh:.3f} m std")

        with col4:
            st.metric(
                label="📈 RH Range",
                value=f"{range_rh:.3f} m",
                help="Total variation in reflector height over the year"
            )
            st.caption(f"{rh_data['rh_median_m'].min():.3f} to {rh_data['rh_median_m'].max():.3f} m")
        
        # Performance indicators
        st.markdown("#### Performance Indicators")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            # Data quality based on retrievals
            if avg_daily_retrievals >= 50:
                quality_color = "🟢"
                quality_text = "Excellent"
            elif avg_daily_retrievals >= 20:
                quality_color = "🟡" 
                quality_text = "Good"
            else:
                quality_color = "🟠"
                quality_text = "Fair"
            
            st.markdown(f"**Data Quality:** {quality_color} {quality_text}")
            
        with perf_col2:
            # Temporal coverage
            if coverage_percent >= 90:
                coverage_color = "🟢"
                coverage_text = "Excellent"
            elif coverage_percent >= 70:
                coverage_color = "🟡"
                coverage_text = "Good" 
            else:
                coverage_color = "🟠"
                coverage_text = "Partial"
            
            st.markdown(f"**Temporal Coverage:** {coverage_color} {coverage_text}")
            
        with perf_col3:
            # Precision indicator
            if std_rh <= 0.1:
                precision_color = "🟢"
                precision_text = "High"
            elif std_rh <= 0.2:
                precision_color = "🟡"
                precision_text = "Medium"
            else:
                precision_color = "🟠"
                precision_text = "Variable"
            
            st.markdown(f"**Precision:** {precision_color} {precision_text}")
    
    else:
        st.warning("⚠️ No GNSS-IR data available for yearly summary")
    
    # Diagnostic Visualizations Section
    st.markdown("---")
    st.markdown("### 🔬 Diagnostic Visualizations")

    # Look for pre-generated visualizations
    results_dir = project_root / "results_annual" / selected_station

    # Find resolution comparison plot
    resolution_plot = results_dir / f"{selected_station}_{selected_year}_resolution_comparison.png"

    # Find polar animation GIF (look for any DOY range)
    gif_files = list(results_dir.glob(f"{selected_station}_{selected_year}_polar_animation_DOY*.gif"))

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("#### 📊 Correlation vs Temporal Resolution")
        if resolution_plot.exists():
            st.image(str(resolution_plot), width='stretch')
            st.caption(f"Full-year correlation analysis showing how aggregation affects RMSE and correlation")
        else:
            st.info(f"Resolution comparison plot not found. Generate with:\n```bash\npython scripts/plot_resolution_comparison.py --station {selected_station} --year {selected_year}\n```")

    with viz_col2:
        st.markdown("#### 🌊 Weekly Polar Animation")
        if gif_files:
            # Use the first GIF found (sorted to be consistent)
            gif_path = sorted(gif_files)[0]
            st.image(str(gif_path), width='stretch')
            # Extract DOY range from filename
            import re
            doy_match = re.search(r'DOY(\d+)-(\d+)', gif_path.name)
            if doy_match:
                doy_start, doy_end = doy_match.groups()
                st.caption(f"Water level visualization with Fresnel zone reflections (DOY {doy_start}-{doy_end})")
            else:
                st.caption("Water level visualization with Fresnel zone reflections")
        else:
            st.info(f"Polar animation not found. Generate with:\n```bash\npython scripts/create_polar_animation.py --station {selected_station} --year {selected_year} --doy_start 260 --doy_end 266\n```")

    # Enhanced Configuration information with station metadata
    with st.expander("📋 Station Configuration & GNSS-IR Parameters", expanded=False):
        try:
            import json

            # Load station configuration directly from stations_config.json
            stations_config_path = project_root / "config" / "stations_config.json"
            station_config = None

            if stations_config_path.exists():
                with open(stations_config_path, 'r') as f:
                    all_stations = json.load(f)
                    station_config = all_stations.get(selected_station)
            
            # Load GNSS-IR parameters if available
            gnssir_params = None
            if station_config and 'gnssir_json_params_path' in station_config:
                params_path = project_root / station_config['gnssir_json_params_path']
                if params_path.exists():
                    with open(params_path, 'r') as f:
                        gnssir_params = json.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🛰️ Station Metadata")
                if station_config:
                    st.markdown(f"""
                    - **Station ID:** {selected_station}
                    - **Latitude:** {station_config.get('latitude_deg', 'N/A'):.6f}°
                    - **Longitude:** {station_config.get('longitude_deg', 'N/A'):.6f}°  
                    - **Ellipsoidal Height:** {station_config.get('ellipsoidal_height_m', 'N/A')} m
                    - **Analysis Year:** {selected_year}
                    """)
                    
                    if 'usgs_comparison' in station_config:
                        usgs_config = station_config['usgs_comparison']
                        target_site = usgs_config.get('target_usgs_site', 'Auto-detected')
                        st.markdown(f"""
                        - **USGS Target:** {target_site}
                        - **Search Radius:** {usgs_config.get('search_radius_km', 'N/A')} km
                        """)
                else:
                    st.error("Station configuration not found")
            
            with col2:
                st.markdown("#### 📏 GNSS-IR Processing Parameters")
                if gnssir_params:
                    st.markdown(f"""
                    - **Reflector Height Range:** {gnssir_params.get('minH', 'N/A')} - {gnssir_params.get('maxH', 'N/A')} m
                    - **Elevation Angles:** {gnssir_params.get('e1', 'N/A')}° - {gnssir_params.get('e2', 'N/A')}°
                    - **Peak Noise Threshold:** {gnssir_params.get('PkNoise', 'N/A')}
                    - **Polynomial Order:** {gnssir_params.get('polyV', 'N/A')}
                    - **Azimuth Constraint:** {gnssir_params.get('azval2', 'All azimuths')}°
                    """)
                    
                    # Show frequency bands if available
                    if 'freqs' in gnssir_params:
                        freq_count = len(gnssir_params['freqs'])
                        st.markdown(f"- **GNSS Frequencies:** {freq_count} bands")
                        
                    # Show processing options
                    processing_opts = []
                    if gnssir_params.get('refraction', False):
                        processing_opts.append("Atmospheric Refraction")
                    if gnssir_params.get('plt_screen', False):
                        processing_opts.append("Screen Plotting")
                    if processing_opts:
                        st.markdown(f"- **Processing Options:** {', '.join(processing_opts)}")
                else:
                    st.warning("GNSS-IR parameters file not found")
            
            # Processing pipeline info
            st.markdown("#### 🔄 Processing Pipeline")
            st.markdown("""
            **Data Flow:** RINEX 3 → RINEX 2.11 → SNR Extraction → GNSS-IR Analysis → Water Level Estimation
            **External APIs:** USGS Water Services, NOAA CO-OPS, ERDDAP
            **Analysis Tools:** Time series comparison, correlation analysis, subdaily validation
            """)
            
        except ImportError as e:
            st.warning(f"Could not load configuration details: {e}")
            # Fallback to basic info
            st.markdown(f"""
            **Station:** {selected_station}
            **Year:** {selected_year}
            **Processing Pipeline:** RINEX 3 → RINEX 2.11 → SNR → GNSS-IR
            **External APIs:** USGS, NOAA CO-OPS, ERDDAP
            **Analysis Tools:** Time series comparison, correlation analysis
            """)
        except Exception as e:
            st.error(f"Error loading station configuration: {e}")
            st.markdown(f"""
            **Station:** {selected_station}  
            **Year:** {selected_year}  
            **Status:** Configuration loading error
            """)


# Export the render function
__all__ = ['render_overview_tab']