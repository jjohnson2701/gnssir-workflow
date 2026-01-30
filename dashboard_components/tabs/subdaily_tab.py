# ABOUTME: Implements subdaily comparison tab for GNSS-IR dashboard.
# ABOUTME: Shows individual retrievals vs reference at full temporal resolution.

"""
Subdaily Comparison Tab Implementation

Displays annual subdaily comparison between GNSS-IR retrievals and
reference gauge data (USGS or CO-OPS) at full temporal resolution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import station metadata helper
from dashboard_components.station_metadata import (
    get_antenna_height,
    get_reference_source_info,
    get_station_display_info
)

from dashboard_components.constants import ENHANCED_COLORS


def load_subdaily_matched_data(station_id: str, year: int) -> tuple:
    """
    Load subdaily matched data for a station.

    Returns:
        Tuple of (DataFrame or None, error_message or None)
    """
    results_dir = project_root / "results_annual" / station_id
    matched_file = results_dir / f"{station_id}_{year}_subdaily_matched.csv"

    if not matched_file.exists():
        return None, f"Subdaily matched file not found: `{matched_file}`"

    try:
        df = pd.read_csv(matched_file)
        df['gnss_datetime'] = pd.to_datetime(df['gnss_datetime'], format='mixed', utc=True)
        df = df.sort_values('gnss_datetime').reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, f"Error loading subdaily data: {e}"


def detect_column_names(df: pd.DataFrame) -> tuple:
    """
    Detect GNSS and reference column names from dataframe.

    Returns:
        Tuple of (gnss_col, ref_col, ref_source)
    """
    # GNSS-IR demeaned column
    if 'gnss_wse_dm' in df.columns:
        gnss_dm_col = 'gnss_wse_dm'
    elif 'gnss_dm' in df.columns:
        gnss_dm_col = 'gnss_dm'
    else:
        gnss_dm_col = None

    # Reference demeaned column (USGS or CO-OPS)
    if 'usgs_wl_dm' in df.columns:
        ref_dm_col = 'usgs_wl_dm'
        ref_source = 'USGS'
    elif 'coops_dm' in df.columns:
        ref_dm_col = 'coops_dm'
        ref_source = 'CO-OPS'
    else:
        ref_dm_col = None
        ref_source = 'Unknown'

    return gnss_dm_col, ref_dm_col, ref_source


def create_subdaily_plot(
    df: pd.DataFrame,
    station_name: str,
    year: int,
    gnss_col: str,
    ref_col: str,
    ref_source: str,
    ref_site_name: str = "Reference",
    distance_km: float = 0.0,
    show_ribbon: bool = True,
    ribbon_window: int = 50
) -> plt.Figure:
    """Create subdaily comparison plot with statistics."""

    # Calculate statistics
    correlation = df[gnss_col].corr(df[ref_col])
    rmse = np.sqrt(np.mean((df[gnss_col] - df[ref_col])**2))
    n_points = len(df)
    residuals = df[gnss_col] - df[ref_col]
    resid_mean = residuals.mean()
    resid_std = residuals.std()

    # Create figure with two panels - explicit white background
    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True,
        facecolor='white'
    )
    ax_main.set_facecolor('white')
    ax_resid.set_facecolor('white')

    # Set spine colors explicitly to black
    for ax in [ax_main, ax_resid]:
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.0)

    plt.subplots_adjust(hspace=0.05)

    # Main panel: both datasets demeaned
    # Reference as continuous line - group by hour
    df['hour'] = df['gnss_datetime'].dt.floor('h')
    hourly_ref = df.groupby('hour')[ref_col].mean()

    # Plot reference line (USGS or CO-OPS) - make it prominent
    ref_line, = ax_main.plot(hourly_ref.index, hourly_ref.values,
                              color='#C0392B', linewidth=2.5, alpha=0.9, zorder=5,
                              label=f'{ref_source}')

    # GNSS-IR ribbon (rolling ±σ) if enabled
    legend_handles = []
    legend_labels = []

    if show_ribbon:
        rolling_mean = df[gnss_col].rolling(ribbon_window, center=True, min_periods=5).mean()
        rolling_std = df[gnss_col].rolling(ribbon_window, center=True, min_periods=5).std()

        # Plot the ribbon (±σ band)
        ax_main.fill_between(df['gnss_datetime'],
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.25, color='#3498DB', zorder=1)

        # Plot the central rolling mean line
        ax_main.plot(df['gnss_datetime'], rolling_mean,
                     color='#2471A3', linewidth=2.0, alpha=0.8, zorder=3)

    # GNSS-IR as scatter points - use darker blue for contrast with ribbon
    gnss_scatter = ax_main.scatter(df['gnss_datetime'], df[gnss_col],
                                   c='#1A5276', s=8, alpha=0.6, zorder=4)

    # Build legend with proper handles
    from matplotlib.lines import Line2D

    # Reference line (USGS or CO-OPS) - thick red line
    ref_marker = Line2D([0], [0], color='#C0392B', linewidth=2.5)

    # GNSS-IR scatter points
    gnss_scatter_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A5276',
                                  markersize=6, linestyle='None')

    # Build legend - order depends on whether ribbon is shown
    if show_ribbon:
        # GNSS-IR rolling mean - blue line
        gnss_mean_marker = Line2D([0], [0], color='#2471A3', linewidth=2)
        from matplotlib.patches import Patch
        ribbon_patch = Patch(facecolor='#3498DB', alpha=0.25)
        legend_handles = [ref_marker, gnss_mean_marker, ribbon_patch, gnss_scatter_marker]
        legend_labels = [f'{ref_source} ({ref_site_name[:20]})', 'GNSS-IR mean', 'GNSS-IR ±1σ', 'GNSS-IR points']
    else:
        legend_handles = [ref_marker, gnss_scatter_marker]
        legend_labels = [f'{ref_source} ({ref_site_name[:20]})', 'GNSS-IR points']

    # Zero line
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=0)

    # Labels and formatting - explicit black text
    ax_main.set_ylabel('Demeaned Water Level (m)', fontsize=12, color='black')
    ax_main.set_title(f'{station_name} Subdaily: GNSS-IR WSE vs {ref_source} ({year})', fontsize=14, color='black')
    ax_main.legend(handles=legend_handles, labels=legend_labels, loc='upper left', fontsize=10)
    ax_main.tick_params(colors='black')
    ax_main.grid(True, alpha=0.3)

    # Set y-axis limits based on reference data range (prevents outliers from compressing ref line)
    ref_range = df[ref_col].max() - df[ref_col].min()
    y_limit = max(1.0, ref_range * 2.5)  # At least 1m, or 2.5x the reference range
    ax_main.set_ylim(-y_limit, y_limit)

    # Info box
    info_lines = [
        f"r = {correlation:.3f}",
        f"RMSE = {rmse:.3f} m",
        f"N = {n_points:,}",
    ]
    if distance_km > 0:
        info_lines.append(f"Distance: {distance_km:.2f} km")

    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    ax_main.text(0.99, 0.98, '\n'.join(info_lines), transform=ax_main.transAxes,
                 fontsize=10, va='top', ha='right', bbox=props, color='black')

    # Residual panel
    ax_resid.scatter(df['gnss_datetime'], residuals,
                     c='#8E44AD', s=2, alpha=0.4)
    ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Residual stats
    ax_resid.text(0.02, 0.95, f'μ={resid_mean:.3f}m, σ={resid_std:.3f}m',
                  transform=ax_resid.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), color='black')

    ax_resid.set_ylabel('Residual (m)', fontsize=11, color='black')
    ax_resid.set_xlabel('Date', fontsize=12, color='black')
    ax_resid.tick_params(colors='black')
    ax_resid.grid(True, alpha=0.3)
    ax_resid.set_ylim(-1.5, 1.5)

    # X-axis formatting - monthly ticks
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator())
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout()
    return fig


def render_subdaily_tab(
    station_id: str,
    year: int,
    rh_data=None,
    comparison_data=None
):
    """
    Render the subdaily comparison tab.

    Args:
        station_id: Station identifier
        year: Analysis year
        rh_data: GNSS-IR daily data (for context)
        comparison_data: Enhanced comparison data (for context)
    """
    st.header("🌊 Subdaily Comparison")

    st.markdown("""
    **Full-resolution comparison** of GNSS-IR individual retrievals versus reference gauge data.
    This view shows all subdaily measurements, revealing tidal patterns and retrieval scatter.
    """)

    # Get station metadata
    station_info = get_station_display_info(station_id)
    ref_info = get_reference_source_info(station_id)

    # Display reference source info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"📍 **Reference Source:** {ref_info['primary_source']} - {ref_info['station_name']}")
    with col2:
        if ref_info['distance_km']:
            st.metric("Distance to Reference", f"{ref_info['distance_km']:.1f} km")

    # Load subdaily data
    df, error = load_subdaily_matched_data(station_id, year)

    if error:
        st.error(f"❌ {error}")
        st.markdown("""
        **To generate subdaily matched data:**

        1. Ensure GNSS-IR processing is complete for this station/year
        2. Run the subdaily matching script:
           ```bash
           python scripts/generate_subdaily_matched.py --station {station_id} --year {year}
           ```

        This creates the `{station_id}_{year}_subdaily_matched.csv` file needed for this view.
        """.format(station_id=station_id, year=year))
        return

    # Detect column names
    gnss_col, ref_col, detected_source = detect_column_names(df)

    if gnss_col is None or ref_col is None:
        st.error(f"❌ Could not find required columns. Available: {df.columns.tolist()}")
        return

    # Display data summary
    st.success(f"✅ Loaded **{len(df):,}** matched subdaily points")

    # Plot settings
    st.markdown("### ⚙️ Plot Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_ribbon = st.checkbox("Show scatter ribbon (±σ)", value=True,
                                  help="Display rolling standard deviation band around GNSS-IR data")

    with col2:
        ribbon_window = st.slider("Ribbon window size", 20, 100, 50,
                                  help="Number of points for rolling statistics",
                                  disabled=not show_ribbon)

    with col3:
        # Date range filter
        min_date = df['gnss_datetime'].min().date()
        max_date = df['gnss_datetime'].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # Filter by date range if specified
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['gnss_datetime'].dt.date >= start_date) & (df['gnss_datetime'].dt.date <= end_date)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    if len(df_filtered) == 0:
        st.warning("No data in selected date range")
        return

    # Generate plot
    with st.spinner("Generating subdaily comparison plot..."):
        fig = create_subdaily_plot(
            df=df_filtered,
            station_name=station_id,
            year=year,
            gnss_col=gnss_col,
            ref_col=ref_col,
            ref_source=detected_source,
            ref_site_name=ref_info['station_name'],
            distance_km=ref_info['distance_km'] or 0.0,
            show_ribbon=show_ribbon,
            ribbon_window=ribbon_window
        )

    st.pyplot(fig)
    plt.close()

    # Statistics summary
    st.markdown("### 📊 Statistics Summary")

    correlation = df_filtered[gnss_col].corr(df_filtered[ref_col])
    rmse = np.sqrt(np.mean((df_filtered[gnss_col] - df_filtered[ref_col])**2))
    residuals = df_filtered[gnss_col] - df_filtered[ref_col]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Correlation (r)", f"{correlation:.3f}")

    with col2:
        st.metric("RMSE", f"{rmse:.3f} m")

    with col3:
        st.metric("Residual Std", f"{residuals.std():.3f} m")

    with col4:
        st.metric("N Points", f"{len(df_filtered):,}")

    # Export option
    st.markdown("### 📥 Export")

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"{station_id}_{year}_subdaily_filtered.csv",
            mime="text/csv"
        )

    with col2:
        # PNG export - save to buffer
        import io
        buf = io.BytesIO()
        fig_export = create_subdaily_plot(
            df=df_filtered,
            station_name=station_id,
            year=year,
            gnss_col=gnss_col,
            ref_col=ref_col,
            ref_source=detected_source,
            ref_site_name=ref_info['station_name'],
            distance_km=ref_info['distance_km'] or 0.0,
            show_ribbon=show_ribbon,
            ribbon_window=ribbon_window
        )
        fig_export.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig_export)

        st.download_button(
            label="⬇️ Download Plot (PNG)",
            data=buf.getvalue(),
            file_name=f"{station_id}_{year}_subdaily_comparison.png",
            mime="image/png"
        )

    # Interpretation guide
    with st.expander("📖 Interpretation Guide"):
        st.markdown("""
        **Understanding the Subdaily Comparison Plot:**

        **Top Panel - Time Series:**
        - **Blue scatter points**: Individual GNSS-IR water surface elevation retrievals
        - **Blue ribbon** (if enabled): Rolling ±1σ band showing GNSS-IR scatter
        - **Red line**: Reference gauge (USGS or CO-OPS) hourly averages

        **Bottom Panel - Residuals:**
        - Shows difference: GNSS-IR - Reference
        - **μ**: Mean residual (bias)
        - **σ**: Standard deviation of residuals (precision)

        **Quality Indicators:**
        - **Correlation > 0.9**: Excellent agreement
        - **RMSE < 0.10 m**: High precision
        - **RMSE 0.10-0.20 m**: Good precision
        - **RMSE > 0.20 m**: May indicate issues with processing parameters

        **Typical Patterns:**
        - Wider scatter at high/low tide extremes is common
        - Gaps indicate periods with insufficient satellite visibility
        - Systematic offsets suggest datum alignment issues
        """)


# Export function
__all__ = ['render_subdaily_tab']
