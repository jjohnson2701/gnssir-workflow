# ABOUTME: Overview tab showing data availability and summary statistics
# ABOUTME: Displays station info, data counts, and reference source details

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import station metadata helper
from dashboard_components.station_metadata import get_reference_source_info  # noqa: E402
from dashboard_components.data_loader import safe_read_parquet  # noqa: E402

AZ_BIN_LABELS = {0: "0-90 (N-E)", 1: "90-180 (E-S)", 2: "180-270 (S-W)", 3: "270-360 (W-N)"}


def _render_quality_summary(station_id, year):
    """Signal quality overview from enriched and per-arc data.

    Shows azimuth coverage, per-sector quality, and frequency band performance
    at a glance. Works for any station — not ice-specific.
    """
    enriched_path = project_root / "results_annual" / station_id / f"{station_id}_{year}_daily_enriched.parquet"
    per_arc_path = project_root / "results_annual" / station_id / f"{station_id}_{year}_per_arc.parquet"

    if not enriched_path.exists() and not per_arc_path.exists():
        st.caption("Run `python scripts/backfill_enriched.py` to enable signal quality analysis.")
        return

    enriched = safe_read_parquet(enriched_path)
    per_arc = safe_read_parquet(per_arc_path)

    # --- Per-azimuth quality summary ---
    if enriched is not None:
        pooled = enriched[
            (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
        ]

        has_per_arc = per_arc is not None and not per_arc.empty

        if has_per_arc or not pooled.empty:
            fig = plt.figure(figsize=(14, 5))
            ax_polar = fig.add_subplot(131, projection="polar")
            ax_freq = fig.add_subplot(132)
            ax_timeline = fig.add_subplot(133)

            # -- Left: Polar azimuth quality at 10-degree resolution --
            ax_polar.set_theta_zero_location("N")
            ax_polar.set_theta_direction(-1)

            if has_per_arc:
                az_step = 10
                pa = per_arc.copy()
                pa["az_10"] = (pa["Azim"] // az_step * az_step).astype(int)

                az_stats = []
                for az in sorted(pa["az_10"].unique()):
                    ad = pa[pa["az_10"] == az]
                    if len(ad) >= 20:
                        daily_rh_std = ad.groupby("date")["RH"].std().mean()
                        az_stats.append({
                            "az": az,
                            "n_arcs": len(ad),
                            "amp_mean": ad["Amp"].mean(),
                            "rh_std": daily_rh_std,
                        })

                if az_stats:
                    az_df = pd.DataFrame(az_stats)
                    az_rad = np.radians(az_df["az"] + az_step / 2)

                    # Color by daily RH std (green=precise, red=noisy)
                    rh_std_vals = az_df["rh_std"].values
                    vmin_s = np.percentile(rh_std_vals, 10)
                    vmax_s = np.percentile(rh_std_vals, 90)
                    norm = mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)

                    # Size by arc count
                    sizes = 30 + 150 * (az_df["n_arcs"] / az_df["n_arcs"].max())

                    sc = ax_polar.scatter(az_rad, az_df["amp_mean"], c=rh_std_vals,
                                          cmap="RdYlGn_r", norm=norm, s=sizes,
                                          alpha=0.8, edgecolors="white", linewidths=0.5)
                    fig.colorbar(sc, ax=ax_polar, label="Daily RH std (m)",
                                 shrink=0.6, pad=0.08)

                    # Compass labels
                    for deg, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
                        r_max = az_df["amp_mean"].max() * 1.15
                        ax_polar.text(np.radians(deg), r_max, label, ha="center",
                                      va="center", fontsize=8, fontweight="bold",
                                      color="#666666")

            ax_polar.set_rlabel_position(225)
            ax_polar.set_title("Azimuth quality\nradius=amp, color=RH scatter\nsize=arc count",
                               fontsize=8, pad=15)

            # -- Frequency bar chart --
            freq_data = enriched[
                (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] != "ALL")
            ]
            freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c",
                           "E6": "#d62728", "B3": "#9467bd"}

            if not freq_data.empty:
                freq_summary = freq_data.groupby("freq_group").agg(
                    total_arcs=("rh_count", "sum"),
                    mean_std=("rh_std", "mean"),
                    mean_amp=("amp_mean", "mean"),
                ).reset_index()
                freq_summary = freq_summary.sort_values("total_arcs", ascending=True)

                bars = ax_freq.barh(
                    freq_summary["freq_group"],
                    freq_summary["total_arcs"],
                    color=[freq_colors.get(fg, "#7f7f7f") for fg in freq_summary["freq_group"]],
                    alpha=0.7,
                )
                for bar, (_, row) in zip(bars, freq_summary.iterrows()):
                    ax_freq.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                                 f"std:{row['mean_std']:.3f}m",
                                 va="center", fontsize=7)

                ax_freq.set_xlabel("Total arcs")
                ax_freq.set_title("Frequency bands", fontsize=9)

            # -- Daily arc count timeline --
            if not pooled.empty:
                pooled_sorted = pooled.sort_values("date")
                pooled_sorted["date_dt"] = pd.to_datetime(pooled_sorted["date"])
                ax_timeline.fill_between(pooled_sorted["date_dt"], 0, pooled_sorted["rh_count"],
                                         color="#1f77b4", alpha=0.4)
                ax_timeline.plot(pooled_sorted["date_dt"], pooled_sorted["rh_count"],
                                 color="#1f77b4", linewidth=0.8)
                median_count = pooled_sorted["rh_count"].median()
                ax_timeline.axhline(median_count, color="red", linestyle="--", linewidth=0.8)
                ax_timeline.text(pooled_sorted["date_dt"].iloc[0], median_count * 1.05,
                                 f"median: {median_count:.0f}", fontsize=7, color="red")
                ax_timeline.set_ylabel("Arcs/day")
                ax_timeline.set_title("Daily coverage", fontsize=9)
                ax_timeline.tick_params(axis="x", labelsize=7, rotation=30)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Per-sector summary table (10-degree resolution from per-arc)
            if has_per_arc:
                az_step = 10
                pa_t = per_arc.copy()
                pa_t["az_10"] = (pa_t["Azim"] // az_step * az_step).astype(int)
                rows = []
                for az in sorted(pa_t["az_10"].unique()):
                    ad = pa_t[pa_t["az_10"] == az]
                    if len(ad) >= 20:
                        daily_cv = ad.groupby("date")["Amp"].agg(
                            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) >= 3 else np.nan
                        ).dropna()
                        rows.append({
                            "Azimuth": f"{az}-{az+az_step}",
                            "Arcs": len(ad),
                            "Days": ad["date"].nunique(),
                            "Amp": f"{ad['Amp'].mean():.1f}",
                            "CV": f"{daily_cv.mean():.3f}" if len(daily_cv) > 0 else "-",
                            "RH std": f"{ad.groupby('date')['RH'].std().mean():.3f}m",
                            "P2N": f"{ad['PkNoise'].mean():.1f}",
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_overview_tab(
    rh_data,
    usgs_data,
    coops_data,
    erddap_data,
    selected_station,
    selected_year,
    coops_station_id=None,
):
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
    primary_source = ref_info["primary_source"]  # 'ERDDAP', 'USGS', or 'CO-OPS'

    # Data availability status bar (horizontal red/green indicator)
    st.markdown("### 🚦 Data Source Availability")

    # Determine primary reference data based on station config
    if primary_source == "ERDDAP":
        primary_ref_data = erddap_data
        primary_ref_name = "ERDDAP"
        primary_ref_emoji = "🌐"
        primary_ref_details = (
            ref_info["station_name"]
            if erddap_data is not None and not erddap_data.empty
            else "No data"
        )
        if erddap_data is not None and not erddap_data.empty:
            primary_ref_details = f"{ref_info['station_name']} ({len(erddap_data)} records)"
    elif primary_source == "CO-OPS":
        primary_ref_data = coops_data
        primary_ref_name = "CO-OPS"
        primary_ref_emoji = "🌀"
        primary_ref_details = (
            f"Station {coops_station_id}"
            if coops_station_id and coops_data is not None and not coops_data.empty
            else ref_info["station_name"]
        )
    else:
        primary_ref_data = usgs_data
        primary_ref_name = "USGS"
        primary_ref_emoji = "🌊"
        primary_ref_details = (
            f"{len(usgs_data)} records"
            if usgs_data is not None and not usgs_data.empty
            else "No data"
        )

    # Create the horizontal status bar - show GNSS-IR and primary reference first
    data_sources = [
        {
            "name": "GNSS-IR",
            "emoji": "🛰️",
            "available": rh_data is not None and not rh_data.empty,
            "count": len(rh_data) if rh_data is not None and not rh_data.empty else 0,
            "details": (
                f"{len(rh_data)} days" if rh_data is not None and not rh_data.empty else "No data"
            ),
            "is_primary": False,
        },
        {
            "name": f"{primary_ref_name} (Primary)",
            "emoji": primary_ref_emoji,
            "available": primary_ref_data is not None and not primary_ref_data.empty,
            "count": (
                len(primary_ref_data)
                if primary_ref_data is not None and not primary_ref_data.empty
                else 0
            ),
            "details": (
                primary_ref_details
                if primary_ref_data is not None and not primary_ref_data.empty
                else "No data"
            ),
            "is_primary": True,
        },
    ]

    # Add secondary reference sources if data is available
    if primary_source == "ERDDAP":
        # For ERDDAP primary, show USGS and CO-OPS as secondary if available
        if usgs_data is not None and not usgs_data.empty:
            data_sources.append(
                {
                    "name": "USGS",
                    "emoji": "🌊",
                    "available": True,
                    "count": len(usgs_data),
                    "details": f"{len(usgs_data)} records",
                    "is_primary": False,
                }
            )
        if coops_data is not None and not coops_data.empty:
            data_sources.append(
                {
                    "name": "CO-OPS",
                    "emoji": "🌀",
                    "available": True,
                    "count": len(coops_data),
                    "details": f"Station {coops_station_id}" if coops_station_id else "Available",
                    "is_primary": False,
                }
            )
    elif primary_source == "CO-OPS" and usgs_data is not None and not usgs_data.empty:
        data_sources.insert(
            2,
            {
                "name": "USGS",
                "emoji": "🌊",
                "available": True,
                "count": len(usgs_data),
                "details": f"{len(usgs_data)} records",
                "is_primary": False,
            },
        )
    elif primary_source == "USGS" and coops_data is not None and not coops_data.empty:
        data_sources.insert(
            2,
            {
                "name": "CO-OPS",
                "emoji": "🌀",
                "available": True,
                "count": len(coops_data),
                "details": f"Station {coops_station_id}" if coops_station_id else "Available",
                "is_primary": False,
            },
        )

    # Create horizontal status bar using Streamlit columns (more reliable than HTML)
    cols = st.columns(len(data_sources))

    for i, source in enumerate(data_sources):
        with cols[i]:
            # Determine styling based on availability
            if source["available"]:
                st.success(f"{source['emoji']} **{source['name']}**\n\n{source['details']}")
            else:
                st.error(f"{source['emoji']} **{source['name']}**\n\n{source['details']}")

    # Data source details in expandable sections
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🛰️ GNSS-IR Details", expanded=True):
            if rh_data is not None and not rh_data.empty:
                gnss_start = rh_data["date"].min().strftime("%Y-%m-%d")
                gnss_end = rh_data["date"].max().strftime("%Y-%m-%d")
                avg_retrievals = rh_data["rh_count"].mean()

                st.success(f"**Active** ({len(rh_data)} processing days)")
                if avg_retrievals >= 50:
                    quality = "Excellent"
                elif avg_retrievals >= 20:
                    quality = "Good"
                else:
                    quality = "Fair"
                st.markdown(
                    f"""
                - **Date Range:** {gnss_start} to {gnss_end}
                - **Daily Retrievals:** {avg_retrievals:.1f} average
                - **Data Quality:** {quality}
                """
                )
            else:
                st.error("❌ **No GNSS-IR data available**")

    with col2:
        # Show primary reference source expander
        if primary_source == "ERDDAP":
            with st.expander(f"🌐 {ref_info['station_name']} (Primary Reference)", expanded=True):
                if erddap_data is not None and not erddap_data.empty:
                    st.success(f"✅ **Active** ({len(erddap_data)} records)")

                    # Find water level column (auto-detect from various naming conventions)
                    # ERDDAP column naming varies by station (e.g., bartlett_cove_wl for GLBX)
                    water_col = None
                    if "water_level_m" in erddap_data.columns:
                        water_col = "water_level_m"
                    else:
                        # Look for any column ending in _wl that isn't gnss
                        wl_cols = [
                            col
                            for col in erddap_data.columns
                            if col.endswith("_wl") and not col.startswith("gnss")
                        ]
                        if wl_cols:
                            water_col = wl_cols[0]

                    if water_col:
                        mean_level = erddap_data[water_col].mean()
                        std_level = erddap_data[water_col].std()
                        wl_min = erddap_data[water_col].min()
                        wl_max = erddap_data[water_col].max()
                        st.markdown(
                            f"""
                        - **Water Level:** {mean_level:.2f} +/- {std_level:.2f} m
                        - **Range:** {wl_min:.2f} - {wl_max:.2f} m
                        - **Dataset:** {ref_info['station_id']}
                        """
                        )
                        if ref_info["distance_km"]:
                            st.markdown(
                                f"- **Distance:** {ref_info['distance_km']*1000:.0f} m (co-located)"
                            )
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

        elif primary_source == "CO-OPS":
            with st.expander(f"🌀 {ref_info['station_name']} (Primary Reference)", expanded=True):
                if coops_data is not None and not coops_data.empty:
                    st.success(f"✅ **Active** ({len(coops_data)} records)")

                    # Find water level column
                    water_col = None
                    for col in ["water_level_m", "water_level", "v"]:
                        if col in coops_data.columns:
                            water_col = col
                            break

                    if water_col:
                        mean_level = coops_data[water_col].mean()
                        std_level = coops_data[water_col].std()
                        wl_min = coops_data[water_col].min()
                        wl_max = coops_data[water_col].max()
                        st.markdown(
                            f"""
                        - **Water Level:** {mean_level:.2f} +/- {std_level:.2f} m
                        - **Range:** {wl_min:.2f} - {wl_max:.2f} m
                        - **Station ID:** {ref_info['station_id']}
                        """
                        )
                        if ref_info["distance_km"]:
                            st.markdown(f"- **Distance:** {ref_info['distance_km']:.1f} km")
                else:
                    st.warning(f"⚠️ CO-OPS data not loaded. Station: {ref_info['station_name']}")

            # Show USGS as secondary if available
            if usgs_data is not None and not usgs_data.empty:
                with st.expander("🌊 USGS Water Levels (Secondary)"):
                    st.info(f"✅ **Available** ({len(usgs_data)} records)")

        else:
            # USGS is primary
            with st.expander("USGS Water Levels (Primary Reference)", expanded=True):
                if usgs_data is not None and not usgs_data.empty:
                    st.success(f"✅ **Active** ({len(usgs_data)} records)")

                    # Find water level column
                    water_col = None
                    for col in ["water_level_m", "usgs_value", "usgs_value_m_median", "value"]:
                        if col in usgs_data.columns:
                            water_col = col
                            break

                    if water_col:
                        mean_level = usgs_data[water_col].mean()
                        std_level = usgs_data[water_col].std()
                        wl_min = usgs_data[water_col].min()
                        wl_max = usgs_data[water_col].max()
                        st.markdown(
                            f"""
                        - **Water Level:** {mean_level:.2f} +/- {std_level:.2f} m
                        - **Range:** {wl_min:.2f} - {wl_max:.2f} m
                        """
                        )

                    if "site_name" in usgs_data.columns:
                        st.markdown(f"- **Site:** {usgs_data['site_name'].iloc[0]}")
                    if ref_info["distance_km"]:
                        st.markdown(f"- **Distance:** {ref_info['distance_km']:.1f} km")
                else:
                    st.error("❌ **No USGS data available**")

            # Show CO-OPS as secondary if available
            if coops_data is not None and not coops_data.empty:
                with st.expander("🌀 NOAA CO-OPS Tides (Secondary)"):
                    st.info(f"✅ **Available** (Station {coops_station_id})")
                    st.markdown(f"- **Records:** {len(coops_data)}")
                    if "water_level_m" in coops_data.columns:
                        wl_min = coops_data["water_level_m"].min()
                        wl_max = coops_data["water_level_m"].max()
                        st.markdown(f"- **Tide Range:** {wl_min:.2f} to {wl_max:.2f} m")

    # Yearly Summary Section
    st.markdown("---")
    st.markdown("### 📊 Yearly Summary")

    if rh_data is not None and not rh_data.empty:
        # Calculate key yearly statistics
        total_days = len(rh_data)
        total_retrievals = rh_data["rh_count"].sum()
        avg_daily_retrievals = rh_data["rh_count"].mean()
        coverage_percent = (total_days / 365) * 100

        # RH statistics
        mean_rh = rh_data["rh_median_m"].mean()
        std_rh = rh_data["rh_median_m"].std()
        range_rh = rh_data["rh_median_m"].max() - rh_data["rh_median_m"].min()

        # Create summary metrics (using help text instead of delta to avoid misleading arrows)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📅 Data Coverage",
                value=f"{coverage_percent:.1f}%",
                help=f"{total_days} days with data out of 365",
            )
            st.caption(f"{total_days} days")

        with col2:
            st.metric(
                label="📡 Total Retrievals",
                value=f"{total_retrievals:,}",
                help="Total number of individual GNSS-IR water level retrievals",
            )
            st.caption(f"{avg_daily_retrievals:.1f}/day avg")

        with col3:
            st.metric(
                label="📏 Mean RH",
                value=f"{mean_rh:.3f} m",
                help="Average reflector height (antenna to water surface)",
            )
            st.caption(f"±{std_rh:.3f} m std")

        with col4:
            st.metric(
                label="📈 RH Range",
                value=f"{range_rh:.3f} m",
                help="Total variation in reflector height over the year",
            )
            st.caption(
                f"{rh_data['rh_median_m'].min():.3f} to {rh_data['rh_median_m'].max():.3f} m"
            )

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

    # Signal Quality Summary (from enriched parquet)
    st.markdown("---")
    st.markdown("### Signal Quality Summary")
    _render_quality_summary(selected_station, selected_year)

    # Diagnostic Visualizations Section
    st.markdown("---")
    st.markdown("### Diagnostic Visualizations")

    # Look for pre-generated visualizations
    results_dir = project_root / "results_annual" / selected_station

    # Find resolution comparison plot
    resolution_plot = results_dir / f"{selected_station}_{selected_year}_resolution_comparison.png"

    # Find polar animation GIF (look for any DOY range)
    gif_files = list(
        results_dir.glob(f"{selected_station}_{selected_year}_polar_animation_DOY*.gif")
    )

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("#### 📊 Correlation vs Temporal Resolution")
        if resolution_plot.exists():
            st.image(str(resolution_plot), use_container_width=True)
            st.caption("Full-year correlation analysis showing how aggregation affects RMSE")
        else:
            st.info(
                f"Resolution comparison plot not found. Generate with:\n"
                f"```bash\npython scripts/plot_resolution_comparison.py "
                f"--station {selected_station} --year {selected_year}\n```"
            )

    with viz_col2:
        st.markdown("#### 🌊 Weekly Polar Animation")
        if gif_files:
            import re

            # Pick the GIF with the widest DOY span (most data coverage)
            def _doy_span(path):
                m = re.search(r"DOY(\d+)-(\d+)", path.name)
                return int(m.group(2)) - int(m.group(1)) if m else 0

            gif_path = max(gif_files, key=_doy_span)
            st.image(str(gif_path), use_container_width=True)
            # Extract DOY range from filename
            doy_match = re.search(r"DOY(\d+)-(\d+)", gif_path.name)
            if doy_match:
                doy_start, doy_end = doy_match.groups()
                st.caption(f"Water level with Fresnel zone reflections (DOY {doy_start}-{doy_end})")
            else:
                st.caption("Water level visualization with Fresnel zone reflections")
        else:
            st.info(
                f"Polar animation not found. Generate with:\n"
                f"```bash\npython scripts/create_polar_animation.py "
                f"--station {selected_station} --year {selected_year} "
                f"--doy_start 260 --doy_end 266\n```"
            )

    # Enhanced Configuration information with station metadata
    with st.expander("📋 Station Configuration & GNSS-IR Parameters", expanded=False):
        try:
            import json

            # Load station configuration directly from stations_config.json
            stations_config_path = project_root / "config" / "stations_config.json"
            station_config = None

            if stations_config_path.exists():
                with open(stations_config_path, "r") as f:
                    all_stations = json.load(f)
                    station_config = all_stations.get(selected_station)

            # Load GNSS-IR parameters if available
            gnssir_params = None
            if station_config and "gnssir_json_params_path" in station_config:
                params_path = project_root / station_config["gnssir_json_params_path"]
                if params_path.exists():
                    with open(params_path, "r") as f:
                        gnssir_params = json.load(f)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🛰️ Station Metadata")
                if station_config:
                    st.markdown(
                        f"""
                    - **Station ID:** {selected_station}
                    - **Latitude:** {station_config.get('latitude_deg', 'N/A'):.6f}°
                    - **Longitude:** {station_config.get('longitude_deg', 'N/A'):.6f}
                    - **Ellipsoidal Height:** {station_config.get('ellipsoidal_height_m', 'N/A')} m
                    - **Analysis Year:** {selected_year}
                    """
                    )

                    if "usgs_comparison" in station_config:
                        usgs_config = station_config["usgs_comparison"]
                        target_site = usgs_config.get("target_usgs_site", "Auto-detected")
                        st.markdown(
                            f"""
                        - **USGS Target:** {target_site}
                        - **Search Radius:** {usgs_config.get('search_radius_km', 'N/A')} km
                        """
                        )
                else:
                    st.error("Station configuration not found")

            with col2:
                st.markdown("#### 📏 GNSS-IR Processing Parameters")
                if gnssir_params:
                    minH = gnssir_params.get("minH", "N/A")
                    maxH = gnssir_params.get("maxH", "N/A")
                    e1 = gnssir_params.get("e1", "N/A")
                    e2 = gnssir_params.get("e2", "N/A")
                    st.markdown(
                        f"""
                    - **Reflector Height Range:** {minH} - {maxH} m
                    - **Elevation Angles:** {e1} - {e2} deg
                    - **Peak Noise Threshold:** {gnssir_params.get('PkNoise', 'N/A')}
                    - **Polynomial Order:** {gnssir_params.get('polyV', 'N/A')}
                    - **Azimuth Constraint:** {gnssir_params.get('azval2', 'All azimuths')}
                    """
                    )

                    # Show frequency bands if available
                    if "freqs" in gnssir_params:
                        freq_count = len(gnssir_params["freqs"])
                        st.markdown(f"- **GNSS Frequencies:** {freq_count} bands")

                    # Show processing options
                    processing_opts = []
                    if gnssir_params.get("refraction", False):
                        processing_opts.append("Atmospheric Refraction")
                    if gnssir_params.get("plt_screen", False):
                        processing_opts.append("Screen Plotting")
                    if processing_opts:
                        st.markdown(f"- **Processing Options:** {', '.join(processing_opts)}")
                else:
                    st.warning("GNSS-IR parameters file not found")

            # Processing pipeline info
            st.markdown("#### Processing Pipeline")
            st.markdown(
                """
            **Data Flow:** RINEX 3 -> RINEX 2.11 -> SNR Extraction -> GNSS-IR -> Water Level
            **External APIs:** USGS Water Services, NOAA CO-OPS, ERDDAP
            **Analysis Tools:** Time series comparison, correlation analysis, subdaily validation
            """
            )

        except ImportError as e:
            st.warning(f"Could not load configuration details: {e}")
            # Fallback to basic info
            st.markdown(
                f"""
            **Station:** {selected_station}
            **Year:** {selected_year}
            **Processing Pipeline:** RINEX 3 → RINEX 2.11 → SNR → GNSS-IR
            **External APIs:** USGS, NOAA CO-OPS, ERDDAP
            **Analysis Tools:** Time series comparison, correlation analysis
            """
            )
        except Exception as e:
            st.error(f"Error loading station configuration: {e}")
            st.markdown(
                f"""
            **Station:** {selected_station}
            **Year:** {selected_year}
            **Status:** Configuration loading error
            """
            )


# Export the render function
__all__ = ["render_overview_tab"]
