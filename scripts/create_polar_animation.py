# ABOUTME: Creates animated GIF showing water level on satellite imagery with regional context
# ABOUTME: Overlays GNSS-IR data directly on Fresnel zone satellite view

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import imageio.v2 as imageio
import tempfile
import json
import contextily as ctx
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import box

# GPS L1 wavelength for Fresnel zone calculations
GPS_L1_WAVELENGTH = 0.1903  # meters


def render_cached_basemaps(
    metadata, transformer, station_x, station_y, gauge_x, gauge_y, region_bounds, cache_dir: Path
) -> dict:
    """
    Pre-render the three map basemaps once and cache them as images.
    Returns paths to the cached images and their extents.
    """
    from math import sqrt

    cache_paths = {}

    # Calculate buffer sizes based on station-gauge distance
    gauge_distance_m = sqrt((gauge_x - station_x) ** 2 + (gauge_y - station_y) ** 2)

    if gauge_distance_m < 1000:
        buffer_wide = 2500
        zoom_level = 14
    elif gauge_distance_m < 8000:
        buffer_wide = 5000
        zoom_level = 12
    elif gauge_distance_m < 15000:
        buffer_wide = 7500
        zoom_level = 11
    else:
        buffer_wide = 30000
        zoom_level = 9

    outer_refl_dist = metadata.get("outer_reflection_dist", 230)
    outer_fresnel_r = metadata.get("outer_fresnel_radius", 5)
    buffer_close = int(outer_refl_dist + outer_fresnel_r + 20)

    # Store geometry info
    cache_paths["buffer_wide"] = buffer_wide
    cache_paths["buffer_close"] = buffer_close
    cache_paths["zoom_level"] = zoom_level

    # === Render Regional Overview (CartoDB Positron) ===
    print("  Caching regional overview basemap...")
    fig_coast, ax_coast = plt.subplots(figsize=(8, 8))

    reg_west, reg_east = region_bounds["west"], region_bounds["east"]
    reg_south, reg_north = region_bounds["south"], region_bounds["north"]
    ec_x_min, ec_y_min = transformer.transform(reg_west, reg_south)
    ec_x_max, ec_y_max = transformer.transform(reg_east, reg_north)

    ax_coast.set_xlim(ec_x_min, ec_x_max)
    ax_coast.set_ylim(ec_y_min, ec_y_max)

    try:
        ctx.add_basemap(ax_coast, source=ctx.providers.CartoDB.Positron, zoom=7)
    except Exception:
        ax_coast.set_facecolor("#c6e2ff")

    ax_coast.set_aspect("equal")
    ax_coast.axis("off")

    coast_path = cache_dir / "basemap_coast.png"
    fig_coast.savefig(coast_path, dpi=100, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig_coast)

    cache_paths["coast"] = coast_path
    cache_paths["coast_extent"] = [ec_x_min, ec_x_max, ec_y_min, ec_y_max]

    # === Render Regional Context (Esri WorldImagery) ===
    print("  Caching regional context basemap...")
    fig_map, ax_map = plt.subplots(figsize=(10, 10))

    center_x = (station_x + gauge_x) / 2
    center_y = (station_y + gauge_y) / 2

    ax_map.set_xlim(center_x - buffer_wide, center_x + buffer_wide)
    ax_map.set_ylim(center_y - buffer_wide, center_y + buffer_wide)

    try:
        ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, zoom=zoom_level)
    except Exception:
        ax_map.set_facecolor("lightblue")

    ax_map.set_aspect("equal")
    ax_map.axis("off")

    map_path = cache_dir / "basemap_regional.png"
    fig_map.savefig(map_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig_map)

    cache_paths["regional"] = map_path
    cache_paths["regional_extent"] = [
        center_x - buffer_wide,
        center_x + buffer_wide,
        center_y - buffer_wide,
        center_y + buffer_wide,
    ]
    cache_paths["center_x"] = center_x
    cache_paths["center_y"] = center_y

    # === Render Fresnel Zone Close-up (Esri WorldImagery high zoom) ===
    print("  Caching Fresnel zone basemap...")
    fig_sat, ax_sat = plt.subplots(figsize=(12, 12))

    ax_sat.set_xlim(station_x - buffer_close, station_x + buffer_close)
    ax_sat.set_ylim(station_y - buffer_close, station_y + buffer_close)

    try:
        ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom="auto")
    except Exception:
        try:
            ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom=17)
        except Exception:
            ax_sat.set_facecolor("lightblue")

    ax_sat.set_aspect("equal")
    ax_sat.axis("off")

    sat_path = cache_dir / "basemap_fresnel.png"
    fig_sat.savefig(sat_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig_sat)

    cache_paths["fresnel"] = sat_path
    cache_paths["fresnel_extent"] = [
        station_x - buffer_close,
        station_x + buffer_close,
        station_y - buffer_close,
        station_y + buffer_close,
    ]

    print("  Basemap caching complete.")
    return cache_paths


def calculate_fresnel_radius(reflector_height: float, elevation_deg: float) -> float:
    """
    Calculate first Fresnel zone radius on reflecting surface.

    Args:
        reflector_height: Distance from antenna to reflecting surface (meters)
        elevation_deg: Satellite elevation angle (degrees)

    Returns:
        First Fresnel zone radius (meters)
    """
    elev_rad = np.radians(elevation_deg)
    slant_distance = reflector_height / np.sin(elev_rad)
    fresnel_radius = np.sqrt((GPS_L1_WAVELENGTH * slant_distance) / 2)
    return fresnel_radius


def load_data(station: str, year: int, results_dir: Path):
    """Load raw GNSS-IR data, matched data with reference, and reference instantaneous values."""
    raw_file = results_dir / station / f"{station}_{year}_combined_raw.csv"
    matched_file = results_dir / station / f"{station}_{year}_subdaily_matched.csv"

    # Reference file paths (will be set based on config)
    usgs_file = results_dir / station / f"{station}_{year}_usgs_iv.csv"
    coops_file = results_dir / station / f"{station}_{year}_coops_6min.csv"

    print(f"Loading raw data from {raw_file}")
    df = pd.read_csv(raw_file)
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["UTCtime"], unit="h")

    # Get station config (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_file = project_root / "config" / "stations_config.json"
    with open(config_file) as f:
        config = json.load(f)

    station_config = config[station]
    antenna_height = station_config["ellipsoidal_height_m"]

    # Get station coordinates from config
    station_lat = station_config.get("latitude", station_config.get("latitude_deg"))
    station_lon = station_config.get("longitude", station_config.get("longitude_deg"))

    # Get reference gauge info
    usgs_info = station_config.get("usgs_comparison", {})
    coops_info = station_config.get("coops_comparison", {})

    # Also check for external_data_sources structure
    ext_sources = station_config.get("external_data_sources", {})
    noaa_coops = ext_sources.get("noaa_coops", {})
    erddap_config = ext_sources.get("erddap", {})

    # Build ERDDAP filename from config if available
    erddap_file = None
    erddap_station_name = None
    if erddap_config.get("enabled"):
        erddap_station_name = erddap_config.get("station_name", "")
        if erddap_station_name:
            # Convert "Bartlett Cove, AK" -> "bartlett_cove_ak"
            station_name_clean = (
                erddap_station_name.lower().replace(" ", "_").replace(",", "").replace(".", "")
            )
            erddap_file = results_dir / station / f"{station_name_clean}_{year}_raw.csv"
            # Also try without state suffix
            if not erddap_file.exists():
                parts = station_name_clean.split("_")
                if len(parts) > 1:
                    erddap_file = results_dir / station / f"{'_'.join(parts[:-1])}_{year}_raw.csv"

    # Determine reference type and coordinates
    ref_source = "Unknown"
    ref_site_id = "Unknown"
    gauge_lat, gauge_lon = station_lat, station_lon  # Default to station location

    # Check ERDDAP first if configured and file exists
    if erddap_file and erddap_file.exists():
        ref_source = f"{erddap_station_name} ERDDAP" if erddap_station_name else "ERDDAP"
        ref_site_id = erddap_station_name or "ERDDAP Station"
        if "latitude" in erddap_config and "longitude" in erddap_config:
            gauge_lat = erddap_config["latitude"]
            gauge_lon = erddap_config["longitude"]
            print(f"Using ERDDAP station {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif noaa_coops.get("enabled") and noaa_coops.get("nearest_station"):
        # Check external_data_sources.noaa_coops structure first (more detailed)
        ref_source = "CO-OPS"
        nearest = noaa_coops["nearest_station"]
        ref_site_id = nearest.get("id", "Unknown")
        if "latitude" in nearest and "longitude" in nearest:
            gauge_lat = nearest["latitude"]
            gauge_lon = nearest["longitude"]
            print(f"Using CO-OPS station {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif usgs_info and usgs_info.get("target_usgs_site"):
        ref_source = "USGS"
        ref_site_id = usgs_info.get("target_usgs_site", "Unknown")
        if "usgs_latitude" in usgs_info and "usgs_longitude" in usgs_info:
            gauge_lat = usgs_info["usgs_latitude"]
            gauge_lon = usgs_info["usgs_longitude"]
            print(f"Using USGS gauge {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif coops_info and coops_info.get("target_station"):
        ref_source = "CO-OPS"
        ref_site_id = coops_info.get("target_station", "Unknown")
        if "station_latitude" in coops_info and "station_longitude" in coops_info:
            gauge_lat = coops_info["station_latitude"]
            gauge_lon = coops_info["station_longitude"]
            print(f"Using CO-OPS station {ref_site_id} at ({gauge_lat}, {gauge_lon})")

    # Get azimuth mask and elevation angles from GNSS-IR config
    gnssir_config_path = project_root / station_config.get("gnssir_json_params_path", "")
    az_ranges = [[0, 80], [330, 360]]
    e1, e2 = 5.0, 15.0  # Default elevation angles
    if gnssir_config_path.exists():
        with open(gnssir_config_path) as f:
            gnssir_config = json.load(f)
            azval = gnssir_config.get("azval2", [])
            if len(azval) >= 2:
                az_ranges = [
                    [azval[i], azval[i + 1]] for i in range(0, len(azval), 2) if i + 1 < len(azval)
                ]
            e1 = gnssir_config.get("e1", e1)  # Min elevation (outer Fresnel zone)
            e2 = gnssir_config.get("e2", e2)  # Max elevation (inner Fresnel zone)

    # Calculate actual RH and elevation ranges from data
    mean_rh = df["RH"].mean()
    min_rh = df["RH"].min()
    max_rh = df["RH"].max()

    df["elev_avg"] = (df["eminO"] + df["emaxO"]) / 2.0
    actual_elev_min = df["elev_avg"].min()
    actual_elev_max = df["elev_avg"].max()

    # Calculate Fresnel zone boundaries using min/max RH and elevation
    # Inner zone: closest reflections (low RH, high elevation)
    inner_fresnel_radius = calculate_fresnel_radius(min_rh, actual_elev_max)
    inner_reflection_dist = min_rh / np.tan(np.radians(actual_elev_max))

    # Outer zone: farthest reflections (high RH, low elevation)
    outer_fresnel_radius = calculate_fresnel_radius(max_rh, actual_elev_min)
    outer_reflection_dist = max_rh / np.tan(np.radians(actual_elev_min))

    print(f"Station {station}:")
    print(f"  RH range: {min_rh:.2f}m to {max_rh:.2f}m (mean: {mean_rh:.2f}m)")
    print(f"  Config elevation range: {e1}°-{e2}°")
    print(f"  Actual elevation range: {actual_elev_min:.2f}°-{actual_elev_max:.2f}°")
    print(
        f"  Inner reflection: {inner_reflection_dist:.1f}m from antenna, Fresnel radius: {inner_fresnel_radius:.1f}m"
    )
    print(
        f"  Outer reflection: {outer_reflection_dist:.1f}m from antenna, Fresnel radius: {outer_fresnel_radius:.1f}m"
    )

    metadata = {
        "ref_source": ref_source,
        "ref_site_id": ref_site_id,
        "az_ranges": az_ranges,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "gauge_lat": gauge_lat,
        "gauge_lon": gauge_lon,
        "station_name": station,
        "inner_fresnel_radius": inner_fresnel_radius,
        "outer_fresnel_radius": outer_fresnel_radius,
        "inner_reflection_dist": inner_reflection_dist,
        "outer_reflection_dist": outer_reflection_dist,
        "mean_rh": mean_rh,
        "elev_min": e1,
        "elev_max": e2,
    }

    df["WSE"] = antenna_height - df["RH"]
    df["WSE_dm"] = df["WSE"] - df["WSE"].mean()

    # Load matched data for residuals
    matched_df = None
    if matched_file.exists():
        matched_df = pd.read_csv(matched_file)
        matched_df["gnss_datetime"] = pd.to_datetime(
            matched_df["gnss_datetime"], format="ISO8601", utc=True
        )
        matched_df["gnss_datetime"] = matched_df["gnss_datetime"].dt.tz_convert(None)

        # Handle residual calculation for different reference sources
        if "residual" not in matched_df.columns:
            gnss_dm_col = "gnss_wse_dm" if "gnss_wse_dm" in matched_df.columns else "gnss_dm"
            # Find reference demeaned column
            ref_dm_cols = [
                col for col in matched_df.columns if col.endswith("_dm") and col != gnss_dm_col
            ]
            if ref_dm_cols:
                matched_df["residual"] = matched_df[gnss_dm_col] - matched_df[ref_dm_cols[0]]

    # Load reference data (ERDDAP, USGS, or CO-OPS)
    ref_df = None
    if erddap_file and erddap_file.exists():
        ref_df = pd.read_csv(erddap_file, skiprows=[1])  # Skip units row
        # ERDDAP files have 'time' and water level columns
        ref_df["datetime"] = pd.to_datetime(ref_df["time"], utc=True).dt.tz_convert(None)
        # Use NAVD88 datum for consistency
        if "water_surface_above_navd88" in ref_df.columns:
            ref_df["wl"] = ref_df["water_surface_above_navd88"]
            ref_df["wl_dm"] = ref_df["wl"] - ref_df["wl"].mean()
    elif usgs_file.exists():
        ref_df = pd.read_csv(usgs_file)
        dt_col = [c for c in ref_df.columns if "datetime" in c.lower()][0]
        ref_df["datetime"] = pd.to_datetime(ref_df[dt_col], utc=True).dt.tz_convert(None)
        if "value_m" in ref_df.columns:
            ref_df["wl"] = ref_df["value_m"]
            ref_df["wl_dm"] = ref_df["wl"] - ref_df["wl"].mean()
    elif coops_file.exists():
        ref_df = pd.read_csv(coops_file)
        # Filter to observations only (exclude predictions)
        if "is_observation" in ref_df.columns:
            ref_df = ref_df[ref_df["is_observation"] == True].copy()
        dt_col = [c for c in ref_df.columns if "datetime" in c.lower()][0]
        ref_df["datetime"] = pd.to_datetime(ref_df[dt_col], utc=True).dt.tz_convert(None)
        # CO-OPS files typically have water_level or predicted_wl columns
        wl_col = None
        for col in ref_df.columns:
            if "water_level" in col.lower() or "wl" in col.lower():
                wl_col = col
                break
        if wl_col:
            ref_df["wl"] = ref_df[wl_col]
            ref_df["wl_dm"] = ref_df["wl"] - ref_df["wl"].mean()

    # Merge residuals
    if matched_df is not None and "residual" in matched_df.columns:
        df["MJD_round"] = np.round(df["MJD"], 3)
        matched_df["MJD_approx"] = (
            matched_df["gnss_datetime"] - pd.Timestamp("1858-11-17")
        ).dt.total_seconds() / 86400
        matched_df["MJD_round"] = np.round(matched_df["MJD_approx"], 3)
        residual_lookup = matched_df.groupby("MJD_round")["residual"].mean().reset_index()
        df = df.merge(residual_lookup, on="MJD_round", how="left")

    return df, ref_df, metadata


def create_frame(
    df_all,
    df_current,
    df_accumulated,
    df_filtered_out,
    ref_df,
    metadata,
    frame_time,
    frame_num,
    total_frames,
    output_path,
    start_time,
    end_time,
    vmin_wl,
    vmax_wl,
    transformer,
    station_x,
    station_y,
    gauge_x,
    gauge_y,
    region_bounds,
    cached_basemaps=None,
):
    """Create a single frame with regional context + satellite overlay."""

    # Three-panel bottom layout: Regional Overview | Regional | Fresnel Zone
    # Reduced from 22x10 to 18x8 for smaller GIF size
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.8, 1.6],
        width_ratios=[0.8, 1, 1.2],
        hspace=0.2,
        wspace=0.08,
        top=0.92,
        bottom=0.05,
        left=0.03,
        right=0.97,
    )

    ref_source = metadata.get("ref_source", "Unknown")
    ref_site_id = metadata.get("ref_site_id", "Unknown")
    station_name = metadata.get("station_name", "Unknown")
    az_ranges = metadata.get("az_ranges", [[0, 80], [330, 360]])

    # Get Fresnel zone geometry
    inner_fresnel_r = metadata.get("inner_fresnel_radius", 3)
    outer_fresnel_r = metadata.get("outer_fresnel_radius", 5)
    inner_refl_dist = metadata.get("inner_reflection_dist", 90)
    outer_refl_dist = metadata.get("outer_reflection_dist", 230)
    mean_rh = metadata.get("mean_rh", 10.0)  # Mean reflector height for distance calculations

    # === Top panel: Time series (spans both columns) ===
    ax_ts = fig.add_subplot(gs[0, :])

    # Create appropriate reference label based on source type
    if "ERDDAP" in ref_source or "CO-OPS" in ref_source:
        ref_legend_label = ref_source
    else:
        ref_legend_label = f"{ref_source} {ref_site_id}"

    if ref_df is not None:
        ref_window = ref_df[
            (ref_df["datetime"] >= start_time - timedelta(hours=6))
            & (ref_df["datetime"] <= end_time + timedelta(hours=6))
        ]
        if len(ref_window) > 0 and "wl_dm" in ref_window.columns:
            ax_ts.plot(
                ref_window["datetime"],
                ref_window["wl_dm"] * 100,
                "r-",
                linewidth=2.5,
                alpha=0.9,
                label=ref_legend_label,
                zorder=2,
            )

    ax_ts.scatter(
        df_all["datetime"],
        df_all["WSE_dm"] * 100,
        c="lightgray",
        s=8,
        alpha=0.3,
        label="GNSS-IR (all)",
        zorder=1,
    )

    if len(df_accumulated) > 0:
        ax_ts.scatter(
            df_accumulated["datetime"],
            df_accumulated["WSE_dm"] * 100,
            c="steelblue",
            s=18,
            alpha=0.7,
            label="GNSS-IR (processed)",
            zorder=3,
        )

    if len(df_current) > 0:
        ax_ts.scatter(
            df_current["datetime"],
            df_current["WSE_dm"] * 100,
            c="gold",
            s=80,
            alpha=0.9,
            label="Current (passed)",
            edgecolors="darkorange",
            linewidths=1.5,
            zorder=5,
        )

    # Show filtered-out (low quality) points in current bin with X markers
    if len(df_filtered_out) > 0:
        ax_ts.scatter(
            df_filtered_out["datetime"],
            df_filtered_out["WSE_dm"] * 100,
            c="red",
            s=60,
            alpha=0.7,
            marker="x",
            linewidths=1.5,
            zorder=4,
            label="Current (filtered)",
        )

    window_start = frame_time - timedelta(hours=3)
    ax_ts.axvspan(window_start, frame_time, alpha=0.15, color="gold", zorder=0)
    ax_ts.axvline(frame_time, color="darkorange", linestyle="-", alpha=0.7, linewidth=2)

    ax_ts.set_xlim(start_time - timedelta(hours=2), end_time + timedelta(hours=2))
    # Calculate y-limits based on actual data range (with 10% padding, asymmetric)
    y_padding = (vmax_wl - vmin_wl) * 0.1
    ax_ts.set_ylim(vmin_wl - y_padding, vmax_wl + y_padding)
    ax_ts.set_ylabel("Demeaned Water Level (cm)", fontsize=11)
    ax_ts.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d %H:%M"))

    # Create appropriate title based on reference source type
    if "ERDDAP" in ref_source or "CO-OPS" in ref_source:
        ref_label = f"{ref_source}"
    else:
        ref_label = f"{ref_source} Gauge {ref_site_id}"

    ax_ts.set_title(
        f"{station_name} Water Level: GNSS-IR vs {ref_label}\n"
        f'Frame {frame_num}/{total_frames} — {frame_time.strftime("%Y-%m-%d %H:%M")} UTC',
        fontsize=12,
        fontweight="bold",
    )

    # === Bottom left: Regional overview ===
    ax_coast = fig.add_subplot(gs[1, 0])

    # Regional bounding box (calculated from station location)
    # Note: State labels below are currently hardcoded for US East Coast
    #       For other regions, these labels should be loaded from a shapefile
    reg_west = region_bounds["west"]
    reg_east = region_bounds["east"]
    reg_south = region_bounds["south"]
    reg_north = region_bounds["north"]

    ec_x_min, ec_y_min = transformer.transform(reg_west, reg_south)
    ec_x_max, ec_y_max = transformer.transform(reg_east, reg_north)

    ax_coast.set_xlim(ec_x_min, ec_x_max)
    ax_coast.set_ylim(ec_y_min, ec_y_max)

    # Use cached basemap if available, otherwise fetch
    if cached_basemaps and "coast" in cached_basemaps:
        coast_img = plt.imread(cached_basemaps["coast"])
        ext = cached_basemaps["coast_extent"]
        ax_coast.imshow(coast_img, extent=ext, aspect="auto", zorder=0)
    else:
        try:
            ctx.add_basemap(ax_coast, source=ctx.providers.CartoDB.Positron, zoom=7)
        except Exception:
            ax_coast.set_facecolor("#c6e2ff")

    # Add state labels
    state_labels = {
        "MD": (-76.7, 39.1),
        "VA": (-78.8, 37.6),
        "DE": (-75.5, 39.15),
        "NJ": (-74.6, 40.1),
        "NC": (-79.5, 36.3),
        "PA": (-77.5, 40.7),
        "WV": (-80.5, 38.8),
    }
    for state, (lon, lat) in state_labels.items():
        sx, sy = transformer.transform(lon, lat)
        if ec_x_min < sx < ec_x_max and ec_y_min < sy < ec_y_max:
            ax_coast.text(
                sx,
                sy,
                state,
                fontsize=9,
                ha="center",
                va="center",
                color="#333333",
                fontweight="bold",
                alpha=0.7,
            )

    # Mark station and gauge locations
    ax_coast.plot(
        station_x,
        station_y,
        "r^",
        markersize=14,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=10,
        label="GNSS Station",
    )
    ax_coast.plot(
        gauge_x,
        gauge_y,
        "bs",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=10,
        label="Reference",
    )

    # Draw connection line between station and gauge
    ax_coast.plot(
        [station_x, gauge_x],
        [station_y, gauge_y],
        "gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        zorder=9,
    )

    # Calculate appropriate buffer size based on station and gauge separation
    # Distance in Web Mercator meters
    from math import sqrt

    gauge_distance_m = sqrt((gauge_x - station_x) ** 2 + (gauge_y - station_y) ** 2)

    # Set buffer to show both stations with comfortable margins
    if gauge_distance_m < 1000:  # Very close (<1km) - like GLBX/Bartlett Cove
        buffer_wide = 2500  # 5km × 5km view
        zoom_level = 14
    elif gauge_distance_m < 8000:  # Close (1-8km)
        buffer_wide = 5000  # 10km × 10km view
        zoom_level = 12
    elif gauge_distance_m < 15000:  # Medium (8-15km) - like VALR
        buffer_wide = 7500  # 15km × 15km view
        zoom_level = 11
    else:  # Far (>15km) - like FORA
        buffer_wide = 30000  # 60km × 60km view
        zoom_level = 9

    # Draw box showing regional view extent
    box_x = [
        station_x - buffer_wide,
        station_x + buffer_wide,
        station_x + buffer_wide,
        station_x - buffer_wide,
        station_x - buffer_wide,
    ]
    box_y = [
        station_y - buffer_wide,
        station_y - buffer_wide,
        station_y + buffer_wide,
        station_y + buffer_wide,
        station_y - buffer_wide,
    ]
    ax_coast.plot(box_x, box_y, "r-", linewidth=2, zorder=9)

    ax_coast.set_aspect("equal")
    ax_coast.axis("off")
    ax_coast.set_title("Regional Overview", fontsize=11, fontweight="bold")

    # === Bottom middle: Regional context (variable scale) ===
    ax_map = fig.add_subplot(gs[1, 1])

    # Center the map between station and gauge for better view
    center_x = (station_x + gauge_x) / 2
    center_y = (station_y + gauge_y) / 2

    ax_map.set_xlim(center_x - buffer_wide, center_x + buffer_wide)
    ax_map.set_ylim(center_y - buffer_wide, center_y + buffer_wide)

    # Use cached basemap if available, otherwise fetch
    if cached_basemaps and "regional" in cached_basemaps:
        regional_img = plt.imread(cached_basemaps["regional"])
        ext = cached_basemaps["regional_extent"]
        ax_map.imshow(regional_img, extent=ext, aspect="auto", zorder=0)
    else:
        try:
            ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, zoom=zoom_level)
        except Exception:
            ax_map.set_facecolor("lightblue")

    # Scaled Fresnel zone indicator (scale with buffer size)
    fresnel_indicator_radius = min(200, buffer_wide * 0.08)  # Scale with map size
    for az_start, az_end in az_ranges:
        theta1, theta2 = 90 - az_end, 90 - az_start
        wedge = Wedge(
            (station_x, station_y),
            fresnel_indicator_radius,
            theta1,
            theta2,
            facecolor="cyan",
            edgecolor="blue",
            alpha=0.35,
            linewidth=1.5,
        )
        ax_map.add_patch(wedge)

    # Station and gauge markers
    ax_map.plot(
        station_x,
        station_y,
        "r^",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=1.5,
        alpha=0.8,
        zorder=10,
        label="GNSS Station",
    )

    # Use appropriate label based on reference source
    if "ERDDAP" in ref_source or "CO-OPS" in ref_source:
        gauge_label = ref_source
    else:
        gauge_label = f"{ref_source} {ref_site_id}"
    ax_map.plot(
        gauge_x,
        gauge_y,
        "bs",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
        alpha=0.8,
        zorder=10,
        label=gauge_label,
    )
    ax_map.plot([station_x, gauge_x], [station_y, gauge_y], "w-", linewidth=1.5, alpha=0.5)

    # Draw box showing Fresnel zone extent
    buffer_close = 60
    box_x2 = [
        station_x - buffer_close,
        station_x + buffer_close,
        station_x + buffer_close,
        station_x - buffer_close,
        station_x - buffer_close,
    ]
    box_y2 = [
        station_y - buffer_close,
        station_y - buffer_close,
        station_y + buffer_close,
        station_y + buffer_close,
        station_y - buffer_close,
    ]
    ax_map.plot(box_x2, box_y2, "cyan", linewidth=2, zorder=9)

    ax_map.annotate(
        "N",
        (center_x, center_y + buffer_wide * 0.85),
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )
    ax_map.legend(loc="lower right", fontsize=8, facecolor="white", framealpha=0.9)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    # Dynamic title based on map scale
    map_scale_km = (buffer_wide * 2) / 1000  # Total width in km
    ax_map.set_title(f"Regional ({map_scale_km:.0f} km)", fontsize=11, fontweight="bold")

    # === Bottom right: Fresnel zone close-up with data ===
    ax_sat = fig.add_subplot(gs[1, 2])
    # Buffer needs to encompass outer reflection zone plus some margin
    buffer_close = int(outer_refl_dist + outer_fresnel_r + 20)

    ax_sat.set_xlim(station_x - buffer_close, station_x + buffer_close)
    ax_sat.set_ylim(station_y - buffer_close, station_y + buffer_close)

    # Use cached basemap if available, otherwise fetch
    if cached_basemaps and "fresnel" in cached_basemaps:
        fresnel_img = plt.imread(cached_basemaps["fresnel"])
        ext = cached_basemaps["fresnel_extent"]
        ax_sat.imshow(fresnel_img, extent=ext, aspect="auto", zorder=0)
    else:
        try:
            ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom="auto")
        except Exception as e:
            print(f"Warning: Could not load Fresnel zone basemap at zoom 18: {e}")
            try:
                ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom=17)
                print("  Loaded with zoom=17 instead")
            except Exception as e2:
                print(f"  Fallback to zoom 17 also failed: {e2}")
                ax_sat.set_facecolor("lightblue")

    # Draw Fresnel zone boundaries as annular rings at reflection distances
    for az_start, az_end in az_ranges:
        theta1, theta2 = 90 - az_end, 90 - az_start

        # Inner Fresnel zone (high elevation, close to antenna)
        inner_annulus_inner = Wedge(
            (station_x, station_y),
            inner_refl_dist - inner_fresnel_r,
            theta1,
            theta2,
            facecolor="none",
            edgecolor="yellow",
            alpha=0.6,
            linewidth=1.5,
            linestyle="--",
        )
        inner_annulus_outer = Wedge(
            (station_x, station_y),
            inner_refl_dist + inner_fresnel_r,
            theta1,
            theta2,
            facecolor="none",
            edgecolor="yellow",
            alpha=0.6,
            linewidth=1.5,
            linestyle="--",
        )
        ax_sat.add_patch(inner_annulus_inner)
        ax_sat.add_patch(inner_annulus_outer)

        # Outer Fresnel zone (low elevation, far from antenna)
        outer_annulus_inner = Wedge(
            (station_x, station_y),
            outer_refl_dist - outer_fresnel_r,
            theta1,
            theta2,
            facecolor="none",
            edgecolor="cyan",
            alpha=0.6,
            linewidth=2,
            linestyle="-",
        )
        outer_annulus_outer = Wedge(
            (station_x, station_y),
            outer_refl_dist + outer_fresnel_r,
            theta1,
            theta2,
            facecolor="none",
            edgecolor="cyan",
            alpha=0.6,
            linewidth=2,
            linestyle="-",
        )
        ax_sat.add_patch(outer_annulus_inner)
        ax_sat.add_patch(outer_annulus_outer)

    # Station marker
    ax_sat.plot(
        station_x,
        station_y,
        "r^",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=10,
    )

    # Colormap for water level
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=vmin_wl, vmax=vmax_wl)

    # Plot accumulated data with varying reflection distances
    if len(df_accumulated) > 0:
        for _, row in df_accumulated.iterrows():
            az_rad = np.radians(row["Azim"])
            # Calculate elevation angle (average of min and max observed)
            elev_deg = (row["eminO"] + row["emaxO"]) / 2.0
            elev_rad = np.radians(elev_deg)

            # Calculate reflection distance using actual RH for this retrieval
            # As water level changes, the reflection point moves horizontally
            reflection_dist = row["RH"] / np.tan(elev_rad)

            # Convert to Cartesian (N=up, E=right)
            dx = reflection_dist * np.sin(az_rad)
            dy = reflection_dist * np.cos(az_rad)

            color = cmap(norm(row["WSE_dm"] * 100))
            ax_sat.plot(
                station_x + dx,
                station_y + dy,
                "o",
                markersize=4,
                color=color,
                alpha=0.5,
                markeredgecolor="none",
            )

    # Plot current bin data larger
    if len(df_current) > 0:
        for _, row in df_current.iterrows():
            az_rad = np.radians(row["Azim"])
            # Calculate elevation angle (average of min and max observed)
            elev_deg = (row["eminO"] + row["emaxO"]) / 2.0
            elev_rad = np.radians(elev_deg)

            # Calculate reflection distance using actual RH for this retrieval
            reflection_dist = row["RH"] / np.tan(elev_rad)

            dx = reflection_dist * np.sin(az_rad)
            dy = reflection_dist * np.cos(az_rad)

            color = cmap(norm(row["WSE_dm"] * 100))
            ax_sat.plot(
                station_x + dx,
                station_y + dy,
                "o",
                markersize=12,
                color=color,
                alpha=0.95,
                markeredgecolor="darkorange",
                markeredgewidth=2,
                zorder=5,
            )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_sat, pad=0.02, shrink=0.7, aspect=20)
    cbar.set_label("Water Level (cm)", fontsize=9)

    # Compass and labels (positioned for larger view extent)
    compass_offset = buffer_close * 0.8  # Position near edge of view
    ax_sat.annotate(
        "N",
        (station_x, station_y + compass_offset),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax_sat.annotate(
        "E",
        (station_x + compass_offset, station_y),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    ax_sat.set_aspect("equal")
    ax_sat.axis("off")
    ax_sat.set_title(
        f"Reflection Distances ({inner_refl_dist:.0f}-{outer_refl_dist:.0f}m) | Current: {len(df_current)} pts",
        fontsize=10,
    )

    plt.savefig(output_path, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close()


def create_animation(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    results_dir: Path,
    output_path: Path,
    quality_filter: bool = True,
    fps: int = 4,
    bin_hours: int = 12,
):
    """Create the full animation."""

    df, ref_df, metadata = load_data(station, year, results_dir)

    df = df[(df["doy"] >= doy_start) & (df["doy"] <= doy_end)].copy()
    print(f"Data in range DOY {doy_start}-{doy_end}: {len(df)} retrievals")

    # Keep both filtered and unfiltered data to show filtering visually
    df_all_unfiltered = df.copy()
    pknoise_median = df["PkNoise"].median()

    if quality_filter:
        df = df[df["PkNoise"] > pknoise_median].copy()
        print(f"After quality filter (PkNoise > {pknoise_median:.2f}): {len(df)} retrievals")
        print(f"Filtered out: {len(df_all_unfiltered) - len(df)} low-quality points")

    # Store threshold in metadata for display
    metadata["pknoise_threshold"] = pknoise_median

    start_time = df["datetime"].min()
    end_time = df["datetime"].max()
    print(f"Time range: {start_time} to {end_time}")

    # Water level range for colormap - include reference data for proper y-axis scaling
    vmin_wl = df["WSE_dm"].quantile(0.05) * 100
    vmax_wl = df["WSE_dm"].quantile(0.95) * 100

    # Include reference data range if available
    if ref_df is not None and "wl_dm" in ref_df.columns:
        ref_window = ref_df[
            (ref_df["datetime"] >= start_time - timedelta(days=1))
            & (ref_df["datetime"] <= end_time + timedelta(days=1))
        ]
        if len(ref_window) > 0:
            ref_min = ref_window["wl_dm"].min() * 100
            ref_max = ref_window["wl_dm"].max() * 100
            vmin_wl = min(vmin_wl, ref_min)
            vmax_wl = max(vmax_wl, ref_max)

    print(f"Water level range: {vmin_wl:.1f} to {vmax_wl:.1f} cm")

    # Transform coordinates from metadata
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    station_lon = metadata["station_lon"]
    station_lat = metadata["station_lat"]
    gauge_lon = metadata["gauge_lon"]
    gauge_lat = metadata["gauge_lat"]

    station_x, station_y = transformer.transform(station_lon, station_lat)
    gauge_x, gauge_y = transformer.transform(gauge_lon, gauge_lat)

    # Calculate regional overview bounds (±2-3 degrees around station)
    # Note: State labels in create_frame are currently hardcoded for US East Coast
    region_bounds = {
        "west": station_lon - 3,
        "east": station_lon + 3,
        "south": station_lat - 2,
        "north": station_lat + 2,
    }

    if ref_df is not None:
        ref_df = ref_df[
            (ref_df["datetime"] >= start_time - timedelta(days=1))
            & (ref_df["datetime"] <= end_time + timedelta(days=1))
        ].copy()

    bin_times = pd.date_range(
        start=start_time.floor(f"{bin_hours}h") + timedelta(hours=bin_hours),
        end=end_time.ceil(f"{bin_hours}h"),
        freq=f"{bin_hours}h",
    )

    total_frames = len(bin_times)
    print(f"Creating {total_frames} frames...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Pre-render and cache basemaps (huge speedup - only fetch tiles once)
        print("Pre-rendering basemaps (this happens once)...")
        cached_basemaps = render_cached_basemaps(
            metadata,
            transformer,
            station_x,
            station_y,
            gauge_x,
            gauge_y,
            region_bounds,
            tmpdir_path,
        )

        frame_paths = []
        df_accumulated = pd.DataFrame()

        for i, bin_end in enumerate(bin_times):
            bin_start = bin_end - timedelta(hours=bin_hours)

            # Get quality-passed points for current bin (these accumulate)
            df_current = df[(df["datetime"] >= bin_start) & (df["datetime"] < bin_end)].copy()

            # Get filtered-out (low quality) points for current bin (shown but don't accumulate)
            df_current_all = df_all_unfiltered[
                (df_all_unfiltered["datetime"] >= bin_start)
                & (df_all_unfiltered["datetime"] < bin_end)
            ]
            df_filtered_out = df_current_all[df_current_all["PkNoise"] <= pknoise_median].copy()

            df_accumulated = pd.concat([df_accumulated, df_current], ignore_index=True)

            frame_path = tmpdir_path / f"frame_{i:04d}.png"
            create_frame(
                df,
                df_current,
                df_accumulated,
                df_filtered_out,
                ref_df,
                metadata,
                bin_end,
                i + 1,
                total_frames,
                frame_path,
                start_time,
                end_time,
                vmin_wl,
                vmax_wl,
                transformer,
                station_x,
                station_y,
                gauge_x,
                gauge_y,
                region_bounds,
                cached_basemaps=cached_basemaps,
            )
            frame_paths.append(frame_path)

            if (i + 1) % 10 == 0:
                print(f"  Created frame {i+1}/{total_frames}")

        print(f"Compiling GIF at {fps} fps...")
        images = [imageio.imread(str(fp)) for fp in frame_paths]
        # Add pause frames at the end
        for _ in range(fps * 2):
            images.append(images[-1])

        # Use pillow plugin with optimization for smaller file size
        imageio.mimsave(
            str(output_path),
            images,
            fps=fps,
            loop=0,
            plugin="pillow",
            optimize=True,
            quantizer="nq",
        )

    print(f"Saved animation to {output_path}")
    return total_frames


def main():
    parser = argparse.ArgumentParser(description="Create animated polar water level GIF")
    parser.add_argument("--station", type=str, default="MDAI")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doy_start", type=int, default=244)
    parser.add_argument("--doy_end", type=int, default=250)
    parser.add_argument(
        "--results_dir", type=str, default=str(Path(__file__).parent.parent / "results_annual")
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--bin_hours", type=int, default=12, help="Time bin size in hours (default: 12)"
    )
    parser.add_argument("--no_quality_filter", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = (
        results_dir
        / args.station
        / f"{args.station}_{args.year}_polar_animation_DOY{args.doy_start}-{args.doy_end}.gif"
    )

    create_animation(
        args.station,
        args.year,
        args.doy_start,
        args.doy_end,
        results_dir,
        output_path,
        quality_filter=not args.no_quality_filter,
        fps=args.fps,
        bin_hours=args.bin_hours,
    )


if __name__ == "__main__":
    main()
