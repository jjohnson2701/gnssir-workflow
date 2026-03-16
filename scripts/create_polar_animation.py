# ABOUTME: Creates animated GIF showing water level on satellite imagery with regional context.
# ABOUTME: Overlays GNSS-IR reflection points on satellite basemap with time series.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from pathlib import Path
import argparse
from datetime import timedelta
import imageio.v2 as imageio
import json
import multiprocessing as mp
import contextily as ctx
from pyproj import Transformer
from matplotlib.colors import LightSource

# Approximate number of days per season (quarter year)
SEASONAL_CHUNK_DAYS = 91


def split_into_seasons(doy_start, doy_end, chunk_days=SEASONAL_CHUNK_DAYS):
    """Split a DOY range into seasonal chunks, merging small remainders."""
    chunks = []
    start = doy_start
    while start <= doy_end:
        end = min(start + chunk_days - 1, doy_end)
        # Merge small remainder (<30 days) into this chunk
        if end < doy_end and (doy_end - end) < 30:
            end = doy_end
        chunks.append((start, end))
        start = end + 1
    return chunks


def detect_tidal_extremes(ref_df, wl_col="wl_dm", min_separation_hours=4):
    """
    Detect high and low tide times from regularly-sampled reference data.

    Uses scipy peak detection on the water level signal to find alternating
    highs (peaks) and lows (troughs).

    Args:
        ref_df: DataFrame with 'datetime' and water level column
        wl_col: Name of the water level column
        min_separation_hours: Minimum hours between consecutive extremes

    Returns:
        numpy array of datetime64 values at each tidal extreme
    """
    from scipy.signal import find_peaks

    df = ref_df.dropna(subset=[wl_col]).sort_values("datetime").reset_index(drop=True)
    if len(df) < 3:
        return np.array([], dtype="datetime64[ns]")

    wl = df[wl_col].values
    times = df["datetime"].values

    # Compute minimum distance in samples from the median time step
    dt_median = np.median(np.diff(times) / np.timedelta64(1, "h"))
    min_distance = max(1, int(min_separation_hours / dt_median))

    highs, _ = find_peaks(wl, distance=min_distance)
    lows, _ = find_peaks(-wl, distance=min_distance)

    extreme_indices = np.sort(np.concatenate([highs, lows]))
    return times[extreme_indices]


def detect_tidal_extremes_from_gnssir(df, wl_col="WSE_dm", min_separation_hours=4):
    """
    Detect tidal extremes from irregularly-sampled GNSS-IR water surface data.

    Resamples the noisy signal to a regular 30-minute grid using a rolling
    median, then applies peak detection.

    Args:
        df: DataFrame with 'datetime' and water level column
        wl_col: Name of the water level column
        min_separation_hours: Minimum hours between consecutive extremes

    Returns:
        numpy array of datetime64 values at each tidal extreme
    """
    from scipy.signal import find_peaks

    df_sorted = df.dropna(subset=[wl_col]).sort_values("datetime").copy()
    if len(df_sorted) < 10:
        return np.array([], dtype="datetime64[ns]")

    # Resample to regular 30-min grid
    df_sorted = df_sorted.set_index("datetime")
    resampled = df_sorted[wl_col].resample("30min").median().dropna()

    if len(resampled) < 6:
        return np.array([], dtype="datetime64[ns]")

    # Smooth with 3h rolling window to suppress noise
    smoothed = resampled.rolling(6, center=True, min_periods=3).mean().dropna()

    if len(smoothed) < 6:
        return np.array([], dtype="datetime64[ns]")

    wl = smoothed.values
    times = smoothed.index.values

    min_distance = max(1, int(min_separation_hours / 0.5))  # 0.5h sample interval

    highs, _ = find_peaks(wl, distance=min_distance)
    lows, _ = find_peaks(-wl, distance=min_distance)

    extreme_indices = np.sort(np.concatenate([highs, lows]))
    return times[extreme_indices]


def render_satellite_fresnel_basemap(sat_path, station_lat, station_lon, buffer_m, cache_dir):
    """
    Generate reflection point basemap from a satellite image GeoTIFF.

    Clips the image to a buffer around the station and renders it in
    local-meter coordinates (origin at station).

    Args:
        sat_path: Path to satellite GeoTIFF (e.g., Sentinel-2 TCI)
        station_lat: Station latitude in degrees
        station_lon: Station longitude in degrees
        buffer_m: Buffer radius in meters around station
        cache_dir: Directory to save the cached basemap PNG

    Returns:
        dict with 'path', 'extent', 'local_coords' or None on failure.
    """
    import rasterio
    from rasterio.windows import from_bounds

    try:
        with rasterio.open(sat_path) as src:
            sat_crs = src.crs
            t = Transformer.from_crs("EPSG:4326", str(sat_crs), always_xy=True)
            sx, sy = t.transform(station_lon, station_lat)

            # Clip to buffer around station in the image's native CRS
            window = from_bounds(
                sx - buffer_m,
                sy - buffer_m,
                sx + buffer_m,
                sy + buffer_m,
                transform=src.transform,
            )
            rgb = src.read(window=window)

        if rgb.size == 0:
            return None

        # Transpose to (rows, cols, bands) for imshow
        img = np.moveaxis(rgb, 0, -1)

        fig, ax = plt.subplots(figsize=(10, 10))
        extent = [-buffer_m, buffer_m, -buffer_m, buffer_m]
        ax.imshow(img, extent=extent, origin="upper")
        ax.axis("off")
        basemap_path = Path(cache_dir) / "basemap_fresnel_local.png"
        fig.savefig(basemap_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return {
            "path": basemap_path,
            "extent": extent,
            "local_coords": True,
        }
    except Exception as e:
        print(f"  WARNING: Satellite basemap failed: {e}")
        return None


def render_local_fresnel_basemap(dem_array, dem_resolution, buffer_m, cache_dir):
    """
    Generate reflection point basemap from a DEM array when tile servers are unavailable.

    Creates a hillshaded terrain image with water bodies colored blue. The station
    is assumed to be at the center of the DEM array.

    Args:
        dem_array: 2D numpy array of elevation values (station at center)
        dem_resolution: DEM pixel size in meters
        buffer_m: Buffer radius in meters around station
        cache_dir: Directory to save the cached basemap PNG

    Returns:
        dict with 'path' (Path to PNG), 'extent' (local-meter extent), 'local_coords' (True)
        or None if DEM is all nodata.
    """
    valid = dem_array[dem_array > -9999]
    if len(valid) == 0:
        return None

    # Detect water: pixels near or below sea level (coastal GNSS-IR stations)
    # Stereo DEMs produce noisy low/negative elevations over featureless water
    water_mask = (dem_array < 2.0) & (dem_array > -9999)

    # Generate hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_array, vert_exag=3, dx=dem_resolution, dy=dem_resolution)

    # Build RGBA composite: terrain hillshade with blue water
    rows, cols = dem_array.shape
    rgba = np.zeros((rows, cols, 4))

    # Land: brown/tan hillshade
    land_cmap = plt.cm.YlOrBr
    land_norm = (dem_array - valid.min()) / max(valid.max() - valid.min(), 1.0)
    land_norm = np.clip(land_norm, 0, 1)
    land_colors = land_cmap(land_norm)
    land_colors[..., :3] = land_colors[..., :3] * 0.6 + hs[..., np.newaxis] * 0.4

    # Water: dark blue
    water_colors = np.zeros((rows, cols, 4))
    water_colors[..., 0] = 0.15  # R
    water_colors[..., 1] = 0.3  # G
    water_colors[..., 2] = 0.55  # B
    water_colors[..., 3] = 1.0

    # Composite
    rgba[~water_mask] = land_colors[~water_mask]
    rgba[water_mask] = water_colors[water_mask]

    # Clip to buffer around center
    center_r, center_c = rows // 2, cols // 2
    buf_px = int(buffer_m / dem_resolution)
    r0 = max(0, center_r - buf_px)
    r1 = min(rows, center_r + buf_px)
    c0 = max(0, center_c - buf_px)
    c1 = min(cols, center_c + buf_px)
    clip = rgba[r0:r1, c0:c1]

    # Save as PNG
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = [-buffer_m, buffer_m, -buffer_m, buffer_m]
    ax.imshow(clip, extent=extent, origin="upper")
    ax.axis("off")
    basemap_path = Path(cache_dir) / "basemap_fresnel_local.png"
    fig.savefig(basemap_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "path": basemap_path,
        "extent": extent,
        "local_coords": True,
    }


def render_cached_basemaps(
    metadata,
    transformer,
    station_x,
    station_y,
    gauge_x,
    gauge_y,
    region_bounds,
    cache_dir: Path,
    local_dem=None,
    local_satellite=None,
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
    buffer_close = int(outer_refl_dist + 20)

    # Store geometry info (buffer_wide may be updated by satellite imagery below)
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

    # === Render Regional Context (Esri WorldImagery or local satellite) ===
    print("  Caching regional context basemap...")

    center_x = (station_x + gauge_x) / 2
    center_y = (station_y + gauge_y) / 2
    regional_rendered = False

    if local_satellite is not None:
        sat_path_str, station_lat, station_lon = local_satellite
        import rasterio as _rio

        with _rio.open(sat_path_str) as _src:
            sat_half = (
                min(_src.bounds.right - _src.bounds.left, _src.bounds.top - _src.bounds.bottom) / 2
            )
        sat_result = render_satellite_fresnel_basemap(
            sat_path=sat_path_str,
            station_lat=station_lat,
            station_lon=station_lon,
            buffer_m=int(sat_half),
            cache_dir=cache_dir,
        )
        if sat_result is not None:
            # Rename to regional basemap path
            regional_path = Path(cache_dir) / "basemap_regional.png"
            Path(sat_result["path"]).rename(regional_path)
            cache_paths["regional"] = regional_path
            buffer_wide = int(sat_half)
            cache_paths["regional_extent"] = sat_result["extent"]
            cache_paths["regional_local_coords"] = True
            regional_rendered = True

    if not regional_rendered:
        fig_map, ax_map = plt.subplots(figsize=(10, 10))

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
        cache_paths["regional_local_coords"] = False

    cache_paths["buffer_wide"] = buffer_wide
    cache_paths["center_x"] = center_x
    cache_paths["center_y"] = center_y

    # === Render reflection point close-up basemap ===
    # Priority: satellite imagery > DEM > tile server
    print("  Caching reflection point basemap...")
    fresnel_rendered = False

    if local_satellite is not None:
        sat_path, station_lat, station_lon = local_satellite
        print(f"  Using satellite imagery basemap ({Path(sat_path).name})...")
        sat_result = render_satellite_fresnel_basemap(
            sat_path=sat_path,
            station_lat=station_lat,
            station_lon=station_lon,
            buffer_m=buffer_close,
            cache_dir=cache_dir,
        )
        if sat_result is not None:
            cache_paths["fresnel"] = sat_result["path"]
            cache_paths["fresnel_extent"] = sat_result["extent"]
            cache_paths["fresnel_local_coords"] = True
            fresnel_rendered = True

    if not fresnel_rendered and local_dem is not None:
        dem_array, dem_resolution = local_dem
        print("  Using local DEM basemap (2m resolution)...")
        local_result = render_local_fresnel_basemap(
            dem_array=dem_array,
            dem_resolution=dem_resolution,
            buffer_m=buffer_close,
            cache_dir=cache_dir,
        )
        if local_result is not None:
            cache_paths["fresnel"] = local_result["path"]
            cache_paths["fresnel_extent"] = local_result["extent"]
            cache_paths["fresnel_local_coords"] = True
            fresnel_rendered = True

    if not fresnel_rendered:
        # Fall back to tile server imagery
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
        cache_paths["fresnel_local_coords"] = False

    print("  Basemap caching complete.")
    return cache_paths


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

    # Apply RHdot + IF bias corrections if available
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    station_lower = station.lower()
    local_refl = project_root / "gnssrefl_data_workspace" / "refl_code"
    if_corrected_path = (
        local_refl / "Files" / station_lower
        / f"{station_lower}_{year}_subdaily_edit.txt.withrhdotIF"
    )
    if if_corrected_path.exists():
        from scripts.utils.subdaily_loader import load_corrected_retrievals

        corrected = load_corrected_retrievals(if_corrected_path)
        # Match on MJD+sat+freq (unique per retrieval) and replace RH
        if len(corrected) > 0 and "MJD" in df.columns:
            merge_keys = ["MJD", "sat", "freq"]
            available_keys = [k for k in merge_keys if k in df.columns and k in corrected.columns]
            merged = df.merge(
                corrected[available_keys + ["rh_if_corrected"]],
                on=available_keys, how="left",
            )
            n_matched = merged["rh_if_corrected"].notna().sum()
            if n_matched > 0:
                df.loc[merged["rh_if_corrected"].notna(), "RH"] = (
                    merged.loc[merged["rh_if_corrected"].notna(), "rh_if_corrected"].values
                )
                print(f"Applied RHdot+IF corrections to {n_matched}/{len(df)} retrievals")

    # Get station config
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
    has_reference = False
    gauge_lat, gauge_lon = station_lat, station_lon  # Default to station location

    # Check ERDDAP first if configured and file exists
    if erddap_file and erddap_file.exists():
        ref_source = f"{erddap_station_name} ERDDAP" if erddap_station_name else "ERDDAP"
        ref_site_id = erddap_station_name or "ERDDAP Station"
        has_reference = True
        if "latitude" in erddap_config and "longitude" in erddap_config:
            gauge_lat = erddap_config["latitude"]
            gauge_lon = erddap_config["longitude"]
            print(f"Using ERDDAP station {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif noaa_coops.get("enabled") and noaa_coops.get("nearest_station"):
        # Check external_data_sources.noaa_coops structure first (more detailed)
        ref_source = "CO-OPS"
        nearest = noaa_coops["nearest_station"]
        ref_site_id = nearest.get("id", "Unknown")
        has_reference = True
        if "latitude" in nearest and "longitude" in nearest:
            gauge_lat = nearest["latitude"]
            gauge_lon = nearest["longitude"]
            print(f"Using CO-OPS station {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif usgs_info and usgs_info.get("target_usgs_site"):
        ref_source = "USGS"
        ref_site_id = usgs_info.get("target_usgs_site", "Unknown")
        has_reference = True
        if "usgs_latitude" in usgs_info and "usgs_longitude" in usgs_info:
            gauge_lat = usgs_info["usgs_latitude"]
            gauge_lon = usgs_info["usgs_longitude"]
            print(f"Using USGS gauge {ref_site_id} at ({gauge_lat}, {gauge_lon})")
    elif coops_info and coops_info.get("target_station"):
        ref_source = "CO-OPS"
        ref_site_id = coops_info.get("target_station", "Unknown")
        has_reference = True
        if "station_latitude" in coops_info and "station_longitude" in coops_info:
            gauge_lat = coops_info["station_latitude"]
            gauge_lon = coops_info["station_longitude"]
            print(f"Using CO-OPS station {ref_site_id} at ({gauge_lat}, {gauge_lon})")

    # Calculate reflection distance range from data (for view extent)
    mean_rh = df["RH"].mean()
    max_rh = df["RH"].max()

    df["elev_avg"] = (df["eminO"] + df["emaxO"]) / 2.0
    actual_elev_min = df["elev_avg"].min()

    # Farthest reflection distance (high RH, low elevation) sets the view extent
    outer_reflection_dist = max_rh / np.tan(np.radians(actual_elev_min))

    print(f"Station {station}:")
    print(f"  RH range: {df['RH'].min():.2f}m to {max_rh:.2f}m (mean: {mean_rh:.2f}m)")
    print(f"  Actual elevation range: {actual_elev_min:.2f}°-{df['elev_avg'].max():.2f}°")
    print(f"  Max reflection distance: {outer_reflection_dist:.1f}m from antenna")

    metadata = {
        "ref_source": ref_source,
        "ref_site_id": ref_site_id,
        "has_reference": has_reference,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "gauge_lat": gauge_lat,
        "gauge_lon": gauge_lon,
        "station_name": station,
        "outer_reflection_dist": outer_reflection_dist,
        "mean_rh": mean_rh,
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
        # Find water level column from config or common names
        wl_col = erddap_config.get("variables", {}).get("water_level", None)
        if wl_col is None or wl_col not in ref_df.columns:
            for col in [
                "water_surface_above_navd88",
                "sea_surface_height_above_geopotential_datum",
                "sea_level",
                "water_level",
                "wl",
            ]:
                if col in ref_df.columns:
                    wl_col = col
                    break
        if wl_col and wl_col in ref_df.columns:
            units_scale = erddap_config.get("variables", {}).get("units_scale", 1.0)
            ref_df["wl"] = ref_df[wl_col] * units_scale
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
            ref_df = ref_df[ref_df["is_observation"]].copy()
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


def render_cover_frame(metadata, frame_config, output_path):
    """
    Render a single cover frame with map context panels and station metadata.

    Shown for ~3 seconds at animation start. Contains all 3 basemap panels
    (regional overview, regional context, reflection point basemap) plus
    station info text. Dimensions match animation frames for ffmpeg concat.
    """
    figsize = frame_config.get("figsize", (10, 10))
    dpi = frame_config.get("dpi", 100)
    cached_basemaps = frame_config["cached_basemaps"]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.15,
        wspace=0.1,
        top=0.93,
        bottom=0.03,
        left=0.03,
        right=0.97,
    )

    station_name = metadata.get("station_name", "Unknown")
    ref_source = metadata.get("ref_source", "Unknown")
    ref_site_id = metadata.get("ref_site_id", "")
    year = frame_config.get("year", "")
    doy_start = frame_config.get("doy_start", "")
    doy_end = frame_config.get("doy_end", "")

    # Top-left: Regional overview
    ax_coast = fig.add_subplot(gs[0, 0])
    if cached_basemaps and "coast" in cached_basemaps:
        coast_img = plt.imread(cached_basemaps["coast"])
        ext = cached_basemaps["coast_extent"]
        ax_coast.imshow(coast_img, extent=ext, aspect="auto")
    ax_coast.set_aspect("equal")
    ax_coast.axis("off")
    ax_coast.set_title("Regional Overview", fontsize=11, fontweight="bold")

    # Top-right: Regional context
    ax_regional = fig.add_subplot(gs[0, 1])
    if cached_basemaps and "regional" in cached_basemaps:
        regional_img = plt.imread(cached_basemaps["regional"])
        ext = cached_basemaps["regional_extent"]
        ax_regional.imshow(regional_img, extent=ext, aspect="auto")
    ax_regional.set_aspect("equal")
    ax_regional.axis("off")
    buffer_wide = cached_basemaps.get("buffer_wide", 5000)
    map_scale_km = (buffer_wide * 2) / 1000
    ax_regional.set_title(
        f"Regional ({map_scale_km:.0f} km)", fontsize=11, fontweight="bold"
    )

    # Bottom-left: Reflection point basemap
    ax_fresnel = fig.add_subplot(gs[1, 0])
    if cached_basemaps and "fresnel" in cached_basemaps:
        fresnel_img = plt.imread(cached_basemaps["fresnel"])
        ext = cached_basemaps["fresnel_extent"]
        ax_fresnel.imshow(fresnel_img, extent=ext, aspect="auto")
    ax_fresnel.set_aspect("equal")
    ax_fresnel.axis("off")
    outer_dist = metadata.get("outer_reflection_dist", 0)
    ax_fresnel.set_title(
        f"Reflection Zone ({outer_dist:.0f}m)", fontsize=11, fontweight="bold"
    )

    # Bottom-right: Station metadata text
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")

    lat = metadata.get("station_lat", 0)
    lon = metadata.get("station_lon", 0)
    has_ref = metadata.get("has_reference", False)

    info_lines = [
        f"Station: {station_name}",
        f"Year: {year}",
        f"DOY: {doy_start} – {doy_end}",
        f"Lat: {lat:.4f}°  Lon: {lon:.4f}°",
        "",
    ]
    if has_ref:
        ref_label = ref_source
        if ref_site_id and ref_site_id != "Unknown":
            ref_label += f" ({ref_site_id})"
        info_lines.append(f"Reference: {ref_label}")
        gauge_lat = metadata.get("gauge_lat", 0)
        gauge_lon = metadata.get("gauge_lon", 0)
        info_lines.append(f"Gauge: {gauge_lat:.4f}°, {gauge_lon:.4f}°")
    else:
        info_lines.append("Reference: None")

    ax_text.text(
        0.1, 0.85,
        "\n".join(info_lines),
        transform=ax_text.transAxes,
        fontsize=13,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        f"{station_name} — GNSS-IR Polar Animation",
        fontsize=16, fontweight="bold",
    )

    plt.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_path


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

    # Three-panel bottom layout: Regional Overview | Regional | Reflection Points
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
    has_reference = metadata.get("has_reference", False)
    station_name = metadata.get("station_name", "Unknown")

    outer_refl_dist = metadata.get("outer_reflection_dist", 230)

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

    # Highlight the current window on the time series
    if len(df_current) > 0:
        window_start = df_current["datetime"].min()
        window_end = df_current["datetime"].max()
    else:
        window_start = frame_time - timedelta(hours=3)
        window_end = frame_time
    ax_ts.axvspan(window_start, window_end, alpha=0.15, color="gold", zorder=0)
    window_center = window_start + (window_end - window_start) / 2
    ax_ts.axvline(window_center, color="darkorange", linestyle="-", alpha=0.7, linewidth=2)

    ax_ts.set_xlim(start_time - timedelta(hours=2), end_time + timedelta(hours=2))
    # Calculate y-limits based on actual data range (with 10% padding, asymmetric)
    y_padding = (vmax_wl - vmin_wl) * 0.1
    ax_ts.set_ylim(vmin_wl - y_padding, vmax_wl + y_padding)
    ax_ts.set_ylabel("Demeaned Water Level (cm)", fontsize=11)
    ax_ts.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d %H:%M"))

    # Create appropriate title based on reference source type
    if ref_df is not None:
        if "ERDDAP" in ref_source or "CO-OPS" in ref_source:
            ref_label = f"{ref_source}"
        else:
            ref_label = f"{ref_source} Gauge {ref_site_id}"
        title_line1 = f"{station_name} Water Level: GNSS-IR vs {ref_label}"
    else:
        title_line1 = f"{station_name} Water Level: GNSS-IR"

    ax_ts.set_title(
        f"{title_line1}\n"
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
    if has_reference:
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
    from math import sqrt

    if has_reference:
        gauge_distance_m = sqrt((gauge_x - station_x) ** 2 + (gauge_y - station_y) ** 2)
    else:
        gauge_distance_m = 0

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

    # Determine coordinate origin for regional panel
    use_regional_local = (
        cached_basemaps.get("regional_local_coords", False) if cached_basemaps else False
    )
    if use_regional_local:
        reg_sx, reg_sy = 0, 0
        reg_gx = gauge_x - station_x
        reg_gy = gauge_y - station_y
        reg_cx, reg_cy = reg_sx, reg_sy
    else:
        reg_sx, reg_sy = station_x, station_y
        reg_gx, reg_gy = gauge_x, gauge_y
        reg_cx = (station_x + gauge_x) / 2
        reg_cy = (station_y + gauge_y) / 2

    ax_map.set_xlim(reg_cx - buffer_wide, reg_cx + buffer_wide)
    ax_map.set_ylim(reg_cy - buffer_wide, reg_cy + buffer_wide)

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

    # Station and gauge markers
    ax_map.plot(
        reg_sx,
        reg_sy,
        "r^",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=1.5,
        alpha=0.8,
        zorder=10,
        label="GNSS Station",
    )

    if has_reference:
        # Use appropriate label based on reference source
        if "ERDDAP" in ref_source or "CO-OPS" in ref_source:
            gauge_label = ref_source
        else:
            gauge_label = f"{ref_source} {ref_site_id}"
        ax_map.plot(
            reg_gx,
            reg_gy,
            "bs",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=0.8,
            zorder=10,
            label=gauge_label,
        )
        ax_map.plot([reg_sx, reg_gx], [reg_sy, reg_gy], "w-", linewidth=1.5, alpha=0.5)

    # Draw box showing reflection point view extent
    buffer_close = 60
    box_x2 = [
        reg_sx - buffer_close,
        reg_sx + buffer_close,
        reg_sx + buffer_close,
        reg_sx - buffer_close,
        reg_sx - buffer_close,
    ]
    box_y2 = [
        reg_sy - buffer_close,
        reg_sy - buffer_close,
        reg_sy + buffer_close,
        reg_sy + buffer_close,
        reg_sy - buffer_close,
    ]
    ax_map.plot(box_x2, box_y2, "cyan", linewidth=2, zorder=9)

    ax_map.annotate(
        "N",
        (reg_cx, reg_cy + buffer_wide * 0.85),
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

    # === Bottom right: Reflection point close-up with data ===
    ax_sat = fig.add_subplot(gs[1, 2])
    # Buffer needs to encompass outer reflection zone plus some margin
    buffer_close = int(outer_refl_dist + 20)

    # Local coords: origin at (0,0) in real meters from antenna
    # Web Mercator coords: origin at (station_x, station_y)
    use_local = cached_basemaps.get("fresnel_local_coords", False) if cached_basemaps else False
    origin_x = 0 if use_local else station_x
    origin_y = 0 if use_local else station_y

    ax_sat.set_xlim(origin_x - buffer_close, origin_x + buffer_close)
    ax_sat.set_ylim(origin_y - buffer_close, origin_y + buffer_close)

    # Use cached basemap if available, otherwise fetch
    if cached_basemaps and "fresnel" in cached_basemaps:
        fresnel_img = plt.imread(cached_basemaps["fresnel"])
        ext = cached_basemaps["fresnel_extent"]
        ax_sat.imshow(fresnel_img, extent=ext, aspect="auto", zorder=0)
    else:
        try:
            ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom="auto")
        except Exception as e:
            print(f"Warning: Could not load basemap at zoom 18: {e}")
            try:
                ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery, zoom=17)
                print("  Loaded with zoom=17 instead")
            except Exception as e2:
                print(f"  Fallback to zoom 17 also failed: {e2}")
                ax_sat.set_facecolor("lightblue")

    # Station marker
    ax_sat.plot(
        origin_x,
        origin_y,
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
            elev_deg = (row["eminO"] + row["emaxO"]) / 2.0
            elev_rad = np.radians(elev_deg)
            reflection_dist = row["RH"] / np.tan(elev_rad)

            # Convert to Cartesian (N=up, E=right)
            dx = reflection_dist * np.sin(az_rad)
            dy = reflection_dist * np.cos(az_rad)

            color = cmap(norm(row["WSE_dm"] * 100))
            ax_sat.plot(
                origin_x + dx,
                origin_y + dy,
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
            elev_deg = (row["eminO"] + row["emaxO"]) / 2.0
            elev_rad = np.radians(elev_deg)
            reflection_dist = row["RH"] / np.tan(elev_rad)

            dx = reflection_dist * np.sin(az_rad)
            dy = reflection_dist * np.cos(az_rad)

            color = cmap(norm(row["WSE_dm"] * 100))
            ax_sat.plot(
                origin_x + dx,
                origin_y + dy,
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
    compass_offset = buffer_close * 0.8
    ax_sat.annotate(
        "N",
        (origin_x, origin_y + compass_offset),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax_sat.annotate(
        "E",
        (origin_x + compass_offset, origin_y),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    ax_sat.set_aspect("equal")
    ax_sat.axis("off")
    ax_sat.set_title(
        f"Reflection Points (max {outer_refl_dist:.0f}m) | "
        f"Current: {len(df_current)} pts",
        fontsize=10,
    )

    plt.savefig(output_path, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close()


# Module-level state for multiprocessing workers
_worker_shared = {}


def _init_frame_worker(shared_data):
    """Initialize each worker process with shared data."""
    global _worker_shared
    _worker_shared = shared_data
    import matplotlib

    matplotlib.use("Agg")


def _render_frame_task(frame_args):
    """Render a single animation frame (called in worker process)."""
    i, bin_start, bin_end, frame_path_str = frame_args
    d = _worker_shared
    frame_path = Path(frame_path_str)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    df = d["df"]
    df_all_unfiltered = d["df_all_unfiltered"]

    df_current = df[(df["datetime"] >= bin_start) & (df["datetime"] < bin_end)].copy()

    df_current_all = df_all_unfiltered[
        (df_all_unfiltered["datetime"] >= bin_start) & (df_all_unfiltered["datetime"] < bin_end)
    ]
    df_filtered_out = df_current_all[df_current_all["PkNoise"] <= d["pknoise_median"]].copy()

    # Windowed data (same as current bin for tide-synced frames)
    df_accumulated = df_current.copy()

    create_frame(
        df,
        df_current,
        df_accumulated,
        df_filtered_out,
        d["ref_df"],
        d["metadata"],
        bin_end,
        i + 1,
        d["total_frames"],
        frame_path,
        d["start_time"],
        d["end_time"],
        d["vmin_wl"],
        d["vmax_wl"],
        transformer,
        d["station_x"],
        d["station_y"],
        d["gauge_x"],
        d["gauge_y"],
        d["region_bounds"],
        cached_basemaps=d["cached_basemaps"],
    )

    if (i + 1) % 10 == 0:
        print(f"  Created frame {i + 1}/{d['total_frames']}")

    return frame_path_str


def create_animation(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    results_dir: Path,
    output_path: Path,
    quality_filter: bool = True,
    fps: int = 4,
    bin_hours: int = 6,
    window_hours: int = 6,
    n_workers: int = 0,
    mode: str = "presentation",
    output_format: str = "gif",
):
    """Create the full animation. Uses multiprocessing when n_workers > 1."""

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

    # Sync frames to tidal extremes (high/low tide) when possible
    frame_center_times = None
    if ref_df is not None and "wl_dm" in ref_df.columns and len(ref_df) > 10:
        extremes = detect_tidal_extremes(ref_df)
        # Filter to data range
        extremes = extremes[
            (extremes >= np.datetime64(start_time))
            & (extremes <= np.datetime64(end_time))
        ]
        if len(extremes) >= 2:
            frame_center_times = pd.DatetimeIndex(extremes)
            print(f"Synced {len(frame_center_times)} frames to tidal extremes from reference data")

    if frame_center_times is None:
        # Fallback: detect from GNSS-IR data
        extremes = detect_tidal_extremes_from_gnssir(df)
        extremes = extremes[
            (extremes >= np.datetime64(start_time))
            & (extremes <= np.datetime64(end_time))
        ]
        if len(extremes) >= 2:
            frame_center_times = pd.DatetimeIndex(extremes)
            print(f"Synced {len(frame_center_times)} frames to tidal extremes from GNSS-IR data")

    if frame_center_times is None:
        # Last resort: regular bins
        frame_center_times = pd.date_range(
            start=start_time.floor(f"{bin_hours}h") + timedelta(hours=bin_hours),
            end=end_time.ceil(f"{bin_hours}h"),
            freq=f"{bin_hours}h",
        )
        print(f"No tidal signal detected, using regular {bin_hours}h bins")

    total_frames = len(frame_center_times)
    print(f"Creating {total_frames} frames...")

    frames_dir = (
        results_dir / station / "animation_frames" / f"{station}_{year}_DOY{doy_start}-{doy_end}"
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving frames to {frames_dir}")

    # Load local satellite imagery or DEM for reflection point basemap
    local_satellite = None
    local_dem = None
    sat_path = results_dir / station / f"{station.lower()}_sentinel2_10m.tif"
    dem_path = results_dir / station / f"{station.lower()}_arcticdem_2m.tif"

    if sat_path.exists():
        local_satellite = (str(sat_path), metadata["station_lat"], metadata["station_lon"])
        print(f"Loaded satellite imagery: {sat_path.name}")
    elif dem_path.exists():
        try:
            import rasterio

            with rasterio.open(dem_path) as src:
                local_dem = (src.read(1).astype(float), src.res[0])
                print(f"Loaded local DEM: {dem_path.name} ({local_dem[0].shape})")
        except ImportError:
            print("rasterio not available, skipping local DEM")
        except Exception as e:
            print(f"Failed to load local DEM: {e}")

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
        frames_dir,
        local_dem=local_dem,
        local_satellite=local_satellite,
    )

    # Build frame arguments using window_hours centered on each frame time
    half_window = timedelta(hours=window_hours / 2)
    frame_args = []
    for i, center_time in enumerate(frame_center_times):
        bin_start = center_time - half_window
        bin_end = center_time + half_window
        frame_path = frames_dir / f"frame_{i:04d}.png"
        frame_args.append((i, bin_start, bin_end, str(frame_path)))

    if n_workers == 0:
        n_workers = min(mp.cpu_count(), 8)

    if n_workers > 1:
        shared_data = {
            "df": df,
            "df_all_unfiltered": df_all_unfiltered,
            "pknoise_median": pknoise_median,
            "ref_df": ref_df,
            "metadata": metadata,
            "total_frames": total_frames,
            "start_time": start_time,
            "end_time": end_time,
            "vmin_wl": vmin_wl,
            "vmax_wl": vmax_wl,
            "station_x": station_x,
            "station_y": station_y,
            "gauge_x": gauge_x,
            "gauge_y": gauge_y,
            "region_bounds": region_bounds,
            "cached_basemaps": cached_basemaps,
        }
        print(f"Rendering {total_frames} frames using {n_workers} workers...")
        with mp.Pool(n_workers, initializer=_init_frame_worker, initargs=(shared_data,)) as pool:
            results = pool.map(_render_frame_task, frame_args)
        frame_paths = [Path(p) for p in results]
    else:
        # Serial fallback
        frame_paths = []
        for i, bin_start, bin_end, frame_path_str in frame_args:
            frame_path = Path(frame_path_str)
            df_current = df[(df["datetime"] >= bin_start) & (df["datetime"] < bin_end)].copy()
            df_current_all = df_all_unfiltered[
                (df_all_unfiltered["datetime"] >= bin_start)
                & (df_all_unfiltered["datetime"] < bin_end)
            ]
            df_filtered_out = df_current_all[df_current_all["PkNoise"] <= pknoise_median].copy()
            df_accumulated = df_current.copy()
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
        "--bin_hours", type=int, default=6,
        help="Fallback time bin size in hours when no tidal signal detected (default: 6)",
    )
    parser.add_argument(
        "--window_hours", type=int, default=6,
        help="Plotting window size in hours centered on each frame (default: 6)",
    )
    parser.add_argument("--no_quality_filter", action="store_true")
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Disable automatic seasonal splitting for large DOY ranges",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for frame rendering (0=auto, 1=serial)",
    )
    parser.add_argument(
        "--mode",
        choices=["presentation", "analysis"],
        default="presentation",
        help="Animation mode: presentation (NPS, fixed bins, 14-day rolling TS) "
        "or analysis (EarthScope, tidal-synced, 3-day rolling TS) (default: presentation)",
    )
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format (default: gif)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    quality_filter = not args.no_quality_filter

    ext = args.format
    doy_range = args.doy_end - args.doy_start + 1
    if doy_range > SEASONAL_CHUNK_DAYS and not args.no_split:
        chunks = split_into_seasons(args.doy_start, args.doy_end)
        print(
            f"Splitting {doy_range}-day range into {len(chunks)} seasonal animations: "
            + ", ".join(f"DOY {s}-{e}" for s, e in chunks)
        )
        for doy_s, doy_e in chunks:
            output_path = (
                results_dir
                / args.station
                / f"{args.station}_{args.year}_polar_animation_DOY{doy_s}-{doy_e}.{ext}"
            )
            create_animation(
                args.station,
                args.year,
                doy_s,
                doy_e,
                results_dir,
                output_path,
                quality_filter=quality_filter,
                fps=args.fps,
                bin_hours=args.bin_hours,
                window_hours=args.window_hours,
                n_workers=args.workers,
                mode=args.mode,
                output_format=args.format,
            )
    else:
        output_path = (
            results_dir / args.station / f"{args.station}_{args.year}_polar_animation_"
            f"DOY{args.doy_start}-{args.doy_end}.{ext}"
        )
        create_animation(
            args.station,
            args.year,
            args.doy_start,
            args.doy_end,
            results_dir,
            output_path,
            quality_filter=quality_filter,
            fps=args.fps,
            bin_hours=args.bin_hours,
            window_hours=args.window_hours,
            n_workers=args.workers,
            mode=args.mode,
            output_format=args.format,
        )


if __name__ == "__main__":
    main()
