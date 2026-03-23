# ABOUTME: Shared utilities for OPERA RTC-S1 Fresnel zone pipeline
# ABOUTME: Fresnel bbox computation, station config, scene name parsing, path helpers

import json
import re
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default clip extent (meters) — half-width of the square clip around the station
DEFAULT_CLIP_HALF_EXTENT_M = 2500  # 5 km total


def get_station_s1_config(station, project_root=None):
    """Load station coordinates and GNSS-IR params relevant to S1 pipeline.

    Returns dict with: lat, lon, ht, azval2, e1, e2, minH, maxH
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    config_path = root / "config" / "stations_config.json"
    with open(config_path) as f:
        all_config = json.load(f)

    if station not in all_config:
        raise ValueError(f"Station {station} not found in stations_config.json")

    sc = all_config[station]
    result = {
        "lat": sc["latitude_deg"],
        "lon": sc["longitude_deg"],
        "ht": sc["ellipsoidal_height_m"],
    }

    # Load gnssir params for azimuth/elevation ranges
    params_path = root / sc.get("gnssir_json_params_path", f"config/{station.lower()}.json")
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        result["azval2"] = params.get("azval2", [0, 360])
        result["e1"] = params.get("e1", 5)
        result["e2"] = params.get("e2", 25)
        result["minH"] = params.get("minH", 1)
        result["maxH"] = params.get("maxH", 15)
    else:
        logger.warning(f"No gnssir params file at {params_path}, using defaults")
        result.update({"azval2": [0, 360], "e1": 5, "e2": 25, "minH": 1, "maxH": 15})

    return result


def compute_fresnel_bbox(station, per_arc_path=None, buffer_m=None, project_root=None):
    """Compute geographic bounding box for S1 clip around the station.

    Uses a fixed square extent (5 km default) centered on the station.
    The per_arc_path and buffer_m params are accepted but the clip is always
    station-centered — the Fresnel zone is drawn as an overlay on the clip.

    Returns dict with: lon_min, lon_max, lat_min, lat_max, center_lat, center_lon
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    config = get_station_s1_config(station, root)
    lat, lon = config["lat"], config["lon"]

    half_extent = buffer_m if buffer_m is not None else DEFAULT_CLIP_HALF_EXTENT_M

    # Meters per degree at this latitude
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat))

    bbox = {
        "lon_min": lon - half_extent / m_per_deg_lon,
        "lon_max": lon + half_extent / m_per_deg_lon,
        "lat_min": lat - half_extent / m_per_deg_lat,
        "lat_max": lat + half_extent / m_per_deg_lat,
        "center_lat": lat,
        "center_lon": lon,
        "half_extent_m": half_extent,
    }
    return bbox


def compute_fresnel_footprint(station, per_arc_path=None, project_root=None):
    """Compute the Fresnel reflection footprint from per-arc data.

    Returns arrays of reflection point offsets (dx_m, dy_m) relative to station,
    plus summary stats. Used for overlay on S1 clips, not for clip extent.
    """
    import pandas as pd

    root = Path(project_root) if project_root else PROJECT_ROOT
    config = get_station_s1_config(station, root)

    if per_arc_path is None:
        # Find the most recent per-arc parquet
        results_dir = root / "results_annual" / station
        parquets = sorted(results_dir.glob(f"{station}_*_per_arc.parquet"))
        if not parquets:
            logger.warning(f"No per-arc parquet for {station}, estimating from config")
            return _estimate_footprint_from_config(config)
        per_arc_path = parquets[-1]

    df = pd.read_parquet(per_arc_path, columns=["RH", "Azim", "eminO", "emaxO"])
    elev_mid = (df["eminO"] + df["emaxO"]) / 2.0
    refl_dist = df["RH"] / np.tan(np.radians(elev_mid))

    dx = refl_dist * np.sin(np.radians(df["Azim"]))
    dy = refl_dist * np.cos(np.radians(df["Azim"]))

    return {
        "dx_m": dx.values,
        "dy_m": dy.values,
        "refl_dist_m": refl_dist.values,
        "azim_deg": df["Azim"].values,
        "dx_range": (float(dx.min()), float(dx.max())),
        "dy_range": (float(dy.min()), float(dy.max())),
        "dist_range": (float(refl_dist.min()), float(refl_dist.max())),
        "n_arcs": len(df),
    }


def _estimate_footprint_from_config(config):
    """Estimate Fresnel footprint from station config when no per-arc data exists."""
    az_ranges = config["azval2"]
    e1, e2 = config["e1"], config["e2"]
    min_h, max_h = config["minH"], config["maxH"]

    # Generate synthetic reflection points at azimuth/elevation extremes
    azimuths = []
    for i in range(0, len(az_ranges), 2):
        az_start, az_end = az_ranges[i], az_ranges[i + 1]
        azimuths.extend(np.linspace(az_start, az_end, 36).tolist())

    elevations = [e1, (e1 + e2) / 2, e2]
    rh_values = [min_h, (min_h + max_h) / 2, max_h]

    dx_all, dy_all, dist_all, az_all = [], [], [], []
    for az in azimuths:
        for elev in elevations:
            for rh in rh_values:
                dist = rh / np.tan(np.radians(elev))
                dx_all.append(dist * np.sin(np.radians(az)))
                dy_all.append(dist * np.cos(np.radians(az)))
                dist_all.append(dist)
                az_all.append(az)

    dx = np.array(dx_all)
    dy = np.array(dy_all)
    return {
        "dx_m": dx,
        "dy_m": dy,
        "refl_dist_m": np.array(dist_all),
        "azim_deg": np.array(az_all),
        "dx_range": (float(dx.min()), float(dx.max())),
        "dy_range": (float(dy.min()), float(dy.max())),
        "dist_range": (float(min(dist_all)), float(max(dist_all))),
        "n_arcs": len(dx_all),
        "estimated": True,
    }


def parse_opera_rtc_scene_name(scene_name):
    """Parse OPERA RTC-S1 scene name into components.

    Input: 'OPERA_L2_RTC-S1_T025-052202-IW3_20250711T095922Z_20250927T073443Z_S1C_30_v1.0'
    """
    # Pattern: OPERA_L2_RTC-S1_{burst_id}_{acq_datetime}Z_{proc_datetime}Z_{platform}_{res}_{version}
    m = re.match(
        r"OPERA_L2_RTC-S1_(T\d+-\d+-IW\d)_(\d{8}T\d{6})Z_(\d{8}T\d{6})Z_(S1[ABC])_(\d+)_(v[\d.]+)",
        scene_name,
    )
    if not m:
        logger.warning(f"Could not parse scene name: {scene_name}")
        return {"scene_name": scene_name, "parsed": False}

    acq_dt = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")
    return {
        "scene_name": scene_name,
        "burst_id": m.group(1),
        "acquisition_date": acq_dt.strftime("%Y-%m-%d"),
        "acquisition_datetime": acq_dt,
        "processing_date": datetime.strptime(m.group(3), "%Y%m%dT%H%M%S").strftime("%Y-%m-%d"),
        "platform": m.group(4),
        "resolution": int(m.group(5)),
        "version": m.group(6),
        "parsed": True,
    }


# --- Sector-based S1 analysis ---


def _load_gsw_water_mask(station, s1_crs, s1_transform, s1_shape, project_root=None):
    """Load GSW occurrence raster, reproject to S1 grid, return boolean water mask.

    Water pixel = GSW occurrence >= 50%.
    Returns None if GSW file doesn't exist.
    """
    import rasterio
    from pyproj import Transformer

    root = Path(project_root) if project_root else PROJECT_ROOT
    gsw_path = root / "data" / station / "s1_fresnel" / f"{station}_gsw_occurrence.tif"
    if not gsw_path.exists():
        return None

    with rasterio.open(gsw_path) as gsw_src:
        gsw_data = gsw_src.read(1)
        gsw_transform = gsw_src.transform

    # Build S1 pixel centers → WGS84 → GSW pixel lookup
    rows_grid, cols_grid = np.meshgrid(
        np.arange(s1_shape[0]), np.arange(s1_shape[1]), indexing="ij"
    )
    s1_x = s1_transform.c + (cols_grid + 0.5) * s1_transform.a
    s1_y = s1_transform.f + (rows_grid + 0.5) * s1_transform.e

    t = Transformer.from_crs(s1_crs, "EPSG:4326", always_xy=True)
    s1_lon, s1_lat = t.transform(s1_x.ravel(), s1_y.ravel())
    s1_lon = s1_lon.reshape(s1_shape)
    s1_lat = s1_lat.reshape(s1_shape)

    gsw_col = ((s1_lon - gsw_transform.c) / gsw_transform.a).astype(int)
    gsw_row = ((s1_lat - gsw_transform.f) / gsw_transform.e).astype(int)
    gsw_col = np.clip(gsw_col, 0, gsw_data.shape[1] - 1)
    gsw_row = np.clip(gsw_row, 0, gsw_data.shape[0] - 1)

    gsw_at_s1 = gsw_data[gsw_row, gsw_col]
    water_mask = (gsw_at_s1 >= 50) & (gsw_at_s1 != 255)
    return water_mask


def compute_sector_stats(tif_path, station, per_arc_path=None, window_arcs=None,
                         project_root=None):
    """Two-scale S1 backscatter analysis from a cropped GeoTIFF.

    Scale 1 — Regional (full 5km clip):
        Per-quadrant sector stats. With GSW water mask: water-only and land-only stats.

    Scale 2 — Fresnel neighborhood:
        If window_arcs (DataFrame) is provided, computes a DYNAMIC Fresnel zone
        from those specific arcs — the actual reflection area at the time of the
        S1 pass. Uses a 60m buffer around reflection points (roughly 2 pixels,
        accounts for first Fresnel zone ellipse size).

        If no window_arcs, falls back to a static zone from station config
        (azimuth range + max reflection distance from all arcs).

    GSW water mask applied when available:
        - Regional stats split into water-only and land-only
        - Fresnel zone stats reported for water pixels only
        - Removes land contamination from backscatter averages

    Returns dict with:
        'whole_clip': {mean_hh_db, ...}
        'water_only': {mean_hh_db, ...}  (GSW-masked, if available)
        'sectors': {0: {...}, 1: {...}, ...}
        'fresnel_zone': {mean_hh_db, ..., n_pixels}
        'fresnel_water': {mean_hh_db, ...}  (Fresnel zone, water pixels only)
        'fresnel_sectors': {0: {...}, 1: {...}}
        'has_water_mask': bool
    """
    import rasterio
    from pyproj import Transformer

    root = Path(project_root) if project_root else PROJECT_ROOT
    config = get_station_s1_config(station, root)
    lat, lon = config["lat"], config["lon"]

    with rasterio.open(tif_path) as src:
        hh_db = src.read(1)
        hv_db = src.read(2)
        transform = src.transform
        crs = src.crs

    # Build coordinate grids in scene CRS
    rows_grid, cols_grid = np.meshgrid(
        np.arange(hh_db.shape[0]), np.arange(hh_db.shape[1]), indexing="ij"
    )
    x_coords = transform.c + (cols_grid + 0.5) * transform.a
    y_coords = transform.f + (rows_grid + 0.5) * transform.e

    # Station position in scene CRS
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = t.transform(lon, lat)

    # Per-pixel azimuth and distance from station
    dx = x_coords - cx
    dy = y_coords - cy
    dist_m = np.sqrt(dx**2 + dy**2)
    az_deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    az_bins = (az_deg // 90).astype(int)

    valid_mask = ~np.isnan(hh_db)

    # Load GSW water mask if available
    water_mask = _load_gsw_water_mask(station, crs, transform, hh_db.shape, root)
    has_water_mask = water_mask is not None
    if water_mask is None:
        water_mask = np.ones_like(hh_db, dtype=bool)  # no mask = treat all as water

    def _stats(mask):
        m = mask & valid_mask
        n = int(np.sum(m))
        if n == 0:
            return {"n_pixels": 0, "mean_hh_db": np.nan, "mean_hv_db": np.nan,
                    "std_hh_db": np.nan, "std_hv_db": np.nan, "hh_hv_ratio_db": np.nan}
        hh_vals = hh_db[m]
        hv_vals = hv_db[m]
        return {
            "n_pixels": n,
            "mean_hh_db": float(np.nanmean(hh_vals)),
            "mean_hv_db": float(np.nanmean(hv_vals)),
            "std_hh_db": float(np.nanstd(hh_vals)),
            "std_hv_db": float(np.nanstd(hv_vals)),
            "hh_hv_ratio_db": float(np.nanmean(hh_vals) - np.nanmean(hv_vals)),
        }

    # --- Scale 1: Regional ---
    result = {
        "whole_clip": _stats(np.ones_like(hh_db, dtype=bool)),
        "water_only": _stats(water_mask),
        "land_only": _stats(~water_mask),
        "sectors": {},
        "fresnel_sectors": {},
        "has_water_mask": has_water_mask,
    }

    for b in range(4):
        result["sectors"][b] = _stats(az_bins == b)
        # Also water-only per sector
        result["sectors"][b]["water"] = _stats((az_bins == b) & water_mask)
        result["sectors"][b]["land"] = _stats((az_bins == b) & ~water_mask)

    # --- Scale 2: Fresnel zone ---
    fresnel_buffer_m = 60  # ~2 pixels, covers first Fresnel zone ellipse

    if window_arcs is not None and len(window_arcs) >= 3:
        # DYNAMIC Fresnel zone from specific arcs near S1 pass
        elev_mid = (window_arcs["eminO"].values + window_arcs["emaxO"].values) / 2.0
        refl_dist = window_arcs["RH"].values / np.tan(np.radians(elev_mid))
        arc_dx = refl_dist * np.sin(np.radians(window_arcs["Azim"].values))
        arc_dy = refl_dist * np.cos(np.radians(window_arcs["Azim"].values))

        # Reflection points in scene CRS
        refl_x = cx + arc_dx
        refl_y = cy + arc_dy

        # Convert to pixel coords
        refl_col = ((refl_x - transform.c) / transform.a).astype(int)
        refl_row = ((refl_y - transform.f) / transform.e).astype(int)

        # Build Fresnel mask: all pixels within buffer_m of any reflection point
        # (distance in scene CRS meters, not pixel coords)
        fresnel_mask = np.zeros_like(hh_db, dtype=bool)
        for rx, ry in zip(refl_x, refl_y):
            pixel_dist = np.sqrt((x_coords - rx)**2 + (y_coords - ry)**2)
            fresnel_mask |= pixel_dist <= fresnel_buffer_m

        result["fresnel_zone"] = _stats(fresnel_mask)
        result["fresnel_zone"]["method"] = "dynamic"
        result["fresnel_zone"]["buffer_m"] = fresnel_buffer_m
        result["fresnel_zone"]["n_arcs"] = len(window_arcs)

        # Fresnel zone, water pixels only
        result["fresnel_water"] = _stats(fresnel_mask & water_mask)

        # Per-sector within Fresnel zone (water only)
        for b in range(4):
            sector_fresnel = fresnel_mask & (az_bins == b)
            if np.any(sector_fresnel):
                s = _stats(sector_fresnel)
                s["water"] = _stats(sector_fresnel & water_mask)
                result["fresnel_sectors"][b] = s

    else:
        # STATIC fallback: use config azimuth range + max reflection distance
        footprint = compute_fresnel_footprint(station, per_arc_path, root)
        if footprint is not None:
            max_dist = footprint["dist_range"][1] + fresnel_buffer_m
            az_range = config.get("azval2", [0, 360])
            fresnel_mask = dist_m <= max_dist
            az_mask = np.zeros_like(az_deg, dtype=bool)
            for i in range(0, len(az_range), 2):
                az_min, az_max = az_range[i], az_range[i + 1]
                if az_min < az_max:
                    az_mask |= (az_deg >= az_min) & (az_deg <= az_max)
                else:
                    az_mask |= (az_deg >= az_min) | (az_deg <= az_max)
            fresnel_mask &= az_mask
            result["fresnel_zone"] = _stats(fresnel_mask)
            result["fresnel_zone"]["method"] = "static"
            result["fresnel_water"] = _stats(fresnel_mask & water_mask)

            for b in range(4):
                sector_fresnel = fresnel_mask & (az_bins == b)
                if np.any(sector_fresnel):
                    s = _stats(sector_fresnel)
                    s["water"] = _stats(sector_fresnel & water_mask)
                    result["fresnel_sectors"][b] = s
        else:
            result["fresnel_zone"] = _stats(np.ones_like(hh_db, dtype=bool))
            result["fresnel_zone"]["method"] = "fallback"
            result["fresnel_water"] = _stats(water_mask)

    result["az_deg"] = az_deg
    result["dist_m"] = dist_m

    return result


# --- Path helpers ---


def get_s1_fresnel_dir(station, project_root=None):
    """Return data/{STATION}/s1_fresnel/ path, creating if needed."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    d = root / "data" / station / "s1_fresnel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_s1_catalog_path(station, project_root=None):
    """Return path to the S1 RTC catalog CSV."""
    return get_s1_fresnel_dir(station, project_root) / f"{station}_s1_rtc_catalog.csv"


def get_s1_index_path(station, project_root=None):
    """Return path to the S1 Fresnel index CSV (post-crop stats)."""
    return get_s1_fresnel_dir(station, project_root) / f"{station}_s1_fresnel_index.csv"


def get_s1_crop_filename(station, date_str, burst_id=None):
    """Return filename for a cropped S1 GeoTIFF. date_str like '2025-07-11'.

    Includes burst_id suffix (e.g., IW3) to disambiguate multiple passes per day.
    """
    date_compact = date_str.replace("-", "")
    if burst_id:
        # Extract subswath from burst_id like 'T025-052202-IW3' -> 'IW3'
        parts = burst_id.split("-")
        subswath = parts[-1] if parts else burst_id
        return f"{station}_{date_compact}_{subswath}_s1_fresnel.tif"
    return f"{station}_{date_compact}_s1_fresnel.tif"
