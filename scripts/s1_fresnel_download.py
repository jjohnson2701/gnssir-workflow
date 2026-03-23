# ABOUTME: Download OPERA RTC-S1 HH+HV GeoTIFFs, crop to 5km around station, save small GeoTIFF
# ABOUTME: Builds index CSV with pre-computed backscatter stats for dashboard

"""
Download and crop OPERA RTC-S1 scenes to the area around a GNSS-IR station.

Usage:
    # Download all scenes in catalog
    python scripts/s1_fresnel_download.py --station UMNQ

    # Download first 10 scenes (test run)
    python scripts/s1_fresnel_download.py --station UMNQ --max-scenes 10

    # Skip already-downloaded scenes
    python scripts/s1_fresnel_download.py --station UMNQ --skip-existing

    # Prioritize transition months (freeze-up/break-up) first
    python scripts/s1_fresnel_download.py --station UMNQ --priority-months 5,6,10,11,12
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.s1_fresnel_utils import (
    compute_fresnel_bbox,
    get_s1_catalog_path,
    get_s1_crop_filename,
    get_s1_fresnel_dir,
    get_s1_index_path,
    parse_opera_rtc_scene_name,
)

logger = logging.getLogger(__name__)


def download_with_wget(url, output_path, timeout=300):
    """Download a file using wget with .netrc auth. Returns True on success."""
    cmd = [
        "wget", "-q", "--netrc",
        "-O", str(output_path),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            logger.error(f"wget failed (rc={result.returncode}): {result.stderr[:200]}")
            if output_path.exists():
                output_path.unlink()
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"wget timed out after {timeout}s: {url}")
        if output_path.exists():
            output_path.unlink()
        return False


def crop_and_save(hh_path, hv_path, bbox, output_path):
    """Crop HH+HV GeoTIFFs to bbox, convert to dB, save as 2-band GeoTIFF.

    Returns dict with pre-computed stats, or None on failure.
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.windows import from_bounds

    try:
        with rasterio.open(hh_path) as src_hh:
            # Transform bbox from WGS84 to scene CRS
            t = Transformer.from_crs("EPSG:4326", src_hh.crs, always_xy=True)
            x_min, y_min = t.transform(bbox["lon_min"], bbox["lat_min"])
            x_max, y_max = t.transform(bbox["lon_max"], bbox["lat_max"])

            # Ensure correct ordering (y might flip in polar projections)
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            # Compute pixel window
            window = from_bounds(x_min, y_min, x_max, y_max, transform=src_hh.transform)
            window = window.round_offsets().round_lengths()

            # Read HH data
            hh_data = src_hh.read(1, window=window)
            crop_transform = src_hh.window_transform(window)
            crop_crs = src_hh.crs

        with rasterio.open(hv_path) as src_hv:
            hv_data = src_hv.read(1, window=window)

        # Convert linear power to dB
        with np.errstate(divide="ignore", invalid="ignore"):
            hh_db = np.where(
                np.isnan(hh_data) | (hh_data <= 0),
                np.nan,
                10.0 * np.log10(np.maximum(hh_data, 1e-10)),
            )
            hv_db = np.where(
                np.isnan(hv_data) | (hv_data <= 0),
                np.nan,
                10.0 * np.log10(np.maximum(hv_data, 1e-10)),
            )

        # Write 2-band GeoTIFF
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": hh_db.shape[1],
            "height": hh_db.shape[0],
            "count": 2,
            "crs": crop_crs,
            "transform": crop_transform,
            "nodata": np.nan,
            "compress": "deflate",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(hh_db.astype(np.float32), 1)
            dst.write(hv_db.astype(np.float32), 2)
            dst.set_band_description(1, "HH_dB")
            dst.set_band_description(2, "HV_dB")

        # Compute stats (excluding NaN)
        hh_valid = hh_db[~np.isnan(hh_db)]
        hv_valid = hv_db[~np.isnan(hv_db)]
        n_valid = len(hh_valid)

        stats = {
            "n_valid_pixels": n_valid,
            "n_total_pixels": hh_db.size,
            "shape_rows": hh_db.shape[0],
            "shape_cols": hh_db.shape[1],
        }

        if n_valid > 0:
            stats["mean_hh_db"] = float(np.nanmean(hh_db))
            stats["mean_hv_db"] = float(np.nanmean(hv_db))
            stats["std_hh_db"] = float(np.nanstd(hh_db))
            stats["std_hv_db"] = float(np.nanstd(hv_db))
            stats["hh_hv_ratio_db"] = stats["mean_hh_db"] - stats["mean_hv_db"]
        else:
            stats["mean_hh_db"] = np.nan
            stats["mean_hv_db"] = np.nan
            stats["std_hh_db"] = np.nan
            stats["std_hv_db"] = np.nan
            stats["hh_hv_ratio_db"] = np.nan

        return stats

    except Exception as e:
        logger.error(f"Crop failed: {e}")
        return None


def download_and_crop_scene(row, bbox, output_dir, temp_dir):
    """Download HH+HV, crop, save. Returns index row dict or None."""
    scene_name = row["scene_name"]
    parsed = parse_opera_rtc_scene_name(scene_name)
    acq_date = parsed.get("acquisition_date", row.get("acquisition_date", "unknown"))

    burst_id = parsed.get("burst_id", "")
    crop_filename = get_s1_crop_filename(row.get("station", "XXXX"), acq_date, burst_id)
    output_path = output_dir / crop_filename

    hh_url = row.get("hh_url", "")
    hv_url = row.get("hv_url", "")

    # Handle NaN/empty URLs
    if not isinstance(hh_url, str) or not hh_url or not isinstance(hv_url, str) or not hv_url:
        logger.warning(f"  Missing HH/HV URL for {scene_name}, skipping")
        return None

    # Download HH
    hh_tmp = Path(temp_dir) / f"{scene_name}_HH.tif"
    hv_tmp = Path(temp_dir) / f"{scene_name}_HV.tif"

    logger.info(f"  Downloading HH ({row.get('hh_bytes', 0) / 1024 / 1024:.1f} MB)...")
    if not download_with_wget(hh_url, hh_tmp):
        return None

    logger.info(f"  Downloading HV ({row.get('hv_bytes', 0) / 1024 / 1024:.1f} MB)...")
    if not download_with_wget(hv_url, hv_tmp):
        hh_tmp.unlink(missing_ok=True)
        return None

    # Crop and save
    logger.info(f"  Cropping to {bbox['half_extent_m'] * 2 / 1000:.0f}km extent...")
    stats = crop_and_save(hh_tmp, hv_tmp, bbox, output_path)

    # Clean up temp files
    hh_tmp.unlink(missing_ok=True)
    hv_tmp.unlink(missing_ok=True)

    if stats is None:
        return None

    crop_size_kb = output_path.stat().st_size / 1024
    logger.info(
        f"  Saved: {crop_filename} ({stats['shape_rows']}x{stats['shape_cols']}, "
        f"{crop_size_kb:.0f} KB, mean HH={stats.get('mean_hh_db', 0):.1f} dB)"
    )

    return {
        "acquisition_date": acq_date,
        "file_path": str(output_path.name),
        "scene_name": scene_name,
        "burst_id": parsed.get("burst_id", ""),
        "platform": parsed.get("platform", ""),
        "orbit": row.get("orbit", ""),
        "flight_direction": row.get("flight_direction", ""),
        "mean_hh_db": stats.get("mean_hh_db", np.nan),
        "mean_hv_db": stats.get("mean_hv_db", np.nan),
        "std_hh_db": stats.get("std_hh_db", np.nan),
        "std_hv_db": stats.get("std_hv_db", np.nan),
        "hh_hv_ratio_db": stats.get("hh_hv_ratio_db", np.nan),
        "n_valid_pixels": stats["n_valid_pixels"],
        "n_total_pixels": stats["n_total_pixels"],
        "shape": f"{stats['shape_rows']}x{stats['shape_cols']}",
    }


def load_or_create_index(station):
    """Load existing index or return empty DataFrame."""
    index_path = get_s1_index_path(station)
    if index_path.exists():
        return pd.read_csv(index_path)
    return pd.DataFrame()


def save_index(station, df):
    """Save index CSV."""
    index_path = get_s1_index_path(station)
    df.to_csv(index_path, index=False)
    logger.info(f"Index saved: {index_path} ({len(df)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Download and crop OPERA RTC-S1 for GNSS-IR")
    parser.add_argument("--station", required=True, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--max-scenes", type=int, help="Limit number of scenes to download")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-cropped scenes")
    parser.add_argument(
        "--priority-months",
        help="Comma-separated months to download first (e.g., 5,6,10,11,12)",
    )
    parser.add_argument(
        "--clip-extent-km",
        type=float,
        default=5.0,
        help="Clip extent in km (default: 5.0)",
    )
    parser.add_argument("--temp-dir", help="Temp directory for downloads (default: system tmp)")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    station = args.station

    # Load catalog
    catalog_path = get_s1_catalog_path(station)
    if not catalog_path.exists():
        logger.error(f"No catalog found at {catalog_path}. Run s1_fresnel_search.py first.")
        sys.exit(1)

    catalog = pd.read_csv(catalog_path)
    catalog["station"] = station
    logger.info(f"Loaded catalog: {len(catalog)} scenes")

    # Compute bbox
    half_extent = args.clip_extent_km * 1000 / 2
    bbox = compute_fresnel_bbox(station, buffer_m=half_extent)
    logger.info(
        f"Clip extent: {args.clip_extent_km}km "
        f"({bbox['lat_min']:.4f}-{bbox['lat_max']:.4f}N, "
        f"{bbox['lon_min']:.4f}-{bbox['lon_max']:.4f}E)"
    )

    # Load existing index for skip-existing
    existing_index = load_or_create_index(station)
    existing_scenes = set(existing_index["scene_name"].tolist()) if len(existing_index) > 0 else set()

    # Filter catalog
    if args.skip_existing and existing_scenes:
        before = len(catalog)
        catalog = catalog[~catalog["scene_name"].isin(existing_scenes)].reset_index(drop=True)
        logger.info(f"Skipping {before - len(catalog)} existing scenes, {len(catalog)} remaining")

    if catalog.empty:
        logger.info("Nothing to download — all scenes already exist")
        return

    # Priority ordering
    if args.priority_months:
        priority = [int(m) for m in args.priority_months.split(",")]
        catalog["_month"] = pd.to_datetime(catalog["acquisition_date"]).dt.month
        catalog["_priority"] = catalog["_month"].isin(priority).astype(int) * -1  # -1 = first
        catalog = catalog.sort_values(["_priority", "acquisition_date"]).reset_index(drop=True)
        catalog = catalog.drop(columns=["_month", "_priority"])
        logger.info(f"Priority months: {priority}")

    # Limit
    if args.max_scenes:
        catalog = catalog.head(args.max_scenes)
        logger.info(f"Limited to {args.max_scenes} scenes")

    # Output directory
    output_dir = get_s1_fresnel_dir(station)
    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix=f"s1_{station}_")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Temp: {temp_dir}")

    # Download loop
    new_rows = []
    for i, (_, row) in enumerate(catalog.iterrows(), 1):
        logger.info(f"[{i}/{len(catalog)}] {row['acquisition_date']} — {row['scene_name']}")
        result = download_and_crop_scene(row, bbox, output_dir, temp_dir)
        if result:
            new_rows.append(result)

            # Incremental index save every 10 scenes
            if len(new_rows) % 10 == 0:
                merged = pd.concat(
                    [existing_index, pd.DataFrame(new_rows)], ignore_index=True
                ).drop_duplicates(subset="scene_name")
                save_index(station, merged)

    # Final index save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        merged = pd.concat(
            [existing_index, new_df], ignore_index=True
        ).drop_duplicates(subset="scene_name")
        merged = merged.sort_values("acquisition_date").reset_index(drop=True)
        save_index(station, merged)

    # Summary
    total = len(catalog)
    success = len(new_rows)
    failed = total - success
    logger.info("=" * 60)
    logger.info(f"Download complete: {success}/{total} succeeded, {failed} failed")
    if new_rows:
        hh_vals = [r["mean_hh_db"] for r in new_rows if not np.isnan(r["mean_hh_db"])]
        if hh_vals:
            logger.info(f"Mean HH range: [{min(hh_vals):.1f}, {max(hh_vals):.1f}] dB")

    # Clean up temp dir if empty
    try:
        Path(temp_dir).rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
