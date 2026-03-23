# ABOUTME: Render S1 backscatter thumbnails for gallery/mosaic view
# ABOUTME: Produces consistent PNGs with water mask, station marker, date label

"""
Render S1 scene thumbnails for gallery viewing.

Usage:
    python scripts/s1_render_thumbnails.py --station UMNQ
    python scripts/s1_render_thumbnails.py --station NKAR --force
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.s1_fresnel_utils import (
    get_s1_fresnel_dir,
    get_s1_index_path,
    get_station_s1_config,
    _load_gsw_water_mask,
)

logger = logging.getLogger(__name__)


def render_thumbnail(tif_path, station, output_path, date_label="", hh_range=(-25, 0),
                     fresnel_inner_m=None, fresnel_outer_m=None):
    """Render a single S1 scene as a thumbnail PNG.

    Shows the Fresnel annulus (green arcs) when radii are provided,
    distinguishing the actual GNSS-IR sensing zone from regional context.
    """
    config = get_station_s1_config(station)
    az_range = config.get("azval2", [0, 360])

    with rasterio.open(tif_path) as src:
        hh_db = src.read(1)
        s1_transform = src.transform
        s1_crs = src.crs

    # Station pixel coords
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", s1_crs, always_xy=True)
    cx, cy = t.transform(config["lon"], config["lat"])
    sta_col = (cx - s1_transform.c) / s1_transform.a
    sta_row = (cy - s1_transform.f) / s1_transform.e

    # Water mask
    water_mask = _load_gsw_water_mask(station, s1_crs, s1_transform, hh_db.shape)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    # S1 image
    ax.imshow(hh_db, cmap="gray", vmin=hh_range[0], vmax=hh_range[1], origin="upper")

    # Land overlay
    if water_mask is not None:
        land_ov = np.zeros((*hh_db.shape, 4))
        land_ov[~water_mask] = [1, 0.2, 0.2, 0.12]
        ax.imshow(land_ov, origin="upper")

    # Station marker
    ax.plot(sta_col, sta_row, "r^", markersize=6, markeredgecolor="white", markeredgewidth=1, zorder=10)

    px_per_m = 1.0 / abs(s1_transform.a)

    # Azimuth range lines
    fan_r = 200 * px_per_m
    for i in range(0, len(az_range), 2):
        for az_val in [az_range[i], az_range[i + 1]]:
            ddx = fan_r * np.sin(np.radians(az_val))
            ddy = -fan_r * np.cos(np.radians(az_val))
            ax.plot([sta_col, sta_col + ddx], [sta_row, sta_row + ddy],
                    "-", color="cyan", linewidth=0.6, alpha=0.5)

    # Fresnel annulus arcs (green)
    if fresnel_inner_m is not None and fresnel_outer_m is not None:
        angles = np.linspace(0, 2 * np.pi, 200)
        for radius_m, ls, lw in [(fresnel_inner_m, "--", 0.6), (fresnel_outer_m, "-", 0.8)]:
            r_px = radius_m * px_per_m
            arc_x = sta_col + r_px * np.sin(angles)
            arc_y = sta_row - r_px * np.cos(angles)
            ax.plot(arc_x, arc_y, ls, color="lime", linewidth=lw, alpha=0.6)

    # Date label
    ax.text(0.02, 0.98, date_label, transform=ax.transAxes, fontsize=8, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7))

    # Compute stats for both scales
    if water_mask is not None:
        rows_grid, cols_grid = np.meshgrid(
            np.arange(hh_db.shape[0]), np.arange(hh_db.shape[1]), indexing="ij"
        )
        x_coords = s1_transform.c + (cols_grid + 0.5) * s1_transform.a
        y_coords = s1_transform.f + (rows_grid + 0.5) * s1_transform.e
        dx = x_coords - cx
        dy = y_coords - cy
        dist_m = np.sqrt(dx**2 + dy**2)
        az_deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

        az_mask = np.zeros_like(az_deg, dtype=bool)
        for i in range(0, len(az_range), 2):
            az_min, az_max = az_range[i], az_range[i + 1]
            if az_min < az_max:
                az_mask |= (az_deg >= az_min) & (az_deg <= az_max)
            else:
                az_mask |= (az_deg >= az_min) | (az_deg <= az_max)

        # Regional: water in azimuth
        regional_valid = ~np.isnan(hh_db) & water_mask & az_mask
        # Fresnel: water in azimuth AND within annulus
        if fresnel_inner_m is not None and fresnel_outer_m is not None:
            fresnel_valid = regional_valid & (dist_m >= fresnel_inner_m) & (dist_m <= fresnel_outer_m)
        else:
            fresnel_valid = None

        # Annotate: top-right shows regional, bottom-right shows Fresnel
        if np.any(regional_valid):
            mean_hh_reg = float(np.nanmean(hh_db[regional_valid]))
            ax.text(0.98, 0.98, f"R:{mean_hh_reg:.0f}", transform=ax.transAxes, fontsize=6,
                    color="white", va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#1565c0", alpha=0.7))

        if fresnel_valid is not None and np.any(fresnel_valid):
            mean_hh_fz = float(np.nanmean(hh_db[fresnel_valid]))
            ax.text(0.98, 0.86, f"F:{mean_hh_fz:.0f}", transform=ax.transAxes, fontsize=6,
                    color="white", va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#2e7d32", alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _compute_fresnel_radii(station):
    """Compute Fresnel zone radii from per-arc data or config."""
    # Try per-arc data (any year)
    results_dir = PROJECT_ROOT / "results_annual" / station
    if results_dir.exists():
        pa_files = sorted(results_dir.glob(f"{station}_*_per_arc.parquet"))
        if pa_files:
            pa = pd.read_parquet(pa_files[-1])  # most recent year
            elev_mid = (pa["eminO"] + pa["emaxO"]) / 2.0
            refl_dist = pa["RH"] / np.tan(np.radians(elev_mid))
            inner = max(0, float(refl_dist.quantile(0.02)) - 30)
            outer = float(refl_dist.quantile(0.98)) + 30
            logger.info(f"Fresnel radii from per-arc: {inner:.0f}-{outer:.0f}m")
            return inner, outer

    # Config fallback
    import json
    cfg_path = PROJECT_ROOT / "config" / f"{station.lower()}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        inner = max(0, cfg["minH"] / np.tan(np.radians(cfg["e2"])) - 30)
        outer = cfg["maxH"] / np.tan(np.radians(cfg["e1"])) + 30
        logger.info(f"Fresnel radii from config: {inner:.0f}-{outer:.0f}m")
        return inner, outer

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Render S1 scene thumbnails")
    parser.add_argument("--station", required=True, help="Station ID")
    parser.add_argument("--force", action="store_true", help="Re-render existing thumbnails")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    station = args.station
    idx_path = get_s1_index_path(station)
    if not idx_path.exists():
        logger.error(f"No index found: {idx_path}")
        sys.exit(1)

    idx = pd.read_csv(idx_path)
    fresnel_dir = get_s1_fresnel_dir(station)
    thumb_dir = fresnel_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)

    # Compute Fresnel radii once for all thumbnails
    fresnel_inner, fresnel_outer = _compute_fresnel_radii(station)

    rendered = 0
    skipped = 0
    for _, row in idx.iterrows():
        tif_path = fresnel_dir / row["file_path"]
        date_str = row["acquisition_date"]
        thumb_path = thumb_dir / f"{station}_{date_str.replace('-', '')}_thumb.png"

        if thumb_path.exists() and not args.force:
            skipped += 1
            continue

        if not tif_path.exists():
            continue

        render_thumbnail(tif_path, station, thumb_path, date_label=date_str,
                         fresnel_inner_m=fresnel_inner, fresnel_outer_m=fresnel_outer)
        rendered += 1
        if rendered % 20 == 0:
            logger.info(f"  Rendered {rendered}...")

    logger.info(f"Done: {rendered} rendered, {skipped} skipped, saved to {thumb_dir}")


if __name__ == "__main__":
    main()
