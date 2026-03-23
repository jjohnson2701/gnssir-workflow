# ABOUTME: Search ASF for OPERA RTC-S1 products covering a GNSS-IR station
# ABOUTME: Outputs a catalog CSV with scene metadata and HH/HV download URLs

"""
Search ASF for OPERA RTC-S1 scenes at a GNSS-IR station location.

Usage:
    python scripts/s1_fresnel_search.py --station UMNQ
    python scripts/s1_fresnel_search.py --station UMNQ --start-date 2025-01-01 --end-date 2025-12-31
    python scripts/s1_fresnel_search.py --station NKAR --start-date 2024-06-01 --end-date 2025-03-22
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.s1_fresnel_utils import (
    get_s1_catalog_path,
    get_station_s1_config,
    parse_opera_rtc_scene_name,
)

logger = logging.getLogger(__name__)


def search_opera_rtc_s1(station, start_date, end_date):
    """Search ASF for OPERA RTC-S1 products at station location.

    Returns a DataFrame with one row per scene.
    """
    import asf_search as asf

    config = get_station_s1_config(station)
    lat, lon = config["lat"], config["lon"]

    logger.info(f"Searching ASF for OPERA RTC-S1 at {station} ({lat:.4f}, {lon:.4f})")
    logger.info(f"Date range: {start_date} to {end_date}")

    # ASF uses exclusive end date — add 1 day
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_padded = end_dt.strftime("%Y-%m-%d")

    results = asf.search(
        platform="SENTINEL-1",
        processingLevel="RTC",
        intersectsWith=f"POINT({lon} {lat})",
        start=start_date,
        end=end_padded,
        maxResults=5000,
    )

    logger.info(f"ASF returned {len(results)} results")

    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        props = r.properties
        scene_name = props.get("sceneName", "")
        parsed = parse_opera_rtc_scene_name(scene_name)

        # Collect all candidate URLs: primary URL + additionalUrls
        primary_url = props.get("url", "")
        additional = props.get("additionalUrls", [])
        all_urls = [primary_url] + additional

        # Extract HH/VV and HV/VH URLs from all candidates
        hh_url = next((u for u in all_urls if "_HH.tif" in u), None)
        hv_url = next((u for u in all_urls if "_HV.tif" in u), None)

        # Also check for VV/VH (lower latitude stations)
        if not hh_url:
            hh_url = next((u for u in all_urls if "_VV.tif" in u), None)
        if not hv_url:
            hv_url = next((u for u in all_urls if "_VH.tif" in u), None)

        # Get file sizes from bytes dict
        bytes_info = props.get("bytes", {})
        hh_bytes = 0
        hv_bytes = 0
        for fname, info in bytes_info.items():
            if "_HH.tif" in fname or "_VV.tif" in fname:
                hh_bytes = info.get("bytes", 0)
            elif "_HV.tif" in fname or "_VH.tif" in fname:
                hv_bytes = info.get("bytes", 0)

        # Determine polarization type
        pol = props.get("polarization", [])
        pol_str = "+".join(pol) if isinstance(pol, list) else str(pol)

        rows.append(
            {
                "scene_name": scene_name,
                "acquisition_date": parsed.get("acquisition_date", props.get("startTime", "")[:10]),
                "burst_id": parsed.get("burst_id", ""),
                "platform": parsed.get("platform", props.get("platform", "")),
                "polarization": pol_str,
                "orbit": props.get("orbit", ""),
                "flight_direction": props.get("flightDirection", ""),
                "hh_url": hh_url or "",
                "hv_url": hv_url or "",
                "hh_bytes": hh_bytes,
                "hv_bytes": hv_bytes,
                "primary_url": props.get("url", ""),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("acquisition_date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Search ASF for OPERA RTC-S1 at a GNSS-IR station")
    parser.add_argument("--station", required=True, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--append", action="store_true", help="Merge with existing catalog")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    start_date = args.start_date or "2024-01-01"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    df = search_opera_rtc_s1(args.station, start_date, end_date)

    if df.empty:
        logger.error("No results found")
        sys.exit(1)

    # Merge with existing catalog if --append
    catalog_path = get_s1_catalog_path(args.station)
    if args.append and catalog_path.exists():
        existing = pd.read_csv(catalog_path)
        df = pd.concat([existing, df]).drop_duplicates(subset="scene_name").reset_index(drop=True)
        df = df.sort_values("acquisition_date").reset_index(drop=True)
        logger.info(f"Merged with existing catalog: {len(existing)} existing + new = {len(df)} total")

    df.to_csv(catalog_path, index=False)
    logger.info(f"Saved catalog: {catalog_path}")

    # Summary
    unique_dates = df["acquisition_date"].nunique()
    unique_bursts = df["burst_id"].nunique()
    total_mb = (df["hh_bytes"].sum() + df["hv_bytes"].sum()) / 1024 / 1024
    logger.info(f"Summary for {args.station}:")
    logger.info(f"  Total scenes: {len(df)}")
    logger.info(f"  Unique dates: {unique_dates}")
    logger.info(f"  Unique burst IDs: {unique_bursts}")
    logger.info(f"  Date range: {df['acquisition_date'].min()} to {df['acquisition_date'].max()}")
    logger.info(f"  Platforms: {df['platform'].value_counts().to_dict()}")
    logger.info(f"  Polarizations: {df['polarization'].unique().tolist()}")
    logger.info(f"  Total download size (HH+HV): {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
