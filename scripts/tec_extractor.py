#!/usr/bin/env python3
"""Extract TEC values at GNSS-IR station locations from IGS IONEX files.

Downloads IGS Global Ionospheric Maps (GIM) from CDDIS and extracts
Total Electron Content interpolated to station lat/lon.

Usage:
    python scripts/tec_extractor.py --station ROSS --year 2024
    python scripts/tec_extractor.py --station ROSS --year 2024 --skip_download
"""

import argparse
import gzip
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.external_data import load_station_coords, get_cache_dir, get_output_path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CDDIS IONEX base URL (requires Earthdata auth via .netrc)
CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"


# ---------------------------------------------------------------------------
# IONEX parser
# ---------------------------------------------------------------------------

def parse_ionex(filepath):
    """Parse an IONEX file into a list of TEC maps.

    Returns list of dicts:
        {epoch: datetime, lat_grid: array, lon_grid: array, tec: 2D array (TECU)}
    """
    fpath = str(filepath)
    if fpath.endswith(".Z"):
        # Unix compress — decompress with subprocess
        import subprocess
        import tempfile
        result = subprocess.run(["gzip", "-dc", fpath], capture_output=True)
        if result.returncode != 0:
            # Try unlzw or zcat
            result = subprocess.run(["zcat", fpath], capture_output=True)
        if result.returncode != 0:
            logger.warning(f"Cannot decompress {fpath}")
            return []
        import io
        opener = io.StringIO(result.stdout.decode("ascii", errors="ignore"))
    elif fpath.endswith(".gz"):
        opener = gzip.open(filepath, "rt")
    else:
        opener = open(filepath, "r")

    maps = []
    current_map = None
    lat_grid = None
    lon_grid = None
    hgt = None

    with opener as f:
        in_map = False
        lat_rows = []
        current_lat = None

        for line in f:
            label = line[60:].strip() if len(line) > 60 else ""

            if "LAT1 / LAT2 / DLAT" in label:
                parts = line[:60].split()
                lat1, lat2, dlat = float(parts[0]), float(parts[1]), float(parts[2])
                lat_grid = np.arange(lat1, lat2 + dlat/2, dlat)

            elif "LON1 / LON2 / DLON" in label:
                parts = line[:60].split()
                lon1, lon2, dlon = float(parts[0]), float(parts[1]), float(parts[2])
                lon_grid = np.arange(lon1, lon2 + dlon/2, dlon)

            elif "HGT1 / HGT2 / DHGT" in label:
                parts = line[:60].split()
                hgt = float(parts[0])

            elif "START OF TEC MAP" in label:
                in_map = True
                lat_rows = []
                current_lat = None

            elif "EPOCH OF CURRENT MAP" in label:
                parts = line[:60].split()
                try:
                    yr, mo, dy, hr, mi, se = [int(x) for x in parts[:6]]
                    hr = min(hr, 23)  # Clamp hour=24 to 23
                    epoch = datetime(yr, mo, dy, hr, mi, se)
                except (ValueError, IndexError):
                    pass

            elif "LAT/LON1/LON2/DLON/H" in label and in_map:
                # Start of a new latitude row
                # Fixed-width: cols 2-8=lat, 8-14=lon1, 14-20=lon2, 20-26=dlon, 26-32=h
                try:
                    current_lat = float(line[2:8])
                except ValueError:
                    current_lat = None
                lat_rows.append([])

            elif "END OF TEC MAP" in label:
                in_map = False
                if lat_rows and lon_grid is not None:
                    # Build 2D TEC array
                    tec_2d = np.array([row for row in lat_rows if len(row) == len(lon_grid)])
                    if tec_2d.shape[0] == len(lat_grid):
                        maps.append({
                            "epoch": epoch,
                            "lat_grid": lat_grid.copy(),
                            "lon_grid": lon_grid.copy(),
                            "tec": tec_2d * 0.1,  # IONEX stores in 0.1 TECU
                        })

            elif in_map and current_lat is not None:
                # TEC data line — integers only, no label keywords
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip lines that contain known IONEX labels
                if any(kw in line for kw in ["MAP", "EPOCH", "LAT", "LON",
                                              "END OF", "START OF", "COMMENT",
                                              "DESCRIPTION"]):
                    continue
                try:
                    vals = [int(v) for v in stripped.split()]
                    lat_rows[-1].extend(vals)
                except ValueError:
                    pass

    return maps


def interpolate_tec(maps, station_lat, station_lon):
    """Bilinear interpolation of TEC at station location for each epoch.

    Returns DataFrame with columns: datetime, tec_tecu, tec_rate
    """
    rows = []
    prev_tec = None

    for m in maps:
        lat_grid = m["lat_grid"]
        lon_grid = m["lon_grid"]
        tec = m["tec"]

        # Handle longitude wrapping (IONEX uses -180 to 180)
        slon = station_lon
        if slon > 180:
            slon -= 360

        # Find bounding indices
        lat_idx = np.searchsorted(lat_grid, station_lat) - 1
        lon_idx = np.searchsorted(lon_grid, slon) - 1

        lat_idx = np.clip(lat_idx, 0, len(lat_grid) - 2)
        lon_idx = np.clip(lon_idx, 0, len(lon_grid) - 2)

        # Bilinear interpolation
        lat_frac = (station_lat - lat_grid[lat_idx]) / (lat_grid[lat_idx + 1] - lat_grid[lat_idx])
        lon_frac = (slon - lon_grid[lon_idx]) / (lon_grid[lon_idx + 1] - lon_grid[lon_idx])

        lat_frac = np.clip(lat_frac, 0, 1)
        lon_frac = np.clip(lon_frac, 0, 1)

        t00 = tec[lat_idx, lon_idx]
        t01 = tec[lat_idx, lon_idx + 1]
        t10 = tec[lat_idx + 1, lon_idx]
        t11 = tec[lat_idx + 1, lon_idx + 1]

        tec_val = (t00 * (1 - lat_frac) * (1 - lon_frac) +
                   t01 * (1 - lat_frac) * lon_frac +
                   t10 * lat_frac * (1 - lon_frac) +
                   t11 * lat_frac * lon_frac)

        # TEC rate (TECU/hr)
        tec_rate = np.nan
        if prev_tec is not None:
            dt_hr = (m["epoch"] - prev_epoch).total_seconds() / 3600
            if dt_hr > 0:
                tec_rate = (tec_val - prev_tec) / dt_hr

        rows.append({
            "datetime": m["epoch"],
            "tec_tecu": float(tec_val),
            "tec_rate": float(tec_rate) if not np.isnan(tec_rate) else np.nan,
        })

        prev_tec = tec_val
        prev_epoch = m["epoch"]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get_cddis_session():
    """Create an authenticated session for CDDIS (Earthdata OAuth)."""
    import requests
    from netrc import netrc

    session = requests.Session()
    try:
        nrc = netrc()
        login, _, password = nrc.authenticators("urs.earthdata.nasa.gov")
        session.auth = (login, password)
    except Exception as e:
        logger.warning(f"Cannot read .netrc for Earthdata: {e}")
    return session


def download_ionex(year, doy, cache_dir):
    """Download IGS final IONEX file for a given day.

    Tries multiple analysis centers: IGS combined (igsg), then CODE (codg).
    Uses Earthdata OAuth redirect for CDDIS authentication.
    Returns path to downloaded file or None.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    doy_str = f"{doy:03d}"
    yy = str(year)[-2:]

    # Check if already cached (uncompressed)
    # Check cache with all known prefixes
    all_prefixes = ["casg", "uqrg", "igsg", "codg", "jplg", "c1pg", "carg"]
    for prefix in all_prefixes:
        for ext in [f".{yy}i", f".{yy}i.Z", f".{yy}i.gz"]:
            cached = cache_dir / f"{prefix}{doy_str}0{ext}"
            if cached.exists() and cached.stat().st_size > 1000:
                return cached

    session = _get_cddis_session()

    for prefix in all_prefixes:
        for ext in [".Z", ".gz"]:
            fname = f"{prefix}{doy_str}0.{yy}i{ext}"
            url = f"{CDDIS_BASE}/{year}/{doy_str}/{fname}"
            local = cache_dir / fname

            try:
                resp = session.get(url, allow_redirects=True, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    # Verify it's not an HTML error page
                    if resp.content[:5] != b"<!DOC":
                        local.write_bytes(resp.content)
                        logger.debug(f"Downloaded {fname}")
                        return local
            except Exception:
                continue

    return None


def extract_tec_for_year(station, year, cache_dir=None,
                          skip_download=False):
    """Extract daily TEC time series for a station.

    Returns DataFrame with columns: date, doy, tec_daily_mean, tec_daily_max,
    tec_daily_std, tec_rate_max
    """
    coords = load_station_coords(station)
    station_lat, station_lon = coords["lat"], coords["lon"]
    if cache_dir is None:
        cache_dir = str(get_cache_dir("tec"))

    all_daily = []

    for doy in range(1, 366):
        if skip_download:
            # Look for any cached IONEX file for this DOY
            from glob import glob
            pattern = str(Path(cache_dir) / f"*{doy:03d}0.{str(year)[-2:]}i*")
            files = glob(pattern)
            ionex_path = files[0] if files else None
        else:
            ionex_path = download_ionex(year, doy, cache_dir)

        if ionex_path is None:
            continue

        try:
            maps = parse_ionex(ionex_path)
            if not maps:
                continue

            tec_df = interpolate_tec(maps, station_lat, station_lon)
            if tec_df.empty:
                continue

            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            all_daily.append({
                "date": date.strftime("%Y-%m-%d"),
                "doy": doy,
                "tec_daily_mean": tec_df["tec_tecu"].mean(),
                "tec_daily_max": tec_df["tec_tecu"].max(),
                "tec_daily_min": tec_df["tec_tecu"].min(),
                "tec_daily_std": tec_df["tec_tecu"].std(),
                "tec_rate_max": tec_df["tec_rate"].abs().max(),
                "tec_noon": tec_df.iloc[len(tec_df) // 2]["tec_tecu"],
            })

        except Exception as e:
            logger.warning(f"Error processing DOY {doy}: {e}")

    if not all_daily:
        return pd.DataFrame()

    return pd.DataFrame(all_daily)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract TEC at GNSS-IR station")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--cache_dir", help="Override cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info(f"Extracting TEC for {args.station} {args.year}")

    tec_df = extract_tec_for_year(
        args.station, args.year,
        cache_dir=args.cache_dir,
        skip_download=args.skip_download,
    )

    if tec_df.empty:
        logger.error("No TEC data extracted")
        return

    out_path = get_output_path(args.station, args.year, "tec.parquet")
    tec_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(tec_df)} days)")

    print(f"\nTEC Summary: {args.station} {args.year}")
    print(f"  Days extracted: {len(tec_df)}")
    print(f"  TEC mean: {tec_df['tec_daily_mean'].mean():.1f} TECU")
    print(f"  TEC range: {tec_df['tec_daily_min'].min():.1f} - {tec_df['tec_daily_max'].max():.1f} TECU")
    print(f"  Max TEC rate: {tec_df['tec_rate_max'].max():.2f} TECU/hr")


if __name__ == "__main__":
    main()
