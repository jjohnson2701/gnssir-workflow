#!/usr/bin/env python3
"""Extract ERA5 meteorological variables at GNSS-IR station locations.

Downloads ERA5 reanalysis from CDS API and extracts daily statistics
at the nearest grid point to each station.

Usage:
    python scripts/era5_extractor.py --station ROSS --year 2024
    python scripts/era5_extractor.py --station ROSS --year 2024 --skip_download
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.external_data import load_station_coords, get_cache_dir, get_output_path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ERA5 variables to download
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "snow_depth",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
]

ERA5_SHORT = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "snow_depth": "sd",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_pressure": "sp",
    "volumetric_soil_water_layer_1": "swvl1",
    "soil_temperature_level_1": "stl1",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_era5(station_lat, station_lon, year, cache_dir):
    """Download ERA5 hourly data for a single grid point via CDS API.

    Downloads the nearest 0.25° grid cell for the full year.
    Returns path to downloaded NetCDF file.
    """
    import cdsapi

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Snap to nearest 0.25° grid point
    lat_snap = round(station_lat * 4) / 4
    lon_snap = round(station_lon * 4) / 4

    # Small bounding box around the point (CDS needs area as [N, W, S, E])
    area = [lat_snap + 0.125, lon_snap - 0.125,
            lat_snap - 0.125, lon_snap + 0.125]

    out_path = cache_dir / f"era5_{lat_snap:.2f}_{lon_snap:.2f}_{year}.nc"
    if out_path.exists():
        logger.info(f"ERA5 cached: {out_path}")
        return out_path

    logger.info(f"Downloading ERA5 for ({lat_snap}, {lon_snap}), {year}")

    c = cdsapi.Client()

    # Download month by month to stay within CDS cost limits
    monthly_files = []
    for month in range(1, 13):
        month_path = cache_dir / f"era5_{lat_snap:.2f}_{lon_snap:.2f}_{year}_{month:02d}.nc"
        if month_path.exists():
            monthly_files.append(month_path)
            continue

        logger.info(f"  Downloading {year}-{month:02d}")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ERA5_VARIABLES,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(0, 24)],
                    "area": area,
                    "format": "netcdf",
                },
                str(month_path),
            )
            monthly_files.append(month_path)
        except Exception as e:
            logger.warning(f"  Failed {year}-{month:02d}: {e}")

    if not monthly_files:
        logger.error("No ERA5 monthly files downloaded")
        return None

    # Handle CDS zip format: some months come as zip with instant + accumulated files
    import zipfile
    resolved_files = []
    for mf in monthly_files:
        mf = Path(mf)
        # Check if it's a zip
        if zipfile.is_zipfile(str(mf)):
            logger.info(f"  Unzipping {mf.name}")
            extract_dir = mf.parent / f"tmp_{mf.stem}"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(str(mf), "r") as zf:
                zf.extractall(str(extract_dir))
            # Find all .nc files inside
            nc_files = list(extract_dir.glob("*.nc"))
            if nc_files:
                # Merge instant + accumulated into one dataset
                import xarray as xr
                parts = [xr.open_dataset(f) for f in nc_files]
                combined = xr.merge(parts)
                merged_path = mf.parent / f"{mf.stem}_merged.nc"
                combined.to_netcdf(str(merged_path))
                for p in parts:
                    p.close()
                combined.close()
                resolved_files.append(merged_path)
                # Clean up
                import shutil
                shutil.rmtree(str(extract_dir))
            else:
                resolved_files.append(mf)
        else:
            resolved_files.append(mf)

    # Merge monthly files into one annual file
    import xarray as xr
    datasets = [xr.open_dataset(str(f)) for f in resolved_files]
    time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
    merged = xr.concat(datasets, dim=time_dim)
    merged.to_netcdf(str(out_path))
    for ds in datasets:
        ds.close()
    merged.close()

    logger.info(f"Merged {len(resolved_files)} months → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_daily_era5(nc_path, station_lat, station_lon):
    """Extract daily statistics from ERA5 NetCDF at nearest grid point.

    Returns DataFrame with daily aggregated values.
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Find nearest grid point
    if "latitude" in ds.dims:
        lat_dim, lon_dim = "latitude", "longitude"
    else:
        lat_dim, lon_dim = "lat", "lon"

    ds_point = ds.sel({lat_dim: station_lat, lon_dim: station_lon},
                      method="nearest")

    # Compute daily statistics
    time_var = "valid_time" if "valid_time" in ds_point.dims else "time"
    daily = ds_point.resample({time_var: "1D"})

    rows = []
    dates = []

    # Get the time coordinate values
    for date, grp in daily:
        date_val = pd.Timestamp(date).date()
        dates.append(date_val)

        row = {"date": str(date_val), "doy": date_val.timetuple().tm_yday}

        for long_name, short in ERA5_SHORT.items():
            if short in grp:
                vals = grp[short].values
                vals = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else np.array([vals])

                if len(vals) == 0:
                    continue

                if short == "t2m" or short == "stl1":
                    # Temperature: convert K to °C, compute min/mean/max
                    vals_c = vals - 273.15
                    row[f"{short}_mean"] = float(np.mean(vals_c))
                    row[f"{short}_min"] = float(np.min(vals_c))
                    row[f"{short}_max"] = float(np.max(vals_c))
                elif short == "tp":
                    # Precipitation: hourly accumulations, sum to daily total (m)
                    row["precip_mm"] = float(np.sum(vals) * 1000)
                elif short == "sd":
                    # Snow depth: daily mean (m)
                    row["snow_depth_m"] = float(np.mean(vals))
                elif short in ("u10", "v10"):
                    row[f"{short}_mean"] = float(np.mean(vals))
                elif short == "sp":
                    # Surface pressure: convert Pa to hPa
                    row["sp_hpa"] = float(np.mean(vals)) / 100
                elif short == "swvl1":
                    row["soil_moisture"] = float(np.mean(vals))

        # Compute wind speed from u10, v10
        if "u10_mean" in row and "v10_mean" in row:
            row["wind_speed_ms"] = float(np.sqrt(row["u10_mean"]**2 + row["v10_mean"]**2))

        rows.append(row)

    ds.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Drop intermediate u10, v10 columns
    df = df.drop(columns=["u10_mean", "v10_mean"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract ERA5 at GNSS-IR station")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--cache_dir", help="Override cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    coords = load_station_coords(args.station)
    station_lat, station_lon = coords["lat"], coords["lon"]
    cache_dir = args.cache_dir or str(get_cache_dir("era5"))

    logger.info(f"ERA5 extraction: {args.station} ({station_lat:.3f}, {station_lon:.3f}), {args.year}")

    if args.skip_download:
        import glob
        files = glob.glob(str(Path(cache_dir) / f"era5_*_{args.year}.nc"))
        if not files:
            logger.error("No cached ERA5 files found")
            return
        nc_path = files[0]
    else:
        nc_path = download_era5(station_lat, station_lon, args.year, cache_dir)

    era5_df = extract_daily_era5(nc_path, station_lat, station_lon)

    if era5_df.empty:
        logger.error("No ERA5 data extracted")
        return

    out_path = get_output_path(args.station, args.year, "era5.parquet")
    era5_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(era5_df)} days)")

    print(f"\nERA5 Summary: {args.station} {args.year}")
    print(f"  Days: {len(era5_df)}")
    for col in ["t2m_mean", "t2m_min", "t2m_max", "precip_mm", "snow_depth_m",
                 "wind_speed_ms", "soil_moisture", "sp_hpa"]:
        if col in era5_df.columns:
            print(f"  {col}: {era5_df[col].mean():.2f} (mean), "
                  f"[{era5_df[col].min():.2f}, {era5_df[col].max():.2f}]")


if __name__ == "__main__":
    main()
