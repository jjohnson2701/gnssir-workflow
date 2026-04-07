#!/usr/bin/env python3
"""Download CYGNSS L2 data near GNSS-IR stations and compare surface reflectivity.

Extracts CYGNSS specular-point observations within a radius of a ground station,
then compares spaceborne Fresnel reflectivity and NBRCS with ground-based SNR
features (AF, CLR, gamma, phase).

Usage:
    python scripts/cygnss_comparison.py --station FORA --year 2024
    python scripts/cygnss_comparison.py --station FORA --year 2024 --skip_download
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.utils.external_data import load_station_coords, get_cache_dir, get_results_dir

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_station_lat_lon(station):
    """Return (lat, lon) tuple for a station."""
    coords = load_station_coords(station)
    return coords["lat"], coords["lon"]


# ---------------------------------------------------------------------------
# CYGNSS download
# ---------------------------------------------------------------------------

def download_cygnss_l2(year, doy_start, doy_end, cache_dir):
    """Download CYGNSS L2 V3.2 daily files via earthaccess.

    Returns list of downloaded file paths.
    """
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        logger.error("Earthdata authentication failed. Check ~/.netrc")
        sys.exit(1)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(year, 1, 1) + timedelta(days=doy_start - 1)
    end_date = datetime(year, 1, 1) + timedelta(days=doy_end - 1)

    logger.info(f"Searching CYGNSS L2 V3.2: {start_date.date()} to {end_date.date()}")
    results = earthaccess.search_data(
        short_name="CYGNSS_L2_V3.2",
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    logger.info(f"Found {len(results)} granules")

    if not results:
        return []

    # Filter out already-downloaded files
    to_download = []
    existing = []
    for r in results:
        fname = r.data_links()[0].split("/")[-1]
        local_path = cache_dir / fname
        if local_path.exists():
            existing.append(str(local_path))
        else:
            to_download.append(r)

    logger.info(f"Already cached: {len(existing)}, to download: {len(to_download)}")

    downloaded = []
    if to_download:
        downloaded = earthaccess.download(to_download, str(cache_dir))
        downloaded = [str(p) for p in downloaded]

    return existing + downloaded


# ---------------------------------------------------------------------------
# CYGNSS extraction
# ---------------------------------------------------------------------------

def extract_cygnss_near_station(nc_files, station_lat, station_lon,
                                 radius_deg=1.0):
    """Extract CYGNSS specular points within radius of station.

    Returns DataFrame with columns:
        datetime, lat, lon, dist_deg, fresnel_coeff, nbrcs_mean,
        incidence_angle, wind_speed, prn_code, spacecraft_num
    """
    import netCDF4 as nc

    # Convert station lon to 0-360 convention used by CYGNSS
    slon360 = station_lon % 360

    all_rows = []
    for fpath in nc_files:
        try:
            ds = nc.Dataset(fpath)
        except Exception as e:
            logger.warning(f"Failed to open {fpath}: {e}")
            continue

        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]

        # Spatial filter
        mask = (np.abs(lat - station_lat) < radius_deg) & \
               (np.abs(lon - slon360) < radius_deg)

        n_near = mask.sum()
        if n_near == 0:
            ds.close()
            continue

        idx = np.where(mask)[0]

        # Parse file date from filename: cyg.ddmi.sYYYYMMDD-...
        fname = Path(fpath).name
        try:
            date_part = fname.split(".")[2]  # sYYYYMMDD-HHMMSS-eYYYYMMDD-HHMMSS
            file_date_str = date_part[1:9]  # YYYYMMDD after 's'
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
        except (IndexError, ValueError):
            logger.warning(f"Cannot parse date from {fname}")
            ds.close()
            continue

        # sample_time is seconds within the day (0-86400)
        fc = ds.variables["fresnel_coeff"][idx]
        nbrcs = ds.variables["nbrcs_mean"][idx]
        inc = ds.variables["incidence_angle"][idx]
        ws = ds.variables["wind_speed"][idx]
        prn = ds.variables["prn_code"][idx]
        sc = ds.variables["spacecraft_num"][idx]
        times = ds.variables["sample_time"][idx]
        lats = lat[idx]
        lons = lon[idx]

        for i in range(len(idx)):
            def _val(arr, i):
                v = arr[i]
                if np.ma.is_masked(v):
                    return np.nan
                return float(v)

            dt = file_date + timedelta(seconds=float(times[i]))
            dist = np.sqrt((float(lats[i]) - station_lat)**2 +
                           (float(lons[i]) - slon360)**2)
            all_rows.append({
                "datetime": dt,
                "lat": float(lats[i]),
                "lon": float(lons[i]) - 360 if float(lons[i]) > 180 else float(lons[i]),
                "dist_deg": dist,
                "fresnel_coeff": _val(fc, i),
                "nbrcs_mean": _val(nbrcs, i),
                "incidence_angle": _val(inc, i),
                "wind_speed": _val(ws, i),
                "prn_code": int(prn[i]),
                "spacecraft_num": int(sc[i]),
            })

        ds.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = df["datetime"].dt.date
    df["doy"] = df["datetime"].dt.dayofyear
    logger.info(f"Extracted {len(df)} CYGNSS points near station")
    return df


# ---------------------------------------------------------------------------
# Ground-based feature loading
# ---------------------------------------------------------------------------

def load_ground_features(station, year):
    """Load SNR features from arc_table (or legacy snr_features), compute daily medians."""
    results_dir = PROJECT_ROOT / "results_annual" / station
    # Prefer arc_table (4-layer), fall back to snr_features (legacy)
    sf_path = results_dir / f"{station}_{year}_arc_table.parquet"
    if not sf_path.exists():
        sf_path = results_dir / f"{station}_{year}_snr_features.parquet"
    if not sf_path.exists():
        logger.error(f"No SNR features for {station} {year}")
        return None

    sf = pd.read_parquet(sf_path)
    if "CLR" not in sf.columns:
        logger.error(f"No SNR feature columns in {sf_path.name}")
        return None
    logger.info(f"Ground features: {len(sf)} arcs, columns: {list(sf.columns)}")

    # Daily medians of key features
    features = ["CLR", "AF", "PR", "gamma", "phase", "RH", "MS", "VS"]
    features = [f for f in features if f in sf.columns]

    daily = sf.groupby("doy")[features].median().reset_index()
    # Also get arc count per day
    daily["n_arcs"] = sf.groupby("doy").size().values
    return daily


# ---------------------------------------------------------------------------
# Comparison and visualization
# ---------------------------------------------------------------------------

def build_comparison(cygnss_df, ground_df, radius_deg=1.0):
    """Merge CYGNSS daily stats with ground daily medians on DOY."""
    if cygnss_df.empty:
        return pd.DataFrame()

    # Filter by radius
    cygnss_near = cygnss_df[cygnss_df["dist_deg"] <= radius_deg].copy()

    # Daily CYGNSS medians
    cygnss_daily = cygnss_near.groupby("doy").agg(
        fresnel_coeff_median=("fresnel_coeff", "median"),
        fresnel_coeff_std=("fresnel_coeff", "std"),
        nbrcs_median=("nbrcs_mean", "median"),
        nbrcs_std=("nbrcs_mean", "std"),
        wind_speed_median=("wind_speed", "median"),
        inc_angle_median=("incidence_angle", "median"),
        n_cygnss=("fresnel_coeff", "count"),
    ).reset_index()

    # Merge on DOY
    merged = pd.merge(cygnss_daily, ground_df, on="doy", how="inner")
    logger.info(f"Matched {len(merged)} days with both CYGNSS and ground data")
    return merged


def plot_timeseries(merged, station, year, out_dir):
    """Dual-axis time series: CYGNSS reflectivity + ground features."""
    if merged.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    doy = merged["doy"]

    # Panel 1: CYGNSS Fresnel coeff + ground CLR
    ax1 = axes[0]
    ax1.plot(doy, merged["fresnel_coeff_median"], "b.-", label="CYGNSS Fresnel coeff", alpha=0.7)
    ax1.fill_between(doy,
                     merged["fresnel_coeff_median"] - merged["fresnel_coeff_std"],
                     merged["fresnel_coeff_median"] + merged["fresnel_coeff_std"],
                     alpha=0.15, color="blue")
    ax1.set_ylabel("CYGNSS Fresnel Coeff", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    if "CLR" in merged.columns:
        ax1r = ax1.twinx()
        ax1r.plot(doy, merged["CLR"], "r.-", label="Ground CLR", alpha=0.7)
        ax1r.set_ylabel("Ground CLR", color="red")
        ax1r.tick_params(axis="y", labelcolor="red")
    ax1.set_title(f"{station} {year}: CYGNSS vs Ground-Based GNSS-IR")

    # Panel 2: CYGNSS NBRCS + ground AF
    ax2 = axes[1]
    ax2.plot(doy, merged["nbrcs_median"], "b.-", label="CYGNSS NBRCS", alpha=0.7)
    ax2.set_ylabel("CYGNSS NBRCS", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    if "AF" in merged.columns:
        ax2r = ax2.twinx()
        ax2r.plot(doy, merged["AF"], "r.-", label="Ground AF", alpha=0.7)
        ax2r.set_ylabel("Ground AF", color="red")
        ax2r.tick_params(axis="y", labelcolor="red")

    # Panel 3: Wind speed + ground RH variance
    ax3 = axes[2]
    ax3.plot(doy, merged["wind_speed_median"], "g.-", label="CYGNSS wind speed", alpha=0.7)
    ax3.set_ylabel("Wind Speed (m/s)", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    if "VS" in merged.columns:
        ax3r = ax3.twinx()
        ax3r.plot(doy, merged["VS"], "m.-", label="Ground VS", alpha=0.7)
        ax3r.set_ylabel("Ground SNR Variance", color="purple")
        ax3r.tick_params(axis="y", labelcolor="purple")
    ax3.set_xlabel("Day of Year")

    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_cygnss_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_scatter(merged, station, year, out_dir):
    """Scatter plots: CYGNSS observables vs ground features."""
    if merged.empty:
        return

    pairs = [
        ("fresnel_coeff_median", "CLR", "CYGNSS Fresnel Coeff", "Ground CLR"),
        ("fresnel_coeff_median", "AF", "CYGNSS Fresnel Coeff", "Ground AF"),
        ("nbrcs_median", "CLR", "CYGNSS NBRCS", "Ground CLR"),
        ("nbrcs_median", "AF", "CYGNSS NBRCS", "Ground AF"),
        ("fresnel_coeff_median", "gamma", "CYGNSS Fresnel Coeff", "Ground gamma"),
        ("wind_speed_median", "VS", "CYGNSS Wind Speed", "Ground SNR Variance"),
    ]
    pairs = [(cx, gx, cl, gl) for cx, gx, cl, gl in pairs
             if cx in merged.columns and gx in merged.columns]

    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (cx, gx, cl, gl) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        valid = merged[[cx, gx]].dropna()
        ax.scatter(valid[cx], valid[gx], c=valid.index, cmap="viridis",
                   s=15, alpha=0.6, edgecolors="none")
        ax.set_xlabel(cl)
        ax.set_ylabel(gl)

        # Correlation
        if len(valid) > 5:
            corr = valid[cx].corr(valid[gx])
            ax.set_title(f"r = {corr:.3f}  (n={len(valid)})")
        else:
            ax.set_title(f"n={len(valid)}")

    for idx in range(len(pairs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"{station} {year}: CYGNSS vs Ground GNSS-IR", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_cygnss_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_map(cygnss_df, station_lat, station_lon, station, year, out_dir):
    """Map of CYGNSS specular points colored by Fresnel coefficient."""
    if cygnss_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(cygnss_df["lon"], cygnss_df["lat"],
                    c=cygnss_df["fresnel_coeff"], cmap="viridis",
                    s=3, alpha=0.3, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Fresnel Coefficient", shrink=0.7)

    ax.plot(station_lon, station_lat, "r*", markersize=15, label=station,
            zorder=10)

    # Draw radius circles at 0.5° and 1.0°
    for r in [0.5, 1.0]:
        circle = plt.Circle((station_lon, station_lat), r,
                             fill=False, color="red", ls="--", lw=1)
        ax.add_patch(circle)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{station} {year}: CYGNSS Specular Points")
    ax.legend()
    ax.set_aspect("equal")

    # Set axis limits based on data
    margin = 0.2
    ax.set_xlim(cygnss_df["lon"].min() - margin, cygnss_df["lon"].max() + margin)
    ax.set_ylim(cygnss_df["lat"].min() - margin, cygnss_df["lat"].max() + margin)

    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_cygnss_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def print_summary(merged, cygnss_df, station, year):
    """Print correlation summary."""
    print(f"\n{'=' * 60}")
    print(f"CYGNSS Comparison Summary: {station} {year}")
    print(f"{'=' * 60}")
    print(f"CYGNSS specular points extracted: {len(cygnss_df)}")
    print(f"Days with matched data: {len(merged)}")

    if merged.empty:
        print("No matched data — cannot compute correlations.")
        return

    print(f"\nCYGNSS coverage: DOY {merged['doy'].min()}-{merged['doy'].max()}")
    print(f"Median CYGNSS points per day: {merged['n_cygnss'].median():.0f}")
    print(f"Median CYGNSS Fresnel coeff: {merged['fresnel_coeff_median'].median():.4f}")

    print(f"\nCorrelations (CYGNSS daily median vs ground daily median):")
    cygnss_vars = ["fresnel_coeff_median", "nbrcs_median", "wind_speed_median"]
    ground_vars = ["CLR", "AF", "PR", "gamma", "phase", "RH", "VS"]
    ground_vars = [g for g in ground_vars if g in merged.columns]

    for cv in cygnss_vars:
        cv_short = cv.replace("_median", "")
        for gv in ground_vars:
            valid = merged[[cv, gv]].dropna()
            if len(valid) > 10:
                r = valid[cv].corr(valid[gv])
                print(f"  {cv_short:>16s} vs {gv:<8s}: r={r:+.3f}  (n={len(valid)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CYGNSS vs ground GNSS-IR comparison")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy_start", type=int, default=1)
    parser.add_argument("--doy_end", type=int, default=365)
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Search radius in degrees (default 1.0)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Use cached CYGNSS files only")
    parser.add_argument("--cache_dir", help="Override cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    station_lat, station_lon = _load_station_lat_lon(args.station)
    logger.info(f"Station {args.station}: lat={station_lat:.3f}, lon={station_lon:.3f}")

    # Check latitude
    if abs(station_lat) > 38:
        logger.error(f"Station {args.station} at {station_lat:.1f}°N is outside "
                     f"CYGNSS coverage (±38°). Aborting.")
        sys.exit(1)

    out_dir = get_results_dir(args.station) / "cygnss"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download CYGNSS L2
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir("cygnss")
    if args.skip_download:
        import glob
        nc_files = sorted(glob.glob(str(cache_dir / "cyg.ddmi.*.l2.wind-mss.*.nc")))
        logger.info(f"Using {len(nc_files)} cached CYGNSS files")
    else:
        nc_files = download_cygnss_l2(args.year, args.doy_start, args.doy_end,
                                       cache_dir)

    if not nc_files:
        logger.error("No CYGNSS files available")
        sys.exit(1)

    # Step 2: Extract near station
    logger.info(f"Extracting CYGNSS points within {args.radius}° of {args.station}")
    cygnss_df = extract_cygnss_near_station(
        nc_files, station_lat, station_lon, radius_deg=args.radius
    )
    if cygnss_df.empty:
        logger.error("No CYGNSS points found near station")
        sys.exit(1)

    # Save extracted CYGNSS data
    cygnss_path = out_dir / f"{args.station}_{args.year}_cygnss_extracted.parquet"
    cygnss_df.to_parquet(cygnss_path, index=False)
    logger.info(f"Saved {cygnss_path}")

    # Step 3: Load ground features
    ground_df = load_ground_features(args.station, args.year)
    if ground_df is None:
        logger.error("No ground features — run snr_feature_extractor.py first")
        sys.exit(1)

    # Step 4: Build comparison
    merged = build_comparison(cygnss_df, ground_df, radius_deg=args.radius)

    # Save merged comparison
    if not merged.empty:
        merged_path = out_dir / f"{args.station}_{args.year}_cygnss_comparison.parquet"
        merged.to_parquet(merged_path, index=False)

    # Step 5: Visualize
    plot_timeseries(merged, args.station, args.year, out_dir)
    plot_scatter(merged, args.station, args.year, out_dir)
    plot_map(cygnss_df, station_lat, station_lon, args.station, args.year, out_dir)

    # Step 6: Summary
    print_summary(merged, cygnss_df, args.station, args.year)
    print(f"\nOutputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
