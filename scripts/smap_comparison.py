#!/usr/bin/env python3
"""Compare SMAP L-band radiometry with ground-based GNSS-IR ice classification.

Downloads SMAP Enhanced L3 Freeze/Thaw (SPL3FTP_E, 9km daily) and extracts
brightness temperature, normalized polarization ratio, and freeze/thaw state
at GNSS-IR station locations. Compares with ice_classification.parquet.

The SMAP freeze/thaw product is designed for soil, so it returns fill values
over large water bodies. However, the underlying radiometric observables
(TBv, TBh, NPR) still respond to surface ice — frozen lakes have lower
emissivity and different polarization signatures than open water.

Usage:
    python scripts/smap_comparison.py --station ROSS --year 2024
    python scripts/smap_comparison.py --station ROSS --year 2024 --skip_download
    python scripts/smap_comparison.py --station ROSS --years 2020-2024
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

SMAP_SHORT_NAME = "SPL3FTP_E"
SMAP_GROUP = "Freeze_Thaw_Retrieval_Data_Global"
# Fill/invalid value for freeze_thaw field
FT_FILL = 254


# ---------------------------------------------------------------------------
# Station config
# ---------------------------------------------------------------------------

def _load_station_lat_lon(station):
    """Return (lat, lon) tuple for a station."""
    coords = load_station_coords(station)
    return coords["lat"], coords["lon"]


# ---------------------------------------------------------------------------
# SMAP download
# ---------------------------------------------------------------------------

def download_smap_ft(year, doy_start, doy_end, cache_dir):
    """Download SMAP SPL3FTP_E daily files via earthaccess."""
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        logger.error("Earthdata auth failed")
        sys.exit(1)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(year, 1, 1) + timedelta(days=doy_start - 1)
    end_date = datetime(year, 1, 1) + timedelta(days=doy_end - 1)

    logger.info(f"Searching {SMAP_SHORT_NAME}: {start_date.date()} to {end_date.date()}")
    results = earthaccess.search_data(
        short_name=SMAP_SHORT_NAME,
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    logger.info(f"Found {len(results)} granules")

    # Filter already-downloaded
    to_download = []
    existing = []
    for r in results:
        fname = r.data_links()[0].split("/")[-1]
        local = cache_dir / fname
        if local.exists():
            existing.append(str(local))
        else:
            to_download.append(r)

    logger.info(f"Cached: {len(existing)}, to download: {len(to_download)}")
    downloaded = []
    if to_download:
        downloaded = earthaccess.download(to_download, str(cache_dir))
        downloaded = [str(p) for p in downloaded]

    return sorted(existing + downloaded)


# ---------------------------------------------------------------------------
# SMAP extraction
# ---------------------------------------------------------------------------

def find_nearest_pixel(h5_file, station_lat, station_lon):
    """Find the nearest EASE-Grid pixel indices for a station.

    Returns (row, col) in the Global grid.
    """
    import h5py

    with h5py.File(h5_file, "r") as f:
        lat = f[f"{SMAP_GROUP}/latitude"][0]  # AM pass
        lon = f[f"{SMAP_GROUP}/longitude"][0]

    dist = np.sqrt((lat - station_lat)**2 + (lon - station_lon)**2)
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    return int(r), int(c), float(lat[r, c]), float(lon[r, c])


def extract_smap_timeseries(h5_files, pixel_row, pixel_col):
    """Extract daily SMAP values at a fixed pixel location.

    Returns DataFrame with columns:
        date, ft_am, ft_pm, npr_am, npr_pm, tbv_am, tbv_pm,
        tbh_am, tbh_pm, water_frac, transition_flag
    """
    import h5py

    rows = []
    for fpath in h5_files:
        # Parse date from filename: SMAP_L3_FT_P_E_YYYYMMDD_...
        fname = Path(fpath).name
        parts = fname.split("_")
        try:
            date_str = parts[5]
            date = datetime.strptime(date_str, "%Y%m%d").date()
        except (IndexError, ValueError):
            logger.warning(f"Cannot parse date from {fname}")
            continue

        try:
            with h5py.File(fpath, "r") as f:
                r, c = pixel_row, pixel_col
                g = SMAP_GROUP

                ft_am = int(f[f"{g}/freeze_thaw"][0, r, c])
                ft_pm = int(f[f"{g}/freeze_thaw"][1, r, c])
                npr_am = float(f[f"{g}/normalized_polarization_ratio"][0, r, c])
                npr_pm = float(f[f"{g}/normalized_polarization_ratio"][1, r, c])
                tbv_am = float(f[f"{g}/tbv_mean"][0, r, c])
                tbv_pm = float(f[f"{g}/tbv_mean"][1, r, c])
                tbh_am = float(f[f"{g}/tbh_mean"][0, r, c])
                tbh_pm = float(f[f"{g}/tbh_mean"][1, r, c])
                water_frac = float(f[f"{g}/open_water_body_fraction"][0, r, c])
                trans = int(f[f"{g}/transition_state_flag"][r, c])

            # Mask fill values
            row = {
                "date": date,
                "doy": date.timetuple().tm_yday,
                "ft_am": ft_am if ft_am <= 1 else np.nan,
                "ft_pm": ft_pm if ft_pm <= 1 else np.nan,
                "npr_am": npr_am if abs(npr_am) < 100 else np.nan,
                "npr_pm": npr_pm if abs(npr_pm) < 100 else np.nan,
                "tbv_am": tbv_am if tbv_am > 0 else np.nan,
                "tbv_pm": tbv_pm if tbv_pm > 0 else np.nan,
                "tbh_am": tbh_am if tbh_am > 0 else np.nan,
                "tbh_pm": tbh_pm if tbh_pm > 0 else np.nan,
                "water_frac": water_frac,
                "transition_flag": trans,
            }
            rows.append(row)

        except Exception as e:
            logger.warning(f"Error reading {fname}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Derived features
    df["emissivity_v"] = df["tbv_am"] / 273.15  # rough approximation
    df["polarization_diff"] = df["tbv_am"] - df["tbh_am"]
    logger.info(f"Extracted {len(df)} SMAP days")
    return df


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def load_ice_classification(station, years):
    """Load ice classification, return daily labels."""
    all_dfs = []
    for year in years:
        ic_path = (PROJECT_ROOT / "results_annual" / station /
                   f"{station}_{year}_ice_classification.parquet")
        if not ic_path.exists():
            continue
        ic = pd.read_parquet(ic_path)
        if "classification" not in ic.columns:
            continue
        ic["date_dt"] = pd.to_datetime(ic["date"])
        ic["doy"] = ic["date_dt"].dt.dayofyear
        ic["year"] = ic["date_dt"].dt.year
        all_dfs.append(ic[["date", "doy", "year", "classification", "ice_score"]].copy()
                       if "ice_score" in ic.columns
                       else ic[["date", "doy", "year", "classification"]].copy())

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def load_ground_features(station, year):
    """Load SNR features from arc_table (or legacy snr_features), compute daily medians."""
    # Prefer arc_table (4-layer), fall back to snr_features (legacy)
    results_dir = PROJECT_ROOT / "results_annual" / station
    sf_path = results_dir / f"{station}_{year}_arc_table.parquet"
    if not sf_path.exists():
        sf_path = results_dir / f"{station}_{year}_snr_features.parquet"
    if not sf_path.exists():
        return None
    sf = pd.read_parquet(sf_path)
    if "CLR" not in sf.columns:
        return None
    features = ["CLR", "AF", "PR", "gamma", "phase", "RH", "MS", "VS"]
    features = [f for f in features if f in sf.columns]
    daily = sf.groupby("doy")[features].median().reset_index()
    daily["n_arcs"] = sf.groupby("doy").size().values
    return daily


def build_comparison(smap_df, ice_df, year, station=None):
    """Merge SMAP, ice classification, and ground SNR features on DOY."""
    ice_year = ice_df[ice_df["year"] == year].copy()
    merged = pd.merge(smap_df, ice_year, on="doy", how="inner",
                      suffixes=("_smap", "_ice"))

    # Also merge ground SNR features if available
    if station:
        ground = load_ground_features(station, year)
        if ground is not None:
            merged = pd.merge(merged, ground, on="doy", how="left",
                              suffixes=("", "_ground"))
            logger.info(f"Merged ground SNR features: {[c for c in ground.columns if c != 'doy']}")

    logger.info(f"Matched {len(merged)} days for {year}")
    return merged


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_timeseries(merged, station, year, out_dir):
    """Time series: SMAP TB + ice classification."""
    if merged.empty:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    doy = merged["doy"]

    # Color mapping for classification
    cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}
    cls_numeric = merged["classification"].map(
        {"ice": 1.0, "transition": 0.5, "water": 0.0}
    )

    # Panel 1: TBv (AM)
    ax = axes[0]
    ax.plot(doy, merged["tbv_am"], "b.-", alpha=0.7, label="TBv AM", ms=3)
    ax.plot(doy, merged["tbh_am"], "r.-", alpha=0.7, label="TBh AM", ms=3)
    ax.set_ylabel("Brightness Temp (K)")
    ax.legend(fontsize=8)
    ax.set_title(f"{station} {year}: SMAP L-band vs GNSS-IR Ice Classification")

    # Panel 2: Polarization difference (TBv - TBh)
    ax = axes[1]
    ax.plot(doy, merged["polarization_diff"], "k.-", alpha=0.7, ms=3)
    ax.set_ylabel("TBv - TBh (K)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    # Panel 3: NPR
    ax = axes[2]
    valid_npr = merged["npr_am"].notna()
    if valid_npr.sum() > 0:
        ax.plot(doy[valid_npr], merged.loc[valid_npr, "npr_am"], "m.-",
                alpha=0.7, ms=3)
    ax.set_ylabel("NPR (AM)")

    # Panel 4: Ice classification + ice_score
    ax = axes[3]
    for cls, color in cls_colors.items():
        mask = merged["classification"] == cls
        if mask.sum() > 0:
            ax.scatter(doy[mask], cls_numeric[mask], c=color, label=cls,
                       s=20, zorder=5)
    if "ice_score" in merged.columns:
        ax_r = ax.twinx()
        ax_r.plot(doy, merged["ice_score"], "k-", alpha=0.4, label="ice_score")
        ax_r.set_ylabel("Ice Score", color="gray")
    ax.set_ylabel("Classification")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["water", "transition", "ice"])
    ax.legend(fontsize=8)
    ax.set_xlabel("Day of Year")

    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_smap_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_scatter(merged, station, year, out_dir):
    """Scatter: SMAP observables vs DOY, colored by classification."""
    if merged.empty:
        return

    cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}

    smap_vars = [
        ("tbv_am", "TBv AM (K)"),
        ("tbh_am", "TBh AM (K)"),
        ("polarization_diff", "TBv - TBh (K)"),
        ("npr_am", "NPR AM"),
    ]
    smap_vars = [(v, l) for v, l in smap_vars if v in merged.columns]

    fig, axes = plt.subplots(1, len(smap_vars), figsize=(5 * len(smap_vars), 4.5))
    if len(smap_vars) == 1:
        axes = [axes]

    for ax, (var, label) in zip(axes, smap_vars):
        for cls in ["transition", "water", "ice"]:
            mask = merged["classification"] == cls
            valid = mask & merged[var].notna()
            if valid.sum() > 0:
                ax.scatter(merged.loc[valid, "doy"],
                           merged.loc[valid, var],
                           c=cls_colors[cls], label=cls, s=15, alpha=0.6,
                           edgecolors="none")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, markerscale=2)

    fig.suptitle(f"{station} {year}: SMAP Observables by Season and Classification", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_smap_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_smap_vs_ground(merged, station, year, out_dir):
    """Scatter: SMAP radiometry vs ground-based SNR features."""
    if merged.empty:
        return

    ground_features = ["CLR", "AF", "gamma", "VS"]
    ground_features = [f for f in ground_features if f in merged.columns]
    if not ground_features:
        logger.info("No ground SNR features in merged data, skipping smap_vs_ground plot")
        return

    smap_vars = [
        ("polarization_diff", "SMAP TBv-TBh (K)"),
        ("tbv_am", "SMAP TBv AM (K)"),
    ]
    smap_vars = [(v, l) for v, l in smap_vars if v in merged.columns]

    cls_colors = {"ice": "#1f77b4", "water": "#ff7f0e", "transition": "#2ca02c"}

    nrows = len(smap_vars)
    ncols = len(ground_features)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, (sv, sl) in enumerate(smap_vars):
        for ci, gf in enumerate(ground_features):
            ax = axes[ri, ci]
            valid = merged[[sv, gf, "classification"]].dropna()
            if len(valid) < 5:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
                continue

            for cls in ["transition", "water", "ice"]:
                mask = valid["classification"] == cls
                if mask.sum() > 0:
                    ax.scatter(valid.loc[mask, gf], valid.loc[mask, sv],
                               c=cls_colors[cls], label=cls, s=12, alpha=0.5,
                               edgecolors="none")

            r = valid[sv].corr(valid[gf])
            ax.set_xlabel(f"Ground {gf}", fontsize=9)
            ax.set_ylabel(sl, fontsize=9)
            ax.set_title(f"r = {r:+.3f}", fontsize=10,
                         fontweight="bold" if abs(r) > 0.2 else "normal",
                         color="darkred" if abs(r) > 0.3 else "black")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(f"{station} {year}: SMAP Radiometry vs Ground GNSS-IR Features\n"
                 f"(SMAP: single 9 km pixel, Ground: daily median across all arcs)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_smap_vs_ground.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_tb_by_class(merged, station, year, out_dir):
    """Box plots of SMAP TB by ice classification."""
    if merged.empty:
        return

    smap_vars = ["tbv_am", "tbh_am", "polarization_diff", "npr_am"]
    smap_vars = [v for v in smap_vars if v in merged.columns]
    labels = {
        "tbv_am": "TBv AM (K)", "tbh_am": "TBh AM (K)",
        "polarization_diff": "TBv-TBh (K)", "npr_am": "NPR AM",
    }

    fig, axes = plt.subplots(1, len(smap_vars), figsize=(4.5 * len(smap_vars), 5))
    if len(smap_vars) == 1:
        axes = [axes]

    order = ["water", "transition", "ice"]
    colors = {"water": "#ff7f0e", "transition": "#2ca02c", "ice": "#1f77b4"}

    for ax, var in zip(axes, smap_vars):
        data = []
        group_labels = []
        group_colors = []
        for cls in order:
            vals = merged.loc[merged["classification"] == cls, var].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                group_labels.append(f"{cls}\n(n={len(vals)})")
                group_colors.append(colors[cls])

        if data:
            bp = ax.boxplot(data, labels=group_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], group_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.set_ylabel(labels.get(var, var))
        ax.set_title(var)

    fig.suptitle(f"{station} {year}: SMAP by GNSS-IR Class", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{station}_{year}_smap_boxplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def print_summary(merged, smap_df, station, year):
    """Print comparison statistics."""
    print(f"\n{'=' * 60}")
    print(f"SMAP Comparison Summary: {station} {year}")
    print(f"{'=' * 60}")
    print(f"SMAP days extracted: {len(smap_df)}")
    print(f"Days with ice classification: {len(merged)}")

    if merged.empty:
        return

    # FT agreement
    ft_valid = merged["ft_am"].notna()
    if ft_valid.sum() > 0:
        # Map: SMAP frozen=1 → "ice", thawed=0 → "water"
        smap_ice = merged.loc[ft_valid, "ft_am"] == 1
        gnss_ice = merged.loc[ft_valid, "classification"] == "ice"
        gnss_water = merged.loc[ft_valid, "classification"] == "water"
        agree_ice = (smap_ice & gnss_ice).sum()
        agree_water = (~smap_ice & gnss_water).sum()
        total = ft_valid.sum()
        print(f"\nFT flag agreement (where available, n={total}):")
        print(f"  Both frozen/ice: {agree_ice}")
        print(f"  Both thawed/water: {agree_water}")
        print(f"  Simple agreement: {(agree_ice + agree_water) / total:.1%}")
    else:
        print("\nFT flag: all fill values (water-dominated pixel)")

    # TB statistics by class
    print(f"\nBrightness temperature by GNSS-IR class:")
    for cls in ["ice", "transition", "water"]:
        mask = merged["classification"] == cls
        n = mask.sum()
        if n > 0:
            tbv = merged.loc[mask, "tbv_am"].dropna()
            tbh = merged.loc[mask, "tbh_am"].dropna()
            pdiff = merged.loc[mask, "polarization_diff"].dropna()
            print(f"  {cls:>11s} (n={n:3d}): TBv={tbv.mean():.1f}±{tbv.std():.1f}K  "
                  f"TBh={tbh.mean():.1f}±{tbh.std():.1f}K  "
                  f"ΔTB={pdiff.mean():.1f}±{pdiff.std():.1f}K")

    # Correlations with ice_score
    if "ice_score" in merged.columns:
        print(f"\nCorrelations with ice_score:")
        for var in ["tbv_am", "tbh_am", "polarization_diff", "npr_am"]:
            if var in merged.columns:
                valid = merged[["ice_score", var]].dropna()
                if len(valid) > 10:
                    r = valid["ice_score"].corr(valid[var])
                    print(f"  {var:>20s}: r={r:+.3f}  (n={len(valid)})")

    # Correlations: SMAP vs ground SNR features
    ground_feats = ["CLR", "AF", "gamma", "VS", "phase", "RH"]
    ground_feats = [f for f in ground_feats if f in merged.columns]
    if ground_feats:
        print(f"\nCorrelations: SMAP vs ground SNR features:")
        for sv in ["polarization_diff", "tbv_am", "tbh_am"]:
            if sv not in merged.columns:
                continue
            for gf in ground_feats:
                valid = merged[[sv, gf]].dropna()
                if len(valid) > 10:
                    r = valid[sv].corr(valid[gf])
                    flag = " <--" if abs(r) > 0.3 else ""
                    print(f"  {sv:>20s} vs {gf:<8s}: r={r:+.3f}  (n={len(valid)}){flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SMAP freeze/thaw vs GNSS-IR comparison")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, help="Single year")
    parser.add_argument("--years", help="Year range, e.g. 2020-2024")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--cache_dir", help="Override cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.years:
        start_yr, end_yr = map(int, args.years.split("-"))
        years = list(range(start_yr, end_yr + 1))
    elif args.year:
        years = [args.year]
    else:
        parser.error("Must specify --year or --years")

    station_lat, station_lon = _load_station_lat_lon(args.station)
    logger.info(f"Station {args.station}: lat={station_lat:.3f}, lon={station_lon:.3f}")

    out_dir = get_results_dir(args.station) / "smap"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir("smap")

    # Load ice classification
    ice_df = load_ice_classification(args.station, years)
    if ice_df is None:
        logger.error("No ice classification data found")
        sys.exit(1)
    logger.info(f"Ice classification: {len(ice_df)} days across {years}")

    for year in years:
        print(f"\n{'=' * 40}")
        print(f"Processing {args.station} {year}")
        print(f"{'=' * 40}")

        # Download SMAP
        if args.skip_download:
            import glob
            h5_files = sorted(glob.glob(str(cache_dir / f"SMAP_L3_FT_P_E_{year}*.h5")))
            logger.info(f"Using {len(h5_files)} cached SMAP files for {year}")
        else:
            h5_files = download_smap_ft(year, 1, 366, cache_dir)

        if not h5_files:
            logger.warning(f"No SMAP files for {year}")
            continue

        # Find nearest pixel (use first file)
        r, c, plat, plon = find_nearest_pixel(h5_files[0], station_lat, station_lon)
        logger.info(f"Nearest SMAP pixel: ({r},{c}) at ({plat:.3f}, {plon:.3f})")

        # Extract time series
        smap_df = extract_smap_timeseries(h5_files, r, c)
        if smap_df.empty:
            logger.warning(f"No SMAP data extracted for {year}")
            continue

        # Save SMAP time series
        smap_path = out_dir / f"{args.station}_{year}_smap_extracted.parquet"
        smap_df.to_parquet(smap_path, index=False)
        logger.info(f"Saved {smap_path}")

        # Build comparison (includes ground SNR features)
        merged = build_comparison(smap_df, ice_df, year, station=args.station)
        if merged.empty:
            logger.warning(f"No matched data for {year}")
            continue

        merged_path = out_dir / f"{args.station}_{year}_smap_comparison.parquet"
        merged.to_parquet(merged_path, index=False)

        # Visualize
        plot_timeseries(merged, args.station, year, out_dir)
        plot_scatter(merged, args.station, year, out_dir)
        plot_tb_by_class(merged, args.station, year, out_dir)
        plot_smap_vs_ground(merged, args.station, year, out_dir)

        # Summary
        print_summary(merged, smap_df, args.station, year)

    print(f"\nOutputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
