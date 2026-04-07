#!/usr/bin/env python3
"""
Investigation B4: Simultaneous satellite geometry census.

For selected days, reads SNR files and identifies co-temporal satellite pairs
where both are at similar elevation but different azimuth. This assesses the
feasibility of spatial heterogeneity detection (e.g., ice edges).

Usage:
    python scripts/investigate_geometry_census.py --station UMNQ --year 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.snr_feature_extractor import read_snr_file, _snr_file_path

log = logging.getLogger(__name__)


def load_station_azimuth_mask(station):
    """Load azimuth mask from station config."""
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            all_cfg = json.load(f)
        st_cfg = all_cfg.get(station, {})
        return st_cfg.get("azimuth_min", 0), st_cfg.get("azimuth_max", 360)
    return 0, 360


def census_day(station, year, doy, e1, e2, az_min, az_max,
               elev_tol=1.0, az_sep_min=30.0, epoch_step=30):
    """Find co-temporal satellite pairs for one day.

    Args:
        station, year, doy: data identifiers
        e1, e2: elevation window
        az_min, az_max: azimuth mask
        elev_tol: max elevation difference for a "pair" (degrees)
        az_sep_min: min azimuth separation for a "pair" (degrees)
        epoch_step: bin epochs in this many seconds (reduces combinatorics)

    Returns:
        list of dicts, one per usable pair found
    """
    snr_path = _snr_file_path(station, year, doy)
    if not snr_path.exists():
        gz = Path(str(snr_path) + ".gz")
        if gz.exists():
            snr_path = gz
        else:
            return []

    try:
        snr_data = read_snr_file(snr_path)
    except Exception:
        return []

    # Filter to elevation and azimuth window
    mask = ((snr_data[:, 1] >= e1) & (snr_data[:, 1] <= e2) &
            (snr_data[:, 2] >= az_min) & (snr_data[:, 2] <= az_max))
    data = snr_data[mask]
    if len(data) < 10:
        return []

    # Bin by epoch (seconds of day)
    sod = data[:, 3]
    epoch_bins = (sod // epoch_step).astype(int)

    pairs = []
    for epoch_bin in np.unique(epoch_bins):
        epoch_mask = epoch_bins == epoch_bin
        epoch_data = data[epoch_mask]

        # Get unique satellites in this epoch
        sats = np.unique(epoch_data[:, 0]).astype(int)
        if len(sats) < 2:
            continue

        # For each satellite, get median elevation and azimuth
        sat_info = {}
        for sat in sats:
            sat_mask = epoch_data[:, 0] == sat
            sat_info[sat] = {
                "elev": np.median(epoch_data[sat_mask, 1]),
                "az": np.median(epoch_data[sat_mask, 2]),
                "sod": np.median(epoch_data[sat_mask, 3]),
            }

        # Find pairs
        sat_list = list(sat_info.keys())
        for i in range(len(sat_list)):
            for j in range(i + 1, len(sat_list)):
                si = sat_info[sat_list[i]]
                sj = sat_info[sat_list[j]]

                elev_diff = abs(si["elev"] - sj["elev"])
                az_diff = abs(si["az"] - sj["az"])
                if az_diff > 180:
                    az_diff = 360 - az_diff

                if elev_diff <= elev_tol and az_diff >= az_sep_min:
                    pairs.append({
                        "doy": doy,
                        "epoch_sod": float(epoch_bin * epoch_step),
                        "sat_i": sat_list[i],
                        "sat_j": sat_list[j],
                        "elev_i": si["elev"],
                        "elev_j": sj["elev"],
                        "az_i": si["az"],
                        "az_j": sj["az"],
                        "elev_diff": elev_diff,
                        "az_separation": az_diff,
                        "mean_elev": (si["elev"] + sj["elev"]) / 2,
                    })

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Simultaneous satellite geometry census"
    )
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--doys", type=str, default=None,
                        help="Comma-separated DOYs (default: 4 seasonal samples)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load station config for e1/e2 and azimuth
    from scripts.snr_feature_extractor import _load_station_config
    config = _load_station_config(args.station)
    if config is None:
        print(f"No config for {args.station}")
        sys.exit(1)

    e1, e2 = config["e1"], config["e2"]
    az_min, az_max = load_station_azimuth_mask(args.station)
    log.info(f"Station {args.station}: e=[{e1},{e2}], az=[{az_min},{az_max}]")

    # Select representative DOYs
    if args.doys:
        doys = [int(d) for d in args.doys.split(",")]
    else:
        # Pick 4 seasonal samples — check which SNR files exist
        snr_dir = (PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code"
                   / str(args.year) / "snr" / args.station.lower())
        if snr_dir.exists():
            available = sorted([
                int(f.name.split(args.station.lower())[1][:3])
                for f in snr_dir.iterdir()
                if f.name.endswith(".snr66") or f.name.endswith(".snr66.gz")
            ])
        else:
            available = list(range(1, 366))

        # Pick 4 spread across the year
        if len(available) >= 4:
            n = len(available)
            doys = [available[n // 8], available[3 * n // 8],
                    available[5 * n // 8], available[7 * n // 8]]
        else:
            doys = available[:4]

    log.info(f"Census DOYs: {doys}")

    all_pairs = []
    for doy in doys:
        pairs = census_day(args.station, args.year, doy, e1, e2,
                           az_min, az_max)
        log.info(f"  DOY {doy}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("No usable satellite pairs found.")
        sys.exit(0)

    df = pd.DataFrame(all_pairs)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Geometry Census: {args.station} {args.year}")
    print(f"{'='*60}")
    print(f"DOYs sampled: {doys}")
    print(f"Total pairs: {len(df)}")
    print(f"Unique satellite pair combos: {df.groupby(['sat_i','sat_j']).ngroups}")

    for doy in doys:
        sub = df[df["doy"] == doy]
        n_hours = sub["epoch_sod"].nunique() * 30 / 3600  # approx hours
        print(f"\n  DOY {doy}: {len(sub)} pairs across ~{n_hours:.1f} hours")
        print(f"    Elevation range: {sub['mean_elev'].min():.1f}–{sub['mean_elev'].max():.1f}°")
        print(f"    Az separation: {sub['az_separation'].min():.0f}–{sub['az_separation'].max():.0f}°")
        print(f"    Per hour: ~{len(sub) / max(n_hours, 1):.0f} pairs")

    # --- Figures ---
    out_dir = PROJECT_ROOT / "results_annual" / args.station / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram of elevations
    axes[0, 0].hist(df["mean_elev"], bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Mean elevation (°)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Elevation distribution of pairs")

    # Histogram of azimuth separations
    axes[0, 1].hist(df["az_separation"], bins=20, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Azimuth separation (°)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Azimuth separation distribution")

    # Pairs per hour by DOY
    for doy in doys:
        sub = df[df["doy"] == doy]
        hours = sub["epoch_sod"] / 3600
        axes[1, 0].hist(hours, bins=24, alpha=0.5, label=f"DOY {doy}")
    axes[1, 0].set_xlabel("Hour of day (UTC)")
    axes[1, 0].set_ylabel("Pairs per hour")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Pair availability by hour")

    # Scatter: elevation vs azimuth separation
    axes[1, 1].scatter(df["mean_elev"], df["az_separation"], s=3, alpha=0.3)
    axes[1, 1].set_xlabel("Mean elevation (°)")
    axes[1, 1].set_ylabel("Azimuth separation (°)")
    axes[1, 1].set_title("Elevation vs azimuth geometry")

    fig.suptitle(f"{args.station} {args.year}: Satellite pair geometry census",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.station}_{args.year}_geometry_census.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved figure to {out_dir}")

    # Save data
    df.to_parquet(out_dir / f"{args.station}_{args.year}_geometry_census.parquet",
                  index=False)


if __name__ == "__main__":
    main()
