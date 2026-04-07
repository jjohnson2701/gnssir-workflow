#!/usr/bin/env python3
"""
Investigation B1: edot as geometry-informed QC layer.

Column 4 of every SNR file contains edot (elevation rate in deg/s).
Low edot arcs (satellite near apex) have slowly oscillating SNR that's
hard to distinguish from detrending residuals, potentially producing
noisier gamma and AF estimates.

This script extracts median |edot| per arc, then correlates it with
gamma_r2 (fit quality) and gamma outlier magnitude.

Usage:
    python scripts/investigate_edot_qc.py --station UMNQ --year 2025
    python scripts/investigate_edot_qc.py --station ROSS --year 2024
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

from scripts.snr_feature_extractor import (
    read_snr_file, segment_satellite_arcs, find_matching_segment,
    _snr_file_path, _load_station_config,
)

log = logging.getLogger(__name__)


def extract_edot_for_day(station, year, doy, day_arcs, config, snr_data):
    """Extract median |edot| for each arc from SNR file.

    Args:
        station, year, doy: identifiers
        day_arcs: DataFrame of arc_table rows for this day
        config: station gnssir config dict
        snr_data: pre-loaded SNR array

    Returns:
        list of dicts with arc identifiers + edot_med
    """
    e1, e2 = config["e1"], config["e2"]
    arc_groups = day_arcs.groupby(["sat", "UTCtime", "rise"])
    results = []
    segment_cache = {}

    for (sat, utctime, rise), freq_rows in arc_groups:
        sat = int(sat)
        if sat not in segment_cache:
            sat_mask = snr_data[:, 0] == sat
            sat_data = snr_data[sat_mask]
            if len(sat_data) < 5:
                segment_cache[sat] = ([], None)
            else:
                arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
                segment_cache[sat] = (arcs, sat_data)

        arcs, sat_data = segment_cache[sat]
        if sat_data is None or len(arcs) == 0:
            continue

        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime, target_rise=rise, e1=e1, e2=e2,
        )
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        # Window to [e1, e2]
        mask = (arc_ele >= e1) & (arc_ele <= e2)
        if mask.sum() < 10:
            continue

        # Column 4 is edot (elevation rate in deg/s)
        edot_windowed = arc_data[mask, 4]
        edot_med = float(np.median(np.abs(edot_windowed)))

        # Attach to all frequencies of this physical arc
        for _, row in freq_rows.iterrows():
            results.append({
                "doy": doy,
                "sat": sat,
                "UTCtime": utctime,
                "rise": int(rise),
                "freq": int(row["freq"]),
                "edot_med": edot_med,
                "Azim": float(row.get("Azim", np.nan)),
                "azimuth_bin": row.get("azimuth_bin", np.nan),
            })

    return results


def process_station_year(station, year, max_doys=None):
    """Extract edot for all arcs and merge with existing features."""
    config = _load_station_config(station)
    if config is None:
        log.error(f"No config for {station}")
        return None

    at = pd.read_parquet(
        PROJECT_ROOT / "results_annual" / station
        / f"{station}_{year}_arc_table.parquet"
    )
    log.info(f"Loaded {len(at)} arcs for {station} {year}")

    doys = sorted(at["doy"].unique())
    if max_doys is not None:
        doys = doys[:max_doys]
    log.info(f"Processing {len(doys)} DOYs")

    all_results = []
    for i, doy in enumerate(doys):
        if (i + 1) % 50 == 0:
            log.info(f"  Day {i+1}/{len(doys)}")

        snr_path = _snr_file_path(station, year, doy)
        if not snr_path.exists():
            gz = Path(str(snr_path) + ".gz")
            if gz.exists():
                snr_path = gz
            else:
                continue

        try:
            snr_data = read_snr_file(snr_path)
        except Exception:
            continue

        day_arcs = at[at["doy"] == doy]
        all_results.extend(
            extract_edot_for_day(station, year, doy, day_arcs, config, snr_data)
        )

    if not all_results:
        return None

    edot_df = pd.DataFrame(all_results)

    # Merge with arc_table features
    join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]
    feat_cols = ["gamma", "gamma_r2", "AF", "CLR", "full_arc"]
    available = [c for c in feat_cols if c in at.columns]
    merged = edot_df.merge(at[join_cols + available], on=join_cols, how="left")

    log.info(f"Extracted edot for {len(merged)} arcs "
             f"({merged['edot_med'].notna().sum()} valid)")
    return merged


def make_diagnostics(df, station, year, out_dir):
    """Produce diagnostic figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df["month"] = (pd.to_datetime(
        pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(df["doy"] - 1, unit="D")
    )).dt.month if False else None
    # Compute month properly
    dates = pd.to_datetime(year * 1000 + df["doy"], format="%Y%j")
    df["month"] = dates.dt.month

    # gamma_r2 may not exist yet if features haven't been re-extracted
    has_gamma_r2 = "gamma_r2" in df.columns
    gamma_col = "gamma_r2" if has_gamma_r2 else "gamma"
    drop_cols = ["edot_med"] + ([gamma_col] if gamma_col in df.columns else [])
    valid = df.dropna(subset=drop_cols)
    log.info(f"Valid rows with edot + {gamma_col}: {len(valid)}")

    # --- Figure 1: edot vs gamma_r2 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(valid["edot_med"], valid[gamma_col], s=1, alpha=0.1)
    ax.set_xlabel("median |edot| (deg/s)")
    ax.set_ylabel(gamma_col)
    ax.set_title(f"edot vs {gamma_col} (n={len(valid)})")
    # Binned median
    bins = pd.qcut(valid["edot_med"], 20, duplicates="drop")
    binned = valid.groupby(bins)[gamma_col].median()
    bin_centers = [b.mid for b in binned.index]
    ax.plot(bin_centers, binned.values, "r-", lw=2, label="binned median")
    ax.legend()

    # --- Figure 1b: edot vs |gamma - sector median gamma| (outlier magnitude) ---
    ax = axes[1]
    if "gamma" in valid.columns:
        # Compute sector-daily median gamma
        sector_med = valid.groupby(["doy", "azimuth_bin"])["gamma"].transform("median")
        valid_copy = valid.copy()
        valid_copy["gamma_deviation"] = (valid_copy["gamma"] - sector_med).abs()
        v2 = valid_copy.dropna(subset=["gamma_deviation"])

        ax.scatter(v2["edot_med"], v2["gamma_deviation"], s=1, alpha=0.1)
        ax.set_xlabel("median |edot| (deg/s)")
        ax.set_ylabel("|gamma - sector median|")
        ax.set_title(f"edot vs gamma deviation")

        bins2 = pd.qcut(v2["edot_med"], 20, duplicates="drop")
        binned2 = v2.groupby(bins2)["gamma_deviation"].median()
        bc2 = [b.mid for b in binned2.index]
        ax.plot(bc2, binned2.values, "r-", lw=2, label="binned median")
        ax.legend()
    else:
        ax.set_title("No gamma data")

    # --- Figure 1c: edot histogram per azimuth sector ---
    ax = axes[2]
    if "azimuth_bin" in df.columns:
        sectors = sorted(df["azimuth_bin"].dropna().unique())[:8]
        for sector in sectors:
            sec_data = df[df["azimuth_bin"] == sector]["edot_med"].dropna()
            ax.hist(sec_data, bins=30, alpha=0.4, label=f"az={sector}",
                    density=True)
        ax.set_xlabel("median |edot| (deg/s)")
        ax.set_ylabel("Density")
        ax.set_title("edot distribution by azimuth sector")
        ax.legend(fontsize=7)

    fig.suptitle(f"{station} {year}: edot as QC diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_{year}_edot_diagnostic.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: edot time series by sector ---
    fig, ax = plt.subplots(figsize=(14, 5))
    daily = df.groupby("doy")["edot_med"].agg(["median", "std", "count"])
    daily = daily[daily["count"] >= 5]
    ax.plot(daily.index, daily["median"], ".", ms=2)
    ax.fill_between(daily.index,
                    daily["median"] - daily["std"],
                    daily["median"] + daily["std"], alpha=0.2)
    ax.set_xlabel("DOY")
    ax.set_ylabel("median |edot| (deg/s)")
    ax.set_title(f"{station} {year}: daily median edot — confirms geometry is ~static")
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_{year}_edot_timeseries.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Summary statistics ---
    print(f"\n{'='*60}")
    print(f"edot QC diagnostic: {station} {year}")
    print(f"{'='*60}")
    print(f"Total arcs with edot: {df['edot_med'].notna().sum()}")
    print(f"edot range: {df['edot_med'].min():.4f} – {df['edot_med'].max():.4f} deg/s")
    print(f"edot median: {df['edot_med'].median():.4f} deg/s")
    print(f"edot Q10: {df['edot_med'].quantile(0.1):.4f} deg/s")

    if gamma_col in valid.columns and len(valid) > 0:
        # Split by edot quartile
        valid_q = valid.copy()
        valid_q["edot_q"] = pd.qcut(valid_q["edot_med"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        print(f"\n{gamma_col} by edot quartile:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = valid_q[valid_q["edot_q"] == q]
            print(f"  {q} (edot ≤ {sub['edot_med'].max():.4f}): "
                  f"{gamma_col} median={sub[gamma_col].median():.3f}, "
                  f"mean={sub[gamma_col].mean():.3f}")

        # Correlation
        r_edot_gc = valid["edot_med"].corr(valid[gamma_col])
        print(f"\nPearson r(edot, {gamma_col}): {r_edot_gc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="edot QC diagnostic")
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--max_doys", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    df = process_station_year(args.station, args.year, max_doys=args.max_doys)
    if df is None:
        print("No results")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "results_annual" / args.station / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{args.station}_{args.year}_edot.parquet", index=False)

    make_diagnostics(df, args.station, args.year, out_dir)


if __name__ == "__main__":
    main()
