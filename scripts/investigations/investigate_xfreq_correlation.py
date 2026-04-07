#!/usr/bin/env python3
"""
Investigation B2: Cross-frequency SNR correlation diagnostic.

For each physical arc with multiple frequency bands, rescales the detrended
signals to a common sin(ε)/(λ/2) domain and computes Pearson correlation.
This bypasses LSP estimation noise and may be a more sensitive detector of
layered media (ice, snow-on-ice).

Usage:
    python scripts/investigate_xfreq_correlation.py --station UMNQ --year 2025
    python scripts/investigate_xfreq_correlation.py --station ROSS --year 2024
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
    detrend_arc, _snr_file_path, _load_station_config,
    FREQ_TO_COL, FREQ_WAVELENGTH,
)

log = logging.getLogger(__name__)

# Pairs of interest: (freq_a, freq_b, label)
# Within each constellation, pair L1 with other bands
PAIR_DEFS = [
    # GPS
    (1, 20, "GPS_L1_L2C"),
    (1, 5, "GPS_L1_L5"),
    # GLONASS
    (101, 102, "GLO_L1_L2"),
    # Galileo
    (201, 205, "GAL_L1_L5a"),
    (201, 206, "GAL_L1_E5b"),
    (201, 207, "GAL_L1_E5"),
    (201, 208, "GAL_L1_E6"),
]

# Build lookup: frozenset({fa, fb}) → label
PAIR_LABELS = {frozenset({fa, fb}): label for fa, fb, label in PAIR_DEFS}


def compute_xfreq_corr(sin_e, dsnr_a, dsnr_b, wl_a, wl_b, n_interp=200):
    """Compute cross-frequency correlation between two detrended signals.

    Rescales both to sin(ε)/(λ/2) domain, interpolates onto a common grid,
    then computes Pearson correlation.

    Args:
        sin_e: sin(elevation) array (shared by both bands)
        dsnr_a, dsnr_b: detrended SNR arrays for bands a and b
        wl_a, wl_b: carrier wavelengths in meters
        n_interp: number of points in common interpolation grid

    Returns:
        float: Pearson correlation coefficient, or NaN if computation fails
    """
    if len(sin_e) < 20:
        return np.nan

    cf_a = wl_a / 2
    cf_b = wl_b / 2

    x_a = sin_e / cf_a
    x_b = sin_e / cf_b

    # Common grid in overlapping region
    x_lo = max(x_a.min(), x_b.min())
    x_hi = min(x_a.max(), x_b.max())
    if x_hi <= x_lo:
        return np.nan

    x_common = np.linspace(x_lo, x_hi, n_interp)

    # Interpolate both onto common grid
    y_a = np.interp(x_common, x_a, dsnr_a)
    y_b = np.interp(x_common, x_b, dsnr_b)

    # Pearson correlation
    if np.std(y_a) < 1e-10 or np.std(y_b) < 1e-10:
        return np.nan

    r = np.corrcoef(y_a, y_b)[0, 1]
    return float(r)


def process_station_year(station, year, max_doys=None):
    """Process all multi-frequency arcs for one station-year.

    Returns DataFrame with one row per frequency pair per physical arc.
    """
    config = _load_station_config(station)
    if config is None:
        log.error(f"No config for {station}")
        return None

    e1, e2 = config["e1"], config["e2"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))

    at = pd.read_parquet(
        PROJECT_ROOT / "results_annual" / station
        / f"{station}_{year}_arc_table.parquet"
    )
    log.info(f"Loaded {len(at)} arcs for {station} {year}")

    # Group by physical arc
    arc_groups = at.groupby(["doy", "sat", "UTCtime", "rise"])
    # Keep only multi-freq arcs
    multi_groups = {k: g for k, g in arc_groups if g["freq"].nunique() > 1}
    log.info(f"Multi-freq physical arcs: {len(multi_groups)}")

    doys = sorted({k[0] for k in multi_groups})
    if max_doys is not None:
        doys = doys[:max_doys]
        multi_groups = {k: g for k, g in multi_groups.items() if k[0] in set(doys)}
        log.info(f"Limited to {len(doys)} DOYs, {len(multi_groups)} arcs")

    results = []
    n_processed = 0
    n_failed = 0
    last_snr_doy = None
    snr_data = None

    for (doy, sat, utctime, rise), freq_rows in sorted(multi_groups.items()):
        # Read SNR file (cache by DOY)
        if doy != last_snr_doy:
            snr_path = _snr_file_path(station, year, doy)
            if not snr_path.exists():
                gz = Path(str(snr_path) + ".gz")
                if gz.exists():
                    snr_path = gz
                else:
                    n_failed += len(freq_rows)
                    continue
            try:
                snr_data = read_snr_file(snr_path)
            except Exception:
                snr_data = None
                n_failed += len(freq_rows)
                continue
            last_snr_doy = doy

        if snr_data is None:
            continue

        # Get arc segments for this satellite
        sat_mask = snr_data[:, 0] == int(sat)
        sat_data = snr_data[sat_mask]
        if len(sat_data) < 5:
            continue

        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime, target_rise=rise, e1=e1, e2=e2,
        )
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        # Extract detrended signals for all available frequencies
        signals = {}  # freq → (sin_e_windowed, dsnr_windowed)
        for _, row in freq_rows.iterrows():
            freq = int(row["freq"])
            col_idx = FREQ_TO_COL.get(freq)
            wl = FREQ_WAVELENGTH.get(freq)
            if col_idx is None or wl is None or col_idx >= arc_data.shape[1]:
                continue

            snr_db_col = arc_data[:, col_idx]
            if np.all(snr_db_col == 0):
                continue

            snr_lin = np.power(10, snr_db_col / 20)
            detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

            mask = (arc_ele >= e1) & (arc_ele <= e2)
            if mask.sum() < 20:
                continue

            sin_e = np.sin(np.radians(arc_ele[mask]))
            dsnr = detrended[mask]
            signals[freq] = (sin_e, dsnr)

        # Compute cross-frequency correlations for all defined pairs
        freqs_available = set(signals.keys())
        for fa, fb, pair_label in PAIR_DEFS:
            if fa not in freqs_available or fb not in freqs_available:
                continue

            sin_e_a, dsnr_a = signals[fa]
            sin_e_b, dsnr_b = signals[fb]

            # Both signals share the same arc_ele, so sin_e should be identical
            # Use the common mask (already applied)
            wl_a = FREQ_WAVELENGTH[fa]
            wl_b = FREQ_WAVELENGTH[fb]

            r = compute_xfreq_corr(sin_e_a, dsnr_a, dsnr_b, wl_a, wl_b)

            # Get arc metadata
            row_a = freq_rows[freq_rows["freq"] == fa].iloc[0]
            results.append({
                "doy": doy, "sat": int(sat), "UTCtime": utctime,
                "rise": int(rise), "pair": pair_label,
                "xfreq_corr": r,
                "date": row_a.get("date"),
                "Azim": float(row_a.get("Azim", np.nan)),
                "azimuth_bin": row_a.get("azimuth_bin", np.nan),
                "RH_a": float(row_a.get("RH", np.nan)),
                "CLR_a": float(row_a.get("CLR", np.nan)) if "CLR" in row_a.index else np.nan,
                "AF_a": float(row_a.get("AF", np.nan)) if "AF" in row_a.index else np.nan,
                "gamma_a": float(row_a.get("gamma", np.nan)) if "gamma" in row_a.index else np.nan,
            })
            n_processed += 1

        if n_processed % 2000 == 0 and n_processed > 0:
            log.info(f"  Processed {n_processed} pairs...")

    log.info(f"Done: {n_processed} pairs, {n_failed} failed")

    if not results:
        return None
    return pd.DataFrame(results)


def make_diagnostics(df, station, year, out_dir):
    """Produce diagnostic figures from cross-frequency correlation results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add month column
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    # Load v3 classification if available for ice/water labels
    v3_path = (PROJECT_ROOT / "results_annual" / station
               / f"{station}_{year}_ice_classification_v3.parquet")
    v3_labels = None
    if v3_path.exists():
        v3 = pd.read_parquet(v3_path)
        v3["doy"] = pd.to_datetime(v3["date"]).dt.dayofyear
        v3_labels = dict(zip(v3["doy"], v3["v3_state"]))
        df["v3_state"] = df["doy"].map(v3_labels).fillna("unknown")
    else:
        df["v3_state"] = "unknown"

    pairs = sorted(df["pair"].unique())

    # --- Figure 1: Time series of daily median xfreq_corr per pair ---
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3 * len(pairs)),
                             sharex=True)
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        sub = df[df["pair"] == pair]
        daily = sub.groupby("doy")["xfreq_corr"].agg(["median", "std", "count"])
        daily = daily[daily["count"] >= 3]

        ax.plot(daily.index, daily["median"], ".", ms=2, alpha=0.7)
        ax.fill_between(daily.index,
                        daily["median"] - daily["std"],
                        daily["median"] + daily["std"],
                        alpha=0.2)
        ax.set_ylabel("Corr")
        ax.set_title(f"{pair} (n={len(sub)})", fontsize=10)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", lw=0.5, ls="--")

        # Shade ice periods if v3 available
        if v3_labels is not None:
            doys_ice = [d for d, s in v3_labels.items()
                        if s in ("ice_surface", "ice_layered", "ice_decaying",
                                 "anomalous")]
            if doys_ice:
                for d in doys_ice:
                    ax.axvspan(d - 0.5, d + 0.5, alpha=0.1, color="blue",
                               lw=0)

    axes[-1].set_xlabel("DOY")
    fig.suptitle(f"{station} {year}: Cross-frequency correlation time series",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_{year}_xfreq_corr_timeseries.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved timeseries plot")

    # --- Figure 2: xfreq_corr vs delta_rh_mean (if available) ---
    # Compute per-arc delta_rh from the pair RH values
    # For now, just distribution by v3_state
    if df["v3_state"].nunique() > 1:
        states_of_interest = ["open_water", "ice_surface", "ice_layered",
                              "ice_decaying", "baseline", "anomalous"]
        active_states = [s for s in states_of_interest if s in df["v3_state"].values]

        if active_states:
            fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4),
                                     sharey=True)
            if len(pairs) == 1:
                axes = [axes]

            for ax, pair in zip(axes, pairs):
                sub = df[df["pair"] == pair]
                box_data = []
                box_labels = []
                for state in active_states:
                    vals = sub[sub["v3_state"] == state]["xfreq_corr"].dropna()
                    if len(vals) >= 10:
                        box_data.append(vals.values)
                        box_labels.append(f"{state}\n(n={len(vals)})")

                if box_data:
                    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                    ax.set_title(pair, fontsize=10)
                    ax.set_ylim(-1, 1)
                    ax.axhline(0, color="gray", lw=0.5, ls="--")

            axes[0].set_ylabel("Cross-freq correlation")
            fig.suptitle(f"{station} {year}: xfreq_corr by surface state",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(out_dir / f"{station}_{year}_xfreq_corr_by_state.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved state boxplot")

    # --- Figure 3: xfreq_corr vs CLR and PR ---
    for feat_name in ["CLR_a", "AF_a", "gamma_a"]:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4),
                                 sharey=True)
        if len(pairs) == 1:
            axes = [axes]

        for ax, pair in zip(axes, pairs):
            sub = df[df["pair"] == pair].dropna(subset=["xfreq_corr", feat_name])
            if len(sub) < 10:
                ax.set_title(f"{pair}: insufficient data")
                continue

            ax.scatter(sub[feat_name], sub["xfreq_corr"], s=1, alpha=0.2)
            ax.set_xlabel(feat_name.replace("_a", ""))
            ax.set_title(f"{pair} (r={sub['xfreq_corr'].corr(sub[feat_name]):.3f})")
            ax.set_ylim(-1, 1)

        axes[0].set_ylabel("Cross-freq correlation")
        fig.suptitle(f"{station} {year}: xfreq_corr vs {feat_name.replace('_a', '')}",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{station}_{year}_xfreq_corr_vs_{feat_name.replace('_a','')}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    log.info(f"Saved scatter plots")

    # --- Summary statistics ---
    print(f"\n{'='*70}")
    print(f"Cross-frequency correlation summary: {station} {year}")
    print(f"{'='*70}")
    for pair in pairs:
        sub = df[df["pair"] == pair]
        valid = sub["xfreq_corr"].dropna()
        print(f"\n{pair}: {len(valid)} arcs")
        print(f"  Median: {valid.median():.3f}")
        print(f"  Mean:   {valid.mean():.3f} ± {valid.std():.3f}")
        print(f"  Q25-Q75: {valid.quantile(0.25):.3f} – {valid.quantile(0.75):.3f}")

        if df["v3_state"].nunique() > 1:
            for state in sorted(sub["v3_state"].unique()):
                s_vals = sub[sub["v3_state"] == state]["xfreq_corr"].dropna()
                if len(s_vals) >= 5:
                    print(f"  {state:>15s}: {s_vals.median():.3f} "
                          f"(n={len(s_vals)})")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-frequency SNR correlation diagnostic"
    )
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--max_doys", type=int, default=None,
                        help="Limit to first N DOYs for quick test")
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

    # Save raw results
    out_dir = PROJECT_ROOT / "results_annual" / args.station / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.station}_{args.year}_xfreq_corr.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} results to {out_path}")

    # Diagnostics
    make_diagnostics(df, args.station, args.year, out_dir)


if __name__ == "__main__":
    main()
