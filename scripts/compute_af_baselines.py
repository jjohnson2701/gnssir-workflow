# ABOUTME: Precompute per-PRN average CWT power curves from ice-free months for AF correction
# ABOUTME: Produces an optional npz file that snr_feature_extractor.py loads to subtract antenna gain patterns

"""
Precompute per-PRN average power curves from ice-free months for AF correction.

Reads SNR files from the configured ice-free months, computes CWT power curves
at the dominant RH for each satellite pass, and averages per (satellite, frequency)
to produce a baseline. The baseline is saved as an npz file that
snr_feature_extractor.py optionally loads.

Usage:
    python scripts/compute_af_baselines.py --station UMNQ --year 2025
    python scripts/compute_af_baselines.py --station UMNQ --year 2025 --max_arcs 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.snr_feature_extractor import (
    read_snr_file,
    segment_satellite_arcs,
    find_matching_segment,
    detrend_arc,
    compute_area_factor,
    _snr_file_path,
    _load_station_config,
    FREQ_TO_COL,
    FREQ_WAVELENGTH,
)

logger = logging.getLogger(__name__)

# Number of points in the common sin(ε) grid for baseline interpolation
SIN_GRID_SIZE = 50


def compute_baselines(station, year, max_arcs=500):
    """Compute per-PRN average CWT power curves from ice-free months.

    For each sampled arc, extracts the CWT power curve at the dominant RH
    and interpolates it onto a common sin(ε) grid. Power curves are then
    averaged per (satellite PRN, frequency) to produce the baseline.

    Args:
        station: station ID (e.g., "UMNQ")
        year: processing year
        max_arcs: maximum number of arcs to sample from ice-free months

    Returns:
        (sin_grid, baselines_dict) where baselines_dict maps
        (sat, freq) → averaged power curve array of shape (SIN_GRID_SIZE,).
        Returns (None, None) if insufficient data.
    """
    config = _load_station_config(station)
    if config is None:
        logger.error(f"No gnssir config found for {station}")
        return None, None

    # Load ice_free_months from station config
    stations_cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    ice_free_months = []
    if stations_cfg_path.exists():
        with open(stations_cfg_path) as f:
            all_cfg = json.load(f)
        ice_free_months = all_cfg.get(station, {}).get("ice_free_months", [])

    if not ice_free_months:
        logger.error(f"No ice_free_months configured for {station}")
        return None, None

    # Load per-arc parquet to identify summer arcs
    per_arc_path = (PROJECT_ROOT / "results_annual" / station
                    / f"{station}_{year}_per_arc.parquet")
    if not per_arc_path.exists():
        logger.error(f"Per-arc parquet not found: {per_arc_path}")
        return None, None

    per_arc = pd.read_parquet(per_arc_path)
    per_arc["month"] = pd.to_datetime(per_arc["date"]).dt.month
    summer_arcs = per_arc[per_arc["month"].isin(ice_free_months)]

    if len(summer_arcs) < 50:
        logger.error(f"Only {len(summer_arcs)} arcs in ice-free months — need ≥50")
        return None, None

    logger.info(f"Ice-free months {ice_free_months}: {len(summer_arcs)} arcs available")

    # Sample arcs
    sample = summer_arcs.sample(min(max_arcs, len(summer_arcs)), random_state=42)
    logger.info(f"Sampling {len(sample)} arcs for baseline computation")

    e1 = config["e1"]
    e2 = config["e2"]
    min_rh = config["minH"]
    max_rh = config["maxH"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))

    # Common sin(ε) grid for interpolation
    sin_grid = np.linspace(
        np.sin(np.radians(e1)),
        np.sin(np.radians(e2)),
        SIN_GRID_SIZE,
    )
    # Convert to sin(ε)/(λ/2) domain — but λ varies by freq.
    # Store baselines in the sin(ε) domain; compute_area_factor converts internally.

    # Accumulate power curves per (sat, freq)
    accumulator = {}  # (sat, freq) → list of power curve arrays

    # Group sample by DOY for efficient SNR file reads
    processed = 0
    failed = 0
    for doy, doy_rows in sample.groupby("doy"):
        snr_path = _snr_file_path(station, year, doy)
        if not snr_path.exists():
            gz_path = Path(str(snr_path) + ".gz")
            if gz_path.exists():
                snr_path = gz_path
            else:
                failed += len(doy_rows)
                continue

        try:
            snr_data = read_snr_file(snr_path)
        except Exception:
            failed += len(doy_rows)
            continue

        segment_cache = {}
        for _, row in doy_rows.iterrows():
            sat = int(row["sat"])
            freq = int(row["freq"])
            utctime = float(row["UTCtime"])
            rise = int(row["rise"])

            wavelength = FREQ_WAVELENGTH.get(freq)
            col_idx = FREQ_TO_COL.get(freq)
            if wavelength is None or col_idx is None:
                failed += 1
                continue

            # Segment satellite arcs (cached per PRN per DOY)
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
                failed += 1
                continue

            seg_idx = find_matching_segment(
                arcs, sat_data[:, 3], sat_data[:, 1],
                target_utctime=utctime, target_rise=rise,
                e1=e1, e2=e2,
            )
            if seg_idx < 0:
                failed += 1
                continue

            arc = arcs[seg_idx]
            arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
            arc_ele = arc_data[:, 1]

            if col_idx >= arc_data.shape[1]:
                failed += 1
                continue

            snr_db_col = arc_data[:, col_idx]
            if np.all(snr_db_col == 0):
                failed += 1
                continue

            snr_lin = np.power(10, snr_db_col / 20)
            detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

            # Window to [e1, e2]
            mask = (arc_ele >= e1) & (arc_ele <= e2)
            if mask.sum() < 15:
                failed += 1
                continue

            sin_e = np.sin(np.radians(arc_ele[mask]))
            dsnr = detrended[mask]

            # Get the power curve at dominant RH via return_power=True
            af, power_info = compute_area_factor(
                sin_e, dsnr, wavelength, min_rh, max_rh,
                return_power=True,
            )
            if power_info is None:
                failed += 1
                continue

            # Interpolate power curve onto common sin(ε) grid
            power_curve = power_info["power"][power_info["dominant_idx"], :]
            arc_sin_grid = power_info["sin_elev"]

            from scipy.interpolate import interp1d
            try:
                interp_fn = interp1d(
                    arc_sin_grid, power_curve,
                    bounds_error=False, fill_value=0.0,
                )
                interp_curve = interp_fn(sin_grid)
            except Exception:
                failed += 1
                continue

            key = (sat, freq)
            if key not in accumulator:
                accumulator[key] = []
            accumulator[key].append(interp_curve)
            processed += 1

    logger.info(f"Processed {processed} arcs, {failed} failed/skipped")

    if not accumulator:
        logger.error("No power curves collected — cannot compute baselines")
        return None, None

    # Average per (sat, freq)
    baselines = {}
    for key, curves in accumulator.items():
        baselines[key] = np.mean(curves, axis=0)

    logger.info(f"Computed baselines for {len(baselines)} (sat, freq) combinations")
    for key, curve in sorted(baselines.items()):
        n = len(accumulator[key])
        logger.debug(f"  PRN {key[0]} freq {key[1]}: {n} arcs, "
                     f"mean power {curve.mean():.2f}")

    return sin_grid, baselines


def save_baselines(station, year, sin_grid, baselines):
    """Save baselines to npz file.

    Args:
        station: station ID
        year: processing year
        sin_grid: common sin(ε) grid, shape (SIN_GRID_SIZE,)
        baselines: dict of (sat, freq) → power curve array
    """
    out_dir = PROJECT_ROOT / "results_annual" / station
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station}_{year}_af_baselines.npz"

    keys_array = np.array(list(baselines.keys()))
    baselines_array = np.array(list(baselines.values()))

    np.savez(
        out_path,
        sin_grid=sin_grid,
        keys=keys_array,
        baselines=baselines_array,
    )
    logger.info(f"Saved baselines to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-PRN AF baselines from ice-free months"
    )
    parser.add_argument("--station", required=True, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--max_arcs", type=int, default=500,
                        help="Maximum arcs to sample from ice-free months")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    sin_grid, baselines = compute_baselines(
        args.station, args.year, max_arcs=args.max_arcs
    )
    if baselines is None:
        sys.exit(1)

    out_path = save_baselines(args.station, args.year, sin_grid, baselines)

    print(f"\n{'='*60}")
    print(f"AF Baselines: {args.station} {args.year}")
    print(f"{'='*60}")
    print(f"Baselines computed for {len(baselines)} (sat, freq) combinations")
    print(f"Saved to: {out_path}")

    # Summary per constellation
    constellations = {}
    for (sat, freq), curve in baselines.items():
        if sat < 100:
            const = "GPS"
        elif sat < 200:
            const = "GLONASS"
        elif sat < 300:
            const = "Galileo"
        elif sat < 400:
            const = "BeiDou"
        else:
            const = "Other"
        constellations.setdefault(const, []).append((sat, freq))

    for const, keys in sorted(constellations.items()):
        print(f"  {const}: {len(keys)} baselines")


if __name__ == "__main__":
    main()
