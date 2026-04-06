# ABOUTME: Layer 2 feature aggregator — produces daily_features.parquet from arc_table
# ABOUTME: Aggregates per-arc observables to daily × sector resolution with z-score normalization,
# ABOUTME: matched-arc ΔRH, circular phase stats, and interfrequency spread

"""
Daily feature aggregator for GNSS-IR arc-level data (Layer 2).

Reads arc_table.parquet (Layer 1 output) and station config, computes
daily × azimuth-sector aggregates of all physical observables, then
writes daily_features.parquet.

This is the central feature table consumed by ice classifiers, dashboards,
and notebooks. It replaces transient computations previously scattered
across ice_classifier._extract_indicators(), generate_ice_notebook.py cells,
and ad-hoc dashboard code.

Usage:
    python scripts/feature_aggregator.py --station ROSS --year 2022
    python scripts/feature_aggregator.py --all
    python scripts/feature_aggregator.py --all --force --log-level DEBUG
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

logger = logging.getLogger(__name__)

# SNR feature columns eligible for z-score normalization
_ZSCORE_FEATURES = ["CLR", "AF", "PR", "gamma"]

# SNR feature columns to compute medians for (raw + z-scored)
_SNR_FEATURE_COLS = ["CLR", "PR", "AF", "gamma", "MS", "VS"]

# Minimum arcs per satellite to compute z-score reference stats
_MIN_REF_ARCS = 10


# ---------------------------------------------------------------------------
# Station config loading
# ---------------------------------------------------------------------------

def _load_station_config(station):
    """Load station entry from stations_config.json."""
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        all_cfg = json.load(f)
    return all_cfg.get(station, {})


# ---------------------------------------------------------------------------
# Per-satellite z-score normalization
# ---------------------------------------------------------------------------

def normalize_features_per_satellite(arc_table, feature_cols=None,
                                     ice_free_months=None):
    """Z-score normalize SNR features per satellite PRN.

    For each feature, compute mean and std per satellite from ice-free
    months (or full year if not specified), then apply (value - mean) / std.
    Writes new columns with '_z' suffix (e.g., CLR_z, AF_z).

    Satellites only seen during ice months fall back to full-year stats.
    Satellites with < 10 reference arcs are left un-normalized (NaN in _z column).

    Args:
        arc_table: DataFrame with per-arc data (must have 'date', 'sat' columns)
        feature_cols: list of column names to normalize (default: CLR, AF, PR, gamma)
        ice_free_months: list of month ints for reference period, or None for full year

    Returns:
        arc_table with added _z columns (modified in place for efficiency).
    """
    if feature_cols is None:
        feature_cols = [c for c in _ZSCORE_FEATURES if c in arc_table.columns]

    if not feature_cols:
        return arc_table

    # Compute month for filtering
    months = pd.to_datetime(arc_table["date"]).dt.month

    if ice_free_months:
        ref_mask = months.isin(ice_free_months)
        ref_data = arc_table[ref_mask]
    else:
        ref_data = arc_table

    for col in feature_cols:
        if col not in arc_table.columns:
            continue

        z_col = f"{col}_z"
        arc_table[z_col] = np.nan

        sat_stats = ref_data.groupby("sat")[col].agg(["mean", "std", "count"])

        # Fall back to full-year stats for satellites only seen during ice months
        if ice_free_months:
            all_sats = arc_table["sat"].unique()
            missing_sats = set(all_sats) - set(sat_stats.index)
            if missing_sats:
                fallback = arc_table.groupby("sat")[col].agg(
                    ["mean", "std", "count"]
                )
                for sat in missing_sats:
                    if sat in fallback.index:
                        sat_stats.loc[sat] = fallback.loc[sat]
                        logger.debug(
                            f"Satellite {sat}: no ice-free data for {col}, "
                            f"using full-year stats"
                        )

        normalized = 0
        skipped = 0
        for sat, row in sat_stats.iterrows():
            mask = arc_table["sat"] == sat
            if row["count"] < _MIN_REF_ARCS:
                skipped += 1
                continue
            if row["std"] > 0:
                arc_table.loc[mask, z_col] = (
                    (arc_table.loc[mask, col] - row["mean"]) / row["std"]
                )
            else:
                arc_table.loc[mask, z_col] = 0.0
            normalized += 1

        logger.debug(
            f"Z-score {col}: {normalized} satellites normalized, "
            f"{skipped} skipped (<{_MIN_REF_ARCS} arcs)"
        )

    return arc_table


# ---------------------------------------------------------------------------
# Matched-arc ΔRH (cross-frequency)
# ---------------------------------------------------------------------------

def compute_matched_delta_rh(sector_arcs):
    """Compute ΔRH from matched satellite passes across frequency bands.

    Groups by (sat, rise) to compare L1 and L2 reflections from the exact
    same satellite at the exact same time. This avoids the seiche/tide trap
    where daily averaging mixes different water levels.

    Args:
        sector_arcs: DataFrame with columns [sat, rise, freq_group, RH]

    Returns:
        (delta_rh_mean, delta_rh_std, n_pairs) — NaN/NaN/0 if no matched pairs
    """
    drh_values = []
    for (sat, rise), pass_arcs in sector_arcs.groupby(["sat", "rise"]):
        band_rh = pass_arcs.groupby("freq_group")["RH"].median()
        if len(band_rh) >= 2:
            drh_values.append(float(band_rh.std()))

    if not drh_values:
        return np.nan, np.nan, 0

    return float(np.mean(drh_values)), float(np.std(drh_values)), len(drh_values)


# ---------------------------------------------------------------------------
# Phase circular statistics
# ---------------------------------------------------------------------------

def _compute_phase_stats(phase_values):
    """Compute circular mean and std of phase values.

    Args:
        phase_values: array-like of phase in radians

    Returns:
        (circ_mean, circ_std) or (NaN, NaN) if insufficient data
    """
    vals = np.asarray(phase_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan

    from scipy.stats import circmean, circstd
    cm = circmean(vals, high=np.pi, low=-np.pi)
    cs = circstd(vals, high=np.pi, low=-np.pi)
    return float(cm), float(cs)


def compute_differential_phase(sector_arcs):
    """Compute L1 − L2C differential phase for matched satellite passes.

    Args:
        sector_arcs: DataFrame with columns [sat, rise, freq_group, phase]

    Returns:
        (diff_phase_mean, diff_phase_std, n_pairs)
    """
    if "phase" not in sector_arcs.columns:
        return np.nan, np.nan, 0

    from scipy.stats import circmean

    diff_values = []
    for (sat, rise), pass_arcs in sector_arcs.groupby(["sat", "rise"]):
        band_phase = {}
        for fg, fg_arcs in pass_arcs.groupby("freq_group"):
            valid = fg_arcs["phase"].dropna()
            if len(valid) > 0:
                band_phase[fg] = circmean(valid.values, high=np.pi, low=-np.pi)

        if "L1" in band_phase and "L2C" in band_phase:
            diff = band_phase["L1"] - band_phase["L2C"]
            # Wrap to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            diff_values.append(float(diff))

    if not diff_values:
        return np.nan, np.nan, 0

    arr = np.array(diff_values)
    return float(np.mean(arr)), float(np.std(arr)), len(arr)


# ---------------------------------------------------------------------------
# Per-sector daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_sector(sector_arcs, has_snr, has_phase, has_zscore):
    """Aggregate one (date, azimuth_bin) group into a feature row.

    Returns a dict with all computed features for this sector-day.
    """
    row = {}
    n = len(sector_arcs)
    row["n_arcs"] = n
    row["n_sats"] = int(sector_arcs["sat"].nunique())

    if "full_arc" in sector_arcs.columns:
        fa = sector_arcs["full_arc"]
        row["n_full_arcs"] = int(fa.sum())
        row["frac_full_arc"] = float(fa.mean())
    else:
        row["n_full_arcs"] = np.nan
        row["frac_full_arc"] = np.nan

    # RH stats
    rh = sector_arcs["RH"]
    row["rh_mean"] = float(rh.mean())
    row["rh_median"] = float(rh.median())
    row["rh_std"] = float(rh.std())
    row["rh_range"] = float(rh.max() - rh.min())
    row["rh_count"] = n

    # Amplitude stats
    amp = sector_arcs["Amp"]
    amp_mean = float(amp.mean())
    row["amp_mean"] = amp_mean
    row["amp_std"] = float(amp.std())
    row["amp_cv"] = float(amp.std() / amp_mean) if amp_mean > 0 else np.nan

    # PkNoise stats
    if "PkNoise" in sector_arcs.columns:
        pk = sector_arcs["PkNoise"]
        row["p2n_mean"] = float(pk.mean())
        row["p2n_std"] = float(pk.std())

    # WSE stats
    if "wse" in sector_arcs.columns:
        wse = sector_arcs["wse"].dropna()
        if len(wse) > 0:
            row["wse_mean"] = float(wse.mean())
            row["wse_std"] = float(wse.std())

    # SNR feature medians (raw)
    if has_snr:
        for col in _SNR_FEATURE_COLS:
            if col in sector_arcs.columns:
                vals = sector_arcs[col].dropna()
                row[f"{col.lower()}_med"] = float(vals.median()) if len(vals) > 0 else np.nan

    # SNR feature medians (z-scored)
    if has_zscore:
        for col in _ZSCORE_FEATURES:
            z_col = f"{col}_z"
            if z_col in sector_arcs.columns:
                vals = sector_arcs[z_col].dropna()
                row[f"{col.lower()}_z"] = float(vals.median()) if len(vals) > 0 else np.nan

    # Phase circular stats
    if has_phase and "phase" in sector_arcs.columns:
        phase_vals = sector_arcs["phase"].dropna()
        if len(phase_vals) >= 2:
            cm, cs = _compute_phase_stats(phase_vals.values)
            row["phase_circ_mean"] = cm
            row["phase_circ_std"] = cs

        # Differential phase (L1 - L2C)
        if "freq_group" in sector_arcs.columns:
            dp_mean, dp_std, dp_n = compute_differential_phase(sector_arcs)
            row["diff_phase_L1_L2"] = dp_mean
            row["diff_phase_std"] = dp_std
            row["diff_phase_n_pairs"] = dp_n

    # Matched-arc ΔRH
    if "freq_group" in sector_arcs.columns:
        drh_mean, drh_std, drh_n = compute_matched_delta_rh(sector_arcs)
        row["delta_rh_mean"] = drh_mean
        row["delta_rh_std"] = drh_std
        row["delta_rh_n_pairs"] = drh_n

    # Per-band RH medians
    if "freq_group" in sector_arcs.columns:
        for band, band_arcs in sector_arcs.groupby("freq_group"):
            if band == "OTHER":
                continue
            row[f"rh_{band}_median"] = float(band_arcs["RH"].median())
            row[f"rh_{band}_count"] = len(band_arcs)

    return row


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_features(station, year, results_dir=None):
    """Produce daily_features.parquet from arc_table.parquet.

    Aggregates per-arc data to daily × azimuth_bin resolution.
    Includes pooled rows (azimuth_bin = -1) for station-level daily values.

    Args:
        station: Station ID (e.g., "ROSS")
        year: Processing year
        results_dir: Path to results directory (default: results_annual/{station})

    Returns:
        Path to daily_features.parquet, or None on failure.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results_annual" / station
    results_dir = Path(results_dir)

    arc_path = results_dir / f"{station}_{year}_arc_table.parquet"
    out_path = results_dir / f"{station}_{year}_daily_features.parquet"

    if not arc_path.exists():
        logger.error(f"arc_table not found: {arc_path}")
        return None

    arc_table = pd.read_parquet(arc_path)
    logger.info(f"Loaded arc_table: {len(arc_table)} arcs, "
                f"{arc_table['date'].nunique()} days")

    # Detect available columns
    has_snr = "CLR" in arc_table.columns
    has_phase = "phase" in arc_table.columns
    has_freq_group = "freq_group" in arc_table.columns

    if has_snr:
        logger.info("SNR features detected — will compute feature medians and z-scores")
    else:
        logger.info("No SNR features — computing RH/Amp/PkNoise stats only")

    # Load station config for ice_free_months
    station_cfg = _load_station_config(station)
    ice_free_months = station_cfg.get("ice_free_months")

    # Z-score normalization (adds _z columns to arc_table)
    has_zscore = False
    if has_snr:
        normalize_features_per_satellite(
            arc_table,
            feature_cols=[c for c in _ZSCORE_FEATURES if c in arc_table.columns],
            ice_free_months=ice_free_months,
        )
        has_zscore = any(f"{c}_z" in arc_table.columns for c in _ZSCORE_FEATURES)
        if has_zscore:
            logger.info(f"Z-score normalization applied "
                        f"(ice_free_months={ice_free_months})")

    # Group by (date, azimuth_bin) and aggregate
    dates = sorted(arc_table["date"].unique())
    az_bins = sorted(arc_table["azimuth_bin"].unique())
    logger.info(f"Aggregating: {len(dates)} days × {len(az_bins)} sectors")

    rows = []
    for date in dates:
        day_arcs = arc_table[arc_table["date"] == date]

        # Per-sector rows
        for az_bin in az_bins:
            sector = day_arcs[day_arcs["azimuth_bin"] == az_bin]
            if len(sector) == 0:
                continue
            entry = _aggregate_sector(sector, has_snr, has_phase, has_zscore)
            entry["date"] = date
            entry["azimuth_bin"] = az_bin
            rows.append(entry)

        # Pooled row (azimuth_bin = -1): station-level daily values
        if len(day_arcs) > 0:
            pooled = _aggregate_sector(day_arcs, has_snr, has_phase, has_zscore)
            pooled["date"] = date
            pooled["azimuth_bin"] = -1

            # Station-level interfreq_spread: range of per-band RH medians
            if has_freq_group:
                band_rh = day_arcs.groupby("freq_group")["RH"].median()
                # Exclude OTHER
                band_rh = band_rh.drop("OTHER", errors="ignore")
                if len(band_rh) >= 2:
                    pooled["interfreq_spread"] = float(
                        band_rh.max() - band_rh.min()
                    )

            rows.append(pooled)

    if not rows:
        logger.error("No feature rows produced")
        return None

    daily_features = pd.DataFrame(rows)

    # Ensure key columns are first
    key_cols = ["date", "azimuth_bin"]
    other_cols = [c for c in daily_features.columns if c not in key_cols]
    daily_features = daily_features[key_cols + sorted(other_cols)]

    daily_features.to_parquet(out_path, index=False, engine="pyarrow")

    n_sectors = daily_features[daily_features["azimuth_bin"] >= 0]["azimuth_bin"].nunique()
    n_pooled = (daily_features["azimuth_bin"] == -1).sum()
    logger.info(
        f"daily_features saved: {out_path} "
        f"({len(daily_features)} rows: {len(daily_features) - n_pooled} sector + "
        f"{n_pooled} pooled, {n_sectors} sectors, "
        f"{out_path.stat().st_size / 1024:.0f} KB)"
    )

    # Summary stats
    pooled = daily_features[daily_features["azimuth_bin"] == -1]
    for col in ["rh_mean", "amp_mean", "clr_med", "gamma_med", "delta_rh_mean"]:
        if col in pooled.columns:
            vals = pooled[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col}: median={vals.median():.4f}, "
                            f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: aggregate arc_table → daily_features.parquet"
    )
    parser.add_argument("--station", help="Station ID (e.g., ROSS)")
    parser.add_argument("--year", type=int, help="Year (e.g., 2022)")
    parser.add_argument("--all", action="store_true",
                        help="Process all station-years with arc_table.parquet")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite even if daily_features already exists")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all:
        from scripts.results_handler import discover_station_years
        targets = discover_station_years(require_file="arc_table.parquet")
        logger.info(f"Found {len(targets)} station-years with arc_table.parquet")
    elif args.station and args.year:
        targets = [(args.station, args.year)]
    else:
        parser.error("Specify --station and --year, or use --all")

    success = 0
    for station, year in targets:
        results_dir = PROJECT_ROOT / "results_annual" / station
        out_path = results_dir / f"{station}_{year}_daily_features.parquet"
        if out_path.exists() and not args.force:
            logger.info(f"Skipping {station} {year} (daily_features exists, use --force)")
            continue
        result = aggregate_daily_features(station, year, results_dir=results_dir)
        if result is not None:
            success += 1

    logger.info(f"Done: {success}/{len(targets)} daily_features created")


if __name__ == "__main__":
    main()
