# ABOUTME: Match S1 RTC acquisitions to GNSS-IR per-arc data with time-aware windowing
# ABOUTME: Computes amp_cv and RH stats from arcs near the S1 overpass time, not daily aggregates

"""
Match S1 acquisition times to GNSS-IR per-arc data.

For each S1 scene, computes GNSS-IR statistics from arcs within ±2 hours
of the S1 overpass time on the same date. This ensures we're comparing
surface conditions at the same moment, not daily averages.

Usage:
    python scripts/s1_gnssir_match.py --station UMNQ --year 2025
    python scripts/s1_gnssir_match.py --station NKAR --year 2025 --window-hours 2
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.s1_fresnel_utils import (
    get_s1_index_path,
    parse_opera_rtc_scene_name,
)

logger = logging.getLogger(__name__)


def _parse_utc_hour_from_scene(scene_name):
    """Extract UTC hour (float) from scene name."""
    m = re.search(r"_(\d{8}T(\d{2})(\d{2})\d{2})Z_", scene_name)
    if m:
        return int(m.group(2)) + int(m.group(3)) / 60.0
    return None


def compute_arc_stats_near_time(arcs, utc_hour, window_hours=2):
    """Compute amplitude and RH stats from arcs within ±window of utc_hour.

    Returns dict with stats, or None if no arcs in window.
    """
    # Handle midnight wrap (e.g., pass at 23:30, window includes 0:00-1:30)
    t_min = utc_hour - window_hours
    t_max = utc_hour + window_hours

    if t_min < 0:
        mask = (arcs["UTCtime"] >= (t_min + 24)) | (arcs["UTCtime"] <= t_max)
    elif t_max > 24:
        mask = (arcs["UTCtime"] >= t_min) | (arcs["UTCtime"] <= (t_max - 24))
    else:
        mask = (arcs["UTCtime"] >= t_min) & (arcs["UTCtime"] <= t_max)

    window_arcs = arcs[mask]
    if len(window_arcs) < 3:
        return None

    amp = window_arcs["Amp"]
    rh = window_arcs["RH"]

    stats = {
        "gnssir_arc_count": len(window_arcs),
        "gnssir_window_hours": window_hours,
        "gnssir_amp_mean": float(amp.mean()),
        "gnssir_amp_std": float(amp.std()),
        "gnssir_amp_cv": float(amp.std() / amp.mean()) if amp.mean() > 0 else np.nan,
        "gnssir_rh_mean": float(rh.mean()),
        "gnssir_rh_std": float(rh.std()),
        "gnssir_rh_range": float(rh.max() - rh.min()),
    }

    # Per-azimuth stats
    if "azimuth_bin" in window_arcs.columns:
        for b in sorted(window_arcs["azimuth_bin"].unique()):
            bin_arcs = window_arcs[window_arcs["azimuth_bin"] == b]
            if len(bin_arcs) >= 2:
                b_amp = bin_arcs["Amp"]
                stats[f"gnssir_amp_cv_az{b}"] = float(b_amp.std() / b_amp.mean()) if b_amp.mean() > 0 else np.nan
                stats[f"gnssir_arc_count_az{b}"] = len(bin_arcs)

    return stats


def match_s1_to_gnssir(station, year, window_hours=2):
    """Match each S1 scene to GNSS-IR arcs near its overpass time.

    Returns one row per S1 scene with both S1 stats and time-windowed GNSS-IR stats.
    """
    results_dir = PROJECT_ROOT / "results_annual" / station

    # Load per-arc data (need individual arcs for time windowing)
    per_arc_path = results_dir / f"{station}_{year}_per_arc.parquet"
    if not per_arc_path.exists():
        logger.error(f"Per-arc parquet not found: {per_arc_path}")
        return None

    per_arc = pd.read_parquet(per_arc_path)
    logger.info(f"Loaded per-arc: {len(per_arc)} arcs, {per_arc['date'].nunique()} dates")

    # Also load daily enriched for full-day stats (for comparison)
    enriched_path = results_dir / f"{station}_{year}_daily_enriched.parquet"
    enriched = None
    if enriched_path.exists():
        enriched = pd.read_parquet(enriched_path)

    # Load S1 index
    s1_index_path = get_s1_index_path(station)
    if not s1_index_path.exists():
        logger.error(f"S1 index not found: {s1_index_path}")
        return None

    s1_index = pd.read_csv(s1_index_path)
    s1_index["acquisition_date_dt"] = pd.to_datetime(s1_index["acquisition_date"])

    # Filter to requested year
    s1_year = s1_index[s1_index["acquisition_date_dt"].dt.year == year].copy()
    logger.info(f"S1 scenes for {year}: {len(s1_year)}")

    if s1_year.empty:
        logger.warning(f"No S1 data for {year}")
        return None

    # Parse UTC hour from scene names
    s1_year["s1_utc_hour"] = s1_year["scene_name"].apply(_parse_utc_hour_from_scene)

    # Match each S1 scene to GNSS-IR arcs
    matched_rows = []
    for _, s1_row in s1_year.iterrows():
        acq_date = s1_row["acquisition_date"]
        utc_hour = s1_row["s1_utc_hour"]

        if utc_hour is None:
            continue

        # Get arcs for same date
        day_arcs = per_arc[per_arc["date"] == acq_date]

        # If no arcs on exact date, try ±1 day (S1 might be just after midnight)
        if day_arcs.empty:
            acq_dt = pd.Timestamp(acq_date)
            for offset in [-1, 1]:
                alt_date = (acq_dt + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                day_arcs = per_arc[per_arc["date"] == alt_date]
                if not day_arcs.empty:
                    acq_date = alt_date
                    break

        if day_arcs.empty:
            continue

        # Compute time-windowed stats
        window_stats = compute_arc_stats_near_time(day_arcs, utc_hour, window_hours)
        if window_stats is None:
            continue

        # Also compute full-day stats for comparison
        full_day_amp = day_arcs["Amp"]
        full_day_cv = float(full_day_amp.std() / full_day_amp.mean()) if full_day_amp.mean() > 0 else np.nan

        # Get daily enriched pooled stats if available
        daily_amp_cv = np.nan
        if enriched is not None:
            pooled = enriched[
                (enriched["date"] == acq_date)
                & (enriched["azimuth_bin"] == -1)
                & (enriched["freq_group"] == "ALL")
            ]
            if len(pooled) > 0:
                daily_amp_cv = pooled.iloc[0]["amp_cv"]

        row = {
            # S1 metadata
            "acquisition_date": s1_row["acquisition_date"],
            "s1_utc_hour": utc_hour,
            "scene_name": s1_row["scene_name"],
            "burst_id": s1_row.get("burst_id", ""),
            "flight_direction": s1_row.get("flight_direction", ""),
            "file_path": s1_row["file_path"],
            # S1 backscatter (whole-clip — will be replaced by sector stats in dashboard)
            "s1_mean_hh_db": s1_row["mean_hh_db"],
            "s1_mean_hv_db": s1_row["mean_hv_db"],
            "s1_hh_hv_ratio_db": s1_row.get("hh_hv_ratio_db", np.nan),
            # Time-windowed GNSS-IR stats (±window_hours of S1 pass)
            **window_stats,
            # Full-day stats for comparison
            "gnssir_fullday_amp_cv": full_day_cv,
            "gnssir_fullday_arc_count": len(day_arcs),
            "gnssir_daily_enriched_amp_cv": daily_amp_cv,
        }
        matched_rows.append(row)

    if not matched_rows:
        logger.warning("No matches found")
        return None

    result = pd.DataFrame(matched_rows)
    logger.info(f"Matched {len(result)} S1 scenes to GNSS-IR arcs (±{window_hours}hr window)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Match S1 to GNSS-IR per-arc data (time-aware)")
    parser.add_argument("--station", required=True, help="Station ID (e.g., UMNQ)")
    parser.add_argument("--year", required=True, type=int, help="Year (e.g., 2025)")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=2.0,
        help="Hours before/after S1 pass to include GNSS-IR arcs (default: 2)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = match_s1_to_gnssir(args.station, args.year, args.window_hours)

    if result is None:
        sys.exit(1)

    # Save
    output_path = (
        PROJECT_ROOT
        / "results_annual"
        / args.station
        / f"{args.station}_{args.year}_s1_gnssir_matched.parquet"
    )
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved: {output_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Matched {len(result)} S1 scenes")

    # Compare time-windowed vs full-day amp_cv
    valid = result.dropna(subset=["gnssir_amp_cv", "gnssir_fullday_amp_cv"])
    if len(valid) > 0:
        logger.info(f"\nTime-windowed amp_cv (±{args.window_hours}hr):")
        logger.info(f"  Range: [{valid['gnssir_amp_cv'].min():.3f}, {valid['gnssir_amp_cv'].max():.3f}]")
        logger.info(f"  Mean arcs/window: {valid['gnssir_arc_count'].mean():.0f}")
        logger.info(f"\nFull-day amp_cv:")
        logger.info(f"  Range: [{valid['gnssir_fullday_amp_cv'].min():.3f}, {valid['gnssir_fullday_amp_cv'].max():.3f}]")

        # Correlation with S1
        for label, col in [("time-windowed", "gnssir_amp_cv"), ("full-day", "gnssir_fullday_amp_cv")]:
            v = valid.dropna(subset=[col, "s1_mean_hh_db"])
            if len(v) >= 5:
                corr = v[col].corr(v["s1_mean_hh_db"])
                logger.info(f"\n  {label} amp_cv vs S1 HH: r = {corr:.3f} (n={len(v)})")

    # Per flight direction
    for fd in result["flight_direction"].unique():
        fd_data = result[result["flight_direction"] == fd]
        logger.info(f"\n{fd} passes ({len(fd_data)} scenes, ~{fd_data['s1_utc_hour'].mean():.1f} UTC):")
        v = fd_data.dropna(subset=["gnssir_amp_cv", "s1_mean_hh_db"])
        if len(v) >= 5:
            corr = v["gnssir_amp_cv"].corr(v["s1_mean_hh_db"])
            logger.info(f"  amp_cv vs S1 HH: r = {corr:.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
