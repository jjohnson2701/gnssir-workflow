# ABOUTME: Per-azimuth-sector ice/water/transition classifier for GNSS-IR stations
# ABOUTME: Classifies each azimuth sector independently, flags land contamination

"""
Per-sector ice classifier.

Classifies each azimuth sector independently rather than pooling all azimuths.
This prevents land-contaminated sectors (e.g., 50-60 deg at UMNQ) from drowning
out genuine ice signals in other sectors.

Thresholds are computed per-sector from the sector's own amplitude range.
Sectors with no seasonal amplitude change are flagged as potential land.

Usage:
    python scripts/ice_classifier.py --station UMNQ --year 2025
    python scripts/ice_classifier.py --station NKAR --year 2025 --no-s1
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

ICE = "ice"
TRANSITION = "transition"
WATER = "water"

# Azimuth bin labels
AZ_BIN_LABELS = {0: "0-90", 1: "90-180", 2: "180-270", 3: "270-360"}


def _vote(value, ice_thresh, water_thresh, high_is_ice=True):
    """Return classification vote for a single value."""
    if np.isnan(value):
        return None
    if high_is_ice:
        if value >= ice_thresh:
            return ICE
        elif value <= water_thresh:
            return WATER
        return TRANSITION
    else:
        if value <= ice_thresh:
            return ICE
        elif value >= water_thresh:
            return WATER
        return TRANSITION


def _score(vote):
    """ice=-1, transition=0, water=+1."""
    return {"ice": -1.0, "transition": 0.0, "water": 1.0}.get(vote, np.nan)


def _load_station_config(station):
    """Load the full station config dict from stations_config.json."""
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f).get(station, {})


def _load_suspect_azimuths(station_cfg):
    """Load suspect azimuth ranges from station config dict.

    Returns set of 10-degree azimuth bin lower bounds that are flagged.
    """
    suspect = station_cfg.get("suspect_azimuths", {})
    flagged_bins = set()
    for key, val in suspect.items():
        if key == "notes":
            continue
        if "-" in key:
            try:
                lo, hi = key.split("-")
                for az in range(int(lo), int(hi), 10):
                    flagged_bins.add(az)
            except ValueError:
                pass
    return flagged_bins


def compute_sector_thresholds(per_arc, az_bin, suspect_bins=None,
                              ice_free_months=None):
    """Compute thresholds for a single azimuth sector from its own data.

    When ice_free_months is provided, thresholds are anchored to the summer
    (ice-free) window. This prevents bias at stations with long ice seasons
    where full-year percentiles are dominated by ice values.

    Returns dict with amp/cv/rh thresholds, optional SNR feature thresholds,
    plus a land_flag.
    """
    sector = per_arc[per_arc["azimuth_bin"] == az_bin]
    if len(sector) < 100:
        return None

    config_flagged = suspect_bins is not None and az_bin in suspect_bins

    # Check for seasonal amplitude change (land detection)
    sector_copy = sector.copy()
    sector_copy["month"] = pd.to_datetime(sector_copy["date"]).dt.month
    monthly_amp = sector_copy.groupby("month")["Amp"].mean()
    if len(monthly_amp) >= 4:
        amp_ratio = monthly_amp.max() / monthly_amp.min() if monthly_amp.min() > 0 else 1.0
        daily_rh_range = sector.groupby("date")["RH"].agg(lambda x: x.max() - x.min())
        has_tidal = daily_rh_range.median() > 2.0
        land_flag = (amp_ratio < 1.15 and not has_tidal) or config_flagged
    else:
        amp_ratio = np.nan
        land_flag = config_flagged

    # Choose threshold source: summer anchor or full-year percentiles
    if ice_free_months and len(ice_free_months) > 0:
        summer = sector_copy[sector_copy["month"].isin(ice_free_months)]
        # Fall back to full sector if insufficient summer data
        thresh_source = summer if len(summer) >= 50 else sector_copy
        if len(summer) >= 50:
            logger.debug(f"  Sector {az_bin}: using summer-anchored thresholds "
                         f"({len(summer)} arcs from months {ice_free_months})")
    else:
        thresh_source = sector_copy

    # Amplitude mean thresholds
    daily_amp = thresh_source.groupby("date")["Amp"].mean()
    if len(daily_amp) >= 10:
        amp_p70 = float(daily_amp.quantile(0.70))
        amp_p30 = float(daily_amp.quantile(0.30))
    else:
        amp_p70 = float(sector.groupby("date")["Amp"].mean().quantile(0.70))
        amp_p30 = float(sector.groupby("date")["Amp"].mean().quantile(0.30))

    # Amplitude CV thresholds
    daily_cv = thresh_source.groupby("date")["Amp"].agg(
        lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) >= 3 else np.nan
    ).dropna()
    if len(daily_cv) >= 10:
        cv_p25 = float(daily_cv.quantile(0.25))
        cv_p75 = float(daily_cv.quantile(0.75))
    else:
        cv_p25 = 0.25
        cv_p75 = 0.40

    # RH std thresholds
    daily_rh_std = thresh_source.groupby("date")["RH"].std().dropna()
    if len(daily_rh_std) >= 10:
        rh_p30 = float(daily_rh_std.quantile(0.30))
        rh_p70 = float(daily_rh_std.quantile(0.70))
    else:
        rh_p30 = 0.4
        rh_p70 = 0.8

    result = {
        "amp_mean_ice": amp_p70,
        "amp_mean_water": amp_p30,
        "amp_cv_ice": cv_p25,
        "amp_cv_water": cv_p75,
        "rh_std_ice": rh_p30,
        "rh_std_water": rh_p70,
        "land_flag": land_flag,
        "seasonal_ratio": float(amp_ratio) if not np.isnan(amp_ratio) else None,
        "n_arcs": len(sector),
    }

    # SNR feature thresholds (if features are merged into per_arc)
    for feat_col in ["CLR", "AF", "PR", "gamma"]:
        if feat_col not in thresh_source.columns:
            continue
        daily_feat = thresh_source.groupby("date")[feat_col].median().dropna()
        if len(daily_feat) >= 10:
            result[f"{feat_col}_ice"] = float(daily_feat.quantile(0.70))
            result[f"{feat_col}_water"] = float(daily_feat.quantile(0.30))

    return result


def classify_daily(per_arc, enriched, s1_matched=None, s1_index=None,
                   station=None, snr_features=None, ice_free_months=None):
    """Classify each day per azimuth sector, then compute station-level consensus.

    When snr_features is provided, merges CLR/AF/PR/gamma into per_arc and uses
    them as additional voting indicators.

    Returns DataFrame with per-date, per-sector classifications and overall score.
    """
    # Merge SNR features into per_arc if available
    if snr_features is not None and len(snr_features) > 0:
        join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]
        # Avoid duplicate columns
        feat_cols = [c for c in snr_features.columns
                     if c not in per_arc.columns or c in join_cols]
        per_arc = per_arc.merge(snr_features[feat_cols], on=join_cols, how="left")
        n_matched = per_arc["CLR"].notna().sum() if "CLR" in per_arc.columns else 0
        logger.info(f"Merged SNR features: {n_matched}/{len(per_arc)} arcs matched")

    dates = sorted(per_arc["date"].unique())
    az_bins = sorted(per_arc["azimuth_bin"].unique())

    # Load station config
    station_cfg = _load_station_config(station) if station else {}
    suspect_bins = _load_suspect_azimuths(station_cfg)
    if ice_free_months is None:
        ice_free_months = station_cfg.get("ice_free_months", [])
    if suspect_bins:
        logger.info(f"Suspect azimuths from config: {sorted(suspect_bins)}")
    if ice_free_months:
        logger.info(f"Summer-anchored thresholds using months: {ice_free_months}")

    has_snr_features = "CLR" in per_arc.columns

    # Compute per-sector thresholds
    sector_thresholds = {}
    for b in az_bins:
        t = compute_sector_thresholds(per_arc, b, suspect_bins=suspect_bins,
                                      ice_free_months=ice_free_months)
        if t is not None:
            sector_thresholds[b] = t
            flag = " [LAND?]" if t["land_flag"] else ""
            snr_info = ""
            if "CLR_ice" in t:
                snr_info = (f", CLR [{t['CLR_water']:.1f},{t['CLR_ice']:.1f}]"
                            f", AF [{t['AF_water']:.0f},{t['AF_ice']:.0f}]")
            logger.info(
                f"  Sector {AZ_BIN_LABELS.get(b, str(b))}: "
                f"amp ice>{t['amp_mean_ice']:.1f} water<{t['amp_mean_water']:.1f}, "
                f"cv ice<{t['amp_cv_ice']:.3f} water>{t['amp_cv_water']:.3f}, "
                f"seasonal ratio={t['seasonal_ratio']:.2f}x, n={t['n_arcs']}"
                f"{snr_info}{flag}"
            )

    if not sector_thresholds:
        logger.error("No sectors with enough data")
        return None

    # S1 HH lookup (same for all sectors — regional context)
    s1_lookup = {}
    if s1_index is not None and "water_az_mean_hh_db" in s1_index.columns:
        for _, row in s1_index.iterrows():
            s1_lookup[row["acquisition_date"]] = row["water_az_mean_hh_db"]

    # Interfreq spread from enriched
    interfreq = {}
    if enriched is not None and len(enriched) > 0:
        freq_daily = enriched[
            (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] != "ALL")
        ]
        for date, grp in freq_daily.groupby("date"):
            if len(grp) >= 2:
                interfreq[date] = grp["rh_mean"].max() - grp["rh_mean"].min()

    rows = []
    for date in dates:
        day_arcs = per_arc[per_arc["date"] == date]
        row = {"date": date}

        sector_scores = []
        sector_weights = []

        for b, thresh in sector_thresholds.items():
            sector_arcs = day_arcs[day_arcs["azimuth_bin"] == b]
            prefix = f"az{b}"

            if len(sector_arcs) < 3:
                row[f"{prefix}_class"] = None
                row[f"{prefix}_score"] = np.nan
                continue

            # Skip land-flagged sectors
            if thresh["land_flag"]:
                row[f"{prefix}_class"] = "land"
                row[f"{prefix}_score"] = np.nan
                continue

            amp_mean = sector_arcs["Amp"].mean()
            amp_cv = sector_arcs["Amp"].std() / amp_mean if amp_mean > 0 else np.nan
            rh_std = sector_arcs["RH"].std()

            row[f"{prefix}_amp"] = amp_mean
            row[f"{prefix}_cv"] = amp_cv
            row[f"{prefix}_rh_std"] = rh_std

            # Vote per indicator for this sector
            votes = []
            weights_i = []

            # Amp mean (weight 2)
            v = _vote(amp_mean, thresh["amp_mean_ice"], thresh["amp_mean_water"], high_is_ice=True)
            if v:
                votes.append(_score(v))
                weights_i.append(2.0)
            row[f"{prefix}_amp_vote"] = v

            # Amp CV (weight 1.5)
            v = _vote(amp_cv, thresh["amp_cv_ice"], thresh["amp_cv_water"], high_is_ice=False)
            if v:
                votes.append(_score(v))
                weights_i.append(1.5)
            row[f"{prefix}_cv_vote"] = v

            # RH std (weight 1)
            v = _vote(rh_std, thresh["rh_std_ice"], thresh["rh_std_water"], high_is_ice=False)
            if v:
                votes.append(_score(v))
                weights_i.append(1.0)
            row[f"{prefix}_rh_vote"] = v

            # SNR-derived features (literature-based indicators)
            if has_snr_features:
                # CLR — clarity ratio (Purnell 2024, weight 2)
                if "CLR_ice" in thresh and "CLR" in sector_arcs.columns:
                    clr_med = sector_arcs["CLR"].median()
                    row[f"{prefix}_clr"] = clr_med
                    v = _vote(clr_med, thresh["CLR_ice"], thresh["CLR_water"],
                              high_is_ice=True)
                    if v:
                        votes.append(_score(v))
                        weights_i.append(2.0)
                    row[f"{prefix}_clr_vote"] = v

                # AF — area factor (Song 2022, weight 1.5)
                if "AF_ice" in thresh and "AF" in sector_arcs.columns:
                    af_med = sector_arcs["AF"].median()
                    row[f"{prefix}_af"] = af_med
                    v = _vote(af_med, thresh["AF_ice"], thresh["AF_water"],
                              high_is_ice=True)
                    if v:
                        votes.append(_score(v))
                        weights_i.append(1.5)
                    row[f"{prefix}_af_vote"] = v

                # PR — peak ratio (Purnell 2024, weight 1)
                if "PR_ice" in thresh and "PR" in sector_arcs.columns:
                    pr_med = sector_arcs["PR"].median()
                    row[f"{prefix}_pr"] = pr_med
                    v = _vote(pr_med, thresh["PR_ice"], thresh["PR_water"],
                              high_is_ice=True)
                    if v:
                        votes.append(_score(v))
                        weights_i.append(1.0)
                    row[f"{prefix}_pr_vote"] = v

                # gamma — damping (Strandberg 2017, weight 1)
                if "gamma_ice" in thresh and "gamma" in sector_arcs.columns:
                    gamma_med = sector_arcs["gamma"].median()
                    row[f"{prefix}_gamma"] = gamma_med
                    v = _vote(gamma_med, thresh["gamma_ice"], thresh["gamma_water"],
                              high_is_ice=False)
                    if v:
                        votes.append(_score(v))
                        weights_i.append(1.0)
                    row[f"{prefix}_gamma_vote"] = v

            # Sector score
            if votes:
                w_arr = np.array(weights_i)
                v_arr = np.array(votes)
                sector_score = float(np.average(v_arr, weights=w_arr))
            else:
                sector_score = np.nan

            row[f"{prefix}_score"] = sector_score
            if sector_score <= -0.33:
                row[f"{prefix}_class"] = ICE
            elif sector_score >= 0.33:
                row[f"{prefix}_class"] = WATER
            else:
                row[f"{prefix}_class"] = TRANSITION

            # Contribute to station-level, weighted by arc count
            if not np.isnan(sector_score):
                sector_scores.append(sector_score)
                sector_weights.append(len(sector_arcs))

        # Interfreq spread (station-level indicator)
        row["interfreq_spread"] = interfreq.get(date, np.nan)

        # S1 HH (station-level indicator)
        row["s1_hh"] = s1_lookup.get(date, np.nan)

        # Station-level consensus from sector scores + station-level indicators
        all_scores = list(sector_scores)
        all_weights = list(sector_weights)

        # Add interfreq spread vote
        ifs = row["interfreq_spread"]
        if not np.isnan(ifs):
            v = _vote(ifs, 0.03, 0.08, high_is_ice=False)
            row["interfreq_vote"] = v
            if v:
                all_scores.append(_score(v))
                all_weights.append(sum(sector_weights) * 0.3 if sector_weights else 1.0)

        # Add S1 HH vote
        s1 = row["s1_hh"]
        if not np.isnan(s1):
            v = _vote(s1, -17.0, -20.0, high_is_ice=True)
            row["s1_vote"] = v
            if v:
                all_scores.append(_score(v))
                all_weights.append(sum(sector_weights) * 0.5 if sector_weights else 1.0)

        # Weighted consensus
        if all_scores:
            w_arr = np.array(all_weights)
            s_arr = np.array(all_scores)
            consensus = float(np.average(s_arr, weights=w_arr))
            row["ice_score"] = consensus
            if consensus <= -0.33:
                row["classification"] = ICE
            elif consensus >= 0.33:
                row["classification"] = WATER
            else:
                row["classification"] = TRANSITION
        else:
            row["ice_score"] = np.nan
            row["classification"] = None

        # Also store pooled amp stats for backward compatibility
        row["amp_mean"] = day_arcs["Amp"].mean()
        row["amp_cv"] = day_arcs["Amp"].std() / day_arcs["Amp"].mean() if day_arcs["Amp"].mean() > 0 else np.nan
        row["rh_std"] = day_arcs["RH"].std()

        rows.append(row)

    result = pd.DataFrame(rows)
    result["date_dt"] = pd.to_datetime(result["date"])

    counts = result["classification"].value_counts()
    logger.info(f"Classification summary: {dict(counts)}")

    return result


def classify_station(station, year, include_s1=True):
    """Load data and classify a station-year.

    Automatically loads SNR features and ice_free_months if available.

    Returns (classification_df, sector_thresholds).
    """
    results_dir = PROJECT_ROOT / "results_annual" / station

    pa_path = results_dir / f"{station}_{year}_per_arc.parquet"
    if not pa_path.exists():
        logger.error(f"Per-arc not found: {pa_path}")
        return None, None
    per_arc = pd.read_parquet(pa_path)
    logger.info(f"Loaded {len(per_arc)} arcs for {station} {year}")

    en_path = results_dir / f"{station}_{year}_daily_enriched.parquet"
    enriched = pd.read_parquet(en_path) if en_path.exists() else pd.DataFrame()

    s1_matched = None
    s1_index = None
    if include_s1:
        m_path = results_dir / f"{station}_{year}_s1_gnssir_matched.parquet"
        if m_path.exists():
            s1_matched = pd.read_parquet(m_path)
        idx_path = PROJECT_ROOT / "data" / station / "s1_fresnel" / f"{station}_s1_fresnel_index.csv"
        if idx_path.exists():
            s1_index = pd.read_csv(idx_path)

    # Load SNR features if available
    snr_features = None
    feat_path = results_dir / f"{station}_{year}_snr_features.parquet"
    if feat_path.exists():
        snr_features = pd.read_parquet(feat_path)
        logger.info(f"Loaded {len(snr_features)} SNR feature rows from {feat_path.name}")

    # Load ice_free_months from station config
    station_cfg = _load_station_config(station)
    ice_free_months = station_cfg.get("ice_free_months", [])

    # Compute sector thresholds for reporting (before merge, for display)
    az_bins = sorted(per_arc["azimuth_bin"].unique())
    sector_thresholds = {}
    for b in az_bins:
        t = compute_sector_thresholds(per_arc, b,
                                      ice_free_months=ice_free_months)
        if t is not None:
            sector_thresholds[b] = t

    result = classify_daily(per_arc, enriched, s1_matched, s1_index,
                            station=station, snr_features=snr_features,
                            ice_free_months=ice_free_months)

    return result, sector_thresholds


def main():
    parser = argparse.ArgumentParser(description="Per-sector ice classifier")
    parser.add_argument("--station", required=True, help="Station ID")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--no-s1", action="store_true", help="Exclude S1 data")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    result, sector_thresholds = classify_station(
        args.station, args.year, include_s1=not args.no_s1
    )
    if result is None:
        sys.exit(1)

    # Save
    out_path = (PROJECT_ROOT / "results_annual" / args.station
                / f"{args.station}_{args.year}_ice_classification.parquet")
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Ice Classification: {args.station} {args.year}")
    print(f"{'='*60}")

    print(f"\nSector thresholds:")
    for b, t in sector_thresholds.items():
        flag = " ** LAND **" if t["land_flag"] else ""
        print(f"  {AZ_BIN_LABELS.get(b, str(b))} deg: "
              f"amp [{t['amp_mean_water']:.1f}, {t['amp_mean_ice']:.1f}], "
              f"cv [{t['amp_cv_ice']:.3f}, {t['amp_cv_water']:.3f}], "
              f"ratio={t['seasonal_ratio']:.2f}x, n={t['n_arcs']}{flag}")

    print(f"\nDays classified: {len(result)}")
    print(f"\n{result['classification'].value_counts().to_string()}")

    # Per-sector classification counts
    az_class_cols = [c for c in result.columns if c.endswith("_class")]
    for col in sorted(az_class_cols):
        counts = result[col].value_counts()
        print(f"\n  {col}: {dict(counts)}")

    # Monthly breakdown
    result["month"] = result["date_dt"].dt.month
    print(f"\nMonthly breakdown:")
    for m in sorted(result["month"].unique()):
        md = result[result["month"] == m]
        counts = md["classification"].value_counts()
        score = md["ice_score"].mean()

        # Per-sector scores
        sector_scores = []
        for col in sorted(az_class_cols):
            sector_counts = md[col].value_counts()
            ice_n = sector_counts.get("ice", 0)
            if ice_n > 0:
                sector_scores.append(f"{col.replace('_class', '')}:{ice_n}ice")

        sector_text = f" [{', '.join(sector_scores)}]" if sector_scores else ""
        print(f"  Month {m:2d}: score={score:+.2f}  {dict(counts)}{sector_text}")


if __name__ == "__main__":
    main()
