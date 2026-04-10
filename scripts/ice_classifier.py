# ABOUTME: Per-azimuth-sector ice/water/transition classifier for GNSS-IR stations
# ABOUTME: Reads daily_features.parquet (Layer 2) and applies threshold voting per sector

"""
Per-sector ice classifier (v2 — reads daily_features.parquet).

Classifies each azimuth sector independently rather than pooling all azimuths.
This prevents land-contaminated sectors (e.g., 50-60 deg at UMNQ) from drowning
out genuine ice signals in other sectors.

Thresholds are computed per-sector from the sector's own daily feature values.
Sectors with no seasonal amplitude change are flagged as potential land.

Input: daily_features.parquet (produced by feature_aggregator.py — Layer 2)
Output: ice_state.parquet (Layer 3)

Usage:
    python scripts/ice_classifier.py --station UMNQ --year 2025
    python scripts/ice_classifier.py --station NKAR --year 2025 --no-s1
    python scripts/ice_classifier.py --all
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


def _normalize_features_in_place(per_arc, feature_cols, ice_free_months=None):
    """Z-score normalize SNR features per satellite PRN (v1-compatible).

    Modifies per_arc in place. Satellites with < 10 reference arcs keep
    their raw values (not set to NaN). This matches the original v1
    classifier behavior for threshold computation.
    """
    if ice_free_months:
        months = pd.to_datetime(per_arc["date"]).dt.month
        ref_mask = months.isin(ice_free_months)
        ref_data = per_arc[ref_mask]
    else:
        ref_data = per_arc

    for col in feature_cols:
        if col not in per_arc.columns:
            continue

        sat_stats = ref_data.groupby("sat")[col].agg(["mean", "std", "count"])

        if ice_free_months:
            all_sats = per_arc["sat"].unique()
            missing_sats = set(all_sats) - set(sat_stats.index)
            if missing_sats:
                fallback = per_arc.groupby("sat")[col].agg(["mean", "std", "count"])
                for sat in missing_sats:
                    if sat in fallback.index:
                        sat_stats.loc[sat] = fallback.loc[sat]
                        logger.warning(
                            f"Satellite {sat}: no ice-free data for {col}, "
                            f"using full-year stats"
                        )

        for sat, row in sat_stats.iterrows():
            mask = per_arc["sat"] == sat
            if row["count"] < 10:
                continue  # keep raw values
            if row["std"] > 0:
                per_arc.loc[mask, col] = (
                    (per_arc.loc[mask, col] - row["mean"]) / row["std"]
                )
            else:
                per_arc.loc[mask, col] = 0.0


# Frequency code → band group (for per-sector ΔRH in threshold computation)
_FREQ_TO_BAND = {
    1: "L1", 2: "L2", 20: "L2C", 5: "L5",          # GPS
    101: "E1", 102: "E5a", 105: "E5a",               # Galileo (E1≈L1, E5a≈L5)
    201: "G1", 205: "G2",                             # GLONASS
    206: "E5b", 207: "E5", 208: "E6",                 # Galileo cont.
    301: "B1", 302: "B1", 306: "B3", 307: "B2a",     # BeiDou
}


# ---------------------------------------------------------------------------
# Threshold computation from arc_table (Option B — preserves exact v1 thresholds)
# ---------------------------------------------------------------------------

def compute_sector_thresholds(per_arc, az_bin, suspect_bins=None,
                              ice_free_months=None):
    """Compute thresholds for a single azimuth sector from per-arc data.

    Uses arc_table (per-arc data) for threshold computation to produce
    identical thresholds to the v1 classifier. The voting and smoothing
    stages use daily_features.parquet instead.

    When ice_free_months is provided, thresholds are anchored to the summer
    (ice-free) window. SNR feature thresholds use full-year data since
    features are already z-scored per satellite.
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

    # SNR feature thresholds — always use full-year data.
    # Features are already z-scored per satellite using ice-free months,
    # so full-year percentiles capture the seasonal range.
    for feat_col in ["CLR", "AF", "PR", "gamma", "phase"]:
        if feat_col not in sector_copy.columns:
            continue
        daily_feat = sector_copy.groupby("date")[feat_col].median().dropna()
        if len(daily_feat) >= 10:
            result[f"{feat_col}_ice"] = float(daily_feat.quantile(0.70))
            result[f"{feat_col}_water"] = float(daily_feat.quantile(0.30))

    # Per-sector ΔRH thresholds (interfrequency spread)
    if "freq" in sector_copy.columns and "RH" in sector_copy.columns:
        sc = sector_copy.copy()
        sc["_band"] = sc["freq"].map(_FREQ_TO_BAND)
        daily_drh = []
        for date, grp in sc.groupby("date"):
            band_rh = grp.groupby("_band")["RH"].median()
            if len(band_rh) >= 2:
                daily_drh.append(float(band_rh.std()))
        if len(daily_drh) >= 10:
            drh_series = pd.Series(daily_drh)
            result["delta_rh_ice"] = float(drh_series.quantile(0.70))
            result["delta_rh_water"] = float(drh_series.quantile(0.30))

    return result


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth_daily_features(df, window=3):
    """Apply centered rolling median to daily feature values per sector.

    n_arcs and metadata columns are not smoothed. Only numeric feature
    columns are smoothed. Setting window=1 returns the input unchanged.
    """
    if window <= 1:
        return df

    smoothed = df.copy()
    smoothed["_date_dt"] = pd.to_datetime(smoothed["date"])
    smoothed = smoothed.sort_values(["azimuth_bin", "_date_dt"])

    value_cols = [
        "amp_mean", "amp_cv", "rh_std",
        "clr_med", "af_med", "pr_med", "gamma_med",
        "clr_z", "af_z", "pr_z", "gamma_z",
        "delta_rh_mean", "phase_circ_mean", "phase_circ_std",
        "diff_phase_L1_L2",
    ]
    value_cols = [c for c in value_cols if c in smoothed.columns]

    for sector in smoothed["azimuth_bin"].unique():
        mask = smoothed["azimuth_bin"] == sector
        sector_data = smoothed.loc[mask]
        for col in value_cols:
            smoothed.loc[mask, col] = (
                sector_data[col]
                .rolling(window, min_periods=1, center=True)
                .median()
            )

    smoothed = smoothed.drop(columns=["_date_dt"])
    return smoothed


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_daily(daily_features, sector_thresholds, station=None,
                   s1_index=None, smoothing_window=3, min_arcs=3):
    """Classify each day per azimuth sector from pre-computed daily features.

    Reads daily_features.parquet (Layer 2 output). Thresholds are computed
    externally from arc_table (Option B) and passed in.

    Three-pass architecture:
      1. Thresholds computed externally from arc_table (passed in)
      2. Apply centered rolling median (smoothing_window days) per sector
      3. Vote on smoothed values, compute sector and station-level scores
    """
    station_cfg = _load_station_config(station) if station else {}
    feature_overrides = station_cfg.get("feature_overrides", {})

    # Separate sector rows from pooled rows
    sector_features = daily_features[daily_features["azimuth_bin"] >= 0].copy()
    pooled_features = daily_features[daily_features["azimuth_bin"] == -1].copy()

    has_snr = "clr_z" in sector_features.columns

    dates = sorted(sector_features["date"].unique())

    if not sector_thresholds:
        logger.error("No sectors with enough data for thresholds")
        return None

    # --- Pass 2: Temporal smoothing ---
    # Mask feature values for days below min_arcs or flagged as land
    # (matches v1 behavior: _extract_indicators set NaN for these)
    value_cols_to_mask = [
        "amp_mean", "amp_cv", "rh_std",
        "clr_med", "af_med", "pr_med", "gamma_med",
        "clr_z", "af_z", "pr_z", "gamma_z",
        "delta_rh_mean", "phase_circ_mean", "phase_circ_std",
        "diff_phase_L1_L2",
    ]
    value_cols_to_mask = [c for c in value_cols_to_mask if c in sector_features.columns]
    for b, thresh in sector_thresholds.items():
        mask = (sector_features["azimuth_bin"] == b)
        low_arcs = mask & (sector_features["n_arcs"] < min_arcs)
        land = mask & thresh["land_flag"]
        null_mask = low_arcs | land
        if null_mask.any():
            sector_features.loc[null_mask, value_cols_to_mask] = np.nan

    sector_features = _smooth_daily_features(sector_features, window=smoothing_window)
    logger.info(f"Min arcs per sector: {min_arcs}, smoothing window: {smoothing_window}")
    if smoothing_window > 1:
        logger.info(f"Applied {smoothing_window}-day rolling median smoothing")

    # Build lookups
    # Sector features: (date, azimuth_bin) → row
    sf_lookup = {}
    for _, row in sector_features.iterrows():
        sf_lookup[(row["date"], int(row["azimuth_bin"]))] = row

    # Interfreq spread from pooled rows
    interfreq = {}
    if "interfreq_spread" in pooled_features.columns:
        for _, row in pooled_features.iterrows():
            val = row["interfreq_spread"]
            if pd.notna(val):
                interfreq[row["date"]] = val

    # S1 HH lookup
    s1_lookup = {}
    if s1_index is not None and "water_az_mean_hh_db" in s1_index.columns:
        for _, row in s1_index.iterrows():
            s1_lookup[row["acquisition_date"]] = row["water_az_mean_hh_db"]

    # --- Indicator voting config ---
    # (feature_col_in_daily_features, thresh_ice_key, thresh_water_key,
    #  high_is_ice, weight, output_suffix)
    # Threshold keys match compute_sector_thresholds() output (arc_table based).
    # Feature cols read from daily_features (z-scored medians for SNR features).
    base_indicators = [
        ("amp_mean", "amp_mean_ice", "amp_mean_water", True, 2.0, "amp"),
        ("amp_cv", "amp_cv_ice", "amp_cv_water", False, 1.5, "cv"),
        ("rh_std", "rh_std_ice", "rh_std_water", False, 1.0, "rh"),
    ]
    snr_indicators = [
        ("clr_z", "CLR_ice", "CLR_water", True, 2.0, "clr"),
        ("af_z", "AF_ice", "AF_water", True, 1.5, "af"),
        ("pr_z", "PR_ice", "PR_water", True, 1.0, "pr"),
        ("gamma_z", "gamma_ice", "gamma_water", False, 1.0, "gamma"),
        ("phase_circ_mean", "phase_ice", "phase_water", True, 0.5, "phase"),
        ("delta_rh_mean", "delta_rh_ice", "delta_rh_water", True, 0.5, "delta_rh"),
    ]

    # Apply station-specific feature overrides
    if feature_overrides:
        logger.info(f"Applying feature overrides: {feature_overrides}")
        for indicators in [base_indicators, snr_indicators]:
            for i, (ind_key, ice_key, water_key, high_ice, weight, suffix) in enumerate(indicators):
                if suffix in feature_overrides:
                    ovr = feature_overrides[suffix]
                    if "high_is_ice" in ovr:
                        high_ice = ovr["high_is_ice"]
                    if "weight" in ovr:
                        weight = ovr["weight"]
                    indicators[i] = (ind_key, ice_key, water_key, high_ice, weight, suffix)
                    logger.info(f"  {suffix}: high_is_ice={high_ice}, weight={weight}")

    # --- Pass 3: Vote on smoothed features ---
    rows = []
    for date in dates:
        row = {"date": date}
        sector_scores = []
        sector_weights = []

        for b, thresh in sector_thresholds.items():
            prefix = f"az{b}"
            sr = sf_lookup.get((date, b))

            if sr is None:
                row[f"{prefix}_class"] = None
                row[f"{prefix}_score"] = np.nan
                continue

            n_arcs = sr.get("n_arcs", 0)
            if n_arcs < min_arcs:
                row[f"{prefix}_class"] = None
                row[f"{prefix}_score"] = np.nan
                continue

            if thresh["land_flag"]:
                row[f"{prefix}_class"] = "land"
                row[f"{prefix}_score"] = np.nan
                continue

            votes = []
            weights_i = []

            # Vote on base indicators
            for ind_key, ice_key, water_key, high_ice, weight, suffix in base_indicators:
                val = sr.get(ind_key, np.nan)
                if np.isnan(val) if isinstance(val, float) else pd.isna(val):
                    continue
                row[f"{prefix}_{suffix}"] = val
                v = _vote(val, thresh[ice_key], thresh[water_key],
                          high_is_ice=high_ice)
                row[f"{prefix}_{suffix}_vote"] = v
                if v:
                    votes.append(_score(v))
                    weights_i.append(weight)

            # Vote on SNR-derived indicators
            if has_snr:
                for ind_key, ice_key, water_key, high_ice, weight, suffix in snr_indicators:
                    if ice_key not in thresh:
                        continue
                    val = sr.get(ind_key, np.nan)
                    if np.isnan(val) if isinstance(val, float) else pd.isna(val):
                        continue
                    row[f"{prefix}_{suffix}"] = val
                    v = _vote(val, thresh[ice_key], thresh[water_key],
                              high_is_ice=high_ice)
                    row[f"{prefix}_{suffix}_vote"] = v
                    if v:
                        votes.append(_score(v))
                        weights_i.append(weight)

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

            if not np.isnan(sector_score):
                sector_scores.append(sector_score)
                sector_weights.append(int(n_arcs))

        # Interfreq spread (station-level indicator)
        ifs = interfreq.get(date, np.nan)
        row["interfreq_spread"] = ifs if not pd.isna(ifs) else np.nan
        if pd.notna(ifs):
            v = _vote(ifs, 0.03, 0.08, high_is_ice=False)
            row["interfreq_vote"] = v
        else:
            row["interfreq_vote"] = None

        # S1 HH (station-level indicator)
        s1 = s1_lookup.get(date, np.nan)
        row["s1_hh"] = s1 if not pd.isna(s1) else np.nan

        # Station-level consensus
        all_scores = list(sector_scores)
        all_weights = list(sector_weights)

        # Add interfreq spread vote
        if pd.notna(ifs):
            v = _vote(ifs, 0.03, 0.08, high_is_ice=False)
            if v:
                all_scores.append(_score(v))
                all_weights.append(sum(sector_weights) * 0.3 if sector_weights else 1.0)

        # Add S1 HH vote
        if pd.notna(s1):
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

        # Pooled station-level stats from pooled row
        pooled_row = pooled_features[pooled_features["date"] == date]
        if len(pooled_row) > 0:
            pr = pooled_row.iloc[0]
            row["amp_mean"] = pr.get("amp_mean", np.nan)
            row["amp_cv"] = pr.get("amp_cv", np.nan)
            row["rh_std"] = pr.get("rh_std", np.nan)
        else:
            row["amp_mean"] = np.nan
            row["amp_cv"] = np.nan
            row["rh_std"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    result["date_dt"] = pd.to_datetime(result["date"])

    counts = result["classification"].value_counts()
    logger.info(f"Classification summary: {dict(counts)}")

    return result


def classify_station(station, year, include_s1=True):
    """Load daily_features + arc_table and classify a station-year.

    Uses arc_table for threshold computation (preserves v1 thresholds)
    and daily_features for smoothing and voting (Layer 2 aggregates).

    Returns (classification_df, sector_thresholds).
    """
    results_dir = PROJECT_ROOT / "results_annual" / station

    # Load daily features (Layer 2 — used for smoothing and voting)
    df_path = results_dir / f"{station}_{year}_daily_features.parquet"
    if not df_path.exists():
        logger.error(f"daily_features not found: {df_path}")
        logger.error(
            "Run feature_aggregator.py first: "
            f"python scripts/feature_aggregator.py --station {station} --year {year}"
        )
        return None, None
    daily_features = pd.read_parquet(df_path)
    logger.info(f"Loaded {len(daily_features)} daily feature rows for {station} {year}")

    # Load arc_table (Layer 1 — used for threshold computation)
    at_path = results_dir / f"{station}_{year}_arc_table.parquet"
    if not at_path.exists():
        logger.error(f"arc_table not found: {at_path}")
        return None, None
    arc_table = pd.read_parquet(at_path)
    logger.info(f"Loaded {len(arc_table)} arcs for threshold computation")

    # S1 data
    s1_index = None
    if include_s1:
        idx_path = (PROJECT_ROOT / "data" / station / "s1_fresnel"
                    / f"{station}_s1_fresnel_index.csv")
        if idx_path.exists():
            s1_index = pd.read_csv(idx_path)

    station_cfg = _load_station_config(station)
    ice_free_months = station_cfg.get("baseline_period",
                                     station_cfg.get("ice_free_months", []))
    smoothing_window = station_cfg.get("smoothing_window", 3)
    min_arcs = station_cfg.get("min_arcs_per_sector", 3)
    suspect_bins = _load_suspect_azimuths(station_cfg)

    # Z-score normalize SNR features in arc_table (v1-compatible: in-place)
    # Satellites with < 10 reference arcs keep their raw values (v1 behavior).
    has_snr = "CLR" in arc_table.columns
    if has_snr:
        _normalize_features_in_place(
            arc_table,
            feature_cols=["CLR", "AF", "PR", "gamma"],
            ice_free_months=ice_free_months,
        )

    # Compute per-sector thresholds from arc_table (preserves v1 thresholds)
    az_bins = sorted(arc_table["azimuth_bin"].unique())
    sector_thresholds = {}
    for b in az_bins:
        t = compute_sector_thresholds(
            arc_table, b,
            suspect_bins=suspect_bins,
            ice_free_months=ice_free_months,
        )
        if t is not None:
            sector_thresholds[b] = t
            flag = " [LAND?]" if t["land_flag"] else ""
            snr_info = ""
            if "CLR_ice" in t:
                snr_info = f", CLR [{t['CLR_water']:.1f},{t['CLR_ice']:.1f}]"
                if "AF_ice" in t:
                    snr_info += f", AF [{t['AF_water']:.0f},{t['AF_ice']:.0f}]"
            ratio_str = (f"{t['seasonal_ratio']:.2f}x"
                         if t["seasonal_ratio"] is not None else "N/A")
            logger.info(
                f"  Sector {AZ_BIN_LABELS.get(b, str(b))}: "
                f"amp [{t['amp_mean_water']:.1f},{t['amp_mean_ice']:.1f}], "
                f"cv [{t['amp_cv_ice']:.3f},{t['amp_cv_water']:.3f}], "
                f"ratio={ratio_str}, n={t['n_arcs']}"
                f"{snr_info}{flag}"
            )

    result = classify_daily(
        daily_features, sector_thresholds,
        station=station, s1_index=s1_index,
        smoothing_window=smoothing_window, min_arcs=min_arcs,
    )

    return result, sector_thresholds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-sector ice classifier (v2)")
    parser.add_argument("--station", help="Station ID")
    parser.add_argument("--year", type=int, help="Year")
    parser.add_argument("--no-s1", action="store_true", help="Exclude S1 data")
    parser.add_argument("--all", action="store_true",
                        help="Classify all station-years with daily_features")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing classification")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.all:
        from scripts.results_handler import discover_station_years
        targets = discover_station_years(require_file="daily_features.parquet")
        logger.info(f"Found {len(targets)} station-years with daily_features")
    elif args.station and args.year:
        targets = [(args.station, args.year)]
    else:
        parser.error("Specify --station and --year, or use --all")

    for station, year in targets:
        out_path = (PROJECT_ROOT / "results_annual" / station
                    / f"{station}_{year}_ice_state.parquet")
        if out_path.exists() and not args.force and args.all:
            continue

        result, sector_thresholds = classify_station(
            station, year, include_s1=not args.no_s1
        )
        if result is None:
            continue

        result.to_parquet(out_path, index=False)
        logger.info(f"Saved: {out_path} (Layer 3)")

        if not args.all:
            # Print summary for single station
            print(f"\n{'='*60}")
            print(f"Ice Classification: {station} {year}")
            print(f"{'='*60}")

            print(f"\nSector thresholds:")
            for b, t in sector_thresholds.items():
                flag = " ** LAND **" if t["land_flag"] else ""
                ratio_str = (f"{t['seasonal_ratio']:.2f}x"
                             if t["seasonal_ratio"] is not None else "N/A")
                print(f"  {AZ_BIN_LABELS.get(b, str(b))} deg: "
                      f"amp [{t['amp_mean_water']:.1f}, {t['amp_mean_ice']:.1f}], "
                      f"cv [{t['amp_cv_ice']:.3f}, {t['amp_cv_water']:.3f}], "
                      f"ratio={ratio_str}, n={t['n_arcs']}{flag}")

            print(f"\nDays classified: {len(result)}")
            print(f"\n{result['classification'].value_counts().to_string()}")

            az_class_cols = [c for c in result.columns if c.endswith("_class")]
            for col in sorted(az_class_cols):
                counts = result[col].value_counts()
                print(f"\n  {col}: {dict(counts)}")

            result["month"] = result["date_dt"].dt.month
            print(f"\nMonthly breakdown:")
            for m in sorted(result["month"].unique()):
                md = result[result["month"] == m]
                counts = md["classification"].value_counts()
                score = md["ice_score"].mean()
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
