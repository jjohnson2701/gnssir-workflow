#!/usr/bin/env python3
"""
Classifier v3: Mahalanobis Sub-State Resolution (Discovery / Validated)

Builds on v2's water-default Mahalanobis approach and adds:
1. Sub-states: ice_surface vs ice_layered (from interfrequency ΔRH)
2. Ice_decaying: anomalous regime persisting above freezing (ERA5 temperature)
3. Preserves v1 and v2 outputs — writes to _ice_classification_v3.parquet

Two classification modes based on station config:

**Validated mode** (classification.validated=true, use_case="ice"):
    open_water    — Mahalanobis < threshold
    freeze_up     — transition into anomalous regime
    ice_surface   — Mahalanobis > threshold, |ΔRH| < 0.3m
    ice_layered   — Mahalanobis > threshold, |ΔRH| >= 0.3m
    ice_decaying  — Mahalanobis > threshold, ERA5 t2m_mean > freeze_temp
    break_up      — transition out of anomalous regime

**Discovery mode** (default — no validated reference data):
    baseline      — Mahalanobis < threshold
    anomalous     — Mahalanobis > threshold (any sub-type)
    regime_change — transition into/out of anomalous regime
    Metadata columns (has_interfreq_divergence, above_freezing) still computed.

Usage:
    python scripts/classifier_v3.py --station ROSS --years 2020-2024
    python scripts/classifier_v3.py --station GLBX --years 2023-2024
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_annual"
CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"


def load_station_config(station):
    with open(CONFIG_PATH) as f:
        return json.load(f).get(station, {})


def get_classification_mode(cfg):
    """Determine classification mode from station config.

    Returns "validated" if classification.validated is True and use_case is set,
    otherwise "discovery".
    """
    cls_cfg = cfg.get("classification", {})
    if cls_cfg.get("validated", False) and cls_cfg.get("use_case"):
        return "validated"
    return "discovery"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v2(station, year):
    """Load v2 classification (must already exist)."""
    path = RESULTS_DIR / station / f"{station}_{year}_ice_classification_v2.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    return df


def load_interfreq(station, year):
    """Load interfrequency divergence, return daily median ΔRH."""
    path = RESULTS_DIR / station / f"{station}_{year}_combined_interfreq.parquet"
    if not path.exists():
        return None
    interfreq = pd.read_parquet(path)
    if "L1_L2C_rh_diff" not in interfreq.columns:
        return None  # L2C not available this year
    daily = interfreq.groupby(
        interfreq["date"].apply(lambda d: pd.Timestamp(d).timetuple().tm_yday)
    ).agg(
        rh_diff_median=("L1_L2C_rh_diff", "median"),
        rh_diff_std=("L1_L2C_rh_diff", "std"),
        amp_diff_median=("L1_L2C_amp_diff", "median"),
        n_sectors=("L1_L2C_rh_diff", "count"),
    ).reset_index()
    daily.columns = ["doy", "rh_diff_median", "rh_diff_std", "amp_diff_median", "n_interfreq"]
    return daily


def load_era5(station, year):
    """Load ERA5 daily temperature."""
    path = RESULTS_DIR / station / f"{station}_{year}_era5.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[["doy", "t2m_mean", "t2m_min", "t2m_max"]].copy()


def load_v1(station, year):
    """Load v1 classification for comparison column."""
    path = RESULTS_DIR / station / f"{station}_{year}_ice_classification.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    return df[["doy", "classification", "ice_score"]].rename(
        columns={"classification": "v1_class", "ice_score": "v1_score"}
    )


# ---------------------------------------------------------------------------
# v3 classification
# ---------------------------------------------------------------------------

def classify_v3(v2_df, interfreq_df=None, era5_df=None,
                rh_diff_threshold=0.3, freeze_temp=0.0,
                classification_mode="discovery"):
    """Apply v3 sub-state labels on top of v2 results.

    Args:
        v2_df: v2 classification with columns [date, doy, state, mahal_dist, ...]
        interfreq_df: daily interfrequency ΔRH (optional)
        era5_df: daily ERA5 temperature (optional)
        rh_diff_threshold: |ΔRH| threshold for ice_layered (meters)
        freeze_temp: temperature below which ice can form (°C).
        classification_mode: "validated" for ice vocabulary, "discovery" for
                             generic baseline/anomalous labels.

    Returns:
        DataFrame with v3_state, metadata columns, and classification_mode.
    """
    df = v2_df.copy()

    # Merge interfreq
    if interfreq_df is not None:
        df = df.merge(interfreq_df, on="doy", how="left")
    else:
        df["rh_diff_median"] = np.nan

    # Merge ERA5
    if era5_df is not None:
        df = df.merge(era5_df, on="doy", how="left")
    else:
        df["t2m_mean"] = np.nan
        df["t2m_min"] = np.nan

    # --- Compute metadata flags (always, regardless of mode) ---
    df["has_interfreq_divergence"] = df["rh_diff_median"].abs() >= rh_diff_threshold
    df["above_freezing"] = df["t2m_mean"] > freeze_temp
    # NaN → False for boolean columns
    df["has_interfreq_divergence"] = df["has_interfreq_divergence"].fillna(False)
    df["above_freezing"] = df["above_freezing"].fillna(False)

    # --- Apply v3 sub-state logic ---
    v3_states = []
    for _, row in df.iterrows():
        v2_state = row["state"]
        rh_diff = row.get("rh_diff_median", np.nan)
        t2m = row.get("t2m_mean", np.nan)

        if classification_mode == "validated":
            # Full ice vocabulary
            if v2_state == "water":
                v3_states.append("open_water")
            elif v2_state == "freeze_up":
                v3_states.append("freeze_up")
            elif v2_state == "break_up":
                v3_states.append("break_up")
            elif v2_state == "ice":
                if not np.isnan(t2m) and t2m > freeze_temp:
                    v3_states.append("ice_decaying")
                elif not np.isnan(rh_diff) and abs(rh_diff) > rh_diff_threshold:
                    v3_states.append("ice_layered")
                else:
                    v3_states.append("ice_surface")
            else:
                v3_states.append(v2_state if pd.notna(v2_state) else "unknown")
        else:
            # Discovery mode — generic labels
            if v2_state == "water":
                v3_states.append("baseline")
            elif v2_state in ("freeze_up", "break_up"):
                v3_states.append("regime_change")
            elif v2_state == "ice":
                v3_states.append("anomalous")
            else:
                v3_states.append(v2_state if pd.notna(v2_state) else "unknown")

    df["v3_state"] = v3_states
    df["classification_mode"] = classification_mode

    # Retain v2 state for comparison
    df = df.rename(columns={"state": "v2_state"})

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_V3_STATES = ["open_water", "freeze_up", "ice_surface", "ice_layered",
                 "ice_decaying", "break_up"]
DISCOVERY_STATES = ["baseline", "anomalous", "regime_change"]


def run_station(station, years, skip_era5=False):
    """Run v3 for one station across multiple years."""
    cfg = load_station_config(station)
    freeze_temp = cfg.get("freeze_temp_c", 0.0)
    mode = get_classification_mode(cfg)

    log.info(f"{'=' * 60}")
    log.info(f"{station}: v3 classification, years={years}, "
             f"mode={mode}, freeze_temp={freeze_temp}°C")

    state_labels = ALL_V3_STATES if mode == "validated" else DISCOVERY_STATES
    all_results = []

    for year in years:
        v2 = load_v2(station, year)
        if v2 is None:
            log.warning(f"  {year}: no v2 classification, skipping")
            continue

        interfreq = load_interfreq(station, year)
        era5 = load_era5(station, year) if not skip_era5 else None
        v1 = load_v1(station, year)

        v3 = classify_v3(v2, interfreq, era5, freeze_temp=freeze_temp,
                         classification_mode=mode)

        # Add v1 for comparison
        if v1 is not None:
            v3 = v3.merge(v1, on="doy", how="left")

        # Save
        out_path = RESULTS_DIR / station / f"{station}_{year}_ice_classification_v3.parquet"
        v3.to_parquet(out_path, index=False)

        # Summary
        state_counts = v3["v3_state"].value_counts()
        total = len(v3)

        log.info(f"  {year}: {total} days [{mode}] → {state_counts.to_dict()}")

        all_results.append({
            "station": station, "year": year, "n_days": total,
            "classification_mode": mode,
            **{f"n_{s}": state_counts.get(s, 0) for s in state_labels},
        })

    return pd.DataFrame(all_results) if all_results else None


def main():
    parser = argparse.ArgumentParser(description="Classifier v3")
    parser.add_argument("--station", required=True)
    parser.add_argument("--years", required=True, help="e.g. 2020-2024")
    parser.add_argument("--skip_era5", action="store_true",
                        help="Run without ERA5 (no ice_decaying detection)")
    parser.add_argument("--freeze_temp", type=float, default=None,
                        help="Override freeze temperature (°C). Default: from config or 0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    start, end = map(int, args.years.split("-"))
    years = list(range(start, end + 1))

    if args.freeze_temp is not None:
        # Temporarily override in config
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        if args.station in cfg:
            cfg[args.station]["freeze_temp_c"] = args.freeze_temp
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)

    results = run_station(args.station, years, skip_era5=args.skip_era5)

    if results is not None and len(results) > 0:
        mode = results["classification_mode"].iloc[0]
        state_cols = [c for c in results.columns if c.startswith("n_") and c != "n_days"]
        print(f"\n{'=' * 70}")
        print(f"V3 SUMMARY: {args.station} [{mode}]")
        print(f"{'=' * 70}")
        print(results.to_string(index=False))

        totals = results[state_cols].sum()
        total = totals.sum()
        print(f"\nAcross all years ({total} days):")
        for state, count in totals.items():
            pct = count / total * 100
            label = state.replace("n_", "")
            print(f"  {label:>15s}: {int(count):4d} ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
