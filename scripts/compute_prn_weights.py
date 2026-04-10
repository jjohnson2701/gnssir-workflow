# ABOUTME: Compute per-(sat, freq) discriminating power weights from arc_table
# ABOUTME: Produces a JSON lookup of Cohen's d per PRN per feature for weighted aggregation

"""
Compute per-(satellite, frequency) discriminating-power weights.

For each feature in the arc_table, computes Cohen's d between ice-free and
winter months per (sat, freq) combo.  Stores the result as a JSON file that
feature_aggregator.py can load to produce weighted medians — PRNs with higher
|d| contribute more to the daily aggregate.

The weight for a given (sat, freq, feature) is |d|, floored at 0 to avoid
negative weights.  PRNs with fewer than _MIN_ARCS_PER_SEASON in either season
are assigned weight 0 (insufficient data to estimate d reliably).

Usage:
    python scripts/compute_prn_weights.py --station UMNQ --year 2025
    python scripts/compute_prn_weights.py --station ROSS --year 2024
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

# Features to compute weights for — matches _SNR_FEATURE_COLS in aggregator
_WEIGHT_FEATURES = ["CLR", "PR", "AF", "gamma", "MS", "VS"]

# Minimum arcs per season per (sat, freq) to compute a reliable d
_MIN_ARCS_PER_SEASON = 15

# Winter months — the "ice" reference season
_WINTER_MONTHS_DEFAULT = [12, 1, 2, 3]


def _load_station_config(station):
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        all_cfg = json.load(f)
    return all_cfg.get(station, {})


def _cohens_d(group_a, group_b):
    """Compute Cohen's d (pooled SD) between two groups."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    ma, mb = group_a.mean(), group_b.mean()
    va, vb = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (ma - mb) / pooled_std


def compute_prn_weights(station, year, winter_months=None):
    """Compute per-(sat, freq) Cohen's d for each feature.

    Args:
        station: station ID
        year: processing year
        winter_months: months to use as "ice" season (default [12, 1, 2, 3])

    Returns:
        dict with structure:
            {
                "station": str,
                "year": int,
                "baseline_period": [...],
                "winter_months": [...],
                "features": {
                    "CLR": {"1_1": {"d": 0.85, "weight": 0.85, "n_summer": 40, "n_winter": 35}, ...},
                    "AF": {...},
                    ...
                },
                "summary": {feature: {"n_combos": N, "n_weighted": N, "mean_weight": f, "max_weight": f}}
            }
    """
    station_cfg = _load_station_config(station)
    baseline_period = station_cfg.get("baseline_period",
                                      station_cfg.get("ice_free_months", [6, 7, 8]))
    if winter_months is None:
        winter_months = _WINTER_MONTHS_DEFAULT

    results_dir = PROJECT_ROOT / "results_annual" / station
    arc_path = results_dir / f"{station}_{year}_arc_table.parquet"
    if not arc_path.exists():
        logger.error(f"arc_table not found: {arc_path}")
        return None

    arc_table = pd.read_parquet(arc_path)
    months = pd.to_datetime(arc_table["date"]).dt.month

    summer_mask = months.isin(baseline_period)
    winter_mask = months.isin(winter_months)

    logger.info(f"Arcs: {len(arc_table)} total, "
                f"{summer_mask.sum()} baseline, {winter_mask.sum()} winter")

    if summer_mask.sum() < 50 or winter_mask.sum() < 50:
        logger.error("Insufficient seasonal data for weight computation")
        return None

    baseline_arcs = arc_table[summer_mask]
    winter_arcs = arc_table[winter_mask]

    output = {
        "station": station,
        "year": year,
        "baseline_period": baseline_period,
        "winter_months": winter_months,
        "features": {},
        "summary": {},
    }

    for feature in _WEIGHT_FEATURES:
        if feature not in arc_table.columns:
            logger.debug(f"Skipping {feature} — not in arc_table")
            continue

        feature_weights = {}
        # Group by (sat, freq)
        all_combos = arc_table.groupby(["sat", "freq"]).groups.keys()

        for sat, freq in all_combos:
            key = f"{sat}_{freq}"

            summer_vals = baseline_arcs.loc[
                (baseline_arcs["sat"] == sat) & (baseline_arcs["freq"] == freq),
                feature
            ].dropna()
            winter_vals = winter_arcs.loc[
                (winter_arcs["sat"] == sat) & (winter_arcs["freq"] == freq),
                feature
            ].dropna()

            n_s, n_w = len(summer_vals), len(winter_vals)
            if n_s < _MIN_ARCS_PER_SEASON or n_w < _MIN_ARCS_PER_SEASON:
                feature_weights[key] = {
                    "d": None, "weight": 0.0,
                    "n_summer": n_s, "n_winter": n_w,
                    "reason": "insufficient_data",
                }
                continue

            d = _cohens_d(summer_vals.values, winter_vals.values)
            w = abs(d) if np.isfinite(d) else 0.0

            feature_weights[key] = {
                "d": round(float(d), 4) if np.isfinite(d) else None,
                "weight": round(w, 4),
                "n_summer": n_s,
                "n_winter": n_w,
            }

        output["features"][feature] = feature_weights

        # Summary stats
        weights = [v["weight"] for v in feature_weights.values()]
        n_weighted = sum(1 for w in weights if w > 0)
        output["summary"][feature] = {
            "n_combos": len(feature_weights),
            "n_weighted": n_weighted,
            "mean_weight": round(float(np.mean(weights)), 4) if weights else 0.0,
            "max_weight": round(float(np.max(weights)), 4) if weights else 0.0,
            "median_weight": round(float(np.median([w for w in weights if w > 0])), 4) if n_weighted > 0 else 0.0,
        }

        logger.info(f"  {feature}: {n_weighted}/{len(feature_weights)} combos weighted, "
                     f"mean |d|={output['summary'][feature]['mean_weight']:.3f}, "
                     f"max |d|={output['summary'][feature]['max_weight']:.3f}")

    return output


def save_prn_weights(output, station, year):
    """Save PRN weights to JSON."""
    results_dir = PROJECT_ROOT / "results_annual" / station
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{station}_{year}_prn_weights.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved PRN weights to {out_path}")
    return out_path


def load_prn_weights(station, year):
    """Load PRN weights from JSON. Returns None if not found."""
    path = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_prn_weights.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-PRN discriminating-power weights"
    )
    parser.add_argument("--station", required=True, help="Station ID")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--winter-months", type=int, nargs="+",
                        default=_WINTER_MONTHS_DEFAULT,
                        help="Winter months for ice reference (default: 12 1 2 3)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output = compute_prn_weights(args.station, args.year,
                                  winter_months=args.winter_months)
    if output is None:
        sys.exit(1)

    out_path = save_prn_weights(output, args.station, args.year)

    print(f"\n{'='*60}")
    print(f"PRN Weights: {args.station} {args.year}")
    print(f"{'='*60}")
    for feature, summary in output["summary"].items():
        print(f"  {feature:8s}: {summary['n_weighted']:3d}/{summary['n_combos']:3d} combos, "
              f"mean |d|={summary['mean_weight']:.3f}, "
              f"median |d|={summary['median_weight']:.3f}, "
              f"max |d|={summary['max_weight']:.3f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
