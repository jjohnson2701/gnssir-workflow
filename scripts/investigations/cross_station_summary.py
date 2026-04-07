#!/usr/bin/env python3
# ABOUTME: Cross-station comparison of v3 classification and feature seasonality.
# ABOUTME: Produces summary parquet and markdown report showing polarity inversions.

"""
Cross-Station Classification Summary

Scans all ice_classification_v3.parquet and daily_features.parquet files,
computes per-station-year phenology and feature seasonality, and produces:
  - results_annual/cross_station_summary.parquet  (one row per station-year)
  - results_annual/cross_station_report.md         (human-readable report)

The feature seasonality columns reveal which features flip polarity across
stations (the non-transferability problem for cross-station classifiers).

Usage:
    python scripts/cross_station_summary.py
    python scripts/cross_station_summary.py --results-dir results_annual/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "stations_config.json"

log = logging.getLogger(__name__)

FEATURES = ["amp_mean", "gamma_med", "clr_med", "af_med", "pr_med", "rh_std", "vs_med"]
V3_STATES = ["open_water", "freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"]
DISCOVERY_STATES = ["baseline", "anomalous", "regime_change"]
ICE_STATES = {"freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"}
ANOMALOUS_STATES = {"anomalous", "regime_change"}


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def discover_v3_files(results_dir: Path) -> list[tuple[str, int, Path]]:
    """Find all ice_classification_v3.parquet files."""
    found = []
    for f in sorted(results_dir.glob("*/*_ice_classification_v3.parquet")):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            station = parts[0]
            try:
                year = int(parts[1])
                found.append((station, year, f))
            except ValueError:
                continue
    return found


def _get_mode(v3_df):
    """Detect classification mode from v3 DataFrame."""
    if "classification_mode" in v3_df.columns:
        return v3_df["classification_mode"].iloc[0]
    if v3_df["v3_state"].isin(["baseline", "anomalous", "regime_change"]).any():
        return "discovery"
    return "validated"


def compute_phenology(v3_df: pd.DataFrame) -> dict:
    """Compute ice/anomaly season phenology from v3 classification."""
    mode = _get_mode(v3_df)
    counts = v3_df["v3_state"].value_counts()

    # State counts — include both vocabularies
    state_list = DISCOVERY_STATES if mode == "discovery" else V3_STATES
    row = {f"n_{s}": int(counts.get(s, 0)) for s in state_list}
    # Ensure all standard columns exist (zeroed) for consistent schema
    for s in V3_STATES + DISCOVERY_STATES:
        row.setdefault(f"n_{s}", 0)
    row["n_days"] = len(v3_df)
    row["classification_mode"] = mode

    non_baseline = {"anomalous", "regime_change"} if mode == "discovery" else ICE_STATES
    active_days = v3_df[v3_df["v3_state"].isin(non_baseline)]
    if len(active_days) > 0:
        row["first_freeze_doy"] = int(active_days["doy"].min())
        row["last_ice_doy"] = int(active_days["doy"].max())
        row["ice_season_length"] = row["last_ice_doy"] - row["first_freeze_doy"]
    else:
        row["first_freeze_doy"] = np.nan
        row["last_ice_doy"] = np.nan
        row["ice_season_length"] = np.nan

    return row


def cohens_d(group_a, group_b):
    """Compute Cohen's d effect size between two groups."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    ma, mb = group_a.mean(), group_b.mean()
    va, vb = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (mb - ma) / pooled_std


def compute_feature_seasonality(
    features_df: pd.DataFrame,
    v3_df: pd.DataFrame,
    ice_free_months: list[int],
) -> dict:
    """Compute summer vs winter feature medians and Cohen's d.

    Winter months are determined as the 3 months with the most ice days,
    falling back to [12, 1, 2] if no ice days exist.
    """
    pooled = features_df[features_df["azimuth_bin"] == -1].copy()
    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled["month"] = pooled["date"].dt.month
    pooled["doy"] = pooled["date"].dt.dayofyear

    # Determine winter months from actual ice/anomalous days
    mode = _get_mode(v3_df)
    non_baseline = ANOMALOUS_STATES if mode == "discovery" else ICE_STATES
    ice_days = v3_df[v3_df["v3_state"].isin(non_baseline)]
    if len(ice_days) > 0:
        ice_months = (
            ice_days.assign(month=pd.to_datetime(ice_days["date"]).dt.month)
            ["month"].value_counts()
            .head(3).index.tolist()
        )
    else:
        ice_months = [12, 1, 2]

    summer = pooled[pooled["month"].isin(ice_free_months)]
    winter = pooled[pooled["month"].isin(ice_months)]

    row = {}
    for feat in FEATURES:
        if feat not in pooled.columns:
            for suffix in ["_summer_median", "_winter_median", "_direction", "_separation"]:
                row[f"{feat}{suffix}"] = np.nan
            continue

        s_vals = summer[feat].dropna()
        w_vals = winter[feat].dropna()

        s_med = float(s_vals.median()) if len(s_vals) > 0 else np.nan
        w_med = float(w_vals.median()) if len(w_vals) > 0 else np.nan

        row[f"{feat}_summer_median"] = s_med
        row[f"{feat}_winter_median"] = w_med

        if np.isnan(s_med) or np.isnan(w_med):
            row[f"{feat}_direction"] = "insufficient_data"
            row[f"{feat}_separation"] = np.nan
        else:
            row[f"{feat}_direction"] = "winter_high" if w_med > s_med else "summer_high"
            row[f"{feat}_separation"] = cohens_d(s_vals, w_vals)

    row["winter_months"] = ice_months
    return row


def build_summary(results_dir: Path) -> pd.DataFrame:
    """Build the full cross-station summary DataFrame."""
    cfg = load_config()
    v3_files = discover_v3_files(results_dir)

    if not v3_files:
        log.error("No ice_classification_v3.parquet files found")
        return pd.DataFrame()

    log.info(f"Found {len(v3_files)} station-years with v3 classification")

    rows = []
    for station, year, v3_path in v3_files:
        log.info(f"  {station} {year}...")
        v3_df = pd.read_parquet(v3_path)
        v3_df["date"] = pd.to_datetime(v3_df["date"])
        if "doy" not in v3_df.columns:
            v3_df["doy"] = v3_df["date"].dt.dayofyear

        # Metadata
        scfg = cfg.get(station, {})
        row = {
            "station": station,
            "year": year,
            "lat": scfg.get("latitude_deg", np.nan),
            "lon": scfg.get("longitude_deg", np.nan),
        }

        # Phenology
        row.update(compute_phenology(v3_df))

        # Feature seasonality
        feat_path = results_dir / station / f"{station}_{year}_daily_features.parquet"
        if feat_path.exists():
            feat_df = pd.read_parquet(feat_path)
            ice_free = scfg.get("ice_free_months", [6, 7, 8])
            row.update(compute_feature_seasonality(feat_df, v3_df, ice_free))
        else:
            log.warning(f"    No daily_features for {station} {year}")

        rows.append(row)

    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame) -> str:
    """Generate the markdown report from the summary DataFrame."""
    lines = []
    lines.append("# Cross-Station Classification Report")
    lines.append("")
    lines.append(f"Generated from {len(df)} station-years across "
                 f"{df['station'].nunique()} stations.")
    lines.append("")

    # ---- Section 1: Station inventory ----
    lines.append("## 1. Station Inventory")
    lines.append("")
    inv = (
        df.groupby("station")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            years=("year", lambda x: f"{x.min()}-{x.max()}"),
            n_years=("year", "count"),
            total_days=("n_days", "sum"),
        )
        .reset_index()
        .sort_values("station")
    )
    # Add mode column
    mode_map = df.groupby("station")["classification_mode"].first().to_dict() if "classification_mode" in df.columns else {}
    lines.append("| Station | Lat | Lon | Mode | Years | N Years | Total Days |")
    lines.append("|---------|-----|-----|------|-------|---------|------------|")
    for _, r in inv.iterrows():
        mode = mode_map.get(r["station"], "?")
        lines.append(
            f"| {r['station']} | {r['lat']:.2f} | {r['lon']:.2f} | "
            f"{mode} | {r['years']} | {r['n_years']} | {r['total_days']} |"
        )
    lines.append("")

    # ---- Section 2: Ice season phenology ----
    lines.append("## 2. Ice Season Phenology")
    lines.append("")
    lines.append("Station-years with at least one ice/anomalous day.")
    lines.append("")
    ice_df = df[df["first_freeze_doy"].notna()].copy()
    if len(ice_df) > 0:
        lines.append("| Station | Year | Mode | First DOY | Last DOY | Length | Active Days |")
        lines.append("|---------|------|------|-----------|----------|--------|-------------|")
        for _, r in ice_df.sort_values(["station", "year"]).iterrows():
            mode = r.get("classification_mode", "validated")
            if mode == "discovery":
                active = sum(r.get(f"n_{s}", 0) for s in ANOMALOUS_STATES)
            else:
                active = sum(r.get(f"n_{s}", 0) for s in ICE_STATES)
            lines.append(
                f"| {r['station']} | {r['year']} | {mode} | {int(r['first_freeze_doy'])} | "
                f"{int(r['last_ice_doy'])} | {int(r['ice_season_length'])} | {active} |"
            )
    else:
        lines.append("No station-years with ice days found.")
    lines.append("")

    # ---- Section 3: Feature direction comparison ----
    lines.append("## 3. Feature Direction Comparison")
    lines.append("")
    lines.append("Direction indicates whether the feature is higher in winter (ice season) "
                 "or summer (open water). Cohen's d measures effect size. "
                 "**Bold** entries disagree with the majority direction across stations.")
    lines.append("")

    # Compute per-station median direction across years
    station_feat = []
    for station in sorted(df["station"].unique()):
        sdf = df[df["station"] == station]
        row = {"station": station}
        for feat in FEATURES:
            dirs = sdf[f"{feat}_direction"].dropna()
            seps = sdf[f"{feat}_separation"].dropna()
            if len(dirs) == 0 or (dirs == "insufficient_data").all():
                row[f"{feat}_dir"] = "?"
                row[f"{feat}_d"] = np.nan
            else:
                valid = dirs[dirs != "insufficient_data"]
                row[f"{feat}_dir"] = valid.mode().iloc[0] if len(valid) > 0 else "?"
                row[f"{feat}_d"] = float(seps.median()) if len(seps) > 0 else np.nan
        station_feat.append(row)
    sfdf = pd.DataFrame(station_feat)

    # Determine majority direction per feature
    majority = {}
    for feat in FEATURES:
        col = f"{feat}_dir"
        valid = sfdf[col][(sfdf[col] != "?") & (sfdf[col] != "insufficient_data")]
        if len(valid) > 0:
            majority[feat] = valid.mode().iloc[0]
        else:
            majority[feat] = None

    # Build table
    header = "| Station | " + " | ".join(FEATURES) + " |"
    sep = "|---------|" + "|".join(["------" for _ in FEATURES]) + "|"
    lines.append(header)
    lines.append(sep)

    for _, r in sfdf.iterrows():
        cells = [r["station"]]
        for feat in FEATURES:
            d_val = r[f"{feat}_d"]
            direction = r[f"{feat}_dir"]
            if direction == "?" or pd.isna(d_val):
                cells.append("?")
                continue

            arrow = "\u2191" if direction == "winter_high" else "\u2193"
            d_str = f"{abs(d_val):.2f}"
            cell = f"{arrow} {d_str}"

            # Bold if disagrees with majority
            if majority[feat] and direction != majority[feat]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Legend
    lines.append("\u2191 = winter_high (feature increases with ice), "
                 "\u2193 = summer_high (feature decreases with ice)")
    lines.append("")
    lines.append("Majority direction per feature:")
    for feat in FEATURES:
        maj = majority.get(feat, "?")
        arrow = "\u2191" if maj == "winter_high" else ("\u2193" if maj == "summer_high" else "?")
        lines.append(f"  - **{feat}**: {arrow} {maj}")
    lines.append("")

    # ---- Section 4: State distribution ----
    lines.append("## 4. State Distribution")
    lines.append("")

    # Validated stations
    val_df = df[df.get("classification_mode", "validated") == "validated"] if "classification_mode" in df.columns else df
    if len(val_df) > 0:
        lines.append("### Validated stations (ice vocabulary)")
        lines.append("")
        lines.append("| Station | Year | Days | " + " | ".join(V3_STATES) + " |")
        lines.append("|---------|------|------|" + "|".join(["----" for _ in V3_STATES]) + "|")
        for _, r in val_df.sort_values(["station", "year"]).iterrows():
            cells = [r["station"], str(r["year"]), str(r["n_days"])]
            for s in V3_STATES:
                v = int(r.get(f"n_{s}", 0))
                cells.append(str(v) if v > 0 else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Discovery stations
    disc_df = df[df["classification_mode"] == "discovery"] if "classification_mode" in df.columns else pd.DataFrame()
    if len(disc_df) > 0:
        lines.append("### Discovery stations (generic labels)")
        lines.append("")
        lines.append("| Station | Year | Days | " + " | ".join(DISCOVERY_STATES) + " |")
        lines.append("|---------|------|------|" + "|".join(["----" for _ in DISCOVERY_STATES]) + "|")
        for _, r in disc_df.sort_values(["station", "year"]).iterrows():
            cells = [r["station"], str(r["year"]), str(r["n_days"])]
            for s in DISCOVERY_STATES:
                v = int(r.get(f"n_{s}", 0))
                cells.append(str(v) if v > 0 else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-station classification summary and report"
    )
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: results_annual/)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results_annual"

    # Build summary
    summary = build_summary(results_dir)
    if summary.empty:
        print("ERROR: No data found")
        sys.exit(1)

    # Save parquet
    out_parquet = results_dir / "cross_station_summary.parquet"
    summary.to_parquet(out_parquet, index=False)
    log.info(f"Saved {out_parquet} ({len(summary)} rows, {len(summary.columns)} columns)")

    # Generate and save report
    report = generate_report(summary)
    out_md = results_dir / "cross_station_report.md"
    out_md.write_text(report)
    log.info(f"Saved {out_md}")

    # Print summary stats
    n_stations = summary["station"].nunique()
    n_years = len(summary)
    n_ice = summary["first_freeze_doy"].notna().sum()
    print(f"\n{n_stations} stations, {n_years} station-years, {n_ice} with ice")
    print(f"Outputs:")
    print(f"  {out_parquet}")
    print(f"  {out_md}")


if __name__ == "__main__":
    main()
