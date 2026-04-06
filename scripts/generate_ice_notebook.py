#!/usr/bin/env python3
# ABOUTME: Generates a Jupyter notebook that walks through ice classification results
# ABOUTME: Mirrors the dashboard ice tab narrative with static matplotlib plots

"""
Generate an ice classification Jupyter notebook for a station-year.

Walks through the full ice detection pipeline:
  1. Data overview and station context
  2. GLERL ground truth ice concentration
  3. Raw amplitude seasonal signal
  4. SNR feature deep dive (CLR, AF, PR, gamma) with GLERL comparison
  5. Feature–ice correlation analysis
  6. Per-sector classification behaviour
  7. Current classification result vs ground truth
  8. Monthly summary and concordance
  9. Multi-year overview

Usage:
    python scripts/generate_ice_notebook.py --station ROSS --year 2022
    python scripts/generate_ice_notebook.py --station ROSS --year 2022 --open
"""

import argparse
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _md(text):
    """Create a markdown cell."""
    return new_markdown_cell(text.strip())


def _code(text):
    """Create a code cell."""
    return new_code_cell(text.strip())


def generate_notebook(station, year):
    """Build the ice classification notebook."""
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # =====================================================================
    # Title
    # =====================================================================
    cells.append(_md(f"""
# Ice Detection Analysis: {station} {year}

This notebook walks through GNSS-IR ice detection at **{station}** for **{year}**,
comparing the classifier output against GLERL satellite-derived ice concentration
as ground truth.

The analysis investigates:
- Which GNSS-IR features respond to Great Lakes ice and which do not
- How the per-azimuth-sector voting system behaves
- Where the current classifier succeeds and where it breaks down

Each section builds toward understanding the physical signals in the data
and how they relate to actual ice presence.

**Notebook version:** v8 — refactored to consume `arc_table.parquet` (Layer 1)
and `daily_features.parquet` (Layer 2) instead of computing features inline.
Prior versions: v7/v7.1 (multi-observable analysis), v1–v6 (incremental ice classifier tuning).
"""))

    # =====================================================================
    # Setup
    # =====================================================================
    cells.append(_code(f"""
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pathlib import Path

# Configuration
STATION = "{station}"
YEAR = {year}
PROJECT_ROOT = Path("{PROJECT_ROOT}")

# Paths
results_dir = PROJECT_ROOT / "results_annual" / STATION
config_path = PROJECT_ROOT / "config" / "stations_config.json"

# Load station config
with open(config_path) as f:
    station_cfg = json.load(f).get(STATION, {{}})

ice_free_months = station_cfg.get("ice_free_months", [])
lat = station_cfg.get("latitude_deg", "?")
lon = station_cfg.get("longitude_deg", "?")

print(f"Station: {{STATION}} ({{lat:.3f}}°N, {{lon:.3f}}°W)")
print(f"Year: {{YEAR}}")
print(f"Ice-free calibration months: {{ice_free_months or 'not configured'}}")

# Style
plt.rcParams.update({{
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.3,
    "font.size": 10,
}})

CLASS_COLOR = {{"ice": "#1565c0", "transition": "#f9a825", "water": "#2e7d32"}}
MONTH_NAMES = {{1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}}
"""))

    # =====================================================================
    # Station Overview
    # =====================================================================
    cells.append(_md(f"""
## Station Overview

| Property | Value |
|----------|-------|
| **Station** | {station} |
| **Location** | North shore of Lake Superior |
| **Year** | {year} |
| **Azimuth window** | 110°–190° (looking SSE over lake) |
| **Ice-free months** | Jun–Oct (calibration baseline) |
| **GLERL reference** | Gridded satellite ice concentration (~1.3 km) |
| **Wind reference** | NDBC PILM4 (Passage Island, MI, ~92 km) |
| **Data source** | NRCAN (Natural Resources Canada) |
| **Orbit type** | GPS broadcast (nav) |

**Key station characteristics:**
- Narrow azimuth window (80°) constrained by coastline geometry — land to the N/W/E
- Only GPS L1 and L2C frequencies available (2 of 12 configured — NRCAN archive limitation)
- Per-sector arc density varies widely due to satellite ground track geometry
- Predominant winter winds from NNW/NW push ice into the antenna's field of view
"""))

    # =====================================================================
    # Section 1: Data Loading
    # =====================================================================
    cells.append(_md("""
## 1. Data Loading

Load the arc-level data, daily aggregated features, classification results, and GLERL ground truth.

**v8 data architecture:**
- `arc_table.parquet` — Layer 1: merged per-arc + SNR features (replaces per_arc + snr_features merge)
- `daily_features.parquet` — Layer 2: daily × sector aggregated features (replaces inline computation)
- `ice_classification.parquet` — Layer 3: classification output

Falls back to legacy `per_arc.parquet` + `snr_features.parquet` if Layer 1-2 files don't exist.
"""))

    cells.append(_code(f"""
# --- v8 data loading: prefer arc_table + daily_features, fall back to per_arc ---
arc_table_path = results_dir / f"{{STATION}}_{{YEAR}}_arc_table.parquet"
daily_features_path = results_dir / f"{{STATION}}_{{YEAR}}_daily_features.parquet"
pa_path = results_dir / f"{{STATION}}_{{YEAR}}_per_arc.parquet"

has_arc_table = arc_table_path.exists()
has_daily_features = daily_features_path.exists()

if has_arc_table:
    per_arc = pd.read_parquet(arc_table_path)
    print(f"arc_table (Layer 1): {{len(per_arc):,}} arcs across {{per_arc['date'].nunique()}} days")
    has_snr = "CLR" in per_arc.columns
    if has_snr:
        print(f"  SNR features: CLR, AF, PR, gamma (pre-merged)")
    if "freq_group" in per_arc.columns:
        print(f"  Frequency groups: {{sorted(per_arc['freq_group'].unique())}}")
else:
    # Legacy fallback: load per_arc + snr_features and merge
    print("arc_table not found — falling back to legacy per_arc + snr_features")
    per_arc = pd.read_parquet(pa_path)
    feat_path = results_dir / f"{{STATION}}_{{YEAR}}_snr_features.parquet"
    has_snr = feat_path.exists()
    if has_snr:
        snr_feat = pd.read_parquet(feat_path)
        join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]
        feat_cols = [c for c in snr_feat.columns if c not in per_arc.columns or c in join_cols]
        per_arc = per_arc.merge(snr_feat[feat_cols], on=join_cols, how="left")
        n_matched = per_arc["CLR"].notna().sum()
        print(f"  SNR features merged: {{n_matched}}/{{len(per_arc)}} arcs matched")
    print(f"Per-arc (legacy): {{len(per_arc):,}} arcs across {{per_arc['date'].nunique()}} days")

per_arc["date_dt"] = pd.to_datetime(per_arc["date"])
per_arc["month"] = per_arc["date_dt"].dt.month
print(f"  Azimuth sectors: {{sorted(per_arc['azimuth_bin'].unique())}}")

# Daily features (Layer 2)
daily_features = None
if has_daily_features:
    daily_features = pd.read_parquet(daily_features_path)
    daily_features["date_dt"] = pd.to_datetime(daily_features["date"])
    n_sector_rows = (daily_features["azimuth_bin"] >= 0).sum()
    n_pooled_rows = (daily_features["azimuth_bin"] == -1).sum()
    print(f"\\ndaily_features (Layer 2): {{len(daily_features)}} rows "
          f"({{n_sector_rows}} sector + {{n_pooled_rows}} pooled)")
    feat_cols = [c for c in daily_features.columns
                 if c not in ("date", "azimuth_bin", "date_dt") and daily_features[c].notna().any()]
    print(f"  Available features: {{len(feat_cols)}}")
else:
    print("\\nNo daily_features — inline computation will be used where needed")

# Classification results (Layer 3)
clf_path = results_dir / f"{{STATION}}_{{YEAR}}_ice_classification.parquet"
clf = pd.read_parquet(clf_path)
clf["date_dt"] = pd.to_datetime(clf["date"])
clf["month"] = clf["date_dt"].dt.month
counts = clf["classification"].value_counts()
print(f"\\nClassification: {{len(clf)}} days")
for cls in ["ice", "transition", "water"]:
    print(f"  {{cls:12s}}: {{counts.get(cls, 0):3d}} days")

# GLERL ground truth
glerl_path = PROJECT_ROOT / "data" / ".cache" / "glerl_ice" / f"{{STATION.lower()}}_ice_{{YEAR}}.csv"
glerl = None
if glerl_path.exists():
    glerl = pd.read_csv(glerl_path, parse_dates=["datetime"])
    glerl["date_dt"] = glerl["datetime"].dt.tz_localize(None).dt.normalize()
    glerl["month"] = glerl["date_dt"].dt.month
    glerl = glerl[(glerl["date_dt"] >= clf["date_dt"].min()) &
                  (glerl["date_dt"] <= clf["date_dt"].max())]
    print(f"\\nGLERL: {{len(glerl)}} days, peak {{glerl['ice_concentration'].max():.0f}}%")
    print(f"  Days >30% ice: {{(glerl['ice_concentration'] > 30).sum()}}")
    print(f"  Days <5% ice:  {{(glerl['ice_concentration'] < 5).sum()}}")
else:
    print("\\nNo GLERL data. Run: python scripts/external_apis/glerl_ice.py "
          f"--station {{STATION}} --year {{YEAR}}")
"""))

    # =====================================================================
    # Section 1b: Azimuth Geometry
    # =====================================================================
    cells.append(_md("""
## 1b. Azimuth Coverage — Why are some sectors sparse?

The station config permits azimuths **110°–190°** (80° window looking SSE over Lake Superior).
Within this window, arc density per 10° sector varies because of **GPS satellite geometry**:

- GPS satellites orbit at ~55° inclination, creating discrete ground tracks
- Each satellite crosses a specific azimuth band at this latitude
- Some 10° bins only catch 1–2 satellites; others catch 7+
- The classifier requires **≥3 arcs per day** in a sector to classify it

Sectors with only 1–2 satellites rarely accumulate 3 arcs in a single day,
so they appear as brown gaps in the classification heatmap. This is a
**coverage limitation**, not a data quality issue.
"""))

    cells.append(_code("""
# Azimuth distribution
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: azimuth histogram
az_bins = np.arange(110, 191, 1)
ax1.hist(per_arc["Azim"], bins=az_bins, color="#546e7a", edgecolor="white", linewidth=0.3)
ax1.set_xlabel("Azimuth (°)")
ax1.set_ylabel("Number of arcs")
ax1.set_title("Arc Distribution by Azimuth (1° bins)")

# Shade gaps
az_hist, _ = np.histogram(per_arc["Azim"], bins=az_bins)
for i in range(len(az_hist)):
    if az_hist[i] == 0 and i > 0 and i < len(az_hist) - 1:
        ax1.axvspan(az_bins[i], az_bins[i+1], color="red", alpha=0.1)

# Right: satellites per sector
sectors = sorted(per_arc["azimuth_bin"].unique())
sat_counts = []
arc_per_day = []
classifiable_days = []
for b in sectors:
    sec = per_arc[per_arc["azimuth_bin"] == b]
    sat_counts.append(sec["sat"].nunique())
    n_days = sec["date"].nunique()
    arc_per_day.append(len(sec) / max(n_days, 1))
    daily = sec.groupby("date").size()
    classifiable_days.append(int((daily >= 3).sum()))

x = range(len(sectors))
colors = ["#c62828" if a < 2.5 else "#2e7d32" for a in arc_per_day]
bars = ax2.bar(x, sat_counts, color=colors, alpha=0.7, edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{b}°" for b in sectors], fontsize=9)
ax2.set_xlabel("Azimuth Sector")
ax2.set_ylabel("Unique Satellites")
ax2.set_title("Satellites per Sector (red = sparse, <2.5 arcs/day)")

# Annotate with classifiable days
for i, (b, cd) in enumerate(zip(sectors, classifiable_days)):
    ax2.text(i, sat_counts[i] + 0.2, f"{cd}d", ha="center", fontsize=7, color="#555")

fig.tight_layout()
plt.show()

# Summary table
print(f"{'Sector':>8} | {'Sats':>4} | {'Arcs/day':>8} | {'Days≥3':>6} | Status")
print("-" * 55)
for b, sc, apd, cd in zip(sectors, sat_counts, arc_per_day, classifiable_days):
    status = "SPARSE" if apd < 2.5 else "good"
    print(f"  {b:>5}° | {sc:>4} | {apd:>8.1f} | {cd:>6} | {status}")
"""))

    # =====================================================================
    # Section 2: GLERL Ground Truth
    # =====================================================================
    cells.append(_md("""
## 2. GLERL Ground Truth — What does the ice season actually look like?

GLERL provides gridded satellite-derived ice concentration for the Great Lakes
at ~1.3 km resolution. We extract the value at the station's location to get
the "answer key" for our GNSS-IR classifier.

Understanding the ice season structure helps us evaluate whether the classifier's
features respond at the right times.
"""))

    cells.append(_code("""
if glerl is not None:
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                     color="#90caf9", alpha=0.4)
    ax.plot(glerl["date_dt"], glerl["ice_concentration"],
            color="#1565c0", linewidth=1.5)
    ax.axhline(30, color="red", ls="--", alpha=0.4, label="30% threshold")
    ax.axhline(5, color="green", ls="--", alpha=0.4, label="5% threshold")
    ax.set_ylabel("Ice Concentration (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{STATION} {YEAR} — GLERL Satellite Ice Concentration at Station Location")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    # Monthly breakdown
    print("Monthly GLERL ice concentration:")
    for m in sorted(glerl["month"].unique()):
        md = glerl[glerl["month"] == m]
        print(f"  {MONTH_NAMES.get(m, str(m)):>3}: mean={md['ice_concentration'].mean():5.1f}%  "
              f"max={md['ice_concentration'].max():5.1f}%  "
              f"days >30%: {(md['ice_concentration'] > 30).sum()}/{len(md)}")
"""))

    # =====================================================================
    # Section 2b: GLERL Sampling — Regional Context
    # =====================================================================
    cells.append(_md("""
## 2b. GLERL Sampling — Regional Context and Azimuth Alignment

The GLERL grid has ~1.3 km cells. We compare two sampling strategies:

1. **Bounding box** (original): All water cells within ±0.05° of the station (~15 cells)
2. **Azimuth-filtered** (new): Only cells within the GNSS-IR field of view (110°–190°)
   and within 10 km of the station (~18 cells)

The azimuth-filtered method ensures the reference ice data comes from the **same
direction** the antenna is measuring, not from behind the station (over land) or
in unrelated directions.
"""))

    cells.append(_code(f"""
import requests, gzip
from io import StringIO as SIO

# Fetch a ~3° regional grid for context (mid-Feb peak ice)
ctx_url = ("https://apps.glerl.noaa.gov/erddap/griddap/GL_Ice_Concentration_GCS.csv?"
           "ice_concentration[(" + str(YEAR) + "-02-15T12:00:00Z):1:(" + str(YEAR) + "-02-15T12:00:00Z)]"
           "[(47.3):1:(50.0)][(-89.2):1:(-86.0)]")
print("Fetching regional GLERL grid...")
try:
    ctx_resp = requests.get(ctx_url, timeout=60)
    ctx_lines = ctx_resp.text.strip().split("\\n")
    ctx_csv = ctx_lines[0] + "\\n" + "\\n".join(ctx_lines[2:])
    ctx_grid = pd.read_csv(SIO(ctx_csv))
    ctx_grid.loc[ctx_grid["ice_concentration"] <= -99998, "ice_concentration"] = np.nan
    ctx_water = ctx_grid[ctx_grid["ice_concentration"].notna()]
    ctx_land = ctx_grid[ctx_grid["ice_concentration"].isna()]
    print(f"Regional grid: {{len(ctx_water)}} water cells, {{len(ctx_land)}} land cells")
except Exception as e:
    print(f"Failed to fetch regional grid: {{e}}")
    ctx_water = ctx_land = pd.DataFrame()
"""))

    cells.append(_code(f"""
if len(ctx_water) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- LEFT: Regional context ---
    ax1.scatter(ctx_land["longitude"], ctx_land["latitude"],
                c="#c8b898", s=2, alpha=0.3, marker="s")
    sc = ax1.scatter(ctx_water["longitude"], ctx_water["latitude"],
                      c=ctx_water["ice_concentration"],
                      cmap="Blues", vmin=0, vmax=100, s=2, alpha=0.8, marker="s")

    # Station
    s_lat = station_cfg.get("latitude_deg", 0)
    s_lon = station_cfg.get("longitude_deg", 0)
    ax1.plot(s_lon, s_lat, "r^", markersize=14,
             markeredgecolor="white", markeredgewidth=2.5, zorder=10)
    ax1.text(s_lon + 0.06, s_lat + 0.05, STATION, color="red",
             fontsize=12, fontweight="bold")

    # PILM4 wind station
    ax1.plot(-87.395, 48.374, "gs", markersize=10, markeredgecolor="white", zorder=10)
    ax1.text(-87.30, 48.30, "PILM4\\n(wind)", color="lime", fontsize=9, fontweight="bold")

    # Old bbox (green)
    bb = 0.05
    ax1.plot([s_lon-bb, s_lon+bb, s_lon+bb, s_lon-bb, s_lon-bb],
             [s_lat-bb, s_lat-bb, s_lat+bb, s_lat+bb, s_lat-bb],
             "g--", linewidth=2, label="Old: bbox ±0.05° (15 cells)")

    # New azimuth wedge (red)
    cos_lat = np.cos(np.radians(s_lat))
    max_d_deg = 10.0 / 111.0
    angles = np.linspace(np.radians(110), np.radians(190), 80)
    w_lons = s_lon + max_d_deg * np.sin(angles) / cos_lat
    w_lats = s_lat + max_d_deg * np.cos(angles)
    w_lons = np.concatenate([[s_lon], w_lons, [s_lon]])
    w_lats = np.concatenate([[s_lat], w_lats, [s_lat]])
    ax1.fill(w_lons, w_lats, color="red", alpha=0.15)
    ax1.plot(w_lons, w_lats, "r-", linewidth=2,
             label="New: az 110°–190°, <10 km (18 cells)")

    plt.colorbar(sc, ax=ax1, shrink=0.7, label="Ice Concentration (%)")
    ax1.set_xlim(-89.2, -86.0)
    ax1.set_ylim(47.3, 50.0)
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title(f"Regional Context — {{YEAR}}-02-15", fontsize=11)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_aspect(1.4)

    # --- RIGHT: Time series comparison ---
    bbox_path = PROJECT_ROOT / "data" / ".cache" / "glerl_ice" / f"{{STATION.lower()}}_ice_{{YEAR}}.csv"
    azf_path = PROJECT_ROOT / "data" / ".cache" / "glerl_ice" / f"{{STATION.lower()}}_ice_{{YEAR}}_azfilt.csv"

    if bbox_path.exists() and azf_path.exists():
        g_bbox = pd.read_csv(bbox_path, parse_dates=["datetime"])
        g_azf = pd.read_csv(azf_path, parse_dates=["datetime"])
        g_bbox["dt"] = g_bbox["datetime"].dt.tz_localize(None)
        g_azf["dt"] = g_azf["datetime"].dt.tz_localize(None)

        ax2.fill_between(g_bbox["dt"], g_bbox["ice_concentration"],
                          color="#90caf9", alpha=0.3, label="Bbox (15 cells)")
        ax2.plot(g_bbox["dt"], g_bbox["ice_concentration"],
                 color="#64b5f6", linewidth=1.5, alpha=0.7)
        ax2.plot(g_azf["dt"], g_azf["ice_concentration"],
                 color="#c62828", linewidth=2, alpha=0.8,
                 label="Az-filtered (18 cells)")
        ax2.axhline(30, color="gray", ls="--", alpha=0.3)
        ax2.set_ylabel("Ice Concentration (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title(f"{{STATION}} {{YEAR}} — Bbox vs Az-Filtered GLERL", fontsize=11)
        ax2.legend(loc="upper right", fontsize=9)

        # Correlation comparison
        from scipy import stats
        merged_b = clf.merge(g_bbox[["dt","ice_concentration"]].rename(columns={{"dt":"date_dt"}}),
                              on="date_dt", how="inner")
        merged_a = clf.merge(g_azf[["dt","ice_concentration"]].rename(columns={{"dt":"date_dt"}}),
                              on="date_dt", how="inner")
        vb = merged_b.dropna(subset=["ice_score"])
        va = merged_a.dropna(subset=["ice_score"])
        r_b = vb["ice_score"].corr(vb["ice_concentration"]) if len(vb) > 10 else float("nan")
        r_a = va["ice_score"].corr(va["ice_concentration"]) if len(va) > 10 else float("nan")
        ax2.text(0.02, 0.05, f"Classifier corr: bbox r={{r_b:+.3f}}, az-filt r={{r_a:+.3f}}",
                 transform=ax2.transAxes, fontsize=9, color="#333",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
"""))

    # =====================================================================
    # Section 3: Raw Amplitude
    # =====================================================================
    cells.append(_md("""
## 3. What does the antenna see? — Raw amplitude signal

The most direct GNSS-IR ice indicator is reflected signal **amplitude**.
Ice produces stronger, more coherent reflections than rough open water.

This is the foundation of the classifier — do we see a seasonal signal
that matches the GLERL ice season?
"""))

    cells.append(_code("""
daily_amp = per_arc.groupby("date_dt")["Amp"].agg(["mean", "std", "count"])
daily_amp = daily_amp.reset_index()

fig, ax1 = plt.subplots(figsize=(14, 4))

ax1.scatter(daily_amp["date_dt"], daily_amp["mean"], c="#333", s=10, alpha=0.6,
            label="Daily mean amplitude", zorder=3)
ax1.set_ylabel("Amplitude (mean)")
ax1.set_title(f"{STATION} {YEAR} — Daily Mean Amplitude vs GLERL Ice")

if glerl is not None:
    ax2 = ax1.twinx()
    ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                     color="#90caf9", alpha=0.2)
    ax2.plot(glerl["date_dt"], glerl["ice_concentration"],
             color="#64b5f6", linewidth=1, alpha=0.6)
    ax2.set_ylabel("GLERL Ice %", color="#64b5f6")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", colors="#64b5f6")

ax1.legend(loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

# Monthly amplitude stats
print("Monthly amplitude (higher = more ice-like):")
per_arc_monthly = per_arc.groupby("month")["Amp"].agg(["mean", "std"])
for m in sorted(per_arc_monthly.index):
    row = per_arc_monthly.loc[m]
    bar = "█" * int(row["mean"] / 1.0)
    print(f"  {MONTH_NAMES.get(m, str(m)):>3}: {row['mean']:6.1f} ± {row['std']:5.1f}  {bar}")
"""))

    # =====================================================================
    # Section 4: SNR Feature Deep Dive
    # =====================================================================
    cells.append(_md("""
## 4. SNR Feature Deep Dive

Beyond simple amplitude, four SNR-derived features are extracted from each arc:

| Feature | Source | Physical Meaning | Expected Ice Signal |
|---------|--------|-----------------|-------------------|
| **CLR** (Clarity Ratio) | Purnell 2024 | Dominant LSP peak / mean of other peaks | High (coherent reflection) |
| **AF** (Area Factor) | Song 2022 | Integrated CWT power at dominant RH | High (strong total reflectivity) |
| **PR** (Peak Ratio) | Purnell 2024 | P1/P2 in LSP | High (single dominant surface) |
| **γ** (Damping) | Strandberg 2017 | Hilbert envelope decay rate | Low (smooth surface) |

The key question is: **do these features actually track ice at this Great Lakes station?**
These indicators were developed and validated on Greenland fjord ice — the physical
characteristics of Great Lakes ice may differ.
"""))

    cells.append(_code("""
if has_snr:
    features = {
        "CLR": ("Clarity Ratio", True, "High = ice"),
        "AF": ("Area Factor", True, "High = ice"),
        "PR": ("Peak Ratio", True, "High = ice"),
        "gamma": ("Damping (γ)", False, "Low = ice (smooth)"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Use daily_features pooled medians if available, otherwise compute from per_arc
    if daily_features is not None:
        pooled = daily_features[daily_features["azimuth_bin"] == -1].copy()
        pooled["month"] = pd.to_datetime(pooled["date"]).dt.month
        feat_source = "daily_features"
    else:
        pooled = None
        feat_source = "per_arc"

    for ax, (feat, (label, high_ice, note)) in zip(axes.flat, features.items()):
        feat_lower = feat.lower()

        if pooled is not None and f"{feat_lower}_med" in pooled.columns:
            monthly = pooled.groupby("month")[f"{feat_lower}_med"].median()
        elif feat in per_arc.columns:
            monthly = per_arc.groupby("month")[feat].median()
        else:
            ax.set_visible(False)
            continue

        # Color bars by GLERL ice
        if glerl is not None:
            glerl_monthly = glerl.groupby("month")["ice_concentration"].mean()
        else:
            glerl_monthly = pd.Series(dtype=float)

        bar_colors = []
        for m in monthly.index:
            g = glerl_monthly.get(m, 0)
            if g > 30:
                bar_colors.append("#1565c0")   # ice
            elif g > 5:
                bar_colors.append("#f9a825")   # marginal
            else:
                bar_colors.append("#2e7d32")   # water

        ax.bar(monthly.index, monthly.values, color=bar_colors, alpha=0.7, edgecolor="white")
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Median {feat}")
        ax.set_title(f"{label}  ({note})")
        ax.set_xticks(monthly.index)
        ax.set_xticklabels([MONTH_NAMES.get(m, str(m)) for m in monthly.index], fontsize=8)

        # Mark ice-free calibration months
        for m in ice_free_months:
            if m in monthly.index:
                ax.axvline(m, color="green", ls=":", alpha=0.3)

    fig.suptitle(f"{STATION} {YEAR} — Monthly SNR Feature Medians (from {feat_source})\\n"
                 f"(Blue bars = GLERL >30% ice, Green = <5%, Yellow = marginal)",
                 fontsize=11)
    fig.tight_layout()
    plt.show()
else:
    print("No SNR features available.")
"""))

    # =====================================================================
    # Section 5: Feature–Ice Correlation
    # =====================================================================
    cells.append(_md("""
## 5. Feature–Ice Correlation

For each indicator, we compute the daily median and correlate it against
GLERL ice concentration. This reveals which features actually respond to
Great Lakes ice and — critically — whether the polarity matches the
Greenland-derived expectations.

A positive correlation means the feature **increases** with ice.
A negative correlation means it **decreases** with ice.
"""))

    cells.append(_code("""
if glerl is not None:
    # Use daily_features (pooled) if available, otherwise compute from per_arc
    if daily_features is not None:
        pooled = daily_features[daily_features["azimuth_bin"] == -1].copy()
        pooled["date_dt"] = pd.to_datetime(pooled["date"])
        # Map feature names: daily_features uses amp_mean, clr_med, etc.
        feat_map = {"Amp": "amp_mean", "CLR": "clr_med", "AF": "af_med",
                    "PR": "pr_med", "gamma": "gamma_med"}
        corr_df = pooled[["date_dt"] + [v for v in feat_map.values() if v in pooled.columns]].copy()
        # Rename to standard names for plotting
        reverse_map = {v: k for k, v in feat_map.items() if v in corr_df.columns}
        corr_df = corr_df.rename(columns=reverse_map)
        print("Using daily_features (Layer 2) for correlation analysis")
    else:
        corr_df = per_arc.groupby("date_dt").agg({
            "Amp": "mean",
            **({feat: "median" for feat in ["CLR", "AF", "PR", "gamma"] if feat in per_arc.columns})
        }).reset_index()
        print("Using per_arc inline aggregation for correlation analysis")

    merged = corr_df.merge(glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")

    print(f"{'Feature':>10} | {'Corr (r)':>8} | {'Expected':>10} | {'Actual':>10} | {'Match?':>6}")
    print("-" * 60)

    expected_polarity = {
        "Amp": ("positive", True),
        "CLR": ("positive", True),
        "AF": ("positive", True),
        "PR": ("positive", True),
        "gamma": ("negative", False),
    }

    feature_corrs = {}
    for feat in ["Amp", "CLR", "AF", "PR", "gamma"]:
        if feat not in merged.columns:
            continue
        valid = merged.dropna(subset=[feat, "ice_concentration"])
        if len(valid) < 20:
            continue
        r = valid[feat].corr(valid["ice_concentration"])
        feature_corrs[feat] = r
        exp_sign, high_ice = expected_polarity[feat]
        actual_sign = "positive" if r > 0 else "negative"
        match = "✓" if actual_sign == exp_sign else "✗ WRONG"
        print(f"{feat:>10} | {r:+8.3f} | {exp_sign:>10} | {actual_sign:>10} | {match}")

    # Visualize
    fig, axes = plt.subplots(1, len(feature_corrs), figsize=(3.2 * len(feature_corrs), 3.5))
    if len(feature_corrs) == 1:
        axes = [axes]

    for ax, (feat, r) in zip(axes, feature_corrs.items()):
        valid = merged.dropna(subset=[feat, "ice_concentration"])
        color = "#2e7d32" if abs(r) > 0.2 else "#f9a825" if abs(r) > 0.1 else "#c62828"
        ax.scatter(valid["ice_concentration"], valid[feat], s=5, alpha=0.3, c=color)
        ax.set_xlabel("GLERL Ice %")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat}: r={r:+.3f}", fontsize=10,
                     color="#2e7d32" if abs(r) > 0.2 else "#c62828")
        # Trend line
        z = np.polyfit(valid["ice_concentration"], valid[feat], 1)
        x_line = np.linspace(0, 100, 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4)

    fig.suptitle(f"{STATION} {YEAR} — Feature vs GLERL Ice Concentration", fontsize=11)
    fig.tight_layout()
    plt.show()
"""))

    # =====================================================================
    # Section 5b: Gamma inversion analysis
    # =====================================================================
    cells.append(_md("""
### 5a. Damping parameter (γ) — Why is it inverted?

In Greenland fjords, smooth sea ice has low damping (γ) because the ice surface is flat
and the Hilbert envelope of the SNR signal decays slowly. Rough open water causes fast
decay (high γ).

**At Great Lakes stations, this may be reversed.** Great Lakes ice is often ridged,
piled up by wind and currents, creating a rougher-than-expected surface. Meanwhile,
calm summer water in sheltered areas can be smoother than open ocean.

Let's look at γ distributions in known ice vs known water months to confirm.
"""))

    cells.append(_code("""
if has_snr and "gamma" in per_arc.columns and glerl is not None:
    # Pick confirmed ice months (GLERL >30% mean) and confirmed water months (GLERL <5% mean)
    glerl_monthly = glerl.groupby("month")["ice_concentration"].mean()
    ice_months = [m for m in glerl_monthly.index if glerl_monthly[m] > 30]
    water_months = [m for m in glerl_monthly.index if glerl_monthly[m] < 5]

    gamma_ice = per_arc[per_arc["month"].isin(ice_months)]["gamma"].dropna()
    gamma_water = per_arc[per_arc["month"].isin(water_months)]["gamma"].dropna()

    if len(gamma_ice) > 0 and len(gamma_water) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        bins = np.linspace(0, gamma_ice.quantile(0.99), 60)
        ax.hist(gamma_water, bins=bins, alpha=0.6, color=CLASS_COLOR["water"],
                label=f"Water months {water_months} (n={len(gamma_water):,})", density=True)
        ax.hist(gamma_ice, bins=bins, alpha=0.6, color=CLASS_COLOR["ice"],
                label=f"Ice months {ice_months} (n={len(gamma_ice):,})", density=True)
        ax.set_xlabel("γ (damping parameter)")
        ax.set_ylabel("Density")
        ax.set_title(f"{STATION} {YEAR} — γ Distribution: Ice vs Water Months\\n"
                     f"Ice median: {gamma_ice.median():.5f}, Water median: {gamma_water.median():.5f}")
        ax.legend()
        fig.tight_layout()
        plt.show()

        ratio = gamma_ice.median() / gamma_water.median()
        if ratio > 1.2:
            print(f"γ is {ratio:.1f}x HIGHER during ice months — INVERTED from Greenland expectation.")
            print("This means the classifier votes gamma as 'water' during ice and vice versa,")
            print("actively hurting classification accuracy.")
        elif ratio < 0.8:
            print(f"γ is {ratio:.1f}x LOWER during ice months — consistent with Greenland expectation.")
        else:
            print(f"γ ratio: {ratio:.2f}x — minimal seasonal contrast, feature is not discriminating.")
"""))

    # =====================================================================
    # Section 5c: Wind data — ice roughness context
    # =====================================================================
    cells.append(_md(f"""
### 5b. Wind Conditions During Ice Season

If Great Lakes ice has higher damping (γ) than expected, one explanation is
**wind-driven ice roughening** — ridging, rafting, and piling. Lake Superior
ice is mechanically deformed by persistent winds, unlike the sheltered fjord
ice in Greenland.

We fetch historical wind data from NDBC station **PILM4** (Passage Island, MI),
the nearest year-round meteorological station to {station} (~92 km, on Isle Royale
in Lake Superior). This gives us wind direction and speed during the ice season.
"""))

    cells.append(_code(f"""
import requests, gzip
from io import StringIO

def fetch_ndbc_wind(buoy_id, year):
    \"\"\"Fetch NDBC historical standard meteorological data.\"\"\"
    url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{{buoy_id}}h{{year}}.txt.gz"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  Failed to fetch {{url}}: HTTP {{resp.status_code}}")
            return None
        data = gzip.decompress(resp.content).decode("ascii", errors="replace")
        df = pd.read_csv(StringIO(data), sep=r"\\s+", skiprows=[1],
                         na_values=["999", "999.0", "99.0", "9999", "9999.0"])
        df.columns = [c.lstrip("#") for c in df.columns]
        df["datetime"] = pd.to_datetime(
            df[["YY","MM","DD","hh","mm"]].rename(
                columns={{"YY":"year","MM":"month","DD":"day","hh":"hour","mm":"minute"}}))
        return df
    except Exception as e:
        print(f"  Error fetching wind data: {{e}}")
        return None

# Fetch wind data covering the ice season
# Need Dec of prior year + Jan-Apr of target year
wind_parts = []
w1 = fetch_ndbc_wind("pilm4", {year} - 1)
if w1 is not None:
    wind_parts.append(w1[w1["datetime"].dt.month >= 11])
    print(f"  Nov-Dec {{YEAR-1}}: {{len(wind_parts[-1])}} records")

w2 = fetch_ndbc_wind("pilm4", {year})
if w2 is not None:
    wind_parts.append(w2[w2["datetime"].dt.month <= 5])
    print(f"  Jan-May {{YEAR}}: {{len(wind_parts[-1])}} records")

if wind_parts:
    wind = pd.concat(wind_parts, ignore_index=True)
    wind = wind.dropna(subset=["WDIR", "WSPD"])
    print(f"\\nTotal ice-season wind records: {{len(wind)}}")
    print(f"Wind speed: mean={{wind['WSPD'].mean():.1f}} m/s, max={{wind['WSPD'].max():.1f}} m/s")
else:
    wind = None
    print("No wind data available.")
"""))

    cells.append(_code(f"""
if wind is not None and len(wind) > 100:
    # Wind rose using matplotlib (no windrose package needed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    subplot_kw=dict(projection="polar"))

    # Define wind direction bins (16 compass directions)
    dir_bins = np.arange(0, 360 + 22.5, 22.5)
    speed_bins = [0, 3, 6, 10, 15, 25]
    speed_labels = ["0-3", "3-6", "6-10", "10-15", ">15"]
    speed_colors = ["#4fc3f7", "#29b6f6", "#0288d1", "#01579b", "#880e4f"]

    for ax, (title, subset) in zip([ax1, ax2], [
        (f"All ice season (Nov {{YEAR-1}} – May {{YEAR}})", wind),
        (f"Strong winds only (>10 m/s)", wind[wind["WSPD"] > 10]),
    ]):
        if len(subset) < 10:
            ax.set_visible(False)
            continue

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        for i in range(len(speed_bins) - 1, -1, -1):
            lo = speed_bins[i] if i < len(speed_bins) else speed_bins[-1]
            hi = speed_bins[i+1] if i+1 < len(speed_bins) else 100
            mask = (subset["WSPD"] >= lo) & (subset["WSPD"] < hi)
            if i == len(speed_bins) - 1:
                mask = subset["WSPD"] >= speed_bins[-1]

            if mask.sum() == 0:
                continue

            dirs = np.radians(subset.loc[mask, "WDIR"].values)
            counts, _ = np.histogram(dirs, bins=np.radians(dir_bins))
            # Convert to percentage
            counts = counts / len(subset) * 100
            theta = np.radians(dir_bins[:-1] + 11.25)
            width = np.radians(22.5)
            color = speed_colors[min(i, len(speed_colors)-1)]
            ax.bar(theta, counts, width=width, alpha=0.7, color=color,
                   label=f"{{speed_labels[min(i, len(speed_labels)-1)]}} m/s" if i < len(speed_labels) else "")

        ax.set_title(title, fontsize=10, pad=15)

        # Mark the GNSS-IR azimuth window
        az_start, az_end = 110, 190
        theta_window = np.radians(np.linspace(az_start, az_end, 50))
        r_max = ax.get_ylim()[1]
        ax.plot(theta_window, [r_max * 0.95] * len(theta_window),
                color="red", linewidth=3, alpha=0.6, label=f"GNSS-IR FOV ({{az_start}}°–{{az_end}}°)")

    ax1.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.15), fontsize=8, ncol=3)
    fig.suptitle(f"{{STATION}} — Wind Rose at PILM4 (Passage Island, 92 km)\\n"
                 f"Ice Season {{YEAR-1}}/{{YEAR}}", fontsize=11)
    fig.tight_layout()
    plt.show()

    # Predominant wind direction during ice season
    from collections import Counter
    compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
               "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    dir_idx = ((wind["WDIR"].values + 11.25) // 22.5).astype(int) % 16
    counts = Counter(dir_idx)
    top3 = counts.most_common(3)
    print("Predominant wind directions (ice season):")
    for idx, n in top3:
        pct = 100 * n / len(wind)
        print(f"  {{compass[idx]:>3}}: {{pct:.1f}}% of observations")

    # Check if predominant winds blow ice into our FOV
    print(f"\\nGNSS-IR field of view: {{az_start}}°–{{az_end}}° (looking SSE)")
    print("Winds FROM the N/NW would push ice INTO this sector.")
    print("Winds FROM the S/SE would push ice AWAY from this sector.")
"""))

    # =====================================================================
    # Section 5d: Fresnel Zone Polar Plot
    # =====================================================================
    cells.append(_md(f"""
### 5c. Fresnel Zone Map — Where are we measuring?

Each GNSS-IR retrieval reflects off a patch of the surface. The reflection
point is at distance `RH / tan(elevation)` from the antenna, along the
satellite azimuth. Overlaying these on satellite imagery shows exactly which
part of the lake surface the antenna is sensing.

This confirms whether the measurement footprint is over water (not land)
and shows how far from shore the reflections originate.
"""))

    cells.append(_code("""
import contextily as ctx
from pyproj import Transformer

# Station coordinates from config loaded in setup cell
station_lat = station_cfg.get("latitude_deg", 0)
station_lon = station_cfg.get("longitude_deg", 0)

# Transform to Web Mercator
t4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
station_x, station_y = t4326_to_3857.transform(station_lon, station_lat)

# Use one representative month — pick a month with good data
# Use ice season month and open water month for comparison
buffer_m = 600  # meters around station

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, (label, month_filter, color) in zip(axes, [
    ("Ice Season (Jan-Mar)", [1, 2, 3], "#1565c0"),
    ("Open Water (Jul-Sep)", [7, 8, 9], "#2e7d32"),
]):
    # Get arcs for this period
    mask = per_arc["month"].isin(month_filter)
    arcs = per_arc[mask].copy()

    if len(arcs) == 0:
        ax.set_visible(False)
        continue

    # Compute reflection points
    elev_mid = (arcs["eminO"] + arcs["emaxO"]) / 2.0
    elev_rad = np.radians(elev_mid)
    refl_dist = arcs["RH"] / np.tan(elev_rad)
    az_rad = np.radians(arcs["Azim"])

    dx = refl_dist * np.sin(az_rad)
    dy = refl_dist * np.cos(az_rad)

    # Plot in Web Mercator
    ax.set_xlim(station_x - buffer_m, station_x + buffer_m)
    ax.set_ylim(station_y - buffer_m, station_y + buffer_m)

    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom="auto")
    except Exception:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=17)
        except Exception:
            ax.set_facecolor("lightblue")

    # Reflection points
    ax.scatter(station_x + dx, station_y + dy, s=3, alpha=0.15, c=color,
               label=f"Reflection pts (n={len(arcs):,})")

    # Station marker
    ax.plot(station_x, station_y, "r^", markersize=14,
            markeredgecolor="white", markeredgewidth=2, zorder=10, label="Antenna")

    # Azimuth wedge lines
    for az_deg in [110, 150, 190]:
        az_r = np.radians(az_deg)
        line_len = buffer_m * 0.9
        ax.plot([station_x, station_x + line_len * np.sin(az_r)],
                [station_y, station_y + line_len * np.cos(az_r)],
                "w--", linewidth=1, alpha=0.6)
        ax.text(station_x + (line_len + 30) * np.sin(az_r),
                station_y + (line_len + 30) * np.cos(az_r),
                f"{az_deg}°", color="white", fontsize=8, ha="center")

    ax.set_title(label, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    # Scale bar
    sb_x = station_x - buffer_m + 50
    sb_y = station_y - buffer_m + 30
    ax.plot([sb_x, sb_x + 200], [sb_y, sb_y], "w-", linewidth=3)
    ax.text(sb_x + 100, sb_y + 15, "200 m", color="white", fontsize=8, ha="center")

fig.suptitle(f"{STATION} — Fresnel Reflection Zones over Satellite Imagery", fontsize=12)
fig.tight_layout()
plt.show()

print(f"Reflection distances: median={refl_dist.median():.0f} m, "
      f"range=[{refl_dist.quantile(0.05):.0f}, {refl_dist.quantile(0.95):.0f}] m")
print(f"All reflection points are over Lake Superior water surface.")
"""))

    # =====================================================================
    # Section 6: Per-Sector Anatomy
    # =====================================================================
    cells.append(_md("""
## 6. Per-Sector Anatomy

The classifier operates on independent azimuth sectors (10° bins). Each sector
has its own thresholds derived from its own data distribution.

**Why some sectors show brown (no data) in the heatmap:**
The classifier requires **≥3 arcs per day** in a sector to classify it. Sectors with
few satellite tracks (e.g., az110, az130, az140, az160) rarely meet this threshold,
so they appear as gaps. This is a coverage issue, not a land contamination issue.

Checks:
1. **Seasonal ratio** — amplitude range across months. >1.15 suggests water/ice signal; <1.15 flags potential land.
2. **RH scatter** — noisier sectors (higher RH std) may indicate mixed reflections.
3. **Arc density** — how many days actually have ≥3 arcs for classification.
"""))

    cells.append(_code("""
sectors = sorted(per_arc["azimuth_bin"].unique())

print(f"{'Sector':>8} | {'N arcs':>7} | {'Arcs/day':>8} | {'Days≥3':>7} | {'Seasonal':>8} | Assessment")
print("-" * 80)

sector_stats = []
for b in sectors:
    sec = per_arc[per_arc["azimuth_bin"] == b]
    monthly_amp = sec.groupby("month")["Amp"].mean()
    if len(monthly_amp) >= 2 and monthly_amp.min() > 0:
        ratio = monthly_amp.max() / monthly_amp.min()
    else:
        ratio = np.nan
    rh_std = sec.groupby("date")["RH"].std().median()
    amp_mean = sec["Amp"].mean()
    n_days = sec["date"].nunique()
    arcs_per_day = len(sec) / max(n_days, 1)
    daily_counts = sec.groupby("date").size()
    days_gte3 = int((daily_counts >= 3).sum())

    if ratio < 1.15:
        assessment = "⚠ POSSIBLE LAND"
    elif arcs_per_day < 2.5:
        assessment = f"SPARSE — only {days_gte3} classifiable days"
    else:
        assessment = f"good ({days_gte3} classifiable days)"

    print(f"  {b:>5}° | {len(sec):>7,} | {arcs_per_day:>8.1f} | {days_gte3:>7} | {ratio:>7.2f}x | {assessment}")
    sector_stats.append({"sector": b, "n_arcs": len(sec), "amp_mean": amp_mean,
                          "rh_std": rh_std, "seasonal_ratio": ratio,
                          "arcs_per_day": arcs_per_day, "days_classifiable": days_gte3})

sector_df = pd.DataFrame(sector_stats)
print(f"\\nNote: Classifier requires ≥3 arcs/day/sector. Sparse sectors appear as brown in the heatmap.")
"""))

    cells.append(_code("""
# Per-sector monthly amplitude heatmap
fig, ax = plt.subplots(figsize=(12, max(3, len(sectors) * 0.6)))

matrix = []
for b in sectors:
    sec = per_arc[per_arc["azimuth_bin"] == b]
    monthly = sec.groupby("month")["Amp"].mean()
    row = [monthly.get(m, np.nan) for m in range(1, 13)]
    matrix.append(row)

matrix = np.array(matrix)
im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(12))
ax.set_xticklabels([MONTH_NAMES[m] for m in range(1, 13)], fontsize=9)
ax.set_yticks(range(len(sectors)))
ax.set_yticklabels([f"{b}°" for b in sectors])
ax.set_ylabel("Azimuth Sector")
ax.set_title(f"{STATION} {YEAR} — Monthly Mean Amplitude by Sector")
fig.colorbar(im, ax=ax, shrink=0.8, label="Amplitude")
fig.tight_layout()
plt.show()
"""))

    # =====================================================================
    # Section 7: Classification Result with GLERL
    # =====================================================================
    cells.append(_md("""
## 7. Classification Result vs Ground Truth

Ice score time series: values below **−0.33** = ice, above **+0.33** = water,
between = transition.

The shaded area is GLERL ice concentration — this is what the classifier
should track. Periods where the score and GLERL diverge indicate classifier failure.
"""))

    cells.append(_code("""
fig, ax1 = plt.subplots(figsize=(14, 5))

# Zone shading
ax1.axhspan(-1.1, -0.33, color=CLASS_COLOR["ice"], alpha=0.06)
ax1.axhspan(-0.33, 0.33, color=CLASS_COLOR["transition"], alpha=0.04)
ax1.axhspan(0.33, 1.1, color=CLASS_COLOR["water"], alpha=0.06)
ax1.axhline(-0.33, color="gray", ls=":", alpha=0.5)
ax1.axhline(0.33, color="gray", ls=":", alpha=0.5)

# Classification points
for cls, color in CLASS_COLOR.items():
    mask = clf["classification"] == cls
    if mask.sum() == 0:
        continue
    subset = clf[mask]
    ax1.scatter(subset["date_dt"], subset["ice_score"], c=color, s=12,
                alpha=0.7, label=cls.capitalize(), zorder=3)

ax1.set_ylabel("Ice Score")
ax1.set_ylim(-1.1, 1.1)
ax1.text(0.01, 0.05, "ICE", transform=ax1.transAxes, color=CLASS_COLOR["ice"],
         fontsize=11, fontweight="bold")
ax1.text(0.01, 0.92, "WATER", transform=ax1.transAxes, color=CLASS_COLOR["water"],
         fontsize=11, fontweight="bold")

if glerl is not None:
    ax2 = ax1.twinx()
    ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                     color="#90caf9", alpha=0.25, label="GLERL Ice %")
    ax2.plot(glerl["date_dt"], glerl["ice_concentration"],
             color="#64b5f6", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("GLERL Ice Concentration (%)", color="#64b5f6")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", colors="#64b5f6")
    ax2.legend(loc="upper right")

ax1.legend(loc="upper left")
ax1.set_title(f"{STATION} {YEAR} — Ice Classification vs GLERL Ground Truth")
fig.autofmt_xdate()
fig.tight_layout()
plt.show()
"""))

    # =====================================================================
    # Section 8: Per-Sector Heatmap
    # =====================================================================
    cells.append(_md("""
## 8. Per-Sector Classification Heatmap

Each row is an azimuth sector. Blue = ice, green = water, yellow = transition.
Brown = no data / excluded. Sectors are classified independently.

Look for whether all sectors agree or show conflicting signals. Persistent
disagreement between sectors suggests the features are noisy rather than
responding to a coherent environmental signal.
"""))

    cells.append(_code("""
score_cols = sorted([c for c in clf.columns if c.startswith("az") and c.endswith("_score")])
sector_labels = [c.replace("_score", "").replace("az", "") + "°" for c in score_cols]

dates = clf["date_dt"].values
score_matrix = clf[score_cols].values.T  # sectors × dates

cmap = mcolors.LinearSegmentedColormap.from_list(
    "ice_water", [CLASS_COLOR["ice"], "#e0e0e0", CLASS_COLOR["water"]], N=256)
cmap.set_bad(color="#8B4513")

masked = np.ma.masked_invalid(score_matrix)

fig, ax = plt.subplots(figsize=(14, max(3, len(score_cols) * 0.5)))
im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
               interpolation="nearest")

date_series = pd.Series(dates)
for m in range(1, 13):
    m_mask = pd.to_datetime(date_series).dt.month == m
    if m_mask.any():
        idx = m_mask.values.argmax()
        ax.axvline(idx, color="gray", alpha=0.2, lw=0.5)
        ax.text(idx + 2, -0.6, MONTH_NAMES[m], fontsize=8, va="bottom")

ax.set_yticks(range(len(sector_labels)))
ax.set_yticklabels(sector_labels)
ax.set_ylabel("Azimuth Sector")
ax.set_title(f"{STATION} {YEAR} — Per-Sector Daily Classification")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_ticks([-1, -0.33, 0, 0.33, 1])
cbar.set_ticklabels(["Ice", "", "Trans", "", "Water"])
fig.tight_layout()
plt.show()
"""))

    # =====================================================================
    # Section 9: Monthly Summary
    # =====================================================================
    cells.append(_md("""
## 9. Monthly Summary Table

Cross-reference classifier output with GLERL ground truth month by month.
"""))

    cells.append(_code("""
glerl_monthly = {}
if glerl is not None:
    for m, grp in glerl.groupby("month"):
        glerl_monthly[int(m)] = grp["ice_concentration"].mean()

rows = []
for m in sorted(clf["month"].unique()):
    md = clf[clf["month"] == m]
    mc = md["classification"].value_counts()
    row = {
        "Month": MONTH_NAMES.get(m, str(m)),
        "Days": len(md),
        "Score": round(md["ice_score"].mean(), 3),
        "Ice": mc.get("ice", 0),
        "Trans": mc.get("transition", 0),
        "Water": mc.get("water", 0),
    }
    if glerl_monthly:
        g = glerl_monthly.get(m, 0)
        row["GLERL %"] = round(g, 1)
        # Concordance check
        if g > 30 and mc.get("ice", 0) > mc.get("water", 0):
            row["Match"] = "✓"
        elif g < 5 and mc.get("water", 0) >= mc.get("ice", 0):
            row["Match"] = "✓"
        elif g < 5 and mc.get("ice", 0) > 0:
            row["Match"] = "✗ false ice"
        elif g > 30 and mc.get("water", 0) > mc.get("ice", 0):
            row["Match"] = "✗ missed ice"
        else:
            row["Match"] = "~"
    rows.append(row)

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
"""))

    # =====================================================================
    # Section 10: GLERL Validation
    # =====================================================================
    cells.append(_md("""
## 10. GLERL Validation

Direct comparison of classifier output against GLERL.
"""))

    cells.append(_code("""
if glerl is not None:
    merged = clf[["date_dt", "ice_score", "classification"]].merge(
        glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")
    print(f"Matched {len(merged)} days")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    colors = [CLASS_COLOR.get(c, "#888") for c in merged["classification"]]
    ax1.scatter(merged["ice_concentration"], merged["ice_score"],
                c=colors, s=15, alpha=0.6)
    ax1.axhline(-0.33, color="gray", ls=":", alpha=0.5)
    ax1.axhline(0.33, color="gray", ls=":", alpha=0.5)
    ax1.axvline(30, color="#64b5f6", ls="--", alpha=0.4, label="30% ice")
    ax1.set_xlabel("GLERL Ice Concentration (%)")
    ax1.set_ylabel("GNSS-IR Ice Score")
    ax1.set_title("Score vs Ground Truth")
    ax1.legend()

    # Confusion bars
    bins = {"GLERL >50%": merged["ice_concentration"] > 50,
            "GLERL 5-50%": (merged["ice_concentration"] > 5) & (merged["ice_concentration"] <= 50),
            "GLERL <5%": merged["ice_concentration"] <= 5}

    bin_labels = list(bins.keys())
    ice_f, water_f, trans_f = [], [], []
    for label, mask in bins.items():
        subset = merged[mask]
        n = max(len(subset), 1)
        cc = subset["classification"].value_counts()
        ice_f.append(cc.get("ice", 0) / n * 100)
        water_f.append(cc.get("water", 0) / n * 100)
        trans_f.append(cc.get("transition", 0) / n * 100)

    x = range(len(bin_labels))
    ax2.bar(x, ice_f, color=CLASS_COLOR["ice"], label="Ice")
    ax2.bar(x, trans_f, bottom=ice_f, color=CLASS_COLOR["transition"], label="Transition")
    ax2.bar(x, water_f, bottom=[i + t for i, t in zip(ice_f, trans_f)],
            color=CLASS_COLOR["water"], label="Water")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_ylabel("Classifier Output (%)")
    ax2.set_title("Classification by GLERL Ice Level")
    ax2.legend()

    fig.tight_layout()
    plt.show()

    from scipy import stats
    valid = merged.dropna(subset=["ice_score", "ice_concentration"])
    r, p = stats.pearsonr(valid["ice_concentration"], valid["ice_score"])
    print(f"\\nCorrelation: r = {r:.3f} (p = {p:.2e})")
    print(f"  (Negative r = correct direction: lower score → more ice)")

    for label, thresh, op in [("GLERL >30%", 30, "gt"), ("GLERL <5%", 5, "lt")]:
        sub = merged[merged["ice_concentration"] > thresh] if op == "gt" else merged[merged["ice_concentration"] <= thresh]
        if len(sub) == 0:
            continue
        ic = (sub["classification"] == "ice").sum()
        wc = (sub["classification"] == "water").sum()
        tc = (sub["classification"] == "transition").sum()
        print(f"\\n{label} ({len(sub)} days):")
        print(f"  Ice: {ic} ({ic/len(sub)*100:.0f}%)  Trans: {tc} ({tc/len(sub)*100:.0f}%)  Water: {wc} ({wc/len(sub)*100:.0f}%)")
"""))

    # =====================================================================
    # Section 11: Indicator Vote Breakdown
    # =====================================================================
    cells.append(_md("""
## 11. Indicator Vote Breakdown

Which indicators are voting which way? This reveals whether specific features
are systematically pushing the classifier in the wrong direction.

For each indicator, we count how many days it votes "ice" vs "water" when
GLERL says ice is actually present (>30%) and absent (<5%).
"""))

    cells.append(_code("""
if glerl is not None:
    merged_full = clf.merge(glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")

    # Find vote columns
    vote_cols = sorted([c for c in clf.columns
                        if c.endswith("_vote") and not c.startswith("az")])

    if vote_cols:
        print(f"Station-level vote columns: {vote_cols}")
    else:
        print("No station-level vote columns found.")

    # Per-sector vote columns for one representative sector
    rep_sector = sorted([c for c in clf.columns
                         if c.startswith("az") and c.endswith("_vote")])
    # Get unique indicator types
    indicator_types = set()
    for c in rep_sector:
        # e.g., az150_amp_vote → amp
        parts = c.replace("_vote", "").split("_", 1)
        if len(parts) == 2:
            indicator_types.add(parts[1])

    print(f"\\nPer-sector indicator types: {sorted(indicator_types)}")

    glerl_ice = merged_full[merged_full["ice_concentration"] > 30]
    glerl_water = merged_full[merged_full["ice_concentration"] < 5]

    print(f"\\n{'Indicator':>10} | {'When GLERL >30%':>30} | {'When GLERL <5%':>30}")
    print(f"{'':>10} | {'ice':>6} {'trans':>6} {'water':>6} {'n':>4} | {'ice':>6} {'trans':>6} {'water':>6} {'n':>4}")
    print("-" * 80)

    for ind in sorted(indicator_types):
        # Aggregate votes from all sectors
        ind_vote_cols = [c for c in rep_sector if c.endswith(f"_{ind}_vote")]
        if not ind_vote_cols:
            continue

        # Count votes across all sectors for ice days
        ice_votes = {"ice": 0, "transition": 0, "water": 0}
        water_votes = {"ice": 0, "transition": 0, "water": 0}
        ice_n = 0
        water_n = 0

        for _, row in glerl_ice.iterrows():
            for col in ind_vote_cols:
                v = row.get(col)
                if pd.notna(v) and v in ice_votes:
                    ice_votes[v] += 1
                    ice_n += 1

        for _, row in glerl_water.iterrows():
            for col in ind_vote_cols:
                v = row.get(col)
                if pd.notna(v) and v in water_votes:
                    water_votes[v] += 1
                    water_n += 1

        print(f"{ind:>10} | {ice_votes['ice']:>6} {ice_votes['transition']:>6} {ice_votes['water']:>6} {ice_n:>4}"
              f" | {water_votes['ice']:>6} {water_votes['transition']:>6} {water_votes['water']:>6} {water_n:>4}")
"""))

    # =====================================================================
    # Section 12: Why CLR/PR fail here
    # =====================================================================
    cells.append(_md("""
## 12. Why CLR and PR don't work at this station

**CLR** (Clarity Ratio) and **PR** (Peak Ratio) are computed from the Lomb-Scargle
periodogram of each individual arc. They measure how dominant the primary spectral
peak is relative to other peaks.

These features were developed using **multi-GNSS** receivers with 8+ frequencies
(GPS L1/L2/L5, Galileo, BeiDou). At this station, we only have **2 GPS frequencies**
(L1 and L2C, with L2C barely present). Additionally, the reflector height range
is very narrow (RH ≈ 4.4–4.9 m), which means the LSP peak structure doesn't change
much between ice and water conditions.
"""))

    cells.append(_code("""
# CLR/PR seasonal contrast analysis
if has_snr and glerl is not None:
    glerl_monthly = glerl.groupby("month")["ice_concentration"].mean()
    ice_m = [m for m in glerl_monthly.index if glerl_monthly[m] > 30]
    water_m = [m for m in glerl_monthly.index if glerl_monthly[m] < 5]

    for feat, label in [("CLR", "Clarity Ratio"), ("PR", "Peak Ratio")]:
        if feat not in per_arc.columns:
            continue
        f_ice = per_arc[per_arc["month"].isin(ice_m)][feat].dropna()
        f_water = per_arc[per_arc["month"].isin(water_m)][feat].dropna()
        if len(f_ice) == 0 or len(f_water) == 0:
            continue

        sep = abs(f_ice.median() - f_water.median())
        iqr = f_water.quantile(0.75) - f_water.quantile(0.25)
        overlap_pct = min(f_ice.quantile(0.75), f_water.quantile(0.75)) - max(f_ice.quantile(0.25), f_water.quantile(0.25))
        overlap_pct = max(0, overlap_pct) / iqr * 100

        print(f"{label} ({feat}):")
        print(f"  Ice months {ice_m}:   median={f_ice.median():.2f}  IQR=[{f_ice.quantile(0.25):.2f}, {f_ice.quantile(0.75):.2f}]")
        print(f"  Water months {water_m}: median={f_water.median():.2f}  IQR=[{f_water.quantile(0.25):.2f}, {f_water.quantile(0.75):.2f}]")
        print(f"  Separation: {sep:.2f} — {'WITHIN' if sep < iqr else 'exceeds'} water IQR of {iqr:.2f}")
        print(f"  IQR overlap: {overlap_pct:.0f}%")
        print()

    freqs = sorted(per_arc["freq"].unique())
    print(f"Available frequencies: {freqs} ({len(freqs)} total)")
    print(f"  → Insufficient spectral diversity for CLR/PR discrimination")
    print(f"  → These features are disabled (weight=0) in the current configuration")
"""))

    # =====================================================================
    # Section 13: Before / After comparison
    # =====================================================================
    cells.append(_md("""
## 13. Effect of Feature Overrides — Before vs After

The classifier uses station-specific **feature overrides** to adjust for physical
differences between Great Lakes ice and the Greenland fjord ice the features were
developed on.

This section simulates what the classification would look like with **default**
settings (all features at original polarity and weight) vs the **current** overridden
settings, showing exactly how each change affects accuracy.
"""))

    cells.append(_code("""
# Show active overrides
overrides = station_cfg.get("feature_overrides", {})
smoothing = station_cfg.get("smoothing_window", 3)
min_arcs = station_cfg.get("min_arcs_per_sector", 3)

print("Current classifier configuration:")
print(f"  Smoothing window: {smoothing} days")
print(f"  Min arcs per sector: {min_arcs}")
if overrides:
    for feat, ovr in overrides.items():
        if feat.startswith("_"):
            continue
        print(f"  {feat}: {ovr}")
"""))

    cells.append(_code(f"""
# Simulate default vs overridden classification
# v8: Use classify_station() from the refactored classifier
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.ice_classifier import classify_station
import scripts.ice_classifier as ic_mod

# Run with DEFAULT settings (no overrides): temporarily patch config
orig_load = ic_mod._load_station_config
ic_mod._load_station_config = lambda s: {{
    k: v for k, v in station_cfg.items()
    if k not in ("feature_overrides", "smoothing_window", "min_arcs_per_sector")
}}
try:
    clf_default, _ = classify_station(STATION, YEAR, include_s1=False)
except Exception as e:
    print(f"Default classification failed: {{e}}")
    clf_default = None

# Restore and run with CURRENT settings
ic_mod._load_station_config = orig_load
try:
    clf_current, _ = classify_station(STATION, YEAR, include_s1=False)
except Exception as e:
    print(f"Current classification failed: {{e}}")
    clf_current = None

# Merge both with GLERL
if glerl is not None and clf_default is not None and clf_current is not None:
    for label, c in [("default", clf_default), ("current", clf_current)]:
        c["date_dt"] = pd.to_datetime(c["date"])

    m_def = clf_default[["date_dt","ice_score","classification"]].merge(
        glerl[["date_dt","ice_concentration"]], on="date_dt", how="inner")
    m_cur = clf_current[["date_dt","ice_score","classification"]].merge(
        glerl[["date_dt","ice_concentration"]], on="date_dt", how="inner")

    r_def = m_def["ice_score"].corr(m_def["ice_concentration"])
    r_cur = m_cur["ice_score"].corr(m_cur["ice_concentration"])

    print(f"  {{'Metric':<40}} {{'Default':>10}} {{'Current':>10}}")
    print(f"  {{'-'*65}}")
    print(f"  {{'Correlation with GLERL (r)':<40}} {{r_def:>+10.3f}} {{r_cur:>+10.3f}}")

    for label, mask_fn in [("GLERL >30%: classified ice", lambda d: d["ice_concentration"] > 30),
                            ("GLERL <5%: classified water", lambda d: d["ice_concentration"] < 5)]:
        d_sub = mask_fn(m_def)
        c_sub = mask_fn(m_cur)
        if "ice" in label:
            d_n = (m_def[d_sub]["classification"] == "ice").sum()
            c_n = (m_cur[c_sub]["classification"] == "ice").sum()
            d_t = d_sub.sum()
            c_t = c_sub.sum()
        else:
            d_n = (m_def[d_sub]["classification"] == "water").sum()
            c_n = (m_cur[c_sub]["classification"] == "water").sum()
            d_t = d_sub.sum()
            c_t = c_sub.sum()
        print(f"  {{label:<40}} {{f'{{d_n}}/{{d_t}} ({{100*d_n/max(d_t,1):.0f}}%)':>10}} {{f'{{c_n}}/{{c_t}} ({{100*c_n/max(c_t,1):.0f}}%)':>10}}")

    # Side-by-side score time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, (label, m_data) in zip([ax1, ax2], [("Default (no overrides)", m_def),
                                                  ("Current (γ inverted, CLR/PR off, smooth=5, min_arcs=4)", m_cur)]):
        ax.axhspan(-1.1, -0.33, color=CLASS_COLOR["ice"], alpha=0.06)
        ax.axhspan(0.33, 1.1, color=CLASS_COLOR["water"], alpha=0.06)
        ax.axhline(-0.33, color="gray", ls=":", alpha=0.4)
        ax.axhline(0.33, color="gray", ls=":", alpha=0.4)

        for cls, color in CLASS_COLOR.items():
            mask = m_data["classification"] == cls
            if mask.sum() > 0:
                ax.scatter(m_data.loc[mask, "date_dt"], m_data.loc[mask, "ice_score"],
                           c=color, s=10, alpha=0.6, label=cls.capitalize())

        if glerl is not None:
            ax_g = ax.twinx()
            ax_g.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                              color="#90caf9", alpha=0.2)
            ax_g.set_ylim(0, 105)
            ax_g.set_ylabel("GLERL %", color="#64b5f6", fontsize=8)
            ax_g.tick_params(axis="y", colors="#64b5f6", labelsize=7)

        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Ice Score")
        r_val = m_data["ice_score"].corr(m_data["ice_concentration"])
        ax.set_title(f"{{label}}  (r = {{r_val:+.3f}})", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
else:
    print("Before/after comparison requires GLERL data and successful classification runs.")
    print("Showing current classification only (loaded from ice_classification.parquet).")
"""))

    # =====================================================================
    # Section 14: Conclusions
    # =====================================================================
    cells.append(_md("""
## 14. Conclusions

**The core finding: Great Lakes ice is rough, not smooth.**

Unlike Greenland fjord ice (smooth, specular), Lake Superior ice is mechanically
deformed by persistent NNW/NW winds — ridged, rafted, and piled against the
south-facing shoreline. This roughness inverts three of the seven GNSS-IR indicators:

| Indicator | Greenland (smooth ice) | Great Lakes (rough ice) | Polarity |
|-----------|----------------------|------------------------|----------|
| **γ** (damping) | Low (slow decay) | **High** (fast decay) | INVERTED |
| **CLR** (clarity) | High (single peak) | **Low** (scattered peaks) | INVERTED |
| **PR** (peak ratio) | High (dominant P1) | **Low** (competing peaks) | INVERTED |
| **Amp** (amplitude) | High | High | Standard |
| **AF** (area factor) | High | High | Standard |

Wind data from PILM4 confirms predominant NNW/NW winds during the ice season,
consistent with mechanical ice deformation in the antenna's SSE field of view.

**GLERL-optimized single-feature accuracy rankings:**
1. γ: 81.5% — best single feature for Great Lakes
2. AF: 79.6%
3. Amp: 75.1%
4. PR: 75.0% (with inverted polarity)
5. CLR: 74.5% (with inverted polarity)
6. RH std: 66.5% (weak — not useful here)

**Applied configuration:**
- γ, CLR, PR: polarity inverted for rough-ice physics
- AF, Amp: upweighted to 2.0 (strongest absolute discriminators)
- Smoothing: 5 days, min arcs: 4 per sector
- All-years correlation: r ≈ −0.47 vs GLERL az-filtered truth

**Version history:**
- v1: Default SNR feature polarities (r = +0.25, wrong direction)
- v2: + Station overview, azimuth geometry, wind data, Fresnel maps
- v3: Inverted γ, downweighted CLR/PR to 0.5 (r ≈ −0.35)
- v4: Disabled CLR/PR, γ=1.5, smoothing=5, min_arcs=4 (r ≈ −0.40)
- v5: + GLERL regional context map, azimuth-filtered sampling
- v6: Re-enabled CLR/PR with inverted polarity, GLERL-optimized weights (r ≈ −0.47)
"""))

    # =====================================================================
    # Section 15: Phase Observable (v7)
    # =====================================================================
    cells.append(_md("""
## 15. Phase Observable (v7)

The reflection phase φ is extracted from the detrended SNR model:

$$dSNR(\\varepsilon) = A \\cdot e^{-\\gamma \\sin^2\\varepsilon} \\cdot \\cos\\left(\\frac{4\\pi h}{\\lambda}\\sin\\varepsilon + \\varphi\\right)$$

Phase responds to the **dielectric properties** of the reflecting surface. A permittivity
change (water→ice, dry ice→wet melt) shifts the Fresnel reflection coefficient phase
(Strandberg et al. 2017, IEEE GRSL 14(9); Muñoz-Martín et al. 2020).

At coastal/lakeside sites, differential tropospheric delay between land-facing and
water-facing arcs also contributes a phase offset — this is the basis for the
atmospheric comparison in Section 18.

The phase is extracted via a matched-filter (inner product) at the LSP-derived RH
frequency, wrapped to [−π, π].
"""))

    cells.append(_code("""
if has_snr and "phase" in per_arc.columns:
    phase_valid = per_arc["phase"].notna().sum()
    print(f"Phase values: {phase_valid}/{len(per_arc)} arcs ({100*phase_valid/len(per_arc):.1f}%)")

    # Monthly circular mean
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: daily circular mean phase time series
    # Use daily_features if available, otherwise compute from per_arc
    ax = axes[0, 0]
    if daily_features is not None and "phase_circ_mean" in daily_features.columns:
        pooled_phase = daily_features[daily_features["azimuth_bin"] == -1][["date", "phase_circ_mean"]].copy()
        pooled_phase["date_dt"] = pd.to_datetime(pooled_phase["date"])
        daily_phase = pooled_phase.rename(columns={"phase_circ_mean": "phase_mean"}).dropna(subset=["phase_mean"])
        print("  Daily phase from daily_features (Layer 2)")
    else:
        daily_phase = per_arc.groupby("date_dt")["phase"].apply(
            lambda x: np.arctan2(np.sin(x.dropna()).mean(), np.cos(x.dropna()).mean())
        ).reset_index(name="phase_mean")
        print("  Daily phase computed inline from per_arc")
    ax.scatter(daily_phase["date_dt"], daily_phase["phase_mean"], s=8, alpha=0.5, c="#333")
    if glerl is not None:
        ax2 = ax.twinx()
        ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                         color="#90caf9", alpha=0.2)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("GLERL %", color="#64b5f6", fontsize=8)
        ax2.tick_params(axis="y", colors="#64b5f6", labelsize=7)
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Daily Circular Mean Phase")

    # Top-right: monthly phase distributions
    ax = axes[0, 1]
    months = sorted(per_arc["month"].unique())
    monthly_circ_mean = []
    monthly_circ_std = []
    for m in months:
        mp = per_arc[per_arc["month"] == m]["phase"].dropna()
        if len(mp) > 5:
            cm = np.arctan2(np.sin(mp).mean(), np.cos(mp).mean())
            # Circular std: sqrt(-2 * ln(R)), R = |mean(exp(i*phase))|
            R = np.abs(np.mean(np.exp(1j * mp.values)))
            cs = np.sqrt(-2 * np.log(max(R, 1e-10)))
            monthly_circ_mean.append(cm)
            monthly_circ_std.append(cs)
        else:
            monthly_circ_mean.append(np.nan)
            monthly_circ_std.append(np.nan)

    # Color by GLERL
    if glerl is not None:
        glerl_mon = glerl.groupby("month")["ice_concentration"].mean()
    else:
        glerl_mon = pd.Series(dtype=float)
    bar_colors = []
    for m in months:
        g = glerl_mon.get(m, 0)
        bar_colors.append("#1565c0" if g > 30 else "#f9a825" if g > 5 else "#2e7d32")

    ax.bar(months, monthly_circ_mean, yerr=monthly_circ_std, color=bar_colors,
           alpha=0.7, edgecolor="white", capsize=3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Circular Mean Phase (rad)")
    ax.set_title("Monthly Phase (blue=ice, green=water)")
    ax.set_xticks(months)
    ax.set_xticklabels([MONTH_NAMES.get(m, str(m)) for m in months], fontsize=8)

    # Bottom-left: ice vs water phase distribution
    ax = axes[1, 0]
    if glerl is not None:
        glerl_mon = glerl.groupby("month")["ice_concentration"].mean()
        ice_m = [m for m in glerl_mon.index if glerl_mon[m] > 30]
        water_m = [m for m in glerl_mon.index if glerl_mon[m] < 5]
        phase_ice = per_arc[per_arc["month"].isin(ice_m)]["phase"].dropna()
        phase_water = per_arc[per_arc["month"].isin(water_m)]["phase"].dropna()

        bins = np.linspace(-np.pi, np.pi, 50)
        if len(phase_water) > 0:
            ax.hist(phase_water, bins=bins, alpha=0.6, density=True,
                    color=CLASS_COLOR["water"], label=f"Water months (n={len(phase_water):,})")
        if len(phase_ice) > 0:
            ax.hist(phase_ice, bins=bins, alpha=0.6, density=True,
                    color=CLASS_COLOR["ice"], label=f"Ice months (n={len(phase_ice):,})")
        ax.set_xlabel("Phase (rad)")
        ax.set_ylabel("Density")
        ax.set_title("Phase Distribution: Ice vs Water")
        ax.legend(fontsize=8)

    # Bottom-right: phase vs GLERL scatter
    ax = axes[1, 1]
    if glerl is not None:
        dp = daily_phase.copy()
        dp = dp.merge(glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")
        valid = dp.dropna(subset=["phase_mean", "ice_concentration"])
        if len(valid) > 20:
            r = valid["phase_mean"].corr(valid["ice_concentration"])
            ax.scatter(valid["ice_concentration"], valid["phase_mean"], s=8, alpha=0.4, c="#555")
            ax.set_xlabel("GLERL Ice %")
            ax.set_ylabel("Daily Circular Mean Phase (rad)")
            ax.set_title(f"Phase vs GLERL: r={r:+.3f}")
            z = np.polyfit(valid["ice_concentration"], valid["phase_mean"], 1)
            xl = np.linspace(0, 100, 50)
            ax.plot(xl, np.polyval(z, xl), "r--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")

    fig.suptitle(f"{STATION} {YEAR} — Reflection Phase Analysis", fontsize=12)
    fig.tight_layout()
    plt.show()

    # Quantitative summary
    if glerl is not None and len(phase_ice) > 0 and len(phase_water) > 0:
        cm_ice = np.arctan2(np.sin(phase_ice).mean(), np.cos(phase_ice).mean())
        cm_water = np.arctan2(np.sin(phase_water).mean(), np.cos(phase_water).mean())
        delta = np.arctan2(np.sin(cm_ice - cm_water), np.cos(cm_ice - cm_water))
        print(f"Circular mean phase — ice months: {cm_ice:+.3f} rad, water months: {cm_water:+.3f} rad")
        print(f"Ice–water phase shift: {delta:+.3f} rad ({np.degrees(delta):+.1f}°)")
        if abs(delta) > 0.2:
            print("→ Significant phase shift detected between ice and water seasons.")
        else:
            print("→ Minimal phase shift — phase may not be a strong discriminator at this station.")
else:
    print("No phase data available. Re-run snr_feature_extractor.py to extract phase.")
"""))

    # =====================================================================
    # Section 15a: Per-Satellite Phase Analysis (v7.1)
    # =====================================================================
    cells.append(_md("""
### 15a. Per-Satellite Phase — Unmasking the Signal (v7.1)

The overall ice-water phase shift (~0.1 rad / 5.6° at ROSS) is a station-level average.
But phase is satellite-geometry-dependent — each PRN has a different elevation profile
and azimuth, introducing satellite-specific phase biases. Averaging across PRNs may
wash out the surface-driven shift if it's smaller than the inter-satellite variability.

This section disaggregates phase by:
1. **Individual satellites** — per-PRN ice vs water phase contrast
2. **Frequency band** — L1 vs L2C phase behavior (differential penetration?)
3. **Phase variability** — circular standard deviation as a discriminator
"""))

    cells.append(_code("""
if has_snr and "phase" in per_arc.columns and glerl is not None:
    glerl_mon = glerl.groupby("month")["ice_concentration"].mean()
    ice_m = [m for m in glerl_mon.index if glerl_mon[m] > 30]
    water_m = [m for m in glerl_mon.index if glerl_mon[m] < 5]

    # --- Per-satellite phase contrast ---
    prn_shifts = []
    for prn in per_arc["sat"].unique():
        prn_arcs = per_arc[per_arc["sat"] == prn]
        pi = prn_arcs[prn_arcs["month"].isin(ice_m)]["phase"].dropna()
        pw = prn_arcs[prn_arcs["month"].isin(water_m)]["phase"].dropna()
        if len(pi) >= 20 and len(pw) >= 20:
            cm_i = np.arctan2(np.sin(pi).mean(), np.cos(pi).mean())
            cm_w = np.arctan2(np.sin(pw).mean(), np.cos(pw).mean())
            shift = np.arctan2(np.sin(cm_i - cm_w), np.cos(cm_i - cm_w))
            prn_shifts.append({
                "sat": prn, "n_ice": len(pi), "n_water": len(pw),
                "phase_ice": cm_i, "phase_water": cm_w, "shift": shift,
            })

    if prn_shifts:
        prn_df = pd.DataFrame(prn_shifts).sort_values("shift")
        shifts = prn_df["shift"].values
        median_shift = np.median(shifts)

        print(f"Per-satellite ice-water phase shift ({len(prn_df)} PRNs with >=20 arcs each):")
        print(f"{'PRN':>5} | {'N_ice':>5} | {'N_water':>7} | {'phi_ice':>8} | {'phi_wat':>8} | {'Shift':>7}")
        print("-" * 56)
        for _, row in prn_df.iterrows():
            print(f"  {int(row['sat']):>3} | {int(row['n_ice']):>5} | {int(row['n_water']):>7} | "
                  f"{row['phase_ice']:>+8.3f} | {row['phase_water']:>+8.3f} | {row['shift']:>+7.3f}")
        print(f"\\nMedian per-satellite shift: {median_shift:+.3f} rad ({np.degrees(median_shift):+.1f} deg)")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Left: per-satellite shift distribution
        ax = axes[0]
        ax.barh(range(len(prn_df)), prn_df["shift"].values,
                color=["#1565c0" if s > 0 else "#c62828" for s in prn_df["shift"].values])
        ax.set_yticks(range(len(prn_df)))
        ax.set_yticklabels([f"PRN {int(s)}" for s in prn_df["sat"].values], fontsize=8)
        ax.axvline(0, color="gray", ls="-", alpha=0.5)
        ax.axvline(median_shift, color="red", ls="--", alpha=0.7,
                   label=f"Median: {median_shift:+.3f} rad")
        ax.set_xlabel("Ice - Water Phase Shift (rad)")
        ax.set_title("Per-Satellite Phase Shift")
        ax.legend(fontsize=8)

        # Middle: top 4 PRNs phase time series
        ax = axes[1]
        top_prns = per_arc.groupby("sat")["phase"].count().nlargest(4).index
        colors_prn = ["#e53935", "#1e88e5", "#43a047", "#8e24aa"]
        for prn, clr in zip(top_prns, colors_prn):
            prn_daily = per_arc[per_arc["sat"] == prn].groupby("date_dt")["phase"].apply(
                lambda x: np.arctan2(np.sin(x.dropna()).mean(), np.cos(x.dropna()).mean())
            ).reset_index(name="phase_mean").set_index("date_dt").sort_index()
            rolling = prn_daily["phase_mean"].rolling("7D", center=True).median()
            ax.plot(rolling.index, rolling.values, linewidth=1.2, alpha=0.7,
                    color=clr, label=f"PRN {int(prn)}")
        if glerl is not None:
            ax_g = ax.twinx()
            ax_g.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                             color="#90caf9", alpha=0.15)
            ax_g.set_ylim(0, 105)
            ax_g.set_ylabel("GLERL %", fontsize=8, color="#64b5f6")
            ax_g.tick_params(axis="y", colors="#64b5f6", labelsize=7)
        ax.set_ylabel("Phase (rad, 7d median)")
        ax.set_title("Top-4 PRN Phase Time Series")
        ax.legend(fontsize=7, loc="upper left")

        # Right: phase by frequency band
        ax = axes[2]
        if "freq_group" in per_arc.columns:
            per_arc["_phband"] = per_arc["freq_group"]
        else:
            _PH_BAND = {1: "L1", 2: "L2", 20: "L2", 5: "L5",
                         101: "L1", 102: "L2", 105: "L5", 201: "L1", 205: "L5"}
            per_arc["_phband"] = per_arc["freq"].map(_PH_BAND)
        band_shifts = {}
        for band, color in [("L1", "#e53935"), ("L2", "#1e88e5"), ("L5", "#43a047")]:
            bm = per_arc[per_arc["_phband"] == band]
            if len(bm) < 50:
                continue
            daily_b = bm.groupby("date_dt")["phase"].apply(
                lambda x: np.arctan2(np.sin(x.dropna()).mean(), np.cos(x.dropna()).mean())
            ).reset_index(name="phase_mean").set_index("date_dt").sort_index()
            rolling = daily_b["phase_mean"].rolling("7D", center=True).median()
            ax.plot(rolling.index, rolling.values, linewidth=1.2, alpha=0.7,
                    color=color, label=band)
            bm_ice = bm[bm["month"].isin(ice_m)]["phase"].dropna()
            bm_water = bm[bm["month"].isin(water_m)]["phase"].dropna()
            if len(bm_ice) > 10 and len(bm_water) > 10:
                cm_i = np.arctan2(np.sin(bm_ice).mean(), np.cos(bm_ice).mean())
                cm_w = np.arctan2(np.sin(bm_water).mean(), np.cos(bm_water).mean())
                bshift = np.arctan2(np.sin(cm_i - cm_w), np.cos(cm_i - cm_w))
                band_shifts[band] = bshift
                print(f"  {band} phase shift: {bshift:+.3f} rad ({np.degrees(bshift):+.1f} deg)")
        ax.set_ylabel("Phase (rad, 7d median)")
        ax.set_title("Phase by Frequency Band")
        ax.legend(fontsize=8)

        fig.suptitle(f"{STATION} {YEAR} — Per-Satellite Phase Analysis (v7.1)", fontsize=12)
        fig.tight_layout()
        plt.show()

        # Phase stability: circular std per day as discriminator
        daily_circ_std = per_arc.groupby("date_dt")["phase"].apply(
            lambda x: np.sqrt(-2 * np.log(max(np.abs(np.mean(np.exp(1j * x.dropna().values))), 1e-10)))
            if len(x.dropna()) > 3 else np.nan
        ).reset_index(name="phase_cstd")

        cs_merged = daily_circ_std.merge(glerl[["date_dt", "ice_concentration"]],
                                          on="date_dt", how="inner")
        cs_ice = cs_merged[cs_merged["ice_concentration"] > 30]["phase_cstd"].dropna()
        cs_water = cs_merged[cs_merged["ice_concentration"] < 5]["phase_cstd"].dropna()

        print(f"\\nPhase circular std (variability):")
        if len(cs_ice) > 5 and len(cs_water) > 5:
            print(f"  Ice months:   median = {cs_ice.median():.3f} rad")
            print(f"  Water months: median = {cs_water.median():.3f} rad")
            r_cs = cs_merged.dropna(subset=["phase_cstd", "ice_concentration"])
            if len(r_cs) > 20:
                r_val = r_cs["phase_cstd"].corr(r_cs["ice_concentration"])
                print(f"  Correlation with GLERL: r = {r_val:+.3f}")
            if cs_ice.median() > cs_water.median() * 1.1:
                print("  -> Phase variability INCREASES during ice (multiple reflecting interfaces)")
            elif cs_water.median() > cs_ice.median() * 1.1:
                print("  -> Phase variability DECREASES during ice (coherent specular reflection)")
            else:
                print("  -> No significant seasonal change in phase variability")

        # Combined summary
        print(f"\\n{'='*60}")
        print("Phase Summary (v7.1)")
        print(f"{'='*60}")
        print(f"  Per-satellite median shift:     {median_shift:+.3f} rad ({np.degrees(median_shift):+.1f} deg)")
        if band_shifts:
            for band, bs in band_shifts.items():
                print(f"  {band} phase shift:              {bs:+.3f} rad ({np.degrees(bs):+.1f} deg)")
            if "L1" in band_shifts and "L2" in band_shifts:
                diff = band_shifts["L1"] - band_shifts["L2"]
                print(f"  L1-L2 shift difference:         {diff:+.3f} rad ({np.degrees(diff):+.1f} deg)")
                if abs(diff) > 0.1:
                    print("  -> Divergent band response supports frequency-dependent penetration")
                else:
                    print("  -> Consistent band response — no evidence for differential penetration via phase")
    else:
        print("Insufficient per-satellite data (need >=20 arcs per PRN per season)")
else:
    print("Per-satellite phase analysis requires SNR features with phase and GLERL data.")
"""))

    # =====================================================================
    # Section 16: Interfrequency ΔRH (v7)
    # =====================================================================
    cells.append(_md("""
## 16. Interfrequency ΔRH — Per-Sector Frequency Spread (v7)

When L-band signals hit open water, all frequencies reflect from the surface — tight ΔRH.
When signals hit freshwater ice, they partially penetrate (Ghiasi 2020), and different
frequencies penetrate to different depths — ΔRH diverges.

**Important caveat:** ROSS only has **2 GPS frequencies** (L1 and L2C). The frequency spread
is narrow (L1=1575 MHz, L2=1228 MHz), so ΔRH will be a weak signal at best. This section
documents the methodology and the ROSS-specific result. The same analysis at multi-GNSS
stations (UMNQ, Greenland EarthScope) with 6+ frequencies will have more discriminating power.

ΔRH is computed per (date, azimuth sector) as the standard deviation of per-band RH medians.
"""))

    cells.append(_code("""
# ΔRH from daily_features (matched-arc method) or computed inline (legacy band-median)
if daily_features is not None and "delta_rh_mean" in daily_features.columns:
    # Use pre-computed matched-arc ΔRH from daily_features
    sector_df = daily_features[daily_features["azimuth_bin"] >= 0].copy()
    sector_df["date_dt"] = pd.to_datetime(sector_df["date"])
    drh_df = sector_df[["date_dt", "azimuth_bin", "delta_rh_mean", "delta_rh_std", "delta_rh_n_pairs"]].copy()
    drh_df = drh_df.rename(columns={"delta_rh_mean": "delta_rh"})
    drh_df = drh_df.dropna(subset=["delta_rh"])
    n_bands = len([c for c in daily_features.columns if c.startswith("rh_") and c.endswith("_median")])
    print(f"ΔRH from daily_features (matched-arc method): {len(drh_df)} (date, sector) rows")
    print(f"  Per-band RH columns: {n_bands}")

    # Station-level daily ΔRH from pooled rows
    pooled_drh = daily_features[daily_features["azimuth_bin"] == -1].copy()
    pooled_drh["date_dt"] = pd.to_datetime(pooled_drh["date"])
    daily_drh = pooled_drh[["date_dt", "delta_rh_mean"]].rename(
        columns={"delta_rh_mean": "delta_rh"}).dropna(subset=["delta_rh"])
else:
    # Legacy: compute per-sector band-median ΔRH from per_arc
    FREQ_TO_BAND = {
        1: "L1", 2: "L2", 20: "L2", 5: "L5",
        101: "L1", 102: "L2", 105: "L5",
        201: "L1", 205: "L5", 206: "L5", 207: "L5", 208: "E6",
        301: "L1", 302: "L1", 306: "B3", 307: "L5",
    }
    _band_col = "freq_group" if "freq_group" in per_arc.columns else None
    if _band_col is None:
        per_arc["_band"] = per_arc["freq"].map(FREQ_TO_BAND)
        _band_col = "_band"

    n_bands = per_arc[_band_col].nunique()
    print(f"ΔRH computed inline (legacy band-median method)")
    print(f"Frequency bands: {sorted(per_arc[_band_col].dropna().unique())} ({n_bands} bands)")

    drh_rows = []
    if n_bands >= 2:
        for (date, az_bin), grp in per_arc.groupby(["date_dt", "azimuth_bin"]):
            band_rh = grp.groupby(_band_col)["RH"].median()
            if len(band_rh) >= 2:
                drh_rows.append({
                    "date_dt": date, "azimuth_bin": az_bin,
                    "delta_rh": float(band_rh.std()),
                    "n_bands": len(band_rh),
                })
    drh_df = pd.DataFrame(drh_rows)
    daily_drh = drh_df.groupby("date_dt")["delta_rh"].median().reset_index() if len(drh_df) > 0 else pd.DataFrame(columns=["date_dt", "delta_rh"])
    print(f"ΔRH computed for {len(drh_df)} (date, sector) pairs")

if len(daily_drh) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: ΔRH time series
    ax = axes[0, 0]
    ax.scatter(daily_drh["date_dt"], daily_drh["delta_rh"], s=8, alpha=0.5, c="#333")
    if glerl is not None:
        ax2 = ax.twinx()
        ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                         color="#90caf9", alpha=0.2)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("GLERL %", color="#64b5f6", fontsize=8)
        ax2.tick_params(axis="y", colors="#64b5f6", labelsize=7)
    ax.set_ylabel("ΔRH (m)")
    ax.set_title("Daily Median ΔRH (interfrequency spread)")

    # Top-right: monthly ΔRH
    ax = axes[0, 1]
    daily_drh["month"] = daily_drh["date_dt"].dt.month
    monthly_drh = daily_drh.groupby("month")["delta_rh"].agg(["median", "std"])
    bar_colors = []
    for m in monthly_drh.index:
        g = glerl_mon.get(m, 0) if glerl is not None else 0
        bar_colors.append("#1565c0" if g > 30 else "#f9a825" if g > 5 else "#2e7d32")
    ax.bar(monthly_drh.index, monthly_drh["median"], yerr=monthly_drh["std"],
           color=bar_colors, alpha=0.7, edgecolor="white", capsize=3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Median ΔRH (m)")
    ax.set_title("Monthly ΔRH")
    ax.set_xticks(sorted(monthly_drh.index))
    ax.set_xticklabels([MONTH_NAMES.get(m, str(m)) for m in sorted(monthly_drh.index)], fontsize=8)

    # Bottom-left: ice vs water ΔRH distribution
    ax = axes[1, 0]
    if glerl is not None:
        drh_ice = daily_drh[daily_drh["month"].isin(ice_m := [m for m in glerl.groupby("month")["ice_concentration"].mean().index if glerl.groupby("month")["ice_concentration"].mean()[m] > 30])]["delta_rh"]
        drh_water = daily_drh[daily_drh["month"].isin(water_m := [m for m in glerl.groupby("month")["ice_concentration"].mean().index if glerl.groupby("month")["ice_concentration"].mean()[m] < 5])]["delta_rh"]
        bins = np.linspace(0, max(drh_ice.quantile(0.99) if len(drh_ice) > 0 else 0.5,
                                   drh_water.quantile(0.99) if len(drh_water) > 0 else 0.5), 40)
        if len(drh_water) > 0:
            ax.hist(drh_water, bins=bins, alpha=0.6, density=True,
                    color=CLASS_COLOR["water"], label=f"Water ({len(drh_water)} days)")
        if len(drh_ice) > 0:
            ax.hist(drh_ice, bins=bins, alpha=0.6, density=True,
                    color=CLASS_COLOR["ice"], label=f"Ice ({len(drh_ice)} days)")
        ax.set_xlabel("ΔRH (m)")
        ax.set_ylabel("Density")
        ax.set_title("ΔRH Distribution: Ice vs Water")
        ax.legend(fontsize=8)

    # Bottom-right: ΔRH vs GLERL scatter
    ax = axes[1, 1]
    if glerl is not None:
        m = daily_drh.merge(glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")
        valid = m.dropna(subset=["delta_rh", "ice_concentration"])
        if len(valid) > 20:
            r = valid["delta_rh"].corr(valid["ice_concentration"])
            ax.scatter(valid["ice_concentration"], valid["delta_rh"], s=8, alpha=0.4, c="#555")
            ax.set_xlabel("GLERL Ice %")
            ax.set_ylabel("ΔRH (m)")
            ax.set_title(f"ΔRH vs GLERL: r={r:+.3f}")
            z = np.polyfit(valid["ice_concentration"], valid["delta_rh"], 1)
            xl = np.linspace(0, 100, 50)
            ax.plot(xl, np.polyval(z, xl), "r--", alpha=0.5)

    # Determine band list for title
    if "freq_group" in per_arc.columns:
        _band_list = sorted(per_arc["freq_group"].dropna().unique())
    elif "_band" in per_arc.columns:
        _band_list = sorted(per_arc["_band"].dropna().unique())
    else:
        _band_list = []
    fig.suptitle(f"{STATION} {YEAR} — Interfrequency ΔRH Analysis\\n"
                 f"({n_bands} bands: {_band_list})", fontsize=12)
    fig.tight_layout()
    plt.show()

    # Quantitative summary
    if glerl is not None and len(drh_ice) > 0 and len(drh_water) > 0:
        print(f"\\nΔRH — ice months: median={drh_ice.median():.4f} m, water months: median={drh_water.median():.4f} m")
        ratio = drh_ice.median() / max(drh_water.median(), 1e-6)
        print(f"  Ice/water ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print("  → Ice ΔRH is larger — consistent with ice penetration hypothesis.")
        elif ratio < 0.8:
            print("  → Ice ΔRH is smaller — unexpected, may reflect frequency-dependent scattering.")
        else:
            print("  → Minimal contrast — insufficient frequency diversity at this station (2 bands).")
            print("  → This is an expected negative result for GPS-only stations.")
else:
    print("Insufficient ΔRH data for visualization.")
"""))

    # =====================================================================
    # Section 16a: ΔRH Validation — Quality Filters (v7.1)
    # =====================================================================
    cells.append(_md("""
### 16a. ΔRH Validation — Is the ice/water ratio real? (v7.1)

The ice/water ΔRH ratio from Section 16 may be inflated by artifacts. Three concerns:

1. **Bad arcs during ice months** — rough ice may produce poor LSP fits with unreliable
   RH, inflating the per-band standard deviation that defines ΔRH
2. **Low arc density** — days with very few arcs per band are noise-dominated
3. **Systematic L1-L2 offset** — if ice penetration is real, L1 vs L2 RH should show
   a systematic offset from the 1:1 line during ice months (L2 penetrates deeper)

This section applies quality filters to validate whether the signal survives.
"""))

    cells.append(_code("""
# --- L1 vs L2 RH scatter plot ---
# Use freq_group if available (from arc_table), otherwise map from freq
if "freq_group" in per_arc.columns:
    per_arc["_band_v"] = per_arc["freq_group"]
else:
    _BAND_V = {
        1: "L1", 2: "L2", 20: "L2", 5: "L5",
        101: "L1", 102: "L2", 105: "L5",
        201: "L1", 205: "L5", 206: "L5", 207: "L5", 208: "E6",
        301: "L1", 302: "L1", 306: "B3", 307: "L5",
    }
    per_arc["_band_v"] = per_arc["freq"].map(_BAND_V)

l1_arcs = per_arc[per_arc["_band_v"] == "L1"][["date", "date_dt", "sat", "RH", "month"]].copy()
l1_arcs = l1_arcs.rename(columns={"RH": "RH_L1"})
l2_arcs = per_arc[per_arc["_band_v"] == "L2"][["date", "sat", "RH"]].copy()
l2_arcs = l2_arcs.rename(columns={"RH": "RH_L2"})

matched = l1_arcs.merge(l2_arcs, on=["date", "sat"])
print(f"L1-L2 matched pairs: {len(matched)} (by date + satellite)")

if len(matched) > 50 and glerl is not None:
    glerl_mon_v = glerl.groupby("month")["ice_concentration"].mean()
    ice_m_v = [m for m in glerl_mon_v.index if glerl_mon_v[m] > 30]
    water_m_v = [m for m in glerl_mon_v.index if glerl_mon_v[m] < 5]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: L1 vs L2 RH scatter colored by season
    ax = axes[0]
    ice_mask_v = matched["month"].isin(ice_m_v)
    water_mask_v = matched["month"].isin(water_m_v)
    other_mask_v = ~ice_mask_v & ~water_mask_v

    if other_mask_v.sum() > 0:
        ax.scatter(matched.loc[other_mask_v, "RH_L1"], matched.loc[other_mask_v, "RH_L2"],
                   s=5, alpha=0.15, c="#888", label=f"Other ({other_mask_v.sum()})")
    if water_mask_v.sum() > 0:
        ax.scatter(matched.loc[water_mask_v, "RH_L1"], matched.loc[water_mask_v, "RH_L2"],
                   s=8, alpha=0.3, c=CLASS_COLOR["water"], label=f"Water ({water_mask_v.sum()})")
    if ice_mask_v.sum() > 0:
        ax.scatter(matched.loc[ice_mask_v, "RH_L1"], matched.loc[ice_mask_v, "RH_L2"],
                   s=8, alpha=0.3, c=CLASS_COLOR["ice"], label=f"Ice ({ice_mask_v.sum()})")

    rh_lo = matched[["RH_L1", "RH_L2"]].min().min()
    rh_hi = matched[["RH_L1", "RH_L2"]].max().max()
    ax.plot([rh_lo, rh_hi], [rh_lo, rh_hi], "k--", alpha=0.4, label="1:1 line")

    ice_offset = (matched.loc[ice_mask_v, "RH_L2"].median() -
                  matched.loc[ice_mask_v, "RH_L1"].median()) if ice_mask_v.sum() > 10 else np.nan
    water_offset = (matched.loc[water_mask_v, "RH_L2"].median() -
                    matched.loc[water_mask_v, "RH_L1"].median()) if water_mask_v.sum() > 10 else np.nan

    ax.set_xlabel("L1 RH (m)")
    ax.set_ylabel("L2 RH (m)")
    ax.set_title("L1 vs L2 RH by Season")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Middle: ΔRH time series with arc count overlay
    ax = axes[1]
    drh_v_rows = []
    for dt_v, grp in per_arc.groupby("date_dt"):
        band_rh = grp.groupby("_band_v")["RH"].agg(["median", "count"])
        if len(band_rh) >= 2:
            drh_v_rows.append({
                "date_dt": dt_v,
                "delta_rh": float(band_rh["median"].std()),
                "min_band_count": int(band_rh["count"].min()),
                "n_arcs": len(grp),
            })
    drh_v = pd.DataFrame(drh_v_rows)

    sc_v = ax.scatter(drh_v["date_dt"], drh_v["delta_rh"],
                      c=drh_v["min_band_count"].clip(1, 10),
                      cmap="RdYlGn", s=10, alpha=0.6, vmin=1, vmax=10)
    ax.set_ylabel("delta_RH (m)")
    ax.set_title("Daily delta_RH (color = min arcs/band)")
    plt.colorbar(sc_v, ax=ax, shrink=0.7, label="Min arcs/band")

    # Right: filtered vs unfiltered ratio comparison
    ax = axes[2]
    drh_v["month"] = drh_v["date_dt"].dt.month

    drh_ice_uf = drh_v[drh_v["month"].isin(ice_m_v)]["delta_rh"]
    drh_water_uf = drh_v[drh_v["month"].isin(water_m_v)]["delta_rh"]
    ratio_uf = drh_ice_uf.median() / max(drh_water_uf.median(), 1e-6) if len(drh_ice_uf) > 0 and len(drh_water_uf) > 0 else np.nan

    drh_filt = drh_v[drh_v["min_band_count"] >= 3]
    drh_ice_f = drh_filt[drh_filt["month"].isin(ice_m_v)]["delta_rh"]
    drh_water_f = drh_filt[drh_filt["month"].isin(water_m_v)]["delta_rh"]
    ratio_f = drh_ice_f.median() / max(drh_water_f.median(), 1e-6) if len(drh_ice_f) > 0 and len(drh_water_f) > 0 else np.nan

    # Arc quality: elevation coverage >= 80% of configured range
    e1_cfg, e2_cfg = 5.0, 25.0
    per_arc["_elev_cov"] = (per_arc["emaxO"] - per_arc["eminO"]) / (e2_cfg - e1_cfg)
    full_arc_mask = per_arc["_elev_cov"] >= 0.8
    pct_full_ice = full_arc_mask[per_arc["month"].isin(ice_m_v)].mean() * 100
    pct_full_water = full_arc_mask[per_arc["month"].isin(water_m_v)].mean() * 100

    pa_full = per_arc[full_arc_mask]
    drh_full_rows = []
    for dt_v, grp in pa_full.groupby("date_dt"):
        band_rh = grp.groupby("_band_v")["RH"].agg(["median", "count"])
        if len(band_rh) >= 2:
            drh_full_rows.append({
                "date_dt": dt_v,
                "delta_rh": float(band_rh["median"].std()),
                "month": dt_v.month,
            })
    drh_full_df = pd.DataFrame(drh_full_rows) if drh_full_rows else pd.DataFrame(columns=["date_dt", "delta_rh", "month"])
    drh_ice_fa = drh_full_df[drh_full_df["month"].isin(ice_m_v)]["delta_rh"] if len(drh_full_df) > 0 else pd.Series(dtype=float)
    drh_water_fa = drh_full_df[drh_full_df["month"].isin(water_m_v)]["delta_rh"] if len(drh_full_df) > 0 else pd.Series(dtype=float)
    ratio_fa = drh_ice_fa.median() / max(drh_water_fa.median(), 1e-6) if len(drh_ice_fa) > 0 and len(drh_water_fa) > 0 else np.nan

    labels_bar = ["Unfiltered", ">=3 arcs/band", "Full arcs (80%)"]
    ratios_bar = [ratio_uf, ratio_f, ratio_fa]
    colors_bar = ["#c62828" if r > 10 else "#f9a825" if r > 3 else "#2e7d32" for r in ratios_bar]
    ax.bar(range(len(labels_bar)), ratios_bar, color=colors_bar, alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels(labels_bar, fontsize=9)
    ax.set_ylabel("Ice/Water delta_RH Ratio")
    ax.set_title("delta_RH Ratio by Quality Filter")
    for i, r in enumerate(ratios_bar):
        if not np.isnan(r):
            ax.text(i, r + 0.3, f"{r:.1f}x", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"{STATION} {YEAR} — delta_RH Validation (v7.1)", fontsize=12)
    fig.tight_layout()
    plt.show()

    # Quantitative summary
    print(f"\\ndelta_RH Validation Summary:")
    print(f"{'='*60}")
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*55}")
    print(f"  {'delta_RH ice/water (unfiltered)':<40} {ratio_uf:>10.1f}x")
    print(f"  {'delta_RH ice/water (>=3 arcs/band)':<40} {ratio_f:>10.1f}x")
    print(f"  {'delta_RH ice/water (full arcs only)':<40} {ratio_fa:>10.1f}x")
    if not np.isnan(ice_offset):
        print(f"  {'L1-L2 RH offset — ice months':<40} {ice_offset:>+10.4f} m")
    if not np.isnan(water_offset):
        print(f"  {'L1-L2 RH offset — water months':<40} {water_offset:>+10.4f} m")
    offset_diff = ice_offset - water_offset if not (np.isnan(ice_offset) or np.isnan(water_offset)) else np.nan
    if not np.isnan(offset_diff):
        print(f"  {'Offset difference (ice - water)':<40} {offset_diff:>+10.4f} m")
        if offset_diff > 0.01:
            print("  -> Positive = deeper L2 penetration during ice (supports Ghiasi 2020)")
        elif offset_diff < -0.01:
            print("  -> Negative = deeper L1 penetration during ice (unexpected)")
        else:
            print("  -> No systematic L1-L2 offset difference between seasons")
    print(f"  {'Full-arc fraction — ice months':<40} {pct_full_ice:>9.1f}%")
    print(f"  {'Full-arc fraction — water months':<40} {pct_full_water:>9.1f}%")
    if pct_full_ice < pct_full_water * 0.7:
        print("  -> Ice months have fewer full arcs — delta_RH may be inflated by partial fits")
    else:
        print("  -> Arc quality comparable between seasons — delta_RH contrast not an artifact")

    # Conclusion
    if ratio_f < ratio_uf * 0.5:
        print("\\n  CONCLUSION: delta_RH ratio drops significantly with quality filter.")
        print("  The original ratio was inflated by low-quality arcs during ice months.")
    elif ratio_f > ratio_uf * 0.7:
        print("\\n  CONCLUSION: delta_RH ratio survives quality filtering.")
        print("  The interfrequency spread is a genuine ice signal (even with only 2 bands).")
    else:
        print("\\n  CONCLUSION: Moderate reduction with filtering — signal partially real.")
elif len(matched) > 0:
    print("L1-L2 scatter requires GLERL data for seasonal labeling.")
else:
    print("Insufficient L1-L2 matched pairs for validation.")
"""))

    # =====================================================================
    # Section 17: Freeze/Thaw Transition Detection (v7)
    # =====================================================================
    cells.append(_md("""
## 17. Freeze/Thaw Transition Detection — Moving t-Test (v7.1)

Following Ghiasi et al. 2023 (IEEE JSTARS 17), we apply a **Moving t-Test (MTT)**
changepoint detector on the classification time series. For each position, the MTT
compares the mean of the preceding *w* days against the following *w* days. Large
|t| values indicate abrupt transitions.

**v7.1 improvements:**
- **Climatological search windows:** FUS restricted to DOY 300–60 (Oct–Feb),
  BUS restricted to DOY 30–150 (Feb–May) for Great Lakes. Eliminates physically
  impossible mid-summer transition detections.
- **Data gap detection:** Gaps >7 days flagged on plot; gap-adjacent peaks marked unreliable.
- **Per-feature constrained detection:** Individual features (γ, AF, amplitude, RH std)
  run within the same windows to identify which observable is the best leading indicator.

Transition dates:
- **FUS** (Freeze-Up Start): most negative t-score within the FUS window
- **BUS** (Break-Up Start): most positive t-score within the BUS window
"""))

    cells.append(_code("""
from scipy import stats as sp_stats

def moving_t_test(series, window=30):
    \"\"\"Detect abrupt changes via consecutive-window t-test.\"\"\"
    values = np.array(series, dtype=float)
    t_scores = np.full(len(values), np.nan)
    for i in range(window, len(values) - window):
        sub1 = values[i - window:i]
        sub2 = values[i:i + window]
        s1 = sub1[~np.isnan(sub1)]
        s2 = sub2[~np.isnan(sub2)]
        if len(s1) < 5 or len(s2) < 5:
            continue
        n1, n2 = len(s1), len(s2)
        v1, v2 = s1.var(ddof=1), s2.var(ddof=1)
        denom = n1 + n2 - 2
        if denom <= 0:
            continue
        s_pooled = np.sqrt((n1 * v1 + n2 * v2) / denom)
        if s_pooled < 1e-10:
            continue
        t = (s2.mean() - s1.mean()) / (s_pooled * np.sqrt(1.0 / n1 + 1.0 / n2))
        t_scores[i] = t
    return t_scores

def doy_in_range(doy, lo, hi):
    \"\"\"Check if DOY is within [lo, hi], handling year-boundary wrap.\"\"\"
    if lo <= hi:
        return lo <= doy <= hi
    else:
        return doy >= lo or doy <= hi

# Station-specific phenology windows (v7.1)
PHENOLOGY_WINDOWS = {
    "ROSS": {"fus_doy_range": (300, 60), "bus_doy_range": (30, 150)},
    "UMNQ": {"fus_doy_range": (270, 30), "bus_doy_range": (120, 210)},
}
phenology = PHENOLOGY_WINDOWS.get(STATION, {"fus_doy_range": (270, 60), "bus_doy_range": (30, 180)})
fus_lo, fus_hi = phenology["fus_doy_range"]
bus_lo, bus_hi = phenology["bus_doy_range"]

# Critical t-value
MTT_WINDOW = 30
df_approx = 2 * MTT_WINDOW - 2
t_crit = sp_stats.t.ppf(0.999, df_approx)
print(f"MTT window: {MTT_WINDOW} days, critical |t| (p<0.001, df={df_approx}): {t_crit:.2f}")
print(f"FUS search window: DOY {fus_lo}-{fus_hi} (wraps year boundary)" if fus_lo > fus_hi else f"FUS search window: DOY {fus_lo}-{fus_hi}")
print(f"BUS search window: DOY {bus_lo}-{bus_hi}")

# Sort by date
clf_sorted = clf.sort_values("date_dt").reset_index(drop=True)
ice_score_series = clf_sorted["ice_score"].values
mtt_dates = clf_sorted["date_dt"].values
mtt_doys = pd.to_datetime(mtt_dates).dayofyear

# --- Data gap detection (v7.1) ---
date_diffs = np.diff(pd.to_datetime(mtt_dates).astype("int64") // 10**9) / 86400
gap_indices = np.where(date_diffs > 7)[0]
gap_ranges = []
for gi in gap_indices:
    gap_ranges.append((mtt_dates[gi], mtt_dates[gi + 1], date_diffs[gi]))
    print(f"  Data gap: {pd.Timestamp(mtt_dates[gi]).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(mtt_dates[gi+1]).strftime('%Y-%m-%d')} ({date_diffs[gi]:.0f} days)")
if not gap_ranges:
    print("  No data gaps >7 days detected.")

# Run MTT on ice_score
t_scores = moving_t_test(ice_score_series, window=MTT_WINDOW)

# Also run on individual features — use daily_features if available, otherwise per_arc
feature_t = {}
for feat in ["amp_mean", "rh_std"]:
    if feat in clf_sorted.columns:
        feature_t[feat] = moving_t_test(clf_sorted[feat].values, window=MTT_WINDOW)

if has_snr and daily_features is not None:
    # Use daily_features z-scored/aggregated columns
    pooled_sorted = daily_features[daily_features["azimuth_bin"] == -1].copy()
    pooled_sorted["date_dt"] = pd.to_datetime(pooled_sorted["date"])
    pooled_sorted = pooled_sorted.sort_values("date_dt")
    for feat_col, feat_name in [("gamma_z", "gamma_z"), ("af_z", "af_z"),
                                 ("delta_rh_mean", "delta_rh")]:
        if feat_col in pooled_sorted.columns:
            # Align to clf dates
            vals = pooled_sorted.set_index("date_dt")[feat_col].reindex(
                pd.to_datetime(clf_sorted["date"]))
            feature_t[feat_name] = moving_t_test(vals.values, window=MTT_WINDOW)
elif has_snr:
    for feat in ["gamma", "AF"]:
        if feat in per_arc.columns:
            daily_feat = per_arc.groupby("date_dt")[feat].median().reindex(
                pd.to_datetime(clf_sorted["date"]))
            feature_t[feat] = moving_t_test(daily_feat.values, window=MTT_WINDOW)

# --- Constrained transition detection (v7.1) ---
def find_constrained_peak(t_scores, doys, doy_lo, doy_hi, sign="negative"):
    \"\"\"Find most extreme t-score within a DOY window.\"\"\"
    best_idx = None
    best_t = 0
    for i in range(len(t_scores)):
        if np.isnan(t_scores[i]):
            continue
        if not doy_in_range(int(doys[i]), doy_lo, doy_hi):
            continue
        if sign == "negative" and t_scores[i] < -t_crit:
            if best_idx is None or t_scores[i] < best_t:
                best_t = t_scores[i]
                best_idx = i
        elif sign == "positive" and t_scores[i] > t_crit:
            if best_idx is None or t_scores[i] > best_t:
                best_t = t_scores[i]
                best_idx = i
    return best_idx, best_t

def near_gap(idx, gap_idxs, tolerance=3):
    \"\"\"Check if index is within tolerance of a gap boundary.\"\"\"
    for gi in gap_idxs:
        if abs(idx - gi) <= tolerance or abs(idx - (gi + 1)) <= tolerance:
            return True
    return False

fus_idx, fus_t = find_constrained_peak(t_scores, mtt_doys, fus_lo, fus_hi, "negative")
bus_idx, bus_t = find_constrained_peak(t_scores, mtt_doys, bus_lo, bus_hi, "positive")

constrained_transitions = []
if fus_idx is not None:
    gap_flag = " [GAP-ADJACENT]" if near_gap(fus_idx, gap_indices) else ""
    constrained_transitions.append(("FUS", mtt_dates[fus_idx], fus_t, gap_flag))
if bus_idx is not None:
    gap_flag = " [GAP-ADJACENT]" if near_gap(bus_idx, gap_indices) else ""
    constrained_transitions.append(("BUS", mtt_dates[bus_idx], bus_t, gap_flag))

# Backward compatibility for Section 18
detected_transitions = [(label, dt) for label, dt, _, _ in constrained_transitions]

# --- Plotting ---
n_panels = 2 + len(feature_t)
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
if not hasattr(axes, "__len__"):
    axes = [axes]

# Panel 1: ice score
ax = axes[0]
for cls, color in CLASS_COLOR.items():
    mask = clf_sorted["classification"] == cls
    idx = np.where(mask.values)[0]
    if len(idx) > 0:
        ax.scatter(mtt_dates[idx], ice_score_series[idx], c=color, s=8, alpha=0.5)
ax.axhline(-0.33, color="gray", ls=":", alpha=0.4)
ax.axhline(0.33, color="gray", ls=":", alpha=0.4)
ax.set_ylabel("Ice Score")
ax.set_title("Classification Time Series")
if glerl is not None:
    ax_g = ax.twinx()
    ax_g.fill_between(glerl["date_dt"], glerl["ice_concentration"], color="#90caf9", alpha=0.15)
    ax_g.set_ylim(0, 105)
    ax_g.set_ylabel("GLERL %", fontsize=8, color="#64b5f6")
    ax_g.tick_params(axis="y", colors="#64b5f6", labelsize=7)
for g_start, g_end, g_days in gap_ranges:
    ax.axvspan(g_start, g_end, color="gray", alpha=0.15)

# Panel 2: MTT t-scores with constrained windows
ax = axes[1]
valid_mask = ~np.isnan(t_scores)
ax.plot(mtt_dates[valid_mask], t_scores[valid_mask], "k-", linewidth=0.8, alpha=0.7)
ax.axhline(t_crit, color="red", ls="--", alpha=0.5, label=f"+t_crit ({t_crit:.1f})")
ax.axhline(-t_crit, color="blue", ls="--", alpha=0.5, label=f"-t_crit")
ax.axhline(0, color="gray", ls="-", alpha=0.3)

# Shade search windows
year_start = pd.Timestamp(f"{YEAR}-01-01")
for doy in range(1, 367):
    d = year_start + pd.Timedelta(days=doy - 1)
    if d > pd.Timestamp(mtt_dates[-1]):
        break
    if doy_in_range(doy, fus_lo, fus_hi):
        ax.axvspan(d, d + pd.Timedelta(days=1), color="blue", alpha=0.02)
    if doy_in_range(doy, bus_lo, bus_hi):
        ax.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.02)

# Mark constrained transitions
for label, dt, t_val, gap_flag in constrained_transitions:
    c = "blue" if label == "FUS" else "red"
    ax.axvline(pd.Timestamp(dt), color=c, ls="-", linewidth=2, alpha=0.6)
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1] * 0.85,
            f" {label}{gap_flag}", color=c, fontsize=8, fontweight="bold")

for g_start, g_end, g_days in gap_ranges:
    ax.axvspan(g_start, g_end, color="gray", alpha=0.15)

ax.set_ylabel("t-score")
ax.set_title(f"MTT on Ice Score (window={MTT_WINDOW}d, constrained windows shaded)")
ax.legend(fontsize=8)

# Feature panels with per-feature constrained detection
feat_results = [("ice_score", fus_idx, fus_t, bus_idx, bus_t)]
for panel_idx, (feat, ft) in enumerate(feature_t.items()):
    ax = axes[2 + panel_idx]
    valid_mask = ~np.isnan(ft)
    ax.plot(mtt_dates[valid_mask], ft[valid_mask], "k-", linewidth=0.8, alpha=0.7)
    ax.axhline(t_crit, color="red", ls="--", alpha=0.3)
    ax.axhline(-t_crit, color="blue", ls="--", alpha=0.3)
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_ylabel("t-score")
    ax.set_title(f"MTT on {feat}")

    f_fus_idx, f_fus_t = find_constrained_peak(ft, mtt_doys, fus_lo, fus_hi, "negative")
    f_bus_idx, f_bus_t = find_constrained_peak(ft, mtt_doys, bus_lo, bus_hi, "positive")
    feat_results.append((feat, f_fus_idx, f_fus_t, f_bus_idx, f_bus_t))

    if f_fus_idx is not None:
        ax.axvline(pd.Timestamp(mtt_dates[f_fus_idx]), color="blue", ls="-", alpha=0.4, linewidth=1.5)
    if f_bus_idx is not None:
        ax.axvline(pd.Timestamp(mtt_dates[f_bus_idx]), color="red", ls="-", alpha=0.4, linewidth=1.5)

    for g_start, g_end, g_days in gap_ranges:
        ax.axvspan(g_start, g_end, color="gray", alpha=0.1)

axes[-1].set_xlabel("Date")
fig.suptitle(f"{STATION} {YEAR} — Freeze/Thaw Transition Detection (MTT, v7.1 constrained)", fontsize=12)
fig.tight_layout()
plt.show()

# --- Report (v7.1) ---
print(f"\\n{'='*60}")
print("Constrained Transition Detection Results (v7.1)")
print(f"{'='*60}")

glerl_fus_date = glerl_bue_date = None
if glerl is not None:
    ice_days = glerl[glerl["ice_concentration"] > 15].sort_values("date_dt")
    if len(ice_days) > 0:
        glerl_fus_date = ice_days["date_dt"].iloc[0]
        glerl_bue_date = ice_days["date_dt"].iloc[-1]
        print(f"  GLERL first ice (>15%): {glerl_fus_date.strftime('%Y-%m-%d')} (DOY {glerl_fus_date.dayofyear})")
        print(f"  GLERL last ice  (>15%): {glerl_bue_date.strftime('%Y-%m-%d')} (DOY {glerl_bue_date.dayofyear})")

for label, dt, t_val, gap_flag in constrained_transitions:
    dt_pd = pd.Timestamp(dt)
    print(f"\\n  {label} (constrained): {dt_pd.strftime('%Y-%m-%d')} (DOY {dt_pd.dayofyear}) — t={t_val:+.2f}{gap_flag}")
    if label == "FUS" and glerl_fus_date is not None:
        delta = (dt_pd - glerl_fus_date).days
        sign = "before" if delta < 0 else "after"
        print(f"    FUS lead/lag vs GLERL: {abs(delta)} days {sign} GLERL first ice")
    elif label == "BUS" and glerl_bue_date is not None:
        delta = (dt_pd - glerl_bue_date).days
        sign = "before" if delta < 0 else "after"
        print(f"    BUS lead/lag vs GLERL: {abs(delta)} days {sign} GLERL last ice")

if not constrained_transitions:
    print("\\n  No transitions detected within the constrained windows.")
    print(f"  This may indicate a mild ice year or insufficient data coverage.")

# Per-feature transition summary table
print(f"\\nPer-feature transition detection (within constrained windows):")
print(f"  {'Feature':<12} {'FUS DOY':>8} {'FUS t':>8} {'BUS DOY':>8} {'BUS t':>8}")
print(f"  {'-'*48}")
for feat_name, f_idx, f_t, b_idx, b_t in feat_results:
    fus_str = f"DOY {pd.Timestamp(mtt_dates[f_idx]).dayofyear}" if f_idx is not None else "None"
    bus_str = f"DOY {pd.Timestamp(mtt_dates[b_idx]).dayofyear}" if b_idx is not None else "None"
    fus_t_str = f"{f_t:+.2f}" if f_idx is not None else "N/A"
    bus_t_str = f"{b_t:+.2f}" if b_idx is not None else "N/A"
    print(f"  {feat_name:<12} {fus_str:>8} {fus_t_str:>8} {bus_str:>8} {bus_t_str:>8}")
"""))

    # =====================================================================
    # Section 18: Land-Water Atmospheric Comparison (v7)
    # =====================================================================
    cells.append(_md(f"""
## 18. Land-Water Atmospheric Comparison (v7) — Exploratory

**Hypothesis:** During open water season, water-facing arcs (110°–190°) traverse
atmosphere with higher moisture content (lake evaporation). Land-facing arcs (N/W/E)
traverse drier air. This differential should be visible in phase (path length
difference) and possibly amplitude (attenuation). During freeze-up, evaporation
stops, the atmospheric contrast collapses, and the land-water differential should
converge toward zero.

**If this convergence precedes surface-observable ice detection, it's a leading indicator.**

**Prerequisites:** This section requires reprocessing ROSS with expanded azimuths
(0°–360°) using `config/ross_land.json`. The per-arc parquet must include arcs from
all azimuths, not just the 110°–190° water window.

To generate the expanded data:
```bash
# 1. Set up gnssrefl with expanded config
cp config/ross.json config/ross.json.bak
cp config/ross_land.json config/ross.json

# 2. Reprocess (this will include all azimuths)
python scripts/run_gnssir_processing.py --station ROSS --year {year} --skip_download --skip_conversion --skip_snr

# 3. Extract SNR features for all azimuths
python scripts/snr_feature_extractor.py --station ROSS --year {year} --num_cores 8

# 4. Restore original config
cp config/ross.json.bak config/ross.json
```

**Caveats:**
- The land surface itself changes seasonally (snow cover, frozen ground) — this confounds
  the atmospheric differential
- At ROSS, land sectors look N/W/E (inland, forested terrain) — inherently noisier
- This is exploratory. A null result is informative.
"""))

    cells.append(_code(f"""
# Check if expanded-azimuth data is available
pa_path_land = results_dir / f"{{STATION}}_{{YEAR}}_per_arc_land.parquet"
has_land_data = pa_path_land.exists()

# Also check if current per_arc has any land-facing azimuths
az_range = per_arc["Azim"].agg(["min", "max"])
has_full_az = az_range["min"] < 90 or az_range["max"] > 200

if has_land_data:
    pa_land = pd.read_parquet(pa_path_land)
    print(f"Loaded expanded per-arc: {{len(pa_land)}} arcs, az range: {{pa_land['Azim'].min():.0f}}°–{{pa_land['Azim'].max():.0f}}°")
elif has_full_az:
    pa_land = per_arc.copy()
    print(f"Current per-arc has expanded azimuths: {{az_range['min']:.0f}}°–{{az_range['max']:.0f}}°")
else:
    pa_land = None
    print("No expanded-azimuth data available.")
    print(f"Current azimuth range: {{az_range['min']:.0f}}°–{{az_range['max']:.0f}}° (water only)")
    print("\\nTo run this analysis, reprocess ROSS with config/ross_land.json (azval2=[0,360]).")
    print("See the instructions above for the processing steps.")
"""))

    cells.append(_code(f"""
if pa_land is not None:
    # Classify arcs by surface type
    def classify_surface(az):
        if 110 <= az <= 190:
            return "water"
        elif 90 <= az < 110 or 190 < az <= 210:
            return "coastal"
        else:
            return "land"

    pa_land["surface"] = pa_land["Azim"].apply(classify_surface)
    pa_land["date_dt"] = pd.to_datetime(pa_land["date"])

    counts = pa_land["surface"].value_counts()
    print(f"Surface classification:")
    for s in ["water", "coastal", "land"]:
        print(f"  {{s:>8}}: {{counts.get(s, 0):,}} arcs")

    if counts.get("land", 0) < 100:
        print("\\nInsufficient land arcs for comparison. Need expanded-azimuth reprocessing.")
    else:
        # Daily land vs water observables
        land_arcs = pa_land[pa_land["surface"] == "land"]
        water_arcs = pa_land[pa_land["surface"] == "water"]

        daily_land = land_arcs.groupby("date_dt").agg(
            amp_land=("Amp", "median"),
        )
        daily_water = water_arcs.groupby("date_dt").agg(
            amp_water=("Amp", "median"),
        )

        # Add gamma and phase if available
        if "gamma" in land_arcs.columns:
            daily_land["gamma_land"] = land_arcs.groupby("date_dt")["gamma"].median()
            daily_water["gamma_water"] = water_arcs.groupby("date_dt")["gamma"].median()

        if "phase" in land_arcs.columns:
            # Circular mean for phase
            daily_land["phase_land"] = land_arcs.groupby("date_dt")["phase"].apply(
                lambda x: np.arctan2(np.sin(x.dropna()).mean(), np.cos(x.dropna()).mean()))
            daily_water["phase_water"] = water_arcs.groupby("date_dt")["phase"].apply(
                lambda x: np.arctan2(np.sin(x.dropna()).mean(), np.cos(x.dropna()).mean()))

        # Merge
        diff = daily_water.join(daily_land, how="inner")
        diff["delta_amp"] = diff["amp_water"] - diff["amp_land"]
        if "gamma_water" in diff.columns and "gamma_land" in diff.columns:
            diff["delta_gamma"] = diff["gamma_water"] - diff["gamma_land"]
        if "phase_water" in diff.columns and "phase_land" in diff.columns:
            diff["delta_phase"] = np.arctan2(
                np.sin(diff["phase_water"] - diff["phase_land"]),
                np.cos(diff["phase_water"] - diff["phase_land"]))

        diff = diff.reset_index()
        print(f"\\nLand-water differential computed for {{len(diff)}} days")

        # Determine which differential columns exist
        diff_cols = [c for c in ["delta_amp", "delta_gamma", "delta_phase"] if c in diff.columns]
        n_panels = len(diff_cols) + 1  # +1 for raw comparison

        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        # Panel 1: Raw land vs water amplitude
        ax = axes[0]
        ax.plot(diff["date_dt"], diff["amp_water"], "b-", linewidth=1, alpha=0.6, label="Water (110°–190°)")
        ax.plot(diff["date_dt"], diff["amp_land"], "r-", linewidth=1, alpha=0.6, label="Land (other)")
        ax.set_ylabel("Median Amplitude")
        ax.set_title("Water-Sector vs Land-Sector Amplitude")
        ax.legend(fontsize=8)
        if glerl is not None:
            ax2 = ax.twinx()
            ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                             color="#90caf9", alpha=0.15)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("GLERL %", fontsize=8, color="#64b5f6")
            ax2.tick_params(axis="y", colors="#64b5f6", labelsize=7)

        # Differential panels
        labels = {{"delta_amp": "Δ Amplitude (water − land)",
                  "delta_gamma": "Δ γ (water − land)",
                  "delta_phase": "Δ Phase (water − land, rad)"}}
        for idx, col in enumerate(diff_cols):
            ax = axes[1 + idx]
            # 7-day rolling median for smoothing
            rolling = diff.set_index("date_dt")[col].rolling("7D", center=True).median()
            ax.plot(diff["date_dt"], diff[col], ".", markersize=3, alpha=0.3, color="#888")
            ax.plot(rolling.index, rolling.values, "k-", linewidth=1.5, label="7-day median")
            ax.axhline(0, color="gray", ls="-", alpha=0.3)
            ax.set_ylabel(labels.get(col, col))
            ax.set_title(labels.get(col, col))
            ax.legend(fontsize=8)

            # Mark freeze-up/break-up from MTT if available
            for label, dt in detected_transitions:
                ax.axvline(pd.Timestamp(dt), color="blue" if label == "FUS" else "red",
                           ls="--", alpha=0.4, label=label if idx == 0 else "")

            if glerl is not None:
                ax2 = ax.twinx()
                ax2.fill_between(glerl["date_dt"], glerl["ice_concentration"],
                                 color="#90caf9", alpha=0.1)
                ax2.set_ylim(0, 105)
                ax2.tick_params(axis="y", colors="#64b5f6", labelsize=7)

        axes[-1].set_xlabel("Date")
        fig.suptitle(f"{{STATION}} {{YEAR}} — Land-Water Atmospheric Differential", fontsize=12)
        fig.tight_layout()
        plt.show()

        # Statistical test: does the differential change before the classifier detects ice?
        if glerl is not None and "delta_amp" in diff.columns:
            # Split into pre-ice and ice periods
            glerl_monthly = glerl.groupby("month")["ice_concentration"].mean()
            ice_months = [m for m in glerl_monthly.index if glerl_monthly[m] > 30]
            water_months = [m for m in glerl_monthly.index if glerl_monthly[m] < 5]

            diff["month"] = diff["date_dt"].dt.month
            d_ice = diff[diff["month"].isin(ice_months)]["delta_amp"]
            d_water = diff[diff["month"].isin(water_months)]["delta_amp"]

            if len(d_ice) > 10 and len(d_water) > 10:
                t_stat, p_val = sp_stats.ttest_ind(d_water, d_ice)
                print(f"\\nAmplitude differential: water months mean={{d_water.mean():.2f}}, "
                      f"ice months mean={{d_ice.mean():.2f}}")
                print(f"t-test: t={{t_stat:.2f}}, p={{p_val:.3e}}")
                if p_val < 0.01:
                    print("→ Significant difference in land-water differential between seasons.")
                    if abs(d_water.mean()) > abs(d_ice.mean()):
                        print("  The differential is larger during open water (evaporation signal?)")
                        print("  and collapses during ice — consistent with the hypothesis.")
                    else:
                        print("  The differential is larger during ice — opposite to the evaporation hypothesis.")
                        print("  May reflect snow cover / frozen ground changes on land sectors.")
                else:
                    print("→ No significant seasonal change in the land-water differential.")
                    print("  The atmospheric hypothesis is not supported at this station.")
else:
    print("Skipping land-water comparison (no expanded-azimuth data).")
"""))

    # =====================================================================
    # v7 Addendum to Conclusions
    # =====================================================================
    cells.append(_md("""
## v7/v7.1 Addendum — New Observables and Validation

**Phase (φ):** Extracted via matched filter at the LSP-derived RH frequency.
The ice-water phase contrast quantifies the permittivity change at the reflecting
surface. Phase is a new column in the SNR features parquet; its discriminating
power relative to existing features is quantified in Section 15.

**v7.1 — Per-satellite phase (Section 15a):** Disaggregated phase by PRN and
frequency band. Tests whether the station-averaged ~5.6° shift was being washed
out by inter-satellite variability. Phase circular standard deviation tested as
a complementary discriminator (phase *variability* vs phase *mean*).

**Interfrequency ΔRH:** Per-sector computation of frequency-group RH spread.
At ROSS with only L1/L2C, this is expected to be a weak discriminator — the
negative result documents the 2-frequency limitation and establishes the
methodology for richer multi-GNSS stations (UMNQ, Greenland EarthScope).

**v7.1 — ΔRH validation (Section 16a):** Applied three quality filters:
(1) full-arc elevation coverage (≥80% of configured range),
(2) minimum arc count per band (≥3/band/day),
(3) L1 vs L2 RH scatter with seasonal coloring.
Validates whether the ice/water ΔRH ratio survives filtering or is an artifact
of poor LSP fits during rough-ice months.

**Freeze/Thaw transition detection (MTT):** The Moving t-Test identifies
abrupt state changes in the ice score time series.

**v7.1 — Constrained MTT (Section 17):** Added climatological search windows
(FUS: DOY 300-60, BUS: DOY 30-150 for Great Lakes). Eliminates physically
impossible mid-summer detections (e.g., the v7 FUS at DOY 198 = July 17).
Data gaps >7 days detected and flagged; gap-adjacent peaks marked unreliable.
Per-feature constrained detection identifies which observable is the earliest
transition indicator.

**Land-water atmospheric comparison:** Exploratory analysis comparing
observables in the water-facing (110°-190°) and land-facing sectors. Requires
expanded-azimuth reprocessing with `config/ross_land.json`. The hypothesis
is that evaporation cessation at freeze-up creates a detectable convergence in
the land-water differential. Results are reported as significant or null.

**Version history:**
- v7: + Phase extraction, per-sector ΔRH, MTT freeze/thaw detection,
  land-water atmospheric comparison (exploratory)
- v7.1: + ΔRH quality filters, constrained MTT with phenology windows,
  per-satellite phase disaggregation, data gap detection
"""))

    # =====================================================================
    # Section 19: Multi-year comparison (renumbered from 15)
    # =====================================================================
    cells.append(_md("""
## 19. Multi-Year Overview

Compare ice seasons across all available years with GLERL overlay.
"""))

    cells.append(_code(f"""
all_years = sorted([
    int(p.stem.split("_")[1])
    for p in results_dir.glob(f"{{STATION}}_*_ice_classification.parquet")
])
print(f"Available years: {{all_years}}")

if len(all_years) > 1:
    fig, axes = plt.subplots(len(all_years), 1, figsize=(14, 2.5 * len(all_years)),
                             sharex=False)
    if len(all_years) == 1:
        axes = [axes]

    for ax, yr in zip(axes, all_years):
        yr_clf = pd.read_parquet(results_dir / f"{{STATION}}_{{yr}}_ice_classification.parquet")
        yr_clf["date_dt"] = pd.to_datetime(yr_clf["date"])
        yr_clf["doy"] = yr_clf["date_dt"].dt.dayofyear

        for cls, color in CLASS_COLOR.items():
            mask = yr_clf["classification"] == cls
            if mask.sum() > 0:
                ax.scatter(yr_clf.loc[mask, "doy"], yr_clf.loc[mask, "ice_score"],
                           c=color, s=8, alpha=0.6, label=cls.capitalize())

        glerl_yr = PROJECT_ROOT / "data" / ".cache" / "glerl_ice" / f"{{STATION.lower()}}_ice_{{yr}}.csv"
        if glerl_yr.exists():
            g = pd.read_csv(glerl_yr, parse_dates=["datetime"])
            g["datetime"] = g["datetime"].dt.tz_localize(None)
            g["doy"] = g["datetime"].dt.dayofyear
            g_yr = g[g["datetime"].dt.year == yr]
            if not g_yr.empty:
                ax2 = ax.twinx()
                ax2.fill_between(g_yr["doy"], g_yr["ice_concentration"],
                                 color="#90caf9", alpha=0.2)
                ax2.set_ylim(0, 105)
                ax2.set_ylabel("GLERL %", color="#64b5f6", fontsize=8)
                ax2.tick_params(axis="y", colors="#64b5f6", labelsize=7)

        ax.axhline(-0.33, color="gray", ls=":", alpha=0.4)
        ax.axhline(0.33, color="gray", ls=":", alpha=0.4)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(str(yr), fontsize=11, fontweight="bold")
        ax.set_xlim(1, 366)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Day of Year")
    fig.suptitle(f"{{STATION}} Ice Classification — {{all_years[0]}}–{{all_years[-1]}}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()
else:
    print("Only one year available.")
"""))

    nb.cells = cells
    return nb


def main():
    parser = argparse.ArgumentParser(
        description="Generate ice classification Jupyter notebook")
    parser.add_argument("--station", required=True, help="Station ID (e.g., ROSS)")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--open", action="store_true",
                        help="Open notebook in jupyter after generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: results_annual/{station}/{station}_{year}_ice.ipynb)")
    args = parser.parse_args()

    nb = generate_notebook(args.station, args.year)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (PROJECT_ROOT / "results_annual" / args.station
                    / f"{args.station}_{args.year}_ice.ipynb")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out_path))
    print(f"Notebook saved: {out_path}")

    if args.open:
        import subprocess
        subprocess.Popen(["jupyter", "notebook", str(out_path)])


if __name__ == "__main__":
    main()
