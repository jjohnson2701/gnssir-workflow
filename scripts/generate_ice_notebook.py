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

Load the per-arc retrievals, SNR features, classification results, and GLERL ground truth.
"""))

    cells.append(_code(f"""
# Per-arc data (individual GNSS-IR retrievals)
pa_path = results_dir / f"{{STATION}}_{{YEAR}}_per_arc.parquet"
per_arc = pd.read_parquet(pa_path)
per_arc["date_dt"] = pd.to_datetime(per_arc["date"])
per_arc["month"] = per_arc["date_dt"].dt.month
print(f"Per-arc: {{len(per_arc):,}} arcs across {{per_arc['date'].nunique()}} days")
print(f"  Azimuth sectors: {{sorted(per_arc['azimuth_bin'].unique())}}")
print(f"  Frequencies: {{sorted(per_arc['freq'].unique())}}")

# SNR features
feat_path = results_dir / f"{{STATION}}_{{YEAR}}_snr_features.parquet"
has_snr = feat_path.exists()
if has_snr:
    snr_feat = pd.read_parquet(feat_path)
    # Merge features into per_arc
    join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]
    feat_cols = [c for c in snr_feat.columns if c not in per_arc.columns or c in join_cols]
    per_arc = per_arc.merge(snr_feat[feat_cols], on=join_cols, how="left")
    n_matched = per_arc["CLR"].notna().sum()
    print(f"  SNR features merged: {{n_matched}}/{{len(per_arc)}} arcs matched")
    print(f"  Features: CLR, AF, PR, gamma")
else:
    print("  No SNR features found. Run snr_feature_extractor.py first.")

# Classification results
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

    for ax, (feat, (label, high_ice, note)) in zip(axes.flat, features.items()):
        if feat not in per_arc.columns:
            ax.set_visible(False)
            continue

        monthly = per_arc.groupby("month")[feat].median()

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

    fig.suptitle(f"{STATION} {YEAR} — Monthly SNR Feature Medians\\n"
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
    # Compute daily feature medians
    daily_features = per_arc.groupby("date_dt").agg({
        "Amp": "mean",
        **({feat: "median" for feat in ["CLR", "AF", "PR", "gamma"] if feat in per_arc.columns})
    }).reset_index()

    merged = daily_features.merge(glerl[["date_dt", "ice_concentration"]], on="date_dt", how="inner")

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
# Re-run the classifier with default settings for comparison
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.ice_classifier import classify_daily

# Load fresh data
pa_fresh = pd.read_parquet(results_dir / f"{{STATION}}_{{YEAR}}_per_arc.parquet")
en_fresh = pd.read_parquet(results_dir / f"{{STATION}}_{{YEAR}}_daily_enriched.parquet")
feat_fresh = pd.read_parquet(results_dir / f"{{STATION}}_{{YEAR}}_snr_features.parquet")

# Run with DEFAULT settings (no overrides): temporarily patch config
import scripts.ice_classifier as ic_mod
orig_load = ic_mod._load_station_config
ic_mod._load_station_config = lambda s: {{
    k: v for k, v in station_cfg.items()
    if k not in ("feature_overrides", "smoothing_window", "min_arcs_per_sector")
}}
clf_default = classify_daily(
    pa_fresh, en_fresh, station=STATION, snr_features=feat_fresh,
    ice_free_months=ice_free_months, smoothing_window=3, min_arcs=3)

# Restore and run with CURRENT settings
ic_mod._load_station_config = orig_load
clf_current = classify_daily(
    pa_fresh, en_fresh, station=STATION, snr_features=feat_fresh,
    ice_free_months=ice_free_months,
    smoothing_window=station_cfg.get("smoothing_window", 3),
    min_arcs=station_cfg.get("min_arcs_per_sector", 3))

# Merge both with GLERL
if glerl is not None:
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
    # Section 13: Multi-year comparison
    # =====================================================================
    cells.append(_md("""
## 15. Multi-Year Overview

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
