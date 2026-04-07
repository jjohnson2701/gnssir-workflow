# ABOUTME: B4 geometry census internal consistency check.
# ABOUTME: Computes per-sector ice scores across the full year and checks inter-sector
# ABOUTME: disagreement as a function of DOY and v3 surface state. Uses geometry census
# ABOUTME: satellite pairs on 4 sample days to validate azimuth-pair agreement directly.

"""
B4: Sector-level classification consistency check.

For each day, independently scores each azimuth sector using corrected af_med
(z-scored against the ice-free summer baseline), then measures inter-sector
disagreement. High disagreement during transition states = spatial heterogeneity
(physically real). High disagreement during stable states = classifier noise.

Also cross-checks simultaneous satellite pairs from the geometry census (4 sample
days) to directly compare adjacent-azimuth sector scores at the same epoch.

Outputs (results_annual/UMNQ/diagnostics/):
  UMNQ_2025_b4_sector_consistency.parquet  — daily disagreement + v3 state
  UMNQ_2025_b4_consistency.png             — inter-sector std by DOY, colored by state
  UMNQ_2025_b4_census_pairs.png            — sector-pair score comparison (4 sample days)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_annual" / "UMNQ"
DIAG    = RESULTS / "diagnostics"
DIAG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
feat = pd.read_parquet(RESULTS / "UMNQ_2025_daily_features.parquet")
clf  = pd.read_parquet(RESULTS / "UMNQ_2025_ice_classification_v3.parquet")
gc   = pd.read_parquet(DIAG / "UMNQ_2025_geometry_census.parquet")

feat["date"] = pd.to_datetime(feat["date"])
clf["date"]  = pd.to_datetime(clf["date"])

# Sector rows only (exclude pooled azimuth_bin == -1)
sectors = feat[feat["azimuth_bin"] != -1].copy()

# ---------------------------------------------------------------------------
# Compute per-sector ice score from af_med
# Baseline: summer ice-free period DOY 181-240 (Jul-Aug 2025)
# Score = z-score of af_med relative to summer stats per sector
# ---------------------------------------------------------------------------
summer_doys = range(181, 241)
summer = sectors[sectors["date"].dt.dayofyear.isin(summer_doys)]
summer_stats = summer.groupby("azimuth_bin")["af_med"].agg(
    summer_mean="mean", summer_std="std"
).reset_index()
summer_stats["summer_std"] = summer_stats["summer_std"].replace(0, np.nan)

sectors = sectors.merge(summer_stats, on="azimuth_bin", how="left")
sectors["ice_score"] = (
    (sectors["af_med"] - sectors["summer_mean"]) / sectors["summer_std"]
)

# ---------------------------------------------------------------------------
# Daily inter-sector disagreement
# ---------------------------------------------------------------------------
daily_diag = (
    sectors.groupby("date")["ice_score"]
    .agg(
        sector_score_mean="mean",
        sector_score_std="std",
        sector_score_min="min",
        sector_score_max="max",
        n_sectors="count",
    )
    .reset_index()
)
daily_diag["sector_score_range"] = (
    daily_diag["sector_score_max"] - daily_diag["sector_score_min"]
)

# Merge v3 state
daily_diag = daily_diag.merge(clf[["date", "v3_state", "doy"]], on="date", how="left")

# Save
out_parquet = DIAG / "UMNQ_2025_b4_sector_consistency.parquet"
daily_diag.to_parquet(out_parquet, index=False)
print(f"Saved: {out_parquet} ({len(daily_diag)} rows)")

# Summary statistics
print("\n=== Inter-sector ice_score std by v3_state ===")
print(daily_diag.groupby("v3_state")["sector_score_std"].describe().round(3))

print("\n=== Inter-sector ice_score range by v3_state ===")
print(daily_diag.groupby("v3_state")["sector_score_range"].describe().round(3))

# ---------------------------------------------------------------------------
# Plot 1: inter-sector std by DOY, colored by state
# ---------------------------------------------------------------------------
state_colors = {
    "baseline":      "#2196F3",  # blue
    "anomalous":     "#F44336",  # red
    "regime_change": "#FF9800",  # orange
}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
for state, grp in daily_diag.groupby("v3_state"):
    ax.scatter(grp["doy"], grp["sector_score_std"],
               color=state_colors.get(state, "gray"),
               alpha=0.6, s=20, label=state)
ax.set_ylabel("Inter-sector std\n(af_med ice score)")
ax.set_title("UMNQ 2025 — Sector-level Classification Consistency (B4)\n"
             "Inter-sector disagreement by day and surface state")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(bottom=0)

# Moving median
dd_sorted = daily_diag.sort_values("doy")
ax.plot(dd_sorted["doy"],
        dd_sorted["sector_score_std"].rolling(14, center=True, min_periods=3).median(),
        "k-", lw=1.5, alpha=0.7, label="14-day median")
ax.legend(loc="upper right", fontsize=9)

ax2 = axes[1]
ax2.scatter(daily_diag["doy"], daily_diag["sector_score_mean"],
            c=[state_colors.get(s, "gray") for s in daily_diag["v3_state"]],
            alpha=0.6, s=20)
ax2.axhline(0, color="k", lw=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("Mean sector ice score\n(z-score vs summer)")
ax2.set_xlabel("DOY 2025")

patches = [mpatches.Patch(color=c, label=s) for s, c in state_colors.items()]
ax2.legend(handles=patches, loc="upper right", fontsize=9)

plt.tight_layout()
out1 = DIAG / "UMNQ_2025_b4_consistency.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2: geometry census satellite-pair sector comparison (4 sample days)
# ---------------------------------------------------------------------------
# Bin census azimuths to match feature azimuth_bins (40-180, step 10)
def az_to_bin(az):
    """Map azimuth to nearest 10-degree bin in [40, 180]."""
    b = round(az / 10) * 10
    return int(np.clip(b, 40, 180))

gc["bin_i"] = gc["az_i"].apply(az_to_bin)
gc["bin_j"] = gc["az_j"].apply(az_to_bin)

# Map doy → date (use 2025)
doy_to_date = {
    row["doy"]: row["date"]
    for _, row in clf[["doy", "date"]].iterrows()
}
gc["date"] = gc["doy"].map(doy_to_date)
gc["date"] = pd.to_datetime(gc["date"])

# Look up sector ice_score for each satellite in pair
score_lookup = sectors.set_index(["date", "azimuth_bin"])["ice_score"]

def lookup(date, az_bin):
    try:
        return score_lookup.loc[(date, az_bin)]
    except KeyError:
        return np.nan

gc["score_i"] = [lookup(d, b) for d, b in zip(gc["date"], gc["bin_i"])]
gc["score_j"] = [lookup(d, b) for d, b in zip(gc["date"], gc["bin_j"])]
gc["score_diff"] = np.abs(gc["score_i"] - gc["score_j"])
gc_valid = gc.dropna(subset=["score_i", "score_j"])

print(f"\n=== Geometry census: sector-pair score agreement ===")
print(f"Valid pairs: {len(gc_valid)} / {len(gc)}")

# Merge state for sample days
gc_valid = gc_valid.merge(clf[["date", "v3_state"]], on="date", how="left")
print(gc_valid.groupby(["doy", "v3_state"])["score_diff"].agg(["median", "mean", "count"]).round(3))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sample_doys = sorted(gc_valid["doy"].unique())
for ax, doy in zip(axes.flat, sample_doys):
    sub = gc_valid[gc_valid["doy"] == doy]
    state = sub["v3_state"].iloc[0] if len(sub) > 0 else "unknown"
    ax.scatter(sub["az_separation"], sub["score_diff"],
               alpha=0.3, s=8,
               color=state_colors.get(state, "gray"))
    ax.set_title(f"DOY {doy} ({state})\nmedian |Δscore|={sub['score_diff'].median():.2f}")
    ax.set_xlabel("Azimuth separation (deg)")
    ax.set_ylabel("|score_i − score_j|")
    ax.set_ylim(0, max(gc_valid["score_diff"].quantile(0.95), 0.1))

plt.suptitle("UMNQ 2025 — Geometry Census: Sector-pair Score Differences (B4)", y=1.01)
plt.tight_layout()
out2 = DIAG / "UMNQ_2025_b4_census_pairs.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Summary: high-disagreement days
# ---------------------------------------------------------------------------
threshold = daily_diag["sector_score_std"].quantile(0.90)
high_disagree = daily_diag[daily_diag["sector_score_std"] >= threshold].sort_values("doy")
print(f"\n=== Top 10% disagreement days (std >= {threshold:.2f}) ===")
print(f"State breakdown: {high_disagree['v3_state'].value_counts().to_dict()}")
print(high_disagree[["doy", "v3_state", "sector_score_std", "sector_score_range"]].to_string(index=False))
