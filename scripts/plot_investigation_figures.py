#!/usr/bin/env python3
"""Generate figures for the feature investigation log (Stories 1–5).

Story 1 — AF baseline domain fix:
  Fig 1a: Baseline power curve heatmaps (all-zero vs real)
  Fig 1b: AF distribution shift (pre-fix vs post-fix)
  Fig 1c: Per-PRN AF variance reduction

Story 2 — Gamma extraction hardening:
  Fig 2a: Full-arc guard impact (gamma and PkNoise, full vs partial arcs)
  Fig 2b: gamma vs gamma_r2 (damping value vs model fit quality)

Story 3 — Feature candidate evaluation (tested and ruled out):
  Fig 3a: Envelope ratio noise floor (theoretical vs real data)
  Fig 3b: Cross-frequency correlation by state (overlapping distributions)
  Fig 3c: Geometry census (simultaneous satellite pairs for spatial heterogeneity)

Story 4 — Post-fix feature reassessment:
  Fig 4a: Cohen's d comparison across features (ROSS vs UMNQ)
  Fig 4b: CLR blindness at UMNQ vs ROSS
  Fig 4c: Sector consistency (spring breakup vs autumn freeze-up)

Story 5 — Station-specific hardware artifact (GLBX L1 suppression):
  Fig 5a: Frequency-resolved seasonal amplitude at GLBX
  Fig 5b: Three-station comparison (GLBX vs UMNQ vs NIAQ)
  Fig 5c: DOY crossover (rolling Cohen's d for L1 vs pooled at UMNQ)

Usage:
    python scripts/plot_investigation_figures.py [--outdir docs/images/investigation]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results_annual")

# ── Constellation helpers ────────────────────────────────────────────

def sat_constellation(sat_id):
    if sat_id <= 32:
        return "GPS"
    elif 101 <= sat_id <= 132:
        return "GLONASS"
    elif 201 <= sat_id <= 250:
        return "Galileo"
    return "Other"

CONST_COLORS = {"GPS": "#1f77b4", "GLONASS": "#d62728", "Galileo": "#2ca02c"}
CONST_ORDER = ["GPS", "GLONASS", "Galileo"]


# ── Figure 1a: Baseline heatmaps ────────────────────────────────────

def fig1a_baselines(outdir):
    """Side-by-side heatmaps of the 196x50 baseline array: all-zero vs real."""
    old = np.load(RESULTS / "UMNQ/UMNQ_2025_af_baselines_old.npz", allow_pickle=True)
    new = np.load(RESULTS / "UMNQ/UMNQ_2025_af_baselines.npz", allow_pickle=True)

    bl_old = old["baselines"]  # 196 x 50, all zeros
    bl_new = new["baselines"]  # 196 x 50, real curves
    keys = new["keys"]         # (sat, freq) pairs
    sin_grid = new["sin_grid"]

    # Sort rows by constellation then sat number for visual grouping
    consts = np.array([sat_constellation(k[0]) for k in keys])
    order_map = {c: i for i, c in enumerate(CONST_ORDER)}
    sort_key = np.array([order_map.get(c, 9) * 1000 + keys[i][0] for i, c in enumerate(consts)])
    sort_idx = np.argsort(sort_key)

    bl_old_s = bl_old[sort_idx]
    bl_new_s = bl_new[sort_idx]
    consts_s = consts[sort_idx]

    # Constellation boundaries for tick labels
    boundaries = []
    for c in CONST_ORDER:
        idxs = np.where(consts_s == c)[0]
        if len(idxs):
            boundaries.append((c, idxs[0], idxs[-1]))

    elev_deg = np.degrees(np.arcsin(sin_grid))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # X-axis: satellite elevation angle during the arc (sin_grid back-converted to degrees).
    # Each row is one (satellite, frequency) combination.
    # Each streak = CWT power at that sat/freq combo's dominant reflector height,
    # measured as a function of elevation. This is the antenna gain pattern to be subtracted from AF.

    # Left: before fix (all zeros — subtraction was a silent no-op)
    ax1.imshow(bl_old_s, aspect="auto", cmap="inferno",
               extent=[elev_deg[0], elev_deg[-1], len(bl_old_s) - 0.5, -0.5],
               vmin=0, vmax=bl_new_s.max())
    ax1.set_title("Before fix: all baselines zero\n(Song 2022 correction was a no-op)", fontsize=11)
    ax1.set_xlabel("Satellite elevation angle (deg)\n[each streak = CWT power vs elevation for one sat/freq combo]",
                   fontsize=9)
    ax1.set_ylabel("Each row = one (satellite, frequency) combination\n"
                   "sorted by constellation then PRN number", fontsize=9)

    # Right: after fix (real per-PRN antenna gain curves)
    im = ax2.imshow(bl_new_s, aspect="auto", cmap="inferno",
                    extent=[elev_deg[0], elev_deg[-1], len(bl_new_s) - 0.5, -0.5],
                    vmin=0, vmax=bl_new_s.max())
    ax2.set_title("After fix: 196 antenna gain baselines recovered\n(per-PRN power curves, UMNQ 2025)", fontsize=11)
    ax2.set_xlabel("Satellite elevation angle (deg)\n[brighter = stronger antenna gain at that elevation]",
                   fontsize=9)

    # Constellation boundary lines + labels overlaid inside ax1
    # Number in parentheses = count of satellite×frequency combos in that constellation
    for c, start, end in boundaries:
        mid = (start + end) / 2
        for ax in (ax1, ax2):
            ax.axhline(start - 0.5, color="white", lw=0.5, alpha=0.6)
        ax1.text(0.02, mid, f"{c}  ({end - start + 1} combos)",
                 fontsize=8, ha="left", va="center",
                 transform=ax1.get_yaxis_transform(),
                 color=CONST_COLORS[c], fontweight="bold",
                 bbox=dict(fc="black", alpha=0.45, pad=1.5, lw=0, boxstyle="round,pad=0.2"))

    # Annotation in right panel explaining what the streaks are
    ax2.text(0.98, 0.02,
             "Each horizontal streak is the satellite's\n"
             "antenna gain pattern: how much CWT power\n"
             "the antenna alone produces at each elevation.\n"
             "Subtracting this isolates surface scattering (AF).",
             transform=ax2.transAxes, fontsize=7.5, ha="right", va="bottom",
             color="white", style="italic",
             bbox=dict(fc="black", alpha=0.55, pad=4, lw=0, boxstyle="round,pad=0.3"))

    # Manually placed colorbar — fixed position below panels, cannot overlap
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.18, wspace=0.04)
    cbar_ax = fig.add_axes([0.15, 0.07, 0.70, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_title("CWT power (arbitrary units) — brighter = stronger antenna gain at that elevation angle",
                      fontsize=8.5, color="white", pad=4)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")

    fig.suptitle("Fig 1a — AF Baseline Domain Bug: sin(ε) vs sin(ε)/cf Confusion",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.savefig(outdir / "fig1a_baselines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1a_baselines.png")


# ── Figure 1b: AF distribution shift ────────────────────────────────

def fig1b_af_shift(outdir):
    """AF distribution before and after baseline subtraction (full arcs only)."""
    cur = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table.parquet")
    bak = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table_backup.parquet")

    af_pre = bak.loc[bak["full_arc"] == True, "AF"].dropna()
    af_post = cur.loc[cur["full_arc"] == True, "AF"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlapping histograms
    ax = axes[0]
    bins = np.linspace(0, 120000, 80)
    ax.hist(af_pre, bins=bins, alpha=0.6, color="#d62728", label="Before fix (uncorrected)", density=True)
    ax.hist(af_post, bins=bins, alpha=0.6, color="#1f77b4", label="After fix (baseline-subtracted)", density=True)
    ax.axvline(af_pre.median(), color="#d62728", ls="--", lw=1.5, label=f"Pre-fix median: {af_pre.median():,.0f}")
    ax.axvline(af_post.median(), color="#1f77b4", ls="--", lw=1.5, label=f"Post-fix median: {af_post.median():,.0f}")
    ax.set_xlabel("Area Factor (CWT power integral)")
    ax.set_ylabel("Density")
    ax.set_title("AF distribution: full arcs, UMNQ 2025")
    ax.legend(fontsize=8)

    # Right: per-constellation violin
    ax2 = axes[1]
    bak["constellation"] = bak["sat"].apply(sat_constellation)
    cur["constellation"] = cur["sat"].apply(sat_constellation)

    positions = []
    labels = []
    violins_data = []
    colors = []
    for i, c in enumerate(CONST_ORDER):
        pre_c = bak.loc[(bak["full_arc"] == True) & (bak["constellation"] == c), "AF"].dropna()
        post_c = cur.loc[(cur["full_arc"] == True) & (cur["constellation"] == c), "AF"].dropna()
        if len(pre_c) > 10 and len(post_c) > 10:
            # Clip for visualization
            violins_data.append(np.clip(pre_c.values, 0, 150000))
            violins_data.append(np.clip(post_c.values, 0, 150000))
            positions.extend([i * 3, i * 3 + 1])
            labels.append(c)
            colors.extend(["#d62728", "#1f77b4"])

    vp = ax2.violinplot(violins_data, positions=positions, showmedians=True, widths=0.8)
    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[j])
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")

    ax2.set_xticks([i * 3 + 0.5 for i in range(len(labels))])
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Area Factor")
    ax2.set_title("AF by constellation (red = before, blue = after)")

    drop_pct = (1 - af_post.median() / af_pre.median()) * 100
    fig.suptitle(f"Fig 1b — AF Median Drops {drop_pct:.0f}% After Antenna Gain Removal",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "fig1b_af_shift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1b_af_shift.png")


# ── Figure 1c: Per-PRN AF variance reduction ────────────────────────

def fig1c_prn_variance(outdir):
    """Scatter of per-PRN AF std: pre-fix vs post-fix."""
    cur = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table.parquet")
    bak = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table_backup.parquet")

    # Group by (sat, freq_group) for both
    grp_pre = bak.loc[bak["full_arc"] == True].groupby(["sat", "freq_group"])["AF"]
    grp_post = cur.loc[cur["full_arc"] == True].groupby(["sat", "freq_group"])["AF"]

    stats_pre = grp_pre.agg(["std", "median", "count"]).rename(columns={"std": "std_pre", "median": "med_pre", "count": "n_pre"})
    stats_post = grp_post.agg(["std", "median", "count"]).rename(columns={"std": "std_post", "median": "med_post", "count": "n_post"})
    merged = stats_pre.join(stats_post, how="inner")
    merged = merged[merged["n_pre"] >= 20]  # meaningful sample

    merged = merged.reset_index()
    merged["constellation"] = merged["sat"].apply(sat_constellation)

    fig, ax = plt.subplots(figsize=(8, 8))

    for c in CONST_ORDER:
        mask = merged["constellation"] == c
        sub = merged[mask]
        # x = after fix, y = before fix — improvement = points above diagonal
        ax.scatter(sub["std_post"], sub["std_pre"],
                   c=CONST_COLORS[c], label=f"{c} ({len(sub)} combos)",
                   alpha=0.6, s=30, edgecolors="none")

    # Diagonal (no-change line)
    lim = max(merged["std_pre"].max(), merged["std_post"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="No change (equal std)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # Shade the improvement region
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="#2ca02c")
    ax.text(lim * 0.55, lim * 0.92, "← variance reduced\n   (fix helped)",
            fontsize=8, color="#2ca02c", style="italic")

    # Aggregate reduction
    mean_pre = merged["std_pre"].mean()
    mean_post = merged["std_post"].mean()
    reduction = (1 - mean_post / mean_pre) * 100
    above = (merged["std_pre"] > merged["std_post"]).sum()
    total = len(merged)

    # x = after-fix, y = before-fix
    ax.set_xlabel("Per-PRN AF standard deviation  (after fix — corrected AF)", fontsize=11)
    ax.set_ylabel("Per-PRN AF standard deviation  (before fix — raw AF with antenna gain)", fontsize=11)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

    ax.text(0.97, 0.03,
            f"{above}/{total} combos above diagonal\n"
            f"Mean AF σ reduction: {reduction:.1f}%\n"
            f"Points above line = antenna gain removed",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

    ax.set_title("Fig 1c — Per-PRN AF Variance: Baseline Subtraction Removes Satellite Bias\n"
                 "(each point = one satellite × frequency combo; above diagonal = improvement)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig1c_prn_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1c_prn_variance.png")


# ── Figure 2a: Full-arc guard impact ────────────────────────────────

def fig2a_arc_guard(outdir):
    """Gamma and PkNoise distributions: full vs partial arcs."""
    bak = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table_backup.parquet")

    full = bak[bak["full_arc"] == True]
    part = bak[bak["full_arc"] == False]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: gamma distributions
    ax = axes[0]
    bins_g = np.linspace(0, 0.025, 60)
    ax.hist(full["gamma"].dropna(), bins=bins_g, alpha=0.6, color="#1f77b4",
            density=True, label=f"Full arcs (n={len(full):,})")
    ax.hist(part["gamma"].dropna(), bins=bins_g, alpha=0.6, color="#ff7f0e",
            density=True, label=f"Partial arcs (n={len(part):,})")
    ax.axvline(full["gamma"].median(), color="#1f77b4", ls="--", lw=1.5)
    ax.axvline(part["gamma"].median(), color="#ff7f0e", ls="--", lw=1.5)
    ax.set_xlabel("γ  (damping coefficient, Strandberg 2017)")
    ax.set_ylabel("Density")
    ax.set_title("γ: damping rate of fringe amplitude\nwith elevation angle")
    ax.legend(fontsize=8)

    # Panel 2: PkNoise distributions
    ax2 = axes[1]
    bins_p = np.linspace(0, 30, 60)
    ax2.hist(full["PkNoise"].dropna(), bins=bins_p, alpha=0.6, color="#1f77b4",
             density=True, label=f"Full arcs")
    ax2.hist(part["PkNoise"].dropna(), bins=bins_p, alpha=0.6, color="#ff7f0e",
             density=True, label=f"Partial arcs")
    ax2.axvline(full["PkNoise"].median(), color="#1f77b4", ls="--", lw=1.5)
    ax2.axvline(part["PkNoise"].median(), color="#ff7f0e", ls="--", lw=1.5)
    pk_drop = (1 - part["PkNoise"].median() / full["PkNoise"].median()) * 100
    ax2.set_xlabel("PkNoise (peak-to-noise ratio)")
    ax2.set_title(f"PkNoise: partial arcs {pk_drop:.0f}% lower\n(weaker spectral peak)")
    ax2.legend(fontsize=8)

    # Panel 3: elevation coverage
    ax3 = axes[2]
    bak["ele_range"] = bak["emaxO"] - bak["eminO"]
    expected = 20.0  # e2 - e1 for UMNQ (5-25 deg)
    bak["coverage"] = bak["ele_range"] / expected
    bins_c = np.linspace(0, 1.2, 50)
    ax3.hist(bak.loc[bak["full_arc"] == True, "coverage"], bins=bins_c, alpha=0.6,
             color="#1f77b4", density=True, label="Full arcs")
    ax3.hist(bak.loc[bak["full_arc"] == False, "coverage"], bins=bins_c, alpha=0.6,
             color="#ff7f0e", density=True, label="Partial arcs")
    ax3.axvline(0.8, color="black", ls=":", lw=1.5, label="80% threshold (Song 2022)")
    ax3.set_xlabel("Elevation coverage fraction")
    ax3.set_title(f"Coverage gate: {len(part):,} arcs ({len(part)/len(bak)*100:.1f}%)\nexcluded from γ and AF")
    ax3.legend(fontsize=8)

    fig.suptitle("Fig 2a — Truncated Arc Guard: Partial Arcs Add Noise, Not Signal (UMNQ 2025)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig2a_arc_guard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2a_arc_guard.png")


# ── Figure 2b: gamma vs gamma_r2 ────────────────────────────────────

def fig2b_gamma_r2(outdir):
    """gamma (the physical quantity) vs gamma_r2 (how much to trust it).

    Three panels:
      Left:   gamma_r2 distributions at UMNQ vs ROSS — shows the Strandberg
              model fits Arctic fjord ice but not rough Great Lakes ice.
      Center: gamma vs gamma_r2 scatter at UMNQ, colored by DOY — shows which
              arcs have trustworthy gamma values.
      Right:  gamma distributions split by gamma_r2 quality band — shows how
              the gamma distribution sharpens when you only trust high-R2 arcs.
    """
    umnq = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table.parquet")
    ross = pd.read_parquet(RESULTS / "ROSS/ROSS_2024_arc_table.parquet")

    umnq_full = umnq[umnq["full_arc"] == True].copy()
    ross_full = ross[ross["full_arc"] == True].copy()

    fig = plt.figure(figsize=(16, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.3)

    # ── Panel 1: gamma_r2 distributions ──
    ax1 = fig.add_subplot(gs[0])
    bins_r2 = np.linspace(0, 0.9, 50)
    ax1.hist(umnq_full["gamma_r2"], bins=bins_r2, alpha=0.6, color="#2ca02c",
             density=True, label=f"UMNQ 2025\nmedian = {umnq_full['gamma_r2'].median():.3f}")
    ax1.hist(ross_full["gamma_r2"], bins=bins_r2, alpha=0.6, color="#9467bd",
             density=True, label=f"ROSS 2024\nmedian = {ross_full['gamma_r2'].median():.3f}")
    ax1.set_xlabel("γ R²  (fit quality: does the Strandberg model apply?)", fontsize=10)
    ax1.set_ylabel("Density")
    ax1.set_title("γ R² tells you whether γ is\nmeaningful at this station", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.text(0.5, 0.55, "ROSS: smooth-surface\nmodel fails on\nrough Great Lakes ice",
             transform=ax1.transAxes, fontsize=8, ha="center", style="italic",
             color="#9467bd")

    # ── Panel 2: gamma vs gamma_r2 scatter ──
    ax2 = fig.add_subplot(gs[1])
    # Subsample for scatter readability
    rng = np.random.default_rng(42)
    n_plot = min(8000, len(umnq_full))
    idx = rng.choice(len(umnq_full), n_plot, replace=False)
    sub = umnq_full.iloc[idx]

    sc = ax2.scatter(sub["gamma_r2"], sub["gamma"], c=sub["doy"],
                     cmap="twilight_shifted", s=4, alpha=0.4, edgecolors="none",
                     vmin=1, vmax=365)
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("Day of year", fontsize=9)

    # Annotate regions
    ax2.axvline(0.1, color="black", ls=":", lw=1, alpha=0.6)
    ax2.text(0.03, 0.022, "γ R² < 0.1\nγ unreliable", fontsize=8,
             ha="center", va="bottom", style="italic", color="#888")
    ax2.text(0.5, 0.022, "γ R² > 0.1\nγ trustworthy", fontsize=8,
             ha="center", va="bottom", style="italic", color="#222")

    ax2.set_xlabel("γ R²  (model fit quality)", fontsize=10)
    ax2.set_ylabel("γ  (damping coefficient)", fontsize=10)
    ax2.set_title("γ is the value; γ R² says\nhow much to trust it (UMNQ 2025)", fontsize=11)
    ax2.set_xlim(-0.02, 0.9)
    ax2.set_ylim(0, 0.027)

    # ── Panel 3: gamma distributions by R2 band ──
    ax3 = fig.add_subplot(gs[2])
    bands = [
        (0.0, 0.05, "R² < 0.05\n(model fails)", "#d62728"),
        (0.05, 0.2, "R² 0.05–0.2\n(marginal fit)", "#ff7f0e"),
        (0.2, 1.0, "R² > 0.2\n(good fit)", "#2ca02c"),
    ]
    bins_g = np.linspace(0, 0.025, 50)
    for lo, hi, label, color in bands:
        mask = (umnq_full["gamma_r2"] >= lo) & (umnq_full["gamma_r2"] < hi)
        vals = umnq_full.loc[mask, "gamma"].dropna()
        ax3.hist(vals, bins=bins_g, alpha=0.5, color=color, density=True,
                 label=f"{label}  (n={len(vals):,})")

    ax3.set_xlabel("γ  (damping coefficient)", fontsize=10)
    ax3.set_ylabel("Density")
    ax3.set_title("Higher R² → sharper γ distribution\n→ more informative damping estimate", fontsize=11)
    ax3.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Fig 2b — γ (Damping) vs γ R² (Fit Quality): Two Different Quantities",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(outdir / "fig2b_gamma_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2b_gamma_r2.png")


# ── Helpers ──────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def load_daily_with_state(station, year):
    """Load daily_features (pooled rows) merged with v3 state labels."""
    df = pd.read_parquet(RESULTS / f"{station}/{station}_{year}_daily_features.parquet")
    df = df[df["azimuth_bin"] == -1].copy()
    v3 = pd.read_parquet(RESULTS / f"{station}/{station}_{year}_ice_classification_v3.parquet")
    v3 = v3[["date", "v3_state"]].copy()
    # Normalize date types for merge
    df["date"] = pd.to_datetime(df["date"])
    v3["date"] = pd.to_datetime(v3["date"])
    return df.merge(v3, on="date", how="inner")


# ── Figure 3a: Envelope ratio noise floor ────────────────────────────

def fig3a_envelope_ratio(outdir):
    """Theoretical envelope ratio signal vs real data noise floor."""
    er = pd.read_parquet(RESULTS / "UMNQ/diagnostics/UMNQ_2025_envelope_ratio.parquet")

    # Synthetic model results from the log (two-layer Fresnel, Table in §8)
    snow_cm = np.array([0, 2, 5, 10, 15, 20])
    slope_L1_L2C = np.array([0.010, -0.013, -0.072, -0.242, -0.619, -1.164])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: synthetic signal
    ax = axes[0]
    ax.plot(snow_cm, slope_L1_L2C, "o-", color="#1f77b4", lw=2, ms=8, zorder=5)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Snow depth (cm)", fontsize=11)
    ax.set_ylabel("L1/L2C envelope ratio slope", fontsize=11)
    ax.set_title("Synthetic two-layer Fresnel model\n(air → snow → ice)", fontsize=11)

    # Shade the real-data noise floor for context
    real_std = er["env_ratio_slope"].std()
    ax.axhspan(-real_std, real_std, color="#d62728", alpha=0.12, zorder=1)
    ax.text(10, real_std * 0.6, f"Real data noise floor\n(1σ = ±{real_std:.1f})",
            fontsize=9, color="#d62728", ha="center")

    ax.set_ylim(-2, real_std * 1.3)
    ax.text(15, -0.8, "Detectable signal:\nslope < −0.07 (≥5 cm snow)",
            fontsize=8, style="italic", color="#1f77b4")

    # Right: real data distribution
    ax2 = axes[1]
    for state, color, label in [("baseline", "#1f77b4", "Baseline (open water)"),
                                 ("anomalous", "#d62728", "Anomalous (ice)"),
                                 ("regime_change", "#ff7f0e", "Regime change")]:
        vals = er.loc[er["v3_state"] == state, "env_ratio_slope"]
        ax2.hist(vals, bins=60, alpha=0.5, color=color, density=True, label=f"{label} (n={len(vals):,})")

    # Mark the theoretical signals
    for sc, sl in zip([5, 10, 20], [-0.072, -0.242, -1.164]):
        ax2.axvline(sl, color="#2ca02c", ls="--", lw=1, alpha=0.7)
        ax2.text(sl, ax2.get_ylim()[1] * 0.92, f"{sc}cm", fontsize=7,
                 ha="center", color="#2ca02c", rotation=90)

    ax2.set_xlabel("L1/L2C envelope ratio slope", fontsize=11)
    ax2.set_ylabel("Density")
    ax2.set_xlim(-80, 80)
    ax2.set_title(f"Real data: std = {real_std:.1f}\n(theoretical 5cm signal = −0.07 → {real_std/0.072:.0f}× below noise)",
                  fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle("Fig 3a — Envelope Ratio: Theoretical Signal Buried Under Real-Data Noise (UMNQ 2025)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig3a_envelope_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3a_envelope_ratio.png")


# ── Figure 3b: Cross-frequency correlation by state ──────────────────

def fig3b_xfreq_correlation(outdir):
    """Boxplots of cross-freq correlation for ice vs water — overlapping distributions."""
    xf = pd.read_parquet(RESULTS / "UMNQ/diagnostics/UMNQ_2025_xfreq_corr.parquet")

    # Merge v3 state
    v3 = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_ice_classification_v3.parquet")
    v3["date"] = pd.to_datetime(v3["date"])
    v3["doy"] = v3["date"].dt.dayofyear
    xf = xf.merge(v3[["doy", "v3_state"]], on="doy", how="left")
    xf = xf.dropna(subset=["v3_state"])

    # Select well-behaved pairs (positive median)
    good_pairs = ["GPS_L1_L2C", "GPS_L1_L5", "GLO_L1_L2", "GAL_L1_L5a"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    positions = []
    tick_labels = []
    all_boxes = []
    state_colors = {"baseline": "#1f77b4", "anomalous": "#d62728", "regime_change": "#ff7f0e"}

    for i, pair in enumerate(good_pairs):
        sub = xf[xf["pair"] == pair]
        for j, (state, color) in enumerate(state_colors.items()):
            vals = sub.loc[sub["v3_state"] == state, "xfreq_corr"].dropna().values
            if len(vals) < 5:
                continue
            pos = i * 4 + j
            bp = ax.boxplot([vals], positions=[pos], widths=0.7, patch_artist=True,
                           showfliers=False, medianprops=dict(color="black", lw=1.5))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.6)
            positions.append(pos)
            if j == 1:  # center label on middle box
                tick_labels.append(pair.replace("_", " "))

    ax.set_xticks([i * 4 + 1 for i in range(len(good_pairs))])
    ax.set_xticklabels([p.replace("_", " ") for p in good_pairs], fontsize=9)
    ax.set_ylabel("Cross-frequency Pearson r", fontsize=11)
    ax.axhline(0, color="gray", ls=":", lw=0.8)

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, alpha=0.6, label=s.capitalize())
                      for s, c in state_colors.items()]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # Annotate median shifts
    for i, pair in enumerate(good_pairs):
        sub = xf[xf["pair"] == pair]
        med_b = sub.loc[sub["v3_state"] == "baseline", "xfreq_corr"].median()
        med_a = sub.loc[sub["v3_state"] == "anomalous", "xfreq_corr"].median()
        diff = med_a - med_b
        ax.text(i * 4 + 1, -0.42, f"Δmedian = {diff:+.3f}",
                ha="center", fontsize=8, color="#666")

    ax.set_title("Fig 3b — Cross-Frequency Correlation: Median Shifts Are Real but Tiny (UMNQ 2025)",
                 fontsize=13, fontweight="bold")
    ax.text(0.02, 0.02, "IQR ~0.3–0.7 in both states; Δmedian 0.02–0.09 → not usable for per-arc classification",
            transform=ax.transAxes, fontsize=9, style="italic", color="#666")
    fig.tight_layout()
    fig.savefig(outdir / "fig3b_xfreq_corr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3b_xfreq_corr.png")


# ── Figure 3c: Geometry census ───────────────────────────────────────

def fig3c_geometry_census(outdir):
    """Polar scatter of simultaneous satellite pairs on sample days."""
    gc = pd.read_parquet(RESULTS / "UMNQ/diagnostics/UMNQ_2025_geometry_census.parquet")

    sample_doys = sorted(gc["doy"].unique())  # Should be 4 sample days
    n_days = len(sample_doys)

    fig, axes = plt.subplots(1, n_days, figsize=(4 * n_days, 4.5),
                             subplot_kw=dict(projection="polar"))
    if n_days == 1:
        axes = [axes]

    season_names = {15: "Winter (Jan 15)", 105: "Spring (Apr 15)",
                    120: "Spring (Apr 30)", 190: "Summer (Jul 9)",
                    195: "Summer (Jul 14)", 285: "Autumn (Oct 12)"}

    for ax, doy in zip(axes, sample_doys):
        sub = gc[gc["doy"] == doy]
        az_rad = np.radians(sub["az_i"])
        elev = sub["mean_elev"]
        sep = sub["az_separation"]

        sc = ax.scatter(az_rad, elev, c=sep, cmap="viridis", s=6,
                        alpha=0.5, edgecolors="none", vmin=30, vmax=180)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(4, 16)
        ax.set_rticks([6, 10, 14])
        ax.set_rlabel_position(45)
        name = season_names.get(doy, f"DOY {doy}")
        ax.set_title(f"{name}\n{len(sub):,} pairs", fontsize=9, pad=12)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.08, aspect=30)
    cbar.set_label("Azimuth separation (deg)", fontsize=9)

    fig.suptitle("Fig 3c — Geometry Census: Abundant Simultaneous Multi-Azimuth Coverage (UMNQ 2025)",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.savefig(outdir / "fig3c_geometry_census.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3c_geometry_census.png")


# ── Figure 4a: Cohen's d comparison ──────────────────────────────────

def fig4a_cohens_d(outdir):
    """Side-by-side Cohen's d for all features at ROSS vs UMNQ."""
    ross = load_daily_with_state("ROSS", 2024)
    umnq = load_daily_with_state("UMNQ", 2025)

    # Normalize state labels: ROSS uses validated names, UMNQ uses discovery
    # Map to binary: ice-like vs water-like
    ross_ice_states = {"ice_surface", "ice_decaying", "ice_layered", "freeze_up", "break_up"}
    ross["is_ice"] = ross["v3_state"].isin(ross_ice_states)
    umnq["is_ice"] = umnq["v3_state"] == "anomalous"

    features = [
        ("af_med", "AF (corrected)"),
        ("clr_med", "CLR (clarity)"),
        ("amp_mean", "Amplitude"),
        ("p2n_mean", "PkNoise"),
        ("gamma_med", "γ (damping)"),
        ("rh_std", "RH std dev"),
        ("pr_med", "Peak ratio"),
        ("vs_med", "SNR variance"),
        ("ms_med", "Mean SNR"),
    ]

    d_ross = []
    d_umnq = []
    labels = []
    for col, label in features:
        if col not in ross.columns or col not in umnq.columns:
            continue
        r_ice = ross.loc[ross["is_ice"], col].dropna()
        r_wat = ross.loc[~ross["is_ice"], col].dropna()
        u_ice = umnq.loc[umnq["is_ice"], col].dropna()
        u_wat = umnq.loc[~umnq["is_ice"], col].dropna()
        d_ross.append(cohens_d(r_ice, r_wat))
        d_umnq.append(cohens_d(u_ice, u_wat))
        labels.append(label)

    d_ross = np.array(d_ross)
    d_umnq = np.array(d_umnq)

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y + h/2, np.abs(d_ross), h, color="#9467bd", alpha=0.7, label="ROSS 2024 (Great Lakes)")
    ax.barh(y - h/2, np.abs(d_umnq), h, color="#2ca02c", alpha=0.7, label="UMNQ 2025 (Greenland)")

    # Highlight CLR blindness
    clr_idx = labels.index("CLR (clarity)") if "CLR (clarity)" in labels else None
    if clr_idx is not None:
        ax.barh(clr_idx - h/2, np.abs(d_umnq[clr_idx]), h, color="#d62728", alpha=0.8)
        ax.annotate("CLR blind at UMNQ\n(d = 0.06)",
                    xy=(np.abs(d_umnq[clr_idx]) + 0.05, clr_idx - h/2),
                    fontsize=8, color="#d62728", fontweight="bold",
                    va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("|Cohen's d|  (ice vs water daily features, pooled rows)", fontsize=11)
    ax.axvline(0.2, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(0.8, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.2, len(labels) - 0.3, "small", fontsize=7, color="gray", ha="center")
    ax.text(0.8, len(labels) - 0.3, "large", fontsize=7, color="gray", ha="center")
    ax.legend(loc="lower right", fontsize=10)
    ax.invert_yaxis()

    ax.set_title("Fig 4a — Feature Discriminability: Same Features, Different Physics (ROSS vs UMNQ)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig4a_cohens_d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4a_cohens_d.png")


# ── Figure 4b: CLR blindness ────────────────────────────────────────

def fig4b_clr_blind(outdir):
    """CLR distributions at ROSS (separated) vs UMNQ (overlapping)."""
    ross = load_daily_with_state("ROSS", 2024)
    umnq = load_daily_with_state("UMNQ", 2025)

    ross_ice_states = {"ice_surface", "ice_decaying", "ice_layered", "freeze_up", "break_up"}
    ross["is_ice"] = ross["v3_state"].isin(ross_ice_states)
    umnq["is_ice"] = umnq["v3_state"] == "anomalous"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # ROSS
    bins = np.linspace(0, 15, 50)
    r_ice = ross.loc[ross["is_ice"], "clr_med"].dropna()
    r_wat = ross.loc[~ross["is_ice"], "clr_med"].dropna()
    ax1.hist(r_wat, bins=bins, alpha=0.6, color="#1f77b4", density=True, label=f"Open water (n={len(r_wat)})")
    ax1.hist(r_ice, bins=bins, alpha=0.6, color="#d62728", density=True, label=f"Ice states (n={len(r_ice)})")
    d_r = cohens_d(r_ice, r_wat)
    ax1.set_title(f"ROSS 2024 (Great Lakes)\nCohen's d = {d_r:.2f} — CLR discriminates", fontsize=11)
    ax1.set_xlabel("CLR (clarity ratio: P1 / mean other peaks)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)

    # UMNQ
    u_ice = umnq.loc[umnq["is_ice"], "clr_med"].dropna()
    u_wat = umnq.loc[~umnq["is_ice"], "clr_med"].dropna()
    ax2.hist(u_wat, bins=bins, alpha=0.6, color="#1f77b4", density=True, label=f"Baseline (n={len(u_wat)})")
    ax2.hist(u_ice, bins=bins, alpha=0.6, color="#d62728", density=True, label=f"Anomalous (n={len(u_ice)})")
    d_u = cohens_d(u_ice, u_wat)
    ax2.set_title(f"UMNQ 2025 (Greenland, 70.7°N)\nCohen's d = {d_u:.2f} — CLR blind", fontsize=11)
    ax2.set_xlabel("CLR (clarity ratio: P1 / mean other peaks)")
    ax2.legend(fontsize=9)

    ax2.text(0.97, 0.65,
             "At 70.7°N all satellites track\nthrough a narrow elevation band.\n"
             "LSP always has one dominant\npeak regardless of surface state\n"
             "→ CLR ≈ constant → useless.",
             transform=ax2.transAxes, fontsize=8, ha="right", va="top",
             style="italic", color="#666",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    fig.suptitle("Fig 4b — CLR Blindness: High-Latitude Geometry Kills Spectral Clarity Discrimination",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig4b_clr_blind.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4b_clr_blind.png")


# ── Figure 4c: Sector consistency ────────────────────────────────────

def fig4c_sector_consistency(outdir):
    """Inter-sector disagreement: spring breakup (patchy) vs autumn freeze-up (uniform)."""
    b4 = pd.read_parquet(RESULTS / "UMNQ/diagnostics/UMNQ_2025_b4_sector_consistency.parquet")

    state_colors = {"baseline": "#1f77b4", "anomalous": "#d62728", "regime_change": "#ff7f0e"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: time series of inter-sector std
    for state, color in state_colors.items():
        mask = b4["v3_state"] == state
        ax1.scatter(b4.loc[mask, "doy"], b4.loc[mask, "sector_score_std"],
                    c=color, s=15, alpha=0.7, label=state.capitalize(), edgecolors="none")

    # Highlight spring breakup window
    ax1.axvspan(85, 117, alpha=0.1, color="#d62728")
    ax1.text(101, b4["sector_score_std"].max() * 0.95, "Spring breakup\n(patchy ice)",
             ha="center", fontsize=9, color="#d62728", fontweight="bold")

    # Highlight autumn freeze-up
    ax1.axvspan(300, 366, alpha=0.1, color="#ff7f0e")
    ax1.text(333, b4["sector_score_std"].max() * 0.95, "Autumn freeze-up\n(spatially uniform)",
             ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")

    # Noise floor
    baseline_std = b4.loc[b4["v3_state"] == "baseline", "sector_score_std"].median()
    ax1.axhline(baseline_std, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.text(365, baseline_std + 0.1, f"Baseline noise floor = {baseline_std:.2f}",
             ha="right", fontsize=8, color="gray")

    ax1.set_ylabel("Inter-sector AF score σ\n(higher = sectors disagree)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)

    # Bottom: state timeline
    for _, row in b4.iterrows():
        color = state_colors.get(row["v3_state"], "gray")
        ax2.barh(0, 1, left=row["doy"], color=color, height=0.6, edgecolor="none")
    ax2.set_yticks([])
    ax2.set_xlabel("Day of year (UMNQ 2025)", fontsize=10)
    ax2.set_ylabel("State", fontsize=9)
    ax2.set_xlim(1, 366)

    fig.suptitle("Fig 4c — Sector Consistency: Spring Breakup Is Patchy, Autumn Freeze-Up Is Uniform",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "fig4c_sector_consistency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4c_sector_consistency.png")


# ── Figure 5a: GLBX frequency-resolved amplitude ────────────────────

def fig5a_glbx_amplitude(outdir):
    """Monthly L1 vs L5 vs pooled amplitude at GLBX — L1 collapse in winter."""
    df = pd.read_parquet(RESULTS / "GLBX/GLBX_2024_daily_features.parquet")
    df = df[df["azimuth_bin"] == -1].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    monthly = df.groupby("month").agg(
        amp_mean=("amp_mean", "median"),
        amp_L1=("amp_L1_mean", "median"),
        amp_L5=("amp_L5_mean", "median"),
        ms_med=("ms_med", "median"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: amplitude by band
    ax1.plot(monthly["month"], monthly["amp_L1"], "o-", color="#1f77b4", lw=2, ms=7, label="L1 amplitude")
    ax1.plot(monthly["month"], monthly["amp_L5"], "s-", color="#2ca02c", lw=2, ms=7, label="L5 amplitude")
    ax1.plot(monthly["month"], monthly["amp_mean"], "D-", color="#666", lw=2, ms=6, label="Pooled amp_mean")
    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_ylabel("Amplitude (monthly median)", fontsize=11)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax1.legend(fontsize=9)

    # Annotate the collapse
    ax1.annotate("L1 collapses 67%\nin winter",
                 xy=(1, monthly.loc[monthly["month"] == 1, "amp_L1"].values[0]),
                 xytext=(3, 15), fontsize=9, color="#1f77b4", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax1.annotate("L5 is HIGHER\nin winter (ice = more reflective)",
                 xy=(1, monthly.loc[monthly["month"] == 1, "amp_L5"].values[0]),
                 xytext=(4, 50), fontsize=9, color="#2ca02c", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    ax1.set_title("Amplitude by frequency band\n(GLBX Bartlett Cove 2024)", fontsize=11)

    # Right: input SNR (MS) showing the hardware-level cause
    ax2.plot(monthly["month"], monthly["ms_med"], "o-", color="#9467bd", lw=2, ms=7)
    ax2.set_xlabel("Month", fontsize=11)
    ax2.set_ylabel("Mean SNR (dBHz, monthly median)", fontsize=11)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_title("Input signal quality (MS)\ndrops ~8 dBHz in winter", fontsize=11)
    ax2.annotate("~8 dBHz drop\n(hardware/atmospheric)",
                 xy=(1, monthly.loc[monthly["month"] == 1, "ms_med"].values[0]),
                 xytext=(4, 36), fontsize=9, color="#9467bd", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#9467bd"))

    fig.suptitle("Fig 5a — GLBX: L1 Suppression Inverts Pooled Amplitude (Hardware Artifact, Not Surface Physics)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig5a_glbx_amplitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5a_glbx_amplitude.png")


# ── Figure 5b: Three-station comparison ──────────────────────────────

def fig5b_three_station(outdir):
    """Same frequency-split plot for GLBX, UMNQ, NIAQ — only GLBX is contaminated."""

    stations = [
        ("GLBX", 2024, "Bartlett Cove, AK"),
        ("UMNQ", 2025, "Upernavik, Greenland"),
        ("NIAQ", 2025, "Niakornat, Greenland"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (station, year, location) in zip(axes, stations):
        df = pd.read_parquet(RESULTS / f"{station}/{station}_{year}_daily_features.parquet")
        df = df[df["azimuth_bin"] == -1].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month

        # Use arc table for per-band amplitude if daily_features lacks freq-split cols
        has_L1 = "amp_L1_mean" in df.columns
        if has_L1:
            monthly = df.groupby("month").agg(
                amp_L1=("amp_L1_mean", "median"),
                amp_L5=("amp_L5_mean", "median"),
                amp_pool=("amp_mean", "median"),
            ).reset_index()
        else:
            # Compute from arc table
            arc = pd.read_parquet(RESULTS / f"{station}/{station}_{year}_arc_table.parquet")
            arc["date"] = pd.to_datetime(arc["date"])
            arc["month"] = arc["date"].dt.month
            l1 = arc[arc["freq_group"] == "L1"].groupby("month")["Amp"].median().rename("amp_L1")
            l5 = arc[arc["freq_group"].isin(["L5", "L5a"])].groupby("month")["Amp"].median().rename("amp_L5")
            pool = df.groupby("month")["amp_mean"].median().rename("amp_pool")
            monthly = pd.concat([l1, l5, pool], axis=1).reset_index()

        ax.plot(monthly["month"], monthly["amp_L1"], "o-", color="#1f77b4", lw=2, ms=6, label="L1")
        ax.plot(monthly["month"], monthly["amp_L5"], "s-", color="#2ca02c", lw=2, ms=6, label="L5")
        ax.plot(monthly["month"], monthly["amp_pool"], "D-", color="#666", lw=1.5, ms=5, label="Pooled", alpha=0.7)

        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8)
        ax.set_title(f"{station} {year}\n{location}", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

    # Verdict annotations
    axes[0].text(0.5, 0.02, "CONTAMINATED\nL1 collapse inverts pool",
                 transform=axes[0].transAxes, ha="center", fontsize=9,
                 fontweight="bold", color="#d62728",
                 bbox=dict(boxstyle="round", fc="white", ec="#d62728", alpha=0.9))
    axes[1].text(0.5, 0.02, "CLEAN\nBoth bands rise in winter",
                 transform=axes[1].transAxes, ha="center", fontsize=9,
                 fontweight="bold", color="#2ca02c",
                 bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.9))
    axes[2].text(0.5, 0.02, "CLEAN\nMinimal seasonal variation",
                 transform=axes[2].transAxes, ha="center", fontsize=9,
                 fontweight="bold", color="#2ca02c",
                 bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.9))

    axes[0].set_ylabel("Amplitude (monthly median)", fontsize=11)

    fig.suptitle("Fig 5b — Three-Station Validation: L1 Suppression Is GLBX-Specific",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig5b_three_station.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5b_three_station.png")


# ── Figure 5c: DOY crossover ────────────────────────────────────────

def fig5c_doy_crossover(outdir):
    """Rolling 30-day Cohen's d for amp_L1_mean vs pooled amp_mean at UMNQ."""
    # Compute per-band amplitude from arc table (not yet in daily_features for UMNQ)
    arc = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_arc_table.parquet")
    v3 = pd.read_parquet(RESULTS / "UMNQ/UMNQ_2025_ice_classification_v3.parquet")

    # Daily per-band amplitude
    arc["date"] = pd.to_datetime(arc["date"])
    arc_L1 = arc[arc["freq_group"] == "L1"].groupby("doy")["Amp"].median().rename("amp_L1")
    arc_L5 = arc[arc["freq_group"].isin(["L5"])].groupby("doy")["Amp"].median().rename("amp_L5")
    arc_pool = arc.groupby("doy")["Amp"].median().rename("amp_pool")

    daily = pd.concat([arc_L1, arc_L5, arc_pool], axis=1).reset_index()
    v3["date"] = pd.to_datetime(v3["date"])
    v3["doy"] = v3["date"].dt.dayofyear
    daily = daily.merge(v3[["doy", "v3_state"]], on="doy", how="inner")

    # Exclude regime_change for cleaner d computation
    daily = daily[daily["v3_state"] != "regime_change"]

    # Rolling Cohen's d (30-day window, 7-day step)
    window = 30
    step = 7
    centers = list(range(window // 2, 366 - window // 2, step))

    results = []
    for center in centers:
        lo, hi = center - window // 2, center + window // 2
        w = daily[(daily["doy"] >= lo) & (daily["doy"] <= hi)]
        anom = w[w["v3_state"] == "anomalous"]
        base = w[w["v3_state"] == "baseline"]
        if len(anom) < 3 or len(base) < 3:
            continue
        results.append({
            "doy": center,
            "d_L1": cohens_d(anom["amp_L1"].dropna(), base["amp_L1"].dropna()),
            "d_pool": cohens_d(anom["amp_pool"].dropna(), base["amp_pool"].dropna()),
            "d_L5": cohens_d(anom["amp_L5"].dropna(), base["amp_L5"].dropna()),
            "n_anom": len(anom),
            "n_base": len(base),
        })

    rd = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(rd["doy"], rd["d_L1"], "o-", color="#1f77b4", lw=2, ms=5, label="amp_L1 Cohen's d")
    ax.plot(rd["doy"], rd["d_pool"], "D-", color="#666", lw=2, ms=5, label="pooled amp Cohen's d")
    ax.plot(rd["doy"], rd["d_L5"], "s-", color="#2ca02c", lw=1.5, ms=4, alpha=0.6, label="amp_L5 Cohen's d")

    ax.axhline(0, color="black", ls="-", lw=0.5)
    ax.axhline(0.8, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(30, 0.85, "large effect", fontsize=7, color="gray")

    # Highlight the crossover window
    # Find where L1 crosses zero
    cross_mask = (rd["d_L1"].shift(1) < 0) & (rd["d_L1"] >= 0)
    cross_doys = rd.loc[cross_mask, "doy"]
    if len(cross_doys) > 0:
        cross_doy = cross_doys.iloc[-1]  # Take the autumn crossing
        ax.axvline(cross_doy, color="#d62728", ls="--", lw=1.5, alpha=0.7)
        ax.text(cross_doy + 3, ax.get_ylim()[0] + 0.3,
                f"L1 crosses zero\nat DOY {cross_doy}\n(pooled still negative)",
                fontsize=9, color="#d62728", fontweight="bold")

    # Shade the gap region
    autumn = rd[rd["doy"] >= 250]
    if len(autumn) > 2:
        ax.fill_between(autumn["doy"], autumn["d_L1"], autumn["d_pool"],
                        alpha=0.15, color="#d62728", label="L1 advantage window")

    ax.set_xlabel("Day of year (UMNQ 2025)", fontsize=11)
    ax.set_ylabel("Cohen's d  (anomalous vs baseline)\n+d = correct ice direction", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(15, 360)

    ax.set_title("Fig 5c — DOY Crossover: L1 Detects Freeze-Onset a Month Before Pooled Amplitude",
                 fontsize=13, fontweight="bold")
    ax.text(0.98, 0.02,
            "L1 (19 cm) is sensitive to thin new ice;\n"
            "L5 (25 cm) lags → dilutes pooled mean\n"
            "during the critical autumn transition.",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            style="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    fig.savefig(outdir / "fig5c_doy_crossover.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5c_doy_crossover.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/images/investigation",
                        help="Output directory for figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating Story 1 — AF Baseline Fix:")
    fig1a_baselines(outdir)
    fig1b_af_shift(outdir)
    fig1c_prn_variance(outdir)

    print("\nGenerating Story 2 — Gamma Hardening:")
    fig2a_arc_guard(outdir)
    fig2b_gamma_r2(outdir)

    print("\nGenerating Story 3 — Feature Candidates (Tested & Ruled Out):")
    fig3a_envelope_ratio(outdir)
    fig3b_xfreq_correlation(outdir)
    fig3c_geometry_census(outdir)

    print("\nGenerating Story 4 — Post-Fix Feature Reassessment:")
    fig4a_cohens_d(outdir)
    fig4b_clr_blind(outdir)
    fig4c_sector_consistency(outdir)

    print("\nGenerating Story 5 — GLBX Hardware Artifact:")
    fig5a_glbx_amplitude(outdir)
    fig5b_three_station(outdir)
    fig5c_doy_crossover(outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
