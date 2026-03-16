# ABOUTME: Audits station GNSS-IR configuration by analyzing per-frequency performance,
# ABOUTME: arc duration tradeoffs, RHdot correction impact, and sample rate adequacy.

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.visualizer.base import PLOT_COLORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# GNSS frequency code labels (gnssrefl convention)
FREQ_LABELS = {
    1: "GPS L1",
    2: "GPS L2",
    5: "GPS L5",
    20: "GPS L2C",
    101: "GLO L1",
    102: "GLO L2",
    201: "GAL E1",
    205: "GAL E5a",
    206: "GAL E5b",
    207: "GAL E5",
    208: "GAL E6",
    302: "BDS B1",
    306: "BDS B3",
}


def compute_frequency_stats(df):
    """Compute per-frequency retrieval statistics.

    Args:
        df: DataFrame with columns: freq, Amp, PkNoise, RH.
            Optionally includes rh_residual for spline-based quality.

    Returns:
        DataFrame indexed by freq with columns: count, median_amp,
        median_pknoise, rh_std. Includes residual_rmse if rh_residual present.
    """
    grouped = df.groupby("freq")
    stats = pd.DataFrame({
        "count": grouped["RH"].count(),
        "median_amp": grouped["Amp"].median(),
        "median_pknoise": grouped["PkNoise"].median(),
        "rh_std": grouped["RH"].std(),
    })

    if "rh_residual" in df.columns:
        stats["residual_rmse"] = grouped["rh_residual"].apply(
            lambda x: np.sqrt(np.mean(x ** 2))
        )

    return stats


def compute_delTmax_tradeoff(df, thresholds):
    """Compute data retention and RMSE at various delTmax thresholds.

    Args:
        df: DataFrame with columns DelT (arc duration in minutes)
            and rh_residual (residual from spline or median).
        thresholds: List of delTmax values (minutes) to evaluate.

    Returns:
        DataFrame with columns: delTmax, n_retained, pct_retained, rmse.
    """
    total = len(df)
    rows = []
    for t in thresholds:
        mask = df["DelT"] <= t
        subset = df[mask]
        n = len(subset)
        if n > 0:
            rmse = np.sqrt(np.mean(subset["rh_residual"] ** 2))
        else:
            rmse = np.nan
        rows.append({
            "delTmax": t,
            "n_retained": n,
            "pct_retained": 100.0 * n / total if total > 0 else 0.0,
            "rmse": rmse,
        })
    return pd.DataFrame(rows)


def compute_rhdot_impact(df_raw, df_corrected):
    """Compute before/after statistics for RHdot+IF corrections.

    Merges raw and corrected data on [MJD, sat, freq] to compare
    raw RH vs IF-corrected RH.

    Args:
        df_raw: DataFrame with columns MJD, sat, freq, RH.
        df_corrected: DataFrame with columns MJD, sat, freq,
            rh_if_corrected, rhdot_correction.

    Returns:
        Dict with raw_rh_std, corrected_rh_std, median_correction_magnitude,
        n_matched, and per_freq breakdown.
    """
    merge_keys = ["MJD", "sat", "freq"]
    available_keys = [k for k in merge_keys
                      if k in df_raw.columns and k in df_corrected.columns]

    merged = df_raw.merge(
        df_corrected[available_keys + ["rh_if_corrected", "rhdot_correction"]],
        on=available_keys,
        how="inner",
    )

    n_matched = len(merged)
    if n_matched == 0:
        return {
            "raw_rh_std": np.nan,
            "corrected_rh_std": np.nan,
            "median_correction_magnitude": np.nan,
            "n_matched": 0,
            "per_freq": {},
        }

    raw_rh_std = merged["RH"].std()
    corrected_rh_std = merged["rh_if_corrected"].std()
    median_correction = merged["rhdot_correction"].abs().median()

    # Per-frequency breakdown
    per_freq = {}
    for freq, grp in merged.groupby("freq"):
        per_freq[freq] = {
            "count": len(grp),
            "raw_std": grp["RH"].std(),
            "corrected_std": grp["rh_if_corrected"].std(),
            "median_correction": grp["rhdot_correction"].abs().median(),
        }

    return {
        "raw_rh_std": raw_rh_std,
        "corrected_rh_std": corrected_rh_std,
        "median_correction_magnitude": median_correction,
        "n_matched": n_matched,
        "per_freq": per_freq,
    }


def check_nyquist_adequacy(max_rh_m, sample_rate_sec):
    """Check if sample rate is adequate for the reflector height range.

    Uses the GNSS-IR Nyquist relationship: the maximum resolvable RH
    depends on the sampling rate in sin(elevation) space.

    Conservative empirical thresholds from gnssrefl documentation:
      1s  → ~100m max RH
      5s  → ~40m
      15s → ~30m
      30s → ~12m

    Args:
        max_rh_m: Maximum expected reflector height in meters.
        sample_rate_sec: Data sample rate in seconds.

    Returns:
        Dict with max_resolvable_rh, is_adequate, margin_pct, sample_rate_sec.
    """
    # Empirical Nyquist model for GNSS-IR based on gnssrefl docs and use cases.
    # At typical satellite elevation rates (~0.005 deg/s), the max resolvable
    # RH scales inversely with sample rate. Calibrated against:
    #   tggo: 15s needed for ~12m RH (30s "inadequate")
    #   tnpp: 2s decimation for 55-70m RH
    #   general docs: >6m needs better than 30s
    #
    # Conservative model: max_rh ≈ 400 / sample_rate_sec
    # This gives: 1s→400m, 2s→200m, 5s→80m, 15s→26.7m, 30s→13.3m
    max_resolvable_rh = 400.0 / sample_rate_sec

    margin_pct = 100.0 * (max_resolvable_rh - max_rh_m) / max_rh_m
    is_adequate = max_resolvable_rh > max_rh_m * 1.3  # require 30% margin

    return {
        "max_resolvable_rh": max_resolvable_rh,
        "is_adequate": is_adequate,
        "margin_pct": margin_pct,
        "sample_rate_sec": sample_rate_sec,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _freq_label(code):
    return FREQ_LABELS.get(code, f"F{code}")


def _add_residuals_from_rolling_median(df, window_hours=6.0):
    """Add rh_residual column: deviation from rolling median RH."""
    df = df.sort_values("MJD").copy()
    # Approximate time step between unique epochs (1 day = 1.0 MJD)
    mjd_diffs = df["MJD"].diff()
    median_step = mjd_diffs[mjd_diffs > 0].median()
    if median_step > 0:
        window_pts = max(5, int(window_hours / 24.0 / median_step))
    else:
        window_pts = max(5, len(df) // 50)
    df["rh_residual"] = df["RH"] - df["RH"].rolling(
        window_pts, center=True, min_periods=3
    ).median()
    df = df.dropna(subset=["rh_residual"])
    return df


def plot_frequency_performance(freq_stats, station, year, output_dir,
                               config_freqs=None, reqAmp=None, pkNoise=None):
    """Figure 1: Per-frequency retrieval performance (2x2 grid).

    Shows count, amplitude, peak2noise, and RH scatter per frequency.
    Configured but absent frequencies are shown as empty bars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{station} {year} — Per-Frequency Performance Audit",
                 fontsize=14, fontweight="bold")

    # Build label order: configured freqs first, then any extras in data
    if config_freqs:
        all_freqs = list(config_freqs)
        for f in freq_stats.index:
            if f not in all_freqs:
                all_freqs.append(f)
    else:
        all_freqs = sorted(freq_stats.index)

    labels = [_freq_label(f) for f in all_freqs]
    x = np.arange(len(all_freqs))

    def _get_values(col):
        return [freq_stats.loc[f, col] if f in freq_stats.index else 0
                for f in all_freqs]

    def _color_bars(values, all_freqs_list):
        colors = []
        for i, f in enumerate(all_freqs_list):
            if f not in freq_stats.index:
                colors.append(PLOT_COLORS["quality_poor"])
            else:
                colors.append(PLOT_COLORS["gnssir"])
        return colors

    # Panel A: Retrieval count
    ax = axes[0, 0]
    counts = _get_values("count")
    colors = _color_bars(counts, all_freqs)
    ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Retrievals")
    ax.set_title("Retrieval Count per Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Annotate absent freqs
    for i, f in enumerate(all_freqs):
        if f not in freq_stats.index:
            ax.annotate("No data", (x[i], 0), ha="center", va="bottom",
                        fontsize=7, color=PLOT_COLORS["quality_poor"])

    # Panel B: Median amplitude
    ax = axes[0, 1]
    amps = _get_values("median_amp")
    ax.bar(x, amps, color=_color_bars(amps, all_freqs),
           edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Amplitude (V/V)")
    ax.set_title("Median Amplitude per Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if reqAmp is not None:
        ax.axhline(reqAmp, color=PLOT_COLORS["quality_poor"],
                    linestyle="--", alpha=0.7, label=f"reqAmp={reqAmp}")
        ax.legend(fontsize=8)

    # Panel C: Median peak2noise
    ax = axes[1, 0]
    pns = _get_values("median_pknoise")
    ax.bar(x, pns, color=_color_bars(pns, all_freqs),
           edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Peak-to-Noise")
    ax.set_title("Median Peak-to-Noise per Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if pkNoise is not None:
        ax.axhline(pkNoise, color=PLOT_COLORS["quality_poor"],
                    linestyle="--", alpha=0.7, label=f"PkNoise threshold={pkNoise}")
        ax.legend(fontsize=8)

    # Panel D: RH scatter (std dev)
    ax = axes[1, 1]
    stds = _get_values("rh_std")
    # Color by quality: lower std is better
    std_colors = []
    for i, f in enumerate(all_freqs):
        if f not in freq_stats.index:
            std_colors.append(PLOT_COLORS["quality_poor"])
        elif stds[i] > 0 and stds[i] < np.median([s for s in stds if s > 0]) * 0.8:
            std_colors.append(PLOT_COLORS["quality_good"])
        elif stds[i] > np.median([s for s in stds if s > 0]) * 1.5:
            std_colors.append(PLOT_COLORS["quality_poor"])
        else:
            std_colors.append(PLOT_COLORS["quality_medium"])
    ax.bar(x, stds, color=std_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("RH Std Dev (m)")
    ax.set_title("RH Scatter per Frequency (lower = tighter)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Add residual RMSE if available
    if "residual_rmse" in freq_stats.columns:
        ax2 = ax.twinx()
        rmses = [freq_stats.loc[f, "residual_rmse"] if f in freq_stats.index else 0
                 for f in all_freqs]
        ax2.plot(x, rmses, "D", color=PLOT_COLORS["correlation"],
                 markersize=6, label="Residual RMSE")
        ax2.set_ylabel("Residual RMSE (m)", color=PLOT_COLORS["correlation"])
        ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / f"{station}_{year}_freq_performance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved frequency performance audit: {out_path}")
    return out_path


def plot_delTmax_tradeoff(df, tradeoff_df, current_delTmax,
                          station, year, output_dir):
    """Figure 2: Arc duration analysis (2x2 grid).

    Shows arc duration distribution, residual vs duration scatter,
    data retention curve, and RMSE tradeoff curve.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{station} {year} — Arc Duration (delTmax) Analysis",
                 fontsize=14, fontweight="bold")

    # Panel A: Histogram of arc durations
    ax = axes[0, 0]
    ax.hist(df["DelT"], bins=50, color=PLOT_COLORS["gnssir"],
            edgecolor="white", alpha=0.8)
    ax.axvline(current_delTmax, color=PLOT_COLORS["quality_poor"],
               linestyle="--", linewidth=2,
               label=f"Current delTmax={current_delTmax:.0f} min")
    ax.set_xlabel("Arc Duration (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Satellite Arc Durations")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: |Residual| vs arc duration scatter
    ax = axes[0, 1]
    ax.scatter(df["DelT"], df["rh_residual"].abs(),
               alpha=0.3, s=8, color=PLOT_COLORS["scatter"])
    # Bin medians for trend
    bins = np.arange(0, df["DelT"].max() + 5, 5)
    df_binned = df.copy()
    df_binned["DelT_bin"] = pd.cut(df_binned["DelT"], bins)
    medians = df_binned.groupby("DelT_bin", observed=True)["rh_residual"].apply(
        lambda x: x.abs().median()
    )
    bin_centers = [(b.left + b.right) / 2 for b in medians.index]
    ax.plot(bin_centers, medians.values, "o-",
            color=PLOT_COLORS["quality_poor"], markersize=5, linewidth=2,
            label="Binned median")
    ax.axvline(current_delTmax, color=PLOT_COLORS["quality_poor"],
               linestyle="--", alpha=0.5)
    ax.set_xlabel("Arc Duration (minutes)")
    ax.set_ylabel("|RH Residual| (m)")
    ax.set_title("Retrieval Scatter vs Arc Duration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel C: Data retention vs delTmax
    ax = axes[1, 0]
    ax.plot(tradeoff_df["delTmax"], tradeoff_df["pct_retained"],
            "o-", color=PLOT_COLORS["gnssir"], markersize=5, linewidth=2)
    ax.axvline(current_delTmax, color=PLOT_COLORS["quality_poor"],
               linestyle="--", linewidth=2, label=f"Current={current_delTmax:.0f}")
    ax.set_xlabel("delTmax Threshold (minutes)")
    ax.set_ylabel("Data Retained (%)")
    ax.set_title("Data Retention vs delTmax")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel D: RMSE vs delTmax
    ax = axes[1, 1]
    ax.plot(tradeoff_df["delTmax"], tradeoff_df["rmse"] * 100,
            "o-", color=PLOT_COLORS["correlation"], markersize=5, linewidth=2)
    ax.axvline(current_delTmax, color=PLOT_COLORS["quality_poor"],
               linestyle="--", linewidth=2, label=f"Current={current_delTmax:.0f}")
    ax.set_xlabel("delTmax Threshold (minutes)")
    ax.set_ylabel("RH Residual RMSE (cm)")
    ax.set_title("RMSE vs delTmax (lower-left is better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add annotation showing the tradeoff
    current_row = tradeoff_df[tradeoff_df["delTmax"] >= current_delTmax]
    if len(current_row) > 0:
        row = current_row.iloc[0]
        ax.annotate(
            f"{row['pct_retained']:.0f}% data, {row['rmse']*100:.1f}cm RMSE",
            xy=(current_delTmax, row["rmse"] * 100),
            xytext=(current_delTmax - 10, row["rmse"] * 100 + 1),
            fontsize=8, ha="right",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    plt.tight_layout()
    out_path = output_dir / f"{station}_{year}_delTmax_tradeoff.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved delTmax tradeoff analysis: {out_path}")
    return out_path


def plot_rhdot_impact(merged_df, impact_stats, station, year, output_dir):
    """Figure 3: RHdot+IF correction impact (2x2 grid).

    Shows raw vs corrected time series, correction magnitude histogram,
    per-retrieval scatter comparison, and per-frequency RMSE improvement.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{station} {year} — RHdot + IF Bias Correction Impact",
                 fontsize=14, fontweight="bold")

    # Panel A: Time series of raw vs corrected (zoomed to ~5 days)
    ax = axes[0, 0]
    # Pick a representative 5-day window near the middle
    mjd_mid = merged_df["MJD"].median()
    window = merged_df[(merged_df["MJD"] >= mjd_mid - 2.5) &
                       (merged_df["MJD"] <= mjd_mid + 2.5)].copy()
    if len(window) < 10:
        window = merged_df.head(200)
    hours = (window["MJD"] - window["MJD"].iloc[0]) * 24
    ax.scatter(hours, window["RH"], s=10, alpha=0.5,
               color=PLOT_COLORS["quality_poor"], label="Raw RH", zorder=2)
    ax.scatter(hours, window["rh_if_corrected"], s=10, alpha=0.5,
               color=PLOT_COLORS["quality_good"], label="Corrected RH", zorder=3)
    ax.set_xlabel("Hours from window start")
    ax.set_ylabel("Reflector Height (m)")
    ax.set_title("Raw vs Corrected RH (5-day window)")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

    # Panel B: Histogram of correction magnitudes
    ax = axes[0, 1]
    corrections = merged_df["rhdot_correction"]
    ax.hist(corrections * 100, bins=60, color=PLOT_COLORS["scatter"],
            edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    med = corrections.abs().median() * 100
    ax.axvline(med, color=PLOT_COLORS["correlation"], linestyle="--",
               label=f"Median |corr|={med:.2f} cm")
    ax.axvline(-med, color=PLOT_COLORS["correlation"], linestyle="--")
    ax.set_xlabel("RHdot Correction (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RHdot Corrections")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Residual comparison scatter
    ax = axes[1, 0]
    # Compute residuals from rolling median for both raw and corrected
    sorted_m = merged_df.sort_values("MJD").copy()
    window_pts = max(5, len(sorted_m) // 50)
    median_rh = sorted_m["RH"].rolling(window_pts, center=True, min_periods=3).median()
    raw_resid = (sorted_m["RH"] - median_rh).dropna().abs() * 100
    corr_resid = (sorted_m["rh_if_corrected"] - median_rh).dropna().abs() * 100
    common_idx = raw_resid.index.intersection(corr_resid.index)
    if len(common_idx) > 0:
        ax.scatter(raw_resid[common_idx], corr_resid[common_idx],
                   s=5, alpha=0.2, color=PLOT_COLORS["scatter"])
        max_val = max(raw_resid[common_idx].quantile(0.98),
                      corr_resid[common_idx].quantile(0.98))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="1:1 line")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    ax.set_xlabel("|Raw RH Residual| (cm)")
    ax.set_ylabel("|Corrected RH Residual| (cm)")
    ax.set_title("Per-Retrieval: Raw vs Corrected Scatter")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Count points below the 1:1 line (improved)
    if len(common_idx) > 0:
        n_improved = (corr_resid[common_idx] < raw_resid[common_idx]).sum()
        pct = 100 * n_improved / len(common_idx)
        ax.text(0.95, 0.05, f"{pct:.0f}% of retrievals\nimproved by correction",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Panel D: Per-frequency improvement
    ax = axes[1, 1]
    freqs = sorted(impact_stats["per_freq"].keys())
    labels = [_freq_label(f) for f in freqs]
    raw_stds = [impact_stats["per_freq"][f]["raw_std"] * 100 for f in freqs]
    corr_stds = [impact_stats["per_freq"][f]["corrected_std"] * 100 for f in freqs]
    x = np.arange(len(freqs))
    width = 0.35
    ax.bar(x - width / 2, raw_stds, width, color=PLOT_COLORS["quality_poor"],
           label="Raw RH", edgecolor="white")
    ax.bar(x + width / 2, corr_stds, width, color=PLOT_COLORS["quality_good"],
           label="Corrected RH", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RH Std Dev (cm)")
    ax.set_title("Per-Frequency: Raw vs Corrected Scatter")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Summary textbox
    raw_s = impact_stats["raw_rh_std"] * 100
    corr_s = impact_stats["corrected_rh_std"] * 100
    improvement = (1 - corr_s / raw_s) * 100 if raw_s > 0 else 0
    summary = (
        f"Matched retrievals: {impact_stats['n_matched']}\n"
        f"Raw RH std: {raw_s:.2f} cm\n"
        f"Corrected RH std: {corr_s:.2f} cm\n"
        f"Overall improvement: {improvement:.1f}%\n"
        f"Median |correction|: {impact_stats['median_correction_magnitude']*100:.2f} cm"
    )
    fig.text(0.02, 0.02, summary, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = output_dir / f"{station}_{year}_rhdot_impact.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved RHdot correction impact: {out_path}")
    return out_path


def plot_sample_rate_summary(station_results, output_dir):
    """Figure 4: Sample rate adequacy summary across stations.

    Args:
        station_results: List of dicts with station, year, max_rh, nyquist info.
        output_dir: Where to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Sample Rate Adequacy — Nyquist Check",
                 fontsize=14, fontweight="bold")

    stations = [r["station"] for r in station_results]
    max_rhs = [r["max_rh"] for r in station_results]
    max_resolvable = [r["max_resolvable_rh"] for r in station_results]
    sample_rates = [r["sample_rate_sec"] for r in station_results]

    x = np.arange(len(stations))
    width = 0.35

    bars_rh = ax.bar(x - width / 2, max_rhs, width, color=PLOT_COLORS["gnssir"],
                     label="Config maxH", edgecolor="white")
    bars_nyq = ax.bar(x + width / 2, max_resolvable, width,
                      color=PLOT_COLORS["quality_good"],
                      label="Nyquist max RH", edgecolor="white")

    # Cap the Nyquist bar display for readability
    y_max = max(max_rhs) * 3
    ax.set_ylim(0, y_max)
    for i, (rh, nyq, sr) in enumerate(zip(max_rhs, max_resolvable, sample_rates)):
        if nyq > y_max:
            ax.annotate(f"{nyq:.0f}m", (x[i] + width / 2, y_max * 0.95),
                        ha="center", va="top", fontsize=8, fontweight="bold",
                        color=PLOT_COLORS["quality_good"])
        margin = 100 * (nyq - rh) / rh
        adequate = nyq > rh * 1.3
        color = PLOT_COLORS["quality_good"] if adequate else PLOT_COLORS["quality_poor"]
        ax.text(x[i], max(rh, min(nyq, y_max)) + y_max * 0.02,
                f"{sr:.0f}s\n{margin:.0f}% margin",
                ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(stations, fontsize=10)
    ax.set_ylabel("Reflector Height (m)")
    ax.set_title("Configured maxH vs Nyquist-Resolvable RH per Station")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "sample_rate_adequacy.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved sample rate summary: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _detect_sample_rate(station_lower, year):
    """Detect SNR file sample rate by reading the first file found."""
    project_root = Path(__file__).parent.parent
    snr_dir = (project_root / "gnssrefl_data_workspace" / "refl_code"
               / str(year) / "snr" / station_lower)
    if not snr_dir.exists():
        return None

    snr_files = sorted(snr_dir.glob("*.snr66*"))
    if not snr_files:
        return None

    # Read first non-comment lines and check time column spacing
    import gzip
    path = snr_files[0]
    opener = gzip.open if path.suffix == ".gz" else open
    times = []
    try:
        with opener(path, "rt") as f:
            for line in f:
                if line.startswith("%") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # Column 4 is seconds-of-day in SNR files
                    times.append(float(parts[3]))
                if len(times) > 200:
                    break
    except Exception:
        return None

    if len(times) < 10:
        return None

    # Find the most common time step
    diffs = np.diff(sorted(set(times)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    # Use the mode of small positive diffs
    rounded = np.round(diffs).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    return float(values[counts.argmax()])


def run_audit(station, year):
    """Run the full configuration audit for a station.

    Loads data, computes statistics, generates diagnostic figures.
    Returns dict of output paths.
    """
    project_root = Path(__file__).parent.parent
    station_upper = station.upper()
    station_lower = station.lower()
    output_dir = project_root / "results_annual" / station_upper
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load station config
    config_path = project_root / "config" / f"{station_lower}.json"
    if not config_path.exists():
        # Try alternate naming
        alt = list((project_root / "config").glob(f"{station_lower}*.json"))
        config_path = alt[0] if alt else None

    config = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logging.info(f"Loaded config from {config_path.name}")

    config_freqs = config.get("freqs", [])
    current_delTmax = config.get("delTmax", 75.0)
    max_rh = config.get("maxH", 15.0)
    reqAmp_val = None
    reqAmp_list = config.get("reqAmp", [])
    if reqAmp_list:
        reqAmp_val = reqAmp_list[0] if isinstance(reqAmp_list, list) else reqAmp_list
    pkNoise_val = config.get("PkNoise", None)

    # Load raw results
    raw_path = output_dir / f"{station_upper}_{year}_combined_raw.csv"
    if not raw_path.exists():
        logging.error(f"Combined raw CSV not found: {raw_path}")
        return {}
    df_raw = pd.read_csv(raw_path)
    logging.info(f"Loaded {len(df_raw)} raw retrievals from {raw_path.name}")

    # Load corrected retrievals
    files_dir = (project_root / "gnssrefl_data_workspace" / "refl_code"
                 / "Files" / station_lower)
    if_path = files_dir / f"{station_lower}_{year}_subdaily_edit.txt.withrhdotIF"
    df_corrected = None
    if if_path.exists():
        from scripts.utils.subdaily_loader import load_corrected_retrievals
        df_corrected = load_corrected_retrievals(if_path)
        logging.info(f"Loaded {len(df_corrected)} corrected retrievals")
    else:
        logging.warning(f"No .withrhdotIF file found at {if_path}")

    # Add residuals (rolling median)
    df_with_resid = _add_residuals_from_rolling_median(df_raw)

    outputs = {}

    # --- Figure 1: Frequency performance ---
    freq_stats = compute_frequency_stats(df_with_resid)
    logging.info(f"Frequency stats:\n{freq_stats.to_string()}")
    outputs["freq_performance"] = plot_frequency_performance(
        freq_stats, station_upper, year, output_dir,
        config_freqs=config_freqs, reqAmp=reqAmp_val, pkNoise=pkNoise_val,
    )

    # --- Figure 2: delTmax tradeoff ---
    thresholds = list(range(5, 80, 5))
    tradeoff = compute_delTmax_tradeoff(df_with_resid, thresholds)
    outputs["delTmax_tradeoff"] = plot_delTmax_tradeoff(
        df_with_resid, tradeoff, current_delTmax,
        station_upper, year, output_dir,
    )

    # --- Figure 3: RHdot impact ---
    if df_corrected is not None and len(df_corrected) > 0:
        impact = compute_rhdot_impact(df_raw, df_corrected)
        if impact["n_matched"] > 0:
            # Build merged df for plotting
            merge_keys = ["MJD", "sat", "freq"]
            avail = [k for k in merge_keys
                     if k in df_raw.columns and k in df_corrected.columns]
            merged_plot = df_raw.merge(
                df_corrected[avail + ["rh_if_corrected", "rhdot_correction"]],
                on=avail, how="inner",
            )
            outputs["rhdot_impact"] = plot_rhdot_impact(
                merged_plot, impact, station_upper, year, output_dir,
            )

    # --- Figure 4 data: Nyquist check ---
    sample_rate = _detect_sample_rate(station_lower, year)
    nyquist_info = None
    if sample_rate is not None:
        nyquist_info = check_nyquist_adequacy(max_rh, sample_rate)
        logging.info(
            f"Sample rate: {sample_rate}s, max resolvable RH: "
            f"{nyquist_info['max_resolvable_rh']:.0f}m, "
            f"adequate: {nyquist_info['is_adequate']}, "
            f"margin: {nyquist_info['margin_pct']:.0f}%"
        )
    else:
        logging.warning("Could not detect sample rate from SNR files")

    outputs["nyquist"] = {
        "station": station_upper,
        "year": year,
        "max_rh": max_rh,
        "sample_rate_sec": sample_rate,
        **(nyquist_info or {}),
    }

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Audit GNSS-IR station configuration"
    )
    parser.add_argument("--station", nargs="+", required=True,
                        help="Station name(s) to audit (e.g., GLBX NKAR)")
    parser.add_argument("--year", type=int, nargs="+", required=True,
                        help="Year(s) to audit (matched 1:1 with stations)")
    args = parser.parse_args()

    if len(args.year) == 1 and len(args.station) > 1:
        years = args.year * len(args.station)
    else:
        years = args.year
    if len(years) != len(args.station):
        parser.error("Provide one year per station, or one year for all")

    project_root = Path(__file__).parent.parent
    all_nyquist = []

    for station, year in zip(args.station, years):
        logging.info(f"\n{'='*60}")
        logging.info(f"AUDITING {station.upper()} {year}")
        logging.info(f"{'='*60}")
        outputs = run_audit(station, year)
        if "nyquist" in outputs:
            all_nyquist.append(outputs["nyquist"])

    # Cross-station Nyquist summary if multiple stations
    if len(all_nyquist) > 1:
        valid = [r for r in all_nyquist if r.get("sample_rate_sec") is not None]
        if valid:
            output_dir = project_root / "results_annual"
            plot_sample_rate_summary(valid, output_dir)

    logging.info("\nAudit complete.")


if __name__ == "__main__":
    main()
