# ABOUTME: Data-driven diagnostics tab for GNSS-IR dashboard
# ABOUTME: Azimuth quality flags, frequency performance, arc quality from enriched/per-arc data

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dashboard_components.data_loader import (
    get_available_diagnostic_days,
    get_quicklook_plots_for_day,
    doy_to_date,
    date_to_doy,
    load_enriched,
    load_per_arc,
)

AZ_BIN_LABELS = {0: "0-90 (N-E)", 1: "90-180 (E-S)", 2: "180-270 (S-W)", 3: "270-360 (W-N)"}


def _render_azimuth_quality(per_arc):
    """Azimuth sector quality assessment — flags suspect directions."""
    az_step = 10
    per_arc = per_arc.copy()
    per_arc["az_10"] = (per_arc["Azim"] // az_step * az_step).astype(int)
    per_arc["date_dt"] = pd.to_datetime(per_arc["date"])
    per_arc["month"] = per_arc["date_dt"].dt.month

    az_bins = sorted(per_arc["az_10"].unique())

    # Compute per-azimuth statistics
    rows = []
    for az in az_bins:
        ad = per_arc[per_arc["az_10"] == az]
        if len(ad) < 20:
            continue

        daily_cv = ad.groupby("date")["Amp"].agg(
            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) >= 3 else np.nan
        ).dropna()

        # Seasonal ratio (early vs mid-year)
        months = sorted(ad["month"].unique())
        early = ad[ad["month"].isin(months[:2])]
        mid = ad[ad["month"].isin([m for m in months if 6 <= m <= 8])]
        if not mid.empty:
            mid_data = mid
        else:
            mid_data = ad[ad["month"].isin(months[len(months)//2:len(months)//2+2])]

        if len(early) >= 10 and len(mid_data) >= 10:
            ratio = early["Amp"].mean() / mid_data["Amp"].mean()
        else:
            ratio = np.nan

        rh_std = ad.groupby("date")["RH"].std().mean()

        rows.append({
            "az": az,
            "n_arcs": len(ad),
            "n_days": ad["date"].nunique(),
            "amp_mean": ad["Amp"].mean(),
            "amp_cv_mean": daily_cv.mean() if len(daily_cv) > 0 else np.nan,
            "rh_mean": ad["RH"].mean(),
            "rh_std_daily": rh_std,
            "p2n_mean": ad["PkNoise"].mean(),
            "seasonal_ratio": ratio,
        })

    if not rows:
        st.warning("Not enough data for azimuth quality analysis")
        return

    df = pd.DataFrame(rows)

    # Flag suspect sectors
    df["flag"] = ""
    if not df["seasonal_ratio"].isna().all():
        df.loc[df["seasonal_ratio"] < 1.1, "flag"] = "low seasonal change"
    df.loc[df["rh_std_daily"] > df["rh_std_daily"].quantile(0.85), "flag"] = df["flag"] + " high RH scatter"
    df.loc[df["amp_cv_mean"] > df["amp_cv_mean"].quantile(0.85), "flag"] = df["flag"] + " high amp CV"
    df["flag"] = df["flag"].str.strip()

    # --- Two-panel figure: polar quality map + bar chart ---
    fig, (ax_polar, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                            subplot_kw={"projection": "polar"})
    # Fix: ax_bar needs to not be polar
    fig.clear()
    ax_polar = fig.add_subplot(121, projection="polar")
    ax_bar = fig.add_subplot(122)

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    # Polar: amplitude mean by azimuth, sized by arc count
    az_rad = np.radians(df["az"] + az_step / 2)
    sizes = 20 + 80 * (df["n_arcs"] / df["n_arcs"].max())
    norm = mcolors.Normalize(vmin=df["amp_mean"].quantile(0.05),
                              vmax=df["amp_mean"].quantile(0.95))
    sc = ax_polar.scatter(az_rad, df["amp_mean"], c=df["amp_mean"], cmap="magma",
                          norm=norm, s=sizes, alpha=0.8, edgecolors="none")

    # Mark suspect sectors
    suspect = df[df["flag"] != ""]
    if not suspect.empty:
        sus_rad = np.radians(suspect["az"] + az_step / 2)
        ax_polar.scatter(sus_rad, suspect["amp_mean"], c="none", s=sizes[suspect.index],
                         edgecolors="red", linewidths=2, zorder=10)

    ax_polar.set_rlabel_position(225)
    ax_polar.set_title("Amplitude by azimuth\n(red ring = suspect)", fontsize=10, pad=15)
    fig.colorbar(sc, ax=ax_polar, label="Amp mean", shrink=0.7, pad=0.08)

    # Bar chart: RH daily std by azimuth (quality indicator)
    colors = ["#e53935" if f else "#43a047" for f in df["flag"]]
    ax_bar.bar(range(len(df)), df["rh_std_daily"], color=colors, alpha=0.7)
    ax_bar.set_xticks(range(len(df)))
    ax_bar.set_xticklabels([f"{a}" for a in df["az"]], fontsize=6, rotation=45)
    ax_bar.set_xlabel("Azimuth (deg)")
    ax_bar.set_ylabel("Daily RH std (m)")
    ax_bar.set_title("RH scatter by azimuth (red = suspect)", fontsize=10)
    ax_bar.axhline(df["rh_std_daily"].median(), color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Table
    table_rows = []
    for _, r in df.iterrows():
        table_rows.append({
            "Azimuth": f"{int(r['az'])}-{int(r['az'])+az_step}",
            "Arcs": int(r["n_arcs"]),
            "Days": int(r["n_days"]),
            "Amp": f"{r['amp_mean']:.1f}",
            "CV": f"{r['amp_cv_mean']:.3f}" if not np.isnan(r["amp_cv_mean"]) else "-",
            "RH std": f"{r['rh_std_daily']:.3f}",
            "P2N": f"{r['p2n_mean']:.1f}",
            "Ratio": f"{r['seasonal_ratio']:.2f}x" if not np.isnan(r["seasonal_ratio"]) else "-",
            "Flag": r["flag"] if r["flag"] else "",
        })

    table_df = pd.DataFrame(table_rows)

    def _flag_style(val):
        if val and len(val) > 0:
            return "background-color: #ffcdd2"
        return ""

    styled = table_df.style.map(_flag_style, subset=["Flag"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Seasonal ratio = early/mid-year amplitude. <1.1 may indicate land. High RH scatter or amp CV = noisy sector.")


def _render_frequency_performance(enriched):
    """Frequency band performance comparison over time."""
    freq_data = enriched[
        (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] != "ALL")
    ].copy()
    freq_data["date_dt"] = pd.to_datetime(freq_data["date"])

    if freq_data.empty:
        st.info("No per-frequency data available")
        return

    freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c", "E6": "#d62728",
                   "B3": "#9467bd", "OTHER": "#7f7f7f"}

    fig, (ax_count, ax_std) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for fg in sorted(freq_data["freq_group"].unique()):
        fd = freq_data[freq_data["freq_group"] == fg].sort_values("date_dt")
        color = freq_colors.get(fg, "#7f7f7f")
        ax_count.plot(fd["date_dt"], fd["rh_count"], color=color, label=fg,
                      linewidth=1, alpha=0.7)
        ax_std.plot(fd["date_dt"], fd["rh_std"], color=color, label=fg,
                    linewidth=1, alpha=0.7)

    ax_count.set_ylabel("Daily arc count")
    ax_count.set_title("Arc count by frequency band", fontsize=10)
    ax_count.legend(loc="upper right", fontsize=8, ncol=5)

    ax_std.set_ylabel("RH std (m)")
    ax_std.set_xlabel("Date")
    ax_std.set_title("RH scatter by frequency band (lower = more precise)", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    rows = []
    for fg in sorted(freq_data["freq_group"].unique()):
        fd = freq_data[freq_data["freq_group"] == fg]
        rows.append({
            "Band": fg,
            "Total arcs": int(fd["rh_count"].sum()),
            "Days active": len(fd),
            "Avg arcs/day": f"{fd['rh_count'].mean():.0f}",
            "Mean RH std": f"{fd['rh_std'].mean():.3f}m",
            "Mean amp": f"{fd['amp_mean'].mean():.1f}",
            "Mean P2N": f"{fd['p2n_mean'].mean():.1f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_arc_quality(per_arc):
    """NumbOf and DelT distribution analysis."""
    has_numbof = "NumbOf" in per_arc.columns
    has_delt = "DelT" in per_arc.columns

    if not has_numbof and not has_delt:
        st.info("No arc quality fields (NumbOf, DelT) available")
        return

    n_plots = sum([has_numbof, has_delt])
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if has_numbof:
        ax = axes[idx]
        ax.hist(per_arc["NumbOf"], bins=50, color="#1f77b4", alpha=0.7, edgecolor="none")
        median_nof = per_arc["NumbOf"].median()
        ax.axvline(median_nof, color="red", linestyle="--", linewidth=1)
        ax.text(median_nof + 2, ax.get_ylim()[1] * 0.9, f"median={median_nof:.0f}",
                fontsize=9, color="red")
        ax.set_xlabel("NumbOf (samples per arc)")
        ax.set_ylabel("Count")
        ax.set_title("Arc quality distribution", fontsize=10)

        # What % would be filtered at various thresholds
        for thresh in [30, 50, 70]:
            pct = 100 * (per_arc["NumbOf"] < thresh).sum() / len(per_arc)
            if pct > 0.5:
                ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
                ax.text(thresh, ax.get_ylim()[1] * 0.05, f"<{thresh}: {pct:.0f}%",
                        fontsize=7, color="gray", rotation=90, va="bottom")
        idx += 1

    if has_delt:
        ax = axes[idx]
        ax.hist(per_arc["DelT"], bins=50, color="#2ca02c", alpha=0.7, edgecolor="none")
        median_dt = per_arc["DelT"].median()
        ax.axvline(median_dt, color="red", linestyle="--", linewidth=1)
        ax.text(median_dt + 1, ax.get_ylim()[1] * 0.9, f"median={median_dt:.1f}min",
                fontsize=9, color="red")
        ax.set_xlabel("Arc duration (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("Arc duration distribution", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_daily_quality_timeline(enriched):
    """Daily quality timeline: arc count, amp_cv, rh_std over time."""
    pooled = enriched[
        (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
    ].copy()
    pooled["date_dt"] = pd.to_datetime(pooled["date"])
    pooled = pooled.sort_values("date_dt")

    if pooled.empty:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # Arc count
    ax1.bar(pooled["date_dt"], pooled["rh_count"], width=1, color="#1f77b4", alpha=0.6)
    ax1.axhline(pooled["rh_count"].median(), color="red", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Arc count")
    ax1.set_title("Daily data quality timeline", fontsize=11)

    # Amp CV
    ax2.plot(pooled["date_dt"], pooled["amp_cv"], color="#ff7f0e", linewidth=1)
    ax2.fill_between(pooled["date_dt"], 0, pooled["amp_cv"], color="#ff7f0e", alpha=0.15)
    ax2.set_ylabel("Amplitude CV")

    # RH std
    ax3.plot(pooled["date_dt"], pooled["rh_std"], color="#2ca02c", linewidth=1)
    ax3.fill_between(pooled["date_dt"], 0, pooled["rh_std"], color="#2ca02c", alpha=0.15)
    ax3.set_ylabel("RH std (m)")
    ax3.set_xlabel("Date")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_annual_qc(per_arc, station_id):
    """Aggregated quickLook-style view: RH, PkNoise, Amplitude vs azimuth for full year.

    Mirrors the 3-panel quickLook summary but with all days overlaid, showing
    systematic azimuth-dependent patterns and QC threshold lines from config.
    """
    import json

    cfg_path = project_root / "config" / f"{station_id.lower()}.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    req_amp = cfg.get("reqAmp", [5])[0] if isinstance(cfg.get("reqAmp"), list) else cfg.get("reqAmp", 5)
    pk_noise_min = cfg.get("PkNoise", 0)
    min_h = cfg.get("minH", 0)
    max_h = cfg.get("maxH", 100)

    # Compute daily median RH per arc for outlier flagging
    daily_median = per_arc.groupby("date")["RH"].median()
    daily_mad = per_arc.groupby("date")["RH"].apply(
        lambda x: (x - x.median()).abs().median()
    )

    pa = per_arc.copy()
    pa["daily_rh_median"] = pa["date"].map(daily_median)
    pa["daily_rh_mad"] = pa["date"].map(daily_mad)
    pa["rh_deviation"] = (pa["RH"] - pa["daily_rh_median"]).abs()
    # Outlier: deviates from daily median by more than 3x MAD (minimum MAD = 0.1m)
    mad_threshold = pa["daily_rh_mad"].clip(lower=0.1) * 3
    pa["is_outlier"] = pa["rh_deviation"] > mad_threshold

    n_outlier = pa["is_outlier"].sum()
    n_total = len(pa)
    good = pa[~pa["is_outlier"]]
    bad = pa[pa["is_outlier"]]

    fig, (ax_rh, ax_pn, ax_amp) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Subsample for rendering speed (keep all outliers, sample good arcs)
    if len(good) > 10000:
        good_sample = good.sample(10000, random_state=42)
    else:
        good_sample = good

    # --- RH vs Azimuth ---
    ax_rh.scatter(good_sample["Azim"], good_sample["RH"], c="#1f77b4", s=2, alpha=0.08,
                  edgecolors="none", rasterized=True, label=f"good ({n_total - n_outlier})")
    if len(bad) > 0:
        ax_rh.scatter(bad["Azim"], bad["RH"], c="#e53935", s=6, alpha=0.3,
                      edgecolors="none", rasterized=True, label=f"outlier ({n_outlier})")
    ax_rh.axhline(min_h, color="black", linewidth=1, linestyle="--")
    ax_rh.axhline(max_h, color="black", linewidth=1, linestyle="--")
    ax_rh.set_ylabel("Refl. Ht. (m)")
    ax_rh.set_title(f"Annual QC — {station_id} ({n_total} arcs, {n_outlier} outliers flagged)",
                     fontsize=11)
    ax_rh.legend(fontsize=7, loc="upper right")
    # Show the dense band clearly
    rh_p02 = per_arc["RH"].quantile(0.02)
    rh_p98 = per_arc["RH"].quantile(0.98)
    ax_rh.set_ylim(max(0, rh_p02 - 1), rh_p98 + 2)

    # --- PkNoise vs Azimuth ---
    ax_pn.scatter(good_sample["Azim"], good_sample["PkNoise"], c="#1f77b4", s=2, alpha=0.08,
                  edgecolors="none", rasterized=True)
    if len(bad) > 0:
        ax_pn.scatter(bad["Azim"], bad["PkNoise"], c="#e53935", s=6, alpha=0.3,
                      edgecolors="none", rasterized=True)
    ax_pn.axhline(pk_noise_min, color="black", linewidth=1.5, linestyle="--",
                  label=f"QC value used: {pk_noise_min}")
    ax_pn.set_ylabel("peak2noise")
    ax_pn.legend(fontsize=7, loc="upper right")

    # --- Amplitude vs Azimuth ---
    ax_amp.scatter(good_sample["Azim"], good_sample["Amp"], c="#1f77b4", s=2, alpha=0.08,
                   edgecolors="none", rasterized=True)
    if len(bad) > 0:
        ax_amp.scatter(bad["Azim"], bad["Amp"], c="#e53935", s=6, alpha=0.3,
                       edgecolors="none", rasterized=True)
    ax_amp.axhline(req_amp, color="black", linewidth=1.5, linestyle="--",
                   label=f"QC value used: {req_amp}")
    ax_amp.set_ylabel("Spectral Peak Ampl.")
    ax_amp.set_xlabel("Azimuth (degrees)")
    ax_amp.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Stats summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Arcs passed QC", f"{n_total - n_outlier:,}")
    c2.metric("Outliers (>3x daily MAD)", f"{n_outlier:,}")
    c3.metric("Outlier rate", f"{100 * n_outlier / n_total:.1f}%")


def _render_filter_analysis(per_arc, station_id):
    """Identify RH outliers, show where they originate, and suggest tighter config."""
    import json

    cfg_path = project_root / "config" / f"{station_id.lower()}.json"
    if not cfg_path.exists():
        st.info(f"No config file found at {cfg_path}")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    min_h = cfg.get("minH", 0)
    max_h = cfg.get("maxH", 100)
    req_amp = cfg.get("reqAmp", [5])[0] if isinstance(cfg.get("reqAmp"), list) else cfg.get("reqAmp", 5)
    pk_noise_min = cfg.get("PkNoise", 0)
    az_range = cfg.get("azval2", [0, 360])

    n_total = len(per_arc)
    rh_p01 = per_arc["RH"].quantile(0.01)
    rh_p50 = per_arc["RH"].median()
    rh_p95 = per_arc["RH"].quantile(0.95)
    rh_p99 = per_arc["RH"].quantile(0.99)

    # --- Identify outliers: arcs deviating >3x daily MAD from daily median ---
    daily_median = per_arc.groupby("date")["RH"].median()
    daily_mad = per_arc.groupby("date")["RH"].apply(
        lambda x: (x - x.median()).abs().median()
    )
    pa = per_arc.copy()
    pa["daily_rh_median"] = pa["date"].map(daily_median)
    pa["daily_rh_mad"] = pa["date"].map(daily_mad).clip(lower=0.1)
    pa["rh_deviation"] = (pa["RH"] - pa["daily_rh_median"]).abs()
    outlier_mask = pa["rh_deviation"] > (pa["daily_rh_mad"] * 3)

    outliers = pa[outlier_mask]
    good = pa[~outlier_mask]

    # Baseline precision
    baseline_daily_std = per_arc.groupby("date")["RH"].std().mean()
    cleaned_daily_std = good.groupby("date")["RH"].std().mean()

    # --- Figure: outlier anatomy ---
    fig = plt.figure(figsize=(14, 8))
    ax_rh_az = fig.add_subplot(221, projection="polar")
    ax_rh_hist = fig.add_subplot(222)
    ax_amp_pn = fig.add_subplot(223)
    ax_combined = fig.add_subplot(224)

    # Top-left: Polar view — all arcs gray, outliers red
    ax_rh_az.set_theta_zero_location("N")
    ax_rh_az.set_theta_direction(-1)
    ax_rh_az.scatter(np.radians(good["Azim"]), good["RH"], c="#cccccc", s=2,
                     alpha=0.1, edgecolors="none", rasterized=True)
    if len(outliers) > 0:
        ax_rh_az.scatter(np.radians(outliers["Azim"]), outliers["RH"], c="red", s=15,
                         alpha=0.7, edgecolors="none", zorder=5, label=f"outliers ({len(outliers)})")
        ax_rh_az.legend(fontsize=7, loc="upper right")
    ax_rh_az.set_title(f"RH outliers by azimuth\n(red = beyond p1/p99)", fontsize=9, pad=12)
    ax_rh_az.set_rlabel_position(225)

    # Top-right: RH histogram with suggested boundaries
    ax_rh_hist.hist(per_arc["RH"], bins=150, color="#1f77b4", alpha=0.5, edgecolor="none",
                    label="all arcs")
    ax_rh_hist.axvline(min_h, color="red", linewidth=2, linestyle="--", label=f"config minH={min_h}")
    ax_rh_hist.axvline(max_h, color="red", linewidth=2, linestyle="--", label=f"config maxH={max_h}")
    ax_rh_hist.axvline(rh_p01, color="green", linewidth=1.5, linestyle="-",
                       label=f"suggested minH={rh_p01:.1f} (p1)")
    ax_rh_hist.axvline(rh_p99, color="green", linewidth=1.5, linestyle="-",
                       label=f"suggested maxH={rh_p99:.1f} (p99)")
    ax_rh_hist.set_xlabel("RH (m)")
    ax_rh_hist.set_ylabel("Count")
    ax_rh_hist.set_title("RH distribution — current vs suggested bounds", fontsize=9)
    ax_rh_hist.legend(fontsize=6)
    # Zoom to where data actually lives
    ax_rh_hist.set_xlim(max(0, rh_p01 - 2), rh_p99 + 5)

    # Bottom-left: Amplitude vs PkNoise — outliers highlighted
    ax_amp_pn.scatter(good["Amp"], good["PkNoise"], c="#cccccc", s=2, alpha=0.1,
                      edgecolors="none", rasterized=True)
    if len(outliers) > 0:
        ax_amp_pn.scatter(outliers["Amp"], outliers["PkNoise"], c="red", s=10,
                          alpha=0.6, edgecolors="none", zorder=5)
    ax_amp_pn.axvline(req_amp, color="red", linewidth=1.5, linestyle="--",
                      label=f"config reqAmp={req_amp}")
    ax_amp_pn.axhline(pk_noise_min, color="red", linewidth=1.5, linestyle="--",
                      label=f"config PkNoise={pk_noise_min}")
    # Suggested thresholds
    if len(outliers) > 10:
        out_amp_p50 = outliers["Amp"].median()
        out_pn_p50 = outliers["PkNoise"].median()
        ax_amp_pn.axvline(out_amp_p50, color="orange", linewidth=1, linestyle=":",
                          label=f"outlier median amp={out_amp_p50:.0f}")
    ax_amp_pn.set_xlabel("Amplitude")
    ax_amp_pn.set_ylabel("PkNoise")
    ax_amp_pn.set_title("Amp vs PkNoise — where do outliers sit?", fontsize=9)
    ax_amp_pn.legend(fontsize=6)

    # Bottom-right: Combined filter impact
    # Try progressively tighter combined filters and show precision improvement
    filter_combos = [
        ("Current config", per_arc),
        (f"RH [{rh_p01:.1f}, {rh_p99:.1f}]", per_arc[(per_arc["RH"] >= rh_p01) & (per_arc["RH"] <= rh_p99)]),
        (f"+ Amp >= {max(req_amp, 8)}", per_arc[(per_arc["RH"] >= rh_p01) & (per_arc["RH"] <= rh_p99) & (per_arc["Amp"] >= max(req_amp, 8))]),
        (f"+ PkNoise >= {max(pk_noise_min, 3)}", per_arc[(per_arc["RH"] >= rh_p01) & (per_arc["RH"] <= rh_p99) & (per_arc["Amp"] >= max(req_amp, 8)) & (per_arc["PkNoise"] >= max(pk_noise_min, 3))]),
    ]

    labels = []
    stds = []
    counts = []
    for label, filtered in filter_combos:
        if len(filtered) > 50:
            daily = filtered.groupby("date")["RH"].std()
            labels.append(label)
            stds.append(daily.mean())
            counts.append(len(filtered))

    if labels:
        x = np.arange(len(labels))
        bars = ax_combined.barh(x, stds, color=["#1f77b4", "#2ca02c", "#2ca02c", "#2ca02c"],
                                alpha=0.7)
        ax_combined.set_yticks(x)
        ax_combined.set_yticklabels(labels, fontsize=7)
        ax_combined.set_xlabel("Daily RH std (m)")
        ax_combined.set_title("Cumulative filter tightening — precision impact", fontsize=9)
        # Annotate arc counts
        for i, (s, n) in enumerate(zip(stds, counts)):
            pct = 100 * n / n_total
            ax_combined.text(s + 0.002, i, f"{n:,} arcs ({pct:.0f}%)", va="center", fontsize=7)

    fig.suptitle(f"Filter Analysis — {station_id}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Outlier source breakdown ---
    if len(outliers) > 0:
        st.markdown("**Where do outliers originate?**")

        out_az = outliers.copy()
        out_az["az_10"] = (out_az["Azim"] // 10 * 10).astype(int)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("By azimuth (10-deg bins):")
            az_counts = out_az.groupby("az_10").size().reset_index(name="outliers")
            total_by_az = per_arc.copy()
            total_by_az["az_10"] = (total_by_az["Azim"] // 10 * 10).astype(int)
            total_counts = total_by_az.groupby("az_10").size().reset_index(name="total")
            merged = az_counts.merge(total_counts, on="az_10")
            merged["outlier_rate"] = (100 * merged["outliers"] / merged["total"]).round(1)
            merged = merged.sort_values("outlier_rate", ascending=False)
            merged.columns = ["Azimuth", "Outliers", "Total", "Rate %"]
            merged["Azimuth"] = merged["Azimuth"].apply(lambda x: f"{x}-{x+10}")
            st.dataframe(merged.head(10), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("By frequency:")
            freq_counts = outliers.groupby("freq_group").size().reset_index(name="outliers")
            freq_total = per_arc.groupby("freq_group").size().reset_index(name="total")
            freq_m = freq_counts.merge(freq_total, on="freq_group")
            freq_m["outlier_rate"] = (100 * freq_m["outliers"] / freq_m["total"]).round(1)
            freq_m = freq_m.sort_values("outlier_rate", ascending=False)
            freq_m.columns = ["Band", "Outliers", "Total", "Rate %"]
            st.dataframe(freq_m, use_container_width=True, hide_index=True)

    # --- Suggested config ---
    st.markdown("**Suggested config changes:**")
    suggestions = []
    if rh_p99 < max_h * 0.7:
        suggestions.append(f"`maxH`: {max_h} -> **{rh_p99:.1f}** (p99 of data, current allows {max_h - rh_p99:.0f}m of empty range)")
    if rh_p01 > min_h + 0.5:
        suggestions.append(f"`minH`: {min_h} -> **{rh_p01:.1f}** (p1 of data)")
    if req_amp < 8 and per_arc["Amp"].median() > 15:
        suggestions.append(f"`reqAmp`: {req_amp} -> **8** (median amp is {per_arc['Amp'].median():.0f}, current threshold accepts very weak signals)")
    if pk_noise_min < 2:
        low_pn = (per_arc["PkNoise"] < 2).sum()
        if low_pn < n_total * 0.05:
            suggestions.append(f"`PkNoise`: {pk_noise_min} -> **2.0** (only {low_pn} arcs below 2.0)")

    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
        st.caption("These suggestions are based on the data distribution. "
                   "Tighter filters mean fewer arcs but more precise RH estimates. "
                   "Changes require reprocessing with gnssir.")
    else:
        st.caption("Current filters appear well-tuned for this dataset.")


def render_diagnostics_tab(station_id="UMNQ", year=2025, data_dict=None):
    """Render data-driven diagnostics tab."""
    st.header("Diagnostics")
    st.warning(
        "**This tab is deprecated.** "
        "Use **Data Quality** and **Validation** instead. "
        "This tab will be removed in a future update."
    )

    enriched = load_enriched(station_id, year)
    per_arc = load_per_arc(station_id, year)

    if per_arc is None and enriched is None:
        st.warning(
            f"No per-arc or enriched data for {station_id} {year}. "
            f"Run processing first, then `python scripts/backfill_enriched.py`."
        )
        # Fall back to QuickLook PNGs if available
        _render_quicklook_section(station_id, year)
        return

    # --- Azimuth Quality ---
    st.subheader("Azimuth Sector Quality")
    st.caption("Which directions produce reliable data? Red-flagged sectors may be looking at land or have multipath issues.")
    if per_arc is not None:
        _render_azimuth_quality(per_arc)

    # --- Daily Quality Timeline ---
    st.subheader("Daily Quality Timeline")
    if enriched is not None:
        _render_daily_quality_timeline(enriched)

    # --- Frequency Performance ---
    st.subheader("Frequency Band Performance")
    st.caption("Compares arc count, RH precision, and signal strength across GNSS frequency bands.")
    if enriched is not None:
        _render_frequency_performance(enriched)

    # --- Arc Quality ---
    st.subheader("Arc Quality Distribution")
    if per_arc is not None:
        _render_arc_quality(per_arc)

    # --- Annual QC Summary (aggregated quickLook) ---
    st.subheader("Annual QC Summary")
    st.caption("Same view as quickLook summary plots, but aggregated over the full year. Shows systematic patterns invisible in single-day views.")
    if per_arc is not None:
        _render_annual_qc(per_arc, station_id)

    # --- Filter Effectiveness ---
    st.subheader("Filter Effectiveness")
    st.caption("Identifies bad arcs, shows where they originate, and suggests config changes.")
    if per_arc is not None:
        _render_filter_analysis(per_arc, station_id)

    # --- QuickLook PNGs (keep as optional section) ---
    with st.expander("QuickLook diagnostic plots (LSP/summary PNGs)"):
        _render_quicklook_section(station_id, year)


def _render_quicklook_section(station_id, year, key_prefix="ql"):
    """Original QuickLook PNG viewer, now as a subsection."""
    try:
        available_days = get_available_diagnostic_days(station_id, year)
    except Exception:
        available_days = []

    if not available_days:
        st.info(f"No QuickLook plots found for {station_id} {year}")
        return

    st.caption(f"{len(available_days)} days with QuickLook plots")

    selected_doy = st.selectbox(
        "Select DOY", available_days, index=len(available_days) - 1,
        key=f"{key_prefix}_doy_{station_id}_{year}",
    )

    plot_files = get_quicklook_plots_for_day(station_id, year, selected_doy)
    if not plot_files:
        st.warning(f"No plots for DOY {selected_doy}")
        return

    selected_date = doy_to_date(year, selected_doy)
    st.markdown(f"**{selected_date.strftime('%Y-%m-%d')} (DOY {selected_doy})**")

    col1, col2 = st.columns(2)
    with col1:
        if "lsp" in plot_files:
            st.image(Image.open(plot_files["lsp"]), caption="LSP", use_container_width=True)
    with col2:
        if "summary" in plot_files:
            st.image(Image.open(plot_files["summary"]), caption="Summary", use_container_width=True)


__all__ = ["render_diagnostics_tab"]
