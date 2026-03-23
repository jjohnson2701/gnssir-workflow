# ABOUTME: Per-Arc Explorer dashboard tab — subdaily GNSS-IR data by azimuth, frequency, amplitude
# ABOUTME: Phase A: Panels 1-3 (arc timeline, frequency comparison, amplitude distribution)
# ABOUTME: Phase B: Panels 4-5 (S1 backscatter map + cross-validation time series)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from dashboard_components.data_loader import load_per_arc, load_enriched

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

AZ_BIN_LABELS = {0: "0-90 (N-E)", 1: "90-180 (E-S)", 2: "180-270 (S-W)", 3: "270-360 (W-N)"}

# amp_cv thresholds for ice/water classification
AMP_CV_ICE = 0.20
AMP_CV_WATER = 0.35


def _load_s1_index(station):
    """Load S1 Fresnel index if it exists."""
    path = PROJECT_ROOT / "data" / station / "s1_fresnel" / f"{station}_s1_fresnel_index.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_s1_matched(station, year):
    """Load S1-GNSSIR matched parquet if it exists."""
    path = (
        PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_s1_gnssir_matched.parquet"
    )
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 0: Multi-Day Subdaily Heatmap
# ──────────────────────────────────────────────────────────────────────────────


def _classify_constellation(sat):
    """Map satellite PRN to constellation name."""
    if 1 <= sat <= 32:
        return "GPS"
    elif 101 <= sat <= 132:
        return "GLONASS"
    elif 201 <= sat <= 250:
        return "Galileo"
    elif sat >= 301:
        return "BeiDou"
    return "Other"


def _render_panel0_multiday_heatmap(per_arc, selected_date, heatmap_field,
                                     min_numbof, constellation_filter="All"):
    """2D heatmap: date (Y) x UTC hour (X), colored by RH/Amp/arc count.

    Reveals seasonal patterns in sub-daily structure — freeze-up/break-up
    transitions become visible as color shifts across rows.
    Diagonal striping = satellite orbital repeat (GPS -3.3 min/day,
    GLONASS +9.6 min/day sidereal drift).
    """
    df = per_arc.copy()
    if min_numbof > 0 and "NumbOf" in df.columns:
        df = df[df["NumbOf"] >= min_numbof]

    # Constellation filter
    if constellation_filter != "All" and "sat" in df.columns:
        df["constellation"] = df["sat"].apply(_classify_constellation)
        df = df[df["constellation"] == constellation_filter]

    if df.empty:
        st.warning("No data after filtering")
        return

    df["date_dt"] = pd.to_datetime(df["date"])
    df["utc_bin"] = df["UTCtime"].round().astype(int).clip(0, 23)

    # Aggregate by date x UTC hour
    if heatmap_field == "arc_count":
        pivot = df.groupby(["date", "utc_bin"]).size().reset_index(name="value")
    elif heatmap_field == "amp_cv":
        grp = df.groupby(["date", "utc_bin"])["Amp"]
        pivot = grp.agg(["mean", "std"]).reset_index()
        pivot["value"] = pivot["std"] / pivot["mean"].replace(0, np.nan)
        pivot = pivot[["date", "utc_bin", "value"]]
    else:
        pivot = df.groupby(["date", "utc_bin"])[heatmap_field].mean().reset_index(name="value")

    if pivot.empty:
        st.warning("No data for heatmap")
        return

    # Build 2D array: rows = dates, cols = UTC hours 0-23
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    grid = np.full((len(all_dates), 24), np.nan)
    for _, row in pivot.iterrows():
        grid[date_to_idx[row["date"]], int(row["utc_bin"])] = row["value"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_dates) * 0.04)))

    # Choose colormap based on field
    cmap_map = {"RH": "viridis_r", "Amp": "magma", "amp_cv": "RdYlBu_r", "arc_count": "YlOrRd"}
    cmap = cmap_map.get(heatmap_field, "viridis")

    # Robust vmin/vmax
    vals = grid[~np.isnan(grid)]
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")

    # Y-axis: dates (show monthly ticks)
    n_labels = min(12, len(all_dates))
    tick_step = max(1, len(all_dates) // n_labels)
    y_ticks = list(range(0, len(all_dates), tick_step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([all_dates[i][:10] for i in y_ticks], fontsize=7)

    # X-axis: UTC hours
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Date")

    # Highlight selected date row
    if selected_date in date_to_idx:
        sel_idx = date_to_idx[selected_date]
        ax.axhline(sel_idx, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(24.3, sel_idx, selected_date[:10], fontsize=7, color="red",
                va="center", clip_on=False)

    label_map = {"RH": "Reflector Height (m)", "Amp": "Amplitude (v/v)",
                 "amp_cv": "Amplitude CV", "arc_count": "Arc Count"}
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label_map.get(heatmap_field, heatmap_field), fontsize=9)

    # Title with constellation + sidereal drift note
    title_parts = [f"Subdaily Structure — {heatmap_field}"]
    if constellation_filter != "All":
        drift_note = {"GPS": "~-3.3 min/day", "GLONASS": "~+9.6 min/day",
                      "Galileo": "~+2.6 min/day"}.get(constellation_filter, "")
        title_parts.append(f"[{constellation_filter} {drift_note}]")
    if min_numbof > 0:
        title_parts.append(f"(NumbOf >= {min_numbof})")
    ax.set_title(" ".join(title_parts), fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 0b: Azimuth-Resolved Amplitude Map
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel0b_azimuth_map(per_arc, selected_date, min_numbof):
    """Heatmap: date (Y) x azimuth (X), colored by amplitude mean or CV.

    Reveals which azimuths see ice vs land vs water. Sectors with no seasonal
    amplitude change are flagged as potential land contamination.
    """
    df = per_arc.copy()
    if min_numbof > 0 and "NumbOf" in df.columns:
        df = df[df["NumbOf"] >= min_numbof]

    if df.empty:
        st.warning("No data after filtering")
        return

    # 10-degree azimuth bins
    az_step = 10
    df["az_bin"] = (df["Azim"] // az_step * az_step).astype(int)
    df["date_dt"] = pd.to_datetime(df["date"])

    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    az_bins = sorted(df["az_bin"].unique())
    az_to_col = {a: i for i, a in enumerate(az_bins)}

    # --- Two heatmaps side by side: Amp mean + Amp CV ---
    fig, (ax_amp, ax_cv) = plt.subplots(1, 2, figsize=(14, max(4, len(all_dates) * 0.04)),
                                         sharey=True)

    grid_amp = np.full((len(all_dates), len(az_bins)), np.nan)
    grid_cv = np.full((len(all_dates), len(az_bins)), np.nan)

    for (date, az), grp in df.groupby(["date", "az_bin"]):
        r = date_to_idx.get(date)
        c = az_to_col.get(az)
        if r is not None and c is not None and len(grp) >= 3:
            grid_amp[r, c] = grp["Amp"].mean()
            mean_amp = grp["Amp"].mean()
            if mean_amp > 0:
                grid_cv[r, c] = grp["Amp"].std() / mean_amp

    # Amp heatmap
    vals = grid_amp[~np.isnan(grid_amp)]
    vmin_a, vmax_a = (np.percentile(vals, [5, 95]) if len(vals) > 0 else (0, 1))
    im_amp = ax_amp.imshow(grid_amp, aspect="auto", cmap="magma", vmin=vmin_a, vmax=vmax_a,
                            origin="upper", interpolation="nearest")
    plt.colorbar(im_amp, ax=ax_amp, shrink=0.7, pad=0.02, label="Amplitude")
    ax_amp.set_title("Amplitude by azimuth", fontsize=10)

    # CV heatmap
    im_cv = ax_cv.imshow(grid_cv, aspect="auto", cmap="RdYlBu_r", vmin=0.15, vmax=0.55,
                          origin="upper", interpolation="nearest")
    plt.colorbar(im_cv, ax=ax_cv, shrink=0.7, pad=0.02, label="Amp CV")
    ax_cv.set_title("Amplitude CV by azimuth", fontsize=10)

    # Axis labels
    for ax in [ax_amp, ax_cv]:
        ax.set_xticks(range(len(az_bins)))
        ax.set_xticklabels([f"{a}" for a in az_bins], fontsize=6, rotation=45)
        ax.set_xlabel("Azimuth (deg)")

        if selected_date in date_to_idx:
            sel_idx = date_to_idx[selected_date]
            ax.axhline(sel_idx, color="red", linewidth=1, linestyle="--", alpha=0.8)

    # Y-axis: dates
    n_labels = min(12, len(all_dates))
    tick_step = max(1, len(all_dates) // n_labels)
    y_ticks = list(range(0, len(all_dates), tick_step))
    ax_amp.set_yticks(y_ticks)
    ax_amp.set_yticklabels([all_dates[i][:10] for i in y_ticks], fontsize=7)
    ax_amp.set_ylabel("Date")

    fig.suptitle("Azimuth-Resolved View — which directions see ice?", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Seasonal amplitude ratio per azimuth (flag land contamination) ---
    df["month"] = df["date_dt"].dt.month
    # Define "winter" and "summer" from available months
    available_months = sorted(df["month"].unique())
    # Use first 2 months as "early" and middle months as "summer"
    early_months = available_months[:2]
    mid_months = [m for m in available_months if 6 <= m <= 8]
    if not mid_months:
        mid_months = available_months[len(available_months)//2 : len(available_months)//2 + 2]

    early = df[df["month"].isin(early_months)]
    summer = df[df["month"].isin(mid_months)]

    if not early.empty and not summer.empty:
        rows = []
        for az in az_bins:
            e = early[early["az_bin"] == az]
            s = summer[summer["az_bin"] == az]
            if len(e) >= 10 and len(s) >= 10:
                ratio = e["Amp"].mean() / s["Amp"].mean()
                e_cv = e.groupby("date")["Amp"].agg(lambda x: x.std()/x.mean() if x.mean() > 0 else np.nan).mean()
                s_cv = s.groupby("date")["Amp"].agg(lambda x: x.std()/x.mean() if x.mean() > 0 else np.nan).mean()
                flag = ""
                if ratio < 1.1:
                    flag = "no seasonal change"
                elif ratio > 1.5:
                    flag = "strong seasonal signal"
                rows.append({
                    "Azimuth": f"{az}-{az+az_step}",
                    f"Early ({','.join(str(m) for m in early_months)})": f"{e['Amp'].mean():.1f}",
                    f"Summer ({','.join(str(m) for m in mid_months)})": f"{s['Amp'].mean():.1f}",
                    "Ratio": f"{ratio:.2f}x",
                    f"CV early": f"{e_cv:.3f}",
                    f"CV summer": f"{s_cv:.3f}",
                    "Flag": flag,
                })

        if rows:
            st.markdown("**Seasonal amplitude ratio by azimuth**")
            table = pd.DataFrame(rows)

            def _flag_color(val):
                if val == "no seasonal change":
                    return "background-color: #ffcdd2"  # red — suspect
                elif val == "strong seasonal signal":
                    return "background-color: #c8e6c9"  # green
                return ""

            styled = table.style.map(_flag_color, subset=["Flag"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                "Ratio = early/summer amplitude. "
                "Sectors with ratio < 1.1 (red) may be looking at land. "
                "Sectors with ratio > 1.5 (green) show strong seasonal ice/water signal."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Panel 1: Subdaily Arc Timeline
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel1_arc_polar(per_arc, selected_date, station_id, min_numbof=0):
    """Two polar plots showing individual arcs — no sector aggregation.

    Left:  Each arc at its reflection point (azimuth=angle, distance=radius),
           colored by amplitude. Shows where high/low amplitude arcs land.
    Right: Amplitude vs azimuth (angle=azimuth, radius=amplitude),
           colored by frequency. Shows amplitude structure without binning.
    """
    day = per_arc[per_arc["date"] == selected_date]
    if day.empty:
        st.warning(f"No arcs on {selected_date}")
        return

    has_numbof = "NumbOf" in day.columns
    if has_numbof and min_numbof > 0:
        good = day[day["NumbOf"] >= min_numbof]
        weak = day[day["NumbOf"] < min_numbof]
    else:
        good = day
        weak = day.iloc[:0]

    # Reflection geometry
    elev_mid = (good["eminO"] + good["emaxO"]) / 2.0
    refl_dist = good["RH"] / np.tan(np.radians(elev_mid))
    az_rad = np.radians(good["Azim"])

    # Dot sizes from NumbOf
    if has_numbof and len(good) > 0:
        nof = good["NumbOf"].values.astype(float)
        nof_min, nof_max = nof.min(), nof.max()
        sizes = 10 + 50 * (nof - nof_min) / max(nof_max - nof_min, 1)
    else:
        sizes = np.full(len(good), 25.0)

    fig = plt.figure(figsize=(14, 6))
    ax_refl = fig.add_subplot(121, projection="polar")
    ax_amp = fig.add_subplot(122, projection="polar")

    for ax in [ax_refl, ax_amp]:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for boundary in [0, 90, 180, 270]:
            ax.axvline(np.radians(boundary), color="#cccccc", linewidth=0.5, linestyle="--")

    # -- Left: Reflection points colored by amplitude --
    amp_vals = good["Amp"].values
    if len(amp_vals) > 0:
        vmin_a = np.percentile(amp_vals, 5) if len(amp_vals) > 10 else amp_vals.min()
        vmax_a = np.percentile(amp_vals, 95) if len(amp_vals) > 10 else amp_vals.max()
    else:
        vmin_a, vmax_a = 0, 1
    norm = mcolors.Normalize(vmin=vmin_a, vmax=vmax_a)

    sc = ax_refl.scatter(az_rad, refl_dist, c=amp_vals, cmap="magma", norm=norm,
                         s=sizes, alpha=0.8, edgecolors="none", zorder=5)
    fig.colorbar(sc, ax=ax_refl, label="Amplitude", shrink=0.7, pad=0.08)

    # Weak arcs
    if len(weak) > 0:
        w_elev = (weak["eminO"] + weak["emaxO"]) / 2.0
        w_dist = weak["RH"] / np.tan(np.radians(w_elev))
        ax_refl.scatter(np.radians(weak["Azim"]), w_dist, c="#999999", s=8, alpha=0.2,
                        marker="x", linewidths=0.5)

    ax_refl.plot(0, 0, "r*", markersize=10, zorder=10)
    ax_refl.set_rlabel_position(225)
    ax_refl.set_title("Reflection points (color = amplitude, size = quality)", fontsize=10, pad=15)

    # -- Right: Amplitude vs azimuth, colored by frequency --
    freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c", "E6": "#d62728",
                   "B3": "#9467bd", "OTHER": "#7f7f7f"}
    for fg in sorted(good["freq_group"].unique()):
        fg_data = good[good["freq_group"] == fg]
        ax_amp.scatter(np.radians(fg_data["Azim"]), fg_data["Amp"],
                       c=freq_colors.get(fg, "#7f7f7f"), s=15, alpha=0.6,
                       label=fg, edgecolors="none")

    ax_amp.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8,
                  bbox_to_anchor=(1.15, 1.1))
    ax_amp.set_rlabel_position(225)
    ax_amp.set_title("Amplitude vs azimuth (color = frequency)", fontsize=10, pad=15)

    # Summary
    n_good = len(good)
    n_weak = len(weak)
    quality_text = f"{n_good} arcs"
    if n_weak > 0:
        quality_text += f" ({n_weak} below NumbOf {min_numbof})"
    if has_numbof and n_good > 0:
        quality_text += f" | dot size = NumbOf"

    fig.suptitle(f"{selected_date} — {quality_text}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 2: Frequency Band Comparison
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel2_freq_comparison(enriched, date_range, y_col):
    """Time series of RH/WSE by frequency group, showing interfrequency divergence."""
    # Filter to per-frequency, pooled across azimuths
    freq_daily = enriched[
        (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] != "ALL")
    ].copy()
    freq_daily["date_dt"] = pd.to_datetime(freq_daily["date"])
    freq_daily = freq_daily[
        (freq_daily["date_dt"] >= pd.Timestamp(date_range[0]))
        & (freq_daily["date_dt"] <= pd.Timestamp(date_range[1]))
    ]

    if freq_daily.empty:
        st.warning("No frequency-grouped data in selected range")
        return

    freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c", "E6": "#d62728",
                   "B3": "#9467bd", "OTHER": "#7f7f7f"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1], sharex=True)

    for fg in sorted(freq_daily["freq_group"].unique()):
        fg_data = freq_daily[freq_daily["freq_group"] == fg].sort_values("date_dt")
        color = freq_colors.get(fg, "#7f7f7f")
        ax1.plot(fg_data["date_dt"], fg_data[y_col], color=color, label=fg, linewidth=1.2)
        if f"{y_col.replace('_mean', '_std')}" in fg_data.columns:
            std_col = y_col.replace("_mean", "_std")
            ax1.fill_between(
                fg_data["date_dt"],
                fg_data[y_col] - fg_data[std_col],
                fg_data[y_col] + fg_data[std_col],
                color=color, alpha=0.15,
            )

    ax1.set_ylabel(y_col.replace("_", " ").title(), fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, ncol=6)
    ax1.set_title("Frequency Band Comparison", fontsize=12)

    # Bottom subplot: interfrequency spread
    dates = sorted(freq_daily["date_dt"].unique())
    spreads = []
    for dt in dates:
        day_freqs = freq_daily[freq_daily["date_dt"] == dt]
        if len(day_freqs) >= 2:
            spread = day_freqs[y_col].max() - day_freqs[y_col].min()
            spreads.append({"date_dt": dt, "spread": spread})

    if spreads:
        spread_df = pd.DataFrame(spreads)
        ax2.bar(spread_df["date_dt"], spread_df["spread"], width=1, color="#888888", alpha=0.7)
        ax2.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="0.05m threshold")
        ax2.set_ylabel("Interfreq\nSpread (m)", fontsize=9)
        ax2.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_panel2_detail(per_arc, selected_date, y_col):
    """Single-day frequency detail: individual arcs by freq at their UTC time.

    Shows same-moment L1 vs L5 RH divergence — direct ice penetration signature.
    Different frequencies penetrate ice differently, so divergence at the same
    azimuth/time is diagnostic of surface type.
    """
    day = per_arc[per_arc["date"] == selected_date]
    if day.empty:
        st.warning(f"No arcs on {selected_date}")
        return

    freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c", "E6": "#d62728",
                   "B3": "#9467bd", "OTHER": "#7f7f7f"}

    has_numbof = "NumbOf" in day.columns

    fig, (ax_time, ax_az) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: UTC time vs RH/WSE, colored by frequency --
    for fg in sorted(day["freq_group"].unique()):
        fg_data = day[day["freq_group"] == fg]
        color = freq_colors.get(fg, "#7f7f7f")
        if has_numbof:
            nof = fg_data["NumbOf"].values.astype(float)
            nof_clip = np.clip(nof, 20, 150)
            sizes = 10 + 40 * (nof_clip - 20) / 130
        else:
            sizes = 25
        ax_time.scatter(fg_data["UTCtime"], fg_data[y_col], c=color, s=sizes,
                        alpha=0.7, label=f"{fg} ({len(fg_data)})", edgecolors="none")

    ax_time.set_xlabel("UTC Hour")
    ax_time.set_ylabel(y_col)
    ax_time.set_xlim(0, 24)
    ax_time.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    ax_time.set_title("Individual arcs by frequency", fontsize=11)

    # -- Right: Azimuth vs RH/WSE, colored by frequency --
    # This shows if L1/L5 diverge at the same azimuth (same reflection zone)
    for fg in sorted(day["freq_group"].unique()):
        fg_data = day[day["freq_group"] == fg]
        color = freq_colors.get(fg, "#7f7f7f")
        if has_numbof:
            nof = fg_data["NumbOf"].values.astype(float)
            nof_clip = np.clip(nof, 20, 150)
            sizes = 10 + 40 * (nof_clip - 20) / 130
        else:
            sizes = 25
        ax_az.scatter(fg_data["Azim"], fg_data[y_col], c=color, s=sizes,
                      alpha=0.7, label=fg, edgecolors="none")

    ax_az.set_xlabel("Azimuth (deg)")
    ax_az.set_ylabel(y_col)
    ax_az.set_title("Frequency divergence by azimuth", fontsize=11)

    # Compute per-frequency means for annotation
    freq_means = day.groupby("freq_group")[y_col].mean()
    if len(freq_means) >= 2:
        spread = freq_means.max() - freq_means.min()
        spread_label = f"Interfreq spread: {spread:.3f}m"
        color = "red" if spread > 0.05 else "gray"
        ax_az.text(0.02, 0.02, spread_label, transform=ax_az.transAxes,
                   fontsize=9, color=color, fontweight="bold" if spread > 0.05 else "normal")

    quality_note = ""
    if has_numbof:
        quality_note = " (dot size = NumbOf)"
    fig.suptitle(f"{selected_date} — {len(day)} arcs, {len(day['freq_group'].unique())} bands{quality_note}",
                 fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 3: Amplitude and Quality Distribution
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel3_amplitude(per_arc, enriched, selected_date, station_id):
    """Polar quality views for selected date — no sector aggregation.

    Left:  RH by azimuth, colored by constellation. Shows surface variability
           spatially — consistent RH = stable surface, scattered = variable.
    Right: PkNoise by azimuth, colored by frequency. High PkNoise = strong
           coherent return, low = scattered signal.
    """
    day_arcs = per_arc[per_arc["date"] == selected_date].copy()

    if day_arcs.empty:
        st.warning(f"No arcs on {selected_date}")
        return

    has_sat = "sat" in day_arcs.columns
    if has_sat:
        day_arcs["constellation"] = day_arcs["sat"].apply(_classify_constellation)

    freq_colors = {"L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c", "E6": "#d62728",
                   "B3": "#9467bd", "OTHER": "#7f7f7f"}
    const_colors = {"GPS": "#1f77b4", "Galileo": "#2ca02c", "GLONASS": "#d62728",
                    "BeiDou": "#ff7f0e", "Other": "#7f7f7f"}

    fig = plt.figure(figsize=(14, 6))
    ax_rh = fig.add_subplot(121, projection="polar")
    ax_pn = fig.add_subplot(122, projection="polar")

    for ax in [ax_rh, ax_pn]:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for boundary in [0, 90, 180, 270]:
            ax.axvline(np.radians(boundary), color="#cccccc", linewidth=0.5, linestyle="--")

    # -- Left: RH by azimuth, colored by constellation --
    if has_sat:
        for const in sorted(day_arcs["constellation"].unique()):
            cd = day_arcs[day_arcs["constellation"] == const]
            ax_rh.scatter(np.radians(cd["Azim"]), cd["RH"], c=const_colors.get(const, "#7f7f7f"),
                          s=15, alpha=0.6, label=f"{const} ({len(cd)})", edgecolors="none")
        ax_rh.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.8,
                     bbox_to_anchor=(1.15, 1.1))
    else:
        ax_rh.scatter(np.radians(day_arcs["Azim"]), day_arcs["RH"],
                      c="#1f77b4", s=15, alpha=0.6, edgecolors="none")

    ax_rh.set_rlabel_position(225)
    ax_rh.set_title("RH by azimuth (color = constellation)", fontsize=10, pad=15)

    # -- Right: PkNoise by azimuth, colored by frequency --
    for fg in sorted(day_arcs["freq_group"].unique()):
        fd = day_arcs[day_arcs["freq_group"] == fg]
        ax_pn.scatter(np.radians(fd["Azim"]), fd["PkNoise"],
                      c=freq_colors.get(fg, "#7f7f7f"), s=15, alpha=0.6,
                      label=fg, edgecolors="none")

    ax_pn.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.8,
                 bbox_to_anchor=(1.15, 1.1))
    ax_pn.set_rlabel_position(225)
    ax_pn.set_title("PkNoise by azimuth (color = frequency)", fontsize=10, pad=15)

    fig.suptitle(f"Spatial Quality — {selected_date} ({len(day_arcs)} arcs)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Per-sector metrics table (10-degree resolution) ---
    az_step = 10
    day_arcs["az_10"] = (day_arcs["Azim"] // az_step * az_step).astype(int)
    rows = []
    for az in sorted(day_arcs["az_10"].unique()):
        ad = day_arcs[day_arcs["az_10"] == az]
        if len(ad) >= 3:
            amp_cv = ad["Amp"].std() / ad["Amp"].mean() if ad["Amp"].mean() > 0 else np.nan
            rows.append({
                "Azimuth": f"{az}-{az + az_step}",
                "Arcs": len(ad),
                "Amp": f"{ad['Amp'].mean():.1f}",
                "CV": f"{amp_cv:.3f}",
                "PkNoise": f"{ad['PkNoise'].mean():.1f}",
                "RH": f"{ad['RH'].mean():.3f}",
                "RH std": f"{ad['RH'].std():.3f}",
            })

    if rows:
        table_df = pd.DataFrame(rows)
        st.markdown("**Per-azimuth metrics (10-degree resolution)**")

        def _cv_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v < AMP_CV_ICE:
                return "background-color: #bbdefb"
            elif v > AMP_CV_WATER:
                return "background-color: #c8e6c9"
            return "background-color: #fff9c4"

        styled = table_df.style.map(_cv_color, subset=["CV"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 4: S1 Backscatter Map with sector overlay
# ──────────────────────────────────────────────────────────────────────────────


def _find_nearest_s1_scene(station_id, target_date, s1_index):
    """Find the S1 scene closest to target_date. Returns (row, days_offset) or (None, None)."""
    s1 = s1_index.copy()
    s1["date_dt"] = pd.to_datetime(s1["acquisition_date"])
    target_dt = pd.Timestamp(target_date)
    s1["offset"] = (s1["date_dt"] - target_dt).abs().dt.days
    best = s1.loc[s1["offset"].idxmin()]
    return best, int(best["offset"])


def _get_s1_pass_utc(scene_name):
    """Extract UTC hour from scene name."""
    import re
    m = re.search(r"_\d{8}T(\d{2})(\d{2})\d{2}Z_", scene_name)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    return None


def _render_panel4_s1_map(station_id, selected_date, s1_index, per_arc):
    """Two-scale S1 view: regional context (5km) + zoomed Fresnel neighborhood (~500m)."""
    from scripts.s1_fresnel_utils import compute_sector_stats, get_s1_fresnel_dir, get_station_s1_config

    best_scene, days_offset = _find_nearest_s1_scene(station_id, selected_date, s1_index)
    tif_name = best_scene["file_path"]
    tif_path = get_s1_fresnel_dir(station_id) / tif_name

    if not tif_path.exists():
        st.warning(f"GeoTIFF not found: {tif_path}")
        return

    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        st.warning("S1 map requires `rasterio` and `pyproj`. Install with: `pip install -e .[dashboard]`")
        return

    with rasterio.open(tif_path) as src:
        hh_db = src.read(1)
        s1_transform = src.transform
        s1_crs = src.crs

    # S1 pass time → filter arcs
    s1_utc = _get_s1_pass_utc(best_scene.get("scene_name", ""))
    day_arcs = per_arc[per_arc["date"] == selected_date]
    if s1_utc is not None and not day_arcs.empty:
        window_arcs = day_arcs[
            (day_arcs["UTCtime"] >= s1_utc - 2) & (day_arcs["UTCtime"] <= s1_utc + 2)
        ]
    else:
        window_arcs = day_arcs

    sector = compute_sector_stats(tif_path, station_id, window_arcs=window_arcs)

    # Station pixel coords
    config = get_station_s1_config(station_id)
    t_proj = Transformer.from_crs("EPSG:4326", s1_crs, always_xy=True)
    cx_scene, cy_scene = t_proj.transform(config["lon"], config["lat"])
    sta_col = (cx_scene - s1_transform.c) / s1_transform.a
    sta_row = (cy_scene - s1_transform.f) / s1_transform.e

    # Reflection points in pixel coords
    refl_col, refl_row = np.array([]), np.array([])
    if not window_arcs.empty:
        elev_mid = (window_arcs["eminO"] + window_arcs["emaxO"]) / 2.0
        refl_dist = window_arcs["RH"] / np.tan(np.radians(elev_mid))
        refl_x = cx_scene + refl_dist * np.sin(np.radians(window_arcs["Azim"]))
        refl_y = cy_scene + refl_dist * np.cos(np.radians(window_arcs["Azim"]))
        refl_col = ((refl_x - s1_transform.c) / s1_transform.a).values
        refl_row = ((refl_y - s1_transform.f) / s1_transform.e).values

    # Load water mask
    water_mask = None
    gsw_path = PROJECT_ROOT / "data" / station_id / "s1_fresnel" / f"{station_id}_gsw_occurrence.tif"
    if gsw_path.exists():
        from scripts.s1_fresnel_utils import _load_gsw_water_mask
        water_mask = _load_gsw_water_mask(station_id, s1_crs, s1_transform, hh_db.shape)

    # --- Two panels: regional (left) + Fresnel reflection points (right) ---
    fig, (ax_reg, ax_refl) = plt.subplots(1, 2, figsize=(14, 6))

    vmin, vmax = -25, 0
    px_per_m = 1.0 / abs(s1_transform.a)  # pixels per meter

    # -- Left: Regional 5km context --
    ax_reg.imshow(hh_db, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    if water_mask is not None:
        land_ov = np.zeros((*hh_db.shape, 4))
        land_ov[~water_mask] = [1, 0.2, 0.2, 0.15]
        ax_reg.imshow(land_ov, origin="upper")

    ax_reg.plot(sta_col, sta_row, "r^", markersize=10, markeredgecolor="white",
                markeredgewidth=1.5, zorder=10)

    # Draw azimuth range fan
    az_range = config.get("azval2", [0, 360])
    fan_r = 250 * px_per_m
    for i in range(0, len(az_range), 2):
        for az_val in [az_range[i], az_range[i + 1]]:
            ddx = fan_r * np.sin(np.radians(az_val))
            ddy = -fan_r * np.cos(np.radians(az_val))
            ax_reg.plot([sta_col, sta_col + ddx], [sta_row, sta_row + ddy],
                        "-", color="cyan", linewidth=1.5, alpha=0.8)

    # Draw zoom box
    zoom_half = 250 * px_per_m
    zoom_rect = plt.Rectangle(
        (sta_col - zoom_half, sta_row - zoom_half), 2 * zoom_half, 2 * zoom_half,
        fill=False, edgecolor="lime", linewidth=2, linestyle="-",
    )
    ax_reg.add_patch(zoom_rect)
    ax_reg.set_title("Regional context (5km)", fontsize=10)

    # -- Right: Reflection points on S1 (GIF-style) --
    # Use local-meter coords centered on station, S1 as basemap
    # Compute the view extent from max reflection distance
    all_day_arcs = per_arc[per_arc["date"] == selected_date]
    if not all_day_arcs.empty:
        all_elev = (all_day_arcs["eminO"] + all_day_arcs["emaxO"]) / 2.0
        all_refl_dist = all_day_arcs["RH"] / np.tan(np.radians(all_elev))
        outer_dist = all_refl_dist.max()
    else:
        outer_dist = 200
    buffer_m = outer_dist + 30

    # S1 image extent in meters from station
    s1_extent_m = [
        (0 - sta_col) / px_per_m,
        (hh_db.shape[1] - sta_col) / px_per_m,
        -(hh_db.shape[0] - sta_row) / px_per_m,  # bottom (south)
        -(0 - sta_row) / px_per_m,                # top (north)
    ]
    ax_refl.imshow(hh_db, cmap="gray", vmin=vmin, vmax=vmax, extent=s1_extent_m,
                   origin="upper", aspect="equal")
    if water_mask is not None:
        ax_refl.imshow(land_ov, extent=s1_extent_m, origin="upper", aspect="equal")

    ax_refl.set_xlim(-buffer_m, buffer_m)
    ax_refl.set_ylim(-buffer_m, buffer_m)

    # Station marker
    ax_refl.plot(0, 0, "r^", markersize=14, markeredgecolor="white",
                 markeredgewidth=2, zorder=10)

    # Azimuth range boundary lines
    for i in range(0, len(az_range), 2):
        for az_val in [az_range[i], az_range[i + 1]]:
            ddx = buffer_m * np.sin(np.radians(az_val))
            ddy = buffer_m * np.cos(np.radians(az_val))
            ax_refl.plot([0, ddx], [0, ddy], "-", color="cyan", linewidth=1, alpha=0.5)

    # Plot ALL arcs for the day (faded, small)
    if not all_day_arcs.empty:
        all_elev = (all_day_arcs["eminO"] + all_day_arcs["emaxO"]) / 2.0
        all_dist = all_day_arcs["RH"] / np.tan(np.radians(all_elev))
        all_dx = all_dist * np.sin(np.radians(all_day_arcs["Azim"]))
        all_dy = all_dist * np.cos(np.radians(all_day_arcs["Azim"]))
        ax_refl.scatter(all_dx, all_dy, c=all_day_arcs["Amp"], cmap="coolwarm",
                        s=8, alpha=0.3, edgecolors="none", zorder=3)

    # Plot WINDOW arcs (±2hr of S1 pass) prominently
    if not window_arcs.empty:
        w_elev = (window_arcs["eminO"] + window_arcs["emaxO"]) / 2.0
        w_dist = window_arcs["RH"] / np.tan(np.radians(w_elev))
        w_dx = w_dist * np.sin(np.radians(window_arcs["Azim"]))
        w_dy = w_dist * np.cos(np.radians(window_arcs["Azim"]))
        sc = ax_refl.scatter(w_dx, w_dy, c=window_arcs["Amp"], cmap="coolwarm",
                             s=50, alpha=0.95, edgecolors="darkorange", linewidths=1.5,
                             zorder=5)
        plt.colorbar(sc, ax=ax_refl, label="Amplitude", shrink=0.7, pad=0.02)

    ax_refl.set_xlabel("East (m)")
    ax_refl.set_ylabel("North (m)")
    ax_refl.set_aspect("equal")

    utc_text = f" {s1_utc:.0f}:00 UTC" if s1_utc else ""
    offset_text = f" ({days_offset}d offset)" if days_offset > 0 else ""
    n_window = len(window_arcs)
    n_all = len(all_day_arcs)
    ax_refl.set_title(
        f"Reflection points on S1 HH\n"
        f"Orange border = ±2hr of pass ({n_window} arcs), faded = full day ({n_all})",
        fontsize=9,
    )

    fig.suptitle(
        f"S1 HH — {best_scene['acquisition_date']}{utc_text}{offset_text}",
        fontsize=11,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Stats tables ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backscatter (water pixels only)**")
        rows = []
        wo = sector.get("water_only", {})
        fw = sector.get("fresnel_water", {})
        if wo.get("n_pixels", 0) > 0:
            rows.append({"Zone": "Regional water", "HH (dB)": f"{wo['mean_hh_db']:.1f}",
                         "Pixels": wo["n_pixels"]})
        if fw.get("n_pixels", 0) > 0:
            rows.append({"Zone": "Fresnel water", "HH (dB)": f"{fw['mean_hh_db']:.1f}",
                         "Pixels": fw["n_pixels"]})
        fz_sectors = sector.get("fresnel_sectors", {})
        for b, s in sorted(fz_sectors.items()):
            w = s.get("water", {})
            if w.get("n_pixels", 0) > 0:
                rows.append({"Zone": f"Fresnel {AZ_BIN_LABELS.get(b, str(b))}",
                              "HH (dB)": f"{w['mean_hh_db']:.1f}", "Pixels": w["n_pixels"]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        if not window_arcs.empty:
            st.markdown("**GNSS-IR (±2hr of S1 pass)**")
            gnss_rows = []
            for b in sorted(window_arcs["azimuth_bin"].unique()):
                ba = window_arcs[window_arcs["azimuth_bin"] == b]
                amp = ba["Amp"]
                cv = amp.std() / amp.mean() if amp.mean() > 0 else float("nan")
                gnss_rows.append({
                    "Sector": AZ_BIN_LABELS.get(b, str(b)),
                    "Arcs": len(ba),
                    "Amp CV": f"{cv:.3f}",
                    "Amp": f"{amp.mean():.1f}",
                })
            st.dataframe(pd.DataFrame(gnss_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 5: S1 + GNSS-IR Cross-Validation Time Series
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel5_s1_timeseries(s1_index, enriched, station_id, year):
    """Time series: regional water-only S1 HH, GNSS-IR amp_cv, and regional ice fraction."""
    # Use pre-computed index stats for the time series (fast, no per-scene recomputation)
    # The water-masked Fresnel stats would require recomputing each scene — too slow for
    # a dashboard. Instead, use the regional water-only stats (which the index already has
    # as mean_hh_db for the whole clip) and the matched parquet for time-windowed amp_cv.

    s1 = s1_index.copy()
    s1["date_dt"] = pd.to_datetime(s1["acquisition_date"])
    s1 = s1[s1["date_dt"].dt.year == year].sort_values("date_dt")

    # Load matched data if available (has time-windowed amp_cv)
    matched = _load_s1_matched(station_id, year)

    # Fall back to enriched pooled daily
    pooled = pd.DataFrame()
    if enriched is not None and len(enriched) > 0:
        pooled = enriched[
            (enriched["azimuth_bin"] == -1) & (enriched["freq_group"] == "ALL")
        ].copy()
        pooled["date_dt"] = pd.to_datetime(pooled["date"])
        pooled = pooled[pooled["date_dt"].dt.year == year].sort_values("date_dt")

    if pooled.empty and s1.empty:
        st.warning("No data for cross-validation")
        return

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    # --- Top panel: amp_cv + S1 HH (water-only, sensing azimuths) ---
    color_cv = "#1f77b4"
    if not pooled.empty:
        ax1.plot(pooled["date_dt"], pooled["amp_cv"], color=color_cv, linewidth=1.2,
                 alpha=0.5, label="GNSS-IR amp_cv (daily)")

    # If matched data exists, overlay time-windowed amp_cv
    if matched is not None and len(matched) > 0:
        matched["date_dt"] = pd.to_datetime(matched["acquisition_date"])
        m_year = matched[matched["date_dt"].dt.year == year].sort_values("date_dt")
        if not m_year.empty and "gnssir_amp_cv" in m_year.columns:
            ax1.scatter(m_year["date_dt"], m_year["gnssir_amp_cv"], color=color_cv,
                        s=40, zorder=6, label="amp_cv (±2hr of S1 pass)", edgecolors="navy",
                        linewidths=0.5)

    ax1.set_ylabel("Amplitude CV", color=color_cv, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_cv)
    ax1.axhspan(0, AMP_CV_ICE, alpha=0.08, color="blue")
    ax1.axhspan(AMP_CV_WATER, 1.0, alpha=0.08, color="green")
    ax1.axhline(AMP_CV_ICE, color="blue", linewidth=0.5, alpha=0.3)
    ax1.axhline(AMP_CV_WATER, color="green", linewidth=0.5, alpha=0.3)

    # S1 HH — water-only in sensing azimuths
    ax2 = ax1.twinx()
    if not s1.empty:
        hh_col = "water_az_mean_hh_db" if "water_az_mean_hh_db" in s1.columns else "mean_hh_db"
        hh_label = "S1 HH (water, sensing az)" if hh_col.startswith("water") else "S1 HH"
        ax2.scatter(s1["date_dt"], s1[hh_col], color="#ff7f0e", s=25, alpha=0.7,
                    zorder=5, label=hh_label, marker="o")
        ax2.set_ylabel("S1 HH Backscatter (dB)", color="#ff7f0e", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=2)
    ax1.set_title(f"S1 HH vs GNSS-IR Amplitude CV — {year}", fontsize=12)

    # --- Bottom panel: amp_mean (signal strength tracks surface coherence) ---
    if not pooled.empty and "amp_mean" in pooled.columns:
        ax3.plot(pooled["date_dt"], pooled["amp_mean"], color="#2ca02c", linewidth=1)
        ax3.fill_between(pooled["date_dt"], 0, pooled["amp_mean"], color="#2ca02c", alpha=0.15)
        ax3.set_ylabel("Amp mean", fontsize=10)
        ax3.set_xlabel("Date")
        ax3.set_title("GNSS-IR amplitude (high + steady = coherent surface)", fontsize=9)
    else:
        ax3.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Correlation metrics — merge matched data with water-filtered S1
    if matched is not None and not s1.empty:
        m_year = matched[matched["date_dt"].dt.year == year].copy()

        # Join water-filtered S1 HH onto matched data
        hh_col = "water_az_mean_hh_db" if "water_az_mean_hh_db" in s1.columns else "mean_hh_db"
        s1_lookup = s1[["date_dt", hh_col]].rename(columns={hh_col: "s1_water_hh"})
        m_merged = pd.merge_asof(
            m_year.sort_values("date_dt"),
            s1_lookup.sort_values("date_dt"),
            on="date_dt", tolerance=pd.Timedelta("1D"), direction="nearest",
        )
        valid = m_merged.dropna(subset=["gnssir_amp_cv", "s1_water_hh"])

        if len(valid) >= 5:
            c1, c2, c3 = st.columns(3)
            corr_tw = valid["gnssir_amp_cv"].corr(valid["s1_water_hh"])
            c1.metric("±2hr amp_cv vs S1 (water)", f"r = {corr_tw:.3f}", delta=f"n={len(valid)}")

            valid_fd = valid.dropna(subset=["gnssir_fullday_amp_cv"])
            if len(valid_fd) >= 5:
                corr_fd = valid_fd["gnssir_fullday_amp_cv"].corr(valid_fd["s1_water_hh"])
                c2.metric("Daily amp_cv vs S1 (water)", f"r = {corr_fd:.3f}", delta=f"n={len(valid_fd)}")

            desc = valid[valid["flight_direction"] == "DESCENDING"]
            if len(desc) >= 5:
                corr_desc = desc["gnssir_amp_cv"].corr(desc["s1_water_hh"])
                c3.metric("Descending only", f"r = {corr_desc:.3f}", delta=f"n={len(desc)}")


# ──────────────────────────────────────────────────────────────────────────────
# Panel 6: S1 Gallery / Mosaic
# ──────────────────────────────────────────────────────────────────────────────


def _render_panel6_s1_gallery(station_id, s1_index, year):
    """Show a chronological mosaic of S1 thumbnails for the year."""
    thumb_dir = PROJECT_ROOT / "data" / station_id / "s1_fresnel" / "thumbnails"

    s1 = s1_index.copy()
    s1["date_dt"] = pd.to_datetime(s1["acquisition_date"])
    s1_year = s1[s1["date_dt"].dt.year == year].sort_values("date_dt")

    if s1_year.empty:
        st.info(f"No S1 data for {year}")
        return

    # Find available thumbnails
    thumbs = []
    for _, row in s1_year.iterrows():
        date_str = row["acquisition_date"]
        thumb_path = thumb_dir / f"{station_id}_{date_str.replace('-', '')}_thumb.png"
        if thumb_path.exists():
            thumbs.append({"date": date_str, "path": thumb_path, "month": row["date_dt"].month})

    if not thumbs:
        st.warning(
            f"No thumbnails rendered yet. Run:\n"
            f"`python scripts/s1_render_thumbnails.py --station {station_id}`"
        )
        return

    # Month filter
    months_available = sorted(set(t["month"] for t in thumbs))
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_options = ["All"] + [month_names[m] for m in months_available]
    selected_month = st.selectbox("Filter by month", month_options, index=0, key="gallery_month")

    if selected_month != "All":
        month_num = [k for k, v in month_names.items() if v == selected_month][0]
        thumbs = [t for t in thumbs if t["month"] == month_num]

    st.caption(f"{len(thumbs)} scenes")

    # Render as grid
    cols_per_row = 6
    for row_start in range(0, len(thumbs), cols_per_row):
        row_thumbs = thumbs[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, thumb in enumerate(row_thumbs):
            with cols[j]:
                st.image(str(thumb["path"]), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main tab renderer
# ──────────────────────────────────────────────────────────────────────────────


def render_per_arc_tab(station_id, year):
    """Render the Arc-Level Analysis tab."""
    st.header("Arc-Level Analysis")

    per_arc = load_per_arc(station_id, year)
    enriched = load_enriched(station_id, year)

    if per_arc is None:
        st.error(
            f"No per-arc parquet found for {station_id} {year}. "
            f"Run: `python scripts/backfill_enriched.py --station {station_id} --year {year}`"
        )
        return

    if enriched is None:
        st.warning("No enriched daily parquet — some panels will be limited.")

    # Available dates
    available_dates = sorted(per_arc["date"].unique())
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Per-Arc Controls")

    selected_date = st.sidebar.selectbox(
        "Select Date", available_dates, index=len(available_dates) // 2
    )

    has_wse = "wse" in per_arc.columns
    y_options = ["RH", "wse"] if has_wse else ["RH"]
    y_toggle = st.sidebar.radio("Y-axis", y_options, horizontal=True)
    y_col = y_toggle.lower() if y_toggle == "wse" else "RH"

    # Arc quality filter
    has_numbof = "NumbOf" in per_arc.columns
    min_numbof = 0
    if has_numbof:
        nof_range = (int(per_arc["NumbOf"].min()), int(per_arc["NumbOf"].max()))
        min_numbof = st.sidebar.slider(
            "Min NumbOf (arc quality)",
            min_value=0, max_value=nof_range[1],
            value=0, step=5,
            help=f"Filter arcs with fewer than N samples. Range: {nof_range[0]}-{nof_range[1]}",
        )

    # ── Panel 0: Multi-Day Subdaily Heatmap ──
    st.subheader("0. Multi-Day Subdaily Overview")
    hm_col1, hm_col2 = st.columns([3, 1])
    with hm_col1:
        heatmap_field = st.radio(
            "Heatmap value", ["RH", "Amp", "amp_cv", "arc_count"],
            horizontal=True, key="heatmap_field",
        )
    with hm_col2:
        const_options = ["All"]
        if "sat" in per_arc.columns:
            consts = sorted(per_arc["sat"].apply(_classify_constellation).unique())
            const_options += consts
        const_filter = st.selectbox(
            "Constellation", const_options, index=0, key="const_filter",
            help="Filter by constellation to isolate sidereal repeat patterns",
        )
    _render_panel0_multiday_heatmap(per_arc, selected_date, heatmap_field,
                                     min_numbof, const_filter)
    st.caption(
        "Diagonal stripes = satellite orbital repeat. "
        "GPS: -3.3 min/day, GLONASS: +9.6 min/day, Galileo: +2.6 min/day (sidereal drift)."
    )

    # ── Panel 0b: Azimuth-Resolved Map ──
    st.subheader("0b. Azimuth-Resolved Amplitude")
    _render_panel0b_azimuth_map(per_arc, selected_date, min_numbof)

    # ── Panel 1: Arc Polar View ──
    st.subheader("1. Reflection Points + Amplitude Structure")
    _render_panel1_arc_polar(per_arc, selected_date, station_id, min_numbof)

    # ── Panel 2: Frequency Band Comparison ──
    st.subheader("2. Frequency Band Comparison")
    freq_view = st.radio(
        "View", ["Time series (all dates)", "Single-day detail"],
        horizontal=True, key="freq_view",
    )
    if freq_view == "Time series (all dates)":
        if enriched is not None:
            if y_col != "RH" and "wse_mean" in enriched.columns:
                freq_y_col = "wse_mean"
            else:
                freq_y_col = "rh_mean"
            date_min = pd.Timestamp(available_dates[0])
            date_max = pd.Timestamp(available_dates[-1])
            _render_panel2_freq_comparison(enriched, (date_min, date_max), freq_y_col)
        else:
            st.info("Requires daily enriched parquet")
    else:
        _render_panel2_detail(per_arc, selected_date, y_col)

    # ── Panel 3: Spatial Quality ──
    st.subheader("3. Spatial Quality Distribution")
    _render_panel3_amplitude(per_arc, enriched, selected_date, station_id)

    # ── Phase B: S1 Panels (conditional) ──
    st.markdown("---")
    s1_index = _load_s1_index(station_id)

    if s1_index is not None and len(s1_index) > 0:
        s1_year = s1_index[pd.to_datetime(s1_index["acquisition_date"]).dt.year == year]
        st.caption(f"{len(s1_year)} S1 scenes available for {year} ({len(s1_index)} total)")

        # ── Panel 4: S1 Map with sector overlay ──
        st.subheader("4. S1 Backscatter Map + Sectors")
        try:
            _render_panel4_s1_map(station_id, selected_date, s1_index, per_arc)
        except Exception as e:
            st.warning(f"S1 map unavailable: {e}")

        # ── Panel 5: Cross-validation time series ──
        st.subheader("5. S1 vs GNSS-IR Cross-Validation")
        _render_panel5_s1_timeseries(
            s1_index, enriched if enriched is not None else pd.DataFrame(), station_id, year
        )

        # ── Panel 6: S1 Gallery ──
        st.subheader("6. S1 Scene Gallery")
        _render_panel6_s1_gallery(station_id, s1_index, year)
    else:
        st.info(
            f"No S1 backscatter data for {station_id}. "
            f"To download, run:\n"
            f"```\n"
            f"python scripts/s1_fresnel_search.py --station {station_id}\n"
            f"python scripts/s1_fresnel_download.py --station {station_id}\n"
            f"```"
        )
