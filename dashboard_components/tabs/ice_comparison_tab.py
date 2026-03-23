# ABOUTME: Ice classification tab — literature-based scoring with indicator time series
# ABOUTME: Walks through voting methodology, shows CLR/AF/damping evidence and sector analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Classification colors
CLASS_COLOR = {"ice": "#1565c0", "transition": "#f9a825", "water": "#2e7d32"}
CLASS_BORDER = CLASS_COLOR  # backward compat for thumbnail grid
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def _load_classification(station, year):
    path = (PROJECT_ROOT / "results_annual" / station
            / f"{station}_{year}_ice_classification.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date_dt"] = pd.to_datetime(df["date"])
    return df


def _load_s1_index(station):
    path = PROJECT_ROOT / "data" / station / "s1_fresnel" / f"{station}_s1_fresnel_index.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date_dt"] = pd.to_datetime(df["acquisition_date"])
    return df


def _match_s1_to_classification(s1_index, clf):
    merged = pd.merge_asof(
        s1_index.sort_values("date_dt"),
        clf[["date_dt", "classification", "ice_score",
             "amp_mean", "amp_cv", "rh_std"]].sort_values("date_dt"),
        on="date_dt", tolerance=pd.Timedelta("1D"), direction="nearest",
    )
    return merged.dropna(subset=["classification"])


def _get_thumb_path(station, date_str):
    d = date_str.replace("-", "")
    return PROJECT_ROOT / "data" / station / "s1_fresnel" / "thumbnails" / f"{station}_{d}_thumb.png"


def _compute_fresnel_radii(station, year):
    """Compute Fresnel zone inner/outer radii from per-arc data or config.

    Returns (inner_m, outer_m, method) or (None, None, None) if no data.
    """
    import json

    # Try per-arc data first (actual observed reflection distances)
    pa_path = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_per_arc.parquet"
    if pa_path.exists():
        pa = pd.read_parquet(pa_path)
        elev_mid = (pa["eminO"] + pa["emaxO"]) / 2.0
        refl_dist = pa["RH"] / np.tan(np.radians(elev_mid))
        # Use p2-p98 with 30m buffer (1 S1 pixel) on each side
        inner = max(0, refl_dist.quantile(0.02) - 30)
        outer = refl_dist.quantile(0.98) + 30
        return float(inner), float(outer), "per-arc p2-p98 +30m buffer"

    # Fallback: config-based
    cfg_path = PROJECT_ROOT / "config" / f"{station.lower()}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Widest range: maxH at lowest elev, minH at highest elev
        inner = max(0, cfg["minH"] / np.tan(np.radians(cfg["e2"])) - 30)
        outer = cfg["maxH"] / np.tan(np.radians(cfg["e1"])) + 30
        return float(inner), float(outer), "config (minH/maxH, e1/e2)"

    return None, None, None


def _render_analysis_zone_legend(station, year):
    """Render reference image showing both analysis scales:

    Scale 1 (regional): all water pixels in azimuth range (faded blue)
    Scale 2 (Fresnel):  water pixels within Fresnel annulus (bright green)
    """
    from scripts.s1_fresnel_utils import get_station_s1_config, get_s1_fresnel_dir, _load_gsw_water_mask

    config = get_station_s1_config(station)
    az_range = config.get("azval2", [0, 360])

    fresnel_dir = get_s1_fresnel_dir(station)
    tifs = sorted(fresnel_dir.glob(f"{station}_*_s1_fresnel.tif"))
    if not tifs:
        return

    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        st.warning("Analysis zone overlay requires `rasterio` and `pyproj`. Install with: `pip install -e .[dashboard]`")
        return

    tif_path = tifs[len(tifs) // 2]
    with rasterio.open(tif_path) as src:
        hh_db = src.read(1)
        s1_transform = src.transform
        s1_crs = src.crs

    t_proj = Transformer.from_crs("EPSG:4326", s1_crs, always_xy=True)
    cx, cy = t_proj.transform(config["lon"], config["lat"])
    sta_col = (cx - s1_transform.c) / s1_transform.a
    sta_row = (cy - s1_transform.f) / s1_transform.e
    px_per_m = 1.0 / abs(s1_transform.a)

    water_mask = _load_gsw_water_mask(station, s1_crs, s1_transform, hh_db.shape)

    # Build coordinate grids
    rows_grid, cols_grid = np.meshgrid(
        np.arange(hh_db.shape[0]), np.arange(hh_db.shape[1]), indexing="ij"
    )
    x_coords = s1_transform.c + (cols_grid + 0.5) * s1_transform.a
    y_coords = s1_transform.f + (rows_grid + 0.5) * s1_transform.e
    dx = x_coords - cx
    dy = y_coords - cy
    dist_m = np.sqrt(dx**2 + dy**2)
    az_deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

    # Azimuth mask
    az_mask = np.zeros_like(az_deg, dtype=bool)
    for i in range(0, len(az_range), 2):
        az_min, az_max = az_range[i], az_range[i + 1]
        if az_min < az_max:
            az_mask |= (az_deg >= az_min) & (az_deg <= az_max)
        else:
            az_mask |= (az_deg >= az_min) | (az_deg <= az_max)

    # Fresnel annulus
    inner_m, outer_m, fresnel_method = _compute_fresnel_radii(station, year)
    has_fresnel = inner_m is not None

    # Two-panel figure: full scene (left) + zoomed Fresnel (right)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    for ax in [ax_full, ax_zoom]:
        ax.imshow(hh_db, cmap="gray", vmin=-25, vmax=0, origin="upper")

    # --- Overlays ---
    if water_mask is not None:
        # Land (red tint)
        land_ov = np.zeros((*hh_db.shape, 4))
        land_ov[~water_mask] = [1, 0.2, 0.2, 0.25]

        # Regional water in azimuth (faded blue)
        regional_mask = water_mask & az_mask & ~np.isnan(hh_db)
        regional_ov = np.zeros((*hh_db.shape, 4))
        regional_ov[regional_mask] = [0.3, 0.5, 1.0, 0.15]
        n_regional = int(np.sum(regional_mask))

        # Fresnel zone water (bright green)
        if has_fresnel:
            fresnel_annulus = (dist_m >= inner_m) & (dist_m <= outer_m)
            fresnel_mask = water_mask & az_mask & fresnel_annulus & ~np.isnan(hh_db)
            fresnel_ov = np.zeros((*hh_db.shape, 4))
            fresnel_ov[fresnel_mask] = [0.0, 1.0, 0.3, 0.4]
            n_fresnel = int(np.sum(fresnel_mask))
        else:
            fresnel_mask = np.zeros_like(hh_db, dtype=bool)
            n_fresnel = 0

        for ax in [ax_full, ax_zoom]:
            ax.imshow(land_ov, origin="upper")
            ax.imshow(regional_ov, origin="upper")
            if has_fresnel:
                ax.imshow(fresnel_ov, origin="upper")
    else:
        n_regional = 0
        n_fresnel = 0

    # Station + azimuth lines on both panels
    for ax in [ax_full, ax_zoom]:
        ax.plot(sta_col, sta_row, "r^", markersize=8, markeredgecolor="white",
                markeredgewidth=1, zorder=10)
        fan_r = 300 * px_per_m
        for i in range(0, len(az_range), 2):
            for az_val in [az_range[i], az_range[i + 1]]:
                ddx = fan_r * np.sin(np.radians(az_val))
                ddy = -fan_r * np.cos(np.radians(az_val))
                ax.plot([sta_col, sta_col + ddx], [sta_row, sta_row + ddy],
                        "-", color="cyan", linewidth=1.5, alpha=0.7)

    # Draw Fresnel annulus arcs
    if has_fresnel:
        for ax in [ax_full, ax_zoom]:
            for radius_m, ls in [(inner_m, "--"), (outer_m, "-")]:
                r_px = radius_m * px_per_m
                angles = np.linspace(0, 2 * np.pi, 200)
                arc_x = sta_col + r_px * np.sin(angles)
                arc_y = sta_row - r_px * np.cos(angles)
                ax.plot(arc_x, arc_y, ls, color="lime", linewidth=1.2, alpha=0.7)

    # Left: full scene
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_title("Full S1 scene (5km clip)", fontsize=10)

    # Right: zoomed to Fresnel neighborhood
    if has_fresnel:
        zoom_r = (outer_m + 50) * px_per_m
    else:
        zoom_r = 200 * px_per_m
    ax_zoom.set_xlim(sta_col - zoom_r, sta_col + zoom_r)
    ax_zoom.set_ylim(sta_row + zoom_r, sta_row - zoom_r)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_title("Zoomed to Fresnel zone", fontsize=10)

    # Legend text
    lines = [
        f"Red tint    = land (GSW mask, excluded)",
        f"Blue tint   = regional water context ({n_regional} px)",
        f"             (water in {az_range[0]:.0f}-{az_range[1]:.0f} deg azimuth)",
    ]
    if has_fresnel:
        lines += [
            f"Green tint  = Fresnel reflection zone ({n_fresnel} px)",
            f"             ({inner_m:.0f}-{outer_m:.0f}m from station)",
            f"Dashed arc  = inner Fresnel radius ({inner_m:.0f}m)",
            f"Solid arc   = outer Fresnel radius ({outer_m:.0f}m)",
            f"Source: {fresnel_method}",
        ]
    lines += [
        f"Cyan lines  = azimuth range boundaries",
        f"Red triangle = GNSS station",
    ]

    ax_full.text(0.02, 0.02, "\n".join(lines), transform=ax_full.transAxes, fontsize=7,
                 color="white", va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.85),
                 family="monospace")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)


def _render_thumbnail_grid(matched, station, cols_per_row=8):
    """Render chronological thumbnail grid with classification-colored borders."""
    scenes = matched.sort_values("date_dt")
    n = len(scenes)
    if n == 0:
        return

    n_rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(cols_per_row * 2, n_rows * 2.4),
                             dpi=100)

    if n_rows == 1:
        axes = axes.reshape(1, -1) if cols_per_row > 1 else np.array([[axes]])

    for idx, (_, scene) in enumerate(scenes.iterrows()):
        r, c = divmod(idx, cols_per_row)
        ax = axes[r, c]

        thumb_path = _get_thumb_path(station, scene["acquisition_date"])
        if thumb_path.exists():
            img = mpimg.imread(str(thumb_path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        # Classification-colored border
        cls = scene["classification"]
        border_color = CLASS_BORDER.get(cls, "#999999")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

        # Date + score annotation below image
        date_short = scene["acquisition_date"][5:]  # MM-DD
        score = scene["ice_score"]
        amp = scene["amp_mean"]
        label = f"{date_short}\n{score:+.2f} | {amp:.0f}"
        ax.set_xlabel(label, fontsize=7, labelpad=2, color=border_color, fontweight="bold")

    # Hide unused axes
    for idx in range(n, n_rows * cols_per_row):
        r, c = divmod(idx, cols_per_row)
        axes[r, c].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="white", edgecolor=CLASS_BORDER["ice"], linewidth=2,
              label=f"Ice (score < -0.33)"),
        Patch(facecolor="white", edgecolor=CLASS_BORDER["transition"], linewidth=2,
              label=f"Transition (-0.33 to +0.33)"),
        Patch(facecolor="white", edgecolor=CLASS_BORDER["water"], linewidth=2,
              label=f"Water (score > +0.33)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3, fontsize=9,
               framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        f"S1 Scenes — Chronological, border = classification\n"
        f"Label: MM-DD / ice_score | amp_mean",
        fontsize=10, y=1.04,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _has_snr_features(clf):
    """Check if classification data includes SNR-derived feature columns."""
    return any(c.endswith("_clr") and not c.endswith("_vote") for c in clf.columns)


def _median_across_sectors(clf, suffix):
    """Compute median of az*_{suffix} columns (excluding land/missing)."""
    cols = [c for c in clf.columns
            if c.endswith(f"_{suffix}") and not c.endswith("_vote")
            and c.startswith("az")]
    if not cols:
        return pd.Series(np.nan, index=clf.index)
    return clf[cols].median(axis=1)


def _render_score_timeseries(clf):
    """Ice score time series with classification zone shading."""
    fig = go.Figure()

    # Shaded classification zones
    dates = clf["date_dt"]
    fig.add_hrect(y0=-1.1, y1=-0.33, fillcolor=CLASS_COLOR["ice"],
                  opacity=0.08, line_width=0)
    fig.add_hrect(y0=-0.33, y1=0.33, fillcolor=CLASS_COLOR["transition"],
                  opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.33, y1=1.1, fillcolor=CLASS_COLOR["water"],
                  opacity=0.08, line_width=0)

    # Threshold lines
    fig.add_hline(y=-0.33, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.33, line_dash="dot", line_color="gray", opacity=0.5)

    # Points colored by classification
    for cls, color in CLASS_COLOR.items():
        mask = clf["classification"] == cls
        if mask.sum() == 0:
            continue
        subset = clf[mask]
        fig.add_trace(go.Scatter(
            x=subset["date_dt"], y=subset["ice_score"],
            mode="markers", marker=dict(color=color, size=5),
            name=cls.capitalize(),
            hovertemplate="%{x|%b %d}: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        height=280, margin=dict(l=50, r=20, t=30, b=30),
        yaxis=dict(title="Ice Score", range=[-1.1, 1.1]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        annotations=[
            dict(x=0.01, y=-0.7, xref="paper", yref="y", text="ICE",
                 showarrow=False, font=dict(color=CLASS_COLOR["ice"], size=10)),
            dict(x=0.01, y=0.7, xref="paper", yref="y", text="WATER",
                 showarrow=False, font=dict(color=CLASS_COLOR["water"], size=10)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_methodology(clf):
    """Explain the scoring system with indicator table."""
    has_features = _has_snr_features(clf)

    st.markdown("""
Each azimuth sector votes independently using weighted indicators.
Sector scores are combined (weighted by arc count) into a station-level
**ice_score**: values below **-0.33** classify as ice, above **+0.33** as water.
""")

    # Build indicator table
    rows = [
        ("Amplitude mean", "2.0", "High", "Low", "Standard GNSS-IR"),
        ("Amplitude CV", "1.5", "Low", "High", "Standard GNSS-IR"),
        ("RH std dev", "1.0", "Low", "High", "Standard GNSS-IR"),
    ]
    if has_features:
        rows.insert(1, ("**CLR** (clarity ratio)", "2.0", "High", "Low", "Purnell 2024"))
        rows.insert(3, ("**AF** (area factor)", "1.5", "High", "Low", "Song 2022"))
        rows.insert(5, ("**PR** (peak ratio)", "1.0", "High", "Low", "Purnell 2024"))
        rows.insert(6, ("**gamma** (damping)", "1.0", "Low", "High", "Strandberg 2017"))

    header = "| Indicator | Weight | Ice Signal | Water Signal | Source |\n"
    header += "|-----------|--------|-----------|--------------|--------|\n"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows)
    st.markdown(header + body)

    if has_features:
        st.caption(
            "Thresholds are **summer-anchored**: computed from the ice-free months "
            "(Jul-Aug) rather than the full year, avoiding bias at stations with "
            "long ice seasons. SNR-derived indicators (CLR, AF, PR, gamma) are "
            "extracted from raw SNR arcs using wavelet and spectral analysis."
        )
    else:
        st.caption(
            "SNR-derived features (CLR, AF, damping) are not available for this "
            "station-year. Run `snr_feature_extractor.py` then re-run the classifier "
            "to enable literature-based indicators."
        )


def _render_indicator_timeseries(clf):
    """4-panel time series of key indicators with classification coloring."""
    # Compute cross-sector medians
    clf = clf.copy()
    clf["clr_med"] = _median_across_sectors(clf, "clr")
    clf["af_med"] = _median_across_sectors(clf, "af")
    clf["gamma_med"] = _median_across_sectors(clf, "gamma")

    panels = [
        ("clr_med", "CLR (Clarity Ratio)", "Purnell 2024"),
        ("af_med", "AF (Area Factor)", "Song 2022"),
        ("amp_mean", "Amplitude Mean", "Standard GNSS-IR"),
        ("gamma_med", "Damping (gamma)", "Strandberg 2017"),
    ]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{p[1]}  ({p[2]})" for p in panels])

    for i, (col, label, _source) in enumerate(panels, 1):
        if col not in clf.columns:
            continue
        for cls, color in CLASS_COLOR.items():
            mask = clf["classification"] == cls
            if mask.sum() == 0:
                continue
            subset = clf[mask]
            fig.add_trace(go.Scatter(
                x=subset["date_dt"], y=subset[col],
                mode="markers", marker=dict(color=color, size=4, opacity=0.7),
                name=cls.capitalize(), showlegend=(i == 1),
                hovertemplate=f"{label}: %{{y:.2f}}<br>%{{x|%b %d}}<extra></extra>",
            ), row=i, col=1)

    fig.update_layout(
        height=700, margin=dict(l=60, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.5, xanchor="center"),
    )
    for i in range(1, 5):
        fig.update_yaxes(title_text="", row=i, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_sector_heatmap(clf):
    """Date x sector heatmap showing per-sector classification scores."""
    # Find all sector score columns
    score_cols = sorted([c for c in clf.columns
                         if c.endswith("_score") and c.startswith("az")])
    if not score_cols:
        return

    # Extract sector labels and data
    sector_labels = [c.replace("_score", "").replace("az", "") + "\u00b0"
                     for c in score_cols]
    class_cols = [c.replace("_score", "_class") for c in score_cols]

    dates = clf["date_dt"].values
    score_matrix = clf[score_cols].values.T  # sectors × dates

    # Mark land sectors
    land_sectors = set()
    for i, cc in enumerate(class_cols):
        if cc in clf.columns and (clf[cc] == "land").any():
            land_sectors.add(i)

    # Custom colormap: blue (ice) → yellow (transition) → green (water)
    cmap_colors = [CLASS_COLOR["ice"], "#e0e0e0", CLASS_COLOR["water"]]
    cmap = mcolors.LinearSegmentedColormap.from_list("ice_water", cmap_colors, N=256)
    cmap.set_bad(color="#888888")  # gray for NaN/land

    # Mask land sectors
    masked = np.ma.array(score_matrix)
    for i in land_sectors:
        masked[i, :] = np.ma.masked

    fig, ax = plt.subplots(figsize=(14, max(3, len(score_cols) * 0.35)), dpi=100)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest")

    # X-axis: show month labels
    date_series = pd.Series(dates)
    month_starts = []
    for m in range(1, 13):
        mask = pd.to_datetime(date_series).month == m
        if mask.any():
            first_idx = mask.values.argmax()
            month_starts.append((first_idx, MONTH_NAMES[m]))
    ax.set_xticks([x[0] for x in month_starts])
    ax.set_xticklabels([x[1] for x in month_starts], fontsize=8)

    # Y-axis: sector labels
    ax.set_yticks(range(len(sector_labels)))
    ax.set_yticklabels(sector_labels, fontsize=8)
    ax.set_ylabel("Azimuth Sector")

    # Mark land sectors
    for i in land_sectors:
        ax.text(len(dates) + 2, i, "LAND", fontsize=7, va="center", color="#888888")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_ticks([-1, -0.33, 0, 0.33, 1])
    cbar.set_ticklabels(["Ice", "", "Trans", "", "Water"])

    ax.set_title("Per-Sector Classification Score (blue=ice, green=water, gray=land)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_monthly_summary(clf):
    """Monthly summary table with optional SNR feature columns."""
    has_features = _has_snr_features(clf)
    clf = clf.copy()
    clf["month"] = clf["date_dt"].dt.month

    if has_features:
        clf["clr_med"] = _median_across_sectors(clf, "clr")
        clf["af_med"] = _median_across_sectors(clf, "af")

    rows = []
    for m in sorted(clf["month"].unique()):
        md = clf[clf["month"] == m]
        mc = md["classification"].value_counts()
        row = {
            "Month": MONTH_NAMES.get(m, str(m)),
            "Days": len(md),
            "Score": f"{md['ice_score'].mean():+.2f}",
            "Ice": mc.get("ice", 0),
            "Trans": mc.get("transition", 0),
            "Water": mc.get("water", 0),
            "Amp": f"{md['amp_mean'].mean():.1f}",
        }
        if has_features:
            row["CLR"] = f"{md['clr_med'].mean():.1f}"
            row["AF"] = f"{md['af_med'].mean():.0f}"
        rows.append(row)

    table_df = pd.DataFrame(rows)

    def _score_color(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v <= -0.33:
            return "background-color: #bbdefb"
        elif v >= 0.33:
            return "background-color: #c8e6c9"
        return "background-color: #fff9c4"

    styled = table_df.style.map(_score_color, subset=["Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_ice_comparison_tab(station_id, year):
    """Render ice classification tab with literature-based scoring methodology."""
    st.header("Ice Classification")

    clf = _load_classification(station_id, year)
    if clf is None:
        st.info(
            f"No classification found. Run:\n"
            f"```\npython scripts/ice_classifier.py --station {station_id} --year {year}\n```"
        )
        return

    has_features = _has_snr_features(clf)

    # --- Section 1: Overview metrics + score time series ---
    counts = clf["classification"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days Classified", len(clf))
    c2.metric("Ice", counts.get("ice", 0))
    c3.metric("Transition", counts.get("transition", 0))
    c4.metric("Water", counts.get("water", 0))

    _render_score_timeseries(clf)

    # --- Section 2: Methodology ---
    with st.expander("Scoring Methodology", expanded=False):
        _render_methodology(clf)

    # --- Section 3: Indicator time series (if SNR features available) ---
    if has_features:
        st.subheader("Indicator Evidence")
        st.caption(
            "Daily median of each indicator across active azimuth sectors. "
            "Points colored by station-level classification."
        )
        _render_indicator_timeseries(clf)

    # --- Section 4: Per-sector heatmap ---
    st.subheader("Per-Sector Analysis")
    st.caption(
        "Each row is an azimuth sector. Blue = ice vote, green = water vote. "
        "Gray = land-flagged (excluded from consensus)."
    )
    _render_sector_heatmap(clf)

    # --- Section 5: S1 SAR validation (expander) ---
    s1_index = _load_s1_index(station_id)
    if s1_index is not None:
        s1_year = s1_index[s1_index["date_dt"].dt.year == year].copy()
        matched = _match_s1_to_classification(s1_year, clf)
        matched["thumb_exists"] = matched["acquisition_date"].apply(
            lambda d: _get_thumb_path(station_id, d).exists()
        )
        matched = matched[matched["thumb_exists"]]

        if not matched.empty:
            with st.expander(f"S1 SAR Validation ({len(matched)} scenes)", expanded=False):
                st.caption(
                    "Sentinel-1 thumbnails with classification-colored borders. "
                    "Purnell 2024 found r = -0.8 between GNSS-IR spectral power "
                    "and C-band SAR backscatter."
                )

                # Analysis zone
                with st.expander("Analysis zone overlay", expanded=False):
                    _render_analysis_zone_legend(station_id, year)

                _render_thumbnail_grid(matched, station_id)

    # --- Section 6: Monthly summary ---
    st.subheader("Monthly Summary")
    _render_monthly_summary(clf)
