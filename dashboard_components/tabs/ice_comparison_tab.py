# ABOUTME: Ice classification tab — literature-based scoring with indicator time series
# ABOUTME: Walks through voting methodology, shows CLR/AF/damping evidence and sector analysis

import json
import warnings

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

from dashboard_components.s1_helpers import (
    load_s1_index,
    get_thumb_path,
    compute_fresnel_radii,
)
from dashboard_components.data_loader import safe_read_parquet

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
    df = safe_read_parquet(path)
    if df is None:
        return None
    df["date_dt"] = pd.to_datetime(df["date"])
    return df


def _load_met_data(station, year, project_root=None):
    """Load cached met data CSV. Returns DataFrame or None."""
    root = project_root or PROJECT_ROOT
    path = root / "results_annual" / station / f"{station}_{year}_met_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_freezing_point(station, config_path=None):
    """Get local freezing point from config, defaulting by surface type.

    Stations with ice_free_months configured → seawater (-1.8°C).
    Other stations → freshwater (0.0°C).
    Explicit met_data.freezing_point_c overrides both defaults.
    """
    cfg_path = config_path or (PROJECT_ROOT / "config" / "stations_config.json")
    if not cfg_path.exists():
        return -1.8

    with open(cfg_path) as f:
        all_cfg = json.load(f)

    station_cfg = all_cfg.get(station, {})
    met_cfg = station_cfg.get("met_data", {})
    if "freezing_point_c" in met_cfg:
        return met_cfg["freezing_point_c"]

    # Default: seawater freezing if ice classification is configured
    if station_cfg.get("ice_free_months"):
        return -1.8
    return 0.0


def _match_s1_to_classification(s1_index, clf):
    merged = pd.merge_asof(
        s1_index.sort_values("date_dt"),
        clf[["date_dt", "classification", "ice_score",
             "amp_mean", "amp_cv", "rh_std"]].sort_values("date_dt"),
        on="date_dt", tolerance=pd.Timedelta("1D"), direction="nearest",
    )
    return merged.dropna(subset=["classification"])



def _render_analysis_zone_legend(station, year):
    """Render reference image showing both analysis scales:

    Scale 1 (regional): all water pixels in azimuth range (faded blue)
    Scale 2 (Fresnel):  water pixels within Fresnel annulus (bright green)
    """
    from scripts.s1_fresnel_utils import get_station_s1_config, get_s1_fresnel_dir, _load_gsw_water_mask, resolve_crs

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

    t_proj = Transformer.from_crs("EPSG:4326", resolve_crs(s1_crs), always_xy=True)
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
    inner_m, outer_m, fresnel_method = compute_fresnel_radii(station, year)
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

        thumb_path = get_thumb_path(station, scene["acquisition_date"])
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


def _render_scene_detail(matched, station):
    """Scene picker: select a thumbnail to view full-size with classification details."""
    scenes = matched.sort_values("date_dt")
    options = scenes["acquisition_date"].tolist()
    labels = {
        d: f"{d}  —  {row['classification']}  (score {row['ice_score']:+.2f}, amp {row['amp_mean']:.0f})"
        for d, (_, row) in zip(options, scenes.iterrows())
    }

    selected = st.selectbox(
        "Select scene to view",
        options,
        format_func=lambda d: labels[d],
        key=f"ice_s1_scene_{station}",
    )

    if selected:
        row = scenes[scenes["acquisition_date"] == selected].iloc[0]
        thumb_path = get_thumb_path(station, selected)

        col_img, col_info = st.columns([3, 1])
        with col_img:
            if thumb_path.exists():
                st.image(str(thumb_path), use_container_width=True)
            else:
                st.warning("Thumbnail file not found.")

        with col_info:
            cls = row["classification"]
            color = CLASS_COLOR.get(cls, "#999999")
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 8px;'>"
                f"<b>{selected}</b><br>"
                f"Classification: <b>{cls}</b><br>"
                f"Ice score: <b>{row['ice_score']:+.3f}</b><br>"
                f"Amplitude: <b>{row['amp_mean']:.1f}</b><br>"
                f"Amp CV: <b>{row['amp_cv']:.3f}</b><br>"
                f"RH std: <b>{row['rh_std']:.3f} m</b>"
                f"</div>",
                unsafe_allow_html=True,
            )


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


@st.cache_data(ttl=3600, show_spinner="Computing wavelet scalograms...")
def _compute_example_scalograms(station, year):
    """Compute CWT scalograms for representative ice and water arcs.

    Returns dict with 'ice' and 'water' keys, each containing:
        elevation_deg, rh_values, power (2D array), date, score, sat
    Returns None on any failure.
    """
    try:
        from scripts.snr_feature_extractor import (
            read_snr_file, segment_satellite_arcs, find_matching_segment,
            detrend_arc, FREQ_TO_COL, FREQ_WAVELENGTH, _snr_file_path,
            _load_station_config as _load_gnssir_config,
        )
        from scripts.snr_feature_extractor import _cwt as cwt, _morlet2 as morlet2
    except ImportError:
        return None

    clf = _load_classification(station, year)
    if clf is None or len(clf) < 10:
        return None

    # Representative days: strongest ice and strongest water
    ice_row = clf.loc[clf["ice_score"].idxmin()]
    water_row = clf.loc[clf["ice_score"].idxmax()]

    pa_path = (PROJECT_ROOT / "results_annual" / station
               / f"{station}_{year}_per_arc.parquet")
    per_arc = safe_read_parquet(pa_path)
    if per_arc is None:
        return None

    ice_doy = ice_row["date_dt"].timetuple().tm_yday
    water_doy = water_row["date_dt"].timetuple().tm_yday

    # Find satellite with best amplitude contrast on both days
    ice_arcs = per_arc[per_arc["doy"] == ice_doy]
    water_arcs = per_arc[per_arc["doy"] == water_doy]
    common_sats = set(ice_arcs["sat"].unique()) & set(water_arcs["sat"].unique())
    if not common_sats:
        return None

    best_sat, best_contrast = None, 0
    for sat in common_sats:
        ia = ice_arcs[ice_arcs["sat"] == sat]["Amp"].mean()
        wa = water_arcs[water_arcs["sat"] == sat]["Amp"].mean()
        contrast = ia / wa if wa > 0 else 0
        if contrast > best_contrast:
            best_contrast = contrast
            best_sat = sat
    if best_sat is None:
        return None

    config = _load_gnssir_config(station)
    if config is None:
        return None

    e1, e2 = config["e1"], config["e2"]
    min_rh, max_rh = config["minH"], config["maxH"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))

    results = {}
    for label, doy, row in [("ice", ice_doy, ice_row),
                              ("water", water_doy, water_row)]:
        snr_path = _snr_file_path(station, year, doy)
        if not snr_path.exists():
            gz = Path(str(snr_path) + ".gz")
            if gz.exists():
                snr_path = gz
            else:
                return None

        snr_data = read_snr_file(snr_path)
        sat_mask = snr_data[:, 0] == best_sat
        sat_data = snr_data[sat_mask]
        if len(sat_data) < 10:
            return None

        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
        day_pa = per_arc[(per_arc["doy"] == doy) & (per_arc["sat"] == best_sat)]
        if len(day_pa) == 0:
            return None

        arc_entry = day_pa.iloc[0]
        freq = int(arc_entry["freq"])
        col_idx = FREQ_TO_COL.get(freq)
        wavelength = FREQ_WAVELENGTH.get(freq)
        if col_idx is None or wavelength is None:
            return None

        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=arc_entry["UTCtime"],
            target_rise=arc_entry["rise"],
            e1=e1, e2=e2,
        )
        if seg_idx < 0:
            return None

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]
        snr_db = arc_data[:, col_idx]
        if np.all(snr_db == 0):
            return None

        snr_lin = np.power(10, snr_db / 20)
        detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

        # Window to [e1, e2]
        mask = (arc_ele >= e1) & (arc_ele <= e2)
        if mask.sum() < 20:
            return None
        ele_w = arc_ele[mask]
        dsnr = detrended[mask]
        sin_e = np.sin(np.radians(ele_w))

        # CWT computation
        cf = wavelength / 2
        sort_idx = np.argsort(sin_e)
        x = sin_e[sort_idx] / cf
        y = dsnr[sort_idx]
        ele_sorted = ele_w[sort_idx]

        dx = np.mean(np.diff(x))
        if dx <= 0:
            return None

        w = 5.0
        rh_values = np.linspace(min_rh, max_rh, 100)
        scales = w / (2 * np.pi * rh_values * dx)
        valid = scales > 1
        if valid.sum() < 5:
            return None
        scales = scales[valid]
        rh_values = rh_values[valid]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cwtmatr = cwt(y, morlet2, scales, w=w)
        power = np.abs(cwtmatr) ** 2

        results[label] = {
            "elevation_deg": ele_sorted,
            "rh_values": rh_values,
            "power": power,
            "date": row["date"],
            "score": float(row["ice_score"]),
            "sat": int(best_sat),
        }

    return results


def _render_wavelet_comparison(station, year):
    """Render side-by-side CWT scalograms for representative ice and water arcs."""
    data = _compute_example_scalograms(station, year)
    if data is None:
        st.caption("Wavelet comparison unavailable (missing SNR files or per-arc data).")
        return

    fig, (ax_ice, ax_water) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    # Common color scale
    vmax = max(data["ice"]["power"].max(), data["water"]["power"].max())

    for ax, label, title_color in [(ax_ice, "ice", CLASS_COLOR["ice"]),
                                    (ax_water, "water", CLASS_COLOR["water"])]:
        d = data[label]
        im = ax.pcolormesh(
            d["elevation_deg"], d["rh_values"], d["power"],
            shading="auto", cmap="inferno", vmin=0, vmax=vmax,
        )
        ax.set_xlabel("Elevation (deg)", fontsize=9)
        ax.set_ylabel("Reflector Height (m)", fontsize=9)
        ax.set_title(
            "{} day: {} (score {:.2f})".format(
                label.capitalize(), d["date"], d["score"]),
            fontsize=10, color=title_color, fontweight="bold",
        )
        ax.tick_params(labelsize=8)

    fig.colorbar(im, ax=[ax_ice, ax_water], label="CWT Power",
                 shrink=0.8, pad=0.02)
    fig.suptitle(
        "Wavelet Scalogram — SAT {} (ice shows concentrated power at one RH)".format(
            data["ice"]["sat"]),
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_score_timeseries(clf, station_id=None, year=None):
    """Ice score time series with classification zone shading and optional temperature overlay."""
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

    # Temperature overlay on secondary y-axis
    met = _load_met_data(station_id, year) if station_id and year else None

    layout_kwargs = dict(
        height=320, margin=dict(l=50, r=60, t=30, b=30),
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

    if met is not None:
        freezing_pt = _get_freezing_point(station_id)

        fig.add_trace(go.Scatter(
            x=met["date"],
            y=met["temp_mean_c"],
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5),
            opacity=0.6,
            name="Air Temp (\u00b0C)",
            yaxis="y2",
            hovertemplate="%{x|%b %d}: %{y:.1f}\u00b0C<extra></extra>",
        ))

        # SST trace if available
        if "sst_max_c" in met.columns:
            sst_data = met.dropna(subset=["sst_max_c"])
            if not sst_data.empty:
                fig.add_trace(go.Scatter(
                    x=sst_data["date"],
                    y=sst_data["sst_max_c"],
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    opacity=0.7,
                    name="SST (\u00b0C)",
                    yaxis="y2",
                    hovertemplate="%{x|%b %d}: SST %{y:.1f}\u00b0C<extra></extra>",
                ))

        # Freezing point reference line
        fig.add_shape(
            type="line", y0=freezing_pt, y1=freezing_pt,
            x0=0, x1=1, xref="paper", yref="y2",
            line=dict(color="#ff7f0e", width=1, dash="dash"),
            opacity=0.4,
        )
        fig.add_annotation(
            x=1.0, y=freezing_pt, xref="paper", yref="y2",
            text=f"Freezing ({freezing_pt}\u00b0C)",
            showarrow=False, font=dict(color="#ff7f0e", size=9),
            xanchor="right", yanchor="bottom",
        )

        layout_kwargs["yaxis2"] = dict(
            title=dict(text="Temperature (\u00b0C)", font=dict(color="#ff7f0e")),
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color="#ff7f0e"),
        )

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)

    if met is None and station_id:
        st.caption(
            f"Add temperature context: "
            f"`python scripts/fetch_met_data.py --station {station_id} --year {year}`"
        )


def _load_ice_free_months(station):
    """Load ice_free_months from stations_config.json."""
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not cfg_path.exists():
        return []
    with open(cfg_path) as f:
        all_cfg = json.load(f)
    return all_cfg.get(station, {}).get("ice_free_months", [])


def _render_methodology(clf):
    """Explain the scoring system with equations and indicator table."""
    has_features = _has_snr_features(clf)

    st.markdown("""
Each azimuth sector votes independently using weighted indicators.
Sector scores are combined (weighted by arc count) into a station-level
**ice_score**: values below **-0.33** classify as ice, above **+0.33** as water.
""")

    # --- Equations ---
    st.markdown("#### Signal Processing")
    st.markdown(
        "Raw SNR is converted to linear units and detrended with a 4th-order "
        "polynomial to isolate the interference fringes:"
    )
    st.latex(r"d\text{SNR}(\varepsilon) = 10^{\text{SNR}_{dB}/20} - P_4(\varepsilon)")

    st.markdown(
        "The detrended signal is analyzed in the $\\sin(\\varepsilon) / (\\lambda/2)$ "
        "domain, where peaks in the Lomb-Scargle periodogram correspond directly "
        "to reflector height (RH) in meters."
    )

    if has_features:
        st.markdown("#### Spectral Indicators")

        st.markdown(
            "**CLR** (Clarity Ratio) — Purnell 2024's best single-feature "
            "discriminator (77.7% accuracy alone). Ratio of the dominant LSP peak "
            "to the mean of all other peaks:"
        )
        st.latex(r"CLR = \frac{P_1}{\bar{P}_{2 \ldots N}}")

        st.markdown(
            "**PR** (Peak Ratio) — ratio of the two strongest LSP peaks. "
            "High PR indicates a single dominant reflection surface (ice):"
        )
        st.latex(r"PR = \frac{P_1}{P_2}")

        st.markdown("#### Wavelet Indicator")
        st.markdown(
            "**AF** (Area Factor) — Song 2022's snow-on-ice indicator. "
            "Continuous wavelet transform (Morlet, $\\omega_0 = 5$) of "
            "$d\\text{SNR}$ in the $\\sin(\\varepsilon)/(\\lambda/2)$ domain. "
            "The power curve at the dominant RH scale $s_0$ is integrated:"
        )
        st.latex(r"AF = \int \left| W_{CWT}(s_0,\, \tau) \right|^2 \, d\tau")
        st.markdown(
            "AF measures total reflected power across all elevation angles, which "
            "depends on both the Fresnel reflectivity of the surface layers and "
            "their roughness. Ice and snow-on-ice produce higher total RHCP "
            "reflectivity than open water — despite water's higher permittivity, "
            "the air-water interface converts most reflected energy to LHCP which "
            "the geodetic antenna rejects. The area factor captures both "
            "reflectivity and roughness effects without parametric assumptions, "
            "making it robust to multilayer conditions (Song 2022)."
        )

        st.markdown("#### Envelope Indicator")
        st.markdown(
            "**$\\gamma$** (Damping) — Strandberg 2017. The Hilbert transform "
            "envelope of the detrended signal decays with elevation. Fitting "
            "the log-envelope gives the damping coefficient:"
        )
        st.latex(
            r"\ln \left| \mathcal{H}\{d\text{SNR}\} \right| = "
            r"\ln A_0 - 4 k^2 \gamma \sin^2 \varepsilon"
            r"\qquad \text{where } k = 2\pi / \lambda"
        )
        st.markdown(
            "Low $\\gamma$ → slow decay → smooth surface (ice). "
            "High $\\gamma$ → fast decay → rough surface (water)."
        )

    st.markdown("#### Voting and Scoring")
    st.markdown(
        "Each indicator compares the daily value against thresholds derived "
        "from the **ice-free calibration months** (p30 and p70 of summer daily "
        "medians). Values beyond the ice threshold vote $v_i = -1$ (ice), "
        "beyond the water threshold vote $v_i = +1$ (water), between = transition "
        "($v_i = 0$). Sector score is the weighted mean:"
    )
    st.latex(r"S_{\text{sector}} = \frac{\sum_i w_i \, v_i}{\sum_i w_i}")
    st.markdown(
        "Station-level score combines sectors weighted by arc count $n_j$:"
    )
    st.latex(r"S_{\text{station}} = \frac{\sum_j n_j \, S_j}{\sum_j n_j}")

    # --- Indicator table ---
    st.markdown("#### Indicator Weights")
    rows = [
        ("Amplitude mean", "2.0", "High (≥ p70)", "Low (≤ p30)", "Standard GNSS-IR"),
        ("Amplitude CV", "1.5", "Low (≤ p25)", "High (≥ p75)", "Standard GNSS-IR"),
        ("RH std dev", "1.0", "Low (≤ p30)", "High (≥ p70)", "Standard GNSS-IR"),
    ]
    if has_features:
        rows.insert(1, ("**CLR** (clarity ratio)", "2.0", "High (≥ p70)", "Low (≤ p30)", "Purnell 2024"))
        rows.insert(3, ("**AF** (area factor)", "1.5", "High (≥ p70)", "Low (≤ p30)", "Song 2022"))
        rows.insert(5, ("**PR** (peak ratio)", "1.0", "High (≥ p70)", "Low (≤ p30)", "Purnell 2024"))
        rows.insert(6, ("**γ** (damping)", "1.0", "Low (≤ p30)", "High (≥ p70)", "Strandberg 2017"))

    header = "| Indicator | Weight | Ice Signal | Water Signal | Source |\n"
    header += "|-----------|--------|-----------|--------------|--------|\n"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows)
    st.markdown(header + body)

    if has_features:
        st.caption(
            "Thresholds are **summer-anchored**: percentiles computed from the "
            "ice-free months rather than the full year, avoiding bias at stations "
            "with long ice seasons where full-year percentiles are dominated by "
            "ice values."
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
    # Find all sector score columns, sorted numerically by azimuth
    score_cols = [c for c in clf.columns
                  if c.endswith("_score") and c.startswith("az")]
    score_cols.sort(key=lambda c: int(c.replace("az", "").replace("_score", "")))
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
    cmap.set_bad(color="#8B4513")  # brown for land/missing

    # Mask NaN values (including land sectors which have NaN scores)
    masked = np.ma.masked_invalid(score_matrix)

    fig, ax = plt.subplots(figsize=(14, max(3, len(score_cols) * 0.35)), dpi=100)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest")

    # X-axis: show month labels
    date_series = pd.Series(dates)
    month_starts = []
    for m in range(1, 13):
        mask = pd.to_datetime(date_series).dt.month == m
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
        ax.text(len(dates) + 2, i, "LAND", fontsize=7, va="center", color="#8B4513")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_ticks([-1, -0.33, 0, 0.33, 1])
    cbar.set_ticklabels(["Ice", "", "Trans", "", "Water"])

    ax.set_title("Per-Sector Classification Score (blue=ice, gray=transition, green=water, brown=land)")
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
            return "background-color: #bbdefb; color: #1a1a1a"
        elif v >= 0.33:
            return "background-color: #c8e6c9; color: #1a1a1a"
        return "background-color: #fff9c4; color: #1a1a1a"

    styled = table_df.style.map(_score_color, subset=["Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_calibration_validation(clf, ice_free_months):
    """Show whether ice-free calibration months are appropriate.

    Displays monthly box plots of key indicators with the calibration window
    highlighted, plus percentile analysis.
    """
    if not ice_free_months:
        st.caption("No ice-free calibration months configured for this station.")
        return

    has_features = _has_snr_features(clf)
    clf = clf.copy()
    clf["month"] = clf["date_dt"].dt.month

    if has_features:
        clf["clr_med"] = _median_across_sectors(clf, "clr")
        clf["af_med"] = _median_across_sectors(clf, "af")
        clf["gamma_med"] = _median_across_sectors(clf, "gamma")

    # Determine which indicators to show
    panels = [("amp_mean", "Amplitude Mean", True)]
    if has_features:
        panels = [
            ("clr_med", "CLR (Clarity Ratio)", True),
            ("af_med", "AF (Area Factor)", True),
            ("amp_mean", "Amplitude Mean", True),
            ("gamma_med", "Damping (γ)", False),
        ]

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4), dpi=100)
    if n_panels == 1:
        axes = [axes]

    cal_label = ", ".join(MONTH_NAMES.get(m, str(m)) for m in ice_free_months)
    months_present = sorted(clf["month"].unique())

    for ax, (col, label, high_is_ice) in zip(axes, panels):
        # Monthly box plot data
        month_data = []
        month_positions = []
        month_labels = []
        for m in months_present:
            vals = clf[clf["month"] == m][col].dropna().values
            if len(vals) > 0:
                month_data.append(vals)
                month_positions.append(m)
                month_labels.append(MONTH_NAMES.get(m, str(m)))

        if not month_data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(month_data, positions=month_positions, widths=0.6,
                        patch_artist=True, showfliers=False)

        # Color boxes: highlight calibration months
        for i, (patch, pos) in enumerate(zip(bp["boxes"], month_positions)):
            if pos in ice_free_months:
                patch.set_facecolor("#c8e6c9")
                patch.set_edgecolor(CLASS_COLOR["water"])
                patch.set_linewidth(1.5)
            else:
                patch.set_facecolor("#e8e8e8")
                patch.set_edgecolor("#999999")

        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels, fontsize=7, rotation=45)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=8)

        # Add ice/water arrow annotation
        direction = "↑ ice" if high_is_ice else "↓ ice"
        ax.annotate(direction, xy=(0.02, 0.98), xycoords="axes fraction",
                    fontsize=7, color=CLASS_COLOR["ice"], va="top",
                    fontweight="bold")

    fig.suptitle(
        "Monthly Distributions — green = ice-free calibration ({})".format(cal_label),
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Percentile analysis
    summer = clf[clf["month"].isin(ice_free_months)]
    rest = clf[~clf["month"].isin(ice_free_months)]
    if len(summer) < 5:
        st.caption("Insufficient data in calibration months for percentile analysis.")
        return

    pct_rows = []
    for col, label, high_is_ice in panels:
        all_vals = clf[col].dropna().sort_values().values
        summer_med = summer[col].median()
        if len(all_vals) == 0 or np.isnan(summer_med):
            continue
        pctile = np.searchsorted(all_vals, summer_med) / len(all_vals) * 100
        # For high_is_ice indicators, summer should be at LOW percentile
        # For low_is_ice (damping), summer should be at HIGH percentile
        expected = "low" if high_is_ice else "high"
        actual_pos = "low" if pctile < 40 else ("mid" if pctile < 60 else "high")
        ok = (expected == actual_pos) or (expected == "low" and pctile < 50) or (expected == "high" and pctile > 50)
        pct_rows.append({
            "Indicator": label,
            "Summer Median": "{:.1f}".format(summer_med),
            "Percentile": "{:.0f}%".format(pctile),
            "Expected": expected,
            "Valid": "yes" if ok else "marginal",
        })

    if pct_rows:
        st.markdown("**Percentile of calibration months within full-year distribution:**")
        pct_df = pd.DataFrame(pct_rows)

        def _valid_color(val):
            if val == "yes":
                return "background-color: #c8e6c9; color: #1a1a1a"
            return "background-color: #fff9c4; color: #1a1a1a"

        styled = pct_df.style.map(_valid_color, subset=["Valid"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Summary text
        n_valid = sum(1 for r in pct_rows if r["Valid"] == "yes")
        st.caption(
            "{}/{} indicators confirm that {} values represent the water "
            "baseline. {}".format(
                n_valid, len(pct_rows), cal_label,
                "Calibration months are appropriate."
                if n_valid >= len(pct_rows) * 0.6
                else "Consider adjusting ice-free month selection."
            )
        )


def _load_example_arcs(station, year):
    """Load precomputed example arcs (ice and water) for single-arc displays."""
    path = (PROJECT_ROOT / "results_annual" / station
            / f"{station}_{year}_example_arcs.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_per_arc(station, year):
    """Load per-arc parquet for raw amplitude and standard indicator plots."""
    path = (PROJECT_ROOT / "results_annual" / station
            / f"{station}_{year}_per_arc.parquet")
    if not path.exists():
        return None
    return safe_read_parquet(path)


def _render_raw_amplitude(per_arc):
    """Section 1: Raw daily mean amplitude scatter — what the antenna sees."""
    daily_amp = per_arc.groupby("date")["Amp"].mean()
    dates = pd.to_datetime(daily_amp.index)
    months = dates.month

    # Color by season
    season_colors = []
    for m in months:
        if m in (12, 1, 2, 3):
            season_colors.append("#1565c0")   # winter blue
        elif m in (4, 5):
            season_colors.append("#7986cb")   # spring
        elif m in (6, 7, 8):
            season_colors.append("#2e7d32")   # summer green
        elif m in (9, 10, 11):
            season_colors.append("#f9a825")   # autumn
        else:
            season_colors.append("#888888")

    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    ax.scatter(dates, daily_amp.values, c=season_colors, s=8, alpha=0.7)
    ax.set_ylabel("Mean Amplitude")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Each point is one day's mean reflected signal strength across all "
        "azimuth sectors. Blue = winter, green = summer, yellow = autumn."
    )


def _render_standard_indicators(clf):
    """Section 2: Three-panel figure of amplitude mean, CV, RH std."""
    clf = clf.copy()
    clf["month"] = clf["date_dt"].dt.month

    panels = [
        ("amp_mean", "Amplitude Mean"),
        ("amp_cv", "Amplitude CV"),
        ("rh_std", "RH Std Dev (m)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=100)

    for ax, (col, label) in zip(axes, panels):
        if col not in clf.columns:
            ax.set_visible(False)
            continue

        vals = clf[col].values
        dates = clf["date_dt"].values

        ax.scatter(dates, vals, s=6, alpha=0.6, c="#555555")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Shade Mar-May (ice) and Jul-Aug (water)
        year = clf["date_dt"].dt.year.iloc[0]
        ice_start = pd.Timestamp(year=year, month=3, day=1)
        ice_end = pd.Timestamp(year=year, month=5, day=31)
        water_start = pd.Timestamp(year=year, month=7, day=1)
        water_end = pd.Timestamp(year=year, month=8, day=31)
        ax.axvspan(ice_start, ice_end, alpha=0.1, color="#1565c0", label="Ice season")
        ax.axvspan(water_start, water_end, alpha=0.1, color="#2e7d32", label="Water season")

        # Rotate date labels to prevent overlap
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Standard GNSS-IR indicators over the year. Blue shading = typical ice "
        "season (Mar-May), green = ice-free calibration (Jul-Aug). "
        "Amplitude alone cannot resolve the freeze-up transition — we need "
        "indicators that measure *how* the signal reflects, not just *how strong* it is."
    )


def _render_damping_section(clf, examples):
    """Section 3: Damping parameter with single-arc envelope display."""
    st.markdown(
        "**$\\gamma$** (Damping) — Strandberg 2017. The Hilbert transform "
        "envelope of the detrended signal decays with elevation. Fitting "
        "the log-envelope gives the damping coefficient:"
    )
    st.latex(
        r"\ln \left| \mathcal{H}\{d\text{SNR}\} \right| = "
        r"\ln A_0 - 4 k^2 \gamma \sin^2 \varepsilon"
        r"\qquad \text{where } k = 2\pi / \lambda"
    )
    st.markdown(
        "Low $\\gamma$ → slow decay → smooth surface (ice). "
        "High $\\gamma$ → fast decay → rough surface (water)."
    )

    # Single-arc example
    if examples and "ice" in examples and "water" in examples:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)

        for ax, label, color in [(ax1, "ice", "#1565c0"), (ax2, "water", "#2e7d32")]:
            arc = examples[label]
            ele = np.array(arc["elevation"])
            dsnr = np.array(arc["dsnr"])
            env = np.array(arc["envelope"])

            ax.plot(ele, dsnr, color="#999999", linewidth=0.5, alpha=0.7)
            ax.plot(ele, env, color=color, linewidth=2,
                    label=f"Envelope (γ={arc['gamma']:.4f})")
            ax.plot(ele, -env, color=color, linewidth=2, alpha=0.5)
            ax.set_title(
                f"{label.capitalize()} — {arc['date']} PRN {arc['sat']}",
                fontsize=10,
            )
            ax.set_xlabel("Elevation (°)")
            ax.set_ylabel("dSNR")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "The envelope decays faster over rough water (high γ) than smooth "
            "ice (low γ). The Hilbert transform extracts this decay rate per "
            "arc without needing the full Strandberg multi-day inversion."
        )
    else:
        st.caption(
            "Single-arc example not available. Re-run the feature extractor "
            "to generate example arcs."
        )

    # γ timeseries
    clf = clf.copy()
    clf["gamma_med"] = _median_across_sectors(clf, "gamma")
    if "gamma_med" in clf.columns and clf["gamma_med"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        for cls, color in CLASS_COLOR.items():
            mask = clf["classification"] == cls
            if mask.sum() == 0:
                continue
            subset = clf[mask]
            ax.scatter(subset["date_dt"], subset["gamma_med"],
                       c=color, s=6, alpha=0.7, label=cls.capitalize())
        ax.set_ylabel("γ (damping)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_area_factor_section(clf, station_id, year, examples):
    """Section 4: Area factor with corrected physics and wavelet scalograms."""
    st.markdown(
        "**AF** (Area Factor) — Song 2022. "
        "Continuous wavelet transform (Morlet, $\\omega_0 = 5$) of "
        "$d\\text{SNR}$ in the $\\sin(\\varepsilon)/(\\lambda/2)$ domain. "
        "The power curve at the dominant RH scale $s_0$ is integrated:"
    )
    st.latex(r"AF = \int \left| W_{CWT}(s_0,\, \tau) \right|^2 \, d\tau")
    st.markdown(
        "AF measures total reflected power across all elevation angles, which "
        "depends on both the Fresnel reflectivity of the surface layers and "
        "their roughness. Ice and snow-on-ice produce higher total RHCP "
        "reflectivity than open water — despite water's higher permittivity, "
        "the air-water interface converts most reflected energy to LHCP which "
        "the geodetic antenna rejects. The area factor captures both "
        "reflectivity and roughness effects without parametric assumptions, "
        "making it robust to multilayer conditions (Song 2022)."
    )

    # Wavelet scalograms (existing)
    try:
        _render_wavelet_comparison(station_id, year)
    except Exception as e:
        st.warning(f"Wavelet comparison unavailable: {e}")

    # AF timeseries
    clf = clf.copy()
    clf["af_med"] = _median_across_sectors(clf, "af")
    if "af_med" in clf.columns and clf["af_med"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        for cls, color in CLASS_COLOR.items():
            mask = clf["classification"] == cls
            if mask.sum() == 0:
                continue
            subset = clf[mask]
            ax.scatter(subset["date_dt"], subset["af_med"],
                       c=color, s=6, alpha=0.7, label=cls.capitalize())
        ax.set_ylabel("Area Factor")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_voting_section(clf):
    """Section 5: Purnell 2024 — CLR/PR intro, weight table, all indicators."""
    has_features = _has_snr_features(clf)

    if has_features:
        st.markdown(
            "**CLR** (Clarity Ratio) — Purnell 2024's best single-feature "
            "discriminator (77.7% accuracy alone). Ratio of the dominant LSP peak "
            "to the mean of all other peaks:"
        )
        st.latex(r"CLR = \frac{P_1}{\bar{P}_{2 \ldots N}}")
        st.markdown(
            "**PR** (Peak Ratio) — ratio of the two strongest LSP peaks. "
            "High PR indicates a single dominant reflection surface (ice):"
        )
        st.latex(r"PR = \frac{P_1}{P_2}")

    st.markdown(
        "Each indicator compares the daily value against thresholds derived "
        "from the **ice-free calibration months** (p30 and p70 of summer daily "
        "medians). Sector score is the weighted mean of all indicator votes:"
    )
    st.latex(r"S_{\text{sector}} = \frac{\sum_i w_i \, v_i}{\sum_i w_i}")

    # Indicator weight table
    rows = [
        ("Amplitude mean", "2.0", "High (≥ p70)", "Low (≤ p30)", "Standard GNSS-IR"),
        ("Amplitude CV", "1.5", "Low (≤ p25)", "High (≥ p75)", "Standard GNSS-IR"),
        ("RH std dev", "1.0", "Low (≤ p30)", "High (≥ p70)", "Standard GNSS-IR"),
    ]
    if has_features:
        rows.insert(1, ("**CLR** (clarity ratio)", "2.0", "High (≥ p70)", "Low (≤ p30)", "Purnell 2024"))
        rows.insert(3, ("**AF** (area factor)", "1.5", "High (≥ p70)", "Low (≤ p30)", "Song 2022"))
        rows.insert(5, ("**PR** (peak ratio)", "1.0", "High (≥ p70)", "Low (≤ p30)", "Purnell 2024"))
        rows.insert(6, ("**γ** (damping)", "1.0", "Low (≤ p30)", "High (≥ p70)", "Strandberg 2017"))

    header = "| Indicator | Weight | Ice Signal | Water Signal | Source |\n"
    header += "|-----------|--------|-----------|--------------|--------|\n"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows)
    st.markdown(header + body)

    # All-indicator evidence plot
    if has_features:
        st.markdown(
            "Different indicators respond to different physical processes. "
            "γ detects pre-freeze surface calming (roughness); CLR only responds "
            "when a coherent ice sheet forms. Their divergence during freeze-up "
            "reveals the transition sequence."
        )
        _render_indicator_timeseries(clf)


def render_ice_comparison_tab(station_id, year):
    """Render ice classification tab as a paper-by-paper narrative.

    Walks from raw signal → standard indicators → damping → area factor →
    voting → classification result → validation → summary.
    """
    st.header("Ice Classification")

    clf = _load_classification(station_id, year)
    if clf is None:
        st.info(
            f"No classification found. Run:\n"
            f"```\npython scripts/ice_classifier.py --station {station_id} --year {year}\n```"
        )
        return

    has_features = _has_snr_features(clf)
    per_arc = _load_per_arc(station_id, year)
    examples = _load_example_arcs(station_id, year)
    ice_free_months = _load_ice_free_months(station_id)

    # --- Section 0: Metrics strip ---
    counts = clf["classification"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days Classified", len(clf))
    c2.metric("Ice", counts.get("ice", 0))
    c3.metric("Transition", counts.get("transition", 0))
    c4.metric("Water", counts.get("water", 0))

    # --- Section 1: What does the antenna see? ---
    if per_arc is not None:
        st.subheader("What does the antenna see?")
        _render_raw_amplitude(per_arc)

    # --- Section 2: Amplitude tells part of the story ---
    st.subheader("Amplitude tells part of the story")
    _render_standard_indicators(clf)

    # --- Section 3: Damping sees roughness changes (Strandberg 2017) ---
    if has_features:
        st.subheader("The damping parameter sees roughness changes")
        _render_damping_section(clf, examples)

    # --- Section 4: The area factor handles snow-on-ice (Song 2022) ---
    if has_features:
        st.subheader("The area factor handles snow-on-ice")
        _render_area_factor_section(clf, station_id, year, examples)

    # --- Section 5: Multiple indicators vote together (Purnell 2024) ---
    st.subheader("Multiple indicators vote together")
    _render_voting_section(clf)

    # --- Section 6: The classification result ---
    st.subheader("The classification result")
    _render_score_timeseries(clf, station_id=station_id, year=year)

    # --- Section 7: Validation ---
    st.subheader("Validation")

    if ice_free_months:
        with st.expander("Calibration validation", expanded=False):
            st.caption(
                "Are the ice-free calibration months ({}) representative of open "
                "water conditions?".format(
                    ", ".join(MONTH_NAMES.get(m, str(m)) for m in ice_free_months))
            )
            _render_calibration_validation(clf, ice_free_months)

    with st.expander("Per-sector analysis", expanded=False):
        st.caption(
            "Each row is an azimuth sector. Blue = ice vote, green = water vote. "
            "Brown = land-flagged (excluded from consensus)."
        )
        _render_sector_heatmap(clf)

    # S1 SAR validation
    s1_index = load_s1_index(station_id)
    if s1_index is not None:
        s1_year = s1_index[s1_index["date_dt"].dt.year == year].copy()
        matched = _match_s1_to_classification(s1_year, clf)
        matched["thumb_exists"] = matched["acquisition_date"].apply(
            lambda d: get_thumb_path(station_id, d).exists()
        )
        matched = matched[matched["thumb_exists"]]

        if not matched.empty:
            with st.expander(
                f"S1 SAR validation ({len(matched)} scenes)", expanded=False
            ):
                _render_thumbnail_grid(matched, station_id)
                _render_scene_detail(matched, station_id)

                try:
                    _render_analysis_zone_legend(station_id, year)
                except Exception:
                    pass

    # --- Section 8: Summary ---
    st.subheader("Summary")
    _render_monthly_summary(clf)
