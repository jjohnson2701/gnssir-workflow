#!/usr/bin/env python3
"""GNSS-IR Dashboard — Plotly Dash v2.

Five-tab layout:
  1. Overview:  Map + station selector + data quality
  2. Time Series: RH + v3 classification + reference gauge
  3. Polar: Reflection points by frequency, date-selectable
  4. Ice Classification: v3 state timeline + features + SMAP
  5. Imagery: S1/S2 viewer with Fresnel zone overlay

Usage:
    python dashboard_dash.py
    python dashboard_dash.py --port 8050
"""

import argparse
import base64
import glob
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ALL, MATCH
import dash_leaflet as dl
# dash_leaflet.express not used — plain markers with click events

# Dark theme for all Plotly figures
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
)
DARK_BG = "#0d1117"
DARK_CARD = "#161b22"
DARK_BORDER = "#30363d"
DARK_TEXT = "#c9d1d9"

PROJECT_ROOT = Path(__file__).resolve().parent

FREQ_COLORS = {
    "L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c",
    "E6": "#d62728", "B3": "#9467bd", "OTHER": "#8c564b",
}

V3_COLORS = {
    "open_water": "#2d9a6b",
    "freeze_up": "#f0ad4e",
    "ice_surface": "#4a90d9",
    "ice_layered": "#7b4fbf",
    "ice_decaying": "#e07b39",
    "break_up": "#d94452",
    # Discovery mode
    "baseline": "#2d9a6b",
    "anomalous": "#e07b39",
    "regime_change": "#f0ad4e",
}

V3_ORDER = ["open_water", "freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"]
DISCOVERY_ORDER = ["baseline", "regime_change", "anomalous"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stations_config():
    with open(PROJECT_ROOT / "config" / "stations_config.json") as f:
        return json.load(f)


def list_available_stations():
    results = PROJECT_ROOT / "results_annual"
    stations = {}
    cfg = load_stations_config()
    for d in sorted(results.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Prefer arc_table (4-layer), fall back to per_arc (legacy)
        parquets = sorted(d.glob(f"{name}_*_arc_table.parquet"))
        if not parquets:
            parquets = sorted(d.glob(f"{name}_*_per_arc.parquet"))
        if parquets and name in cfg:
            years = [int(p.stem.split("_")[1]) for p in parquets]
            stations[name] = {
                "years": years,
                "lat": cfg[name].get("latitude_deg", 0),
                "lon": cfg[name].get("longitude_deg", 0),
            }
    return stations


def load_per_arc(station, year):
    """Load Layer 1 arc-level data (arc_table preferred, per_arc fallback)."""
    for name in ["arc_table", "per_arc"]:
        p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "date" in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"])
            return df
    return None


def load_v3(station, year):
    """Load Layer 3 ice state (v3 preferred, v2 fallback, v1 legacy)."""
    for name in ["ice_classification_v3", "ice_classification_v2", "ice_state"]:
        p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date_dt"] = pd.to_datetime(df["date"])
            # Normalize to v3_state column expected by all callbacks
            if "v3_state" not in df.columns:
                if "state" in df.columns:
                    # v2: water/freeze_up/ice/break_up → map to v3 names
                    v2_to_v3 = {"water": "open_water", "ice": "ice_surface"}
                    df["v3_state"] = df["state"].map(
                        lambda s: v2_to_v3.get(s, s))
                elif "classification" in df.columns:
                    # v1: ice/water/transition → map to v3 names
                    v1_to_v3 = {"water": "open_water", "ice": "ice_surface",
                                "transition": "freeze_up"}
                    df["v3_state"] = df["classification"].map(v1_to_v3)
            # Ensure doy column exists (needed by features panel)
            if "doy" not in df.columns:
                df["doy"] = df["date_dt"].dt.dayofyear
            return df
    return None


def load_snr_features(station, year):
    """Load SNR features — embedded in arc_table or standalone snr_features."""
    # Check arc_table first (4-layer: features are columns in arc_table)
    at = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_arc_table.parquet"
    if at.exists():
        df = pd.read_parquet(at)
        if "CLR" in df.columns:
            return df
    # Fall back to standalone snr_features (legacy)
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_snr_features.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_smap(station, year):
    p = PROJECT_ROOT / "results_annual" / station / "smap" / f"{station}_{year}_smap_comparison.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_era5(station, year):
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_era5.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_v2(station, year):
    """Load v2 classification with Mahalanobis distances."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_ice_classification_v2.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    if "doy" not in df.columns:
        df["doy"] = df["date"].dt.dayofyear
    return df


def load_mahal_threshold(station):
    """Load Mahalanobis threshold from saved model."""
    import pickle
    model_path = PROJECT_ROOT / "models" / f"{station}_water_state_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model.get("optimal_threshold")


def load_cross_station_summary():
    """Load cross-station summary parquet (if it exists)."""
    p = PROJECT_ROOT / "results_annual" / "cross_station_summary.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_daily_features(station, year):
    """Load Layer 2 daily features (aggregated from arc_table by feature_aggregator.py)."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_daily_features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])
        df["doy"] = df["date_dt"].dt.dayofyear
    return df


def load_prn_weights(station, year):
    """Load per-PRN discriminating power weights JSON (from compute_prn_weights.py)."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_prn_weights.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_s1_images(station):
    s1_dir = PROJECT_ROOT / "data" / station / "s1_fresnel"
    if not s1_dir.exists():
        return []
    tifs = sorted(s1_dir.glob(f"{station}_*_s1_fresnel.tif"))
    dates = []
    for t in tifs:
        ds = t.stem.split("_")[1]
        dates.append({"date": f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", "path": str(t)})
    return dates


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_polar_figure(per_arc, title="Polar Diagram"):
    if per_arc is None or per_arc.empty:
        return go.Figure().update_layout(title="No data")

    df = per_arc.copy()
    if "freq_group" not in df.columns:
        df["freq_group"] = "L1"

    if "eminO" in df.columns and "emaxO" in df.columns:
        mid_elev = (df["eminO"] + df["emaxO"]) / 2
    else:
        mid_elev = pd.Series(10.0, index=df.index)
    mid_elev_rad = np.radians(mid_elev.clip(lower=1))
    df["refl_dist"] = df["RH"] / np.tan(mid_elev_rad)

    fig = go.Figure()
    for freq in sorted(df["freq_group"].unique()):
        sub = df[df["freq_group"] == freq]
        color = FREQ_COLORS.get(freq, "#888888")
        fig.add_trace(go.Scatterpolar(
            r=sub["refl_dist"], theta=sub["Azim"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.5),
            name=freq,
            hovertemplate="Az: %{theta:.0f}°<br>Dist: %{r:.0f}m<br>"
                          "RH: %{customdata[0]:.2f}m<br>Amp: %{customdata[1]:.1f}<extra></extra>",
            customdata=sub[["RH", "Amp"]].values,
        ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(range=[0, df["refl_dist"].quantile(0.95)]),
            bgcolor=DARK_CARD,
        ),
        title=dict(text=title, font=dict(size=13)),
        showlegend=True, legend=dict(font=dict(size=10)),
        margin=dict(l=40, r=40, t=50, b=40), height=450,
        **PLOTLY_DARK,
    )
    return fig


def build_timeseries_figure(per_arc, v3=None, station="", year=0, s1_dates=None):
    if per_arc is None or per_arc.empty:
        return go.Figure().update_layout(title="No data", **PLOTLY_DARK)

    df = per_arc.copy()
    if "date_dt" not in df.columns and "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])
    if "freq_group" not in df.columns:
        df["freq_group"] = "L1"

    n_rows = 2 if v3 is not None else 1
    row_heights = [0.75, 0.25] if n_rows == 2 else [1.0]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.05)

    # RH by frequency
    for freq in sorted(df["freq_group"].unique()):
        sub = df[df["freq_group"] == freq]
        daily = sub.groupby("date_dt").agg(
            rh_median=("RH", "median"), n_arcs=("RH", "count")).reset_index()
        color = FREQ_COLORS.get(freq, "#888888")
        fig.add_trace(go.Scatter(
            x=daily["date_dt"], y=daily["rh_median"],
            mode="markers", name=freq,
            marker=dict(size=4, color=color, opacity=0.6),
            hovertemplate=f"{freq}<br>RH: %{{y:.3f}}m<br>n=%{{customdata}}<extra></extra>",
            customdata=daily["n_arcs"],
        ), row=1, col=1)

    # Overall daily median
    daily_all = df.groupby("date_dt")["RH"].median().reset_index()
    fig.add_trace(go.Scatter(
        x=daily_all["date_dt"], y=daily_all["RH"],
        mode="lines", name="Daily median",
        line=dict(color="white", width=1.5),
    ), row=1, col=1)
    fig.update_yaxes(title_text="Reflector Height (m)", row=1, col=1)

    # S1 acquisition date markers — clickable triangles along top
    if s1_dates:
        # Get v3 state labels for S1 dates
        v3_map = {}
        if v3 is not None and "v3_state" in v3.columns:
            v3_map = dict(zip(v3["date_dt"].dt.strftime("%Y-%m-%d"), v3["v3_state"]))

        s1_x = [pd.Timestamp(d["date"]) for d in s1_dates]
        rh_max = daily_all["RH"].max()
        s1_y = [rh_max + 0.05] * len(s1_x)
        s1_labels = [v3_map.get(d["date"], "?") for d in s1_dates]
        s1_colors = [V3_COLORS.get(v3_map.get(d["date"], ""), "#aaa") for d in s1_dates]

        fig.add_trace(go.Scatter(
            x=s1_x, y=s1_y,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=16, color=s1_colors,
                        line=dict(color="white", width=1.5)),
            name="S1 imagery (click ▼)",
            customdata=[d["date"] for d in s1_dates],
            hovertemplate="<b>S1: %{customdata}</b><br>State: %{text}<br><i>Click to view imagery</i><extra></extra>",
            text=s1_labels,
            textposition="top center",
            textfont=dict(size=8, color="rgba(0,0,0,0)"),  # hidden text, just for click area
        ), row=1, col=1)

    # v3 classification strip
    if v3 is not None and not v3.empty and "v3_state" in v3.columns:
        for state in _state_order(_get_mode(v3)):
            mask = v3["v3_state"] == state
            if mask.sum() > 0:
                sub = v3[mask]
                fig.add_trace(go.Bar(
                    x=sub["date_dt"], y=[1] * len(sub),
                    marker=dict(color=V3_COLORS.get(state, "#999")),
                    name=state, showlegend=True,
                ), row=2, col=1)
        fig.update_yaxes(title_text="State", showticklabels=False, range=[0, 1.2], row=2, col=1)

    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_layout(
        title=dict(text=f"{station} {year} — click ▼ markers to view S1 imagery", font=dict(size=13)),
        height=400, margin=dict(l=50, r=20, t=50, b=30),
        barmode="stack", legend=dict(font=dict(size=9), orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def build_ice_features_figure(snr_features, v3, era5=None, smap=None,
                               station="", year=0, daily_features=None):
    """Multi-panel: SNR features + temperature + SMAP, colored by v3 state.

    If daily_features is provided, uses pre-aggregated best-variant columns
    from daily_features.parquet. Otherwise falls back to computing daily
    medians from arc-level snr_features.
    """
    if v3 is None:
        return go.Figure().update_layout(title="No data")

    mode = _get_mode(v3)
    order = _state_order(mode)
    v3c = v3[["doy", "v3_state"]].drop_duplicates("doy")

    # --- Determine feature panels ---
    # Try daily_features (best variants) first, fall back to arc-level
    use_daily = False
    if daily_features is not None:
        pooled = daily_features[daily_features["azimuth_bin"] == -1].copy() if "azimuth_bin" in daily_features.columns else daily_features.copy()
        if not pooled.empty and "doy" in pooled.columns:
            pooled = pooled.merge(v3c, on="doy", how="left")
            use_daily = True

    if not use_daily:
        if snr_features is None:
            return go.Figure().update_layout(title="No data")
        pooled = snr_features.groupby("doy")[["CLR", "AF", "gamma", "VS"]].median().reset_index()
        pooled = pooled.merge(v3c, on="doy", how="left")

    # Build panel list: (column_name, y_label)
    if use_daily:
        panels = []
        # Core features in preferred order, using best variant when available
        panel_candidates = [
            ("gamma", "gamma_med", "Damping (γ)"),
            ("VS", "vs_wmed", "SNR Variance (PRN-wt)"),
            ("AF", "af_wmed", "Area Factor (PRN-wt)"),
            ("MS", "ms_z", "Mean Spectral Power (z)"),
            ("CLR", "clr_wmed", "CLR (PRN-wt)"),
            ("gamma_r2", "gamma_r2_med", "Surface Coherence (γ_r2)"),
        ]
        for _base, best_col, label in panel_candidates:
            if best_col in pooled.columns:
                panels.append((best_col, label))
        # Fallback: if no best-variant columns found, use raw arc-level names
        if not panels:
            for col in ["gamma", "VS", "AF"]:
                if col in pooled.columns:
                    panels.append((col, FEATURE_LABELS.get(col + "_med", col)))
    else:
        panels = [("gamma", "Damping (γ)"), ("VS", "SNR Variance"), ("AF", "Area Factor")]

    n_feature_panels = len(panels)
    n_panels = n_feature_panels
    has_era5 = era5 is not None and "t2m_mean" in era5.columns
    has_smap = smap is not None and "polarization_diff" in smap.columns
    if has_era5:
        n_panels += 1
    if has_smap:
        n_panels += 1

    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03)

    row = 1
    for feat_col, label in panels:
        if feat_col not in pooled.columns:
            row += 1
            continue
        for state in order:
            mask = pooled["v3_state"] == state
            if mask.sum() > 0:
                sub = pooled[mask]
                fig.add_trace(go.Scatter(
                    x=sub["doy"], y=sub[feat_col], mode="markers",
                    marker=dict(size=4, color=V3_COLORS.get(state, "#999"), opacity=0.6),
                    name=state, showlegend=(row == 1),
                    legendgroup=state,
                ), row=row, col=1)
        fig.update_yaxes(title_text=label, row=row, col=1)
        row += 1

    if has_era5:
        era5m = era5.copy()
        if "doy" in era5m.columns:
            era5m = era5m.merge(v3c, on="doy", how="left")
            for state in order:
                mask = era5m["v3_state"] == state
                if mask.sum() > 0:
                    sub = era5m[mask]
                    fig.add_trace(go.Scatter(
                        x=sub["doy"], y=sub["t2m_mean"], mode="markers",
                        marker=dict(size=4, color=V3_COLORS.get(state, "#999"), opacity=0.6),
                        showlegend=False, legendgroup=state,
                    ), row=row, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)
            fig.update_yaxes(title_text="ERA5 Temp (°C)", row=row, col=1)
            row += 1

    if has_smap:
        smapm = smap.copy()
        if "doy" in smapm.columns:
            smapm = smapm.merge(v3c, on="doy", how="left")
            for state in order:
                mask = smapm["v3_state"] == state
                if mask.sum() > 0:
                    sub = smapm[mask]
                    fig.add_trace(go.Scatter(
                        x=sub["doy"], y=sub["polarization_diff"], mode="markers",
                        marker=dict(size=4, color=V3_COLORS.get(state, "#999"), opacity=0.6),
                        showlegend=False, legendgroup=state,
                    ), row=row, col=1)
            fig.update_yaxes(title_text="SMAP TBv-TBh (K)", row=row, col=1)
            row += 1

    tab_title = "Regime Detection Features" if mode == "discovery" else "Ice Classification Features"
    if use_daily:
        tab_title += " (best variants)"
    fig.update_xaxes(title_text="Day of Year", row=n_panels, col=1)
    fig.update_layout(
        title=f"{station} {year}: {tab_title}",
        height=200 * n_panels, margin=dict(l=60, r=20, t=50, b=30),
        legend=dict(font=dict(size=9), orientation="h", y=-0.05),
        **PLOTLY_DARK,
    )
    return fig


# ---------------------------------------------------------------------------
# Ice tab: new components
# ---------------------------------------------------------------------------

ICE_STATES_SET = {"freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"}
ANOMALOUS_STATES_SET = {"anomalous", "regime_change"}


def _get_mode(v3):
    """Detect classification mode from v3 DataFrame."""
    if v3 is None:
        return "validated"
    if "classification_mode" in v3.columns:
        return v3["classification_mode"].iloc[0]
    # Legacy fallback: if any discovery labels present, it's discovery
    if v3["v3_state"].isin(["baseline", "anomalous", "regime_change"]).any():
        return "discovery"
    return "validated"


def _state_order(mode):
    return DISCOVERY_ORDER if mode == "discovery" else V3_ORDER


def _non_baseline_states(mode):
    return ANOMALOUS_STATES_SET if mode == "discovery" else ICE_STATES_SET


FEATURE_LABELS = {
    # Raw medians
    "amp_mean": "Amplitude", "gamma_med": "Damping γ", "clr_med": "CLR",
    "af_med": "Area Factor", "pr_med": "Peak Ratio", "rh_std": "RH Std",
    "vs_med": "SNR Variance", "ms_med": "Mean Spectral Power",
    "gamma_r2_med": "Surface Coherence (γ_r2)",
    # Z-scored
    "clr_z": "CLR (z-scored)", "af_z": "AF (z-scored)", "pr_z": "PR (z-scored)",
    "gamma_z": "γ (z-scored)", "ms_z": "MS (z-scored)", "vs_z": "VS (z-scored)",
    # PRN-weighted medians
    "clr_wmed": "CLR (PRN-weighted)", "af_wmed": "AF (PRN-weighted)",
    "pr_wmed": "PR (PRN-weighted)", "gamma_wmed": "γ (PRN-weighted)",
    "ms_wmed": "MS (PRN-weighted)", "vs_wmed": "VS (PRN-weighted)",
    # Amplitude ratios
    "amp_ratio_L1_L5": "Amp ratio L1/L5", "amp_ratio_L1_L2C": "Amp ratio L1/L2C",
    # Other
    "delta_rh_mean": "ΔRH mean", "phase_circ_std": "Phase circ. std",
    "n_arcs": "Arc count", "frac_full_arc": "Full-arc fraction",
}

# Best normalization variant per feature (from literature review tests, Apr 2026)
BEST_VARIANT = {
    "AF": "af_wmed", "MS": "ms_z", "VS": "vs_wmed",
    "CLR": "clr_wmed", "PR": "pr_wmed", "gamma": "gamma_med",
    "gamma_r2": "gamma_r2_med", "amp": "amp_mean", "rh_std": "rh_std",
}

# Majority directions from cross-station analysis (network consensus)
MAJORITY_DIRECTION = {
    "amp_mean": "winter_high", "gamma_med": "summer_high", "clr_med": "summer_high",
    "af_med": "winter_high", "pr_med": "winter_high", "rh_std": "summer_high",
    "vs_med": "winter_high",
}

# Feature definitions for the reference panel
FEATURE_DEFINITIONS = {
    "CLR": {
        "name": "Clarity Ratio (CLR)",
        "source": "Purnell 2024, Eq. 4",
        "desc": "Ratio of dominant LSP peak power to mean of remaining peaks. "
                "Higher = cleaner single-reflector signal (smooth ice/water). "
                "Lower = multipath interference (rough surfaces, layered ice).",
        "variants": "clr_med (raw), clr_z (z-scored), clr_wmed (PRN-weighted)",
        "best": "clr_wmed (d=0.82)",
    },
    "AF": {
        "name": "Area Factor (AF)",
        "source": "Song 2022, Eq. 22-23",
        "desc": "CWT-based spectral power integrated around the dominant RH, "
                "with per-PRN ice-free baseline subtracted. Measures excess or "
                "deficit spectral energy relative to open-water conditions. "
                "Polarity is site-dependent (drops during ice at ROSS, rises at UMNQ).",
        "variants": "af_med (raw), af_z (z-scored), af_wmed (PRN-weighted)",
        "best": "af_wmed (d=1.68)",
    },
    "gamma": {
        "name": "Damping Parameter (γ)",
        "source": "Strandberg 2017",
        "desc": "Rate of SNR envelope decay with elevation. Fit per-arc via "
                "Hilbert envelope + Siegel robust regression on log(envelope) vs sin²(ε). "
                "Higher = rougher/wetter surface. Lower = smoother/frozen surface.",
        "variants": "gamma_med (raw), gamma_z (z-scored), gamma_wmed (PRN-weighted)",
        "best": "gamma_med (d=0.74)",
    },
    "gamma_r2": {
        "name": "Surface Coherence (γ_r2)",
        "source": "Strandberg 2017 (reinterpreted)",
        "desc": "R² of the damping envelope fit. NOT a quality gate — it is a surface "
                "coherence metric. Higher = more specular/smooth surface (ice, calm water, snow). "
                "Lower = more diffuse scattering (rough open water, wet snow).",
        "variants": "gamma_r2_med",
        "best": "gamma_r2_med",
    },
    "MS": {
        "name": "Mean Spectral Power (MS)",
        "source": "Derived from LSP",
        "desc": "Mean power across the LSP spectrum. Strongly PRN-dependent (ICC=88%), "
                "so raw values are dominated by satellite hardware differences. "
                "Z-scoring per (sat, freq) removes the hardware bias and reveals the "
                "surface-state signal.",
        "variants": "ms_med (raw), ms_z (z-scored), ms_wmed (PRN-weighted)",
        "best": "ms_z (d=1.92)",
    },
    "VS": {
        "name": "SNR Variance (VS)",
        "source": "Derived from detrended SNR",
        "desc": "Variance of the detrended SNR arc. Higher variance indicates stronger "
                "multipath interference patterns, typically from rough or layered surfaces.",
        "variants": "vs_med (raw), vs_wmed (PRN-weighted)",
        "best": "vs_wmed (d=1.84)",
    },
    "PR": {
        "name": "Peak Ratio (PR)",
        "source": "Derived from LSP",
        "desc": "Ratio of the dominant LSP peak to the noise floor. Similar concept to CLR "
                "but measures absolute peak prominence rather than relative clarity.",
        "variants": "pr_med (raw), pr_z (z-scored), pr_wmed (PRN-weighted)",
        "best": "pr_wmed (d=0.72)",
    },
    "amp": {
        "name": "Amplitude (amp_mean)",
        "source": "gnssrefl output",
        "desc": "Mean amplitude of the dominant LSP peak across arcs. Reflects overall "
                "signal strength of the reflected signal.",
        "variants": "amp_mean, amp_std, amp_cv, per-band (amp_L1_mean, etc.)",
        "best": "amp_mean",
    },
    "amp_ratio": {
        "name": "Amplitude Ratios (L1/L5, L1/L2C)",
        "source": "Literature review, Apr 2026",
        "desc": "Ratio of mean amplitude on L1 to L5 (or L2C). Different frequencies "
                "penetrate ice/snow differently, so the ratio can indicate surface layering "
                "or material changes invisible to single-frequency features.",
        "variants": "amp_ratio_L1_L5, amp_ratio_L1_L2C",
        "best": "TBD — newly added",
    },
    "rh_std": {
        "name": "RH Standard Deviation",
        "source": "Derived from gnssrefl RH",
        "desc": "Daily standard deviation of reflector height estimates. Higher during "
                "open water (waves, tides) than during stable ice cover.",
        "variants": "rh_std",
        "best": "rh_std",
    },
    "normalization": {
        "name": "Normalization Variants",
        "source": "feature_aggregator.py",
        "desc": "_med = raw median across arcs. "
                "_z = per-(satellite, frequency) z-scored before aggregating — removes "
                "hardware bias (critical for MS where ICC=88%). "
                "_wmed = PRN-weighted median where weight = |Cohen's d| from a training "
                "period — PRNs with stronger ice/water discrimination contribute more.",
        "variants": "Suffix: _med, _z, _wmed",
        "best": "Depends on feature — see individual entries",
    },
}


def build_feature_definitions_panel():
    """Collapsible reference panel with feature definitions."""
    rows = []
    for key, info in FEATURE_DEFINITIONS.items():
        rows.append(html.Tr([
            html.Td(info["name"], style={"fontWeight": "bold", "color": "#58a6ff",
                                          "verticalAlign": "top", "padding": "6px 12px 6px 0",
                                          "whiteSpace": "nowrap", "fontSize": "0.85rem"}),
            html.Td(info["desc"], style={"color": DARK_TEXT, "padding": "6px 12px 6px 0",
                                          "fontSize": "0.85rem", "lineHeight": "1.4"}),
            html.Td(info.get("best", ""), style={"color": "#2d9a6b", "padding": "6px 0",
                                                   "fontSize": "0.8rem", "whiteSpace": "nowrap"}),
        ]))

    return html.Details([
        html.Summary("Feature Definitions & Best Variants",
                     style={"cursor": "pointer", "color": "#58a6ff", "fontWeight": "bold",
                            "fontSize": "0.95rem", "padding": "8px 0"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Feature", style={"color": "#8b949e", "padding": "4px 12px 4px 0",
                                           "fontSize": "0.8rem", "fontWeight": "normal",
                                           "borderBottom": f"1px solid {DARK_BORDER}"}),
                html.Th("Description", style={"color": "#8b949e", "padding": "4px 12px 4px 0",
                                               "fontSize": "0.8rem", "fontWeight": "normal",
                                               "borderBottom": f"1px solid {DARK_BORDER}"}),
                html.Th("Best Variant", style={"color": "#8b949e", "padding": "4px 0",
                                                "fontSize": "0.8rem", "fontWeight": "normal",
                                                "borderBottom": f"1px solid {DARK_BORDER}"}),
            ])),
            html.Tbody(rows),
        ], style={"borderCollapse": "collapse", "width": "100%"}),
    ], style={"backgroundColor": DARK_CARD, "borderRadius": "6px",
              "border": f"1px solid {DARK_BORDER}", "padding": "8px 16px",
              "marginBottom": "8px"})


def build_state_timeline(v3, station, year):
    """Horizontal bar showing v3_state day-by-day across the year."""
    if v3 is None or "v3_state" not in v3.columns:
        return html.Div("No classification data", style={"color": DARK_TEXT, "padding": "8px"})

    fig = go.Figure()
    df = v3.sort_values("doy")

    # Build segments of consecutive same-state days for efficient rendering
    segments = []
    prev_state, seg_start = None, None
    for _, row in df.iterrows():
        if row["v3_state"] != prev_state:
            if prev_state is not None:
                segments.append((seg_start, prev_doy, prev_state))
            seg_start = row["doy"]
            prev_state = row["v3_state"]
        prev_doy = row["doy"]
    if prev_state is not None:
        segments.append((seg_start, prev_doy, prev_state))

    for start, end, state in segments:
        fig.add_trace(go.Bar(
            x=[end - start + 1], y=[f"{station} {year}"],
            base=[start - 0.5], orientation="h",
            marker_color=V3_COLORS.get(state, "#666"),
            name=state, showlegend=False,
            hovertemplate=f"{state}<br>DOY {start}–{end}<extra></extra>",
        ))

    # Add legend entries (one per state that appears)
    seen = set()
    for _, _, state in segments:
        if state not in seen:
            fig.add_trace(go.Bar(
                x=[0], y=[f"{station} {year}"], base=[0], orientation="h",
                marker_color=V3_COLORS.get(state, "#666"),
                name=state, showlegend=True, legendgroup=state,
                hoverinfo="skip",
            ))
            seen.add(state)

    fig.update_layout(
        barmode="stack", height=90,
        margin=dict(l=10, r=10, t=5, b=5),
        xaxis=dict(range=[0.5, 366.5], title=None, showticklabels=True,
                   dtick=30, gridcolor="#30363d"),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", y=-0.6, font=dict(size=10)),
        **PLOTLY_DARK,
    )
    return dcc.Graph(figure=fig, style={"height": "90px"},
                     config={"displayModeBar": False})


def build_metrics_strip(v3):
    """Row of summary metric cards, adapting labels to classification mode."""
    if v3 is None or "v3_state" not in v3.columns:
        return html.Div()

    mode = _get_mode(v3)
    counts = v3["v3_state"].value_counts()
    non_baseline = v3[v3["v3_state"].isin(_non_baseline_states(mode))]
    total_nb = len(non_baseline)

    if total_nb > 0:
        first_doy = int(non_baseline["doy"].min())
        last_doy = int(non_baseline["doy"].max())
        regime_len = last_doy - first_doy
    else:
        first_doy, last_doy, regime_len = None, None, 0

    def card(label, value, color="#c9d1d9"):
        return html.Div([
            html.Div(str(value) if value is not None else "—",
                     style={"fontSize": "1.6rem", "fontWeight": "bold", "color": color}),
            html.Div(label, style={"fontSize": "0.75rem", "color": "#8b949e"}),
        ], style={
            "textAlign": "center", "padding": "8px 16px",
            "backgroundColor": DARK_CARD, "borderRadius": "6px",
            "border": f"1px solid {DARK_BORDER}", "minWidth": "100px",
        })

    if mode == "discovery":
        n_divergence = int(v3["has_interfreq_divergence"].sum()) if "has_interfreq_divergence" in v3.columns else 0
        n_above_freeze = int(v3["above_freezing"].sum()) if "above_freezing" in v3.columns else 0
        cards = [
            card("Anomalous Days", total_nb, "#e07b39"),
            card("Regime Length", f"{regime_len}d" if total_nb > 0 else "—", "#e07b39"),
            card("First Anomaly DOY", first_doy, "#f0ad4e"),
            card("Last Anomaly DOY", last_doy, "#f0ad4e"),
            card("IF Divergence Days", n_divergence, "#7b4fbf"),
            card("Above Freezing Days", n_above_freeze, "#2d9a6b"),
        ]
    else:
        n_layered = int(counts.get("ice_layered", 0))
        n_decaying = int(counts.get("ice_decaying", 0))
        cards = [
            card("Ice Days", total_nb, "#4a90d9"),
            card("Season Length", f"{regime_len}d" if total_nb > 0 else "—", "#4a90d9"),
            card("First Freeze DOY", first_doy, "#f0ad4e"),
            card("Last Ice DOY", last_doy, "#d94452"),
            card("Layered Days", n_layered, "#7b4fbf"),
            card("Decaying Days", n_decaying, "#e07b39"),
        ]
    return html.Div(cards, style={
        "display": "flex", "gap": "10px", "flexWrap": "wrap",
        "justifyContent": "center", "marginBottom": "8px",
    })


def build_feature_direction_summary(station, year):
    """Show feature seasonal directions with polarity inversion flags."""
    css = load_cross_station_summary()
    if css is None:
        return html.Div("cross_station_summary.parquet not found",
                         style={"color": "#8b949e", "fontSize": "0.85rem", "padding": "4px"})

    row = css[(css["station"] == station) & (css["year"] == year)]
    if row.empty:
        return html.Div(f"No feature summary for {station} {year}",
                         style={"color": "#8b949e", "fontSize": "0.85rem", "padding": "4px"})
    row = row.iloc[0]

    cells = []
    for feat in FEATURE_LABELS:
        direction = row.get(f"{feat}_direction", "?")
        d_val = row.get(f"{feat}_separation", np.nan)
        majority = MAJORITY_DIRECTION.get(feat)

        if direction == "insufficient_data" or pd.isna(d_val):
            cells.append(html.Td("?", style={"color": "#8b949e", "padding": "4px 8px"}))
            continue

        arrow = "↑" if direction == "winter_high" else "↓"
        inverted = majority is not None and direction != majority
        color = "#e07b39" if inverted else "#2d9a6b"
        label = f"{arrow} {abs(d_val):.1f}"
        if inverted:
            label += " ⚠"

        cells.append(html.Td(label, style={
            "color": color, "fontWeight": "bold" if inverted else "normal",
            "padding": "4px 8px", "fontSize": "0.85rem",
        }))

    header_cells = [html.Th(FEATURE_LABELS[f], style={
        "padding": "4px 8px", "fontSize": "0.75rem", "color": "#8b949e",
        "fontWeight": "normal",
    }) for f in FEATURE_LABELS]

    return html.Div([
        html.Div("Feature Seasonal Direction (↑ winter-high, ↓ summer-high, ⚠ inverted vs network)",
                 style={"fontSize": "0.75rem", "color": "#8b949e", "marginBottom": "4px"}),
        html.Table([
            html.Tr(header_cells),
            html.Tr(cells),
        ], style={"borderCollapse": "collapse"}),
    ], style={
        "backgroundColor": DARK_CARD, "borderRadius": "6px",
        "border": f"1px solid {DARK_BORDER}", "padding": "8px",
    })


def _cohens_d(group_a, group_b):
    """Cohen's d (pooled SD) between two groups."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def build_discrimination_summary(daily_features, v3, station, year):
    """Horizontal bar chart: Cohen's d for each numeric column in daily_features."""
    if daily_features is None or v3 is None:
        return html.Div("Need daily_features and v3 classification.",
                        style={"color": DARK_TEXT, "padding": "16px"})

    pooled = daily_features[daily_features["azimuth_bin"] == -1].copy() if "azimuth_bin" in daily_features.columns else daily_features.copy()
    v3c = v3[["doy", "v3_state"]].drop_duplicates("doy")
    pooled = pooled.merge(v3c, on="doy", how="left")

    mode = _get_mode(v3)
    non_baseline = _non_baseline_states(mode)
    water_label = "baseline" if mode == "discovery" else "open_water"

    water = pooled[pooled["v3_state"] == water_label]
    ice = pooled[pooled["v3_state"].isin(non_baseline)]

    if len(water) < 5 or len(ice) < 5:
        return html.Div("Insufficient classified days for discrimination analysis.",
                        style={"color": DARK_TEXT, "padding": "16px"})

    skip = {"date", "date_dt", "doy", "azimuth_bin", "v3_state"}
    numeric_cols = [c for c in pooled.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(pooled[c])]

    results = []
    for col in numeric_cols:
        iv, wv = ice[col].dropna(), water[col].dropna()
        if len(iv) > 5 and len(wv) > 5:
            d = _cohens_d(iv, wv)
            if np.isfinite(d):
                results.append({"feature": col,
                                "label": FEATURE_LABELS.get(col, col),
                                "d": d, "abs_d": abs(d)})

    if not results:
        return html.Div("No features with sufficient data.",
                        style={"color": DARK_TEXT, "padding": "16px"})

    rdf = pd.DataFrame(results).sort_values("abs_d", ascending=True)

    colors = ["#2d9a6b" if ad >= 0.8 else "#f0ad4e" if ad >= 0.5 else "#8b949e"
              for ad in rdf["abs_d"]]

    sep = "ice" if mode != "discovery" else "anomalous"
    fig = go.Figure(go.Bar(
        y=rdf["label"], x=rdf["d"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>d = %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{station} {year}: Feature Discrimination (Cohen's d, {sep} vs {water_label})",
        xaxis_title=f"Cohen's d (positive = higher during {sep})",
        height=max(400, len(rdf) * 22),
        margin=dict(l=200, r=20, t=50, b=30),
        **PLOTLY_DARK,
    )
    return dcc.Graph(figure=fig)


def build_normalization_comparison(daily_features, v3, station, year, base_feature="CLR"):
    """3-panel comparison of raw, z-scored, and weighted median for one feature."""
    variant_map = {
        "CLR": {"Raw": "clr_med", "Z-scored": "clr_z", "PRN-weighted": "clr_wmed"},
        "AF": {"Raw": "af_med", "Z-scored": "af_z", "PRN-weighted": "af_wmed"},
        "PR": {"Raw": "pr_med", "Z-scored": "pr_z", "PRN-weighted": "pr_wmed"},
        "gamma": {"Raw": "gamma_med", "Z-scored": "gamma_z", "PRN-weighted": "gamma_wmed"},
        "MS": {"Raw": "ms_med", "Z-scored": "ms_z", "PRN-weighted": "ms_wmed"},
        "VS": {"Raw": "vs_med", "Z-scored": "vs_z", "PRN-weighted": "vs_wmed"},
    }

    if daily_features is None or v3 is None:
        return html.Div("Need daily_features and v3 classification.",
                        style={"color": DARK_TEXT, "padding": "16px"})

    variants = variant_map.get(base_feature, {})
    pooled = daily_features[daily_features["azimuth_bin"] == -1].copy() if "azimuth_bin" in daily_features.columns else daily_features.copy()
    variants = {k: v for k, v in variants.items() if v in pooled.columns}

    if not variants:
        return html.Div(f"No variants found for {base_feature}",
                        style={"color": DARK_TEXT, "padding": "16px"})

    v3c = v3[["doy", "v3_state"]].drop_duplicates("doy")
    pooled = pooled.merge(v3c, on="doy", how="left")

    mode = _get_mode(v3)
    order = _state_order(mode)
    non_baseline = _non_baseline_states(mode)
    water_label = "baseline" if mode == "discovery" else "open_water"

    n = len(variants)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=list(variants.keys()),
                        vertical_spacing=0.06)

    for i, (label, col) in enumerate(variants.items(), 1):
        # Compute Cohen's d for annotation
        water = pooled[pooled["v3_state"] == water_label][col].dropna()
        ice = pooled[pooled["v3_state"].isin(non_baseline)][col].dropna()
        d_ann = ""
        if len(water) > 5 and len(ice) > 5:
            d_val = _cohens_d(ice, water)
            if np.isfinite(d_val):
                d_ann = f" (d={d_val:.2f})"

        for state in order:
            mask = pooled["v3_state"] == state
            if mask.sum() == 0:
                continue
            sub = pooled[mask]
            fig.add_trace(go.Scatter(
                x=sub["doy"], y=sub[col], mode="markers",
                marker=dict(size=4, color=V3_COLORS.get(state, "#999"), opacity=0.6),
                name=state, showlegend=(i == 1), legendgroup=state,
            ), row=i, col=1)

        fig.update_yaxes(title_text=f"{label}{d_ann}", row=i, col=1)

    fig.update_xaxes(title_text="Day of Year", row=n, col=1)
    fig.update_layout(
        title=f"{station} {year}: {base_feature} — Normalization Comparison",
        height=250 * n,
        margin=dict(l=80, r=20, t=50, b=30),
        legend=dict(font=dict(size=9), orientation="h", y=-0.05),
        **PLOTLY_DARK,
    )
    return dcc.Graph(figure=fig)


def build_prn_weight_heatmap(station, year):
    """Heatmap of per-PRN |Cohen's d| across features from PRN weights JSON."""
    weights = load_prn_weights(station, year)
    if weights is None:
        return html.Div([
            html.P(f"No PRN weight file for {station} {year}.",
                   style={"color": "#8b949e", "padding": "4px"}),
            html.Code(f"python scripts/compute_prn_weights.py --station {station} --year {year}",
                      style={"color": "#4a90d9", "fontSize": "0.9rem"}),
        ], style={"padding": "16px"})

    features_dict = weights.get("features", {})
    if not features_dict:
        return html.Div("No feature data in weight file.",
                        style={"color": DARK_TEXT, "padding": "16px"})

    # Collect all PRN keys across features
    all_prns = set()
    feature_names = list(features_dict.keys())
    for feat_data in features_dict.values():
        all_prns.update(feat_data.keys())
    prn_list = sorted(all_prns, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))

    # Build z matrix: rows=PRNs, cols=features
    z = []
    for prn in prn_list:
        row = []
        for feat in feature_names:
            entry = features_dict[feat].get(prn, {})
            row.append(entry.get("weight", 0.0))
        z.append(row)

    feat_labels = [FEATURE_LABELS.get(f.lower() + "_med", f) for f in feature_names]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=feat_labels,
        y=prn_list,
        colorscale="YlOrRd",
        colorbar_title="|d|",
        hovertemplate="PRN %{y}<br>Feature: %{x}<br>|d| = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{station} {year}: Per-PRN Discriminating Power (|Cohen's d|)",
        xaxis_title="Feature", yaxis_title="Satellite_Frequency",
        height=max(400, len(prn_list) * 18),
        margin=dict(l=80, r=20, t=50, b=80),
        **PLOTLY_DARK,
    )

    # Summary from the weights file
    summary_dict = weights.get("summary", {})
    summary_items = []
    for feat, s in summary_dict.items():
        summary_items.append(
            f"{feat}: {s.get('n_weighted', 0)}/{s.get('n_combos', 0)} weighted, "
            f"mean |d|={s.get('mean_weight', 0):.3f}, max={s.get('max_weight', 0):.3f}"
        )

    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([html.P(line, style={"margin": "2px 0", "fontSize": "0.85rem"})
                  for line in summary_items],
                 style={"color": DARK_TEXT, "padding": "8px 16px"}),
    ])


# ---------------------------------------------------------------------------
# Investigation tab — static story viewer
# ---------------------------------------------------------------------------

INVESTIGATION_IMG_DIR = PROJECT_ROOT / "docs" / "images" / "investigation"


def _encode_inv_img(fname):
    """Return base64 data URI for a PNG in the investigation image dir, or None."""
    p = INVESTIGATION_IMG_DIR / fname
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _inv_fig(fname, caption):
    """Render one investigation figure with a caption below."""
    src = _encode_inv_img(fname)
    if src is None:
        return html.P(f"[Figure not found: {fname}]",
                      style={"color": "#666", "fontStyle": "italic", "padding": "4px"})
    return html.Div([
        html.Img(src=src, style={"maxWidth": "100%", "borderRadius": "4px",
                                  "border": f"1px solid {DARK_BORDER}",
                                  "display": "block"}),
        html.P(caption, style={"fontSize": "0.78rem", "color": "#8b949e",
                                "marginTop": "4px", "textAlign": "center",
                                "fontStyle": "italic", "lineHeight": "1.4"}),
    ], style={"marginBottom": "14px"})


def _inv_two_col(fig_a, fig_b):
    """Lay two investigation figures side-by-side."""
    return html.Div([
        html.Div(fig_a, style={"flex": "1", "minWidth": "0"}),
        html.Div(fig_b, style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "8px"})


def _inv_story_header(title, status, status_color, summary):
    return html.Div([
        html.Div([
            html.Span(title, style={"color": "#e0e0e0", "fontSize": "1.05rem",
                                     "fontWeight": "bold"}),
            html.Span(f"  [{status}]",
                      style={"color": status_color, "fontSize": "0.82rem",
                             "fontWeight": "bold", "marginLeft": "8px"}),
        ]),
        html.P(summary, style={"color": DARK_TEXT, "fontSize": "0.86rem",
                                "marginTop": "6px", "lineHeight": "1.5",
                                "margin": "6px 0 0 0"}),
    ], style={"padding": "10px 14px", "backgroundColor": DARK_CARD,
               "borderRadius": "4px", "borderLeft": f"3px solid {status_color}",
               "marginBottom": "10px"})


def _inv_findings(findings):
    """findings: list of (label, text, color)."""
    items = [
        html.Div([
            html.Span(f"{lbl}: ", style={"color": col, "fontWeight": "bold"}),
            html.Span(txt, style={"color": DARK_TEXT}),
        ], style={"marginBottom": "3px", "fontSize": "0.84rem"})
        for lbl, txt, col in findings
    ]
    return html.Div(items, style={"backgroundColor": "#0d1117",
                                   "border": f"1px solid {DARK_BORDER}",
                                   "borderRadius": "4px", "padding": "8px 14px",
                                   "marginBottom": "12px"})


def build_investigation_tab():
    """Build the Investigation tab — 5 story sub-tabs, image-based narrative."""
    _ss = {"backgroundColor": "#161b22", "color": "#8b949e",
           "border": "1px solid #30363d", "padding": "6px 14px"}
    _ss_sel = {"backgroundColor": "#21262d", "color": "#e0e0e0",
               "borderTop": "2px solid #f0ad4e", "padding": "6px 14px"}

    # ── Story 1: AF Baseline Fix ─────────────────────────────────────
    s1 = html.Div([
        _inv_story_header(
            "AF Baseline Domain Fix",
            "PRODUCTION BUG FIX",
            "#f0ad4e",
            "The Song 2022 antenna gain pattern correction was a complete no-op on every prior run. "
            "compute_af_baselines.py stored baselines on a sin(e) grid (0.09–0.42). "
            "snr_feature_extractor.py queried at sin(e)/cf (0.92–4.44 for GPS L1) — zero overlap, "
            "every query returned NaN, baseline subtraction was silently skipped. "
            "196 UMNQ baselines and 38 ROSS baselines were all-zero for the lifetime of the pipeline.",
        ),
        _inv_findings([
            ("Root cause", "power_info['sin_elev'] returns x/cf (rescaled), not x (raw sin(e)). "
             "Two-site fix: baseline accumulation and extraction query both corrected.", "#f0ad4e"),
            ("Impact UMNQ", "Median AF drops 77.5%; per-PRN AF variance reduced 27.3%", "#2ca02c"),
            ("Impact ROSS", "AF median drops 80.2%. Only 38 combos (GPS-only) vs 196 at UMNQ.", "#2ca02c"),
            ("Consequence", "All AF thresholds in v1/v2/v3 classifiers are calibrated against uncorrected "
             "antenna gain patterns and need recalibration.", "#d62728"),
        ]),
        _inv_two_col(
            _inv_fig("fig1a_baselines.png",
                     "Fig 1a — Before (all-zero) vs after (real curves): 196 antenna gain baselines "
                     "recovered for UMNQ 2025. GPS / GLONASS / Galileo constellation bands visible."),
            _inv_fig("fig1b_af_shift.png",
                     "Fig 1b — AF distribution before (red) vs after (blue) baseline subtraction. "
                     "Median drops 77.5%. Per-constellation violins show the shift is consistent."),
        ),
        _inv_fig("fig1c_prn_variance.png",
                 "Fig 1c — Per-PRN AF standard deviation: before vs after. Most points fall below the "
                 "diagonal (lower variance post-fix). Mean σ reduction: 27.3%. The gain pattern was "
                 "adding satellite-specific bias on top of surface-state variance."),
    ], style={"padding": "10px 0"})

    # ── Story 2: Gamma Hardening ─────────────────────────────────────
    s2 = html.Div([
        _inv_story_header(
            "Per-Arc γ Extraction: Three Reliability Improvements",
            "PRODUCTION IMPROVEMENT",
            "#2ca02c",
            "γ and gamma_r2 — what they are and why per-arc extraction needs explicit hardening.",
        ),
        html.Div([
            html.P(
                "γ (gamma) is the damping coefficient from Strandberg 2017: the rate at which the SNR "
                "interference fringe amplitude decays with increasing elevation angle, modeled as "
                "log(envelope) = intercept − γ · sin²(ε). A smooth specular surface (calm ice) produces "
                "slow decay — low γ. A rough diffuse surface (open water) produces faster decay — high γ.",
                style={"color": DARK_TEXT, "fontSize": "0.87rem", "lineHeight": "1.6", "marginBottom": "8px"}),
            html.P(
                "Strandberg extracted γ by fitting this model jointly to all arcs collected over a "
                "72-hour window using nonlinear least squares (NLLS). With dozens of satellite passes "
                "pooled together, random noise sources — Hilbert edge artifacts, destructive interference "
                "nulls, truncated arcs — appear at different positions in each arc and cancel out in the "
                "aggregate. Our pipeline extracts γ per-arc from individual 20–40 minute satellite passes, "
                "so every noise source that would average away over 72 hours is fully present and can "
                "dominate a single fit. Three explicit software fixes compensate:",
                style={"color": DARK_TEXT, "fontSize": "0.87rem", "lineHeight": "1.6", "marginBottom": "8px"}),
        ], style={"padding": "4px 14px 0 14px"}),
        _inv_findings([
            ("1 · Tukey tapering", "Problem: Hilbert transform assumes a periodic signal. A real arc ends "
             "mid-oscillation — the boundary discontinuity leaks energy into the computed envelope, "
             "corrupting the edge samples that γ is fit to. In 72-hour averaging this averages away; "
             "per-arc it does not. "
             "Fix: Tukey window (α=0.2, 10% cosine taper each edge) before the transform; tapered samples "
             "trimmed from the fit region. Verified on synthetic signal: 1.5% estimation error. "
             "Not visualized — no pre-Tukey arc_table snapshot was saved.", "#8b949e"),
            ("2 · Siegelslopes", "Problem: at destructive interference nulls the envelope approaches zero. "
             "log(near-zero) → large negative values that dominate an OLS fit (high leverage). "
             "In 72-hour averaging, null positions vary across arcs and wash out; per-arc they do not. "
             "Fix: Siegel repeated-medians regression (~50% breakdown point). "
             "Verified qualitatively: eliminates extreme γ outliers. Not visualized — no pre-implementation snapshot.", "#8b949e"),
            ("3 · Full-arc guard (Fig 2a)", "Problem: UMNQ's gnssrefl config uses ediff=200 (effectively "
             "disabled), passing arcs that span <80% of [e1, e2]. Shorter elevation range = less lever arm "
             "for the sin²(ε) fit → biased γ; shorter integration domain → biased AF. "
             "Song 2022 avoided this with strict ediff; our config did not. "
             "Fix: exclude arcs with ele_range < 0.8×expected_range from γ and AF. "
             "UMNQ: 14.7% excluded (partial arcs had 23% lower PkNoise). ROSS: 0% (ediff=2.0 catches them).", "#f0ad4e"),
            ("gamma_r2 (Fig 2b)", "R² of the log-envelope fit — how well the Strandberg smooth-surface "
             "model describes a given arc. Where all three hardening improvements still cannot tame a noisy "
             "arc, gamma_r2 quantifies how much to trust the resulting γ. "
             "UMNQ median = 0.123 (model partially applies). ROSS median = 0.000 — rough Great Lakes ice "
             "breaks the smooth-surface assumption regardless of extraction method. "
             "Reframed as surface coherence: high R² = the surface itself is specular enough for the model "
             "to apply, not simply 'good data'.", "#58a6ff"),
        ]),
        _inv_two_col(
            _inv_fig("fig2a_arc_guard.png",
                     "Fig 2a — Full (blue) vs partial (orange) arc distributions for gamma, PkNoise, and "
                     "elevation coverage at UMNQ 2025. Partial arcs cluster below the 80% threshold (dashed). "
                     "Their lower PkNoise confirms they add noise, not signal, to daily medians."),
            _inv_fig("fig2b_gamma_r2.png",
                     "Fig 2b — (Left) gamma_r2 distributions: UMNQ (median 0.123) vs ROSS (median 0.000). "
                     "ROSS near-zero confirms the Strandberg smooth-surface model fails for rough Great Lakes ice. "
                     "(Center) gamma vs gamma_r2 colored by DOY at UMNQ — R² < 0.1 marks unreliable gamma. "
                     "(Right) gamma distributions by R² band — higher R² → sharper, more informative fit."),
        ),
    ], style={"padding": "10px 0"})

    # ── Story 3: Ruled-Out Features ───────────────────────────────────
    s3 = html.Div([
        _inv_story_header(
            "Feature Candidates: Tested and Ruled Out",
            "DEAD / PARKED",
            "#d62728",
            "Three candidates targeting different physical signals — snow layering, surface complexity, "
            "and spatial heterogeneity. B3 and B2 are ruled out on noise grounds. B4 is parked: "
            "the geometry is sufficient but no classification method has been built to use it.",
        ),
        html.Div([
            html.P(
                "B3 (envelope ratio) attempts to detect a snow layer on top of ice using "
                "frequency-dependent Fresnel reflections. A two-layer air→snow→ice system produces "
                "different L1 vs L2C Hilbert envelope shapes as a function of elevation angle — "
                "the ratio of those envelopes carries a slope that scales with snow depth and permittivity. "
                "Fig 3a shows the theoretical signal and the real data side by side: the noise-to-signal "
                "ratio makes the approach unworkable with current Hilbert envelope quality.",
                style={"color": DARK_TEXT, "fontSize": "0.87rem", "lineHeight": "1.6", "marginBottom": "8px"}),
            html.P(
                "B2 (cross-frequency correlation) attempts to use waveform similarity between frequency "
                "bands as a surface complexity proxy. For the same physical arc, L1 and L2C observe "
                "identical geometry and should oscillate at the same dominant frequency. "
                "Pearson r measures the linear correlation between the two rescaled waveforms — "
                "r=1 means the bands track each other perfectly, r=0 means no relationship. "
                "A single specular reflector (smooth ice) should push r toward 1; a rough or layered "
                "surface should decorrelate them toward 0. "
                "Fig 3b shows the median ice/water separation is real (0.02–0.09) but the per-arc "
                "distribution width (~0.5 IQR) buries it completely.",
                style={"color": DARK_TEXT, "fontSize": "0.87rem", "lineHeight": "1.6", "marginBottom": "8px"}),
            html.P(
                "B4 (geometry census) is different in character — it was not ruled out on signal grounds, "
                "it was parked for lack of a defined use. The census confirmed that at UMNQ, 2300–3200 "
                "simultaneous satellite pairs exist per day at matching elevations but different azimuths. "
                "If a classification method were built around them, they would enable spatial surface maps "
                "at sub-daily resolution: which azimuth sectors see ice vs open water at the same time, "
                "rather than one label per station per day. During spring breakup (DOY 85–117), where "
                "sector consistency analysis showed inter-sector AF disagreement up to 9.6, B4 pairs could "
                "distinguish a mixed fjord — some sectors ice, others open water — from a uniformly "
                "transitioning one. Fig 3c shows the coverage: the infrastructure is there, "
                "the analysis method is not.",
                style={"color": DARK_TEXT, "fontSize": "0.87rem", "lineHeight": "1.6", "marginBottom": "4px"}),
        ], style={"padding": "4px 14px 0 14px"}),
        _inv_findings([
            ("B3 verdict", "Theoretical 5cm snow signal (slope=−0.07) is ~400× below real Hilbert "
             "envelope noise floor (std=28.83). Dead.", "#d62728"),
            ("B2 verdict", "Median ice/water Pearson r shift 0.02–0.09; per-arc IQR ~0.5. Signal "
             "completely buried. GAL L1-E5b persistently negative in both states (receiver artifact). Dead.", "#d62728"),
            ("B4 verdict", "2300–3200 pairs/day at UMNQ (>200/hour), consistent across all seasons. "
             "Parked — needs a spatial classification method before it becomes useful.", "#f0ad4e"),
            ("Noise floor note", "Any Hilbert-envelope feature must produce signals >10× σ=29 to be "
             "usable. Rules out all subtle multi-layer approaches.", "#8b949e"),
        ]),
        _inv_fig("fig3a_envelope_ratio.png",
                 "Fig 3a — (Left) Theoretical L1/L2C envelope ratio slope vs snow depth from a two-layer "
                 "Fresnel model: 5cm snow produces slope=−0.07. (Right) Real data distribution by surface "
                 "state: std=28.8. The theoretical signal is invisible at this scale — green dashes mark "
                 "the 5, 10, and 20cm theoretical values. The noise-to-signal ratio is ~400:1."),
        _inv_two_col(
            _inv_fig("fig3b_xfreq_corr.png",
                     "Fig 3b — Pearson r between frequency-pair waveforms by surface state (boxplots, "
                     "outliers hidden). Blue=baseline, red=anomalous, orange=regime change. "
                     "Δmedian labels show the ice/water separation per pair: real but ~0.02–0.09. "
                     "IQR of ~0.5 in both states means individual arcs are not separable."),
            _inv_fig("fig3c_geometry_census.png",
                     "Fig 3c — Simultaneous satellite pairs (|Δelev| < 1°, |Δaz| > 30°) across four "
                     "seasonal sample days at UMNQ 2025. Color = azimuth separation. Consistent "
                     "200+ pairs/hour across all seasons confirms adequate geometry for spatial "
                     "heterogeneity mapping — the missing piece is the classification method."),
        ),
    ], style={"padding": "10px 0"})

    # ── Story 4: Post-Fix Assessment ──────────────────────────────────
    s4 = html.Div([
        _inv_story_header(
            "Post-Fix Feature Reassessment",
            "DECISIONS MADE",
            "#58a6ff",
            "After correcting the AF baseline bug and re-extracting all features, a full discriminability "
            "audit across ROSS 2024 and UMNQ 2025 drove several concrete decisions: "
            "CLR is blind at UMNQ due to polar geometry; af_z adds nothing over af_med post-correction; "
            "spring breakup is spatially patchy while autumn freeze-up is spatially uniform.",
        ),
        _inv_findings([
            ("CLR at UMNQ", "d = 0.057 — effectively blind. All GPS+GLONASS+Galileo satellites track "
             "through a narrow elevation band; LSP always has one dominant peak. Drop from v1/Mahal.", "#d62728"),
            ("CLR at ROSS", "d = 1.30 — strongly discriminating at mid-latitude (Great Lakes). "
             "Same feature, completely different behavior due to geometry.", "#2ca02c"),
            ("af_z vs af_med", "r = 0.99 at both stations. af_med strictly outperforms af_z. "
             "Post-correction, z-scoring adds nothing. Drop af_z from Mahalanobis feature set.", "#f0ad4e"),
            ("Spring breakup", "DOY 85–117: inter-sector AF score σ up to 9.6 — real spatial heterogeneity. "
             "Multiple sectors showing ice while others show open water simultaneously.", "#f0ad4e"),
            ("Autumn freeze-up", "DOY 300+: sectors agree (σ ≈ 1.3 = regime_change noise floor). "
             "New thin ice forms uniformly across the fjord. Classifier signal is real and coherent.", "#58a6ff"),
        ]),
        _inv_fig("fig4a_cohens_d.png",
                 "Fig 4a — |Cohen's d| for 9 features: ROSS 2024 (purple) vs UMNQ 2025 (green). "
                 "CLR highlighted in red at UMNQ (d=0.06). AF (corrected) is strongest at both stations. "
                 "Dashed lines mark small (0.2) and large (0.8) effect thresholds."),
        _inv_two_col(
            _inv_fig("fig4b_clr_blind.png",
                     "Fig 4b — CLR distributions for ice vs water. ROSS (d=1.30): well-separated. "
                     "UMNQ (d=0.06): completely overlapping. At 70.7°N, polar geometry makes CLR useless — "
                     "every arc looks spectrally clear regardless of surface state."),
            _inv_fig("fig4c_sector_consistency.png",
                     "Fig 4c — Inter-sector AF score disagreement over the UMNQ 2025 year. "
                     "Spring (DOY 85–117): high disagreement = patchy ice, real spatial heterogeneity. "
                     "Autumn (DOY 300+): sectors agree = uniform new ice formation."),
        ),
    ], style={"padding": "10px 0"})

    # ── Story 5: GLBX Hardware Artifact ──────────────────────────────
    s5 = html.Div([
        _inv_story_header(
            "GLBX Hardware Artifact: L1 Suppression",
            "STATION-SPECIFIC",
            "#9467bd",
            "At GLBX (Bartlett Cove, AK), L1 input SNR drops ~8 dBHz in winter. This collapses "
            "L1 amplitude 67% (from 29 to 9 units), dragging the frequency-pooled amp_mean in the "
            "wrong direction during ice periods. L5 amplitude is correctly higher in winter (46 vs 41). "
            "UMNQ 2025 and NIAQ 2025 are both clean. Additionally, L1 amplitude at UMNQ detects "
            "autumn freeze-onset about one month earlier than the pooled amplitude.",
        ),
        _inv_findings([
            ("GLBX verdict", "CONTAMINATED. L1 input SNR drops 8 dBHz; amplitude collapses 67%. "
             "Pooled amp_mean moves DOWN during ice (wrong direction). Fix: use amp_L5_mean.", "#d62728"),
            ("UMNQ verdict", "CLEAN. Both L1 (+5.3 units) and L5 (+4.1 units) rise in winter. "
             "L1 marginally better discriminator (d=1.18 vs d=1.05 for L5).", "#2ca02c"),
            ("NIAQ verdict", "CLEAN. Minimal seasonal variation (<1 unit), MS flat (<0.5 dBHz). "
             "Pool correlations r > 0.99 in winter for all bands.", "#2ca02c"),
            ("UMNQ autumn", "DOY 300–366: amp_L1 d=0.72, pooled d=0.25, amp_L5 d=−0.003. "
             "L5 actively dilutes L1's freeze-onset signal in the pooled mean.", "#f0ad4e"),
            ("DOY boundary", "DOY 285 (Oct 12): switch to amp_L1_mean for UMNQ autumn classification. "
             "At this point d_L1 ≈ 0 while d_pooled = −1.11 (actively wrong direction).", "#58a6ff"),
            ("Action GLBX", "Replace amp_mean with amp_L5_mean in GLBX mahal_features config "
             "and switch polarity from winter_low to winter_high.", "#58a6ff"),
        ]),
        _inv_two_col(
            _inv_fig("fig5a_glbx_amplitude.png",
                     "Fig 5a — GLBX monthly amplitude by band. L1 (blue) collapses Jan–Mar; "
                     "L5 (green) is flat/higher in winter. Input MS (right) drops 8 dBHz — "
                     "hardware-level cause, not surface physics."),
            _inv_fig("fig5b_three_station.png",
                     "Fig 5b — Three-station comparison: GLBX (CONTAMINATED, left), UMNQ (CLEAN, center), "
                     "NIAQ (CLEAN, right). L1 suppression mechanism is GLBX-specific — absent at "
                     "both Greenland stations."),
        ),
        _inv_fig("fig5c_doy_crossover.png",
                 "Fig 5c — Rolling 30-day Cohen's d at UMNQ 2025. amp_L1 (blue) crosses from wrong- to "
                 "correct-ice direction at ~DOY 285, a full month before pooled amp_mean (gray, ~DOY 334). "
                 "amp_L5 (green) remains strongly negative (dilutes pool). Shaded = L1 advantage window."),
    ], style={"padding": "10px 0"})

    return html.Div([
        html.Div([
            html.H4("Feature Investigation Log",
                    style={"color": "#e0e0e0", "margin": "0 0 3px 0", "fontSize": "1.0rem"}),
            html.P("Systematic pipeline debugging and feature evaluation — UMNQ 2025 and ROSS 2024. "
                   "Generated figures from scripts/plot_investigation_figures.py.",
                   style={"color": "#8b949e", "fontSize": "0.82rem", "margin": "0 0 10px 0"}),
        ]),
        dcc.Tabs(id="investigation-subtabs", value="story1", children=[
            dcc.Tab(label="1 · AF Baseline Fix", value="story1",
                    style=_ss, selected_style=_ss_sel, children=[s1]),
            dcc.Tab(label="2 · Per-Arc γ Reliability", value="story2",
                    style=_ss, selected_style=_ss_sel, children=[s2]),
            dcc.Tab(label="3 · Ruled-Out Features", value="story3",
                    style=_ss, selected_style=_ss_sel, children=[s3]),
            dcc.Tab(label="4 · Post-Fix Assessment", value="story4",
                    style=_ss, selected_style=_ss_sel, children=[s4]),
            dcc.Tab(label="5 · GLBX Hardware Artifact", value="story5",
                    style=_ss, selected_style=_ss_sel, children=[s5]),
        ]),
    ])


def build_mahalanobis_figure(v2, v3, threshold, station, year):
    """Mahalanobis distance time series with threshold line and v3 state coloring."""
    if v2 is None or "mahal_dist" not in v2.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Mahalanobis distance: no v2 data",
            height=200, **PLOTLY_DARK,
        )
        return fig

    df = v2.sort_values("doy").copy()

    # Merge v3 states for coloring
    mode = _get_mode(v3)
    if v3 is not None and "v3_state" in v3.columns:
        v3c = v3[["doy", "v3_state"]].drop_duplicates("doy")
        df = df.merge(v3c, on="doy", how="left")
    else:
        df["v3_state"] = np.where(df["above_threshold"], "ice_surface", "open_water")

    fig = go.Figure()

    # Plot points colored by v3 state
    for state in _state_order(mode):
        mask = df["v3_state"] == state
        if mask.sum() == 0:
            continue
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["doy"], y=sub["mahal_dist"], mode="markers",
            marker=dict(size=5, color=V3_COLORS.get(state, "#999"), opacity=0.7),
            name=state, legendgroup=state, showlegend=False,
        ))

    # Threshold line
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#d94452",
                      annotation_text=f"threshold = {threshold:.1f}",
                      annotation_font_color="#d94452",
                      annotation_font_size=10)

    fig.update_layout(
        title=f"Mahalanobis Distance from Water-State Baseline",
        xaxis_title="Day of Year",
        yaxis_title="Mahalanobis Distance",
        height=220,
        margin=dict(l=60, r=20, t=35, b=30),
        **PLOTLY_DARK,
    )
    return fig


def render_s1_imagery(station, date_str, s1_images, v3):
    """Render S1 VV/VH as inline base64 images for the imagery tab."""
    if not date_str or not s1_images:
        return html.P("No imagery available", style={"color": DARK_TEXT})

    # Find the GeoTIFF for this date
    tif_path = None
    for img in s1_images:
        if img["date"] == date_str:
            tif_path = img["path"]
            break

    if tif_path is None or not Path(tif_path).exists():
        return html.P(f"No S1 file for {date_str}", style={"color": DARK_TEXT})

    # Get v3 state for this date
    state_label = "?"
    if v3 is not None:
        v3_match = v3[v3["date_dt"].dt.strftime("%Y-%m-%d") == date_str]
        if len(v3_match) > 0:
            state_label = v3_match.iloc[0].get("v3_state", "?")

    try:
        import rioxarray
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc
        from pyproj import Transformer
        import json as _json

        with open(PROJECT_ROOT / "config" / "stations_config.json") as f:
            cfg = _json.load(f)

        lat = cfg[station]["latitude_deg"]
        lon = cfg[station]["longitude_deg"]

        # Get azimuth range from gnssir params
        params_path = cfg[station].get("gnssir_json_params_path", "")
        az_start, az_end = 0, 360
        try:
            with open(PROJECT_ROOT / params_path) as f:
                params = _json.load(f)
            azval = params.get("azval2", [0, 360])
            if len(azval) >= 2:
                az_start, az_end = azval[0], azval[1]
        except:
            pass

        da = rioxarray.open_rasterio(tif_path)
        t = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
        x_s, y_s = t.transform(lon, lat)

        # Full 5km crop
        vv_full = da.sel(band=1).values.astype(float)
        vh_full = da.sel(band=2).values.astype(float)
        bounds_full = da.rio.bounds()
        extent_full = [bounds_full[0], bounds_full[2], bounds_full[1], bounds_full[3]]

        # Close-up 500m crop
        pad_close = 300
        clip = da.rio.clip_box(x_s - pad_close, y_s - pad_close, x_s + pad_close, y_s + pad_close)
        vv_close = clip.sel(band=1).values.astype(float)
        vh_close = clip.sel(band=2).values.astype(float)
        bounds_close = clip.rio.bounds()
        extent_close = [bounds_close[0], bounds_close[2], bounds_close[1], bounds_close[3]]
        clip.close()
        da.close()

        # 2x2 grid: top row = VV (full + close), bottom = VH (full + close)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor=DARK_BG,
                                  gridspec_kw={"width_ratios": [1.2, 1]})

        RH = 5.0
        panels = [
            (axes[0, 0], vv_full, extent_full, bounds_full, "VV — 5 km context", -25, -5, 1000, False),
            (axes[0, 1], vv_close, extent_close, bounds_close, "VV — Fresnel zone", -25, -5, 100, True),
            (axes[1, 0], vh_full, extent_full, bounds_full, "VH — 5 km context", -30, -12, 1000, False),
            (axes[1, 1], vh_close, extent_close, bounds_close, "VH — Fresnel zone", -30, -12, 100, True),
        ]

        for ax, data, ext, bnds, label, vmin, vmax, bar_m, show_fresnel in panels:
            ax.set_facecolor(DARK_CARD)
            im = ax.imshow(data, extent=ext, origin="upper", cmap="gray",
                           vmin=vmin, vmax=vmax)
            ax.plot(x_s, y_s, "r*", markersize=14 if not show_fresnel else 20,
                    markeredgecolor="white", markeredgewidth=1.5)

            if show_fresnel:
                # Detailed Fresnel cone on close-up
                for elev, color, ls, lw in [(5, "cyan", "-", 2), (10, "lime", "--", 1.5), (15, "yellow", ":", 1.2)]:
                    d = RH / np.tan(np.radians(elev))
                    t1 = 90 - az_end
                    t2 = 90 - az_start
                    arc = Arc((x_s, y_s), 2*d, 2*d, angle=0, theta1=t1, theta2=t2,
                              edgecolor=color, lw=lw, ls=ls)
                    ax.add_patch(arc)
                for az in [az_start, az_end]:
                    d_max = RH / np.tan(np.radians(5))
                    ax.plot([x_s, x_s + d_max * np.sin(np.radians(az))],
                            [y_s, y_s + d_max * np.cos(np.radians(az))],
                            "w-", lw=1, alpha=0.7)
            else:
                # On full view, draw a box showing the close-up extent
                from matplotlib.patches import Rectangle
                rect = Rectangle((bounds_close[0], bounds_close[1]),
                                  bounds_close[2] - bounds_close[0],
                                  bounds_close[3] - bounds_close[1],
                                  linewidth=1.5, edgecolor="cyan", facecolor="none", ls="--")
                ax.add_patch(rect)

            ax.set_title(label, color=DARK_TEXT, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

            # Scale bar
            bar_x = bnds[0] + 30
            bar_y = bnds[1] + 30
            ax.plot([bar_x, bar_x + bar_m], [bar_y, bar_y], "w-", lw=3)
            bar_label = f"{bar_m} m" if bar_m < 1000 else f"{bar_m // 1000} km"
            ax.text(bar_x + bar_m/2, bar_y + bar_m * 0.06, bar_label, color="white",
                    ha="center", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5))

        # Colorbar for the right column
        fig.subplots_adjust(right=0.93)
        cax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cax, label="Backscatter (dB)")

        state_color = V3_COLORS.get(state_label, "#999")
        fig.suptitle(f"{station}  {date_str}  —  {state_label}",
                     color=state_color, fontsize=14, fontweight="bold")
        fig.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return html.Div([
            html.Img(src=f"data:image/png;base64,{img_b64}",
                     style={"width": "100%", "borderRadius": "6px"}),
        ])

    except Exception as e:
        return html.P(f"Error rendering S1: {e}", style={"color": "#ff6b6b"})


def build_quality_stats(per_arc):
    if per_arc is None or per_arc.empty:
        return {}
    df = per_arc
    freq_counts = df["freq_group"].value_counts() if "freq_group" in df.columns else pd.Series()
    total = len(df)
    n_days = df["date_dt"].nunique() if "date_dt" in df.columns else df["doy"].nunique() if "doy" in df.columns else 0
    return {
        "total_arcs": total, "n_days": n_days,
        "arcs_per_day": total / max(n_days, 1),
        "freq_pcts": {k: v / total * 100 for k, v in freq_counts.items()},
        "rh_median": df["RH"].median(),
        "rh_std": df["RH"].std(),
    }


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def create_app():
    stations = list_available_stations()
    station_names = sorted(stations.keys())

    app = Dash(__name__, title="GNSS-IR Dashboard", suppress_callback_exceptions=True)

    # Put ROSS first, then alphabetical
    default_station = "ROSS" if "ROSS" in station_names else (station_names[0] if station_names else None)
    ordered_names = ([default_station] + [s for s in station_names if s != default_station]) if default_station else station_names

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("GNSS-IR Dashboard", style={"margin": "0", "fontSize": "1.4rem", "color": "#e0e0e0"}),
            html.Div([
                html.Label("Station:", style={"marginRight": "6px", "fontWeight": "bold", "color": "#ccc"}),
                dcc.Dropdown(id="station-select",
                    options=[{"label": s, "value": s} for s in ordered_names],
                    value=default_station,
                    style={"width": "140px", "color": "#000"}, clearable=False),
                html.Label("Year:", style={"marginLeft": "16px", "marginRight": "6px", "fontWeight": "bold", "color": "#ccc"}),
                dcc.Dropdown(id="year-select", style={"width": "100px", "color": "#000"}, clearable=False),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "10px 20px", "background": "#0d1117", "color": "#e0e0e0",
            "borderBottom": "2px solid #4a90d9",
        }),

        # Tabs
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Time Series", value="timeseries",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Polar", value="polar",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Ice Classification", value="ice",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Imagery", value="imagery",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Features", value="features",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}),
            dcc.Tab(label="Investigation", value="investigation",
                    style={"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"},
                    selected_style={"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #f0ad4e"}),
        ], style={"marginBottom": "0"}),

        # Tab content with loading spinner
        dcc.Loading(
            id="loading",
            type="circle",
            color="#4a90d9",
            children=html.Div(id="tab-content", style={"padding": "8px", "minHeight": "500px"}),
        ),

        # Quality bar
        html.Div(id="quality-bar", style={
            "padding": "10px 20px", "background": "#161b22",
            "borderTop": "1px solid #30363d", "fontSize": "0.9rem",
            "display": "flex", "gap": "20px", "flexWrap": "wrap", "color": "#c9d1d9",
        }),
    ], style={
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "backgroundColor": "#0d1117", "color": "#e0e0e0",
        "minHeight": "100vh", "width": "100%",
    })

    # --- Callbacks ---

    @callback(
        Output("year-select", "options"),
        Output("year-select", "value"),
        Input("station-select", "value"),
    )
    def update_year_options(station):
        if not station or station not in stations:
            return [], None
        years = stations[station]["years"]
        options = [{"label": str(y), "value": y} for y in sorted(years, reverse=True)]
        # Default to 2024 if available, otherwise latest
        default = 2024 if 2024 in years else years[-1]
        return options, default

    # Overview time series brush → update polar diagram
    @callback(
        Output("overview-polar", "figure"),
        Output("overview-selection-info", "children"),
        Input("overview-ts", "relayoutData"),
        State("overview-per-arc-store", "data"),
        State("station-select", "value"),
        State("year-select", "value"),
        prevent_initial_call=True,
    )
    def overview_brush(relayout_data, per_arc_records, station, year):
        if not relayout_data or not per_arc_records:
            return no_update, no_update

        # Check if there's a zoom/selection range
        x_start = relayout_data.get("xaxis.range[0]") or relayout_data.get("xaxis.range")
        x_end = relayout_data.get("xaxis.range[1]")

        if x_start is None:
            # Check for autorange reset
            if "xaxis.autorange" in relayout_data:
                # Reset — show all arcs
                df = pd.DataFrame(per_arc_records)
                if "date_dt" in df.columns:
                    df["date_dt"] = pd.to_datetime(df["date_dt"])
                fig = build_polar_figure(df, title="All Arcs")
                fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30))
                info = html.P(f"Showing all arcs", style={"color": DARK_TEXT})
                return fig, info
            return no_update, no_update

        if isinstance(x_start, list):
            x_start = x_start[0]
            x_end = x_start[-1] if len(x_start) > 1 else x_start

        # Filter per_arc to the selected date range
        df = pd.DataFrame(per_arc_records)
        if "date_dt" in df.columns:
            df["date_dt"] = pd.to_datetime(df["date_dt"])
            try:
                start_dt = pd.Timestamp(x_start)
                end_dt = pd.Timestamp(x_end)
                mask = (df["date_dt"] >= start_dt) & (df["date_dt"] <= end_dt)
                filtered = df[mask]
            except:
                return no_update, no_update
        else:
            return no_update, no_update

        if filtered.empty:
            return no_update, no_update

        n_days = filtered["date_dt"].dt.date.nunique()
        n_arcs = len(filtered)
        date_range = f"{filtered['date_dt'].min().strftime('%b %d')} – {filtered['date_dt'].max().strftime('%b %d')}"

        fig = build_polar_figure(filtered, title=f"Arcs: {date_range}")
        fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30))

        info = html.Div([
            html.P(f"{date_range}", style={"color": "#4a90d9", "fontWeight": "bold", "margin": "0"}),
            html.P(f"{n_arcs:,} arcs across {n_days} days", style={"color": DARK_TEXT, "margin": "2px 0"}),
            html.P("Zoom out or double-click to reset", style={"color": "#8b949e", "fontSize": "0.75rem"}),
        ])

        return fig, info

    # Map marker click → update station dropdown
    @callback(
        Output("station-select", "value", allow_duplicate=True),
        Input({"type": "station-marker", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def map_click(n_clicks_list):
        from dash import ctx
        if not ctx.triggered_id:
            return no_update
        # Guard: only respond to real clicks, not marker re-creation.
        # When the overview tab re-renders, new markers are created with
        # n_clicks=None, which can fire this callback spuriously.
        if not any(n and n > 0 for n in n_clicks_list):
            return no_update
        # ctx.triggered_id is {"type": "station-marker", "index": "ROSS"}
        clicked_station = ctx.triggered_id.get("index")
        if clicked_station:
            return clicked_station
        return no_update

    # Imagery grid click → show full S1 image
    @callback(
        Output("imagery-display", "children"),
        Input({"type": "imagery-thumb", "index": ALL}, "n_clicks"),
        State("station-select", "value"),
        State("year-select", "value"),
        prevent_initial_call=True,
    )
    def imagery_grid_click(n_clicks_list, station, year):
        from dash import ctx
        if not ctx.triggered_id or not station:
            return no_update
        clicked_date = ctx.triggered_id.get("index")
        if not clicked_date:
            return no_update
        s1_images = list_s1_images(station)
        v3 = load_v3(station, year)
        return render_s1_imagery(station, clicked_date, s1_images, v3)

    # Time series click → show nearest S1 imagery (±3 days)
    @callback(
        Output("ts-imagery-panel", "children"),
        Input("ts-graph", "clickData"),
        State("station-select", "value"),
        State("year-select", "value"),
        prevent_initial_call=True,
    )
    def ts_click_imagery(click_data, station, year):
        if not click_data or not station:
            return no_update

        point = click_data.get("points", [{}])[0]

        # Get the clicked date — either from customdata (S1 marker) or x axis
        clicked_date = None
        custom = point.get("customdata")
        if isinstance(custom, list):
            custom = custom[0] if custom else None
        if custom and isinstance(custom, str) and len(custom) == 10 and custom[4] == "-":
            clicked_date = custom
        else:
            # Try to get date from the x axis value
            x_val = point.get("x")
            if x_val and isinstance(x_val, str) and len(x_val) >= 10:
                clicked_date = x_val[:10]

        if not clicked_date:
            return no_update

        # Find nearest S1 image within ±3 days
        s1_images = list_s1_images(station)
        if not s1_images:
            return html.P(f"No S1 imagery for {station}", style={"color": "#8b949e", "padding": "8px"})

        from datetime import datetime, timedelta
        try:
            clicked_dt = datetime.strptime(clicked_date, "%Y-%m-%d")
        except ValueError:
            return no_update

        best_match = None
        best_offset = 999
        for img in s1_images:
            try:
                img_dt = datetime.strptime(img["date"], "%Y-%m-%d")
                offset = abs((img_dt - clicked_dt).days)
                if offset <= 3 and offset < best_offset:
                    best_match = img["date"]
                    best_offset = offset
            except ValueError:
                continue

        if best_match is None:
            return html.P(f"No S1 image within ±3 days of {clicked_date}",
                          style={"color": "#8b949e", "padding": "8px"})

        v3 = load_v3(station, year)
        offset_text = f" (S1 from {best_match}, {best_offset}d offset)" if best_offset > 0 else ""
        imagery = render_s1_imagery(station, best_match, s1_images, v3)
        if best_offset > 0:
            return html.Div([
                html.P(f"Clicked {clicked_date} → nearest S1: {best_match} ({best_offset} day offset)",
                       style={"color": "#4a90d9", "fontSize": "0.85rem", "marginBottom": "4px"}),
                imagery,
            ])
        return imagery

    @callback(
        Output("tab-content", "children"),
        Output("quality-bar", "children"),
        Input("tabs", "value"),
        Input("station-select", "value"),
        Input("year-select", "value"),
    )
    def update_tab(tab, station, year):
        if not station or not year:
            return html.P("Select a station and year"), []

        per_arc = load_per_arc(station, year)
        stats = build_quality_stats(per_arc)

        # Quality bar (shared across tabs)
        quality_items = []
        if stats:
            quality_items.append(html.Span(
                f"{stats['total_arcs']:,} arcs | {stats['n_days']} days | "
                f"{stats['arcs_per_day']:.0f} arcs/day",
                style={"fontWeight": "bold"}))
            quality_items.append(html.Span(f"RH: {stats['rh_median']:.2f} +/- {stats['rh_std']:.2f} m"))
            for freq, pct in sorted(stats.get("freq_pcts", {}).items()):
                color = FREQ_COLORS.get(freq, "#888")
                quality_items.append(html.Span([
                    html.Span("| ", style={"color": color, "fontSize": "1.2em"}),
                    f"{freq}: {pct:.0f}%"]))

        if tab == "overview":
            v3 = load_v3(station, year)



            center = [stations[station]["lat"], stations[station]["lon"]] if station in stations else [45, -80]
            # High-latitude stations need lower zoom (fewer satellite tile levels)
            station_lat = stations[station]["lat"] if station in stations else 45
            map_zoom = 14 if abs(station_lat) > 60 else 16

            # v3 summary
            v3_lines = []
            if v3 is not None and "v3_state" in v3.columns:
                counts = v3["v3_state"].value_counts()
                total = len(v3)
                for s in V3_ORDER:
                    c = counts.get(s, 0)
                    if c > 0:
                        color = V3_COLORS.get(s, "#999")
                        v3_lines.append(html.Div([
                            html.Span("■ ", style={"color": color, "fontSize": "1.3em"}),
                            f"{s}: {c} days ({c/total*100:.0f}%)",
                        ], style={"marginBottom": "4px"}))

            # Polar diagram (compact, top right)
            polar_fig = build_polar_figure(per_arc, title="Reflection Points")
            polar_fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30))

            # Time series (below map) — include S1 dates
            s1_images = list_s1_images(station)
            year_s1 = [img for img in s1_images if img["date"].startswith(str(year))]
            ts_fig = build_timeseries_figure(per_arc, v3, station, year, s1_dates=year_s1)

            # Build Fresnel zone overlay for the map
            fresnel_layers = []
            station_cfg = load_stations_config().get(station, {})
            params_path = station_cfg.get("gnssir_json_params_path", "")
            az_start, az_end = 0, 360
            try:
                with open(PROJECT_ROOT / params_path) as f:
                    params = json.load(f)
                azval = params.get("azval2", [0, 360])
                if len(azval) >= 2:
                    az_start, az_end = azval[0], azval[1]
            except:
                pass

            slat = stations[station]["lat"] if station in stations else 0
            slon = stations[station]["lon"] if station in stations else 0
            RH = 5.0  # typical reflector height

            # Draw Fresnel cone arcs and radial lines
            deg_per_m_lat = 1.0 / 111000
            deg_per_m_lon = 1.0 / (111000 * np.cos(np.radians(slat)))

            for elev_deg, color in [(5, "cyan"), (10, "lime"), (15, "yellow")]:
                d = RH / np.tan(np.radians(elev_deg))
                # Arc: series of points at distance d from station
                arc_pts = []
                for az in np.arange(az_start, az_end + 1, 2):
                    az_rad = np.radians(az)
                    pt_lat = slat + d * np.cos(az_rad) * deg_per_m_lat
                    pt_lon = slon + d * np.sin(az_rad) * deg_per_m_lon
                    arc_pts.append([pt_lat, pt_lon])
                if arc_pts:
                    fresnel_layers.append(dl.Polyline(
                        positions=arc_pts, color=color, weight=2,
                        dashArray="5,5" if elev_deg > 5 else "",
                    ))

            # Radial lines at azimuth boundaries
            for az_deg in [az_start, az_end]:
                d_max = RH / np.tan(np.radians(5))
                az_rad = np.radians(az_deg)
                end_lat = slat + d_max * np.cos(az_rad) * deg_per_m_lat
                end_lon = slon + d_max * np.sin(az_rad) * deg_per_m_lon
                fresnel_layers.append(dl.Polyline(
                    positions=[[slat, slon], [end_lat, end_lon]],
                    color="white", weight=1, opacity=0.7,
                ))

            # Build reference station markers for the selected station
            ref_markers = []
            ref_legend = []
            ext = station_cfg.get("external_data_sources", {})

            hydat_cfg = ext.get("hydat", {})
            if hydat_cfg.get("enabled") and hydat_cfg.get("station_id"):
                # Look up HYDAT station coordinates from the GeoMet API response cache
                # or use approximate position (same lake, very close)
                h_name = hydat_cfg.get("station_name", hydat_cfg["station_id"])
                h_dist = hydat_cfg.get("distance_km", "?")
                ref_legend.append(html.Div([
                    html.Span("◆ ", style={"color": "#4FC3F7", "fontSize": "1.2em"}),
                    f"HYDAT: {h_name} ({h_dist} km)",
                ], style={"fontSize": "0.75rem", "color": "#ccc"}))

            coops_cfg = ext.get("noaa_coops", {})
            if coops_cfg.get("enabled") and coops_cfg.get("nearest_station"):
                ns = coops_cfg["nearest_station"]
                c_lat = ns.get("latitude")
                c_lon = ns.get("longitude")
                c_name = ns.get("name", ns.get("id", ""))
                c_dist = ns.get("distance_km", "?")
                if c_lat and c_lon:
                    ref_markers.append(dl.CircleMarker(
                        center=[c_lat, c_lon], radius=7,
                        color="#FF69B4", fillColor="#FF69B4", fillOpacity=0.8,
                        children=[dl.Tooltip(f"CO-OPS: {c_name} ({c_dist} km)")],
                    ))
                ref_legend.append(html.Div([
                    html.Span("● ", style={"color": "#FF69B4", "fontSize": "1.2em"}),
                    f"CO-OPS: {c_name} ({c_dist} km)",
                ], style={"fontSize": "0.75rem", "color": "#ccc"}))

            ec_cfg = ext.get("ec_climate", {})
            if ec_cfg.get("enabled") and ec_cfg.get("station_id"):
                e_lat = ec_cfg.get("latitude")
                e_lon = ec_cfg.get("longitude")
                e_name = ec_cfg.get("station_name", "EC Climate")
                e_dist = ec_cfg.get("distance_km", "?")
                if e_lat and e_lon and (not isinstance(e_dist, (int, float)) or e_dist > 5):
                    # Only show marker if station is far enough to not overlap GNSS marker
                    ref_markers.append(dl.CircleMarker(
                        center=[e_lat, e_lon], radius=7,
                        color="#FFA726", fillColor="#FFA726", fillOpacity=0.8,
                        children=[dl.Tooltip(f"Temp: {e_name} ({e_dist} km)")],
                    ))
                ref_legend.append(html.Div([
                    html.Span("● ", style={"color": "#FFA726", "fontSize": "1.2em"}),
                    f"Temp: {e_name} ({e_dist} km)",
                ], style={"fontSize": "0.75rem", "color": "#ccc"}))

            content = html.Div([
                # Top row: Map + Polar + Info
                html.Div([
                    html.Div([
                        dl.Map([
                            dl.TileLayer(url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                        ] + [
                            dl.Marker(
                                position=[info["lat"], info["lon"]],
                                id={"type": "station-marker", "index": name},
                                children=[dl.Tooltip(name)],
                            ) for name, info in stations.items()
                        ] + fresnel_layers + ref_markers, center=center, zoom=map_zoom,
                           id=f"map-{station}-{year}",
                           style={"height": "400px", "borderRadius": "6px", "width": "100%"}),
                    ], style={"flex": "2"}),

                    # Polar + Info stacked
                    html.Div([
                        dcc.Graph(id="overview-polar", figure=polar_fig, style={"height": "350px"},
                                  config={"displayModeBar": False}),
                        html.Div(id="overview-selection-info",
                                 children=[
                                     html.H4(f"{station} {year}", style={"margin": "0 0 8px 0", "color": "#e0e0e0"}),
                                     html.Div(v3_lines if v3_lines else [html.P("No v3 data", style={"color": DARK_TEXT})]),
                                 ] + ([html.Div([
                                     html.Hr(style={"borderColor": DARK_BORDER, "margin": "6px 0"}),
                                     html.Span("Reference Stations", style={"color": "#8b949e", "fontSize": "0.75rem", "fontWeight": "bold"}),
                                 ] + ref_legend, style={"marginTop": "4px"})] if ref_legend else []) + [
                                     html.P("Drag on time series to filter polar diagram",
                                            style={"color": "#8b949e", "fontSize": "0.75rem", "marginTop": "8px"}),
                                 ],
                                 style={"padding": "8px 12px", "backgroundColor": DARK_CARD,
                                        "borderRadius": "6px", "border": f"1px solid {DARK_BORDER}"}),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "8px"}),
                ], style={"display": "flex", "gap": "12px", "width": "100%"}),

                # Time series below — with selection enabled
                html.Div([
                    dcc.Graph(id="overview-ts", figure=ts_fig, style={"height": "350px"},
                              config={"modeBarButtonsToAdd": ["select2d"]}),
                ], style={"marginTop": "8px"}),

                # Hidden store for per_arc data (subset of columns for performance)
                dcc.Store(id="overview-per-arc-store",
                          data=per_arc[["doy", "date_dt", "Azim", "RH", "Amp", "freq_group",
                                        "eminO", "emaxO"]].to_dict("records")
                          if per_arc is not None and not per_arc.empty else []),
            ])

        elif tab == "timeseries":
            v3 = load_v3(station, year)
            s1_images = list_s1_images(station)
            year_s1 = [img for img in s1_images if img["date"].startswith(str(year))]
            era5 = load_era5(station, year)
            smap = load_smap(station, year)

            # Determine how many rows we need
            has_era5 = era5 is not None and "t2m_mean" in era5.columns
            has_smap = smap is not None and "polarization_diff" in smap.columns
            has_v3 = v3 is not None and "v3_state" in v3.columns

            n_rows = 1  # RH always
            if has_v3: n_rows += 1
            if has_era5: n_rows += 1
            if has_smap: n_rows += 1

            row_heights = [0.5] + [0.15] * (n_rows - 1)  # RH gets most space
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                row_heights=row_heights, vertical_spacing=0.03)

            df = per_arc.copy() if per_arc is not None else pd.DataFrame()
            if not df.empty:
                if "date_dt" not in df.columns and "date" in df.columns:
                    df["date_dt"] = pd.to_datetime(df["date"])
                if "freq_group" not in df.columns:
                    df["freq_group"] = "L1"

                # RH by frequency
                for freq in sorted(df["freq_group"].unique()):
                    sub = df[df["freq_group"] == freq]
                    daily = sub.groupby("date_dt").agg(rh_median=("RH", "median"), n_arcs=("RH", "count")).reset_index()
                    color = FREQ_COLORS.get(freq, "#888")
                    fig.add_trace(go.Scatter(
                        x=daily["date_dt"], y=daily["rh_median"], mode="markers", name=freq,
                        marker=dict(size=4, color=color, opacity=0.6),
                        hovertemplate=f"{freq}<br>RH: %{{y:.3f}}m<br>n=%{{customdata}}<extra></extra>",
                        customdata=daily["n_arcs"],
                    ), row=1, col=1)

                daily_all = df.groupby("date_dt")["RH"].median().reset_index()
                fig.add_trace(go.Scatter(
                    x=daily_all["date_dt"], y=daily_all["RH"], mode="lines", name="Median",
                    line=dict(color="white", width=1.5),
                ), row=1, col=1)

                # S1 markers
                if year_s1:
                    v3_map = {}
                    if has_v3:
                        v3_map = dict(zip(v3["date_dt"].dt.strftime("%Y-%m-%d"), v3["v3_state"]))
                    s1_x = [pd.Timestamp(d["date"]) for d in year_s1]
                    rh_max = daily_all["RH"].max()
                    s1_colors = [V3_COLORS.get(v3_map.get(d["date"], ""), "#aaa") for d in year_s1]
                    fig.add_trace(go.Scatter(
                        x=s1_x, y=[rh_max + 0.05] * len(s1_x), mode="markers",
                        marker=dict(symbol="triangle-down", size=14, color=s1_colors,
                                    line=dict(color="white", width=1)),
                        name="S1 imagery", customdata=[d["date"] for d in year_s1],
                        hovertemplate="<b>S1: %{customdata}</b><extra>Click for imagery</extra>",
                    ), row=1, col=1)

            fig.update_yaxes(title_text="RH (m)", row=1, col=1)

            current_row = 2
            from datetime import datetime as _dt, timedelta as _td

            # ERA5 temperature
            if has_era5:
                era5_dates = [_dt(year, 1, 1) + _td(days=int(d) - 1) for d in era5["doy"]]
                fig.add_trace(go.Scatter(
                    x=era5_dates, y=era5["t2m_mean"], mode="lines",
                    name="ERA5 Temp", line=dict(color="#ff6b6b", width=1.5),
                    hovertemplate="Temp: %{y:.1f}°C<extra></extra>",
                ), row=current_row, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=current_row, col=1)
                fig.update_yaxes(title_text="°C", row=current_row, col=1)
                current_row += 1

            # SMAP polarization
            if has_smap:
                smap_dates = [_dt(year, 1, 1) + _td(days=int(d) - 1) for d in smap["doy"]]
                fig.add_trace(go.Scatter(
                    x=smap_dates, y=smap["polarization_diff"], mode="lines",
                    name="SMAP TBv-TBh", line=dict(color="#a855f7", width=1.5),
                    hovertemplate="SMAP pol: %{y:.1f}K<extra></extra>",
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="K", row=current_row, col=1)
                current_row += 1

            # v3 classification strip
            if has_v3:
                for state in _state_order(_get_mode(v3)):
                    mask = v3["v3_state"] == state
                    if mask.sum() > 0:
                        sub = v3[mask]
                        fig.add_trace(go.Bar(
                            x=sub["date_dt"], y=[1] * len(sub),
                            marker=dict(color=V3_COLORS.get(state, "#999")),
                            name=state, showlegend=True,
                        ), row=current_row, col=1)
                fig.update_yaxes(showticklabels=False, range=[0, 1.2], row=current_row, col=1)

            fig.update_xaxes(title_text="Date", row=n_rows, col=1)
            fig.update_layout(
                title=dict(text=f"{station} {year}", font=dict(size=14)),
                height=max(400, 180 * n_rows),
                margin=dict(l=50, r=20, t=50, b=30),
                barmode="stack",
                legend=dict(font=dict(size=8), orientation="h", y=-0.08),
                **PLOTLY_DARK,
            )

            content = html.Div([
                html.P("Click ▼ or any point to view nearest S1 imagery below.",
                       style={"color": "#8b949e", "fontSize": "0.8rem", "margin": "4px 0"}),
                dcc.Graph(id="ts-graph", figure=fig, style={"height": "60vh"}),
                html.Div(id="ts-imagery-panel",
                         style={"marginTop": "8px", "minHeight": "50px"}),
            ])

        elif tab == "polar":
            fig = build_polar_figure(per_arc, title=f"{station} {year} — Reflection Points")
            fig.update_layout(height=550)

            # Station metadata
            sinfo = stations.get(station, {})
            n_arcs = len(per_arc) if per_arc is not None else 0
            freq_list = sorted(per_arc["freq_group"].unique()) if per_arc is not None and "freq_group" in per_arc.columns else []
            az_range = f"{per_arc['Azim'].min():.0f}°–{per_arc['Azim'].max():.0f}°" if per_arc is not None and "Azim" in per_arc.columns else "?"
            rh_range = f"{per_arc['RH'].min():.2f}–{per_arc['RH'].max():.2f} m" if per_arc is not None else "?"

            # Get gnssir params for azimuth config
            station_cfg = load_stations_config().get(station, {})
            params_path = station_cfg.get("gnssir_json_params_path", "")
            az_config = "?"
            try:
                with open(PROJECT_ROOT / params_path) as f:
                    params = json.load(f)
                azval = params.get("azval2", [])
                if len(azval) >= 2:
                    az_config = f"{azval[0]:.0f}°–{azval[1]:.0f}°"
            except:
                pass

            # Additional context
            elev_range = "?"
            try:
                with open(PROJECT_ROOT / params_path) as f:
                    params = json.load(f)
                elev_range = f"{params.get('e1', '?')}°–{params.get('e2', '?')}°"
            except:
                pass

            suspect_az = station_cfg.get("suspect_azimuths", {})
            ice_free = station_cfg.get("ice_free_months", [])
            ant_height = station_cfg.get("ellipsoidal_height_m", "?")

            # Data availability
            import glob as _glob
            n_v3 = len(_glob.glob(str(PROJECT_ROOT / "results_annual" / station / f"{station}_*_ice_classification_v3.parquet")))
            n_s1 = len(_glob.glob(str(PROJECT_ROOT / "data" / station / "s1_fresnel" / f"{station}_*_s1_fresnel.tif")))
            has_era5 = (PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_era5.parquet").exists()
            has_smap = (PROJECT_ROOT / "results_annual" / station / "smap" / f"{station}_{year}_smap_comparison.parquet").exists()
            has_tec = (PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_tec.parquet").exists()

            def data_badge(label, available):
                color = "#2ea043" if available else "#484f58"
                return html.Span(label, style={
                    "backgroundColor": color, "color": "white", "padding": "2px 8px",
                    "borderRadius": "12px", "fontSize": "0.75rem", "marginRight": "4px",
                })

            def table_row(label, value):
                return html.Tr([
                    html.Td(label, style={"color": "#8b949e", "paddingRight": "12px", "paddingBottom": "6px"}),
                    html.Td(value, style={"paddingBottom": "6px"}),
                ])

            metadata = html.Div([
                html.H3(f"{station}", style={"color": "#e0e0e0", "marginTop": "0", "marginBottom": "12px"}),

                # Data badges
                html.Div([
                    data_badge(f"v3: {n_v3}yr", n_v3 > 0),
                    data_badge(f"S1: {n_s1}", n_s1 > 0),
                    data_badge("ERA5", has_era5),
                    data_badge("SMAP", has_smap),
                    data_badge("TEC", has_tec),
                ], style={"marginBottom": "12px"}),

                html.Table([
                    table_row("Location", f"{sinfo.get('lat', 0):.4f}°N, {abs(sinfo.get('lon', 0)):.4f}°W"),
                    table_row("Antenna ht", f"{ant_height} m"),
                    table_row("Year", str(year)),
                    table_row("Arcs", f"{n_arcs:,}"),
                    table_row("Frequencies", ", ".join(freq_list)),
                    table_row("Elevation", elev_range),
                    table_row("Az config", az_config),
                    table_row("Az observed", az_range),
                    table_row("RH range", rh_range),
                    table_row("Ice-free months", ", ".join(str(m) for m in ice_free) if ice_free else "not set"),
                    table_row("Available years", ", ".join(str(y) for y in sinfo.get("years", []))),
                ], style={"width": "100%", "color": DARK_TEXT, "fontSize": "0.85rem",
                          "borderCollapse": "collapse"}),

                # Suspect azimuths warning
            ] + ([
                html.Div([
                    html.Span("⚠ Suspect azimuths: ", style={"color": "#d29922", "fontWeight": "bold"}),
                    html.Span(str(suspect_az), style={"color": "#d29922", "fontSize": "0.8rem"}),
                ], style={"marginTop": "8px"})
            ] if suspect_az else []),
            style={"padding": "16px", "backgroundColor": DARK_CARD,
                      "borderRadius": "6px", "border": f"1px solid {DARK_BORDER}"})

            content = html.Div([
                html.Div([
                    dcc.Graph(figure=fig, style={"height": "550px"},
                              config={"displayModeBar": True}),
                ], style={"flex": "2"}),
                html.Div([metadata], style={"flex": "1", "minWidth": "250px"}),
            ], style={"display": "flex", "gap": "16px", "width": "100%"})

        elif tab == "ice":
            v3 = load_v3(station, year)
            v2 = load_v2(station, year)
            snr = load_snr_features(station, year)
            era5 = load_era5(station, year)
            smap = load_smap(station, year)
            threshold = load_mahal_threshold(station)
            daily_feat = load_daily_features(station, year)

            # State timeline strip
            timeline = build_state_timeline(v3, station, year)

            # Metrics strip
            metrics = build_metrics_strip(v3)

            # Feature direction summary
            feat_dir = build_feature_direction_summary(station, year)

            # Mahalanobis distance plot
            mahal_fig = build_mahalanobis_figure(v2, v3, threshold, station, year)

            # SNR features figure — uses daily_features best variants when available
            snr_fig = build_ice_features_figure(snr, v3, era5, smap, station, year,
                                                 daily_features=daily_feat)

            content = html.Div([
                timeline,
                metrics,
                html.Div([
                    html.Div([feat_dir], style={"flex": "1", "minWidth": "300px"}),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "8px"}),
                dcc.Graph(figure=mahal_fig, style={"height": "220px"},
                          config={"displayModeBar": True}),
                dcc.Graph(figure=snr_fig, config={"displayModeBar": True}),
            ])

        elif tab == "imagery":
            s1_images = list_s1_images(station)
            v3 = load_v3(station, year)

            if not s1_images:
                content = html.Div([
                    html.P(f"No S1 imagery downloaded for {station}.",
                           style={"color": DARK_TEXT, "fontSize": "1.1rem"}),
                    html.Code(f"python scripts/s1_fresnel_search.py --station {station} && "
                              f"python scripts/s1_fresnel_download.py --station {station}",
                              style={"color": "#4a90d9", "fontSize": "0.9rem"}),
                ], style={"padding": "20px"})
            else:
                # Filter to current year
                year_images = [img for img in s1_images if img["date"].startswith(str(year))]
                if not year_images:
                    year_images = s1_images

                v3_map = {}
                if v3 is not None and "v3_state" in v3.columns:
                    v3_map = dict(zip(v3["date_dt"].dt.strftime("%Y-%m-%d"), v3["v3_state"]))

                # Build thumbnail grid — small VV preview for each date
                grid_items = []
                for img in year_images:
                    state = v3_map.get(img["date"], "?")
                    state_color = V3_COLORS.get(state, "#666")
                    grid_items.append(
                        html.Div([
                            html.Div(img["date"][-5:], style={"fontSize": "0.75rem", "color": DARK_TEXT}),
                            html.Div(state, style={"fontSize": "0.65rem", "color": state_color,
                                                    "fontWeight": "bold"}),
                        ], id={"type": "imagery-thumb", "index": img["date"]},
                           n_clicks=0,
                           style={
                               "padding": "6px 8px", "cursor": "pointer",
                               "backgroundColor": DARK_CARD, "borderRadius": "4px",
                               "border": f"2px solid {state_color}",
                               "textAlign": "center", "minWidth": "70px",
                               "transition": "background-color 0.2s",
                           })
                    )

                # Show first image by default
                first_date = year_images[0]["date"] if year_images else None
                initial_imagery = render_s1_imagery(station, first_date, s1_images, v3) if first_date else html.P("No images")

                content = html.Div([
                    # Date grid along top
                    html.Div([
                        html.Label("Click a date to view:", style={"color": DARK_TEXT, "fontWeight": "bold",
                                                                    "marginRight": "12px", "whiteSpace": "nowrap"}),
                        html.Div(grid_items, style={
                            "display": "flex", "gap": "4px", "flexWrap": "wrap", "flex": "1",
                        }),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px",
                              "overflowX": "auto", "padding": "4px"}),
                    # Main imagery display
                    html.Div(id="imagery-display", children=initial_imagery),
                ])
        elif tab == "features":
            _subtab_style = {"backgroundColor": "#161b22", "color": "#8b949e",
                             "border": "1px solid #30363d", "padding": "6px 12px"}
            _subtab_sel = {"backgroundColor": "#21262d", "color": "#e0e0e0",
                           "borderTop": "2px solid #58a6ff", "padding": "6px 12px"}
            content = html.Div([
                build_feature_definitions_panel(),
                dcc.Tabs(id="features-subtabs", value="clustering", children=[
                    dcc.Tab(label="Feature Clustering", value="clustering",
                            style=_subtab_style, selected_style=_subtab_sel),
                    dcc.Tab(label="Discrimination Summary", value="discrimination",
                            style=_subtab_style, selected_style=_subtab_sel),
                    dcc.Tab(label="Normalization Comparison", value="normalization",
                            style=_subtab_style, selected_style=_subtab_sel),
                    dcc.Tab(label="PRN Weights", value="prn_weights",
                            style=_subtab_style, selected_style=_subtab_sel),
                ]),
                # Normalization feature selector — always in DOM, shown/hidden by sub-tab callback
                html.Div([
                    html.Label("Base feature:", style={"color": DARK_TEXT, "fontWeight": "bold",
                                                        "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="norm-base-feature",
                        options=[{"label": f, "value": f}
                                 for f in ["CLR", "AF", "PR", "gamma", "MS", "VS"]],
                        value="MS",
                        style={"width": "200px", "color": "#000"},
                        clearable=False,
                    ),
                ], id="norm-selector-row",
                   style={"display": "none", "alignItems": "center", "padding": "8px 0"}),
                html.Div(id="features-subtab-content",
                         style={"minHeight": "400px", "padding": "8px 0"}),
            ])

        elif tab == "investigation":
            content = build_investigation_tab()

        else:
            content = html.P("Unknown tab")

        return content, quality_items

    # ----- Feature clustering callback (threshold slider) -----

    @callback(
        Output("cluster-heatmap", "figure"),
        Output("cluster-summary", "children"),
        Input("cluster-threshold", "value"),
        Input("cluster-metric", "value"),
        Input("cluster-init-trigger", "data"),
        State("station-select", "value"),
        State("year-select", "value"),
    )
    def update_cluster_heatmap(threshold, metric, _trigger, station, year):
        """Recompute the clustered heatmap when threshold or metric changes."""
        from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram as scipy_dendrogram
        from scipy.spatial.distance import squareform
        from sklearn.preprocessing import StandardScaler

        snr = load_snr_features(station, year)
        if snr is None:
            return go.Figure(), "No data"

        feature_cols = [c for c in ["CLR", "PR", "AF", "gamma", "phase", "MS", "VS", "SP"]
                        if c in snr.columns]
        if len(feature_cols) < 3:
            return go.Figure(), "Too few features"

        df = snr[feature_cols].dropna()

        if metric == "partial":
            # Partial correlation from precision matrix (inverse covariance).
            # pcorr(i,j) = -P(i,j) / sqrt(P(i,i)*P(j,j)) where P = cov^{-1}
            # This is the same covariance structure that Mahalanobis distance uses.
            scaled = StandardScaler().fit_transform(df)
            cov = np.cov(scaled, rowvar=False)
            try:
                precision = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Near-singular: add small ridge
                precision = np.linalg.inv(cov + 1e-6 * np.eye(len(feature_cols)))

            # Convert precision to partial correlation
            d = np.sqrt(np.diag(precision))
            pcorr = -precision / np.outer(d, d)
            np.fill_diagonal(pcorr, 1.0)
            corr_matrix = pd.DataFrame(pcorr, index=feature_cols, columns=feature_cols)
            dist_matrix = 1 - np.abs(pcorr)
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            heatmap_label = "Partial r"
            metric_label = "Partial Correlation (Precision Matrix)"
        else:
            # Standard Pearson correlation
            corr_matrix = df.corr()
            dist_matrix = (1 - corr_matrix.abs()).values
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            heatmap_label = "Pearson r"
            metric_label = "Correlation"

        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        coph_r, _ = cophenet(Z, condensed)
        cluster_ids = fcluster(Z, t=threshold, criterion="distance")

        # Cluster colors
        palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
        unique_clusters = sorted(set(cluster_ids))
        color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}

        # Build node->cluster mapping for coloring
        n = len(feature_cols)
        node_cluster = {}
        for i in range(n):
            node_cluster[i] = cluster_ids[i]
        for i in range(len(Z)):
            left, right = int(Z[i, 0]), int(Z[i, 1])
            h = Z[i, 2]
            lc, rc = node_cluster.get(left), node_cluster.get(right)
            node_cluster[n + i] = lc if (h <= threshold and lc == rc and lc is not None) else None

        def link_color(nid):
            c = node_cluster.get(nid)
            return color_map[c] if c is not None else "#888888"

        dend = scipy_dendrogram(Z, labels=feature_cols, no_plot=True,
                                link_color_func=link_color)
        order = dend["leaves"]
        ordered_labels = [feature_cols[i] for i in order]
        corr_vals = corr_matrix.values
        corr_ordered = corr_vals[np.ix_(order, order)]

        # Build Plotly figure
        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.2, 0.8],
            vertical_spacing=0.02, shared_xaxes=True,
        )

        # Dendrogram traces
        icoord = np.array(dend["icoord"])
        dcoord = np.array(dend["dcoord"])
        for i in range(len(icoord)):
            fig.add_trace(go.Scatter(
                x=icoord[i], y=dcoord[i], mode="lines",
                line=dict(color=dend["color_list"][i], width=2.5),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

        # Threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="#888888",
                      line_width=1, row=1, col=1)

        # Heatmap
        hover_text = []
        for i, rl in enumerate(ordered_labels):
            row_hover = []
            for j, cl in enumerate(ordered_labels):
                row_hover.append(f"{rl} vs {cl}<br>{heatmap_label} = {corr_ordered[i, j]:.3f}")
            hover_text.append(row_hover)

        fig.add_trace(go.Heatmap(
            z=corr_ordered,
            x=ordered_labels, y=ordered_labels,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_ordered],
            texttemplate="%{text}", textfont={"size": 11},
            hovertext=hover_text, hoverinfo="text",
            colorbar=dict(title=heatmap_label, thickness=15, len=0.6, y=0.35),
        ), row=2, col=1)

        # Build cluster legend as subtitle
        cluster_info = []
        for cid in unique_clusters:
            members = [feature_cols[i] for i in range(n) if cluster_ids[i] == cid]
            color = color_map[cid]
            cluster_info.append(
                f"<span style='color:{color}'><b>C{cid}</b>: {', '.join(members)}</span>"
            )

        fig.update_layout(
            title=dict(
                text=(f"{station} {year} — {metric_label} (threshold={threshold})"
                      f"<br><sub>cophenetic r={coph_r:.3f} | "
                      f"{' &nbsp;|&nbsp; '.join(cluster_info)}</sub>"),
                font=dict(size=14),
            ),
            **PLOTLY_DARK,
            height=700,
            margin=dict(l=80, r=40, t=100, b=80),
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(title_text="Distance", showgrid=False, zeroline=False, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)

        # Summary text
        summary = html.Div([
            html.Span(f"{len(unique_clusters)} clusters at threshold {threshold}",
                       style={"fontWeight": "bold", "marginRight": "16px"}),
            html.Span(f"cophenetic r = {coph_r:.3f}", style={"marginRight": "16px"}),
            html.Span(f"{len(df):,} arcs", style={"marginRight": "16px"}),
            html.Span(f"metric: {metric_label}", style={"color": "#8b949e", "marginRight": "16px"}),
            html.Br(),
            *[html.Span([
                html.Span(f"C{cid}", style={"color": color_map[cid], "fontWeight": "bold"}),
                f": {', '.join(feature_cols[i] for i in range(n) if cluster_ids[i] == cid)}  ",
            ]) for cid in unique_clusters],
        ])

        return fig, summary

    # ----- Features sub-tab routing -----

    @callback(
        Output("features-subtab-content", "children"),
        Output("norm-selector-row", "style"),
        Input("features-subtabs", "value"),
        Input("norm-base-feature", "value"),
        State("station-select", "value"),
        State("year-select", "value"),
    )
    def update_features_subtab(subtab, norm_feature, station, year):
        norm_hidden = {"display": "none"}
        norm_visible = {"display": "flex", "alignItems": "center", "padding": "8px 0"}

        if not station or not year:
            return html.Div(), norm_hidden

        if subtab == "clustering":
            snr = load_snr_features(station, year)
            if snr is None:
                return html.Div(html.P(f"No SNR features for {station} {year}.",
                                       style={"color": DARK_TEXT, "padding": "20px"})), norm_hidden
            return html.Div([
                html.Div([
                    html.Div([
                        html.Label("Distance metric:",
                                   style={"color": DARK_TEXT, "fontWeight": "bold",
                                          "marginBottom": "4px", "display": "block"}),
                        dcc.RadioItems(
                            id="cluster-metric",
                            options=[
                                {"label": " Correlation (1-|r|)", "value": "correlation"},
                                {"label": " Partial correlation (precision matrix)",
                                 "value": "partial"},
                            ],
                            value="partial",
                            style={"color": "#e0e0e0", "fontSize": "0.95rem"},
                            inputStyle={"marginRight": "8px", "accentColor": "#58a6ff"},
                            labelStyle={"display": "block", "marginBottom": "6px",
                                        "color": "#e0e0e0", "cursor": "pointer"},
                        ),
                        html.Div([
                            html.Span("Correlation: ", style={"color": "#8b949e"}),
                            html.Span("do these features move together?",
                                      style={"color": "#8b949e", "fontSize": "0.8rem"}),
                            html.Br(),
                            html.Span("Partial: ", style={"color": "#8b949e"}),
                            html.Span("are they directly related, or only through other features?",
                                      style={"color": "#8b949e", "fontSize": "0.8rem"}),
                        ], style={"marginTop": "6px"}),
                    ], style={"flex": "0 0 340px", "padding": "8px 20px"}),
                    html.Div([
                        html.Label("Cluster threshold:",
                                   style={"color": DARK_TEXT, "fontWeight": "bold",
                                          "marginBottom": "4px", "display": "block"}),
                        dcc.Slider(
                            id="cluster-threshold",
                            min=0.1, max=0.9, step=0.05, value=0.4,
                            marks={v: {"label": f"{v}", "style": {"color": "#c9d1d9"}}
                                   for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Span("Low = strict (only near-identical cluster) | "
                                  "High = loose (more features merge)",
                                  style={"color": "#8b949e", "fontSize": "0.8rem"}),
                    ], style={"flex": "1 1 400px", "padding": "8px 20px",
                              "minWidth": "300px"}),
                ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start",
                          "backgroundColor": DARK_CARD, "borderRadius": "8px",
                          "margin": "8px", "border": f"1px solid {DARK_BORDER}"}),
                html.Div([
                    dcc.Graph(id="cluster-heatmap", style={"height": "700px"}),
                ], style={"width": "100%"}),
                html.Div(id="cluster-summary", style={
                    "padding": "12px 20px", "color": DARK_TEXT, "fontSize": "0.9rem",
                }),
                dcc.Store(id="cluster-init-trigger", data=True),
            ]), norm_hidden

        elif subtab == "discrimination":
            daily_feat = load_daily_features(station, year)
            v3 = load_v3(station, year)
            if daily_feat is None:
                return html.Div("No daily_features.parquet found. Run feature_aggregator.py first.",
                                style={"color": DARK_TEXT, "padding": "16px"}), norm_hidden
            return build_discrimination_summary(daily_feat, v3, station, year), norm_hidden

        elif subtab == "normalization":
            daily_feat = load_daily_features(station, year)
            if daily_feat is None:
                return html.Div("No daily_features.parquet found. Run feature_aggregator.py first.",
                                style={"color": DARK_TEXT, "padding": "16px"}), norm_hidden
            v3 = load_v3(station, year)
            feature = norm_feature or "MS"
            return build_normalization_comparison(
                daily_feat, v3, station, year, feature), norm_visible

        elif subtab == "prn_weights":
            return build_prn_weight_heatmap(station, year), norm_hidden

        return html.Div(), norm_hidden

    return app


def main():
    parser = argparse.ArgumentParser(description="GNSS-IR Dash Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"Starting dashboard at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
