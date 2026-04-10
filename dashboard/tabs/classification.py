# ABOUTME: Analysis tab — time-aligned panels: arc density, Mahalanobis, state, contributions
# ABOUTME: Integrates scorecard reliability into feature display; falls back to legacy v3

"""Analysis tab for the GNSS-IR dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc

from dashboard.utils import (
    PLOTLY_DARK, DARK_BG, DARK_CARD, DARK_BORDER, DARK_TEXT,
    V3_COLORS, FEATURE_LABELS,
    get_mode, state_order, non_baseline_states,
)
from dashboard.data_loader import (
    load_v3, load_v2, load_snr_features, load_era5, load_smap,
    load_mahal_threshold, load_daily_features,
    load_anomaly_scores, load_feature_scorecard, load_baseline_definition,
)


def render(station, year):
    """Render the Analysis tab.

    If anomaly_scores.csv exists, shows time-aligned analysis panels with
    observational labels. Otherwise falls back to legacy v3 display.
    """
    scores = load_anomaly_scores(station, year)
    scorecard = load_feature_scorecard(station, year)
    baseline_def = load_baseline_definition(station, year)

    if scores is not None:
        return _render_analysis(station, year, scores, scorecard, baseline_def)
    else:
        return _render_legacy(station, year)


def _render_analysis(station, year, scores, scorecard, baseline_def):
    """Render new pipeline analysis view with time-aligned panels."""
    v3 = load_v3(station, year)
    daily_feat = load_daily_features(station, year)

    # Metrics strip
    metrics = _build_analysis_metrics(scores, baseline_def)

    # Main time-aligned figure: arc density + Mahal + state strip + contributions
    main_fig = _build_time_aligned_figure(scores, daily_feat, scorecard,
                                          baseline_def, station, year)

    # Feature reliability heatmap
    heatmap = _build_reliability_heatmap(scorecard, station, year) if scorecard is not None else html.Div()

    # Scorecard summary
    scorecard_panel = _build_scorecard_panel(scorecard, station, year) if scorecard is not None else html.Div()

    # SNR features below if available
    snr = load_snr_features(station, year)
    era5 = load_era5(station, year)
    smap = load_smap(station, year)
    snr_section = []
    if snr is not None and v3 is not None:
        from dashboard.app import build_ice_features_figure
        snr_fig = build_ice_features_figure(snr, v3, era5, smap, station, year,
                                             daily_features=daily_feat)
        snr_section = [dcc.Graph(figure=snr_fig, config={"displayModeBar": True})]

    return html.Div([
        metrics,
        dcc.Graph(figure=main_fig, config={"displayModeBar": True}),
        heatmap,
        scorecard_panel,
    ] + snr_section)


def _render_legacy(station, year):
    """Fallback: render legacy v3 classification view."""
    from dashboard.app import (
        build_state_timeline, build_metrics_strip,
        build_feature_direction_summary, build_mahalanobis_figure,
        build_ice_features_figure,
    )

    v3 = load_v3(station, year)
    v2 = load_v2(station, year)
    snr = load_snr_features(station, year)
    era5 = load_era5(station, year)
    smap = load_smap(station, year)
    threshold = load_mahal_threshold(station)
    daily_feat = load_daily_features(station, year)

    timeline = build_state_timeline(v3, station, year)
    metrics = build_metrics_strip(v3)
    feat_dir = build_feature_direction_summary(station, year)
    mahal_fig = build_mahalanobis_figure(v2, v3, threshold, station, year)
    snr_fig = build_ice_features_figure(snr, v3, era5, smap, station, year,
                                         daily_features=daily_feat)

    return html.Div([
        timeline,
        metrics,
        html.Div([
            html.Div([feat_dir], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "8px"}),
        dcc.Graph(figure=mahal_fig, style={"height": "220px"}, config={"displayModeBar": True}),
        dcc.Graph(figure=snr_fig, config={"displayModeBar": True}),
    ])


# ---------------------------------------------------------------------------
# Time-aligned multi-panel figure
# ---------------------------------------------------------------------------

def _build_time_aligned_figure(scores, daily_feat, scorecard, baseline_def,
                                station, year):
    """Build a single figure with shared x-axis: arc density, Mahal, state, contributions.

    All panels share the DOY x-axis so the user can visually correlate
    data gaps with anomaly scores and feature contributions.
    """
    # Determine top contributing features
    contrib_cols = [c for c in scores.columns if c.endswith("_contribution")]
    anomalous = scores[scores["state"] == "anomalous"]
    if len(anomalous) < 3:
        anomalous = scores

    if contrib_cols:
        mean_contrib = anomalous[contrib_cols].abs().mean().sort_values(ascending=False)
        top_features = mean_contrib.head(6).index.tolist()
    else:
        top_features = []

    n_feat_rows = min(len(top_features), 6)
    # Rows: arc density, Mahal distance, state strip, then one row per top feature
    n_rows = 3 + n_feat_rows
    row_heights = [0.08, 0.25, 0.04] + [0.63 / max(n_feat_rows, 1)] * n_feat_rows

    subplot_titles = ["Arc Density", "Anomaly Score (Mahalanobis)", "State"]
    for col in top_features:
        feat_name = col.replace("_contribution", "")
        subplot_titles.append(FEATURE_LABELS.get(feat_name, feat_name))

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.015,
        subplot_titles=subplot_titles,
    )

    bl_start = baseline_def["start_doy"] if baseline_def else None
    bl_end = baseline_def["end_doy"] if baseline_def else None

    # --- Row 1: Arc density histogram ---
    _add_arc_density(fig, scores, daily_feat, row=1, bl_start=bl_start, bl_end=bl_end)

    # --- Row 2: Mahalanobis distance ---
    _add_mahal_panel(fig, scores, row=2, bl_start=bl_start, bl_end=bl_end)

    # --- Row 3: State strip ---
    _add_state_strip(fig, scores, row=3)

    # --- Rows 4+: Feature contributions ---
    # Build reliability lookup from scorecard
    reliable_windows = _build_reliability_lookup(scorecard) if scorecard is not None else {}

    for i, col in enumerate(top_features):
        feat_name = col.replace("_contribution", "")
        _add_feature_panel(fig, scores, daily_feat, col, feat_name,
                           row=4 + i, reliable_windows=reliable_windows,
                           bl_start=bl_start, bl_end=bl_end)

    fig.update_xaxes(title_text="Day of Year", row=n_rows, col=1)
    fig.update_layout(
        height=max(600, 120 + 100 * n_rows),
        margin=dict(l=60, r=20, t=30, b=30),
        showlegend=False,
        **PLOTLY_DARK,
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=10, color="#8b949e"), x=0.01, xanchor="left")

    return fig


def _add_arc_density(fig, scores, daily_feat, row, bl_start, bl_end):
    """Arc density bar chart — bars colored by count relative to median."""
    # Use daily_features if available (has all days), otherwise scores
    if daily_feat is not None and "doy" in daily_feat.columns:
        pooled = daily_feat[daily_feat.get("azimuth_bin", -1) == -1] if "azimuth_bin" in daily_feat.columns else daily_feat
        doys = pooled["doy"].values
        arcs = pooled["n_arcs"].values
    elif "n_arcs" in scores.columns:
        doys = scores["doy"].values
        arcs = scores["n_arcs"].values
    else:
        return

    median_arcs = np.median(arcs)
    colors = ["#2d9a6b" if a >= median_arcs else "#d29922" if a >= median_arcs * 0.5 else "#d94452"
              for a in arcs]

    fig.add_trace(go.Bar(
        x=doys, y=arcs, marker_color=colors,
        hovertemplate="DOY %{x}<br>%{y} arcs<extra></extra>",
        showlegend=False,
    ), row=row, col=1)

    fig.update_yaxes(title_text="arcs", row=row, col=1, tickfont=dict(size=8),
                     title_font=dict(size=9))

    # Baseline shading (blue)
    if bl_start and bl_end:
        fig.add_vrect(x0=bl_start - 0.5, x1=bl_end + 0.5, row=row, col=1,
                      fillcolor="rgba(74,144,217,0.12)", line_width=0)


def _add_mahal_panel(fig, scores, row, bl_start, bl_end):
    """Mahalanobis distance with marker opacity scaled by n_arcs."""
    if "mahal_distance" not in scores.columns:
        return

    # Use opacity for confidence (more arcs = more opaque)
    n_arcs = scores["n_arcs"].values if "n_arcs" in scores.columns else np.ones(len(scores)) * 30
    opacity_min, opacity_max = 0.3, 1.0
    if n_arcs.max() > n_arcs.min():
        opacities = opacity_min + (n_arcs - n_arcs.min()) / (n_arcs.max() - n_arcs.min()) * (opacity_max - opacity_min)
    else:
        opacities = np.full(len(scores), 0.8)

    for state in ["baseline", "transition_in", "anomalous", "transition_out"]:
        mask = scores["state"] == state
        if mask.sum() == 0:
            continue
        sub = scores[mask]
        idx = mask.values.nonzero()[0]
        fig.add_trace(go.Scatter(
            x=sub["doy"].values, y=sub["mahal_distance"].values,
            mode="markers", name=state,
            marker=dict(
                size=6,
                color=V3_COLORS.get(state, "#999"),
                opacity=opacities[idx],
            ),
            hovertemplate=(f"{state}<br>DOY %{{x}}<br>d=%{{y:.1f}}"
                           f"<br>n_arcs=%{{customdata}}<extra></extra>"),
            customdata=sub["n_arcs"].values if "n_arcs" in sub.columns else None,
            showlegend=False,
        ), row=row, col=1)

    # Annotation explaining opacity
    fig.add_annotation(
        text="opacity = arc count",
        xref="paper", yref=f"y{row}" if row > 1 else "y",
        x=1.0, y=1.0, xanchor="right", yanchor="top",
        showarrow=False, font=dict(size=8, color="#8b949e"),
        row=row, col=1,
    )

    fig.update_yaxes(title_text="Mahal d", row=row, col=1, type="log",
                     tickfont=dict(size=8), title_font=dict(size=9))

    # Baseline shading (blue)
    if bl_start and bl_end:
        fig.add_vrect(x0=bl_start - 0.5, x1=bl_end + 0.5, row=row, col=1,
                      fillcolor="rgba(74,144,217,0.12)", line_width=0)


def _add_state_strip(fig, scores, row):
    """Thin colored bar of state per day."""
    for state in ["baseline", "transition_in", "anomalous", "transition_out"]:
        mask = scores["state"] == state
        if mask.sum() == 0:
            continue
        sub = scores[mask]
        fig.add_trace(go.Bar(
            x=sub["doy"].values, y=[1] * len(sub),
            marker_color=V3_COLORS.get(state, "#999"),
            showlegend=False,
            hovertemplate=f"{state}<br>DOY %{{x}}<extra></extra>",
        ), row=row, col=1)

    fig.update_yaxes(showticklabels=False, range=[0, 1.2], row=row, col=1,
                     fixedrange=True)


def _add_feature_panel(fig, scores, daily_feat, contrib_col, feat_name,
                        row, reliable_windows, bl_start, bl_end):
    """Feature contribution panel — line + filled area.

    Reliability info is shown in the heatmap below, not overlaid here.
    """
    if contrib_col in scores.columns:
        vals = scores[contrib_col].values
        fig.add_trace(go.Scatter(
            x=scores["doy"].values, y=vals,
            mode="lines", line=dict(width=1.5, color="#58a6ff"),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
            hovertemplate=f"DOY %{{x}}<br>contrib=%{{y:.2f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=1)

    fig.update_yaxes(tickfont=dict(size=7), title_font=dict(size=9), row=row, col=1)

    # Baseline shading (blue)
    if bl_start and bl_end:
        fig.add_vrect(x0=bl_start - 0.5, x1=bl_end + 0.5, row=row, col=1,
                      fillcolor="rgba(74,144,217,0.12)", line_width=0)


def _build_reliability_lookup(scorecard):
    """Build lookup of unreliable windows per feature from scorecard.

    Returns dict: {feature_name: [(start_doy, end_doy), ...]} for FAILED windows.
    """
    if scorecard is None or scorecard.empty:
        return {}

    failed = scorecard[scorecard["usable"] == False]
    lookup = {}
    for _, row in failed.iterrows():
        feat = row["feature"]
        w_start = row["window_start_doy"]
        ws = row["window_size"]
        w_days = 14 if ws == "2w" else 28
        w_end = w_start + w_days

        lookup.setdefault(feat, []).append((w_start, w_end))

    return lookup


# ---------------------------------------------------------------------------
# Feature reliability heatmap
# ---------------------------------------------------------------------------

def _build_reliability_heatmap(scorecard, station, year):
    """Heatmap: features x DOY windows, colored by |Cohen's d|.

    Green cells = usable (intensity shows effect size). Dark cells = failed.
    Only features that pass in at least one window are shown.
    """
    if scorecard is None or scorecard.empty:
        return html.Div()

    # Use 4w windows for a cleaner view
    sc = scorecard[scorecard["window_size"] == "4w"].copy()
    if sc.empty:
        sc = scorecard.copy()

    # Only keep features that are usable in at least one window
    usable_features = set(sc[sc["usable"] == True]["feature"].unique())
    if not usable_features:
        return html.Div("No features passed all 5 gates in any window.",
                        style={"color": DARK_TEXT, "padding": "8px"})

    # Sort by max |d| (best at top of chart = bottom of list for plotly)
    feat_order = (sc[sc["feature"].isin(usable_features)]
                   .groupby("feature")["gate4_cohens_d"]
                   .apply(lambda x: x.abs().max())
                   .sort_values(ascending=True))
    features = feat_order.index.tolist()

    windows = sorted(sc["window_start_doy"].unique())
    if not features or not windows:
        return html.Div()

    # Build matrix: |d| for usable, NaN for failed/missing
    z_matrix = np.full((len(features), len(windows)), np.nan)
    text_matrix = [[""] * len(windows) for _ in range(len(features))]
    hover_matrix = [[""] * len(windows) for _ in range(len(features))]

    for i, feat in enumerate(features):
        for j, w in enumerate(windows):
            row = sc[(sc["feature"] == feat) & (sc["window_start_doy"] == w)]
            if row.empty:
                continue
            r = row.iloc[0]
            if r["usable"]:
                d_val = abs(r["gate4_cohens_d"]) if pd.notna(r["gate4_cohens_d"]) else 0
                z_matrix[i, j] = d_val
                text_matrix[i][j] = f"{d_val:.1f}"
                hover_matrix[i][j] = f"|d|={d_val:.2f}"
            else:
                z_matrix[i, j] = 0  # dark cell
                text_matrix[i][j] = ""
                hover_matrix[i][j] = r.get("failure_reason", "failed")

    labels = [FEATURE_LABELS.get(f, f) for f in features]

    # Green-only colorscale: dark = no signal, bright green = strong signal
    z_max = max(3, np.nanmax(z_matrix) if np.any(np.isfinite(z_matrix)) else 3)
    colorscale = [
        [0.0, "#161b22"],    # no discrimination (dark background)
        [0.15, "#1a3a28"],   # weak
        [0.35, "#2d9a6b"],   # medium effect
        [0.6, "#4ac89a"],    # strong
        [1.0, "#a8f0d4"],    # very strong
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[f"DOY {w}" for w in windows],
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=9, color="#e0e0e0"),
        customdata=hover_matrix,
        colorscale=colorscale,
        zmin=0, zmax=z_max,
        colorbar=dict(
            title=dict(text="Cohen's |d|", font=dict(size=10)),
            tickfont=dict(size=8),
            tickvals=[0, 0.5, 1, 2, 3] if z_max <= 5 else None,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"{station} {year}: Feature Discriminability by Window (4-week, stride 1-week)",
            font=dict(size=13),
        ),
        height=max(250, 24 * len(features) + 100),
        margin=dict(l=180, r=60, t=50, b=50),
        xaxis=dict(tickangle=45, tickfont=dict(size=8), side="bottom"),
        yaxis=dict(tickfont=dict(size=9)),
        **PLOTLY_DARK,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True},
                     style={"marginTop": "8px"})


# ---------------------------------------------------------------------------
# Metrics strip
# ---------------------------------------------------------------------------

def _build_analysis_metrics(scores, baseline_def):
    """Summary metric cards for observational analysis."""
    if scores is None or "state" not in scores.columns:
        return html.Div()

    counts = scores["state"].value_counts()
    total = len(scores)

    def card(label, value, color="#c9d1d9"):
        return html.Div([
            html.Div(str(value), style={"fontSize": "1.5rem", "fontWeight": "bold", "color": color}),
            html.Div(label, style={"fontSize": "0.75rem", "color": "#8b949e"}),
        ], style={"padding": "8px 16px", "backgroundColor": DARK_CARD,
                  "borderRadius": "6px", "border": f"1px solid {DARK_BORDER}",
                  "textAlign": "center", "minWidth": "80px"})

    cards = [
        card("Total Days", total),
        card("Baseline", counts.get("baseline", 0), "#2d9a6b"),
        card("Anomalous", counts.get("anomalous", 0), "#e07b39"),
        card("Trans. In", counts.get("transition_in", 0), "#f0ad4e"),
        card("Trans. Out", counts.get("transition_out", 0), "#d94452"),
    ]

    if baseline_def:
        prov = " (prov.)" if baseline_def.get("provisional") else ""
        cards.append(card(
            f"Baseline DOY{prov}",
            f"{int(baseline_def['start_doy'])}-{int(baseline_def['end_doy'])}",
            "#58a6ff",
        ))

    if "n_arcs" in scores.columns:
        cards.append(card("Median arcs/day", f"{scores['n_arcs'].median():.0f}"))

    return html.Div(cards, style={
        "display": "flex", "gap": "8px", "flexWrap": "wrap",
        "marginBottom": "8px", "marginTop": "4px",
    })


# ---------------------------------------------------------------------------
# Scorecard summary panel
# ---------------------------------------------------------------------------

def _build_scorecard_panel(scorecard, station, year):
    """Collapsible panel showing top features with gate details."""
    if scorecard is None or scorecard.empty:
        return html.Div()

    usable = scorecard[scorecard["usable"] == True]
    if usable.empty:
        return html.Div("No features passed all 5 gates.",
                        style={"color": DARK_TEXT, "padding": "8px"})

    best_4w = usable[usable["window_size"] == "4w"]
    if best_4w.empty:
        best_4w = usable

    best_per_feat = best_4w.groupby("feature").agg(
        max_d=("gate4_cohens_d", lambda x: x.abs().max()),
        n_windows=("gate4_cohens_d", "count"),
        cv=("gate3_cv_baseline", "first"),
    ).sort_values("max_d", ascending=False).head(10).reset_index()

    rows = []
    for _, r in best_per_feat.iterrows():
        feat = r["feature"]
        label = FEATURE_LABELS.get(feat, feat)
        rows.append(html.Tr([
            html.Td(label, style={"color": "#58a6ff", "padding": "4px 12px 4px 0",
                                   "fontSize": "0.85rem"}),
            html.Td(f"|d|={r['max_d']:.2f}", style={"color": "#2d9a6b", "padding": "4px 8px",
                                                      "fontSize": "0.85rem"}),
            html.Td(f"CV={r['cv']:.3f}" if np.isfinite(r['cv']) else "",
                    style={"color": "#8b949e", "padding": "4px 8px", "fontSize": "0.8rem"}),
            html.Td(f"{int(r['n_windows'])} windows",
                    style={"color": "#8b949e", "padding": "4px 0", "fontSize": "0.8rem"}),
        ]))

    fails = scorecard[scorecard["usable"] == False]
    fail_counts = fails["failure_reason"].value_counts()
    fail_summary = " | ".join(f"{reason}: {count}" for reason, count in fail_counts.items())

    return html.Details([
        html.Summary(f"Feature Scorecard: {len(best_per_feat)} top features, "
                     f"{len(usable)} usable windows",
                     style={"cursor": "pointer", "color": "#58a6ff", "fontWeight": "bold",
                            "fontSize": "0.95rem", "padding": "8px 0"}),
        html.Table([html.Tbody(rows)],
                   style={"borderCollapse": "collapse", "width": "100%", "color": DARK_TEXT}),
        html.P(f"Failures: {fail_summary}",
               style={"color": "#8b949e", "fontSize": "0.8rem", "marginTop": "8px"}),
    ], open=True,
       style={"backgroundColor": DARK_CARD, "borderRadius": "6px",
              "border": f"1px solid {DARK_BORDER}", "padding": "8px 16px",
              "marginTop": "8px"})
