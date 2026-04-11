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

    # Main figure: arc density + Mahal + state strip
    main_fig = _build_time_aligned_figure(scores, daily_feat, scorecard,
                                          baseline_def, station, year)

    # Feature values figure: top features from scorecard, actual values, state-colored
    feat_fig = _build_feature_values_figure(daily_feat, scores, scorecard,
                                            baseline_def, station, year)

    # Scorecard summary
    scorecard_panel = _build_scorecard_panel(scorecard, station, year) if scorecard is not None else html.Div()

    # Feature coverage view: usable features per window + failure breakdown
    coverage_fig = _build_confidence_view(scores, scorecard, baseline_def,
                                          station, year) if scorecard is not None else html.Div()

    # Reliability heatmap (collapsible)
    heatmap = _build_reliability_heatmap(scorecard, station, year) if scorecard is not None else html.Div()

    return html.Div([
        metrics,
        dcc.Graph(figure=main_fig, config={"displayModeBar": True}),
        feat_fig,
        scorecard_panel,
        coverage_fig,
        html.Details([
            html.Summary("Feature Discriminability Heatmap",
                         style={"cursor": "pointer", "color": "#58a6ff",
                                "fontWeight": "bold", "fontSize": "0.9rem",
                                "padding": "8px 0"}),
            heatmap,
        ], style={"marginTop": "4px"}),
    ])


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
    """Build a compact figure: arc density + Mahalanobis distance + state strip.

    All panels share the DOY x-axis for visual correlation.
    """
    n_rows = 3
    row_heights = [0.15, 0.65, 0.08]
    subplot_titles = ["Arc Density", "Anomaly Score (Mahalanobis)", "State"]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    bl_start = baseline_def["start_doy"] if baseline_def else None
    bl_end = baseline_def["end_doy"] if baseline_def else None

    _add_arc_density(fig, scores, daily_feat, row=1, bl_start=bl_start, bl_end=bl_end)
    _add_mahal_panel(fig, scores, row=2, bl_start=bl_start, bl_end=bl_end)
    _add_state_strip(fig, scores, row=3)

    fig.update_xaxes(title_text="Day of Year", row=n_rows, col=1)
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=30, b=30),
        showlegend=False,
        **PLOTLY_DARK,
    )

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
    """Mahalanobis distance scatter, state-colored."""
    if "mahal_distance" not in scores.columns:
        return

    for state in ["baseline", "transition_in", "anomalous", "transition_out"]:
        mask = scores["state"] == state
        if mask.sum() == 0:
            continue
        sub = scores[mask]
        fig.add_trace(go.Scatter(
            x=sub["doy"].values, y=sub["mahal_distance"].values,
            mode="markers", name=state,
            marker=dict(size=5, color=V3_COLORS.get(state, "#999"), opacity=0.8),
            hovertemplate=(f"{state}<br>DOY %{{x}}<br>d=%{{y:.1f}}"
                           f"<br>n_arcs=%{{customdata}}<extra></extra>"),
            customdata=sub["n_arcs"].values if "n_arcs" in sub.columns else None,
            showlegend=True,
        ), row=row, col=1)

    fig.update_yaxes(title_text="Mahal d", row=row, col=1, type="log",
                     tickfont=dict(size=8), title_font=dict(size=9),
                     dtick=1,  # log scale: 1 = 10^1 steps (1, 10, 100)
                     minor=dict(showgrid=False))

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


def _build_feature_values_figure(daily_feat, scores, scorecard, baseline_def,
                                  station, year):
    """Multi-panel figure showing top feature VALUES colored by state.

    Like the old Regime Detection Features, but using scorecard-validated
    best features. Shows actual physical values (gamma, CLR, AF, etc.)
    so the user can see what changed and by how much.
    """
    if daily_feat is None or scores is None:
        return html.Div("No daily features available.",
                        style={"color": DARK_TEXT, "padding": "8px"})

    # Get pooled daily features
    if "azimuth_bin" in daily_feat.columns:
        pooled = daily_feat[daily_feat["azimuth_bin"] == -1].copy()
    else:
        pooled = daily_feat.copy()

    if "doy" not in pooled.columns:
        return html.Div()

    # Merge state labels onto daily features
    state_map = dict(zip(scores["doy"], scores["state"]))
    pooled["state"] = pooled["doy"].map(state_map)

    # Select top features to display
    top_features = _select_top_features(scorecard, pooled)
    if not top_features:
        # Fallback: show standard features
        top_features = [f for f in ["gamma_med", "clr_med", "af_med", "vs_med",
                                     "rh_std", "amp_mean"]
                        if f in pooled.columns][:5]

    n_panels = len(top_features)
    if n_panels == 0:
        return html.Div()

    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03)

    bl_start = baseline_def["start_doy"] if baseline_def else None
    bl_end = baseline_def["end_doy"] if baseline_def else None

    state_order_list = ["baseline", "transition_in", "anomalous", "transition_out"]

    for i, feat in enumerate(top_features):
        row = i + 1

        # Compute baseline mean for reference line
        if bl_start and bl_end:
            bl_vals = pooled[(pooled["doy"] >= bl_start) & (pooled["doy"] <= bl_end)][feat].dropna()
            bl_mean = bl_vals.mean() if len(bl_vals) > 0 else None
        else:
            bl_mean = None

        # Plot points colored by state
        for state in state_order_list:
            mask = pooled["state"] == state
            if mask.sum() == 0:
                continue
            sub = pooled[mask]
            fig.add_trace(go.Scatter(
                x=sub["doy"].values, y=sub[feat].values,
                mode="markers", name=state,
                marker=dict(size=4, color=V3_COLORS.get(state, "#999"), opacity=0.7),
                showlegend=(row == 1),
                legendgroup=state,
                hovertemplate=f"{state}<br>DOY %{{x}}<br>{feat}=%{{y:.3f}}<extra></extra>",
            ), row=row, col=1)

        # Baseline mean reference line
        if bl_mean is not None:
            fig.add_hline(y=bl_mean, line_dash="dash", line_color="#58a6ff",
                          line_width=1, opacity=0.5, row=row, col=1)

        # Baseline shading
        if bl_start and bl_end:
            fig.add_vrect(x0=bl_start - 0.5, x1=bl_end + 0.5, row=row, col=1,
                          fillcolor="rgba(74,144,217,0.08)", line_width=0)

        label = FEATURE_LABELS.get(feat, feat)
        fig.update_yaxes(title_text=label, row=row, col=1,
                         tickfont=dict(size=8), title_font=dict(size=9))

    fig.update_xaxes(title_text="Day of Year", row=n_panels, col=1)
    fig.update_layout(
        title=f"{station} {year}: Top Features (scorecard-validated)",
        height=max(400, 130 * n_panels),
        margin=dict(l=80, r=20, t=40, b=30),
        legend=dict(orientation="h", y=-0.05, font=dict(size=9)),
        **PLOTLY_DARK,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True})


def _select_top_features(scorecard, daily_feat, n=5):
    """Select top N features from scorecard that exist in daily_feat.

    Picks features with highest peak |d| across any window, excluding
    redundant pairs (keeps the stronger one).
    """
    if scorecard is None or scorecard.empty:
        return []

    usable = scorecard[scorecard["usable"] == True]
    if usable.empty:
        return []

    # Best |d| per feature
    best = usable.groupby("feature")["gate4_cohens_d"].apply(
        lambda x: x.abs().max()
    ).sort_values(ascending=False)

    # Filter to features present in daily_feat
    available = set(daily_feat.columns)
    selected = []
    for feat in best.index:
        if feat in available and len(selected) < n:
            selected.append(feat)

    return selected


# ---------------------------------------------------------------------------
# Failure timeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Confidence-filtered view
# ---------------------------------------------------------------------------

def _build_confidence_view(scores, scorecard, baseline_def, station, year):
    """Intersection view: arc density adequacy × feature gate reliability, per window.

    Top panel: arc density (arcs/day) for each 4-week window vs the baseline rate.
               Bars are green when window arc count >= baseline rate, red when below.
               Dotted line shows baseline arc density.

    Bottom panel: stacked feature counts per window split into three categories:
      - Usable (green): passed all 5 gates — arc density OK + feature quality OK
      - Arc-limited (red): failed gate 1 only — would be usable if arc count were higher
      - Quality-limited (yellow/gray stacked): passed gate 1 but failed gates 2-5

    The green usable stack is the intersection of both criteria.
    """
    if scores is None or scorecard is None or scorecard.empty:
        return html.Div()

    sc = scorecard[scorecard["window_size"] == "4w"].copy()
    if sc.empty:
        return html.Div()

    # Per-window aggregation
    # gate1_n_arcs_window and gate1_n_arcs_baseline are the same for all features
    # in a given window, so we take the first value per window.
    windows = sorted(sc["window_start_doy"].unique())

    # Baseline arc rate: mean across all windows' baseline arcs
    baseline_arc_rate = sc["gate1_n_arcs_baseline"].mean()

    w_data = []
    for w in windows:
        wsc = sc[sc["window_start_doy"] == w]
        arc_window = wsc["gate1_n_arcs_window"].mean()  # same for all features

        n_usable = int(wsc["usable"].sum())
        fails = wsc[~wsc["usable"]]

        # Arc-limited: failed gate1 only (removing gate1, they'd be usable)
        # Proxy: failure_reason == "insufficient_data"
        n_arc_limited = int((fails["failure_reason"] == "insufficient_data").sum())

        # Quality-limited: passed gate1 but failed gates 2-5
        # i.e. gate1_pass=True but usable=False
        passed_gate1_but_failed = fails[fails["gate1_pass"] == True]
        reason_counts = passed_gate1_but_failed["failure_reason"].value_counts().to_dict()

        w_data.append({
            "window_mid": w + 14,
            "arc_window": arc_window,
            "arc_ok": arc_window >= baseline_arc_rate * 0.7,  # >70% of baseline
            "n_usable": n_usable,
            "n_arc_limited": n_arc_limited,
            "n_untrustworthy": reason_counts.get("untrustworthy_computation", 0),
            "n_unstable": reason_counts.get("unstable_baseline", 0),
            "n_nondiscriminant": reason_counts.get("non_discriminant", 0),
            "n_redundant": reason_counts.get("redundant", 0),
        })

    wdf = pd.DataFrame(w_data)
    total_features = sc["feature"].nunique()
    trusted_mask = (wdf["n_usable"] >= max(1, total_features * 0.25)) & wdf["arc_ok"]
    n_trusted = int(trusted_mask.sum())

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.38, 0.62], vertical_spacing=0.06,
        subplot_titles=["Arc Density per Window vs Baseline",
                        "Feature Reliability (arc-limited vs quality-limited vs usable)"],
    )

    # ── Panel 1: arc density bars ──────────────────────────────────────────
    arc_colors = ["#2d9a6b" if ok else "#d94452" for ok in wdf["arc_ok"]]
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["arc_window"].round(1),
        marker_color=arc_colors, name="Arc density",
        hovertemplate="DOY %{x}<br>%{y:.1f} arcs/day<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Baseline reference line
    fig.add_hline(
        y=baseline_arc_rate, line_dash="dot",
        line_color="rgba(74,144,217,0.7)", line_width=1.5,
        annotation_text=f"baseline avg ({baseline_arc_rate:.1f})",
        annotation_font_size=8, annotation_font_color="#8b949e",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="arcs/day", row=1, col=1,
                     tickfont=dict(size=8), title_font=dict(size=9))

    # ── Panel 2: stacked feature breakdown ────────────────────────────────
    # Order: arc-limited (bottom, red) → quality failures (middle) → usable (top, green)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_arc_limited"],
        name="Arc-limited (gate 1 fail)", marker_color=_FAILURE_COLORS["insufficient_data"],
        hovertemplate="DOY %{x}<br>%{y} features: insufficient arc density<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_untrustworthy"],
        name="Computation health fail", marker_color=_FAILURE_COLORS["untrustworthy_computation"],
        hovertemplate="DOY %{x}<br>%{y} features: untrustworthy computation<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_unstable"],
        name="Unstable baseline", marker_color=_FAILURE_COLORS["unstable_baseline"],
        hovertemplate="DOY %{x}<br>%{y} features: unstable baseline<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_nondiscriminant"],
        name="Non-discriminant", marker_color=_FAILURE_COLORS["non_discriminant"],
        hovertemplate="DOY %{x}<br>%{y} features: no signal vs baseline<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_redundant"],
        name="Redundant", marker_color=_FAILURE_COLORS["redundant"],
        hovertemplate="DOY %{x}<br>%{y} features: redundant with stronger feature<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=wdf["window_mid"], y=wdf["n_usable"],
        name="Usable (all gates pass)", marker_color="#2d9a6b",
        hovertemplate="DOY %{x}<br>%{y} features: arc density OK + feature quality OK<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="feature count", row=2, col=1,
                     tickfont=dict(size=8), title_font=dict(size=9))

    # Baseline shading both panels
    bl_start = baseline_def.get("start_doy") if baseline_def else None
    bl_end = baseline_def.get("end_doy") if baseline_def else None
    if bl_start and bl_end:
        for r in [1, 2]:
            fig.add_vrect(x0=bl_start - 0.5, x1=bl_end + 0.5, row=r, col=1,
                          fillcolor="rgba(74,144,217,0.10)", line_width=0)

    # Mark trusted windows (arc ok + usable features) in panel 2 with a small marker
    trusted_doys = wdf.loc[trusted_mask, "window_mid"].tolist()
    if trusted_doys:
        max_stack = (wdf["n_arc_limited"] + wdf["n_untrustworthy"] + wdf["n_unstable"] +
                     wdf["n_nondiscriminant"] + wdf["n_redundant"] + wdf["n_usable"]).max()
        fig.add_trace(go.Scatter(
            x=trusted_doys,
            y=[max_stack * 1.08] * len(trusted_doys),
            mode="markers", marker=dict(symbol="triangle-down", size=7, color="#58a6ff"),
            name="Trusted window", showlegend=True,
            hovertemplate="DOY %{x}: arc density OK + ≥25% features usable<extra></extra>",
        ), row=2, col=1)

    fig.update_xaxes(title_text="Window Center DOY", row=2, col=1,
                     tickfont=dict(size=8))
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=40, b=30),
        barmode="stack",
        legend=dict(orientation="h", y=-0.18, font=dict(size=8)),
        **PLOTLY_DARK,
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=10, color="#8b949e"), x=0.01, xanchor="left")

    # Summary sentence
    n_arc_fail_windows = int((~wdf["arc_ok"]).sum())
    n_windows = len(wdf)
    summary = (
        f"{n_trusted}/{n_windows} windows are trusted (arc density ≥70% of baseline "
        f"AND ≥25% of features usable). "
        f"{n_arc_fail_windows} window(s) fail the arc-density threshold (red bars, top). "
        f"Green stack = intersection of both criteria."
    )

    return html.Div([
        html.P(summary, style={"color": "#8b949e", "fontSize": "0.8rem", "margin": "4px 0"}),
        dcc.Graph(figure=fig, config={"displayModeBar": True}),
    ], style={"marginTop": "8px"})


_FAILURE_COLORS = {
    "insufficient_data": "#d94452",       # red — not enough observations
    "untrustworthy_computation": "#e07b39",  # orange — health check failed
    "unstable_baseline": "#f0ad4e",       # yellow — noisy during calm period
    "non_discriminant": "#8b949e",        # gray — no signal
    "redundant": "#6e7681",              # dim gray — correlated with better feature
}


def _build_failure_timeline(scorecard, station, year):
    """Stacked area chart showing failure counts by reason across DOY windows.

    Shows WHERE features fail: are failures concentrated in winter
    (data gaps) or distributed across the year (feature limitations)?
    """
    if scorecard is None or scorecard.empty:
        return html.Div()

    sc = scorecard[scorecard["window_size"] == "4w"].copy()
    if sc.empty:
        return html.Div()

    fails = sc[sc["usable"] == False]
    if fails.empty:
        return html.Div()

    # Count failures per window per reason
    windows = sorted(sc["window_start_doy"].unique())
    reasons = ["insufficient_data", "untrustworthy_computation",
               "unstable_baseline", "non_discriminant", "redundant"]

    fig = go.Figure()

    for reason in reasons:
        counts = []
        for w in windows:
            n = len(fails[(fails["window_start_doy"] == w) & (fails["failure_reason"] == reason)])
            counts.append(n)

        label = reason.replace("_", " ").title()
        fig.add_trace(go.Scatter(
            x=windows, y=counts, name=label,
            mode="lines", stackgroup="failures",
            line=dict(width=0.5, color=_FAILURE_COLORS.get(reason, "#666")),
            fillcolor=_FAILURE_COLORS.get(reason, "#666"),
            hovertemplate=f"{label}<br>DOY %{{x}}<br>%{{y}} features<extra></extra>",
        ))

    fig.update_layout(
        title=f"{station} {year}: Gate Failures by Window (why features are excluded)",
        xaxis_title="Window Start DOY",
        yaxis_title="Features failing",
        height=250,
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        **PLOTLY_DARK,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                     style={"marginTop": "8px"})


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

    return html.Div([
        html.P("Cohen's d measures how strongly a feature separates baseline from each "
               "evaluation window. Brighter green = stronger separation (|d| > 0.8 is large). "
               "Dark cells = feature failed a validation gate in that window.",
               style={"color": "#8b949e", "fontSize": "0.8rem", "margin": "4px 0"}),
        dcc.Graph(figure=fig, config={"displayModeBar": True}),
    ], style={"marginTop": "4px"})


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
    """Feature scorecard: top features table + per-window reliability dot chart.

    Shows which features are most discriminating (by max Cohen's d across windows)
    and a small heatmap-style dot chart showing per-window pass/fail for each top
    feature — so you can immediately see WHICH DOY windows have reliable features.
    """
    if scorecard is None or scorecard.empty:
        return html.Div()

    sc4 = scorecard[scorecard["window_size"] == "4w"].copy()
    usable = sc4[sc4["usable"] == True]
    if usable.empty:
        return html.Div("No features passed all 5 gates.",
                        style={"color": DARK_TEXT, "padding": "8px"})

    # Top features ranked by max |Cohen's d|
    best_per_feat = usable.groupby("feature").agg(
        max_d=("gate4_cohens_d", lambda x: x.abs().max()),
        n_windows=("gate4_cohens_d", "count"),
        cv=("gate3_cv_baseline", "first"),
    ).sort_values("max_d", ascending=False).head(8).reset_index()

    top_features = best_per_feat["feature"].tolist()
    windows = sorted(sc4["window_start_doy"].unique())

    # Build per-window reliability figure for top features
    # Each cell: green = usable, colored by failure reason if not usable
    fig = go.Figure()

    # Arc density row (top row in chart)
    baseline_arc_rate = sc4["gate1_n_arcs_baseline"].mean()
    arc_per_window = {w: sc4[sc4["window_start_doy"] == w]["gate1_n_arcs_window"].mean()
                      for w in windows}
    arc_ok = {w: arc_per_window[w] >= baseline_arc_rate * 0.7 for w in windows}

    arc_colors_dot = ["#2d9a6b" if arc_ok[w] else "#d94452" for w in windows]
    arc_text = [f"{arc_per_window[w]:.1f} arcs/day" for w in windows]
    fig.add_trace(go.Scatter(
        x=[w + 14 for w in windows],
        y=["Arc density"] * len(windows),
        mode="markers+text",
        marker=dict(
            color=arc_colors_dot,
            size=10,
            symbol=["circle" if arc_ok[w] else "x" for w in windows],
            line=dict(width=0.5, color="#30363d"),
        ),
        text=["✓" if arc_ok[w] else "✗" for w in windows],
        textfont=dict(size=7, color="#ffffff"),
        textposition="middle center",
        hovertext=[f"DOY {w+14}: {t}" for w, t in zip(windows, arc_text)],
        hoverinfo="text",
        showlegend=False,
        name="Arc density",
    ))

    # Feature rows
    _reason_dot_color = {
        "insufficient_data": "#d94452",
        "untrustworthy_computation": "#e07b39",
        "unstable_baseline": "#f0ad4e",
        "non_discriminant": "#6e7681",
        "redundant": "#444c56",
        "": "#2d9a6b",
    }

    for feat in top_features:
        label = FEATURE_LABELS.get(feat, feat)
        feat_sc = sc4[sc4["feature"] == feat].set_index("window_start_doy")

        colors, symbols, hover, texts = [], [], [], []
        for w in windows:
            if w in feat_sc.index:
                row = feat_sc.loc[w]
                reason = row.get("failure_reason", "")
                if isinstance(reason, pd.Series):
                    reason = reason.iloc[0]
                is_usable = bool(row["usable"]) if not isinstance(row["usable"], pd.Series) \
                    else bool(row["usable"].iloc[0])
                d_val = row["gate4_cohens_d"] if not isinstance(row["gate4_cohens_d"], pd.Series) \
                    else row["gate4_cohens_d"].iloc[0]
                color = "#2d9a6b" if is_usable else _reason_dot_color.get(reason, "#444")
                sym = "circle" if is_usable else "x"
                label_text = f"|d|={abs(d_val):.2f}" if is_usable else (reason or "fail")
                h = f"DOY {w+14}: {'USABLE |d|='+str(round(abs(d_val),2)) if is_usable else reason}"
            else:
                color, sym, label_text, h = "#1c2128", "circle", "", f"DOY {w+14}: no data"
            colors.append(color)
            symbols.append(sym)
            hover.append(h)
            texts.append(label_text)

        fig.add_trace(go.Scatter(
            x=[w + 14 for w in windows],
            y=[label] * len(windows),
            mode="markers",
            marker=dict(color=colors, size=10, symbol=symbols,
                        line=dict(width=0.5, color="#30363d")),
            hovertext=hover, hoverinfo="text",
            showlegend=False,
            name=label,
        ))

    fig.update_layout(
        height=max(220, (len(top_features) + 2) * 30 + 60),
        margin=dict(l=140, r=20, t=30, b=40),
        xaxis=dict(title="Window Center DOY", tickfont=dict(size=8)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
        **PLOTLY_DARK,
    )

    # Summary table (compact, same order)
    rows = []
    for _, r in best_per_feat.iterrows():
        feat = r["feature"]
        label = FEATURE_LABELS.get(feat, feat)
        rows.append(html.Tr([
            html.Td(label, style={"color": "#58a6ff", "padding": "3px 10px 3px 0",
                                   "fontSize": "0.82rem"}),
            html.Td(f"|d|={r['max_d']:.2f}", style={"color": "#2d9a6b", "padding": "3px 8px",
                                                      "fontSize": "0.82rem"}),
            html.Td(f"CV={r['cv']:.3f}" if np.isfinite(r["cv"]) else "—",
                    style={"color": "#8b949e", "padding": "3px 8px", "fontSize": "0.78rem"}),
            html.Td(f"{int(r['n_windows'])} windows",
                    style={"color": "#8b949e", "padding": "3px 0", "fontSize": "0.78rem"}),
        ]))

    fails = sc4[sc4["usable"] == False]
    fail_counts = fails["failure_reason"].value_counts()
    fail_summary = " | ".join(f"{r}: {c}" for r, c in fail_counts.items())

    return html.Details([
        html.Summary(
            f"Feature Scorecard — top {len(best_per_feat)} features "
            f"({int(usable['window_start_doy'].nunique())} windows with usable features)",
            style={"cursor": "pointer", "color": "#58a6ff", "fontWeight": "bold",
                   "fontSize": "0.9rem", "padding": "8px 0"},
        ),
        html.P(
            "Each dot = one 4-week window. Green circle = passed all gates (arc density ok + "
            "feature discriminates). Red ✗ = arc-density failure. Colored ✗ = quality gate failure.",
            style={"color": "#8b949e", "fontSize": "0.78rem", "margin": "4px 0 6px 0"},
        ),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Table([html.Tbody(rows)],
                   style={"borderCollapse": "collapse", "width": "100%", "marginTop": "8px"}),
        html.P(f"Gate failures: {fail_summary}",
               style={"color": "#8b949e", "fontSize": "0.78rem", "marginTop": "6px"}),
    ], open=True,
       style={"backgroundColor": DARK_CARD, "borderRadius": "6px",
              "border": f"1px solid {DARK_BORDER}", "padding": "8px 16px",
              "marginTop": "8px"})
