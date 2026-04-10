#!/usr/bin/env python3
# ABOUTME: Dash app initialization, layout shell, tab routing, and callbacks
# ABOUTME: Modular entry point — replaces monolithic dashboard_dash.py

"""
GNSS-IR Dashboard — Modular Plotly Dash application.

Usage:
    python -m dashboard.app
    python dashboard/app.py
    python dashboard/app.py --port 8050
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ALL

from dashboard.utils import (
    PLOTLY_DARK, DARK_BG, DARK_CARD, DARK_BORDER, DARK_TEXT,
    FREQ_COLORS, V3_COLORS, V3_ORDER, DISCOVERY_ORDER,
    TAB_STYLE, TAB_SELECTED, SUBTAB_STYLE, SUBTAB_SELECTED,
    get_mode, state_order, non_baseline_states,
)
from dashboard.data_loader import (
    list_available_stations, load_stations_config, load_per_arc, load_v3,
    load_snr_features, load_smap, load_era5, load_v2, load_mahal_threshold,
    load_daily_features, load_prn_weights, list_s1_images, build_quality_stats,
)

# Import figure builders and tab renderers from the legacy module.
# This guarantees pixel-perfect functional equivalence while the code lives
# in a modular structure. Figure builders can be progressively extracted
# into dashboard/components/ as needed.
import dashboard_dash_legacy as _legacy

# Re-export figure builders so tabs can import them
build_polar_figure = _legacy.build_polar_figure
build_timeseries_figure = _legacy.build_timeseries_figure
build_ice_features_figure = _legacy.build_ice_features_figure
build_feature_definitions_panel = _legacy.build_feature_definitions_panel
build_state_timeline = _legacy.build_state_timeline
build_metrics_strip = _legacy.build_metrics_strip
build_feature_direction_summary = _legacy.build_feature_direction_summary
build_discrimination_summary = _legacy.build_discrimination_summary
build_normalization_comparison = _legacy.build_normalization_comparison
build_prn_weight_heatmap = _legacy.build_prn_weight_heatmap
build_investigation_tab = _legacy.build_investigation_tab
build_mahalanobis_figure = _legacy.build_mahalanobis_figure
render_s1_imagery = _legacy.render_s1_imagery

# Import tab renderers
from dashboard.tabs import overview, timeseries, polar, classification, imagery, features


def create_app():
    """Create and configure the Dash application."""
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
            "padding": "10px 20px", "background": DARK_BG, "color": "#e0e0e0",
            "borderBottom": "2px solid #4a90d9",
        }),

        # Tabs
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Time Series", value="timeseries", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Polar", value="polar", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Ice Classification", value="ice", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Imagery", value="imagery", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Features", value="features", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Investigation", value="investigation", style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "2px solid #f0ad4e"}),
        ], style={"marginBottom": "0"}),

        # Tab content with loading spinner
        dcc.Loading(
            id="loading", type="circle", color="#4a90d9",
            children=html.Div(id="tab-content", style={"padding": "8px", "minHeight": "500px"}),
        ),

        # Quality bar
        html.Div(id="quality-bar", style={
            "padding": "10px 20px", "background": DARK_CARD,
            "borderTop": f"1px solid {DARK_BORDER}", "fontSize": "0.9rem",
            "display": "flex", "gap": "20px", "flexWrap": "wrap", "color": "#c9d1d9",
        }),
    ], style={
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "backgroundColor": DARK_BG, "color": "#e0e0e0",
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
        default = 2024 if 2024 in years else years[-1]
        return options, default

    # Overview brush → update polar
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
        import pandas as pd
        if not relayout_data or not per_arc_records:
            return no_update, no_update

        x_start = relayout_data.get("xaxis.range[0]") or relayout_data.get("xaxis.range")
        x_end = relayout_data.get("xaxis.range[1]")

        if x_start is None:
            if "xaxis.autorange" in relayout_data:
                df = pd.DataFrame(per_arc_records)
                if "date_dt" in df.columns:
                    df["date_dt"] = pd.to_datetime(df["date_dt"])
                fig = build_polar_figure(df, title="All Arcs")
                fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30))
                return fig, html.P("Showing all arcs", style={"color": DARK_TEXT})
            return no_update, no_update

        if isinstance(x_start, list):
            x_start = x_start[0]
            x_end = x_start[-1] if len(x_start) > 1 else x_start

        df = pd.DataFrame(per_arc_records)
        if "date_dt" in df.columns:
            df["date_dt"] = pd.to_datetime(df["date_dt"])
            try:
                start_dt = pd.Timestamp(x_start)
                end_dt = pd.Timestamp(x_end)
                mask = (df["date_dt"] >= start_dt) & (df["date_dt"] <= end_dt)
                filtered = df[mask]
            except Exception:
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
            html.P(date_range, style={"color": "#4a90d9", "fontWeight": "bold", "margin": "0"}),
            html.P(f"{n_arcs:,} arcs across {n_days} days", style={"color": DARK_TEXT, "margin": "2px 0"}),
            html.P("Zoom out or double-click to reset", style={"color": "#8b949e", "fontSize": "0.75rem"}),
        ])
        return fig, info

    # Map marker click → update station
    @callback(
        Output("station-select", "value", allow_duplicate=True),
        Input({"type": "station-marker", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def map_click(n_clicks_list):
        from dash import ctx
        if not ctx.triggered_id:
            return no_update
        if not any(n and n > 0 for n in n_clicks_list):
            return no_update
        clicked_station = ctx.triggered_id.get("index")
        return clicked_station if clicked_station else no_update

    # Imagery grid click
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

    # Time series click → nearest S1
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
        clicked_date = None
        custom = point.get("customdata")
        if isinstance(custom, list):
            custom = custom[0] if custom else None
        if custom and isinstance(custom, str) and len(custom) == 10 and custom[4] == "-":
            clicked_date = custom
        else:
            x_val = point.get("x")
            if x_val and isinstance(x_val, str) and len(x_val) >= 10:
                clicked_date = x_val[:10]
        if not clicked_date:
            return no_update

        s1_images = list_s1_images(station)
        if not s1_images:
            return html.P(f"No S1 imagery for {station}", style={"color": "#8b949e", "padding": "8px"})

        from datetime import datetime, timedelta
        try:
            clicked_dt = datetime.strptime(clicked_date, "%Y-%m-%d")
        except ValueError:
            return no_update

        best_match, best_offset = None, 999
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
            return html.P(f"No S1 image within +/-3 days of {clicked_date}",
                          style={"color": "#8b949e", "padding": "8px"})

        v3 = load_v3(station, year)
        result = render_s1_imagery(station, best_match, s1_images, v3)
        if best_offset > 0:
            return html.Div([
                html.P(f"Clicked {clicked_date} -> nearest S1: {best_match} ({best_offset} day offset)",
                       style={"color": "#4a90d9", "fontSize": "0.85rem", "marginBottom": "4px"}),
                result,
            ])
        return result

    # Master tab routing
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

        # Quality bar
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

        # Dispatch to tab modules
        if tab == "overview":
            content = overview.render(station, year, stations, per_arc)
        elif tab == "timeseries":
            content = timeseries.render(station, year, per_arc)
        elif tab == "polar":
            content = polar.render(station, year, stations, per_arc)
        elif tab == "ice":
            content = classification.render(station, year)
        elif tab == "imagery":
            content = imagery.render(station, year)
        elif tab == "features":
            content = features.render()
        elif tab == "investigation":
            content = build_investigation_tab()
        else:
            content = html.P("Unknown tab")

        return content, quality_items

    # Clustering callback
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
        return _legacy_cluster_fallback(threshold, metric, _trigger, station, year)

    # Features sub-tab routing
    @callback(
        Output("features-subtab-content", "children"),
        Output("norm-selector-row", "style"),
        Input("features-subtabs", "value"),
        Input("norm-base-feature", "value"),
        State("station-select", "value"),
        State("year-select", "value"),
    )
    def update_features_subtab(subtab, base_feature, station, year):
        import plotly.graph_objects as go
        norm_hidden = {"display": "none"}
        norm_visible = {"display": "flex", "alignItems": "center", "padding": "8px 0"}

        if not station or not year:
            return html.Div(), norm_hidden

        if subtab == "clustering":
            return features.render_clustering(station, year), norm_hidden
        elif subtab == "discrimination":
            daily_feat = load_daily_features(station, year)
            v3 = load_v3(station, year)
            return build_discrimination_summary(daily_feat, v3, station, year), norm_hidden
        elif subtab == "normalization":
            daily_feat = load_daily_features(station, year)
            v3 = load_v3(station, year)
            return build_normalization_comparison(
                daily_feat, v3, station, year, base_feature), norm_visible
        elif subtab == "prn_weights":
            return build_prn_weight_heatmap(station, year), norm_hidden

        return html.Div(), norm_hidden

    return app


def _legacy_cluster_fallback(threshold, metric, _trigger, station, year):
    """Fallback clustering callback using legacy code."""
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendrogram
    from scipy.spatial.distance import squareform
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    snr = load_snr_features(station, year)
    if snr is None:
        return go.Figure(), "No data"

    feature_cols = [c for c in ["CLR", "PR", "AF", "gamma", "phase", "MS", "VS", "SP"]
                    if c in snr.columns]
    if len(feature_cols) < 3:
        return go.Figure(), "Too few features"

    df = snr[feature_cols].dropna()

    if metric == "partial":
        scaled = StandardScaler().fit_transform(df)
        cov = np.cov(scaled, rowvar=False)
        try:
            prec = np.linalg.inv(cov + np.eye(len(feature_cols)) * 1e-6)
            d = np.sqrt(np.diag(prec))
            pcorr = -prec / np.outer(d, d)
            np.fill_diagonal(pcorr, 1.0)
            corr_matrix = pcorr
        except np.linalg.LinAlgError:
            corr_matrix = df.corr().values
    else:
        corr_matrix = df.corr().values

    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")

    clusters = fcluster(Z, t=threshold, criterion="distance")
    dendro = scipy_dendrogram(Z, labels=feature_cols, no_plot=True)
    order = dendro["leaves"]

    ordered_cols = [feature_cols[i] for i in order]
    ordered_corr = corr_matrix[np.ix_(order, order)]

    from dashboard.utils import FEATURE_LABELS, DARK_TEXT
    labels = [FEATURE_LABELS.get(c, c) for c in ordered_cols]

    fig = go.Figure(data=go.Heatmap(
        z=ordered_corr, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(ordered_corr, 2), texttemplate="%{text}",
        textfont={"size": 9},
    ))
    metric_label = "Partial Correlation" if metric == "partial" else "Pearson Correlation"
    fig.update_layout(
        title=f"Feature {metric_label} (threshold={threshold})",
        height=500, margin=dict(l=120, r=20, t=50, b=120),
        xaxis=dict(tickangle=45),
        **PLOTLY_DARK,
    )

    from dashboard.utils import DARK_BORDER
    cluster_map = {}
    for i, c in enumerate(clusters):
        cluster_map.setdefault(c, []).append(feature_cols[i])

    summary_parts = []
    for cid in sorted(cluster_map):
        members = cluster_map[cid]
        member_labels = [FEATURE_LABELS.get(m, m) for m in members]
        summary_parts.append(
            html.Div([
                html.Span(f"Cluster {cid}: ", style={"fontWeight": "bold", "color": "#58a6ff"}),
                html.Span(", ".join(member_labels), style={"color": DARK_TEXT}),
            ], style={"marginBottom": "4px"})
        )

    return fig, html.Div(summary_parts)


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
