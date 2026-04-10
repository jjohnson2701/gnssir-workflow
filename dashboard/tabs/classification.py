# ABOUTME: Ice Classification tab — timeline + metrics + Mahalanobis + SNR features
# ABOUTME: Renders state timeline, direction summary, Mahalanobis plot, feature panels

"""Classification tab for the GNSS-IR dashboard."""

from dash import html, dcc

from dashboard.data_loader import (
    load_v3, load_v2, load_snr_features, load_era5, load_smap,
    load_mahal_threshold, load_daily_features,
)


def render(station, year):
    """Render the Ice Classification tab content."""
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
