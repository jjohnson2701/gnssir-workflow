# ABOUTME: Features tab — definitions panel + 4 sub-tabs (clustering, discrimination, normalization, PRN)
# ABOUTME: Layout and clustering sub-tab rendering

"""Features tab for the GNSS-IR dashboard."""

from dash import html, dcc

from dashboard.utils import SUBTAB_STYLE, SUBTAB_SELECTED, DARK_TEXT


def render():
    """Render the Features tab content (layout only — sub-tabs handled by callback)."""
    from dashboard.app import build_feature_definitions_panel

    return html.Div([
        build_feature_definitions_panel(),
        dcc.Tabs(id="features-subtabs", value="clustering", children=[
            dcc.Tab(label="Feature Clustering", value="clustering",
                    style=SUBTAB_STYLE, selected_style=SUBTAB_SELECTED),
            dcc.Tab(label="Discrimination Summary", value="discrimination",
                    style=SUBTAB_STYLE, selected_style=SUBTAB_SELECTED),
            dcc.Tab(label="Normalization Comparison", value="normalization",
                    style=SUBTAB_STYLE, selected_style=SUBTAB_SELECTED),
            dcc.Tab(label="PRN Weights", value="prn_weights",
                    style=SUBTAB_STYLE, selected_style=SUBTAB_SELECTED),
        ]),
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


def render_clustering(station, year):
    """Render the clustering sub-tab content (initial state)."""
    return html.Div([
        html.Div([
            html.Label("Cluster threshold:", style={"color": DARK_TEXT, "marginRight": "8px"}),
            dcc.Slider(id="cluster-threshold", min=0.1, max=0.9, step=0.05, value=0.4,
                       marks={i/10: str(i/10) for i in range(1, 10)},
                       tooltip={"placement": "bottom"}),
        ], style={"width": "400px", "display": "inline-block", "marginRight": "30px"}),
        html.Div([
            dcc.RadioItems(
                id="cluster-metric",
                options=[
                    {"label": " Correlation", "value": "correlation"},
                    {"label": " Partial Correlation", "value": "partial"},
                ],
                value="correlation",
                inline=True,
                style={"color": DARK_TEXT},
                labelStyle={"marginRight": "16px"},
            ),
        ], style={"display": "inline-block"}),
        dcc.Graph(id="cluster-heatmap", style={"height": "500px"}),
        html.Div(id="cluster-summary", style={"padding": "8px"}),
        dcc.Store(id="cluster-init-trigger", data=True),
    ])
