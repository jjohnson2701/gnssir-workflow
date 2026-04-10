# ABOUTME: Imagery tab — S1 date grid + thumbnail viewer
# ABOUTME: Click dates to view S1 VV/VH with Fresnel zone overlay

"""Imagery tab for the GNSS-IR dashboard."""

from dash import html

from dashboard.utils import DARK_CARD, DARK_TEXT, V3_COLORS
from dashboard.data_loader import load_v3, list_s1_images


def render(station, year):
    """Render the Imagery tab content."""
    from dashboard.app import render_s1_imagery

    s1_images = list_s1_images(station)
    v3 = load_v3(station, year)

    if not s1_images:
        return html.Div([
            html.P(f"No S1 imagery downloaded for {station}.",
                   style={"color": DARK_TEXT, "fontSize": "1.1rem"}),
            html.Code(f"python scripts/s1_fresnel_search.py --station {station} && "
                      f"python scripts/s1_fresnel_download.py --station {station}",
                      style={"color": "#4a90d9", "fontSize": "0.9rem"}),
        ], style={"padding": "20px"})

    year_images = [img for img in s1_images if img["date"].startswith(str(year))]
    if not year_images:
        year_images = s1_images

    v3_map = {}
    if v3 is not None and "v3_state" in v3.columns:
        v3_map = dict(zip(v3["date_dt"].dt.strftime("%Y-%m-%d"), v3["v3_state"]))

    grid_items = []
    for img in year_images:
        state = v3_map.get(img["date"], "?")
        state_color = V3_COLORS.get(state, "#666")
        grid_items.append(
            html.Div([
                html.Div(img["date"][-5:], style={"fontSize": "0.75rem", "color": DARK_TEXT}),
                html.Div(state, style={"fontSize": "0.65rem", "color": state_color, "fontWeight": "bold"}),
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

    first_date = year_images[0]["date"] if year_images else None
    initial_imagery = render_s1_imagery(station, first_date, s1_images, v3) if first_date else html.P("No images")

    return html.Div([
        html.Div([
            html.Label("Click a date to view:", style={"color": DARK_TEXT, "fontWeight": "bold",
                                                        "marginRight": "12px", "whiteSpace": "nowrap"}),
            html.Div(grid_items, style={"display": "flex", "gap": "4px", "flexWrap": "wrap", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px",
                  "overflowX": "auto", "padding": "4px"}),
        html.Div(id="imagery-display", children=initial_imagery),
    ])
