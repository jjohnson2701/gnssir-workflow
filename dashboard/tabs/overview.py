# ABOUTME: Overview tab — map + polar + time series with brush interaction
# ABOUTME: Renders Fresnel zones, station markers, reference stations, v3 summary

"""Overview tab for the GNSS-IR dashboard."""

import json

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
import dash_leaflet as dl

from dashboard.utils import (
    PROJECT_ROOT, DARK_BG, DARK_CARD, DARK_BORDER, DARK_TEXT,
    PLOTLY_DARK, V3_COLORS, V3_ORDER, get_mode,
)
from dashboard.data_loader import (
    load_v3, load_stations_config, list_s1_images,
    load_baseline_definition,
)


def render(station, year, stations, per_arc):
    """Render the Overview tab content."""
    from dashboard.app import build_polar_figure, build_timeseries_figure

    v3 = load_v3(station, year)

    center = [stations[station]["lat"], stations[station]["lon"]] if station in stations else [45, -80]
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
                    html.Span("| ", style={"color": color, "fontSize": "1.3em"}),
                    f"{s}: {c} days ({c/total*100:.0f}%)",
                ], style={"marginBottom": "4px"}))

    # Polar diagram
    polar_fig = build_polar_figure(per_arc, title="Reflection Points")
    polar_fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30))

    # Time series with S1 markers
    s1_images = list_s1_images(station)
    year_s1 = [img for img in s1_images if img["date"].startswith(str(year))]
    ts_fig = build_timeseries_figure(per_arc, v3, station, year, s1_dates=year_s1)

    # Baseline shading on RH panel
    baseline_def = load_baseline_definition(station, year)
    if baseline_def:
        import pandas as pd
        bl_start = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=baseline_def["start_doy"] - 1)
        bl_end = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=baseline_def["end_doy"] - 1)
        ts_fig.add_vrect(
            x0=bl_start, x1=bl_end, row=1, col=1,
            fillcolor="rgba(74,144,217,0.12)", line_width=0,
            annotation_text="baseline", annotation_position="top left",
            annotation_font=dict(size=9, color="rgba(74,144,217,0.8)"),
        )

    # Fresnel zone overlay
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
    except Exception:
        pass

    slat = stations[station]["lat"] if station in stations else 0
    slon = stations[station]["lon"] if station in stations else 0
    RH = 5.0
    deg_per_m_lat = 1.0 / 111000
    deg_per_m_lon = 1.0 / (111000 * np.cos(np.radians(slat)))

    for elev_deg, color in [(5, "cyan"), (10, "lime"), (15, "yellow")]:
        d = RH / np.tan(np.radians(elev_deg))
        arc_pts = []
        for az in np.arange(az_start, az_end + 1, 2):
            az_rad = np.radians(az)
            arc_pts.append([
                slat + d * np.cos(az_rad) * deg_per_m_lat,
                slon + d * np.sin(az_rad) * deg_per_m_lon,
            ])
        if arc_pts:
            fresnel_layers.append(dl.Polyline(
                positions=arc_pts, color=color, weight=2,
                dashArray="5,5" if elev_deg > 5 else "",
            ))

    for az_deg in [az_start, az_end]:
        d_max = RH / np.tan(np.radians(5))
        az_rad = np.radians(az_deg)
        fresnel_layers.append(dl.Polyline(
            positions=[[slat, slon], [
                slat + d_max * np.cos(az_rad) * deg_per_m_lat,
                slon + d_max * np.sin(az_rad) * deg_per_m_lon,
            ]], color="white", weight=1, opacity=0.7,
        ))

    # Reference station markers and legend
    ref_markers, ref_legend = _build_reference_markers(station_cfg)

    # Regional context inset
    context_map = _build_context_map(slat, slon, station)

    return html.Div([
        # Top row: Map (context inset overlaid) + Polar + Info
        html.Div([
            # Map container — position:relative so the inset can be absolutely pinned
            html.Div([
                dl.Map([
                    dl.TileLayer(
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attribution="",
                    ),
                ] + [
                    dl.Marker(
                        position=[info["lat"], info["lon"]],
                        id={"type": "station-marker", "index": name},
                        children=[dl.Tooltip(name)],
                    ) for name, info in stations.items()
                ] + fresnel_layers + ref_markers, center=center, zoom=map_zoom,
                   id=f"map-{station}-{year}",
                   attributionControl=False,
                   style={"height": "100%", "minHeight": "400px", "borderRadius": "6px", "width": "100%"}),
                # Inset context map — absolutely positioned in bottom-left corner
                # resize:both gives a native drag handle; overflow:hidden clips content
                html.Div(
                    context_map,
                    style={
                        "position": "absolute", "bottom": "10px", "left": "10px",
                        "width": "170px", "height": "180px",
                        "minWidth": "100px", "minHeight": "100px",
                        "zIndex": "1000",
                        "borderRadius": "6px", "overflow": "hidden",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.6)",
                        "resize": "both",
                    },
                ),
            ], style={"flex": "2", "display": "flex", "flexDirection": "column",
                      "position": "relative"}),

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
        ], style={"display": "flex", "gap": "12px", "width": "100%", "alignItems": "stretch"}),

        # Time series
        html.Div([
            dcc.Graph(id="overview-ts", figure=ts_fig, style={"height": "350px"},
                      config={"modeBarButtonsToAdd": ["select2d"]}),
        ], style={"marginTop": "8px"}),

        # Hidden store for brush interaction
        dcc.Store(id="overview-per-arc-store",
                  data=per_arc[["doy", "date_dt", "Azim", "RH", "Amp", "freq_group",
                                "eminO", "emaxO"]].to_dict("records")
                  if per_arc is not None and not per_arc.empty else []),
    ])


def _build_reference_markers(station_cfg):
    """Build reference station markers and legend entries."""
    import dash_leaflet as dl

    ref_markers = []
    ref_legend = []
    ext = station_cfg.get("external_data_sources", {})

    hydat_cfg = ext.get("hydat", {})
    if hydat_cfg.get("enabled") and hydat_cfg.get("station_id"):
        h_name = hydat_cfg.get("station_name", hydat_cfg["station_id"])
        h_dist = hydat_cfg.get("distance_km", "?")
        ref_legend.append(html.Div([
            html.Span("* ", style={"color": "#4FC3F7", "fontSize": "1.2em"}),
            f"HYDAT: {h_name} ({h_dist} km)",
        ], style={"fontSize": "0.75rem", "color": "#ccc"}))

    coops_cfg = ext.get("noaa_coops", {})
    if coops_cfg.get("enabled") and coops_cfg.get("nearest_station"):
        ns = coops_cfg["nearest_station"]
        c_lat, c_lon = ns.get("latitude"), ns.get("longitude")
        c_name = ns.get("name", ns.get("id", ""))
        c_dist = ns.get("distance_km", "?")
        if c_lat and c_lon:
            ref_markers.append(dl.CircleMarker(
                center=[c_lat, c_lon], radius=7,
                color="#FF69B4", fillColor="#FF69B4", fillOpacity=0.8,
                children=[dl.Tooltip(f"CO-OPS: {c_name} ({c_dist} km)")],
            ))
        ref_legend.append(html.Div([
            html.Span("* ", style={"color": "#FF69B4", "fontSize": "1.2em"}),
            f"CO-OPS: {c_name} ({c_dist} km)",
        ], style={"fontSize": "0.75rem", "color": "#ccc"}))

    ec_cfg = ext.get("ec_climate", {})
    if ec_cfg.get("enabled") and ec_cfg.get("station_id"):
        e_lat, e_lon = ec_cfg.get("latitude"), ec_cfg.get("longitude")
        e_name = ec_cfg.get("station_name", "EC Climate")
        e_dist = ec_cfg.get("distance_km", "?")
        if e_lat and e_lon and (not isinstance(e_dist, (int, float)) or e_dist > 5):
            ref_markers.append(dl.CircleMarker(
                center=[e_lat, e_lon], radius=7,
                color="#FFA726", fillColor="#FFA726", fillOpacity=0.8,
                children=[dl.Tooltip(f"Temp: {e_name} ({e_dist} km)")],
            ))
        ref_legend.append(html.Div([
            html.Span("* ", style={"color": "#FFA726", "fontSize": "1.2em"}),
            f"Temp: {e_name} ({e_dist} km)",
        ], style={"fontSize": "0.75rem", "color": "#ccc"}))

    return ref_markers, ref_legend


def _build_context_map(lat, lon, station):
    """Small scattergeo panel showing regional location of the station.

    Uses orthographic projection for polar sites (|lat| > 55), natural earth
    for everything else. A star marker pinpoints the station; the map is
    zoomed to roughly a 2000 km radius.
    """
    is_polar = abs(lat) > 55

    if is_polar:
        projection = dict(type="orthographic", rotation=dict(lon=lon, lat=lat, roll=0))
    else:
        projection = dict(type="natural earth")

    # Approximate degree range for ~2000 km context window
    lat_range = [max(-90, lat - 18), min(90, lat + 18)]
    lon_range = [lon - 25, lon + 25]

    fig = go.Figure()

    # Landmass fill
    fig.add_trace(go.Scattergeo(
        lat=[lat], lon=[lon],
        mode="markers+text",
        marker=dict(size=10, color="#ff6b6b", symbol="star",
                    line=dict(color="white", width=1)),
        text=[station],
        textposition="bottom center",
        textfont=dict(size=9, color="white"),
        showlegend=False,
        hovertemplate=f"{station}<br>{lat:.2f}°, {lon:.2f}°<extra></extra>",
    ))

    geo_kwargs = dict(
        showland=True, landcolor="#2d3748",
        showocean=True, oceancolor="#1a2535",
        showcoastlines=True, coastlinecolor="#4a5568", coastlinewidth=0.8,
        showlakes=True, lakecolor="#1a2535",
        showframe=False,
        bgcolor="#0d1117",
        projection=projection,
    )
    if not is_polar:
        geo_kwargs["lataxis_range"] = lat_range
        geo_kwargs["lonaxis_range"] = lon_range

    fig.update_geos(**geo_kwargs)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        autosize=True,
        paper_bgcolor="#0d1117",
        title=dict(text="Regional context", font=dict(size=9, color="#8b949e"),
                   x=0.5, xanchor="center", y=0.98),
    )

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "scrollZoom": False},
        responsive=True,
        style={"height": "100%", "width": "100%"},
    )
