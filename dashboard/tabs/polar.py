# ABOUTME: Polar tab — reflection points + station metadata table
# ABOUTME: Full polar diagram with data badges and configuration summary

"""Polar tab for the GNSS-IR dashboard."""

import json
import glob as _glob

from dash import html, dcc

from dashboard.utils import PROJECT_ROOT, DARK_CARD, DARK_BORDER, DARK_TEXT
from dashboard.data_loader import load_stations_config


def render(station, year, stations, per_arc):
    """Render the Polar tab content."""
    from dashboard.app import build_polar_figure

    fig = build_polar_figure(per_arc, title=f"{station} {year} — Reflection Points")
    fig.update_layout(height=550)

    sinfo = stations.get(station, {})
    n_arcs = len(per_arc) if per_arc is not None else 0
    freq_list = sorted(per_arc["freq_group"].unique()) if per_arc is not None and "freq_group" in per_arc.columns else []
    az_range = f"{per_arc['Azim'].min():.0f}-{per_arc['Azim'].max():.0f}" if per_arc is not None and "Azim" in per_arc.columns else "?"
    rh_range = f"{per_arc['RH'].min():.2f}-{per_arc['RH'].max():.2f} m" if per_arc is not None else "?"

    station_cfg = load_stations_config().get(station, {})
    params_path = station_cfg.get("gnssir_json_params_path", "")
    az_config, elev_range = "?", "?"
    try:
        with open(PROJECT_ROOT / params_path) as f:
            params = json.load(f)
        azval = params.get("azval2", [])
        if len(azval) >= 2:
            az_config = f"{azval[0]:.0f}-{azval[1]:.0f}"
        elev_range = f"{params.get('e1', '?')}-{params.get('e2', '?')}"
    except Exception:
        pass

    suspect_az = station_cfg.get("suspect_azimuths", {})
    baseline_period = station_cfg.get("baseline_period", station_cfg.get("ice_free_months", []))
    ant_height = station_cfg.get("ellipsoidal_height_m", "?")

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
        html.H3(station, style={"color": "#e0e0e0", "marginTop": "0", "marginBottom": "12px"}),
        html.Div([
            data_badge(f"v3: {n_v3}yr", n_v3 > 0),
            data_badge(f"S1: {n_s1}", n_s1 > 0),
            data_badge("ERA5", has_era5),
            data_badge("SMAP", has_smap),
            data_badge("TEC", has_tec),
        ], style={"marginBottom": "12px"}),
        html.Table([
            table_row("Location", f"{sinfo.get('lat', 0):.4f}N, {abs(sinfo.get('lon', 0)):.4f}W"),
            table_row("Antenna ht", f"{ant_height} m"),
            table_row("Year", str(year)),
            table_row("Arcs", f"{n_arcs:,}"),
            table_row("Frequencies", ", ".join(freq_list)),
            table_row("Elevation", elev_range),
            table_row("Az config", az_config),
            table_row("Az observed", az_range),
            table_row("RH range", rh_range),
            table_row("Baseline period", ", ".join(str(m) for m in baseline_period) if baseline_period else "not set"),
            table_row("Available years", ", ".join(str(y) for y in sinfo.get("years", []))),
        ], style={"width": "100%", "color": DARK_TEXT, "fontSize": "0.85rem", "borderCollapse": "collapse"}),
    ] + ([html.Div([
        html.Span("Suspect azimuths: ", style={"color": "#d29922", "fontWeight": "bold"}),
        html.Span(str(suspect_az), style={"color": "#d29922", "fontSize": "0.8rem"}),
    ], style={"marginTop": "8px"})] if suspect_az else []),
    style={"padding": "16px", "backgroundColor": DARK_CARD,
           "borderRadius": "6px", "border": f"1px solid {DARK_BORDER}"})

    return html.Div([
        html.Div([
            dcc.Graph(figure=fig, style={"height": "550px"}, config={"displayModeBar": True}),
        ], style={"flex": "2"}),
        html.Div([metadata], style={"flex": "1", "minWidth": "250px"}),
    ], style={"display": "flex", "gap": "16px", "width": "100%"})
