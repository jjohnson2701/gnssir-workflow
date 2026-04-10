# ABOUTME: Time Series tab — multi-panel RH + ERA5 + SMAP + v3 strip + S1 click
# ABOUTME: Renders per-frequency RH, contextual data, and S1 imagery interaction

"""Time Series tab for the GNSS-IR dashboard."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc

from dashboard.utils import (
    PLOTLY_DARK, DARK_TEXT, FREQ_COLORS, V3_COLORS,
    get_mode, state_order,
)
from dashboard.data_loader import load_v3, load_era5, load_smap, list_s1_images


def render(station, year, per_arc):
    """Render the Time Series tab content."""
    v3 = load_v3(station, year)
    s1_images = list_s1_images(station)
    year_s1 = [img for img in s1_images if img["date"].startswith(str(year))]
    era5 = load_era5(station, year)
    smap = load_smap(station, year)

    has_era5 = era5 is not None and "t2m_mean" in era5.columns
    has_smap = smap is not None and "polarization_diff" in smap.columns
    has_v3 = v3 is not None and "v3_state" in v3.columns

    n_rows = 1
    if has_v3: n_rows += 1
    if has_era5: n_rows += 1
    if has_smap: n_rows += 1

    row_heights = [0.5] + [0.15] * (n_rows - 1)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.03)

    df = per_arc.copy() if per_arc is not None else pd.DataFrame()
    if not df.empty:
        if "date_dt" not in df.columns and "date" in df.columns:
            df["date_dt"] = pd.to_datetime(df["date"])
        if "freq_group" not in df.columns:
            df["freq_group"] = "L1"

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

    if has_era5:
        era5_dates = [_dt(year, 1, 1) + _td(days=int(d) - 1) for d in era5["doy"]]
        fig.add_trace(go.Scatter(
            x=era5_dates, y=era5["t2m_mean"], mode="lines",
            name="ERA5 Temp", line=dict(color="#ff6b6b", width=1.5),
            hovertemplate="Temp: %{y:.1f} C<extra></extra>",
        ), row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=current_row, col=1)
        fig.update_yaxes(title_text="C", row=current_row, col=1)
        current_row += 1

    if has_smap:
        smap_dates = [_dt(year, 1, 1) + _td(days=int(d) - 1) for d in smap["doy"]]
        fig.add_trace(go.Scatter(
            x=smap_dates, y=smap["polarization_diff"], mode="lines",
            name="SMAP TBv-TBh", line=dict(color="#a855f7", width=1.5),
            hovertemplate="SMAP pol: %{y:.1f}K<extra></extra>",
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="K", row=current_row, col=1)
        current_row += 1

    if has_v3:
        for state in state_order(get_mode(v3)):
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

    return html.Div([
        html.P("Click a point or S1 marker to view nearest imagery below.",
               style={"color": "#8b949e", "fontSize": "0.8rem", "margin": "4px 0"}),
        dcc.Graph(id="ts-graph", figure=fig, style={"height": "60vh"}),
        html.Div(id="ts-imagery-panel", style={"marginTop": "8px", "minHeight": "50px"}),
    ])
