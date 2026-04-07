# ABOUTME: Reusable polar diagram renderer for GNSS-IR reflection points.
# ABOUTME: Supports polar and Cartesian projections, 5 color modes, and suspect azimuth shading.

"""
# In progress — not yet integrated
PolarDiagram — a reusable renderer for GNSS-IR reflection geometry.

Accepts per-arc DataFrames (from reference_data.load_per_arc) and renders
polar or Cartesian scatter diagrams colored by various fields.

Color modes:
    wse        — water surface elevation (demeaned, cm)
    amplitude  — SNR amplitude
    pknoise    — peak-to-noise ratio
    frequency  — frequency group (L1, L2, L5, etc.)
    ice        — v3 ice classification state

Usage:
    from scripts.visualizer.polar_diagram import PolarDiagram, PolarDiagramConfig

    diagram = PolarDiagram(per_arc_df, station_config=cfg)
    subset = diagram.select_time_window(doy_range=(180, 200))
    fig = diagram.render(PolarDiagramConfig(color_by="wse"), data=subset)
    fig.savefig("polar.png", dpi=150, bbox_inches="tight")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Wedge

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Frequency color palette (matches dashboard)
FREQ_COLORS = {
    "L1": "#1f77b4",
    "L2": "#ff7f0e",
    "L2C": "#2ca02c",
    "L5": "#d62728",
    "L6": "#9467bd",
    "L1C": "#8c564b",
}

# v3 ice classification colors (matches dashboard)
V3_COLORS = {
    "open_water": "#2196F3",
    "freeze_up": "#FF9800",
    "ice_surface": "#90CAF9",
    "ice_layered": "#E1BEE7",
    "ice_decaying": "#FFCC80",
    "break_up": "#F44336",
    "water": "#2196F3",
    "ice": "#90CAF9",
}

ColorMode = Literal["wse", "amplitude", "pknoise", "frequency", "ice"]


@dataclass
class PolarDiagramConfig:
    """Configuration for a single polar diagram render."""

    color_by: ColorMode = "wse"
    title: str = ""
    figsize: tuple = (8, 8)
    dpi: int = 150
    marker_size: float = 8
    marker_alpha: float = 0.7
    show_suspect: bool = True
    show_compass: bool = True
    cmap: str = "coolwarm"
    dark_mode: bool = False
    sector_size_deg: float = 10.0


class PolarDiagram:
    """Renders GNSS-IR per-arc data as polar or Cartesian reflection diagrams.

    Args:
        per_arc_df: DataFrame with columns [Azim, RH, eminO, emaxO, WSE_dm, ...]
        station_config: station configuration dict (for suspect azimuths, gnssir params)
    """

    def __init__(self, per_arc_df: pd.DataFrame, station_config: dict | None = None):
        self.df = per_arc_df.copy()
        self.station_config = station_config or {}

        # Pre-compute reflection geometry
        self.df["elev_avg"] = (self.df["eminO"] + self.df["emaxO"]) / 2.0
        self.df["refl_dist"] = self.df["RH"] / np.tan(
            np.radians(self.df["elev_avg"])
        )

        # Parse suspect azimuths from config
        self._suspect_sectors = self._parse_suspect_azimuths()

        # Parse valid azimuth range from gnssir params
        self._az_ranges = self._parse_azimuth_ranges()

    def _parse_suspect_azimuths(self) -> list[tuple[float, float, str]]:
        """Parse suspect_azimuths from station config into (az_start, az_end, reason) tuples."""
        suspect = self.station_config.get("suspect_azimuths", {})
        sectors = []
        for key, value in suspect.items():
            if key == "notes":
                continue
            try:
                parts = key.split("-")
                az_start = float(parts[0])
                az_end = float(parts[1])
                reason = value if isinstance(value, str) else str(value)
                sectors.append((az_start, az_end, reason))
            except (ValueError, IndexError):
                continue
        return sectors

    def _parse_azimuth_ranges(self) -> list[tuple[float, float]]:
        """Parse valid azimuth ranges from gnssir JSON params."""
        params_path = self.station_config.get("gnssir_json_params_path")
        if not params_path:
            return [(0, 360)]

        full_path = PROJECT_ROOT / params_path
        if not full_path.exists():
            return [(0, 360)]

        try:
            with open(full_path) as f:
                params = json.load(f)
            azval = params.get("azval2", params.get("azval", [0, 360]))
            # azval is a flat list: [start1, end1, start2, end2, ...]
            ranges = []
            for i in range(0, len(azval), 2):
                if i + 1 < len(azval):
                    ranges.append((float(azval[i]), float(azval[i + 1])))
            return ranges if ranges else [(0, 360)]
        except Exception:
            return [(0, 360)]

    def select_time_window(
        self,
        doy_range: tuple[int, int] | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """Filter per-arc data to a time window.

        Args:
            doy_range: (start_doy, end_doy) inclusive
            date_range: (start_date, end_date) as strings parseable by pd.Timestamp

        Returns:
            Filtered DataFrame (view into self.df).
        """
        mask = pd.Series(True, index=self.df.index)

        if doy_range is not None:
            if "doy" in self.df.columns:
                mask &= (self.df["doy"] >= doy_range[0]) & (
                    self.df["doy"] <= doy_range[1]
                )

        if date_range is not None and "datetime" in self.df.columns:
            start = pd.Timestamp(date_range[0])
            end = pd.Timestamp(date_range[1])
            mask &= (self.df["datetime"] >= start) & (self.df["datetime"] <= end)

        return self.df[mask].copy()

    def render(
        self,
        config: PolarDiagramConfig | None = None,
        data: pd.DataFrame | None = None,
    ) -> plt.Figure:
        """Render a polar projection diagram.

        Args:
            config: rendering options (defaults to PolarDiagramConfig())
            data: subset to render (defaults to all data)

        Returns:
            matplotlib Figure
        """
        if config is None:
            config = PolarDiagramConfig()
        if data is None:
            data = self.df

        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        ax = fig.add_subplot(111, projection="polar")

        if config.dark_mode:
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")
            text_color = "#e0e0e0"
        else:
            text_color = "black"

        # Convention: azimuth=0 at top (North), increasing clockwise
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        az_rad = np.radians(data["Azim"].values)
        dist = data["refl_dist"].values if "refl_dist" in data.columns else (
            data["RH"].values / np.tan(np.radians((data["eminO"] + data["emaxO"]) / 2.0))
        )

        # Color mapping
        colors, cmap, norm, legend_handles = self._compute_colors(data, config)

        sc = ax.scatter(
            az_rad,
            dist,
            c=colors,
            s=config.marker_size,
            alpha=config.marker_alpha,
            edgecolors="none",
            zorder=3,
        )

        # Suspect azimuth shading
        if config.show_suspect and self._suspect_sectors:
            max_dist = np.nanmax(dist) if len(dist) > 0 else 100
            for az_start, az_end, reason in self._suspect_sectors:
                theta_start = np.radians(az_start)
                theta_end = np.radians(az_end)
                thetas = np.linspace(theta_start, theta_end, 50)
                ax.fill_between(
                    thetas,
                    0,
                    max_dist * 1.1,
                    alpha=0.15,
                    color="red",
                    zorder=1,
                    label=f"Suspect: {reason}" if az_start == self._suspect_sectors[0][0] else None,
                )

        # Colorbar or legend
        if config.color_by in ("frequency", "ice"):
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    bbox_to_anchor=(1.3, 1.0),
                    fontsize=8,
                    framealpha=0.8,
                )
        elif cmap is not None and norm is not None:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            label_map = {
                "wse": "Water Level (cm)",
                "amplitude": "Amplitude",
                "pknoise": "Peak/Noise",
            }
            cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label(label_map.get(config.color_by, config.color_by))

        # Labels
        ax.set_rlabel_position(45)
        ax.set_ylabel("Distance (m)", labelpad=30, color=text_color)
        if config.title:
            ax.set_title(config.title, pad=20, fontsize=12, fontweight="bold", color=text_color)

        fig.tight_layout()
        return fig

    def render_cartesian(
        self,
        config: PolarDiagramConfig | None = None,
        data: pd.DataFrame | None = None,
    ) -> plt.Figure:
        """Render a Cartesian (x-y meters from station) reflection diagram.

        Positive x = East, positive y = North.

        Args:
            config: rendering options
            data: subset to render

        Returns:
            matplotlib Figure
        """
        if config is None:
            config = PolarDiagramConfig()
        if data is None:
            data = self.df

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        if config.dark_mode:
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")
            text_color = "#e0e0e0"
        else:
            text_color = "black"

        dist = data["refl_dist"].values if "refl_dist" in data.columns else (
            data["RH"].values / np.tan(np.radians((data["eminO"] + data["emaxO"]) / 2.0))
        )
        az_rad = np.radians(data["Azim"].values)
        dx = dist * np.sin(az_rad)
        dy = dist * np.cos(az_rad)

        colors, cmap, norm, legend_handles = self._compute_colors(data, config)

        sc = ax.scatter(
            dx, dy, c=colors,
            s=config.marker_size, alpha=config.marker_alpha,
            edgecolors="none", zorder=3,
        )

        # Station marker at origin
        ax.plot(0, 0, "r^", markersize=12, markeredgecolor="white", markeredgewidth=2, zorder=10)

        # Suspect azimuth shading
        if config.show_suspect and self._suspect_sectors:
            max_dist = np.nanmax(dist) if len(dist) > 0 else 100
            for az_start, az_end, reason in self._suspect_sectors:
                wedge = Wedge(
                    (0, 0), max_dist * 1.1,
                    90 - az_end, 90 - az_start,  # matplotlib wedge uses math convention
                    alpha=0.15, color="red", zorder=1,
                )
                ax.add_patch(wedge)

        # Colorbar or legend
        if config.color_by in ("frequency", "ice"):
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.8)
        elif cmap is not None and norm is not None:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            label_map = {
                "wse": "Water Level (cm)",
                "amplitude": "Amplitude",
                "pknoise": "Peak/Noise",
            }
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
            cbar.set_label(label_map.get(config.color_by, config.color_by))

        # Compass annotations
        if config.show_compass:
            limit = np.nanmax(np.abs(np.concatenate([dx, dy]))) * 1.1 if len(dx) > 0 else 100
            ax.annotate("N", (0, limit * 0.9), ha="center", fontsize=11, fontweight="bold", color=text_color)
            ax.annotate("E", (limit * 0.9, 0), ha="center", fontsize=11, fontweight="bold", color=text_color)
            ax.annotate("S", (0, -limit * 0.9), ha="center", fontsize=11, fontweight="bold", color=text_color)
            ax.annotate("W", (-limit * 0.9, 0), ha="center", fontsize=11, fontweight="bold", color=text_color)

        ax.set_aspect("equal")
        ax.set_xlabel("East (m)", color=text_color)
        ax.set_ylabel("North (m)", color=text_color)
        if config.title:
            ax.set_title(config.title, fontsize=12, fontweight="bold", color=text_color)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def sector_summary(
        self,
        data: pd.DataFrame | None = None,
        sector_size_deg: float = 10.0,
    ) -> pd.DataFrame:
        """Compute statistics per azimuth sector.

        Returns DataFrame with columns: sector_center, az_start, az_end,
        n_arcs, mean_rh, std_rh, mean_amp, has_data, is_suspect.
        """
        if data is None:
            data = self.df

        sectors = []
        for az_start in np.arange(0, 360, sector_size_deg):
            az_end = az_start + sector_size_deg
            center = az_start + sector_size_deg / 2.0
            mask = (data["Azim"] >= az_start) & (data["Azim"] < az_end)
            subset = data[mask]

            is_suspect = any(
                az_start < s_end and az_end > s_start
                for s_start, s_end, _ in self._suspect_sectors
            )

            if len(subset) > 0:
                sectors.append({
                    "sector_center": center,
                    "az_start": az_start,
                    "az_end": az_end,
                    "n_arcs": len(subset),
                    "mean_rh": subset["RH"].mean(),
                    "std_rh": subset["RH"].std(),
                    "mean_amp": subset["Amp"].mean() if "Amp" in subset.columns else np.nan,
                    "has_data": True,
                    "is_suspect": is_suspect,
                })
            else:
                sectors.append({
                    "sector_center": center,
                    "az_start": az_start,
                    "az_end": az_end,
                    "n_arcs": 0,
                    "mean_rh": np.nan,
                    "std_rh": np.nan,
                    "mean_amp": np.nan,
                    "has_data": False,
                    "is_suspect": is_suspect,
                })

        return pd.DataFrame(sectors)

    # --- internal helpers ---

    def _compute_colors(self, data, config):
        """Return (colors_array, cmap, norm, legend_handles) for the given color mode."""
        cmap_obj = None
        norm = None
        legend_handles = None

        if config.color_by == "wse":
            values = data["WSE_dm"].values * 100
            cmap_obj = plt.cm.get_cmap(config.cmap)
            vmin, vmax = np.nanpercentile(values, [5, 95]) if len(values) > 0 else (-10, 10)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            colors = cmap_obj(norm(values))

        elif config.color_by == "amplitude":
            values = data["Amp"].values if "Amp" in data.columns else np.zeros(len(data))
            cmap_obj = plt.cm.get_cmap("viridis")
            vmin, vmax = np.nanpercentile(values, [5, 95]) if len(values) > 0 else (0, 50)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            colors = cmap_obj(norm(values))

        elif config.color_by == "pknoise":
            values = data["PkNoise"].values if "PkNoise" in data.columns else np.zeros(len(data))
            cmap_obj = plt.cm.get_cmap("plasma")
            vmin, vmax = np.nanpercentile(values, [5, 95]) if len(values) > 0 else (1, 5)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            colors = cmap_obj(norm(values))

        elif config.color_by == "frequency":
            freq_col = "freq_group" if "freq_group" in data.columns else "freq"
            groups = data[freq_col].values
            unique_groups = sorted(set(str(g) for g in groups))

            color_map = {}
            for g in unique_groups:
                color_map[g] = FREQ_COLORS.get(str(g), "#888888")

            colors = [color_map.get(str(g), "#888888") for g in groups]
            legend_handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[g],
                           markersize=8, label=str(g))
                for g in unique_groups if g in color_map
            ]

        elif config.color_by == "ice":
            state_col = "v3_state" if "v3_state" in data.columns else None
            if state_col is None:
                # Fallback: uniform color
                colors = ["#2196F3"] * len(data)
                legend_handles = []
            else:
                states = data[state_col].values
                unique_states = sorted(set(str(s) for s in states if pd.notna(s)))
                colors = [V3_COLORS.get(str(s), "#888888") for s in states]
                legend_handles = [
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=V3_COLORS.get(s, "#888888"),
                               markersize=8, label=s)
                    for s in unique_states
                ]

        else:
            colors = ["#1f77b4"] * len(data)

        return colors, cmap_obj, norm, legend_handles
