#!/usr/bin/env python3
"""Generate publication-quality CYGNSS comparison plots.

Reads the extracted CYGNSS parquet and ground comparison parquet,
produces improved visualizations.

Usage:
    python scripts/cygnss_plots.py --station FORA --year 2024
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

from scripts.utils.external_data import load_station_coords, get_results_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_station_lat_lon(station):
    coords = load_station_coords(station)
    return coords["lat"], coords["lon"]


def plot_fresnel_heatmap(cyg, station_lat, station_lon, station, year, out_dir):
    """Gridded Fresnel coefficient map revealing land/ocean boundary."""
    res = 0.05  # degrees
    lat_bins = np.arange(cyg["lat"].min() - res, cyg["lat"].max() + res, res)
    lon_bins = np.arange(cyg["lon"].min() - res, cyg["lon"].max() + res, res)

    fc_grid = np.full((len(lat_bins) - 1, len(lon_bins) - 1), np.nan)
    count_grid = np.zeros_like(fc_grid)

    lat_idx = np.digitize(cyg["lat"].values, lat_bins) - 1
    lon_idx = np.digitize(cyg["lon"].values, lon_bins) - 1

    for i in range(len(cyg)):
        li, lj = lat_idx[i], lon_idx[i]
        if 0 <= li < fc_grid.shape[0] and 0 <= lj < fc_grid.shape[1]:
            fc = cyg.iloc[i]["fresnel_coeff"]
            if np.isfinite(fc):
                if np.isnan(fc_grid[li, lj]):
                    fc_grid[li, lj] = fc
                    count_grid[li, lj] = 1
                else:
                    fc_grid[li, lj] += fc
                    count_grid[li, lj] += 1

    mask = count_grid > 0
    fc_grid[mask] /= count_grid[mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    im = ax.pcolormesh(lon_centers, lat_centers, fc_grid,
                       cmap="RdYlBu_r", vmin=0.60, vmax=0.68, shading="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Fresnel Coefficient")

    ax.plot(station_lon, station_lat, "k*", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate(station, (station_lon, station_lat),
                xytext=(8, 8), textcoords="offset points",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    for r in [0.5, 1.0]:
        circle = plt.Circle((station_lon, station_lat), r,
                             fill=False, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(f"{r}°", (station_lon + r * 0.7, station_lat + r * 0.7),
                    fontsize=8, color="gray")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{station} {year}: CYGNSS Fresnel Coefficient\n"
                 f"Gridded to {res}° ({len(cyg):,} specular points, {cyg['doy'].nunique()} days)")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_cygnss_fresnel_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_density_map(cyg, station_lat, station_lon, station, year, out_dir):
    """Specular point density map showing CYGNSS sampling pattern."""
    fig, ax = plt.subplots(figsize=(10, 8))

    h = ax.hist2d(cyg["lon"], cyg["lat"], bins=80, cmap="inferno",
                  norm=LogNorm(vmin=1, vmax=500))
    fig.colorbar(h[3], ax=ax, shrink=0.8, label="Specular Point Count (log scale)")

    ax.plot(station_lon, station_lat, "c*", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate(station, (station_lon, station_lat),
                xytext=(8, 8), textcoords="offset points",
                fontsize=11, fontweight="bold", color="cyan",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{station} {year}: CYGNSS Sampling Density\n"
                 f"{len(cyg):,} specular points over {cyg['doy'].nunique()} days")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_cygnss_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fresnel_vs_incidence(cyg, station, year, out_dir):
    """Empirical Fresnel curve: reflectivity vs incidence angle.

    Overlays theoretical Fresnel curves for water (eps=80) and ice (eps=3.2).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bin by incidence angle
    inc_bins = np.arange(0, 70, 2)
    inc_centers = inc_bins[:-1] + 1
    fc_medians = []
    fc_q25 = []
    fc_q75 = []
    counts = []
    for i in range(len(inc_bins) - 1):
        mask = (cyg["incidence_angle"] >= inc_bins[i]) & (cyg["incidence_angle"] < inc_bins[i + 1])
        vals = cyg.loc[mask, "fresnel_coeff"].dropna()
        if len(vals) > 10:
            fc_medians.append(vals.median())
            fc_q25.append(vals.quantile(0.25))
            fc_q75.append(vals.quantile(0.75))
            counts.append(len(vals))
        else:
            fc_medians.append(np.nan)
            fc_q25.append(np.nan)
            fc_q75.append(np.nan)
            counts.append(0)

    fc_medians = np.array(fc_medians)
    fc_q25 = np.array(fc_q25)
    fc_q75 = np.array(fc_q75)

    # Plot empirical curve
    valid = np.isfinite(fc_medians)
    ax.fill_between(inc_centers[valid], fc_q25[valid], fc_q75[valid],
                     alpha=0.3, color="steelblue", label="CYGNSS IQR")
    ax.plot(inc_centers[valid], fc_medians[valid], "o-", color="steelblue",
            ms=5, lw=2, label="CYGNSS median")

    # Theoretical Fresnel curves (power reflection coefficient, V-pol, L-band)
    theta = np.linspace(0.5, 70, 200)
    theta_rad = np.radians(theta)

    for eps_r, label, color, ls in [
        (80.0, "Water (ε=80)", "navy", "--"),
        (3.2, "Ice (ε=3.2)", "cornflowerblue", ":"),
    ]:
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        # Fresnel reflection coefficient (V-pol) — using incidence angle convention
        n = np.sqrt(eps_r)
        # R_v = (eps * cos(theta) - sqrt(eps - sin²(theta))) / (eps * cos(theta) + sqrt(eps - sin²(theta)))
        sqrt_term = np.sqrt(eps_r - sin_t**2 + 0j)
        Rv = (eps_r * cos_t - sqrt_term) / (eps_r * cos_t + sqrt_term)
        power_R = np.abs(Rv)**2

        ax.plot(theta, power_R, color=color, ls=ls, lw=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Incidence Angle (°)")
    ax.set_ylabel("Fresnel Power Reflection Coefficient")
    ax.set_title(f"{station} {year}: Empirical vs Theoretical Fresnel Curves\n"
                 f"CYGNSS L1 GPS reflections off ocean surface")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Secondary axis: sample count
    ax2 = ax.twinx()
    ax2.bar(inc_centers, counts, width=1.8, alpha=0.15, color="gray")
    ax2.set_ylabel("Sample Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, max(counts) * 5)

    fig.tight_layout()
    path = out_dir / f"{station}_{year}_cygnss_fresnel_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_seasonal_timeseries(cyg, merged, station, year, out_dir):
    """Full-year CYGNSS time series with ground-truth overlay on matched window."""
    # Daily CYGNSS stats (full year)
    cyg_daily = cyg.groupby("doy").agg(
        fc_med=("fresnel_coeff", "median"),
        fc_std=("fresnel_coeff", "std"),
        nbrcs_med=("nbrcs_mean", "median"),
        ws_med=("wind_speed", "median"),
        n_pts=("fresnel_coeff", "count"),
        inc_med=("incidence_angle", "median"),
    ).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    doy = cyg_daily["doy"]

    # Panel 1: Fresnel coefficient
    ax = axes[0]
    ax.fill_between(doy, cyg_daily["fc_med"] - cyg_daily["fc_std"],
                     cyg_daily["fc_med"] + cyg_daily["fc_std"],
                     alpha=0.2, color="steelblue")
    ax.plot(doy, cyg_daily["fc_med"], "-", color="steelblue", lw=1,
            label="CYGNSS Fresnel coeff (daily median)")
    if not merged.empty:
        ax.axvspan(merged["doy"].min(), merged["doy"].max(),
                   alpha=0.08, color="red", label="Ground-truth window")
    ax.set_ylabel("Fresnel Coeff")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{station} {year}: CYGNSS Full-Year Time Series")

    # Panel 2: Wind speed
    ax = axes[1]
    ax.plot(doy, cyg_daily["ws_med"], "-", color="green", lw=0.8)
    ax.set_ylabel("Wind Speed (m/s)")
    # Highlight storms (>10 m/s)
    storm_mask = cyg_daily["ws_med"] > 10
    if storm_mask.any():
        storm_doys = doy[storm_mask]
        ax.scatter(storm_doys, cyg_daily.loc[storm_mask, "ws_med"],
                   c="red", s=20, zorder=5, label=f"High wind (>10 m/s, n={storm_mask.sum()})")
        ax.legend(fontsize=8)

    # Panel 3: NBRCS (surface roughness proxy)
    ax = axes[2]
    ax.plot(doy, cyg_daily["nbrcs_med"], "-", color="purple", lw=0.8)
    ax.set_ylabel("NBRCS (median)")
    ax.set_yscale("log")

    # Panel 4: Point count (data density)
    ax = axes[3]
    ax.bar(doy, cyg_daily["n_pts"], width=1, color="gray", alpha=0.6)
    ax.set_ylabel("CYGNSS Points/Day")
    ax.set_xlabel("Day of Year")

    # Add month labels
    import calendar
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = list(calendar.month_abbr[1:])
    axes[3].set_xticks(month_starts)
    axes[3].set_xticklabels(month_names, fontsize=9)

    fig.tight_layout()
    path = out_dir / f"{station}_{year}_cygnss_seasonal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_wind_vs_ground(merged, station, year, out_dir):
    """CYGNSS wind speed vs ground GNSS-IR features with regression lines."""
    if merged.empty or len(merged) < 10:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    pairs = [
        ("wind_speed_median", "VS", "Wind Speed (m/s)", "SNR Variance"),
        ("wind_speed_median", "RH", "Wind Speed (m/s)", "Reflector Height (m)"),
        ("wind_speed_median", "gamma", "Wind Speed (m/s)", "Damping (γ)"),
        ("fresnel_coeff_median", "phase", "Fresnel Coeff", "Phase (rad)"),
        ("fresnel_coeff_median", "CLR", "Fresnel Coeff", "CLR"),
        ("nbrcs_median", "AF", "NBRCS", "Area Factor"),
    ]

    for idx, (cx, gx, cl, gl) in enumerate(pairs):
        if cx not in merged.columns or gx not in merged.columns:
            continue
        r, c = divmod(idx, 3)
        ax = axes[r, c]

        valid = merged[[cx, gx, "doy"]].dropna()
        if len(valid) < 5:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        sc = ax.scatter(valid[cx], valid[gx], c=valid["doy"], cmap="Spectral",
                        s=30, edgecolors="black", linewidths=0.3, zorder=5)
        ax.set_xlabel(cl, fontsize=10)
        ax.set_ylabel(gl, fontsize=10)

        # Regression line
        corr = valid[cx].corr(valid[gx])
        z = np.polyfit(valid[cx], valid[gx], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[cx].min(), valid[cx].max(), 50)
        ax.plot(x_line, p(x_line), "k--", lw=1, alpha=0.5)

        ax.set_title(f"r = {corr:+.3f}", fontsize=11,
                     fontweight="bold" if abs(corr) > 0.2 else "normal",
                     color="darkred" if abs(corr) > 0.3 else "black")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sc, cax=cax, label="DOY")

    fig.suptitle(f"{station} {year}: CYGNSS vs Ground GNSS-IR (n={len(merged)} days)",
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    path = out_dir / f"{station}_{year}_cygnss_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_storm_events(cyg, merged, station, year, out_dir):
    """Case study: high-wind days vs ground features."""
    if merged.empty or len(merged) < 10:
        return

    # Find high-wind days in the matched window
    high_wind = merged[merged["wind_speed_median"] > merged["wind_speed_median"].quantile(0.8)]
    low_wind = merged[merged["wind_speed_median"] < merged["wind_speed_median"].quantile(0.2)]

    if len(high_wind) < 3 or len(low_wind) < 3:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    features = [("VS", "SNR Variance"), ("RH", "Reflector Height (m)"),
                ("gamma", "Damping (γ)"), ("CLR", "CLR")]

    for ax, (feat, label) in zip(axes, features):
        if feat not in merged.columns:
            continue
        data_high = high_wind[feat].dropna()
        data_low = low_wind[feat].dropna()
        if len(data_high) < 2 or len(data_low) < 2:
            continue

        bp = ax.boxplot([data_low.values, data_high.values],
                        labels=[f"Calm\n(n={len(data_low)})",
                                f"Windy\n(n={len(data_high)})"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#e74c3c")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_ylabel(label)

        # t-test
        from scipy import stats
        t, p = stats.ttest_ind(data_low, data_high)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"p={p:.3f} {sig}", fontsize=10)

    fig.suptitle(f"{station} {year}: Ground Feature Response to CYGNSS Wind\n"
                 f"Calm (<P20) vs Windy (>P80) days", fontsize=12, y=1.02)
    fig.tight_layout()
    path = out_dir / f"{station}_{year}_cygnss_wind_response.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    station_lat, station_lon = _load_station_lat_lon(args.station)

    cygnss_dir = get_results_dir(args.station) / "cygnss"
    cyg_path = cygnss_dir / f"{args.station}_{args.year}_cygnss_extracted.parquet"
    merged_path = cygnss_dir / f"{args.station}_{args.year}_cygnss_comparison.parquet"

    cyg = pd.read_parquet(cyg_path)
    merged = pd.read_parquet(merged_path) if merged_path.exists() else pd.DataFrame()

    print(f"=== Generating CYGNSS plots: {args.station} {args.year} ===")
    print(f"  CYGNSS points: {len(cyg):,}, Matched days: {len(merged)}")

    plot_fresnel_heatmap(cyg, station_lat, station_lon, args.station, args.year, cygnss_dir)
    plot_density_map(cyg, station_lat, station_lon, args.station, args.year, cygnss_dir)
    plot_fresnel_vs_incidence(cyg, args.station, args.year, cygnss_dir)
    plot_seasonal_timeseries(cyg, merged, args.station, args.year, cygnss_dir)
    plot_wind_vs_ground(merged, args.station, args.year, cygnss_dir)
    plot_storm_events(cyg, merged, args.station, args.year, cygnss_dir)

    print(f"\nAll plots saved to {cygnss_dir}/")


if __name__ == "__main__":
    main()
