#!/usr/bin/env python3
"""
Investigation B3: Multi-frequency envelope ratio for snow-on-ice detection.

A snow layer creates frequency-dependent partial reflections. L1 (shorter λ)
reflects more from the snow surface, while L2C (longer λ) penetrates deeper.
This makes the Hilbert envelope ratio between bands vary with elevation.

Phase 1: Synthetic two-layer Fresnel model — quantify theoretical sensitivity.
Phase 2: Real data test on UMNQ 2025 (if synthetic looks promising).

Usage:
    python scripts/investigate_envelope_ratio.py --station UMNQ --year 2025
    python scripts/investigate_envelope_ratio.py --synthetic-only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal.windows import tukey

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# Physical constants
C_LIGHT = 299792458.0  # m/s
L1_WAVELENGTH = 0.19029  # m
L2C_WAVELENGTH = 0.24421  # m
L5_WAVELENGTH = 0.25482  # m


# ---------------------------------------------------------------------------
# Phase 1: Synthetic two-layer Fresnel model
# ---------------------------------------------------------------------------

def fresnel_reflection_two_layer(elev_deg, rh_ice, snow_depth, eps_snow, eps_ice,
                                  wavelength):
    """Model reflected signal through air → snow → ice with Fresnel coefficients.

    Simplified model: vertical polarization, normal incidence approximation
    for thin layers. The reflected signal is a superposition of the snow-surface
    reflection and the ice-surface reflection with a phase delay.

    Args:
        elev_deg: elevation angles (degrees)
        rh_ice: reflector height to ice surface (meters)
        snow_depth: snow layer thickness (meters)
        eps_snow: relative permittivity of snow (~1.5-2.0)
        eps_ice: relative permittivity of ice (~3.0-3.5)
        wavelength: carrier wavelength (meters)

    Returns:
        detrended signal (real part of reflected composite)
    """
    elev_rad = np.radians(elev_deg)
    sin_e = np.sin(elev_rad)
    k = 2 * np.pi / wavelength

    # Fresnel reflection coefficients (simplified, RHCP → approximation)
    # Air-snow interface
    n_air = 1.0
    n_snow = np.sqrt(eps_snow)
    n_ice = np.sqrt(eps_ice)

    # At grazing angles for GNSS-IR (5-25°), use approximate Fresnel for
    # circular polarization: r ≈ (n1 - n2) / (n1 + n2)
    r_air_snow = (n_air - n_snow) / (n_air + n_snow)
    r_snow_ice = (n_snow - n_ice) / (n_snow + n_ice)
    t_air_snow = 1 - abs(r_air_snow) ** 2  # power transmission

    # Phase from antenna to snow surface and back
    rh_snow = rh_ice + snow_depth  # snow surface is higher (closer to antenna)
    phase_snow = 2 * k * rh_snow * sin_e

    # Phase from antenna to ice surface through snow
    # Extra path in snow: 2 * snow_depth * n_snow * sin(refracted angle)
    # Snell: sin(θ_air)/sin(θ_snow) = n_snow → sin(θ_snow) = sin(θ_air)/n_snow
    # For reflector geometry: sin(ε) maps to the reflection phase
    phase_ice = 2 * k * (rh_ice * sin_e + snow_depth * n_snow * sin_e)

    # Composite reflected signal
    signal = (r_air_snow * np.cos(phase_snow) +
              t_air_snow * r_snow_ice * np.cos(phase_ice))

    return signal


def run_synthetic_test(out_dir):
    """Run synthetic two-layer Fresnel model for various snow depths.

    Produces: envelope ratio slope vs snow depth for L1 and L2C.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    elev_deg = np.linspace(5, 25, 500)
    rh_ice = 4.0  # meters (typical for UMNQ/ROSS)
    eps_snow = 1.8
    eps_ice = 3.2

    snow_depths = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]  # meters
    wavelengths = {"L1": L1_WAVELENGTH, "L2C": L2C_WAVELENGTH, "L5": L5_WAVELENGTH}

    results = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, d_snow in enumerate(snow_depths):
        envelopes = {}
        for band, wl in wavelengths.items():
            signal = fresnel_reflection_two_layer(
                elev_deg, rh_ice, d_snow, eps_snow, eps_ice, wl
            )
            # Hilbert envelope with Tukey taper
            window = tukey(len(signal), alpha=0.2)
            analytic = hilbert(signal * window)
            env = np.abs(analytic)
            # Trim tapered edges
            trim = int(len(signal) * 0.1)
            envelopes[band] = env[trim:-trim]

        elev_trim = elev_deg[trim:-trim]
        sin_e = np.sin(np.radians(elev_trim))

        # Compute envelope ratios
        ratio_L1_L2 = envelopes["L1"] / np.maximum(envelopes["L2C"], 1e-10)
        ratio_L1_L5 = envelopes["L1"] / np.maximum(envelopes["L5"], 1e-10)

        # Fit linear slope to ratio vs sin²(ε)
        sin2_e = sin_e ** 2
        for ratio, pair_name in [(ratio_L1_L2, "L1/L2C"), (ratio_L1_L5, "L1/L5")]:
            # Robust fit
            mask = np.isfinite(ratio)
            if mask.sum() < 10:
                continue
            coeffs = np.polyfit(sin2_e[mask], ratio[mask], 1)
            results.append({
                "snow_depth_cm": d_snow * 100,
                "pair": pair_name,
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "ratio_mean": np.mean(ratio[mask]),
                "ratio_std": np.std(ratio[mask]),
            })

        # Plot envelopes
        row, col = divmod(i, 3)
        ax = axes[row, col]
        for band, env in envelopes.items():
            ax.plot(elev_trim, env, label=band, alpha=0.8)
        ax.set_title(f"Snow = {d_snow*100:.0f} cm")
        ax.set_xlabel("Elevation (°)")
        ax.set_ylabel("Envelope")
        ax.legend(fontsize=8)

    fig.suptitle("Synthetic Hilbert envelopes: air → snow → ice", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_envelopes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results_df = pd.DataFrame(results)

    # Plot: slope vs snow depth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pair in zip(axes, ["L1/L2C", "L1/L5"]):
        sub = results_df[results_df["pair"] == pair]
        ax.plot(sub["snow_depth_cm"], sub["slope"], "o-", ms=8)
        ax.set_xlabel("Snow depth (cm)")
        ax.set_ylabel("Envelope ratio slope vs sin²(ε)")
        ax.set_title(f"{pair} envelope ratio slope")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Theoretical sensitivity: envelope ratio slope vs snow depth",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_slope_vs_snow_depth.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print("Synthetic two-layer model results")
    print(f"{'='*60}")
    print(f"Parameters: RH_ice={rh_ice}m, ε_snow={eps_snow}, ε_ice={eps_ice}")
    print(f"Elevation range: 5–25°")
    print(results_df.to_string(index=False))

    # Compare against typical noise levels
    # gamma_r2 < 0.5 means poor fit → envelope is noisy
    # Typical envelope noise: ~20% of mean envelope
    print(f"\nNoise context:")
    print(f"  If typical envelope noise ~ 20% of mean, slope noise ~ 0.2 * mean/range")
    print(f"  For L1/L2C, 0cm slope = {results_df[(results_df['pair']=='L1/L2C') & (results_df['snow_depth_cm']==0)]['slope'].values}")
    print(f"  Detection requires slope signal > noise — see figures.")

    return results_df


# ---------------------------------------------------------------------------
# Phase 2: Real data test
# ---------------------------------------------------------------------------

def run_real_data_test(station, year, out_dir):
    """Compute envelope ratios for real multi-frequency arcs."""
    from scripts.snr_feature_extractor import (
        read_snr_file, segment_satellite_arcs, find_matching_segment,
        detrend_arc, _snr_file_path, _load_station_config,
        FREQ_TO_COL, FREQ_WAVELENGTH,
    )

    out_dir = Path(out_dir)
    config = _load_station_config(station)
    if config is None:
        log.error(f"No config for {station}")
        return None

    e1, e2 = config["e1"], config["e2"]
    poly_order = config.get("polyV", 4)
    pele = tuple(config.get("pele", [5, 30]))

    at = pd.read_parquet(
        PROJECT_ROOT / "results_annual" / station
        / f"{station}_{year}_arc_table.parquet"
    )

    # Focus on GPS L1-L2C pairs
    gps = at[at["freq"].isin([1, 20])]
    arc_groups = gps.groupby(["doy", "sat", "UTCtime", "rise"])
    multi = {k: g for k, g in arc_groups if set(g["freq"].unique()) >= {1, 20}}
    log.info(f"GPS L1+L2C arcs: {len(multi)}")

    results = []
    last_snr_doy = None
    snr_data = None

    for (doy, sat, utctime, rise), freq_rows in sorted(multi.items()):
        if doy != last_snr_doy:
            snr_path = _snr_file_path(station, year, doy)
            if not snr_path.exists():
                gz = Path(str(snr_path) + ".gz")
                if gz.exists():
                    snr_path = gz
                else:
                    continue
            try:
                snr_data = read_snr_file(snr_path)
            except Exception:
                snr_data = None
                continue
            last_snr_doy = doy

        if snr_data is None:
            continue

        sat_mask = snr_data[:, 0] == int(sat)
        sat_data = snr_data[sat_mask]
        if len(sat_data) < 5:
            continue

        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime, target_rise=rise, e1=e1, e2=e2,
        )
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        mask = (arc_ele >= e1) & (arc_ele <= e2)
        if mask.sum() < 30:
            continue

        ele_w = arc_ele[mask]
        sin_e = np.sin(np.radians(ele_w))

        # Extract detrended and envelope for both bands
        envs = {}
        for freq in [1, 20]:
            col_idx = FREQ_TO_COL[freq]
            if col_idx >= arc_data.shape[1]:
                continue
            snr_db = arc_data[:, col_idx]
            if np.all(snr_db == 0):
                continue
            snr_lin = np.power(10, snr_db / 20)
            detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)
            dsnr = detrended[mask]

            # Hilbert envelope with Tukey taper
            n = len(dsnr)
            window = tukey(n, alpha=0.2)
            analytic = hilbert(dsnr * window)
            env = np.abs(analytic)
            trim = max(1, int(n * 0.1))
            envs[freq] = env[trim:-trim]

        if 1 not in envs or 20 not in envs:
            continue
        if len(envs[1]) != len(envs[20]):
            min_len = min(len(envs[1]), len(envs[20]))
            envs[1] = envs[1][:min_len]
            envs[20] = envs[20][:min_len]

        trim = max(1, int(mask.sum() * 0.1))
        sin_e_trim = sin_e[trim:-trim]
        if len(sin_e_trim) != len(envs[1]):
            min_len = min(len(sin_e_trim), len(envs[1]))
            sin_e_trim = sin_e_trim[:min_len]
            envs[1] = envs[1][:min_len]
            envs[20] = envs[20][:min_len]

        if len(sin_e_trim) < 15:
            continue

        # Envelope ratio
        ratio = envs[1] / np.maximum(envs[20], 1e-10)
        sin2_e = sin_e_trim ** 2

        # Fit slope
        finite = np.isfinite(ratio) & np.isfinite(sin2_e)
        if finite.sum() < 10:
            continue

        try:
            coeffs = np.polyfit(sin2_e[finite], ratio[finite], 1)
        except Exception:
            continue

        row0 = freq_rows.iloc[0]
        results.append({
            "doy": doy, "sat": int(sat), "UTCtime": utctime, "rise": int(rise),
            "env_ratio_slope": coeffs[0],
            "env_ratio_intercept": coeffs[1],
            "env_ratio_mean": float(np.mean(ratio[finite])),
            "date": row0.get("date"),
            "Azim": float(row0.get("Azim", np.nan)),
        })

    if not results:
        return None

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    # Load v3 classification
    v3_path = (PROJECT_ROOT / "results_annual" / station
               / f"{station}_{year}_ice_classification_v3.parquet")
    if v3_path.exists():
        v3 = pd.read_parquet(v3_path)
        v3["doy"] = pd.to_datetime(v3["date"]).dt.dayofyear
        v3_map = dict(zip(v3["doy"], v3["v3_state"]))
        df["v3_state"] = df["doy"].map(v3_map).fillna("unknown")
    else:
        df["v3_state"] = "unknown"

    # --- Diagnostics ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Time series
    daily = df.groupby("doy")["env_ratio_slope"].agg(["median", "std", "count"])
    daily = daily[daily["count"] >= 2]
    axes[0].plot(daily.index, daily["median"], ".", ms=2)
    axes[0].fill_between(daily.index,
                         daily["median"] - daily["std"],
                         daily["median"] + daily["std"], alpha=0.2)
    axes[0].set_xlabel("DOY")
    axes[0].set_ylabel("Envelope ratio slope (L1/L2C)")
    axes[0].set_title("Time series")
    axes[0].axhline(0, color="gray", ls="--")

    # By state
    states = ["open_water", "ice_surface", "ice_layered", "ice_decaying",
              "baseline", "anomalous"]
    active = [s for s in states if s in df["v3_state"].values]
    if active:
        data_by_state = [df[df["v3_state"] == s]["env_ratio_slope"].dropna()
                         for s in active]
        data_by_state = [d for d, s in zip(data_by_state, active) if len(d) >= 5]
        active = [s for s, d in zip(active, [df[df["v3_state"] == s]["env_ratio_slope"].dropna() for s in active]) if len(d) >= 5]
        if data_by_state:
            axes[1].boxplot([d.values for d in data_by_state],
                           labels=[f"{s}\nn={len(d)}" for s, d in zip(active, data_by_state)])
            axes[1].set_ylabel("Envelope ratio slope")
            axes[1].set_title("By surface state")
            axes[1].axhline(0, color="gray", ls="--")

    # Distribution
    axes[2].hist(df["env_ratio_slope"].dropna(), bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Envelope ratio slope")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Distribution (n={df['env_ratio_slope'].notna().sum()})")
    axes[2].axhline(0, color="gray", ls="--")

    fig.suptitle(f"{station} {year}: L1/L2C envelope ratio slope", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_{year}_envelope_ratio.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print(f"\n{'='*60}")
    print(f"Real data envelope ratio: {station} {year}")
    print(f"{'='*60}")
    print(f"GPS L1+L2C arcs analyzed: {len(df)}")
    print(f"Slope: median={df['env_ratio_slope'].median():.4f}, "
          f"std={df['env_ratio_slope'].std():.4f}")

    if df["v3_state"].nunique() > 1:
        print(f"\nBy state:")
        for state in sorted(df["v3_state"].unique()):
            sub = df[df["v3_state"] == state]["env_ratio_slope"].dropna()
            if len(sub) >= 5:
                print(f"  {state:>15s}: median={sub.median():.4f}, "
                      f"std={sub.std():.4f}, n={len(sub)}")

    df.to_parquet(out_dir / f"{station}_{year}_envelope_ratio.parquet", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Multi-frequency envelope ratio diagnostic"
    )
    parser.add_argument("--station", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Only run synthetic test")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = PROJECT_ROOT / "results_annual"
    if args.station:
        out_dir = out_dir / args.station / "diagnostics"
    else:
        out_dir = out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Synthetic test (always)
    syn_results = run_synthetic_test(out_dir)

    # Phase 2: Real data (if station provided)
    if not args.synthetic_only and args.station and args.year:
        run_real_data_test(args.station, args.year, out_dir)


if __name__ == "__main__":
    main()
