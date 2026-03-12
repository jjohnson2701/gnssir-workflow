# ABOUTME: Compares our RH aggregation pipeline against gnssrefl subdaily output.
# ABOUTME: Shows before/after with confidence annotations, gap detection, and datum reporting.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.subdaily_loader import (
    annotate_confidence,
    detect_observation_gaps,
    load_observation_times,
    load_subdaily_results,
)


# --- Station-specific configuration ---

STATION_CONFIG = {
    "GLBX": {
        "year": 2024,
        "antenna_height": -12.535,
        "ref_file": "bartlett_cove_2024_raw.csv",
        "ref_name": "Bartlett Cove",
        "ref_loader": "glbx_erddap",
        "matched_ref_col": "bartlett_cove_wl",
        "matched_ref_dm_col": "bartlett_cove_dm",
    },
    "NKAR": {
        "year": 2025,
        "antenna_height": 40.8208,
        "ref_file": "nuuk_godthaab_2025_raw.csv",
        "ref_name": "Nuuk (UHSLC)",
        "ref_loader": "nkar_uhslc",
        "matched_ref_col": "nuuk_wl",
        "matched_ref_dm_col": "nuuk_dm",
    },
}


def load_erddap_reference(ref_path, loader_type):
    """Load ERDDAP reference data with station-specific parsing."""
    df = pd.read_csv(ref_path, skiprows=1)
    if loader_type == "glbx_erddap":
        df.columns = ["time", "wl_navd88", "wl_msl", "wl_mllw", "wl_station_datum"]
    elif loader_type == "nkar_uhslc":
        df.columns = ["time", "sea_level_mm", "station_name", "uhslc_id"]
        df["wl_navd88"] = pd.to_numeric(df["sea_level_mm"], errors="coerce") * 0.001
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
    dt = pd.to_datetime(df["time"])
    df["datetime"] = dt if dt.dt.tz is not None else dt.dt.tz_localize("UTC")
    df = df.dropna(subset=["wl_navd88"])
    return df


def compute_demeaned_stats(series_a, series_b):
    """Compute correlation and RMSE between two demeaned series."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return {"correlation": np.nan, "rmse_demeaned": np.nan, "n": len(a)}
    a_dm = a - a.mean()
    b_dm = b - b.mean()
    correlation = np.corrcoef(a_dm, b_dm)[0, 1]
    rmse_demeaned = np.sqrt(np.mean((a_dm - b_dm) ** 2))
    return {"correlation": correlation, "rmse_demeaned": rmse_demeaned, "n": len(a)}


def match_by_nearest_time(df_gnss, df_ref, max_diff_sec=1800):
    """Match GNSS observations to reference by nearest timestamp."""
    gnss = df_gnss.sort_values("datetime").reset_index(drop=True)
    ref = df_ref[["datetime", "wl_navd88"]].sort_values("datetime").reset_index(drop=True)
    merged = pd.merge_asof(
        gnss,
        ref.rename(columns={"datetime": "ref_datetime", "wl_navd88": "ref_wl"}),
        left_on="datetime",
        right_on="ref_datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=max_diff_sec),
    )
    merged = merged.dropna(subset=["ref_wl"]).reset_index(drop=True)
    merged["time_diff_sec"] = (
        (merged["datetime"] - merged["ref_datetime"]).dt.total_seconds().abs()
    )
    merged = merged.rename(
        columns={"datetime": "gnss_datetime", "wse_ortho_m": "gnss_wse", "rh_m": "gnss_rh"}
    )
    return merged


def resample_to_daily(df, datetime_col, value_col):
    """Resample a time series to daily means."""
    df = df.copy()
    df["date"] = df[datetime_col].dt.date
    daily = df.groupby("date")[value_col].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def plot_spline_with_confidence(ax, spline_annotated, ref_matched, label_prefix,
                                color="#2980B9"):
    """Plot spline time series with solid/dashed for observed/interpolated."""
    df = spline_annotated.sort_values("datetime").copy()
    df["wse_dm"] = df["wse_ortho_m"] - df["wse_ortho_m"].mean()

    # Plot observed regions as solid, interpolated as dashed
    observed = df[~df["is_interpolated"]]
    interpolated = df[df["is_interpolated"]]

    ax.plot(observed["datetime"], observed["wse_dm"], "-", color=color,
            linewidth=1.2, alpha=0.9, label=f"{label_prefix} (observed)")

    if len(interpolated) > 0:
        ax.plot(interpolated["datetime"], interpolated["wse_dm"], "--", color=color,
                linewidth=1.0, alpha=0.5, label=f"{label_prefix} (interpolated)")


def shade_gaps(ax, gaps, ymin, ymax):
    """Shade observation gap regions on a plot."""
    for gap in gaps:
        ax.axvspan(gap["start"], gap["end"], alpha=0.1, color="red",
                   label=f"Gap ({gap['duration_hours']:.0f}h)" if gap == gaps[0] else None)


def plot_before_after(station, cfg, spline_annotated, spline_matched, our_matched,
                      our_daily, ref_df, gaps, results_dir):
    """Generate the before/after comparison figure."""
    year = cfg["year"]
    ref_name = cfg["ref_name"]
    ref_wl_col = cfg["matched_ref_col"]
    ref_dm_col = cfg["matched_ref_dm_col"]
    antenna_height = cfg["antenna_height"]

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # --- Row 0: Subdaily time series zoom (pick first 14 days with data) ---
    spline_sorted = spline_annotated.sort_values("datetime")
    zoom_start = spline_sorted["datetime"].iloc[0]
    zoom_end = zoom_start + pd.Timedelta(days=14)

    # Panel (0,0): BEFORE - Our raw retrievals
    ax = fig.add_subplot(gs[0, 0])
    zoom_ours = our_matched[
        (our_matched["gnss_datetime"] >= zoom_start) &
        (our_matched["gnss_datetime"] <= zoom_end)
    ].copy()

    if len(zoom_ours) > 0:
        zoom_ref_slice = ref_df[
            (ref_df["datetime"] >= zoom_start) & (ref_df["datetime"] <= zoom_end)
        ].copy()
        ref_dm = zoom_ref_slice["wl_navd88"] - ref_df["wl_navd88"].mean()
        ax.plot(zoom_ref_slice["datetime"], ref_dm, "-", color="black",
                linewidth=1.0, alpha=0.6, label=ref_name)
        our_dm = zoom_ours["gnss_wse"] - our_matched["gnss_wse"].mean()
        ax.scatter(zoom_ours["gnss_datetime"], our_dm, s=15, color="#E74C3C",
                   alpha=0.7, zorder=3, label="Raw retrievals")
    ax.set_title("BEFORE: Raw Retrievals (no corrections)", fontweight="bold")
    ax.set_ylabel("Demeaned WL (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.text(0.02, 0.02, f"Datum: ellipsoidal (WGS84)\nAntenna ht: {antenna_height} m",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Panel (0,1): AFTER - Subdaily spline with confidence
    ax = fig.add_subplot(gs[0, 1])
    zoom_spline = spline_annotated[
        (spline_annotated["datetime"] >= zoom_start) &
        (spline_annotated["datetime"] <= zoom_end)
    ].copy()

    if len(zoom_spline) > 0:
        zoom_ref_slice = ref_df[
            (ref_df["datetime"] >= zoom_start) & (ref_df["datetime"] <= zoom_end)
        ].copy()
        ref_dm = zoom_ref_slice["wl_navd88"] - ref_df["wl_navd88"].mean()
        ax.plot(zoom_ref_slice["datetime"], ref_dm, "-", color="black",
                linewidth=1.0, alpha=0.6, label=ref_name)
        plot_spline_with_confidence(ax, zoom_spline, None, "Subdaily spline")
        shade_gaps(ax, [g for g in gaps if g["start"] >= zoom_start and g["end"] <= zoom_end],
                   ax.get_ylim()[0], ax.get_ylim()[1])
    ax.set_title("AFTER: Subdaily Spline (IF+RHdot corrected)", fontweight="bold")
    ax.set_ylabel("Demeaned WL (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.text(0.02, 0.02, "Datum: orthometric (EGM96)\nSolid=observed, Dashed=interpolated",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Row 1: Full period daily comparison ---

    # Resample spline to daily for comparison
    spline_daily = resample_to_daily(spline_matched, "gnss_datetime", "gnss_wse")
    spline_daily.rename(columns={"gnss_wse": "spline_wse_daily"}, inplace=True)
    ref_daily = resample_to_daily(spline_matched, "ref_datetime", "ref_wl")
    ref_daily.rename(columns={"ref_wl": "ref_wl_daily"}, inplace=True)

    daily_merged = our_daily[["date", "wse_ellips_m"]].merge(
        spline_daily, on="date", how="inner"
    ).merge(ref_daily, on="date", how="inner")

    # Panel (1,0): BEFORE daily
    ax = fig.add_subplot(gs[1, 0])
    if len(daily_merged) > 0:
        our_dm = daily_merged["wse_ellips_m"] - daily_merged["wse_ellips_m"].mean()
        ref_dm = daily_merged["ref_wl_daily"] - daily_merged["ref_wl_daily"].mean()
        ax.plot(daily_merged["date"], ref_dm, "k-", linewidth=1.5, alpha=0.7,
                label=ref_name)
        ax.plot(daily_merged["date"], our_dm, "o-", color="#E74C3C", markersize=5,
                linewidth=1, alpha=0.8, label="Our daily median")
        our_daily_stats = compute_demeaned_stats(
            daily_merged["wse_ellips_m"], daily_merged["ref_wl_daily"])
        ax.set_title(f"BEFORE: Daily Median WSE\n"
                     f"r={our_daily_stats['correlation']:.3f}, "
                     f"RMSE={our_daily_stats['rmse_demeaned']:.3f} m, "
                     f"N={our_daily_stats['n']}", fontweight="bold")
    ax.set_ylabel("Demeaned WL (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Panel (1,1): AFTER daily
    ax = fig.add_subplot(gs[1, 1])
    if len(daily_merged) > 0:
        sp_dm = daily_merged["spline_wse_daily"] - daily_merged["spline_wse_daily"].mean()
        ref_dm = daily_merged["ref_wl_daily"] - daily_merged["ref_wl_daily"].mean()
        ax.plot(daily_merged["date"], ref_dm, "k-", linewidth=1.5, alpha=0.7,
                label=ref_name)
        ax.plot(daily_merged["date"], sp_dm, "s-", color="#2980B9", markersize=5,
                linewidth=1, alpha=0.8, label="Subdaily daily avg")
        sp_daily_stats = compute_demeaned_stats(
            daily_merged["spline_wse_daily"], daily_merged["ref_wl_daily"])
        ax.set_title(f"AFTER: Subdaily Daily Average WSE\n"
                     f"r={sp_daily_stats['correlation']:.3f}, "
                     f"RMSE={sp_daily_stats['rmse_demeaned']:.3f} m, "
                     f"N={sp_daily_stats['n']}", fontweight="bold")
    ax.set_ylabel("Demeaned WL (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # --- Row 2: Scatter plots ---

    # Panel (2,0): BEFORE scatter
    ax = fig.add_subplot(gs[2, 0])
    if ref_dm_col in our_matched.columns:
        gnss_dm = our_matched["gnss_dm"]
        ref_dm = our_matched[ref_dm_col]
        ax.scatter(ref_dm, gnss_dm, s=4, alpha=0.3, color="#E74C3C")
        stats = compute_demeaned_stats(our_matched["gnss_wse"], our_matched[ref_wl_col])
        lims = [min(ref_dm.min(), gnss_dm.min()), max(ref_dm.max(), gnss_dm.max())]
        ax.plot(lims, lims, "k--", linewidth=0.5)
        ax.set_title(f"BEFORE: Subdaily Scatter\n"
                     f"r={stats['correlation']:.4f}, RMSE={stats['rmse_demeaned']:.3f} m, "
                     f"N={stats['n']}", fontweight="bold")
    ax.set_xlabel(f"{ref_name} demeaned (m)")
    ax.set_ylabel("GNSS-IR demeaned (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Panel (2,1): AFTER scatter
    ax = fig.add_subplot(gs[2, 1])
    if len(spline_matched) > 0:
        sp_dm = spline_matched["gnss_wse"] - spline_matched["gnss_wse"].mean()
        ref_dm = spline_matched["ref_wl"] - spline_matched["ref_wl"].mean()
        ax.scatter(ref_dm, sp_dm, s=4, alpha=0.3, color="#2980B9")
        stats = compute_demeaned_stats(spline_matched["gnss_wse"], spline_matched["ref_wl"])
        lims = [min(ref_dm.min(), sp_dm.min()), max(ref_dm.max(), sp_dm.max())]
        ax.plot(lims, lims, "k--", linewidth=0.5)
        ax.set_title(f"AFTER: Subdaily Spline Scatter\n"
                     f"r={stats['correlation']:.4f}, RMSE={stats['rmse_demeaned']:.3f} m, "
                     f"N={stats['n']}", fontweight="bold")
    ax.set_xlabel(f"{ref_name} demeaned (m)")
    ax.set_ylabel("Spline WSE demeaned (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # --- Gap summary annotation ---
    gap_text = f"{len(gaps)} observation gaps > 6h detected"
    if gaps:
        max_gap = max(g["duration_hours"] for g in gaps)
        gap_text += f" (longest: {max_gap:.1f}h)"

    fig.suptitle(
        f"{station} {year}: Pipeline Comparison\n"
        f"Before (raw median, ellipsoidal) vs After (subdaily spline, orthometric EGM96)\n"
        f"{gap_text}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    out_path = results_dir / f"{station}_{year}_pipeline_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.close()
    return out_path


def run_station(station):
    """Run the full comparison for one station."""
    if station not in STATION_CONFIG:
        print(f"ERROR: No config for {station}. Available: {list(STATION_CONFIG.keys())}")
        sys.exit(1)

    cfg = STATION_CONFIG[station]
    year = cfg["year"]
    station_lower = station.lower()
    results_dir = PROJECT_ROOT / "results_annual" / station
    files_dir = (
        PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code" / "Files" / station_lower
    )

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPARISON: {station} {year}")
    print(f"{'='*70}")

    # --- Load subdaily spline with confidence ---
    spline_path = files_dir / f"{station_lower}_spline_out.txt"
    if not spline_path.exists():
        print(f"ERROR: Run 'subdaily {station_lower} {year}' first.")
        return
    spline_df = load_subdaily_results(spline_path)
    print(f"  Subdaily spline: {len(spline_df)} records, "
          f"{spline_df['datetime'].min().date()} to {spline_df['datetime'].max().date()}")

    # Load observation times for confidence annotation
    if_path = files_dir / f"{station_lower}_{year}_subdaily_edit.txt.withrhdotIF"
    if if_path.exists():
        obs_times = load_observation_times(if_path)
        print(f"  Observations: {len(obs_times)} retrievals")
        spline_annotated = annotate_confidence(spline_df, obs_times, threshold_hours=3.0)
        gaps = detect_observation_gaps(obs_times, min_gap_hours=6.0)
        n_interp = spline_annotated["is_interpolated"].sum()
        pct_interp = 100 * n_interp / len(spline_annotated)
        print(f"  Confidence: {n_interp}/{len(spline_annotated)} spline points "
              f"interpolated ({pct_interp:.1f}%)")
        print(f"  Gaps > 6h: {len(gaps)}")
        for g in gaps[:5]:
            print(f"    {g['start'].strftime('%Y-%m-%d %H:%M')} to "
                  f"{g['end'].strftime('%Y-%m-%d %H:%M')} ({g['duration_hours']:.1f}h)")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more")
    else:
        print(f"  WARNING: No IF-corrected file found, skipping confidence annotation")
        spline_annotated = spline_df.copy()
        spline_annotated["nearest_obs_hours"] = 0.0
        spline_annotated["is_interpolated"] = False
        gaps = []

    # --- Load ERDDAP reference ---
    ref_path = results_dir / cfg["ref_file"]
    if not ref_path.exists():
        print(f"ERROR: Reference not found at {ref_path}")
        return
    ref_df = load_erddap_reference(ref_path, cfg["ref_loader"])
    print(f"  Reference ({cfg['ref_name']}): {len(ref_df)} records")

    # --- Load our pipeline data ---
    daily_path = results_dir / f"{station}_{year}_combined_rh.csv"
    our_daily = pd.read_csv(daily_path)
    our_daily["date"] = pd.to_datetime(our_daily["date"])
    our_daily["wse_ellips_m"] = cfg["antenna_height"] - our_daily["rh_median_m"]
    print(f"  Our daily: {len(our_daily)} days")

    matched_path = results_dir / f"{station}_{year}_subdaily_matched.csv"
    if matched_path.exists():
        our_matched = pd.read_csv(matched_path)
        our_matched["gnss_datetime"] = pd.to_datetime(
            our_matched["gnss_datetime"], format="ISO8601", utc=True)
        print(f"  Our subdaily matched: {len(our_matched)} records")
    else:
        print(f"  WARNING: No subdaily matched file, skipping retrieval comparison")
        our_matched = pd.DataFrame()

    # --- Match spline to reference ---
    print(f"\n  Matching subdaily spline to {cfg['ref_name']}...")
    spline_matched = match_by_nearest_time(spline_annotated, ref_df)
    print(f"  Matched: {len(spline_matched)} of {len(spline_annotated)} spline points")

    # --- Statistics ---
    print(f"\n  [STATISTICS]")
    if len(our_matched) > 0 and cfg["matched_ref_col"] in our_matched.columns:
        our_stats = compute_demeaned_stats(
            our_matched["gnss_wse"], our_matched[cfg["matched_ref_col"]])
        print(f"  Our subdaily retrievals: r={our_stats['correlation']:.4f}, "
              f"RMSE={our_stats['rmse_demeaned']:.3f} m (N={our_stats['n']})")

    if len(spline_matched) > 0:
        sp_stats = compute_demeaned_stats(
            spline_matched["gnss_wse"], spline_matched["ref_wl"])
        print(f"  Subdaily spline:         r={sp_stats['correlation']:.4f}, "
              f"RMSE={sp_stats['rmse_demeaned']:.3f} m (N={sp_stats['n']})")

    # --- Plot ---
    plot_before_after(
        station, cfg, spline_annotated, spline_matched, our_matched,
        our_daily, ref_df, gaps, results_dir,
    )


def main():
    stations = ["GLBX"]
    if len(sys.argv) > 1:
        stations = sys.argv[1:]

    for station in stations:
        run_station(station)


if __name__ == "__main__":
    main()
