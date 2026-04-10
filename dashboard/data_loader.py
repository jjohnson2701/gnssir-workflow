# ABOUTME: Centralized data loading for the dashboard
# ABOUTME: All file I/O and caching in one place — tabs import from here

"""Dashboard data loading functions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.utils import PROJECT_ROOT


def load_stations_config():
    """Load station metadata from stations_config.json."""
    cfg_path = PROJECT_ROOT / "config" / "stations_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def list_available_stations():
    """Enumerate stations with available data (arc_table or per_arc parquets).

    Returns dict: {station_name: {years: [...], lat: float, lon: float}}
    """
    results_dir = PROJECT_ROOT / "results_annual"
    if not results_dir.exists():
        return {}

    stations = {}
    cfg = load_stations_config()

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        parquets = sorted(d.glob(f"{name}_*_arc_table.parquet"))
        if not parquets:
            parquets = sorted(d.glob(f"{name}_*_per_arc.parquet"))
        if not parquets:
            continue

        years = []
        for p in parquets:
            parts = p.stem.split("_")
            if len(parts) >= 2:
                try:
                    years.append(int(parts[1]))
                except ValueError:
                    continue

        if years:
            lat = cfg.get(name, {}).get("latitude_deg", 0)
            lon = cfg.get(name, {}).get("longitude_deg", 0)
            stations[name] = {"years": sorted(years), "lat": lat, "lon": lon}

    return stations


def load_per_arc(station, year):
    """Load Layer 1 arc-level data (arc_table preferred, per_arc fallback)."""
    for name in ["arc_table", "per_arc"]:
        p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "date" in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"])
                df["doy"] = df["date_dt"].dt.dayofyear
            if "freq_group" not in df.columns and "freq" in df.columns:
                freq_map = {1: "L1", 101: "L1", 201: "L1", 301: "L1",
                            2: "L2C", 20: "L2C", 102: "L2C", 302: "L2C",
                            5: "L5", 205: "L5",
                            206: "E6", 306: "E6", 208: "E6",
                            207: "B3", 307: "B3"}
                df["freq_group"] = df["freq"].map(freq_map).fillna("OTHER")
            return df
    return None


def load_v3(station, year):
    """Load v3 classification (ice_classification_v3 preferred)."""
    for name in ["ice_classification_v3", "ice_classification_v2", "ice_state"]:
        p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "v3_state" not in df.columns:
                if "state" in df.columns:
                    df["v3_state"] = df["state"]
                elif "ice_state" in df.columns:
                    df["v3_state"] = df["ice_state"]
            if "date" in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"])
                df["doy"] = df["date_dt"].dt.dayofyear
            return df
    return None


def load_snr_features(station, year):
    """Load SNR features — embedded in arc_table or standalone snr_features."""
    at = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_arc_table.parquet"
    if at.exists():
        df = pd.read_parquet(at)
        if "CLR" in df.columns:
            if "date" in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"])
                df["doy"] = df["date_dt"].dt.dayofyear
            return df
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_snr_features.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_smap(station, year):
    """Load SMAP soil moisture comparison data."""
    p = PROJECT_ROOT / "results_annual" / station / "smap" / f"{station}_{year}_smap_comparison.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_era5(station, year):
    """Load ERA5 temperature data."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_era5.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_v2(station, year):
    """Load v2 classification (Mahalanobis distances)."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_ice_classification_v2.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])
        df["doy"] = df["date_dt"].dt.dayofyear
    return df


def load_mahal_threshold(station):
    """Load Mahalanobis threshold from station config."""
    cfg = load_stations_config()
    scfg = cfg.get(station, {})
    cls = scfg.get("classification", {})
    return cls.get("threshold", 3.0)


def load_cross_station_summary():
    """Load cross-station summary parquet."""
    p = PROJECT_ROOT / "results_annual" / "cross_station_summary.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_daily_features(station, year):
    """Load Layer 2 daily features (aggregated from arc_table by feature_aggregator)."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_daily_features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])
        df["doy"] = df["date_dt"].dt.dayofyear
    return df


def load_prn_weights(station, year):
    """Load per-PRN discriminating power weights JSON."""
    p = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_prn_weights.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_s1_images(station):
    """List available S1 Fresnel zone images for a station.

    Returns list of dicts: [{date: "YYYY-MM-DD", vv_path: Path, vh_path: Path}, ...]
    """
    s1_dir = PROJECT_ROOT / "data" / station / "s1_fresnel"
    if not s1_dir.exists():
        return []

    images = []
    for tif in sorted(s1_dir.glob(f"{station}_*_s1_fresnel.tif")):
        parts = tif.stem.split("_")
        if len(parts) >= 3:
            date_str = parts[1]
            if len(date_str) == 8:
                date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                images.append({
                    "date": date_fmt,
                    "vv_path": tif,
                    "vh_path": s1_dir / tif.name.replace("_s1_fresnel.tif", "_s1_fresnel_vh.tif"),
                })
    return images


def build_quality_stats(per_arc):
    """Compute quality summary statistics from per-arc data."""
    if per_arc is None or per_arc.empty:
        return None

    stats = {
        "total_arcs": len(per_arc),
        "n_days": per_arc["doy"].nunique() if "doy" in per_arc.columns else 0,
        "arcs_per_day": len(per_arc) / max(1, per_arc["doy"].nunique()) if "doy" in per_arc.columns else 0,
        "rh_median": float(per_arc["RH"].median()) if "RH" in per_arc.columns else 0,
        "rh_std": float(per_arc["RH"].std()) if "RH" in per_arc.columns else 0,
    }

    if "freq_group" in per_arc.columns:
        freq_counts = per_arc["freq_group"].value_counts()
        total = len(per_arc)
        stats["freq_pcts"] = {f: c / total * 100 for f, c in freq_counts.items()}
    else:
        stats["freq_pcts"] = {}

    return stats
