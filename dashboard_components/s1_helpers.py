# ABOUTME: Shared S1/SAR utilities for dashboard tabs (per-arc and ice classification).
# ABOUTME: Provides S1 index loading, thumbnail paths, scene matching, and Fresnel geometry.

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_s1_index(station: str) -> pd.DataFrame | None:
    """Load S1 Fresnel index CSV with parsed acquisition dates."""
    path = PROJECT_ROOT / "data" / station / "s1_fresnel" / f"{station}_s1_fresnel_index.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date_dt"] = pd.to_datetime(df["acquisition_date"])
    return df


def get_thumb_path(station: str, date_str: str) -> Path:
    """Return path to S1 thumbnail PNG for a given acquisition date string."""
    d = date_str.replace("-", "")
    return PROJECT_ROOT / "data" / station / "s1_fresnel" / "thumbnails" / f"{station}_{d}_thumb.png"


def find_nearest_s1_scene(station_id: str, target_date, s1_index: pd.DataFrame):
    """Find the S1 scene closest in time to target_date.

    Returns (row, days_offset) or (None, None) if index is empty.
    """
    if s1_index is None or s1_index.empty:
        return None, None
    s1 = s1_index.copy()
    if "date_dt" not in s1.columns:
        s1["date_dt"] = pd.to_datetime(s1["acquisition_date"])
    target_dt = pd.Timestamp(target_date)
    s1["offset"] = (s1["date_dt"] - target_dt).abs().dt.days
    best = s1.loc[s1["offset"].idxmin()]
    return best, int(best["offset"])


def compute_fresnel_radii(station: str, year: int):
    """Compute Fresnel zone inner/outer radii from per-arc data or config.

    Returns (inner_m, outer_m, method) or (None, None, None) if no data.
    """
    # Try per-arc data first (actual observed reflection distances)
    pa_path = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_per_arc.parquet"
    pa = None
    if pa_path.exists():
        try:
            pa = pd.read_parquet(pa_path)
        except Exception:
            pass
    if pa is not None:
        elev_mid = (pa["eminO"] + pa["emaxO"]) / 2.0
        refl_dist = pa["RH"] / np.tan(np.radians(elev_mid))
        # Use p2-p98 with 30m buffer (1 S1 pixel) on each side
        inner = max(0, refl_dist.quantile(0.02) - 30)
        outer = refl_dist.quantile(0.98) + 30
        return float(inner), float(outer), "per-arc p2-p98 +30m buffer"

    # Fallback: config-based
    cfg_path = PROJECT_ROOT / "config" / f"{station.lower()}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Widest range: maxH at lowest elev, minH at highest elev
        inner = max(0, cfg["minH"] / np.tan(np.radians(cfg["e2"])) - 30)
        outer = cfg["maxH"] / np.tan(np.radians(cfg["e1"])) + 30
        return float(inner), float(outer), "config (minH/maxH, e1/e2)"

    return None, None, None


def load_s1_matched(station: str, year: int) -> pd.DataFrame | None:
    """Load S1-GNSSIR matched parquet if it exists."""
    path = PROJECT_ROOT / "results_annual" / station / f"{station}_{year}_s1_gnssir_matched.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
