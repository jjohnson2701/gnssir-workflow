# ABOUTME: Tests for validation tab column detection, daily aggregation, and bias computation.
# ABOUTME: Covers all three station column formats (ERDDAP, USGS, CO-OPS) and missing-column edge cases.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_components.tabs.validation_tab import (
    _detect_columns,
    _aggregate_to_daily,
    _compute_stats,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_subdaily(columns, n=48, seed=42):
    """Build a synthetic subdaily_matched DataFrame with requested columns."""
    rng = np.random.default_rng(seed)
    base = datetime(2024, 6, 1, tzinfo=None)
    dt = [base + timedelta(hours=i) for i in range(n)]

    df = pd.DataFrame({"gnss_datetime": pd.to_datetime(dt, utc=True)})

    # Deterministic tidal signal so gnss_wse and ref track each other
    hours = np.arange(n)
    tidal = 0.5 * np.sin(2 * np.pi * hours / 12.42)

    if "gnss_wse" in columns:
        df["gnss_wse"] = 20.0 + tidal + rng.normal(0, 0.05, n)
    if "gnss_dm" in columns:
        df["gnss_dm"] = tidal + rng.normal(0, 0.05, n)
    if "gnss_wse_dm" in columns:
        df["gnss_wse_dm"] = tidal + rng.normal(0, 0.05, n)
    if "gnss_rh" in columns:
        df["gnss_rh"] = 5.0 - tidal + rng.normal(0, 0.05, n)

    # Reference columns — offset by a real datum shift
    datum_offset = -22.0  # like GLBX ellipsoidal-to-orthometric
    if "bartlett_cove_wl" in columns:
        df["bartlett_cove_wl"] = (20.0 + datum_offset) + tidal + rng.normal(0, 0.03, n)
    if "bartlett_cove_dm" in columns:
        df["bartlett_cove_dm"] = tidal + rng.normal(0, 0.03, n)
    if "bartlett_cove_datetime" in columns:
        df["bartlett_cove_datetime"] = df["gnss_datetime"]

    if "usgs_wl_m" in columns:
        df["usgs_wl_m"] = (20.0 + datum_offset) + tidal + rng.normal(0, 0.03, n)
    if "usgs_wl_dm" in columns:
        df["usgs_wl_dm"] = tidal + rng.normal(0, 0.03, n)
    if "usgs_datetime" in columns:
        df["usgs_datetime"] = df["gnss_datetime"]

    if "coops_wl" in columns:
        df["coops_wl"] = (20.0 + datum_offset) + tidal + rng.normal(0, 0.03, n)
    if "coops_dm" in columns:
        df["coops_dm"] = tidal + rng.normal(0, 0.03, n)
    if "coops_datetime" in columns:
        df["coops_datetime"] = df["gnss_datetime"]

    if "freq" in columns:
        df["freq"] = rng.choice([1, 2, 5, 20], n)
    if "satellite" in columns:
        df["satellite"] = rng.integers(1, 32, n)
    if "azimuth" in columns:
        df["azimuth"] = rng.uniform(0, 360, n)
    if "amplitude" in columns:
        df["amplitude"] = rng.uniform(5, 15, n)

    return df


@pytest.fixture
def glbx_df():
    """GLBX-style subdaily matched (ERDDAP, has freq/satellite/azimuth)."""
    return _make_subdaily([
        "gnss_wse", "gnss_rh", "gnss_dm",
        "bartlett_cove_wl", "bartlett_cove_dm", "bartlett_cove_datetime",
        "freq", "satellite", "azimuth", "amplitude",
    ])


@pytest.fixture
def fora_df():
    """FORA-style subdaily matched (USGS, no freq/satellite)."""
    return _make_subdaily([
        "gnss_wse", "gnss_rh", "gnss_wse_dm",
        "usgs_wl_m", "usgs_wl_dm", "usgs_datetime",
    ])


@pytest.fixture
def no_gnss_wse_df():
    """Hypothetical DataFrame missing gnss_wse column entirely."""
    return _make_subdaily([
        "gnss_dm",
        "bartlett_cove_wl", "bartlett_cove_dm", "bartlett_cove_datetime",
    ])


# ── _detect_columns ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDetectColumns:
    def test_glbx_erddap_columns(self, glbx_df):
        cols = _detect_columns(glbx_df)
        assert cols["gnss_wse"] == "gnss_wse"
        assert cols["gnss_dm"] == "gnss_dm"
        assert cols["ref_dm"] == "bartlett_cove_dm"
        assert cols["ref_wl"] == "bartlett_cove_wl"
        assert cols["ref_source"] == "ERDDAP"
        assert cols["freq"] == "freq"

    def test_fora_usgs_columns(self, fora_df):
        cols = _detect_columns(fora_df)
        assert cols["gnss_wse"] == "gnss_wse"
        assert cols["gnss_dm"] == "gnss_wse_dm"
        assert cols["ref_dm"] == "usgs_wl_dm"
        assert cols["ref_wl"] == "usgs_wl_m"
        assert cols["ref_source"] == "USGS"
        assert cols["freq"] is None

    def test_missing_gnss_wse(self, no_gnss_wse_df):
        cols = _detect_columns(no_gnss_wse_df)
        assert cols["gnss_wse"] is None

    def test_coops_detection(self):
        df = _make_subdaily([
            "gnss_wse", "gnss_dm",
            "coops_wl", "coops_dm", "coops_datetime",
        ])
        cols = _detect_columns(df)
        assert cols["ref_source"] == "CO-OPS"
        assert cols["ref_dm"] == "coops_dm"
        assert cols["ref_wl"] == "coops_wl"


# ── _aggregate_to_daily ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestAggregateToDaily:
    def test_daily_has_expected_columns(self, glbx_df):
        cols = _detect_columns(glbx_df)
        daily = _aggregate_to_daily(glbx_df, cols)
        assert "gnss_wse_median" in daily.columns
        assert "gnss_wse_std" in daily.columns
        assert "gnss_dm_median" in daily.columns
        assert "ref_wl_mean" in daily.columns
        assert "ref_dm_mean" in daily.columns
        assert "n_retrievals" in daily.columns
        assert "residual" in daily.columns

    def test_daily_without_gnss_wse(self, no_gnss_wse_df):
        cols = _detect_columns(no_gnss_wse_df)
        daily = _aggregate_to_daily(no_gnss_wse_df, cols)
        assert "gnss_wse_median" not in daily.columns
        assert "gnss_dm_median" in daily.columns
        assert "ref_dm_mean" in daily.columns

    def test_daily_row_count(self, glbx_df):
        """48 hours of data spanning 2 days → 2 daily rows."""
        cols = _detect_columns(glbx_df)
        daily = _aggregate_to_daily(glbx_df, cols)
        # 48 hourly points starting at midnight → 2 full days (day 0: hours 0-23, day 1: hours 24-47)
        assert len(daily) == 2


# ── _compute_stats ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeStats:
    def test_perfect_correlation(self):
        g = pd.Series([1.0, 2.0, 3.0, 4.0])
        stats = _compute_stats(g, g)
        assert stats["correlation"] == pytest.approx(1.0)
        assert stats["rmse"] == pytest.approx(0.0)
        assert stats["bias"] == pytest.approx(0.0)
        assert stats["n"] == 4

    def test_bias_nonzero(self):
        g = pd.Series([10.0, 20.0, 30.0])
        r = pd.Series([12.0, 22.0, 32.0])
        stats = _compute_stats(g, r)
        assert stats["bias"] == pytest.approx(-2.0)

    def test_handles_nans(self):
        g = pd.Series([1.0, np.nan, 3.0])
        r = pd.Series([1.0, 2.0, 3.0])
        stats = _compute_stats(g, r)
        assert stats["n"] == 2

    def test_too_few_points(self):
        g = pd.Series([1.0])
        r = pd.Series([2.0])
        stats = _compute_stats(g, r)
        assert np.isnan(stats["correlation"])


# ── Bias from absolute values (the critical correctness test) ─────────────


@pytest.mark.unit
class TestBiasComputation:
    def test_demeaned_bias_near_zero(self, glbx_df):
        """Bias computed from demeaned columns should be near zero (by construction)."""
        cols = _detect_columns(glbx_df)
        daily = _aggregate_to_daily(glbx_df, cols)
        stats = _compute_stats(daily["gnss_dm_median"], daily["ref_dm_mean"])
        # Demeaned bias should be small (noise-level), not a real datum offset
        assert abs(stats["bias"]) < 0.5

    def test_absolute_datum_offset_meaningful(self, glbx_df):
        """Datum offset from absolute values should reflect the real ~22m separation."""
        cols = _detect_columns(glbx_df)
        daily = _aggregate_to_daily(glbx_df, cols)
        # The offset should be large — roughly the datum_offset we baked in (-22.0)
        abs_bias = daily["gnss_wse_median"].mean() - daily["ref_wl_mean"].mean()
        assert abs(abs_bias - 22.0) < 1.0  # within 1m of the true offset

    def test_no_absolute_bias_when_gnss_wse_missing(self, no_gnss_wse_df):
        """When gnss_wse is missing, daily should not contain gnss_wse_median."""
        cols = _detect_columns(no_gnss_wse_df)
        daily = _aggregate_to_daily(no_gnss_wse_df, cols)
        assert "gnss_wse_median" not in daily.columns
