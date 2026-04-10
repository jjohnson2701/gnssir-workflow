# ABOUTME: Tests for baseline period detection (auto, user-months, user-doy)
# ABOUTME: Validates auto-detection on synthetic data with known calm period

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def _make_synthetic_daily_features(n_days=365, calm_start=150, calm_end=243):
    """Create synthetic daily features with a known calm period.

    The calm period (calm_start to calm_end) has low variance.
    Outside the calm period, features have higher variance and shifted means.
    """
    rng = np.random.RandomState(42)
    doys = np.arange(1, n_days + 1)

    data = {"doy": doys, "n_arcs": np.zeros(n_days)}
    feature_names = ["clr_med", "gamma_med", "af_med", "vs_med"]

    for feat in feature_names:
        values = np.zeros(n_days)
        for i, doy in enumerate(doys):
            if calm_start <= doy <= calm_end:
                # Calm period: low variance, stable mean
                values[i] = 5.0 + rng.normal(0, 0.3)
                data["n_arcs"][i] = 40 + rng.randint(-5, 5)
            else:
                # Active period: higher variance, shifted mean
                values[i] = 5.0 + 3.0 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1.5)
                data["n_arcs"][i] = 20 + rng.randint(-10, 10)
        data[feat] = values

    # Ensure n_arcs is positive
    data["n_arcs"] = np.maximum(data["n_arcs"], 5)

    return pd.DataFrame(data)


class TestAutoDetection:
    """Test automatic baseline detection on synthetic data."""

    def test_finds_calm_period(self):
        from scripts.baseline_detection import detect_baseline

        df = _make_synthetic_daily_features(calm_start=150, calm_end=243)
        result = detect_baseline(df, method="auto")

        assert result is not None
        assert result["method"] == "auto"
        assert result["provisional"] is True
        # Should detect a window overlapping significantly with the calm period
        assert result["start_doy"] >= 100  # not too early
        assert result["end_doy"] <= 280  # not too late
        assert result["n_days"] >= 30  # at least minimum window

    def test_returns_quality_metrics(self):
        from scripts.baseline_detection import detect_baseline

        df = _make_synthetic_daily_features()
        result = detect_baseline(df)

        assert "mean_n_arcs" in result
        assert result["mean_n_arcs"] > 0
        assert "n_days" in result
        assert result["n_days"] > 0


class TestUserSpecified:
    """Test user-specified baseline methods."""

    def test_user_doy_range(self):
        from scripts.baseline_detection import detect_baseline

        df = _make_synthetic_daily_features()
        result = detect_baseline(df, method="user_doy", doy_range=(100, 200))

        assert result is not None
        assert result["method"] == "user_doy"
        assert result["provisional"] is False
        assert result["start_doy"] == 100
        assert result["end_doy"] == 200

    def test_user_months(self):
        from scripts.baseline_detection import detect_baseline

        df = _make_synthetic_daily_features()
        # Add date column for month extraction
        df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["doy"] - 1, unit="D")
        result = detect_baseline(df, method="user_months", months=[6, 7, 8])

        assert result is not None
        assert result["method"] == "user_months"
        assert result["provisional"] is False
        assert result["n_days"] > 0

    def test_user_overrides_auto(self):
        """User-specified baseline should not trigger auto-detection."""
        from scripts.baseline_detection import detect_baseline

        df = _make_synthetic_daily_features()
        result = detect_baseline(df, method="user_doy", doy_range=(1, 30))

        assert result["start_doy"] == 1
        assert result["end_doy"] == 30
        assert result["method"] == "user_doy"


class TestSaveLoad:
    """Test saving and loading baseline definitions."""

    def test_round_trip(self, tmp_path):
        from scripts.baseline_detection import save_baseline_definition, load_baseline_definition

        baseline_def = {
            "start_doy": 150,
            "end_doy": 243,
            "n_days": 94,
            "mean_n_arcs": 42.5,
            "method": "auto",
            "provisional": True,
        }

        save_baseline_definition(baseline_def, "TEST", 2024, tmp_path)
        loaded = load_baseline_definition("TEST", 2024, tmp_path)

        assert loaded is not None
        assert loaded["start_doy"] == 150
        assert loaded["end_doy"] == 243
        assert loaded["method"] == "auto"


class TestEdgeCases:
    """Test edge cases and degraded modes."""

    def test_insufficient_data(self):
        from scripts.baseline_detection import detect_baseline

        # Very short dataset — fewer days than minimum window
        df = pd.DataFrame({
            "doy": range(1, 15),
            "n_arcs": [20] * 14,
            "clr_med": np.random.randn(14),
        })
        result = detect_baseline(df)
        # Should still return something (fallback), or None
        # The key is it shouldn't crash
        assert result is None or result["n_days"] > 0

    def test_no_feature_columns(self):
        from scripts.baseline_detection import detect_baseline

        df = pd.DataFrame({"doy": range(1, 100), "n_arcs": [30] * 99})
        result = detect_baseline(df)
        assert result is None  # no features to analyze
