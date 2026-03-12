# ABOUTME: Tests for the pipeline comparison script that compares our aggregation
# ABOUTME: against gnssrefl subdaily output using ERDDAP reference data.

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDemeanedComparison:
    """Tests for demeaned comparison statistics."""

    @pytest.mark.unit
    def test_demeaned_correlation(self):
        """Demeaned series should correlate when they track the same signal."""
        from scripts.compare_pipelines import compute_demeaned_stats

        # Simulated tidal signal with offset
        t = np.linspace(0, 10 * np.pi, 200)
        series_a = 2.0 * np.sin(t) + 10.0  # offset = 10
        series_b = 2.0 * np.sin(t) + 50.0  # offset = 50, same signal

        stats = compute_demeaned_stats(series_a, series_b)
        assert stats["correlation"] > 0.99
        assert stats["rmse_demeaned"] < 0.01

    @pytest.mark.unit
    def test_demeaned_stats_with_noise(self):
        """Noisy series should have lower correlation and higher RMSE."""
        from scripts.compare_pipelines import compute_demeaned_stats

        np.random.seed(42)
        t = np.linspace(0, 10 * np.pi, 200)
        series_a = 2.0 * np.sin(t)
        series_b = 2.0 * np.sin(t) + np.random.normal(0, 0.5, 200)

        stats = compute_demeaned_stats(series_a, series_b)
        assert 0.7 < stats["correlation"] < 1.0
        assert stats["rmse_demeaned"] > 0.1

    @pytest.mark.unit
    def test_demeaned_stats_insufficient_data(self):
        """Should handle series with fewer than 3 points gracefully."""
        from scripts.compare_pipelines import compute_demeaned_stats

        stats = compute_demeaned_stats([1.0, 2.0], [3.0, 4.0])
        assert np.isnan(stats["correlation"])
        assert stats["n"] == 2
