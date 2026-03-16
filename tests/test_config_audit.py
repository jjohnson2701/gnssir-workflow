# ABOUTME: Tests for station configuration audit analysis functions.
# ABOUTME: Validates per-frequency stats, delTmax tradeoffs, RHdot impact, and Nyquist checks.

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestComputeFrequencyStats:
    """Tests for per-frequency retrieval statistics."""

    @pytest.mark.unit
    def test_returns_stats_for_each_frequency(self):
        """Should return one row per unique frequency."""
        from scripts.audit_station_config import compute_frequency_stats

        df = pd.DataFrame({
            "freq": [1, 1, 1, 5, 5, 20],
            "Amp": [10.0, 12.0, 11.0, 8.0, 9.0, 15.0],
            "PkNoise": [3.0, 3.5, 2.8, 2.0, 2.5, 4.0],
            "RH": [7.0, 7.1, 6.9, 7.5, 7.3, 7.0],
        })

        result = compute_frequency_stats(df)
        assert len(result) == 3
        assert set(result.index) == {1, 5, 20}

    @pytest.mark.unit
    def test_count_is_correct(self):
        """Count should reflect number of retrievals per frequency."""
        from scripts.audit_station_config import compute_frequency_stats

        df = pd.DataFrame({
            "freq": [1, 1, 1, 1, 5, 5],
            "Amp": [10.0] * 6,
            "PkNoise": [3.0] * 6,
            "RH": [7.0] * 6,
        })

        result = compute_frequency_stats(df)
        assert result.loc[1, "count"] == 4
        assert result.loc[5, "count"] == 2

    @pytest.mark.unit
    def test_rh_std_reflects_scatter(self):
        """Frequency with more RH scatter should have higher std."""
        from scripts.audit_station_config import compute_frequency_stats

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "freq": [1] * n + [5] * n,
            "Amp": [10.0] * (2 * n),
            "PkNoise": [3.0] * (2 * n),
            "RH": np.concatenate([
                np.random.normal(7.0, 0.05, n),  # tight
                np.random.normal(7.0, 0.50, n),  # scattered
            ]),
        })

        result = compute_frequency_stats(df)
        assert result.loc[5, "rh_std"] > result.loc[1, "rh_std"]

    @pytest.mark.unit
    def test_includes_residual_stats_when_available(self):
        """If rh_residual column exists, should include residual RMSE."""
        from scripts.audit_station_config import compute_frequency_stats

        df = pd.DataFrame({
            "freq": [1, 1, 5, 5],
            "Amp": [10.0] * 4,
            "PkNoise": [3.0] * 4,
            "RH": [7.0] * 4,
            "rh_residual": [0.01, -0.02, 0.10, -0.15],
        })

        result = compute_frequency_stats(df)
        assert "residual_rmse" in result.columns
        assert result.loc[5, "residual_rmse"] > result.loc[1, "residual_rmse"]


class TestComputeDelTmaxTradeoff:
    """Tests for arc duration vs quality tradeoff analysis."""

    @pytest.mark.unit
    def test_returns_row_per_threshold(self):
        """Should return one row per delTmax threshold."""
        from scripts.audit_station_config import compute_delTmax_tradeoff

        df = pd.DataFrame({
            "DelT": [5.0, 10.0, 20.0, 40.0, 60.0],
            "rh_residual": [0.01, 0.02, 0.05, 0.10, 0.20],
        })

        thresholds = [15.0, 30.0, 50.0, 75.0]
        result = compute_delTmax_tradeoff(df, thresholds)
        assert len(result) == 4
        assert list(result["delTmax"]) == thresholds

    @pytest.mark.unit
    def test_lower_threshold_retains_fewer_points(self):
        """Stricter delTmax should retain fewer data points."""
        from scripts.audit_station_config import compute_delTmax_tradeoff

        df = pd.DataFrame({
            "DelT": [5.0, 10.0, 20.0, 40.0, 60.0],
            "rh_residual": [0.01, 0.02, 0.05, 0.10, 0.20],
        })

        thresholds = [15.0, 75.0]
        result = compute_delTmax_tradeoff(df, thresholds)
        strict = result[result["delTmax"] == 15.0].iloc[0]
        loose = result[result["delTmax"] == 75.0].iloc[0]
        assert strict["n_retained"] < loose["n_retained"]

    @pytest.mark.unit
    def test_stricter_threshold_improves_rmse(self):
        """Removing long arcs with large residuals should lower RMSE."""
        from scripts.audit_station_config import compute_delTmax_tradeoff

        # Short arcs have small residuals, long arcs have large residuals
        df = pd.DataFrame({
            "DelT": [5.0, 5.0, 5.0, 10.0, 10.0, 60.0, 60.0],
            "rh_residual": [0.01, -0.01, 0.02, 0.03, -0.02, 0.30, -0.25],
        })

        thresholds = [15.0, 75.0]
        result = compute_delTmax_tradeoff(df, thresholds)
        strict_rmse = result[result["delTmax"] == 15.0].iloc[0]["rmse"]
        loose_rmse = result[result["delTmax"] == 75.0].iloc[0]["rmse"]
        assert strict_rmse < loose_rmse


class TestComputeRhdotImpact:
    """Tests for RHdot+IF correction impact analysis."""

    @pytest.mark.unit
    def test_returns_expected_keys(self):
        """Result should contain before/after statistics."""
        from scripts.audit_station_config import compute_rhdot_impact

        df_raw = pd.DataFrame({
            "MJD": [60310.0, 60310.1, 60310.2],
            "sat": [1, 2, 3],
            "freq": [1, 1, 5],
            "RH": [7.0, 7.2, 6.8],
        })
        df_corrected = pd.DataFrame({
            "MJD": [60310.0, 60310.1, 60310.2],
            "sat": [1, 2, 3],
            "freq": [1, 1, 5],
            "rh_if_corrected": [6.98, 7.18, 6.82],
            "rhdot_correction": [-0.01, -0.01, 0.01],
        })

        result = compute_rhdot_impact(df_raw, df_corrected)
        assert "raw_rh_std" in result
        assert "corrected_rh_std" in result
        assert "median_correction_magnitude" in result
        assert "n_matched" in result

    @pytest.mark.unit
    def test_corrections_reduce_scatter(self):
        """Corrected RH should have less scatter when corrections are meaningful."""
        from scripts.audit_station_config import compute_rhdot_impact

        np.random.seed(42)
        n = 200
        true_rh = 7.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
        noise = np.random.normal(0, 0.1, n)
        bias = 0.15 * np.sin(np.linspace(0, 8 * np.pi, n))  # systematic error

        df_raw = pd.DataFrame({
            "MJD": np.linspace(60310, 60312, n),
            "sat": np.tile(np.arange(1, 11), n // 10),
            "freq": np.tile([1, 1, 5, 5, 20], n // 5),
            "RH": true_rh + noise + bias,
        })
        df_corrected = pd.DataFrame({
            "MJD": np.linspace(60310, 60312, n),
            "sat": np.tile(np.arange(1, 11), n // 10),
            "freq": np.tile([1, 1, 5, 5, 20], n // 5),
            "rh_if_corrected": true_rh + noise,  # bias removed
            "rhdot_correction": -bias,
        })

        result = compute_rhdot_impact(df_raw, df_corrected)
        assert result["corrected_rh_std"] < result["raw_rh_std"]

    @pytest.mark.unit
    def test_per_frequency_breakdown(self):
        """Should include per-frequency correction statistics."""
        from scripts.audit_station_config import compute_rhdot_impact

        df_raw = pd.DataFrame({
            "MJD": [60310.0, 60310.1, 60310.2, 60310.3],
            "sat": [1, 2, 3, 4],
            "freq": [1, 1, 5, 5],
            "RH": [7.0, 7.2, 6.8, 7.1],
        })
        df_corrected = pd.DataFrame({
            "MJD": [60310.0, 60310.1, 60310.2, 60310.3],
            "sat": [1, 2, 3, 4],
            "freq": [1, 1, 5, 5],
            "rh_if_corrected": [6.99, 7.19, 6.82, 7.08],
            "rhdot_correction": [-0.005, -0.005, 0.01, -0.01],
        })

        result = compute_rhdot_impact(df_raw, df_corrected)
        assert "per_freq" in result
        assert 1 in result["per_freq"]
        assert 5 in result["per_freq"]


class TestCheckNyquistAdequacy:
    """Tests for sample rate vs reflector height Nyquist check."""

    @pytest.mark.unit
    def test_1hz_adequate_for_low_rh(self):
        """1 Hz sampling should be adequate for RH < 15m."""
        from scripts.audit_station_config import check_nyquist_adequacy

        result = check_nyquist_adequacy(max_rh_m=12.0, sample_rate_sec=1.0)
        assert result["is_adequate"] is True

    @pytest.mark.unit
    def test_30s_inadequate_for_high_rh(self):
        """30s sampling should be flagged for RH > 15m."""
        from scripts.audit_station_config import check_nyquist_adequacy

        result = check_nyquist_adequacy(max_rh_m=25.0, sample_rate_sec=30.0)
        assert result["is_adequate"] is False

    @pytest.mark.unit
    def test_15s_marginal_for_25m(self):
        """15s sampling for 25m RH should report low margin."""
        from scripts.audit_station_config import check_nyquist_adequacy

        result = check_nyquist_adequacy(max_rh_m=25.0, sample_rate_sec=15.0)
        # Should be adequate but with limited margin
        assert "max_resolvable_rh" in result
        assert result["max_resolvable_rh"] > 25.0  # should resolve, but...
        assert result["margin_pct"] < 100  # ...not with tons of headroom

    @pytest.mark.unit
    def test_returns_expected_keys(self):
        """Result dict should have standard keys."""
        from scripts.audit_station_config import check_nyquist_adequacy

        result = check_nyquist_adequacy(max_rh_m=10.0, sample_rate_sec=15.0)
        assert "max_resolvable_rh" in result
        assert "is_adequate" in result
        assert "margin_pct" in result
        assert "sample_rate_sec" in result
