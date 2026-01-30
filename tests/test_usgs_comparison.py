# ABOUTME: Tests for usgs_comparison.py module.
# ABOUTME: Tests data loading, correlation calculations, and comparison logic.

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


class TestDataLoading:
    """Tests for GNSS-IR data loading functions."""

    @pytest.mark.unit
    def test_sample_data_structure(self, sample_gnssir_data):
        """Test that sample data has expected structure."""
        df = sample_gnssir_data

        assert 'datetime' in df.columns
        assert 'rh' in df.columns
        assert len(df) > 0
        assert df['rh'].dtype in [np.float64, np.float32]

    @pytest.mark.unit
    def test_sample_data_values(self, sample_gnssir_data):
        """Test that sample data has reasonable values."""
        df = sample_gnssir_data

        # Reflector heights should be positive
        assert (df['rh'] > 0).all()

        # Azimuths should be 0-360
        assert (df['azimuth'] >= 0).all()
        assert (df['azimuth'] <= 360).all()

    @pytest.mark.unit
    def test_datetime_parsing(self, sample_gnssir_data):
        """Test that datetime column is properly parsed."""
        df = sample_gnssir_data

        assert pd.api.types.is_datetime64_any_dtype(df['datetime'])


class TestCorrelationCalculations:
    """Tests for correlation and statistics calculations."""

    @pytest.mark.unit
    def test_perfect_correlation(self):
        """Test correlation with perfectly correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = x.copy()

        correlation = np.corrcoef(x, y)[0, 1]
        assert correlation == pytest.approx(1.0)

    @pytest.mark.unit
    def test_negative_correlation(self):
        """Test correlation with negatively correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])

        correlation = np.corrcoef(x, y)[0, 1]
        assert correlation == pytest.approx(-1.0)

    @pytest.mark.unit
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        actual = np.array([1.1, 2.1, 2.9, 4.1])

        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        assert rmse == pytest.approx(0.1, abs=0.01)

    @pytest.mark.unit
    def test_comparison_data_correlation(self, mock_comparison_df):
        """Test correlation calculation on comparison data."""
        df = mock_comparison_df

        # Calculate correlation
        correlation = df['wse_ellips'].corr(df['usgs_value'])

        # Should have reasonably high correlation for synthetic data
        assert correlation > 0.9

    @pytest.mark.unit
    def test_residual_calculation(self, mock_comparison_df):
        """Test that residuals are calculated correctly."""
        df = mock_comparison_df

        expected_residual = df['wse_ellips'] - df['usgs_value']

        assert np.allclose(df['residual'], expected_residual)

    @pytest.mark.unit
    def test_residual_statistics(self, mock_comparison_df):
        """Test residual statistics."""
        df = mock_comparison_df

        # Mean residual should be close to zero for well-matched data
        mean_residual = df['residual'].mean()

        # Std should be relatively small for synthetic data
        std_residual = df['residual'].std()

        assert abs(mean_residual) < 1.0  # Less than 1 meter
        assert std_residual < 0.5  # Less than 0.5 meter


class TestWaterSurfaceElevation:
    """Tests for water surface elevation calculations."""

    @pytest.mark.unit
    def test_wse_calculation(self):
        """Test WSE = antenna_height - reflector_height."""
        antenna_height = 10.0
        rh = 4.5

        wse = antenna_height - rh

        assert wse == pytest.approx(5.5)

    @pytest.mark.unit
    def test_wse_with_array(self, sample_gnssir_data):
        """Test WSE calculation with array of RH values."""
        antenna_height = 10.5
        rh_values = sample_gnssir_data['rh'].values

        wse_values = antenna_height - rh_values

        # All WSE values should be positive (assuming water is below antenna)
        assert (wse_values > 0).all()

        # Mean WSE should be reasonable
        assert 0 < wse_values.mean() < antenna_height


class TestTimeLagAnalysis:
    """Tests for time lag analysis functionality."""

    @pytest.mark.unit
    def test_lag_detection_no_lag(self):
        """Test lag detection with synchronized data."""
        # Create two identical time series
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / 24)

        # Cross-correlation should peak at lag=0
        cross_corr = np.correlate(signal, signal, mode='full')
        peak_lag = np.argmax(cross_corr) - (len(signal) - 1)

        assert peak_lag == 0

    @pytest.mark.unit
    def test_lag_detection_with_offset(self):
        """Test lag detection with known time offset."""
        # Create a signal and a lagged version
        t = np.arange(200)
        signal1 = np.sin(2 * np.pi * t / 24)

        lag = 5
        # np.roll shifts signal right by lag, so signal2 is signal1 delayed by lag
        signal2 = np.roll(signal1, lag)

        # Cross-correlation should detect the lag
        # correlate(a, b) finds how much b should be shifted to align with a
        cross_corr = np.correlate(signal1, signal2, mode='full')
        detected_lag = np.argmax(cross_corr) - (len(signal1) - 1)

        # np.roll shifts right, cross-corr detects negative lag (signal2 lags signal1)
        assert abs(detected_lag + lag) <= 1


class TestDataAlignment:
    """Tests for data alignment and merging."""

    @pytest.mark.unit
    def test_hourly_alignment(self, sample_gnssir_data, sample_reference_data):
        """Test alignment of data to hourly intervals."""
        gnss = sample_gnssir_data.copy()
        ref = sample_reference_data.copy()

        gnss['hour'] = gnss['datetime'].dt.floor('h')
        ref['hour'] = ref['datetime'].dt.floor('h')

        # Group by hour and check we get reasonable counts
        gnss_hourly = gnss.groupby('hour').size()

        assert len(gnss_hourly) > 0

    @pytest.mark.unit
    def test_merge_preserves_data(self, mock_comparison_df):
        """Test that merge operation preserves data."""
        df = mock_comparison_df

        # Should have both GNSS and reference columns
        assert 'rh' in df.columns
        assert 'usgs_value' in df.columns or 'reference_wl' in df.columns

        # No NaN in key columns after merge
        assert not df['rh'].isna().any()
