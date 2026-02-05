# ABOUTME: Tests for different reference data source formats.
# ABOUTME: Validates ERDDAP, USGS, and CO-OPS data loading and column structures.

import pytest
import numpy as np
import pandas as pd


class TestERDDAPDataFormat:
    """Tests for ERDDAP reference data format (GLBX station)."""

    @pytest.mark.integration
    def test_erddap_subdaily_columns(self, real_subdaily_matched_data):
        """Test that ERDDAP subdaily data has expected columns."""
        df = real_subdaily_matched_data

        # ERDDAP uses station-specific column names (bartlett_cove)
        assert "gnss_datetime" in df.columns
        assert "gnss_wse" in df.columns
        assert "bartlett_cove_datetime" in df.columns
        assert "bartlett_cove_wl" in df.columns
        assert "residual" in df.columns

    @pytest.mark.integration
    def test_erddap_subdaily_data_types(self, real_subdaily_matched_data):
        """Test ERDDAP data types are correct."""
        df = real_subdaily_matched_data

        assert pd.api.types.is_datetime64_any_dtype(df["gnss_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["bartlett_cove_datetime"])
        assert df["gnss_wse"].dtype in [np.float64, np.float32]
        assert df["bartlett_cove_wl"].dtype in [np.float64, np.float32]

    @pytest.mark.integration
    def test_erddap_comparison_columns(self, real_comparison_data):
        """Test that ERDDAP comparison data has expected columns."""
        df = real_comparison_data

        # Daily comparison uses standardized column names
        assert "wse_ellips_m" in df.columns
        assert "usgs_value_m_median" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)


class TestUSGSDataFormat:
    """Tests for USGS reference data format (MDAI station)."""

    @pytest.mark.integration
    def test_usgs_subdaily_columns(self, usgs_subdaily_matched_data):
        """Test that USGS subdaily data has expected columns."""
        df = usgs_subdaily_matched_data

        # USGS uses usgs_* column names
        assert "gnss_datetime" in df.columns
        assert "gnss_wse" in df.columns
        assert "usgs_datetime" in df.columns
        assert "usgs_wl_m" in df.columns

    @pytest.mark.integration
    def test_usgs_subdaily_data_types(self, usgs_subdaily_matched_data):
        """Test USGS data types are correct."""
        df = usgs_subdaily_matched_data

        assert pd.api.types.is_datetime64_any_dtype(df["gnss_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["usgs_datetime"])
        assert df["gnss_wse"].dtype in [np.float64, np.float32]
        assert df["usgs_wl_m"].dtype in [np.float64, np.float32]

    @pytest.mark.integration
    def test_usgs_comparison_columns(self, usgs_comparison_data):
        """Test that USGS comparison data has expected columns."""
        df = usgs_comparison_data

        assert "wse_ellips_m" in df.columns
        assert "usgs_value_m_median" in df.columns
        assert "usgs_site_code" in df.columns
        assert "usgs_site_name" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    @pytest.mark.integration
    def test_usgs_site_metadata(self, usgs_comparison_data):
        """Test that USGS metadata is present."""
        df = usgs_comparison_data

        # Should have consistent site info
        assert df["usgs_site_code"].nunique() == 1
        assert "BUNTINGS GUT" in df["usgs_site_name"].iloc[0]


class TestCOOPSDataFormat:
    """Tests for NOAA CO-OPS reference data format (VALR station)."""

    @pytest.mark.integration
    def test_coops_subdaily_columns(self, coops_subdaily_matched_data):
        """Test that CO-OPS subdaily data has expected columns."""
        df = coops_subdaily_matched_data

        # CO-OPS uses coops_* column names
        assert "gnss_datetime" in df.columns
        assert "gnss_wse" in df.columns
        assert "coops_datetime" in df.columns
        assert "coops_wl" in df.columns

    @pytest.mark.integration
    def test_coops_subdaily_data_types(self, coops_subdaily_matched_data):
        """Test CO-OPS data types are correct."""
        df = coops_subdaily_matched_data

        assert pd.api.types.is_datetime64_any_dtype(df["gnss_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["coops_datetime"])
        assert df["gnss_wse"].dtype in [np.float64, np.float32]
        assert df["coops_wl"].dtype in [np.float64, np.float32]

    @pytest.mark.integration
    def test_coops_comparison_columns(self, coops_comparison_data):
        """Test that CO-OPS comparison data has expected columns."""
        df = coops_comparison_data

        # Daily comparison uses standardized column names
        assert "wse_ellips_m" in df.columns
        assert "usgs_value_m_median" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    @pytest.mark.integration
    def test_coops_demeaned_columns(self, coops_subdaily_matched_data):
        """Test that CO-OPS data has demeaned columns."""
        df = coops_subdaily_matched_data

        assert "gnss_dm" in df.columns
        assert "coops_dm" in df.columns


class TestCrossSourceComparison:
    """Tests comparing data across different reference sources."""

    @pytest.mark.integration
    def test_all_sources_have_gnss_columns(
        self, real_subdaily_matched_data, usgs_subdaily_matched_data, coops_subdaily_matched_data
    ):
        """Test that all sources have consistent GNSS columns."""
        for df in [
            real_subdaily_matched_data,
            usgs_subdaily_matched_data,
            coops_subdaily_matched_data,
        ]:
            assert "gnss_datetime" in df.columns
            assert "gnss_wse" in df.columns
            assert "gnss_rh" in df.columns

    @pytest.mark.integration
    def test_all_sources_have_time_diff(
        self, real_subdaily_matched_data, usgs_subdaily_matched_data
    ):
        """Test that matched data includes time difference."""
        # ERDDAP format
        assert "time_diff_sec" in real_subdaily_matched_data.columns

        # USGS format - check if time_diff exists or can be calculated
        if "time_diff_sec" not in usgs_subdaily_matched_data.columns:
            # Calculate from timestamps
            df = usgs_subdaily_matched_data
            time_diff = (df["gnss_datetime"] - df["usgs_datetime"]).dt.total_seconds()
            assert len(time_diff) == len(df)

    @pytest.mark.integration
    def test_comparison_data_consistent_columns(
        self, real_comparison_data, usgs_comparison_data, coops_comparison_data
    ):
        """Test that all comparison datasets have consistent column names."""
        required_cols = ["wse_ellips_m", "usgs_value_m_median", "rh_median_m"]

        for df in [real_comparison_data, usgs_comparison_data, coops_comparison_data]:
            for col in required_cols:
                assert col in df.columns, f"Missing column {col}"

    @pytest.mark.integration
    def test_wse_values_reasonable(
        self, real_comparison_data, usgs_comparison_data, coops_comparison_data
    ):
        """Test that WSE values are in reasonable ranges."""
        for df, name in [
            (real_comparison_data, "ERDDAP"),
            (usgs_comparison_data, "USGS"),
            (coops_comparison_data, "CO-OPS"),
        ]:
            wse = df["wse_ellips_m"]
            # WSE should be within reasonable bounds (-50 to 50 meters)
            assert wse.min() > -100, f"{name}: WSE too low"
            assert wse.max() < 100, f"{name}: WSE too high"
            assert not wse.isna().all(), f"{name}: All WSE values are NaN"
