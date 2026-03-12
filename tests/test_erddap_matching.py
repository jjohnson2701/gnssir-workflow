# ABOUTME: Tests for ERDDAP matching including spline-based comparison.
# ABOUTME: Validates that subdaily spline output is matched against reference data.

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMatchSplineToReference:
    """Tests for matching subdaily spline output to reference data."""

    @pytest.mark.unit
    def test_match_produces_required_columns(self):
        """Spline-matched output should have spline WSE and reference columns."""
        from scripts.generate_erddap_matched import match_spline_to_reference

        # Spline at 30-min intervals
        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "wse_ortho_m": np.sin(np.linspace(0, 4 * np.pi, 48)) + 1.0,
            "datum": "orthometric_EGM96",
        })

        # Reference at 6-min intervals
        ref_times = pd.date_range("2024-01-01", periods=240, freq="6min", tz="UTC")
        ref_df = pd.DataFrame({
            "datetime": ref_times,
            "wl": np.sin(np.linspace(0, 4 * np.pi, 240)) + 1.0,
        })

        matched = match_spline_to_reference(spline_df, ref_df, ref_name="test")
        assert "spline_wse" in matched.columns
        assert "spline_datetime" in matched.columns
        assert "test_wl" in matched.columns
        assert "datum" in matched.columns

    @pytest.mark.unit
    def test_match_computes_demeaned_values(self):
        """Matched output should include demeaned spline and reference columns."""
        from scripts.generate_erddap_matched import match_spline_to_reference

        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        signal = np.sin(np.linspace(0, 4 * np.pi, 48))
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "wse_ortho_m": signal + 10.0,  # offset from reference
            "datum": "orthometric_EGM96",
        })

        ref_times = pd.date_range("2024-01-01", periods=240, freq="6min", tz="UTC")
        ref_df = pd.DataFrame({
            "datetime": ref_times,
            "wl": np.sin(np.linspace(0, 4 * np.pi, 240)) + 1.0,
        })

        matched = match_spline_to_reference(spline_df, ref_df, ref_name="test")
        assert "spline_dm" in matched.columns
        assert "test_dm" in matched.columns

        # Demeaned values should be close since same signal shape
        correlation = matched["spline_dm"].corr(matched["test_dm"])
        assert correlation > 0.95

    @pytest.mark.unit
    def test_match_respects_time_tolerance(self):
        """Points with no reference within tolerance should be excluded."""
        from scripts.generate_erddap_matched import match_spline_to_reference

        # Spline covering 24 hours
        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "wse_ortho_m": np.ones(48),
            "datum": "orthometric_EGM96",
        })

        # Reference only for first 6 hours
        ref_times = pd.date_range("2024-01-01", periods=60, freq="6min", tz="UTC")
        ref_df = pd.DataFrame({
            "datetime": ref_times,
            "wl": np.ones(60),
        })

        matched = match_spline_to_reference(
            spline_df, ref_df, ref_name="test", max_time_diff_min=30,
        )
        # Only spline points within first ~6.5 hours should match
        assert len(matched) < 48
        assert len(matched) >= 12  # At least 6 hours worth
