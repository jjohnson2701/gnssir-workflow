# ABOUTME: Tests for the subdaily spline output loader with confidence annotations.
# ABOUTME: Validates parsing, gap detection, and observation proximity classification.

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

GLBX_FILES_DIR = (
    project_root
    / "gnssrefl_data_workspace"
    / "refl_code"
    / "Files"
    / "glbx"
)


class TestLoadSplineOutput:
    """Tests for loading and parsing subdaily spline output."""

    @pytest.mark.unit
    def test_required_columns(self):
        """Loader output should have required columns."""
        from scripts.utils.subdaily_loader import load_subdaily_results

        spline_path = GLBX_FILES_DIR / "glbx_spline_out.txt"
        if not spline_path.exists():
            pytest.skip("GLBX subdaily output not available")

        df = load_subdaily_results(spline_path)
        required = ["datetime", "rh_m", "wse_ortho_m", "datum"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.unit
    def test_datum_is_orthometric(self):
        """WSE datum should be labeled orthometric (EGM96)."""
        from scripts.utils.subdaily_loader import load_subdaily_results

        spline_path = GLBX_FILES_DIR / "glbx_spline_out.txt"
        if not spline_path.exists():
            pytest.skip("GLBX subdaily output not available")

        df = load_subdaily_results(spline_path)
        assert (df["datum"] == "orthometric_EGM96").all()

    @pytest.mark.unit
    def test_gap_values_excluded(self):
        """999 gap-fill values should be excluded from output."""
        from scripts.utils.subdaily_loader import load_subdaily_results

        spline_path = GLBX_FILES_DIR / "glbx_spline_out.txt"
        if not spline_path.exists():
            pytest.skip("GLBX subdaily output not available")

        df = load_subdaily_results(spline_path)
        assert (df["rh_m"] < 900).all()
        assert (df["wse_ortho_m"] > -900).all()


class TestConfidenceAnnotation:
    """Tests for observation proximity and confidence flagging."""

    @pytest.mark.unit
    def test_annotate_adds_confidence_columns(self):
        """Annotation should add nearest_obs_hours and is_interpolated columns."""
        from scripts.utils.subdaily_loader import annotate_confidence

        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "rh_m": np.random.uniform(6, 9, 48),
            "wse_ortho_m": np.random.uniform(-23, -20, 48),
        })

        # Observations at hours 0, 3, 6, 9, 12, 15, 18, 21
        obs_times = pd.date_range("2024-01-01", periods=8, freq="3h", tz="UTC")

        result = annotate_confidence(spline_df, obs_times)
        assert "nearest_obs_hours" in result.columns
        assert "is_interpolated" in result.columns

    @pytest.mark.unit
    def test_points_near_observations_are_not_interpolated(self):
        """Points within threshold of an observation should be flagged as observed."""
        from scripts.utils.subdaily_loader import annotate_confidence

        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "rh_m": np.ones(48) * 7.0,
            "wse_ortho_m": np.ones(48) * -20.0,
        })

        # Observations every hour — all spline points should be within 0.5 hours
        obs_times = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")

        result = annotate_confidence(spline_df, obs_times, threshold_hours=1.0)
        assert not result["is_interpolated"].any()

    @pytest.mark.unit
    def test_gap_region_flagged_as_interpolated(self):
        """Points far from any observation should be flagged as interpolated."""
        from scripts.utils.subdaily_loader import annotate_confidence

        spline_times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
        spline_df = pd.DataFrame({
            "datetime": spline_times,
            "rh_m": np.ones(48) * 7.0,
            "wse_ortho_m": np.ones(48) * -20.0,
        })

        # Only 2 observations: one at start and one 20 hours later
        obs_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 20:00", tz="UTC"),
        ])

        result = annotate_confidence(spline_df, obs_times, threshold_hours=3.0)
        # Points in the middle (hours 4-16) should be interpolated
        # (hours 0-3 are near the 00:00 obs, hours 17-20 near the 20:00 obs)
        mid_points = result[(result["datetime"].dt.hour >= 4) &
                           (result["datetime"].dt.hour <= 16)]
        assert mid_points["is_interpolated"].all()


class TestLoadCorrectedRetrievals:
    """Tests for loading RHdot+IF corrected retrievals from .withrhdotIF file."""

    @pytest.mark.unit
    def test_corrected_retrievals_have_required_columns(self):
        """Corrected retrieval output should have all raw columns plus corrections."""
        from scripts.utils.subdaily_loader import load_corrected_retrievals

        if_path = GLBX_FILES_DIR / "glbx_2024_subdaily_edit.txt.withrhdotIF"
        if not if_path.exists():
            pytest.skip("GLBX withrhdotIF file not available")

        df = load_corrected_retrievals(if_path)
        required = [
            "year", "doy", "RH", "sat", "UTCtime", "Azim", "Amp",
            "eminO", "emaxO", "NumbOf", "freq", "rise", "EdotF",
            "PkNoise", "DelT", "MJD", "rh_rhdot_corrected",
            "rhdot_correction", "rh_if_corrected",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.unit
    def test_corrected_rh_differs_from_raw(self):
        """IF-corrected RH should differ from raw RH for most retrievals."""
        from scripts.utils.subdaily_loader import load_corrected_retrievals

        if_path = GLBX_FILES_DIR / "glbx_2024_subdaily_edit.txt.withrhdotIF"
        if not if_path.exists():
            pytest.skip("GLBX withrhdotIF file not available")

        df = load_corrected_retrievals(if_path)
        # At least some rows should have different raw vs corrected RH
        diff = (df["RH"] - df["rh_if_corrected"]).abs()
        assert (diff > 0.001).any(), "Expected corrected RH to differ from raw"

    @pytest.mark.unit
    def test_rhdot_correction_is_small(self):
        """RHdot corrections should be small (typically < 0.2m)."""
        from scripts.utils.subdaily_loader import load_corrected_retrievals

        if_path = GLBX_FILES_DIR / "glbx_2024_subdaily_edit.txt.withrhdotIF"
        if not if_path.exists():
            pytest.skip("GLBX withrhdotIF file not available")

        df = load_corrected_retrievals(if_path)
        assert df["rhdot_correction"].abs().max() < 0.5, "RHdot corrections unexpectedly large"


class TestGapDetection:
    """Tests for gap detection in observation time series."""

    @pytest.mark.unit
    def test_detect_gaps(self):
        """Should identify gaps larger than threshold."""
        from scripts.utils.subdaily_loader import detect_observation_gaps

        obs_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            pd.Timestamp("2024-01-01 02:00", tz="UTC"),
            # 8-hour gap
            pd.Timestamp("2024-01-01 10:00", tz="UTC"),
            pd.Timestamp("2024-01-01 11:00", tz="UTC"),
        ])

        gaps = detect_observation_gaps(obs_times, min_gap_hours=6.0)
        assert len(gaps) == 1
        assert gaps[0]["start"] == pd.Timestamp("2024-01-01 02:00", tz="UTC")
        assert gaps[0]["end"] == pd.Timestamp("2024-01-01 10:00", tz="UTC")
        assert gaps[0]["duration_hours"] == pytest.approx(8.0)

    @pytest.mark.unit
    def test_no_gaps(self):
        """Dense observations should report no gaps."""
        from scripts.utils.subdaily_loader import detect_observation_gaps

        obs_times = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")

        gaps = detect_observation_gaps(obs_times, min_gap_hours=6.0)
        assert len(gaps) == 0
