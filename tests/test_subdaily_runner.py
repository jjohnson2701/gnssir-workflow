# ABOUTME: Tests for the subdaily processing phase that integrates gnssrefl subdaily into the pipeline.
# ABOUTME: Validates output CSV format, file discovery, and confidence annotation inclusion.

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFindSplineOutput:
    """Tests for locating subdaily spline output files."""

    @pytest.mark.unit
    def test_finds_spline_output(self, tmp_path):
        """Should find spline output file in expected location."""
        from scripts.utils.subdaily_runner import find_spline_output

        spline_dir = tmp_path / "Files" / "mysta"
        spline_dir.mkdir(parents=True)
        spline_file = spline_dir / "mysta_spline_out.txt"
        spline_file.write_text("% test data\n")

        result = find_spline_output("mysta", tmp_path)
        assert result == spline_file

    @pytest.mark.unit
    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when spline output doesn't exist."""
        from scripts.utils.subdaily_runner import find_spline_output

        result = find_spline_output("missing", tmp_path)
        assert result is None


class TestFindObservationFile:
    """Tests for locating IF-corrected observation files."""

    @pytest.mark.unit
    def test_finds_obs_file(self, tmp_path):
        """Should find the IF-corrected observation file."""
        from scripts.utils.subdaily_runner import find_observation_file

        obs_dir = tmp_path / "Files" / "test"
        obs_dir.mkdir(parents=True)
        obs_file = obs_dir / "test_2024_subdaily_edit.txt.withrhdotIF"
        obs_file.write_text("% test\n")

        result = find_observation_file("test", 2024, tmp_path)
        assert result == obs_file

    @pytest.mark.unit
    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when observation file doesn't exist."""
        from scripts.utils.subdaily_runner import find_observation_file

        result = find_observation_file("test", 2024, tmp_path)
        assert result is None


def _write_spline_file(path, n_points=6):
    """Helper to create a synthetic spline output file."""
    lines = ["% MJD RH year month day hour minute second WSE\n"]
    for i in range(n_points):
        h = i // 2
        m = (i % 2) * 30
        lines.append(
            f"60310.{i * 20833:06d}  7.{500 + i:03d}  2024  1  1  {h}  {m}  0  -22.{500 - i:03d}\n"
        )
    path.write_text("".join(lines))


class TestProcessSubdailyOutput:
    """Tests for subdaily output processing and CSV generation."""

    @pytest.mark.unit
    def test_output_csv_has_required_columns(self, tmp_path):
        """Processed subdaily output CSV should contain all required columns."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        spline_dir = tmp_path / "Files" / "test"
        spline_dir.mkdir(parents=True)
        _write_spline_file(spline_dir / "test_spline_out.txt")

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result_path = process_subdaily_output(
            station="test", year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )

        assert result_path is not None
        assert result_path.exists()

        df = pd.read_csv(result_path)
        required_cols = ["datetime", "rh_m", "wse_ortho_m", "datum"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.unit
    def test_output_csv_reports_datum(self, tmp_path):
        """Output CSV should report orthometric EGM96 datum."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        spline_dir = tmp_path / "Files" / "test"
        spline_dir.mkdir(parents=True)
        _write_spline_file(spline_dir / "test_spline_out.txt")

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result_path = process_subdaily_output(
            station="test", year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )

        df = pd.read_csv(result_path)
        assert (df["datum"] == "orthometric_EGM96").all()

    @pytest.mark.unit
    def test_returns_none_when_no_spline_output(self, tmp_path):
        """Should return None when subdaily output file doesn't exist."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = process_subdaily_output(
            station="nonexistent", year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )
        assert result is None

    @pytest.mark.unit
    def test_output_includes_confidence_when_obs_available(self, tmp_path):
        """Output should include confidence columns when observation file exists."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        station = "test"
        spline_dir = tmp_path / "Files" / station
        spline_dir.mkdir(parents=True)

        # Create spline output (48 half-hourly points covering 24 hours)
        spline_file = spline_dir / f"{station}_spline_out.txt"
        lines = ["% comment\n"]
        for i in range(48):
            h = i // 2
            m = (i % 2) * 30
            mjd = 60310.0 + i * (0.5 / 24.0)
            lines.append(
                f"{mjd:.6f}  7.500  2024  1  1  {h}  {m}  0  -22.500\n"
            )
        spline_file.write_text("".join(lines))

        # Create IF-corrected observation file (21+ columns)
        # Indices: 0=year, 17=month, 18=day, 19=hour, 20=minute
        obs_file = spline_dir / f"{station}_2024_subdaily_edit.txt.withrhdotIF"
        obs_lines = ["% comment\n"]
        for h in [0, 6, 12, 18]:
            parts = ["2024"] + ["0.0"] * 16 + ["1", "1", str(h), "0"]
            obs_lines.append("  ".join(parts) + "\n")
        obs_file.write_text("".join(obs_lines))

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result_path = process_subdaily_output(
            station=station, year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )

        df = pd.read_csv(result_path)
        assert "nearest_obs_hours" in df.columns
        assert "is_interpolated" in df.columns

    @pytest.mark.unit
    def test_output_without_obs_file_lacks_confidence(self, tmp_path):
        """Output without observation file should omit confidence columns."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        station = "test"
        spline_dir = tmp_path / "Files" / station
        spline_dir.mkdir(parents=True)
        _write_spline_file(spline_dir / f"{station}_spline_out.txt")

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result_path = process_subdaily_output(
            station=station, year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )

        df = pd.read_csv(result_path)
        assert "nearest_obs_hours" not in df.columns
        assert "is_interpolated" not in df.columns

    @pytest.mark.unit
    def test_output_filename_convention(self, tmp_path):
        """Output CSV should follow station_year_subdaily.csv naming."""
        from scripts.utils.subdaily_runner import process_subdaily_output

        spline_dir = tmp_path / "Files" / "glbx"
        spline_dir.mkdir(parents=True)
        _write_spline_file(spline_dir / "glbx_spline_out.txt")

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result_path = process_subdaily_output(
            station="glbx", year=2024,
            refl_code_base=tmp_path, output_dir=output_dir,
        )

        assert result_path.name == "glbx_2024_subdaily.csv"
