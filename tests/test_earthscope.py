# ABOUTME: Tests for EarthScope GAGE archive integration
# ABOUTME: Covers rinex2snr archive routing, download helpers, and data source config

import gzip
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.data_manager import (
    download_rinex_earthscope,
    download_from_url,
    EARTHSCOPE_BASE_URL,
    EARTHSCOPE_PATH_PATTERN,
)
from scripts.external_tools.preprocessor import decompress_unix_z
from scripts.external_tools.gnssrefl_executor import execute_rinex2snr


class TestEarthScopeConstants:
    """Test that EarthScope URL constants are correct."""

    @pytest.mark.unit
    def test_base_url(self):
        assert EARTHSCOPE_BASE_URL == "https://gage-data.earthscope.org/archive/gnss/rinex/obs"

    @pytest.mark.unit
    def test_path_pattern_format(self):
        """Test that the path pattern produces correct URLs."""
        result = EARTHSCOPE_PATH_PATTERN.format(
            year=2025, doy=100, station_lower="umnq", yy="25"
        )
        assert result == "2025/100/umnq1000.25o.Z"


class TestDownloadRinexEarthscope:
    """Tests for the EarthScope RINEX download function."""

    @pytest.mark.unit
    def test_missing_token_returns_false(self, tmp_path):
        """download_rinex_earthscope returns False when EARTHSCOPE_TOKEN is not set."""
        target = tmp_path / "test.o.Z"
        with patch.dict("os.environ", {}, clear=True):
            result = download_rinex_earthscope("UMNQ", 2025, 100, target)
        assert result is False

    @pytest.mark.unit
    def test_constructs_correct_url(self, tmp_path):
        """Verify the correct URL and auth header are passed to download_from_url."""
        target = tmp_path / "test.o.Z"
        with patch.dict("os.environ", {"EARTHSCOPE_TOKEN": "fake_token"}):
            with patch(
                "scripts.utils.data_manager.download_from_url", return_value=True
            ) as mock_dl:
                result = download_rinex_earthscope("UMNQ", 2025, 100, target)

        assert result is True
        mock_dl.assert_called_once()
        call_args = mock_dl.call_args
        url = call_args[0][0]
        assert url == (
            "https://gage-data.earthscope.org/archive/gnss/rinex/obs"
            "/2025/100/umnq1000.25o.Z"
        )
        headers = call_args[1]["headers"]
        assert headers == {"Authorization": "Bearer fake_token"}

    @pytest.mark.unit
    def test_station_id_lowercased(self, tmp_path):
        """Station ID is lowercased in the URL regardless of input case."""
        target = tmp_path / "test.o.Z"
        with patch.dict("os.environ", {"EARTHSCOPE_TOKEN": "tok"}):
            with patch(
                "scripts.utils.data_manager.download_from_url", return_value=True
            ) as mock_dl:
                download_rinex_earthscope("NIAQ", 2026, 44, target)

        url = mock_dl.call_args[0][0]
        assert "niaq0440.26o.Z" in url

    @pytest.mark.unit
    def test_doy_zero_padded(self, tmp_path):
        """DOY is zero-padded to 3 digits."""
        target = tmp_path / "test.o.Z"
        with patch.dict("os.environ", {"EARTHSCOPE_TOKEN": "tok"}):
            with patch(
                "scripts.utils.data_manager.download_from_url", return_value=True
            ) as mock_dl:
                download_rinex_earthscope("NKAR", 2025, 5, target)

        url = mock_dl.call_args[0][0]
        assert "nkar0050.25o.Z" in url


class TestDownloadFromUrlHeaders:
    """Test that download_from_url accepts and uses headers."""

    @pytest.mark.unit
    def test_headers_passed_to_requests(self, tmp_path):
        """Headers are forwarded to requests.get."""
        target = tmp_path / "testfile.dat"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"fake data"]

        with patch("scripts.utils.data_manager.requests.get", return_value=mock_response) as mock_get:
            download_from_url(
                "https://example.com/file.Z",
                target,
                headers={"Authorization": "Bearer test123"},
            )

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] == {"Authorization": "Bearer test123"}

    @pytest.mark.unit
    def test_no_headers_by_default(self, tmp_path):
        """Without headers argument, None is passed to requests.get."""
        target = tmp_path / "testfile.dat"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"fake data"]

        with patch("scripts.utils.data_manager.requests.get", return_value=mock_response) as mock_get:
            download_from_url("https://example.com/file.dat", target)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] is None


class TestDecompressUnixZ:
    """Tests for Unix .Z decompression."""

    @pytest.mark.unit
    def test_decompress_in_place(self, tmp_path):
        """Decompressing a .gz file returns the decompressed path."""
        content = b"fake rinex data\n" * 100
        compressed = tmp_path / "test.o.gz"
        with gzip.open(compressed, "wb") as f:
            f.write(content)
        assert compressed.exists()

        decompressed = decompress_unix_z(compressed)

        assert decompressed is not None
        assert decompressed.exists()
        assert decompressed.name == "test.o"
        assert not compressed.exists()  # compressed file removed by gzip -d

    @pytest.mark.unit
    def test_decompress_to_output_dir(self, tmp_path):
        """Decompressing to a different output directory moves the file."""
        content = b"fake rinex data\n" * 100
        compressed = tmp_path / "test.o.gz"
        with gzip.open(compressed, "wb") as f:
            f.write(content)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        decompressed = decompress_unix_z(compressed, output_dir=output_dir)

        assert decompressed is not None
        assert decompressed.parent == output_dir
        assert decompressed.name == "test.o"
        assert decompressed.exists()

    @pytest.mark.unit
    def test_decompress_nonexistent_file_returns_none(self, tmp_path):
        """Decompressing a nonexistent file returns None."""
        fake_path = tmp_path / "nonexistent.o.Z"
        result = decompress_unix_z(fake_path)
        assert result is None

    @pytest.mark.unit
    def test_decompress_preserves_content(self, tmp_path):
        """Content is preserved through compress/decompress cycle."""
        original_content = "     2.11           OBSERVATION DATA    M (MIXED)\n" * 50
        compressed = tmp_path / "umnq1000.25o.gz"
        with gzip.open(compressed, "wt") as f:
            f.write(original_content)

        decompressed = decompress_unix_z(compressed)

        assert decompressed.read_text() == original_content


class TestRinex2snrArchiveFlag:
    """Test that execute_rinex2snr passes the correct archive flag."""

    @pytest.mark.unit
    def test_default_uses_nolook(self, tmp_path):
        """Without archive param, command uses -nolook T."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        with patch(
            "scripts.external_tools.gnssrefl_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            execute_rinex2snr(
                "rinex2snr", "umnq", 2025, "100",
                tmp_path, tmp_path, logs_dir,
            )
        cmd = mock_run.call_args[0][0]
        assert "-nolook" in cmd
        assert "T" in cmd
        assert "-archive" not in cmd

    @pytest.mark.unit
    def test_archive_unavco_replaces_nolook(self, tmp_path):
        """With archive='unavco', command uses -archive unavco instead of -nolook T."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        with patch(
            "scripts.external_tools.gnssrefl_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            execute_rinex2snr(
                "rinex2snr", "umnq", 2025, "100",
                tmp_path, tmp_path, logs_dir,
                archive="unavco",
            )
        cmd = mock_run.call_args[0][0]
        assert "-archive" in cmd
        assert "unavco" in cmd
        assert "-nolook" not in cmd


class TestRinex2snrOrbitFlag:
    """Test that execute_rinex2snr passes the correct orbit type flag."""

    @pytest.mark.unit
    def test_no_orbit_flag_by_default(self, tmp_path):
        """Without orbit param, command omits -orb flag (gnssrefl picks default)."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        with patch(
            "scripts.external_tools.gnssrefl_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            execute_rinex2snr(
                "rinex2snr", "umnq", 2025, "100",
                tmp_path, tmp_path, logs_dir,
            )
        cmd = mock_run.call_args[0][0]
        assert "-orb" not in cmd

    @pytest.mark.unit
    def test_orbit_flag_passed_when_specified(self, tmp_path):
        """With orbit='gnss3', command includes -orb gnss3."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        with patch(
            "scripts.external_tools.gnssrefl_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            execute_rinex2snr(
                "rinex2snr", "umnq", 2025, "223",
                tmp_path, tmp_path, logs_dir,
                archive="unavco",
                orbit="gnss3",
            )
        cmd = mock_run.call_args[0][0]
        assert "-orb" in cmd
        assert "gnss3" in cmd
        assert "-archive" in cmd


class TestDataSourceRouting:
    """Test that station config data_source field routes correctly."""

    @pytest.mark.unit
    def test_nps_default_when_no_data_source(self):
        """Stations without data_source field default to nps."""
        config = {"station_id_4char_lower": "fora"}
        assert config.get("data_source", "nps") == "nps"

    @pytest.mark.unit
    def test_earthscope_when_specified(self):
        """Stations with data_source=earthscope are routed correctly."""
        config = {"station_id_4char_lower": "umnq", "data_source": "earthscope"}
        assert config.get("data_source", "nps") == "earthscope"
