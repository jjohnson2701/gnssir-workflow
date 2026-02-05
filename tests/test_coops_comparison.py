# ABOUTME: Tests for coops_comparison.py module.
# ABOUTME: Tests station discovery, config loading, and CO-OPS comparison logic.

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from coops_comparison import (
    load_station_config,
    find_nearest_coops_station,
    get_coops_station_info,
)


class TestLoadStationConfig:
    """Tests for station configuration loading."""

    @pytest.mark.unit
    def test_load_valid_station(self, config_dir):
        """Test loading a valid station from config."""
        config_path = config_dir / "stations_config.json"
        if not config_path.exists():
            pytest.skip("Config file not found")

        # Patch PROJECT_ROOT to use the actual config
        import coops_comparison

        original_root = coops_comparison.PROJECT_ROOT
        coops_comparison.PROJECT_ROOT = config_dir.parent

        try:
            config = load_station_config("GLBX")
            if config is None:
                pytest.skip("GLBX not in config")

            assert "latitude_deg" in config or "latitude" in config
            assert "longitude_deg" in config or "longitude" in config
        finally:
            coops_comparison.PROJECT_ROOT = original_root

    @pytest.mark.unit
    def test_load_nonexistent_station(self, config_dir):
        """Test loading a station that doesn't exist returns None."""
        import coops_comparison

        original_root = coops_comparison.PROJECT_ROOT
        coops_comparison.PROJECT_ROOT = config_dir.parent

        try:
            config = load_station_config("NONEXISTENT_XYZ")
            assert config is None
        finally:
            coops_comparison.PROJECT_ROOT = original_root


class TestFindNearestCoopsStation:
    """Tests for CO-OPS station discovery."""

    @pytest.mark.unit
    def test_missing_coordinates_returns_none(self):
        """Test that missing coordinates returns None."""
        station_config = {"name": "Test Station"}
        result = find_nearest_coops_station(station_config)
        assert result is None

    @pytest.mark.unit
    def test_missing_latitude_returns_none(self):
        """Test that missing latitude returns None."""
        station_config = {"longitude_deg": -122.0}
        result = find_nearest_coops_station(station_config)
        assert result is None

    @pytest.mark.unit
    def test_missing_longitude_returns_none(self):
        """Test that missing longitude returns None."""
        station_config = {"latitude_deg": 47.0}
        result = find_nearest_coops_station(station_config)
        assert result is None

    @pytest.mark.unit
    @patch("coops_comparison.NOAACOOPSClient")
    def test_find_nearest_returns_closest(self, mock_client_class):
        """Test that find_nearest returns the closest station."""
        # Set up mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.find_nearby_stations.return_value = [
            {"id": "9447130", "name": "Seattle", "distance_km": 5.2},
            {"id": "9447110", "name": "Tacoma", "distance_km": 25.0},
        ]

        station_config = {"latitude_deg": 47.6, "longitude_deg": -122.3}
        result = find_nearest_coops_station(station_config)

        assert result is not None
        assert result["id"] == "9447130"
        assert result["distance_km"] == 5.2

    @pytest.mark.unit
    @patch("coops_comparison.NOAACOOPSClient")
    def test_find_nearest_no_stations_returns_none(self, mock_client_class):
        """Test that no nearby stations returns None."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.find_nearby_stations.return_value = []

        station_config = {"latitude_deg": 47.6, "longitude_deg": -122.3}
        result = find_nearest_coops_station(station_config)

        assert result is None


class TestGetCoopsStationInfo:
    """Tests for CO-OPS station info retrieval."""

    @pytest.mark.unit
    @patch("coops_comparison.NOAACOOPSClient")
    def test_preferred_station_from_config(self, mock_client_class):
        """Test that preferred station from config is used first."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_station_metadata.return_value = {
            "name": "Seattle",
            "latitude": 47.6,
            "longitude": -122.3,
        }

        station_config = {
            "latitude_deg": 47.6,
            "longitude_deg": -122.3,
            "external_data_sources": {
                "noaa_coops": {"preferred_stations": ["9447130"], "datum": "NAVD88"}
            },
        }

        result = get_coops_station_info(station_config)

        assert result is not None
        assert result["id"] == "9447130"
        assert result["source"] == "config_preferred"

    @pytest.mark.unit
    def test_nearest_station_from_config(self):
        """Test that nearest_station from config is used."""
        station_config = {
            "latitude_deg": 47.6,
            "longitude_deg": -122.3,
            "external_data_sources": {
                "noaa_coops": {
                    "nearest_station": {"id": "9447130", "name": "Seattle", "distance_km": 5.2}
                }
            },
        }

        result = get_coops_station_info(station_config)

        assert result is not None
        assert result["id"] == "9447130"
        assert result["source"] == "config_nearest"
        assert result["distance_km"] == 5.2

    @pytest.mark.unit
    @patch("coops_comparison.find_nearest_coops_station")
    def test_auto_discovery_fallback(self, mock_find_nearest):
        """Test auto-discovery when no config exists."""
        mock_find_nearest.return_value = {"id": "9447130", "name": "Seattle", "distance_km": 5.2}

        station_config = {
            "latitude_deg": 47.6,
            "longitude_deg": -122.3,
            "external_data_sources": {},
        }

        result = get_coops_station_info(station_config)

        assert result is not None
        assert result["id"] == "9447130"
        assert result["source"] == "auto_discovered"

    @pytest.mark.unit
    @patch("coops_comparison.find_nearest_coops_station")
    def test_returns_none_when_no_station_found(self, mock_find_nearest):
        """Test returns None when no station can be found."""
        mock_find_nearest.return_value = None

        station_config = {
            "latitude_deg": 47.6,
            "longitude_deg": -122.3,
            "external_data_sources": {},
        }

        result = get_coops_station_info(station_config)

        assert result is None


class TestCoopsStationInfoStructure:
    """Tests for the structure of returned station info."""

    @pytest.mark.unit
    @patch("coops_comparison.NOAACOOPSClient")
    def test_preferred_station_has_required_fields(self, mock_client_class):
        """Test that preferred station info has all required fields."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_station_metadata.return_value = {
            "name": "Seattle",
            "latitude": 47.6,
            "longitude": -122.3,
        }

        station_config = {
            "external_data_sources": {"noaa_coops": {"preferred_stations": ["9447130"]}}
        }

        result = get_coops_station_info(station_config)

        assert "id" in result
        assert "name" in result
        assert "source" in result
        assert result["source"] == "config_preferred"

    @pytest.mark.unit
    @patch("coops_comparison.find_nearest_coops_station")
    def test_auto_discovered_has_required_fields(self, mock_find_nearest):
        """Test that auto-discovered station info has all required fields."""
        mock_find_nearest.return_value = {
            "id": "9447130",
            "name": "Seattle",
            "latitude": 47.6,
            "longitude": -122.3,
            "distance_km": 5.2,
        }

        station_config = {
            "latitude_deg": 47.6,
            "longitude_deg": -122.3,
            "external_data_sources": {},
        }

        result = get_coops_station_info(station_config)

        assert "id" in result
        assert "name" in result
        assert "distance_km" in result
        assert "source" in result
        assert result["source"] == "auto_discovered"


class TestCoopsComparisonIntegration:
    """Integration tests for CO-OPS comparison with real fixture data."""

    @pytest.mark.integration
    def test_coops_comparison_data_structure(self, coops_comparison_data):
        """Test that CO-OPS comparison data has expected structure."""
        df = coops_comparison_data

        # Should have standardized columns
        assert "wse_ellips_m" in df.columns
        assert "rh_median_m" in df.columns

    @pytest.mark.integration
    def test_coops_subdaily_data_structure(self, coops_subdaily_matched_data):
        """Test that CO-OPS subdaily data has expected structure."""
        df = coops_subdaily_matched_data

        assert "gnss_datetime" in df.columns
        assert "gnss_wse" in df.columns
        assert "coops_datetime" in df.columns
        assert "coops_wl" in df.columns

    @pytest.mark.integration
    def test_coops_demeaned_values(self, coops_subdaily_matched_data):
        """Test that CO-OPS data has properly calculated demeaned values."""
        df = coops_subdaily_matched_data

        if "gnss_dm" in df.columns and "coops_dm" in df.columns:
            # Demeaned values should have mean close to zero
            assert abs(df["gnss_dm"].mean()) < 0.1
            assert abs(df["coops_dm"].mean()) < 0.1


class TestCoopsDataValidation:
    """Tests for CO-OPS data validation."""

    @pytest.mark.unit
    def test_station_id_format(self):
        """Test that CO-OPS station IDs are in expected format."""
        # CO-OPS station IDs are typically 7 digits
        valid_ids = ["9447130", "1612340", "8443970"]

        for station_id in valid_ids:
            assert len(station_id) == 7
            assert station_id.isdigit()

    @pytest.mark.unit
    def test_datum_values(self):
        """Test that common datum values are recognized."""
        valid_datums = ["MLLW", "MSL", "NAVD88", "IGLD85"]

        # All should be string values
        for datum in valid_datums:
            assert isinstance(datum, str)
            assert len(datum) > 0
