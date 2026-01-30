# ABOUTME: Tests for find_reference_stations.py module.
# ABOUTME: Tests haversine distance calculation, config parsing, and station search.

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from find_reference_stations import (
    haversine_distance,
    load_station_config,
    ReferenceStation
)


class TestHaversineDistance:
    """Tests for the haversine distance calculation."""

    @pytest.mark.unit
    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be zero."""
        lat, lon = 45.0, -122.0
        distance = haversine_distance(lat, lon, lat, lon)
        assert distance == pytest.approx(0.0, abs=0.001)

    @pytest.mark.unit
    def test_known_distance(self):
        """Test with known distance between two cities."""
        # Seattle to Portland is approximately 233 km
        seattle = (47.6062, -122.3321)
        portland = (45.5152, -122.6784)
        distance = haversine_distance(*seattle, *portland)
        assert distance == pytest.approx(233, rel=0.05)  # 5% tolerance

    @pytest.mark.unit
    def test_equator_distance(self):
        """Test distance along equator."""
        # 1 degree of longitude at equator is ~111 km
        distance = haversine_distance(0, 0, 0, 1)
        assert distance == pytest.approx(111, rel=0.02)

    @pytest.mark.unit
    def test_meridian_distance(self):
        """Test distance along a meridian."""
        # 1 degree of latitude is ~111 km
        distance = haversine_distance(0, 0, 1, 0)
        assert distance == pytest.approx(111, rel=0.02)

    @pytest.mark.unit
    def test_symmetry(self):
        """Distance A to B should equal distance B to A."""
        lat1, lon1 = 40.7128, -74.0060  # New York
        lat2, lon2 = 51.5074, -0.1278   # London

        dist_ab = haversine_distance(lat1, lon1, lat2, lon2)
        dist_ba = haversine_distance(lat2, lon2, lat1, lon1)

        assert dist_ab == pytest.approx(dist_ba, abs=0.001)

    @pytest.mark.unit
    def test_negative_coordinates(self):
        """Test with negative (southern/western) coordinates."""
        # Sydney, Australia to Santiago, Chile
        sydney = (-33.8688, 151.2093)
        santiago = (-33.4489, -70.6693)
        distance = haversine_distance(*sydney, *santiago)
        # Known distance is approximately 11,340 km
        assert distance == pytest.approx(11340, rel=0.05)


class TestStationConfig:
    """Tests for station configuration loading."""

    @pytest.mark.unit
    def test_load_valid_station(self, config_dir):
        """Test loading a valid station from config."""
        config_path = config_dir / 'stations_config.json'
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_station_config('GLBX', config_path)
        if config is None:
            pytest.skip("GLBX not in config")

        # Config uses latitude_deg/longitude_deg naming convention
        assert 'latitude_deg' in config or 'latitude' in config or 'lat' in config

    @pytest.mark.unit
    def test_load_nonexistent_station(self, config_dir):
        """Test loading a station that doesn't exist."""
        config_path = config_dir / 'stations_config.json'
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_station_config('NONEXISTENT_STATION_XYZ', config_path)
        assert config is None

    @pytest.mark.unit
    def test_config_has_required_fields(self, sample_station_config):
        """Test that station config has required fields."""
        for station_id, config in sample_station_config.items():
            assert 'latitude' in config
            assert 'longitude' in config
            assert isinstance(config['latitude'], (int, float))
            assert isinstance(config['longitude'], (int, float))


class TestReferenceStation:
    """Tests for the ReferenceStation dataclass."""

    @pytest.mark.unit
    def test_reference_station_creation(self):
        """Test creating a ReferenceStation object."""
        station = ReferenceStation(
            source='CO-OPS',
            station_id='9447130',
            station_name='Seattle',
            latitude=47.6062,
            longitude=-122.3321,
            distance_km=5.2,
            datum='NAVD88',
            notes='Test station'
        )

        assert station.source == 'CO-OPS'
        assert station.station_id == '9447130'
        assert station.distance_km == 5.2

    @pytest.mark.unit
    def test_reference_station_sorting(self):
        """Test that stations can be sorted by distance."""
        stations = [
            ReferenceStation('A', 'id1', 'Name1', 0, 0, 10.0, 'NAVD88', ''),
            ReferenceStation('B', 'id2', 'Name2', 0, 0, 5.0, 'NAVD88', ''),
            ReferenceStation('C', 'id3', 'Name3', 0, 0, 15.0, 'NAVD88', ''),
        ]

        sorted_stations = sorted(stations, key=lambda s: s.distance_km)

        assert sorted_stations[0].distance_km == 5.0
        assert sorted_stations[1].distance_km == 10.0
        assert sorted_stations[2].distance_km == 15.0
