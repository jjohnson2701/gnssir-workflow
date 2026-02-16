# ABOUTME: Unit tests for core utility functions
# ABOUTME: Tests Fresnel zone calculations, distance functions, and data transformations

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFresnelZoneCalculations:
    """Tests for Fresnel zone calculations in create_polar_animation.py"""

    @pytest.mark.unit
    def test_calculate_fresnel_radius_basic(self):
        """Test Fresnel radius calculation with known values."""
        from scripts.create_polar_animation import calculate_fresnel_radius

        # At 10m reflector height and 10 degrees elevation
        # Slant distance = 10 / sin(10°) = 57.6m
        # Fresnel radius = sqrt(wavelength * slant_distance / 2)
        reflector_height = 10.0
        elevation = 10.0

        radius = calculate_fresnel_radius(reflector_height, elevation)

        # Expected: sqrt(0.1903 * 57.6 / 2) ≈ 2.34m
        assert radius > 2.0
        assert radius < 3.0

    @pytest.mark.unit
    def test_calculate_fresnel_radius_higher_elevation(self):
        """Test that higher elevation produces smaller Fresnel radius."""
        from scripts.create_polar_animation import calculate_fresnel_radius

        reflector_height = 10.0

        radius_low_elev = calculate_fresnel_radius(reflector_height, 5.0)
        radius_high_elev = calculate_fresnel_radius(reflector_height, 15.0)

        # Higher elevation = closer slant distance = smaller Fresnel zone
        assert radius_high_elev < radius_low_elev

    @pytest.mark.unit
    def test_calculate_fresnel_radius_higher_rh(self):
        """Test that higher reflector height produces larger Fresnel radius."""
        from scripts.create_polar_animation import calculate_fresnel_radius

        elevation = 10.0

        radius_low_rh = calculate_fresnel_radius(5.0, elevation)
        radius_high_rh = calculate_fresnel_radius(15.0, elevation)

        # Higher RH = larger slant distance = larger Fresnel zone
        assert radius_high_rh > radius_low_rh


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    @pytest.mark.unit
    def test_haversine_distance_same_point(self):
        """Test that distance to same point is zero."""
        from scripts.usgs_data_handler import haversine_distance

        lat, lon = 36.0, -75.0
        distance = haversine_distance(lat, lon, lat, lon)

        assert distance == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.unit
    def test_haversine_distance_known_value(self):
        """Test haversine with a known distance."""
        from scripts.usgs_data_handler import haversine_distance

        # New York to Los Angeles is approximately 3944 km
        ny_lat, ny_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437

        distance = haversine_distance(ny_lat, ny_lon, la_lat, la_lon)

        # Should be within 50km of known distance
        assert distance > 3900
        assert distance < 4000

    @pytest.mark.unit
    def test_haversine_distance_symmetry(self):
        """Test that distance is symmetric."""
        from scripts.usgs_data_handler import haversine_distance

        lat1, lon1 = 36.0, -75.0
        lat2, lon2 = 37.0, -76.0

        distance_1_to_2 = haversine_distance(lat1, lon1, lat2, lon2)
        distance_2_to_1 = haversine_distance(lat2, lon2, lat1, lon1)

        assert distance_1_to_2 == pytest.approx(distance_2_to_1, rel=1e-9)


class TestReflectorHeightUtils:
    """Tests for reflector height utility functions."""

    @pytest.mark.unit
    def test_calculate_wse_from_rh(self):
        """Test water surface elevation calculation from reflector height."""
        from scripts.reflector_height_utils import calculate_wse_from_rh

        # Create sample dataframe with daily RH data (using expected column names)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "rh_median_m": [8.0, 8.5, 9.0, 8.5, 8.0],
                "RH_std": [0.1, 0.15, 0.2, 0.15, 0.1],
            }
        )

        antenna_height = 30.0  # meters

        result = calculate_wse_from_rh(df, antenna_height)

        # WSE = antenna_height - RH
        # For rh_median_m = 8.0, WSE should be 30.0 - 8.0 = 22.0
        assert "wse_ellips_m" in result.columns
        assert result["wse_ellips_m"].iloc[0] == pytest.approx(22.0)
        assert result["wse_ellips_m"].iloc[2] == pytest.approx(21.0)


class TestSegmentedAnalysis:
    """Tests for segmented analysis utilities."""

    @pytest.mark.unit
    def test_filter_by_segment_date_range(self):
        """Test filtering by date range tuple."""
        from scripts.utils.segmented_analysis import filter_by_segment

        # Create sample dataframe with datetime index
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)
        df.index.name = "datetime"

        # Filter to February
        result = filter_by_segment(df, ("2024-02-01", "2024-02-29"))

        assert len(result) == 29  # February has 29 days in 2024
        assert result.index.min() >= pd.Timestamp("2024-02-01")
        assert result.index.max() <= pd.Timestamp("2024-02-29")

    @pytest.mark.unit
    def test_filter_by_segment_month_list(self):
        """Test filtering by month numbers as list."""
        from scripts.utils.segmented_analysis import filter_by_segment

        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        df = pd.DataFrame({"value": np.random.randn(365)}, index=dates)
        df.index.name = "datetime"

        # Filter to March using month list [3]
        result = filter_by_segment(df, [3])

        assert len(result) == 31  # March has 31 days
        assert all(result.index.month == 3)


class TestSplitIntoSeasons:
    """Tests for DOY range splitting into seasonal chunks."""

    @pytest.mark.unit
    def test_short_range_no_split(self):
        """A range shorter than one chunk stays as a single chunk."""
        from scripts.create_polar_animation import split_into_seasons

        chunks = split_into_seasons(100, 180)
        assert chunks == [(100, 180)]

    @pytest.mark.unit
    def test_exact_chunk_size(self):
        """A range exactly equal to chunk size stays as a single chunk."""
        from scripts.create_polar_animation import split_into_seasons

        chunks = split_into_seasons(1, 91)
        assert chunks == [(1, 91)]

    @pytest.mark.unit
    def test_full_year_splits_into_three_or_four(self):
        """A full year (DOY 1-365) splits into 4 chunks."""
        from scripts.create_polar_animation import split_into_seasons

        chunks = split_into_seasons(1, 365)
        assert len(chunks) == 4
        # First chunk starts at 1
        assert chunks[0][0] == 1
        # Last chunk ends at 365
        assert chunks[-1][1] == 365
        # Chunks are contiguous
        for i in range(len(chunks) - 1):
            assert chunks[i][1] + 1 == chunks[i + 1][0]

    @pytest.mark.unit
    def test_small_remainder_merged(self):
        """A small remainder (<30 days) is merged into the previous chunk."""
        from scripts.create_polar_animation import split_into_seasons

        # 85 to 365 = 281 days. 3 chunks of 91 = 273, remainder = 8 days
        # The 8-day remainder should merge into the third chunk
        chunks = split_into_seasons(85, 365)
        assert len(chunks) == 3
        assert chunks[-1][1] == 365
        # Last chunk should be > 91 days (got the remainder)
        last_chunk_days = chunks[-1][1] - chunks[-1][0] + 1
        assert last_chunk_days > 91

    @pytest.mark.unit
    def test_custom_chunk_size(self):
        """Custom chunk_days parameter is respected."""
        from scripts.create_polar_animation import split_into_seasons

        chunks = split_into_seasons(1, 365, chunk_days=120)
        assert len(chunks) == 3
        assert chunks[0] == (1, 120)

    @pytest.mark.unit
    def test_single_day_range(self):
        """A single-day range returns one chunk."""
        from scripts.create_polar_animation import split_into_seasons

        chunks = split_into_seasons(100, 100)
        assert chunks == [(100, 100)]


class TestColorScheme:
    """Tests for visualization color scheme."""

    @pytest.mark.unit
    def test_plot_colors_has_required_keys(self):
        """Test that PLOT_COLORS has all required keys."""
        from scripts.visualizer.base import PLOT_COLORS

        required_keys = ["gnssir", "usgs", "highlight", "grid"]

        for key in required_keys:
            assert key in PLOT_COLORS, f"Missing required color key: {key}"

    @pytest.mark.unit
    def test_plot_colors_are_valid_hex(self):
        """Test that all colors are valid hex codes."""
        from scripts.visualizer.base import PLOT_COLORS

        import re

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")

        for key, color in PLOT_COLORS.items():
            assert hex_pattern.match(color), f"Invalid hex color for {key}: {color}"
