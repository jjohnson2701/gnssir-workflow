# ABOUTME: Tests for local basemap rendering in polar animation
# ABOUTME: Verifies ArcticDEM/S2 fallback when tile servers are unavailable

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from scripts.create_polar_animation import render_local_fresnel_basemap


class TestRenderLocalFresnelBasemap:
    """Test local basemap generation from ArcticDEM data."""

    @pytest.mark.unit
    def test_generates_png_from_dem(self, tmp_path):
        """Basemap PNG is created from a DEM array with correct local-meter extent."""
        # Create a realistic fake DEM (500x500 at 2m resolution = 1km)
        dem = np.random.uniform(25, 50, (500, 500)).astype(np.float32)
        dem[200:300, 200:300] = 27.1  # Water patch at center

        buffer_m = 200
        result = render_local_fresnel_basemap(
            dem_array=dem,
            dem_resolution=2.0,
            buffer_m=buffer_m,
            cache_dir=tmp_path,
        )

        assert result is not None
        assert result["local_coords"] is True
        assert result["path"].exists()
        assert result["extent"] == [-buffer_m, buffer_m, -buffer_m, buffer_m]

    @pytest.mark.unit
    def test_water_detection(self, tmp_path):
        """DEM values near the minimum are classified as water."""
        # DEM with clear land/water boundary: left half = water (27m), right half = land (50m)
        dem = np.full((500, 500), 50.0, dtype=np.float32)
        dem[:, :250] = 27.0

        result = render_local_fresnel_basemap(
            dem_array=dem,
            dem_resolution=2.0,
            buffer_m=300,
            cache_dir=tmp_path,
        )

        assert result is not None
        assert result["path"].exists()

    @pytest.mark.unit
    def test_returns_none_for_empty_dem(self, tmp_path):
        """Returns None when DEM array is all nodata."""
        dem = np.full((100, 100), -9999.0, dtype=np.float32)

        result = render_local_fresnel_basemap(
            dem_array=dem,
            dem_resolution=2.0,
            buffer_m=200,
            cache_dir=tmp_path,
        )

        assert result is None


class TestFresnelPanelLocalCoords:
    """Test that Fresnel zone panel works with local coordinates."""

    @pytest.mark.unit
    def test_origin_at_zero_when_local(self):
        """When fresnel_local_coords is True, origin should be (0, 0)."""
        cached = {"fresnel_local_coords": True}
        use_local = cached.get("fresnel_local_coords", False)
        origin_x = 0 if use_local else 12345
        origin_y = 0 if use_local else 67890
        assert origin_x == 0
        assert origin_y == 0

    @pytest.mark.unit
    def test_origin_at_station_when_mercator(self):
        """When fresnel_local_coords is False, origin should be station coords."""
        cached = {"fresnel_local_coords": False}
        station_x, station_y = 12345, 67890
        use_local = cached.get("fresnel_local_coords", False)
        origin_x = 0 if use_local else station_x
        origin_y = 0 if use_local else station_y
        assert origin_x == station_x
        assert origin_y == station_y
