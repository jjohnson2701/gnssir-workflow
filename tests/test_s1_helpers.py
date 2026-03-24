# ABOUTME: Tests for shared S1 helper functions used by dashboard tabs.
# ABOUTME: Validates index loading, thumbnail paths, scene matching, and Fresnel geometry.

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_components.s1_helpers import (
    load_s1_index,
    get_thumb_path,
    find_nearest_s1_scene,
    compute_fresnel_radii,
    load_s1_matched,
)


@pytest.mark.unit
class TestGetThumbPath:
    def test_basic_path(self):
        p = get_thumb_path("UMNQ", "2025-03-15")
        assert p.name == "UMNQ_20250315_thumb.png"
        assert "s1_fresnel" in str(p)
        assert "thumbnails" in str(p)

    def test_already_compact_date(self):
        p = get_thumb_path("GLBX", "20240601")
        assert p.name == "GLBX_20240601_thumb.png"


@pytest.mark.unit
class TestFindNearestS1Scene:
    def test_finds_closest(self):
        s1_index = pd.DataFrame({
            "acquisition_date": ["2024-06-01", "2024-06-07", "2024-06-13"],
        })
        s1_index["date_dt"] = pd.to_datetime(s1_index["acquisition_date"])

        best, offset = find_nearest_s1_scene("TEST", "2024-06-08", s1_index)
        assert best["acquisition_date"] == "2024-06-07"
        assert offset == 1

    def test_empty_index(self):
        best, offset = find_nearest_s1_scene("TEST", "2024-06-01", pd.DataFrame())
        assert best is None
        assert offset is None

    def test_none_index(self):
        best, offset = find_nearest_s1_scene("TEST", "2024-06-01", None)
        assert best is None
        assert offset is None


@pytest.mark.unit
class TestLoadS1Index:
    def test_nonexistent_station(self):
        result = load_s1_index("NONEXISTENT_STATION_XYZ")
        assert result is None


@pytest.mark.unit
class TestComputeFresnelRadii:
    def test_nonexistent_station(self):
        inner, outer, method = compute_fresnel_radii("NONEXISTENT_STATION_XYZ", 2099)
        assert inner is None
        assert outer is None
        assert method is None


@pytest.mark.unit
class TestLoadS1Matched:
    def test_nonexistent(self):
        result = load_s1_matched("NONEXISTENT_STATION_XYZ", 2099)
        assert result is None
