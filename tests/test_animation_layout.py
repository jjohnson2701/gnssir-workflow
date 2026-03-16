# ABOUTME: Tests for the refactored polar animation layout (cover frame + stacked panels).
# ABOUTME: Validates CLI args, frame dimensions, progress bar, and mode-dependent behavior.

import pytest
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

SCRIPT = str(project_root / "scripts" / "create_polar_animation.py")


def _make_synthetic_metadata():
    """Create minimal metadata dict for testing frame rendering."""
    return {
        "ref_source": "Test Gauge",
        "ref_site_id": "T001",
        "has_reference": True,
        "station_name": "TEST",
        "station_lat": 58.45,
        "station_lon": -135.88,
        "gauge_lat": 58.44,
        "gauge_lon": -135.87,
        "outer_reflection_dist": 100,
        "mean_rh": 5.0,
    }


def _make_synthetic_basemaps(tmp_path):
    """Create minimal cached basemap PNGs for testing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basemaps = {}
    for name in ["coast", "regional", "fresnel"]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_facecolor("lightblue")
        ax.text(0.5, 0.5, name, transform=ax.transAxes, ha="center")
        path = tmp_path / f"basemap_{name}.png"
        fig.savefig(path, dpi=50)
        plt.close(fig)
        basemaps[name] = str(path)
        basemaps[f"{name}_extent"] = [-100, 100, -100, 100]

    basemaps["fresnel_local_coords"] = True
    basemaps["regional_local_coords"] = True
    basemaps["buffer_close"] = 120
    basemaps["buffer_wide"] = 5000
    basemaps["zoom_level"] = 14
    basemaps["center_x"] = 0
    basemaps["center_y"] = 0
    return basemaps


class TestCLIArgs:
    """Tests for --mode and --format CLI argument parsing."""

    @pytest.mark.unit
    def test_mode_defaults_to_presentation(self):
        """--mode should default to 'presentation'. We test via --help output."""
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert "--mode" in result.stdout
        assert "presentation" in result.stdout

    @pytest.mark.unit
    def test_format_defaults_to_gif(self):
        """--format should default to 'gif'. We test via --help output."""
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert "--format" in result.stdout
        assert "gif" in result.stdout

    @pytest.mark.unit
    def test_mode_rejects_invalid(self):
        """Invalid --mode value should cause argparse to exit with error."""
        result = subprocess.run(
            [sys.executable, SCRIPT, "--mode", "bogus"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    @pytest.mark.unit
    def test_format_rejects_invalid(self):
        """Invalid --format value should cause argparse to exit with error."""
        result = subprocess.run(
            [sys.executable, SCRIPT, "--format", "avi"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestCoverFrame:
    """Tests for the cover frame renderer."""

    @pytest.mark.unit
    def test_cover_frame_produces_png(self, tmp_path):
        """render_cover_frame() should create a PNG file."""
        from scripts.create_polar_animation import render_cover_frame

        metadata = _make_synthetic_metadata()
        basemaps = _make_synthetic_basemaps(tmp_path)
        output = tmp_path / "cover.png"

        frame_config = {
            "cached_basemaps": basemaps,
            "start_time": datetime(2024, 1, 1),
            "end_time": datetime(2024, 1, 14),
            "doy_start": 1,
            "doy_end": 14,
            "year": 2024,
            "figsize": (10, 10),
            "dpi": 100,
        }

        render_cover_frame(metadata, frame_config, output)
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.unit
    def test_cover_frame_dimensions(self, tmp_path):
        """Cover frame should be approximately 1000x1000 pixels."""
        from scripts.create_polar_animation import render_cover_frame
        from PIL import Image

        metadata = _make_synthetic_metadata()
        basemaps = _make_synthetic_basemaps(tmp_path)
        output = tmp_path / "cover.png"

        frame_config = {
            "cached_basemaps": basemaps,
            "start_time": datetime(2024, 1, 1),
            "end_time": datetime(2024, 1, 14),
            "doy_start": 1,
            "doy_end": 14,
            "year": 2024,
            "figsize": (10, 10),
            "dpi": 100,
        }

        render_cover_frame(metadata, frame_config, output)

        img = Image.open(output)
        width, height = img.size
        # Allow some tolerance for matplotlib bbox_inches
        assert 900 <= width <= 1100, f"Width {width} out of expected range"
        assert 900 <= height <= 1100, f"Height {height} out of expected range"
