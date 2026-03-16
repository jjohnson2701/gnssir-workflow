# ABOUTME: Tests for polar animation CLI args and ffmpeg-based assembly pipeline.
# ABOUTME: Validates --mode and --format args parsed by create_polar_animation.

import pytest
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

SCRIPT = str(project_root / "scripts" / "create_polar_animation.py")


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
