# ABOUTME: Tests for ffmpeg-based animation assembly.
# ABOUTME: Validates GIF/MP4 output from frame PNGs via ffmpeg concat demuxer.

import pytest
import sys
import shutil
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _create_synthetic_frames(tmp_path, n_frames=5, size=(100, 100)):
    """Create small synthetic PNG frames for testing assembly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    for i in range(n_frames):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_facecolor(plt.cm.viridis(i / max(1, n_frames - 1)))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=100)
        plt.close(fig)

    # Also create a cover frame
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_facecolor("white")
    ax.text(0.5, 0.5, "COVER", transform=ax.transAxes, ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(frames_dir / "cover_frame.png", dpi=100)
    plt.close(fig)

    return frames_dir


class TestAssembleWithFfmpeg:
    """Tests for ffmpeg-based animation assembly."""

    @pytest.mark.unit
    def test_produces_gif(self, tmp_path):
        """Should produce a GIF file from frame PNGs."""
        from scripts.create_polar_animation import assemble_with_ffmpeg

        frames_dir = _create_synthetic_frames(tmp_path)
        cover_path = frames_dir / "cover_frame.png"
        output = tmp_path / "test.gif"

        assemble_with_ffmpeg(
            frames_dir=frames_dir,
            cover_frame_path=cover_path,
            output_path=output,
            fps=2,
            output_format="gif",
        )
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.unit
    def test_produces_mp4(self, tmp_path):
        """Should produce an MP4/WebM file from frame PNGs."""
        from scripts.create_polar_animation import assemble_with_ffmpeg

        frames_dir = _create_synthetic_frames(tmp_path)
        cover_path = frames_dir / "cover_frame.png"
        output = tmp_path / "test.mp4"

        assemble_with_ffmpeg(
            frames_dir=frames_dir,
            cover_frame_path=cover_path,
            output_path=output,
            fps=2,
            output_format="mp4",
        )
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.unit
    def test_raises_on_missing_ffmpeg(self, tmp_path, monkeypatch):
        """Should raise a clear error if ffmpeg is not found."""
        from scripts import create_polar_animation as mod

        frames_dir = _create_synthetic_frames(tmp_path)
        cover_path = frames_dir / "cover_frame.png"
        output = tmp_path / "test.gif"

        monkeypatch.setattr(mod, "FFMPEG_PATH", "/nonexistent/ffmpeg")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            mod.assemble_with_ffmpeg(
                frames_dir=frames_dir,
                cover_frame_path=cover_path,
                output_path=output,
                fps=2,
            )

    @pytest.mark.unit
    def test_framelist_includes_cover(self, tmp_path):
        """The generated framelist should reference the cover frame."""
        from scripts.create_polar_animation import assemble_with_ffmpeg

        frames_dir = _create_synthetic_frames(tmp_path)
        cover_path = frames_dir / "cover_frame.png"
        output = tmp_path / "test.gif"

        assemble_with_ffmpeg(
            frames_dir=frames_dir,
            cover_frame_path=cover_path,
            output_path=output,
            fps=2,
            output_format="gif",
        )

        framelist = frames_dir / "framelist.txt"
        assert framelist.exists()
        content = framelist.read_text()
        assert "cover_frame.png" in content
