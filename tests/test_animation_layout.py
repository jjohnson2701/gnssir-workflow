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


def _make_synthetic_gnssir_df(n=50, start=datetime(2024, 1, 1)):
    """Create minimal GNSS-IR DataFrame for testing frame rendering."""
    times = pd.date_range(start, periods=n, freq="3h")
    rng = np.random.RandomState(42)
    rh = 5.0 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12.42) + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "datetime": times,
        "RH": rh,
        "WSE_dm": -rh + rh.mean(),
        "Azim": rng.uniform(0, 360, n),
        "eminO": rng.uniform(5, 10, n),
        "emaxO": rng.uniform(15, 25, n),
        "PkNoise": rng.uniform(2, 6, n),
        "doy": times.dayofyear,
        "date": times.strftime("%Y-%m-%d"),
    })


def _make_synthetic_ref_df(n=200, start=datetime(2024, 1, 1)):
    """Create synthetic reference tide gauge data."""
    times = pd.date_range(start, periods=n, freq="30min")
    wl = 0.5 * np.sin(2 * np.pi * np.arange(n) / (12.42 * 2))  # semidiurnal in 30min steps
    return pd.DataFrame({
        "datetime": times,
        "wl_dm": wl,
    })


class TestAnimationFrame:
    """Tests for the stacked animation frame renderer."""

    @pytest.mark.unit
    def test_animation_frame_produces_png(self, tmp_path):
        """render_animation_frame() should create a PNG file."""
        from scripts.create_polar_animation import render_animation_frame

        df = _make_synthetic_gnssir_df()
        ref_df = _make_synthetic_ref_df()
        metadata = _make_synthetic_metadata()
        basemaps = _make_synthetic_basemaps(tmp_path)
        output = tmp_path / "frame_0000.png"

        start_time = df["datetime"].min()
        end_time = df["datetime"].max()
        mid = start_time + (end_time - start_time) / 2

        frame_config = {
            "mode": "presentation",
            "start_time": start_time,
            "end_time": end_time,
            "vmin_wl": -5.0,
            "vmax_wl": 5.0,
            "figsize": (10, 10),
            "dpi": 100,
            "cached_basemaps": basemaps,
        }

        render_animation_frame(
            df_all=df,
            df_current=df[(df["datetime"] >= mid - timedelta(hours=3))
                          & (df["datetime"] < mid + timedelta(hours=3))],
            df_accumulated=df[(df["datetime"] >= mid - timedelta(hours=3))
                              & (df["datetime"] < mid + timedelta(hours=3))],
            df_filtered_out=pd.DataFrame(columns=df.columns),
            ref_df=ref_df,
            metadata=metadata,
            frame_time=mid,
            frame_num=1,
            total_frames=10,
            output_path=output,
            frame_config=frame_config,
            bin_start=mid - timedelta(hours=3),
            bin_end=mid + timedelta(hours=3),
        )
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.unit
    def test_animation_frame_has_three_axes(self, tmp_path):
        """Frame should have 3 axes: progress bar, time series, reflection points."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.create_polar_animation import render_animation_frame

        df = _make_synthetic_gnssir_df()
        ref_df = _make_synthetic_ref_df()
        metadata = _make_synthetic_metadata()
        basemaps = _make_synthetic_basemaps(tmp_path)
        output = tmp_path / "frame_test.png"

        start_time = df["datetime"].min()
        end_time = df["datetime"].max()
        mid = start_time + (end_time - start_time) / 2

        frame_config = {
            "mode": "presentation",
            "start_time": start_time,
            "end_time": end_time,
            "vmin_wl": -5.0,
            "vmax_wl": 5.0,
            "figsize": (10, 10),
            "dpi": 100,
            "cached_basemaps": basemaps,
        }

        # We check the figure before it's closed by patching plt.close
        captured_fig = []
        orig_close = plt.close

        def capture_close(fig_arg=None):
            if fig_arg is not None:
                captured_fig.append(fig_arg)
            orig_close(fig_arg)

        plt.close = capture_close
        try:
            render_animation_frame(
                df_all=df,
                df_current=df.iloc[:5],
                df_accumulated=df.iloc[:5],
                df_filtered_out=pd.DataFrame(columns=df.columns),
                ref_df=ref_df,
                metadata=metadata,
                frame_time=mid,
                frame_num=1,
                total_frames=10,
                output_path=output,
                frame_config=frame_config,
                bin_start=mid - timedelta(hours=3),
                bin_end=mid + timedelta(hours=3),
            )
        finally:
            plt.close = orig_close

        assert len(captured_fig) == 1
        # 3 main axes + potentially a colorbar axis
        axes = captured_fig[0].get_axes()
        assert len(axes) >= 3, f"Expected >= 3 axes, got {len(axes)}"

    @pytest.mark.unit
    def test_presentation_mode_ts_window_14_days(self, tmp_path):
        """Presentation mode time series should show ~14-day rolling window."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.create_polar_animation import render_animation_frame

        # 30 days of data — TS should only show ~14 days around midpoint
        df = _make_synthetic_gnssir_df(n=240, start=datetime(2024, 1, 1))
        ref_df = _make_synthetic_ref_df(n=1440, start=datetime(2024, 1, 1))
        metadata = _make_synthetic_metadata()
        basemaps = _make_synthetic_basemaps(tmp_path)
        output = tmp_path / "frame_pres.png"

        start_time = df["datetime"].min()
        end_time = df["datetime"].max()
        mid = start_time + timedelta(days=15)

        frame_config = {
            "mode": "presentation",
            "start_time": start_time,
            "end_time": end_time,
            "vmin_wl": -5.0,
            "vmax_wl": 5.0,
            "figsize": (10, 10),
            "dpi": 100,
            "cached_basemaps": basemaps,
        }

        captured_fig = []
        orig_close = plt.close

        def capture_close(fig_arg=None):
            if fig_arg is not None:
                captured_fig.append(fig_arg)
            orig_close(fig_arg)

        plt.close = capture_close
        try:
            render_animation_frame(
                df_all=df,
                df_current=df.iloc[100:110],
                df_accumulated=df.iloc[100:110],
                df_filtered_out=pd.DataFrame(columns=df.columns),
                ref_df=ref_df,
                metadata=metadata,
                frame_time=mid,
                frame_num=5,
                total_frames=10,
                output_path=output,
                frame_config=frame_config,
                bin_start=mid - timedelta(hours=3),
                bin_end=mid + timedelta(hours=3),
            )
        finally:
            plt.close = orig_close

        # The time series axis (second axis in figure)
        fig = captured_fig[0]
        ax_ts = fig.get_axes()[1]  # [0]=progress, [1]=ts, [2]=refl
        xlim = ax_ts.get_xlim()
        from matplotlib.dates import num2date
        ts_span = num2date(xlim[1]) - num2date(xlim[0])
        # Should be ~14 days (with some padding), not 30 days
        assert ts_span.days <= 18, f"TS window too wide: {ts_span.days} days"
        assert ts_span.days >= 10, f"TS window too narrow: {ts_span.days} days"
