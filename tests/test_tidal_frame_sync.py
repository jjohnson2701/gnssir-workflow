# ABOUTME: Tests for tidal extreme detection used to sync animation frames.
# ABOUTME: Validates peak/trough finding from reference and GNSS-IR data.

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDetectTidalExtremes:
    """Tests for detecting high/low tide times from reference data."""

    @pytest.mark.unit
    def test_finds_peaks_and_troughs_in_sinusoidal_signal(self):
        """Should identify alternating highs and lows from clean tidal signal."""
        from scripts.create_polar_animation import detect_tidal_extremes

        # 3 days of hourly data with semidiurnal period
        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        period_hours = 12.42
        wl = np.sin(2 * np.pi * np.arange(72) / period_hours)
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": wl})

        extremes = detect_tidal_extremes(ref_df)

        # 3 days ≈ 5.8 semidiurnal cycles → ~11-12 extremes
        assert len(extremes) >= 10
        assert len(extremes) <= 14

    @pytest.mark.unit
    def test_extremes_are_roughly_half_tidal_period_apart(self):
        """Consecutive extremes should be ~6h apart (half semidiurnal)."""
        from scripts.create_polar_animation import detect_tidal_extremes

        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        period_hours = 12.42
        wl = np.sin(2 * np.pi * np.arange(72) / period_hours)
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": wl})

        extremes = detect_tidal_extremes(ref_df)
        diffs_hours = np.diff(extremes) / np.timedelta64(1, "h")

        assert np.all(diffs_hours > 4), f"Some extremes too close: {diffs_hours.min():.1f}h"
        assert np.all(diffs_hours < 9), f"Some extremes too far apart: {diffs_hours.max():.1f}h"

    @pytest.mark.unit
    def test_extremes_are_sorted_chronologically(self):
        """Returned extreme times should be in ascending order."""
        from scripts.create_polar_animation import detect_tidal_extremes

        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        wl = np.sin(2 * np.pi * np.arange(72) / 12.42)
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": wl})

        extremes = detect_tidal_extremes(ref_df)
        assert np.all(np.diff(extremes) > np.timedelta64(0))


class TestDetectTidalExtremesFromGNSSIR:
    """Tests for detecting tidal extremes from noisy GNSS-IR data."""

    @pytest.mark.unit
    def test_detects_extremes_from_noisy_irregular_data(self):
        """Should find tidal signal in scattered GNSS-IR retrievals."""
        from scripts.create_polar_animation import detect_tidal_extremes_from_gnssir

        np.random.seed(42)
        n_obs = 300
        hours = np.sort(np.random.uniform(0, 72, n_obs))
        period_hours = 12.42
        wse = np.sin(2 * np.pi * hours / period_hours) + np.random.normal(0, 0.15, n_obs)
        times = pd.Timestamp("2024-01-01") + pd.to_timedelta(hours, unit="h")
        df = pd.DataFrame({"datetime": times, "WSE_dm": wse})

        extremes = detect_tidal_extremes_from_gnssir(df)

        # Should find roughly the same number of extremes as clean signal
        assert len(extremes) >= 8
        assert len(extremes) <= 16

    @pytest.mark.unit
    def test_gnssir_extremes_spacing(self):
        """GNSS-IR detected extremes should be roughly half-tidal-period apart."""
        from scripts.create_polar_animation import detect_tidal_extremes_from_gnssir

        np.random.seed(42)
        n_obs = 300
        hours = np.sort(np.random.uniform(0, 72, n_obs))
        wse = np.sin(2 * np.pi * hours / 12.42) + np.random.normal(0, 0.1, n_obs)
        times = pd.Timestamp("2024-01-01") + pd.to_timedelta(hours, unit="h")
        df = pd.DataFrame({"datetime": times, "WSE_dm": wse})

        extremes = detect_tidal_extremes_from_gnssir(df)
        if len(extremes) >= 2:
            diffs_hours = np.diff(extremes) / np.timedelta64(1, "h")
            assert np.all(diffs_hours > 3), f"Some extremes too close: {diffs_hours.min():.1f}h"


class TestBuildFrameWindows:
    """Tests for mode-dependent frame window construction."""

    @pytest.mark.unit
    def test_analysis_windows_span_extreme_to_extreme(self):
        """Analysis mode: each window spans from one extreme to the next."""
        from scripts.create_polar_animation import build_frame_windows

        # 3 days of hourly data
        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        wl = np.sin(2 * np.pi * np.arange(72) / 12.42)
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": wl})

        start_time = times[0]
        end_time = times[-1]

        windows = build_frame_windows(
            mode="analysis",
            ref_df=ref_df,
            df=None,
            start_time=start_time,
            end_time=end_time,
            bin_hours=6,
            window_hours=6,
        )

        # Each window should be (start, end, center)
        assert len(windows) > 0
        for win_start, win_end, center in windows:
            assert win_end > win_start
            assert center >= win_start
            assert center <= win_end

        # Windows should be contiguous (end of one ~ start of next)
        for i in range(len(windows) - 1):
            gap = abs((windows[i + 1][0] - windows[i][1]).total_seconds())
            assert gap < 1, f"Gap between windows {i} and {i+1}: {gap}s"

    @pytest.mark.unit
    def test_presentation_uses_fixed_bins(self):
        """Presentation mode: windows use fixed bin_hours width."""
        from scripts.create_polar_animation import build_frame_windows

        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        wl = np.sin(2 * np.pi * np.arange(72) / 12.42)
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": wl})

        start_time = times[0]
        end_time = times[-1]

        windows = build_frame_windows(
            mode="presentation",
            ref_df=ref_df,
            df=None,
            start_time=start_time,
            end_time=end_time,
            bin_hours=6,
            window_hours=6,
        )

        assert len(windows) > 0
        # All windows should be ~6 hours wide
        for win_start, win_end, center in windows:
            span_hours = (win_end - win_start).total_seconds() / 3600
            assert 5.5 <= span_hours <= 6.5, f"Window span {span_hours}h, expected ~6h"

    @pytest.mark.unit
    def test_analysis_falls_back_when_no_extremes(self):
        """Analysis mode should fall back to regular bins with flat signal."""
        from scripts.create_polar_animation import build_frame_windows

        # Flat signal - no tidal extremes
        times = pd.date_range("2024-01-01", periods=72, freq="1h")
        ref_df = pd.DataFrame({"datetime": times, "wl_dm": np.zeros(72)})

        windows = build_frame_windows(
            mode="analysis",
            ref_df=ref_df,
            df=None,
            start_time=times[0],
            end_time=times[-1],
            bin_hours=6,
            window_hours=6,
        )

        # Should still produce windows (fallback to regular bins)
        assert len(windows) > 0
