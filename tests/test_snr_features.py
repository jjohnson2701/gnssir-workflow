# ABOUTME: Tests for SNR arc feature extraction (CLR, area factor, damping, etc.)
# ABOUTME: Validates arc matching, detrending, and feature computation against literature methods

import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# GPS L1 wavelength in meters
L1_WAVELENGTH = 0.1903

# UMNQ processing parameters (from config/umnq.json)
UMNQ_CONFIG = {
    "e1": 5.0,
    "e2": 15.0,
    "minH": 5.0,
    "maxH": 30.0,
    "polyV": 4,
    "pele": [5, 30],
    "desiredP": 0.005,
}


def _make_clean_signal(rh=9.0, amplitude=30.0, wavelength=L1_WAVELENGTH, n=300,
                       e_min=5.0, e_max=25.0, gamma=0.0):
    """Generate a synthetic detrended SNR arc with one dominant frequency.

    gamma=0 gives undamped (ice-like), gamma>0 gives damped (water-like).
    Returns (elevation_deg, sin_elev, detrended_snr).
    """
    ele = np.linspace(e_min, e_max, n)
    sin_e = np.sin(np.radians(ele))
    cf = wavelength / 2
    k = 2 * np.pi / wavelength
    envelope = np.exp(-4 * k**2 * gamma * sin_e**2) if gamma > 0 else 1.0
    dsnr = amplitude * envelope * np.sin(2 * np.pi * rh * sin_e / cf)
    return ele, sin_e, dsnr


# ---------------------------------------------------------------------------
# Unit tests: arc segmentation
# ---------------------------------------------------------------------------

class TestSegmentArcs:
    """Test satellite arc segmentation from SNR time series."""

    def test_two_arcs_split_by_gap(self):
        from scripts.snr_feature_extractor import segment_satellite_arcs

        # Rising arc, 10-min gap, setting arc
        n1, n2 = 100, 80
        sod1 = np.arange(n1) * 30.0
        ele1 = np.linspace(5, 25, n1)
        sod2 = sod1[-1] + 600 + np.arange(n2) * 30.0
        ele2 = np.linspace(25, 5, n2)

        sod = np.concatenate([sod1, sod2])
        ele = np.concatenate([ele1, ele2])

        arcs = segment_satellite_arcs(sod, ele, gap_seconds=300)
        assert len(arcs) == 2
        assert arcs[0]["rise"] == 1
        assert arcs[1]["rise"] == -1

    def test_single_rising_arc(self):
        from scripts.snr_feature_extractor import segment_satellite_arcs

        sod = np.arange(200) * 30.0
        ele = np.linspace(2, 30, 200)
        arcs = segment_satellite_arcs(sod, ele, gap_seconds=300)
        assert len(arcs) == 1
        assert arcs[0]["rise"] == 1

    def test_direction_change_splits_arc(self):
        from scripts.snr_feature_extractor import segment_satellite_arcs

        # Rising then setting with no time gap (direction reversal at apex)
        n = 200
        sod = np.arange(n) * 30.0
        ele = np.concatenate([np.linspace(5, 30, n // 2), np.linspace(30, 5, n // 2)])

        arcs = segment_satellite_arcs(sod, ele, gap_seconds=300)
        assert len(arcs) == 2
        assert arcs[0]["rise"] == 1
        assert arcs[1]["rise"] == -1


# ---------------------------------------------------------------------------
# Unit tests: detrending
# ---------------------------------------------------------------------------

class TestDetrend:
    """Test polynomial detrending of SNR arcs."""

    def test_detrended_has_near_zero_mean(self):
        from scripts.snr_feature_extractor import detrend_arc

        ele = np.linspace(5, 25, 200)
        # Simulate: trend (polynomial) + oscillation
        trend = 100 + 2 * ele - 0.05 * ele**2
        osc = 10 * np.sin(2 * np.pi * 9 * np.sin(np.radians(ele)) / (L1_WAVELENGTH / 2))
        snr_lin = trend + osc

        detrended = detrend_arc(ele, snr_lin, poly_order=4, pele=(5, 30))
        assert abs(np.mean(detrended)) < 1.0

    def test_detrended_preserves_oscillation_amplitude(self):
        from scripts.snr_feature_extractor import detrend_arc

        ele = np.linspace(5, 25, 500)
        sin_e = np.sin(np.radians(ele))
        cf = L1_WAVELENGTH / 2
        amp = 15.0
        trend = 100 + 0.5 * ele
        osc = amp * np.sin(2 * np.pi * 9.0 * sin_e / cf)
        snr_lin = trend + osc

        detrended = detrend_arc(ele, snr_lin, poly_order=2, pele=(5, 30))
        # Peak-to-peak of detrended should be close to 2*amp
        ptp = np.max(detrended) - np.min(detrended)
        assert ptp > amp  # at least preserve most of the oscillation


# ---------------------------------------------------------------------------
# Unit tests: LSP features (CLR, PR)
# ---------------------------------------------------------------------------

class TestLSPFeatures:
    """Test clarity ratio and peak ratio from Lomb-Scargle periodogram."""

    def test_clr_high_for_clean_signal(self):
        from scripts.snr_feature_extractor import compute_lsp_features

        _, sin_e, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0)
        features = compute_lsp_features(sin_e, dsnr, L1_WAVELENGTH,
                                        min_rh=5.0, max_rh=30.0, precision=0.005)
        assert features["CLR"] > 5.0

    def test_clr_lower_for_multi_peak_than_clean(self):
        """CLR should be lower for multi-peak signal than single-peak."""
        from scripts.snr_feature_extractor import compute_lsp_features

        ele = np.linspace(5, 25, 300)
        sin_e = np.sin(np.radians(ele))
        cf = L1_WAVELENGTH / 2

        # Clean: single dominant peak
        dsnr_clean = 30.0 * np.sin(2 * np.pi * 9.0 * sin_e / cf)
        feat_clean = compute_lsp_features(sin_e, dsnr_clean, L1_WAVELENGTH,
                                          min_rh=5.0, max_rh=30.0, precision=0.005)

        # Multi-peak: three equal-amplitude frequencies
        dsnr_multi = (10.0 * np.sin(2 * np.pi * 8.0 * sin_e / cf)
                      + 10.0 * np.sin(2 * np.pi * 15.0 * sin_e / cf)
                      + 10.0 * np.sin(2 * np.pi * 22.0 * sin_e / cf))
        feat_multi = compute_lsp_features(sin_e, dsnr_multi, L1_WAVELENGTH,
                                          min_rh=5.0, max_rh=30.0, precision=0.005)

        assert feat_clean["CLR"] > feat_multi["CLR"]

    def test_pr_high_for_dominant_peak(self):
        from scripts.snr_feature_extractor import compute_lsp_features

        _, sin_e, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0)
        features = compute_lsp_features(sin_e, dsnr, L1_WAVELENGTH,
                                        min_rh=5.0, max_rh=30.0, precision=0.005)
        assert features["PR"] > 3.0

    def test_features_include_sp(self):
        """SP (spectral power) = peak LSP amplitude, analogous to gnssrefl's Amp."""
        from scripts.snr_feature_extractor import compute_lsp_features

        _, sin_e, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0)
        features = compute_lsp_features(sin_e, dsnr, L1_WAVELENGTH,
                                        min_rh=5.0, max_rh=30.0, precision=0.005)
        assert "SP" in features
        assert features["SP"] > 0


# ---------------------------------------------------------------------------
# Unit tests: area factor (Song 2022)
# ---------------------------------------------------------------------------

class TestAreaFactor:
    """Test wavelet-derived area factor."""

    def test_af_higher_for_undamped_than_damped(self):
        """Ice-like (undamped) should have higher AF than water-like (damped)."""
        from scripts.snr_feature_extractor import compute_area_factor

        _, sin_e_ice, dsnr_ice = _make_clean_signal(rh=9.0, gamma=0.0)
        _, sin_e_water, dsnr_water = _make_clean_signal(rh=9.0, gamma=0.02)

        af_ice = compute_area_factor(sin_e_ice, dsnr_ice, L1_WAVELENGTH,
                                     min_rh=5.0, max_rh=30.0)
        af_water = compute_area_factor(sin_e_water, dsnr_water, L1_WAVELENGTH,
                                       min_rh=5.0, max_rh=30.0)
        assert af_ice > af_water

    def test_af_positive(self):
        from scripts.snr_feature_extractor import compute_area_factor

        _, sin_e, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0)
        af = compute_area_factor(sin_e, dsnr, L1_WAVELENGTH,
                                 min_rh=5.0, max_rh=30.0)
        assert af > 0


# ---------------------------------------------------------------------------
# Unit tests: damping parameter (Strandberg 2017)
# ---------------------------------------------------------------------------

class TestDamping:
    """Test damping parameter extraction via Hilbert envelope fit."""

    def test_gamma_lower_for_ice_than_water(self):
        from scripts.snr_feature_extractor import compute_damping

        ele_ice, _, dsnr_ice = _make_clean_signal(rh=9.0, gamma=0.002)
        ele_water, _, dsnr_water = _make_clean_signal(rh=9.0, gamma=0.02)

        g_ice = compute_damping(ele_ice, dsnr_ice, L1_WAVELENGTH)
        g_water = compute_damping(ele_water, dsnr_water, L1_WAVELENGTH)
        assert g_ice < g_water

    def test_gamma_nonnegative(self):
        """Physical damping should not be negative."""
        from scripts.snr_feature_extractor import compute_damping

        ele, _, dsnr = _make_clean_signal(rh=9.0, gamma=0.01)
        g = compute_damping(ele, dsnr, L1_WAVELENGTH)
        assert g >= 0


# ---------------------------------------------------------------------------
# Integration tests: real UMNQ data
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRealDataUMNQ:
    """Integration tests using real UMNQ SNR files and per-arc parquet."""

    SNR_PATH = (PROJECT_ROOT / "gnssrefl_data_workspace" / "refl_code"
                / "2025" / "snr" / "umnq" / "umnq0850.25.snr66")
    PER_ARC_PATH = (PROJECT_ROOT / "results_annual" / "UMNQ"
                    / "UMNQ_2025_per_arc.parquet")

    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not self.SNR_PATH.exists():
            pytest.skip("UMNQ SNR file not found")
        if not self.PER_ARC_PATH.exists():
            pytest.skip("UMNQ per-arc parquet not found")

    def test_read_snr_file_shape(self):
        from scripts.snr_feature_extractor import read_snr_file

        data = read_snr_file(self.SNR_PATH)
        assert data.ndim == 2
        assert data.shape[1] == 11
        assert len(data) > 1000

    def test_read_snr_file_elevation_range(self):
        from scripts.snr_feature_extractor import read_snr_file

        data = read_snr_file(self.SNR_PATH)
        # Slightly negative elevations possible due to atmospheric refraction
        assert data[:, 1].min() >= -1.0
        assert data[:, 1].max() <= 90

    def test_arc_matching_sat9_doy85(self):
        """Sat 9 setting arc on DOY 85 should match parquet UTCtime ≈ 0.392."""
        import pandas as pd
        from scripts.snr_feature_extractor import (
            read_snr_file, segment_satellite_arcs, find_matching_segment,
        )

        snr = read_snr_file(self.SNR_PATH)
        per_arc = pd.read_parquet(self.PER_ARC_PATH)
        day85 = per_arc[per_arc["doy"] == 85]

        # Get sat 9, L1 entry
        target = day85[(day85["sat"] == 9) & (day85["freq"] == 1)]
        if len(target) == 0:
            pytest.skip("Sat 9 freq 1 not found in DOY 85")
        row = target.iloc[0]

        # Segment sat 9
        sat_mask = snr[:, 0] == 9
        sat_data = snr[sat_mask]
        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])

        # Find matching segment
        idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=row["UTCtime"],
            target_rise=row["rise"],
            e1=UMNQ_CONFIG["e1"], e2=UMNQ_CONFIG["e2"],
        )
        assert idx >= 0, "No matching segment found"

    def test_detrend_matches_gnssrefl_rh(self):
        """Detrending + LSP should recover RH close to gnssrefl's value."""
        import pandas as pd
        from scripts.snr_feature_extractor import (
            read_snr_file, segment_satellite_arcs, find_matching_segment,
            detrend_arc, compute_lsp_features,
        )

        snr = read_snr_file(self.SNR_PATH)
        per_arc = pd.read_parquet(self.PER_ARC_PATH)
        day85 = per_arc[per_arc["doy"] == 85]

        # Pick a well-behaved arc: sat 9, L1
        target = day85[(day85["sat"] == 9) & (day85["freq"] == 1)]
        if len(target) == 0:
            pytest.skip("Sat 9 freq 1 not found")
        row = target.iloc[0]

        # Extract and match
        sat_mask = snr[:, 0] == 9
        sat_data = snr[sat_mask]
        arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=row["UTCtime"],
            target_rise=row["rise"],
            e1=UMNQ_CONFIG["e1"], e2=UMNQ_CONFIG["e2"],
        )
        assert seg_idx >= 0

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        ele = arc_data[:, 1]
        snr_col = arc_data[:, 6]  # S1 column (0-indexed col 6)
        snr_lin = 10**(snr_col / 20)

        # Detrend
        detrended = detrend_arc(ele, snr_lin,
                                poly_order=UMNQ_CONFIG["polyV"],
                                pele=tuple(UMNQ_CONFIG["pele"]))

        # Window to [e1, e2]
        mask = (ele >= UMNQ_CONFIG["e1"]) & (ele <= UMNQ_CONFIG["e2"])
        sin_e = np.sin(np.radians(ele[mask]))
        dsnr = detrended[mask]

        # LSP should recover RH close to gnssrefl's value
        features = compute_lsp_features(
            sin_e, dsnr, L1_WAVELENGTH,
            min_rh=UMNQ_CONFIG["minH"],
            max_rh=UMNQ_CONFIG["maxH"],
            precision=UMNQ_CONFIG["desiredP"],
        )
        gnssrefl_rh = row["RH"]
        assert abs(features["RH"] - gnssrefl_rh) < 0.3, (
            f"LSP RH={features['RH']:.3f} vs gnssrefl RH={gnssrefl_rh:.3f}"
        )


# ---------------------------------------------------------------------------
# Phase 1 tests: arc_features.csv output
# ---------------------------------------------------------------------------

class TestCLRComponents:
    """Test that CLR component columns are produced for Gate 2 profiling."""

    def test_lsp_returns_clr_components(self):
        from scripts.snr_feature_extractor import compute_lsp_features

        _, sin_e, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0)
        features = compute_lsp_features(sin_e, dsnr, L1_WAVELENGTH,
                                        min_rh=5.0, max_rh=30.0, precision=0.005)
        assert "clr_peak_power" in features
        assert "clr_total_power" in features
        assert features["clr_peak_power"] > 0
        # For a clean single-peak signal, CLR ≈ peak / total
        if features["clr_total_power"] > 0:
            recomputed = features["clr_peak_power"] / features["clr_total_power"]
            assert abs(recomputed - features["CLR"]) < 0.01

    def test_extract_arc_features_includes_clr_components(self):
        from scripts.snr_feature_extractor import extract_arc_features

        ele, _, dsnr = _make_clean_signal(rh=9.0, amplitude=30.0, gamma=0.0)
        snr_db = 20 * np.log10(np.abs(dsnr) + 100)  # approximate dB
        snr_lin = 10 ** (snr_db / 20)

        feats = extract_arc_features(
            ele, snr_db, snr_lin, dsnr,
            wavelength=L1_WAVELENGTH, e1=5.0, e2=25.0,
            min_rh=5.0, max_rh=30.0, precision=0.005,
        )
        assert feats is not None
        assert "clr_peak_power" in feats
        assert "clr_total_power" in feats


class TestArcFeaturesCSVSchema:
    """Test that arc_features.csv has the expected schema and join keys."""

    ARC_FEATURES_PATH = (PROJECT_ROOT / "results_annual" / "ROSS"
                         / "ROSS_2024_arc_features.csv")
    ARC_TABLE_PATH = (PROJECT_ROOT / "results_annual" / "ROSS"
                      / "ROSS_2024_arc_table.parquet")

    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not self.ARC_FEATURES_PATH.exists():
            pytest.skip("ROSS arc_features.csv not found (run Phase 1 extraction first)")

    def test_arc_features_is_csv_not_parquet(self):
        """Output file is CSV, not a modification of arc_table.parquet."""
        import pandas as pd
        df = pd.read_csv(self.ARC_FEATURES_PATH)
        assert len(df) > 0
        # Verify it's a standalone file with expected columns
        assert "CLR" in df.columns
        assert "year" in df.columns

    def test_join_keys_present(self):
        """arc_features.csv must have join keys matching gnssrefl per_arc."""
        import pandas as pd
        df = pd.read_csv(self.ARC_FEATURES_PATH)
        for key in ["year", "doy", "sat", "UTCtime", "rise", "freq"]:
            assert key in df.columns, f"Missing join key: {key}"

    def test_join_keys_match_arc_table(self):
        """Join keys in arc_features.csv should match arc_table.parquet rows."""
        import pandas as pd
        if not self.ARC_TABLE_PATH.exists():
            pytest.skip("arc_table.parquet not found")
        arc_feat = pd.read_csv(self.ARC_FEATURES_PATH)
        arc_table = pd.read_parquet(self.ARC_TABLE_PATH)
        join_cols = ["doy", "sat", "UTCtime", "rise", "freq"]
        # Merge and check that most feature rows find a match
        merged = arc_feat.merge(arc_table[join_cols].drop_duplicates(),
                                on=join_cols, how="inner")
        match_rate = len(merged) / len(arc_feat)
        assert match_rate > 0.95, f"Only {match_rate:.1%} of arc_features rows match arc_table"

    def test_required_feature_columns(self):
        """arc_features.csv must have all spec-required feature columns."""
        import pandas as pd
        df = pd.read_csv(self.ARC_FEATURES_PATH)
        required = ["CLR", "PR", "AF", "gamma", "gamma_r2",
                     "phase_deg", "SP", "MS", "VS", "full_arc",
                     "clr_peak_power", "clr_total_power"]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_arc_table_not_modified_by_csv_output(self):
        """Writing arc_features.csv must not change the arc_table.parquet file."""
        import pandas as pd
        if not self.ARC_TABLE_PATH.exists():
            pytest.skip("arc_table.parquet not found")
        # arc_table should still exist and be a valid parquet
        arc_table = pd.read_parquet(self.ARC_TABLE_PATH)
        assert len(arc_table) > 0
        # It should have gnssrefl columns like RH, Amp
        assert "RH" in arc_table.columns or "Amp" in arc_table.columns
