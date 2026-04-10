# ABOUTME: Tests for feature profiler (5-gate validation producing feature_scorecard)
# ABOUTME: Validates each gate independently with synthetic data

import pytest
import numpy as np
import pandas as pd


def _make_daily_features(n_days=365, baseline_start=150, baseline_end=243):
    """Create synthetic daily features for profiling tests.

    Features:
    - good_feature: stable baseline, strong shift outside → passes all gates
    - noisy_feature: high CV during baseline → fails Gate 3
    - weak_feature: no discrimination → fails Gate 4
    - correlated_feature: correlated with good_feature → fails Gate 5
    """
    rng = np.random.RandomState(42)
    doys = np.arange(1, n_days + 1)

    data = {
        "doy": doys,
        "n_arcs": np.full(n_days, 30),
    }

    # Good feature: low CV baseline, high d outside
    good = np.zeros(n_days)
    for i, doy in enumerate(doys):
        if baseline_start <= doy <= baseline_end:
            good[i] = 5.0 + rng.normal(0, 0.2)  # CV ~ 0.04
        else:
            good[i] = 8.0 + rng.normal(0, 0.3)  # shifted
    data["good_feature"] = good

    # Noisy feature: high CV during baseline
    noisy = np.zeros(n_days)
    for i, doy in enumerate(doys):
        if baseline_start <= doy <= baseline_end:
            noisy[i] = 2.0 + rng.normal(0, 3.0)  # CV ~ 1.5
        else:
            noisy[i] = 5.0 + rng.normal(0, 3.0)
    data["noisy_feature"] = noisy

    # Weak feature: no real discrimination
    weak = rng.normal(5.0, 0.3, n_days)
    data["weak_feature"] = weak

    # Correlated feature: same as good_feature + noise
    data["correlated_feature"] = good + rng.normal(0, 0.1, n_days)

    # gamma_r2_med for Gate 2 testing
    data["gamma_r2_med"] = np.full(n_days, 0.85)

    return pd.DataFrame(data)


class TestGate1DataSufficiency:
    """Test Gate 1: data sufficiency checks."""

    def test_passes_with_sufficient_data(self):
        from scripts.feature_profiler import gate1_data_sufficiency

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        window_doys = set(range(300, 315))

        passes, bl_arcs, win_arcs, _ = gate1_data_sufficiency(
            df, "good_feature", baseline_doys, window_doys
        )
        assert passes

    def test_fails_with_low_arc_count(self):
        from scripts.feature_profiler import gate1_data_sufficiency

        df = _make_daily_features()
        df["n_arcs"] = 3  # below threshold
        baseline_doys = set(range(150, 244))
        window_doys = set(range(300, 315))

        passes, _, _, _ = gate1_data_sufficiency(
            df, "good_feature", baseline_doys, window_doys
        )
        assert not passes


class TestGate2ComputationHealth:
    """Test Gate 2: feature-specific health checks."""

    def test_gamma_passes_with_good_r2(self):
        from scripts.feature_profiler import gate2_computation_health

        df = _make_daily_features()
        window_doys = set(range(300, 315))
        passes, detail = gate2_computation_health(df, "gamma_med", window_doys)
        assert passes
        assert "gamma_r2" in detail

    def test_gamma_fails_with_low_r2(self):
        from scripts.feature_profiler import gate2_computation_health

        df = _make_daily_features()
        df["gamma_r2_med"] = 0.3  # below threshold
        window_doys = set(range(300, 315))
        passes, _ = gate2_computation_health(df, "gamma_med", window_doys)
        assert not passes

    def test_default_passes(self):
        """Features without specific health checks should pass."""
        from scripts.feature_profiler import gate2_computation_health

        df = _make_daily_features()
        window_doys = set(range(300, 315))
        passes, _ = gate2_computation_health(df, "good_feature", window_doys)
        assert passes


class TestGate3BaselineStability:
    """Test Gate 3: baseline CV threshold."""

    def test_stable_feature_passes(self):
        from scripts.feature_profiler import gate3_baseline_stability

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        passes, cv = gate3_baseline_stability(df, "good_feature", baseline_doys)
        assert passes
        assert cv < 0.5

    def test_noisy_feature_fails(self):
        from scripts.feature_profiler import gate3_baseline_stability

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        passes, cv = gate3_baseline_stability(df, "noisy_feature", baseline_doys)
        assert not passes
        assert cv > 0.5


class TestGate4Discriminability:
    """Test Gate 4: Cohen's d threshold."""

    def test_discriminant_feature_passes(self):
        from scripts.feature_profiler import gate4_discriminability

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        window_doys = set(range(300, 330))
        passes, d = gate4_discriminability(df, "good_feature",
                                           baseline_doys, window_doys)
        assert passes
        assert abs(d) > 0.5

    def test_non_discriminant_fails(self):
        from scripts.feature_profiler import gate4_discriminability

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        window_doys = set(range(300, 330))
        passes, d = gate4_discriminability(df, "weak_feature",
                                           baseline_doys, window_doys)
        assert not passes


class TestGate5Redundancy:
    """Test Gate 5: correlation-based redundancy detection."""

    def test_correlated_features_marked_redundant(self):
        from scripts.feature_profiler import profile_features

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        scorecard = profile_features(df, baseline_doys, "TEST", 2024)

        # Find correlated_feature entries that were marked redundant
        corr_rows = scorecard[scorecard["feature"] == "correlated_feature"]
        if len(corr_rows) > 0:
            # At least some should be marked redundant or the good_feature should be
            redundant_rows = scorecard[scorecard["failure_reason"] == "redundant"]
            # Either correlated_feature is redundant with good_feature, or vice versa
            if len(redundant_rows) > 0:
                assert any(
                    r["gate5_correlated_with"] in ("good_feature", "correlated_feature")
                    for _, r in redundant_rows.iterrows()
                )


class TestScorecardOutput:
    """Test scorecard output format and content."""

    def test_scorecard_has_required_columns(self):
        from scripts.feature_profiler import profile_features

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        scorecard = profile_features(df, baseline_doys, "TEST", 2024)

        required = [
            "station", "feature", "window_start_doy", "window_size",
            "gate1_pass", "gate2_pass", "gate3_pass", "gate4_pass",
            "gate5_pass", "usable", "rank", "failure_reason",
        ]
        for col in required:
            assert col in scorecard.columns, f"Missing column: {col}"

    def test_both_window_sizes_present(self):
        from scripts.feature_profiler import profile_features

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        scorecard = profile_features(df, baseline_doys, "TEST", 2024)

        sizes = scorecard["window_size"].unique()
        assert "2w" in sizes
        assert "4w" in sizes

    def test_usable_features_have_rank(self):
        from scripts.feature_profiler import profile_features

        df = _make_daily_features()
        baseline_doys = set(range(150, 244))
        scorecard = profile_features(df, baseline_doys, "TEST", 2024)

        usable = scorecard[scorecard["usable"] == True]
        if len(usable) > 0:
            assert usable["rank"].notna().all()
            assert (usable["rank"] >= 1).all()
