# ABOUTME: Tests for anomaly detection (PCA baseline model, scoring, state classification)
# ABOUTME: Validates all-feature model, per-feature contributions, observational state labels

import pytest
import numpy as np
import pandas as pd


def _make_synthetic_data(n_baseline=60, n_anomalous=30, n_features=5, seed=42):
    """Create synthetic daily features with baseline and anomalous periods.

    Baseline period: DOY 150-209 (60 days), features ~ N(0, 0.5)
    Anomalous period: DOY 300-329 (30 days), features shifted by 3 sigma
    Rest: DOY 1-149, 210-299, 330-365 — mixed baseline-like
    """
    rng = np.random.RandomState(seed)
    n_total = 365

    data = {"doy": np.arange(1, n_total + 1), "n_arcs": rng.randint(20, 60, n_total)}

    for i in range(n_features):
        values = np.zeros(n_total)
        for d in range(n_total):
            doy = d + 1
            if 150 <= doy <= 209:
                values[d] = rng.normal(0, 0.5)
            elif 300 <= doy <= 329:
                values[d] = rng.normal(3.0, 0.5)  # shifted
            else:
                values[d] = rng.normal(0, 1.0)
        data[f"feature_{i}"] = values

    return pd.DataFrame(data)


class TestBuildBaselineModel:
    """Test baseline model construction."""

    def test_uses_all_features(self):
        from scripts.anomaly_detector import build_baseline_model, _select_features

        df = _make_synthetic_data()
        features = _select_features(df)
        assert len(features) == 5  # all 5 feature_* columns
        assert all(f.startswith("feature_") for f in features)

    def test_model_has_required_keys(self):
        from scripts.anomaly_detector import build_baseline_model

        df = _make_synthetic_data()
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)

        assert "scaler" in model
        assert "pca" in model
        assert "mean_pc" in model
        assert "cov_pc_inv" in model
        assert "n_components" in model
        assert "feature_names" in model
        assert "pca_components" in model
        assert model["n_training_days"] >= 10

    def test_no_hardcoded_feature_subset(self):
        """Model should use ALL feature columns, not a hardcoded subset."""
        from scripts.anomaly_detector import build_baseline_model

        df = _make_synthetic_data(n_features=10)
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)

        assert len(model["feature_names"]) == 10

    def test_rejects_insufficient_data(self):
        from scripts.anomaly_detector import build_baseline_model

        df = _make_synthetic_data()
        tiny = df.head(5)
        with pytest.raises(ValueError):
            build_baseline_model(tiny)


class TestScoreAnomalies:
    """Test anomaly scoring with per-feature contributions."""

    def test_anomalous_days_have_higher_distance(self):
        from scripts.anomaly_detector import build_baseline_model, score_anomalies

        df = _make_synthetic_data()
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)

        scores = score_anomalies(df, model)
        assert "mahal_distance" in scores.columns

        # Anomalous period should have higher distances
        baseline_scores = scores[scores["doy"].between(150, 209)]
        anomalous_scores = scores[scores["doy"].between(300, 329)]

        assert anomalous_scores["mahal_distance"].median() > baseline_scores["mahal_distance"].median() * 2

    def test_per_feature_contributions_present(self):
        from scripts.anomaly_detector import build_baseline_model, score_anomalies

        df = _make_synthetic_data()
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)
        scores = score_anomalies(df, model)

        # Should have a contribution column for each feature
        for feat in model["feature_names"]:
            assert f"{feat}_contribution" in scores.columns

    def test_contributions_approximately_sum_to_distance_squared(self):
        """Per-feature contributions should approximately sum to D^2."""
        from scripts.anomaly_detector import build_baseline_model, score_anomalies

        df = _make_synthetic_data()
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)
        scores = score_anomalies(df, model)

        contrib_cols = [c for c in scores.columns if c.endswith("_contribution")]
        total_contrib = scores[contrib_cols].sum(axis=1)
        d_squared = scores["mahal_distance"] ** 2

        # Should be approximately equal (exact for orthogonal PCA)
        ratio = total_contrib / d_squared
        # Allow some tolerance since contributions are approximate
        assert ratio.median() > 0.5
        assert ratio.median() < 2.0


class TestClassifyStates:
    """Test observational state classification."""

    def test_state_labels_are_observational(self):
        """State labels must use observational terms, not ice/water/snow."""
        from scripts.anomaly_detector import classify_states

        scores = pd.DataFrame({
            "doy": range(1, 101),
            "mahal_distance": [1.0] * 50 + [5.0] * 50,
        })
        result = classify_states(scores, threshold=3.0)

        valid_states = {"baseline", "anomalous", "transition_in", "transition_out"}
        actual_states = set(result["state"].unique())
        assert actual_states.issubset(valid_states), (
            f"Invalid states: {actual_states - valid_states}"
        )

    def test_temporal_coherence(self):
        """Single-day spikes should not trigger state changes."""
        from scripts.anomaly_detector import classify_states

        # One spike day surrounded by baseline
        distances = [1.0] * 10 + [10.0] + [1.0] * 10
        scores = pd.DataFrame({
            "doy": range(1, 22),
            "mahal_distance": distances,
        })
        result = classify_states(scores, threshold=3.0, min_duration=3)

        # The spike should not create a transition
        assert (result["state"] == "baseline").all()

    def test_sustained_anomaly_detected(self):
        """Sustained high distance should trigger anomalous state."""
        from scripts.anomaly_detector import classify_states

        distances = [1.0] * 20 + [8.0] * 20 + [1.0] * 20
        scores = pd.DataFrame({
            "doy": range(1, 61),
            "mahal_distance": distances,
        })
        result = classify_states(scores, threshold=3.0)

        # Middle section should be anomalous or transition
        mid = result[(result["doy"] >= 25) & (result["doy"] <= 35)]
        assert "anomalous" in mid["state"].values or "transition_in" in mid["state"].values

    def test_n_arcs_preserved(self):
        """n_arcs should be preserved in output."""
        from scripts.anomaly_detector import build_baseline_model, score_anomalies

        df = _make_synthetic_data()
        baseline = df[(df["doy"] >= 150) & (df["doy"] <= 209)]
        model = build_baseline_model(baseline)
        scores = score_anomalies(df, model)

        assert "n_arcs" in scores.columns
        assert scores["n_arcs"].notna().all()
