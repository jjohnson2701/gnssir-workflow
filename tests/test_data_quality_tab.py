# ABOUTME: Tests for data quality tab pooled-daily extraction and fallback aggregation.
# ABOUTME: Verifies _get_pooled_daily handles both sentinel rows and missing sentinel rows.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_components.tabs.data_quality_tab import _get_pooled_daily


def _make_enriched(include_sentinel=True, n_days=30):
    """Build a synthetic enriched DataFrame.

    If include_sentinel=True, includes azimuth_bin=-1 / freq_group="ALL" pooled rows.
    """
    rng = np.random.default_rng(42)
    rows = []
    for d in range(n_days):
        date = f"2024-{(d // 30) + 1:02d}-{(d % 30) + 1:02d}"
        # Per-azimuth rows
        for az_bin in [0, 1, 2, 3]:
            rows.append({
                "date": date,
                "azimuth_bin": az_bin,
                "freq_group": "ALL",
                "rh_count": rng.integers(5, 20),
                "rh_mean": 2.0 + rng.normal(0, 0.1),
                "rh_std": 0.05 + rng.random() * 0.05,
                "amp_mean": 8.0 + rng.normal(0, 2),
                "amp_cv": 0.2 + rng.random() * 0.3,
                "p2n_mean": 3.0 + rng.normal(0, 0.5),
            })

        if include_sentinel:
            # Pooled sentinel row (azimuth_bin=-1)
            rows.append({
                "date": date,
                "azimuth_bin": -1,
                "freq_group": "ALL",
                "rh_count": rng.integers(20, 80),
                "rh_mean": 2.0 + rng.normal(0, 0.05),
                "rh_std": 0.04 + rng.random() * 0.03,
                "amp_mean": 8.0 + rng.normal(0, 1),
                "amp_cv": 0.2 + rng.random() * 0.2,
                "p2n_mean": 3.0 + rng.normal(0, 0.3),
            })

    return pd.DataFrame(rows)


@pytest.mark.unit
class TestGetPooledDaily:
    def test_extracts_sentinel_rows(self):
        enriched = _make_enriched(include_sentinel=True, n_days=10)
        pooled = _get_pooled_daily(enriched)
        # Should return exactly the sentinel rows
        assert len(pooled) == 10
        assert (pooled["azimuth_bin"] == -1).all()

    def test_fallback_when_no_sentinel(self):
        enriched = _make_enriched(include_sentinel=False, n_days=10)
        pooled = _get_pooled_daily(enriched)
        # Should still return 10 daily rows via aggregation
        assert len(pooled) == 10
        assert (pooled["azimuth_bin"] == -1).all()
        assert "rh_count" in pooled.columns

    def test_fallback_produces_reasonable_values(self):
        enriched = _make_enriched(include_sentinel=False, n_days=5)
        pooled = _get_pooled_daily(enriched)
        # Mean of 4 azimuth bins' rh_count should be reasonable
        assert pooled["rh_count"].min() > 0
        assert not pooled["rh_count"].isna().any()

    def test_returns_empty_for_empty_input(self):
        enriched = pd.DataFrame(columns=["date", "azimuth_bin", "freq_group", "rh_count"])
        pooled = _get_pooled_daily(enriched)
        assert pooled.empty
