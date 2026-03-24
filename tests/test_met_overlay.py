# ABOUTME: Tests for temperature overlay on ice score and data quality gap analysis.
# ABOUTME: Covers met data loading, freezing point inference, and gap classification.

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_components.tabs.ice_comparison_tab import (
    _load_met_data,
    _get_freezing_point,
)
from dashboard_components.tabs.data_quality_tab import (
    _analyze_gaps,
    _get_pooled_daily,
)


# ── Met Data Loading ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLoadMetData:
    def test_loads_csv(self, tmp_path):
        """Loads met CSV and returns DataFrame with date column."""
        csv_path = tmp_path / "results_annual" / "UMNQ" / "UMNQ_2025_met_daily.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text(
            "date,temp_mean_c,temp_min_c,temp_max_c\n"
            "2025-01-01,-18.2,-22.4,-14.1\n"
            "2025-01-02,-15.7,-19.8,-11.6\n"
        )
        df = _load_met_data("UMNQ", 2025, project_root=tmp_path)
        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_returns_none_when_missing(self, tmp_path):
        result = _load_met_data("ZZZZ", 2025, project_root=tmp_path)
        assert result is None


# ── Freezing Point ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetFreezingPoint:
    def test_seawater_default_for_ice_station(self, tmp_path):
        """Stations with ice_free_months → seawater freezing (-1.8)."""
        config = {"UMNQ": {"ice_free_months": [7, 8]}}
        cfg_file = tmp_path / "stations_config.json"
        cfg_file.write_text(json.dumps(config))

        fp = _get_freezing_point("UMNQ", config_path=cfg_file)
        assert fp == pytest.approx(-1.8)

    def test_freshwater_default(self, tmp_path):
        """Stations without ice_free_months → freshwater freezing (0.0)."""
        config = {"GLBX": {"latitude_deg": 58.0}}
        cfg_file = tmp_path / "stations_config.json"
        cfg_file.write_text(json.dumps(config))

        fp = _get_freezing_point("GLBX", config_path=cfg_file)
        assert fp == pytest.approx(0.0)

    def test_explicit_override(self, tmp_path):
        """Explicit met_data.freezing_point_c overrides defaults."""
        config = {"UMNQ": {
            "ice_free_months": [7, 8],
            "met_data": {"freezing_point_c": -2.0},
        }}
        cfg_file = tmp_path / "stations_config.json"
        cfg_file.write_text(json.dumps(config))

        fp = _get_freezing_point("UMNQ", config_path=cfg_file)
        assert fp == pytest.approx(-2.0)

    def test_missing_config_file(self, tmp_path):
        """Missing config file → seawater default."""
        fp = _get_freezing_point("UMNQ", config_path=tmp_path / "nonexistent.json")
        assert fp == pytest.approx(-1.8)


# ── Gap Analysis ────────────────────────────────────────────────────────────


def _make_enriched(dates, rh_counts):
    """Build minimal enriched-style DataFrame with pooled sentinel rows."""
    return pd.DataFrame({
        "date": dates,
        "rh_count": rh_counts,
        "azimuth_bin": -1,
        "freq_group": "ALL",
        "amp_mean": 10.0,
        "p2n_mean": 4.0,
        "rh_std": 0.05,
        "amp_cv": 0.2,
    })


def _make_per_arc(dates_with_arcs, arcs_per_day=20):
    """Build minimal per-arc DataFrame."""
    rows = []
    for d in dates_with_arcs:
        for i in range(arcs_per_day):
            rows.append({"date": d, "azimuth": i * 18, "amplitude": 8.0})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "azimuth", "amplitude"])


@pytest.mark.unit
class TestAnalyzeGaps:
    def test_classifies_data_days(self):
        """Days with pooled enriched data → status 'data'."""
        enriched = _make_enriched(["2025-01-01", "2025-01-02"], [30, 25])
        per_arc = _make_per_arc(["2025-01-01", "2025-01-02"])
        gaps = _analyze_gaps(enriched, per_arc, 2025)

        jan1 = gaps[gaps["date"].dt.date == datetime(2025, 1, 1).date()]
        assert jan1.iloc[0]["status"] == "data"

    def test_classifies_qc_gaps(self):
        """Days with arcs but no enriched data → status 'qc_gap'."""
        # Only Jan 1 in enriched, but arcs exist for Jan 1 and Jan 2
        enriched = _make_enriched(["2025-01-01"], [30])
        per_arc = _make_per_arc(["2025-01-01", "2025-01-02"])
        gaps = _analyze_gaps(enriched, per_arc, 2025)

        jan2 = gaps[gaps["date"].dt.date == datetime(2025, 1, 2).date()]
        assert jan2.iloc[0]["status"] == "qc_gap"
        assert jan2.iloc[0]["arc_attempts"] == 20

    def test_classifies_no_data(self):
        """Days with neither arcs nor enriched → status 'no_data'."""
        enriched = _make_enriched(["2025-01-01"], [30])
        per_arc = _make_per_arc(["2025-01-01"])
        gaps = _analyze_gaps(enriched, per_arc, 2025)

        jan3 = gaps[gaps["date"].dt.date == datetime(2025, 1, 3).date()]
        assert jan3.iloc[0]["status"] == "no_data"
        assert jan3.iloc[0]["arc_attempts"] == 0

    def test_covers_full_year(self):
        """Returns 365 rows for non-leap year."""
        enriched = _make_enriched(["2025-06-15"], [30])
        per_arc = _make_per_arc(["2025-06-15"])
        gaps = _analyze_gaps(enriched, per_arc, 2025)
        assert len(gaps) == 365

    def test_handles_empty_data(self):
        """Works with empty enriched and per_arc."""
        gaps = _analyze_gaps(None, None, 2025)
        assert len(gaps) == 365
        assert (gaps["status"] == "no_data").all()
