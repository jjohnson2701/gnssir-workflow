# ABOUTME: Tests for meteorological data fetcher URL construction, CSV output, and config reading.
# ABOUTME: Uses mocked HTTP responses — no network calls in tests.

import pytest
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_met_data import (
    build_open_meteo_url,
    build_marine_url,
    parse_open_meteo_response,
    parse_marine_response,
    get_station_coords,
)


# ── URL Construction ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildUrl:
    def test_basic_url(self):
        url = build_open_meteo_url(70.67755, -52.115436, 2025)
        assert "latitude=70.67755" in url
        assert "longitude=-52.115436" in url
        assert "start_date=2025-01-01" in url
        assert "end_date=2025-12-31" in url
        assert "temperature_2m_mean" in url
        assert "temperature_2m_min" in url
        assert "temperature_2m_max" in url
        assert "timezone=UTC" in url
        assert url.startswith("https://archive-api.open-meteo.com/v1/archive")


# ── Response Parsing ────────────────────────────────────────────────────────


SAMPLE_RESPONSE = {
    "daily": {
        "time": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "temperature_2m_mean": [-18.2, -15.7, -20.1],
        "temperature_2m_min": [-22.4, -19.8, -24.3],
        "temperature_2m_max": [-14.1, -11.6, -16.0],
    }
}


@pytest.mark.unit
class TestParseResponse:
    def test_parses_three_days(self):
        df = parse_open_meteo_response(SAMPLE_RESPONSE)
        assert len(df) == 3
        assert list(df.columns) == ["date", "temp_mean_c", "temp_min_c", "temp_max_c"]

    def test_values_correct(self):
        df = parse_open_meteo_response(SAMPLE_RESPONSE)
        assert df.iloc[0]["temp_mean_c"] == pytest.approx(-18.2)
        assert df.iloc[1]["temp_min_c"] == pytest.approx(-19.8)
        assert df.iloc[2]["temp_max_c"] == pytest.approx(-16.0)

    def test_dates_are_strings(self):
        df = parse_open_meteo_response(SAMPLE_RESPONSE)
        assert df.iloc[0]["date"] == "2025-01-01"

    def test_handles_none_values(self):
        """API sometimes returns None for missing days."""
        response = {
            "daily": {
                "time": ["2025-01-01", "2025-01-02"],
                "temperature_2m_mean": [-18.2, None],
                "temperature_2m_min": [-22.4, None],
                "temperature_2m_max": [-14.1, None],
            }
        }
        df = parse_open_meteo_response(response)
        assert len(df) == 2
        assert pd.isna(df.iloc[1]["temp_mean_c"])


# ── Config Reading ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetStationCoords:
    def test_reads_umnq(self, tmp_path):
        config = {
            "UMNQ": {
                "latitude_deg": 70.67755,
                "longitude_deg": -52.115436,
            }
        }
        cfg_file = tmp_path / "stations_config.json"
        cfg_file.write_text(json.dumps(config))

        lat, lon = get_station_coords("UMNQ", cfg_file)
        assert lat == pytest.approx(70.67755)
        assert lon == pytest.approx(-52.115436)

    def test_unknown_station_raises(self, tmp_path):
        config = {"UMNQ": {"latitude_deg": 70.0, "longitude_deg": -52.0}}
        cfg_file = tmp_path / "stations_config.json"
        cfg_file.write_text(json.dumps(config))

        with pytest.raises(KeyError):
            get_station_coords("ZZZZ", cfg_file)


# ── Marine API (SST) ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestMarineUrl:
    def test_marine_url(self):
        url = build_marine_url(70.67, -52.11, 2025)
        assert "marine-api.open-meteo.com" in url
        assert "sea_surface_temperature_max" in url
        assert "start_date=2025-01-01" in url


@pytest.mark.unit
class TestParseMarineResponse:
    def test_parses_sst(self):
        response = {
            "daily": {
                "time": ["2025-01-01", "2025-01-02"],
                "sea_surface_temperature_max": [-1.7, -1.8],
            }
        }
        df = parse_marine_response(response)
        assert len(df) == 2
        assert "sst_max_c" in df.columns
        assert df.iloc[0]["sst_max_c"] == pytest.approx(-1.7)
