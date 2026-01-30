# ABOUTME: Pytest fixtures and configuration for GNSS-IR test suite.
# ABOUTME: Provides shared fixtures for station data, config, and mock data.

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir(project_root):
    """Return the config directory."""
    return project_root / 'config'


@pytest.fixture
def stations_config(config_dir):
    """Load the stations configuration."""
    config_path = config_dir / 'stations_config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


@pytest.fixture
def sample_station_config():
    """Provide a sample station configuration for testing."""
    return {
        "GLBX": {
            "name": "Glacier Bay",
            "latitude": 58.4556,
            "longitude": -135.8889,
            "ellipsoidal_height_m": 10.5,
            "erddap": {
                "enabled": True,
                "dataset_id": "aoos_bartlett_cove_wl",
                "station_name": "Bartlett Cove, AK",
                "distance_km": 0.004,
                "server": "AOOS",
                "primary_reference": True
            }
        },
        "FORA": {
            "name": "Fort Alava",
            "latitude": 48.1234,
            "longitude": -124.5678,
            "ellipsoidal_height_m": 8.2,
            "usgs_comparison": {
                "target_usgs_site": "12345678",
                "usgs_gauge_stated_datum": "NAVD88",
                "distance_km": 5.2
            }
        }
    }


@pytest.fixture
def sample_gnssir_data():
    """Generate sample GNSS-IR reflector height data."""
    np.random.seed(42)
    n_points = 100

    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i*6) for i in range(n_points)]

    # Simulate tidal signal + noise
    hours = np.arange(n_points) * 6
    tidal_signal = 1.5 * np.sin(2 * np.pi * hours / (12.42 * 24))  # M2 tide period
    noise = np.random.normal(0, 0.1, n_points)
    rh_values = 5.0 + tidal_signal + noise

    return pd.DataFrame({
        'datetime': dates,
        'rh': rh_values,
        'azimuth': np.random.uniform(0, 360, n_points),
        'amplitude': np.random.uniform(5, 15, n_points),
        'peak2noise': np.random.uniform(2.5, 5, n_points),
        'sat': np.random.randint(1, 32, n_points)
    })


@pytest.fixture
def sample_reference_data(sample_gnssir_data):
    """Generate sample reference water level data aligned with GNSS-IR."""
    np.random.seed(43)  # Different seed for reference noise
    df = sample_gnssir_data.copy()
    # Derive reference from WSE (antenna_height - rh) with small noise
    # This ensures proper correlation between WSE and reference
    antenna_height = 10.5
    wse = antenna_height - df['rh']
    df['reference_wl'] = wse + np.random.normal(0, 0.05, len(df))
    return df[['datetime', 'reference_wl']]


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix='gnssir_test_')
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_comparison_df(sample_gnssir_data, sample_reference_data):
    """Create a merged comparison DataFrame for testing with correct column names."""
    gnss = sample_gnssir_data.copy()
    ref = sample_reference_data.copy()

    gnss['merge_date'] = gnss['datetime'].dt.floor('h')
    ref['merge_date'] = ref['datetime'].dt.floor('h')

    merged = gnss.merge(ref, on='merge_date', suffixes=('', '_ref'))
    # Use column names that match the actual codebase conventions
    merged['wse_ellips_m'] = 10.5 - merged['rh']  # Antenna height - RH
    merged['usgs_value_m_median'] = merged['reference_wl']
    merged['residual'] = merged['wse_ellips_m'] - merged['usgs_value_m_median']
    # Also keep lowercase versions for backward compatibility with some tests
    merged['wse_ellips'] = merged['wse_ellips_m']
    merged['usgs_value'] = merged['usgs_value_m_median']

    return merged


# Fixtures for loading real sample data
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


@pytest.fixture
def real_gnssir_raw_data():
    """Load real GNSS-IR raw data sample from fixtures."""
    csv_path = FIXTURES_DIR / 'sample_gnssir_raw.csv'
    if not csv_path.exists():
        pytest.skip("Sample GNSS-IR raw data not found")
    return pd.read_csv(csv_path, parse_dates=['date'])


@pytest.fixture
def real_subdaily_matched_data():
    """Load real subdaily matched data sample from fixtures."""
    csv_path = FIXTURES_DIR / 'sample_subdaily_matched.csv'
    if not csv_path.exists():
        pytest.skip("Sample subdaily matched data not found")
    df = pd.read_csv(csv_path)
    df['gnss_datetime'] = pd.to_datetime(df['gnss_datetime'], format='ISO8601')
    df['bartlett_cove_datetime'] = pd.to_datetime(df['bartlett_cove_datetime'], format='ISO8601')
    return df


@pytest.fixture
def real_comparison_data():
    """Load real comparison data sample from fixtures."""
    csv_path = FIXTURES_DIR / 'sample_comparison.csv'
    if not csv_path.exists():
        pytest.skip("Sample comparison data not found")
    df = pd.read_csv(csv_path, parse_dates=['merge_date', 'date'])
    df.set_index('merge_date', inplace=True)
    return df


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as requiring external API")
