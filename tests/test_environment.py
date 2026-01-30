# ABOUTME: Tests for environment setup and configuration.
# ABOUTME: Tests conda environment, required packages, and project structure.

import pytest
import sys
import os
import subprocess
from pathlib import Path
import json


class TestProjectStructure:
    """Tests for expected project directory structure."""

    @pytest.mark.unit
    def test_scripts_directory_exists(self, project_root):
        """Test that scripts directory exists."""
        scripts_dir = project_root / 'scripts'
        assert scripts_dir.exists()
        assert scripts_dir.is_dir()

    @pytest.mark.unit
    def test_config_directory_exists(self, project_root):
        """Test that config directory exists."""
        config_dir = project_root / 'config'
        assert config_dir.exists()
        assert config_dir.is_dir()

    @pytest.mark.unit
    def test_dashboard_components_exists(self, project_root):
        """Test that dashboard_components directory exists."""
        components_dir = project_root / 'dashboard_components'
        assert components_dir.exists()
        assert components_dir.is_dir()

    @pytest.mark.unit
    def test_required_scripts_exist(self, project_root):
        """Test that required scripts exist."""
        required_scripts = [
            'scripts/usgs_comparison.py',
            'scripts/find_reference_stations.py',
            'scripts/process_station.py',
            'scripts/run_gnssir_processing.py',
        ]

        for script in required_scripts:
            script_path = project_root / script
            assert script_path.exists(), f"Missing required script: {script}"


class TestConfigFiles:
    """Tests for configuration file validity."""

    @pytest.mark.unit
    def test_stations_config_exists(self, config_dir):
        """Test that stations_config.json exists."""
        config_path = config_dir / 'stations_config.json'
        assert config_path.exists()

    @pytest.mark.unit
    def test_stations_config_valid_json(self, config_dir):
        """Test that stations_config.json is valid JSON."""
        config_path = config_dir / 'stations_config.json'

        with open(config_path) as f:
            config = json.load(f)

        assert isinstance(config, dict)

    @pytest.mark.unit
    def test_tool_paths_exists(self, config_dir):
        """Test that tool_paths.json exists."""
        config_path = config_dir / 'tool_paths.json'
        if not config_path.exists():
            pytest.skip("tool_paths.json not present")

        with open(config_path) as f:
            config = json.load(f)

        assert isinstance(config, dict)


class TestPythonEnvironment:
    """Tests for Python environment setup."""

    @pytest.mark.unit
    def test_python_version(self):
        """Test that Python version is 3.9+."""
        major, minor = sys.version_info[:2]
        assert major >= 3
        assert minor >= 9, f"Python 3.9+ required, got {major}.{minor}"

    @pytest.mark.unit
    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy as np
        assert np.__version__ is not None

    @pytest.mark.unit
    def test_pandas_available(self):
        """Test that pandas is available."""
        import pandas as pd
        assert pd.__version__ is not None

    @pytest.mark.unit
    def test_matplotlib_available(self):
        """Test that matplotlib is available."""
        import matplotlib
        assert matplotlib.__version__ is not None

    @pytest.mark.unit
    def test_scipy_available(self):
        """Test that scipy is available."""
        import scipy
        assert scipy.__version__ is not None


class TestOptionalPackages:
    """Tests for optional packages that enhance functionality."""

    @pytest.mark.unit
    def test_dataretrieval_available(self):
        """Test that dataretrieval (USGS API) is available."""
        try:
            import dataretrieval
            assert dataretrieval is not None
        except ImportError:
            pytest.skip("dataretrieval not installed (optional)")

    @pytest.mark.unit
    def test_requests_available(self):
        """Test that requests is available."""
        import requests
        assert requests.__version__ is not None

    @pytest.mark.unit
    def test_streamlit_available(self):
        """Test that streamlit is available."""
        try:
            import streamlit
            assert streamlit.__version__ is not None
        except ImportError:
            pytest.skip("streamlit not installed (optional for CLI usage)")


class TestGnssreflEnvironment:
    """Tests for gnssrefl-specific environment setup."""

    @pytest.mark.unit
    def test_gnssrefl_available(self):
        """Test that gnssrefl package is available."""
        try:
            import gnssrefl
            assert gnssrefl is not None
        except ImportError:
            pytest.skip("gnssrefl not installed")

    @pytest.mark.unit
    def test_refl_code_env_var(self):
        """Test that REFL_CODE environment variable is set."""
        refl_code = os.environ.get('REFL_CODE')
        if refl_code is None:
            pytest.skip("REFL_CODE environment variable not set")
        assert Path(refl_code).exists(), f"REFL_CODE path does not exist: {refl_code}"

    @pytest.mark.unit
    def test_orbits_env_var(self):
        """Test that ORBITS environment variable is set."""
        orbits = os.environ.get('ORBITS')
        if orbits is None:
            pytest.skip("ORBITS environment variable not set")
        assert Path(orbits).exists(), f"ORBITS path does not exist: {orbits}"


class TestExternalTools:
    """Tests for external tool availability."""

    @pytest.mark.integration
    def test_gfzrnx_available(self, config_dir):
        """Test that gfzrnx tool is available."""
        tool_paths_file = config_dir / 'tool_paths.json'
        if not tool_paths_file.exists():
            pytest.skip("tool_paths.json not found")

        with open(tool_paths_file) as f:
            tool_paths = json.load(f)

        gfzrnx_path = tool_paths.get('gfzrnx')
        if not gfzrnx_path:
            pytest.skip("gfzrnx path not configured")

        if Path(gfzrnx_path).exists():
            result = subprocess.run([gfzrnx_path, '-h'],
                                    capture_output=True, text=True, timeout=10)
            # gfzrnx returns 0 for help
            assert result.returncode in [0, 1]
        else:
            pytest.skip(f"gfzrnx not found at {gfzrnx_path}")


class TestImports:
    """Tests for project module imports."""

    @pytest.mark.unit
    def test_import_visualizer(self):
        """Test importing visualizer module."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

        from visualizer import create_comparison_plot
        assert create_comparison_plot is not None

    @pytest.mark.unit
    def test_import_find_reference_stations(self):
        """Test importing find_reference_stations module."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

        from find_reference_stations import haversine_distance
        assert haversine_distance is not None

    @pytest.mark.unit
    def test_import_dashboard_components(self):
        """Test importing dashboard_components module."""
        sys.path.insert(0, str(Path(__file__).parent.parent))

        try:
            from dashboard_components import load_station_data
            assert load_station_data is not None
        except ImportError as e:
            # May fail due to streamlit decorators outside of streamlit
            if 'streamlit' in str(e).lower():
                pytest.skip("Streamlit not running")
            raise
