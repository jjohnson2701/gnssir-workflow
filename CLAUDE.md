# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GNSS Interferometric Reflectometry (GNSS-IR) processing project that downloads RINEX 3 data from AWS S3, processes it to calculate reflector heights for water level estimation, and validates results against USGS gauge data. The system is designed for automated, parallel processing with comprehensive visualization capabilities.

## Development Environment Setup

```bash
# Activate Python virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Verify external tool paths in config/tool_paths.json
# Required tools: gfzrnx, rinex2snr, gnssir, quickLook
```

## Key Commands

### Main Processing Pipeline
```bash
# Run GNSS-IR processing for date range (parallel)
python scripts/run_gnssir_processing.py --station FORA --year 2024 --doy_start 260 --doy_end 266 --num_cores 4 --log_level INFO

# Enhanced USGS comparison with time lag analysis
python scripts/enhanced_usgs_comparison.py --station FORA --year 2024 --doy_start 260 --doy_end 266 --max_lag_days 5 --log_level INFO

# Multi-source analysis with external data integration
python scripts/multi_source_comparison.py --station FORA --year 2024 --doy_start 260 --doy_end 266 --log_level INFO
```

### USGS Integration Testing
```bash
# Quick USGS API test
python test_quick_usgs.py

# Progressive gauge search with visualization
python test_usgs_gauge_search.py --progressive --min_gauges 3 --plot

# Find optimal gauge with mapping
python scripts/find_optimal_gauge.py --station_id FORA --radius 25 --max_radius 100 --min_gauges 3 --plot_results

# Multi-gauge comparison analysis
bash scripts/multi_gauge_comparison.sh
```

### External Data Integration Testing
```bash
# Test NOAA CO-OPS and NDBC API clients
python test_external_apis.py --coops --ndbc --station FORA

# Test multi-source integration
python test_multi_source_integration.py --quick

# Test environmental analysis with synthetic data
python -c "from scripts.environmental_analysis import test_environmental_analyzer; test_environmental_analyzer()"
```

### Enhanced Dashboard Access
```bash
# Enhanced dashboard with multi-source analysis
streamlit run enhanced_dashboard_v2.py

# Original enhanced dashboard
streamlit run enhanced_dashboard.py

# Basic dashboard
streamlit run dashboard.py
```

### Data Quality Reporting
```bash
# Generate comprehensive quality report for full year
python scripts/data_quality_reporter.py --station FORA --year 2024

# Generate report for specific date range
python scripts/data_quality_reporter.py --station FORA --year 2024 --doy_start 260 --doy_end 266

# Generate quality report integrated with USGS comparison
python scripts/enhanced_usgs_comparison.py --station FORA --year 2024 --generate_quality_report
```

### Testing
```bash
# Run full test suite
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow/long-running tests

# Run individual test files
python test_enhanced_improvements.py
python test_segmented_analysis.py
```

## Architecture Overview

### Core Processing Flow
1. **Download**: RINEX 3 files from AWS S3 `doi-gnss` bucket
2. **Convert**: RINEX 3 → RINEX 2.11 using `gfzrnx`
3. **SNR Extraction**: Generate signal-to-noise ratio data with `rinex2snr`
4. **GNSS-IR Processing**: Calculate reflector heights using `gnssir`
5. **USGS Integration**: Find nearby gauges and retrieve water level data
6. **Analysis**: Time lag analysis, correlation statistics, visualization

### Module Organization

- **`scripts/core_processing/`**: Main processing orchestration
  - `config_loader.py`: Configuration management with validation
  - `daily_gnssir_worker.py`: Individual day processing logic
  - `parallel_orchestrator.py`: Multi-core coordination
  - `workspace_setup.py`: gnssrefl environment configuration

- **`scripts/utils/`**: Common utilities
  - `common_helpers.py`: Shared helper functions
  - `logging_config.py`: Centralized logging setup
  - `data_manager.py`: Data file management
  - `visualizer.py`: Base visualization utilities

- **`scripts/usgs_integration/`**: USGS data handling
  - `data_fetcher.py`: USGS API interactions
  - Progressive search algorithms for gauge discovery

- **`scripts/visualizer/`**: Specialized plotting modules
  - `comparison.py`: USGS vs GNSS-IR comparison plots
  - `timeseries.py`: Time series visualization
  - `advanced_viz.py`: Enhanced plot types with correlation analysis

### Configuration System

The project uses JSON-based configuration with three key files:

1. **`config/stations_config.json`**: Station definitions, coordinates, USGS settings
2. **`config/tool_paths.json`**: Paths to external executables (gfzrnx, gnssrefl tools)
3. **`config/fora_params.json`**: Station-specific processing parameters

### Skip Flags and Caching

The processing pipeline supports skip flags to avoid reprocessing completed stages:
- `--skip_download`: Skip RINEX 3 download if files exist
- `--skip_conversion`: Skip RINEX format conversion
- `--skip_snr`: Skip SNR file generation
- `--skip_processing`: Skip GNSS-IR processing

## Development Patterns

### Parallel Processing
The system uses `multiprocessing.Pool` for daily processing parallelization. When developing new processing functions, ensure they are stateless and can be safely parallelized.

### Configuration Loading
Always use `ConfigFactory` for loading configuration files. It provides validation and error handling:
```python
from scripts.utils.config_factory import ConfigFactory
config = ConfigFactory.load_station_config('FORA')
```

### Logging
Use the centralized logging configuration. Each processing run creates both main logs (`logs/`) and daily logs (`data/{station}/{year}/logs_daily/`).

### External Tool Integration
External tools (gfzrnx, gnssrefl) are wrapped in the `scripts/external_tools/` module. Always use these wrappers rather than calling tools directly.

### USGS Data Handling
USGS parameter codes in priority order:
- `62610`: Water level, NAVD88 (preferred)
- `62611`: Water level, NGVD29
- `62620`: Water level, MSL
- `00065`: Gage height (stage)

### External Data Integration
The system integrates multiple external data sources for comprehensive validation and environmental analysis:

#### NOAA CO-OPS API Integration
- **Tide predictions**: Harmonic-based predictions from CO-OPS stations
- **Water level observations**: Real-time and historical water level data
- **Station discovery**: Automatic nearby station identification
- **Multi-datum support**: NAVD88, MSL, MLLW datum options

```python
from scripts.external_apis.noaa_coops import NOAACOOPSClient
client = NOAACOOPSClient()
predictions = client.get_tide_predictions(station_id, start_date, end_date)
```

#### NDBC Buoy Data Integration
- **Meteorological data**: Wind speed/direction, atmospheric pressure
- **Wave measurements**: Significant wave height, wave period, wave direction
- **Environmental context**: Real-time oceanographic conditions

```python
from scripts.external_apis.ndbc_client import NDBCClient
client = NDBCClient()
met_data = client.get_meteorological_data(buoy_id, days_back=7)
```

#### Environmental Analysis Framework
- **Wind/wave correlation**: Impact assessment on measurement quality
- **Quality categorization**: Environmental condition classification
- **Statistical analysis**: Correlation significance testing
- **Automated recommendations**: Optimal measurement condition identification

```python
from scripts.environmental_analysis import EnvironmentalAnalyzer
analyzer = EnvironmentalAnalyzer()
results = analyzer.analyze_environmental_effects(gnssir_data, environmental_data)
```

#### Multi-Source Comparison
- **Integrated validation**: Cross-validation between GNSS-IR, USGS, CO-OPS, and NDBC
- **Temporal alignment**: Advanced algorithms for multi-source data synchronization
- **Comprehensive reporting**: Automated analysis reports with environmental context

```python
from scripts.multi_source_comparison import MultiSourceComparison
comparison = MultiSourceComparison()
results = comparison.run_comprehensive_analysis(station_name, year)
```

### WSL Compatibility
The project is designed for Windows Subsystem for Linux (WSL). File paths use Unix conventions, and external tools are expected to be in Linux format.

## Environment Variables

The system sets up gnssrefl-required environment variables:
- `REFL_CODE`: Points to `gnssrefl_data_workspace/refl_code/`
- `ORBITS`: Points to `gnssrefl_data_workspace/orbits/`

## Data Structures

### Input Data
- RINEX 3 files from AWS S3 in path pattern: `rinex3/{year}/{doy}/`
- Station coordinates in ellipsoidal system
- USGS gauge data via dataretrieval API

### Output Products
- Daily reflector height files: `data/{station}/{year}/rh_daily/`
- Annual combined datasets: `results_annual/{station}/`
- Visualization plots: PNG files with comparison analysis
- Processing logs: Detailed execution logs with timestamps

### Key Data Transformations
- RINEX 3 → RINEX 2.11 format conversion
- Reflector Height (RH) → Water Surface Ellipsoidal Height (WSE_ellips)
- Time series alignment and lag analysis
- Datum conversions for USGS data integration

## Testing Strategy

The project includes multiple testing approaches:
- **Unit tests**: Individual function testing in `tests/unit/`
- **Integration tests**: End-to-end workflow testing in `tests/integration/`
- **Standalone scripts**: Quick verification scripts (e.g., `test_quick_usgs.py`)
- **Pytest markers**: Categorized tests for selective execution

When developing new features, add appropriate tests and use the existing patterns for configuration loading, logging, and error handling.