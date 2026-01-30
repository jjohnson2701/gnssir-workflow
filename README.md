# GNSS-IR Processing Workflow

A comprehensive Python-based workflow for automated processing of GNSS Interferometric Reflectometry (GNSS-IR) data. This system provides end-to-end processing from RINEX 3 data download through multi-source validation, supporting full-year processing with parallel computation and visualization.

## Key Features

### Core Processing
- **Full-Year GNSS Data Processing**: Automated workflow from RINEX 3 download to water level estimation
- **Parallel Processing**: Multi-core support for efficient year-long data processing
- **Smart Caching**: Skip flags to avoid reprocessing completed stages
- **Comprehensive Logging**: Detailed logs at both system and daily processing levels

### Processing Flow
1. **Download**: RINEX 3 files from AWS S3 `doi-gnss` bucket
2. **Convert**: RINEX 3 → RINEX 2.11 using `gfzrnx`
3. **SNR Extraction**: Generate signal-to-noise ratio data with `rinex2snr`
4. **GNSS-IR Processing**: Calculate reflector heights using `gnssir`
5. **Reference Matching**: Find nearby gauges and retrieve water level data
6. **Analysis**: Time lag analysis, correlation statistics, visualization

### External Data Integration

**USGS Water Level Data**:
- Automatic discovery of nearby stream/tide gauges
- Multiple parameter support (water level NAVD88/NGVD29/MSL, gage height)
- Time lag analysis with cross-correlation

**NOAA CO-OPS Tide Data**:
- Real-time tide predictions from harmonic analysis
- Historical water level observations
- Multiple datum support (NAVD88, MSL, MLLW)

**NOAA ERDDAP Data**:
- Regional servers: AOOS (Alaska), PacIOOS (Hawaii), SECOORA (Southeast)
- Co-located water level sensors for high-accuracy validation

**NDBC Buoy Data**:
- Meteorological observations (wind speed/direction, pressure)
- Wave measurements (height, period, direction)

### Dashboard
- **Three-Tab Interface**: Overview, Monthly Analysis, Yearly Analysis
- **Multi-Source Integration**: GNSS-IR + USGS + NOAA CO-OPS + NDBC data
- **Publication-Quality Outputs**: Interactive Streamlit interface with export capabilities

## Project Structure

```
/project_root/
├── config/                       # Station configurations and tool paths
│   ├── stations_config.json     # Station definitions and reference sources
│   └── tool_paths.json          # External tool paths
├── scripts/
│   ├── run_gnssir_processing.py # Main orchestrator for full-year processing
│   ├── usgs_comparison.py       # USGS comparison with time lag analysis
│   ├── process_station.py       # Unified station processing workflow
│   ├── find_reference_stations.py # Reference station discovery
│   ├── core_processing/         # Core processing modules
│   ├── external_apis/           # NOAA CO-OPS and NDBC clients
│   ├── utils/                   # Utilities and helpers
│   └── visualizer/              # Publication-quality plotting
├── dashboard.py                 # Interactive Streamlit dashboard
├── dashboard_components/        # Modular dashboard architecture
│   └── tabs/                    # Tab implementations
├── tests/                       # Unit and integration test suite
│   └── fixtures/                # Test data samples
├── data/                        # Data storage (not in git)
└── results_annual/              # Processing results (not in git)
```

## Prerequisites

- **Python 3.9+** with packages: pandas, matplotlib, numpy, scipy, dataretrieval, streamlit
- **External Tools**: gfzrnx, gnssrefl package (rinex2snr, gnssir, quickLook)
- **Optional**: contextily, geopandas for enhanced mapping

### Conda Environment Setup

```bash
# Create and activate environment
conda create -n py39 python=3.9
conda activate py39

# Install gnssrefl
pip install gnssrefl

# Install additional dependencies
pip install -r requirements.txt

# Verify external tool paths in config/tool_paths.json
```

## Configuration

Each station requires configuration in `config/stations_config.json`:

```json
{
  "GLBX": {
    "station_id_4char_lower": "glbx",
    "ellipsoidal_height_m": -12.535,
    "latitude_deg": 58.455146658,
    "longitude_deg": -135.8884838318,
    "external_data_sources": {
      "erddap": {
        "enabled": true,
        "primary_reference": true,
        "station_name": "Bartlett Cove, AK"
      }
    }
  }
}
```

**Available Stations**: FORA (North Carolina), GLBX (Alaska), MDAI (Maryland), VALR (Hawaii)

## Quick Start

```bash
# 1. Process full year of GNSS data
python scripts/run_gnssir_processing.py --station GLBX --year 2024 --doy_start 1 --doy_end 31 --num_cores 8

# 2. Run unified processing workflow (GNSS-IR + reference matching + visualization)
python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

# 3. USGS comparison with time lag analysis
python scripts/usgs_comparison.py --station FORA --year 2024 --max_lag_days 5

# 4. Launch interactive dashboard
streamlit run dashboard.py

# 5. Find reference stations for a new station
python scripts/find_reference_stations.py --station GLBX --radius 50
```

### Skip Flags

The processing pipeline supports skip flags to avoid reprocessing:
- `--skip_download`: Skip RINEX 3 download if files exist
- `--skip_gnssir`: Skip GNSS-IR processing (use existing results)
- `--skip_comparison`: Skip reference comparison
- `--skip_viz`: Skip visualization

## Testing

```bash
# Run full test suite
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only

# Run with verbose output
pytest -v
```

The test suite includes:
- **Unit tests**: Haversine distance, correlation calculations, data transformations
- **Integration tests**: Real data loading for ERDDAP, USGS, and CO-OPS formats
- **Visualization tests**: Plot generation and export

## Module Organization

- **`scripts/core_processing/`**: Main processing orchestration
  - `config_loader.py`: Configuration management with validation
  - `daily_gnssir_worker.py`: Individual day processing logic
  - `parallel_orchestrator.py`: Multi-core coordination

- **`scripts/visualizer/`**: Specialized plotting modules
  - `comparison_plots.py`: Reference vs GNSS-IR comparison plots
  - `timeseries.py`: Time series visualization
  - `lag_analyzer.py`: Time lag analysis and visualization

- **`dashboard_components/`**: Modular dashboard architecture
  - `data_loader.py`: Multi-source data loading
  - `station_metadata.py`: Station configuration access

## Development Patterns

### Configuration Loading
```python
from scripts.utils.config_factory import ConfigFactory
config = ConfigFactory.load_station_config('FORA')
```

### Adding a New Station
1. Add basic station info to `config/stations_config.json`
2. Run `find_reference_stations.py` to identify nearby reference sources
3. Add the recommended reference configuration
4. Run processing pipeline

### USGS Parameter Codes (Priority Order)
- `62610`: Water level, NAVD88 (preferred)
- `62611`: Water level, NGVD29
- `62620`: Water level, MSL
- `00065`: Gage height (stage)

## Environment Variables

The system configures gnssrefl-required environment variables:
- `REFL_CODE`: Points to `gnssrefl_data_workspace/refl_code/`
- `ORBITS`: Points to `gnssrefl_data_workspace/orbits/`

## Acknowledgments

- gnssrefl package: [https://github.com/kristinemlarson/gnssrefl](https://github.com/kristinemlarson/gnssrefl)
- GFZ for the gfzrnx tool: [https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577895](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577895)
- USGS Water Data: [https://waterdata.usgs.gov/](https://waterdata.usgs.gov/)
- NOAA CO-OPS: [https://tidesandcurrents.noaa.gov/](https://tidesandcurrents.noaa.gov/)
- NOAA NDBC: [https://www.ndbc.noaa.gov/](https://www.ndbc.noaa.gov/)
