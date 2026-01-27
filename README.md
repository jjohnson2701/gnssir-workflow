# GNSS-IR Processing Workflow

A comprehensive Python-based workflow for automated processing of GNSS Interferometric Reflectometry (GNSS-IR) data. This system provides end-to-end processing from RINEX 3 data download through multi-source validation, supporting full-year processing with parallel computation and advanced visualization capabilities.

![FORA Reflector Height](./docs/sample_plot.png)

## Key Features

### Core Processing Capabilities
- **Full-Year GNSS Data Processing**: Automated workflow from RINEX 3 download to water level estimation
- **Parallel Processing**: Multi-core support for efficient year-long data processing
- **Smart Caching**: Skip flags to avoid reprocessing completed stages
- **Comprehensive Logging**: Detailed logs at both system and daily processing levels

### Data Sources and Integration
- **GNSS-IR Processing**: 
  - Downloads RINEX 3 files from AWS S3 (doi-gnss bucket)
  - Converts to RINEX 2.11 using gfzrnx
  - Extracts SNR data and calculates reflector heights
  - Supports multiple GNSS constellations (GPS, GLONASS, Galileo, BeiDou)

### External Data Integration
- **USGS Water Level Data**:
  - Automatic discovery of nearby stream/tide gauges
  - Multiple parameter support (water level NAVD88/NGVD29/MSL, gage height)
  - Time lag analysis with cross-correlation
  - Progressive search algorithms for optimal gauge selection

- **NOAA CO-OPS Tide Data**:
  - Real-time tide predictions from harmonic analysis
  - Historical water level observations
  - Multiple datum support (NAVD88, MSL, MLLW)
  - Automatic nearby station discovery

- **NDBC Buoy Data**:
  - Meteorological observations (wind speed/direction, pressure)
  - Wave measurements (height, period, direction)
  - Environmental condition monitoring

### Multi-Source Comparison and Analysis
- **Comprehensive Validation**: Cross-validates GNSS-IR against USGS, tide predictions, and buoy data
- **Environmental Analysis**: 
  - Wind/wave impact assessment on measurement quality
  - Statistical correlation analysis
  - Quality categorization based on environmental conditions
  - Automated recommendations for optimal measurement windows

- **Time Series Alignment**: Advanced algorithms for multi-source data synchronization
- **Cached Data Management**: Efficient storage and retrieval of paired station data

### Enhanced Dashboard v3 (January 2025)
- **Three-Tab Interface**: Overview, Monthly Analysis, Yearly Analysis with 6 advanced visualization types
- **Multi-Source Integration**: GNSS-IR + USGS + NOAA CO-OPS + NDBC data with automatic discovery
- **Fixed RMSE Calculations**: Proper RH-to-WSE conversion and correlation analysis
- **Publication-Quality Outputs**: Interactive Streamlit interface with caching and export capabilities

## Project Structure

```
/project_root/
|-- config/                       # Station configurations and tool paths
|-- scripts/
|   |-- run_gnssir_processing.py  # Main orchestrator for full-year processing
|   |-- enhanced_usgs_comparison.py # USGS comparison with time lag analysis
|   |-- multi_source_comparison.py # Multi-source validation
|   |-- core_processing/          # Core processing modules
|   |-- external_apis/            # NOAA CO-OPS and NDBC clients
|   |-- utils/ & visualizer/      # Utilities and publication-quality plotting
|-- enhanced_dashboard_v3.py      # Interactive Streamlit dashboard
|-- dashboard_components/         # Modular dashboard architecture (New 2025)
|   |-- tabs/                    # Three-tab interface implementation
|-- tests/                        # Unit and integration test suite
|-- data/ & results_annual/       # Data storage (not in git)
|-- docs/                         # Comprehensive documentation
|-- archived/                     # Legacy code preservation
```

## Prerequisites

- **Python 3.x** with packages: pandas, matplotlib, numpy, scipy, dataretrieval, streamlit
- **External Tools**: gfzrnx, gnssrefl package (rinex2snr, gnssir, quickLook)  
- **Optional**: contextily, geopandas for enhanced mapping

See [INSTALLATION.md](./INSTALLATION.md) for detailed setup instructions.

## Configuration

Each station requires configuration in `config/stations_config.json` and `config/{station}.json` with coordinates, antenna height, and GNSS-IR parameters. Update `config/tool_paths.json` with paths to external tools.

**Available Stations**: FORA (North Carolina), GLBX (Alaska), UMNQ (Greenland)

See configuration examples and detailed setup in [CLAUDE.md](./CLAUDE.md).

## Quick Start

```bash
# 1. Process full year of GNSS data
python scripts/run_gnssir_processing.py --station FORA --year 2024 --doy_start 1 --doy_end 365 --num_cores 8

# 2. Multi-source comparison and validation
python scripts/multi_source_comparison.py --station FORA --year 2024 --doy_start 260 --doy_end 266
python scripts/enhanced_usgs_comparison.py --station FORA --year 2024 --max_lag_days 5

# 3. Launch interactive dashboard
streamlit run enhanced_dashboard_v3.py
# Access at http://localhost:8501 for three-tab interface with 6 visualization types

# 4. Data quality assessment and gauge discovery
python scripts/data_quality_reporter.py --station FORA --year 2024
python scripts/find_optimal_gauge.py --station_id FORA --radius 25 --plot_results

# 5. Station configuration utilities
python scripts/auto_configure_external_stations.py --station GLBX --update-config  # Auto-discover CO-OPS/NDBC
python experiments/add_station.py  # Add new station to configuration
```

See [docs/DASHBOARD_PROJECT_STATUS.md](./docs/DASHBOARD_PROJECT_STATUS.md) for detailed usage and [CLAUDE.md](./CLAUDE.md) for development patterns.

## Recent Improvements (January 2025)

### Dashboard Enhancements
- **Modular Architecture**: Complete refactoring into separate tab modules for better maintainability
- **Fixed RMSE Calculations**: Corrected to use Water Surface Elevation (WSE) instead of raw Reflector Height (RH)
- **Multi-Parameter Timeline**: Moved from monthly to yearly tab with enhanced functionality
- **CO-OPS Integration**: Added support for NOAA CO-OPS tide predictions and observations
- **GLBX Station Support**: Full integration with CO-OPS station 9452634 (Elfin Cove, AK)

### Data Processing Fixes
- **Correlation Analysis**: Fixed to correlate WSE vs water level instead of RH vs water level
- **Multi-Source Data Loading**: Enhanced data loader supports CO-OPS data files
- **Antenna Height Conversions**: Proper WSE = Antenna Height - RH calculations throughout
- **Data Caching**: Improved performance with intelligent caching strategies

### Visualization Improvements
- **Multiple Plot Modes**: Dual-axis, detrended, normalized, and original visualization options
- **Environmental Integration**: NDBC buoy data for wind/wave analysis
- **Publication Quality**: Enhanced plot styling and export capabilities
- **Performance Optimization**: Reduced memory usage and faster rendering

### Project Organization
- **Archived Legacy Code**: Moved old dashboard versions to `archived/` directory
- **Documentation Updates**: Comprehensive documentation in `docs/` directory
- **Code Cleanup**: Removed deprecated functions and improved code organization


## Technical Details

**Architecture**: Modular design with parallel processing, smart caching, and comprehensive error handling
**Data Integration**: Progressive USGS gauge discovery, CO-OPS/NDBC APIs, time lag analysis  
**Environment**: Configures gnssrefl workspace with `REFL_CODE` and `ORBITS` variables

For comprehensive technical documentation:
- [CLAUDE.md](./CLAUDE.md) - Development patterns and architecture
- [docs/USGS_INTEGRATION.md](./docs/USGS_INTEGRATION.md) - Data integration details
- [docs/visualization_progress.md](./docs/visualization_progress.md) - Visualization implementation

## WSL Compatibility

For Windows Subsystem for Linux (WSL) users, see [WSL_USAGE.md](./WSL_USAGE.md) for specific instructions.

## Contributing

When contributing to this project:
1. Follow the patterns described in [CLAUDE.md](./CLAUDE.md)
2. Add appropriate tests for new features
3. Update documentation as needed
4. Use the existing configuration and logging patterns

## Acknowledgments

- gnssrefl package: [https://github.com/kristinemlarson/gnssrefl](https://github.com/kristinemlarson/gnssrefl)
- GFZ for the gfzrnx tool: [https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577895](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577895)
- USGS Water Data: [https://waterdata.usgs.gov/](https://waterdata.usgs.gov/)
- USGS dataretrieval package: [https://github.com/DOI-USGS/dataretrieval-python](https://github.com/DOI-USGS/dataretrieval-python)
- OpenStreetMap contributors: [https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)