# GNSS-IR Processing Workflow

Python-based workflow for GNSS Interferometric Reflectometry (GNSS-IR) data processing. Downloads RINEX 3 data from AWS S3, processes it to calculate reflector heights for water level estimation, and validates results against external reference sources.

## Processing Flow

1. **Download**: RINEX 3 files from AWS S3 `doi-gnss` bucket
2. **Convert**: RINEX 3 → RINEX 2.11 using `gfzrnx`
3. **SNR Extraction**: Generate signal-to-noise ratio data with `rinex2snr`
4. **GNSS-IR Processing**: Calculate reflector heights using `gnssir`
5. **Reference Matching**: Find nearby gauges and retrieve water level data
6. **Analysis**: Time lag analysis, correlation statistics, visualization

## Reference Data Sources

- **USGS**: Stream/tide gauges with water level data (NAVD88/NGVD29/MSL)
- **NOAA CO-OPS**: Tide predictions and water level observations
- **NOAA ERDDAP**: Regional servers (AOOS, PacIOOS, SECOORA) with water level sensors
- **NDBC**: Meteorological and wave measurements from buoys

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
│   └── visualizer/              # Plotting modules
├── dashboard.py                 # Streamlit dashboard
├── dashboard_components/        # Dashboard modules
├── tests/                       # Test suite with real data fixtures
└── data/                        # Data storage (not in git)
```

## Prerequisites

- **Python 3.9+** with packages: pandas, matplotlib, numpy, scipy, dataretrieval, streamlit
- **External Tools**: gfzrnx, gnssrefl package (rinex2snr, gnssir, quickLook)

### Setup

```bash
conda create -n py39 python=3.9
conda activate py39
pip install gnssrefl
pip install -r requirements.txt
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

**Configured Stations**: FORA (North Carolina), GLBX (Alaska), MDAI (Maryland), VALR (Hawaii)

## Usage

```bash
# Process GNSS data for a date range
python scripts/run_gnssir_processing.py --station GLBX --year 2024 --doy_start 1 --doy_end 31 --num_cores 8

# Run unified processing workflow
python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

# USGS comparison with time lag analysis
python scripts/usgs_comparison.py --station FORA --year 2024 --max_lag_days 5

# Launch dashboard
streamlit run dashboard.py

# Find reference stations for a new station
python scripts/find_reference_stations.py --station GLBX --radius 50
```

### Skip Flags

Avoid reprocessing completed stages:
- `--skip_download`: Skip RINEX 3 download if files exist
- `--skip_gnssir`: Skip GNSS-IR processing (use existing results)
- `--skip_comparison`: Skip reference comparison
- `--skip_viz`: Skip visualization

## Testing

```bash
pytest                  # Full test suite
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -v               # Verbose output
```

Tests use real sample data from GLBX (ERDDAP), MDAI (USGS), and VALR (CO-OPS) stations.

## Adding a New Station

1. Add station info to `config/stations_config.json` (coordinates, antenna height)
2. Run `find_reference_stations.py` to identify nearby reference sources
3. Add recommended reference configuration to the station's config
4. Run processing pipeline

## Environment Variables

The system configures gnssrefl-required environment variables:
- `REFL_CODE`: Points to `gnssrefl_data_workspace/refl_code/`
- `ORBITS`: Points to `gnssrefl_data_workspace/orbits/`

## Acknowledgments

- [gnssrefl](https://github.com/kristinemlarson/gnssrefl) - GNSS-IR processing package
- [gfzrnx](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577895) - RINEX format conversion
- [USGS Water Data](https://waterdata.usgs.gov/)
- [NOAA CO-OPS](https://tidesandcurrents.noaa.gov/)
- [NOAA NDBC](https://www.ndbc.noaa.gov/)
