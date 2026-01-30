# GNSS-IR Processing Workflow Methodology

## Overview

This document describes the complete methodology for GNSS Interferometric Reflectometry (GNSS-IR) processing, including data acquisition, processing steps, validation against external water level measurements, and visualization. The workflow transforms raw RINEX observation data into validated water level estimates through a series of automated processing stages.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Software Dependencies](#2-software-dependencies)
3. [Directory Structure](#3-directory-structure)
4. [Configuration Files](#4-configuration-files)
5. [Processing Pipeline](#5-processing-pipeline)
6. [Output Files](#6-output-files)
7. [External Data Integration](#7-external-data-integration)
8. [Validation and Analysis](#8-validation-and-analysis)
9. [Interactive Dashboard](#9-interactive-dashboard)
10. [Steps to Replicate](#10-steps-to-replicate)

---

## 1. Scientific Background

### 1.1 GNSS Interferometric Reflectometry (GNSS-IR)

GNSS-IR uses signal-to-noise ratio (SNR) data from geodetic-quality GNSS receivers to measure environmental parameters. When GNSS signals reflect off nearby surfaces (e.g., water), they interfere with direct signals, creating oscillations in the SNR data. The frequency of these oscillations is related to the height of the antenna above the reflecting surface (the **Reflector Height**, RH).

### 1.2 Key Equations

**Reflector Height from SNR Frequency:**
```
RH = λ / (2 * sin(e))
```
Where:
- RH = Reflector Height (meters)
- λ = Carrier wavelength
- e = Satellite elevation angle

**Water Surface Elevation (WSE):**
```
WSE_ellipsoidal = Antenna_Ellipsoidal_Height - RH
```

**Demeaned Comparison:**
```
WSE_demeaned = WSE - mean(WSE)
```

---

## 2. Software Dependencies

### 2.1 External Command-Line Tools

| Tool | Purpose | Source |
|------|---------|--------|
| `gfzrnx` | RINEX 3 → RINEX 2.11 format conversion | GFZ Potsdam |
| `rinex2snr` | Extract SNR data from RINEX observations | gnssrefl package |
| `gnssir` | Calculate reflector heights from SNR data | gnssrefl package |
| `quickLook` | Generate quality assurance diagnostic plots | gnssrefl package |

Tool paths are configured in `config/tool_paths.json`.

### 2.2 Python Dependencies

**Core Processing:**
- `gnssrefl>=3.12.0` - GNSS-IR processing library
- `pandas>=2.0.3` - Data manipulation
- `numpy>=1.26.4` - Numerical computing
- `scipy>=1.15.2` - Scientific computing

**Data Acquisition:**
- `boto3>=1.28.65` - AWS S3 access for RINEX data
- `requests>=2.31.0` - HTTP requests for external APIs
- `dataretrieval>=1.0.12` - USGS water data API

**Visualization:**
- `matplotlib>=3.7.2` - Plotting
- `streamlit>=1.30.0` - Interactive dashboard

**Geospatial:**
- `geopy>=2.4.1` - Distance calculations
- `contextily>=1.6.2` - Basemap tiles

Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2.3 Environment Variables

The gnssrefl tools require two environment variables:
```bash
export REFL_CODE=/path/to/gnssrefl_data_workspace/refl_code
export ORBITS=/path/to/gnssrefl_data_workspace/orbits
```

These are automatically set by the processing scripts.

---

## 3. Directory Structure

```
GNSSIRWorkflow/
├── config/                          # Configuration files
│   ├── stations_config.json         # Station definitions and metadata
│   ├── tool_paths.json              # Paths to external executables
│   ├── fora_params.json             # Station-specific GNSS-IR parameters
│   └── {station}_params.json        # Additional station parameters
│
├── scripts/                         # Processing scripts
│   ├── run_gnssir_processing.py     # Main processing entry point
│   ├── usgs_comparison.py  # USGS validation analysis
│   ├── core_processing/             # Core processing modules
│   │   ├── config_loader.py         # Configuration management
│   │   ├── daily_gnssir_worker.py   # Single-day processing
│   │   ├── parallel_orchestrator.py # Multi-core coordination
│   │   └── workspace_setup.py       # gnssrefl environment setup
│   ├── external_tools/              # External tool wrappers
│   │   ├── preprocessor.py          # RINEX conversion (gfzrnx)
│   │   └── gnssrefl_executor.py     # gnssrefl tool execution
│   ├── external_apis/               # External data integration
│   │   ├── noaa_coops.py            # NOAA CO-OPS tide data
│   │   └── ndbc_client.py           # NDBC buoy data
│   ├── utils/                       # Utility modules
│   │   ├── data_manager.py          # S3 download, file management
│   │   ├── logging_config.py        # Centralized logging
│   │   └── segmented_analysis.py    # Monthly/seasonal analysis
│   ├── visualizer/                  # Visualization modules
│   │   ├── comparison.py            # Comparison plots
│   │   ├── timeseries.py            # Time series plots
│   │   └── advanced_viz.py          # Enhanced visualizations
│   └── results_handler.py           # Combine daily results
│
├── dashboard_components/            # Streamlit dashboard modules
│   ├── __init__.py
│   ├── data_loader.py               # Data loading with caching
│   ├── station_metadata.py          # Station info utilities
│   └── tabs/                        # Dashboard tabs
│       ├── overview_tab.py
│       ├── monthly_data_tab.py
│       ├── subdaily_tab.py
│       ├── yearly_residual_tab.py
│       └── diagnostics_tab.py
│
├── data/                            # Processing data (per station/year)
│   └── {STATION}/
│       └── {YEAR}/
│           ├── rinex3/              # Downloaded RINEX 3 files
│           ├── rh_daily/            # Daily reflector height results
│           ├── logs_daily/          # Per-day processing logs
│           └── quicklook_plots_daily/ # QA diagnostic plots
│
├── gnssrefl_data_workspace/         # gnssrefl working directory
│   ├── refl_code/                   # REFL_CODE environment
│   │   ├── {YEAR}/
│   │   │   ├── rinex/               # RINEX 2.11 files
│   │   │   ├── snr/                 # SNR files
│   │   │   └── results/             # gnssir output
│   │   ├── input/                   # Station JSON configs
│   │   └── Files/                   # quickLook outputs
│   └── orbits/                      # Satellite orbit files
│
├── results_annual/                  # Combined annual results
│   └── {STATION}/
│       ├── {STATION}_{YEAR}_combined_rh.csv      # Daily aggregated RH
│       ├── {STATION}_{YEAR}_combined_raw.csv     # All subdaily retrievals
│       ├── {STATION}_{YEAR}_enhanced_comparison.csv
│       ├── {STATION}_{YEAR}_wse_usgs_comparison.png
│       ├── {STATION}_{YEAR}_demeaned_comparison.png
│       └── {STATION}_{YEAR}_*.png   # Various analysis plots
│
├── logs/                            # Main processing logs
├── tests/                           # Test suite
├── tools/                           # External binaries (gfzrnx)
│
├── dashboard.py         # Main dashboard application
├── requirements.txt                 # Python dependencies
└── CLAUDE.md                        # Project documentation
```

---

## 4. Configuration Files

### 4.1 Station Configuration (`config/stations_config.json`)

Defines station metadata, coordinates, and data source paths:

```json
{
  "FORA": {
    "station_id_4char_lower": "fora",
    "ellipsoidal_height_m": -30.917,
    "latitude_deg": 35.9393968812,
    "longitude_deg": -75.7081602909,
    "gnssir_json_params_path": "config/fora_params.json",
    "s3_bucket_name": "doi-gnss",
    "s3_rinex_obs_path_template": "Rinex/{YEAR}/{DOY_PADDED}/FORA/FORA{DOY_PADDED}0.{YY}o",
    "usgs_comparison": {
      "target_usgs_site": "02043433",
      "usgs_parameter_code_to_use": "00065",
      "usgs_gauge_stated_datum": "NAVD88"
    },
    "external_data_sources": {
      "noaa_coops": {
        "enabled": true,
        "preferred_stations": ["8651370", "8652587"],
        "datum": "NAVD88"
      },
      "ndbc_buoys": {
        "enabled": true,
        "preferred_buoys": ["44025", "44014", "41025"]
      }
    }
  }
}
```

### 4.2 Tool Paths (`config/tool_paths.json`)

Paths to external command-line tools:

```json
{
  "gfzrnx_path": "/path/to/tools/gfzrnx",
  "rinex2snr_path": "/path/to/conda/env/bin/rinex2snr",
  "gnssir_path": "/path/to/conda/env/bin/gnssir",
  "quicklook_path": "/path/to/conda/env/bin/quickLook"
}
```

### 4.3 GNSS-IR Parameters (`config/{station}_params.json`)

Station-specific processing parameters for gnssir:

```json
{
  "station": "fora",
  "lat": 35.9393968812,
  "lon": -75.7081602909,
  "ht": -30.917,
  "minH": 5.0,
  "maxH": 15.0,
  "e1": 4.0,
  "e2": 13.0,
  "azval2": [0.0, 80.0, 340.0, 360.0],
  "freqs": [1, 20, 5, 101, 102, 201, 205, 206, 207, 208, 302, 306],
  "reqAmp": [6.4],
  "PkNoise": 2.7,
  "refraction": true
}
```

Key parameters:
- `minH`, `maxH`: Expected reflector height range (meters)
- `e1`, `e2`: Elevation angle range for analysis (degrees)
- `azval2`: Azimuth mask (exclude directions with obstructions)
- `freqs`: GNSS frequencies to use (GPS L1=1, L2C=20, L5=5, etc.)
- `reqAmp`: Minimum amplitude threshold
- `PkNoise`: Minimum peak-to-noise ratio

---

## 5. Processing Pipeline

### 5.1 Pipeline Overview

```
┌─────────────────────┐
│  Station Config     │
│  (coordinates, S3   │
│   paths, parameters)│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  For Each Day       │◄──── Parallel Processing
│  in Date Range      │      (multiprocessing.Pool)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Download RINEX 3   │  Source: AWS S3 doi-gnss bucket
│  from AWS S3        │  Tool: boto3/HTTP
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Convert RINEX 3    │  Tool: gfzrnx
│  → RINEX 2.11       │  Output: {station}{doy}0.{yy}o
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Extract SNR Data   │  Tool: rinex2snr
│  by Satellite       │  Output: {station}{doy}0.{yy}.snr66
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Calculate          │  Tool: gnssir
│  Reflector Heights  │  Output: {doy}.txt (multiple RH per day)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Generate QA Plots  │  Tool: quickLook (background thread)
│  (LSP, summary)     │  Output: PNG files
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Combine Daily      │  Aggregate: mean, median, std, min, max, count
│  Results            │  Output: {STATION}_{YEAR}_combined_rh.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transform to WSE   │  WSE = Antenna_Height - RH
│  (Water Surface     │  Then demean for comparison
│   Elevation)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Retrieve External  │  Sources: USGS, NOAA CO-OPS, NDBC
│  Validation Data    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Time Series        │  Cross-correlation analysis
│  Alignment          │  Optimal lag detection
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Statistical        │  Correlation, RMSE, Bias
│  Comparison         │  Monthly/Seasonal segmentation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Generate Plots &   │  Time series, scatter, residuals
│  Dashboard          │  Interactive Streamlit app
└─────────────────────┘
```

### 5.2 Step-by-Step Details

#### Step 1: Load Station Configuration
```python
from scripts.core_processing.config_loader import load_station_config
station_config = load_station_config(stations_config_path, 'FORA', project_root)
```

#### Step 2: Download RINEX 3 from AWS S3
- Source: `s3://doi-gnss/Rinex/{YEAR}/{DOY}/...`
- Uses boto3 for authenticated access or HTTP for public buckets
- Validates file size (minimum 500 KB)

#### Step 3: Convert RINEX 3 → RINEX 2.11
```bash
gfzrnx -finp FORA2600.24o -fout fora2600.24o -vo 2 -f
```
- Converts multi-constellation RINEX 3 to legacy RINEX 2.11 format
- Required because gnssrefl tools expect RINEX 2.11

#### Step 4: Extract SNR Data
```bash
rinex2snr fora 2024 260 -snr 66 -nolook T
```
- Extracts Signal-to-Noise Ratio data for all satellites
- SNR code 66 = both low and high rate observations
- Output: `fora2600.24.snr66`

#### Step 5: Calculate Reflector Heights
```bash
gnssir fora 2024 260
```
- Performs Lomb-Scargle periodogram analysis on SNR data
- Outputs multiple reflector height retrievals per day
- Each retrieval tagged with timestamp, satellite, azimuth, amplitude, etc.

#### Step 6: Generate QA Plots
```bash
quickLook fora 2024 260
```
- Generates diagnostic plots showing:
  - LSP (Lomb-Scargle Periodogram)
  - RH vs azimuth
  - Quality metrics

#### Step 7: Combine Daily Results
- Reads individual `{doy}.txt` files
- Aggregates to daily statistics:
  - `rh_count`: Number of valid retrievals
  - `rh_mean_m`, `rh_median_m`: Central tendency
  - `rh_std_m`: Variability
  - `rh_min_m`, `rh_max_m`: Range

#### Step 8: Transform to Water Surface Elevation
```python
WSE_ellipsoidal = Antenna_Ellipsoidal_Height - RH_median
WSE_demeaned = WSE - WSE.mean()
```

---

## 6. Output Files

### 6.1 Daily Processing Outputs

| File | Location | Description |
|------|----------|-------------|
| `{STATION}{DOY}0.{YY}o` | `data/{STATION}/{YEAR}/rinex3/` | Downloaded RINEX 3 file |
| `{station}{doy}0.{yy}o` | `gnssrefl_data_workspace/refl_code/{YEAR}/rinex/` | Converted RINEX 2.11 |
| `{station}{doy}0.{yy}.snr66` | `gnssrefl_data_workspace/refl_code/{YEAR}/snr/` | SNR data file |
| `{doy}.txt` | `gnssrefl_data_workspace/refl_code/{YEAR}/results/` | gnssir reflector heights |
| `{station}_{year}_{doy}.txt` | `data/{STATION}/{YEAR}/rh_daily/` | Copied RH results |
| `{station}_{year}_{doy}.log` | `data/{STATION}/{YEAR}/logs_daily/` | Processing log |
| `*.png` | `data/{STATION}/{YEAR}/quicklook_plots_daily/` | QA diagnostic plots |

### 6.2 Annual Combined Outputs

| File | Description |
|------|-------------|
| `{STATION}_{YEAR}_combined_rh.csv` | Daily aggregated reflector heights |
| `{STATION}_{YEAR}_combined_raw.csv` | All subdaily retrievals |
| `{STATION}_{YEAR}_enhanced_comparison.csv` | Merged GNSS-IR + USGS data |
| `{STATION}_{YEAR}_wse_usgs_comparison.png` | WSE vs USGS time series |
| `{STATION}_{YEAR}_demeaned_comparison.png` | Demeaned comparison |
| `{STATION}_{YEAR}_monthly_correlation_demeaned.png` | Monthly correlation analysis |
| `{STATION}_{YEAR}_seasonal_correlation_demeaned.png` | Seasonal correlation |
| `{STATION}_{YEAR}_subdaily_overview.png` | Subdaily comparison plot |

### 6.3 Combined RH CSV Format

```csv
date,rh_count,rh_mean_m,rh_median_m,rh_std_m,rh_min_m,rh_max_m,datetime,year,doy
2024-01-12,50,8.024,8.012,0.080,7.83,8.18,2024-01-12,2024,12
2024-01-13,15,8.087,8.090,0.082,7.92,8.23,2024-01-13,2024,13
```

---

## 7. External Data Integration

### 7.1 USGS Water Data

**API:** USGS Water Services (`dataretrieval` package)

**Parameter Codes (priority order):**
- `62610`: Water level, NAVD88 (preferred)
- `62611`: Water level, NGVD29
- `62620`: Water level, MSL
- `00065`: Gage height (stage)

```python
from scripts.usgs_data_handler import fetch_usgs_gauge_data
usgs_df, gauge_info, param_code = fetch_usgs_gauge_data(
    site_code="02043433",
    parameter_code="00065",
    start_date_str="2024-01-01",
    end_date_str="2024-12-31",
    service="iv"  # instantaneous values
)
```

### 7.2 NOAA CO-OPS Tide Data

**API:** `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`

```python
from scripts.external_apis.noaa_coops import NOAACOOPSClient
client = NOAACOOPSClient()

# Tide predictions
predictions = client.get_tide_predictions(
    station_id="8651370",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    datum="NAVD88"
)

# Water level observations
observations = client.get_water_level_observations(
    station_id="8651370",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

### 7.3 NDBC Buoy Data

**API:** `https://www.ndbc.noaa.gov/data/realtime2/`

```python
from scripts.external_apis.ndbc_client import NDBCClient
client = NDBCClient()

# Meteorological data (wind, waves, pressure)
met_data = client.get_meteorological_data(station_id="44025", days_back=45)

# Wave spectral data
wave_data = client.get_wave_data(station_id="44025", days_back=45)
```

---

## 8. Validation and Analysis

### 8.1 Time Lag Analysis

Cross-correlation analysis to find optimal temporal alignment:

```python
from scripts.time_lag_analyzer import calculate_time_lag_correlation

lag_days, correlation, confidence, all_lags = calculate_time_lag_correlation(
    gnssir_series=merged_df['wse_ellips_m'],
    usgs_series=merged_df['usgs_value_m'],
    max_lag_days=10
)
```

### 8.2 Statistical Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Correlation (r) | Pearson correlation | Linear relationship strength |
| RMSE | √(mean((GNSS - USGS)²)) | Root Mean Square Error |
| Bias | mean(GNSS - USGS) | Systematic offset |
| MAE | mean(\|GNSS - USGS\|) | Mean Absolute Error |

### 8.3 Segmented Analysis

Monthly and seasonal correlation analysis:

```python
from scripts.utils.segmented_analysis import (
    generate_monthly_segments,
    generate_seasonal_segments,
    perform_segmented_correlation
)

monthly_segments = generate_monthly_segments(2024)
monthly_correlations, monthly_data = perform_segmented_correlation(
    merged_df, monthly_segments,
    gnss_col='wse_ellips_m_demeaned',
    usgs_col='usgs_value_m_demeaned'
)
```

---

## 9. Interactive Dashboard

### 9.1 Launch Dashboard

```bash
streamlit run dashboard.py
```

### 9.2 Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | Summary statistics, data availability, correlation metrics |
| **Monthly Data** | Month-by-month comparison and correlation |
| **Subdaily** | Individual retrieval comparison against reference |
| **Yearly Residuals** | Annual residual analysis and patterns |
| **Diagnostics** | Quality metrics, outlier detection |

### 9.3 Features

- Station selection from configured stations
- Year and DOY range selection
- Multi-source data integration (USGS, CO-OPS, NDBC)
- Real-time correlation calculations
- Interactive plots with zoom/pan
- Data export capabilities

---

## 10. Steps to Replicate

### 10.1 Environment Setup

```bash
# 1. Clone repository
git clone <repository_url>
cd GNSSIRWorkflow

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install gnssrefl package
pip install gnssrefl

# 5. Download and install gfzrnx
# From: https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577894
# Place in tools/ directory and make executable:
chmod +x tools/gfzrnx

# 6. Update tool_paths.json with correct paths
```

### 10.2 Configure a Station

1. Add station to `config/stations_config.json`
2. Create station parameters file `config/{station}_params.json`
3. Verify S3 path template for RINEX data

### 10.3 Run Processing Pipeline

```bash
# Full-year processing with 8 cores
python scripts/run_gnssir_processing.py \
    --station FORA \
    --year 2024 \
    --doy_start 1 \
    --doy_end 365 \
    --num_cores 8

# Skip already-processed stages
python scripts/run_gnssir_processing.py \
    --station FORA \
    --year 2024 \
    --doy_start 1 \
    --doy_end 365 \
    --skip_all
```

### 10.4 Run Validation Analysis

```bash
# Enhanced USGS comparison with time lag analysis
python scripts/usgs_comparison.py \
    --station FORA \
    --year 2024 \
    --max_lag_days 5 \
    --log_level INFO
```

### 10.5 Launch Dashboard

```bash
streamlit run dashboard.py
```

### 10.6 Quick Test

```bash
# Process 7 days for testing
python scripts/run_gnssir_processing.py \
    --station FORA \
    --year 2024 \
    --doy_start 260 \
    --doy_end 266 \
    --num_cores 4
```

---

## Appendix A: Available Stations

| Station | Location | Coordinates | Reference Source |
|---------|----------|-------------|------------------|
| FORA | North Carolina, USA | 35.94°N, 75.71°W | USGS + CO-OPS |
| GLBX | Alaska, USA | 58.46°N, 135.89°W | CO-OPS |
| VALR | Hawaii, USA | 21.37°N, 157.94°W | CO-OPS + USGS |
| MDAI | Maryland, USA | 38.14°N, 75.19°W | USGS |
| UMNQ | Greenland | 70.68°N, 52.12°W | — |
| DESO | Florida, USA | 27.52°N, 82.64°W | — |

---

## Appendix B: USGS Parameter Codes

| Code | Description | Preferred |
|------|-------------|-----------|
| 62610 | Water level, NAVD88 | ✓ (Primary) |
| 62611 | Water level, NGVD29 | Secondary |
| 62620 | Water level, MSL | Tertiary |
| 00065 | Gage height (stage) | Fallback |

---

## Appendix C: GNSS Frequency Codes

| Code | System | Signal |
|------|--------|--------|
| 1 | GPS | L1 C/A |
| 5 | GPS | L5 |
| 20 | GPS | L2C |
| 101 | GLONASS | L1 |
| 102 | GLONASS | L2 |
| 201-208 | Galileo | E1, E5a, E5b, etc. |
| 302, 306 | BeiDou | B1, B3 |

---

## References

1. Larson, K.M., et al. (2013). "GPS multipath and its relation to near-surface soil moisture content." IEEE J-STARS.
2. Larson, K.M. (2019). "Unanticipated Uses of the Global Positioning System." Annual Review of Earth and Planetary Sciences.
3. gnssrefl documentation: https://github.com/kristinemlarson/gnssrefl

---

*Document generated: January 2025*
*Version: 1.0*
