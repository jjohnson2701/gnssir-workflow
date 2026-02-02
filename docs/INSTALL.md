# Installation Guide

This guide covers setting up the GNSS-IR Workflow package for processing and visualizing
GNSS Interferometric Reflectometry water level data.

## Prerequisites

- **Operating System**: Linux or macOS (Windows may work but is untested)
- **Python**: 3.9 (recommended and tested; later versions may work but are untested)
- **Conda**: Recommended for environment management (Miniconda or Anaconda)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jjohnson2701/gnssir-workflow.git
cd gnssir-workflow
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate gnssir-workflow
```

### 3. Install the Package

```bash
# Core functionality only
pip install -e .

# With Streamlit dashboard
pip install -e .[dashboard]

# With animation generation
pip install -e .[animation]

# Everything (dashboard, animation, development tools)
pip install -e .[all]
```

### 4. Set Environment Variables

The gnssrefl package requires these environment variables:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export REFL_CODE=$HOME/gnssrefl_data
export ORBITS=$REFL_CODE/orbits

# Create the directories
mkdir -p $REFL_CODE $ORBITS
```

### 5. Install gfzrnx (Required)

gfzrnx is a RINEX file manipulation tool from GFZ Potsdam, required for RINEX
format conversion. Current version: 2.2.0

#### Option A: Install via Conda (Recommended)

```bash
conda install -c eumetsat gfzrnx
```

#### Option B: Manual Download

1. Visit: https://gnss.gfz.de/services/gfzrnx
2. Register and download the appropriate binary for your system
3. The binary is named like `gfzrnx_2.2.0_lx64` (Linux) or `gfzrnx_2.2.0_mac` (macOS)

```bash
# Linux example
sudo cp gfzrnx_2.2.0_lx64 /usr/local/bin/gfzrnx
sudo chmod +x /usr/local/bin/gfzrnx

# Verify installation
gfzrnx -h
```

#### Documentation

- [User Guide](https://gnss.git-pages.gfz-potsdam.de/gfzrnx/)
- [PDF Manual](https://gnss.git-pages.gfz-potsdam.de/gfzrnx/pdf/GFZRNX_Users_Guide.pdf)

#### Alternative: Configure Custom Path

If gfzrnx is not in your PATH, update `config/tool_paths.json`:

```json
{
    "gfzrnx_path": "/path/to/your/gfzrnx"
}
```

## Configuration

### Station Configuration

Copy the template and configure your stations:

```bash
cp config/stations_config_template.json config/stations_config.json
```

Edit `stations_config.json` to add your GNSS stations. See the template for
documentation on all available fields.

### gnssrefl Station Setup

For each station, you need a gnssrefl JSON parameter file. Generate one using:

```bash
gnssir_input XXXX -lat 45.0 -lon -122.0 -height 10.0
```

This creates a file at `$REFL_CODE/input/XXXX.json` which you can customize
for your site's specific geometry and analysis parameters.

## Verifying Installation

Run the test suite to verify everything is working:

```bash
# Run unit tests only (fast)
pytest -m unit

# Run all tests including integration tests
pytest

# Check specific functionality
pytest tests/test_environment.py -v
```

## Usage

### Processing a Station

```bash
# Process a station for a date range (DOY = day of year)
python scripts/process_station.py --station GLBX --year 2024 --doy_start 1 --doy_end 31

# Run USGS comparison (for stations with USGS gauge configured)
python scripts/usgs_comparison.py --station FORA --year 2024

# Run CO-OPS comparison (for coastal stations)
python scripts/coops_comparison.py --station VALR --year 2024

# Run ERDDAP comparison (for stations with ERDDAP configured)
python scripts/generate_erddap_matched.py --station GLBX --year 2024
```

### Running the Dashboard

```bash
# Requires: pip install -e .[dashboard]
streamlit run dashboard.py
```

## Troubleshooting

### gnssrefl not found

Ensure the conda environment is activated:
```bash
conda activate gnssir-workflow
```

### gfzrnx not found

Verify gfzrnx is in your PATH:
```bash
which gfzrnx
gfzrnx -h
```

If not in PATH, check `config/tool_paths.json` is configured correctly.

### REFL_CODE not set

Add environment variables to your shell configuration:
```bash
echo 'export REFL_CODE=$HOME/gnssrefl_data' >> ~/.bashrc
echo 'export ORBITS=$REFL_CODE/orbits' >> ~/.bashrc
source ~/.bashrc
```

### Import errors

Ensure the package is installed:
```bash
pip install -e .
```

### Test failures

Some tests require network access or specific data files. Run unit tests only
for basic verification:
```bash
pytest -m unit
```

## Data Sources

The workflow supports multiple reference data sources for validation:

- **USGS**: US Geological Survey water level gauges (dataretrieval package)
- **NOAA CO-OPS**: Tide gauge data from NOAA Center for Operational Oceanographic Products
- **ERDDAP**: Data from ERDDAP servers (AOOS, etc.)
- **NDBC**: Meteorological buoy data (not for water levels)

Configure your preferred reference sources in `stations_config.json`.

## Support

- Issues: https://github.com/jjohnson2701/gnssir-workflow/issues
- gnssrefl documentation: https://github.com/kristinemlarson/gnssrefl
