# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (workflow rework, November 2026)
- gnssrefl 4.2.3 and Python 3.10 or newer. RINEX 3 is read directly; the RINEX 3 to RINEX 2 conversion and `gfzrnx` are removed.
- Station settings: short `config/stations.yaml` plus generated `config/references/<STATION>.json`.
- Days are translated in parallel, then `gnssir` runs one day at a time, in order (complete midnight arcs, no SNR file collisions).
- Removed from this branch: Streamlit and Dash dashboards, ice and camera work (kept on the `gnssir-surface` branch).

### Added (November 2026)
- `scripts/station_check.py`: sampling check, L2 signal check, scatter about a smooth curve.
- `scripts/inverse_heights.py`: smooth water level from the gnssrefl inverse method, with its own elevation range.
- `scripts/visualizer/periodogram_circle.py`: circular plot with site image, in two views (height above the surface, distance from the water level).
- `scripts/station_page.py`: static results page for a list of stations, with optional wind block.
- `scripts/earthscope_rinex3.py`: EarthScope RINEX 3 download and translation.
- Station SC02 (Friday Harbor) as a development station.

### Fixed (November 2026)
- GLBX station height had the wrong sign.

### Added
- Initial public release preparation
- Package infrastructure (pyproject.toml, LICENSE)
- GitHub Actions CI/CD workflow
- Comprehensive test suite

## [0.1.0] - 2024-01-30

### Added
- Core GNSS-IR processing pipeline with parallel orchestration
- Multi-source reference data integration (USGS, NOAA CO-OPS, NDBC, ERDDAP)
- Automated RINEX 3 to 2.11 conversion via gfzrnx
- RINEX data retrieval from NPS GNSS archive
- Time lag analysis and cross-correlation
- Interactive Streamlit dashboard with multi-tab interface
- Polar animation generation with satellite imagery overlay
- Basemap caching for efficient GIF generation
- Configurable time binning for animations
- Publication-quality visualization themes
- Segmented (monthly/seasonal) analysis capabilities
- Six pre-configured GNSS stations (FORA, GLBX, VALR, MDAI, UMNQ, DESO)

### Documentation
- Comprehensive README with usage examples
- Detailed METHODOLOGY.md with scientific background
- Developer notes in CLAUDE.md
