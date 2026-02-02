# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- AWS S3 data retrieval with HTTP fallback
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
