# Architecture Overview

This document maps the module dependencies and categorizes scripts by their role in the processing pipeline.

## Main Entry Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│  process_station.py        ← Unified workflow (recommended entry point) │
│       ↓ subprocess.run()                                                │
│  run_gnssir_processing.py  ← Core GNSS-IR processing                    │
│  dashboard.py              ← Streamlit analysis interface               │
└─────────────────────────────────────────────────────────────────────────┘
```

| Entry Point | Purpose | Usage |
|-------------|---------|-------|
| `process_station.py` | Unified workflow coordinator | `python scripts/process_station.py --station FORA --year 2024 --doy_start 60 --doy_end 120` |
| `run_gnssir_processing.py` | Core GNSS-IR processing only | Called by process_station.py or standalone |
| `dashboard.py` | Interactive analysis UI | `streamlit run dashboard.py` |

## Workflow Visualization

```
process_station.py (UNIFIED ENTRY POINT)
    │
    ├─[Phase 1: Core Processing]
    │   └─► run_gnssir_processing.py
    │           └─► parallel_orchestrator.py
    │                   └─► daily_gnssir_worker.py (per DOY)
    │                           ├─► data_manager.py (S3/HTTP download)
    │                           ├─► preprocessor.py (RINEX3→2.11)
    │                           └─► gnssrefl_executor.py (rinex2snr, gnssir)
    │
    ├─[Phase 2: Reference Matching]
    │   ├─► usgs_comparison.py (for USGS-based stations)
    │   │       └─► usgs_data_handler, usgs_comparison_analyzer, etc.
    │   ├─► coops_comparison.py (for coastal CO-OPS tide gauge stations)
    │   │       └─► noaa_coops.py (CO-OPS API client with auto-discovery)
    │   └─► generate_erddap_matched.py (for ERDDAP-based stations like GLBX)
    │
    └─[Phase 3: Visualization]
        ├─► plot_resolution_comparison.py
        └─► create_polar_animation.py
```

## Module Classification

### Library Modules (imported, no standalone use)

These are pure library code - they define functions/classes but have no `if __name__ == "__main__"` block.

#### Core Processing (`scripts/core_processing/`)
| Module | Purpose | Used By |
|--------|---------|---------|
| `config_loader.py` | Load station/tool configs from JSON | run_gnssir_processing |
| `parallel_orchestrator.py` | Multi-core daily processing | run_gnssir_processing |
| `daily_gnssir_worker.py` | Single-day processing pipeline | parallel_orchestrator |
| `workspace_setup.py` | Create gnssrefl directory structure | daily_gnssir_worker |

#### External Tools (`scripts/external_tools/`)
| Module | Purpose | Used By |
|--------|---------|---------|
| `preprocessor.py` | RINEX 3→2.11 conversion via gfzrnx | daily_gnssir_worker |
| `gnssrefl_executor.py` | Run rinex2snr, gnssir, quickLook | daily_gnssir_worker |

#### Utilities (`scripts/utils/`)
| Module | Purpose | Used By |
|--------|---------|---------|
| `data_manager.py` | S3/HTTP file downloads | daily_gnssir_worker |
| `logging_config.py` | Logger setup | multiple modules |
| `segmented_analysis.py` | Monthly/seasonal filtering | visualizer modules |

#### External APIs (`scripts/external_apis/`)
| Module | Purpose | Used By |
|--------|---------|---------|
| `noaa_coops.py` | NOAA CO-OPS tide data client | dashboard, multi_source_comparison |
| `ndbc_client.py` | NDBC buoy data client | dashboard, multi_source_comparison |

#### Visualization (`scripts/visualizer/`)
| Module | Purpose |
|--------|---------|
| `base.py` | Shared utilities, color scheme (PLOT_COLORS) |
| `comparison.py` | GNSS-IR vs reference time series |
| `comparison_plots.py` | Diagnostic comparison plots |
| `timeseries.py` | Annual RH time series |
| `lag_analyzer.py` | Time lag correlation plots |
| `tide_integration.py` | Tide prediction comparison |
| `segmented_viz.py` | Monthly/seasonal grids |
| `dashboard_plots.py` | Streamlit-specific plots |
| `publication_theme.py` | Dark theme styling |

#### USGS Integration (root `scripts/`)
| Module | Purpose | Used By |
|--------|---------|---------|
| `usgs_gauge_finder.py` | Load configured gauge info | usgs_comparison |
| `usgs_data_handler.py` | Fetch USGS water level data | usgs_comparison |
| `usgs_comparison_analyzer.py` | Compute correlation statistics | usgs_comparison |
| `usgs_progressive_search.py` | Expanding radius gauge search | usgs_comparison |
| `time_lag_analyzer.py` | Cross-correlation lag detection | usgs_comparison |
| `reflector_height_utils.py` | RH → WSE conversion | usgs_comparison |
| `results_handler.py` | Combine daily RH files | parallel_orchestrator |

#### Dashboard Components (`dashboard_components/`)
| Module | Purpose |
|--------|---------|
| `data_loader.py` | Load CSV data with caching |
| `station_metadata.py` | Station config utilities |
| `cache_manager.py` | File/session caching |
| `constants.py` | Colors, theme settings |
| `tabs/*.py` | Individual tab implementations |

### Standalone Scripts (called via subprocess)

These scripts are invoked by `process_station.py` via `subprocess.run()` to maintain process isolation.

| Script | Purpose | Called When |
|--------|---------|-------------|
| `usgs_comparison.py` | Match GNSS-IR to USGS gauge data | Station has USGS reference |
| `coops_comparison.py` | Match GNSS-IR to CO-OPS tide gauge data | Station has CO-OPS reference (coastal) |
| `generate_erddap_matched.py` | Match GNSS-IR to ERDDAP data | Station has ERDDAP reference (e.g., GLBX) |
| `plot_resolution_comparison.py` | Correlation vs temporal resolution | Visualization phase |
| `create_polar_animation.py` | Animated satellite overlay GIFs | Visualization phase |

### Utility Scripts (independent, edge-case use)

These are standalone tools for exploration and debugging, not part of the main pipeline.

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `find_reference_stations.py` | Discover nearby USGS/CO-OPS/NDBC stations | Setting up a new GNSS station |
| `search_erddap_stations.py` | Search NOAA ERDDAP for water level data | Finding alternative reference data |
| `compare_reference_distances.py` | Compare distances to reference sources | Evaluating reference options |
| `multi_source_comparison.py` | Multi-source validation analysis | Advanced validation (dual-mode: library + CLI) |

## Package Structure

```
GNSSIRWorkflow-standalone/
├── scripts/
│   ├── core_processing/      # Pipeline orchestration
│   ├── external_tools/       # gfzrnx, gnssrefl wrappers
│   ├── external_apis/        # NOAA CO-OPS, NDBC clients
│   ├── utils/                # Data management, logging
│   ├── visualizer/           # All plotting modules
│   ├── run_gnssir_processing.py   # [ENTRY] Core processing
│   ├── process_station.py         # [ENTRY] Unified workflow
│   ├── usgs_comparison.py         # [SUBPROCESS] USGS matching
│   ├── generate_erddap_matched.py # [SUBPROCESS] ERDDAP matching
│   ├── create_polar_animation.py  # [SUBPROCESS] Animation
│   ├── plot_resolution_comparison.py # [SUBPROCESS] Resolution plots
│   ├── find_reference_stations.py # [UTILITY] Station discovery
│   └── ...
├── dashboard_components/     # Streamlit UI modules
├── dashboard.py              # [ENTRY] Streamlit app
├── config/                   # Station and tool configs
├── tests/                    # Test suite
└── results_annual/           # Output directory
```

## Design Patterns

1. **Subprocess Isolation**: `process_station.py` uses `subprocess.run()` rather than imports to avoid shared state issues in the parallel processing pipeline.

2. **Reference Source Abstraction**: The pipeline auto-detects whether to use USGS or ERDDAP based on station config, calling the appropriate comparison script.

3. **Lazy Dashboard Imports**: Dashboard components use try/except for optional dependencies (plotly, advanced tabs) to allow graceful degradation.

4. **Shared Color Scheme**: All visualization modules import `PLOT_COLORS` from `visualizer/base.py` for consistency.
