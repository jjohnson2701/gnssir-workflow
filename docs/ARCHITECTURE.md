# Architecture

This document shows which command calls which module, where the settings are, and where the files go. For the method, see [METHODOLOGY.md](METHODOLOGY.md).

## Commands

| Command | Purpose |
|---|---|
| `python scripts/process_station.py` | The main workflow: data, heights, gauge comparison, figures |
| `python -m scripts.earthscope_rinex3` | Download EarthScope RINEX 3 files and make SNR files |
| `python -m scripts.station_check` | Checks of one station: sampling, L2 signal, scatter |
| `python -m scripts.inverse_heights` | Smooth water level series from the inverse method |
| `python -m scripts.visualizer.periodogram_circle` | Circular plot for one day |
| `python -m scripts.station_page` | Static HTML page for a list of stations |
| `python scripts/find_reference_stations.py` | Find gauges near a station and write the reference catalog |

The last five commands read gnssrefl result files only. They work on any gnssrefl workspace (`REFL_CODE`), also one that this workflow did not make.

## The main workflow

```
process_station.py
│
├─ Phase 1    heights of single arcs
│   └─ core_processing/parallel_orchestrator.py
│        pass 1  translate all days (several days at once)
│        │        └─ core_processing/daily_gnssir_worker.py
│        │             ├─ utils/data_manager.py        NPS archive download
│        │             ├─ earthscope_rinex3.py         EarthScope RINEX 3 download
│        │             └─ external_tools/gnssrefl_executor.py   rinex2snr
│        pass 2  gnssir, one day at a time, in day order
│        pass 3  quickLook plots
│        then    results_handler.py   combine the daily files into one table
│
├─ Phase 1.5  utils/subdaily_runner.py   gnssrefl subdaily (spline series)
│
├─ Phase 2    gauge comparison (one of three, from the `reference` setting)
│   ├─ usgs_comparison.py          USGS gauge
│   ├─ coops_comparison.py         NOAA CO-OPS gauge
│   └─ generate_erddap_matched.py  ERDDAP server
│
└─ Phase 3    figures
    ├─ plot_resolution_comparison.py
    └─ create_polar_animation.py
```

Phases 2 and 3 run as separate processes. Each phase has a switch to leave it out (`--skip_gnssir`, `--skip_subdaily`, `--skip_comparison`, `--skip_viz`).

### Why three passes

Two properties of gnssrefl set the order (see "Two rules for runs" in the README):

- `gnssir` unpacks and repacks the SNR file while it runs. Two jobs on one station at once can lose days. So only the translation runs in parallel.
- `gnssir` reads the SNR files of the neighbour days for arcs that cross midnight. So all days are translated before the first `gnssir` run.

### Data sources

The `data_source` setting of a station selects the route in `daily_gnssir_worker.py`:

| `data_source` | Route |
|---|---|
| `nps` (default) | Download the RINEX 3 file from the NPS archive and give it to `rinex2snr -input_file` |
| `earthscope` | Download the RINEX 3 file with `earthscope_rinex3.py`, unpack it, give it to `rinex2snr -input_file`. If the archive has no RINEX 3 file, fall back to `rinex2snr -archive unavco` |
| any other name | `rinex2snr -archive <name>` |

There is no conversion to RINEX 2.

## Settings

| File | Content |
|---|---|
| `config/stations.yaml` | One short block per station: position, data source, reference gauge, optional settings |
| `config/<station>.json` | The gnssrefl station file (elevation and height range, azimuth mask, signals, limits). Copied to `$REFL_CODE/input/` |
| `config/references/<STATION>.json` | Gauges near the station. Made by `find_reference_stations.py`; not written by hand |
| `config/tool_paths.json` | Paths of external programs |

`scripts/utils/station_config.py` is the one loader. It joins `stations.yaml` with the reference catalog and gives the form that the older modules expect.

## Where files go

gnssrefl owns its workspace. The workflow sets `REFL_CODE` and `ORBITS` and does not move gnssrefl files.

| Path | Content |
|---|---|
| `$REFL_CODE/input/<station>.json` | station file in use |
| `$REFL_CODE/<year>/snr/<station>/` | SNR files |
| `$REFL_CODE/<year>/results/<station>/` | heights of single arcs, one file per day. A run with other settings (an "extension") goes to a sub-folder |
| `$REFL_CODE/Files/<station>/` | subdaily output; `inverse/` holds the inverse-method output |
| `results_annual/<STATION>/` | combined tables, comparison tables and figures of the workflow |

## Modules

| Folder | Content |
|---|---|
| `scripts/core_processing/` | orchestration of phase 1, workspace setup |
| `scripts/external_tools/` | the one place that calls gnssrefl commands (`gnssrefl_executor.py`) |
| `scripts/external_apis/` | NOAA CO-OPS and NDBC clients |
| `scripts/utils/` | station settings, downloads, loaders for gnssrefl output, logging |
| `scripts/visualizer/` | figures. `periodogram_circle.py` and `station_figures.py` are new; the others belong to the comparison scripts and the animation |
| `tests/` | `pytest` tests; they need no network. Tests that read local gnssrefl output skip when it is absent |

### Older modules

Some modules come from the earlier version, which had an interactive dashboard: `utils/station_metadata.py`, `visualizer/dashboard_plots.py`, `visualizer/publication_theme.py` and parts of `visualizer/comparison*.py`. The dashboard is not in this branch. These modules were not reviewed in this rework, and the new commands do not use them.

## Design rules

- gnssrefl does the science. The workflow does not copy gnssrefl algorithms; it calls the commands and reads the result files.
- One station file, in the gnssrefl format. No second set of parameter names.
- New commands read result files and write a figure, a table or a page. They do not change the workspace, except `inverse_heights.py`, which writes to `Files/<station>/inverse/`.
- Credentials stay in the user's own login files (`es login` for EarthScope). The code never prints a token.
