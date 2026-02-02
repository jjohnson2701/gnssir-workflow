# Processing Example: GLBX Station

This example demonstrates the complete GNSS-IR processing workflow using the GLBX station (Bartlett Cove, Alaska) for January-February 2024.

## Station Details

| Parameter | Value |
|-----------|-------|
| Station ID | GLBX |
| Location | Bartlett Cove, Alaska |
| Coordinates | 58.46°N, 135.89°W |
| Antenna Height | -12.535 m (ellipsoidal) |
| Reference Source | ERDDAP (Bartlett Cove water level sensor, co-located) |

## Step 1: GNSS-IR Processing

Process RINEX data to calculate reflector heights:

```bash
python scripts/run_gnssir_processing.py \
    --station GLBX \
    --year 2024 \
    --doy_start 1 \
    --doy_end 60 \
    --num_cores 8 \
    --log_level INFO
```

### Expected Output

```
Starting GNSS-IR Processing
Station: GLBX, Year: 2024, DOY range: 1-60
Using 8 cores for parallel processing
...
Processing complete.
```

### Output Files

| File | Description |
|------|-------------|
| `results_annual/GLBX/GLBX_2024_combined_rh.csv` | Daily aggregated reflector heights |
| `results_annual/GLBX/GLBX_2024_combined_raw.csv` | Individual subdaily retrievals |
| `data/GLBX/2024/quicklook_plots_daily/` | QA diagnostic plots |

### Processing Results

| Metric | Value |
|--------|-------|
| Days processed | 59 (1 day missing data) |
| Total retrievals | 1,806 |
| Avg retrievals/day | 30.6 |

## Step 2: Reference Data Matching

Match GNSS-IR observations to the co-located ERDDAP water level sensor:

```bash
python scripts/generate_erddap_matched.py --station GLBX --year 2024
```

### Expected Output

```
Loading GNSS-IR data...
  Loaded 1,806 GNSS-IR observations
  Date range: 2024-01-01 to 2024-02-29

Loading ERDDAP data...
  Loaded 86,874 ERDDAP observations

Matching observations (max 30 min difference)...
  Matched 1,735 observations (96.1% of GNSS-IR data)
  Mean time difference: 93.0 seconds

Computing demeaned values and residuals...
  RMSE: 0.276 m
  Correlation (r): 0.983
```

### Validation Results

| Metric | Value |
|--------|-------|
| Matched observations | 1,735 (96.1%) |
| Correlation (r) | **0.983** |
| RMSE | **0.276 m** |
| Reference distance | 0.004 km (co-located) |

## Step 3: Visualization (Optional)

Generate resolution comparison plot:

```bash
python scripts/plot_resolution_comparison.py --station GLBX --year 2024
```

Generate polar animation:

```bash
python scripts/create_polar_animation.py --station GLBX --year 2024 --doy_start 1 --doy_end 31
```

## Step 4: Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Access at http://localhost:8501

## Alternative: Unified Workflow

For convenience, use `process_station.py` to run all steps automatically:

```bash
python scripts/process_station.py \
    --station GLBX \
    --year 2024 \
    --doy_start 1 \
    --doy_end 60
```

This runs:
1. GNSS-IR processing (or use `--skip_gnssir` if already done)
2. Reference comparison (ERDDAP for GLBX)
3. Visualization generation (or use `--skip_viz` to skip)

## Notes

- The high correlation (r=0.983) is due to the co-located ERDDAP water level sensor
- GLBX is a tidal station in Alaska with ~4m tidal range
- Processing time: ~10 minutes for 60 days with 8 cores
