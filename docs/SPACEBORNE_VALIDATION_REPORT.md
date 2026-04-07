# Spaceborne Validation & Dashboard Redesign — Session Report

**Date:** 2025-04-05  
**Branch:** greenland-earthscope  
**Stations analyzed:** ROSS, MCHN, PSTA, UMNQ (PCA); FORA, DESO (CYGNSS); ROSS (SMAP)

---

## Step 1: PCA Phase Validation

### What was done
Ran `scripts/compare_features_pca.py` on 4 stations to determine whether the SNR phase feature (matched-filter extraction at LSP-derived RH) adds discriminatory power to the ice classifier beyond CLR, AF, gamma, and PR.

### Results

| Feature | ROSS d | MCHN d | PSTA d | UMNQ d |
|---------|--------|--------|--------|--------|
| gamma | **1.11** | 0.30 | 0.11 | 0.05 |
| VS | **0.92** | **0.83** | 0.25 | **1.15** |
| AF | **0.84** | 0.78 | 0.24 | **1.27** |
| CLR | 0.12 | 0.78 | 0.35 | **1.09** |
| **phase** | **0.57** | 0.03 | 0.24 | 0.07 |
| PR | 0.23 | 0.58 | 0.20 | 0.50 |

Phase is fully independent from all other features (max |r| = 0.23 across all stations). Discriminatory power is station-dependent: medium at ROSS (d=0.57), negligible at MCHN and UMNQ.

### Decision
**Keep phase at weight=0.5** in the ice classifier (current setting). It provides independent information at stations where it matters and doesn't hurt where it doesn't. No code changes needed.

### Outputs
- `results_annual/{station}/pca/` — correlation_matrix.png, pca_biplot.png, feature_separation.csv, pairwise_scatter.png, phase_seasonal.png for each station.

---

## Step 2: CYGNSS Comparison

### What was built
- `scripts/cygnss_comparison.py` — Downloads CYGNSS L2 V3.2 from PO.DAAC via `earthaccess`, extracts specular points near a station, merges with ground SNR features, computes correlations.
- `scripts/cygnss_plots.py` — Six publication-quality plots: Fresnel heatmap, density map, empirical Fresnel curve, seasonal time series, correlation scatter, wind response box plots.
- `scripts/generate_html_report.py` — Reusable self-contained HTML report generator with base64-embedded images. Supports CYGNSS, SMAP, and generic report types.

### Data downloaded
- 366 CYGNSS L2 V3.2 daily netCDF4 files for 2024 (~18 GB total, cached in `data/.cache/cygnss/`)
- Each file contains ~1.2M specular point observations globally

### Results — FORA 2024
- **45,529 CYGNSS specular points** within 1° of FORA across 366 days
- **Closest specular point: 32 km offshore** — no Fresnel zone overlap (CYGNSS masks land/near-shore)
- **61 matched days** (FORA ground data covers DOY 60-120 only)

**Correlations (weak, as expected for open ocean):**

| CYGNSS var | Best ground corr | r |
|---|---|---|
| Fresnel coeff | phase | -0.29 |
| Wind speed | PR | +0.26 |
| Wind speed | RH | -0.25 |

**Key finding:** The Fresnel heatmap (gridded to 0.05°) cleanly resolves the land/water boundary at the NC Outer Banks — the same dielectric boundary that ground GNSS-IR detects at ~100m scale. The empirical Fresnel curve tracks the theoretical water permittivity (ε=80) curve closely, validating the shared reflection physics.

### Issues
1. **Datetime bug (fixed):** `sample_time` in CYGNSS L2 is seconds within the day, not since epoch. Initially mapped all 45K points to a single date. Fixed by parsing the file date from the filename.
2. **No Fresnel zone overlap:** CYGNSS masks land returns, so the nearest specular point is 32 km offshore. The comparison is "regional ocean vs local coast," not a direct spatial match.
3. **DESO:** Only 1 day of ground processing (404 arcs), zero matched days — effectively unusable.
4. **VALR:** 364 SNR files but only 37 gnssir result days — needs full processing before CYGNSS comparison is viable.
5. **Coverage limit:** CYGNSS (35° inclination) covers none of the ice stations (Great Lakes 42-49°N, Greenland 64-71°N). Only FORA (35.9°N), DESO (27.5°N), VALR (21.4°N) qualify — all subtropical with no ice.

### Outputs
- `results_annual/FORA/cygnss/` — extracted parquet, comparison parquet, 8 PNG plots, HTML report
- `results_annual/DESO/cygnss/` — extracted parquet, map only (no matched ground data)

---

## Step 2b: SMAP Comparison

### What was built
- `scripts/smap_comparison.py` — Downloads SMAP Enhanced L3 Freeze/Thaw (SPL3FTP_E, 9 km daily) from NSIDC via `earthaccess`, extracts TBv/TBh/NPR/FT at the nearest EASE-Grid pixel, merges with ice classification AND ground SNR features, produces time series, box plots, scatter, and correlation analysis.

### Data downloaded
- 365 SMAP SPL3FTP_E HDF5 files for 2024 (~165 GB download, cached in `data/.cache/smap_ft/`)
- Each file is ~450 MB (global 9 km EASE-Grid, AM + PM passes)

### SMAP data characteristics
- **Format:** HDF5, EASE-Grid 2.0 at 9 km resolution
- **Extraction method:** Single nearest pixel, no spatial averaging
- **ROSS pixel:** (199, 990) at 48.845°N, 87.526°W — 1.3 km from antenna
- **Pixel composition:** 43% open water fraction — mixed land/lake on a Lake Superior inlet where ice accumulates in winter
- **Key variables:** TBv, TBh (brightness temperature, Kelvin), NPR (normalized polarization ratio), freeze_thaw flag (binary, often fill=254 for water-heavy pixels)

### Results — ROSS 2024
- **365 SMAP days extracted, 263 matched** with ice classification
- **SMAP FT flag:** Only 6.1% agreement with GNSS-IR — expected, since the binary FT product is designed for soil, not lake ice

**Strong correlations between SMAP polarization and ground features:**

| SMAP TBv-TBh vs | r | Physical interpretation |
|---|---|---|
| Ground VS | **-0.665** | Both respond to surface roughness/coherence |
| Ground gamma | **-0.654** | Both respond to scattering vs specular reflection |
| Ground AF | **-0.642** | Both respond to reflection strength |
| Ground CLR | **-0.433** | Both respond to spectral peak clarity |
| Ground RH | **-0.388** | Ice changes effective reflection geometry |

These are the strongest correlations in the entire analysis — a 9 km spaceborne L-band radiometer correlates at r=-0.65 with a 100 m ground-based GNSS-IR Fresnel zone. The physical link: both instruments operate at L-band (~1.4-1.6 GHz) and respond to the same dielectric boundary (water ε≈80 vs ice ε≈3.2).

### Issues
1. **Initial report text overclaimed spatial relationship.** Early text said "direct line of sight, no islands or obstructions." The multi-scale satellite imagery showed ROSS is on an inlet with mixed land/water — corrected to describe a mixed coastal environment where both instruments respond to the same freeze/thaw phenomena, not identical spatial sampling.
2. **ROSS 2024 class imbalance:** Only 1 "water" day, 59 "ice" days, 203 "transition" out of 263. This makes box plot comparisons weak. A multi-year run (2015-2024, 10 years of overlap) would provide much better statistical power.
3. **SMAP pixel is mixed:** 43% water means the SMAP signal includes both land and lake contributions. The correlations are real but the SMAP pixel is not a pure lake measurement.

### SMAP temporal availability
SMAP SPL3FTP_E covers **2015-03-31 to present** (daily). ROSS has ice classification for **2003-2024**. This gives **10 years of overlap** (2015-2024) — the multi-year comparison is ready to run with `--years 2015-2024`.

### Outputs
- `results_annual/ROSS/smap/` — extracted parquet, comparison parquet, 7 PNG plots (timeseries, boxplot, scatter, pixel map, multiscale map, temporal evolution, vs_ground), HTML report

---

## Step 3: Dashboard Redesign (Prototype)

### What was built
- `dashboard_dash.py` — Plotly Dash prototype with the 3-panel layout from the spec

**Implemented:**
- Station dropdown auto-populated from `results_annual/` (discovers ~20 stations)
- Year dropdown updates per station based on available parquets
- **Map panel:** Esri satellite imagery basemap via dash-leaflet, all stations as markers, selected station highlighted, auto-centers on selection
- **Polar panel:** Plotly `scatterpolar` — azimuth × reflection distance, colored by frequency band (L1/L2C/L5/E6), hover shows RH + amplitude
- **Time series panel:** Daily median RH per frequency + overall median line. Bottom sub-panel shows ice classification band (ice/transition/water) from ice_classification.parquet
- **Quality bar:** Total arcs, days, arcs/day, RH median ± std, frequency percentage breakdown
- All data loads from existing parquets with no Streamlit dependency

### Not yet implemented (from spec)
- **Time brushing:** Select range on time series → polar and map update to show only those arcs
- **Sector click:** Click azimuth sector on polar → time series highlights arcs from that sector
- **Reference gauge overlay:** USGS/CO-OPS/ERDDAP water level on same time series axis
- **Suspect azimuth shading:** Visual indicator of known bad sectors from config
- **URL routing:** `/station/UMNQ?days=30` style deep links
- **Map reflection points:** Show individual arc reflection points on the satellite basemap (currently just station markers)
- **Hover tooltips on map:** Arc metadata (satellite, frequency, amplitude)

### Dependencies installed
```
dash==4.1.0
dash-leaflet==1.0.15
plotly==6.6.0 (was already installed)
kaleido==0.2.1 (for static image export)
earthaccess==0.11.0
netCDF4==1.7.2
xarray==2024.7.0
h5py==3.14.0
contextily==1.6.2 (was already installed)
```

---

## Infrastructure Built

### HTML Report Generator
`scripts/generate_html_report.py` — reusable tool for creating self-contained HTML reports with embedded base64 images. Supports:
- CYGNSS report template (6 sections with captions)
- SMAP report template (7 sections with spatial context)
- Generic mode (scan directory for PNGs)
- Used with `litterbox.catbox.moe` for 24-hour temporary hosting for remote review

### Data cached
| Dataset | Location | Size | Coverage |
|---|---|---|---|
| CYGNSS L2 V3.2 | `data/.cache/cygnss/` | ~18 GB | 2024 (366 files) |
| SMAP SPL3FTP_E | `data/.cache/smap_ft/` | ~165 GB | 2024 (365 files) |
| GLERL ice | `data/.cache/glerl_ice/` | ~1 MB | 2003-2024 (varies by station) |

---

## What's Left To Do

### High priority
1. **SMAP multi-year run:** `python scripts/smap_comparison.py --station ROSS --years 2015-2024` — needs downloading 2015-2023 SMAP data (~1,500 more HDF5 files). Will give 10 years of SMAP vs ice classifier comparison with much better statistical power.
2. **Dashboard time brushing:** The core interaction from the spec — select a time range on the time series, polar and map update. This is the single feature that makes Dash worth using over Streamlit.
3. **Reference gauge overlay:** Add USGS/CO-OPS water level to the time series panel. The data loaders already exist in `dashboard_components/data_loader.py`.

### Medium priority
4. **SMAP for additional stations:** Run MCHN, PSTA, CLWD, TOBY (other Great Lakes with ice classification). The SMAP data is already cached for 2024; just needs `--station X`.
5. **Dashboard reflection points on map:** Show individual arc positions on the satellite basemap, colored by amplitude or WSE.
6. **Dashboard suspect azimuths:** Read `suspect_azimuths` from station config, shade those sectors on the polar diagram.
7. **VALR full processing:** Run gnssir for the remaining 327 unprocessed DOYs to enable CYGNSS comparison at a tropical station.

### Lower priority
8. **Dashboard URL routing:** Deep links for bookmarkable views.
9. **Dashboard sector click interaction:** Click polar sector → filter time series.
10. **SMAP multi-pixel analysis:** Instead of single nearest pixel, extract a 3×3 or 5×5 grid and weight by water fraction for a better lake-representative signal.
11. **S1 SAR improvements:** User indicated they will lead this — existing `s1_fresnel_*.py` code is the starting point.

---

## New Scripts Created

| Script | Purpose |
|---|---|
| `scripts/compare_features_pca.py` | PCA + Cohen's d analysis of SNR features for ice detection |
| `scripts/cygnss_comparison.py` | Download CYGNSS L2, extract near station, compare with ground |
| `scripts/cygnss_plots.py` | Publication-quality CYGNSS visualization suite |
| `scripts/smap_comparison.py` | Download SMAP FT, extract pixel time series, compare with ice classifier + ground features |
| `scripts/generate_html_report.py` | Self-contained HTML report generator with embedded images |
| `dashboard_dash.py` | Plotly Dash 3-panel dashboard prototype |
