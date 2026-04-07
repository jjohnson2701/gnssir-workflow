# Session Status: Specs vs Delivery

**Session dates:** 2025-04-05 to 2025-04-06 (updated end of session)

---

## Step 1: PCA Phase Validation

**Spec:** Determine whether SNR phase adds discriminatory power to ice detection.

| Item | Status | Notes |
|------|--------|-------|
| `compare_features_pca.py` | Done | Runs on any station/year |
| Run on ROSS | Done | d=0.57, independent (r<0.23) |
| Run on MCHN | Done | d=0.03, weak |
| Run on PSTA | Done | d=0.24, weak |
| Run on UMNQ | Done | d=0.07, weak |
| Decision | Done | Keep phase at weight=0.5, no code change |

**Verdict:** Complete. Phase is independent but station-dependent. Clear recommendation delivered.

---

## Step 2: CYGNSS Comparison

**Spec:** Compare ground GNSS-IR with spaceborne CYGNSS reflectometry.

| Item | Status | Notes |
|------|--------|-------|
| Check coverage | Done | Only 3 stations qualify (FORA, DESO, VALR), none are ice stations |
| Download CYGNSS L2 | Done | 366 files for 2024, cached |
| `cygnss_comparison.py` | Done | Download, extract, match, correlate |
| `cygnss_plots.py` | Done | 6 publication-quality plot types |
| Run FORA | Done | 45K points, 61 matched days, weak correlations (expected — no ice) |
| Run DESO | Done | 0 matched days (only 1 day of ground data) |
| Run VALR | Not done | Needs gnssir processing (37/364 DOYs processed) |
| Datetime bug | Fixed | sample_time was seconds-in-day, not epoch |
| Fresnel zone overlap | Checked | None — nearest CYGNSS point is 32km offshore |
| HTML report | Done | With narrative text |

**Verdict:** Mostly complete. CYGNSS validated the shared reflection physics (Fresnel curve matches theory) but can't validate ice classification — wrong latitudes. VALR remains unprocessed. The spec correctly predicted this would be a dead end for ice and suggested pivoting to SMAP.

**Open thread:** VALR gnssir processing (327 DOYs unprocessed). Low priority — tropical station, no ice.

---

## Step 2b: SMAP Comparison (Pivot from CYGNSS)

**Spec:** Compare SMAP L-band freeze/thaw with ground GNSS-IR ice classification.

| Item | Status | Notes |
|------|--------|-------|
| `smap_comparison.py` | Done | Download, extract, merge with ice_cls + ground features |
| Download SMAP 2024 | Done | 365 files, cached |
| Run ROSS | Done | 263 matched days, strong correlations |
| Pixel map | Done | Shows mixed land/water inlet pixel |
| Multi-scale map | Done | SMAP grid → satellite → Fresnel zone |
| Temporal evolution | Done | Monthly snapshots + time series |
| SMAP vs ground features | Done | r=-0.65 (pol_diff vs VS/gamma/AF) |
| HTML report | Done | 7 sections with corrected text about inlet |
| Run other stations | Not done | Only ROSS has SMAP extraction |
| Multi-year (2015-2024) | Not done | Only 2024 processed |

**Verdict:** Core analysis complete and produced the strongest cross-validation result of the session (r=-0.65). Key finding: SMAP pixel is mixed land/water on an inlet, not open lake — report text was corrected after initial overclaiming.

**Open threads:**
- Run SMAP for MCHN, PSTA, GOD2 (data is cached, just needs `--station X`)
- Multi-year SMAP at ROSS (2015-2024, 10 years overlap — would give much better statistics)
- Retrieve accumulated ERA5 variables (precip, snow depth) lost to CDS zip format

---

## Step 3: Dashboard Redesign

**Spec:** Redesign from QA tool to communication tool with map, polar, time series, quality bar.

| Item | Status | Notes |
|------|--------|-------|
| Technology decision | Done | Plotly Dash selected |
| Install dependencies | Done | dash, dash-leaflet, kaleido |
| `dashboard_dash.py` | Done | Working prototype |
| Map panel | Done | Esri satellite basemap, station markers, auto-center |
| Polar diagram | Done | Scatterpolar, azimuth×distance, by frequency |
| Time series | Done | Daily RH by frequency + ice classification strip |
| Quality bar | Done | Arcs, days, arcs/day, RH stats, freq breakdown |
| Station/year dropdowns | Done | Auto-discovers from parquets |
| Static preview | Done | Rendered to HTML for remote review |
| Time brushing | Not done | Select range → update polar/map |
| Sector click | Not done | Click azimuth → filter time series |
| Reference gauge overlay | Not done | USGS/CO-OPS on same axis |
| Suspect azimuth shading | Not done | Config-driven visual indicator |
| URL routing | Not done | Deep links |
| Map reflection points | Not done | Individual arcs on basemap |

**Verdict:** Functional prototype with the core layout. Missing the key interactive features (brushing, linked views) that motivated choosing Dash over Streamlit. The prototype proves the layout works but isn't yet better than the existing Streamlit dashboard for actual use.

**Open threads:**
- Time brushing is the critical missing feature
- Reference gauge overlay is needed for the water level use case
- The existing Streamlit dashboard still has more features

---

## Step 4: TEC and ERA5 Integration

**Spec:** Add TEC and ERA5 to PCA to determine whether interfrequency divergence is ionospheric or surface-driven.

| Item | Status | Notes |
|------|--------|-------|
| `tec_extractor.py` | Done | IONEX parser, CDDIS auth, bilinear interpolation |
| `era5_extractor.py` | Done | CDS API, monthly chunks, handles zip format |
| `tec_era5_pca.py` | Done | Combined PCA, TEC vs interfreq, ERA5 vs features |
| Download TEC ROSS 2024 | Done | 297/365 days (some IONEX parse errors) |
| Download ERA5 ROSS 2024 | Done | 366 days, temperature + wind |
| TEC vs interfreq analysis | Done | r=-0.31, inconclusive alone |
| Combined PCA | Done | TEC on different axis than ice features — NOT confounding |
| ERA5 vs features | Done | t2m correlates r=-0.67 to -0.71 with ice features |
| ERA5 precip/snow | Not done | Lost to CDS zip split (accumulated vs instant) |
| TEC for other stations | Not done | Only ROSS |
| ERA5 for other stations | Not done | Only ROSS |

**Verdict:** Core question answered — TEC does not confound the ice signal. ERA5 temperature confirms the classifier tracks real freeze/thaw physics. The 32 "warm ice" days (ice features + above-freezing temperature) led directly to the ice_decaying sub-state concept.

**Open threads:**
- Fix ERA5 extractor to also retrieve accumulated variables (precip, snow depth)
- Run TEC + ERA5 for MCHN, PSTA to confirm cross-station
- IONEX parser still fails on ~20% of days (hour=24 edge case needs better handling)

---

## Classifier v3 Design (Emerged from Steps 1-4)

**Not in original spec** — emerged from diagnosing v1's 77% transition problem.

| Item | Status | Notes |
|------|--------|-------|
| Diagnose v1 failure | Done | P30/P70 voting structurally biased toward transition |
| Discover v2 exists | Done | Already built, AUC=0.895, water-default Mahalanobis |
| Compare v1 vs v2 | Done | v2 fixes water identification (4% → 67%) |
| GMM regime discovery | Done | 5 natural clusters found at ROSS |
| Ice_layered analysis | Done | Cross-station: ROSS 25%, MCHN 30%, KNGV 27%, others rare |
| Warm-ice characterization | Done | 32 days, decaying shore-fast ice |
| Define v3 states | In progress | 6 proposed states, 4 confident, 2 need validation |
| Run v2 on new stations | Done | CLWD, TOBY, KNGV, PARY, PWEL all processed |
| Implement v3 | Not done | |
| Validate v3 vs GLERL | Not done | |
| Cross-station transferability | Partially done | Ice_layered checked, other states not yet |

**Key finding:** Ice_layered signal (L2C measuring deeper/higher than L1 by ~1m) appears at 3 of 8 stations (ROSS, MCHN, KNGV at 25-30%). Absent at others. Direction is consistently negative (L2C deeper) at those 3 stations. Likely snow-on-ice or ridged ice rather than signal penetration. Worth investigating with satellite imagery.

**Open threads:**
- What makes ROSS/MCHN/KNGV different geographically?
- Satellite imagery validation of layered-signal days
- Rough water detection was proposed but found no evidence at ROSS
- Frazil vs columnar ice distinction not achievable with current features
- v3 implementation needs to decide on transition handling

---

## HTML Report Infrastructure

**Not in original spec** — built to solve remote review problem.

| Item | Status | Notes |
|------|--------|-------|
| `generate_html_report.py` | Done | Reusable, base64 images, CYGNSS/SMAP/generic templates |
| Litterbox upload workflow | Done | 24-hour temporary URLs |
| Report quality | Improved | Initially had orphaned figures, now has per-figure narratives |

**Reports generated:**
- CYGNSS FORA 2024 (6 sections)
- SMAP ROSS 2024 (7 sections, corrected text)
- TEC + ERA5 PCA (10 sections with full narratives)
- v3 design comparison (10 sections)

---

## Summary: What's Proven vs What's Open

### Proven
1. Phase feature: independent, station-dependent, keep as-is
2. CYGNSS: validates reflection physics but doesn't cover ice stations
3. SMAP: strong correlation (r=-0.65) with ground features at the same mixed pixel
4. TEC: does NOT confound the ice signal
5. ERA5: temperature tracks ice features (r=-0.67 to -0.71)
6. v1 classifier: structurally broken (77% transition)
7. v2 classifier: works (67% water, 33% ice)
8. Ice_layered signal: real, appears at 3/8 stations, ~1m ΔRH magnitude

### Open Threads (by priority)

**High — would strengthen findings:**
1. v3 classifier implementation with the 4 confident states
2. SMAP multi-year at ROSS (2015-2024) for better ice/water statistics
3. ERA5 fix for precip/snow variables
4. Satellite imagery of ice_layered days at ROSS/MCHN to confirm snow-on-ice

**Medium — extend to more stations:**
5. SMAP for MCHN, PSTA, GOD2
6. TEC + ERA5 for MCHN, PSTA
7. Dashboard time brushing
8. Investigate why ROSS/MCHN/KNGV show ice_layered but others don't

**Lower — complete unfinished work:**
9. VALR gnssir processing for CYGNSS comparison
10. Dashboard reference gauge overlay
11. PARY v2 threshold investigation (0 ice days classified)
12. IONEX parser edge case fixes

---

## End-of-Session Update

### v3 Classifier — Full Deployment

- Ran on all 9 Great Lakes stations (2020-2024) 
- ROSS extended to full history (2003-2024, 5,885 days, 22 years)
- Ice_layered first appears 2007 (when L2C became available)
- Interfreq loader fixed to handle pre-L2C years gracefully

### SMAP Multi-Year (2020-2024)

- 5 years downloaded and analyzed for ROSS
- SMAP polarization clearly separates open_water (21.4K) from ice states (18.2-18.5K)
- SMAP **cannot** distinguish ice_surface from ice_layered (p=0.94) — snow-on-ice is too local for 9km pixel
- Correlation pol_diff vs v3_state: r=-0.60

### S2/S1 Imagery Validation

- S2 via Planetary Computer STAC API: 7 thumbnails across 3 stations
- S1 via ASF OPERA RTC: ROSS (178), MCHN (165), KNGV (180+), CLWD (448) scenes downloaded
- **Key confirmation**: 2024-02-02 S2 shows visible snow on ice_layered day at ROSS
- S1 SAR: VV backscatter increases water → ice_surface → ice_layered consistently

### KNGV False Positive — Resolved

- Nov 2020 ice_layered classification at KNGV: GLERL confirms 0% ice
- Root cause: narrow 30° azimuth window → noisy Mahalanobis → false freeze-up trigger
- Needs higher threshold or longer min_duration for KNGV specifically

### Station Config Audit

- DESO: azval2 130-330° captures significant land on Gulf coast
- KNGV: 30° az window is too narrow for reliable classification
- FORA: multi-range (0-80° + 340-360°) is correct
- NIAQ/UMNQ: suspect azimuths documented in config

### Dashboard v2 — 5-Tab Plotly Dash

**Running on port 8052 with --debug (hot-reload)**

Completed features:
- Dark mode throughout
- Loading spinner
- Station dropdown defaults to ROSS 2024
- Overview: satellite map at Fresnel-zone zoom + Fresnel cone overlay + polar + time series
- Time Series: RH by freq + ERA5 temp panel + SMAP panel + v3 strip + S1 ▼ markers + click-to-imagery (±3 day fuzzy match)
- Polar: reflection points + station metadata panel (data badges, config, frequencies, az range)
- Ice Classification: gamma/VS/AF by DOY colored by v3 + ERA5 + SMAP
- Imagery: thumbnail grid with v3 labels, click to expand full 5km S1 crop (VV/VH 2×2 with Fresnel zone close-up)
- Map markers clickable → switch station
- Full-width layout, all charts responsive

Remaining for next session:
- Reference gauge overlay (only FORA/GLBX have data)
- Time series brushing → polar/map update
- Station photos
- Hover thumbnail previews on time series
- DESO config fix
