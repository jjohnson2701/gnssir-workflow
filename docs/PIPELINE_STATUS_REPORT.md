# Ice Classification Pipeline: Status and Decision Report

**Date:** 2025-04-06  
**Branch:** `gnssir-surface`  
**Stations tested today:** UMNQ (2025), ROSS (2024), MCHN (2024), PSTA (2024)

---

## 1. What We Have: Three Classification Lineages

The pipeline has evolved through three distinct approaches, all still present in the codebase:

### v1: Per-Sector Threshold Voting (`ice_classifier.py` — current, committed)

The active classifier. Reads Layer 1 (arc_table) and Layer 2 (daily_features). Per-azimuth-sector voting with 7 weighted indicators.

**Pipeline:**
```
compute_af_baselines.py   (optional: per-PRN power curve baseline for AF)
        ↓
snr_feature_extractor.py  (6 per-arc features: CLR, PR, AF, gamma, phase, MS/VS)
        ↓
feature_aggregator.py     (daily × sector medians, z-scores → daily_features.parquet)
        ↓
ice_classifier.py         (threshold voting → ice_state.parquet)
```

**Key features (all implemented):**
- Per-satellite z-score normalization using ice-free months
- 3-day or 5-day centered rolling median smoothing
- Summer-anchored percentile thresholds per sector
- `feature_overrides` in station config to flip polarity / adjust weights
- Land detection via seasonal amplitude ratio (<1.15x)

**Output:** `{station}_{year}_ice_state.parquet` (Layer 3)

### v2: PCA + Mahalanobis Distance (`scripts/utils/mahalanobis.py` — committed)

A complementary approach. Builds a PCA-reduced "water baseline" from ice-free months, then measures each day's distance from that baseline in feature space. Days far from water = likely ice.

**Pipeline:**
```
mahalanobis.build_water_model()   (PCA + covariance from summer daily_features)
        ↓
mahalanobis.score_mahalanobis()   (project all days, compute distances)
        ↓
mahalanobis.optimize_threshold()  (ROC against GLERL → Youden's J threshold)
        ↓
mahalanobis.detect_states()       (state machine: water → freeze_up → ice → break_up)
```

**Key features:**
- StandardScaler + PCA retaining 95% variance (typically 2-3 components)
- Features: amp_mean, rh_std, clr_med, af_med, gamma_med, pr_med, p2n_mean
- Temporal coherence state machine with min_duration (3 days)
- Validated: AUC=0.895 vs GLERL across 9 Great Lakes stations
- No polarity assumptions — distance is polarity-agnostic

**Output:** `{station}_mahalanobis_distances.parquet` (exists for ROSS, MCHN, GOD2, PSTA, UMNQ)

### v3: Sub-State Resolution (`scripts/classifier_v3.py` — uncommitted)

Adds ice sub-types on top of v2 Mahalanobis output:

| State | Definition |
|-------|-----------|
| `open_water` | Mahal < threshold |
| `freeze_up` | v2 state machine transition |
| `ice_surface` | Mahal > threshold, \|ΔRH\| < 0.3m |
| `ice_layered` | Mahal > threshold, \|ΔRH\| >= 0.3m (multi-layer penetration) |
| `ice_decaying` | Mahal > threshold, ERA5 t2m > freezing |
| `break_up` | v2 state machine transition |

**Output:** `{station}_{year}_ice_classification_v3.parquet` (exists for ROSS 2003-2024, MCHN 2020-2024, GOD2 2020-2024, PSTA 2020-2024)

---

## 2. What We Ran Today

Ran the v1 (threshold voting) classifier through the full pipeline for 4 stations. This is the first time the literature-aligned pipeline (Tasks 1-5 from the implementation plan) has been tested on Great Lakes stations.

### Per-Station Configuration

Default indicator table (Arctic model, from Strandberg/Song/Purnell papers):

| Indicator | Default: High = | Weight | Source |
|-----------|----------------|--------|--------|
| Amplitude | Ice | 2.0 | Standard GNSS-IR |
| Amp CV | Water | 1.5 | Standard GNSS-IR |
| RH std | Water | 1.0 | Standard GNSS-IR |
| CLR | Ice | 2.0 | Purnell 2024 |
| AF | Ice | 1.5 | Song 2022 |
| PR | Ice | 1.0 | Purnell 2024 |
| Gamma | Water | 1.0 | Strandberg 2017 |

Station-specific overrides applied:

| Indicator | UMNQ | ROSS | MCHN | PSTA |
|-----------|------|------|------|------|
| **Amplitude** | High=Ice, w=2.0 | High=Ice, w=2.0 | High=Ice, w=2.0 | High=Ice, w=2.0 |
| **Amp CV** | High=Water, w=1.5 | High=Water, w=1.5 | High=Water, w=1.5 | High=Water, w=1.5 |
| **RH std** | High=Water, w=1.0 | High=Water, w=1.0 | High=Water, w=1.0 | High=Water, w=1.0 |
| **CLR** | High=Ice, w=2.0 | **High=Water, w=1.0** | High=Ice, w=1.0 | High=Ice, w=2.0 |
| **AF** | High=Ice, w=1.5 | High=Ice, **w=2.0** | High=Ice, **w=2.0** | High=Ice, **w=2.0** |
| **PR** | High=Ice, w=1.0 | **High=Water**, w=1.0 | High=Ice, w=1.0 | High=Ice, w=1.0 |
| **Gamma** | High=Water, w=1.0 | **High=Ice, w=1.5** | **High=Ice, w=1.5** | **High=Ice**, w=1.0 |

Bold = changed from default. Note: ROSS and MCHN are both on Lake Superior but have different overrides. This inconsistency was hand-tuned.

Other config differences:

| Setting | UMNQ | ROSS | MCHN | PSTA |
|---------|------|------|------|------|
| ice_free_months | [7, 8] | [6, 7, 8, 9, 10] | [6, 7, 8, 9] | [6, 7, 8, 9] |
| smoothing_window | 3 | 5 | 5 | 5 |
| min_arcs_per_sector | 3 | 4 | 4 | 4 |

### Results

| Station | Location | Days | Ice | Transition | Water | Best Winter | Best Summer |
|---------|----------|------|-----|------------|-------|-------------|-------------|
| UMNQ | Greenland fjord | 281 | 17 (6%) | 177 (63%) | 87 (31%) | -0.32 (Apr) | +0.44 (Oct) |
| ROSS | Lake Superior | 263 | 59 (22%) | 203 (77%) | 1 (<1%) | -0.31 (Feb-Mar) | -0.06 (Jul) |
| MCHN | Lake Superior | 188 | 35 (19%) | 143 (76%) | 7 (4%) | -0.40 (Feb) | +0.09 (Mar) |
| PSTA | Lake Erie | 263 | 100 (38%) | 152 (58%) | 10 (4%) | -0.26 (Feb) | **-0.43 (Aug)** |

### Seasonal Feature Directions (winter vs summer medians)

| Feature | UMNQ | ROSS | MCHN | PSTA | Arctic expects |
|---------|------|------|------|------|---------------|
| Amplitude | W > S (36.7 vs 24.3) | W > S (16.0 vs 9.7) | W > S (15.6 vs 13.1) | W = S (16.5 vs 16.4) | W > S |
| CLR | W > S (11.7 vs 8.7) | W > S (5.3 vs 4.5) | W > S (4.2 vs 4.0) | **S > W** (4.0 vs 3.7) | W > S |
| AF | W > S (57k vs 29k) | W > S (28k vs 10k) | W > S (4.4k vs 3.9k) | **S > W** (5.5k vs 5.4k) | W > S |
| Gamma | S > W | **W > S** | **W > S** | S > W | S > W |
| Z-score Δ (CLR) | 1.33σ | 1.04σ | 0.17σ | 0.32σ | — |
| Z-score Δ (AF) | 2.22σ | 5.64σ | 0.31σ | 0.15σ | — |

**Key finding: Gamma is inverted at Great Lakes stations** (high gamma = ice, not water). Great Lakes ice is rough/ridged/wind-piled, causing faster envelope decay. The Arctic model assumes smooth flat ice.

**PSTA is fully inverted** — summer classified as ice because calm Lake Erie water produces the same smooth-surface signature as Arctic ice.

### AF Baseline Correction Impact

Computed per-PRN power curve baselines for ROSS (38 combos) and MCHN (48 combos). Re-extracted, re-aggregated, re-classified.

**Result: Negligible change.** The z-score normalization already handles per-satellite bias. The classification counts were identical (ROSS) or shifted by +/- 2 days (MCHN).

---

## 3. Uncommitted Work Inventory

### Analysis Scripts (untracked, in `scripts/`)

| Script | Lines | Purpose | Dependencies |
|--------|-------|---------|-------------|
| `analyze_land_water_azimuths.py` | 1,084 | Land vs water azimuth classification, interfreq validation, Mahalanobis prototype | scipy, sklearn |
| `compare_features_pca.py` | 370 | PCA biplot, correlation matrix, Cohen's d effect sizes per feature | sklearn.decomposition |
| `compare_features_dendrogram.py` | 650 | Hierarchical feature clustering, partial correlation distance, per-state dendrograms | scipy.cluster, plotly |
| `tec_era5_pca.py` | 384 | Combined PCA of SNR + TEC + ERA5 — ionosphere vs surface signal separation | sklearn, ERA5/TEC data |
| `cygnss_comparison.py` | 505 | CYGNSS L2 specular-point comparison with ground GNSS-IR features | requests, external_data |
| `cygnss_plots.py` | 406 | Publication plots for CYGNSS comparison results | matplotlib |
| `smap_comparison.py` | 621 | SMAP L-band radiometry vs GNSS-IR ice classification | requests |
| `era5_extractor.py` | 294 | ERA5 meteorological extraction at station locations | CDS API |
| `tec_extractor.py` | 364 | TEC extraction from IGS IONEX files | CDDIS auth |
| `ingest_gnssrefl_archive.py` | 360 | Ingest Canadian Great Lakes gnssir archive into results_annual/ | results_handler |
| `generate_ice_notebook_greenland.py` | 2,732 | Jupyter notebook generator for Greenland ice classification analysis | nbformat |
| `generate_html_report.py` | 431 | Self-contained HTML report with embedded PNGs | base64 |
| `verify_layer2.py` | 216 | QA: verify daily_features matches classifier transient computations | pandas |
| `classifier_v3.py` | 266 | v3 sub-state classifier (builds on v2 Mahalanobis) | v2 output |

### Reusable Modules (untracked)

| Module | Lines | Purpose | Used by |
|--------|-------|---------|---------|
| `scripts/utils/reference_data.py` | 488 | Canonical loading for per-arc data + reference timeseries | polar_diagram, comparison scripts |
| `scripts/visualizer/polar_diagram.py` | 484 | PolarDiagram renderer (5 color modes, Plotly + matplotlib) | dashboard, notebooks |

### Modified Files (tracked, uncommitted changes)

| File | Change | Impact |
|------|--------|--------|
| `config/stations_config.json` | +381 lines: external_data_sources (HYDAT, CO-OPS, EC Climate) for all 10 GL stations | Dashboard map, reference data fetchers |
| `dashboard_dash.py` | +63 lines: reference station markers on map, CBRG snap-back fix | Dashboard UX |
| `scripts/utils/external_data.py` | +293 lines: HYDAT + EC Climate fetchers with parquet caching | Reference data pipeline |
| `scripts/compute_af_baselines.py` | 9-line fix: use resolve_layer1() instead of hardcoded path | Maintainability |

### Generated Outputs (not in git)

- **Mahalanobis distances:** 5 stations (ROSS, MCHN, GOD2, PSTA, UMNQ)
- **v3 classifications:** ROSS 2003-2024 (22 years), MCHN/GOD2/PSTA 2020-2024
- **ice_state (v1/current):** UMNQ 2025, ROSS 2024, MCHN 2024, PSTA 2024
- **Analysis plots:** 46 PNGs in `data/analysis/plots/` (PCA, Mahalanobis timeseries, feature anatomy, ROC curves)

---

## 4. Dashboard State

### Dash (`dashboard_dash.py` — 1,567 lines, active)

6 tabs: Overview, Time Series, Polar, Ice Classification, Imagery, Features

**Ice tab is minimal** — renders 3 SNR features (gamma, VS, AF) colored by v3 state. No methodology narrative, no voting breakdown, no S1 validation.

**Features tab** has the dendrogram/heatmap with partial correlation clustering. This is unique to Dash (not in Streamlit).

**Self-contained** — all data loading inline, no imports from dashboard_components/.

### Streamlit (`dashboard.py` + `dashboard_components/` — 8,123 lines in tabs alone)

10 tabs. The ice comparison tab (`ice_comparison_tab.py`, 1,589 lines) has the full paper-by-paper narrative with 40+ rendering functions. This is comprehensive but **no longer being used**.

**Key Streamlit content not in Dash:**

| Content | Streamlit lines | Dash status |
|---------|----------------|-------------|
| Ice narrative (Sections 0-8, paper-by-paper) | 1,589 | 75 lines (3 features only) |
| Per-arc explorer (subdaily, freq comparison) | 1,172 | Not present |
| Validation (temporal zoom: year/month/week) | 792 | Not present |
| Yearly residual analysis | 1,070 | Not present |
| Subdaily WSE comparison | 785 | Not present |
| Data quality diagnostics | 449 | Partial (in overview) |
| Feature explorer | 347 | Replaced by dendrogram |
| Monthly data | 431 | Not present |

---

## 5. Relationship Between v1 Voting and v2 Mahalanobis

The two approaches have complementary strengths:

| Aspect | v1 (Threshold Voting) | v2 (Mahalanobis) |
|--------|----------------------|-------------------|
| **Polarity handling** | Requires explicit per-station overrides | Polarity-agnostic (distance from water) |
| **Interpretability** | High — each indicator's vote is visible | Medium — PCA components are mixed |
| **Transferability** | Poor — overrides needed per environment | Better — AUC=0.895 across 9 GL stations |
| **Sub-state resolution** | None (ice/transition/water) | v3 adds 6 sub-states |
| **Per-sector** | Yes — per-azimuth classification | No — pooled across azimuths |
| **Temporal coherence** | Rolling median smoothing | State machine with min_duration |
| **Dashboard support** | Streamlit has full narrative; Dash minimal | Dash renders v3 state colors everywhere |

**The Dash dashboard already loads v3 output** (`load_v3()` at line 106) and colors all time series/imagery by v3 state. The v1 `ice_state.parquet` is not currently consumed by the Dash dashboard.

---

## 6. Decision Points

### A. Which classifier to develop further?

**Option 1: Invest in v1 voting.** Fix the feature_overrides for all GL stations. Advantage: interpretable, per-sector, aligns with paper narrative. Disadvantage: the polarity problem is fundamental — requires per-station (or per-ice-regime) tuning that may not generalize.

**Option 2: Invest in v2/v3 Mahalanobis.** Already validated across 9 stations with AUC=0.895. The distance approach sidesteps the polarity problem entirely. Disadvantage: less interpretable, no per-sector resolution.

**Option 3: Both.** Use v1 voting as the interpretable diagnostic layer (shown in ice narrative tab), and v2/v3 Mahalanobis as the operational classification (state colors on timeseries/imagery). The approaches answer different questions: v1 says "which indicator thinks what", v2 says "how anomalous is today relative to summer water".

### B. What to commit?

Suggested commit groupings:

1. **Reference data integration** (ready now):
   - `config/stations_config.json` (external_data_sources)
   - `scripts/utils/external_data.py` (HYDAT + EC Climate fetchers)
   - `dashboard_dash.py` (map markers + CBRG fix)
   - `scripts/compute_af_baselines.py` (resolve_layer1 fix)

2. **Analysis scripts** (ready, standalone):
   - `scripts/compare_features_pca.py`
   - `scripts/compare_features_dendrogram.py`
   - `scripts/tec_era5_pca.py`
   - `scripts/analyze_land_water_azimuths.py`

3. **Spaceborne validation** (ready, standalone):
   - `scripts/cygnss_comparison.py` + `scripts/cygnss_plots.py`
   - `scripts/smap_comparison.py`

4. **Data extractors** (ready, reusable):
   - `scripts/era5_extractor.py`
   - `scripts/tec_extractor.py`
   - `scripts/ingest_gnssrefl_archive.py`

5. **v3 classifier + reusable modules** (ready):
   - `scripts/classifier_v3.py`
   - `scripts/utils/reference_data.py`
   - `scripts/visualizer/polar_diagram.py`

6. **Notebook generators** (lower priority):
   - `scripts/generate_ice_notebook_greenland.py`
   - `scripts/generate_html_report.py`

### C. What to build in the Dash dashboard?

The Streamlit ice narrative (1,589 lines) is comprehensive but needs to be ported to Dash if we want it in the active dashboard. Three approaches:

**Option 1: Port the full narrative.** Rewrite all 8 sections in Dash/Plotly. Large effort (~2-3 sessions) but produces the definitive ice analysis interface.

**Option 2: Port selectively.** Bring over the parts that matter for presentation:
- Metrics strip (Section 0)
- Seasonal feature comparison table (new — the non-transferability finding)
- Score timeseries with temperature overlay (Section 6)  
- Sector heatmap (Section 7)
- Monthly summary (Section 8)

**Option 3: Build for the Bern talk.** Focus on what needs to be shown: the non-transferability result. Add a cross-station comparison view to Dash that doesn't exist in Streamlit — feature direction table, seasonal contrast metrics, side-by-side classification timeseries for UMNQ vs ROSS \vs PSTA.

### D. How to present the non-transferability finding?

The strongest presentation artifact would combine:

1. **Feature direction table** (from today's analysis) — shows which features agree/disagree with Arctic expectations per station
2. **PCA biplots** (from `compare_features_pca.py`) — shows how ice/water clusters separate in PC space differently at each station  
3. **Mahalanobis distance timeseries** (from `mahalanobis.py` outputs) — shows that distance-based classification works across stations without polarity tuning
4. **Classifier comparison** — v1 voting (fails at GL) vs v2 Mahalanobis (AUC=0.895) on the same data

This could be a Dash tab, a standalone report, or both.

---

## 7. What's Working vs What Needs Work

### Working well
- v1 pipeline on Arctic stations (UMNQ)
- v2 Mahalanobis validated on Great Lakes (AUC=0.895)
- v3 sub-state resolution (ice_layered, ice_decaying)
- Dendrogram/clustering in Dash Features tab
- Reference data integration (HYDAT, CO-OPS, EC Climate)
- AF baseline computation pipeline
- Example arc extraction for dashboard display

### Needs work
- v1 feature_overrides inconsistent across Great Lakes (ROSS vs MCHN differ)
- PSTA fully inverted — needs either polarity fix or flagging as "not suited for v1"
- Dash ice tab is skeletal (75 lines vs 1,589 in Streamlit)
- No cross-station comparison view in either dashboard
- Streamlit is abandoned but contains valuable rendering code
- 15 uncommitted scripts (4,685 lines of analysis work)
