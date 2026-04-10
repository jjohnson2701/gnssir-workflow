# Feature Investigation Log

Date: 2026-04-07
Station: UMNQ (Upernavik, Greenland, 70.7N) — 62,079 arcs, 10 frequency bands, 281 DOYs in 2025
Secondary: ROSS (Great Lakes) — 2024 data

Diagnostic output: `results_annual/UMNQ/diagnostics/`

---

## Part 1: QC Fixes and Feature Investigations

### 1. Hilbert Edge Tapering

**Hypothesis:** The FFT-based Hilbert transform assumes periodicity. At arc boundaries where the detrended SNR is mid-oscillation, spectral leakage creates spurious envelope spikes that corrupt gamma fits.

**Implementation:** Tukey window with `alpha=0.2` (10% cosine taper on each edge) applied to the detrended signal before the Hilbert transform. Tapered edges are then trimmed from the fit: `trim = max(1, int(n * 0.1))`, so the gamma regression only sees the flat-windowed interior.

**Verification:** Synthetic test with known gamma on a clean GNSS-IR-like signal. Estimation error: 1.5%.

**Location:** `snr_feature_extractor.py:452-463`

**Status:** Production. All gamma values extracted after this fix include tapering.

---

### 2. Robust Gamma Fitting

**Hypothesis:** OLS `np.polyfit` for the `log(envelope) = intercept + slope * sin^2(e)` fit is sensitive to near-zero envelope values at destructive interference nodes. These outliers can dominate the slope estimate.

**Implementation:** Replaced `np.polyfit` with `scipy.stats.siegelslopes` (Siegel repeated-medians regression). Breakdown point ~50% — half the data can be outliers before the fit degrades.

**Verification:** Qualitative — eliminates the occasional extreme gamma values that appeared with OLS on arcs with deep nulls.

**Location:** `snr_feature_extractor.py:478`

**Status:** Production.

---

### 3. Truncated Arc Guard

**Hypothesis:** gnssrefl's `ediff` parameter controls minimum elevation coverage, but some station configs (e.g., UMNQ with `ediff=200.0`) effectively disable this check. Truncated arcs that span <80% of [e1, e2] produce mechanically biased AF (shorter integration domain) and gamma (less lever arm for the sin^2 fit).

**Implementation:** `full_arc = ele_range >= 0.8 * expected_range` per Song 2022. When `full_arc=False`, AF, gamma, and gamma_r2 are set to NaN. The full_arc flag is propagated to the arc_table and aggregated as `frac_full_arc` at the sector-daily level.

**Impact:**
- ROSS 2024: 0% arcs affected (`ediff=2.0` already catches truncated arcs)
- UMNQ 2025: 14.7% affected (9,112 of 62,079 arcs)
- Partial arcs at UMNQ had 23% lower PkNoise than full arcs — they were adding noise, not signal, to daily medians

**Location:** `snr_feature_extractor.py:580-601`

**Status:** Production. Feature aggregator (`feature_aggregator.py:249-255`) computes `n_full_arcs` and `frac_full_arc` per sector-day.

---

### 4. gamma_r2 Quality Metric

**Hypothesis:** The Strandberg 2017 damping model assumes a smooth, homogeneous reflecting surface. When the surface is rough (e.g., Great Lakes ice) or when the arc quality is poor, the log-envelope vs sin^2(e) relationship breaks down. R^2 quantifies this.

**Implementation:** After the siegelslopes fit, compute standard R^2:
```
predicted = intercept + slope * sin^2(e)
ss_res = sum((log_env - predicted)^2)
ss_tot = sum((log_env - mean(log_env))^2)
gamma_r2 = 1 - ss_res / ss_tot
```
Clamped to [0, 1].

**Results:**
- ROSS 2024: median gamma_r2 = 0.000, max = 0.758. The smooth Strandberg model does not fit rough Great Lakes ice — consistent with the inverted gamma behavior at that station (gamma increases under ice instead of decreasing, because rough ice scatters rather than attenuates coherently).
- UMNQ 2025: not yet re-extracted with gamma_r2 (pending feature re-extraction with corrected baselines)

**Location:** `snr_feature_extractor.py:483-487`, aggregated as `gamma_r2_med` in `feature_aggregator.py:258-262`

**Status:** Production. Available but not yet used in classification. See "Open Questions" below for how it could weight gamma in the voting scheme.

---

### 5. AF Baseline Domain Fix

**The Bug:** `compute_af_baselines.py` precomputes per-PRN/frequency antenna gain pattern baselines from ice-free summer arcs (Song 2022 Eq. 19). These baselines are stored on a `sin(e)` grid (range ~0.087-0.423 for e1=5, e2=25). But `compute_area_factor()` in `snr_feature_extractor.py` was querying the baseline at `sin(e)/cf` coordinates, where `cf = wavelength/2`. For GPS L1, `cf = 0.0953m`, so `sin(e)/cf` ranges from ~0.916 to ~4.44 — **zero overlap** with the 0.087-0.423 baseline grid. Every interpolation query returned NaN (fill_value), the `oob_frac` check failed, and baseline subtraction was silently skipped. **The Song 2022 correction was a complete no-op on all prior runs.**

**Root Cause:** `power_info["sin_elev"]` returns `x/cf` (the rescaled coordinate used for CWT frequency analysis), not `x` (the raw sin(e) coordinate). The power curve has one value per input sample; each sample has both a sin(e) coordinate (`x`) and a sin(e)/cf coordinate (`x/cf`). The code confused the two domains.

**Two Complementary Fixes:**

1. **`compute_af_baselines.py:212-218`** — When accumulating baseline power curves, convert `power_info["sin_elev"]` from sin(e)/cf back to sin(e) before interpolating onto the common grid:
   ```python
   arc_sin_cf_grid = power_info["sin_elev"]   # sin(e)/cf domain
   cf_local = wavelength / 2
   arc_sin_grid = arc_sin_cf_grid * cf_local   # back to sin(e)
   ```

2. **`snr_feature_extractor.py:389-392`** — When querying the baseline during feature extraction, query at `x` (sin(e)), not `x/cf`:
   ```python
   bl_interp = interp1d(
       baseline_sin_grid, baseline_power_curve,
       bounds_error=False, fill_value=np.nan,
   )(x)    # was (x_cf) — wrong domain
   ```

**Verification:** Recomputed baselines for both stations and compared AF distributions.

Before fix (baselines all-zeros, no-op subtraction):
- ROSS 2024 af_baselines.npz: all power curves were zero
- UMNQ 2025 af_baselines.npz: all power curves were zero

After fix:
- ROSS 2024: 38 PRN/freq combos with non-zero baselines, mean power ~2765
- UMNQ 2025: 196 PRN/freq combos (GPS 62, GLONASS 35, Galileo 99)

AF distribution impact (UMNQ 2025, pooled station-level daily medians):
- **Median AF drops 77.5%** after baseline correction (antenna gain pattern removal)
- **Per-PRN AF variance reduced 27.3%** (the gain pattern was adding satellite-specific bias)

**Consequence:** All AF thresholds in v1 (`ice_classifier.py`) and v2/v3 (`classifier_v2.py`, `classifier_v3.py`) classifiers were calibrated against uncorrected AF. They need recalibration after feature re-extraction. The corrected AF should be a cleaner surface-state signal with less antenna contamination.

**Location:** `compute_af_baselines.py:212-218`, `snr_feature_extractor.py:386-392`

**Status:** Production. Old baselines backed up as `*_af_baselines_old.npz`. Feature re-extraction for UMNQ 2025 and ROSS 2024 with corrected baselines is pending.

---

### 6. B1: edot as Geometry-Informed QC Layer

**Hypothesis:** Low elevation rate (edot) arcs have satellites near apex, producing slowly oscillating SNR that is hard to distinguish from detrending residuals. These arcs should produce noisier gamma and AF estimates. edot could weight arc quality.

**Method:** Extract column 4 (edot, deg/s) from SNR files. Compute median |edot| per arc within the [e1, e2] window. Correlate with gamma (gamma_r2 not yet available in arc_table).

**Results (UMNQ 2025, 62,079 arcs):**
- edot range: 0.0043-0.0085 deg/s
- edot median: 0.0061 deg/s
- Pearson r(edot, gamma) = -0.152
- Gamma by edot quartile:
  - Q1 (edot <= 0.0057): gamma median=0.005, mean=0.005
  - Q2 (edot <= 0.0061): gamma median=0.005, mean=0.006
  - Q3 (edot <= 0.0071): gamma median=0.004, mean=0.004
  - Q4 (edot <= 0.0085): gamma median=0.004, mean=0.004

**Interpretation:** At 70.7N, the satellite geometry is heavily constrained — all satellites follow similar low-elevation tracks. The edot range spans only a factor of 2x (0.004-0.009), which is too narrow for meaningful QC discrimination. The 0.001 difference in median gamma across quartiles is negligible compared to the inter-arc variance of gamma itself.

**Conclusion:** Not useful at polar stations. The narrow edot distribution at high latitude makes this a poor QC discriminator. May have value at mid-latitude stations where the edot range is wider (untested).

**Diagnostic outputs:**
- `UMNQ_2025_edot.parquet` (62,079 rows)
- `UMNQ_2025_edot_diagnostic.png`
- `UMNQ_2025_edot_timeseries.png`

**Script:** `scripts/investigate_edot_qc.py`

**Status:** Parked.

---

### 7. B2: Cross-Frequency SNR Correlation

**Hypothesis:** For the same physical arc, two frequency bands observe identical geometry. After rescaling detrended signals from sin(e) to sin(e)/(wavelength/2), both should oscillate at the same frequency (= RH). A single-surface reflector (smooth ice) should produce highly correlated waveforms, while a complex surface (rough water with waves, or layered snow/ice) should produce less correlated waveforms. This correlation bypasses LSP estimation noise.

**Method:** For each multi-frequency physical arc (same satellite, same time, same rise/set), extract detrended SNR for each frequency pair. Rescale both to sin(e)/(wavelength/2) domain. Interpolate onto a common grid. Compute Pearson r.

**Frequency pairs tested:**
- GPS: L1-L2C (1578 arcs), L1-L5 (1627 arcs)
- GLONASS: L1-L2 (2684 arcs)
- Galileo: L1-L5a (1348), L1-E5b (1186), L1-E5 (1405), L1-E6 (1320)
- Total: 21,202 paired arcs

**Results (UMNQ 2025):**

| Pair | Arcs | Median r | Anomalous | Baseline | Regime Change |
|------|------|----------|-----------|----------|---------------|
| GPS L1-L2C | 1578 | 0.537 | 0.551 | 0.528 | 0.522 |
| GPS L1-L5 | 1627 | 0.541 | 0.570 | 0.524 | 0.528 |
| GLO L1-L2 | 2684 | 0.432 | 0.489 | 0.402 | 0.420 |
| GAL L1-L5a | 1348 | 0.588 | 0.638 | 0.558 | 0.569 |
| GAL L1-E5b | 1186 | -0.185 | -0.181 | -0.187 | -0.199 |
| GAL L1-E5 | 1405 | 0.073 | 0.054 | 0.082 | 0.098 |
| GAL L1-E6 | 1320 | 0.281 | 0.301 | 0.265 | 0.288 |

**Interpretation:**

The anomalous-vs-baseline median difference is +0.023 to +0.087 across GPS/GLONASS/Galileo L1-L5 pairs. **However:** per-arc distributions have IQR of ~0.3-0.7 for both classes. The 0.02-0.09 median shift is completely buried in per-arc noise. Time series shows no visible transition at freeze-up or breakup.

GAL L1-E5b is persistently negative (-0.185) in both states — likely a receiver or antenna artifact, not surface physics. GAL L1-E5 is near zero in both states.

The direction of the anomalous/baseline difference is consistent with the hypothesis (ice = simpler reflector = higher correlation), but the effect size is too small for per-arc or even per-day classification.

**Conclusion:** Statistically detectable but practically useless for classification. The per-arc variance drowns the 0.02-0.09 median difference. The infrastructure (frequency pairing within physical arcs) may be useful for other diagnostics.

**Diagnostic outputs:**
- `UMNQ_2025_xfreq_corr.parquet` (21,202 rows)
- `UMNQ_2025_xfreq_corr_timeseries.png`
- `UMNQ_2025_xfreq_corr_by_state.png`
- `UMNQ_2025_xfreq_corr_vs_CLR.png`
- `UMNQ_2025_xfreq_corr_vs_AF.png`
- `UMNQ_2025_xfreq_corr_vs_gamma.png`

**Script:** `scripts/investigate_xfreq_correlation.py`

**Status:** Dead for classification. Infrastructure preserved.

---

### 8. B3: Multi-Frequency Envelope Ratio for Snow-on-Ice

**Hypothesis:** A snow layer above ice causes frequency-dependent partial Fresnel reflections. The ratio of Hilbert envelopes between L1 and L2C should vary with elevation angle in a way that depends on snow depth and permittivity.

**Method:** Two-phase investigation.

**Phase 1 — Synthetic model:** Two-layer Fresnel model (air -> snow -> ice). Parameters: RH_ice=4.0m, epsilon_snow=1.8, epsilon_ice=3.2, elevation 5-25 deg. Fit linear slope to envelope ratio vs sin^2(e) for snow depths 0-20 cm.

**Synthetic results:**

| Snow (cm) | L1/L2C slope | L1/L5 slope |
|-----------|-------------|-------------|
| 0 | 0.010 | -0.006 |
| 2 | -0.013 | -0.033 |
| 5 | -0.072 | -0.104 |
| 10 | -0.242 | -0.277 |
| 15 | -0.619 | -0.674 |
| 20 | -1.164 | -1.286 |

Clear signal in theory: 5cm snow produces slope of -0.072, easily distinguishable from the 0cm slope of +0.010.

**Phase 2 — Real data:** GPS L1/L2C matched arcs at UMNQ 2025 (3,465 arcs). Computed Hilbert envelope for each band, interpolated to common elevation grid, computed envelope ratio and fit slope vs sin^2(e).

**Real data results:**
- Slope median: 4.87, std: 28.83
- By v3 state:
  - anomalous: median=4.42, std=22.14, n=1265
  - baseline: median=5.10, std=33.32, n=1794
  - regime_change: median=5.45, std=25.02, n=406

**Interpretation:** The theoretical 5cm snow signal (slope = -0.072) is ~400x smaller than the real data noise (std = 28.83). The noise is dominated by Hilbert envelope instability on real (non-synthetic) waveforms — multipath, surface roughness, antenna effects, and the inherent noisiness of the Hilbert transform on finite-length discrete signals with imperfect detrending. The anomalous/baseline difference in median slope is 0.68, which is negligible relative to the std of 28.

**Conclusion:** Dead as currently formulated. The Hilbert envelope is far too noisy for this approach. A better forward model incorporating surface roughness, volume scattering, and antenna effects might recover signal, but that's research-paper scope requiring in-situ calibration data.

**Implication for other Hilbert-based methods:** The envelope ratio noise floor of ~29 (std) relative to theoretical signals of ~0.07-1.2 sets a hard constraint. Any Hilbert-envelope-derived feature must produce signals >10x this noise floor to be useful. This effectively rules out subtle multi-layer effects and confirms that gamma (which uses the envelope slope in log space over a wider dynamic range) is operating near its noise floor for rough surfaces (consistent with the low gamma_r2 at ROSS).

**Diagnostic outputs:**
- `UMNQ_2025_envelope_ratio.parquet` (3,465 rows)
- `UMNQ_2025_envelope_ratio.png`
- `synthetic_envelopes.png`
- `synthetic_slope_vs_snow_depth.png`

**Script:** `scripts/investigate_envelope_ratio.py`

**Status:** Dead.

---

### 9. B4: Simultaneous Satellite Geometry Census

**Question:** Do we have enough same-elevation, different-azimuth satellite pairs at any given epoch to enable spatial heterogeneity detection (e.g., ice edge detection, identifying sectors with different surface states)?

**Method:** Read SNR files for 4 seasonal sample days (DOYs 15, 105, 195, 285). Bin epochs to 30-second windows. Find satellite pairs with |delta_elev| < 1 deg and |delta_az| > 30 deg. Count pairs, assess coverage.

**Results (UMNQ 2025):**
- 2,300-3,200 usable pairs per sample day
- ~200+ pairs per hour
- 917 unique satellite pair combinations across 4 sample days
- Consistent across all seasons (no seasonal dropoff)
- Elevation coverage fills the full [5, 15] deg window
- Azimuth separations range from 30 deg to 180 deg

**Interpretation:** The GNSS constellation provides abundant simultaneous multi-azimuth coverage at UMNQ. This is expected at 70.7N where all GPS/GLONASS/Galileo satellites pass through a narrow elevation band. At lower latitudes with wider elevation spread, the same-elevation constraint would reduce pair counts, but there would still likely be enough for epoch-level analysis.

**Conclusion:** Geometrically very feasible at UMNQ. This enables future work on:
1. Epoch-level spatial heterogeneity maps
2. Ice edge detection (one azimuth ice, another water)
3. Validation of sector-based classification by checking whether adjacent sectors agree

**Open concern:** Transferability to GPS-only or sparse stations. UMNQ benefits from GPS + GLONASS + Galileo.

**Diagnostic outputs:**
- `UMNQ_2025_geometry_census.parquet` (11,016 rows)
- `UMNQ_2025_geometry_census.png`

**Script:** `scripts/investigate_geometry_census.py`

**Status:** Feasibility confirmed. Needs a concrete use case and analysis method before building.

---

### 10. gnssrefl Prefilter Behavior

**Context:** gnssrefl applies 6 sequential arc QC checks before an arc enters the output files:
1. **ediff** — minimum elevation coverage (arc must span at least ediff degrees)
2. **tooclose** — RH not at frequency boundary (avoids aliased fits)
3. **noise** — positive noise floor (rejects arcs with no signal)
4. **amp** — minimum amplitude (reqAmp threshold)
5. **pk2noise** — minimum peak-to-noise ratio (PkNoise threshold)
6. **delT** — maximum arc duration (avoids wrapping arcs)

**Station-specific behavior:**
- ROSS: `ediff=2.0` (strict). Result: 100% of arcs that reach the arc_table are full arcs. The truncated arc guard (item 3) is redundant here.
- UMNQ: `ediff=200.0` (effectively disabled). Result: 14.7% of arcs are partial. The truncated arc guard is essential here as a safety net.

**Implication:** The truncated arc guard must remain in production code even though some stations don't need it. Station configs with permissive ediff values will produce partial arcs that bias AF and gamma if not guarded.

---

## Part 2: Current State and Open Questions

### What's Improved

**AF is now a real signal, not antenna noise.** The baseline correction bug meant all prior AF values were uncorrected antenna gain patterns with per-satellite bias. After the fix, AF should reflect surface roughness/scattering as intended by Song 2022. The 77.5% reduction in median AF and 27.3% reduction in per-PRN variance confirm the gain pattern was a dominant component. The corrected AF is expected to be a cleaner discriminator, but all existing thresholds are invalid and need recalibration.

**Gamma is more robust.** Three independent improvements: edge tapering removes Hilbert boundary artifacts, siegelslopes resists envelope nulls, and gamma_r2 quantifies when the Strandberg model doesn't fit. Together these mean gamma values are more trustworthy, and we now know *when* they're not trustworthy.

**Partial arcs are excluded from AF and gamma.** The 14.7% partial-arc rate at UMNQ was injecting noise into daily medians. These arcs now contribute to LSP-derived features (CLR, PR, RH, phase) but not to envelope-derived features (AF, gamma).

### What's Been Ruled Out

**Hilbert-envelope methods for subtle multi-layer effects are dead.** The envelope ratio investigation (B3) established a noise floor of ~29 (std of slope) vs theoretical signals of 0.07-1.2. Any future Hilbert-envelope investigation must produce signals >10x this floor. This rules out:
- Snow depth estimation from envelope ratios
- Subtle permittivity changes from envelope shape
- Any single-arc envelope-derived feature for layer discrimination

**edot as QC at polar stations is dead.** The narrow edot range (0.004-0.009 deg/s) at 70.7N makes it uninformative. Might work at mid-latitudes — not tested, not prioritized.

**Cross-frequency correlation as a classification feature is dead.** The 0.02-0.09 median shift between ice and water is buried in per-arc variance (IQR ~0.3-0.7). No time series signature at freeze-up/breakup.

### Current Feature Set Health

#### Per-Arc Features (Layer 1: arc_table)

**CLR (Clarity Ratio) — P1/mean(other peaks)**
- Physics: measures spectral dominance. A single reflector at stable height produces a sharp LSP peak (high CLR). Water waves, multiple layers, or vegetation create competing peaks (low CLR).
- Extraction: robust. Pure frequency-domain, no Hilbert transform, no domain confusion.
- Discrimination: strongest single ice/water discriminator in v1 voting (weight 2.0). High CLR = ice.
- Failure modes: RFI or multipath from nearby structures can create spurious peaks that reduce CLR even on smooth ice. CLR also degrades when two reflectors are at similar heights (e.g., water very close to ice surface level).

**PR (Peak Ratio) — P1/P2**
- Physics: similar to CLR but only compares top two peaks. Less sensitive to many small peaks, more sensitive to a single competing reflector.
- Extraction: robust. Same as CLR.
- Discrimination: moderate (weight 1.0 in v1). Correlated with CLR but not redundant — CLR uses mean of all other peaks, PR uses only the second.
- Failure modes: same as CLR.

**AF (Area Factor) — NOW WITH BASELINE CORRECTION**
- Physics: integral of CWT power at dominant RH over sin(e)/cf domain. With baseline correction, this measures excess power relative to the satellite-specific antenna gain pattern. High AF = strong surface scattering/roughness.
- Extraction: **improved but needs re-validation.** The domain fix is correct but AF has not been re-extracted for any station. Current arc_table values still use the uncorrected AF. After re-extraction, all thresholds (v1 percentile-based, v2 Mahalanobis) need recalibration.
- Discrimination: **unknown after correction.** Prior results showed high AF under ice, but this was antenna gain pattern, not surface physics. The corrected AF may discriminate differently. Expected: still high under rough ice (surface scattering), but with reduced satellite-to-satellite variance.
- Failure modes: baseline requires sufficient ice-free summer arcs (>=50 per PRN/freq). Stations without a clear ice-free season can't compute baselines. Only computed for full arcs.

**gamma (Damping Parameter)**
- Physics: Strandberg 2017 model — exponential decay of interference fringe amplitude with elevation. Smooth specular surfaces (ice) should have low gamma; rough diffuse surfaces (water) should have high gamma. Reality is more complex: rough Great Lakes ice (ROSS) shows *inverted* gamma behavior (high gamma under ice).
- Extraction: improved (tapering + robust fit). Still fundamentally limited by Hilbert transform noise on real waveforms.
- Discrimination: station-dependent. Works as expected at some stations, inverted at ROSS. Weight 1.0 in v1 voting with `high_is_ice=False`.
- Failure modes: gamma_r2 < ~0.1 means the Strandberg model doesn't fit — gamma value is unreliable. ROSS has median gamma_r2 = 0.000. Only computed for full arcs.

**gamma_r2 (Damping Fit Quality)**
- Physics: R^2 of the log-envelope linear fit. Low R^2 means the surface doesn't follow the smooth Strandberg model.
- Extraction: robust (derived from the same fit as gamma).
- Discrimination: not directly a surface-state feature, but a quality indicator. Could be used to weight gamma in classification.
- Failure modes: none known. Low R^2 is informative, not a failure.

**phase (Reflection Phase)**
- Physics: Fresnel reflection coefficient phase depends on surface permittivity. Water (epsilon ~80) vs ice (epsilon ~3.2) have different reflection phases. Phase should shift at freeze-up/breakup.
- Extraction: robust. Matched-filter (inner product) approach at known RH.
- Discrimination: theoretical basis is strong, but empirical results have been weak. Weight 0.5 in v1 voting. Circular statistics make aggregation and thresholding more complex than linear features.
- Failure modes: phase wrapping ambiguity. Multi-layer surfaces produce phase that depends on layer thicknesses, not just permittivity. Phase is sensitive to antenna phase center offset errors.

**MS (Mean SNR in dB) and VS (Variance of Detrended SNR)**
- Physics: MS reflects signal strength (antenna gain, satellite power, atmospheric attenuation). VS reflects oscillation power (stronger multipath = higher VS).
- Extraction: trivial and robust.
- Discrimination: weak individually. Not used in v1 voting. Aggregated as `ms_med` and `vs_med` in daily features.
- Failure modes: none.

**SP (Peak Amplitude) — LSP peak height**
- Physics: proportional to multipath interference amplitude. Related to VS but in frequency domain.
- Extraction: robust.
- Discrimination: weak individually. Related to Amp (gnssrefl's amplitude).

**RH (Reflector Height)**
- Physics: the primary observable — distance from antenna to reflecting surface.
- Extraction: robust (LSP peak finding).
- Discrimination: not directly, but RH statistics (rh_std, rh_range, delta_rh_mean, interfreq_spread) are key classification features. Ice produces stable RH; water with waves produces variable RH.

**full_arc (Boolean)**
- Physics: elevation coverage flag (>=80% of [e1, e2]).
- Extraction: trivial.
- Discrimination: not a surface-state feature. Quality gating for AF and gamma.

#### Daily Aggregate Features (Layer 2: daily_features)

**amp_mean, amp_std, amp_cv** — Amplitude statistics from gnssrefl output.
- amp_mean is the highest-weighted base indicator (2.0) in v1 voting, `high_is_ice=True`. Rough ice surfaces produce higher amplitude multipath.
- amp_cv (coefficient of variation) has weight 1.5, `high_is_ice=False` (ice is more consistent than water).
- Well-established, robust.

**rh_std, rh_range, rh_mean, rh_median** — Reflector height statistics.
- rh_std has weight 1.0 in v1 voting with inverted polarity (`high_is_ice=False` in base indicators with thresholds labeled as ice=30th percentile, water=70th percentile). This is confusing — in practice, ice typically has *lower* rh_std than wave-dominated water. The thresholding is set up so that low rh_std → ice.
- Robust, physically well-understood.

**p2n_mean, p2n_std** — PkNoise (peak-to-noise ratio) statistics.
- Highest Cohen's d (0.70) among Mahalanobis features in v2.
- Not in v1 voting (added later). Should be considered for higher weight.

**clr_med, af_med, pr_med, gamma_med, ms_med, vs_med** — Medians of per-arc SNR features.
- These are the sector-daily aggregates that feed into both v1 voting and v2/v3 Mahalanobis distance.
- af_med needs re-extraction with corrected baselines.

**clr_z, af_z, pr_z, gamma_z** — Per-satellite z-score normalized features.
- Removes satellite-specific bias (antenna gain pattern, orbit geometry).
- Used in v1 voting. The z-score normalization should make AF z-scores more meaningful after baseline correction (the baseline removes per-PRN gain; z-scoring removes per-PRN residual variance).

**delta_rh_mean, delta_rh_std** — Matched-arc cross-frequency RH divergence.
- Measures whether L1 and L2C see the same reflector height. Ice (single surface) → low delta_rh. Water with waves or layered media → higher delta_rh.
- Weight 0.5 in v1 voting.
- Also used in v3 classifier for ice_layered sub-state (delta_rh_mean >= 0.3m).

**interfreq_spread** — Range of per-band RH medians (station-level daily).
- Similar to delta_rh_mean but at station level. v1 voting thresholds: ice at 0.03m, water at 0.08m.
- Weight 0.3x sum of sector weights.

**phase_circ_mean, phase_circ_std** — Circular phase statistics.
- Aggregated using scipy circular mean/std. Weight 0.5 in v1 voting.

**diff_phase_L1_L2, diff_phase_std** — L1 minus L2C differential phase from matched arcs.
- Permittivity-sensitive: different materials produce different differential phase.
- Not yet in v1 or v2 voting. Potentially valuable but needs validation.

**frac_full_arc** — Fraction of arcs that are full (>=80% elevation coverage).
- Quality indicator for sector-days. Low frac_full_arc means AF and gamma medians are based on fewer arcs.
- Not in voting, but could weight AF and gamma confidence.

**gamma_r2_med** — Median R^2 of gamma fits in sector-day.
- Quality indicator for gamma. Not in voting.

### What Combinations Might Work?

**1. gamma_r2-weighted gamma in the Mahalanobis model.**
The v2 Mahalanobis distance uses gamma_med as one of 7 features. At stations like ROSS where gamma_r2 is near zero, gamma_med is essentially random noise — it adds dimensionality without discrimination. Two approaches:
- **Gating:** only include gamma_med in the Mahalanobis computation when gamma_r2_med > some threshold (e.g., 0.05). When below threshold, reduce the feature set to 6 dimensions for that sector-day. This requires the Mahalanobis model to handle variable-dimension input (non-trivial).
- **Weighting:** multiply gamma_z by gamma_r2_med before feeding into Mahalanobis. When R^2 is near zero, gamma is effectively zeroed out. Simpler to implement but changes the feature distribution.

**2. frac_full_arc-weighted AF and gamma confidence.**
When frac_full_arc < 0.5 for a sector-day, af_med and gamma_med are based on fewer than half the observed arcs. The v1 voting could scale AF and gamma weights by frac_full_arc:
```
effective_weight = base_weight * frac_full_arc
```
This gracefully degrades their influence when data quality is poor.

**3. Corrected AF + CLR as a joint ice signature.**
Both CLR and AF should be high under smooth ice. With corrected AF (gain pattern removed), the correlation between CLR and AF should change — they may become more independent, which would make their joint discriminating power greater than the sum of individual features. This needs empirical verification after re-extraction.

**4. Multi-feature ice signature: high CLR + high AF + low rh_std.**
The v1 voting already combines these, but with percentile-based thresholds that are calibrated per-station. The question is whether there's a *universal* multi-feature boundary that works across stations. The v2 Mahalanobis approach implicitly does this (it finds the multivariate boundary of the water-state distribution), but it requires a training period. A simpler approach: require 2-of-3 features to exceed their respective thresholds for an "ice" call, rather than the current weighted average which can produce ice calls from a single extreme feature.

**5. p2n_mean as a higher-weight feature.**
Cohen's d = 0.70 makes PkNoise the strongest single discriminator in the v2 feature set, yet it's not in v1 voting at all. Adding it with weight >= 1.5 should improve v1.

---

## Part 3: Post-Re-extraction Analysis (2026-04-07)

### Re-extraction gate (six criteria, all must pass)

Before declaring re-extraction complete and proceeding to downstream analysis:

1. **AF baselines non-zero** — `af_baselines.npz` `baselines` array has all non-zero power curves (N×50). Confirms the fix applied; all-zero baselines = silent no-op repeat of original bug.
2. **UMNQ AF median drop ≥ 60%** — arc-level AF median ≤ 40% of backup. (Doc reported 77.5% at daily-median level; per-arc threshold relaxed to 60%.)
3. **ROSS AF changed > 10%** — any meaningful shift confirms baseline subtraction applied.
4. **gamma_r2 non-null for all full arcs** — `arc_table[full_arc==True]['gamma_r2'].isna().sum() == 0` at both stations.
5. **frac_full_arc in expected range** — daily_features `frac_full_arc` mean: UMNQ in [0.80, 0.90], ROSS in [0.95, 1.00].
6. **Non-AF feature preservation + PRN distribution** — CLR and RH medians within 5% of backup; no (sat, freq_group) combination disappeared; no combination lost > 25% of arcs; unique satellite count within 2.

Verification script: `scripts/verify_reextraction.py`.

### Re-extraction outcome

Both UMNQ 2025 and ROSS 2024 re-extracted with corrected AF baselines. All 6 criteria passed:

| Criterion | UMNQ result | ROSS result |
|-----------|-------------|-------------|
| Baselines non-zero | 196/196 ✓ | 38/38 ✓ |
| AF median change | 68.1% drop ✓ | 80.2% drop ✓ |
| gamma_r2 non-null (full arcs) | 0.0% null ✓ | 0.0% null ✓ |
| gamma_r2 median (full arcs) | 0.123 | 0.000 (open-water dominated) |
| frac_full_arc mean | 0.878 ✓ | 1.000 ✓ |
| CLR drift | 0.00% ✓ | 0.00% ✓ |
| Satellites preserved | 76/76 ✓ | 25/25 ✓ |

Bug fixed during re-extraction: `snr_feature_extractor.py` merge logic only wrote *new* columns to arc_table, silently preserving stale feature values on any re-extraction run. Fixed at `snr_feature_extractor.py:897–910` — fresh feature values now always overwrite existing columns.

---

### C1: gamma_r2 at UMNQ — seasonal confound confirmed

**Question:** Is gamma_r2 tracking surface state independently at UMNQ, or is it a calendar proxy?

**Results (UMNQ 2025, daily medians, n=281):**
- r(gamma_r2_med, DOY) = **−0.650** (p=4e-35): strong seasonal decline
- r(gamma_r2_detrended, is_anomalous) = 0.110 (p=0.07): DOY detrending does not improve the state correlation — the seasonal and state effects are nearly orthogonal
- Overall Cohen d (anomalous vs baseline) = **0.247**: too weak for classification use
- Within-season direction reversal:

| Window | anomalous mean | baseline mean | d |
|--------|----------------|--------------|---|
| May-Jun | 0.132 | 0.124 | +0.28 |
| Sep-Oct | 0.080 | 0.111 | **−1.08** |
| Nov-Dec | 0.079 | 0.113 | **−1.25** |

Spring: anomalous (ice) slightly > baseline (water) — consistent with ROSS pattern. Autumn: strongly inverted — baseline (open water) has higher gamma_r2 than anomalous. Thin forming ice in autumn is apparently rougher (lower gamma_r2) than autumn open water.

**Conclusion:** gamma_r2 does NOT transfer as a classifier feature from ROSS to UMNQ. The ROSS signal (open water → near-zero gamma_r2, ice → moderate gamma_r2) reflects Great Lakes surface physics, not a universal pattern. At UMNQ, the relationship reverses seasonally, making gamma_r2 actively harmful in a season-agnostic classifier.

**Decision:** Keep gamma_r2 as a quality weight for gamma only at both stations. Do not add as independent Mahalanobis feature at UMNQ.

**Future note:** The autumn d = −1.25 is a strong signal in the wrong direction. A season-aware classifier (different feature weights per DOY window) could use gamma_r2 with a sign flip in DOY ~241–366. Not a current priority.

---

### C2: af_med vs af_z after baseline correction

**Question:** Does z-score normalization (af_z) still add discriminative value after the baseline correction removes per-PRN gain pattern?

**Results:**

| Feature | ROSS Cohen d | UMNQ Cohen d |
|---------|-------------|-------------|
| af_med | **+2.881** | **+1.251** |
| af_z   | +2.718 | +1.221 |
| clr_med | +1.303 | **+0.057** |
| clr_z  | +1.379 | +0.098 |

r(af_med, af_z): ROSS = 0.987, UMNQ = 0.991.

af_med strictly outperforms af_z at both stations, and they are r > 0.98 correlated. Including both in the Mahalanobis model creates near-perfect collinearity. Per-PRN AF variance reduction: 18.5% at ROSS (GPS-only), 38% at UMNQ (multi-constellation). The baseline correction achieves physically what z-scoring was statistically compensating for.

**Decision:** Drop af_z from the Mahalanobis feature set. Use af_med as the primary AF feature.

---

### C3: CLR is blind at UMNQ

**The finding:** Cohen d(clr_med) = **0.057** at UMNQ — essentially zero. For reference, ROSS has d = 1.303.

**Geometric explanation:** At 70.7°N all GPS+GLONASS+Galileo satellites are constrained to a narrow low-elevation band. Every arc is geometrically similar, and the LSP is always dominated by a single RH peak regardless of surface state. Spectral clarity (CLR = P1/mean other peaks) is structurally high for both ice and water — there's rarely a competing reflector at a different height.

**Variance:** clr_med var = 2.696 at UMNQ (not near-zero, cv = 18%). This variance exists but is uncorrelated with state — it's noise from geometry variation within the elevation band, not surface physics. No singularity risk from variance collapse, but CLR adds a noise dimension to the Mahalanobis model that dilutes the AF signal.

**Consequence for existing results:** The v1 classifier assigns CLR a weight of 2.0 — the highest single feature weight. At UMNQ this feature has been a noise generator, not a discriminator, for all prior classification runs. **All v1 classifier output for UMNQ prior to this correction should be treated as unreliable.** The v2/v3 Mahalanobis model is less affected (it auto-fits the water-state distribution and CLR's noise variance would be captured in the baseline covariance), but recalibration is needed.

**Decision:** Drop CLR entirely from the UMNQ Mahalanobis model and from v1 voting at UMNQ. Do not simply deprioritize — the near-zero d makes it harmful, not neutral.

**Structural shift:** r(af_med, clr_med) = 0.60 at UMNQ. Corrected AF is now doing the work CLR was supposed to do, with d = 1.25 vs CLR's d = 0.057.

---

### C4: B4 — Sector consistency check (internal classifier validation)

**Question:** Does the classifier produce consistent labels across simultaneous arcs at different azimuths? High inter-sector disagreement during stable states = classifier noise. High disagreement during transitions = real spatial heterogeneity.

**Method:** For each of the 281 days at UMNQ 2025, compute a per-sector ice score from corrected af_med (z-scored against the Jul–Aug ice-free summer baseline). Report the inter-sector standard deviation of ice scores per day. Link to v3 state labels.

**Results (daily inter-sector std of af_med ice score):**

| v3_state | median std | median range | n days |
|----------|-----------|-------------|--------|
| baseline | **0.963** | 3.39 | 152 |
| regime_change | 1.306 | 4.52 | 37 |
| anomalous | 2.000 | 7.49 | 92 |

Baseline std = 0.963 is the noise floor — the classifier's natural sector-to-sector variability during confirmed open-water conditions.

**Spring/autumn asymmetry:** The top 10% disagreement days (std ≥ 4.08) are concentrated entirely in DOY 85–117 (late March–late April). All 29 are labeled anomalous or regime_change. No autumn anomalous days (DOY 300–366) appear in the top 10%.

- Spring anomalous (DOY 85–117): inter-sector std up to 9.6, range up to 35.9. Multiple sectors showing ice while others show open water simultaneously.
- Autumn anomalous (DOY ~300–366): inter-sector std comparable to regime_change (~1.3). Sectors agree.

**Physical interpretation:** Spring breakup at Upernavik (70.7°N) produces patchy ice — leads open, sectors sample different surface conditions simultaneously. This is real spatial heterogeneity, not classifier noise. Autumn freeze-up is spatially uniform (new thin ice forms across the fjord together), so sectors agree even though the state is anomalous.

**Geometry census validation (4 sample days):** Direct satellite-pair score comparisons confirm the pattern.

| DOY | v3_state | median |Δscore| |
|-----|----------|------------------|
| 120 | anomalous | **2.157** |
| 190 | baseline | 0.779 |
| 260 | baseline | 0.998 |
| 330 | baseline | 0.478 |

**Conclusions:**
1. The classifier is internally consistent during stable states — baseline std 0.963 is the noise floor.
2. Spring breakup (DOY 85–117) spatial heterogeneity is physically real, not a calibration artifact.
3. The v3 single-daily-label for DOY 85–117 is averaging over real sector-level mixed states; sector-level classification during that window would be more informative than station-level.
4. **Utility as version tracker:** Re-run this check after Mahalanobis recalibration. If baseline std drops below 0.963, recalibration reduced noise. If the spring cluster shifts in DOY, it would indicate a calibration artifact.

**Outputs:** `results_annual/UMNQ/diagnostics/UMNQ_2025_b4_sector_consistency.parquet`, `_b4_consistency.png`, `_b4_census_pairs.png`
**Script:** `scripts/b4_sector_consistency.py`

---

### Concrete Next Steps

Ranked by expected impact on classification quality:

**1. Re-extract features for UMNQ 2025 and ROSS 2024 with corrected AF baselines.** (High impact, straightforward)
All downstream analysis depends on this. Run `snr_feature_extractor.py` to produce new arc_tables, then `feature_aggregator.py` for daily_features. This is a prerequisite for everything else.

**2. Recalibrate classification thresholds against corrected AF.** (High impact, depends on #1)
After re-extraction, re-run v2/v3 classifiers. The v2 Mahalanobis model will auto-recalibrate (it fits the water-state distribution from data). The v1 percentile thresholds will also auto-recalibrate. Compare new classification timelines to GLERL ground truth to assess whether the corrected AF improves discrimination.

**3. Add p2n_mean to v1 voting.** (Medium impact, quick)
PkNoise has Cohen's d = 0.70 (highest in v2 feature set) but is absent from v1 voting. Add it with weight 1.5-2.0 and verify against GLERL ice dates at ROSS.

**4. Implement gamma_r2 gating or weighting.** (Medium impact, depends on #1)
After re-extraction produces gamma_r2 for UMNQ, assess whether gamma_r2 predicts gamma utility. If gamma_r2 < 0.05 correlates with gamma being non-discriminating, implement gating in v1 or weighting in v2. Test at both ROSS (where gamma is known to be unreliable) and UMNQ.

**5. Validate frac_full_arc as AF/gamma confidence weight.** (Low-medium impact, depends on #1)
After re-extraction, compare AF discrimination (ice vs water separation in daily medians) between sector-days with frac_full_arc > 0.8 vs < 0.5. If high frac_full_arc sector-days have cleaner AF distributions, implement confidence weighting in v1 voting.

---

## Part 4: GLBX Station Investigation (2026-04-07)

### C5: GLBX winter amplitude drop — frequency-mixing artifact, not surface contamination

**Question:** Is the January–April amplitude ramp (amp_mean: 21 → 35) at GLBX Bartlett Cove an atmospheric/antenna artifact corrupting the signal, or real surface physics?

**Method:** Three-part investigation: (1) QC metric seasonality, (2) feature coherence on winter anomalous days, (3) September cluster characterization.

---

#### Task 1: QC metric seasonality

Monthly medians of key QC and surface features (pooled daily features, azimuth_bin = -1):

| Month | amp_mean | ms_med | vs_med | p2n_mean | gamma_med | frac_full_arc |
|-------|----------|--------|--------|----------|-----------|---------------|
| Jan | 21.1 | 30.1 | 334.7 | 2.192 | 0.011 | 0.955 |
| Feb | 26.5 | 31.5 | 462.8 | 2.181 | 0.010 | 0.961 |
| Mar | 28.0 | 33.8 | 585.7 | 2.174 | 0.011 | 0.968 |
| Apr | 29.7 | 36.9 | 618.8 | 2.219 | 0.015 | 0.985 |
| May | 35.0 | 38.0 | 704.3 | 2.287 | 0.018 | 0.989 |
| Jun–Nov | 33–35 | 37–38 | 700–810 | 2.28–2.31 | 0.016–0.019 | 0.98–0.99 |

Pearson correlations (amp_mean vs QC metrics, L2 daily features):

| Metric | r_all | r_winter (DOY 1–120, 300–366) | r_summer (DOY 150–270) |
|--------|-------|-------------------------------|------------------------|
| ms_med | +0.902 | +0.828 | +0.561 |
| vs_med | +0.848 | +0.810 | +0.571 |
| af_med | +0.540 | +0.711 | +0.521 |
| p2n_mean | +0.696 | +0.461 | +0.071 |
| gamma_med | +0.426 | +0.255 | +0.191 |
| frac_full_arc | +0.393 | +0.255 | −0.136 |

**Key finding — frequency-resolved arc table analysis:**

The winter amplitude ramp is caused almost entirely by **L1 amplitude suppression**, not by a uniform surface signal change:

- **L5 amplitude is actually higher in winter** (Jan median: 49.0) than summer (Jun median: 40.5, ratio 1.21×). If the surface were simply degrading signal quality, L5 would drop too. It does not. L5 is the cleaner indicator of surface reflectivity, and it correctly shows ice is more reflective than water.
- **L1 amplitude collapses**: Jan median 8.6 vs Jun median 29.5. L1 input SNR (`ms_med`) simultaneously drops ~15 dBHz in January. This is input-level signal quality loss — before spectral decomposition.
- **L1 spectral extraction efficiency also degrades**: The L1 Amp/MS ratio collapses from ~0.80 (summer) to ~0.41 (winter). Two-stage degradation: lower input SNR AND worse extraction efficiency from those arcs.
- **p2n_mean (PkNoise) is flat year-round** (2.17–2.31). The receiver noise floor does not change. The issue is in the reflectometry signal, not the receiver.
- **n_arcs drops from ~155–186 (summer) to ~21–37 (Jan–Feb)**. The pooled `amp_mean` is dominated by the L1 frequency group by arc count (56% of winter arcs are L1). L1 suppression drags down the mixed-frequency daily mean.

**Verdict: The winter anomalous classification is real (Bartlett Cove has seasonal sea ice), but the classifier is partly doing so for the wrong reasons.** `amp_mean` moves in the wrong direction during winter anomalous periods because L1 amplitude suppression dominates the frequency-pooled mean over the correctly elevated L5 signal. The classifier detects "winter is different" via the Mahalanobis distance, but the feature direction for `amp_mean` is opposite what ice physics predicts.

**Action item:** Compute frequency-specific amplitude features (`amp_L5_mean`, `amp_L1_mean`) separately in the feature aggregator. The L5 amplitude is the clean ice signal; the mixed `amp_mean` is contaminated by L1 suppression that dominates by arc count.

---

#### Task 2: Feature coherence on winter anomalous days

Winter (DOY 1–120): 105 anomalous days, 7 baseline days (late April only), 6 regime_change.

Feature direction comparison (anomalous vs late-April baseline):

| Feature | Baseline med | Anomalous med | Direction | Ice expected | Assessment |
|---------|-------------|---------------|-----------|-------------|------------|
| amp_mean | 34.75 | 26.45 | DOWN | UP | Wrong direction |
| vs_med | 709.7 | 505.6 | DOWN | UP | Wrong direction |
| af_med | 4,309,731 | 3,465,451 | DOWN | UP | Wrong direction |
| gamma_med | 0.0164 | 0.0109 | DOWN | UP | Wrong direction |
| clr_med | 4.48 | 4.20 | DOWN | UP | Wrong direction |
| rh_std | 1.59 | 1.39 | DOWN | DOWN | Correct |
| rh_range | 5.34 | 4.51 | DOWN | DOWN | Correct |
| rh_mean | 8.10 | 8.07 | ~stable | stable | Correct |
| ms_med | 38.01 | 31.60 | DOWN | stable | Input quality degraded |
| n_arcs | 167 | 41 | DOWN (−75%) | stable | Severe data sparsity |

Five of six primary reflectometry features (amp, vs, af, gamma, clr) move in the direction opposite to what an ice surface would produce. Only rh_std and rh_range (lower on anomalous days) are consistent with an ice-smoothed surface. This is entirely explained by the L1 suppression finding: the frequency-pooled medians are L1-dominated by arc count, so L1 degradation inverts the apparent feature direction.

**rh_mean is stable** (8.07 vs 8.10, Δ = 0.03 m): no systematic reflector height shift. The water level is not changing.

**n_arcs drops 75%** on winter anomalous days (41 vs 167). This data sparsity severely undersamples the azimuth/elevation space and dominates noise in all feature aggregates.

---

#### Task 3: September cluster characterization

**DOYs:** 261–269 (September 17–25, 2024). Bracketed by regime_change transitions (DOY 261–263 onset, DOY 270–272 return).

Per-day feature values:

| DOY | Date | State | rh_std | p2n_mean | ms_med | amp_mean | n_arcs |
|-----|------|-------|--------|----------|--------|----------|--------|
| 258 | Sep 14 | baseline | 1.167 | 2.274 | 38.56 | 34.63 | 168 |
| 261 | Sep 17 | regime_change | 1.999 | **2.380** | 37.69 | 34.32 | 181 |
| 262 | Sep 18 | regime_change | 1.084 | 2.293 | 38.44 | 34.86 | 53 |
| 263 | Sep 19 | regime_change | 2.276 | **2.372** | 38.26 | 34.44 | 209 |
| 264 | Sep 20 | anomalous | 2.183 | **2.384** | 37.95 | 34.21 | 196 |
| 265 | Sep 21 | anomalous | 1.946 | **2.382** | 37.92 | 35.40 | 184 |
| 266 | Sep 22 | anomalous | 1.741 | 2.296 | 38.31 | 35.94 | 187 |
| 267 | Sep 23 | anomalous | 1.662 | 2.314 | 38.40 | 34.87 | 149 |
| 268 | Sep 24 | anomalous | 1.326 | 2.284 | 38.38 | 33.16 | 119 |
| 269 | Sep 25 | anomalous | 1.182 | 2.273 | 38.30 | 35.83 | 167 |
| 270 | Sep 26 | regime_change | 1.016 | 2.257 | 37.66 | 35.69 | 168 |
| 273 | Sep 29 | baseline | 1.219 | 2.275 | 37.95 | 33.76 | 176 |

Deviations from September baseline (amp_mean 34.37±0.99, rh_std 1.32±0.27, p2n_mean 2.283±0.014):

- **p2n_mean** elevated **+6–7σ** on DOY 261, 263, 264, 265 — the strongest anomaly signal. Returns to normal by DOY 266.
- **rh_std** elevated +1.3–3.6σ on DOY 261–265, decaying back to baseline by DOY 268–269. Genuine within-day RH scatter increase.
- **amp_mean** within ±1.6σ throughout — no consistent direction.
- **ms_med** flat throughout (37.7–38.6 dBHz). Input signal quality unchanged. This confirms the September cluster is a real surface event, not signal degradation.

**Structure:** Spike, not step change. Sharp onset (DOY 261), elevated rh_std and p2n for ~5 days, smooth return by DOY 269–270. Total event duration: ~9 days including transitions. Consistent with a brief sea ice formation and melt event at Bartlett Cove in late September.

**Note for ERDDAP cross-reference:** DOY 261–269 = September 17–25, 2024. The ERDDAP sea surface temperature or ice extent data for Bartlett Cove should be checked against these dates to confirm the event. This is a you-and-me task once the internal signal is characterized.

---

### C6: GLBX station-specific mahal_features configuration

**Config change:** Added `mahal_features` block to GLBX entry in `stations_config.json`. Feature polarity is inverted relative to UMNQ/ROSS because the winter anomalous period shows all primary features (amp, vs, af, clr) pointing DOWN rather than UP on anomalous days — a direct consequence of the L1 suppression artifact documented in C5.

GLBX `mahal_features` polarity assignments:
- `amp_mean`, `clr_med`, `af_med`, `pr_med`: `"winter_low"` (anomalous = lower values due to L1 suppression)
- `rh_std`, `gamma_med`, `p2n_mean`: `"winter_high"` (anomalous = higher, consistent with surface physics)

**Validated (C8):** Three-station frequency-split validation (C8) confirmed that the L1 suppression mechanism is GLBX-specific. UMNQ and NIAQ are both Clean — both bands rise together in winter, MS is flat, pool correlations r > 0.98 to both bands. This validates the GLBX `winter_low` polarity assignments for amp/clr/af/pr: the inversion is real at GLBX and absent elsewhere, not a classification framework error.

**Remaining action:** Once GLBX features are re-aggregated with frequency-split columns, replace `amp_mean` with `amp_L5_mean` in the GLBX `mahal_features` config. L5 amplitude is the physically correct ice signal (higher in winter) and will allow the `winter_high` polarity consistent with UMNQ and ROSS. `amp_mean` and `amp_L1_mean` should be removed from the GLBX Mahalanobis feature set at that point.

---

### C7: Frequency-split amplitude features

**Implementation:** Added `amp_L1_mean`, `amp_L2C_mean`, `amp_L5_mean`, `amp_L2_mean` (GLONASS), `amp_L5a_mean` (Galileo) per-band amplitude aggregates to `feature_aggregator.py`. These are computed in `_aggregate_sector()` from arc_table rows grouped by `freq_group`, alongside the existing per-band RH medians. Only frequency groups with ≥2 arcs on a given sector-day are emitted.

**Rationale:** At GLBX, L1 amplitude collapses 67% in winter (input SNR drops ~15 dBHz) while L5 amplitude is *higher* in winter than summer (49 vs 41 median, ice more reflective). The mixed `amp_mean` is dominated by L1 arcs by count, inverting the apparent feature direction. Separating by band exposes the clean L5 signal.

**Three-station validation (C8):**

| Station | L1 winter Δ | L5 winter Δ | L1 MS Δ | amp_L1 Cohen d | amp_L5 Cohen d | pooled d | Verdict |
|---------|------------|------------|---------|----------------|----------------|----------|---------|
| GLBX 2024 | −12.8 | +0.8 | −8 dBHz | — | — | inverted | **Contaminated** |
| UMNQ 2025 | +5.3 | +4.1 | −0.2 dBHz | **1.18** | 1.05 | 1.12 | **Clean** |
| NIAQ 2025 | +0.7 | +0.2 | −0.0 dBHz | — | — | — | **Clean** |

GLBX is the only station where the frequency-split feature is a bias correction. At UMNQ and NIAQ it is signal-additive. See C8 for full results.

**Location:** `feature_aggregator.py:_aggregate_sector()`, new block after per-band RH medians.

**Status:** Implemented in feature aggregator. GLBX 2024 re-aggregated with frequency-split columns — confirmed present: `amp_L1_mean`, `amp_L2C_mean`, `amp_L5_mean`, `amp_E6_mean`. Next step: replace `amp_mean` with `amp_L5_mean` in GLBX `mahal_features` config and re-run v2/v3 classifiers. The existing `amp_mean` is preserved — this is additive, not a replacement.

---

### C8: Frequency-split amplitude validation — UMNQ 2025 and NIAQ 2024–2025

**Question:** Does the GLBX L1-suppression contamination pattern (pooled amp_mean inverted in winter by L1 SNR collapse) appear at UMNQ or NIAQ?

**Short answer: No. Both Greenland stations are Clean.**

---

#### Band availability

Neither station is L1-dominated. At UMNQ 2025, L5 is the dominant band (~41% of arcs), L1 is 27–33% — nearly inverted from GLBX. At NIAQ 2025, L1 and L5 are near parity (~35% each). L5 arcs per day in winter: UMNQ ≥ 78/day (December minimum), NIAQ ≥ 145/day. The len ≥ 2 gate is not a constraint at either station in any month.

#### L1 vs L5 amplitude seasonal pattern

| Station | L1 winter | L1 summer | L5 winter | L5 summer | L1 Δ | L5 Δ | Verdict |
|---------|-----------|-----------|-----------|-----------|-------|-------|---------|
| GLBX 2024 | 16.41 | 29.22 | 41.54 | 40.78 | **−12.81** | +0.76 | CONTAMINATED |
| UMNQ 2025 | 26.55 | 21.27 | 30.48 | 26.40 | **+5.28** | +4.08 | CLEAN |
| NIAQ 2025 | 20.96 | 20.26 | 22.42 | 22.27 | **+0.70** | +0.15 | CLEAN |

At both Greenland stations both bands rise together in winter — physically correct for ice (elevated coherent backscatter). The GLBX signature (L1 collapses 12.8 units, L5 flat) is absent. The L5−L1 gap at UMNQ narrows slightly in winter (5.1 → 3.9 units), consistent with L1 responding with proportionally greater relative gain to specular ice reflection from a lower baseline.

#### MS (mean SNR) by band

UMNQ 2025: L1 MS winter 39.72 vs summer 39.95 dBHz (Δ = −0.22). L5 MS winter 41.93 vs summer 41.75 (Δ = +0.18). Variation is measurement noise.

NIAQ 2025: L1 Δ = −0.03 dBHz. L5 Δ = −0.42 dBHz. All bands decline < 0.5 dBHz.

GLBX reference: L1 drops 8 dBHz in winter (37.18 → 29.18); L5 flat. The 15 dBHz receiver-level L1 suppression at GLBX is not replicated at either Greenland station.

#### Pool correlation by season (UMNQ 2025)

| Season | r(L1, pooled) | r(L5, pooled) | r(L2C, pooled) |
|--------|--------------|--------------|----------------|
| Winter | 0.987 | **0.997** | 0.993 |
| Summer | 0.728 | **0.902** | 0.727 |

All bands track the pool at r > 0.98 in winter. No pathological decoupling; pool faithfully aggregates both bands. NIAQ is r > 0.99 for all bands in winter.

#### Cohen's d by band (UMNQ 2025, anomalous vs baseline)

| Feature | Cohen's d |
|---------|-----------|
| pooled amp_mean | 1.12 |
| amp_L1_mean | **1.18** |
| amp_L5_mean | 1.05 |

L1-specific amplitude is the strongest discriminator (d = 1.18), not L5. The pooled mean (d = 1.12) is the arithmetic midpoint — fully representative and not masking signal. This is the opposite of GLBX, where L1 suppression inverts the pooled feature direction.

**Autumn sensitivity (DOY 300–366 at UMNQ):** pooled amp_mean d = 0.25, amp_L1_mean d = 0.72, amp_L5_mean d = −0.003. L1 is correctly responding to early thin ice while L5 has not yet diverged and dilutes the pooled mean. See "L1 freeze-onset sensitivity at UMNQ" section below for the full crossover analysis and recommended DOY boundary.

---

#### Summary verdicts

**UMNQ 2025: Clean.** L1 suppression mechanism absent. Both bands rise together in winter (correct ice physics). L1 is marginally the better single-band discriminator (d = 1.18 vs d = 1.05 for L5). Pooled amp_mean is fully representative. Frequency-split features are signal-additive (L1 and L5 carry complementary noise characteristics) but not necessary to correct a contamination artifact.

**NIAQ 2024–2025: Clean.** Even weaker seasonal variation than UMNQ (< 1 unit amplitude change between winter and summer for all bands). MS is flat (< 0.5 dBHz variation). Pool correlations r > 0.99 in winter. No frequency divergence of any kind.

**GLBX: Contaminated (reference).** L1 hardware-level suppression (8 dBHz input SNR drop, 12.8-unit amplitude collapse) is a station-specific hardware or atmospheric/ionospheric effect not present at Greenland stations. The frequency-split features are critical there, additive elsewhere.

---

#### L1 freeze-onset sensitivity at UMNQ

During DOY 300–366 (autumn freeze-up at Upernavik), Cohen's d by feature:

| Feature | Cohen's d (DOY 300–366) |
|---------|------------------------|
| pooled amp_mean | 0.25 |
| amp_L1_mean | **0.72** |
| amp_L5_mean | −0.003 |

L1 amplitude responds to early autumn ice formation (d = 0.72) while L5 has essentially no separation (d = −0.003) and the pooled mean dilutes L1 to d = 0.25. This is not a contamination problem — L1 is physically more sensitive to the early freeze-up transition, not suppressed or artifactually inverted. The pooled mean masks a real signal by averaging in an uninformative band.

**Physical interpretation:** Early autumn sea ice at Upernavik (DOY ~300–330) forms as thin, relatively smooth new ice. The L1 wavelength (19 cm) is comparable to the surface roughness scale of newly forming sea ice, making L1 scattering more sensitive to the transition from open water (diffuse, rough) to thin ice (specular, smooth). L5 (wavelength 25 cm) sits on the longer side of this roughness scale and responds less to early-stage thin ice. The sensitivity difference converges as ice thickens and both bands see a solidly specular surface.

**Connection to B4 autumn results (C4):** The B4 sector consistency check found that autumn anomalous days (DOY 300–366) have inter-sector std comparable to regime_change (~1.3) — sectors agree during autumn freeze-up, unlike the patchy spring breakup. This was interpreted as uniform early freeze signal. The L1 sensitivity finding sharpens that interpretation: the autumn anomalous signal is real and spatially coherent, but it is carried primarily by L1, and the pooled amp_mean classifier is operating at only d = 0.25 on those days when it could be using d = 0.72.

**Implication for classification:** A classifier that uses `amp_L1_mean` specifically during the autumn window would have substantially better sensitivity to freeze onset than the current pooled amp_mean. The precise DOY boundary and crossover mechanism are documented in the section below.

---

#### DOY crossover analysis: L1 vs pooled amplitude discrimination

**Method:** 30-day rolling window Cohen's d, stepped 7 days, for `amp_L1_mean`, `amp_mean` (pooled), and `amp_L5_mean`. Computed from arc_table (freq_group groupby) merged with v3 state labels; regime_change days excluded. `amp_L1_mean` not yet in UMNQ daily_features (feature aggregator not re-run for UMNQ after C7 implementation); computed directly from arc_table for this analysis.

**Rolling d table (condensed, autumn series):**

| Window center DOY | Calendar | n_anom | n_base | d_pooled | d_L1 | d_L5 | gap (L1−pool) |
|-------------------|----------|--------|--------|----------|------|------|---------------|
| 271 | Sep 28 | 16 | 8 | −2.034 | −1.666 | −2.184 | 0.368 |
| 278 | Oct 5 | 16 | 12 | −1.325 | −0.460 | −1.622 | **0.866** |
| 285 | Oct 12 | 16 | 11 | −1.110 | −0.111 | −1.350 | **0.999** |
| 292 | Oct 19 | 18 | 9 | −1.201 | −0.416 | −1.480 | **0.785** |
| 299 | Oct 26 | 17 | 9 | −1.123 | −0.259 | −1.437 | **0.864** |
| 306 | Nov 2 | 7 | 10 | −0.712 | +**0.465** | −1.062 | **1.177** |
| 313 | Nov 9 | 7 | 10 | −0.342 | +0.229 | −0.678 | 0.571 |
| 320 | Nov 16 | 5 | 12 | −0.413 | +0.158 | −0.780 | 0.571 |
| 334 | Nov 30 | 13 | 8 | +0.088 | +0.517 | −0.150 | 0.430 |
| 341 | Dec 7 | — | — | +0.643 | +0.900 | +0.376 | 0.257 |

**Sign interpretation:** In the autumn rolling windows, "baseline" days are late-summer open water (high amplitude, seasonal geometry peak), and "anomalous" days are early freeze-up (transitional lower amplitude). Negative d means anomalous days have lower amplitude than baseline — both features initially point in the wrong direction because seasonal geometry suppression partially offsets the ice-backscatter elevation. L1 d crosses zero (transitions to the correct ice-positive direction) around DOY 285–306, a full month before pooled amp_mean (~DOY 334). L5 remains strongly negative throughout (d = −1.06 to −2.18), actively dragging the pooled mean in the wrong direction.

**Crossover window:**

| Event | DOY | Calendar date | d_L1 | d_pooled | gap |
|-------|-----|---------------|------|----------|-----|
| Entry (gap > 0.2) | 271 | Sep 28 | −1.666 | −2.034 | 0.368 |
| Peak gap | 306 | Nov 2 | **+0.465** | −0.712 | **1.177** |
| Exit (gap drops < 0.2) | 341 | Dec 7 | +0.900 | +0.643 | 0.257 |

L1 d is positive (correct ice direction) from DOY ~285–306 onwards; pooled d becomes positive only at DOY ~334. Gap is robustly > 0.78 across DOY 278–306.

**Alignment with gamma_r2 inversion window (C1):**
- gamma_r2 begins inverting: DOY ~241–300
- gamma_r2 strongly inverted (d = −1.25): DOY ~301–366
- L1 gap entry (robust, > 0.78): DOY 278
- Peak gap: DOY 306 — coincides exactly with the gamma_r2 strong inversion boundary

Both the L1 amplitude decoupling and the gamma_r2 sign inversion onset at the same calendar point (~DOY 285–306). This is not coincidental: both effects reflect thin new ice forming at the same time — a surface that L1 (19 cm wavelength) begins to see specularly while L5 (25 cm) and gamma (Strandberg smooth-surface model) have not yet transitioned. A single autumn season boundary in the DOY 280–300 range accounts for both feature-switching motivations.

**Candidate boundaries:**

| Boundary | Calendar | d_L1 | d_pooled | gap | Notes |
|----------|----------|------|----------|-----|-------|
| DOY 270 | Sep 27 | −1.666 | −2.034 | 0.368 | First marginal entry; gap not yet robust |
| **DOY 285** | **Oct 12** | **−0.111** | **−1.110** | **0.999** | **d_L1 ≈ 0: L1 has crossed from wrong- to right-direction; maximum gap zone** |
| DOY 300 | Oct 27 | −0.259 | −1.123 | 0.864 | Still inside high-gap zone; safely conservative |

**Recommendation: DOY 285 (October 12).** At this boundary, d_L1 collapses to near-zero (the seasonal geometry suppression and ice-backscatter elevation cancel), while d_pooled remains strongly negative (−1.110) — the worst-performing window for the pooled feature. It is inside the robust high-gap zone (not at the marginal entry), round as a parameter, and two weeks before the gamma_r2 strong inversion onset (DOY ~301), giving the season-aware switch a natural lead-in.

**Practical implementation:** After DOY 285, the season-aware classifier should use `amp_L1_mean` as the primary amplitude feature in place of pooled `amp_mean`. From DOY 341 onward both features recover correct-direction d, so `amp_L1_mean` remains preferable (higher d) but pooled amp_mean is no longer actively harmful. The winter period (DOY 1–120) uses pooled amp_mean or amp_L1_mean equivalently — no DOY switch needed there since both are strongly positive.

---

**GLBX re-aggregation confirmed** (monthly medians from re-aggregated `GLBX_2024_daily_features.parquet`, pooled rows):

| Month | amp_L1_mean | amp_L5_mean | amp_E6_mean | pooled amp_mean |
|-------|-------------|-------------|-------------|-----------------|
| Jan | 8.76 | **46.08** | NaN | 21.09 |
| Feb | 9.52 | 42.06 | 57.37 | 26.45 |
| Mar | 9.21 | 41.26 | 57.94 | 28.03 |
| Apr | 9.73 | 40.38 | 57.47 | 29.43 |
| Jun | 30.16 | 41.16 | 57.95 | 33.79 |
| Jul | 29.70 | 40.96 | 57.63 | 33.80 |
| Oct | 30.07 | 41.67 | 58.91 | 34.67 |

L5 is flat or slightly elevated in winter (40–46) vs summer (41–42). L1 collapses to 8–10 in winter vs 29–31 in summer. E6 is flat at 57–59 year-round. The pooled amp_mean (21–29 in winter) faithfully tracks the L1 suppression because L1 arcs dominate by count. Frequency-split columns are now in the feature table and ready for use in classifier config.

**Implication for C7 implementation:** The bias-correction motivation applies only to GLBX. The GLBX `mahal_features` config should replace `amp_mean` with `amp_L5_mean` and switch its polarity from `"winter_low"` to `"winter_high"`. At UMNQ, `amp_L1_mean` is the preferred amplitude feature for season-aware classification, particularly in the autumn window — the pooled amp_mean under-uses the available L1 sensitivity during freeze-up.

---

## Part 3: Literature Cross-Validation (2026-04-08)

Systematic comparison of our implementation against Strandberg 2017, Song 2022, and Purnell 2024.
Tested five hypotheses derived from gaps between our pipeline and the published methods.
Data: UMNQ 2025 (62,079 arcs), ROSS 2024 (8,261 arcs), GLBX 2024.

### C10. γ_rel Reference-Period Normalization (Strandberg 2017)

**Hypothesis:** Strandberg's γ_rel = γ / γ_ice_free cancels antenna gain and local geometry, making γ comparable across sites. Our absolute γ is site-dependent — does normalizing recover discriminability at ROSS?

**Result: γ_rel does not improve Cohen's d.** Dividing every arc's γ by a global scalar (the ice-free median) rescales both the mean difference and the pooled SD identically, so d is mathematically invariant. To make Strandberg's approach useful, you would need per-arc or per-azimuth normalization, not a global divisor.

**Gamma polarity across stations:**
- Both ROSS and UMNQ show γ **decreasing** during ice — same direction, no inversion
- ROSS: γ drops 96% (0.0013 → 0.0000), d = 1.40 — strong
- UMNQ: γ drops 4% (0.0046 → 0.0044), d = 0.04 — negligible

**Implication:** Strandberg's γ_rel worked because their 72-hour pooled NLLS reduced noise enough for the normalized value to be meaningful. Our per-arc γ is too noisy for a simple ratio to help. γ_r2 gating is a better approach than normalization for per-arc extraction.

---

### C11. AF Polarity Disagreement (Purnell 2024 §VIII.C)

**Hypothesis:** Purnell noted AF increases during open water at river sites, opposite to Song's finding at TUKT. Do ROSS and UMNQ agree on AF polarity?

**Result: AF polarity is inverted between stations.**
- ROSS: AF drops 91% during ice (16,505 → 1,429), d = 1.59 — ice destroys amplitude
- UMNQ: AF **increases** 37% during ice (7,145 → 9,795), d = −0.19 — ice enhances amplitude

This matches Purnell's finding exactly. The direction of the AF response to ice is site/ice-type dependent:
- Great Lakes rough freshwater ice (ROSS): scatters signal → AF drops
- Arctic sea ice (UMNQ): smoother specular reflection → AF rises

**Implication:** The classifier already handles this via Cohen's d sign (feature polarity is learned per station). But AF cannot be used as a universal ice indicator without station-specific calibration. Cross-station transfer learning for AF would require polarity awareness.

Additional UMNQ AF monthly pattern: very high AF in Mar–Apr (~39,000), drops to ~7,000 in summer, climbs to ~13,000 in December. The spring spike may reflect snow/ice melt dynamics.

---

### C12. Per-PRN Systematic Biases (Purnell 2024 Table II)

**Hypothesis:** Purnell found per-satellite normalization improved accuracy ~4%. Are there systematic per-PRN biases in our CLR/amplitude that daily median pooling averages over?

**Result: PRN bias is real, magnitude is feature-dependent.**

| Feature | ICC (variance from PRN identity) | Interpretation |
|---------|----------------------------------|----------------|
| CLR     | 11.5%                            | Modest — most variance is surface state |
| PkNoise | 13.4%                           | Similar to CLR |
| Amp/SP  | ~33%                             | Substantial — 1/3 of variance is "which PRN" |
| MS      | **87.8%**                        | Almost entirely PRN-determined (geometry) |
| VS      | 42.1%                            | Large PRN effect |

Per-PRN median CLR ranges from 6.09 to 12.52 (2x range). The bias is partially persistent across seasons (summer↔winter correlation r = 0.41) but partially state-dependent.

**Key finding: PRNs differ dramatically in state-discriminating power.** Per-PRN Cohen's d for CLR ranges from ~0 to 2.87. 25.8% of PRNs have |d| > 0.5 (medium effect); 10.7% have |d| > 0.8 (large effect). Some PRNs are excellent surface-state sensors; others contribute noise.

**Implication:** Simple per-PRN z-score normalization is warranted for Amp/SP (ICC ~33%) and beneficial for CLR (ICC ~11%). But the heterogeneous discriminating power suggests a weighting scheme (inverse noise or by |d|) would be more effective than uniform normalization.

---

### C13. γ_r2 as Surface Complexity Proxy (Song 2022)

**Hypothesis:** Song found damping had "limited utility in multilayer surface" (snow-on-ice). If γ_r2 systematically drops during snow-on-ice periods, it would be a proxy for surface layer count — more than just a quality gate.

**Result: Hypothesis rejected. γ_r2 behaves opposite to prediction.**

UMNQ monthly γ_r2:

| Surface state | Months | Median γ_r2 |
|---------------|--------|-------------|
| Snow-on-ice   | Jan–Apr | **0.199** (highest) |
| Transition    | May, Sep–Oct | 0.122 |
| Open water    | Jun–Aug | 0.112 |
| Clean ice     | Nov–Dec | **0.093** (lowest) |

All differences are highly significant (Mann-Whitney p < 10^-170). Snow-on-ice has the **best** Strandberg model fits, not the worst.

ROSS: universally near zero (overall median = 0.000, 78% of arcs below 0.1), but the same pattern holds — ice-covered months (Feb–Mar, median 0.096) fit better than ice-free (Jun–Sep, median 0.000).

**Revised interpretation:** γ_r2 measures **surface coherence** (how specular/flat the reflecting surface is), not surface complexity. Calm ice/snow produces a clean exponential envelope decay (high R²). Rough open water produces noisy interference that the model cannot fit (low R²). This is consistent with Song's finding — the damping model fails for rough/complex surfaces — but the mechanism is about specularity, not layer count.

Arc-level correlation between γ and γ_r2 at UMNQ: Spearman ρ = 0.77 (substantially correlated but carries independent information). At ROSS: Pearson r = 0.94 (tightly coupled, γ_r2 adds little beyond γ).

**Implication:** Reframe γ_r2 from "quality gate on γ" to "surface coherence metric." It is informative about surface state in its own right, but its meaning is "how specular is the surface" rather than "how trustworthy is γ."

---

### C14. Per-Constellation×Frequency Normalization (Strandberg 2017)

**Hypothesis:** Strandberg normalized amplitude per (constellation × frequency) because it is frequency-dependent. Our AF is baseline-subtracted per-PRN, but other features are not. Does per-combo normalization improve discrimination?

**Result: Feature-dependent — one major win.**

UMNQ ice-free median AF spans 4.7x across (constellation, freq) combos (3,074 to 14,590). Yet only 15.5% of ice-free AF variance is from combo identity (ANOVA η²). Seasonal arc composition is nearly identical between ice-free and winter, so daily median is not biased by composition shifts.

Impact of per-(constellation, freq) z-scoring on Cohen's d:

| Feature | Daily d (raw) | Daily d (z-scored) | Change |
|---------|---------------|--------------------|---------| 
| AF      | 1.73          | 1.63               | −6% (no help — already baseline-subtracted) |
| SP      | 0.86          | 0.89               | +5% (minor) |
| **MS**  | **0.19**      | **1.04**           | **+457%** (transforms useless → strong) |

**MS is the headline finding.** Raw pooled daily MS has near-zero discrimination because per-combo baselines (antenna gain patterns at different frequencies/elevations) swamp the temporal signal. After per-combo z-scoring, MS becomes a strong discriminator comparable to AF. This makes physical sense: MS is mean SNR in dB, which is dominated by satellite geometry and antenna gain unless you remove the per-combo baseline.

**GLBX:** Per-freq normalization changes the daily signal substantially (r = 0.42 between raw and z-scored daily AF, vs r = 0.996 at UMNQ). L1 and L5 AF are anti-correlated (r = −0.39). Normalizing would mask the physically meaningful L1/L5 differential response.

Best-discriminating combos at UMNQ: GPS L5 (d = 1.19), GPS L2C (d = 1.16), GPS L1 (d = 1.15). Weakest: GLONASS G1 (d = 0.57).

**Implication:**
1. **Add MS to z-score normalization** — per-(constellation, freq) z-scoring using ice-free reference, then aggregate. This is the single highest-value change from the literature comparison.
2. Leave AF un-normalized at the daily level (already per-PRN baseline-subtracted).
3. At GLBX, preserve frequency-separated features; do not z-score AF across frequencies.
4. The L1/L5 AF ratio at GLBX is a candidate new feature (frequency-dependent scattering).

---

### Summary: Literature Gap Analysis

| Test | Source | Finding | Action |
|------|--------|---------|--------|
| C10: γ_rel | Strandberg | Global scalar normalization cannot improve d | None — γ_r2 gating is better |
| C11: AF polarity | Purnell | Inverted between ROSS and UMNQ | Classifier already handles via d sign |
| C12: Per-PRN bias | Purnell | ICC 11–33% depending on feature; PRN discriminating power varies 0–2.9 | Add per-PRN z-score for MS; consider PRN weighting |
| C13: γ_r2 meaning | Song | Surface coherence metric, not complexity proxy; ice > water | Reframe from quality gate to coherence feature |
| C14: Per-combo MS | Strandberg | MS d jumps 0.19 → 1.04 with per-combo z-scoring | **Implement immediately** — add MS to `_ZSCORE_FEATURES` |

Figures for C10–C14: generated by `scripts/plot_investigation_figures.py` (Stories 1–5) and ad-hoc analysis scripts (not yet persisted as figures).
