# Investigation Scripts

Exploratory analysis scripts used during feature development and validation.
Findings are documented in [`docs/feature_investigation_log.md`](../../docs/feature_investigation_log.md).

| Script | Log Section | Status |
|--------|-------------|--------|
| `investigate_edot_qc.py` | B1 (§6): edot as geometry-informed QC layer | Parked — uninformative at polar latitudes |
| `investigate_xfreq_correlation.py` | B2 (§7): Cross-frequency SNR correlation | Dead — effect size too small for classification |
| `investigate_envelope_ratio.py` | B3 (§8): Multi-frequency envelope ratio for snow-on-ice | Dead — Hilbert envelope noise floor 400x theoretical signal |
| `investigate_geometry_census.py` | B4 (§9): Simultaneous satellite geometry census | Parked — feasibility confirmed, needs use case |
| `b4_sector_consistency.py` | C4: Sector consistency check (internal classifier validation) | Infrastructure preserved — reusable for recalibration validation |
| `compare_features_pca.py` | Precursor to C2–C3: PCA of SNR features for ice detection | Infrastructure preserved |
| `compare_features_dendrogram.py` | Precursor to C2–C3: Hierarchical clustering of feature redundancy | Infrastructure preserved |
| `analyze_land_water_azimuths.py` | Precursor to C2–C3: Mahalanobis land/water azimuth discrimination | Infrastructure preserved |
| `cross_station_summary.py` | C8: Cross-station frequency-split validation | Infrastructure preserved |
| `cygnss_plots.py` | CYGNSS comparison visualization (companion to cygnss_comparison.py) | Infrastructure preserved |
| `tec_era5_pca.py` | Combined TEC + ERA5 + SNR PCA for RH divergence attribution | Infrastructure preserved |
