# GNSS-IR Parameters Reference

This document explains the parameters in the station params JSON file (e.g., `config/glbx.json`) and how to choose appropriate values for your site.

## Quick Start

Generate a starting params file using gnssrefl:
```bash
gnssir_input STATION -lat 45.0 -lon -122.0 -height 10.0
```

This creates `$REFL_CODE/input/STATION.json`. Copy it to `config/station.json` and customize.

---

## Required Parameters

### Station Identity

| Parameter | Description | Example |
|-----------|-------------|---------|
| `station` | 4-character station ID (lowercase) | `"glbx"` |
| `lat` | Latitude in decimal degrees | `58.455` |
| `lon` | Longitude in decimal degrees | `-135.888` |
| `ht` | Ellipsoidal height in meters | `-12.535` |

**Note on height**: This is the antenna height in ellipsoidal coordinates (WGS84), NOT mean sea level. You can get this from precise GNSS processing or survey records.

---

## Reflector Height Parameters

These define the expected range of reflector heights (distance from antenna to reflecting surface).

| Parameter | Description | How to Choose |
|-----------|-------------|---------------|
| `minH` | Minimum reflector height (m) | Distance to water at highest tide/level |
| `maxH` | Maximum reflector height (m) | Distance to water at lowest tide/level |
| `NReg` | Noise region for LSP analysis | Usually same as `[minH, maxH]` |

### How to Estimate minH/maxH

1. **From site survey**: Measure antenna height above typical water levels
2. **From tidal range**: If antenna is 10m above mean water and tidal range is ±3m:
   - `minH` = 10 - 3 = 7m (high tide)
   - `maxH` = 10 + 3 = 13m (low tide)
3. **Start wide, then narrow**: Begin with a 5-15m range, examine results, then refine

**Example for coastal site**:
```json
"minH": 5.0,
"maxH": 12.0,
"NReg": [5.0, 12.0]
```

---

## Elevation Angle Parameters

Control which satellite elevation angles are used for analysis.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `e1` | Minimum elevation angle (degrees) | 2-5 |
| `e2` | Maximum elevation angle (degrees) | 8-25 |
| `pele` | Elevation range for polynomial fit | `[e1, 30]` |
| `ediff` | Minimum arc length in degrees | 2.0 |

### Guidelines

- **Lower elevation angles** (2-5°): More sensitive to reflections, but more noise
- **Higher elevation angles** (>25°): Less reflection signal
- **Typical coastal setup**: `e1=2, e2=8` or `e1=5, e2=15`
- **Sites with obstructions**: May need higher `e1` to avoid terrain interference

```json
"e1": 2.0,
"e2": 8.0,
"pele": [2.0, 30],
"ediff": 2.0
```

---

## Azimuth Masking (azval2)

Defines which directions (azimuths) to include in analysis. This is critical for excluding obstructions.

| Parameter | Description |
|-----------|-------------|
| `azval2` | List of azimuth ranges to INCLUDE (degrees, 0=North, 90=East) |

### Format

`azval2` contains pairs of [start, end] azimuths that define INCLUDED sectors:
- `[0, 360]` = use all directions
- `[0, 90, 270, 360]` = use 0-90° AND 270-360° (exclude 90-270°)
- `[45, 135, 225, 315]` = use 45-135° AND 225-315°

### How to Determine Azimuth Masks

1. **Site visit**: Note directions with trees, buildings, or land
2. **Satellite imagery**: Check for obstructions in Google Earth
3. **Initial processing**: Run without mask, check polar plots for bad sectors

**Example - coastal site with land to the south**:
```json
"azval2": [270.0, 360.0, 0.0, 90.0]
```
This uses the northern half (270° through 0° to 90°), excluding the southern sector where land interferes.

---

## Signal Quality Parameters

Control which reflections are accepted based on signal quality.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `PkNoise` | Minimum peak-to-noise ratio in LSP | 2.0-3.0 |
| `reqAmp` | Minimum amplitude per frequency | 5.0-8.0 |

### Guidelines

- **Higher PkNoise** (>3.0): Fewer but higher-quality retrievals
- **Lower PkNoise** (<2.0): More retrievals but potentially noisier
- **reqAmp**: Should have one value per frequency in `freqs`

```json
"PkNoise": 2.5,
"reqAmp": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```

---

## Frequency Selection

| Parameter | Description |
|-----------|-------------|
| `freqs` | List of GNSS frequency codes to process |

### Common Frequency Codes

| Code | Signal | Constellation |
|------|--------|---------------|
| 1 | L1 C/A | GPS |
| 5 | L5 | GPS |
| 20 | L2C | GPS |
| 101 | L1 | GLONASS |
| 102 | L2 | GLONASS |
| 201 | E1 | Galileo |
| 205-208 | E5a/E6/E7/E8 | Galileo |
| 302, 306 | B2, B6 | BeiDou |

*Source: gnssrefl/gps.py lines 1790-1830*

### Recommendation

Use all available frequencies for maximum data:
```json
"freqs": [1, 20, 5, 101, 102, 201, 205, 206, 207, 208, 302, 306]
```

---

## Processing Options

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `polyV` | Polynomial order for detrending | 4 |
| `desiredP` | Precision parameter | 0.005 |
| `delTmax` | Max time gap in arc (seconds) | 75 |
| `refraction` | Apply refraction correction | `true` |

These rarely need adjustment from defaults.

---

## Output Options

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `overwriteResults` | Overwrite existing results | `true` |
| `plt_screen` | Show plots on screen | `false` |
| `screenstats` | Print stats to screen | `false` |
| `pltname` | Output plot filename | `"station_lsp.png"` |

---

## Complete Example

Here's a complete params file for a coastal GNSS station:

```json
{
    "station": "glbx",
    "lat": 58.455146658,
    "lon": -135.8884838318,
    "ht": -12.535,
    "minH": 5.0,
    "maxH": 12.0,
    "e1": 2.0,
    "e2": 5.0,
    "NReg": [5.0, 12.0],
    "PkNoise": 2.0,
    "polyV": 4,
    "pele": [2.0, 30],
    "ediff": 2.0,
    "desiredP": 0.005,
    "azval2": [0.0, 45.0, 270.0, 360.0],
    "freqs": [1, 20, 5, 101, 102, 201, 205, 206, 207, 208, 302, 306],
    "reqAmp": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "refraction": true,
    "overwriteResults": true,
    "seekRinex": false,
    "wantCompression": false,
    "plt_screen": false,
    "screenstats": false,
    "pltname": "glbx_lsp.png",
    "delTmax": 75.0,
    "gzip": false,
    "ellist": []
}
```

---

## Troubleshooting

### Low retrieval count
- Widen `minH`/`maxH` range
- Lower `PkNoise` threshold
- Check azimuth mask isn't excluding too much

### Noisy results
- Increase `PkNoise` (try 2.5-3.0)
- Narrow elevation range
- Add azimuth masks for problematic directions

### No results at all
- Verify coordinates are correct
- Check `minH`/`maxH` includes actual reflector heights
- Ensure RINEX data exists for the station/date

---

## Further Reading

- [gnssrefl documentation](https://github.com/kristinemlarson/gnssrefl)
- Larson, K.M. (2016). GPS interferometric reflectometry: applications to surface soil moisture, snow depth, and vegetation water content in the western United States.
