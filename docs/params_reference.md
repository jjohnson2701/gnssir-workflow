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
    "e2": 10.0,
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

## Findings from tests in 2024

From three stations: GLBX (Gustavus AK, elevation 2 to 5 degrees, 6 m tide), SC02 (Friday Harbor WA, 5 to 13 degrees, 3 m tide) and FORA (Fort Raleigh NC, wind-driven water level). Each test had its plan written before the run.

### Peak-to-noise limit (PkNoise)

Process a month with the limits off (`reqAmp` all 0, `PkNoise` 0, as an extension) and plot the scatter against a gauge for each limit. `scripts/station_page.py` makes this plot.

- GLBX: scatter falls from 0.29 m with no limit to 0.20 m at 2.0, and the slope against the gauge moves from 0.954 to 1.000. Weak arcs are biased toward the middle of the height window, not only noisy. Above 2.0 the time coverage drops.
- SC02: the limit changes the scatter by 1% up to 2.8.
- Once the peak-to-noise limit is set, the amplitude limit (`reqAmp`) removed almost nothing at these stations.

Most arcs that the limit rejects still hold the true peak: at GLBX 58% of rejected arcs have a peak at the expected height, against 0.2% at a wrong height. The inverse method uses that information.

### Refraction

Keep `refraction` on. With no correction the heights have a scale error: 4.7% at 2 to 5 degrees, 1.5% at 5 to 13 degrees. The default model (1, Bennett) gives a slope of 1 within 0.4% at both stations. Model 5 (NITE) is not better in gnssrefl, which uses a climatology for all models.

### Sample interval

Slower sampling costs arcs before it costs precision. At GLBX (arcs of about 10 minutes): 5 s keeps 97% of the arcs, 15 s keeps 80%, 30 s keeps 51%. At SC02 30 s keeps 99%. To decimate, use the command-line option `gnssir ... -dec N`; a `dec` value in the station file is ignored.

### Inverse method

`scripts/inverse_heights.py` needs enough arcs per 3 hours. Where the periodogram range is narrow, give the inverse method a wider elevation range with `inverse_elevation` in `config/stations.yaml`. GLBX: 7.1 cm with 2 to 5 degrees, 4.1 cm with 2 to 10 degrees. Use all signals together; single-signal runs were unstable at GLBX.

### L2 signal

A RINEX 3 file can hold both the civil L2 signal (L2C; codes S2L, S2S, S2X) and the legacy signal (S2W, S2P). gnssrefl 4 reads L2C from a RINEX 3 file. A conversion to RINEX 2 can put the legacy signal in its place: its SNR is 7 to 19 dB lower. `scripts/station_check.py` lists what a file holds.

### Height above the surface and distance from the water level

The circular plot (`scripts/visualizer/periodogram_circle.py`) shows both. Use the second view at tidal stations to choose the azimuth mask: reflections that follow the water lie on the ring; fixed reflectors do not.

## Further Reading

- [gnssrefl documentation](https://github.com/kristinemlarson/gnssrefl)
- Larson, K.M. (2016). GPS interferometric reflectometry: applications to surface soil moisture, snow depth, and vegetation water content in the western United States.
