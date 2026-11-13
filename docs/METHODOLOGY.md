# Methodology

This document explains how the workflow gets a water level from a GNSS station, how it compares the result with a gauge, and what our tests showed. For the commands, see the [README](../README.md) and the [worked example](PROCESSING_EXAMPLE.md). For each setting, see the [parameter reference](params_reference.md).

## 1. The measurement

A GNSS antenna receives each satellite signal twice: directly, and after a reflection from the surface around the antenna. The two signals interfere. As the satellite rises or sets, the signal-to-noise ratio (SNR) that the receiver records goes up and down in a regular pattern.

The pattern is regular in the sine of the elevation angle `e`. Its frequency depends on the height `H` of the antenna above the reflecting surface (the reflector height) and on the wavelength `λ` of the signal:

```
SNR pattern  ~  A · cos( 4π H sin(e) / λ  +  phase )
```

A higher antenna gives a faster pattern. The workflow finds the frequency, which gives `H`. The water level is the antenna height minus `H`:

```
water level = antenna height − H
```

The reflected signal is weak compared with the direct signal, and the surface is not the only reflector. The method therefore needs many satellite passes and clear rules for which passes to keep.

**Where the reflection comes from.** The antenna senses an area on the surface, not a point. For one pass this area (the first Fresnel zone) is a long ellipse that points toward the satellite. At a 5 m height and 5 degrees elevation it reaches about 100 m from the antenna. A lower elevation or a higher antenna gives a longer reach. The circular plot draws these zones on an image of the site.

## 2. Processing steps

| Step | Tool | Result |
|---|---|---|
| 1. Get the observation file | NPS archive, EarthScope (`scripts/earthscope_rinex3.py`), or another archive that gnssrefl knows | one RINEX file per day |
| 2. Translate | gnssrefl `rinex2snr` | one SNR file per day: elevation, azimuth, time and SNR for each satellite and signal |
| 3. Heights of single arcs | gnssrefl `gnssir` | one reflector height per arc that passes the checks |
| 4. Smooth series | gnssrefl `subdaily`, or the inverse method (`scripts/inverse_heights.py`) | water level against time |
| 5. Compare | gauge comparison scripts, `scripts/station_page.py` | statistics and figures |

The workflow gives a RINEX 3 file to `rinex2snr` as it is. It does not convert to RINEX 2: a conversion can replace the civil L2 signal (L2C) with the weaker legacy signal (see section 5).

### Arcs and the periodogram (step 3)

An arc is one rising or setting pass of one satellite, for one signal, inside the elevation range and the azimuth mask of the station. For each arc `gnssir`:

1. removes the slow trend of the direct signal with a polynomial;
2. computes a Lomb-Scargle periodogram of the remaining SNR against `sin(e)`, for heights between `minH` and `maxH`;
3. takes the height of the highest peak.

An arc passes when its peak is high enough (`reqAmp`), stands out from the rest of the periodogram (`PkNoise`, the peak-to-noise limit), and the arc covers enough of the elevation range. Arcs outside the azimuth mask are not used.

### Two corrections

**Refraction.** The atmosphere bends the signal, so the satellite appears higher than it is. Without a correction all heights are too small by a constant factor. The factor is larger at lower elevations. gnssrefl corrects the elevation angle with a model; the default (Bennett) is the one the workflow uses.

**Moving surface.** The periodogram assumes that the height is constant during the arc. With a tide, the height changes during the arc, and the measured height is shifted by an amount that is proportional to the rate of the water level:

```
shift = (rate of height change) × tan(e) / (de/dt)
```

gnssrefl gives the factor `tan(e)/(de/dt)` for each arc (column `edotF`). `subdaily` estimates the rate from a first spline fit and removes the shift. At a station with no tide the shift is negligible.

### Smooth series: two methods (step 4)

**`subdaily`** fits a spline (8 knots per day) through the heights of the single arcs, removes outliers, applies the correction for the moving surface, and removes offsets between signals.

**The inverse method** (gnssrefl `invsnr`, after Strandberg et al. 2016) does not use the single heights. It describes the water level as one smooth curve (a spline with a knot every 3 hours) and fits that curve to the SNR data of all arcs together. Arcs that are too weak to pass the periodogram checks still add information. In our tests this method was closer to the gauge than `subdaily` at both tidal stations (section 5). It needs arcs in every knot interval: the wrapper stops at long gaps and removes output where fewer arcs support the curve.

## 3. Comparison with a gauge

A GNSS station and a gauge do not share a height reference in general, and they are often some distance apart. The workflow compares them in a way that does not hide a scale error.

For the single arcs it fits

```
−H = slope × gauge + k × edotF × (gauge rate) + constant
```

and reports two numbers:

- **Slope.** 1.000 means that a 1 m change at the gauge is a 1 m change in the GNSS height. A slope below 1 points to a missing refraction correction or to weak arcs.
- **Scatter.** The spread of the differences after the fit, as a robust standard deviation (1.4826 × the median absolute deviation). A few outliers do not change it.

For a smooth series the workflow shifts the series by one constant (the median difference) and fits no scale.

When no gauge is near the station, `scripts/station_check.py` gives the scatter of the single heights about their own spline. This number shows the precision, not the accuracy.

## 4. Visual checks

- **Circular plot** (`scripts/visualizer/periodogram_circle.py`): the periodograms of one day by azimuth, in two views. *Height above the surface* shows the reflector height. *Distance from the water level* shows each reflection against the water level at the time of the pass, so reflections from the water lie on one ring at any tide and fixed reflectors do not. Use it to set the azimuth mask. The principle is published (Santamaría-Gómez et al. 2015: a stack of spectra sharpens for the sea after the tide is removed, and for fixed reflectors before); the display by azimuth is ours. Three limits: it needs a tide range much larger than the height resolution of one arc; floating objects such as boats and docks follow the water and look like water; and the water level comes from the station's own arcs, so it is not independent of them.
- **Station page** (`scripts/station_page.py`): for each station the circular plot, the water level series against the gauge, the effect of the peak-to-noise limit, and the height quality against wind speed when wind data exist.

## 5. What the tests in 2024 showed

Three stations: GLBX (Gustavus, Alaska; elevation 2 to 5 degrees; 6 m tide; gauge 0.3 km away), SC02 (Friday Harbor, Washington; 5 to 13 degrees; 3 m tide; gauge at Friday Harbor) and FORA (Fort Raleigh, North Carolina; wind-driven water level; no usable gauge). The plan of each test was written before the run. Days 001 to 182 unless stated.

| Question | GLBX | SC02 |
|---|---|---|
| Slope with no refraction correction | 0.953 | 0.985 |
| Slope with the Bennett correction | 0.999 | 0.998 |
| Scatter of single arcs against the gauge | 0.195 m | 0.104 m |
| `subdaily` series against the gauge | 0.10 to 0.11 m | 0.026 m |
| Inverse method against the gauge | 0.041 m | 0.018 m |
| Arcs kept with 30 s data in place of 1 s or 15 s | 51% | 99% |

- **Refraction.** Keep the correction on. The more complex models in gnssrefl did not do better than Bennett; all of them use a climate average, not the weather of the day.
- **Peak-to-noise limit.** At GLBX a limit of 2.0 lowers the scatter from 0.29 m to 0.20 m and brings the slope from 0.954 to 1.000. At SC02 the limit changes little.
- **Inverse method.** A wider elevation range helps: at GLBX 0.071 m with 2 to 5 degrees and 0.041 m with 2 to 10 degrees. Use all signals together.
- **L2 signal.** A RINEX 2 file made from a RINEX 3 file can hold the legacy L2 signal, with an SNR 7 to 19 dB below the civil signal. `scripts/station_check.py` shows which signal a file holds.
- **BeiDou.** The numbers in the table are without BeiDou. With BeiDou, SC02 has 514 arcs a day in place of 350, the same slope (0.999), and a scatter of 0.108 m. BeiDou alone gives a slope of 1.001 and a scatter of 0.113 m.
- **Offsets between signals.** At SC02 the mean height of a signal depends on its carrier band, not on the satellite system: fourteen signals fall in four groups, 15 cm apart from the L1 band to the L5 band. `subdaily` removes such offsets. The cause is not known.
- **Wind.** At SC02 the heights of single arcs are not worse in strong wind (0.111 m scatter below 2 m/s, 0.095 m above 8 m/s). The inverse method is a little worse on windy days (1.6 to 2.2 cm).

## 6. Limits and open points

- The numbers above come from three stations and six months. They are not general limits of the method.
- FORA has no gauge that follows its water level (the nearest gauges are 21 and 27 km away and correlate poorly). Its results show precision only.
- At SC02 the height from the lower half of the elevation range differs from the height of the upper half by −11 to +15 cm, with a different sign for the L1 and the L2/L5 signals. We do not know the cause. It can be an effect of the antenna.
- At GLBX the refraction correction gives the right slope but adds 12 to 15% scatter. We do not know the cause.
- The comparison assumes that the gauge shows the water level under the reflection zones. This is not true where the gauge is far away or behind an inlet.
- Small water bodies, vegetation and ice are outside the tests.

## 7. Signal codes in gnssrefl

| Code | System | Signal |
|---|---|---|
| 1, 20, 5 | GPS | L1, L2C, L5 |
| 101, 102 | GLONASS | L1, L2 |
| 201, 205, 206, 207, 208 | Galileo | E1, E5a, E6, E5b, E5 |
| 301 to 308 | BeiDou | see the gnssrefl documentation |

## References

- Bennett, G. G. (1982). The calculation of astronomical refraction in marine navigation. *Journal of Navigation*, 35, 255–259.
- Larson, K. M. (2024). gnssrefl: an open source software package in python for GNSS interferometric reflectometry applications. *GPS Solutions*, 28, 165.
- Larson, K. M., Ray, R. D., Williams, S. D. P. (2017). A 10-year comparison of water levels measured with a geodetic GPS receiver versus a conventional tide gauge. *Journal of Atmospheric and Oceanic Technology*, 34, 295–307. (Friday Harbor, station SC02.)
- Santamaría-Gómez, A., Watson, C., Gravelle, M., King, M., Wöppelmann, G. (2015). Levelling co-located GNSS and tide gauge stations using GNSS reflectometry. *Journal of Geodesy*, 89, 241–258.
- Strandberg, J., Hobiger, T., Haas, R. (2016). Improving GNSS-R sea level determination through inverse modeling of SNR data. *Radio Science*, 51, 1286–1296.
- Williams, S. D. P., Nievinski, F. G. (2017). Tropospheric delays in ground-based GNSS multipath reflectometry — experimental evidence from coastal sites. *Journal of Geophysical Research: Solid Earth*, 122, 2310–2327.
- gnssrefl documentation: https://gnssrefl.readthedocs.io
