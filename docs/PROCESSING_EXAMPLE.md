# Worked example: Friday Harbor (SC02), March 2024

This example goes from nothing to a results page for one station. SC02 is used because anyone can get its data, a NOAA tide gauge with a wind sensor is about 300 m away, and published results exist to compare with (Larson, Ray and Williams 2017: about 12 cm for one height, about 2 cm for a daily mean, GPS L1 only).

All numbers below are from our own run of these steps for 2024 days 060 to 091.

Set `REFL_CODE`, `ORBITS` and `EXE` first (see [Installation](INSTALL.md)).

## 1. Station file

The settings are those of the gnssrefl documentation for this station: heights 3 to 12 m, elevation 5 to 13 degrees, azimuth 50 to 240 degrees.

```bash
gnssir_input sc02 -h1 3 -h2 12 -e1 5 -e2 13 -azlist2 50 240 -allfreq True
```

The repository already holds the result as `config/sc02.json`. Copy it to `$REFL_CODE/input/sc02.json`.

## 2. Data and SNR files

```bash
python -m scripts.earthscope_rinex3 --station sc02 --year 2024 --doy 60 --doy_end 91 --keep downloads/sc02
```

This downloads the 15-second multi-GNSS RINEX 3 files (about 8 MB a day) and makes one SNR file per day. It takes about 7 seconds a day.

## 3. Heights

One job for the whole period. Do not run several `gnssir` jobs on one station at once.

```bash
gnssir sc02 2024 60 -doy_end 91
```

Expect about 510 heights a day (about 350 without BeiDou).

## 4. Station checks

```bash
python -m scripts.station_check --station sc02 --year 2024 --doy 60 --doy_end 91 --rinex path/to/one_unpacked_file.rnx
```

| Check | Our result |
|---|---|
| Sampling | 15 s resolves heights to 22 m; the height window ends at 12 m |
| L2 signal in the file | L2C on 14 satellites, legacy signal on 16 |
| Scatter of the heights about a smooth curve | 0.10 m |

The last number needs no gauge. Against the gauge the scatter of one height is also 0.10 m.

## 5. Smooth water level series

```bash
python -m scripts.inverse_heights --station sc02 --year 2024 --doy 60 --doy_end 91 --out sc02_inverse.csv
```

This gives a value every 5 minutes. Against the gauge its error is 1.8 cm (RMS). The gnssrefl `subdaily` spline through the single heights gives 2.6 cm for the same times.

## 6. Where the reflections come from

```bash
python -m scripts.visualizer.periodogram_circle --station sc02 --year 2024 --doy 70 --freqs 1,20,5
```

The figure has two rows of circles. "Distance from the water level" is first here, because the water moved by 2.9 m on that day. Good arcs lie on the dashed ring. A strong streak near azimuth 20 degrees does not follow the ring: it is a fixed reflector, outside the azimuth mask.

## 7. Gauge, wind and the results page

Download the gauge and wind data from NOAA station 9449880 (water level and wind, 6-minute values, at most 31 days per request). Save each as a CSV file with a header: `time,level_m` and `time,speed`.

Optional, for the plot of the peak-to-noise limit: a second run with the limits off. Copy `$REFL_CODE/input/sc02.json` to `$REFL_CODE/input/sc02.open.json`, set every value of `reqAmp` to 0 and `PkNoise` to 0, and run:

```bash
gnssir sc02 2024 60 -doy_end 91 -extension open
```

Then write a page file (start from `docs/station_page_example.yaml`) and build the page:

```bash
python -m scripts.station_page --spec my_page.yaml --out sc02_results.html
```

## What to expect

| Quantity | Value at SC02 |
|---|---|
| Heights a day | about 510 with BeiDou, about 350 without |
| Scatter of one height against the gauge | 0.10 m |
| Slope against the gauge, refraction correction on | 0.998 |
| Slope with no refraction correction | 0.985 |
| Inverse method, 5-minute values | 1.8 cm |
| Daily mean, all signals | 0.8 cm |
| Daily mean, GPS L1 only | 2.5 cm |
| Wind up to 8 m/s | no loss of arcs or precision; the L1-band heights move by about 5 cm |

Friday Harbor is an easy station. At a station with short arcs at very low elevation (Gustavus, Alaska: 2 to 5 degrees) one height scatters by 0.20 m, the peak-to-noise limit decides the result, and the inverse method needs a wider elevation range than the periodogram. See [Parameters](params_reference.md).
