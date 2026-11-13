# Installation

## What you need

- Linux or macOS, Python 3.10 or newer.
- [gnssrefl](https://github.com/kristinemlarson/gnssrefl) 4.2.3 or newer. It gives the commands `rinex2snr`, `gnssir`, `gnssir_input`, `invsnr`, `subdaily` and `quickLook`.
- `CRX2RNX` (Hatanaka decompression), only for stations whose files come as compact RINEX 3 from EarthScope.

gnssrefl 4 reads RINEX 3 directly. The earlier RINEX 3 to RINEX 2 conversion with `gfzrnx` is no longer used.

## Steps

```bash
conda create -n gnssrefl4 python=3.12
conda activate gnssrefl4
pip install gnssrefl
git clone https://github.com/jjohnson2701/gnssir-workflow.git
cd gnssir-workflow
pip install -r requirements.txt
```

Put `CRX2RNX` in a folder of your choice and point `EXE` to that folder.

## Environment variables

| Variable | Meaning |
|---|---|
| `REFL_CODE` | The gnssrefl workspace. Station files are in `$REFL_CODE/input/`, SNR files in `$REFL_CODE/<year>/snr/<station>/`, results in `$REFL_CODE/<year>/results/<station>/`. |
| `ORBITS` | Where gnssrefl stores orbit files. |
| `EXE` | Folder with `CRX2RNX`. |

`scripts/process_station.py` sets `REFL_CODE` and `ORBITS` itself, to `gnssrefl_data_workspace/` in the repository. Set them yourself when you run the single commands (`station_check`, `inverse_heights`, `periodogram_circle`, `station_page`, `earthscope_rinex3`).

```bash
export REFL_CODE=$PWD/gnssrefl_data_workspace/refl_code
export ORBITS=$PWD/gnssrefl_data_workspace/orbits
export EXE=$HOME/gnss_tools
```

## EarthScope data

EarthScope needs a free account. Log in once; the login is saved in your home folder:

```bash
pip install earthscope-cli
es login
```

## Check the installation

```bash
pytest
gnssir -h
```

Then follow [the worked example](PROCESSING_EXAMPLE.md).

## Problems seen

- **"SNR file not found" for some days, or days missing in the results.** Several `gnssir` jobs ran on one station at the same time. Run one at a time. See the two run rules in the README.
- **A decimation value in the station file has no effect.** `gnssir` uses only the command-line option `-dec`.
- **`rinex2snr` reports success but no SNR file exists.** Seen when two stations translated the same day at once. Run it again.
