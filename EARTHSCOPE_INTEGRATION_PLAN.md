# EarthScope/UNAVCO RINEX API Integration Plan

## Objective

Add support for downloading RINEX observation files from the **EarthScope (formerly UNAVCO) GAGE archive** alongside the existing NPS GNSS archive. This enables processing of four Greenland GNSS-IR stations: **LRSK, NIAQ, UMNQ, NKAR** (part of the NSF-funded Greenland Hazards GNSS Network).

---

## Background

### Current Architecture
- **RINEX download** is handled by `scripts/utils/data_manager.py`
- It pulls RINEX 3 `.XXo` files from `https://gnss.nps.gov/doi-gnss/Rinex/{year}/{doy}/{station}/{station}{doy}0.{yy}o`
- No authentication is required for NPS
- Downloaded RINEX 3 files are converted to RINEX 2.11 via `gfzrnx`, then fed into `rinex2snr` -> `gnssir`

### EarthScope GAGE Archive
- **Base URL**: `https://gage-data.earthscope.org/archive/gnss/rinex3/obs/`
- **URL pattern**: `https://gage-data.earthscope.org/archive/gnss/rinex3/obs/{year}/{doy:03d}/{station_lower}{doy:03d}0.{yy}d.gz`
- Files are **Hatanaka-compressed** (`.crx` / `.d`) and **gzipped** (`.gz`), unlike NPS which serves uncompressed RINEX 3 `.XXo`
- **Authentication**: Requires an EarthScope bearer token via `Authorization: Bearer <token>` header
- Token is obtained by creating a free account at https://data-idm.unavco.org/ and generating a data access token

### Target Stations
| Station | Location | Lat | Lon | Approx Height |
|---------|----------|-----|-----|----------------|
| UMNQ | Uummannaq, Greenland | 70.6776 | -52.1154 | 37.0 m |
| LRSK | Greenland | TBD | TBD | TBD |
| NIAQ | Greenland | TBD | TBD | TBD |
| NKAR | Nuuk, Greenland | TBD | TBD | TBD |

**Note**: UMNQ is already partially configured in `config/stations_config.json`. The other 3 stations need their coordinates determined (check the RINEX file headers or EarthScope station metadata API).

---

## Implementation Steps

### Step 1: Add Hatanaka Decompression to Preprocessor

**File**: `scripts/external_tools/preprocessor.py`

Add a new function `decompress_hatanaka()` that:
1. Gunzips the `.d.gz` file to `.d` (or `.crx`)
2. Runs `CRX2RNX` (Hatanaka decompression tool) to convert `.d` -> `.XXo` (RINEX 3 observation)
3. Returns the path to the decompressed RINEX 3 file

```python
def decompress_hatanaka(compressed_file_path, output_dir, crx2rnx_exe_path="CRX2RNX"):
    """
    Decompress a Hatanaka-compressed RINEX file (.d.gz or .crx.gz).

    Steps:
      1. gunzip -> .d file
      2. CRX2RNX -> .XXo RINEX 3 observation file

    Args:
        compressed_file_path: Path to .d.gz or .crx.gz file
        output_dir: Directory for output RINEX 3 file
        crx2rnx_exe_path: Path to CRX2RNX executable

    Returns:
        Path to decompressed RINEX 3 .XXo file, or None on failure
    """
```

**Dependencies**: The `CRX2RNX` tool must be installed. It's available from https://terras.gsi.go.jp/ja/crx2rnx.html or via conda (`conda install -c conda-forge hatanaka`). Add `crx2rnx_path` to `config/tool_paths.json`.

### Step 2: Add EarthScope Download Function to Data Manager

**File**: `scripts/utils/data_manager.py`

Add a new function `download_rinex_earthscope()` and a dispatcher:

```python
EARTHSCOPE_BASE_URL = "https://gage-data.earthscope.org/archive/gnss/rinex3/obs"
EARTHSCOPE_PATH_PATTERN = "{year}/{doy:03d}/{station_lower}{doy:03d}0.{yy}d.gz"

def download_rinex_earthscope(station: str, year: int, doy: int, target_path: Path) -> bool:
    """
    Download Hatanaka-compressed RINEX from EarthScope GAGE archive.

    Requires EARTHSCOPE_TOKEN environment variable to be set.
    Downloads .d.gz file (Hatanaka compressed + gzipped).

    Args:
        station: 4-character station ID
        year: 4-digit year
        doy: Day of year
        target_path: Local path to save the .d.gz file

    Returns:
        bool: True if download successful
    """
    token = os.environ.get("EARTHSCOPE_TOKEN")
    if not token:
        logging.error("EARTHSCOPE_TOKEN environment variable not set")
        return False

    station_lower = station.lower()
    yy = str(year)[-2:]
    path = EARTHSCOPE_PATH_PATTERN.format(year=year, doy=doy, station_lower=station_lower, yy=yy)
    url = f"{EARTHSCOPE_BASE_URL}/{path}"

    headers = {"Authorization": f"Bearer {token}"}
    return download_from_url(url, target_path, headers=headers)
```

Also modify `download_from_url()` to accept an optional `headers` parameter:

```python
def download_from_url(url: str, target_path: Path, headers: dict = None) -> bool:
    # ... existing code ...
    response = requests.get(url, stream=True, timeout=120, headers=headers)
    # ... rest unchanged ...
```

### Step 3: Add Data Source Routing to Station Config

**File**: `config/stations_config.json`

Add a `"data_source"` field to each station config to specify which archive to use:

```json
{
  "UMNQ": {
    "station_id_4char_lower": "umnq",
    "data_source": "earthscope",
    ...
  },
  "FORA": {
    "station_id_4char_lower": "fora",
    "data_source": "nps",
    ...
  }
}
```

Default behavior: if `data_source` is absent, assume `"nps"` (backward-compatible).

### Step 4: Update the Daily Worker to Handle Both Sources

**File**: `scripts/core_processing/daily_gnssir_worker.py`

The `process_single_day()` function currently does:
1. Download RINEX 3 `.XXo` from NPS
2. Convert RINEX 3 -> RINEX 2.11 via gfzrnx
3. Run rinex2snr
4. Run gnssir
5. Run quickLook

For EarthScope stations, Step 1 becomes:
1. Download `.d.gz` from EarthScope (with auth)
2. Decompress: gunzip -> Hatanaka CRX2RNX -> RINEX 3 `.XXo`

Then Steps 2-5 remain identical.

**Changes needed in `process_single_day()`**:

```python
# In Step 1, replace the direct download_rinex call with:
data_source = station_config.get("data_source", "nps")

if data_source == "earthscope":
    # Download Hatanaka-compressed file
    hatanaka_gz_path = rinex3_dir / f"{station_id.lower()}{doy_padded}0.{yy}d.gz"

    if not download_rinex_earthscope(station_id, year, doy, hatanaka_gz_path):
        # handle error
        return result

    # Decompress Hatanaka -> RINEX 3
    crx2rnx_exe = tool_paths.get("crx2rnx_path", "CRX2RNX")
    rinex3_from_hatanaka = decompress_hatanaka(hatanaka_gz_path, rinex3_dir, crx2rnx_exe)

    if rinex3_from_hatanaka is None:
        # handle error
        return result

    # Rename/move to expected RINEX 3 path if needed
    if rinex3_from_hatanaka != rinex3_file_path:
        shutil.move(str(rinex3_from_hatanaka), str(rinex3_file_path))

else:  # "nps" (default)
    if not download_rinex(station_id, year, doy, rinex3_file_path):
        # existing error handling
        return result
```

### Step 5: Update Tool Paths Config

**File**: `config/tool_paths.json`

Add the CRX2RNX tool path:

```json
{
    "gfzrnx_path": "gfzrnx",
    "rinex2snr_path": "rinex2snr",
    "gnssir_path": "gnssir",
    "quicklook_path": "quickLook",
    "crx2rnx_path": "CRX2RNX"
}
```

Also update `scripts/core_processing/config_loader.py` `load_tool_paths()` to include the default for `crx2rnx_path`.

### Step 6: Create GNSS-IR Parameter Files for New Stations

Create config JSON files for each new station. These require careful tuning but start with reasonable defaults:

**File**: `config/umnq.json` (UMNQ already has a config entry but no params file)

```json
{
    "station": "umnq",
    "lat": 70.67755,
    "lon": -52.115436,
    "ht": 36.9963,
    "minH": 3.0,
    "maxH": 15.0,
    "e1": 5.0,
    "e2": 15.0,
    "NReg": [3.0, 15.0],
    "PkNoise": 2.5,
    "polyV": 4,
    "pele": [5.0, 30],
    "ediff": 2.0,
    "desiredP": 0.005,
    "azval2": [0.0, 360.0],
    "freqs": [1, 20, 5, 101, 102, 201, 205, 206, 207, 208],
    "reqAmp": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "refraction": true,
    "overwriteResults": true,
    "seekRinex": false,
    "wantCompression": false,
    "plt_screen": false,
    "onesat": null,
    "screenstats": false,
    "pltname": "umnq_lsp.png",
    "delTmax": 75.0,
    "gzip": false,
    "ellist": []
}
```

**Important**: The `azval2`, `minH`, `maxH`, `e1`, `e2` values are placeholders. These MUST be tuned based on the station's physical environment (distance to water, surrounding terrain). Use `quickLook` results from initial runs to calibrate. For the other 3 stations (LRSK, NIAQ, NKAR), get coordinates from EarthScope station metadata or RINEX file headers first.

Similarly create `config/lrsk.json`, `config/niaq.json`, `config/nkar.json` with placeholder parameters.

### Step 7: Add Station Entries to stations_config.json

Add entries for LRSK, NIAQ, and NKAR (UMNQ already exists). Each needs:

```json
{
  "LRSK": {
    "station_id_4char_lower": "lrsk",
    "data_source": "earthscope",
    "ellipsoidal_height_m": null,
    "latitude_deg": null,
    "longitude_deg": null,
    "gnssir_json_params_path": "config/lrsk.json",
    "reference_gauge_id": null,
    "usgs_comparison": {
      "search_radius_km": 100,
      "target_usgs_site": null,
      "usgs_parameter_code_to_use": "00065",
      "usgs_gauge_stated_datum": "Unknown"
    }
  }
}
```

**Important**: Coordinates (`latitude_deg`, `longitude_deg`, `ellipsoidal_height_m`) are critical and must be filled in. You can get them from:
1. EarthScope station metadata API: `https://gage-data.earthscope.org/archive/gnss/stations/{station}`
2. Download one RINEX file and read the header
3. The gnssrefl `query_unr` command

Also add `"data_source": "earthscope"` to the existing UMNQ entry.

### Step 8: Update Environment/Auth Setup

**File**: `scripts/run_daily.sh`

Add EarthScope token to the environment:

```bash
# EarthScope authentication
export EARTHSCOPE_TOKEN="${EARTHSCOPE_TOKEN:-}"
if [ -z "$EARTHSCOPE_TOKEN" ]; then
    echo "WARNING: EARTHSCOPE_TOKEN not set. EarthScope stations will fail." >> "$LOG_DIR/daily_$DATE.log"
fi
```

Also document in a `.env.example` file:
```
# EarthScope/UNAVCO data access token
# Get yours at: https://data-idm.unavco.org/
EARTHSCOPE_TOKEN=your_token_here
```

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `scripts/utils/data_manager.py` | **MODIFY** | Add `download_rinex_earthscope()`, add `headers` param to `download_from_url()` |
| `scripts/external_tools/preprocessor.py` | **MODIFY** | Add `decompress_hatanaka()` function |
| `scripts/core_processing/daily_gnssir_worker.py` | **MODIFY** | Route download based on `data_source`, add Hatanaka decompression step |
| `scripts/core_processing/config_loader.py` | **MODIFY** | Add `crx2rnx_path` default to `load_tool_paths()` |
| `config/tool_paths.json` | **MODIFY** | Add `crx2rnx_path` entry |
| `config/stations_config.json` | **MODIFY** | Add `data_source` field to UMNQ, add LRSK/NIAQ/NKAR entries |
| `config/umnq.json` | **CREATE** | GNSS-IR params for Uummannaq |
| `config/lrsk.json` | **CREATE** | GNSS-IR params for LRSK (placeholder) |
| `config/niaq.json` | **CREATE** | GNSS-IR params for NIAQ (placeholder) |
| `config/nkar.json` | **CREATE** | GNSS-IR params for NKAR (placeholder) |
| `.env.example` | **CREATE** | Document required environment variables |
| `scripts/run_daily.sh` | **MODIFY** | Add EARTHSCOPE_TOKEN handling |

---

## Testing Checklist

1. **Unit test**: `download_rinex_earthscope()` with mock responses
2. **Unit test**: `decompress_hatanaka()` with a sample `.d.gz` file
3. **Integration test**: Download + decompress + convert for UMNQ (requires valid token)
4. **Regression test**: Existing NPS stations (FORA, GLBX, etc.) still work unchanged
5. **Test**: Stations without `data_source` field default to NPS (backward compat)
6. **Test**: Missing `EARTHSCOPE_TOKEN` gives clear error message

---

## Prerequisites / Environment Setup

Before running the new stations:

1. **Install CRX2RNX**: `conda install -c conda-forge hatanaka` or download from https://terras.gsi.go.jp/ja/crx2rnx.html
2. **Get EarthScope token**: Create account at https://data-idm.unavco.org/, generate API token
3. **Set environment variable**: `export EARTHSCOPE_TOKEN=<your_token>`
4. **Verify station availability**: Test that each station has data in the archive for your target date range by attempting a single download
5. **Determine station coordinates**: For LRSK, NIAQ, NKAR - get precise coordinates from RINEX headers or EarthScope metadata

---

## Important Notes

- **EarthScope files use Hatanaka compression** (`.d` extension = compact RINEX). This is NOT the same as standard RINEX 3 (`.XXo`). The decompression step (CRX2RNX) is essential.
- **The RINEX 3 filename conventions differ** between NPS (`STATION{doy}0.{yy}o`) and EarthScope (`station{doy}0.{yy}d.gz`). NPS uses uppercase station IDs, EarthScope uses lowercase.
- **All GNSS-IR params files are placeholders** until you can run `quickLook` on actual data and tune the reflection zone parameters.
- **Greenland stations are at high latitude** (~60-70 N) which affects satellite geometry. You may need wider azimuth windows and different elevation angle ranges than the mid-latitude stations.
- **The existing pipeline from RINEX 3 -> RINEX 2.11 -> SNR -> gnssir is unchanged**. The only new code is in the download + decompression layer.
