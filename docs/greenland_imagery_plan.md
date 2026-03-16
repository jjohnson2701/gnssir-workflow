# Imagery/DEM Acquisition Plan for Greenland GNET Stations

## Context

UMNQ (Uummannaq, 70.68°N) was our first Greenland station. We established a working
pipeline for acquiring remote sensing data in areas where commercial tile servers
(Esri WorldImagery) lack high-resolution coverage. This plan documents the approach
for the remaining Greenland stations.

## Data Sources

### ArcticDEM v4.1 (2m mosaic)
- **Source**: Polar Geospatial Center (PGC) via S3/STAC
- **Resolution**: 2m (also available at 10m and 32m)
- **Coverage**: All of Greenland, derived from stereo satellite imagery
- **CRS**: EPSG:3413 (NSIDC Polar Stereographic North)
- **Access**: Public S3 bucket, no authentication needed
- **STAC endpoint**: `https://stac.pgc.umn.edu/search`
- **Collection**: `arcticdem-mosaics-v4.1-2m`

### Sentinel-2 L2A (10m true color)
- **Source**: Element84 Earth Search (AWS mirror of Copernicus)
- **Resolution**: 10m (TCI band)
- **Coverage**: Global, but Greenland has persistent cloud cover
- **CRS**: UTM (varies by tile)
- **Access**: Public S3 COGs, no authentication needed
- **STAC endpoint**: `https://earth-search.aws.element84.com/v1/search`
- **Collection**: `sentinel-2-l2a`
- **Best window**: June-August for cloud-free scenes

## Step-by-Step Procedure per Station

### 1. Query ArcticDEM tile

```python
import requests, math

def get_arcticdem_url(lat, lon):
    """Query PGC STAC for the ArcticDEM 2m mosaic tile covering a location."""
    bbox = [lon - 0.02, lat - 0.01, lon + 0.02, lat + 0.01]
    resp = requests.post(
        "https://stac.pgc.umn.edu/search",
        json={"collections": ["arcticdem-mosaics-v4.1-2m"], "bbox": bbox, "limit": 1},
    )
    features = resp.json().get("features", [])
    if features:
        return features[0]["assets"]["dem"]["href"]
    return None
```

### 2. Clip ArcticDEM to station area

Convert station coordinates from WGS84 to EPSG:3413 (Polar Stereographic), then
clip a 1km x 1km window centered on the station. Save to
`results_annual/{STATION}/{station}_arcticdem_2m.tif`.

The coordinate conversion formula is in `scripts/create_polar_animation.py` — we
use a manual polar stereo conversion since the py39 environment has a broken
proj.db. Alternatively use GDAL CLI: `gdalwarp -t_srs EPSG:3413`.

### 3. Query cloud-free Sentinel-2 scene

```python
def find_clear_s2(lat, lon, max_cloud=20):
    """Find a low-cloud Sentinel-2 scene near a location."""
    bbox = [lon - 0.05, lat - 0.02, lon + 0.05, lat + 0.02]
    resp = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": "2023-06-01T00:00:00Z/2024-08-31T23:59:59Z",
            "limit": 50,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        },
    )
    features = resp.json().get("features", [])
    clear = [f for f in features if f["properties"].get("eo:cloud_cover", 100) < max_cloud]
    if clear:
        return clear[0]["assets"]["visual"]["href"]  # TCI COG URL
    return None
```

### 4. Clip Sentinel-2 to station area

Use rasterio to read a window from the COG. The S2 TCI is in UTM, so coordinates
must be converted to the appropriate UTM zone first.

Save to `results_annual/{STATION}/{station}_sentinel2_{date}.tif`.

### 5. Generate station context plot

The `umnq_station_context.png` workflow produces a 3-panel figure:
- Sentinel-2 true color with station marker
- ArcticDEM hillshade with land/water classification and reflection zone overlay
- Elevation cross-sections through the reflection zone

### 6. Animation basemap

The animation script (`create_polar_animation.py`) automatically detects and uses
the local DEM file at `results_annual/{STATION}/{station}_arcticdem_2m.tif` when
tile servers lack coverage. The DEM basemap renders as hillshaded terrain with
blue water bodies, using local-meter coordinates aligned with the GNSS-IR data.

## Known Issues

- **EPSG:3413 conversion**: py39's rasterio has a broken proj.db. Use manual
  polar stereographic formulas or GDAL CLI (`gdalwarp`) for coordinate transforms.
- **Esri WorldImagery**: Returns "Map data not yet available" for most Greenland
  locations without throwing an error. The animation script handles this by
  preferring local DEM when available.
- **Sentinel-2 cloud cover**: Greenland has high cloud frequency. Search across
  multiple summers (2-3 years) to find scenes below 20% cloud cover.
- **BeiDou coverage**: At latitudes above ~65°N, BeiDou constellation has minimal
  coverage. Don't configure BeiDou frequencies (302, 306) for Greenland stations.

## File Naming Convention

```
results_annual/{STATION}/
  {station}_arcticdem_2m.tif      # Clipped ArcticDEM (EPSG:3413, 2m)
  {station}_sentinel2_{date}.tif  # Clipped Sentinel-2 TCI (UTM, 10m)
  {station}_station_context.png   # 3-panel context visualization
```
