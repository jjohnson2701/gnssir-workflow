# Greenland Station Imagery Sources

## Problem

Greenland GNSS-IR stations (NKAR, UMNQ, NIAQ, LRSK) lack coverage from
the tile servers (Esri WorldImagery, CartoDB) used by mid-latitude stations
for basemap rendering. Remote Arctic locations need alternative imagery
or elevation data for Fresnel zone visualization.

## Sentinel-2 L2A (Recommended)

10m true-color satellite imagery, freely streamable as Cloud-Optimized GeoTIFFs.

### Access

- **Provider**: AWS `sentinel-cogs` bucket (no auth required)
- **URL pattern**: `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{ZONE}/{BAND}/{SQUARE}/{YEAR}/{MONTH}/{SCENE_ID}/TCI.tif`
- **CRS**: UTM (varies by tile)
- **Format**: COG, directly readable via `/vsicurl/` with GDAL

### Scene Discovery

**AWS Element84 STAC** (covers NKAR/Nuuk area, missing some high-latitude tiles):
```python
import requests
resp = requests.post('https://earth-search.aws.element84.com/v1/search', json={
    'collections': ['sentinel-2-l2a'],
    'intersects': {'type': 'Point', 'coordinates': [lon, lat]},
    'datetime': '2024-06-01/2024-08-31',
    'limit': 5,
    'query': {'eo:cloud_cover': {'lt': 10}}
})
```

**Copernicus Data Space STAC** (complete Sentinel-2 catalog, use for >70°N):
```python
resp = requests.post('https://catalogue.dataspace.copernicus.eu/stac/search', json={
    'collections': ['sentinel-2-l2a'],
    'bbox': [lon-0.5, lat-0.2, lon+0.5, lat+0.2],
    'datetime': '2024-06-01T00:00:00Z/2024-08-31T23:59:59Z',
    'limit': 50,
    'sortby': [{'field': 'properties.eo:cloud_cover', 'direction': 'asc'}],
})
```

### Download (GDAL clip from cloud)

```bash
gdalwarp -te <xmin> <ymin> <xmax> <ymax> -tr 10 10 -r bilinear -co COMPRESS=LZW \
  "/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{ZONE}/{BAND}/{SQUARE}/{YEAR}/{MONTH}/{SCENE_ID}/TCI.tif" \
  results_annual/{STATION}/{station}_sentinel2_10m.tif
```

Bounds are in the tile's UTM CRS. Clip to 5km x 5km centered on station.

### Current Downloads

| Station | MGRS Tile | Scene Date  | Cloud | File |
|---------|-----------|-------------|-------|------|
| NKAR    | 22WDS     | 2024-08-12  | 4.9%  | `nkar_sentinel2_10m.tif` |
| UMNQ    | 22WDD     | 2024-08-06  | 0.2%  | `umnq_sentinel2_10m.tif` |

### Naming Convention

Files are saved as `{station}_sentinel2_10m.tif` in `results_annual/{STATION}/`.
The animation script auto-discovers these (preferring satellite imagery over DEM).

## ArcticDEM (Elevation Data)

### Mosaic vs Strips

**Always use mosaic tiles, never individual strips.** Individual strips have:
- Phantom terrain over water (failed stereo correlation)
- Linear edge artifacts at strip boundaries
- Inconsistent coverage

The mosaic (v4.1) merges many strips with quality filtering.

### Mosaic Access

- **S3 URL pattern**: `https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/{res}m/{row}_{col}/{row}_{col}_{res}m_v4.1_dem.tif`
- **Tile grid**: EPSG:3413, origin (-4000000, -4000000), 100km spacing
- **Subtiles (2m)**: `{row}_{col}_{subrow}_{subcol}_2m_v4.1_dem.tif`
- **Format**: COG, directly readable via `/vsicurl/`

### Tile Lookup

```python
from pyproj import Transformer
t = Transformer.from_crs('EPSG:4326', 'EPSG:3413', always_xy=True)
x, y = t.transform(lon, lat)
col = int((x - (-4000000)) / 100000) + 1
row = int((y - (-4000000)) / 100000) + 1
# NKAR (64.17N, -51.75W) -> tile 12_37, subtile 12_37_2_2
```

### Limitations

ArcticDEM is a Digital Surface Model (DSM) — includes buildings and structures.
In urban areas (e.g., Nuuk), buildings obscure the actual coastline. This makes
elevation-based water detection unreliable near settlements. Sentinel-2 imagery
is strongly preferred for coastal urban stations.

## Copernicus DEM GLO-30

30m global DEM, reliable but coarse. Useful as a fallback.

- **URL**: `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat}_00_W{lon}_00_DEM/Copernicus_DSM_COG_10_N{lat}_00_W{lon}_00_DEM.tif`
- **CRS**: EPSG:4326
- **Format**: COG, no auth required

## Notes

- Sentinel-2 scenes should be from summer (Jun-Aug) for snow-free, ice-free views
- The AWS Element84 STAC catalog has gaps above ~70°N; use Copernicus STAC for
  those latitudes
- Landsat Collection 2 exists but requires USGS authentication for data access
