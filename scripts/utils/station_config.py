# ABOUTME: One loader for station settings: short stations.yaml + generated reference catalog.
# ABOUTME: Returns the full per-station dictionary that the rest of the workflow reads.
"""Station settings.

Three files hold what the workflow knows about a station:

    config/stations.yaml            a few lines per station, edited by hand
    config/<station>.json           the gnssrefl settings file (owned by gnssrefl)
    config/references/<STA>.json    reference catalog, written by find_reference_stations

A station entry in stations.yaml:

    GLBX:
      gnssrefl_json: config/glbx.json
      data_source: nps                 # nps (default), earthscope, nrcan, ...
      orbit_type: gnss                 # optional
      reference: erddap:edu_ua_glos_obs_station_bartlett_cove_ak

`reference` is the one choice a user makes: `usgs:<site>`, `coops:<id>`,
`coops:auto`, `erddap:<dataset_id>` or `none`. Everything else about the
references (names, distances, datums, servers) is in the catalog file.

The station position comes from the gnssrefl file, so it is stored once.
"""
import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATIONS_PATH = PROJECT_ROOT / "config" / "stations.yaml"
REFERENCES_DIR = PROJECT_ROOT / "config" / "references"

# Defaults that used to be repeated for every station
USGS_DEFAULTS = {
    "search_radius_km": 20,
    "priority_param_codes": ["00065", "62610", "62611", "62620"],
    "target_usgs_site": None,
    "usgs_parameter_code_to_use": None,
    "usgs_gauge_stated_datum": None,
}
COOPS_DEFAULTS = {
    "enabled": False,
    "search_radius_km": 100,
    "preferred_stations": [],
    "data_products": ["predictions", "water_level"],
    "datum": "NAVD88",
    "time_zone": "gmt",
    "interval": "h",
}


def parse_reference(text: Optional[str]) -> Tuple[str, Optional[str]]:
    """'usgs:02043433' -> ('usgs', '02043433'); None or 'none' -> ('none', None)."""
    if not text or str(text).strip().lower() == "none":
        return "none", None
    source, _, ref_id = str(text).strip().partition(":")
    source = source.lower()
    if source not in ("usgs", "coops", "erddap"):
        raise ValueError(
            f"Unknown reference source '{source}' in '{text}'. Use usgs, coops, erddap or none."
        )
    if not ref_id:
        raise ValueError(
            f"Reference '{text}' has no id. Write for example usgs:02043433 or coops:auto."
        )
    return source, ref_id


def load_catalog(station: str, references_dir: Path = REFERENCES_DIR) -> Dict[str, Any]:
    """Reference catalog of one station, or an empty dictionary."""
    path = Path(references_dir) / f"{station.upper()}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _expand(
    station: str, entry: Dict[str, Any], root: Path, references_dir: Path
) -> Dict[str, Any]:
    """Build the full station dictionary from the short entry, the gnssrefl file and the catalog."""
    gnssrefl_json = entry.get("gnssrefl_json") or f"config/{station.lower()}.json"
    full: Dict[str, Any] = {
        "station_id_4char_lower": station.lower(),
        "gnssir_json_params_path": gnssrefl_json,
        "reference_gauge_id": None,
    }
    json_path = root / gnssrefl_json
    if json_path.exists():
        with open(json_path) as f:
            g = json.load(f)
        full["latitude_deg"] = g.get("lat")
        full["longitude_deg"] = g.get("lon")
        full["ellipsoidal_height_m"] = g.get("ht")
    for key in ("data_source", "orbit_type", "inverse_elevation", "rinex3_name"):
        if entry.get(key) is not None:
            full[key] = entry[key]

    catalog = load_catalog(station, references_dir)
    usgs = {**USGS_DEFAULTS, **copy.deepcopy(catalog.get("usgs_comparison", {}))}
    external = copy.deepcopy(catalog.get("external_data_sources", {}))
    if "noaa_coops" in external:
        external["noaa_coops"] = {**COOPS_DEFAULTS, **external["noaa_coops"]}

    source, ref_id = parse_reference(entry.get("reference"))
    full["reference"] = {"source": source, "id": ref_id}
    if source == "usgs":
        usgs["target_usgs_site"] = ref_id
    elif source == "coops":
        coops = external.setdefault("noaa_coops", dict(COOPS_DEFAULTS))
        coops["enabled"] = True
        if ref_id != "auto":
            others = [s for s in coops.get("preferred_stations", []) if s != ref_id]
            coops["preferred_stations"] = [ref_id] + others
    elif source == "erddap":
        erddap = external.get("erddap")
        if not erddap or erddap.get("dataset_id") != ref_id:
            raise ValueError(
                f"{station}: reference erddap:{ref_id} is not in "
                f"config/references/{station.upper()}.json. "
                f"Run: python scripts/find_reference_stations.py --station {station.upper()} "
                "--update-config"
            )
        erddap["enabled"] = True
        erddap["primary_reference"] = True
    if source != "erddap" and "erddap" in external:
        external["erddap"]["primary_reference"] = False
    full["usgs_comparison"] = usgs
    if external:
        full["external_data_sources"] = external
    return full


@lru_cache(maxsize=8)
def _load_stations_cached(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    if path.suffix == ".json":  # old single-file format, read as it is
        with open(path) as f:
            return json.load(f)
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    root = path.resolve().parent.parent
    references_dir = path.resolve().parent / "references"
    return {
        str(name).upper(): _expand(str(name), entry or {}, root, references_dir)
        for name, entry in raw.items()
    }


def load_stations(config_path=None) -> Dict[str, Any]:
    """All stations as full dictionaries, keyed by upper-case station name."""
    path = Path(config_path) if config_path else DEFAULT_STATIONS_PATH
    return copy.deepcopy(_load_stations_cached(str(path)))


def load_station(station: str, config_path=None) -> Optional[Dict[str, Any]]:
    """One station as a full dictionary, or None if it is not in the file."""
    return load_stations(config_path).get(station.upper())


def clear_cache() -> None:
    _load_stations_cached.cache_clear()


def write_catalog(
    station: str, catalog: Dict[str, Any], references_dir: Path = REFERENCES_DIR
) -> Path:
    """Write the reference catalog of one station (used by find_reference_stations)."""
    Path(references_dir).mkdir(parents=True, exist_ok=True)
    path = Path(references_dir) / f"{station.upper()}.json"
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")
    clear_cache()
    return path


def set_reference(station: str, reference: str, config_path=None) -> None:
    """Set the `reference:` line of one station in stations.yaml, keeping comments."""
    parse_reference(reference)
    path = Path(config_path) if config_path else DEFAULT_STATIONS_PATH
    lines = path.read_text().splitlines()
    out, in_station, done = [], False, False
    for line in lines:
        # a station header: "NAME:" at the start of a line, with or without a comment after it
        top = bool(re.match(r"^[A-Za-z0-9_]+:\s*(#.*)?$", line))
        if top:
            if in_station and not done:
                out.append(f"  reference: {reference}")
                done = True
            in_station = line.split(":")[0].strip().upper() == station.upper()
        if in_station and line.strip().startswith("reference:"):
            line, done = f"  reference: {reference}", True
        out.append(line)
    if in_station and not done:
        out.append(f"  reference: {reference}")
        done = True
    if not done:
        raise ValueError(f"Station {station} is not in {path}")
    path.write_text("\n".join(out) + "\n")
    clear_cache()
