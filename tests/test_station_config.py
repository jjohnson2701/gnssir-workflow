# ABOUTME: Tests for the station settings loader (stations.yaml + reference catalog).
import json

import pytest

from scripts.utils import station_config as sc


@pytest.fixture
def project(tmp_path):
    cfg = tmp_path / "config"
    (cfg / "references").mkdir(parents=True)
    (cfg / "abcd.json").write_text(json.dumps({"lat": 10.5, "lon": -20.25, "ht": 3.0}))
    (cfg / "references" / "ABCD.json").write_text(
        json.dumps(
            {
                "usgs_comparison": {"target_usgs_site": "111", "usgs_gauge_stated_datum": "NAVD88"},
                "external_data_sources": {
                    "noaa_coops": {"enabled": False, "preferred_stations": ["900", "901"]},
                    "erddap": {
                        "enabled": False,
                        "dataset_id": "ds1",
                        "base_url": "https://example.org/erddap",
                    },
                },
            }
        )
    )
    path = cfg / "stations.yaml"
    path.write_text(
        "# comment kept\nABCD:\n  gnssrefl_json: config/abcd.json\n  data_source: earthscope\n"
        "  reference: usgs:222\n\nEFGH:\n  reference: none\n"
    )
    sc.clear_cache()
    return path


@pytest.mark.unit
def test_parse_reference():
    assert sc.parse_reference("usgs:02043433") == ("usgs", "02043433")
    assert sc.parse_reference("coops:auto") == ("coops", "auto")
    assert sc.parse_reference(None) == ("none", None)
    assert sc.parse_reference("none") == ("none", None)
    with pytest.raises(ValueError):
        sc.parse_reference("tidegauge:1")
    with pytest.raises(ValueError):
        sc.parse_reference("usgs:")


@pytest.mark.unit
def test_position_comes_from_gnssrefl_file(project):
    st = sc.load_station("abcd", project)
    assert (st["latitude_deg"], st["longitude_deg"], st["ellipsoidal_height_m"]) == (
        10.5,
        -20.25,
        3.0,
    )
    assert st["data_source"] == "earthscope"
    assert st["station_id_4char_lower"] == "abcd"


@pytest.mark.unit
def test_reference_line_overrides_catalog(project):
    st = sc.load_station("ABCD", project)
    assert st["reference"] == {"source": "usgs", "id": "222"}
    assert st["usgs_comparison"]["target_usgs_site"] == "222"
    assert st["usgs_comparison"]["usgs_gauge_stated_datum"] == "NAVD88"
    assert st["usgs_comparison"]["search_radius_km"] == 20  # default from code


@pytest.mark.unit
def test_set_reference_keeps_comments_and_other_stations(project):
    sc.set_reference("ABCD", "coops:901", project)
    text = project.read_text()
    assert "# comment kept" in text and "reference: none" in text
    st = sc.load_station("ABCD", project)
    assert st["reference"] == {"source": "coops", "id": "901"}
    coops = st["external_data_sources"]["noaa_coops"]
    assert coops["enabled"] is True and coops["preferred_stations"] == ["901", "900"]


@pytest.mark.unit
def test_erddap_needs_a_catalog_entry(project):
    sc.set_reference("ABCD", "erddap:ds1", project)
    st = sc.load_station("ABCD", project)
    assert st["external_data_sources"]["erddap"]["primary_reference"] is True
    sc.set_reference("ABCD", "erddap:unknown", project)
    with pytest.raises(ValueError):
        sc.load_station("ABCD", project)


@pytest.mark.unit
def test_station_without_reference(project):
    st = sc.load_station("EFGH", project)
    assert st["reference"] == {"source": "none", "id": None}
    assert sc.load_station("ZZZZ", project) is None
