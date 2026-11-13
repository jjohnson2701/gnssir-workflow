# ABOUTME: Tests for defects found in the code review of November 2026.
# ABOUTME: Each test fails on the code before the fix.
import numpy as np
import pytest

from scripts.station_check import scatter_about_spline
from scripts.utils import station_config
from scripts.visualizer import station_figures as sf

YAML = """GLBX:
  reference: erddap:a
TOBY:
  reference: coops:1
SC02:   # Friday Harbor
  reference: coops:9449880
"""


def _references(path):
    import yaml

    return {k: v["reference"] for k, v in yaml.safe_load(path.read_text()).items()}


def test_set_reference_does_not_touch_the_next_station(tmp_path):
    p = tmp_path / "stations.yaml"
    p.write_text(YAML)
    station_config.set_reference("TOBY", "none", p)
    assert _references(p) == {"GLBX": "erddap:a", "TOBY": "none", "SC02": "coops:9449880"}


def test_set_reference_finds_a_header_with_a_comment(tmp_path):
    p = tmp_path / "stations.yaml"
    p.write_text(YAML)
    station_config.set_reference("SC02", "none", p)
    assert _references(p)["SC02"] == "none"
    assert "# Friday Harbor" in p.read_text()


def _tide(n=600, seed=1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(100, 103, n))
    rh = 6 + 1.5 * np.sin(2 * np.pi * t / 0.5175) + rng.normal(0, 0.05, n)
    return t, rh


def test_scatter_with_a_missing_edotf_value():
    t, rh = _tide()
    edotf = np.full(len(t), 0.5)
    edotf[10] = np.nan
    out = scatter_about_spline(t, rh, edotf)
    assert np.isfinite(out["robust_scatter_m"])
    assert not out["used"][10]


def test_scatter_reports_which_arcs_were_rejected():
    t, rh = _tide()
    rh[::40] += 2.5
    out = scatter_about_spline(t, rh, np.zeros(len(t)))
    assert not out["used"][::40].any()
    assert out["used"].sum() > 0.9 * len(t)


def test_tuning_table_has_a_row_for_a_limit_off_the_grid():
    t, rh = _tide()
    gauge_t = np.linspace(99, 104, 2000)
    gauge = -1.5 * np.sin(2 * np.pi * gauge_t / 0.5175)
    pk = np.random.default_rng(2).uniform(2, 5, len(t))
    limits = np.unique(np.r_[np.round(np.arange(0, 6.01, 0.2), 1), 2.5])
    rows = sf.tuning_table(t, rh, pk, np.zeros(len(t)), gauge_t, gauge, limits)
    assert sum(abs(r["limit"] - 2.5) < 1e-9 for r in rows) == 1


def test_arcs_outside_the_gauge_span_are_not_used():
    t, rh = _tide()
    gauge_t = np.linspace(100, 101, 500)   # the gauge covers one of three days
    gauge = -1.5 * np.sin(2 * np.pi * gauge_t / 0.5175)
    rows = sf.tuning_table(t, rh, np.full(len(t), 3.0), np.zeros(len(t)), gauge_t, gauge, [0.0])
    assert rows[0]["n"] == int(((t >= 100) & (t <= 101)).sum())
    assert rows[0]["slope"] == pytest.approx(1.0, abs=0.05)


def test_wind_figure_with_no_usable_bin(tmp_path):
    rows = [{"label": "0 to 2", "lo": 0, "n": 5, "share": 1.0, "scatter_m": None, "outliers": None, "pk": None}]
    assert sf.wind_figure(tmp_path / "w.png", rows) is False
    assert not (tmp_path / "w.png").exists()


def test_wind_table_offset_per_signal_removes_a_signal_offset():
    t, rh = _tide(900)
    freq = np.where(np.arange(len(t)) % 2 == 0, 1, 5)
    rh = rh + np.where(freq == 1, 0.10, -0.05)          # two signals, 15 cm apart
    gauge_t = np.linspace(99, 104, 4000)
    gauge = -1.5 * np.sin(2 * np.pi * gauge_t / 0.5175)
    wind = np.full(len(gauge_t), 1.0)
    args = (t, rh, np.full(len(t), 3.0), np.zeros(len(t)), gauge_t, gauge, gauge_t, wind)
    one = sf.wind_table(*args)[0]["scatter_m"]
    per_signal = sf.wind_table(*args, freq=freq)[0]["scatter_m"]
    assert per_signal == pytest.approx(0.05, abs=0.01)
    assert one > 1.4 * per_signal
