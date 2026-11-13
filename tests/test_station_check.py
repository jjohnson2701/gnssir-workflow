# ABOUTME: Tests for the sampling check, the L2 signal check and the scatter about a spline.
import numpy as np

from scripts.station_check import WAVELENGTH, l2_signal_check, sampling_check, scatter_about_spline


def _snr(interval, rate_deg_per_s=0.005):
    t = np.arange(0, 3600, interval, dtype=float)
    ele = 5 + rate_deg_per_s * t
    a = np.zeros((len(t), 9))
    a[:, 0], a[:, 1], a[:, 2], a[:, 3] = 7, ele, 100.0, t
    return a


def test_sampling_height_scales_with_interval():
    r1 = sampling_check(_snr(1), 5, 20, [0, 360], 10.0, [1])
    r30 = sampling_check(_snr(30), 5, 20, [0, 360], 10.0, [1])
    h1, h30 = r1["signals"][1]["resolvable_height_m"], r30["signals"][1]["resolvable_height_m"]
    assert abs(h1 / h30 - 30) < 0.5
    # step of sin(e) per second at about 5 deg/1000 s: check against the formula
    step = np.radians(0.005) * np.cos(np.radians(5))
    assert abs(h1 - WAVELENGTH[1] / (4 * step)) / h1 < 0.05
    assert r1["signals"][1]["ok"] and r1["interval_s"] == 1


def test_sampling_flags_slow_data():
    r = sampling_check(_snr(30, rate_deg_per_s=0.008), 5, 20, [0, 360], 60.0, [1])
    assert not r["signals"][1]["ok"]
    assert r["signals"][1]["longest_interval_s"] < 30


def test_sampling_respects_azimuth_mask():
    import pytest

    with pytest.raises(ValueError):
        sampling_check(_snr(1), 5, 20, [200, 300], 10.0, [1])


def _rinex3(tmp_path, values):
    """values: per satellite, (S2W, S2X) strings or None."""
    hdr = [
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        "G    4 C1C S1C S2W S2X                                      SYS / # / OBS TYPES ",
        "R    2 C1C S1C                                              SYS / # / OBS TYPES ",
        "                                                            END OF HEADER       ",
    ]
    body = []
    for _ in range(3):
        body.append("> 2024 02 29 00 00  0.0000000  0  %d" % len(values))
        for sat, (w, x) in values.items():
            f = lambda v: ("%14.3f  " % v) if v is not None else " " * 16
            body.append(sat + f(2.0e7) + f(45.0) + f(w) + f(x))
    p = tmp_path / "test.24o"
    p.write_text("\n".join(hdr + body) + "\n")
    return p


def test_l2_check_finds_l2c(tmp_path):
    r = l2_signal_check(_rinex3(tmp_path, {"G01": (30.0, 42.0), "G02": (28.0, None)}))
    assert r["verdict"] == "l2c"
    assert r["counts"] == {"S2W": 6, "S2X": 3}
    assert r["satellites"] == {"S2W": 2, "S2X": 1}


def test_l2_check_legacy_only(tmp_path):
    r = l2_signal_check(_rinex3(tmp_path, {"G01": (30.0, None), "G02": (28.0, None)}))
    assert r["verdict"] == "legacy_only"


def test_l2_check_rinex2(tmp_path):
    p = tmp_path / "test.24o"
    p.write_text(
        "     2.11           OBSERVATION DATA    G (GPS)             RINEX VERSION / TYPE\n"
        "     5    L1    L2    C1    S1    S2                        # / TYPES OF OBSERV\n"
        "                                                            END OF HEADER\n")
    assert l2_signal_check(p)["verdict"] == "unknown_rinex2"


def test_scatter_recovers_noise_and_rate_term():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 1500))
    w = 2 * np.pi / 0.5175  # tide period in days
    rh_true = 8 + 2.0 * np.sin(w * t)
    rate = 2.0 * w * np.cos(w * t) / 24.0  # m per hour
    edotf = rng.choice([-0.4, 0.4], len(t))  # hours
    rh = rh_true + edotf * rate + rng.normal(0, 0.05, len(t))
    off = scatter_about_spline(t, rh, edotf, rate_correction=False)
    on = scatter_about_spline(t, rh, edotf, rate_correction=True)
    assert on["robust_scatter_m"] < 0.07
    assert off["robust_scatter_m"] > 2 * on["robust_scatter_m"]
    assert np.sqrt(np.mean((on["rh_corrected"] - rh_true) ** 2)) < 0.08


def test_scatter_handles_gaps_and_few_points():
    t = np.r_[np.linspace(0, 1, 100), np.linspace(5, 6, 100)]
    r = scatter_about_spline(t, 3 + 0.01 * np.sin(t), None)
    assert np.isfinite(r["robust_scatter_m"])
    assert np.isnan(scatter_about_spline(np.arange(5.0), np.ones(5))["robust_scatter_m"])
