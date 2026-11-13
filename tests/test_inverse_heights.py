# ABOUTME: Tests for the helpers of the inverse-method wrapper (no gnssrefl run needed).
import numpy as np

from scripts.inverse_heights import supported, whole_day_segments, widest_azimuth_range


def test_widest_azimuth_range():
    assert widest_azimuth_range([0, 45, 270, 360]) == (270, 360)
    assert widest_azimuth_range([0, 360]) == (0, 360)


def test_segments_split_at_small_and_missing_days():
    sizes = {d: 100 for d in range(60, 70)}
    sizes[63] = 40      # incomplete day
    del sizes[67]       # missing day
    assert whole_day_segments(sizes, 60, 69) == [(60, 62), (64, 66), (68, 69)]
    assert whole_day_segments({}, 60, 69) == []


def test_supported_needs_an_arc_on_both_sides():
    arcs = np.array([1.00, 1.05, 1.10])
    t = np.array([0.99, 1.02, 1.07, 1.20])
    ok = supported(t, arcs, hours=1.5)
    assert ok.tolist() == [False, True, True, False]
    assert not supported(t, [], 1.5).any()
