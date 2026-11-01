"""The Live speed indicator uses completed frames over wall time, not the cheap cost of a cached frame."""

from collections import deque

from courtvision.server.studio_game import StudioGame


def meter():
    game = object.__new__(StudioGame)
    game._rate_start, game._rate_times = None, deque()
    return game


def test_cached_frame_cost_does_not_inflate_the_live_rate():
    game = meter()
    for idx in range(120):
        start = idx / 30
        fps = game._analysis_fps(start, start + (0.001 if idx % 2 else 0.010))
    assert 29 <= fps <= 31  # varying 1-10 ms processing at a real 30 Hz cadence
    assert len(game._rate_times) <= 61


def test_a_long_processing_stall_is_visible_in_the_speed_indicator():
    game = meter()
    game._analysis_fps(0.0, 0.01)
    assert game._analysis_fps(0.02, 3.0) == 0.5
