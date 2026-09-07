"""Tests for the time-remaining estimate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "critter_counter"))

import timing  # noqa: E402


def test_no_estimate_before_meaningful_progress():
    """Early on the number is dominated by start-up and would swing wildly."""

    assert timing.estimate_remaining_seconds(5, 0.0) is None
    assert timing.estimate_remaining_seconds(5, 0.001) is None
    assert timing.estimate_remaining_seconds(0, 0.5) is None


def test_estimate_extrapolates_from_average_rate():
    # A quarter done after 10 minutes implies 30 minutes left.
    assert timing.estimate_remaining_seconds(600, 0.25) == 1800


def test_no_estimate_once_finished():
    assert timing.estimate_remaining_seconds(600, 1.0) is None


def test_smoothing_damps_a_sudden_jump():
    smoothed = timing.smooth_estimate(1000, 2000)
    assert 1000 < smoothed < 2000, "should move toward the new value, not leap to it"


def test_smoothing_takes_the_first_real_estimate_outright():
    assert timing.smooth_estimate(None, 1200) == 1200


def test_smoothing_keeps_last_value_when_there_is_no_new_one():
    assert timing.smooth_estimate(900, None) == 900


def test_formatting_reads_naturally():
    assert timing.format_duration(None) == "estimating time remaining..."
    assert timing.format_duration(30) == "less than a minute remaining"
    assert timing.format_duration(25 * 60) == "about 25 minutes remaining"
    assert timing.format_duration(60 * 60) == "about 1 hour remaining"
    assert timing.format_duration(2 * 60 * 60) == "about 2 hours remaining"
    assert timing.format_duration(2 * 60 * 60 + 15 * 60) == "about 2 hours 15 min remaining"


def test_realistic_full_card_run():
    """Sanity check against the measured 11.4 hour card: 25% done at ~2.9 hours."""

    elapsed = 2.85 * 3600
    remaining = timing.estimate_remaining_seconds(elapsed, 0.25)
    assert 8.0 < remaining / 3600 < 9.0
    assert "hours" in timing.format_duration(remaining)
