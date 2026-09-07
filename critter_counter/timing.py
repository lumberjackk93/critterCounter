"""Time-remaining estimates for long runs.

Deliberately based on the average rate over the whole run rather than a recent-rate
estimate. The underlying tools print their own smoothed ETA and it is badly wrong -
observed claiming 6.5 hours remaining on a job that finished in 82 minutes - because a
brief slow patch skews a short window. Averaging over the run is steadier, and the
overall progress fraction is already weighted by how long each stage typically takes.
"""

from __future__ import annotations

from typing import Optional

# Below this, the estimate is dominated by start-up costs and swings wildly.
MINIMUM_FRACTION = 0.01

# How much a new estimate moves the displayed one. Low enough that the number does not
# jitter, high enough to follow a genuine change in pace.
SMOOTHING = 0.25


def estimate_remaining_seconds(elapsed_seconds: float, fraction: float) -> Optional[float]:
    """Seconds left, or None while it is too early to say anything useful."""

    if elapsed_seconds <= 0 or fraction < MINIMUM_FRACTION or fraction >= 1.0:
        return None
    return elapsed_seconds * (1.0 - fraction) / fraction


def smooth_estimate(previous: Optional[float], latest: Optional[float]) -> Optional[float]:
    if latest is None:
        return previous
    if previous is None:
        return latest
    return previous + SMOOTHING * (latest - previous)


def format_duration(seconds: Optional[float]) -> str:
    """Human phrasing, rounded to avoid implying precision that isn't there."""

    if seconds is None:
        return "estimating time remaining..."
    if seconds < 90:
        return "less than a minute remaining"

    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"about {minutes} minutes remaining"

    hours, remaining_minutes = divmod(minutes, 60)
    hour_word = "hour" if hours == 1 else "hours"
    if remaining_minutes == 0:
        return f"about {hours} {hour_word} remaining"
    return f"about {hours} {hour_word} {remaining_minutes} min remaining"
