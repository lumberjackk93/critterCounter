"""Persisted user settings for Critter Counter."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".critter_counter" / "config.json"

DEFAULTS = {
    "output_folder": str(Path.home() / "Documents" / "Good Game Pics"),
    "species_confidence_threshold": 0.7,
    "event_gap_seconds": 10,
    # Location narrows which species are considered plausible. Setting the state
    # (a two-letter code like "TX") on top of the country measurably reduces photos
    # that land in the Unknown bucket.
    "country": "USA",
    "state": "",
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        return dict(DEFAULTS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def save_config(config: dict, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
