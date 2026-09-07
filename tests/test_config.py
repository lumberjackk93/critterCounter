"""Tests for settings persistence.

The settings screen writes through this, and the pipeline reads its values, so a
missing key or a dropped override silently changes how photos get sorted.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "critter_counter"))

from config import DEFAULTS, load_config, save_config  # noqa: E402


def test_defaults_used_when_no_file(tmp_path):
    config = load_config(tmp_path / "missing.json")
    assert config == DEFAULTS


def test_round_trip(tmp_path):
    path = tmp_path / "config.json"
    config = load_config(path)
    config["state"] = "TX"
    config["species_confidence_threshold"] = 0.55
    save_config(config, path)

    assert load_config(path)["state"] == "TX"
    assert load_config(path)["species_confidence_threshold"] == 0.55


def test_missing_keys_fall_back_to_defaults(tmp_path):
    """An older config file must not crash a newer build."""

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"state": "MT"}), encoding="utf-8")

    config = load_config(path)
    assert config["state"] == "MT"
    for key in DEFAULTS:
        assert key in config


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "config.json"
    save_config(dict(DEFAULTS), path)
    assert path.exists()


def test_pipeline_reads_every_configured_key():
    """Guards against adding a setting the pipeline never actually consumes."""

    run_card = (
        Path(__file__).resolve().parent.parent / "critter_counter" / "run_card.py"
    ).read_text(encoding="utf-8")
    for key in DEFAULTS:
        assert f'config["{key}"]' in run_card, f"{key} is never read"
