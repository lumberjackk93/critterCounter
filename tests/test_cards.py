"""Tests for card identity and history.

Every card mounts as D:, so a drive path cannot tell one from another. These cover the
distinctions the app has to draw before committing to hours of work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "critter_counter"))

import cards  # noqa: E402
from cards import CardIdentity  # noqa: E402


def _identity(serial="AAAA1111", label="MUDDY", photos=27618, fingerprint="abc123"):
    return CardIdentity(
        drive="D:",
        label=label,
        serial=serial,
        photo_count=photos,
        folders=["100MUDDY"],
        fingerprint=fingerprint,
    )


def test_unseen_card_is_reported_as_new(tmp_path):
    summary = cards.describe(_identity(), history={})
    assert "not been processed before" in summary
    assert "27,618" in summary


def test_same_card_unchanged_is_recognised(tmp_path):
    history_file = tmp_path / "cards.json"
    identity = _identity()
    cards.record_run(identity, tmp_path / "session-1", history_file)

    summary = cards.describe(identity, history=cards.load_history(history_file))
    assert "Already processed" in summary
    assert "session-1" in summary


def test_different_card_in_the_same_slot_is_not_confused(tmp_path):
    """The failure this feature exists to prevent."""

    history_file = tmp_path / "cards.json"
    muddy = _identity(serial="AAAA1111", label="MUDDY")
    cards.record_run(muddy, tmp_path / "muddy-session", history_file)

    stealth = _identity(serial="BBBB2222", label="STLTH", photos=11270, fingerprint="zzz")
    summary = cards.describe(stealth, history=cards.load_history(history_file))

    assert "not been processed before" in summary
    assert "muddy-session" not in summary


def test_same_card_with_new_photos_is_flagged_as_changed(tmp_path):
    history_file = tmp_path / "cards.json"
    before = _identity(photos=20000, fingerprint="one")
    cards.record_run(before, tmp_path / "session-1", history_file)

    after = _identity(photos=27618, fingerprint="two")
    summary = cards.describe(after, history=cards.load_history(history_file))

    assert "contents have changed" in summary
    assert "+7,618" in summary


def test_history_survives_multiple_runs(tmp_path):
    history_file = tmp_path / "cards.json"
    identity = _identity()
    cards.record_run(identity, tmp_path / "s1", history_file)
    cards.record_run(identity, tmp_path / "s2", history_file)

    history = cards.load_history(history_file)
    assert len(history[identity.key]["runs"]) == 2


def test_corrupt_history_does_not_crash(tmp_path):
    history_file = tmp_path / "cards.json"
    history_file.write_text("{not json", encoding="utf-8")
    assert cards.load_history(history_file) == {}


def test_identity_falls_back_to_folder_name_without_a_label():
    identity = CardIdentity(
        drive="D:", label="", serial="X", photo_count=5, folders=["100STLTH"]
    )
    assert "100STLTH" in identity.name


def test_key_is_serial_so_it_survives_adding_photos():
    assert _identity(fingerprint="one").key == _identity(fingerprint="two").key
