"""Tests for the pure (non-model) parts of the pipeline.

These cover the logic that silently corrupts results when it's wrong: filename
collisions between camera folders, cache reuse across different cards, event
grouping, and the unknown-species bucketing rules.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "critter_counter"))

import pipeline  # noqa: E402
from pipeline import (  # noqa: E402
    Event,
    ImageResult,
    card_fingerprint,
    copy_good_photos,
    destination_filename,
    group_into_events,
    load_results,
    sanitize_folder_name,
    species_common_name,
)


def test_destination_filename_prefixes_camera_folder():
    source = Path(r"D:\DCIM")
    assert (
        destination_filename(source / "100MUDDY" / "MUD_0001.JPG", source)
        == "100MUDDY_MUD_0001.JPG"
    )
    assert (
        destination_filename(source / "101MUDDY" / "MUD_0001.JPG", source)
        == "101MUDDY_MUD_0001.JPG"
    )


def test_destination_filename_leaves_flat_folders_alone():
    source = Path(r"C:\photos")
    assert destination_filename(source / "MUD_0001.JPG", source) == "MUD_0001.JPG"


def test_destination_filename_handles_path_outside_source():
    assert (
        destination_filename(Path(r"E:\elsewhere\x.JPG"), Path(r"D:\DCIM")) == "x.JPG"
    )


def _fake_image(tmp_path: Path, name: str, species: str, has_detection: bool = True):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpeg")
    return ImageResult(
        filepath=path,
        timestamp=datetime(2026, 5, 11, 18, 37, 44),
        has_detection=has_detection,
        species=species,
        confidence=0.9,
        animal_box_count=1,
    )


def test_same_filename_in_two_camera_folders_does_not_overwrite(tmp_path):
    source = tmp_path / "DCIM"
    a = _fake_image(source, "100MUDDY/MUD_0001.JPG", "white-tailed deer")
    b = _fake_image(source, "101MUDDY/MUD_0001.JPG", "white-tailed deer")

    dest = tmp_path / "out"
    copied = copy_good_photos([Event(images=[a, b])], dest, source)

    assert copied == 2
    names = sorted(p.name for p in (dest / "white-tailed deer").iterdir())
    assert names == ["100MUDDY_MUD_0001.JPG", "101MUDDY_MUD_0001.JPG"]


def test_blank_photos_are_not_copied(tmp_path):
    source = tmp_path / "DCIM"
    blank = _fake_image(source, "100MUDDY/MUD_0002.JPG", "blank", has_detection=False)

    dest = tmp_path / "out"
    assert copy_good_photos([Event(images=[blank])], dest, source) == 0
    assert not dest.exists() or not any(dest.iterdir())


def test_fingerprint_is_stable_and_card_specific(tmp_path):
    source = tmp_path / "DCIM"
    (source / "100MUDDY").mkdir(parents=True)
    first = source / "100MUDDY" / "MUD_0001.JPG"
    first.write_bytes(b"aaaa")

    images = [first]
    original = card_fingerprint(images, source)
    assert original == card_fingerprint(images, source), "must be stable for resume"

    # A different card mounting at the same drive letter must not reuse cached stages.
    first.write_bytes(b"bbbbbbbb")
    swapped_card = card_fingerprint(images, source)
    assert swapped_card != original

    second = source / "100MUDDY" / "MUD_0002.JPG"
    second.write_bytes(b"cccc")
    assert card_fingerprint([first, second], source) != swapped_card


def test_empty_folder_raises_a_readable_error(tmp_path):
    """Picking the wrong folder should explain itself, not stack-trace."""

    empty = tmp_path / "DCIM"
    empty.mkdir()

    with pytest.raises(pipeline.NoPhotosFound) as excinfo:
        pipeline.run_pipeline(empty, tmp_path / "out", tmp_path / "work")

    assert "No JPG photos found" in str(excinfo.value)
    assert not (tmp_path / "out").exists(), "should not create output for an empty card"


def test_group_into_events_splits_on_gap():
    base = datetime(2026, 5, 11, 18, 0, 0)

    def img(offset_seconds):
        return ImageResult(
            filepath=Path(f"x{offset_seconds}.JPG"),
            timestamp=base + timedelta(seconds=offset_seconds),
            has_detection=False,
            species="blank",
            confidence=1.0,
        )

    # 0s, 2s, 4s form one burst; 60s starts a new event.
    events = group_into_events([img(0), img(2), img(4), img(60)], gap_seconds=10)
    assert [len(e.images) for e in events] == [3, 1]


def _write_ensemble(tmp_path: Path, entries) -> Path:
    predictions = []
    for name, prediction, score, detections in entries:
        path = tmp_path / name
        path.write_bytes(b"fake")
        predictions.append(
            {
                "filepath": str(path),
                "prediction": prediction,
                "prediction_score": score,
                "detections": detections,
            }
        )
    out = tmp_path / "ensemble.json"
    out.write_text(json.dumps({"predictions": predictions}), encoding="utf-8")
    return out


ANIMAL_BOX = [{"category": "1", "label": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}]


def test_no_cv_result_becomes_unknown_even_when_score_is_high(tmp_path):
    ensemble = _write_ensemble(
        tmp_path,
        [("a.JPG", "id;no cv result;no cv result;;;;no cv result", 0.93, ANIMAL_BOX)],
    )
    result = next(iter(load_results(ensemble).values()))
    assert result.has_detection
    assert result.species == pipeline.UNKNOWN_LABEL


def test_bare_animal_label_becomes_unknown(tmp_path):
    """Top-of-taxonomy "animal" says nothing more than Unknown already does."""

    ensemble = _write_ensemble(
        tmp_path, [("a.JPG", "id;;;;;;animal", 0.99, ANIMAL_BOX)]
    )
    result = next(iter(load_results(ensemble).values()))
    assert result.species == pipeline.UNKNOWN_LABEL


def test_partial_taxonomy_labels_are_kept(tmp_path):
    """"bird" is a useful partial ID, unlike a bare "animal"."""

    ensemble = _write_ensemble(
        tmp_path, [("b.JPG", "id;aves;;;;;bird", 0.99, ANIMAL_BOX)]
    )
    result = next(iter(load_results(ensemble).values()))
    assert result.species == "bird"


def test_low_confidence_species_becomes_unknown(tmp_path):
    ensemble = _write_ensemble(
        tmp_path,
        [("b.JPG", "id;mammalia;;;;;eastern cottontail", 0.4, ANIMAL_BOX)],
    )
    result = next(iter(load_results(ensemble, confidence_threshold=0.7).values()))
    assert result.species == pipeline.UNKNOWN_LABEL


def test_confident_species_is_kept(tmp_path):
    ensemble = _write_ensemble(
        tmp_path,
        [("c.JPG", "id;mammalia;;;;;white-tailed deer", 0.95, ANIMAL_BOX)],
    )
    result = next(iter(load_results(ensemble, confidence_threshold=0.7).values()))
    assert result.species == "white-tailed deer"
    assert result.has_detection


def test_detector_box_overridden_to_blank_is_not_a_good_photo(tmp_path):
    """The ensemble rejects low-confidence detector boxes (wind-blown grass)."""

    ensemble = _write_ensemble(
        tmp_path, [("d.JPG", "id;;;;;;blank", 0.99, ANIMAL_BOX)]
    )
    result = next(iter(load_results(ensemble).values()))
    assert not result.has_detection
    assert result.species == "blank"


def test_timestamp_prefers_exif_over_file_mtime(tmp_path):
    """A copy tool that rewrites mtime must not collapse a card into one event."""

    from PIL import Image

    path = tmp_path / "MUD_0001.JPG"
    image = Image.new("RGB", (8, 8))
    exif = image.getexif()
    exif.get_ifd(0x8769)[0x9003] = "2026:05:11 18:37:55"
    image.save(path, exif=exif)

    # Push mtime far away from the real capture time.
    import os

    os.utime(path, (0, 0))

    assert pipeline.image_timestamp(path) == datetime(2026, 5, 11, 18, 37, 55)


def test_timestamp_falls_back_to_mtime_without_exif(tmp_path):
    path = tmp_path / "no_exif.JPG"
    path.write_bytes(b"not really a jpeg")

    import os

    os.utime(path, (1_000_000, 1_000_000))
    assert pipeline.image_timestamp(path) == datetime.fromtimestamp(1_000_000)


def test_species_name_parsing_and_sanitizing():
    assert species_common_name("id;mammalia;;;;;white-tailed deer") == "white-tailed deer"
    assert sanitize_folder_name("Unknown/Unidentified") == "Unknown-Unidentified"
    assert sanitize_folder_name("   ") == pipeline.UNKNOWN_LABEL
