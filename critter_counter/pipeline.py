"""Critter Counter core pipeline.

Runs a three-stage chain over a folder of trail-camera photos:
  1. Detector (MegaDetector v5a, GPU via DirectML when available) -> raw MD-format JSON
  2. Classifier (SpeciesNet, CPU) -> per-image species scores
  3. Ensemble (SpeciesNet) -> final species prediction per image

Detection is the only slow, GPU-bound stage; the classifier is fast enough to run on
CPU for every image regardless of whether a detection was found. Each stage writes an
incrementally-resumable JSON file, so an interrupted run can pick back up.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

VENV_PYTHON = Path(r"C:\Users\txchi\MegaDetector\venv\Scripts\python.exe")
DETECTOR_MODEL = Path(r"C:\Users\txchi\MegaDetector\models\md_v5a.0.1.pt")

CATEGORY_LABEL = {"1": "animal", "2": "human", "3": "vehicle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

# Species confidence below this becomes UNKNOWN_LABEL rather than a guess. SpeciesNet's
# own "no cv result" placeholder (emitted when geofencing/rollup can't settle on a valid
# taxon) is always treated as unknown regardless of its score, since that score reflects
# the raw classifier confidence, not confidence in "no cv result" itself.
SPECIES_CONFIDENCE_THRESHOLD = 0.7
UNKNOWN_LABEL = "Unknown"
_NO_CV_RESULT = "no cv result"
_INVALID_FOLDER_CHARS = str.maketrans({c: "-" for c in '/\\:*?"<>|'})

# Photos taken less than this many seconds apart are treated as one trigger event.
EVENT_GAP_SECONDS = 10


@dataclass
class ImageResult:
    filepath: Path
    timestamp: datetime
    has_detection: bool
    species: str
    confidence: float
    animal_box_count: int = 0


@dataclass
class Event:
    images: list[ImageResult] = field(default_factory=list)

    @property
    def start_time(self) -> datetime:
        return min(img.timestamp for img in self.images)

    @property
    def species_present(self) -> set[str]:
        return {img.species for img in self.images if img.has_detection}

    def max_confidence_for(self, species: str) -> float:
        return max(
            (img.confidence for img in self.images if img.species == species),
            default=0.0,
        )

    def max_individuals_for(self, species: str) -> int:
        return max(
            (img.animal_box_count for img in self.images if img.species == species),
            default=0,
        )


def run_detector_stage(image_folder: Path, output_json: Path, checkpoint_path: Path) -> None:
    """Runs MegaDetector over a folder, resuming from checkpoint if present."""

    if output_json.exists():
        return

    args = [
        str(VENV_PYTHON),
        "-m",
        "megadetector.detection.run_detector_batch",
        str(DETECTOR_MODEL),
        str(image_folder),
        str(output_json),
        "--recursive",
        "--output_relative_filenames",
        "--checkpoint_frequency",
        "300",
        "--checkpoint_path",
        str(checkpoint_path),
    ]
    if checkpoint_path.exists():
        args += ["--resume_from_checkpoint", str(checkpoint_path)]

    subprocess.run(args, check=True)


def convert_md_to_speciesnet_format(
    md_json_path: Path, out_json_path: Path, image_folder: Path, country: str = "USA"
) -> None:
    """Reshapes MegaDetector's {images:[...]} output into SpeciesNet's {predictions:[...]}."""

    with open(md_json_path, "r", encoding="utf-8") as f:
        md = json.load(f)

    predictions = []
    for img in md["images"]:
        detections = []
        for d in img.get("detections", []):
            cat = d["category"]
            detections.append(
                {
                    "category": cat,
                    "label": CATEGORY_LABEL.get(cat, "animal"),
                    "conf": d["conf"],
                    "bbox": d["bbox"],
                }
            )
        filepath = img["file"]
        if not Path(filepath).is_absolute():
            filepath = str(image_folder / filepath)
        predictions.append(
            {
                "filepath": str(Path(filepath)),
                "country": country,
                "detections": detections,
            }
        )

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f)


def run_classifier_stage(
    filepaths_txt: Path, detections_json: Path, output_json: Path, country: str = "USA"
) -> None:
    if output_json.exists():
        return

    args = [
        str(VENV_PYTHON),
        "-m",
        "speciesnet.scripts.run_model",
        "--classifier_only",
        "--detections_json",
        str(detections_json),
        "--filepaths_txt",
        str(filepaths_txt),
        "--predictions_json",
        str(output_json),
        "--country",
        country,
        "--bypass_prompts",
        "--noprogress_bars",
    ]
    subprocess.run(args, check=True)


def run_ensemble_stage(
    filepaths_txt: Path,
    classifications_json: Path,
    detections_json: Path,
    output_json: Path,
) -> None:
    if output_json.exists():
        return

    args = [
        str(VENV_PYTHON),
        "-m",
        "speciesnet.scripts.run_model",
        "--ensemble_only",
        "--classifications_json",
        str(classifications_json),
        "--detections_json",
        str(detections_json),
        "--filepaths_txt",
        str(filepaths_txt),
        "--predictions_json",
        str(output_json),
        "--bypass_prompts",
        "--noprogress_bars",
    ]
    subprocess.run(args, check=True)


def species_common_name(prediction: str) -> str:
    """Extracts the trailing common-name field from a SpeciesNet class string."""

    parts = prediction.split(";")
    return parts[-1] if parts else prediction


def sanitize_folder_name(name: str) -> str:
    """Makes a species/label name safe to use as a single Windows folder segment."""

    return name.translate(_INVALID_FOLDER_CHARS).strip() or UNKNOWN_LABEL


def load_results(ensemble_json: Path) -> dict[str, ImageResult]:
    with open(ensemble_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, ImageResult] = {}
    for pred in data["predictions"]:
        filepath = Path(pred["filepath"])
        timestamp = datetime.fromtimestamp(filepath.stat().st_mtime)
        name = species_common_name(pred.get("prediction", ""))
        score = pred.get("prediction_score", 0.0)
        detections = pred.get("detections") or []
        has_detection = bool(detections) and name != "blank"
        animal_box_count = len(detections)

        if has_detection:
            if name == _NO_CV_RESULT or score < SPECIES_CONFIDENCE_THRESHOLD:
                name = UNKNOWN_LABEL
            else:
                name = sanitize_folder_name(name)
        else:
            name = "blank"

        results[str(filepath)] = ImageResult(
            filepath=filepath,
            timestamp=timestamp,
            has_detection=has_detection,
            species=name,
            confidence=score,
            animal_box_count=animal_box_count,
        )
    return results


def group_into_events(images: list[ImageResult], gap_seconds: int = EVENT_GAP_SECONDS) -> list[Event]:
    images = sorted(images, key=lambda img: img.timestamp)
    events: list[Event] = []
    current: Event | None = None
    last_time: datetime | None = None

    for img in images:
        if current is None or (img.timestamp - last_time).total_seconds() > gap_seconds:
            current = Event()
            events.append(current)
        current.images.append(img)
        last_time = img.timestamp

    return events


def copy_good_photos(events: list[Event], dest_root: Path) -> int:
    """Copies every photo with a detection into dest_root/<species>/<filename>."""

    copied = 0
    for event in events:
        for img in event.images:
            if not img.has_detection:
                continue
            species_folder = dest_root / img.species
            species_folder.mkdir(parents=True, exist_ok=True)
            dest = species_folder / img.filepath.name
            shutil.copy2(img.filepath, dest)
            copied += 1
    return copied


def write_excel_report(events: list[Event], dest_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["Species", "Event Count", "Max Individuals", "Photo Count"])

    species_events: dict[str, int] = {}
    species_photos: dict[str, int] = {}
    species_max_individuals: dict[str, int] = {}
    for event in events:
        for species in event.species_present:
            species_events[species] = species_events.get(species, 0) + 1
            species_max_individuals[species] = max(
                species_max_individuals.get(species, 0),
                event.max_individuals_for(species),
            )
        for img in event.images:
            if img.has_detection:
                species_photos[img.species] = species_photos.get(img.species, 0) + 1

    for species in sorted(species_events):
        summary.append(
            [
                species,
                species_events[species],
                species_max_individuals.get(species, 0),
                species_photos.get(species, 0),
            ]
        )

    detail = wb.create_sheet("Detail")
    detail.append(["Event Start", "Species", "Confidence", "Photo Filenames"])
    for event in events:
        for species in sorted(event.species_present):
            filenames = ", ".join(
                img.filepath.name for img in event.images if img.species == species
            )
            detail.append(
                [
                    event.start_time.isoformat(sep=" "),
                    species,
                    round(event.max_confidence_for(species), 3),
                    filenames,
                ]
            )

    wb.save(dest_path)


def run_pipeline(source_folder: Path, output_root: Path, work_dir: Path) -> Path:
    """Runs the full pipeline for one card and returns the session output folder."""

    work_dir.mkdir(parents=True, exist_ok=True)

    all_images = [
        p
        for p in source_folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    filepaths_txt = work_dir / "filepaths.txt"
    filepaths_txt.write_text("\n".join(str(p) for p in all_images), encoding="utf-8")

    md_json = work_dir / "detections_md_format.json"
    checkpoint = work_dir / "checkpoint.json"
    run_detector_stage(source_folder, md_json, checkpoint)

    detections_json = work_dir / "detections_speciesnet_format.json"
    convert_md_to_speciesnet_format(md_json, detections_json, source_folder)

    classifications_json = work_dir / "classifications.json"
    run_classifier_stage(filepaths_txt, detections_json, classifications_json)

    ensemble_json = work_dir / "ensemble.json"
    run_ensemble_stage(filepaths_txt, classifications_json, detections_json, ensemble_json)

    results = load_results(ensemble_json)
    events = group_into_events(list(results.values()))

    finished = datetime.now()
    source_names = "-".join(
        sorted({p.name for p in source_folder.iterdir() if p.is_dir()})
    ) or source_folder.name
    session_name = f"{finished:%Y-%m-%d %H%M} - {source_names}"
    session_folder = output_root / session_name
    session_folder.mkdir(parents=True, exist_ok=True)

    copy_good_photos(events, session_folder)
    write_excel_report(events, session_folder / "report.xlsx")

    return session_folder
