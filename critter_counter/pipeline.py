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

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]


_LOG_PATH: Optional[Path] = None


def set_log_file(path: Optional[Path]) -> None:
    """Sends subprocess output to a log file as well as stderr.

    The GUI runs under pythonw.exe, which has no console and no stderr, so without a
    log there is no record at all of what the detector or classifier reported.
    """

    global _LOG_PATH
    _LOG_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _emit(text: str) -> None:
    """Passes child output onward, tolerating the absence of a console.

    sys.stderr is None under pythonw.exe; writing to it unguarded crashes the run.
    """

    stream = sys.stderr
    if stream is not None:
        try:
            stream.write(text)
            stream.flush()
        except (ValueError, OSError):
            pass

    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8", errors="replace") as handle:
                handle.write(text)
        except OSError:
            pass


class NoPhotosFound(Exception):
    """Raised when the chosen folder has no photos, so the user gets an explanation
    rather than a stack trace from deep inside the detector."""
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")

# CREATE_NO_WINDOW. Every stage runs as a child process, and without this each one
# opens its own console window on top of whatever the user is doing.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Rough share of total wall-clock time each stage takes, used to blend per-stage
# progress into one overall bar. Detection (GPU) dominates; classification (CPU) is
# fast because it only needs a small crop per image; ensembling is pure Python, no
# model inference, and finishes in seconds regardless of image count.
STAGE_WEIGHTS = {"detector": 0.80, "classifier": 0.18, "ensemble": 0.02}


def _app_root() -> Path:
    """Directory this app is running from, whether frozen (PyInstaller) or not."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def card_fingerprint(images: list[Path], source_folder: Path) -> str:
    """Short digest identifying this exact set of photos.

    Keyed on the file list rather than the folder path because every card mounts at the
    same drive letter: swapping SD cards would otherwise look like the same job, and the
    resume logic (each stage skips if its output JSON exists) would silently report the
    previous card's animals for the new card.
    """

    digest = hashlib.sha1()
    for path in sorted(images):
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            name = str(path.relative_to(source_folder))
        except ValueError:
            name = path.name
        digest.update(f"{name}|{stat.st_size}|{int(stat.st_mtime)}\n".encode())
    return digest.hexdigest()[:12]

def _find_distribution_root() -> Path:
    """Locates the folder containing venv/ and models/, dev or packaged.

    Dev layout: MegaDetector/{venv,models,app/critter_counter/pipeline.py}, so the
    root is two levels above this file. Packaged layout: the launcher exe sits next to
    venv/ and models/ directly. Whichever location actually has a models/ folder wins.
    """

    candidates = [_app_root(), _app_root().parent.parent]
    for candidate in candidates:
        if (candidate / "models").is_dir():
            return candidate
    return candidates[-1]


_DIST_ROOT = _find_distribution_root()
VENV_PYTHON = _DIST_ROOT / "venv" / "Scripts" / "python.exe"
DETECTOR_MODEL = _DIST_ROOT / "models" / "md_v5a.0.1.pt"

# SpeciesNet normally downloads its weights from Kaggle into a per-user cache on first
# run. A packaged copy ships them instead, so point at the local folder when it's
# present - otherwise a fresh machine needs network access (and a working cert store,
# which is exactly what failed here) before it can process a single photo.
_LOCAL_SPECIESNET_MODEL = _DIST_ROOT / "models" / "speciesnet"
DEFAULT_SPECIESNET_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"


def speciesnet_model() -> str:
    if (_LOCAL_SPECIESNET_MODEL / "info.json").exists():
        return str(_LOCAL_SPECIESNET_MODEL)
    return DEFAULT_SPECIESNET_MODEL

CATEGORY_LABEL = {"1": "animal", "2": "human", "3": "vehicle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

# Species confidence below this becomes UNKNOWN_LABEL rather than a guess. SpeciesNet's
# own "no cv result" placeholder (emitted when geofencing/rollup can't settle on a valid
# taxon) is always treated as unknown regardless of its score, since that score reflects
# the raw classifier confidence, not confidence in "no cv result" itself.
SPECIES_CONFIDENCE_THRESHOLD = 0.7
UNKNOWN_LABEL = "Unknown"
# Labels that confirm something was in frame but say nothing about what it was.
# "no cv result" is SpeciesNet's own placeholder; a bare "animal" is the top of the
# taxonomy, meaning it could not roll up to anything more specific. Partial IDs like
# "bird" or "mammal" are genuinely useful, so they are kept as-is.
_UNINFORMATIVE_LABELS = {"no cv result", "animal"}
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


_active_process: Optional[subprocess.Popen] = None
_active_process_lock = threading.Lock()


def terminate_active_process() -> bool:
    """Stops the in-flight model subprocess, if there is one.

    The heavy lifting happens in child processes, so closing the app without this
    would leave one running invisibly for hours with no window to stop it from.
    """

    with _active_process_lock:
        process = _active_process

    if process is None or process.poll() is not None:
        return False

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    return True


def _run_with_progress(
    args: list[str], on_stage_progress: Optional[Callable[[float], None]] = None
) -> None:
    """Runs a subprocess, optionally parsing tqdm-style N/Total progress from stderr.

    tqdm rewrites its line in place with carriage returns rather than newlines, so
    stderr is read character-by-character and split on both \\r and \\n to catch
    every update rather than only the final line.
    """

    if on_stage_progress is None:
        subprocess.run(args, check=True, creationflags=_NO_WINDOW)
        return

    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    assert process.stderr is not None

    global _active_process
    with _active_process_lock:
        _active_process = process

    # Read in chunks, not per-character: tqdm redraws its bar many times a second and
    # a slow reader lets the pipe buffer fill, which blocks the child mid-inference and
    # drags a 1.2s/image job down to 8s/image. read1 returns as soon as any bytes are
    # available rather than waiting for a full buffer. Everything read is echoed onward
    # so stderr redirected to a log file still shows the run's output.
    buf = ""
    while True:
        raw = process.stderr.read1(8192)
        if not raw:
            break
        chunk = raw.decode("utf-8", errors="replace")
        _emit(chunk)

        buf += chunk
        segments = re.split(r"[\r\n]", buf)
        buf = segments.pop()
        last_match = None
        for segment in segments:
            for last_match in _PROGRESS_RE.finditer(segment):
                pass
        if last_match:
            n, total = int(last_match.group(1)), int(last_match.group(2))
            if total > 0:
                on_stage_progress(min(n / total, 1.0))

    process.wait()

    with _active_process_lock:
        _active_process = None

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args)


def run_detector_stage(
    image_folder: Path,
    output_json: Path,
    checkpoint_path: Path,
    on_stage_progress: Optional[Callable[[float], None]] = None,
) -> None:
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
        # --allow_checkpoint_overwrite is required alongside --resume_from_checkpoint:
        # we read from and keep writing to the same file, and MegaDetector otherwise
        # refuses to start rather than clobber an existing checkpoint.
        args += [
            "--resume_from_checkpoint",
            str(checkpoint_path),
            "--allow_checkpoint_overwrite",
        ]

    _run_with_progress(args, on_stage_progress)


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
    filepaths_txt: Path,
    detections_json: Path,
    output_json: Path,
    country: str = "USA",
    on_stage_progress: Optional[Callable[[float], None]] = None,
) -> None:
    if output_json.exists():
        return

    args = [
        str(VENV_PYTHON),
        "-m",
        "speciesnet.scripts.run_model",
        "--model",
        speciesnet_model(),
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
    ]
    _run_with_progress(args, on_stage_progress)


def run_ensemble_stage(
    filepaths_txt: Path,
    classifications_json: Path,
    detections_json: Path,
    output_json: Path,
    country: str = "USA",
    admin1_region: str = "",
) -> None:
    if output_json.exists():
        return

    # Geofencing happens here, not in the classifier, so the location has to be passed
    # to this stage - without it the ensemble discards plausible local species and
    # falls back to its "no cv result" placeholder. A state/province narrows it further.
    args = [
        str(VENV_PYTHON),
        "-m",
        "speciesnet.scripts.run_model",
        "--model",
        speciesnet_model(),
        "--ensemble_only",
        "--classifications_json",
        str(classifications_json),
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
    if admin1_region:
        args += ["--admin1_region", admin1_region]
    # Routed through the tracked runner (with a no-op progress sink) so this stage is
    # also cancellable, even though it finishes in seconds.
    _run_with_progress(args, lambda _fraction: None)


_EXIF_DATETIME_ORIGINAL = 0x9003
_EXIF_IFD_POINTER = 0x8769
_EXIF_DATETIME = 0x0132


def image_timestamp(path: Path) -> datetime:
    """Capture time from EXIF, falling back to the file's modified time.

    Event grouping keys off this, and mtime alone is not trustworthy: copying photos
    off a card with a tool that doesn't preserve timestamps would give every photo the
    same time and collapse a whole card into one enormous "event".
    """

    try:
        from PIL import Image

        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(_EXIF_IFD_POINTER).get(_EXIF_DATETIME_ORIGINAL)
            if not raw:
                raw = exif.get(_EXIF_DATETIME)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    return datetime.fromtimestamp(path.stat().st_mtime)


def species_common_name(prediction: str) -> str:
    """Extracts the trailing common-name field from a SpeciesNet class string."""

    parts = prediction.split(";")
    return parts[-1] if parts else prediction


def sanitize_folder_name(name: str) -> str:
    """Makes a species/label name safe to use as a single Windows folder segment."""

    return name.translate(_INVALID_FOLDER_CHARS).strip() or UNKNOWN_LABEL


def load_results(
    ensemble_json: Path, confidence_threshold: float = SPECIES_CONFIDENCE_THRESHOLD
) -> dict[str, ImageResult]:
    with open(ensemble_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, ImageResult] = {}
    for pred in data["predictions"]:
        filepath = Path(pred["filepath"])
        timestamp = image_timestamp(filepath)
        name = species_common_name(pred.get("prediction", ""))
        score = pred.get("prediction_score", 0.0)
        detections = pred.get("detections") or []
        has_detection = bool(detections) and name != "blank"
        animal_box_count = len(detections)

        if has_detection:
            if name in _UNINFORMATIVE_LABELS or score < confidence_threshold:
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


def destination_filename(image_path: Path, source_folder: Path) -> str:
    """Prefixes the camera's folder name onto the filename.

    Trail cams restart numbering in every folder, so 100MUDDY, 101MUDDY and 102MUDDY
    each contain their own MUD_0001.JPG. Copying those into a single species folder by
    bare filename would silently overwrite all but the last one.
    """

    try:
        relative = image_path.relative_to(source_folder)
    except ValueError:
        return image_path.name

    subfolders = relative.parts[:-1]
    if not subfolders:
        return image_path.name
    return "_".join((*subfolders, image_path.name))


def copy_good_photos(events: list[Event], dest_root: Path, source_folder: Path) -> int:
    """Copies every photo with a detection into dest_root/<species>/<filename>."""

    copied = 0
    for event in events:
        for img in event.images:
            if not img.has_detection:
                continue
            species_folder = dest_root / img.species
            species_folder.mkdir(parents=True, exist_ok=True)
            dest = species_folder / destination_filename(img.filepath, source_folder)
            shutil.copy2(img.filepath, dest)
            copied += 1
    return copied


def write_excel_report(events: list[Event], dest_path: Path, source_folder: Path) -> None:
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
                destination_filename(img.filepath, source_folder)
                for img in event.images
                if img.species == species
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


def folder_image_count(folder: Path) -> int:
    return sum(
        1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_batches(folder: Path) -> list[Path]:
    """Splits a folder into independently-processed jobs.

    Each immediate subfolder holding photos is its own job. Keeping them separate
    matters for more than tidiness: events are grouped by how close together photos
    were taken, so processing two cameras as one batch would merge unrelated
    sightings into a single event whenever their clocks happened to line up.

    A folder with photos sitting directly in it is treated as one job, since that is
    a single camera dump rather than a collection.
    """

    if not folder.is_dir():
        return [folder]

    has_loose_photos = any(
        p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir() if p.is_file()
    )
    if has_loose_photos:
        return [folder]

    subfolders = sorted(
        d for d in folder.iterdir() if d.is_dir() and folder_image_count(d) > 0
    )
    return subfolders or [folder]


def run_pipeline(
    source_folder: Path,
    output_root: Path,
    work_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
    confidence_threshold: float = SPECIES_CONFIDENCE_THRESHOLD,
    event_gap_seconds: int = EVENT_GAP_SECONDS,
    country: str = "USA",
    admin1_region: str = "",
    session_folder: Optional[Path] = None,
) -> tuple[Path, list[Event]]:
    """Runs the full pipeline for one card.

    Returns the session output folder and the grouped events, so a caller (e.g. a GUI)
    can build a results summary without re-parsing the Excel report.
    """

    def report(stage: str, stage_fraction: float) -> None:
        if on_progress is None:
            return
        weight_before = sum(
            w for s, w in STAGE_WEIGHTS.items() if list(STAGE_WEIGHTS).index(s) < list(STAGE_WEIGHTS).index(stage)
        )
        overall = weight_before + stage_fraction * STAGE_WEIGHTS[stage]
        on_progress(overall, f"{stage.capitalize()}: {stage_fraction * 100:.0f}%")

    all_images = [
        p
        for p in source_folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not all_images:
        raise NoPhotosFound(
            f"No JPG photos found in {source_folder}.\n\n"
            "Pick the card's DCIM folder (or the drive itself) and try again."
        )

    # Per-card scratch space, so resuming an interrupted run reuses its own cached
    # stages while a different card always starts clean.
    work_dir = work_dir / card_fingerprint(all_images, source_folder)
    work_dir.mkdir(parents=True, exist_ok=True)

    filepaths_txt = work_dir / "filepaths.txt"
    filepaths_txt.write_text("\n".join(str(p) for p in all_images), encoding="utf-8")

    md_json = work_dir / "detections_md_format.json"
    checkpoint = work_dir / "checkpoint.json"
    run_detector_stage(
        source_folder, md_json, checkpoint,
        on_stage_progress=lambda f: report("detector", f),
    )

    detections_json = work_dir / "detections_speciesnet_format.json"
    convert_md_to_speciesnet_format(md_json, detections_json, source_folder, country)

    classifications_json = work_dir / "classifications.json"
    run_classifier_stage(
        filepaths_txt, detections_json, classifications_json, country,
        on_stage_progress=lambda f: report("classifier", f),
    )

    if on_progress is not None:
        on_progress(sum(list(STAGE_WEIGHTS.values())[:2]), "Finalizing results...")

    ensemble_json = work_dir / "ensemble.json"
    run_ensemble_stage(
        filepaths_txt, classifications_json, detections_json, ensemble_json,
        country, admin1_region,
    )

    results = load_results(ensemble_json, confidence_threshold)
    events = group_into_events(list(results.values()), event_gap_seconds)

    if session_folder is None:
        finished = datetime.now()
        source_names = "-".join(
            sorted({p.name for p in source_folder.iterdir() if p.is_dir()})
        ) or source_folder.name
        session_folder = output_root / f"{finished:%Y-%m-%d %H%M} - {source_names}"
    session_folder.mkdir(parents=True, exist_ok=True)

    copy_good_photos(events, session_folder, source_folder)
    write_excel_report(events, session_folder / "report.xlsx", source_folder)

    if on_progress is not None:
        on_progress(1.0, "Done")

    return session_folder, events


def write_combined_report(
    batches: list[tuple[str, list[Event]]], dest_path: Path
) -> None:
    """One workbook totalling species across every folder in a batch run."""

    from openpyxl import Workbook

    wb = Workbook()
    summary = wb.active
    summary.title = "All folders"
    summary.append(["Species", "Event Count", "Photo Count", "Seen in folders"])

    totals: dict[str, dict] = {}
    for name, events in batches:
        for event in events:
            for species in event.species_present:
                entry = totals.setdefault(
                    species, {"events": 0, "photos": 0, "folders": set()}
                )
                entry["events"] += 1
                entry["folders"].add(name)
            for img in event.images:
                if img.has_detection:
                    totals.setdefault(
                        img.species, {"events": 0, "photos": 0, "folders": set()}
                    )["photos"] += 1

    for species in sorted(totals):
        entry = totals[species]
        summary.append(
            [
                species,
                entry["events"],
                entry["photos"],
                ", ".join(sorted(entry["folders"])),
            ]
        )

    by_folder = wb.create_sheet("By folder")
    by_folder.append(["Folder", "Species", "Event Count", "Photo Count"])
    for name, events in batches:
        per_species: dict[str, dict] = {}
        for event in events:
            for species in event.species_present:
                per_species.setdefault(species, {"events": 0, "photos": 0})["events"] += 1
            for img in event.images:
                if img.has_detection:
                    per_species.setdefault(img.species, {"events": 0, "photos": 0})[
                        "photos"
                    ] += 1
        if not per_species:
            by_folder.append([name, "(nothing found)", 0, 0])
        for species in sorted(per_species):
            by_folder.append(
                [name, species, per_species[species]["events"], per_species[species]["photos"]]
            )

    wb.save(dest_path)


def run_batch(
    parent_folder: Path,
    output_root: Path,
    work_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
    on_batch_start: Optional[Callable[[int, int, Path], None]] = None,
    **pipeline_kwargs,
) -> tuple[Path, list[tuple[str, list[Event]]]]:
    """Processes each subfolder of parent_folder as its own job.

    Returns the session folder and per-folder events. Each subfolder keeps its own
    photo folders and report; a combined workbook totals them.
    """

    batches = find_batches(parent_folder)
    if len(batches) == 1 and batches[0] == parent_folder:
        session, events = run_pipeline(
            parent_folder, output_root, work_dir, on_progress, **pipeline_kwargs
        )
        return session, [(parent_folder.name, events)]

    session_folder = output_root / (
        f"{datetime.now():%Y-%m-%d %H%M} - {parent_folder.name}"
    )
    session_folder.mkdir(parents=True, exist_ok=True)

    # Weight progress by photo count so the bar reflects work done, not folders done.
    sizes = [max(folder_image_count(b), 1) for b in batches]
    total = sum(sizes)
    completed = 0

    results: list[tuple[str, list[Event]]] = []
    for index, (batch, size) in enumerate(zip(batches, sizes)):
        if on_batch_start is not None:
            on_batch_start(index + 1, len(batches), batch)

        def batch_progress(fraction: float, status: str, _s=size, _c=completed) -> None:
            if on_progress is not None:
                on_progress(
                    (_c + fraction * _s) / total,
                    f"[{index + 1}/{len(batches)}] {batch.name} - {status}",
                )

        try:
            _, events = run_pipeline(
                batch,
                output_root,
                work_dir,
                batch_progress,
                session_folder=session_folder / batch.name,
                **pipeline_kwargs,
            )
            results.append((batch.name, events))
        except NoPhotosFound:
            results.append((batch.name, []))

        completed += size

    write_combined_report(results, session_folder / "combined-report.xlsx")

    if on_progress is not None:
        on_progress(1.0, "Done")

    return session_folder, results
