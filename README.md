# critterCounter

Sorts trail-camera photos, tossing false triggers (grass/wind movement) and keeping
real animal photos, organized by species, with an Excel report of counts.

## How it works

Three chained stages, each writing a resumable JSON checkpoint:

1. **Detector** — [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) finds
   animal/human/vehicle bounding boxes. Runs on GPU (CUDA or DirectML) when available,
   CPU otherwise.
2. **Classifier** — [SpeciesNet](https://github.com/google/cameratrapai) identifies
   species from each detection. Runs on CPU; fast enough that GPU isn't needed here.
3. **Ensemble** — combines detector + classifier output into a final per-image
   prediction, with geofencing (country/state) to rule out implausible species.

Photos with a detection are grouped into "trigger events" (photos within 10 seconds of
each other), copied into `Documents/Good Game Pics/<session>/<species>/`, and summarized
in an `report.xlsx` per session. Species below a confidence threshold are labeled
"Unknown" rather than forced into a best guess.

## Setup

```
pip install -r requirements.txt
```

First run downloads model weights automatically (MegaDetector + SpeciesNet, ~1GB
combined). If your network blocks the default downloader (SSL cert issues are common on
locked-down Windows machines), download the files manually and place them where the
scripts expect — see `pipeline.py` for the exact paths.

For GPU acceleration on a machine without an NVIDIA GPU, also install `torch-directml`.

## Usage

```
python run_card.py <path to SD card / DCIM folder>
```

### Folders containing several card dumps

Point it at a parent folder and each subfolder holding photos is processed as its own
job, with its own species folders and report, plus a `combined-report.xlsx` totalling
species across all of them:

```
UnAudited/            ->   2026-09-07 1700 - UnAudited/
  DCIM/                      DCIM/                  (photos + report.xlsx)
  North camera/              North camera/          (photos + report.xlsx)
  South camera/              South camera/          (photos + report.xlsx)
                             combined-report.xlsx
```

This split is a correctness matter, not tidiness. Events are grouped by how close
together photos were taken, so processing two cameras as one batch would merge
unrelated sightings into a single event whenever their clocks happened to line up, and
a census built on that would undercount.

A folder with photos sitting directly in it is treated as one job, since that is a
single camera dump rather than a collection of them. Subfolder names become the result
folder names, so naming them by camera or location pays off in the report.

Re-running the same card is cheap: each stage caches its output, so a repeat run skips
straight to regenerating the photo folders and report. That is the fast way to see the
effect of a different confidence threshold without redoing hours of inference.

### Settings

`~/.critter_counter/config.json`, created on first run (see `config.py` for defaults):

| Setting | Default | What it does |
| --- | --- | --- |
| `output_folder` | `~/Documents/Good Game Pics` | Where sorted photos and reports go |
| `species_confidence_threshold` | `0.7` | Below this, a photo is filed as "Unknown" instead of guessing a species |
| `event_gap_seconds` | `10` | Photos closer together than this count as one trigger event |
| `country` | `USA` | ISO-3166 country used to rule out implausible species |
| `state` | *(unset)* | Two-letter state/region code, e.g. `TX`. Narrows species further — worth setting, since it should move photos out of the Unknown bucket |

### About the "Unknown" bucket

A photo lands in `Unknown` when something is definitely there but the species is
uncertain — either the score is below the threshold, or the model rolls up to a
non-answer (`no cv result`, or a bare `animal`). These are still real animal photos and
are kept; only the species label is withheld. Partial IDs like `bird` or `mammal` are
kept as-is, since they're still informative.

## Packaging for a non-technical user

`gui.py` is the actual app (run it directly with `python gui.py` during development).
`launcher.py` builds into a small standalone `.exe` that does first-run setup and then
starts the GUI.

The download people actually click on is **~10 MB**. Everything heavy is fetched from
its official source on first launch instead of being shipped:

| Fetched on first run | Size | From |
| --- | --- | --- |
| Python + PyTorch + deps | ~2.4 GB | PyPI, via a downloaded `uv` binary |
| MegaDetector weights | 268 MB | GitHub releases |
| SpeciesNet weights | 488 MB | Kaggle, via SpeciesNet's own downloader |

Setup is resumable and idempotent: each piece is skipped if already present, so a failed
or cancelled run picks up where it left off rather than starting over. The heavy stack is
never frozen into the exe, which is slow to build and fragile at this size.

Build it:

```
pyinstaller --onefile --noconsole --name CritterCounter --paths . --hidden-import certifi launcher.py
```

Ship `CritterCounter.exe`, both requirements files, and `critter_counter/`. First run
creates the rest alongside them:

```
CritterCounter.exe
requirements.txt
requirements-speciesnet.txt
critter_counter/     <- app code
tools/uv.exe         <- downloaded
venv/                <- downloaded
models/              <- downloaded
```

**Dependencies install in two passes, and the order is load-bearing.** megadetector
pins `protobuf<=3.20.1` (through ultralytics-yolov5) while speciesnet needs a modern
`onnx` that requires newer protobuf. Resolving both at once — with pip *or* uv — makes
the resolver backtrack to `onnx` 1.12, which has no Python 3.12 wheel and fails to build
without cmake. Installing speciesnet separately lets the second pass upgrade protobuf
and take a prebuilt onnx wheel. uv provisions Python (no admin rights needed) but does
not do the installing.

Two things worth knowing before "simplifying" any of this:

- Downloads here pin **certifi**'s CA bundle rather than the system trust store.
  MegaDetector's own downloader failed on the development machine with
  `CERTIFICATE_VERIFY_FAILED`, while everything using certifi worked.
- `models/speciesnet/` contains a second copy of the MegaDetector weights under a
  URL-derived filename, duplicating `models/md_v5a.0.1.pt` (~268 MB). Deleting it to
  save space does not work: SpeciesNet re-downloads it on load, verified by removing it
  and watching it come back.

## Roadmap

- Integration with Backcountry Steward once that service has a public API
  (`export_hooks.py` has the seam)
