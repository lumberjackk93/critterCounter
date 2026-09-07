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
For a double-click experience with no Python setup required, `launcher.py` builds into
a tiny standalone `.exe` (via PyInstaller) that just hands off to the real GUI running
under a portable Python install shipped alongside it — the heavy ML dependencies
(PyTorch, SpeciesNet, etc.) are never themselves frozen into the exe, which is slow and
fragile for a stack this size.

Build the launcher:

```
pyinstaller --onefile --noconsole --name CritterCounter launcher.py
```

Then assemble a distribution folder with this layout (venv/ and models/ are not part of
this git repo — copy them in from a working dev setup, per Setup above):

```
CritterCounter.exe
venv/                      <- portable Python with all deps (pip install -r requirements.txt)
models/
  md_v5a.0.1.pt            <- MegaDetector weights
  speciesnet/              <- SpeciesNet weights, copied from ~/.cache/kagglehub/
                              models/google/speciesnet/pyTorch/v4.0.3a/1/
critter_counter/           <- this repo's app code
```

Zip that whole folder and it runs anywhere — double-click `CritterCounter.exe`.

Copying `models/speciesnet/` in matters: SpeciesNet otherwise downloads its weights
from Kaggle on first run, so a fresh machine would need working network access and a
healthy certificate store before it could process a single photo. When that folder is
present the app uses it and never reaches the network.

## Roadmap

- Integration with Backcountry Steward once that service has a public API
  (`export_hooks.py` has the seam)
