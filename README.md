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

Settings (output folder, confidence threshold, event grouping window) live in
`~/.critter_counter/config.json`, created on first run — see `config.py` for defaults.

## Roadmap

- GUI wrapper (plug in a card, hit "Go")
- Packaged standalone `.exe` via PyInstaller
- Integration with Backcountry Steward once that service has a public API
  (`export_hooks.py` has the seam)
