"""CLI entry point: process one SD card / folder through the full pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    output_root = Path(config["output_folder"])
    work_dir = args.work_dir or (Path.home() / ".critter_counter" / "work")

    session_folder, events = run_pipeline(
        args.source_folder,
        output_root,
        work_dir,
        confidence_threshold=config["species_confidence_threshold"],
        event_gap_seconds=config["event_gap_seconds"],
        country=config["country"],
        admin1_region=config["state"],
    )
    print(f"Done. {len(events)} events found. Results in: {session_folder}")


if __name__ == "__main__":
    main()
