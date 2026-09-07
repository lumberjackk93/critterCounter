"""CLI entry point: process one SD card / folder through the full pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cards
from config import load_config
from pipeline import NoPhotosFound, find_batches, folder_image_count, run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (for unattended runs).",
    )
    args = parser.parse_args()

    config = load_config()
    output_root = Path(config["output_folder"])
    work_dir = args.work_dir or (Path.home() / ".critter_counter" / "work")

    # A drive letter is not an identity - every card mounts as D:. Say which card this
    # actually is before committing to hours of work on it.
    identity = cards.identify(args.source_folder)
    print(cards.describe(identity))

    batches = find_batches(args.source_folder)
    if len(batches) > 1:
        print(f"\nThis folder holds {len(batches)} separate sets, processed one by one:")
        for batch in batches:
            print(f"  {batch.name}: {folder_image_count(batch):,} photos")

    print()
    if not args.yes:
        answer = input("Process this? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return

    def announce(index: int, count: int, folder: Path) -> None:
        print(f"\n[{index}/{count}] {folder.name}", flush=True)

    try:
        session_folder, results = run_batch(
            args.source_folder,
            output_root,
            work_dir,
            on_batch_start=announce if len(batches) > 1 else None,
            confidence_threshold=config["species_confidence_threshold"],
            event_gap_seconds=config["event_gap_seconds"],
            country=config["country"],
            admin1_region=config["state"],
        )
    except NoPhotosFound as exc:
        print(exc)
        sys.exit(1)

    cards.record_run(identity, session_folder)
    total_events = sum(len(events) for _, events in results)
    print(f"\nDone. {total_events} events across {len(results)} folder(s).")
    print(f"Results in: {session_folder}")


if __name__ == "__main__":
    main()
