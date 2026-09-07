"""Identify which physical card is in the reader, and remember the ones seen before.

A drive letter is not an identity: every card mounts as D:, so "process D:\\DCIM" can
silently mean a completely different card than it did an hour ago. This module names the
card (volume label + serial), counts what's on it, and keeps a history, so the app can
say "this is a new card" or "you already did this one" before starting hours of work.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import IMAGE_EXTENSIONS, card_fingerprint

HISTORY_PATH = Path.home() / ".critter_counter" / "cards.json"


@dataclass
class CardIdentity:
    drive: str
    label: str
    serial: str
    photo_count: int
    folders: list[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def key(self) -> str:
        """Stable across adding photos; the fingerprint deliberately is not."""

        return self.serial or f"{self.drive}|{self.label}"

    @property
    def name(self) -> str:
        if self.label:
            return f"{self.label} ({self.drive})"
        if self.folders:
            return f"{self.folders[0]} card ({self.drive})"
        return self.drive


def _volume_info(drive_root: str) -> tuple[str, str]:
    """Volume label and serial number, or empty strings off Windows / on failure."""

    try:
        label = ctypes.create_unicode_buffer(261)
        filesystem = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_ulong(0)
        max_component = ctypes.c_ulong(0)
        flags = ctypes.c_ulong(0)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
            ctypes.c_wchar_p(drive_root),
            label,
            261,
            ctypes.byref(serial),
            ctypes.byref(max_component),
            ctypes.byref(flags),
            filesystem,
            261,
        )
        if not ok:
            return "", ""
        return label.value, f"{serial.value:08X}"
    except Exception:
        return "", ""


def identify(source_folder: Path) -> CardIdentity:
    images = [
        p for p in source_folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    folders = sorted({p.parent.name for p in images})

    drive = source_folder.drive or str(source_folder)
    label, serial = _volume_info(drive + "\\") if source_folder.drive else ("", "")

    return CardIdentity(
        drive=drive,
        label=label,
        serial=serial,
        photo_count=len(images),
        folders=folders,
        fingerprint=card_fingerprint(images, source_folder),
    )


def load_history(path: Path = HISTORY_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def record_run(
    identity: CardIdentity, session_folder: Path, path: Path = HISTORY_PATH
) -> None:
    history = load_history(path)
    entry = history.setdefault(
        identity.key,
        {"label": identity.label, "serial": identity.serial, "runs": []},
    )
    entry["label"] = identity.label
    entry["runs"].append(
        {
            "when": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": identity.fingerprint,
            "photo_count": identity.photo_count,
            "session": str(session_folder),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def describe(identity: CardIdentity, history: Optional[dict] = None) -> str:
    """One-paragraph summary of what is in the reader and whether it's been done."""

    history = load_history() if history is None else history
    entry = history.get(identity.key)

    lines = [
        f"Card: {identity.name}",
        f"Photos: {identity.photo_count:,}"
        + (f" in {', '.join(identity.folders)}" if identity.folders else ""),
    ]

    if not entry or not entry.get("runs"):
        lines.append("This card has not been processed before.")
        return "\n".join(lines)

    last = entry["runs"][-1]
    when = last["when"].replace("T", " ")
    if last.get("fingerprint") == identity.fingerprint:
        lines.append(
            f"Already processed on {when} - the photos are unchanged since then, so "
            "running again just rebuilds the report."
        )
    else:
        difference = identity.photo_count - last.get("photo_count", 0)
        changed = (
            f"{difference:+,} photos" if difference else "the same count but different photos"
        )
        lines.append(
            f"Processed before on {when}, but the contents have changed ({changed}). "
            "This will be treated as a new job."
        )
    lines.append(f"Last result: {last.get('session', 'unknown')}")
    return "\n".join(lines)
