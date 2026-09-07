"""Double-click entry point.

Two jobs: run first-time setup if the app hasn't been set up yet, then start the GUI.

Kept to the standard library (plus certifi via bootstrap) so PyInstaller freezes it into
a small exe in seconds. Everything heavy - the Python runtime, PyTorch, the model
weights - is downloaded on first run rather than shipped, so the download people
actually click on stays small.

Expected layout once set up (this exe sits at the root):
    CritterCounter.exe
    requirements.txt
    critter_counter\\gui.py
    tools\\uv.exe        (downloaded)
    venv\\               (downloaded)
    models\\             (downloaded)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import bootstrap

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def start_gui(root: Path) -> None:
    pythonw = root / "venv" / "Scripts" / "pythonw.exe"
    gui_script = root / "critter_counter" / "gui.py"
    subprocess.Popen([str(pythonw), str(gui_script)], cwd=str(gui_script.parent))


class SetupWindow(tk.Tk):
    """Shown only on first run, while dependencies download."""

    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.failure: str | None = None

        self.title("Critter Counter - First-time setup")
        self.geometry("440x180")
        self.resizable(False, False)

        tk.Label(
            self,
            text="Setting up Critter Counter",
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(22, 4))
        tk.Label(
            self,
            text="Downloading what it needs. This happens once and\n"
            "takes a while on a slow connection.",
            justify="center",
        ).pack()

        self.bar = ttk.Progressbar(self, length=360, mode="indeterminate")
        self.bar.pack(pady=14)
        self.bar.start(12)

        self.status = tk.Label(self, text="Starting...", fg="gray")
        self.status.pack()

        self.after(100, self._begin)

    def _begin(self) -> None:
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        def on_status(message: str, fraction: float | None) -> None:
            self.after(0, self._apply_status, message, fraction)

        try:
            bootstrap.run_setup(self.root_dir, on_status)
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
        self.after(0, self.destroy)

    def _apply_status(self, message: str, fraction: float | None) -> None:
        self.status.configure(text=message)
        if fraction is None:
            if self.bar["mode"] != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(12)
        else:
            if self.bar["mode"] != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate", maximum=1.0)
            self.bar["value"] = fraction


def main() -> None:
    root = app_root()

    if not bootstrap.is_ready(root):
        window = SetupWindow(root)
        window.mainloop()
        if window.failure or not bootstrap.is_ready(root):
            messagebox.showerror(
                "Setup did not finish",
                "Critter Counter could not finish setting up.\n\n"
                f"{window.failure or 'Setup was closed before it completed.'}\n\n"
                "Check your internet connection and run it again - it picks up where "
                "it left off.",
            )
            return

    start_gui(root)


if __name__ == "__main__":
    main()
