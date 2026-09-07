"""Tiny double-click launcher.

Kept deliberately dependency-free (no torch/customtkinter/etc.) so PyInstaller can
freeze it in seconds. Its only job is to hand off to the real GUI, running under the
portable Python interpreter shipped alongside it, so nothing but this launcher needs
to be a "frozen" executable - the ML stack runs as ordinary, unfrozen Python.

Expected distribution layout (this exe sits at the root):
    CritterCounter.exe
    venv\\Scripts\\pythonw.exe   (portable interpreter with all dependencies)
    critter_counter\\gui.py
    models\\md_v5a.0.1.pt
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent

    pythonw = root / "venv" / "Scripts" / "pythonw.exe"
    gui_script = root / "critter_counter" / "gui.py"

    subprocess.Popen([str(pythonw), str(gui_script)], cwd=str(gui_script.parent))


if __name__ == "__main__":
    main()
