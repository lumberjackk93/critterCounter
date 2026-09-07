"""First-run setup: fetches the Python runtime, dependencies and model weights.

The shipped download is small (just the launcher and this app's code); everything
heavy is pulled from its official source on first launch and cached on disk, so later
launches start immediately.

Deliberately limited to the standard library plus certifi, because this runs inside the
frozen launcher before any environment exists. certifi matters specifically: this
machine's default certificate store rejected the MegaDetector download with
CERTIFICATE_VERIFY_FAILED, so downloads here pin a known-good CA bundle instead.
"""

from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
MEGADETECTOR_URL = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.1.pt"
)
SPECIESNET_MODEL_ID = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
REQUIREMENTS_SPECIESNET = "requirements-speciesnet.txt"

StatusCallback = Callable[[str, Optional[float]], None]

# Hides the console windows that would otherwise flash up for each subprocess.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class SetupFailed(RuntimeError):
    """Setup step failed, carrying the tool's own error text.

    Setup runs windowless, so a bare CalledProcessError would discard the only
    explanation of what went wrong and leave nothing to act on.
    """


def _run(args: list[str], what: str) -> None:
    result = subprocess.run(
        args, capture_output=True, text=True, creationflags=_NO_WINDOW
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        tail = "\n".join(detail.splitlines()[-15:])
        raise SetupFailed(f"{what} failed.\n\n{tail}")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(url: str, destination: Path, on_status: StatusCallback, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "CritterCounter"})
    with urllib.request.urlopen(request, context=_ssl_context()) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    on_status(
                        f"{label} ({done // 1_048_576} of {total // 1_048_576} MB)",
                        done / total,
                    )
                else:
                    on_status(f"{label} ({done // 1_048_576} MB)", None)

    # Rename only once complete, so an interrupted download is never mistaken for a
    # finished one on the next launch.
    partial.replace(destination)


def uv_executable(root: Path) -> Path:
    return root / "tools" / "uv.exe"


def ensure_uv(root: Path, on_status: StatusCallback) -> Path:
    uv = uv_executable(root)
    if uv.exists():
        return uv

    on_status("Downloading package installer...", None)
    archive = root / "tools" / "uv.zip"
    _download(UV_URL, archive, on_status, "Downloading package installer")

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.endswith("uv.exe"):
                with bundle.open(member) as source, open(uv, "wb") as target:
                    shutil.copyfileobj(source, target)
                break
    archive.unlink(missing_ok=True)

    if not uv.exists():
        raise RuntimeError("uv.exe was not found inside the downloaded archive")
    return uv


def venv_python(root: Path) -> Path:
    return root / "venv" / "Scripts" / "python.exe"


def _requirements_stamp(root: Path) -> str:
    import hashlib

    digest = hashlib.sha1()
    for name in ("requirements.txt", REQUIREMENTS_SPECIESNET):
        path = root / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()


def _install_marker(root: Path) -> Path:
    return root / "venv" / ".critter-counter-installed"


def dependencies_installed(root: Path) -> bool:
    """Whether the venv actually has the packages, not merely that it exists.

    Creating the venv takes a second; filling it takes gigabytes. An interrupted setup
    leaves a valid-looking but empty venv, and treating that as done means the next
    launch skips installation and fails much later with a confusing import error. The
    marker is written only after a successful install, and keyed to requirements.txt so
    changing dependencies triggers a reinstall.
    """

    marker = _install_marker(root)
    if not (venv_python(root).exists() and marker.exists()):
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == _requirements_stamp(root)
    except OSError:
        return False


def ensure_venv(root: Path, on_status: StatusCallback) -> Path:
    python = venv_python(root)
    if dependencies_installed(root):
        return python

    uv = ensure_uv(root, on_status)

    on_status("Setting up Python...", None)
    if not python.exists():
        # --seed puts pip in the venv; uv is used to provision Python (it needs no
        # admin rights) but not to install, see below.
        _run(
            [str(uv), "venv", str(root / "venv"), "--python", "3.12", "--seed"],
            "Creating the Python environment",
        )

    # Two passes, with pip rather than uv, both deliberate. megadetector pins
    # protobuf<=3.20.1 (via ultralytics-yolov5) while speciesnet needs a modern onnx
    # that requires newer protobuf. Resolving them together makes any resolver backtrack
    # to onnx 1.12, which has no Python 3.12 wheel and fails to compile without cmake.
    # Installing in sequence lets the second pass upgrade protobuf and take a prebuilt
    # onnx, which is how the tested environment was actually built. See
    # requirements-speciesnet.txt.
    on_status("Downloading AI libraries (this is the long part)...", None)
    _run(
        [str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
        "Installing the AI libraries",
    )

    on_status("Downloading the species identification library...", None)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(root / REQUIREMENTS_SPECIESNET),
        ],
        "Installing the species identification library",
    )

    _install_marker(root).write_text(_requirements_stamp(root), encoding="utf-8")
    return python


def ensure_models(root: Path, on_status: StatusCallback) -> None:
    detector = root / "models" / "md_v5a.0.1.pt"
    if not detector.exists():
        _download(MEGADETECTOR_URL, detector, on_status, "Downloading detector model")

    speciesnet_dir = root / "models" / "speciesnet"
    if (speciesnet_dir / "info.json").exists():
        return

    # SpeciesNet fetches its own weights through kagglehub into a user-level cache;
    # ask it to do that once, then copy the result in so the app owns it.
    on_status("Downloading species model...", None)
    fetch = (
        "from speciesnet.utils import ModelInfo\n"
        f"print(ModelInfo({SPECIESNET_MODEL_ID!r}).classifier)\n"
    )
    result = subprocess.run(
        [str(venv_python(root)), "-c", fetch],
        capture_output=True,
        text=True,
        creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SetupFailed(
            "Downloading the species model failed.\n\n"
            + "\n".join(detail.splitlines()[-15:])
        )

    cached_dir = Path(result.stdout.strip().splitlines()[-1]).parent
    speciesnet_dir.mkdir(parents=True, exist_ok=True)
    for item in cached_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, speciesnet_dir / item.name)


def is_ready(root: Path) -> bool:
    return (
        dependencies_installed(root)
        and (root / "models" / "md_v5a.0.1.pt").exists()
        and (root / "models" / "speciesnet" / "info.json").exists()
    )


def run_setup(root: Path, on_status: StatusCallback) -> None:
    ensure_venv(root, on_status)
    ensure_models(root, on_status)
    on_status("Ready", 1.0)
