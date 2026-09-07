"""Tests for first-run setup state detection.

Setup downloads gigabytes, so it must be able to tell "half finished" from "done".
Getting this wrong sends someone into the app with an empty environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bootstrap  # noqa: E402


def _fake_install(root: Path, *, with_marker: bool, requirements: str = "openpyxl\n"):
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "python.exe").write_bytes(b"")
    if with_marker:
        bootstrap._install_marker(root).write_text(
            bootstrap._requirements_stamp(root), encoding="utf-8"
        )


def test_interrupted_install_is_not_mistaken_for_finished(tmp_path):
    """A venv exists after ~1 second; filling it takes gigabytes."""

    _fake_install(tmp_path, with_marker=False)

    assert bootstrap.venv_python(tmp_path).exists()
    assert not bootstrap.dependencies_installed(tmp_path)
    assert not bootstrap.is_ready(tmp_path)


def test_completed_install_is_recognised(tmp_path):
    _fake_install(tmp_path, with_marker=True)
    assert bootstrap.dependencies_installed(tmp_path)


def test_changed_requirements_trigger_reinstall(tmp_path):
    _fake_install(tmp_path, with_marker=True)
    assert bootstrap.dependencies_installed(tmp_path)

    (tmp_path / "requirements.txt").write_text("openpyxl\nnumpy\n", encoding="utf-8")
    assert not bootstrap.dependencies_installed(tmp_path)


def test_is_ready_requires_models_too(tmp_path):
    _fake_install(tmp_path, with_marker=True)
    assert not bootstrap.is_ready(tmp_path), "models are still missing"

    models = tmp_path / "models"
    (models / "speciesnet").mkdir(parents=True)
    (models / "md_v5a.0.1.pt").write_bytes(b"")
    (models / "speciesnet" / "info.json").write_text("{}", encoding="utf-8")

    assert bootstrap.is_ready(tmp_path)


def test_partial_download_is_not_counted_as_a_model(tmp_path):
    """Interrupted downloads land in .part and must not satisfy the check."""

    _fake_install(tmp_path, with_marker=True)
    models = tmp_path / "models"
    (models / "speciesnet").mkdir(parents=True)
    (models / "md_v5a.0.1.pt.part").write_bytes(b"half a file")
    (models / "speciesnet" / "info.json").write_text("{}", encoding="utf-8")

    assert not bootstrap.is_ready(tmp_path)
