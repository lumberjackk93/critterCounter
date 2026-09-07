"""Tests for subprocess progress streaming.

The reader sits in the data path of a long-running job: if it drains stderr too
slowly the pipe fills and the child blocks mid-inference, so this checks that
progress is parsed, output is passed through, and failures still raise.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "critter_counter"))

from pipeline import _run_with_progress, terminate_active_process  # noqa: E402

# Emits a tqdm-style bar rewritten in place with carriage returns, like the real tools.
EMITTER = (
    "import sys\n"
    "for i in range(1, 51):\n"
    "    sys.stderr.write(f'\\r {i*2}%|##### | {i}/50 [00:01<00:02, 1.5it/s]')\n"
    "    sys.stderr.flush()\n"
    "sys.stderr.write('\\nDone\\n')\n"
)


def test_parses_progress_and_reaches_completion():
    seen = []
    _run_with_progress([sys.executable, "-c", EMITTER], seen.append)

    assert seen, "expected progress callbacks"
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert seen == sorted(seen), "progress should never go backwards"


def test_passes_output_through_to_stderr(capfd):
    _run_with_progress([sys.executable, "-c", EMITTER], lambda _: None)

    captured = capfd.readouterr()
    assert "Done" in captured.err, "child output must still reach the log"


def test_raises_on_failure():
    failing = "import sys; sys.stderr.write('1/2\\n'); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError):
        _run_with_progress([sys.executable, "-c", failing], lambda _: None)


def test_terminate_reports_false_when_nothing_is_running():
    assert terminate_active_process() is False


def test_terminate_stops_a_long_running_child():
    """Closing the window must actually stop hours of background work."""

    long_running = (
        "import sys, time\n"
        "sys.stderr.write('1/1000\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(120)\n"
    )
    failures = []

    def run():
        try:
            _run_with_progress([sys.executable, "-c", long_running], lambda _: None)
        except subprocess.CalledProcessError:
            pass  # expected: terminating the child gives a non-zero exit
        except Exception as exc:  # pragma: no cover - surfaces unexpected breakage
            failures.append(exc)

    worker = threading.Thread(target=run)
    worker.start()

    deadline = time.time() + 15
    while time.time() < deadline and not terminate_active_process():
        time.sleep(0.1)

    worker.join(timeout=20)
    assert not worker.is_alive(), "child was not stopped"
    assert not failures, f"unexpected error: {failures}"


def test_handles_large_output_without_deadlocking():
    """A chatty child must not fill the pipe buffer and stall."""

    noisy = (
        "import sys\n"
        "for i in range(1, 2001):\n"
        "    sys.stderr.write('x' * 200 + f' {i}/2000\\n')\n"
    )
    seen = []
    _run_with_progress([sys.executable, "-c", noisy], seen.append)
    assert seen[-1] == pytest.approx(1.0)
