"""The code tour (docs/#code) must explain the code as it is: every line covered, copies up to date."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_code_tour_notes_cover_current_code():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_code_tour.py"), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
