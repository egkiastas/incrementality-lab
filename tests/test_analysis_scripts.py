"""Every analysis script runs end to end on the synthetic fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = sorted(Path("analysis").glob("0*.py"))


def test_scripts_run_on_fixture() -> None:
    for script in SCRIPTS:
        done = subprocess.run([sys.executable, str(script), "--fixture"], capture_output=True, text=True)
        assert done.returncode == 0, f"{script.name}\n{done.stderr[-2000:]}"
    results = Path("build/fixture/results")
    for name in ("health", "ate", "cuped", "sequential", "uplift", "policy", "learners", "synthetic_check"):
        assert json.loads((results / f"{name}.json").read_text())["meta"]["step"] == name
    assert len(list(Path("build/fixture/figures").glob("*.png"))) >= 14
