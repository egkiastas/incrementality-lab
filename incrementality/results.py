"""Each analysis step saves its numbers as one JSON file in report/results/."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def save(results_dir: Path, name: str, payload: dict[str, Any], seed: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {"step": name, "seed": seed, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    text = json.dumps(_plain({"meta": meta, **payload}), indent=2)
    (results_dir / f"{name}.json").write_text(text)


def load(results_dir: Path, name: str) -> dict[str, Any]:
    path = results_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the earlier analysis step first")
    data: dict[str, Any] = json.loads(path.read_text())
    return data
