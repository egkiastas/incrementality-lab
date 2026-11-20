"""Download, verify, convert and load the Criteo Uplift v2.1 file."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

# The official link has been returning 404; the GitHub mirror serves the same bytes (both
# checksums below are published by independent projects: libuplift and scikit-uplift).
URLS = [
    "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
    "https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/"
    "criteo-research-uplift-v2.1.csv.gz",
]
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
MD5 = "d2236769ef69e9be52556110102911ec"
BINARY_COLS = ["treatment", "exposure", "visit", "conversion"]


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        for url in URLS:
            try:
                tmp = dest.with_suffix(".part")
                req = urllib.request.Request(url, headers={"User-Agent": "incrementality-lab"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out, length=1 << 20)
                tmp.rename(dest)
                break
            except OSError as err:
                print(f"failed {url}: {err}")
        else:
            raise RuntimeError("every download URL failed")
    if _hash(dest, "sha256") != SHA256 or _hash(dest, "md5") != MD5:
        raise ValueError(f"checksum mismatch for {dest}")
    return dest


def convert_to_parquet(raw_csv: Path, parquet: Path, expected_rows: int) -> None:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    feats = ", ".join(f"CAST(f{i} AS DOUBLE) AS f{i}" for i in range(12))
    bins = ", ".join(f"CAST({c} AS TINYINT) AS {c}" for c in BINARY_COLS)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute(
        f"""COPY (SELECT {feats}, {bins} FROM read_csv_auto('{raw_csv}', header=true))
            TO '{parquet}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)"""
    )
    row = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet}')").fetchone()
    con.close()
    if row is None or row[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {row}")


@dataclass
class Dataset:
    X: np.ndarray  # float32, (n, 12)
    treatment: np.ndarray  # int8
    exposure: np.ndarray
    visit: np.ndarray
    conversion: np.ndarray
    feature_names: list[str]

    @property
    def n(self) -> int:
        return int(self.treatment.shape[0])

    def outcome(self, name: str) -> np.ndarray:
        return {"visit": self.visit, "conversion": self.conversion}[name]


def load(parquet: Path, features: tuple[str, ...], with_features: bool = True) -> Dataset:
    """Columns as float32 / int8, so all 14M rows take well under 1 GB."""
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found - run analysis/00_download_data.py first")
    feats = list(features) if with_features else []
    df = pl.read_parquet(parquet, columns=feats + BINARY_COLS)
    X = df.select(feats).to_numpy().astype(np.float32) if feats else np.empty((df.height, 0))
    return Dataset(
        X=X,
        treatment=df["treatment"].to_numpy().astype(np.int8),
        exposure=df["exposure"].to_numpy().astype(np.int8),
        visit=df["visit"].to_numpy().astype(np.int8),
        conversion=df["conversion"].to_numpy().astype(np.int8),
        feature_names=feats,
    )


def strata(ds: Dataset) -> np.ndarray:
    """treatment x visit cell, used to stratify every split."""
    return (ds.treatment * 2 + ds.visit).astype(np.int8)


def stratified_split(
    cells: np.ndarray, fracs: dict[str, float], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    names = list(fracs)
    out: dict[str, list[np.ndarray]] = {k: [] for k in names}
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        rng.shuffle(idx)
        start = 0
        for j, k in enumerate(names):
            stop = len(idx) if j == len(names) - 1 else start + round(fracs[k] * len(idx))
            out[k].append(idx[start:stop])
            start = stop
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def stratified_subsample(idx: np.ndarray, cells: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n >= len(idx):
        return idx
    frac = n / len(idx)
    parts = []
    c_idx = cells[idx]
    for c in np.unique(c_idx):
        pool = idx[c_idx == c]
        parts.append(rng.choice(pool, size=round(frac * len(pool)), replace=False))
    return np.sort(np.concatenate(parts))


def train_val_test_split(
    cells: np.ndarray, test_frac: float, val_frac: float, seed: int
) -> dict[str, np.ndarray]:
    fracs = {"test": test_frac, "val": val_frac, "train": 1 - test_frac - val_frac}
    return stratified_split(cells, fracs, np.random.default_rng(seed))
