"""Shared chart style. Learners keep the same colour in every figure; grey is the baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from cycler import cycler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
BASELINE = "#9b9a96"

LEARNER_COLORS = {
    "s_learner": SERIES[0],
    "t_learner": SERIES[1],
    "x_learner": SERIES[2],
    "dr_learner": SERIES[3],
    "causal_forest": SERIES[6],
    "response_model": SERIES[7],
    "random": BASELINE,
}
LEARNER_LABELS = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "dr_learner": "DR-learner",
    "causal_forest": "Causal forest",
    "response_model": "Response model (ignores T)",
    "random": "Random",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=SERIES),
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def new_figure(w: float = 7.0, h: float = 4.2) -> tuple[Figure, Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def new_figures(ncols: int, w: float = 10.0, h: float = 4.0) -> tuple[Figure, list[Axes]]:
    apply_style()
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), squeeze=False)
    return fig, list(axes[0])


def subtitle(ax: Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def save(fig: Figure, figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def learner_label(name: str) -> str:
    if name.startswith("visit_model__"):
        return LEARNER_LABELS[name.split("__")[1]] + " (visit-trained)"
    return LEARNER_LABELS.get(name, name)
