from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .core import AIRSuiteResult, AIRTrainingResult


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_training_curves(result: AIRTrainingResult, output_path: str | Path) -> None:
    out = _ensure_dir(output_path)

    epochs = np.arange(len(result.loss_history))
    loss = np.asarray(result.loss_history)
    acc = np.asarray(result.accuracy_history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(epochs, loss, marker="o")
    ax1.set_ylabel("Objective")
    ax1.set_title("AIR training history")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, acc, marker="o", color="#1f77b4")
    ax2.set_ylabel("Count accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_suite_summary(
    results: Sequence[AIRSuiteResult], output_path: str | Path
) -> None:
    out = _ensure_dir(output_path)

    names = [r.estimator for r in results]
    accuracies = np.asarray([r.final_accuracy for r in results])
    objectives = np.asarray([r.objective_mean for r in results])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.bar(names, objectives, color="#4e79a7")
    ax1.set_ylabel("Objective mean")
    ax1.set_title("AIR estimator comparison")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(names, accuracies, color="#59a14f")
    ax2.set_ylabel("Final accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
