"""Visualization utilities for heat equation solutions and training diagnostics."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    title: str,
    output_path: Path | None = None,
) -> None:
    """Plot a heatmap of u(x, t) where u shape is (nt, nx)."""
    plt.figure(figsize=(9, 4.5))
    image = plt.imshow(
        u,
        extent=[x.min(), x.max(), t.max(), t.min()],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(image, label="u(x, t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.show()

def plot_time_slices(
    x: np.ndarray,
    t: np.ndarray,
    u_reference: np.ndarray,
    u_prediction: np.ndarray,
    time_indices: Sequence[int],
    output_path: Path | None = None,
) -> None:
    """Plot line comparisons between reference and predicted solutions at selected times."""
    plt.figure(figsize=(9, 5))

    for idx in time_indices:
        label_time = f"t={t[idx]:.3f}"
        plt.plot(x, u_reference[idx], linewidth=2, label=f"Reference ({label_time})")
        plt.plot(x, u_prediction[idx], "--", linewidth=2, label=f"Prediction ({label_time})")

    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("PINN vs Reference at Fixed Time Slices")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.show()

def plot_loss_history(
    loss_history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    """Plot total and component losses over training epochs."""
    plt.figure(figsize=(9, 5))

    for key, values in loss_history.items():
        plt.plot(values, label=key)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Components")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.show()