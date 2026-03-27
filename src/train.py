"""Training pipeline for the 1D heat equation PINN and comparison with FDM."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import torch
from torch import optim
from fdm_solver import solve_heat_equation_ftcs
from pinn_model import PINN, compute_heat_equation_losses
from utils import (
    default_boundary_conditions,
    default_initial_condition,
    heat_equation_analytical_solution,
    l2_relative_error,
    mse,
    set_seed,
    to_numpy,
)

@dataclass
class TrainingConfig:
    """Configuration for PDE domain, model, and optimization."""

    x_domain: tuple[float, float] = (0.0, 1.0)
    t_domain: tuple[float, float] = (0.0, 1.0)
    alpha: float = 0.1

    nx: int = 101
    nt: int = 4001

    n_interior: int = 2000
    n_boundary: int = 256
    n_initial: int = 256

    epochs: int = 3000
    log_every: int = 100
    learning_rate: float = 1e-3
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    seed: int = 42

@dataclass
class TrainingArtifacts:
    """Artifacts produced by training and evaluation."""

    model: PINN
    x: np.ndarray
    t: np.ndarray
    u_fdm: np.ndarray
    u_pinn: np.ndarray
    loss_history: dict[str, list[float]]
    metrics: dict[str, float]


def high_accuracy_config() -> TrainingConfig:
    """Return a stronger training configuration for improved PINN accuracy."""
    return TrainingConfig(
        epochs=3000,
        log_every=200,
        learning_rate=5e-4,
        n_interior=4000,
        n_boundary=512,
        n_initial=512,
    )

def _sample_points(config: TrainingConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Uniformly sample interior, boundary, and initial points from the domain."""
    x_min, x_max = config.x_domain
    t_min, t_max = config.t_domain

    x_interior = torch.rand(config.n_interior, 1, device=device) * (x_max - x_min) + x_min
    t_interior = torch.rand(config.n_interior, 1, device=device) * (t_max - t_min) + t_min
    interior_points = torch.cat([x_interior, t_interior], dim=1)

    x_initial = torch.rand(config.n_initial, 1, device=device) * (x_max - x_min) + x_min
    t_initial = torch.full_like(x_initial, t_min)
    initial_points = torch.cat([x_initial, t_initial], dim=1)
    initial_targets = torch.sin(torch.pi * x_initial)

    t_boundary = torch.rand(config.n_boundary, 1, device=device) * (t_max - t_min) + t_min

    left_x = torch.full_like(t_boundary, x_min)
    right_x = torch.full_like(t_boundary, x_max)

    left_boundary_points = torch.cat([left_x, t_boundary], dim=1)
    right_boundary_points = torch.cat([right_x, t_boundary], dim=1)

    left_boundary_targets = torch.zeros_like(t_boundary)
    right_boundary_targets = torch.zeros_like(t_boundary)

    return {
        "interior_points": interior_points,
        "initial_points": initial_points,
        "initial_targets": initial_targets,
        "left_boundary_points": left_boundary_points,
        "right_boundary_points": right_boundary_points,
        "left_boundary_targets": left_boundary_targets,
        "right_boundary_targets": right_boundary_targets,
    }

def _predict_on_grid(
    model: PINN,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate a trained model on a Cartesian grid and return shape (nt, nx)."""
    x_mesh, t_mesh = np.meshgrid(x, t, indexing="xy")
    coordinates = np.column_stack([x_mesh.ravel(), t_mesh.ravel()])

    with torch.no_grad():
        inputs = torch.tensor(coordinates, dtype=torch.float32, device=device)
        outputs = model(inputs).reshape(len(t), len(x))

    return to_numpy(outputs)

def train_and_compare(config: TrainingConfig | None = None) -> TrainingArtifacts:
    """Train the PINN, solve the same problem with FDM, and return comparison artifacts."""
    config = config or TrainingConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN(in_features=2, hidden_features=24, hidden_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history = {"total": [], "pde": [], "ic": [], "bc": []}

    for epoch in range(1, config.epochs + 1):
        batch = _sample_points(config, device)

        optimizer.zero_grad()
        total_loss, components = compute_heat_equation_losses(
            model=model,
            interior_points=batch["interior_points"],
            initial_points=batch["initial_points"],
            initial_targets=batch["initial_targets"],
            left_boundary_points=batch["left_boundary_points"],
            right_boundary_points=batch["right_boundary_points"],
            left_boundary_targets=batch["left_boundary_targets"],
            right_boundary_targets=batch["right_boundary_targets"],
            alpha=config.alpha,
            weights=config.weights,
        )
        total_loss.backward()
        optimizer.step()

        loss_history["total"].append(float(total_loss.item()))
        loss_history["pde"].append(float(components.pde_loss.item()))
        loss_history["ic"].append(float(components.ic_loss.item()))
        loss_history["bc"].append(float(components.bc_loss.item()))

        should_log = (
            epoch == 1
            or epoch == config.epochs
            or (config.log_every > 0 and epoch % config.log_every == 0)
        )
        if should_log:
            print(
                f"Epoch {epoch:5d}/{config.epochs} "
                f"total={loss_history['total'][-1]:.4e} "
                f"pde={loss_history['pde'][-1]:.4e} "
                f"ic={loss_history['ic'][-1]:.4e} "
                f"bc={loss_history['bc'][-1]:.4e}"
            )

    x_fdm, t_fdm, u_fdm = solve_heat_equation_ftcs(
        x_domain=config.x_domain,
        t_domain=config.t_domain,
        nx=config.nx,
        nt=config.nt,
        alpha=config.alpha,
        initial_condition=default_initial_condition,
        boundary_conditions=default_boundary_conditions(),
    )

    u_pinn = _predict_on_grid(model, x_fdm, t_fdm, device)
    u_true = heat_equation_analytical_solution(x_fdm, t_fdm, config.alpha)

    metrics = {
        "mse_pinn_vs_fdm": mse(u_fdm, u_pinn),
        "l2_pinn_vs_fdm": l2_relative_error(u_fdm, u_pinn),
        "mse_pinn_vs_exact": mse(u_true, u_pinn),
        "l2_pinn_vs_exact": l2_relative_error(u_true, u_pinn),
    }

    return TrainingArtifacts(
        model=model,
        x=x_fdm,
        t=t_fdm,
        u_fdm=u_fdm,
        u_pinn=u_pinn,
        loss_history=loss_history,
        metrics=metrics,
    )

def save_model(model: PINN, output_path: Path) -> None:
    """Save trained model weights to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def export_artifacts(
    artifacts: TrainingArtifacts,
    output_dir: Path | str = Path("outputs"),
    model_path: Path | str | None = None,
) -> dict[str, Path]:
    """Export metrics, loss history, model weights, and plots to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in artifacts.metrics.items():
            writer.writerow([key, value])

    loss_path = output_dir / "loss_history.csv"
    epochs = len(artifacts.loss_history["total"])
    with loss_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "total", "pde", "ic", "bc"])
        for i in range(epochs):
            writer.writerow(
                [
                    i + 1,
                    artifacts.loss_history["total"][i],
                    artifacts.loss_history["pde"][i],
                    artifacts.loss_history["ic"][i],
                    artifacts.loss_history["bc"][i],
                ]
            )

    from visualization import plot_heatmap, plot_loss_history, plot_time_slices

    loss_plot_path = output_dir / "loss_history.png"
    fdm_plot_path = output_dir / "fdm_heatmap.png"
    pinn_plot_path = output_dir / "pinn_heatmap.png"
    comparison_plot_path = output_dir / "comparison_time_slices.png"

    plot_loss_history(artifacts.loss_history, output_path=loss_plot_path)
    plot_heatmap(
        artifacts.u_fdm,
        artifacts.x,
        artifacts.t,
        "FDM Solution",
        output_path=fdm_plot_path,
    )
    plot_heatmap(
        artifacts.u_pinn,
        artifacts.x,
        artifacts.t,
        "PINN Solution",
        output_path=pinn_plot_path,
    )
    plot_time_slices(
        artifacts.x,
        artifacts.t,
        artifacts.u_fdm,
        artifacts.u_pinn,
        time_indices=[0, len(artifacts.t) // 4, len(artifacts.t) // 2, -1],
        output_path=comparison_plot_path,
    )

    resolved_model_path = Path(model_path) if model_path else output_dir / "pinn_model.pt"
    save_model(artifacts.model, resolved_model_path)

    return {
        "metrics_csv": metrics_path,
        "loss_csv": loss_path,
        "loss_plot": loss_plot_path,
        "fdm_heatmap": fdm_plot_path,
        "pinn_heatmap": pinn_plot_path,
        "comparison_plot": comparison_plot_path,
        "model": resolved_model_path,
    }

if __name__ == "__main__":
    artifacts = train_and_compare()
    print("Metrics:")
    for key, value in artifacts.metrics.items():
        print(f"  {key}: {value:.6e}")

    exported = export_artifacts(artifacts, output_dir=Path("outputs"))
    print("Exported files:")
    for key, value in exported.items():
        print(f"  {key}: {value}")