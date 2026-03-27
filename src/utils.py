"""Utility helpers for reproducibility, analytical solutions, and error metrics."""

from __future__ import annotations
import random
from typing import Callable, Tuple
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def heat_equation_analytical_solution(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Analytical solution for u_t = alpha * u_xx with u(x,0)=sin(pi x), u(0,t)=u(1,t)=0.

    Returns a 2D array of shape (len(t), len(x)).
    """
    x_grid, t_grid = np.meshgrid(x, t, indexing="xy")
    return np.exp(-(np.pi**2) * alpha * t_grid) * np.sin(np.pi * x_grid)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error between two arrays."""
    return float(np.mean((y_true - y_pred) ** 2))

def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative L2 error ||y_true - y_pred||_2 / ||y_true||_2."""
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true)
    if denominator == 0.0:
        return float(numerator)
    return float(numerator / denominator)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach and move a tensor to CPU numpy format."""
    return tensor.detach().cpu().numpy()

def default_initial_condition(x: np.ndarray) -> np.ndarray:
    """Default smooth initial condition u(x, 0) = sin(pi x)."""
    return np.sin(np.pi * x)

def default_boundary_conditions() -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """Default homogeneous Dirichlet boundaries u(0,t)=0 and u(1,t)=0."""

    def left_boundary(_time: float) -> float:
        return 0.0

    def right_boundary(_time: float) -> float:
        return 0.0

    return left_boundary, right_boundary