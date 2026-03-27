"""Finite Difference Method (FTCS) solver for the 1D heat equation."""

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

BoundaryFunction = Callable[[float], float]
InitialCondition = Callable[[np.ndarray], np.ndarray]

def solve_heat_equation_ftcs(
    x_domain: Tuple[float, float],
    t_domain: Tuple[float, float],
    nx: int,
    nt: int,
    alpha: float,
    initial_condition: InitialCondition,
    boundary_conditions: Tuple[BoundaryFunction, BoundaryFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve u_t = alpha * u_xx with explicit FTCS.

    Args:
        x_domain: Spatial range as (x_min, x_max).
        t_domain: Time range as (t_min, t_max).
        nx: Number of spatial grid points.
        nt: Number of temporal grid points.
        alpha: Diffusion coefficient.
        initial_condition: Function for u(x, t_min).
        boundary_conditions: Tuple of (left_boundary, right_boundary) functions in time.

    Returns:
        Tuple (x, t, u) where u has shape (nt, nx).

    Raises:
        ValueError: If grid settings violate FTCS stability condition.
    """
    x_min, x_max = x_domain
    t_min, t_max = t_domain
    left_boundary, right_boundary = boundary_conditions

    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)

    dx = (x_max - x_min) / (nx - 1)
    dt = (t_max - t_min) / (nt - 1)

    stability_limit = (dx**2) / (2.0 * alpha)
    if dt > stability_limit:
        raise ValueError(
            "FTCS stability condition violated: "
            f"dt={dt:.6e} > dx^2/(2*alpha)={stability_limit:.6e}. "
            "Increase nt or decrease nx/alpha."
        )

    r = alpha * dt / (dx**2)
    u = np.zeros((nt, nx), dtype=float)

    u[0, :] = initial_condition(x)
    u[0, 0] = left_boundary(t_min)
    u[0, -1] = right_boundary(t_min)

    for n in range(0, nt - 1):
        u[n + 1, 0] = left_boundary(t[n + 1])
        u[n + 1, -1] = right_boundary(t[n + 1])

        u[n + 1, 1:-1] = (
            u[n, 1:-1]
            + r * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2])
        )

    return x, t, u