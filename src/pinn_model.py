"""PINN architecture and physics-informed loss computation for 1D heat equation."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

class PINN(nn.Module):
    """Fully connected PINN mapping (x, t) -> u(x, t)."""

    def __init__(self, in_features: int = 2, hidden_features: int = 24, hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        current_features = in_features

        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_features, hidden_features))
            layers.append(nn.Tanh())
            current_features = hidden_features

        layers.append(nn.Linear(current_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Forward pass for concatenated coordinates (x, t)."""
        return self.network(x_t)

@dataclass
class LossComponents:
    """Container for PINN loss components."""

    pde_loss: torch.Tensor
    ic_loss: torch.Tensor
    bc_loss: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        """Return weighted total loss placeholder sum when already weighted externally."""
        return self.pde_loss + self.ic_loss + self.bc_loss

def _gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute first derivative of outputs with respect to inputs."""
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

def compute_heat_equation_losses(
    model: PINN,
    interior_points: torch.Tensor,
    initial_points: torch.Tensor,
    initial_targets: torch.Tensor,
    left_boundary_points: torch.Tensor,
    right_boundary_points: torch.Tensor,
    left_boundary_targets: torch.Tensor,
    right_boundary_targets: torch.Tensor,
    alpha: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, LossComponents]:
    """Compute weighted total loss and individual components for the 1D heat equation."""
    interior_points = interior_points.requires_grad_(True)
    u_interior = model(interior_points)

    grads = _gradient(u_interior, interior_points)
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = _gradient(u_x, interior_points)[:, 0:1]

    residual = u_t - alpha * u_xx
    pde_loss = torch.mean(residual**2)

    ic_pred = model(initial_points)
    ic_loss = torch.mean((ic_pred - initial_targets) ** 2)

    left_pred = model(left_boundary_points)
    right_pred = model(right_boundary_points)
    bc_loss = torch.mean((left_pred - left_boundary_targets) ** 2) + torch.mean(
        (right_pred - right_boundary_targets) ** 2
    )

    w_pde, w_ic, w_bc = weights
    total_loss = w_pde * pde_loss + w_ic * ic_loss + w_bc * bc_loss

    return total_loss, LossComponents(pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss)