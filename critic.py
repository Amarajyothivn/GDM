"""
Section "Wasserstein Critic-Based Discriminator Network" (Eq. 22-23).

Eq. 22  D_omega(H) = f_omega(H)
Eq. 23  L_C = E_{x~Pg}[D_omega(x~)] - E_{x~Pr}[D_omega(x)]
              + lambda_gp * E_{x_hat}[ (||grad D_omega(x_hat)||_2 - 1)^2 ]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WassersteinCritic(nn.Module):
    """f_omega: scores a flattened [N, N] adaptive-association (graph) matrix H (Eq. 22)."""

    def __init__(self, n_nodes: int, mlp_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes * n_nodes, mlp_hidden), nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden, mlp_hidden), nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.net(H.reshape(-1))


def gradient_penalty(critic: WassersteinCritic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(1)
    x_hat = (eps * real + (1 - eps) * fake).detach().requires_grad_(True)
    score = critic(x_hat)
    grad = torch.autograd.grad(score, x_hat, grad_outputs=torch.ones_like(score),
                                create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2) - 1) ** 2)


def critic_loss(critic: WassersteinCritic, real: torch.Tensor, fake: torch.Tensor, lambda_gp: float = 10.0):
    """Eq. 23."""
    d_fake = critic(fake)
    d_real = critic(real)
    gp = gradient_penalty(critic, real, fake)
    return (d_fake - d_real).mean() + lambda_gp * gp.mean()
