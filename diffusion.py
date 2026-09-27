"""
Section "Graph Diffusion Model for Adaptive Graph Evolution" (Eq. 15-17).

Eq. 15  X_t = sqrt(alpha_bar_t) X_0 + sqrt(1 - alpha_bar_t) eps,  eps ~ N(0, I)
Eq. 16  X0_hat = ( X_t - sqrt(1 - alpha_bar_t) eps_theta(X_t, G^(t), t) ) / sqrt(alpha_bar_t)
Eq. 17  X_{t-1} = 1/sqrt(alpha_t) * ( X_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta(X_t, F^(t), E^(t), t) ) + sigma_t z

Standard linear-beta-schedule DDPM machinery, with the noise estimator
conditioned on the dynamically-learned graph (F^(t)) and entropy state
(E^(t)) as the paper specifies, via a pooled graph-context vector.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiffusionSchedule:
    def __init__(self, n_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


class NoiseEstimator(nn.Module):
    """eps_theta(X_t, F^(t), E^(t), t): an MLP over [flattened X_t | pooled graph context | time embedding]."""

    def __init__(self, node_dim: int, n_nodes: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.time_embed = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        in_dim = n_nodes * node_dim + n_nodes + n_nodes + time_dim  # X_t, pooled F row-sums, E, time
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_nodes * node_dim),
        )

    def forward(self, X_t: torch.Tensor, F_graph: torch.Tensor, E: torch.Tensor, t: int) -> torch.Tensor:
        # X_t: [N, node_dim]; F_graph: [N, N]; E: [N]
        t_tensor = torch.tensor([[float(t)]])
        t_emb = self.time_embed(t_tensor).squeeze(0)
        graph_ctx = F_graph.sum(dim=1)  # [N] pooled association strength per node
        flat = torch.cat([X_t.reshape(-1), graph_ctx, E, t_emb])
        out = self.net(flat)
        return out.view(self.n_nodes, self.node_dim)


def forward_diffuse(X0: torch.Tensor, t: int, schedule: DiffusionSchedule):
    """Eq. 15."""
    alpha_bar_t = schedule.alpha_bars[t]
    eps = torch.randn_like(X0)
    X_t = alpha_bar_t.sqrt() * X0 + (1 - alpha_bar_t).sqrt() * eps
    return X_t, eps


def estimate_x0(X_t: torch.Tensor, eps_pred: torch.Tensor, t: int, schedule: DiffusionSchedule) -> torch.Tensor:
    """Eq. 16."""
    alpha_bar_t = schedule.alpha_bars[t]
    return (X_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()


def reverse_step(X_t: torch.Tensor, eps_pred: torch.Tensor, t: int, schedule: DiffusionSchedule) -> torch.Tensor:
    """Eq. 17: one reverse-diffusion denoising step, X_t -> X_{t-1}."""
    alpha_t = schedule.alphas[t]
    alpha_bar_t = schedule.alpha_bars[t]
    beta_t = schedule.betas[t]
    mean = (1.0 / alpha_t.sqrt()) * (X_t - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eps_pred)
    if t > 0:
        z = torch.randn_like(X_t)
        sigma_t = beta_t.sqrt()
        return mean + sigma_t * z
    return mean
