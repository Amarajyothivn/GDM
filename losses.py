"""
Section "Loss Function and Model Training" (Eq. 24-26).

Eq. 24  L_G = -E[D_omega(x~)] + lambda_diff * L_diff + lambda_DGI * L_DGI
Eq. 25  L_diff = E_{t,eps}[ || eps - eps_theta(X_t, F^(t), E^(t), t) ||_2^2 ]
Eq. 26  min_theta max_omega  E[D_omega(x)] - E[D_omega(x~)] + lambda_diff*L_diff + lambda_DGI*L_DGI

lambda_diff, lambda_DGI are exposed as constructor-style arguments below;
the paper names them but does not give numeric values (same pattern as the
FL-NIDS paper's alpha/beta/gamma, sim_min etc.) -- pick values that suit
your own training run.
"""
from __future__ import annotations

import torch


def diffusion_loss(eps_true: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
    """Eq. 25."""
    return ((eps_true - eps_pred) ** 2).sum()


def generator_loss(critic_score_fake: torch.Tensor, diff_loss: torch.Tensor, dgi_loss: torch.Tensor,
                    lambda_diff: float = 1.0, lambda_dgi: float = 1.0) -> torch.Tensor:
    """Eq. 24: L_G = -E[D_omega(x~)] + lambda_diff*L_diff + lambda_DGI*L_DGI."""
    return -critic_score_fake.mean() + lambda_diff * diff_loss + lambda_dgi * dgi_loss


def joint_objective(critic_score_real: torch.Tensor, critic_score_fake: torch.Tensor,
                     diff_loss: torch.Tensor, dgi_loss: torch.Tensor,
                     lambda_diff: float = 1.0, lambda_dgi: float = 1.0) -> torch.Tensor:
    """Eq. 26's scalar value (for logging/monitoring the minimax game; the actual
    optimization alternates critic-ascent and generator-descent steps, see training.py)."""
    return (critic_score_real.mean() - critic_score_fake.mean()
            + lambda_diff * diff_loss + lambda_dgi * dgi_loss)
