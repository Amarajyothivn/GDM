"""
Section "Disorder Risk Scoring" (Eq. 32-34).

Eq. 32  F_bar^P = (1/m_p) sum_k F_k^P                       (representative disorder-affected graph)
Eq. 33  Dif_r   = sum_i sum_j |F_{r,ij}^G - F_bar_ij^P|      (topological difference)
Eq. 34  Sim_r   = exp( -Dif_r^2 / (2 sigma_r^2) )            (Gaussian-membership risk score)
"""
from __future__ import annotations

from typing import List

import torch


def representative_disorder_graph(disorder_graphs: List[torch.Tensor]) -> torch.Tensor:
    """Eq. 32. disorder_graphs: list of [N, N] association matrices from real disorder-affected samples."""
    return torch.stack(disorder_graphs, dim=0).mean(dim=0)


def topological_difference(generated_graph: torch.Tensor, representative_graph: torch.Tensor) -> torch.Tensor:
    """Eq. 33."""
    return (generated_graph - representative_graph).abs().sum()


def disorder_risk_score(generated_graph: torch.Tensor, representative_graph: torch.Tensor,
                         sigma_r: float = 1.0) -> torch.Tensor:
    """Eq. 34: risk score in (0, 1], 1 meaning the generated graph exactly matches the
    representative disorder-affected pattern."""
    Dif_r = topological_difference(generated_graph, representative_graph)
    return torch.exp(-(Dif_r ** 2) / (2 * sigma_r ** 2))
