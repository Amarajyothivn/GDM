"""
Section "Dynamic Graph Learning-Based Adaptive Graph Construction" (Eq. 1-4),
and the reconstruction / dynamic-correlation-update steps reused later
in the DGI propagation loop (Eq. 10-12).

Eq. 1   D_ij = || f_i - f_j ||_2                          (direct association, Euclidean)
Eq. 2   S_ij = sum_{r != i,j} |C_ir - C_jr|                (indirect association)
Eq. 3   C_ij = Pearson correlation of f_i, f_j
Eq. 4   F_ij^(t) = exp( -( lambda_ij D_ij + (1-lambda_ij) S_ij - mu^2 ) / (2 sigma^2) )
Eq. 10  D^(t+1) = Euclid(H^(t+1)), S^(t+1) from C^(t+1)
Eq. 11  G^(t+1) = Reconst(D^(t+1), S^(t+1), F^(t+1))       (= recompute F via Eq.4)
Eq. 12  C_ij^(t+1) = C_ij^(t) - eta_c |DeltaE_i^(t+1) - DeltaE_j^(t)|

lambda_ij (the direct/indirect balance) is never given a numeric value or an
update rule in the archived methods text, so it is modelled here as a
learnable per-edge parameter in [0,1] (sigmoid-parametrized), consistent with
"adaptive coefficient" in the prose. eta_c in Eq. 12 is likewise not
numerically specified and is exposed as a constructor argument.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pairwise_euclidean(F: torch.Tensor) -> torch.Tensor:
    """Eq. 1: D_ij for all pairs. F: [N, l] -> [N, N]."""
    return torch.cdist(F, F, p=2)


def pearson_matrix(F: torch.Tensor) -> torch.Tensor:
    """Eq. 3: C_ij, Pearson correlation for all node-feature pairs. F: [N, l] -> [N, N]."""
    Fc = F - F.mean(dim=1, keepdim=True)
    num = Fc @ Fc.T
    denom = Fc.norm(dim=1, keepdim=True) @ Fc.norm(dim=1, keepdim=True).T
    return num / denom.clamp_min(1e-12)


def indirect_association(C: torch.Tensor) -> torch.Tensor:
    """Eq. 2: S_ij = sum_{r != i,j} |C_ir - C_jr|, vectorized over all pairs."""
    N = C.size(0)
    # diff[i, j, r] = |C_ir - C_jr|
    diff = (C.unsqueeze(1) - C.unsqueeze(0)).abs()  # [N, N, N] indexed [i, j, r]
    S = diff.sum(dim=2)
    # subtract the r=i and r=j terms (Eq. 2 excludes r in {i, j})
    idx = torch.arange(N, device=C.device)
    S -= diff[idx, :, idx]    # remove r == i term for each (i, j)
    S -= diff[:, idx, idx]    # remove r == j term for each (i, j)
    return S


class DynamicGraphLearning(nn.Module):
    """Eq. 1-4: builds the adaptive association matrix F^(t) from node features."""

    def __init__(self, n_nodes: int):
        super().__init__()
        # Eq. 4's adaptive coefficient lambda_ij^(t): not numerically specified
        # in the paper, so learned per-edge and squashed to [0, 1].
        self.lambda_raw = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def forward(self, node_features: torch.Tensor) -> dict:
        D = pairwise_euclidean(node_features)      # Eq. 1
        C = pearson_matrix(node_features)          # Eq. 3
        S = indirect_association(C)                # Eq. 2
        lam = torch.sigmoid(self.lambda_raw)
        combined = lam * D + (1 - lam) * S
        mu, sigma = combined.mean(), combined.std().clamp_min(1e-8)
        F_assoc = torch.exp(-((combined - mu ** 2) / (2 * sigma ** 2)))  # Eq. 4
        off_diag_mask = 1.0 - torch.eye(F_assoc.size(0), device=F_assoc.device)
        F_assoc = F_assoc * off_diag_mask  # zero the diagonal without an in-place autograd hazard
        return {"D": D, "C": C, "S": S, "F": F_assoc}

    def reconstruct(self, node_features: torch.Tensor) -> dict:
        """Eq. 10-11: recompute D, S (and via forward, F) from updated node features H^(t+1)."""
        return self.forward(node_features)

    @staticmethod
    def update_correlation(C: torch.Tensor, delta_E_next: torch.Tensor, delta_E_curr: torch.Tensor,
                            eta_c: float = 0.05) -> torch.Tensor:
        """Eq. 12: C_ij^(t+1) = C_ij^(t) - eta_c |DeltaE_i^(t+1) - DeltaE_j^(t)|.

        delta_E_next, delta_E_curr: [N] per-node entropy increments (Eq. 9).
        """
        diff = (delta_E_next.unsqueeze(1) - delta_E_curr.unsqueeze(0)).abs()  # [N, N]
        return C - eta_c * diff
