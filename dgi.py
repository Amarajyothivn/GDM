"""
Section "Deep Graph Infomax-Based Information Propagation" (Eq. 6-9).

Eq. 6   Z^(t) = Encoder(H^(t), F^(t)),  s^(t) = sigma_g( mean_i Z_i^(t) )
Eq. 7   L_DGI = - sum_i [ log D(Z_i, s) + log(1 - D(Z~_i, s)) ]
Eq. 8   E_i^(t+1) = E_i^(t) + sum_j F_ij^(t) alpha_ij^(t) E_j^(t)
Eq. 9   DeltaE_i^(t) = E_i^(t+1) - E_i^(t);  H_i^(t+1) = H_i^(t) + DeltaE_i^(t) * H_i^(t)

alpha_ij^(t) (Eq. 8's propagation attention) has no separate definition in
the archived methods text, so it is taken here as the row-normalized
adaptive association itself, alpha_ij = softmax_j(F_ij) -- a standard choice
when a GNN's own edge weights are reused as the propagation attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as _F  # aliased to avoid shadowing the F adjacency-matrix name used elsewhere


class GCNEncoder(nn.Module):
    """Eq. 6's Encoder: one graph-convolution step using the adaptive association F^(t) as adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, hidden_dim)

    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        adj_norm = adj / deg
        return torch.tanh(adj_norm @ self.W(H))


class Discriminator(nn.Module):
    """Bilinear D(z, s) used in Eq. 7."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(hidden_dim))

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z: [N, H] or [H]; s: [H]
        return torch.sigmoid((z @ self.W) @ s)


class DeepGraphInfomax(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim)
        self.disc = Discriminator(hidden_dim)

    def encode_and_summarize(self, H: torch.Tensor, adj: torch.Tensor):
        """Eq. 6."""
        Z = self.encoder(H, adj)
        s = torch.sigmoid(Z.mean(dim=0))
        return Z, s

    def loss(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Eq. 7, with the corrupted graph formed by row-shuffling node features (standard DGI corruption)."""
        Z, s = self.encode_and_summarize(H, adj)
        perm = torch.randperm(H.size(0), device=H.device)
        Z_tilde, _ = self.encode_and_summarize(H[perm], adj)

        pos = self.disc(Z, s).clamp(1e-7, 1 - 1e-7)
        neg = self.disc(Z_tilde, s).clamp(1e-7, 1 - 1e-7)
        return -(torch.log(pos).sum() + torch.log(1 - neg).sum())

    def mi_scores(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Eq. 29 reuses this: MI_i = log D(Z_i, s) - log(1 - D(Z~_i, s))."""
        Z, s = self.encode_and_summarize(H, adj)
        perm = torch.randperm(H.size(0), device=H.device)
        Z_tilde, _ = self.encode_and_summarize(H[perm], adj)
        pos = self.disc(Z, s).clamp(1e-7, 1 - 1e-7)
        neg = self.disc(Z_tilde, s).clamp(1e-7, 1 - 1e-7)
        return torch.log(pos) - torch.log(1 - neg)


def propagate_entropy(E: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Eq. 8: E_i^(t+1) = E_i^(t) + sum_j F_ij alpha_ij E_j, with alpha = row-softmax(F)."""
    alpha = _F.softmax(F, dim=1)
    return E + (alpha * F) @ E


def update_node_features(H: torch.Tensor, E: torch.Tensor, E_next: torch.Tensor):
    """Eq. 9: DeltaE_i = E_i^(t+1) - E_i^(t); H_i^(t+1) = H_i^(t) + DeltaE_i * H_i^(t)."""
    delta_E = E_next - E  # [N]
    H_next = H + delta_E.unsqueeze(1) * H
    return delta_E, H_next
