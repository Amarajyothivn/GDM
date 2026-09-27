"""
Section "Graphormer-Based Graph Representation Layers" (Eq. 18-21).

Eq. 18  A_ij^(u) = softmax_j( Q_i K_j^T / sqrt(d_k) + b_ij^sp + gamma * F_ij^(t) )
Eq. 19  H~^(u+1) = LN[ H^(u) + A^(u) V^(u) ]
Eq. 20  H^(u+1)  = LN[ H~^(u+1) + FFN(H~^(u+1)) ]
Eq. 21  DeltaE^(t) = (B (.) E_p^(t)) (.) I,   E_p^(t) = (E^(t) B) (.) F^(t)

b_ij^sp (structural/spatial-distance encoding) is computed from unweighted
shortest-path distance on the graph thresholded from F^(t), embedded through
a learnable lookup table, in line with the original Graphormer's spatial
encoding. gamma is a learnable scalar gate on the association bias, as the
paper's "regulates its contribution" phrasing implies without giving a value.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_nn


def shortest_path_distances(adj: torch.Tensor, threshold: float = 0.3, max_dist: int = 10) -> torch.Tensor:
    """Unweighted BFS shortest-path distance matrix from a thresholded adjacency (Floyd-Warshall)."""
    N = adj.size(0)
    A = (adj > threshold).float()
    A.fill_diagonal_(1.0)
    dist = torch.where(A > 0, torch.ones(N, N), torch.full((N, N), float(max_dist)))
    dist.fill_diagonal_(0.0)
    for k in range(N):
        dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist.clamp(max=max_dist)


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, max_dist: int = 10, ffn_mult: int = 2):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_embed = nn.Embedding(max_dist + 1, n_heads)  # b_ij^sp per head
        self.gamma = nn.Parameter(torch.tensor(1.0))              # Eq. 18's gamma
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult), nn.GELU(), nn.Linear(hidden_dim * ffn_mult, hidden_dim)
        )
        self.max_dist = max_dist

    def forward(self, H: torch.Tensor, F_graph: torch.Tensor) -> torch.Tensor:
        N, d = H.shape
        Q = self.Q(H).view(N, self.n_heads, self.d_k)
        K = self.K(H).view(N, self.n_heads, self.d_k)
        V = self.V(H).view(N, self.n_heads, self.d_k)

        logits = torch.einsum("ihd,jhd->hij", Q, K) / (self.d_k ** 0.5)  # [heads, N, N]

        dist = shortest_path_distances(F_graph, max_dist=self.max_dist).long()  # [N, N]
        b_sp = self.spatial_embed(dist).permute(2, 0, 1)  # [heads, N, N]

        logits = logits + b_sp + self.gamma * F_graph.unsqueeze(0)  # Eq. 18 (pre-softmax)
        A = F_nn.softmax(logits, dim=-1)  # softmax_j, Eq. 18

        out = torch.einsum("hij,jhd->ihd", A, V).reshape(N, d)
        out = self.out_proj(out)

        H_tilde = self.ln1(H + out)          # Eq. 19
        H_next = self.ln2(H_tilde + self.ffn(H_tilde))  # Eq. 20
        return H_next, A.mean(dim=0)  # also return the averaged attention map for downstream use


class Graphormer(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([GraphormerLayer(hidden_dim, n_heads) for _ in range(n_layers)])

    def forward(self, H: torch.Tensor, F_graph: torch.Tensor):
        attn_maps = []
        for layer in self.layers:
            H, A = layer(H, F_graph)
            attn_maps.append(A)
        return H, attn_maps[-1]


def entropy_increment(E: torch.Tensor, F_graph: torch.Tensor) -> torch.Tensor:
    """Eq. 21: DeltaE^(t) = (B (.) E_p^(t)) (.) I, E_p^(t) = (E^(t) B) (.) F^(t).

    With B the row-broadcast operator (see entropy.pairwise_entropy_diff),
    E^(t)B is E_i replicated across each row i; multiplying elementwise by
    F^(t) gives each neighbour's entropy-weighted association, and the final
    (.) I keeps only the diagonal, yielding one increment value per node.
    """
    N = E.size(0)
    E_broadcast = E.unsqueeze(1).expand(N, N)  # E^(t) B
    E_p = E_broadcast * F_graph                # (.) F^(t)
    diag = (E_p * F_graph).diagonal()          # (B (.) E_p) (.) I -> diagonal entries
    return diag
