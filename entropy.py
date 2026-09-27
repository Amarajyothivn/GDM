"""
Eq. 5   E_i = (1/(l-m+1)) sum_r E_i^m(r) - (1/(l-m)) sum_r E_i^{m+1}(r)
Eq. 13  DeltaE_v^(t)   = |E^(t)B - (E^(t)B)^T|
Eq. 14  DeltaE_v^(t+1) = |E^(t+1)B - (E^(t+1)B)^T|

Eq. 5 is the standard Approximate-Entropy (ApEn) construction: E_i^m(r) is the
log-average self-match count of embedding-dimension-m subsequences of node
i's regional time series (length l), and E_i = Phi^m - Phi^{m+1} is ApEn's
usual definition. This module computes that per node from its raw rs-fMRI
time series, once, to initialize E^(0) (used by Algorithm 1's Initialize step).

Eq. 13/14: B is used as a broadcasting operator so that E^(t)B is an NxN
matrix whose row i is filled with E_i (paper's own description right after
Eq. 14: "these terms represent pairwise variations in entropy information").
That is exactly the outer-broadcast |E_i - E_j|, which is what
`pairwise_entropy_diff` computes directly.
"""
from __future__ import annotations

import torch


def approximate_entropy(series: torch.Tensor, m: int = 2, r: float | None = None) -> float:
    """ApEn(m, r) of a single 1D time series (Eq. 5's E_i for one node)."""
    x = series.detach().cpu().numpy()
    l = len(x)
    if r is None:
        r = 0.2 * x.std() + 1e-8

    def _phi(m_: int) -> float:
        if l - m_ + 1 <= 0:
            return 0.0
        templates = [x[i:i + m_] for i in range(l - m_ + 1)]
        counts = []
        for t_i in templates:
            dist = [max(abs(a - b) for a, b in zip(t_i, t_j)) for t_j in templates]
            counts.append(sum(1 for d in dist if d <= r) / len(templates))
        counts = [c for c in counts if c > 0]
        if not counts:
            return 0.0
        import math
        return sum(math.log(c) for c in counts) / len(templates)

    return _phi(m) - _phi(m + 1)


def node_entropies(time_series: torch.Tensor, m: int = 2) -> torch.Tensor:
    """time_series: [N, l] regional time series -> E^(0): [N] (Eq. 5, per node)."""
    return torch.tensor([approximate_entropy(time_series[i], m=m) for i in range(time_series.size(0))],
                         dtype=torch.float32)


def pairwise_entropy_diff(E: torch.Tensor) -> torch.Tensor:
    """Eq. 13/14: |E_i - E_j| as an [N, N] matrix, via the E^(t)B broadcast."""
    return (E.unsqueeze(1) - E.unsqueeze(0)).abs()
