"""
Sections "Disorder-Related Evolutionary Pattern Extraction" and
"Feature Extraction" (Eq. 27-31).

Eq. 27  Ch_H_i^p = ||H_{i,T}^p - H_{i,0}^p||_2 / (||H_{i,0}^p||_2 + eps)
Eq. 28  Ch_F_ij^p = |F_{ij,T}^p - F_{ij,0}^p| / (|F_{ij,0}^p| + eps)
Eq. 29  MI_i^p = log D(Z_i^p, s^p) - log(1 - D(Z~_i^p, s^p))      (see dgi.mi_scores)
Eq. 30  Ch_i_bar = (1/m) sum_p ( Ch_H_i^p + lambda_F * sum_j Ch_F_ij^p + lambda_I * MI_i^p )
Eq. 31  argmax over per-category top-k subsets of prediction accuracy

lambda_F, lambda_I (Eq. 30's weighting terms) are not given numeric values in
the paper; exposed as arguments below.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def node_diffusion_change(H0: torch.Tensor, HT: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 27, per node. H0, HT: [N, d]."""
    return (HT - H0).norm(dim=1) / (H0.norm(dim=1) + eps)


def edge_association_change(F0: torch.Tensor, FT: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 28, per edge. F0, FT: [N, N]."""
    return (FT - F0).abs() / (F0.abs() + eps)


def node_importance(
    Ch_H_list: List[torch.Tensor],       # per-sample Eq. 27 outputs, each [N]
    Ch_F_list: List[torch.Tensor],       # per-sample Eq. 28 outputs, each [N, N]
    MI_list: List[torch.Tensor],         # per-sample Eq. 29 outputs, each [N]
    lambda_F: float = 1.0,
    lambda_I: float = 1.0,
) -> torch.Tensor:
    """Eq. 30: averaged evolutionary importance per node, across m samples."""
    m = len(Ch_H_list)
    total = torch.zeros_like(Ch_H_list[0])
    for Ch_H, Ch_F, MI in zip(Ch_H_list, Ch_F_list, MI_list):
        total = total + Ch_H + lambda_F * Ch_F.sum(dim=1) + lambda_I * MI
    return total / m


def select_feature_subset(
    importance: torch.Tensor,
    node_categories: Sequence[int],           # category id per node, len N
    features: np.ndarray,                     # [n_samples, N] feature matrix used for accuracy eval
    labels: np.ndarray,                       # [n_samples]
    k_per_category: Dict[int, int] | None = None,
    classifier=None,
) -> Tuple[List[int], float]:
    """
    Eq. 31: max over per-category top-k node subsets of prediction accuracy.
    Exhaustively searches k in {1..k_max} per category (as the paper's
    "incremental search" describes) and returns the subset maximizing
    cross-validated accuracy of a simple classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    classifier = classifier or LogisticRegression(max_iter=200)
    categories = sorted(set(node_categories))
    nodes_by_cat = {c: [i for i, cc in enumerate(node_categories) if cc == c] for c in categories}

    best_acc, best_subset = -1.0, []
    # incremental search: grow each category's k from 1 up to its size, keep
    # the best-scoring node ranking (by `importance`) at each k
    ranked_by_cat = {
        c: sorted(nodes_by_cat[c], key=lambda i: importance[i].item(), reverse=True) for c in categories
    }
    max_k = k_per_category or {c: len(nodes_by_cat[c]) for c in categories}

    current_subset: List[int] = []
    for c in categories:
        best_c_acc, best_c_subset = -1.0, []
        trial_subset = list(current_subset)
        for k in range(1, max_k[c] + 1):
            trial_subset = current_subset + ranked_by_cat[c][:k]
            X = features[:, trial_subset]
            try:
                acc = cross_val_score(classifier, X, labels, cv=min(5, len(labels))).mean()
            except ValueError:
                acc = 0.0
            if acc > best_c_acc:
                best_c_acc, best_c_subset = acc, list(trial_subset)
        current_subset = best_c_subset
        if best_c_acc > best_acc:
            best_acc = best_c_acc
        best_subset = current_subset

    return best_subset, best_acc
