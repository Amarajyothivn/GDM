"""
End-to-end demo: Algorithm 1 (training.py) on synthetic resting-state-fMRI-
like data, followed by disorder-evolutionary feature extraction (Eq. 27-31)
and disorder risk scoring for a new subject (Eq. 32-34).

Uses synthetic node time series so it runs anywhere without ABIDE / ADHD-200
/ UCLA LA5c downloads. Swap `make_synthetic_subject` for a loader that
returns each subject's regional mean time series (nodes x timepoints) from
your own preprocessed atlas parcellation and the rest is unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from .feature_extraction import edge_association_change, node_diffusion_change, node_importance, select_feature_subset
from .risk_scoring import disorder_risk_score, representative_disorder_graph
from .training import Algorithm1Config, GDMFramework


def make_synthetic_subject(n_nodes: int, l: int, disorder: bool, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_nodes, l))
    if disorder:
        # inject a disorder-like structured perturbation on a subset of regions
        affected = rng.choice(n_nodes, size=max(1, n_nodes // 4), replace=False)
        base[affected] += rng.normal(scale=1.5, size=(len(affected), l))
    return torch.tensor(base, dtype=torch.float32)


def run_demo(n_nodes: int = 12, l: int = 40, n_outer_iters: int = 3, m_samples: int = 4, seed: int = 0):
    warnings.filterwarnings("ignore")
    torch.manual_seed(seed)

    cfg = Algorithm1Config(n_nodes=n_nodes, feat_len=l, hidden_dim=16, T1=2, T2=2, Td=5, graphormer_layers=2)
    framework = GDMFramework(cfg)

    # --- Algorithm 1: fit the generator/critic on one control/disorder pair ---
    H_C = make_synthetic_subject(n_nodes, l, disorder=False, seed=seed)
    H_P = make_synthetic_subject(n_nodes, l, disorder=True, seed=seed + 1)
    history = framework.fit(H_C, H_P, n_outer_iters=n_outer_iters)

    # --- Eq. 27-30: evolutionary node-importance over m sample pairs ---
    Ch_H_list, Ch_F_list, MI_list = [], [], []
    for p in range(m_samples):
        H_C_p = make_synthetic_subject(n_nodes, l, disorder=False, seed=100 + p)
        H_T1, F_T1, E_T1, _ = framework._entropy_evolution_phase(H_C_p)
        H_refined, _, _ = framework._diffusion_and_graphormer_phase(H_T1, F_T1, E_T1)

        H0_latent = torch.tanh(framework.encoder_to_latent(H_C_p))
        Ch_H_list.append(node_diffusion_change(H0_latent, H_refined))

        F0 = framework.dgl(H_C_p)["F"]
        FT = framework.dgl(framework.to_graph(H_refined))["F"]
        Ch_F_list.append(edge_association_change(F0, FT))

        MI_list.append(framework.dgi.mi_scores(H_C_p, F0))

    importance = node_importance(Ch_H_list, Ch_F_list, MI_list, lambda_F=1.0, lambda_I=1.0)  # Eq. 30
    print("\nNode evolutionary importance (Eq. 30):", importance.detach().numpy().round(3))

    # --- Eq. 31: greedy per-category feature-subset search using a toy accuracy proxy ---
    n_samples = 30
    categories = [i % 3 for i in range(n_nodes)]  # 3 synthetic "region categories"
    rng = np.random.default_rng(seed)
    feature_matrix = rng.normal(size=(n_samples, n_nodes)) + importance.detach().numpy() * rng.normal(
        scale=0.5, size=(n_samples, n_nodes)
    )
    labels = (feature_matrix[:, :3].sum(axis=1) > 0).astype(int)
    subset, acc = select_feature_subset(importance, categories, feature_matrix, labels)
    print(f"Selected feature subset (Eq. 31): nodes={subset}, cv_accuracy={acc:.3f}")

    # --- Eq. 32-34: disorder risk score for a new subject ---
    disorder_graphs = [framework.dgl(make_synthetic_subject(n_nodes, l, True, seed=200 + p))["F"]
                        for p in range(m_samples)]
    F_bar_P = representative_disorder_graph(disorder_graphs)  # Eq. 32

    new_subject = make_synthetic_subject(n_nodes, l, disorder=True, seed=999)
    H_T1, F_T1, E_T1, _ = framework._entropy_evolution_phase(new_subject)
    H_refined, _, _ = framework._diffusion_and_graphormer_phase(H_T1, F_T1, E_T1)
    F_generated = framework.dgl(framework.to_graph(H_refined))["F"]

    risk = disorder_risk_score(F_generated, F_bar_P, sigma_r=float(F_bar_P.std().item() + 1e-3))  # Eq. 33-34
    print(f"Disorder risk score for new subject (Eq. 34): {risk.item():.4f}")

    return {"history": history, "importance": importance, "subset": subset, "risk": risk.item()}


if __name__ == "__main__":
    run_demo()
