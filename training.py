"""
Algorithm 1: GDM-driven adaptive graph representation.

Wires graph_construction (Eq.1-4,10-12), dgi (Eq.5-9), diffusion (Eq.15-17),
graphormer (Eq.18-21), critic (Eq.22-23) and losses (Eq.24-26) into exactly
the loop the paper's pseudocode describes:

  Input:  H^C (control features), H^P (disorder-associated features)
  Output: G_theta (generator: DynamicGraphLearning + NoiseEstimator + Graphormer),
          D_omega (Wasserstein critic), F^P (evolved adaptive graph)

  Repeat:
    F(0) <- DGL(H^C) via (1)-(4)
    E(0) <- entropy(H^C) via (5); Z(0) <- DGI(H(0), F(0))
    for t in 0..T1-1:
        optimize DGI objective (6)-(7)
        update entropy & node reps (8)-(9)
        update graph (10)-(14)
    X0 <- Encoder(H(T1), F(T1), E(T1))
    for k in 1..Td: forward diffuse X_k (15)
    for k in Td..1: denoise step (16)-(17); Graphormer attention (18);
                     update reps (19)-(20); entropy increment (21)
    F^P <- G_theta(X0)
    for r in 1..T2: critic score (22); update omega via (23)
    update theta via (24)-(25); monitor joint objective (26)
  Until convergence
  Return G_theta, D_omega, F^P
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .critic import WassersteinCritic, critic_loss
from .dgi import DeepGraphInfomax, propagate_entropy, update_node_features
from .diffusion import DiffusionSchedule, NoiseEstimator, forward_diffuse, reverse_step
from .entropy import node_entropies, pairwise_entropy_diff
from .graph_construction import DynamicGraphLearning
from .graphormer import Graphormer, entropy_increment
from .losses import diffusion_loss, generator_loss, joint_objective


@dataclass
class Algorithm1Config:
    n_nodes: int
    feat_len: int                 # l: length of raw regional time series
    hidden_dim: int = 32
    T1: int = 3                   # entropy-evolution iterations
    T2: int = 3                   # critic iterations per outer round
    Td: int = 10                  # diffusion steps
    graphormer_layers: int = 2
    lambda_diff: float = 1.0
    lambda_dgi: float = 1.0
    lambda_gp: float = 10.0
    eta_c: float = 0.05           # Eq. 12's eta_c (not numerically specified in the paper)
    lr_generator: float = 1e-3
    lr_critic: float = 1e-4


class GDMFramework:
    """G_theta (generator side) + D_omega (critic), i.e. Algorithm 1 end to end."""

    def __init__(self, cfg: Algorithm1Config):
        self.cfg = cfg
        N, hid = cfg.n_nodes, cfg.hidden_dim

        self.dgl = DynamicGraphLearning(N)
        self.dgi = DeepGraphInfomax(in_dim=cfg.feat_len, hidden_dim=hid)
        self.encoder_to_latent = torch.nn.Linear(cfg.feat_len, hid)  # H(T1) -> X0's node dimension
        self.noise_estimator = NoiseEstimator(node_dim=hid, n_nodes=N)
        self.graphormer = Graphormer(hid, n_layers=cfg.graphormer_layers)
        self.to_graph = torch.nn.Sequential(torch.nn.Linear(hid, hid), torch.nn.ReLU())  # decoder head for F^P
        self.critic = WassersteinCritic(N)
        self.schedule = DiffusionSchedule(n_steps=cfg.Td)

        gen_params = (
            list(self.dgl.parameters()) + list(self.dgi.parameters()) + list(self.encoder_to_latent.parameters())
            + list(self.noise_estimator.parameters()) + list(self.graphormer.parameters())
            + list(self.to_graph.parameters())
        )
        self.opt_generator = torch.optim.Adam(gen_params, lr=cfg.lr_generator)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def _entropy_evolution_phase(self, H0: torch.Tensor):
        """The T1-step loop: Eq. (6)-(14)."""
        graph = self.dgl(H0)
        F_t, C_t = graph["F"], graph["C"]
        E_t = node_entropies(H0)
        H_t = H0
        dgi_loss_total = torch.tensor(0.0)

        for _ in range(self.cfg.T1):
            dgi_loss = self.dgi.loss(H_t, F_t)  # Eq. 6-7
            dgi_loss_total = dgi_loss_total + dgi_loss

            E_next = propagate_entropy(E_t, F_t)              # Eq. 8
            delta_E, H_next = update_node_features(H_t, E_t, E_next)  # Eq. 9

            recon = self.dgl.reconstruct(H_next)               # Eq. 10-11
            C_next = self.dgl.update_correlation(C_t, delta_E, delta_E, eta_c=self.cfg.eta_c)  # Eq. 12

            H_t, E_t, F_t, C_t = H_next, E_next, recon["F"], C_next

        return H_t, F_t, E_t, dgi_loss_total / self.cfg.T1

    def _diffusion_and_graphormer_phase(self, H_T1: torch.Tensor, F_T1: torch.Tensor, E_T1: torch.Tensor):
        """X0 construction, forward/reverse diffusion, then Graphormer refinement: Eq. (15)-(21)."""
        X0 = torch.tanh(self.encoder_to_latent(H_T1))  # Encoder(H^(T1), F^(T1), E^(T1)) -> X0

        X_k = X0
        eps_history = []
        for k in range(self.cfg.Td):
            X_k, eps = forward_diffuse(X0, k, self.schedule)  # Eq. 15
            eps_history.append((k, X_k.detach(), eps))

        diff_loss_total = torch.tensor(0.0)
        X_cur = X_k
        for k in reversed(range(self.cfg.Td)):
            eps_pred = self.noise_estimator(X_cur, F_T1, E_T1, k)  # eps_theta(X_k, F, E, k)
            _, X_k_true, eps_true = eps_history[k]
            diff_loss_total = diff_loss_total + diffusion_loss(eps_true, eps_pred)  # Eq. 25 accum
            X_cur = reverse_step(X_cur, eps_pred, k, self.schedule)  # Eq. 16-17

        H_refined, attn = self.graphormer(X_cur, F_T1)  # Eq. 18-20
        delta_E_final = entropy_increment(E_T1, F_T1)   # Eq. 21

        return H_refined, diff_loss_total / self.cfg.Td, delta_E_final

    def run_round(self, H_C: torch.Tensor, H_P: torch.Tensor) -> Dict:
        """One outer iteration of Algorithm 1's Repeat...Until loop."""
        # --- entropy-evolution phase on the control sample (reference/control input) ---
        H_T1, F_T1, E_T1, dgi_loss = self._entropy_evolution_phase(H_C)

        # --- diffusion + Graphormer refinement ---
        H_refined, diff_loss, delta_E_final = self._diffusion_and_graphormer_phase(H_T1, F_T1, E_T1)

        # --- generate the evolved graph F^P = G_theta(X0) ---
        H_decoded = self.to_graph(H_refined)
        F_P = self.dgl(H_decoded)["F"]  # reuse Eq. 1-4 on the decoded representation

        # --- real disorder-affected graph, from the actual patient sample H^P ---
        F_real = self.dgl(H_P)["F"]

        # --- critic phase: T2 iterations of Eq. 22-23 ---
        last_critic_loss = torch.tensor(0.0)
        for _ in range(self.cfg.T2):
            self.opt_critic.zero_grad()
            c_loss = critic_loss(self.critic, F_real.detach(), F_P.detach(), lambda_gp=self.cfg.lambda_gp)
            c_loss.backward()
            self.opt_critic.step()
            last_critic_loss = c_loss

        # --- generator update: Eq. 24-25 ---
        self.opt_generator.zero_grad()
        critic_score_fake = self.critic(F_P)
        g_loss = generator_loss(critic_score_fake, diff_loss + dgi_loss, dgi_loss,
                                 lambda_diff=self.cfg.lambda_diff, lambda_dgi=self.cfg.lambda_dgi)
        g_loss.backward()
        self.opt_generator.step()

        with torch.no_grad():
            critic_score_real = self.critic(F_real)
            joint = joint_objective(critic_score_real, self.critic(F_P), diff_loss, dgi_loss,
                                     lambda_diff=self.cfg.lambda_diff, lambda_dgi=self.cfg.lambda_dgi)  # Eq. 26

        return {
            "F_P": F_P.detach(),
            "critic_loss": last_critic_loss.item(),
            "generator_loss": g_loss.item(),
            "joint_objective": joint.item(),
            "dgi_loss": dgi_loss.item(),
            "diffusion_loss": diff_loss.item(),
        }

    def fit(self, H_C: torch.Tensor, H_P: torch.Tensor, n_outer_iters: int = 5, verbose: bool = True) -> List[Dict]:
        history = []
        for i in range(n_outer_iters):
            metrics = self.run_round(H_C, H_P)
            history.append(metrics)
            if verbose:
                print(f"[iter {i}] critic={metrics['critic_loss']:.4f} "
                      f"gen={metrics['generator_loss']:.4f} joint={metrics['joint_objective']:.4f}")
        return history
