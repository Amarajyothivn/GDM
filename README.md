# GDM: Graph Diffusion Model for Adaptive Brain-Network Representation

Reference implementation of *"Graph Diffusion Model-Based Framework for
Adaptive Functional Brain-Network Representation and Neuropsychiatric
Disorder Prediction"*, built directly off the paper's equations. Every
function/class docstring cites the equation(s) it implements.

| File | Section | What it implements |
|---|---|---|
| `entropy.py` | Eq. 5 | Approximate-entropy per brain region; pairwise entropy-difference broadcast (Eq. 13/14) |
| `graph_construction.py` | Eq. 1-4, 10-12 | Direct (Euclidean) + indirect (Pearson-based) association, adaptive graph F^(t), reconstruction, dynamic correlation update |
| `dgi.py` | Eq. 6-9 | GCN encoder + Deep Graph Infomax loss, entropy propagation, node-feature update |
| `diffusion.py` | Eq. 15-17 | Forward/reverse graph diffusion (DDPM-style), graph-conditioned noise estimator |
| `graphormer.py` | Eq. 18-21 | Structural-attention Graphormer layer (spatial + association bias), entropy increment |
| `critic.py` | Eq. 22-23 | Wasserstein critic with gradient penalty |
| `losses.py` | Eq. 24-26 | Generator loss, diffusion loss, joint minimax objective |
| `training.py` | Algorithm 1 | Wires all of the above into the paper's exact Repeat/Until loop |
| `feature_extraction.py` | Eq. 27-31 | Node diffusion/association change, node-importance scoring, greedy per-category feature-subset search |
| `risk_scoring.py` | Eq. 32-34 | Representative disorder graph, topological difference, Gaussian-membership risk score |
| `demo.py` | — | End-to-end run over synthetic rs-fMRI-like data |

## Run it

```bash
pip install torch scikit-learn numpy
python -m gdm.demo
```

Runs Algorithm 1's training loop, then node-importance scoring, feature-subset
search, and a disorder risk score for a new subject — all on synthetic
per-region time series, so it executes in seconds with no ABIDE/ADHD-200/UCLA
LA5c download. Swap `demo.make_synthetic_subject` for a loader returning each
subject's real regional mean time series (from your atlas parcellation of the
preprocessed rs-fMRI) and the rest of the pipeline is unchanged.

## What's a placeholder vs. what's from the paper

Several symbols are named in the text but never given a numeric value or a
concrete construction rule in the archived methods section. Rather than
inventing values and presenting them as if they were reported, each is
exposed as a constructor argument with the assumption spelled out in the
nearest docstring:

- **lambda_ij^(t)** (Eq. 4's direct/indirect balance) — modelled as a
  learnable per-edge parameter (sigmoid-squashed to [0,1]).
- **alpha_ij^(t)** (Eq. 8's propagation attention) — taken as the
  row-softmax of the adaptive association F^(t) itself.
- **eta_c** (Eq. 12's correlation-update step size) — a constructor argument
  (`Algorithm1Config.eta_c`), default `0.05`.
- **B** (Eq. 13/14/21's broadcast operator) — implemented as the literal
  row-broadcast that produces `|E_i - E_j|`, matching the paper's own gloss
  of what those terms represent.
- **gamma** (Eq. 18's association-bias gate) — a learnable scalar.
- **lambda_diff, lambda_DGI, lambda_gp, lambda_F, lambda_I** — all exposed
  as named arguments (`Algorithm1Config`, `node_importance(...)`), not
  hard-coded.

`Algorithm1Config` also carries the reported implementation settings where
the paper does give them: 3 graph-diffusion layers, 2 Graphormer encoder
layers, learning rate 1e-5, batch size 32 (see the Methods' "Implementation
environment" paragraph) — trimmed down in `demo.py` for a fast synthetic run;
scale `Algorithm1Config` back up once you plug in real preprocessed data.

Note: the paper's own "Code availability" statement points to
`https://github.com/Amarajyothivn/GDM` for the archived implementation —
this module is an independent re-derivation from the equations, not a copy
of that repository.
