# SpaGVAE: A Variational Graph Autoencoder with a Structured Spatial Prior for Robust Spatial Domain Identification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Reference implementation for the paper
*"SpaGVAE: A Variational Graph Autoencoder with Structured Spatial Prior for
Robust Spatial Domain Identification."*

## Overview

Graph neural network (GNN) autoencoders such as STAGATE and GraphST are the
state of the art for **spatial domain identification**, but their results are
surprisingly sensitive to the random seed: changing only the seed can move the
Adjusted Rand Index (ARI) by as much as the gap between published methods. We
first quantify this with the largest seed-sensitivity study to date (six
methods × 1,000 seeds × 12 DLPFC sections), then introduce **SpaGVAE** to fix it.

SpaGVAE is a variational graph autoencoder whose central innovation is a
**structured spatial prior**. Instead of the standard
\(\mathcal{N}(\mathbf{0},\mathbf{I})\) prior, the latent space is anchored to a
*deterministic* reference obtained by graph diffusion of PCA embeddings and
shared across all training seeds:

```
mu_prior = (I + alpha * L_norm)^{-1} PCA(X)          # deterministic, seed-independent
ELBO     = || X - X_hat ||^2  +  beta * KL( q(z|X,G) || N(mu_prior, I) )
```

Because every seed is pulled toward the same spatial anchor, the learned
embeddings - and hence the downstream Mclust domains - are far more
reproducible. `beta`-annealing and Stochastic Weight Averaging (SWA) are
included as lightweight, optional components; a leave-one-out ablation shows the
structured prior is the dominant driver of both accuracy and stability (removing
`beta`-annealing or SWA leaves both essentially unchanged).

## Key results (DLPFC, 12 sections, 1,000 seeds each)

| Method     | Mean ARI | Worst-case (P5) ARI | CV (%) |
|------------|:--------:|:-------------------:|:------:|
| SpaGCN     | 0.338    | 0.205 | 27.0 |
| SpaceFlow  | 0.435    | 0.265 | 26.2 |
| stCluster  | 0.435    | 0.264 | 23.0 |
| STAGATE    | 0.488    | 0.246 | 25.2 |
| GraphST    | 0.506    | 0.398 | 14.4 |
| **SpaGVAE**| **0.542**| **0.417** | 16.6 |

SpaGVAE has the highest mean and worst-case ARI and beats the strongest
baseline (GraphST) on **all 12 sections** (Wilcoxon signed-rank *p* = 0.0002).
Removing the structured prior collapses the mean ARI to 0.307 and more than
doubles inter-seed variability.

Two further findings characterise *when* the variational model matters:

- **The structured prior is the workhorse.** Clustering the deterministic prior
  directly (no VAE training) already reaches 0.534 mean ARI on DLPFC - within
  0.008 of the full model and above every baseline - at *zero* seed variance.
- **The VAE is essential under sparse gene coverage.** As the number of highly
  variable genes drops from 3,000 (Visium) to 200 (targeted-panel / imaging
  scale), the static prior collapses while SpaGVAE retains accuracy: the gap
  grows from +0.015 to **+0.277 ARI**.

## Installation

```bash
git clone https://github.com/RRRussell/StaR.git
cd StaR
conda create -n spagvae python=3.8 -y && conda activate spagvae
pip install -r requirements.txt
pip install -e .

# Mclust clustering goes through rpy2 -> R:
#   install.packages("mclust")   # inside R
```

The model also needs `STAGATE_pyG` (used for the spatial-graph construction and
PyG data conversion) and, for the baselines, `GraphST` and `SpaceFlow`.

## Quick start

```python
import scanpy as sc, numpy as np
from STAGATE_pyG.utils import Transfer_pytorch_Data
import STAGATE_pyG as STAGATE
from star.spagvae import train_spagvae, spatial_label_refine

# adata: preprocessed AnnData (3,000 HVGs, normalised+log1p+scaled, .obsm['spatial'])
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

# Train SpaGVAE (structured prior + beta-annealing + SWA)
model, emb = train_spagvae(
    data, adata, n_epochs=800, lr=1e-3,
    beta_max=0.05, beta_warmup=200, beta_ramp=400,
    diffusion_alpha=0.5, latent_dim=30, n_pca=30, swa_start=600)

# Cluster the posterior mean with Mclust, then spatial-refine the labels
#   (see scripts/run_spagvae_dlpfc.py for the full Mclust + refinement helper)
labels = spatial_label_refine(mclust_labels(emb, K=7), adata, rad_cutoff=150, n_iter=2)
```

Set `structured_prior=False` in `train_spagvae` to fall back to the standard
\(\mathcal{N}(\mathbf{0},\mathbf{I})\) prior (the ablation baseline).

## Reproducing the paper

All experiments are launched on SLURM; the wrapper `.sbatch` files are in
[`slurm/`](slurm/). The canonical configuration is
`alpha=0.5, beta_max=0.05, SWA from epoch 600, Mclust seed 2020`.

| What | Script | SLURM |
|------|--------|-------|
| DLPFC main benchmark, SpaGVAE @ 1,000 seeds | `scripts/experiments/star_v2/run_spagvae_1000.py` | `slurm/sbatch_spagvae_1000.sh` |
| DLPFC main benchmark, SpaGCN @ 1,000 seeds | `scripts/experiments/star_v2/run_spagcn_1000.py` | `slurm/sbatch_spagcn_1000.sh` |
| Baseline seed-sensitivity @ 1,000 seeds | `scripts/experiments/seed_sensitivity/run_{stagate,graphst,spaceflow,stcluster}.py` | - |
| Component ablation (leave-one-out) + α sweep | `scripts/experiments/star_v2/run_spagvae_ablation.py` | `slurm/sbatch_spagvae_ablation.sh` |
| β-sweep at the canonical α=0.5 | `scripts/experiments/star_v2/run_beta_alpha05.py` | `slurm/sbatch_beta05.sh` |
| **Prior-only** ablation (cluster the prior, no VAE) | `scripts/experiments/star_v2/ablation_prior_only.py`, `ablation_prior_probe.py` | `slurm/sbatch_ablation_prior.sh` |
| **Gene-coverage** sweep (prior-only vs SpaGVAE, HVG 200–3000) | `scripts/experiments/star_v2/ablation_hvg.py` | `slurm/sbatch_ablation_hvg.sh` |
| **K-robustness** (K_true ± 1) | `scripts/experiments/star_v2/run_k_robustness.py` | `slurm/sbatch_k_robustness.sh` |
| Generalisation - Stereo-seq hemibrain, SpaGVAE/STAGATE | `scripts/experiments/star_v2/run_mosta_hemibrain.py` | - |
| Generalisation - hemibrain GraphST/SpaceFlow | `scripts/experiments/star_v2/run_hemibrain_baselines.py` | - |
| Generalisation - hemibrain SpaGCN/stCluster | `scripts/experiments/star_v2/run_gen_extra_baselines.py` | `slurm/sbatch_gen_extra.sh` |
| Generalisation - consistent 20-seed run (Table 7 + Fig 5 source, saves labels) | `scripts/experiments/star_v2/run_hemibrain_fig20.py` | `slurm/sbatch_fig20.sh` |
| Aggregate 1,000-seed tables (mean/P5/CV/Wilcoxon) | `scripts/analyze_1000seed.py` | - |
| Paper figures | `scripts/figures/generate_paper_figures.py`, `fig_ablation.py`, `fig_hemibrain.py` | - |

Results are written to `outputs/` (per-seed ARI CSVs, used to compute every
statistic in the paper). `scripts/analyze_1000seed.py` recomputes all the main
tables (pooled mean / median / P5 / P95 / Std / CV and the Wilcoxon tests).

## Repository structure

```
StaR/
├── star/
│   └── spagvae.py        # SpaGVAE: model, structured prior, training, refinement
├── scripts/
│   ├── run_spagvae_dlpfc.py              # standalone single-section reproduction
│   ├── analyze_1000seed.py              # aggregate per-seed CSVs -> paper tables
│   ├── experiments/                     # benchmark / ablation / generalisation runners
│   └── figures/generate_paper_figures.py# paper figures
├── slurm/                # SLURM wrappers for every experiment
├── requirements.txt
├── setup.py
└── LICENSE
```

## Data

- **DLPFC** (primary benchmark): 12 × 10x Visium sections (151507–151676),
  obtained from [spatialLIBD](http://spatial.libd.org/spatialLIBD/).
- **Generalisation**: MOSTA Stereo-seq mouse hemibrain (19 anatomical regions,
  10,000 spots). SpaGVAE attains the highest mean ARI of all six methods
  compared, and even its worst seed exceeds the mean of every other method.

Place datasets under the paths referenced at the top of each runner script
(e.g. `DATA_ROOT`), or edit those constants to point at your local copies.

## Citation

```bibtex
@article{spagvae,
  title   = {SpaGVAE: A Variational Graph Autoencoder with Structured Spatial
             Prior for Robust Spatial Domain Identification in Spatial
             Transcriptomics},
  author  = {Wei, Xindi},
  year    = {2026}
}
```

## License

MIT License - see [LICENSE](LICENSE).
