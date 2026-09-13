# SpaGVAE

Variational graph autoencoder with a structured spatial prior for spatial domain identification.

## Install

```bash
pip install -r requirements.txt
```

Clustering uses the R package `mclust` through `rpy2`:

```r
install.packages("mclust")
```

## Usage

```python
import scanpy as sc
from spagvae import run_spagvae

adata = sc.read_h5ad("151673.h5ad")   # raw counts, obsm["spatial"], obs["ground_truth"]
labels, embedding, ari = run_spagvae(adata, n_clusters=7, rad=150, seed=0, gt_key="ground_truth")
```

Default hyperparameters are those used in the paper: 3,000 HVGs, smoothing strength
`diffusion_alpha=0.5`, `beta_max=0.05` with a linear KL warm-up, 800 epochs, 30 latent
dimensions. Pass any `train_spagvae` argument to `run_spagvae` to change them.

To cluster the structured prior directly (SpaGVAE-P), use `compute_spatial_prior`.
