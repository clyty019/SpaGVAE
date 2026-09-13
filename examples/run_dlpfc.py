"""Run SpaGVAE on one DLPFC section.

Expects an AnnData with raw counts, spatial coordinates in obsm['spatial']
and the layer annotation in obs['ground_truth'].
"""
import sys
import scanpy as sc
from spagvae import run_spagvae

adata = sc.read_h5ad(sys.argv[1])
labels, emb, ari = run_spagvae(adata, n_clusters=7, rad=150, seed=0, gt_key="ground_truth")
print("ARI = %.3f" % ari)
