#!/usr/bin/env python
"""Ablation (reviewer attack #2): cluster the deterministic structured prior
DIRECTLY, with no VAE training.

    mu_prior = (I + alpha * L_norm)^{-1} PCA(X)        [alpha=0.5, n_pca=30]

If Mclust on mu_prior already matches SpaGVAE's ARI, the variational autoencoder
adds nothing on top of the smoothed-PCA target. This quantifies the VAE's
incremental value. Reports both raw Mclust and +spatial-refine (to match
SpaGVAE's full pipeline). Deterministic -> one value per section.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np, scanpy as sc, pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import STAGATE_pyG as STAGATE
from star.spagvae import compute_spatial_prior, spatial_label_refine

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

DATA_ROOT = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
ALL_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
N_CLUSTERS = {'151507': 7, '151508': 7, '151509': 7, '151510': 7, '151669': 5,
              '151670': 5, '151671': 5, '151672': 5, '151673': 7, '151674': 7,
              '151675': 7, '151676': 7}
ALPHA, N_PCA, RAD, MC_SEED = 0.5, 30, 150, 2020
OUT = 'outputs/ablation_prior_only.csv'


def load_data(sample):
    """Identical preprocessing to run_spagvae_1000.load_data."""
    fold = os.path.join(DATA_ROOT, sample)
    adata = sc.read_visium(fold, count_file='%s_filtered_feature_bc_matrix.h5' % sample)
    adata.var_names_make_unique()
    gt = pd.read_csv(os.path.join(DATA_ROOT, '%s_truth.txt' % sample),
                     sep='\t', header=None, index_col=0)
    gt.columns = ['ground_truth']
    adata.obs['ground_truth'] = gt.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower()
                  .isin(['nan', 'na', 'none', ''])].copy()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=RAD)
    return adata


def mclust(emb, K):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % MC_SEED)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def main():
    rows = []
    for s in ALL_SAMPLES:
        adata = load_data(s)
        gt = adata.obs['ground_truth'].values
        K = N_CLUSTERS[s]
        prior = compute_spatial_prior(adata, n_pca=N_PCA, alpha=ALPHA, rad_cutoff=RAD)
        lab = mclust(prior, K)
        ari_raw = adjusted_rand_score(gt, lab)
        lab_ref = spatial_label_refine(lab, adata, rad_cutoff=RAD, n_iter=2)
        ari_ref = adjusted_rand_score(gt, lab_ref)
        rows.append({'sample': s, 'ari_raw': ari_raw, 'ari_refined': ari_ref})
        print('  %s  prior-only raw=%.4f  +refine=%.4f' % (s, ari_raw, ari_ref), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print('\nPRIOR-ONLY MEAN: raw=%.4f  +refine=%.4f' % (df.ari_raw.mean(), df.ari_refined.mean()))
    print('(compare: SpaGVAE full = 0.542 ; ablation no-prior = ~0.307)')
    print('PRIOR-ONLY DONE', flush=True)


if __name__ == '__main__':
    main()
