#!/usr/bin/env python
"""Probe WHEN the VAE beats clustering the prior directly.

Two deterministic probes (cluster mu_prior = (I+alpha L)^-1 PCA(X), then Mclust
[+spatial refine]); no VAE training, so each is one value per section:

  task 0 'alpha' : DLPFC 12 sections, alpha in {0.1,0.3,0.5,1.0,2.0}.
                   Compare flatness vs the SpaGVAE alpha-sweep (which is robust).
                   If prior-only is far more alpha-sensitive, the VAE buys
                   robustness to the diffusion hyperparameter.
  task 1 'mob'   : Visium MOB, alpha=0.5. Compare to SpaGVAE 0.453.
  task 2 'embryo': Stereo-seq mouse embryo, alpha=0.5. Compare to SpaGVAE 0.280.

Loaders reopen stdout at import, so MOB/embryo run in separate array tasks to
avoid a double-import crash.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, importlib.util
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
N_PCA, RAD, MC_SEED = 30, 150, 2020


def mclust(emb, K):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % MC_SEED)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def dlpfc_load(sample):
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


def task_alpha():
    ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0]
    rows = []
    for s in ALL_SAMPLES:
        adata = dlpfc_load(s); gt = adata.obs['ground_truth'].values; K = N_CLUSTERS[s]
        for a in ALPHAS:
            prior = compute_spatial_prior(adata, n_pca=N_PCA, alpha=a, rad_cutoff=RAD)
            lab = mclust(prior, K)
            raw = adjusted_rand_score(gt, lab)
            ref = adjusted_rand_score(gt, spatial_label_refine(lab, adata, rad_cutoff=RAD, n_iter=2))
            rows.append({'sample': s, 'alpha': a, 'ari_raw': raw, 'ari_refined': ref})
            print('  %s a=%.1f raw=%.4f refine=%.4f' % (s, a, raw, ref), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('outputs/ablation_prior_alpha.csv', index=False)
    print('\nPRIOR-ONLY alpha-sweep (mean over 12 sections):')
    g = df.groupby('alpha')[['ari_raw', 'ari_refined']].mean()
    print(g.to_string())
    print('\nSpaGVAE alpha-sweep (paper): 0.1=0.551 0.3=0.550 0.5=0.543 1.0=0.528 2.0=0.510')
    print('PROBE alpha DONE', flush=True)


def task_gen(name, spagvae_ref):
    mod = 'run_extra_datasets' if name == 'visium_mob' else 'run_extra_datasets2'
    spec = importlib.util.spec_from_file_location(
        mod, os.path.join(os.path.dirname(__file__), mod + '.py'))
    red = importlib.util.module_from_spec(spec); spec.loader.exec_module(red)
    adata, gt, K, rad = red.load_dataset(name)
    prior = compute_spatial_prior(adata, n_pca=N_PCA, alpha=0.5, rad_cutoff=rad)
    lab = mclust(prior, K)
    raw = adjusted_rand_score(gt, lab)
    ref = adjusted_rand_score(gt, spatial_label_refine(lab, adata, rad_cutoff=rad, n_iter=2))
    print('\n%s prior-only: raw=%.4f  +refine=%.4f   | SpaGVAE=%.3f  (VAE gain=%+.4f)'
          % (name, raw, ref, spagvae_ref, spagvae_ref - ref), flush=True)
    pd.DataFrame([{'dataset': name, 'ari_raw': raw, 'ari_refined': ref,
                   'spagvae': spagvae_ref}]).to_csv(
        'outputs/ablation_prior_gen_%s.csv' % name, index=False)
    print('PROBE %s DONE' % name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', type=int, required=True)
    t = p.parse_args().task
    if t == 0:
        task_alpha()
    elif t == 1:
        task_gen('visium_mob', 0.453)
    elif t == 2:
        task_gen('stereo_embryo', 0.280)
