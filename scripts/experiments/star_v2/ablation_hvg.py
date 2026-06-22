#!/usr/bin/env python
"""Probe (decide VAE vs prior): does the VAE beat clustering the prior directly
when the input is GENE-POOR (targeted panels / MERFISH-like)?

For n_top_genes in {200, 500, 1000, 3000} on each DLPFC section, compare
  - prior-only : Mclust(+refine) on mu_prior = (I+0.5 L)^-1 PCA(X)   [deterministic]
  - SpaGVAE    : mean over 3 seeds, same Mclust(+refine)
If SpaGVAE pulls ahead as HVG shrinks, the VAE is justified for low-gene
platforms; if prior-only stays >= SpaGVAE even at HVG=200, the VAE is redundant.

One SLURM array task per section (0-11). Merge with --summary.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, glob
import numpy as np, scanpy as sc, pandas as pd, torch, random
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import STAGATE_pyG as STAGATE
from STAGATE_pyG.utils import Transfer_pytorch_Data
from star.spagvae import train_spagvae, spatial_label_refine, compute_spatial_prior

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
HVGS = [200, 500, 1000, 3000]
SEEDS = list(range(10))
ALPHA, N_PCA, RAD, MC_SEED = 0.5, 30, 150, 2020
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'outputs/ablation_hvg'
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_data(sample, n_hvg):
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
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_hvg)
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


def run_task(task):
    sample = ALL_SAMPLES[task]
    K = N_CLUSTERS[sample]
    rows = []
    for hvg in HVGS:
        adata = load_data(sample, hvg)
        gt = adata.obs['ground_truth'].values
        data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

        prior = compute_spatial_prior(adata, n_pca=N_PCA, alpha=ALPHA, rad_cutoff=RAD)
        lab = mclust(prior, K)
        ari_prior = adjusted_rand_score(gt, spatial_label_refine(lab, adata, rad_cutoff=RAD, n_iter=2))

        sg = []
        for seed in SEEDS:
            set_seed(seed)
            _, emb = train_spagvae(data, adata, n_epochs=800, lr=1e-3,
                                   beta_max=0.05, beta_warmup=200, beta_ramp=400,
                                   diffusion_alpha=ALPHA, latent_dim=30, n_pca=N_PCA,
                                   swa_start=600, device=DEVICE)
            l = mclust(emb, K)
            sg.append(adjusted_rand_score(gt, spatial_label_refine(l, adata, rad_cutoff=RAD, n_iter=2)))
        rows.append({'sample': sample, 'hvg': hvg, 'ari_prior': ari_prior,
                     'ari_spagvae': float(np.mean(sg)),
                     'spagvae_seeds': ';'.join('%.4f' % x for x in sg)})
        print('  %s hvg=%4d: prior=%.4f  SpaGVAE=%.4f  (VAE gain %+.4f)'
              % (sample, hvg, ari_prior, np.mean(sg), np.mean(sg) - ari_prior), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'results_%s.csv' % sample), index=False)
    print('%s HVG-PROBE DONE' % sample, flush=True)


def summary():
    fs = glob.glob(os.path.join(OUT_DIR, 'results_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = df.groupby('hvg')[['ari_prior', 'ari_spagvae']].mean()
    g['VAE_gain'] = g['ari_spagvae'] - g['ari_prior']
    print('Mean over %d sections, by HVG count:' % df['sample'].nunique())
    print(g.to_string())
    df.to_csv(os.path.join(OUT_DIR, 'all_results.csv'), index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.summary:
        summary()
    elif a.task is not None:
        run_task(a.task)
