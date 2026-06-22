#!/usr/bin/env python
"""Robustness to misspecified number of clusters K (reviewer ask: K is assumed
known). For each DLPFC section we cluster SpaGVAE and STAGATE embeddings with
Mclust at G = K_true-1, K_true, K_true+1 (5 seeds each) and report ARI against
the expert layers. Shows whether SpaGVAE's advantage is preserved when K is
mis-set. One SLURM array task per section (0-11); merge with --summary.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, glob, random
import numpy as np, scanpy as sc, pandas as pd, torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import STAGATE_pyG as STAGATE
from STAGATE_pyG.utils import Transfer_pytorch_Data
from star.spagvae import train_spagvae, spatial_label_refine

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

DATA_ROOT = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
ALL_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
K_TRUE = {'151507': 7, '151508': 7, '151509': 7, '151510': 7, '151669': 5,
          '151670': 5, '151671': 5, '151672': 5, '151673': 7, '151674': 7,
          '151675': 7, '151676': 7}
SEEDS = [0, 1, 2, 3, 4]
RAD, MC_SEED = 150, 2020
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'outputs/k_robustness'
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_data(sample):
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


def spagvae_emb(data, adata, seed):
    set_seed(seed)
    _, emb = train_spagvae(data, adata, n_epochs=800, lr=1e-3, beta_max=0.05,
                           beta_warmup=200, beta_ramp=400, diffusion_alpha=0.5,
                           latent_dim=30, n_pca=30, swa_start=600, device=DEVICE)
    return emb


def stagate_emb(data, seed):
    import torch.nn.functional as F
    set_seed(seed)
    n_feat = data.x.shape[1]
    model = STAGATE.STAGATE(hidden_dims=[n_feat, 512, 30]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    d = data.to(DEVICE); model.train()
    for _ in range(800):
        opt.zero_grad(); z, xr = model(d.x, d.edge_index); F.mse_loss(xr, d.x).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z, _ = model(d.x, d.edge_index)
    return z.cpu().numpy()


def run_task(task):
    sample = ALL_SAMPLES[task]
    kt = K_TRUE[sample]
    adata = load_data(sample)
    gt = adata.obs['ground_truth'].values
    data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])
    csv_out = os.path.join(OUT_DIR, 'results_%s.csv' % sample)
    records = pd.read_csv(csv_out).to_dict('records') if os.path.exists(csv_out) else []
    done = {(r['method'], int(r['K']), int(r['seed'])) for r in records}

    for seed in SEEDS:
        sp = spagvae_emb(data, adata, seed)
        st = stagate_emb(data, seed)
        for K in [kt - 1, kt, kt + 1]:
            for name, emb, refine in [('SpaGVAE', sp, True), ('STAGATE', st, False)]:
                if (name, K, seed) in done:
                    continue
                lab = mclust(emb, K)
                if refine:
                    lab = spatial_label_refine(lab, adata, rad_cutoff=RAD, n_iter=2)
                ari = adjusted_rand_score(gt, lab)
                records.append({'sample': sample, 'method': name, 'K': K,
                                'K_true': kt, 'dK': K - kt, 'seed': seed, 'ari': ari})
                print('  %s %s K=%d (dK%+d) seed=%d ARI=%.4f' % (sample, name, K, K - kt, seed, ari), flush=True)
        pd.DataFrame(records).to_csv(csv_out, index=False)
    print('%s K-ROBUST DONE' % sample, flush=True)


def summary():
    fs = glob.glob(os.path.join(OUT_DIR, 'results_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, 'all_results.csv'), index=False)
    print('Mean ARI over 12 sections x 5 seeds, by K offset:')
    g = df.groupby(['dK', 'method'])['ari'].mean().unstack()
    print(g.to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.summary:
        summary()
    elif a.task is not None:
        run_task(a.task)
