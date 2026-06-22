#!/usr/bin/env python
"""
SpaGVAE component ablation + diffusion-strength (alpha) sensitivity on DLPFC.

Runs the configurations that are NOT already covered by outputs/spagvae_improve
(which holds the canonical full config a0.5_b0.05_swa @ 50 seeds, plus a0.3 and
a1.0). All configs here use the SAME pipeline / preprocessing / Mclust seed as
run_spagvae_improve.py so that results are directly comparable at 50 seeds.

Configs (all alpha/beta/refine kept at canonical values unless ablated):
  ABLATION (leave-one-out from the full config a0.5_b0.05_anneal_swa):
    no_prior   : standard N(0,I) prior (mu_prior=0)            -> isolates structured prior
    no_swa     : SWA disabled                                   -> isolates SWA
    no_anneal  : constant beta=beta_max from epoch 0            -> isolates beta-annealing
    (no_refine is obtained for free: every run records ari_raw)
  ALPHA SENSITIVITY (canonical config, vary diffusion strength):
    a0.1       : alpha=0.1
    a2.0       : alpha=2.0
  (alpha=0.3, 0.5, 1.0 already exist at 50 seeds in outputs/spagvae_improve)

One SLURM array task per DLPFC section (array 0-11).
Records ari_raw AND ari_refined per (sample, config, seed); resumes if killed.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, argparse
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np, scanpy as sc, pandas as pd, torch, random
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import STAGATE_pyG as STAGATE
from STAGATE_pyG.utils import Transfer_pytorch_Data
from star.spagvae import train_spagvae, spatial_label_refine

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

ALL_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
DATA_ROOT = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 50
MCLUST_SEED = 2020
N_CLUSTERS = {'151507': 7, '151508': 7, '151509': 7, '151510': 7, '151669': 5,
              '151670': 5, '151671': 5, '151672': 5, '151673': 7, '151674': 7,
              '151675': 7, '151676': 7}
OUT_DIR = 'outputs/spagvae_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

# (name, alpha, beta_max, anneal, swa_start, structured_prior)
CONFIGS = [
    ('no_prior',  0.5, 0.05, True,  600,  False),
    ('no_swa',    0.5, 0.05, True,  None, True),
    ('no_anneal', 0.5, 0.05, False, 600,  True),
    ('a0.1',      0.1, 0.05, True,  600,  True),
    ('a2.0',      2.0, 0.05, True,  600,  True),
]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_data(sample):
    file_fold = os.path.join(DATA_ROOT, sample)
    adata = sc.read_visium(file_fold, count_file='%s_filtered_feature_bc_matrix.h5' % sample)
    adata.var_names_make_unique()
    gt = pd.read_csv(os.path.join(DATA_ROOT, '%s_truth.txt' % sample),
                     sep='\t', header=None, index_col=0)
    gt.columns = ['ground_truth']
    adata.obs['ground_truth'] = gt.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
    invalid = ['nan', 'na', 'none', '']
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower().isin(invalid)].copy()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    return adata


def mclust_labels(emb, K, seed):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % seed)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True, help='sample index 0-11')
    args = parser.parse_args()

    sample = ALL_SAMPLES[args.task]
    K = N_CLUSTERS[sample]
    csv_out = os.path.join(OUT_DIR, 'results_%s.csv' % sample)

    # resume
    if os.path.exists(csv_out):
        done_df = pd.read_csv(csv_out)
        done = set(zip(done_df['config'], done_df['seed']))
        records = done_df.to_dict('records')
    else:
        done, records = set(), []

    print('SpaGVAE Ablation | sample=%s K=%d device=%s | configs=%d seeds=%d'
          % (sample, K, DEVICE, len(CONFIGS), N_SEEDS), flush=True)

    adata = load_data(sample)
    gt = adata.obs['ground_truth'].values
    data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

    for cfg_name, alpha, beta_max, anneal, swa_start, structured in CONFIGS:
        beta_warmup, beta_ramp = (200, 400) if anneal else (0, 0)
        for seed in range(N_SEEDS):
            if (cfg_name, seed) in done:
                continue
            try:
                set_seed(seed)
                model, emb = train_spagvae(
                    data, adata, n_epochs=800, lr=1e-3,
                    beta_max=beta_max, beta_warmup=beta_warmup, beta_ramp=beta_ramp,
                    diffusion_alpha=alpha, latent_dim=30, n_pca=30,
                    swa_start=swa_start, structured_prior=structured,
                    device=DEVICE)
                labels = mclust_labels(emb, K, MCLUST_SEED)
                ari_raw = adjusted_rand_score(gt, labels)
                labels_ref = spatial_label_refine(labels, adata, rad_cutoff=150, n_iter=2)
                ari_ref = adjusted_rand_score(gt, labels_ref)
            except Exception as e:
                print('  ERROR %s seed=%d: %s' % (cfg_name, seed, e), flush=True)
                ari_raw = ari_ref = np.nan
            records.append({'sample': sample, 'config': cfg_name, 'alpha': alpha,
                            'beta_max': beta_max, 'anneal': anneal,
                            'swa_start': swa_start, 'structured_prior': structured,
                            'seed': seed, 'ari_raw': ari_raw, 'ari_refined': ari_ref})
            pd.DataFrame(records).to_csv(csv_out, index=False)

        a = np.array([r['ari_refined'] for r in records
                      if r['config'] == cfg_name and not np.isnan(r['ari_refined'])])
        if len(a):
            print('  %-10s ARI_refined=%.4f std=%.4f CV=%.1f%% (n=%d)'
                  % (cfg_name, a.mean(), a.std(),
                     a.std() / a.mean() * 100 if a.mean() > 0 else 0, len(a)), flush=True)

    print('\n[%s] done.' % sample, flush=True)


if __name__ == '__main__':
    main()
