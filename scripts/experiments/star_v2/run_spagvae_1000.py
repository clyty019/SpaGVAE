#!/usr/bin/env python
"""
SpaGVAE at 1,000 seeds per DLPFC section, matching the baseline seed-sensitivity
protocol. Canonical config (a0.5, b0.05, beta-anneal 200-400, SWA from 600,
spatial refine n_iter=2, Mclust seed 2020) -- identical to run_spagvae_improve.

Parallelism: one SLURM array task per (section, seed-chunk). 12 sections x 4
chunks of 250 seeds = 48 tasks. Each task writes its own chunk file to avoid
write races; merge with --summary afterwards.

Usage:
  python run_spagvae_1000.py --task <0..47>     # run one (section, chunk)
  python run_spagvae_1000.py --summary          # merge chunks -> per-section table
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, argparse, glob
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
N_SEEDS = 1000
CHUNK = 250                      # seeds per task
N_CHUNKS = N_SEEDS // CHUNK      # 4
MCLUST_SEED = 2020
ALPHA, BETA_MAX, SWA_START = 0.5, 0.05, 600
N_CLUSTERS = {'151507': 7, '151508': 7, '151509': 7, '151510': 7, '151669': 5,
              '151670': 5, '151671': 5, '151672': 5, '151673': 7, '151674': 7,
              '151675': 7, '151676': 7}
OUT_DIR = 'outputs/spagvae_1000'
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
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    return adata


def mclust_labels(emb, K):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % MCLUST_SEED)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def run_task(task):
    sample = ALL_SAMPLES[task // N_CHUNKS]
    chunk = task % N_CHUNKS
    seeds = range(chunk * CHUNK, (chunk + 1) * CHUNK)
    K = N_CLUSTERS[sample]
    csv_out = os.path.join(OUT_DIR, 'results_%s_c%d.csv' % (sample, chunk))

    if os.path.exists(csv_out):
        done_df = pd.read_csv(csv_out)
        done = set(done_df['seed'].tolist())
        records = done_df.to_dict('records')
    else:
        done, records = set(), []

    print('SpaGVAE-1000 | %s chunk %d seeds[%d,%d) K=%d device=%s | already=%d'
          % (sample, chunk, seeds.start, seeds.stop, K, DEVICE, len(done)), flush=True)

    adata = load_data(sample)
    gt = adata.obs['ground_truth'].values
    data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

    for seed in seeds:
        if seed in done:
            continue
        try:
            set_seed(seed)
            _, emb = train_spagvae(data, adata, n_epochs=800, lr=1e-3,
                                   beta_max=BETA_MAX, beta_warmup=200, beta_ramp=400,
                                   diffusion_alpha=ALPHA, latent_dim=30, n_pca=30,
                                   swa_start=SWA_START, device=DEVICE)
            labels = mclust_labels(emb, K)
            ari_raw = adjusted_rand_score(gt, labels)
            labels_ref = spatial_label_refine(labels, adata, rad_cutoff=150, n_iter=2)
            ari_ref = adjusted_rand_score(gt, labels_ref)
        except Exception as e:
            print('  ERROR seed=%d: %s' % (seed, str(e)[:90]), flush=True)
            ari_raw = ari_ref = np.nan
        records.append({'sample': sample, 'seed': seed,
                        'ari_raw': ari_raw, 'ari_refined': ari_ref})
        if seed % 25 == 0:
            pd.DataFrame(records).to_csv(csv_out, index=False)
    pd.DataFrame(records).to_csv(csv_out, index=False)
    a = np.array([r['ari_refined'] for r in records if not np.isnan(r['ari_refined'])])
    print('[%s c%d] done n=%d mean=%.4f' % (sample, chunk, len(a), a.mean() if len(a) else float('nan')), flush=True)


def summary():
    rows = []
    pooled = {}
    for s in ALL_SAMPLES:
        files = glob.glob(os.path.join(OUT_DIR, 'results_%s_c*.csv' % s))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates('seed')
        a = df['ari_refined'].dropna().values
        pooled[s] = a
        rows.append({'sample': s, 'n': len(a), 'mean': a.mean(), 'std': a.std(),
                     'cv_pct': a.std() / a.mean() * 100,
                     'p5': np.percentile(a, 5), 'p95': np.percentile(a, 95)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'summary_1000.csv'), index=False)
    print(out.to_string(index=False))
    allv = np.concatenate(list(pooled.values())) if pooled else np.array([])
    if len(allv):
        print('\nPOOLED over all sections: n=%d mean=%.4f median=%.4f P5=%.4f P95=%.4f std=%.4f'
              % (len(allv), allv.mean(), np.median(allv),
                 np.percentile(allv, 5), np.percentile(allv, 95), allv.std()))
        print('Per-section-mean average: %.4f' % out['mean'].mean())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.summary:
        summary()
    elif a.task is not None:
        run_task(a.task)
    else:
        p.print_help()
