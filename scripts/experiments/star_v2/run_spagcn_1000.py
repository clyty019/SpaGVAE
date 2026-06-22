#!/usr/bin/env python
"""
SpaGCN at 1,000 seeds per DLPFC section, matching the seed-sensitivity protocol.
Pipeline copied verbatim from run_new_baselines_v2.run_spagcn (the run that
produced the 50-seed SpaGCN numbers): louvain-initialised SpaGCN, histology off.

One SLURM array task per (section, seed-chunk): 12 sections x 4 chunks of 250.
Each task writes its own chunk file; merge with --summary.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, argparse, glob, random
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np, scanpy as sc, pandas as pd, torch
from sklearn.metrics import adjusted_rand_score

DATA_ROOT = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
ALL_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
N_CLUSTERS = {'151507': 7, '151508': 7, '151509': 7, '151510': 7, '151669': 5,
              '151670': 5, '151671': 5, '151672': 5, '151673': 7, '151674': 7,
              '151675': 7, '151676': 7}
N_SEEDS, CHUNK = 1000, 250
N_CHUNKS = N_SEEDS // CHUNK
OUT_DIR = 'outputs/spagcn_1000'
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_data_raw(sample):
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
    return adata


def run_spagcn(adata_raw, seed, n_clusters):
    import SpaGCN
    set_seed(seed)
    adata = adata_raw.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    x_pixel = adata.obsm['spatial'][:, 0].astype(float)
    y_pixel = adata.obsm['spatial'][:, 1].astype(float)
    adj = SpaGCN.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    l = SpaGCN.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    clf = SpaGCN.SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init='louvain', num_pcs=50, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    return adjusted_rand_score(adata.obs['ground_truth'], y_pred)


def run_task(task):
    sample = ALL_SAMPLES[task // N_CHUNKS]
    chunk = task % N_CHUNKS
    seeds = range(chunk * CHUNK, (chunk + 1) * CHUNK)
    K = N_CLUSTERS[sample]
    csv_out = os.path.join(OUT_DIR, 'results_%s_c%d.csv' % (sample, chunk))
    if os.path.exists(csv_out):
        done_df = pd.read_csv(csv_out)
        done = set(done_df['seed'].tolist()); records = done_df.to_dict('records')
    else:
        done, records = set(), []

    print('SpaGCN-1000 | %s chunk %d seeds[%d,%d) K=%d | already=%d'
          % (sample, chunk, seeds.start, seeds.stop, K, len(done)), flush=True)
    adata_raw = load_data_raw(sample)

    for seed in seeds:
        if seed in done:
            continue
        try:
            ari = run_spagcn(adata_raw, seed, K)
        except Exception as e:
            print('  ERROR seed=%d: %s' % (seed, str(e)[:90]), flush=True)
            ari = np.nan
        records.append({'sample': sample, 'method': 'SpaGCN', 'seed': seed, 'ari': ari})
        if seed % 25 == 0:
            pd.DataFrame(records).to_csv(csv_out, index=False)
    pd.DataFrame(records).to_csv(csv_out, index=False)
    a = np.array([r['ari'] for r in records if not np.isnan(r['ari'])])
    print('[%s c%d] done n=%d mean=%.4f' % (sample, chunk, len(a), a.mean() if len(a) else float('nan')), flush=True)


def summary():
    rows, pooled = [], {}
    for s in ALL_SAMPLES:
        files = glob.glob(os.path.join(OUT_DIR, 'results_%s_c*.csv' % s))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates('seed')
        a = df['ari'].dropna().values
        pooled[s] = a
        rows.append({'sample': s, 'n': len(a), 'mean': a.mean(), 'std': a.std(),
                     'cv_pct': a.std() / a.mean() * 100,
                     'p5': np.percentile(a, 5), 'p95': np.percentile(a, 95)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'summary_1000.csv'), index=False)
    print(out.to_string(index=False))
    allv = np.concatenate(list(pooled.values())) if pooled else np.array([])
    if len(allv):
        print('\nPOOLED: n=%d mean=%.4f median=%.4f P5=%.4f P95=%.4f std=%.4f CV=%.1f%%'
              % (len(allv), allv.mean(), np.median(allv), np.percentile(allv, 5),
                 np.percentile(allv, 95), allv.std(), allv.std() / allv.mean() * 100))
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
