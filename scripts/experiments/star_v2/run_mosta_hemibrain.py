#!/usr/bin/env python
"""
Run SpaGVAE + STAGATE on MOSTA Mouse Hemibrain (Stereo-seq, bin50).
Ground truth: anatomical region annotations.
  gpuidx=0 -> Mouse_brain.h5ad  (adult hemibrain, 38K spots)
  gpuidx=1 -> P7 Mouse Brain bin50 (62K spots)
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, argparse, random
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np, scanpy as sc, pandas as pd, torch
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import STAGATE_pyG as STAGATE
from STAGATE_pyG.utils import Transfer_pytorch_Data
from star.spagvae import train_spagvae, spatial_label_refine

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS  = 20
MC_SEED  = 2020
N_SAMPLE = 10000
OUT_DIR  = 'outputs/spagvae_extra_datasets'
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    0: ('mosta_hemibrain',
        '/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/MOSTA/MouseHemibrain/Mouse_brain.h5ad',
        'annotation'),
    1: ('stereo_p7_brain',
        '/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/P7MouseBrain/MouseBrain_P7_section1_bin50.h5ad',
        'annotation'),
}

NOISE_LABELS = {'nan', 'NaN', '', 'Blood', 'Cavity', 'unknown', 'Unknown'}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adaptive_rad(adata, k=6):
    coor = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coor)
    dists, _ = nbrs.kneighbors(coor)
    return float(np.median(dists[:, k]) * 1.5)


def stratified_subsample(adata, gt, n_target, seed=42):
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(gt, return_counts=True)
    weights = counts / counts.sum()
    per_class = np.maximum(1, (weights * n_target).astype(int))
    diff = n_target - per_class.sum()
    if diff > 0:
        per_class[np.argsort(-counts)[:diff]] += 1
    elif diff < 0:
        per_class[np.argsort(counts)[:-diff]] -= 1
    indices = []
    for cls, n in zip(unique, per_class):
        cls_idx = np.where(gt == cls)[0]
        chosen = rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices


def load_dataset(name, path, gt_col):
    print(f'Loading {name}...', flush=True)
    adata = sc.read_h5ad(path)

    gt = adata.obs[gt_col].astype(str).values
    keep_mask = ~np.isin(gt, list(NOISE_LABELS))
    adata = adata[keep_mask].copy()
    gt = gt[keep_mask]

    K = len(np.unique(gt))
    print(f'  Full: {adata.shape[0]} spots, K={K}', flush=True)
    print(f'  Classes: {sorted(set(gt))[:15]}', flush=True)

    if adata.shape[0] > N_SAMPLE:
        idx = stratified_subsample(adata, gt, N_SAMPLE, seed=42)
        adata = adata[idx].copy()
        gt = gt[idx]
        print(f'  Subsampled to {len(gt)} spots', flush=True)

    adata.obs['ground_truth'] = gt
    adata.var_names_make_unique()

    adata.X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    n_hvg = min(3000, adata.shape[1])
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # normalise spatial key
    for key in ['X_spatial', 'spatial']:
        if key in adata.obsm:
            adata.obsm['spatial'] = adata.obsm[key]
            break

    rad = adaptive_rad(adata, k=6)
    print(f'  Auto rad_cutoff: {rad:.1f}', flush=True)
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad)

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    return adata, gt, K, rad


def mclust_labels(emb, K, seed):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % seed)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def run_method(name, method_fn, data, adata, gt, K, rad, all_records, dataset, csv_path):
    already_done = {(r['method'], int(r['seed'])) for r in all_records}
    print(f'\n>>> {name}', flush=True)
    if all((name, s) in already_done for s in range(N_SEEDS)):
        print('  Already done, skipping.', flush=True)
        aris = [r['ari'] for r in all_records if r['method'] == name]
        a = np.array([x for x in aris if not np.isnan(x)])
        print(f'  {name}: ARI={a.mean():.4f} std={a.std():.4f}', flush=True)
        return
    aris = []
    for seed in range(N_SEEDS):
        try:
            emb = method_fn(data, K, seed)
            labels = mclust_labels(emb, K, MC_SEED)
            if name == 'SpaGVAE':
                labels = spatial_label_refine(labels, adata, rad_cutoff=rad, n_iter=2)
            ari = adjusted_rand_score(gt, labels)
        except Exception as e:
            print(f'  ERROR seed={seed}: {str(e)[:80]}', flush=True)
            ari = np.nan
        aris.append(ari)
        all_records.append({'dataset': dataset, 'method': name, 'seed': seed, 'ari': ari})
    a = np.array([x for x in aris if not np.isnan(x)])
    print(f'  {name}: ARI={a.mean():.4f} std={a.std():.4f}', flush=True)
    pd.DataFrame(all_records).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuidx', type=int, default=0)
    args = parser.parse_args()

    dataset, path, gt_col = DATASETS[args.gpuidx]
    print(f'=== Dataset: {dataset} | gpuidx={args.gpuidx} ===', flush=True)

    adata, gt, K, rad = load_dataset(dataset, path, gt_col)
    print(f'  N spots: {len(gt)}, K={K}', flush=True)
    data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

    csv_path = f'{OUT_DIR}/results_{dataset}.csv'
    if os.path.exists(csv_path):
        all_records = pd.read_csv(csv_path).to_dict('records')
        print(f'  Loaded {len(all_records)} existing records', flush=True)
    else:
        all_records = []

    def spagvae_run(data_, K_, seed):
        set_seed(seed)
        _, emb = train_spagvae(
            data_, adata, n_epochs=800, lr=1e-3,
            beta_max=0.05, beta_warmup=200, beta_ramp=400,
            diffusion_alpha=0.5, latent_dim=30, n_pca=30,
            swa_start=600, device=DEVICE)
        return emb

    run_method('SpaGVAE', spagvae_run,
               data, adata, gt, K, rad, all_records, dataset, csv_path)

    import torch.nn.functional as _F
    n_feat = data.x.shape[1]

    def stagate_run(data_, K_, seed):
        set_seed(seed)
        model = STAGATE.STAGATE(hidden_dims=[n_feat, 512, 30]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        d = data_.to(DEVICE)
        model.train()
        for _ in range(800):
            opt.zero_grad()
            z, xr = model(d.x, d.edge_index)
            _F.mse_loss(xr, d.x).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z, _ = model(d.x, d.edge_index)
        return z.cpu().numpy()

    run_method('STAGATE', stagate_run,
               data, adata, gt, K, rad, all_records, dataset, csv_path)

    df = pd.DataFrame(all_records)
    summary = []
    for m, grp in df.groupby('method'):
        a = grp['ari'].dropna().values
        if len(a) == 0:
            continue
        summary.append({'dataset': dataset, 'method': m, 'n': len(a),
                        'mean': a.mean(), 'std': a.std(),
                        'cv_pct': a.std()/a.mean()*100 if a.mean() > 0 else 0})
    pd.DataFrame(summary).to_csv(f'{OUT_DIR}/summary_{dataset}.csv', index=False)
    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
