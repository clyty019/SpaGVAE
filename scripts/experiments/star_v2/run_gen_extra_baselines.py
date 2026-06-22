#!/usr/bin/env python
"""Complete the generalisation comparison: run SpaGCN and stCluster (the two
DLPFC baselines missing from the generalisation table) on the Visium MOB and
MOSTA Stereo-seq hemibrain datasets, 20 seeds each, so the table covers the same
six methods as DLPFC. SpaGCN uses raw counts (louvain); stCluster uses the
standard PCA(20)+Mclust readout. One CSV per (dataset, method) pair.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, importlib.util, random
import numpy as np, scanpy as sc, pandas as pd, torch
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

# reuse hemibrain subsample helpers
hspec = importlib.util.spec_from_file_location(
    'hemi', os.path.join(os.path.dirname(__file__), 'run_mosta_hemibrain.py'))
hemi = importlib.util.module_from_spec(hspec); hspec.loader.exec_module(hemi)

OUT = 'outputs/spagvae_extra_datasets/results_gen_extra_baselines.csv'
N_SEEDS, MC_SEED = 20, 2020
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('outputs/spagvae_extra_datasets', exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def adaptive_rad(coor, k=6):
    d = cdist(coor, coor); d.sort(axis=1)
    return float(np.median(d[:, k]) * 1.5)


def load_raw(dataset):
    """Return (counts_adata_with_spatial_and_gt, gt, K, rad). counts in .X (raw)."""
    if dataset == 'visium_mob':
        a = sc.read_h5ad('data_extra/downloads/Visium_MOB.h5ad')
        counts = a.layers['raw_count']
        gt = a.obs['clusters'].astype(str).values
        coor = np.asarray(a.obsm['spatial'])
    else:  # mosta_hemibrain
        a = sc.read_h5ad('/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/MOSTA/MouseHemibrain/Mouse_brain.h5ad')
        gt = a.obs['annotation'].astype(str).values
        keep = ~np.isin(gt, list(hemi.NOISE_LABELS))
        a = a[keep].copy(); gt = gt[keep]
        idx = hemi.stratified_subsample(a, gt, hemi.N_SAMPLE, seed=42)
        a = a[idx].copy(); gt = gt[idx]
        counts = a.layers['count']
        for key in ['X_spatial', 'spatial']:
            if key in a.obsm:
                a.obsm['spatial'] = a.obsm[key]
        coor = np.asarray(a.obsm['spatial'])
    raw = sc.AnnData(X=(counts.toarray() if sp.issparse(counts) else np.asarray(counts)),
                     obs=pd.DataFrame(index=[str(i) for i in range(len(gt))]))
    raw.var_names = [str(g) for g in a.var_names]
    raw.obsm['spatial'] = coor
    raw.obs['ground_truth'] = gt
    K = len(np.unique(gt))
    return raw, gt, K, adaptive_rad(coor)


def mclust(emb, K):
    if emb.shape[1] > 20:
        emb = PCA(n_components=20, random_state=0).fit_transform(emb)
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(%d)' % MC_SEED)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def run_spagcn(raw, K, seed):
    import SpaGCN
    set_seed(seed)
    adata = raw.copy()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    x = adata.obsm['spatial'][:, 0].astype(float); y = adata.obsm['spatial'][:, 1].astype(float)
    adj = SpaGCN.calculate_adj_matrix(x=x, y=y, histology=False)
    l = SpaGCN.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    clf = SpaGCN.SpaGCN(); clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init='louvain', num_pcs=50, lr=0.05, max_epochs=200)
    y_pred, _ = clf.predict()
    return adjusted_rand_score(adata.obs['ground_truth'], y_pred)


def run_stcluster(raw, K, rad, seed):
    from stCluster.stCluster.train import train as stCluster_train
    set_seed(seed)
    adata = raw.copy()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=min(3000, adata.shape[1]))
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var['highly_variable']].copy()
    sd = '/tmp/stcluster_gen/%d' % seed; os.makedirs(sd, exist_ok=True)
    adata_train, _ = stCluster_train(adata, radius=rad, ae_rate=0.8, adj_rate=0.2,
                                     pred_rate=0.3, seed=seed,
                                     save_model='%s/model.pth' % sd, show=False)
    z = None
    for kk in ['embedding', 'stCluster', 'X_pca']:
        if kk in adata_train.obsm:
            z = adata_train.obsm[kk]; break
    return adjusted_rand_score(adata.obs['ground_truth'].values, mclust(z, K))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--task', type=int, required=True); a = p.parse_args()
    # task: 0=MOB/SpaGCN 1=MOB/stCluster 2=hemi/SpaGCN 3=hemi/stCluster
    dataset = 'visium_mob' if a.task < 2 else 'mosta_hemibrain'
    method = 'SpaGCN' if a.task % 2 == 0 else 'stCluster'
    print('=== %s | %s ===' % (dataset, method), flush=True)
    raw, gt, K, rad = load_raw(dataset)
    print('  spots=%d K=%d rad=%.1f' % (len(gt), K, rad), flush=True)
    recs = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
    done = {(r['dataset'], r['method'], int(r['seed'])) for r in recs}
    aris = []
    for seed in range(N_SEEDS):
        if (dataset, method, seed) in done:
            continue
        try:
            ari = run_spagcn(raw, K, seed) if method == 'SpaGCN' else run_stcluster(raw, K, rad, seed)
        except Exception as e:
            print('  ERR seed=%d: %s' % (seed, str(e)[:110]), flush=True); ari = np.nan
        aris.append(ari)
        recs.append({'dataset': dataset, 'method': method, 'seed': seed, 'ari': ari})
        print('  %s %s seed=%d ARI=%.4f' % (dataset, method, seed, ari), flush=True)
        pd.DataFrame(recs).to_csv(OUT, index=False)
    v = np.array([x for x in aris if not np.isnan(x)])
    if len(v):
        print('>>> %s %s mean=%.4f std=%.4f CV=%.1f%%' % (dataset, method, v.mean(), v.std(), v.std()/v.mean()*100), flush=True)
    print('TASK %d DONE' % a.task, flush=True)


if __name__ == '__main__':
    main()
