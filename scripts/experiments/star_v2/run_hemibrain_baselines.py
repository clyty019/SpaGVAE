#!/usr/bin/env python
"""GraphST + SpaceFlow on the MOSTA Stereo-seq hemibrain (10k spots, K=19),
20 seeds each, to complete the generalisation comparison. Protocol matches
run_gen_baselines: PCA(20) + Mclust(EEE), no spatial refinement. Runs both
methods sequentially in one job (writes one CSV, no write races)."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, importlib.util
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
spec = importlib.util.spec_from_file_location(
    'hemi', os.path.join(os.path.dirname(__file__), 'run_mosta_hemibrain.py'))
hemi = importlib.util.module_from_spec(spec); spec.loader.exec_module(hemi)
from STAGATE_pyG.utils import Transfer_pytorch_Data  # noqa

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

OUT = 'outputs/spagvae_extra_datasets/results_hemibrain_genbaselines.csv'
MC_SEED, N_SEEDS = 2020, 20
DEVICE = hemi.DEVICE


def cluster_mclust(emb, K, seed):
    x = PCA(n_components=20, random_state=0).fit_transform(emb) if emb.shape[1] > 20 else emb
    robjects.globalenv['emb_r'] = x.astype(np.float64)
    robjects.r('set.seed(%d)' % seed)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def run_graphst(adata, seed):
    from GraphST import GraphST as G
    from io import StringIO
    from contextlib import redirect_stdout
    hemi.set_seed(seed)
    with redirect_stdout(StringIO()):
        m = G.GraphST(adata.copy(), device=DEVICE, random_seed=seed); m.train()
    m.model.eval()
    with torch.no_grad():
        z, _, _, _ = m.model(m.features, m.features_a, m.adj)
    return z.cpu().numpy()


def run_spaceflow(adata, seed):
    from SpaceFlow import SpaceFlow
    import sys as _s
    from io import StringIO
    hemi.set_seed(seed)
    sf = SpaceFlow.SpaceFlow(adata=adata.copy())
    sf.preprocessing_data(n_top_genes=3000)
    old = _s.stdout; _s.stdout = StringIO()
    try:
        sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-3, epochs=1000,
                 max_patience=50, min_stop=100, random_seed=seed,
                 gpu=0 if torch.cuda.is_available() else -1,
                 regularization_acceleration=True,
                 embedding_save_filepath='/tmp/sf_hemi_%d.tsv' % seed)
    finally:
        _s.stdout = old
    return sf.embedding.copy()


def main():
    ds, path, gtcol = hemi.DATASETS[0]
    print('Loading hemibrain...', flush=True)
    adata, gt, K, rad = hemi.load_dataset(ds, path, gtcol)
    print('  K=%d, spots=%d' % (K, len(gt)), flush=True)
    recs = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
    done = {(r['method'], int(r['seed'])) for r in recs}
    for name, fn in [('GraphST', run_graphst), ('SpaceFlow', run_spaceflow)]:
        aris = []
        for seed in range(N_SEEDS):
            if (name, seed) in done:
                continue
            try:
                ari = adjusted_rand_score(gt, cluster_mclust(fn(adata, seed), K, MC_SEED))
            except Exception as e:
                print('  ERR %s seed=%d: %s' % (name, seed, str(e)[:90]), flush=True); ari = np.nan
            aris.append(ari)
            recs.append({'dataset': ds, 'method': name, 'seed': seed, 'ari': ari})
            print('  %s seed=%d ARI=%.4f' % (name, seed, ari), flush=True)
            pd.DataFrame(recs).to_csv(OUT, index=False)
        v = np.array([x for x in aris if not np.isnan(x)])
        if len(v):
            print('>>> %s mean=%.4f std=%.4f CV=%.1f%%' % (name, v.mean(), v.std(), v.std()/v.mean()*100), flush=True)
    print('HEMIBRAIN BASELINES DONE', flush=True)


if __name__ == '__main__':
    main()
