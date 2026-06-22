#!/usr/bin/env python
"""Single consistent source of truth for the hemibrain SpaGVAE/STAGATE results:
run 20 seeds of each, saving BOTH per-seed ARI and per-seed (GT-aligned) labels
to one .npz. Table 7 (SpaGVAE/STAGATE rows) and Fig 5 (worst/median/best maps)
are then both derived from this file, so the figure's worst/median/best ARIs
match the table's percentiles exactly (no cross-GPU drift between table and
figure)."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, importlib.util
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
spec = importlib.util.spec_from_file_location(
    'hemi', os.path.join(os.path.dirname(__file__), 'run_mosta_hemibrain.py'))
hemi = importlib.util.module_from_spec(spec); spec.loader.exec_module(hemi)
from STAGATE_pyG.utils import Transfer_pytorch_Data
import STAGATE_pyG as STAGATE
from star.spagvae import train_spagvae, spatial_label_refine

DEVICE = hemi.DEVICE
N_SEEDS = 20
OUT = 'outputs/spagvae_extra_datasets/hemibrain_fig20.npz'


def align(pred, gt_int):
    gu, pu = np.unique(gt_int), np.unique(pred)
    cost = np.zeros((len(gu), len(pu)))
    for i, g in enumerate(gu):
        for j, p in enumerate(pu):
            cost[i, j] = -np.sum((gt_int == g) & (pred == p))
    r, c = linear_sum_assignment(cost)
    m = {pu[j]: gu[i] for i, j in zip(r, c)}
    return np.array([m.get(p, p) for p in pred])


ds, path, gtcol = hemi.DATASETS[0]
print('Loading hemibrain...', flush=True)
adata, gt, K, rad = hemi.load_dataset(ds, path, gtcol)
data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])
coor = np.asarray(adata.obsm['spatial'])
gu = sorted(set(gt)); gt_int = np.array([{v: i for i, v in enumerate(gu)}[g] for g in gt])


def spagvae(seed):
    hemi.set_seed(seed)
    _, emb = train_spagvae(data, adata, n_epochs=800, lr=1e-3, beta_max=0.05,
                           beta_warmup=200, beta_ramp=400, diffusion_alpha=0.5,
                           latent_dim=30, n_pca=30, swa_start=600, device=DEVICE)
    lab = spatial_label_refine(hemi.mclust_labels(emb, K, hemi.MC_SEED), adata, rad_cutoff=rad, n_iter=2)
    return adjusted_rand_score(gt, lab), align(lab, gt_int)


def stagate(seed):
    hemi.set_seed(seed)
    nf = data.x.shape[1]
    model = STAGATE.STAGATE(hidden_dims=[nf, 512, 30]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    d = data.to(DEVICE); model.train()
    for _ in range(800):
        opt.zero_grad(); z, xr = model(d.x, d.edge_index); F.mse_loss(xr, d.x).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z, _ = model(d.x, d.edge_index)
    lab = hemi.mclust_labels(z.cpu().numpy(), K, hemi.MC_SEED)
    return adjusted_rand_score(gt, lab), align(lab, gt_int)


res = {}
for name, fn in [('SpaGVAE', spagvae), ('STAGATE', stagate)]:
    aris, labs = [], []
    for s in range(N_SEEDS):
        a, l = fn(s); aris.append(a); labs.append(l)
        print('  %s seed=%d ARI=%.4f' % (name, s, a), flush=True)
    a = np.array(aris)
    res[name + '_ari'] = a
    res[name + '_lab'] = np.array(labs, dtype=np.int16)
    print('>>> %s mean=%.4f median=%.4f P5=%.4f P95=%.4f std=%.4f CV=%.1f%%' % (
        name, a.mean(), np.median(a), np.percentile(a, 5), np.percentile(a, 95),
        a.std(), a.std() / a.mean() * 100), flush=True)

np.savez_compressed(OUT, gt_int=gt_int, coor=coor, region_names=np.array(gu, dtype=object),
                    K=K, **res)
print('Saved', OUT, flush=True)
print('FIG20 DONE', flush=True)
