#!/usr/bin/env python
"""Fill the missing beta-sweep points at the canonical alpha=0.5: run
beta_max in {0.01, 0.10} on all 12 DLPFC sections, 50 seeds each, with the same
pipeline as run_spagvae_improve (so Fig 4a is a genuine alpha=0.5 KL-weight
sweep rather than borrowing alpha=0.3 numbers). beta=0.05 at alpha=0.5 already
exists in outputs/spagvae_improve (a0.5_b0.05_swa_mc2020). One array task per
GPU chunk of 4 sections (--gpuidx 0..2)."""
import warnings; warnings.filterwarnings('ignore')
import sys, os, argparse
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
import numpy as np, pandas as pd, torch, random
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from STAGATE_pyG.utils import Transfer_pytorch_Data
from star.spagvae import train_spagvae, spatial_label_refine
import run_spagvae_improve as imp  # reuse load_data / mclust_labels / set_seed / constants

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 50
OUT_DIR = 'outputs/spagvae_beta05'
os.makedirs(OUT_DIR, exist_ok=True)

# (name, alpha, d_z, refine, beta_max, lr, swa_start, mc_seed) -- mirror improve config
CONFIGS = [
    ('a0.5_b0.01_swa_mc2020', 0.5, 30, True, 0.01, 1e-3, 600, 2020),
    ('a0.5_b0.1_swa_mc2020',  0.5, 30, True, 0.10, 1e-3, 600, 2020),
]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpuidx', type=int, default=0); a = p.parse_args()
    samples = imp.ALL_SAMPLES[a.gpuidx * 4: a.gpuidx * 4 + 4]
    out_csv = '%s/results_gpu%d.csv' % (OUT_DIR, a.gpuidx)
    recs = pd.read_csv(out_csv).to_dict('records') if os.path.exists(out_csv) else []
    done = {(r['sample'], r['config'], int(r['seed'])) for r in recs}
    print('beta@a0.5 | gpuidx=%d samples=%s' % (a.gpuidx, samples), flush=True)
    for sample in samples:
        adata = imp.load_data(sample)
        gt = adata.obs['ground_truth'].values
        K = imp.N_CLUSTERS[sample]
        data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])
        for name, alpha, d_z, refine, beta_max, lr, swa_start, mc_seed in CONFIGS:
            refs = []
            for seed in range(N_SEEDS):
                if (sample, name, seed) in done:
                    continue
                try:
                    imp.set_seed(seed)
                    _, emb = train_spagvae(data, adata, n_epochs=800, lr=lr,
                                           beta_max=beta_max, beta_warmup=200, beta_ramp=400,
                                           diffusion_alpha=alpha, latent_dim=d_z, n_pca=d_z,
                                           swa_start=swa_start, device=DEVICE)
                    labels = imp.mclust_labels(emb, K, mc_seed)
                    lab = spatial_label_refine(labels, adata, rad_cutoff=150, n_iter=2)
                    ari = adjusted_rand_score(gt, lab)
                except Exception as e:
                    print('  ERR %s %s seed=%d: %s' % (sample, name, seed, str(e)[:80]), flush=True)
                    ari = np.nan
                refs.append(ari)
                recs.append({'sample': sample, 'config': name, 'alpha': alpha, 'beta_max': beta_max,
                             'seed': seed, 'ari_refined': ari})
            pd.DataFrame(recs).to_csv(out_csv, index=False)
            v = np.array([x for x in refs if not np.isnan(x)])
            if len(v):
                print('  %s %s ARI=%.4f CV=%.1f%%' % (sample, name, v.mean(), v.std() / v.mean() * 100), flush=True)
    print('GPU%d DONE' % a.gpuidx, flush=True)


if __name__ == '__main__':
    main()
