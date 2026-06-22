"""
SpaGVAE full DLPFC benchmark - 12 samples, 20 seeds each.

Usage (via SLURM array, see slurm/run_spagvae_dlpfc.sbatch):
    python run_spagvae_dlpfc.py --gpuidx 0   # samples 151507-151510
    python run_spagvae_dlpfc.py --gpuidx 1   # samples 151669-151672
    python run_spagvae_dlpfc.py --gpuidx 2   # samples 151673-151676

After all 3 jobs finish, run without --gpuidx to print the summary table:
    python run_spagvae_dlpfc.py --summary
"""

import os, sys, random, argparse, warnings
warnings.filterwarnings('ignore')
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
import scipy.sparse as sp

import STAGATE_pyG as STAGATE
from STAGATE_pyG.utils import Transfer_pytorch_Data

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('suppressMessages(library(mclust))')

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_ROOT  = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
OUT_DIR    = 'outputs/spagvae_dlpfc_full'
os.makedirs(OUT_DIR, exist_ok=True)

ALL_SAMPLES = [
    '151507','151508','151509','151510',
    '151669','151670','151671','151672',
    '151673','151674','151675','151676',
]
GPU_SPLITS = {
    0: ALL_SAMPLES[0:4],
    1: ALL_SAMPLES[4:8],
    2: ALL_SAMPLES[8:12],
}
N_CLUSTERS = {
    '151507':7,'151508':7,'151509':7,'151510':7,
    '151669':5,'151670':5,'151671':5,'151672':5,
    '151673':7,'151674':7,'151675':7,'151676':7,
}
N_SEEDS     = 20
RAD_CUTOFF  = 150
ALPHA       = 0.3
BETA_MAX    = 0.01
MCLUST_SEED = 2020
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
class SpaGVAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, latent_dim=30):
        super().__init__()
        self.conv1   = GATConv(in_dim,     hidden_dim, heads=1, concat=False, dropout=0.0)
        self.conv_mu = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=0.0)
        self.conv_lv = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=0.0)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_lv(h, edge_index)


class SpaGVAEDecoder(nn.Module):
    def __init__(self, latent_dim=30, hidden_dim=512, out_dim=3000):
        super().__init__()
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=1, concat=False, dropout=0.0)
        self.conv2 = GATConv(hidden_dim, out_dim,    heads=1, concat=False, dropout=0.0)

    def forward(self, z, edge_index):
        return self.conv2(F.elu(self.conv1(z, edge_index)), edge_index)


class SpaGVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, latent_dim=30):
        super().__init__()
        self.encoder = SpaGVAEEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = SpaGVAEDecoder(latent_dim, hidden_dim, in_dim)

    def reparameterize(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return mu

    def forward(self, x, edge_index):
        mu, lv = self.encoder(x, edge_index)
        return mu, lv, self.decoder(self.reparameterize(mu, lv), edge_index)


# ─────────────────────────────────────────────────────────────
# Spatial Prior
# ─────────────────────────────────────────────────────────────
def compute_spatial_prior(adata, latent_dim=30, alpha=0.3, rad_cutoff=150):
    hvg = adata[:, adata.var['highly_variable']]
    X   = hvg.X.toarray() if sp.issparse(hvg.X) else np.array(hvg.X)
    z_pca = PCA(n_components=latent_dim, random_state=0).fit_transform(X)

    coor = adata.obsm['spatial']
    adj  = (cdist(coor, coor) < rad_cutoff).astype(np.float32)
    np.fill_diagonal(adj, 0)

    d = adj.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(len(d)) - (d_inv[:, None] * adj * d_inv[None, :])

    return np.linalg.solve(np.eye(len(d)) + alpha * L, z_pca).astype(np.float32)


def kl_structured(mu, lv, mu_prior):
    return 0.5 * torch.sum(lv.exp() + (mu - mu_prior)**2 - 1.0 - lv, dim=-1).mean()


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_spagvae(data, adata, n_epochs=800, lr=1e-3, weight_decay=1e-4,
                  beta_max=0.01, beta_warmup=200, beta_ramp=400,
                  alpha=0.3, latent_dim=30, rad_cutoff=150, swa_start=600):
    data = data.to(DEVICE)
    mu_prior = torch.tensor(
        compute_spatial_prior(adata, latent_dim, alpha, rad_cutoff),
        dtype=torch.float32, device=DEVICE
    )
    model = SpaGVAE(in_dim=data.x.shape[1], latent_dim=latent_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    swa_w, swa_n = None, 0
    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        mu, lv, xr = model(data.x, data.edge_index)

        if epoch < beta_warmup:       beta = 0.0
        elif epoch < beta_ramp:       beta = beta_max * (epoch - beta_warmup) / (beta_ramp - beta_warmup)
        else:                         beta = beta_max

        (F.mse_loss(xr, data.x) + beta * kl_structured(mu, lv, mu_prior)).backward()
        opt.step()

        if epoch >= swa_start:
            swa_n += 1
            if swa_w is None:
                swa_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k in swa_w:
                    swa_w[k] += (model.state_dict()[k].detach() - swa_w[k]) / swa_n

    if swa_w is not None:
        model.load_state_dict(swa_w)
    model.eval()
    with torch.no_grad():
        mu, _, _ = model(data.x, data.edge_index)
    return mu.cpu().numpy()


# ─────────────────────────────────────────────────────────────
# Clustering + Refinement
# ─────────────────────────────────────────────────────────────
def mclust_labels(emb, K):
    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r(f'set.seed({MCLUST_SEED})')
    try:
        robjects.r(f'res <- Mclust(as.matrix(emb_r), G={K}, modelNames="EEE")')
    except Exception:
        robjects.r(f'res <- Mclust(as.matrix(emb_r), G={K})')
    return np.array(robjects.r('res$classification'))


def spatial_refine(labels, adata, rad_cutoff=150, n_iter=2):
    adj = cdist(adata.obsm['spatial'], adata.obsm['spatial']) < rad_cutoff
    np.fill_diagonal(adj, True)
    for _ in range(n_iter):
        new = labels.copy()
        for i in range(len(labels)):
            nbr = labels[adj[i]]
            vals, cnts = np.unique(nbr, return_counts=True)
            new[i] = vals[cnts.argmax()]
        labels = new
    return labels


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
def load_dlpfc(sample):
    fold  = os.path.join(DATA_ROOT, sample)
    adata = sc.read_visium(fold, count_file=f'{sample}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    gt = pd.read_csv(os.path.join(DATA_ROOT, f'{sample}_truth.txt'),
                     sep='\t', header=None, index_col=0)
    gt.columns = ['ground_truth']
    adata.obs['ground_truth'] = gt.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower()
                  .isin(['nan','na','none',''])].copy()

    # Preprocessing - order and parameters are critical
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=RAD_CUTOFF)
    return adata, adata.obs['ground_truth'].astype(str).values


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# Run one sample
# ─────────────────────────────────────────────────────────────
def run_sample(sample):
    K       = N_CLUSTERS[sample]
    csv_out = os.path.join(OUT_DIR, f'results_{sample}.csv')

    # Resume if partially done
    if os.path.exists(csv_out):
        done = pd.read_csv(csv_out)
        done_seeds = set(done['seed'].tolist())
    else:
        done, done_seeds = None, set()

    if len(done_seeds) >= N_SEEDS:
        aris = done['ari'].values
        print(f'[{sample}] already done - ARI={aris.mean():.4f} ± {aris.std():.4f}  CV={aris.std()/aris.mean()*100:.1f}%')
        return

    print(f'\n{"="*50}')
    print(f'[{sample}]  K={K}  device={DEVICE}')
    adata, gt = load_dlpfc(sample)
    data = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])

    records = done.to_dict('records') if done is not None else []
    for seed in range(N_SEEDS):
        if seed in done_seeds:
            continue
        set_seed(seed)
        try:
            emb    = train_spagvae(data, adata, alpha=ALPHA, beta_max=BETA_MAX,
                                   rad_cutoff=RAD_CUTOFF)
            labels = mclust_labels(emb, K)
            labels = spatial_refine(labels, adata, rad_cutoff=RAD_CUTOFF)
            ari    = adjusted_rand_score(gt, labels)
        except Exception as e:
            print(f'  seed={seed} ERROR: {e}')
            ari = float('nan')

        records.append({'sample': sample, 'seed': seed, 'ari': ari})
        pd.DataFrame(records).to_csv(csv_out, index=False)
        print(f'  seed={seed:2d}  ARI={ari:.4f}', flush=True)

    aris = np.array([r['ari'] for r in records if not np.isnan(r['ari'])])
    print(f'[{sample}] DONE - Mean={aris.mean():.4f}  Std={aris.std():.4f}  CV={aris.std()/aris.mean()*100:.1f}%')


# ─────────────────────────────────────────────────────────────
# Summary table (run after all 3 jobs finish)
# ─────────────────────────────────────────────────────────────
def print_summary():
    rows = []
    for s in ALL_SAMPLES:
        path = os.path.join(OUT_DIR, f'results_{s}.csv')
        if not os.path.exists(path):
            rows.append({'sample': s, 'n': 0, 'mean': float('nan'),
                         'std': float('nan'), 'cv': float('nan')})
            continue
        df   = pd.read_csv(path).dropna(subset=['ari'])
        aris = df['ari'].values
        rows.append({'sample': s, 'n': len(aris),
                     'mean': aris.mean(), 'std': aris.std(),
                     'cv': aris.std()/aris.mean()*100})

    df = pd.DataFrame(rows)
    print('\n' + '='*60)
    print('SpaGVAE - DLPFC 12-sample summary')
    print('='*60)
    print(f"{'Sample':>8}  {'N':>4}  {'Mean ARI':>9}  {'Std':>7}  {'CV%':>6}")
    print('-'*60)
    for _, r in df.iterrows():
        print(f"{r['sample']:>8}  {int(r['n']):>4}  {r['mean']:>9.4f}  {r['std']:>7.4f}  {r['cv']:>5.1f}%")
    print('-'*60)
    valid = df.dropna()
    print(f"{'Average':>8}  {'':>4}  {valid['mean'].mean():>9.4f}  "
          f"{valid['std'].mean():>7.4f}  {valid['cv'].mean():>5.1f}%")
    print('='*60)
    df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    print(f'\nSaved → {OUT_DIR}/summary.csv')


# ─────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuidx',  type=int, default=None, choices=[0,1,2])
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.gpuidx is not None:
        for sample in GPU_SPLITS[args.gpuidx]:
            run_sample(sample)
    else:
        parser.print_help()
