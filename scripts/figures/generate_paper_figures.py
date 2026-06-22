#!/usr/bin/env python
"""
Generate all paper figures for SpaGVAE BMC Bioinformatics submission.

Figures produced:
  fig1_spatial.pdf     - Spatial cluster maps: GT + 4 methods on 151673
  fig2_ari_bar.pdf     - Per-section ARI grouped bar chart (12 sections)
  fig3_stability.pdf   - Seed-stability violin plots (all methods, all sections)
  fig4_mob_spatial.pdf - Stereo-seq MOB spatial domain map (SpaGVAE, qualitative)

Usage:
  python scripts/figures/generate_paper_figures.py [--spatial] [--mob]
  --spatial : also generate spatial cluster maps (needs GPU + DLPFC data)
  --mob     : also generate Stereo-seq MOB map (needs GPU)
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

OUT_DIR = 'overleaf/bmc-bioinformatics/figures'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_ROOT = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
STEREO_MOB = '/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/MouseOlfactoryBulb/Preprocessed/filtered_adata.h5ad'

ALL_SAMPLES = ['151507','151508','151509','151510','151669','151670',
               '151671','151672','151673','151674','151675','151676']
N_CLUSTERS  = {'151507':7,'151508':7,'151509':7,'151510':7,
               '151669':5,'151670':5,'151671':5,'151672':5,
               '151673':7,'151674':7,'151675':7,'151676':7}

# Colour palette -- Morandi (muted, low-saturation) tones.
# SpaGVAE uses a deeper sage-teal so our method stands out; baselines are muted.
METHOD_COLORS = {
    'SpaGVAE':  '#5F8575',  # deeper sage-teal (accent)
    'GraphST':  '#C99DA3',  # dusty rose
    'STAGATE':  '#9AAAB8',  # slate blue-grey
    'SpaceFlow':'#A7B59A',  # sage
    'stCluster':'#D2BE9C',  # muted sand
    'SpaGCN':   '#B5A8BE',  # greyish lavender
}
# Medium-saturation categorical palette for spatial-domain maps: distinct enough
# to read each of the K domains on small dots, but soft enough not to be garish.
DOMAIN_PALETTE = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4',
                  '#8C613C', '#DC7EC0', '#82C6E2', '#D5BB67', '#797979']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

BASE_ROOT = '/extra/zhanglab0/INDV/zihend1/StaR'   # 1000-seed baseline checkpoints

def load_method_aris(method):
    """Per-section per-seed ARI at the matched 1000-seed protocol used in the
    paper tables. Returns {sample: np.array}. SpaGVAE/SpaGCN come from the
    1000-seed runs; the four baselines from their seed-sensitivity checkpoints."""
    import glob
    out = {}
    if method in ('SpaGVAE', 'SpaGCN'):
        d = 'spagvae_1000' if method == 'SpaGVAE' else 'spagcn_1000'
        col = 'ari_refined' if method == 'SpaGVAE' else 'ari'
        fs = glob.glob(f'outputs/{d}/results_*_c*.csv')
        if fs:
            df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(['sample','seed'])
            df['sample'] = df['sample'].astype(str)
            for s in ALL_SAMPLES:
                out[s] = df[df['sample'] == s][col].dropna().values
    else:
        for s in ALL_SAMPLES:
            f = f'{BASE_ROOT}/{s}/{method}/ari_checkpoint.csv'
            out[s] = pd.read_csv(f)['ARI'].dropna().values if os.path.exists(f) else np.array([])
    return out


# ---------------------------------------------------------------------------
# Figure 2: Per-section ARI grouped bar chart
# ---------------------------------------------------------------------------

def fig_ari_bar():
    names = ['SpaGVAE', 'GraphST', 'STAGATE', 'SpaceFlow', 'stCluster']
    aris = {m: load_method_aris(m) for m in names}
    methods = [(m, aris[m]) for m in names]
    n_methods = len(methods)
    n_samples = len(ALL_SAMPLES)
    x = np.arange(n_samples)
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 4.6))

    for i, (name, d) in enumerate(methods):
        means = [d[s].mean() if len(d.get(s, [])) else 0 for s in ALL_SAMPLES]
        stds  = [d[s].std()  if len(d.get(s, [])) else 0 for s in ALL_SAMPLES]
        offset = (i - n_methods/2 + 0.5) * width
        edge = 'black' if name == 'SpaGVAE' else 'none'
        bars = ax.bar(x + offset, means, width, label=name,
                      color=METHOD_COLORS[name], alpha=0.95,
                      edgecolor=edge, linewidth=0.8,
                      yerr=stds, error_kw={'elinewidth':0.8,'capsize':2})

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_SAMPLES, rotation=45, ha='right')
    ax.set_xlabel('DLPFC section')
    ax.set_ylabel('ARI (mean $\\pm$ std)')
    ax.set_ylim(0, 0.82)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=5,
              framealpha=0.0, fontsize=13, columnspacing=1.6, handlelength=1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='k', linewidth=0.5)

    # Donor separator lines
    for x_pos in [3.5, 7.5]:
        ax.axvline(x_pos, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(1.5, 0.785, 'Donor 1', ha='center', fontsize=12, color='dimgray')
    ax.text(5.5, 0.785, 'Donor 2', ha='center', fontsize=12, color='dimgray')
    ax.text(9.5, 0.785, 'Donor 3', ha='center', fontsize=12, color='dimgray')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT_DIR}/fig2_ari_bar.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig2_ari_bar', flush=True)


# ---------------------------------------------------------------------------
# Figure 3: Seed-stability violin plots
# ---------------------------------------------------------------------------

def fig_stability_violin():
    VMETHODS = ['SpaGVAE', 'GraphST', 'STAGATE', 'SpaceFlow', 'stCluster', 'SpaGCN']
    methods_ari = [(m, load_method_aris(m)) for m in VMETHODS]

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharey=False)
    axes = axes.flatten()

    for idx, sample in enumerate(ALL_SAMPLES):
        ax = axes[idx]
        plot_data, colors, names = [], [], []
        for name, aris in methods_ari:
            if sample in aris and len(aris[sample]) > 0:
                plot_data.append(aris[sample])
                colors.append(METHOD_COLORS[name])
                names.append(name)

        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                              showmedians=True, showextrema=False, widths=0.85)
        for pc, col, nm in zip(parts['bodies'], colors, names):
            pc.set_facecolor(col)
            pc.set_alpha(0.95)
            pc.set_edgecolor('black' if nm == 'SpaGVAE' else 'none')
            pc.set_linewidth(1.1 if nm == 'SpaGVAE' else 0)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.6)

        # colour-coded tick marks instead of cramped truncated text labels
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([''] * len(names))
        for t, col in zip(ax.get_xticklabels(), colors):
            pass
        ax.set_title(sample, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx % 4 == 0:
            ax.set_ylabel('ARI', fontsize=14)

    # Shared legend (bottom), large
    patches = [mpatches.Patch(facecolor=METHOD_COLORS[n], label=n,
                              edgecolor='black' if n == 'SpaGVAE' else 'none')
               for n in VMETHODS]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, fontsize=13,
               handlelength=1.4, columnspacing=1.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT_DIR}/fig3_stability_violin.{ext}',
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig3_stability_violin', flush=True)


# ---------------------------------------------------------------------------
# Figure 1: Spatial cluster maps (needs GPU + DLPFC data)
# ---------------------------------------------------------------------------

def run_spagvae_once(data, adata, K, seed, rad):
    import torch
    from star.spagvae import train_spagvae, spatial_label_refine
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('suppressMessages(library(mclust))')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    _, emb = train_spagvae(
        data, adata, n_epochs=800, lr=1e-3,
        beta_max=0.05, beta_warmup=200, beta_ramp=400,
        diffusion_alpha=0.5, latent_dim=30, n_pca=30,
        swa_start=600, device=DEVICE)

    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(2020)')
    robjects.r(f'res <- Mclust(as.matrix(emb_r), G={K}, modelNames="EEE")')
    labels = np.array(robjects.r('res$classification'))
    labels = spatial_label_refine(labels, adata, rad_cutoff=rad, n_iter=2)
    return labels


def run_stagate_once(data, K, seed):
    import torch
    import STAGATE_pyG as ST
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('suppressMessages(library(mclust))')

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model = ST.STAGATE(hidden_dims=[3000, 512, 30]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    d = data.to(DEVICE)
    model.train()
    for _ in range(800):
        opt.zero_grad()
        z, xr = model(d.x, d.edge_index)
        import torch.nn.functional as F
        F.mse_loss(xr, d.x).backward()
        opt.step()
    model.eval()
    import torch
    with torch.no_grad():
        z, _ = model(d.x, d.edge_index)
    emb = z.cpu().numpy()

    robjects.globalenv['emb_r'] = emb.astype(np.float64)
    robjects.r('set.seed(2020)')
    robjects.r(f'res <- Mclust(as.matrix(emb_r), G={K}, modelNames="EEE")')
    return np.array(robjects.r('res$classification'))


def _mclust_pca(emb, K, seed=2020):
    """Generalisation-baseline clustering: PCA(20) if high-dim, then Mclust EEE.
    Matches the protocol used for the GraphST/SpaceFlow generalisation numbers."""
    from sklearn.decomposition import PCA
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('suppressMessages(library(mclust))')
    x = PCA(n_components=20, random_state=0).fit_transform(emb) if emb.shape[1] > 20 else emb
    robjects.globalenv['emb_r'] = x.astype(np.float64)
    robjects.r('set.seed(%d)' % seed)
    robjects.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="EEE")' % K)
    return np.array(robjects.r('res$classification'))


def run_graphst_once(adata, seed):
    import torch
    from GraphST import GraphST as GraphSTModule
    from io import StringIO
    from contextlib import redirect_stdout
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with redirect_stdout(StringIO()):
        gst = GraphSTModule.GraphST(adata.copy(), device=DEVICE, random_seed=seed)
        gst.train()
    gst.model.eval()
    with torch.no_grad():
        z, _, _, _ = gst.model(gst.features, gst.features_a, gst.adj)
    return z.cpu().numpy()


def run_spaceflow_once(adata, seed):
    import torch, sys as _sys
    from SpaceFlow import SpaceFlow
    from io import StringIO
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    sf = SpaceFlow.SpaceFlow(adata=adata.copy())
    sf.preprocessing_data(n_top_genes=3000)
    old = _sys.stdout; _sys.stdout = StringIO()
    try:
        sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-3,
                 epochs=1000, max_patience=50, min_stop=100,
                 random_seed=seed, gpu=0 if torch.cuda.is_available() else -1,
                 regularization_acceleration=True,
                 embedding_save_filepath='/tmp/sf_emb_fig_%d.tsv' % seed)
    finally:
        _sys.stdout = old
    return sf.embedding.copy()


def align_labels(pred, gt):
    """Hungarian matching: align predicted cluster IDs to GT IDs."""
    from scipy.optimize import linear_sum_assignment
    gt_u = np.unique(gt)
    pred_u = np.unique(pred)
    cost = np.zeros((len(gt_u), len(pred_u)))
    for i, g in enumerate(gt_u):
        for j, p in enumerate(pred_u):
            cost[i, j] = -np.sum((gt == g) & (pred == p))
    row, col = linear_sum_assignment(cost)
    mapping = {pred_u[j]: gt_u[i] for i, j in zip(row, col)}
    return np.array([mapping.get(p, p) for p in pred])


def fig_spatial_clusters(sample='151669', scan_seeds=range(20)):
    """Two-row spatial figure on a volatile section: for each method we scan a
    set of seeds, then show the WORST / MEDIAN / BEST seed (by ARI) alongside
    the ground truth. This makes the seed-induced variability visible -- on a
    high-variance section STAGATE's worst and best maps differ markedly, whereas
    SpaGVAE stays consistent."""
    import scanpy as sc
    import STAGATE_pyG as STAGATE
    from STAGATE_pyG.utils import Transfer_pytorch_Data
    from sklearn.metrics import adjusted_rand_score
    import scipy.sparse as sp

    print(f'Generating spatial figure for {sample} (scan {len(list(scan_seeds))} seeds)...', flush=True)
    scan_seeds = list(scan_seeds)
    K = N_CLUSTERS[sample]

    adata = sc.read_visium(os.path.join(DATA_ROOT, sample),
                           count_file=f'{sample}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    gt_df = pd.read_csv(os.path.join(DATA_ROOT, f'{sample}_truth.txt'),
                        sep='\t', header=None, index_col=0)
    gt_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = gt_df.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()

    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    data_pyg = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])
    gt = adata.obs['ground_truth'].values
    coor = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])
    gt_unique = sorted(set(gt))
    gt_int = np.array([{v: i for i, v in enumerate(gt_unique)}[g] for g in gt])
    palette = DOMAIN_PALETTE[:K]

    # Scan seeds for both methods, recording aligned labels + ARI
    sp_runs, st_runs = {}, {}
    for seed in scan_seeds:
        sl = align_labels(run_spagvae_once(data_pyg, adata, K, seed, rad=150), gt_int)
        sp_runs[seed] = (sl, adjusted_rand_score(gt_int, sl))
        tl = align_labels(run_stagate_once(data_pyg, K, seed), gt_int)
        st_runs[seed] = (tl, adjusted_rand_score(gt_int, tl))
        print(f'  seed {seed:2d}: SpaGVAE {sp_runs[seed][1]:.3f}  STAGATE {st_runs[seed][1]:.3f}', flush=True)

    def pick3(runs):
        order = sorted(runs, key=lambda s: runs[s][1])  # ascending ARI
        return order[0], order[len(order) // 2], order[-1]  # worst, median, best

    def plot_spots(ax, labels, title):
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            ax.scatter(coor['x'][mask], coor['y'][mask],
                       c=palette[int(lbl) % len(palette)],
                       s=9, linewidths=0, rasterized=True)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        ax.set_title(title, fontsize=13, pad=3)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    col_kind = ['Worst', 'Median', 'Best']

    # Column 0: ground truth (top) and domain colour legend (bottom)
    plot_spots(axes[0, 0], gt_int, 'Ground truth')
    axes[1, 0].axis('off')
    handles = [mpatches.Patch(facecolor=palette[i], edgecolor='none', label=str(lbl))
               for i, lbl in enumerate(gt_unique)]
    axes[1, 0].legend(handles=handles, loc='center', frameon=False,
                      fontsize=12, title='Domain', title_fontsize=13,
                      handlelength=1.1, labelspacing=0.5)

    for r, (runs, rowlab) in enumerate([(sp_runs, 'SpaGVAE'), (st_runs, 'STAGATE')]):
        three = pick3(runs)
        for c, seed in enumerate(three):
            lab, ari = runs[seed]
            axes[r, c + 1].set_title(f'{rowlab} -- {col_kind[c]}\nARI={ari:.3f}',
                                     fontsize=12, pad=3)
            for lbl in sorted(set(lab)):
                mask = lab == lbl
                axes[r, c + 1].scatter(coor['x'][mask], coor['y'][mask],
                                       c=palette[int(lbl) % len(palette)],
                                       s=9, linewidths=0, rasterized=True)
            axes[r, c + 1].set_aspect('equal'); axes[r, c + 1].invert_yaxis()
            axes[r, c + 1].axis('off')

    fig.suptitle(f'DLPFC section {sample} -- worst / median / best seed per method',
                 fontsize=15, y=1.0, fontweight='bold')
    fig.tight_layout(pad=0.5)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT_DIR}/fig1_spatial_{sample}.{ext}',
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved fig1_spatial_{sample}', flush=True)


# ---------------------------------------------------------------------------
# Figure 4: Stereo-seq MOB spatial map (qualitative, no labels)
# ---------------------------------------------------------------------------

def fig_mob_spatial():
    """Visium MOB generalisation: ground truth plus all four methods
    (SpaGVAE, GraphST, STAGATE, SpaceFlow), each displayed at the seed closest
    to its own 20-seed mean (a representative run); panel titles report mean
    +/- std ARI over 20 seeds. On MOB SpaGVAE attains both the highest accuracy
    and the lowest inter-seed variance."""
    import importlib.util
    from STAGATE_pyG.utils import Transfer_pytorch_Data

    # reuse the proven Visium-MOB loader (GraphST/SpaceFlow runners are inlined
    # below to avoid re-importing run_extra_datasets, which reopens stdout)
    spec = importlib.util.spec_from_file_location(
        'red', 'scripts/experiments/star_v2/run_extra_datasets.py')
    red = importlib.util.module_from_spec(spec); spec.loader.exec_module(red)

    print('Loading Visium MOB...', flush=True)
    adata, gt, K, rad = red.load_dataset('visium_mob')
    data_pyg = Transfer_pytorch_Data(adata[:, adata.var['highly_variable']])
    coor = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])

    gt_u = sorted(set(gt))
    gt_to_int = {v: i for i, v in enumerate(gt_u)}
    gt_int = np.array([gt_to_int[g] for g in gt])

    # mean +/- std over 20 seeds, from both summary files (SpaGVAE/STAGATE +
    # GraphST/SpaceFlow generalisation baselines)
    s1 = pd.read_csv('outputs/spagvae_extra_datasets/summary_visium_mob.csv')
    s2 = pd.read_csv('outputs/spagvae_gen_baselines/summary_visium_mob.csv')
    smean = {r['method']: (r['mean'], r['std'])
             for _, r in pd.concat([s1, s2]).iterrows()}

    # representative seed per method (closest to that method's own 20-seed mean)
    SEEDS = {'SpaGVAE': 0, 'STAGATE': 15, 'GraphST': 8, 'SpaceFlow': 4}
    MC_SEED = 2020
    print(f'  K={K}, rad={rad:.1f}; running 4 methods...', flush=True)
    labs = {}
    labs['SpaGVAE'] = align_labels(
        run_spagvae_once(data_pyg, adata, K, SEEDS['SpaGVAE'], rad), gt_int)
    labs['STAGATE'] = align_labels(
        run_stagate_once(data_pyg, K, SEEDS['STAGATE']), gt_int)
    labs['GraphST'] = align_labels(
        _mclust_pca(run_graphst_once(adata, SEEDS['GraphST']), K, MC_SEED), gt_int)
    labs['SpaceFlow'] = align_labels(
        _mclust_pca(run_spaceflow_once(adata, SEEDS['SpaceFlow']), K, MC_SEED), gt_int)

    palette = DOMAIN_PALETTE[:K]

    def plot(ax, lab, title):
        for l in sorted(set(lab)):
            m = lab == l
            ax.scatter(coor['x'][m], coor['y'][m],
                       c=palette[int(l) % len(palette)], s=16,
                       linewidths=0, rasterized=True)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=4)

    # ordered by descending mean ARI; legend occupies the 6th cell
    order = ['SpaGVAE', 'GraphST', 'STAGATE', 'SpaceFlow']
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    plot(axes[0, 0], gt_int, f'Ground truth ($K={K}$)')
    for ax, m in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]], order):
        mu, sd = smean[m]
        plot(ax, labs[m], f'{m} ({mu:.3f} $\\pm$ {sd:.3f})')

    axes[1, 2].axis('off')
    handles = [mpatches.Patch(facecolor=palette[i], edgecolor='none',
                              label=f'Domain {i + 1}') for i in range(K)]
    axes[1, 2].legend(handles=handles, loc='center', ncol=2, frameon=False,
                      fontsize=13, handlelength=1.3, columnspacing=1.4,
                      labelspacing=0.7)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUT_DIR}/fig4_mob_spatial.{ext}',
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig4_mob_spatial (4 methods: ' +
          ', '.join(f'{m} {smean[m][0]:.3f}' for m in order) + ')', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', action='store_true',
                        help='Generate spatial cluster maps (needs GPU)')
    parser.add_argument('--mob', action='store_true',
                        help='Generate Stereo-seq MOB map (needs GPU)')
    parser.add_argument('--sample', default='151669',
                        help='DLPFC sample for spatial figure (volatile section)')
    args = parser.parse_args()

    print('=== Generating paper figures ===', flush=True)

    print('\n[1/4] ARI bar chart...', flush=True)
    fig_ari_bar()

    print('\n[2/4] Stability violin plots...', flush=True)
    fig_stability_violin()

    if args.spatial:
        print(f'\n[3/4] Spatial cluster maps ({args.sample})...', flush=True)
        fig_spatial_clusters(sample=args.sample, scan_seeds=range(20))
    else:
        print('\n[3/4] Skipping spatial maps (use --spatial to enable)', flush=True)

    if args.mob:
        print('\n[4/4] Stereo-seq MOB map...', flush=True)
        fig_mob_spatial()
    else:
        print('\n[4/4] Skipping MOB map (use --mob to enable)', flush=True)

    print('\nAll done! Figures saved to:', OUT_DIR, flush=True)


if __name__ == '__main__':
    main()
