"""End-to-end SpaGVAE pipeline: preprocessing, spatial graph, training,
Mclust clustering and spatial label refinement."""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from torch_geometric.data import Data

from .model import set_seed, spatial_label_refine, train_spagvae

__all__ = ["preprocess", "build_graph", "mclust_labels", "run_spagvae"]


def preprocess(adata, n_hvg=3000):
    """Seurat-v3 HVGs on counts, then normalize_total(1e4), log1p and
    scale(zero_center=False, max_value=10)."""
    import scanpy as sc
    a = adata.copy()
    a.X = sp.csr_matrix(a.layers["counts"] if "counts" in a.layers else a.X).astype(np.float32)
    n_hvg = int(min(n_hvg, a.n_vars))
    if n_hvg >= a.n_vars:
        a.var["highly_variable"] = True
    else:
        try:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        except Exception:
            # seurat_v3 needs integer counts; fall back to the seurat flavour on log data
            tmp = a.copy()
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
            sc.pp.highly_variable_genes(tmp, flavor="seurat", n_top_genes=n_hvg)
            a.var["highly_variable"] = tmp.var["highly_variable"].values
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, zero_center=False, max_value=10)
    if not sp.issparse(a.X):
        a.X = sp.csr_matrix(a.X)
    return a


def build_graph(adata, rad):
    """Radius graph on adata.obsm['spatial'] with self loops, features = HVGs."""
    coor = np.asarray(adata.obsm["spatial"], dtype=float)
    n = coor.shape[0]
    g = NearestNeighbors(radius=rad).fit(coor).radius_neighbors_graph(coor, mode="distance").tocoo()
    keep = g.data > 0
    adj = sp.coo_matrix((np.ones(keep.sum()), (g.row[keep], g.col[keep])), shape=(n, n)) + sp.eye(n)
    row, col = np.nonzero(adj)
    x = adata[:, adata.var["highly_variable"]].X
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    return Data(edge_index=torch.LongTensor(np.array([row, col])), x=torch.FloatTensor(x))


def mclust_labels(emb, n_clusters, seed, model_name="EEE"):
    """Gaussian mixture via R mclust; requires rpy2 and the R package mclust."""
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    ro.r("suppressMessages(library(mclust))")
    ro.globalenv["emb_r"] = np.ascontiguousarray(np.asarray(emb, dtype=np.float64))
    ro.r("set.seed(%d)" % int(seed))
    ro.r('res <- Mclust(as.matrix(emb_r), G=%d, modelNames="%s")' % (int(n_clusters), model_name))
    if bool(ro.r("is.null(res)")[0]):
        raise RuntimeError("Mclust returned NULL (G=%d)" % n_clusters)
    return np.array(ro.r("res$classification")).astype(int)


INVALID_LABELS = {"nan", "na", "none", ""}


def run_spagvae(adata, n_clusters, rad, seed=0, n_hvg=3000, refine=True,
                gt_key=None, device=None, **train_kw):
    """Preprocess, train, cluster (Mclust seed = training seed) and refine.

    When gt_key is given, spots without an annotation are removed before
    preprocessing, and ARI is computed on the remaining spots.
    Returns (labels, embedding, ari); ari is None when gt_key is not given.
    Extra keyword arguments are passed to train_spagvae.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gt = None
    if gt_key is not None:
        g = adata.obs[gt_key].astype(str)
        adata = adata[~g.str.strip().str.lower().isin(INVALID_LABELS).values].copy()
        gt = adata.obs[gt_key].astype(str).values
    a = preprocess(adata, n_hvg=n_hvg)
    data = build_graph(a, rad)
    set_seed(seed)
    _, emb = train_spagvae(data, a, device=device, **train_kw)
    emb = np.asarray(emb)
    labels = mclust_labels(emb, n_clusters, seed=seed)
    if refine:
        labels = spatial_label_refine(labels, coords=np.asarray(a.obsm["spatial"]),
                                      rad_cutoff=rad, n_iter=2)
    ari = adjusted_rand_score(gt, labels) if gt is not None else None
    return np.asarray(labels), emb, ari
