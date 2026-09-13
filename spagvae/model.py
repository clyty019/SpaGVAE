"""
SpaGVAE: Spatial Graph Variational Autoencoder with Structured Prior.

A variational graph autoencoder for spatial domain identification that uses
a spatially-informed prior instead of the standard N(0,I). The prior mean
is derived from graph diffusion of PCA embeddings, providing a deterministic
spatial reference that constrains the learned representation.

Theoretical framework:
  ELBO = E_q[log p(x|z)] - beta * KL(q(z|x,G) || p(z))

  where:
    q(z|x,G) = N(mu_encoder(x,G), diag(sigma_encoder(x,G)^2))  [learned]
    p(z)      = N(mu_spatial, I)                                  [structured prior]
    mu_spatial = (I + alpha * L_norm)^{-1} PCA(X)                [graph diffusion]

The structured prior serves two purposes:
  1. Anchoring: All seeds share the same prior mean, constraining the solution
     space and reducing seed sensitivity.
  2. Adaptivity: The encoder learns per-spot variance sigma_i, allowing it to
     deviate from the prior where the data demands it (e.g., domain boundaries).

This is theoretically grounded in the Variational Information Bottleneck:
  - KL term compresses the representation (removes seed-specific noise)
  - Reconstruction term preserves task-relevant information
  - Structured prior injects spatial inductive bias

=============================================================================
REVISION CHANGELOG (sparse / scalable rewrite)  -- behaviour changes vs legacy
=============================================================================
compute_spatial_prior
  * Sparse by default.  The N x N dense Laplacian + np.linalg.solve (O(N^2)
    memory, O(N^3) time) is replaced by a scipy.sparse normalised Laplacian and
    a sparse direct solve (splu, factorised once and reused for all PCs) for
    N <= 200k, or conjugate gradient per PC above that.  Result agrees with the
    legacy dense solve to ~1e-6 (float64 solve, returned as float32).  The
    legacy dense path is kept as ``sparse=False`` (and ``_compute_spatial_prior_dense``)
    for testing / tiny data.
  * Graph construction: radius graph via sklearn NearestNeighbors (strict
    ``dist < rad_cutoff``, self excluded -- identical adjacency to the legacy
    ``cdist(...) < rad`` rule) OR an explicit ``edge_index`` (the GAT graph),
    symmetrised with self-loops dropped.  Isolated nodes get a Laplacian row
    L_ii = 1 exactly as in the legacy code (=> mu_i = z_i / (1 + alpha) for the
    implicit mode).
  * New keyword-only inputs: ``X`` / ``coords`` (AnnData-free usage), ``z_pca``
    (precomputed PCA; skips the dense HVG matrix + PCA), ``pca_seed``.
  * New diffusion ``mode``: "implicit" (legacy, one backward-Euler step),
    "cn" (n_steps Crank-Nicolson steps of size alpha/n_steps), "explicit"
    (n_steps forward-Euler steps), "heat" (exact heat kernel
    expm_multiply(-alpha L, z)).  All modes coincide to O(alpha^2) for small
    alpha; "implicit" is the default and reproduces the legacy prior.
  * DEFAULTS CHANGED: ``alpha`` 2.0 -> 0.5 (the canonical value used in every
    experiment / the paper), ``rad_cutoff`` 150 -> None (must be given unless
    ``edge_index`` is supplied; legacy callers always passed it explicitly).
    Positional order (adata, n_pca, alpha, rad_cutoff) is preserved.
  * ``adata`` may be None when ``X``/``z_pca`` and ``coords``/``edge_index`` are
    given.  Likewise train_spagvae(data, adata=None, ...) computes the prior PCA
    on ``data.x`` with the encoder graph.

spatial_label_refine
  * Sparse CSR majority vote (bincount-based), identical output to the legacy
    dense version including tie-breaking (smallest label wins, as np.unique +
    argmax did).  Accepts ``coords`` / ``edge_index`` as alternatives to
    ``adata``.  Default ``rad_cutoff`` 150 -> None (explicit).

train_spagvae
  * ``prior_graph="encoder"`` (NEW DEFAULT): the prior Laplacian is built from
    ``data.edge_index`` (same graph as the GAT), so the prior no longer depends
    on a second, independent radius parameter.  ``prior_graph="radius"`` is the
    legacy behaviour (``prior_rad``, default 150 when None).  On DLPFC with the
    STAGATE rad-150 graph both are identical.
  * ``mu_prior`` may be passed precomputed (cache across seeds).
    ``return_prior=True`` returns it as third output.
  * ``n_pca`` default 30 -> None (= latent_dim).  n_pca < latent_dim now pads
    the prior mean with zeros (N(0,1) on the extra dims) instead of crashing;
    n_pca > latent_dim raises ValueError.
  * ``batch_size`` (new): None -> full-batch training, identical to legacy.
    int -> mini-batch training with torch_geometric NeighborLoader (loss on
    seed nodes only) and exact layer-wise inference (``SpaGVAE.inference``).
  * ``beta_max`` default 0.01 -> 0.05 and ``diffusion_alpha`` 2.0 -> 0.5
    (canonical values used by all experiments).  ``beta_warmup=beta_ramp=0``
    -> constant beta_max; ``beta_max=0`` -> no KL.
  * The unused legacy ``beta`` argument is still accepted (ignored) for
    backward compatibility of old call sites.
  * ``prior_mode`` / ``prior_steps`` are forwarded to compute_spatial_prior.
  * ``device`` may be a string; ``verbose`` prints the loss every 100 epochs.

Peak-memory-relevant design choices
  * Prior: O(E) sparse Laplacian; splu factor (fill-in is small for planar
    spatial graphs); solve all n_pca right-hand sides in one call.  The only
    dense N x d objects are the PCA input (N x n_HVG, legacy behaviour -- pass
    ``z_pca`` to avoid it) and the N x n_pca result.
  * Mini-batch: ``data.x`` and ``mu_prior`` stay on CPU; only sampled sub-graphs
    and the corresponding prior rows are moved to the device.  Inference is
    layer-wise with ``num_neighbors=[-1]`` so each node's output is computed
    exactly once from its full 1-hop neighbourhood (no neighbour explosion),
    activations are stored on CPU, the returned embedding is exact and
    deterministic.
  * Full-batch: unchanged (whole graph on device) -- use ``batch_size`` for
    large N.
"""

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv

__all__ = [
    "SpaGVAE", "SpaGVAEEncoder", "SpaGVAEDecoder",
    "compute_spatial_prior", "kl_divergence_structured",
    "spatial_label_refine", "train_spagvae", "set_seed",
]

# N above which the implicit solve switches from splu (direct) to CG.
_SPLU_MAX_N = 200_000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Seed python / numpy / torch (CPU + CUDA) and make cuDNN deterministic."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_numpy_edge_index(edge_index):
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return edge_index.astype(np.int64, copy=False)


def _radius_adjacency(coords, rad_cutoff):
    """Binary symmetric CSR adjacency, strict ``dist < rad_cutoff``, no self loops.

    Identical to the legacy ``cdist(coor, coor) < rad_cutoff`` with the diagonal
    zeroed, but built in O(N log N + E) via a ball tree.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    nn_ = NearestNeighbors(radius=float(rad_cutoff)).fit(coords)
    A = nn_.radius_neighbors_graph(coords, mode="distance")  # includes self (d=0) and d<=rad
    A = A.tocoo()
    keep = (A.data > 0) & (A.data < rad_cutoff)
    rows, cols = A.row[keep], A.col[keep]
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)  # radius graph is symmetric already; be safe
    adj.data[:] = 1.0
    return adj


def _edge_index_adjacency(edge_index, n):
    """Binary symmetric CSR adjacency from an edge_index, self loops removed."""
    ei = _as_numpy_edge_index(edge_index)
    mask = ei[0] != ei[1]
    rows, cols = ei[0][mask], ei[1][mask]
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)
    adj.data[:] = 1.0
    return adj


def _build_adjacency(n, coords=None, rad_cutoff=None, edge_index=None):
    if edge_index is not None:
        return _edge_index_adjacency(edge_index, n)
    if coords is None or rad_cutoff is None:
        raise ValueError("Need either edge_index, or coords (adata.obsm['spatial']) "
                         "together with rad_cutoff to build the spatial graph.")
    return _radius_adjacency(coords, rad_cutoff)


def _normalized_laplacian(adj):
    """L = I - D^-1/2 A D^-1/2 (CSR, float64). Isolated nodes -> L_ii = 1."""
    n = adj.shape[0]
    d = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    Dm = sp.diags(d_inv_sqrt)
    return (sp.eye(n, format="csr") - Dm @ adj @ Dm).tocsr()


def _solve_spd(M, B):
    """Solve M X = B for SPD sparse M and dense B [N, k]: splu if small, CG otherwise."""
    n = M.shape[0]
    M = M.tocsc()
    if n <= _SPLU_MAX_N:
        lu = spla.splu(M)
        return lu.solve(np.ascontiguousarray(B, dtype=np.float64))
    X = np.empty_like(B, dtype=np.float64)
    for j in range(B.shape[1]):
        x, info = spla.cg(M, B[:, j], x0=B[:, j], tol=1e-10, atol=0.0, maxiter=5000)
        if info != 0:
            warnings.warn("CG did not converge for PC %d (info=%d)" % (j, info))
        X[:, j] = x
    return X


def _pca_features(adata=None, X=None, n_pca=30, pca_seed=0):
    """Legacy PCA: sklearn PCA(n_pca, random_state=pca_seed) on the HVG matrix."""
    if X is None:
        if adata is None:
            raise ValueError("Provide adata, X, or z_pca.")
        if "highly_variable" in adata.var:
            hvg = adata[:, adata.var["highly_variable"]]
        else:
            hvg = adata
        X = hvg.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return PCA(n_components=int(min(n_pca, X.shape[1] - 1, X.shape[0] - 1)), random_state=pca_seed).fit_transform(X)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _layerwise_propagate(conv, x_all, edge_index, device, batch_size, act=None):
    """Apply one GATConv to every node exactly, in CPU-resident chunks.

    For each chunk of target nodes the full 1-hop neighbourhood is gathered
    (NeighborLoader num_neighbors=[-1]); the layer output is kept only for the
    seed nodes, so every node is computed exactly once from all its neighbours.
    """
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    data = Data(x=x_all, edge_index=edge_index)
    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size, shuffle=False)
    out = None
    for batch in loader:
        bs = batch.batch_size
        h = conv(batch.x.to(device), batch.edge_index.to(device))[:bs]
        if act is not None:
            h = act(h)
        h = h.cpu()
        if out is None:
            out = torch.empty((x_all.shape[0], h.shape[1]), dtype=h.dtype)
        out[batch.n_id[:bs]] = h
    return out


class SpaGVAEEncoder(nn.Module):
    """
    Variational GAT encoder. Outputs mean and log-variance.
    """
    def __init__(self, in_dim, hidden_dim=512, latent_dim=30):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=0.0)
        self.conv_mu = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=0.0)
        self.conv_logvar = GATConv(hidden_dim, latent_dim, heads=1, concat=False, dropout=0.0)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar

    @torch.no_grad()
    def inference(self, x_cpu, edge_index, device, batch_size=4096, return_logvar=False):
        """Exact layer-wise inference of mu (and logvar) for large graphs."""
        h = _layerwise_propagate(self.conv1, x_cpu, edge_index, device, batch_size, act=F.elu)
        mu = _layerwise_propagate(self.conv_mu, h, edge_index, device, batch_size)
        if not return_logvar:
            return mu
        logvar = _layerwise_propagate(self.conv_logvar, h, edge_index, device, batch_size)
        return mu, logvar


class SpaGVAEDecoder(nn.Module):
    """GAT decoder."""
    def __init__(self, latent_dim=30, hidden_dim=512, out_dim=3000):
        super().__init__()
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=1, concat=False, dropout=0.0)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=0.0)

    def forward(self, z, edge_index):
        h = F.elu(self.conv1(z, edge_index))
        return self.conv2(h, edge_index)


class SpaGVAE(nn.Module):
    """
    Spatial Graph Variational Autoencoder with Structured Prior.
    """
    def __init__(self, in_dim, hidden_dim=512, latent_dim=30):
        super().__init__()
        self.encoder = SpaGVAEEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = SpaGVAEDecoder(latent_dim, hidden_dim, in_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # deterministic at test time

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, edge_index)
        return mu, logvar, z, x_recon

    @torch.no_grad()
    def inference(self, x_cpu, edge_index, device=None, batch_size=None):
        """Posterior mean for all nodes.

        ``batch_size=None``: full-batch forward on ``device``.  Otherwise exact
        layer-wise propagation with CPU-resident features (deterministic,
        identical to the full-batch result up to float rounding).
        Returns a CPU float tensor [N, latent_dim].
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        if batch_size is None:
            mu, _ = self.encoder(x_cpu.to(device), edge_index.to(device))
            return mu.cpu()
        return self.encoder.inference(x_cpu.cpu(), edge_index.cpu(), device, batch_size)


# ---------------------------------------------------------------------------
# Structured Prior
# ---------------------------------------------------------------------------

def compute_spatial_prior(adata=None, n_pca=30, alpha=0.5, rad_cutoff=None, *,
                          X=None, coords=None, z_pca=None, edge_index=None,
                          mode="implicit", n_steps=1, sparse=True, pca_seed=0):
    """
    Compute the structured prior mean via graph diffusion of PCA embeddings.

    mode="implicit" (default, legacy):
        mu_spatial = (I + alpha * L_norm)^{-1} PCA(X)
    mode="cn":       n_steps Crank-Nicolson steps, step h = alpha / n_steps
                     (I + h/2 L) z_{k+1} = (I - h/2 L) z_k
    mode="explicit": n_steps forward-Euler steps  z <- (I - h L) z
    mode="heat":     exact heat kernel  expm(-alpha L) z

    This is deterministic and shared across all training seeds.

    Args:
        adata:      Preprocessed AnnData with spatial coords (obsm['spatial'])
                    and HVGs (var['highly_variable']).  May be None if X/z_pca
                    and coords/edge_index are given.
        n_pca:      PCA dimensions.
        alpha:      Diffusion strength (total diffusion time). Larger = smoother.
        rad_cutoff: Spatial neighbour radius (strict <).  Ignored if edge_index
                    is given.
        X:          [N, G] expression matrix (dense or sparse) used instead of
                    adata's HVG matrix.
        coords:     [N, 2] spatial coordinates (defaults to adata.obsm['spatial']).
        z_pca:      [N, n_pca] precomputed PCA scores (skips PCA).
        edge_index: [2, E] graph (np.ndarray or torch tensor) used instead of the
                    radius graph; symmetrised, self loops removed.
        mode:       "implicit" | "cn" | "explicit" | "heat".
        n_steps:    number of sub-steps for "cn" / "explicit".
        sparse:     False -> legacy dense numpy path (testing only, O(N^2) memory).
        pca_seed:   random_state of sklearn PCA.

    Returns:
        mu_spatial: numpy float32 array [N, n_pca].
    """
    # --- features -----------------------------------------------------------
    if z_pca is None:
        z_pca = _pca_features(adata, X, n_pca, pca_seed)
    z_pca = np.asarray(z_pca, dtype=np.float64)
    n = z_pca.shape[0]
    if n_pca is not None and z_pca.shape[1] != n_pca:
        n_pca = int(z_pca.shape[1])   # PCA was clamped to the data (tiny gene panels)

    if alpha == 0:
        return z_pca.astype(np.float32)

    # --- graph --------------------------------------------------------------
    if coords is None and adata is not None and edge_index is None:
        coords = adata.obsm["spatial"]

    if not sparse:
        return _compute_spatial_prior_dense(z_pca, coords=coords, rad_cutoff=rad_cutoff,
                                            edge_index=edge_index, alpha=alpha,
                                            mode=mode, n_steps=n_steps)

    adj = _build_adjacency(n, coords=coords, rad_cutoff=rad_cutoff, edge_index=edge_index)
    L = _normalized_laplacian(adj)
    I = sp.eye(n, format="csr")

    # --- diffusion ----------------------------------------------------------
    if mode == "implicit":
        mu = _solve_spd(I + alpha * L, z_pca)
    elif mode == "cn":
        h = alpha / float(n_steps)
        A_minus = (I - 0.5 * h * L).tocsr()
        lu = spla.splu((I + 0.5 * h * L).tocsc()) if n <= _SPLU_MAX_N else None
        mu = z_pca
        for _ in range(n_steps):
            rhs = A_minus @ mu
            mu = lu.solve(np.ascontiguousarray(rhs)) if lu is not None else _solve_spd(I + 0.5 * h * L, rhs)
    elif mode == "explicit":
        h = alpha / float(n_steps)
        A_step = (I - h * L).tocsr()
        mu = z_pca
        for _ in range(n_steps):
            mu = A_step @ mu
    elif mode == "heat":
        mu = spla.expm_multiply((-alpha * L).tocsc(), z_pca)
    else:
        raise ValueError("unknown mode %r (implicit|cn|explicit|heat)" % mode)

    return np.asarray(mu, dtype=np.float32)


def _compute_spatial_prior_dense(z_pca, coords=None, rad_cutoff=None, edge_index=None,
                                 alpha=0.5, mode="implicit", n_steps=1):
    """Legacy dense reference implementation (O(N^2) memory). Used by tests."""
    from scipy.spatial.distance import cdist
    from scipy.linalg import expm
    z_pca = np.asarray(z_pca, dtype=np.float64)
    n = z_pca.shape[0]
    if edge_index is not None:
        adj = _edge_index_adjacency(edge_index, n).toarray()
    else:
        coords = np.asarray(coords, dtype=np.float64)
        dist = cdist(coords, coords)
        adj = (dist < rad_cutoff).astype(float)
        np.fill_diagonal(adj, 0)
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    I = np.eye(n)
    if mode == "implicit":
        mu = np.linalg.solve(I + alpha * L, z_pca)
    elif mode == "cn":
        h = alpha / float(n_steps)
        mu = z_pca
        for _ in range(n_steps):
            mu = np.linalg.solve(I + 0.5 * h * L, (I - 0.5 * h * L) @ mu)
    elif mode == "explicit":
        h = alpha / float(n_steps)
        mu = z_pca
        for _ in range(n_steps):
            mu = (I - h * L) @ mu
    elif mode == "heat":
        mu = expm(-alpha * L) @ z_pca
    else:
        raise ValueError("unknown mode %r" % mode)
    return mu.astype(np.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def kl_divergence_structured(mu, logvar, mu_prior):
    """
    KL(q(z|x) || p(z)) where q = N(mu, diag(sigma^2)), p = N(mu_prior, I).

    KL = 0.5 * sum( sigma^2 + (mu - mu_prior)^2 - 1 - log(sigma^2) )

    This is the standard VAE KL but with a non-zero prior mean.
    """
    return 0.5 * torch.sum(
        logvar.exp() + (mu - mu_prior) ** 2 - 1 - logvar,
        dim=-1
    ).mean()


# ---------------------------------------------------------------------------
# Spatial label refinement
# ---------------------------------------------------------------------------

def spatial_label_refine(labels, adata=None, rad_cutoff=None, n_iter=1, *,
                         coords=None, edge_index=None):
    """
    Spatial label refinement via majority vote over spatial neighbors.

    For each spot, replace its label with the most common label among its
    spatial neighbors (including itself). Ties -> smallest label (same as the
    legacy np.unique/argmax implementation). Iterate n_iter times.

    Args:
        labels:     numpy array [N] of cluster labels (any hashable/int dtype).
        adata:      AnnData with spatial coordinates (obsm['spatial']).
        rad_cutoff: Spatial neighbor radius (strict <). Ignored if edge_index.
        n_iter:     Number of refinement iterations.
        coords:     [N, 2] coordinates (alternative to adata).
        edge_index: [2, E] graph (alternative to the radius graph).

    Returns:
        refined: numpy array [N] of refined labels (same dtype as input).
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    if coords is None and adata is not None and edge_index is None:
        coords = adata.obsm["spatial"]
    adj = _build_adjacency(n, coords=coords, rad_cutoff=rad_cutoff, edge_index=edge_index)
    adj = (adj + sp.eye(n, format="csr")).tocsr()  # include self
    adj.data[:] = 1.0

    # encode labels as 0..K-1 (np.unique sorts -> tie-break = smallest label)
    uniq, codes = np.unique(labels, return_inverse=True)
    K = len(uniq)
    rows = np.repeat(np.arange(n), np.diff(adj.indptr))
    cols = adj.indices

    refined = codes.copy()
    for _ in range(n_iter):
        # counts[i, k] = number of neighbours of i (incl. self) with label k
        counts = sp.coo_matrix((np.ones(len(rows)), (rows, refined[cols])), shape=(n, K)).toarray()
        refined = counts.argmax(axis=1)  # first max -> smallest label on ties
    return uniq[refined]


def _spatial_label_refine_dense(labels, coords, rad_cutoff=150, n_iter=1):
    """Legacy dense reference implementation (used by tests)."""
    from scipy.spatial.distance import cdist
    dist = cdist(coords, coords)
    adj = dist < rad_cutoff
    np.fill_diagonal(adj, True)
    refined = np.asarray(labels).copy()
    for _ in range(n_iter):
        new_labels = refined.copy()
        for i in range(len(refined)):
            neighbor_labels = refined[adj[i]]
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            new_labels[i] = vals[counts.argmax()]
        refined = new_labels
    return refined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _beta_at(epoch, beta_max, beta_warmup, beta_ramp):
    if beta_max == 0:
        return 0.0
    if beta_warmup == 0 and beta_ramp == 0:
        return float(beta_max)
    if epoch < beta_warmup:
        return 0.0
    if epoch < beta_ramp:
        return beta_max * (epoch - beta_warmup) / float(beta_ramp - beta_warmup)
    return float(beta_max)


def _swa_update(model, swa_state, swa_count):
    if swa_state is None:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd = model.state_dict()
    for k in swa_state:
        swa_state[k] = swa_state[k] + (sd[k].detach() - swa_state[k]) / swa_count
    return swa_state


def train_spagvae(
    data,
    adata,
    n_epochs=800,
    lr=1e-3,
    weight_decay=1e-4,
    beta_max=0.05,
    beta_warmup=200,
    beta_ramp=400,
    diffusion_alpha=0.5,
    latent_dim=30,
    n_pca=None,
    hidden_dim=512,
    swa_start=None,
    structured_prior=True,
    prior_graph="encoder",
    prior_rad=None,
    prior_mode="implicit",
    prior_steps=1,
    mu_prior=None,
    batch_size=None,
    num_neighbors=(15, 10),
    max_steps=None,
    grad_clip=5.0,
    device=None,
    verbose=False,
    return_prior=False,
    beta=None,  # legacy, ignored
):
    """
    Train SpaGVAE with structured spatial prior.

    The KL weight beta is ramped linearly from 0 to beta_max during
    [beta_warmup, beta_ramp] epochs (beta-VAE annealing); set
    beta_warmup=beta_ramp=0 for a constant beta_max and beta_max=0 to drop
    the KL term entirely.

    Args:
        data:            PyG Data object (x [N, G] float, edge_index [2, E]).
        adata:           AnnData (for the prior: HVGs + obsm['spatial']).  May be
                         None: then the prior PCA is computed on data.x (the HVG
                         matrix) and prior_graph must be "encoder" (or mu_prior /
                         structured_prior=False is used).
        n_epochs:        Training epochs (full passes over the data).
        lr:              Learning rate.
        weight_decay:    L2 regularization.
        beta_max:        Maximum KL weight.
        beta_warmup:     Epoch to start KL annealing.
        beta_ramp:       Epoch to reach beta_max.
        diffusion_alpha: Graph diffusion strength for the prior.
        latent_dim:      Latent dimension d_z.
        n_pca:           PCA dims of the prior (None -> latent_dim; < latent_dim
                         -> zero-padded; > latent_dim -> ValueError).
        hidden_dim:      GAT hidden width.
        swa_start:       Epoch from which weights are averaged (None -> no SWA).
        structured_prior: False -> standard N(0, I) prior (ablation).
        prior_graph:     "encoder" -> prior Laplacian from data.edge_index (default);
                         "radius" -> radius graph from adata.obsm['spatial'] with
                         prior_rad (legacy; None -> 150).
        prior_rad:       Radius for prior_graph="radius".
        prior_mode:      Diffusion mode, see compute_spatial_prior.
        prior_steps:     Sub-steps for prior_mode in {"cn", "explicit"}.
        mu_prior:        Precomputed prior mean [N, n_pca or latent_dim]
                         (np.ndarray or tensor) -- skips the prior computation.
        batch_size:      None -> full-batch; int -> NeighborLoader mini-batches.
        num_neighbors:   Fan-out per layer for NeighborLoader (mini-batch only).
        max_steps:       Mini-batch only. Total optimizer-step budget; when set,
                         training stops after this many steps (possibly mid-epoch)
                         and the beta warm-up/ramp is expressed as the same
                         fractions of the step budget as beta_warmup/beta_ramp
                         are of n_epochs. Makes wall-clock ~independent of N.
        grad_clip:       Max gradient norm (mini-batch only; None disables).
        device:          torch.device or str; None -> cuda if available.
        verbose:         Print loss every 100 epochs.
        return_prior:    Also return the (padded) prior mean as numpy array.

    Returns:
        model:     Trained SpaGVAE (on ``device``).
        embedding: numpy array [N, latent_dim] (posterior mean at eval).
        [mu_prior: numpy array [N, latent_dim], if return_prior]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if n_pca is None:
        n_pca = latent_dim
    if n_pca > latent_dim:
        raise ValueError("n_pca (%d) must be <= latent_dim (%d)" % (n_pca, latent_dim))
    if prior_graph not in ("encoder", "radius"):
        raise ValueError("prior_graph must be 'encoder' or 'radius'")

    N, in_dim = data.x.shape

    # ---- structured prior (deterministic, shared across seeds) -------------
    # structured_prior=False -> standard N(0, I) prior (mu_prior = 0), used for
    # the ablation isolating the contribution of the spatial prior.
    if not structured_prior:
        mu_prior_np = np.zeros((N, latent_dim), dtype=np.float32)
    elif mu_prior is not None:
        mu_prior_np = mu_prior.detach().cpu().numpy() if torch.is_tensor(mu_prior) else np.asarray(mu_prior)
        mu_prior_np = mu_prior_np.astype(np.float32, copy=False)
    else:
        # adata=None -> PCA on data.x (which is the HVG matrix at every call site)
        X_prior = None if adata is not None else data.x.detach().cpu().numpy()
        if prior_graph == "encoder":
            mu_prior_np = compute_spatial_prior(adata, n_pca=n_pca, alpha=diffusion_alpha,
                                                X=X_prior, edge_index=data.edge_index,
                                                mode=prior_mode, n_steps=prior_steps)
        else:
            if adata is None:
                raise ValueError("prior_graph='radius' needs adata (obsm['spatial'])")
            rad = 150 if prior_rad is None else prior_rad
            mu_prior_np = compute_spatial_prior(adata, n_pca=n_pca, alpha=diffusion_alpha,
                                                X=X_prior, rad_cutoff=rad,
                                                mode=prior_mode, n_steps=prior_steps)
    if mu_prior_np.shape[0] != N:
        raise ValueError("mu_prior has %d rows, data has %d nodes" % (mu_prior_np.shape[0], N))
    if mu_prior_np.shape[1] > latent_dim:
        raise ValueError("mu_prior has %d dims > latent_dim %d" % (mu_prior_np.shape[1], latent_dim))
    if mu_prior_np.shape[1] < latent_dim:  # zero-pad: N(0,1) on the extra dims
        pad = np.zeros((N, latent_dim - mu_prior_np.shape[1]), dtype=np.float32)
        mu_prior_np = np.concatenate([mu_prior_np, pad], axis=1)
    mu_prior_t = torch.tensor(mu_prior_np, dtype=torch.float32)

    model = SpaGVAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    swa_state, swa_count = None, 0

    if batch_size is None:
        # ---------------- full-batch (legacy loop) --------------------------
        data = data.to(device)
        mu_prior_dev = mu_prior_t.to(device)
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mu, logvar, z, x_recon = model(data.x, data.edge_index)
            l_recon = F.mse_loss(x_recon, data.x)
            current_beta = _beta_at(epoch, beta_max, beta_warmup, beta_ramp)
            if current_beta > 0:
                l_kl = kl_divergence_structured(mu, logvar, mu_prior_dev)
                loss = l_recon + current_beta * l_kl
            else:
                loss = l_recon
            loss.backward()
            optimizer.step()
            if swa_start is not None and epoch >= swa_start:
                swa_count += 1
                swa_state = _swa_update(model, swa_state, swa_count)
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print("epoch %4d loss %.4f recon %.4f beta %.4f"
                      % (epoch, loss.item(), l_recon.item(), current_beta), flush=True)
        if swa_state is not None:
            model.load_state_dict(swa_state)
        model.eval()
        with torch.no_grad():
            mu, _, _, _ = model(data.x, data.edge_index)
        emb = mu.cpu().numpy()
    else:
        # ---------------- mini-batch (NeighborLoader) -----------------------
        from torch_geometric.loader import NeighborLoader
        x_cpu = data.x.cpu()
        ei_cpu = data.edge_index.cpu()
        from torch_geometric.data import Data
        data_cpu = Data(x=x_cpu, edge_index=ei_cpu)
        loader = NeighborLoader(data_cpu, num_neighbors=list(num_neighbors),
                                batch_size=int(batch_size), shuffle=True)
        model.train()
        steps_per_epoch = max(1, len(loader))
        if max_steps is not None:
            max_steps = int(max_steps)
            # beta schedule as fractions of the step budget (same fractions as epochs)
            f_w = beta_warmup / float(max(n_epochs, 1)); f_r = beta_ramp / float(max(n_epochs, 1))
            sw_step = int(f_w * max_steps); sr_step = int(f_r * max_steps)
            swa_step = None if swa_start is None else int(swa_start / float(max(n_epochs, 1)) * max_steps)
            n_epochs_eff = int(np.ceil(max_steps / float(steps_per_epoch)))
        else:
            n_epochs_eff = n_epochs
        step = 0
        n_skipped = 0
        done = False
        for epoch in range(n_epochs_eff):
            tot, nb = 0.0, 0
            for batch in loader:
                if max_steps is not None:
                    current_beta = _beta_at(step, beta_max, sw_step, sr_step)
                else:
                    current_beta = _beta_at(epoch, beta_max, beta_warmup, beta_ramp)
                bs = batch.batch_size
                optimizer.zero_grad()
                xb = batch.x.to(device)
                mu, logvar, z, x_recon = model(xb, batch.edge_index.to(device))
                l_recon = F.mse_loss(x_recon[:bs], xb[:bs])
                if current_beta > 0:
                    prior_b = mu_prior_t[batch.n_id[:bs]].to(device)
                    l_kl = kl_divergence_structured(mu[:bs], logvar[:bs], prior_b)
                    loss = l_recon + current_beta * l_kl
                else:
                    loss = l_recon
                if not torch.isfinite(loss):
                    n_skipped += 1          # NaN guard: skip the step, keep weights
                    optimizer.zero_grad()
                    step += 1
                    if max_steps is not None and step >= max_steps:
                        done = True; break
                    continue
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
                tot += loss.item(); nb += 1
                step += 1
                if max_steps is not None:
                    if swa_step is not None and step >= swa_step:
                        swa_count += 1
                        swa_state = _swa_update(model, swa_state, swa_count)
                    if step >= max_steps:
                        done = True; break
            if max_steps is None and swa_start is not None and epoch >= swa_start:
                swa_count += 1
                swa_state = _swa_update(model, swa_state, swa_count)
            if verbose and (epoch % 100 == 0 or epoch == n_epochs_eff - 1 or done):
                print("epoch %4d step %6d loss %.4f beta %.4f skipped %d" % (
                    epoch, step, tot / max(nb, 1), current_beta, n_skipped), flush=True)
            if done:
                break
        if n_skipped:
            print("[spagvae] warning: %d non-finite mini-batch steps skipped" % n_skipped, flush=True)
        if swa_state is not None:
            model.load_state_dict(swa_state)
        emb = model.inference(x_cpu, ei_cpu, device=device, batch_size=int(batch_size)).numpy()
        bad = ~np.isfinite(emb).all(axis=1)
        if bad.any():
            print("[spagvae] warning: %d non-finite embedding rows replaced by prior mean" % int(bad.sum()), flush=True)
            emb[bad] = mu_prior_np[bad]

    if return_prior:
        return model, emb, mu_prior_np
    return model, emb
