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
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structured Prior
# ---------------------------------------------------------------------------

def compute_spatial_prior(adata, n_pca=30, alpha=2.0, rad_cutoff=150):
    """
    Compute the structured prior mean via graph diffusion of PCA embeddings.

    mu_spatial = (I + alpha * L_norm)^{-1} PCA(X)

    This is deterministic and shared across all training seeds.

    Args:
        adata:      Preprocessed AnnData with spatial coords and HVGs.
        n_pca:      PCA dimensions.
        alpha:      Diffusion strength. Larger = more spatial smoothing.
        rad_cutoff: Spatial neighbor radius.

    Returns:
        mu_spatial: numpy array [N, n_pca].
    """
    # PCA
    hvg = adata[:, adata.var['highly_variable']]
    X = hvg.X.toarray() if hasattr(hvg.X, 'toarray') else np.array(hvg.X)
    z_pca = PCA(n_components=n_pca, random_state=0).fit_transform(X)

    # Spatial graph
    coor = adata.obsm['spatial']
    dist = cdist(coor, coor)
    adj = (dist < rad_cutoff).astype(float)
    np.fill_diagonal(adj, 0)

    # Normalized Laplacian
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Diffusion
    A = np.eye(L_norm.shape[0]) + alpha * L_norm
    mu_spatial = np.linalg.solve(A, z_pca)

    return mu_spatial.astype(np.float32)


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
# Training
# ---------------------------------------------------------------------------

def spatial_label_refine(labels, adata, rad_cutoff=150, n_iter=1):
    """
    Spatial label refinement via majority vote over spatial neighbors.

    For each spot, replace its label with the most common label among its
    spatial neighbors (including itself). Iterate n_iter times.

    Args:
        labels:     numpy array [N] of cluster labels.
        adata:      AnnData with spatial coordinates.
        rad_cutoff: Spatial neighbor radius.
        n_iter:     Number of refinement iterations.

    Returns:
        refined: numpy array [N] of refined labels.
    """
    coor = adata.obsm['spatial']
    dist = cdist(coor, coor)
    adj = dist < rad_cutoff
    np.fill_diagonal(adj, True)

    refined = labels.copy()
    for _ in range(n_iter):
        new_labels = refined.copy()
        for i in range(len(refined)):
            neighbor_labels = refined[adj[i]]
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            new_labels[i] = vals[counts.argmax()]
        refined = new_labels
    return refined


def train_spagvae(
    data,
    adata,
    n_epochs=800,
    lr=1e-3,
    weight_decay=1e-4,
    beta=0.001,
    beta_warmup=200,
    beta_max=0.01,
    beta_ramp=400,
    diffusion_alpha=2.0,
    latent_dim=30,
    n_pca=30,
    swa_start=None,
    structured_prior=True,
    prior_rad=150,
    device=None,
):
    """
    Train SpaGVAE with structured spatial prior.

    The beta parameter is ramped linearly from 0 to beta_max during
    [beta_warmup, beta_ramp] epochs, following the beta-VAE annealing
    strategy to prevent posterior collapse.

    Args:
        data:            PyG Data object.
        adata:           AnnData (for computing spatial prior).
        n_epochs:        Training epochs.
        lr:              Learning rate.
        weight_decay:    L2 regularization.
        beta:            Initial beta (usually 0 or very small).
        beta_warmup:     Epoch to start KL annealing.
        beta_max:        Maximum beta value.
        beta_ramp:       Epoch to reach beta_max.
        diffusion_alpha: Graph diffusion strength for prior.
        latent_dim:      Latent dimension d_z.
        n_pca:           PCA dimensions for the prior (must equal latent_dim).
        device:          torch.device.

    Returns:
        model:     Trained SpaGVAE.
        embedding: numpy array [N, latent_dim] (posterior mean at eval).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = data.to(device)
    in_dim = data.x.shape[1]

    # Compute structured prior (deterministic, shared across seeds).
    # If structured_prior=False, fall back to the standard N(0, I) prior
    # (mu_prior = 0) -- used for the ablation that isolates the contribution
    # of the spatially-structured prior.
    if structured_prior:
        mu_prior_np = compute_spatial_prior(adata, n_pca=latent_dim, alpha=diffusion_alpha,
                                            rad_cutoff=prior_rad)
        mu_prior = torch.tensor(mu_prior_np, dtype=torch.float32).to(device)
    else:
        mu_prior = torch.zeros((data.x.shape[0], latent_dim),
                               dtype=torch.float32, device=device)

    model = SpaGVAE(in_dim=in_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # SWA: maintain a running average of weights starting from swa_start
    swa_weights = None
    swa_count = 0

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        mu, logvar, z, x_recon = model(data.x, data.edge_index)

        # Reconstruction loss
        l_recon = F.mse_loss(x_recon, data.x)

        # KL with structured prior (annealed)
        if epoch < beta_warmup:
            current_beta = 0.0
        elif epoch < beta_ramp:
            current_beta = beta_max * (epoch - beta_warmup) / (beta_ramp - beta_warmup)
        else:
            current_beta = beta_max

        l_kl = kl_divergence_structured(mu, logvar, mu_prior)
        loss = l_recon + current_beta * l_kl

        loss.backward()
        optimizer.step()

        # SWA update
        if swa_start is not None and epoch >= swa_start:
            swa_count += 1
            if swa_weights is None:
                swa_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k in swa_weights:
                    swa_weights[k] = swa_weights[k] + (model.state_dict()[k].detach() - swa_weights[k]) / swa_count

    # Load SWA weights if used
    if swa_weights is not None:
        model.load_state_dict(swa_weights)

    # Extract posterior mean as embedding (deterministic at eval)
    model.eval()
    with torch.no_grad():
        mu, logvar, z, _ = model(data.x, data.edge_index)
    return model, mu.cpu().numpy()
