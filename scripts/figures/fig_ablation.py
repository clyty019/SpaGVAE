#!/usr/bin/env python
"""
Two-panel enrichment figure for the SpaGVAE paper:
  (a) component ablation  -- mean ARI (+/- SE over 12 section means) for
      Full and each leave-one-out variant.
  (b) alpha sensitivity   -- mean ARI and mean CV vs diffusion strength alpha.

Reads:
  outputs/spagvae_improve/results_gpu*.csv  (full a0.5; a0.3/a1.0; raw+refined)
  outputs/spagvae_ablation/results_*.csv     (no_prior/no_swa/no_anneal; a0.1/a2.0)

Writes:  overleaf/bmc-bioinformatics/figures/fig_ablation.{pdf,png}
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
           '151671', '151672', '151673', '151674', '151675', '151676']
IMPROVE = 'outputs/spagvae_improve'
ABL = 'outputs/spagvae_ablation'
BETA = 'outputs/spagvae_beta05'   # alpha=0.5 beta_max in {0.01, 0.10}; b0.05 lives in IMPROVE
OUT = 'overleaf/bmc-bioinformatics/figures/fig_ablation'


def load(dir_, pattern):
    fs = glob.glob(os.path.join(dir_, pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df['sample'] = df['sample'].astype(str)
    return df


def section_means(df, cfg, col='ari_refined'):
    sub = df[df['config'] == cfg]
    out = []
    for s in SAMPLES:
        a = sub[sub['sample'] == s][col].dropna().values
        if len(a):
            out.append(a.mean())
    return np.array(out)


def section_cv(df, cfg, col='ari_refined'):
    sub = df[df['config'] == cfg]
    out = []
    for s in SAMPLES:
        a = sub[sub['sample'] == s][col].dropna().values
        if len(a) and a.mean() > 0:
            out.append(a.std() / a.mean() * 100)
    return np.array(out)


def main():
    imp = load(IMPROVE, 'results_gpu*.csv')
    abl = load(ABL, 'results_*.csv')
    beta = load(BETA, 'results_gpu*.csv')
    full = 'a0.5_b0.05_swa_mc2020'

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 13,
        'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'savefig.dpi': 300,
    })
    # Morandi accents (consistent with the method/domain palettes)
    C_ACCENT, C_DAMAGE = '#5F8575', '#A85F68'   # SpaGVAE sage-teal; "damage" rose
    C_SLATE, C_SAGE, C_SAND = '#9AAAB8', '#A7B59A', '#C2B280'

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    def plot_sweep(ax, xs, ari, cv, xlabel, title, default_x):
        """ARI (blue, left axis) and CV (red, right axis) vs a hyperparameter;
        shared y-ranges so the relative flatness/trend is honest."""
        c_ari, c_cv = '#4F6D88', '#A85F68'
        pos = list(range(len(xs)))                      # even (categorical) spacing
        ax.plot(pos, ari, 'o-', color=c_ari, lw=2.6, ms=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean ARI', color=c_ari)
        ax.tick_params(axis='y', labelcolor=c_ari)
        ax.set_xticks(pos); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.set_ylim(0.50, 0.56)
        if default_x in list(xs):
            ax.axvline(list(xs).index(default_x), color='gray', ls=':', lw=1.3)
        ax2 = ax.twinx()
        ax2.plot(pos, cv, 's--', color=c_cv, lw=2.6, ms=9)
        ax2.set_ylabel('Mean CV (%)', color=c_cv)
        ax2.tick_params(axis='y', labelcolor=c_cv)
        ax2.set_ylim(7.0, 11.5)
        ax.set_title(title, fontsize=14, loc='left', fontweight='bold')
        ax.spines[['top']].set_visible(False); ax2.spines[['top']].set_visible(False)

    # ---------- Panel A: beta_max sensitivity (at alpha=0.5), from data ----------
    beta_pts = []
    for b, cfg, src in [(0.01, 'a0.5_b0.01_swa_mc2020', beta),
                        (0.05, full,                    imp),
                        (0.10, 'a0.5_b0.1_swa_mc2020',  beta)]:
        m = section_means(src, cfg); cv = section_cv(src, cfg)
        if len(m):
            beta_pts.append((b, m.mean(), cv.mean()))
    beta_pts.sort()
    if beta_pts:
        plot_sweep(axA, [p[0] for p in beta_pts], [p[1] for p in beta_pts],
                   [p[2] for p in beta_pts],
                   r'KL weight $\beta_{\max}$', r'(a) KL-weight sensitivity', 0.05)
    print('beta sweep (beta, ARI, CV):', [(b, round(m, 4), round(c, 1)) for b, m, c in beta_pts])

    # ---------- Panel B: alpha (diffusion strength) sensitivity ----------
    alpha_pts = []
    for a, cfg in [(0.3, 'a0.3_b0.05_swa_mc2020'),
                   (0.5, full),
                   (1.0, 'a1.0_b0.05_swa_mc2020')]:
        m = section_means(imp, cfg); cv = section_cv(imp, cfg)
        if len(m):
            alpha_pts.append((a, m.mean(), cv.mean()))
    for a, cfg in [(0.1, 'a0.1'), (2.0, 'a2.0')]:
        m = section_means(abl, cfg); cv = section_cv(abl, cfg)
        if len(m):
            alpha_pts.append((a, m.mean(), cv.mean()))
    alpha_pts.sort()
    if alpha_pts:
        xs = [p[0] for p in alpha_pts]
        plot_sweep(axB, xs, [p[1] for p in alpha_pts], [p[2] for p in alpha_pts],
                   r'Diffusion strength $\alpha$', r'(b) Diffusion-strength sensitivity', 0.5)

    plt.tight_layout()
    plt.savefig(OUT + '.pdf', bbox_inches='tight')
    plt.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('Saved', OUT + '.pdf')
    print('alpha sweep (alpha, ARI, CV):', [(a, round(m, 4), round(c, 1)) for a, m, c in alpha_pts])


if __name__ == '__main__':
    main()
