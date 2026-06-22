#!/usr/bin/env python
"""Compute all paper-table numbers at the matched 1000-seed protocol for the
six methods, plus per-section winners and Wilcoxon tests."""
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon

SAMPLES = ['151507','151508','151509','151510','151669','151670',
           '151671','151672','151673','151674','151675','151676']
BASE_ROOT = '/extra/zhanglab0/INDV/zihend1/StaR'

def load_method(method):
    """Return dict sample -> np.array of per-seed ARI."""
    out = {}
    if method == 'SpaGVAE':
        fs = glob.glob('outputs/spagvae_1000/results_*_c*.csv')
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(['sample','seed'])
        df['sample'] = df['sample'].astype(str)
        for s in SAMPLES:
            out[s] = df[df['sample']==s]['ari_refined'].dropna().values
    elif method == 'SpaGCN':
        fs = glob.glob('outputs/spagcn_1000/results_*_c*.csv')
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(['sample','seed'])
        df['sample'] = df['sample'].astype(str)
        for s in SAMPLES:
            out[s] = df[df['sample']==s]['ari'].dropna().values
    else:
        for s in SAMPLES:
            f = f'{BASE_ROOT}/{s}/{method}/ari_checkpoint.csv'
            out[s] = pd.read_csv(f)['ARI'].dropna().values
    return out

METHODS = ['STAGATE','GraphST','SpaceFlow','stCluster','SpaGCN','SpaGVAE']
data = {m: load_method(m) for m in METHODS}

def sec_mean(m): return np.array([data[m][s].mean() for s in SAMPLES])
def sec_std(m):  return np.array([data[m][s].std()  for s in SAMPLES])
def pooled(m):   return np.concatenate([data[m][s] for s in SAMPLES])

print('=== per-section mean (rows=section) ===')
hdr = 'sect    ' + ''.join(f'{m[:8]:>10}' for m in METHODS)
print(hdr)
win = {m:0 for m in METHODS}
for i,s in enumerate(SAMPLES):
    means = {m: data[m][s].mean() for m in METHODS}
    best = max(means, key=means.get); win[best]+=1
    print(f'{s} ' + ''.join(f'{means[m]:>10.3f}' for m in METHODS) + f'   best={best}')
print('win counts:', win)

print('\n=== per-section mean +/- std (for tab:per_sample) ===')
for i,s in enumerate(SAMPLES):
    cells=[]
    means={m:data[m][s].mean() for m in METHODS}
    best=max(means,key=means.get)
    for m in METHODS:
        mu,sd=data[m][s].mean(),data[m][s].std()
        cells.append((f'**{mu:.3f}**' if m==best else f'{mu:.3f}')+f'+/-{sd:.3f}')
    print(s, ' & '.join(cells))
print('MEAN row:', ' & '.join(f'{sec_mean(m).mean():.3f}' for m in METHODS))

print('\n=== pooled stats (for tab:main_results) ===')
print(f"{'method':10} {'mean':>7}{'median':>8}{'P5':>7}{'P95':>7}{'std':>7}")
for m in METHODS:
    a=pooled(m)
    print(f'{m:10} {a.mean():7.3f}{np.median(a):8.3f}{np.percentile(a,5):7.3f}{np.percentile(a,95):7.3f}{a.std():7.3f}')

print('\n=== seed-sensitivity (for tab:seed_sensitivity): mean, std, CV%, 5-95 range ===')
for m in ['STAGATE','GraphST','SpaceFlow','stCluster','SpaGCN']:
    a=pooled(m)
    cv=a.std()/a.mean()*100
    rng=np.percentile(a,95)-np.percentile(a,5)
    print(f'{m:10} mean={a.mean():.3f} std={a.std():.3f} CV={cv:.1f}% 5-95range={rng:.3f}')

print('\n=== Wilcoxon: SpaGVAE vs each baseline (paired section means, n=12, one-sided SpaGVAE>baseline) ===')
sg=sec_mean('SpaGVAE')
for m in ['STAGATE','GraphST','SpaceFlow','stCluster','SpaGCN']:
    bm=sec_mean(m)
    w,p=wilcoxon(sg,bm,alternative='greater')
    nwin=int((sg>bm).sum())
    print(f'SpaGVAE vs {m:10}: mean diff={ (sg-bm).mean():+.3f}  W={w:.1f}  p={p:.4g}  sections SpaGVAE>baseline={nwin}/12')
