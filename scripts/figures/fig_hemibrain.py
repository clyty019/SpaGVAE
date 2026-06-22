#!/usr/bin/env python
"""Plot the hemibrain worst/median/best spatial maps for SpaGVAE vs STAGATE from
the single consistent 20-seed run (outputs/.../hemibrain_fig20.npz), so the
figure's worst/median/best ARIs are drawn from the same distribution as Table 7.
Bottom-left panel carries the domain (ground-truth region) colour legend.
Pure plotting -- no GPU, no model training."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NPZ = 'outputs/spagvae_extra_datasets/hemibrain_fig20.npz'
OUT_DIR = 'overleaf/bmc-bioinformatics/figures'

d = np.load(NPZ, allow_pickle=True)
gt_int = d['gt_int']; coor = d['coor']; K = int(d['K'])
regions = list(d['region_names'])
palette = plt.cm.tab20(np.linspace(0, 1, 20))


def pick(aris):
    order = np.argsort(aris)                       # worst -> best
    return [int(order[0]), int(order[len(order) // 2]), int(order[-1])]


def plot(ax, lab, title, bold=False):
    ax.scatter(coor[:, 0], coor[:, 1], c=[palette[int(l) % 20] for l in lab],
               s=4, linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold' if bold else 'normal', pad=4)


METHODS = [('SpaGVAE', True), ('STAGATE', False)]
col_lab = ['Worst seed', 'Median seed', 'Best seed']
fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
plot(axes[0, 0], gt_int, 'Ground truth ($K=%d$)' % K)

for row, (name, bold) in enumerate(METHODS):
    aris = d[name + '_ari']; labs = d[name + '_lab']
    for j, si in enumerate(pick(aris)):
        head = '%s\n' % col_lab[j] if row == 0 else ''
        plot(axes[row, j + 1], labs[si], '%s%s (ARI %.3f)' % (head, name, aris[si]), bold=bold)
    print('%s worst/median/best ARI: %s' % (name, [round(aris[i], 3) for i in pick(aris)]), flush=True)

# bottom-left: domain colour legend (ground-truth regions)
lg = axes[1, 0]; lg.axis('off')
handles = [Patch(facecolor=palette[i % 20], edgecolor='none', label=str(regions[i]))
           for i in range(len(regions))]
ncol = 2 if len(regions) > 10 else 1
lg.legend(handles=handles, loc='center', ncol=ncol, frameon=False,
          fontsize=7, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
          labelspacing=0.3, title='Annotated domains', title_fontsize=9)

fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig('%s/fig_hemibrain.%s' % (OUT_DIR, ext), bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved fig_hemibrain (consistent 20-seed run + domain legend)', flush=True)
