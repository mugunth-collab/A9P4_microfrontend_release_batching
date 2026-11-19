#!/usr/bin/env python3
"""Figures 3-6 from the saved results (called at the end of run_all.py; can also run alone)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

METHODS = ['Monolithic (weekly)', 'Monolithic (tuned)', 'Per-module', 'Greedy batching', 'Random search', 'SCA (proposed)']
SHORT = {'carbon_credits': 'carbon', 'kyc_onboarding': 'kyc', 'meter_link': 'meter', 'nft_certificates': 'nft',
         'notifications': 'notif', 'p2p_energy_marketplace': 'p2p mkt', 'price_charts': 'charts', 'settings': 'settings',
         'staking': 'staking', 'token_swap': 'swap', 'tx_history': 'tx hist', 'wallet_core': 'wallet'}

def make(RES, FIG):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 8.5,
                         'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})
        plt.close(fig)
    met = json.load(open(os.path.join(RES, 'metrics.json')))
    fm = pd.read_csv(os.path.join(RES, 'per_fold_results.csv'))
    comp = pd.read_csv(os.path.join(RES, 'cost_components_per_change.csv'), index_col=0).reindex(METHODS)
    sens = pd.read_csv(os.path.join(RES, 'sensitivity_F0_CI.csv'))

    # Fig 3: stacked test-cost components per change
    fig, ax = plt.subplots(figsize=(3.3, 2.3)); left = np.zeros(len(METHODS))
    for c, col, lab in [('delay_cost', '0.78', r'Delay $\omega D$'), ('overhead', '0.45', 'Overhead $O$'),
                        ('risk', 'C3', 'Batch risk $R$'), ('coupling', 'C0', 'Coupling $X$')]:
        ax.barh(range(len(METHODS)), comp[c].values, left=left, color=col, label=lab, height=0.65); left += comp[c].values
    for i, v in enumerate(left): ax.text(v + 0.08, i, f'{v:.2f}', va='center', fontsize=7.5)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([m.replace(' (proposed)', '') for m in METHODS])
    ax.invert_yaxis(); ax.set_xlabel('Test cost per change (eng-h)'); ax.set_xlim(0, 7.6)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower center', bbox_to_anchor=(0.4, 1.0))
    ax.grid(axis='x', alpha=0.3); save(fig, 'fig3_cost_components')

    # Fig 4: (a) SCA plans per fold, (b) trains and releases vs micro-frontend share
    mods = met['modules']; M = len(mods); plans = met['sca_plans_seed11']
    fig = plt.figure(figsize=(3.4, 3.5))
    ax = fig.add_axes([0.17, 0.52, 0.66, 0.40]); cax = fig.add_axes([0.85, 0.52, 0.03, 0.40])
    mat = np.zeros((5, M)); lab = np.empty((5, M), dtype=object)
    for f in range(1, 6):
        a, taus = plans[str(f)]
        for m in range(M): mat[f - 1, m] = taus[a[m]]; lab[f - 1, m] = chr(65 + a[m])
    im = ax.imshow(mat, cmap='YlGnBu', aspect='auto', vmin=20, vmax=65)
    for f in range(5):
        for m in range(M): ax.text(m, f, lab[f, m], ha='center', va='center', fontsize=7.5,
                                   color='w' if mat[f, m] > 45 else 'k')
    ax.set_xticks(range(M)); ax.set_xticklabels([SHORT.get(x, x) for x in mods], rotation=60, ha='right', fontsize=7.5)
    shares = [100 * F['mfe_share'] for F in met['folds']]
    ax.set_yticks(range(5)); ax.set_yticklabels([f'F{f} ({s:.0f}%)' for f, s in zip(range(1, 6), shares)], fontsize=7.5)
    ax.set_title('(a) SCA trains (letters), cadence; MFE share', fontsize=8)
    cb = fig.colorbar(im, cax=cax); cb.set_label(r'$\tau_k$ (h)', fontsize=7.5); cb.ax.tick_params(labelsize=7.5)
    ax2 = fig.add_axes([0.17, 0.06, 0.66, 0.24])
    sf = fm[fm.method == 'SCA (proposed)'].set_index('fold'); gf = fm[fm.method == 'Greedy batching'].set_index('fold')
    x = np.arange(1, 6)
    ax2.plot(x, sf.trains, 'o-', color='C3', ms=3, lw=1, label='SCA trains')
    ax2.plot(x, gf.trains, 's--', color='0.4', ms=3, lw=1, label='Greedy trains')
    ax3 = ax2.twinx(); ax3.plot(x, sf.releases / 4, '^:', color='C0', ms=3, lw=1, label='SCA releases/wk')
    ax3.set_ylabel('Releases/wk', fontsize=7.5); ax3.tick_params(labelsize=7.5)
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{f}' for f in x]); ax2.set_ylabel('Trains', fontsize=7.5)
    ax2.set_title('(b) Active trains and release rate per fold', fontsize=8); ax2.tick_params(labelsize=7.5)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc='upper left'); ax2.grid(alpha=0.3)
    save(fig, 'fig4_plans')

    # Fig 5: sensitivity heatmaps
    F0s = sorted(sens.F0.unique()); CIs = sorted(sens.CI.unique())
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0))
    for ax, c, t in [(axs[0], 'sca_vs_greedy_pct', 'Saving vs greedy (%)'), (axs[1], 'sca_trains', 'SCA active trains')]:
        P = sens.pivot(index='F0', columns='CI', values=c)
        div = c.startswith('sca_vs'); vm = abs(P.values).max()
        ax.imshow(P.values, cmap='RdBu' if div else 'Greys', aspect='auto',
                  vmin=-vm if div else 0, vmax=vm if div else P.values.max() * 1.6)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(j, i, f'{P.values[i, j]:.1f}', ha='center', va='center', fontsize=7.5,
                        color='w' if div and abs(P.values[i, j]) > 0.6 * vm else 'k')
        ax.set_xticks(range(len(CIs))); ax.set_xticklabels([f'{v:g}' for v in CIs]); ax.set_yticks(range(len(F0s)))
        ax.set_yticklabels([f'{v:g}' for v in F0s]); ax.set_xlabel('$C_I$ (eng-h)'); ax.set_title(t, fontsize=8)
    axs[0].set_ylabel('$F_0$ (eng-h)'); fig.tight_layout(); save(fig, 'fig5_sensitivity')

    # Fig 6: delay vs release rate per fold (seed-mean)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    mk = dict(zip(METHODS, ['v', 'x', 'D', 's', '+', 'o']))
    for m, c in zip(METHODS, ['0.6', '0.3', 'C2', 'C0', 'C1', 'C3']):
        d = fm[fm.method == m]
        ax.scatter(d.delay_h, d.releases / 4, marker=mk[m], s=14, color=c, label=m.replace(' (proposed)', ''))
    ax.set_xscale('log'); ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([5, 10, 20, 40, 80]); ax.set_yticks([1, 3, 10, 30])
    ax.set_xlabel('Mean batching delay per change (h)'); ax.set_ylabel('Releases per week')
    ax.grid(alpha=0.3, which='major'); ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower left')
    save(fig, 'fig6_tradeoff')

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
    make(os.path.join(root, 'results'), os.path.join(root, 'figures'))
