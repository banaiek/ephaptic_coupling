# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
import sys; sys.path.insert(0, V3)
from style import COL_STD, COL_ELIF, COL_TH, apply as _apply
_apply()
OUT = os.path.join(V3, 'figures.make_all')


def _sig(sta, tau, w=1.0):
    sta = np.asarray(sta); tau = np.asarray(tau)
    return float(np.mean(sta[np.abs(tau) <= w]))


def _stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def _holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min((m - i) * pvals[idx], 1.0)); adj[idx] = run
    return adj


def _sigbar(ax, x1, x2, y, padj, praw):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color='k')
    ax.text((x1 + x2) / 2, y + h, f'{_stars(padj)}  p={praw:.1e}',
            ha='center', va='bottom', fontsize=7.2)


def _rgb_heat(ax, M, cmap, vmin, vmax, **kw):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgba = cm(norm(np.asarray(M, float))).copy()
    rng = np.random.default_rng(0)
    rgba[..., :3] = np.clip(rgba[..., :3] + (rng.random(rgba[..., :3].shape) - 0.5) * (1.0 / 255), 0, 1)
    im = ax.imshow(rgba, **kw); im.set_rasterized(True)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    return im, sm


def _vector_cbar(cb):
    if cb is not None and getattr(cb, 'solids', None) is not None:
        cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')
    return cb


def main():
    rows = pickle.load(open(os.path.join(HERE, 'sta.pkl'), 'rb'))
    uc = [r for r in rows if not r['connected']]
    cc = [r for r in rows if r['connected']]
    tau = np.asarray(uc[0]['tau_ms'])
    d = np.array([r['distance'] for r in uc])
    amp = np.array([_sig(r['sta_mV'], tau) for r in uc]) * 1e3
    fig = plt.figure(figsize=(19, 9)); gs = fig.add_gridspec(2, 4, wspace=0.38, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    expts = sorted({r['expt'] for r in uc}); rowsf = []
    for e in expts:
        de = np.array([r['distance'] for r in uc if r['expt'] == e])
        ae = np.array([_sig(r['sta_mV'], tau) for r in uc if r['expt'] == e]) * 1e3
        c = ae[de < 70]; f = ae[de > 150]
        if len(c) >= 2 and len(f) >= 2:
            rowsf.append((c.mean() - f.mean(), c.std(ddof=1) / np.sqrt(len(c))))
    rowsf.sort(); diffs = [r[0] for r in rowsf]; yv = np.arange(len(rowsf))
    ax.errorbar(diffs, yv, xerr=[r[1] for r in rowsf], fmt='o', color=COL_ELIF, capsize=2, ms=4)
    ax.axvline(0, ls='--', color='#999'); ax.axvline(np.mean(diffs), ls='-', color='#7a1f2b', lw=1.5, label=f'mean {np.mean(diffs):.0f} µV')
    ax.set_yticks([]); ax.set_xlabel('close − far STA (µV)'); ax.set_ylabel('experiments')
    ax.set_title(f'(a)  Per-experiment (n={len(rowsf)}; {np.mean(np.array(diffs)>0)*100:.0f}%>0)', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(d, bins=np.arange(0, 320, 20), color=COL_TH, edgecolor='k', linewidth=0.4)
    ax.axvline(70, ls='--', color=COL_ELIF); ax.axvline(150, ls='--', color=COL_STD)
    ax.set_xlabel('inter-soma distance (µm)'); ax.set_ylabel('# pairs')
    ax.set_title(f'(b)  Pair distances (n={len(uc)})', loc='left')

    ax = fig.add_subplot(gs[0, 2])
    if cc:
        ax.plot(tau, np.vstack([r['sta_mV'] for r in cc]).mean(0) * 1e3, color=COL_TH, lw=2, label=f'connected (n={len(cc)})')
    ax.plot(tau, np.vstack([r['sta_mV'] for r in uc if r['distance'] < 70]).mean(0) * 1e3, color=COL_ELIF, lw=2, label='unconn <70µm')
    ax.axvline(0, ls=':', color='k', lw=0.8); ax.axhline(0, ls=':', color='k', lw=0.6)
    ax.set_xlabel('lag (ms)'); ax.set_ylabel('STA (µV)')
    ax.set_title('(c)  Synaptic positive control', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 3]); ax.axhline(0, ls=':', color='k', lw=0.6)
    grp = [amp[d < 70], amp[(d >= 70) & (d <= 150)], amp[d > 150]]
    parts = ax.violinplot([g[np.isfinite(g)] for g in grp], showmeans=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(COL_ELIF); pc.set_alpha(0.45)
    for i, g in enumerate(grp):
        gg = g[np.isfinite(g)]
        ax.errorbar(i + 1, gg.mean(), yerr=gg.std(ddof=1) / np.sqrt(len(gg)), fmt='o',
                    color='k', capsize=4, ms=5, zorder=6)
    sh = np.array([_sig(r['sh_sta_mV'], tau) if r.get('sh_sta_mV') else np.nan for r in uc]) * 1e3
    ax.axhspan(np.nanmean(sh) - np.nanstd(sh) / np.sqrt(np.isfinite(sh).sum()),
               np.nanmean(sh) + np.nanstd(sh) / np.sqrt(np.isfinite(sh).sum()), color='#bbb', alpha=0.4, label='shuffle ±SEM')
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['<70', '70–150', '>150'])
    cmp = [(0, 1, 'close', 'mid'), (1, 2, 'mid', 'far'), (0, 2, 'close', 'far')]
    raw = [mannwhitneyu(grp[i][np.isfinite(grp[i])], grp[j][np.isfinite(grp[j])],
                        alternative='two-sided').pvalue for i, j, _, _ in cmp]
    adj = _holm(raw)
    allv = np.concatenate([g[np.isfinite(g)] for g in grp])
    lo, hi = np.nanpercentile(allv, 2), np.nanpercentile(allv, 96)
    step = (hi - lo) * 0.12
    ax.set_ylim(lo - step, hi + step * (len(cmp) + 1.5))
    for k, (i, j, _, _) in enumerate(cmp):
        _sigbar(ax, i + 1, j + 1, hi + step * (k + 1), adj[k], raw[k])
    ax.set_xlabel('distance (µm)'); ax.set_ylabel('signed STA (µV)')
    ax.set_title('(d)  Pairwise comparisons (Holm-corr.)', loc='left'); ax.legend(fontsize=7.5, loc='lower left')

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(d, amp, s=8, color=COL_ELIF, alpha=0.35, edgecolors='none')
    def field(x, b, A, lam): return b + A * np.exp(-x / lam)
    try:
        p1, _ = curve_fit(field, d, amp, p0=[20, 30, 60], maxfev=20000, bounds=([-50, 0, 10], [200, 500, 400]))
        xx = np.linspace(20, 250, 100); ax.plot(xx, field(xx, *p1), '-', color='#7a1f2b', lw=2, label=f'field fit λ={p1[2]:.0f}µm')
        ax.legend(fontsize=8)
    except Exception:
        pass
    ax.axhline(0, ls=':', color='k', lw=0.6)
    ax.set_xlabel('inter-soma distance (µm)'); ax.set_ylabel('signed STA (µV)'); ax.set_ylim(-200, 300)
    ax.set_title('(e)  Per-pair STA vs distance', loc='left')

    ax = fig.add_subplot(gs[1, 1:3])
    from scipy.ndimage import uniform_filter1d
    order = np.argsort(d); Mh = np.vstack([uc[i]['sta_mV'] for i in order]) * 1e3
    Ms = uniform_filter1d(Mh, size=25, axis=0, mode='nearest'); vmax = np.percentile(np.abs(Ms), 98)
    im, sm = _rgb_heat(ax, Ms, 'RdBu_r', -vmax, vmax, aspect='auto', origin='lower',
                       extent=[tau[0], tau[-1], d[order][0], d[order][-1]])
    ax.axvline(0, ls=':', color='k', lw=0.8); ax.set_ylim(0, 300)
    ax.set_xlabel('lag (ms)'); ax.set_ylabel('distance (µm)')
    ax.set_title('(f)  STA × distance (all pairs)', loc='left')
    _vector_cbar(plt.colorbar(sm, ax=ax, fraction=0.046, label='STA (µV)'))

    ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='#f6f6f6', edgecolor='#ddd'))
    ns = np.array([r['n_spikes'] for r in uc])
    note = ['METHODS', 'STA = mean neighbour V locked to a',
            "cell's ISOLATED spikes (no other recorded", 'cell within ±15 ms); signed value at lag 0.',
            'Synaptic pairs excluded.', f'median {np.median(ns):.0f} spikes/pair.', '',
            'CONTROLS', 'far pairs: distance-indep. baseline',
            'shuffle: jitter null (±20 to 60 ms)', 'connected: synaptic positive ctrl',
            '', 'The effect is SUB-THRESHOLD, seen in',
            'the voltage STA, not in firing rate.',
            '', 'CAVEAT: residual electrode crosstalk',
            'not fully excluded; but the slow (ms-scale)', 'tail is unlike fast capacitive crosstalk.']
    for i, ln in enumerate(note):
        ax.text(0.05, 0.97 - i * 0.051, ln, transform=ax.transAxes, fontsize=8.0,
                fontweight='bold' if ln in ('METHODS', 'CONTROLS', 'CAVEAT: residual electrode crosstalk') else 'normal')

    fig.suptitle('Figure S5  |  Real-data ephaptic test (supports Figure 5)', fontweight='bold', y=0.99)
    import style as _S; _S.finalize(fig, scale=0.85)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'FigureS5.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved FigureS5', flush=True)


if __name__ == '__main__':
    main()
