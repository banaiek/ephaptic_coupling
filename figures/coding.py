# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import style as S
from style import COL_STD, COL_ELIF, COL_RM, COL_TH, COLORS3, LABELS3, holm_pairwise, stars
from figures.common import _compute_vel_tang

S.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures.make_all')
AUX = os.path.join(HERE, 'source_data', 'decoding_aux.pkl')


def _save(fig, num):
    S.finalize(fig)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f'Figure{num}.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT, f'Figure{num}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure{num}.pdf + .png', flush=True)


def _vector_cbar(cb):
    if cb is not None and getattr(cb, 'solids', None) is not None:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor('face')
    return cb


def _heat_cells(ax, M, cmap, vmin, vmax):
    M = np.asarray(M, float); n = M.shape[0]
    im = ax.pcolormesh(M, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors='face', linewidth=0, antialiased=False)
    ax.set_aspect('equal'); ax.set_xlim(0, n); ax.set_ylim(0, n); ax.invert_yaxis()
    ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels(range(n))
    ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels(range(n))
    return im


def _rgb_heat(ax, M, cmap, vmin, vmax, **kw):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgba = cm(norm(np.asarray(M, float))).copy()
    rng = np.random.default_rng(0)
    rgba[..., :3] = np.clip(rgba[..., :3] + (rng.random(rgba[..., :3].shape) - 0.5) * (1.0 / 255), 0, 1)
    im = ax.imshow(rgba, **kw); im.set_rasterized(True)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    return im, sm


def _grid_cells(ax, X, Y, C, cmap, vmin, vmax):
    im = ax.pcolormesh(X, Y, np.asarray(C, float), shading='nearest', cmap=cmap,
                       vmin=vmin, vmax=vmax, edgecolors='face', linewidth=0, antialiased=False)
    return im


def _confusion(ax, C, title, annot=False, cbar_ax=None, vmax=1.0):
    C = np.asarray(C, float); n = C.shape[0]
    im = _heat_cells(ax, C, 'gray_r', 0.0, vmax)
    if annot:
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.5, i + 0.5, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if C[i, j] > vmax * 0.55 else 'black')
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, loc='left')
    return im


def _diffmat(ax, D, title, annot=True):
    D = np.asarray(D, float); v = max(1e-6, np.abs(D).max()); n = D.shape[0]
    im = _heat_cells(ax, D, 'RdBu_r', -v, v)
    if annot:
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.5, i + 0.5, f'{D[i, j]:+.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(D[i, j]) > v * 0.6 else 'black')
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, loc='left')
    return im


def _sig_brackets(ax, groups, xpos, paired=True, labels=('LIF', 'eLIF', 'RM')):
    comps = holm_pairwise(groups, list(labels), paired=paired)
    top = max(np.nanmax(np.asarray(g, float)) for g in groups)
    bot = min(np.nanmin(np.asarray(g, float)) for g in groups)
    h = 0.06 * (top - bot + 1e-9); lvl = 0
    for c in comps:
        if c['sig'] == 'n.s.':
            continue
        i, j = xpos[c['i']], xpos[c['j']]
        y = top + h * (1.2 + 1.8 * lvl)
        ax.plot([i, i, j, j], [y, y + h * 0.4, y + h * 0.4, y], lw=1.0, color='k')
        ax.text((i + j) / 2, y + h * 0.4, c['sig'], ha='center', va='bottom', fontsize=8.5)
        lvl += 1
    if lvl:
        ax.set_ylim(top=top + h * (1.5 + 1.8 * lvl))


def _improve_bars(ax, base, others, labels, colors, ylabel, title):
    from scipy.stats import wilcoxon
    means, sems, ps = [], [], []
    for o in others:
        d = np.asarray(o) - np.asarray(base)
        means.append(np.nanmean(d)); sems.append(np.nanstd(d, ddof=1) / np.sqrt(len(d)))
        try:
            ps.append(wilcoxon(d).pvalue)
        except Exception:
            ps.append(1.0)
    x = np.arange(len(others))
    ax.bar(x, means, yerr=sems, color=colors, width=0.6, edgecolor='k', linewidth=0.6, capsize=4)
    ax.axhline(0, color='k', lw=0.8)
    for i, (mn, p) in enumerate(zip(means, ps)):
        ax.text(i, mn + np.sign(mn) * sems[i], f'{mn:+.1f}\n{stars(p)}', ha='center',
                va='bottom' if mn >= 0 else 'top', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel); ax.set_title(title, loc='left')


def _task_schematic(ax):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for lab, y in [('S1', 0.86), ('S2', 0.68), ('vdots', 0.47), ('S10', 0.20)]:
        if lab == 'vdots':
            ax.scatter([0.0575] * 3, [y - 0.045, y, y + 0.045], s=5, c='#333', zorder=4)
        else:
            ax.add_patch(FancyBboxPatch((0.015, y - 0.055), 0.085, 0.11,
                         boxstyle='round,pad=0.004', fc='#222', ec='none'))
            ax.text(0.0575, y, lab, ha='center', va='center', color='white', fontsize=7.5)
            ax.annotate('', xy=(0.30, 0.52), xytext=(0.11, y),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=0.9))
    ax.add_patch(FancyBboxPatch((0.30, 0.16), 0.30, 0.70, boxstyle='round,pad=0.01',
                 fc='#f3d2d2', ec='#cc9a9a', lw=1.0, alpha=0.65))
    rng = np.random.default_rng(0)
    ax.scatter(0.325 + 0.25 * rng.random(110), 0.20 + 0.62 * rng.random(110),
               s=6, c='white', edgecolors='#cc9a9a', linewidths=0.3, zorder=3)
    ax.text(0.45, 0.905, 'recurrent network', ha='center', fontsize=7.5)
    ax.annotate('', xy=(0.685, 0.52), xytext=(0.60, 0.52),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))
    ax.add_patch(FancyBboxPatch((0.685, 0.16), 0.30, 0.70, boxstyle='round,pad=0.01',
                 fc='none', ec='#555', lw=1.0, ls=(0, (4, 3))))
    ax.text(0.835, 0.905, 'decoder', ha='center', fontsize=7.5)
    rr = np.random.default_rng(1)
    for row in range(16):
        yy = 0.80 - row * 0.041
        col = '#b23b3b' if row < 5 else '#37619b'
        xs = 0.71 + 0.25 * rr.random(rr.integers(7, 15))
        ax.scatter(xs, np.full(len(xs), yy), s=2.2, c=col, marker='|', linewidths=0.7)
    ax.set_title('(a)  Decoding task', loc='left', fontweight='bold')


def _pop_tuning(ax, aux, ref=5):
    labels = np.asarray(aux['labels']); NE = 400; NSTIM = np.asarray(aux['tuning']).shape[1]
    pref = np.argmax(np.asarray(aux['tuning']), axis=1); cells = np.where(pref == ref)[0]

    def poptun(key):
        X = np.asarray(aux[key]); t = np.zeros((NE, NSTIM))
        for s in range(NSTIM):
            t[:, s] = X[labels == s][:, :NE].mean(0)
        return t[cells]
    x = np.arange(1, NSTIM + 1)
    for key, col, lab in [('X_std', COL_STD, 'LIF'), ('X_elif', COL_ELIF, 'eLIF'),
                          ('X_elif_rm', COL_RM, 'RM')]:
        t = poptun(key); nrm = t / (t.max(1, keepdims=True) + 1e-9)
        m = nrm.mean(0); se = nrm.std(0) / np.sqrt(len(cells))
        ax.plot(x, m, 'o-', color=col, ms=3, mfc='white', lw=1.3, label=lab)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
    ax.axvline(ref + 1, ls=':', color='gray', lw=0.8)
    ax.set_xticks([1, ref + 1, NSTIM]); ax.set_xticklabels(['S1', f'S{ref+1}', 'S10'])
    ax.set_xlabel('stimulus'); ax.set_ylabel('normalized response')
    ax.set_title(f'(b)  Tuning of S{ref+1}-preferring cells (n={len(cells)})', loc='left'); ax.legend(fontsize=7)


def _improve_all(ax, g, ns):
    from scipy.stats import wilcoxon
    decs = [('nc', 'NC'), ('lda', 'LDA'), ('svm', 'SVM')]; w = 0.36
    for mi, (mdl, col, lab) in enumerate([('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]):
        mns, sems, ps = [], [], []
        for dk, _ in decs:
            d = np.asarray(g(mdl, dk)) - np.asarray(g('std', dk))
            mns.append(np.nanmean(d)); sems.append(np.nanstd(d, ddof=1) / np.sqrt(ns))
            try:
                ps.append(wilcoxon(d).pvalue)
            except Exception:
                ps.append(1.0)
        xp = np.arange(len(decs)) + (mi - 0.5) * w
        ax.bar(xp, mns, w, yerr=sems, color=col, label=lab, edgecolor='k', linewidth=0.5, capsize=3)
        for x0, mn, se, p in zip(xp, mns, sems, ps):
            ax.text(x0, mn + se + 0.2, stars(p), ha='center', va='bottom', fontsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(decs))); ax.set_xticklabels([d[1] for d in decs])
    ax.set_ylabel('Δ accuracy vs LIF (pts)'); ax.set_title('(e)  Improvement vs LIF', loc='left')
    ax.legend(fontsize=7, loc='upper right')


def _noise_hist(ax, aux, g):
    dists = [('noise_corr_std', COL_STD, 'LIF'), ('noise_corr_elif', COL_ELIF, 'eLIF'),
             ('noise_corr_elif_rm', COL_RM, 'RM')]
    ymax = 0.0
    for k, c, lab in dists:
        v = np.asarray(aux.get(k, [])); v = v[np.isfinite(v)]
        if v.size:
            n, _, _ = ax.hist(v, bins=40, density=True, histtype='stepfilled', alpha=0.4, color=c, label=lab)
            ymax = max(ymax, float(n.max()))
    for k, c, lab in dists:
        v = np.asarray(aux.get(k, [])); v = v[np.isfinite(v)]
        if v.size:
            ax.plot(np.median(v), ymax * 1.07, marker='v', color=c, ms=7,
                    mec='k', mew=0.4, clip_on=False, zorder=5)
    ax.set_ylim(top=ymax * 1.24)
    ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_xlabel('pairwise noise correlation'); ax.set_ylabel('density')
    ax.set_title('(g)  Noise-correlation distribution', loc='left'); ax.legend(fontsize=7, loc='upper right')
    comps = holm_pairwise([g('std', 'noise_corr'), g('elif', 'noise_corr'), g('rm', 'noise_corr')],
                          ['LIF', 'eLIF', 'RM'], paired=True)
    labs = ['LIF', 'eLIF', 'RM']
    txt = '; '.join(f"{labs[c['i']]}–{labs[c['j']]} {c['sig']}" for c in comps if c['sig'] != 'n.s.')
    if txt:
        ax.text(0.02, 0.985, txt, transform=ax.transAxes, va='top', ha='left', fontsize=6.3)


def figure4(num=4):
    src = pickle.load(open(os.path.join(HERE, 'source_data', 'decoding_source.pkl'), 'rb'))
    aux = pickle.load(open(AUX, 'rb'))
    M = src['metrics']; ns = src['n_seeds']; chance = src['chance']
    g = lambda k, m: M[k][m]

    fig = plt.figure(figsize=(18, 18.5)); gs = fig.add_gridspec(5, 4, hspace=0.55, wspace=0.36,
                                                                 height_ratios=[1, 1, 1, 1, 0.82])

    _task_schematic(fig.add_subplot(gs[0, 0:2]))
    _pop_tuning(fig.add_subplot(gs[0, 2]), aux)
    S.bar3(fig.add_subplot(gs[0, 3]), [g('std', 'nc'), g('elif', 'nc'), g('rm', 'nc')],
           'accuracy (%)', '(c)  Decoding (nearest-centroid)', chance=chance)

    ax = fig.add_subplot(gs[1, 0])
    decs = ['nc', 'lda', 'svm']; w = 0.26
    for di, mdl in enumerate(('std', 'elif', 'rm')):
        mns = [np.nanmean(g(mdl, dd)) for dd in decs]
        sem = [np.nanstd(g(mdl, dd), ddof=1) / np.sqrt(ns) for dd in decs]
        ax.bar(np.arange(3) + (di - 1) * w, mns, w, yerr=sem, color=COLORS3[di],
               label=LABELS3[di], edgecolor='k', linewidth=0.5, capsize=3)
    ax.axhline(chance, ls='--', color='gray', lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(['NC', 'LDA', 'SVM']); ax.set_ylabel('accuracy (%)')
    ax.set_title('(d)  Decoder comparison', loc='left'); ax.legend(loc='upper right', fontsize=7)
    _improve_all(fig.add_subplot(gs[1, 1]), g, ns)
    S.bar3(fig.add_subplot(gs[1, 2]), [g('std', 'noise_corr'), g('elif', 'noise_corr'), g('rm', 'noise_corr')],
           'mean noise correlation', '(f)  Noise correlation', fmt='{:.3f}')
    _noise_hist(fig.add_subplot(gs[1, 3]), aux, g)

    S.bar3(fig.add_subplot(gs[2, 0]), [g('std', 'dprime'), g('elif', 'dprime'), g('rm', 'dprime')],
           "mean d'", "(h)  Discriminability d'", fmt='{:.2f}')
    ax_k = None
    for ci, (k, lab) in enumerate([('std', 'LIF'), ('elif', 'eLIF'), ('rm', 'RM')]):
        ax_k = fig.add_subplot(gs[2, ci + 1])
        im = _confusion(ax_k, src['confusion'][k], f'({chr(105+ci)})  NC confusion — {lab}')
    _vector_cbar(fig.colorbar(im, ax=ax_k, fraction=0.046, pad=0.04, label='P(pred | true)'))

    tr, trm = aux['trajectories'], aux.get('trajectories_rm', aux['trajectories'])
    ncl = np.asarray(tr['scores_std']).shape[1]
    tg = {'std': _compute_vel_tang(np.asarray(tr['scores_std']), ncl)[1],
          'elif': _compute_vel_tang(np.asarray(tr['scores_elif']), ncl)[1],
          'rm': _compute_vel_tang(np.asarray(trm['scores_elif']), ncl)[1]}
    tb = np.asarray(tr['time_bins'])[1:]
    for ci, (k, sc, lab) in enumerate([('std', tr['scores_std'], 'LIF'), ('elif', tr['scores_elif'], 'eLIF'),
                                       ('rm', trm['scores_elif'], 'RM')]):
        ax = fig.add_subplot(gs[3, ci]); sc = np.asarray(sc)
        cm = plt.cm.turbo(np.linspace(0.05, 0.95, sc.shape[1]))
        for s in range(sc.shape[1]):
            ax.plot(sc[:, s, 0], sc[:, s, 1], '-', color=cm[s], lw=1.3)
            ax.scatter(sc[0, s, 0], sc[0, s, 1], color=cm[s], s=14, zorder=3)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(f'({chr(108+ci)})  {lab} trajectories', loc='left')
    ax = fig.add_subplot(gs[3, 3])
    for k, c, l in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        ax.plot(tb, tg[k].mean(1), '-', color=c, lw=2, label=l)
    ax.set_xlabel('time (ms)'); ax.set_ylabel('tangling Q'); ax.set_title('(o)  Tangling over time', loc='left'); ax.legend(fontsize=7)

    S.bar3(fig.add_subplot(gs[4, 0]), [tg['std'].mean(0), tg['elif'].mean(0), tg['rm'].mean(0)],
           'tangling Q', '(p)  Mean tangling', fmt='{:.2f}', paired=False)

    fig.suptitle(f'Figure {num}  |  Stimulus decoding: ephaptic coupling preserves single-cell tuning but raises '
                 f'accuracy and discriminability, lowers noise correlations, and untangles trajectories at matched '
                 f'rate (n = {ns} network seeds; mean ± SEM; Holm-corrected)', fontsize=13, fontweight='bold', y=0.995)
    _save(fig, num)


def _sparse_schematic(ax):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    tt = np.linspace(0, 1, 60)
    profs = [np.exp(-2 * tt), np.exp(-2 * (1 - tt)), np.exp(-8 * (tt - 0.5) ** 2), 0.7 * np.ones_like(tt)]
    names = ['early-decay', 'late-rise', 'center-peak', 'sustained']
    pcols = ['#7048e8', '#f76707', '#0ca678', '#1c7ed6']

    ax.text(0.105, 0.965, '4 temporal patterns', ha='center', fontsize=7.5, fontweight='bold')
    ys = [0.80, 0.585, 0.37, 0.155]
    for yc, pr, nm, col in zip(ys, profs, names, pcols):
        x0, w, h = 0.02, 0.13, 0.17
        ax.add_patch(FancyBboxPatch((x0 - 0.01, yc - h / 2 - 0.012), w + 0.02, h + 0.024,
                     boxstyle='round,pad=0.003', fc='#fafafa', ec='#dddddd', lw=0.6))
        ax.plot(x0 + tt * w, yc + (pr - 0.5) * h * 0.82, color=col, lw=1.5)
        ax.text(x0 + w + 0.025, yc, nm, va='center', fontsize=6.8, color=col)
    ax.annotate('', xy=(0.415, 0.5), xytext=(0.315, 0.5),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.1))
    ax.text(0.365, 0.55, 'inject into\nsparse subset', ha='center', va='center', fontsize=6.0, color='#777')

    ax.add_patch(FancyBboxPatch((0.42, 0.14), 0.24, 0.72, boxstyle='round,pad=0.01',
                 fc='#f3d2d2', ec='#cc9a9a', lw=1.0, alpha=0.6))
    ax.text(0.54, 0.905, 'recurrent network', ha='center', fontsize=7.5)
    rng = np.random.default_rng(3)
    px = 0.435 + 0.21 * rng.random(90); py = 0.18 + 0.63 * rng.random(90)
    ax.scatter(px, py, s=6, c='white', edgecolors='#cc9a9a', linewidths=0.3, zorder=3)
    sub = rng.choice(len(px), 14, replace=False)
    ax.scatter(px[sub], py[sub], s=11, c='#7048e8', edgecolors='white', linewidths=0.3, zorder=4)
    ax.text(0.54, 0.095, 'active subset (sparsity)', ha='center', fontsize=6.0, color='#7048e8')
    ax.annotate('', xy=(0.735, 0.5), xytext=(0.665, 0.5),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))

    ax.add_patch(FancyBboxPatch((0.735, 0.14), 0.245, 0.72, boxstyle='round,pad=0.01',
                 fc='none', ec='#555', lw=1.0, ls=(0, (4, 3))))
    ax.text(0.8575, 0.905, 'nearest-centroid decoder', ha='center', fontsize=7.0)
    for i, (nm, col) in enumerate(zip(names, pcols)):
        yy = 0.73 - i * 0.155; hit = (i == 2)
        ax.add_patch(FancyBboxPatch((0.775, yy - 0.05), 0.165, 0.10, boxstyle='round,pad=0.004',
                     fc=col if hit else '#f2f2f2', ec=col, lw=0.9))
        ax.text(0.8575, yy, nm, ha='center', va='center', fontsize=6.2,
                color='white' if hit else '#666')
    ax.text(0.8575, 0.055, 'which pattern?  (chance 25%)', ha='center', fontsize=6.0, color='#777')
    ax.set_title('(a)  Sparse-coding task  (swept over sparsity × noise)', loc='left', fontweight='bold')


def figure5(num=5):
    src = pickle.load(open(os.path.join(HERE, 'source_data', 'sparse_source.pkl'), 'rb'))
    spr = src['sparsity_levels'] * 100; nz = src['noise_levels']; chance = src['chance']
    ns = src['n_seeds']
    bys = src['acc_by_seed']
    grid = src['acc_grid']

    fig = plt.figure(figsize=(18, 21.5))
    gs = fig.add_gridspec(5, 12, hspace=0.55, wspace=1.1, height_ratios=[0.9, 1, 1, 1, 1])

    _sparse_schematic(fig.add_subplot(gs[0, 0:12]))

    ax = fig.add_subplot(gs[1, 0:3])
    for k, c, lab in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        per = bys[k].mean(2)
        ax.errorbar(spr, per.mean(0), yerr=per.std(0, ddof=1) / np.sqrt(ns), fmt='o-', color=c, label=lab, capsize=3, mfc='white')
    ax.axhline(chance, ls='--', color='gray', lw=1.0); ax.set_xlabel('active neurons (%)'); ax.set_ylabel('accuracy (%)')
    ax.set_title('(b)  Accuracy vs sparsity', loc='left'); ax.legend()
    ax = fig.add_subplot(gs[1, 3:6])
    for k, c, lab in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        per = bys[k].mean(1)
        ax.errorbar(nz, per.mean(0), yerr=per.std(0, ddof=1) / np.sqrt(ns), fmt='o-', color=c, label=lab, capsize=3, mfc='white')
    ax.axhline(chance, ls='--', color='gray', lw=1.0); ax.set_xlabel('noise level'); ax.set_ylabel('accuracy (%)')
    ax.set_title('(c)  Accuracy vs noise', loc='left'); ax.legend()
    S.bar3(fig.add_subplot(gs[1, 6:9]), [src['mean']['std'], src['mean']['elif'], src['mean']['rm']],
           'mean accuracy (%)', '(d)  Overall accuracy', chance=chance)
    _improve_bars(fig.add_subplot(gs[1, 9:12]), src['mean']['std'], [src['mean']['elif'], src['mean']['rm']],
                  ['eLIF', 'RM'], [COL_ELIF, COL_RM], 'Δ accuracy (pts)', '(e)  Improvement vs LIF')

    for ci, (k, lab) in enumerate([('std', 'LIF'), ('elif', 'eLIF'), ('rm', 'RM')]):
        ax = fig.add_subplot(gs[2, ci * 4:ci * 4 + 4])
        im = _grid_cells(ax, nz, spr, grid[k], 'viridis', chance, 100)
        ax.set_xlabel('noise level')
        if ci == 0:
            ax.set_ylabel('active neurons (%)')
        ax.set_title(f'({chr(102+ci)})  NC accuracy — {lab}', loc='left')
        _vector_cbar(plt.colorbar(im, ax=ax, fraction=0.046))

    dif_e = grid['elif'] - grid['std']
    dif_r = grid['rm'] - grid['std']
    vdif = max(1e-6, float(np.abs(dif_e).max()), float(np.abs(dif_r).max()))
    for ci, ((dif, tag), (c0, c1)) in enumerate([((dif_e, 'eLIF'), (2, 6)), ((dif_r, 'RM'), (6, 10))]):
        ax = fig.add_subplot(gs[3, c0:c1])
        im = _grid_cells(ax, nz, spr, dif, 'RdBu_r', -vdif, vdif)
        ax.set_xlabel('noise level')
        if ci == 0:
            ax.set_ylabel('active neurons (%)')
        ax.set_title(f'({chr(105+ci)})  Accuracy gain  {tag} − LIF', loc='left')
        _vector_cbar(plt.colorbar(im, ax=ax, fraction=0.046, label='Δ accuracy (points)' if ci == 1 else ''))

    cf = src['confusion']
    _confusion(fig.add_subplot(gs[4, 0:3]), cf['std'], '(k)  NC confusion — LIF', annot=True)
    _confusion(fig.add_subplot(gs[4, 3:6]), cf['elif'], '(l)  NC confusion — eLIF', annot=True)
    _confusion(fig.add_subplot(gs[4, 6:9]), cf['rm'], '(m)  NC confusion — RM', annot=True)
    ax = fig.add_subplot(gs[4, 9:12])
    imd = _diffmat(ax, np.asarray(cf['rm']) - np.asarray(cf['std']), '(n)  Confusion RM − LIF')
    _vector_cbar(plt.colorbar(imd, ax=ax, fraction=0.046))

    fig.suptitle(f'Figure {num}  |  Sparse coding: the matched-rate benefit generalizes across sparsity and noise '
                 f'(nearest-centroid; n = {ns} seeds; mean ± SEM; Holm-corrected)', fontsize=13, fontweight='bold', y=1.0)
    _save(fig, num)


def _raster(ax, spikes, NE, title, dt=0.1):
    sp = np.asarray(spikes)
    yy, xx = np.nonzero(sp)
    tt = xx * dt
    ce = yy < NE
    ax.scatter(tt[ce], yy[ce], s=1.0, c=COL_ELIF, marker='.', alpha=0.5, linewidths=0)
    ax.scatter(tt[~ce], yy[~ce], s=1.0, c=COL_TH, marker='.', alpha=0.5, linewidths=0)
    ax.set_xlabel('time (ms)'); ax.set_ylabel('neuron'); ax.set_title(title, loc='left')


def figure_supp_decoding(num='S3'):
    d = pickle.load(open(AUX, 'rb'))
    src = pickle.load(open(os.path.join(HERE, 'source_data', 'decoding_source.pkl'), 'rb'))
    M = src['metrics']; g = lambda k, m: M[k][m]
    NE = 400; NT = d['counts_std'].shape[1]
    tuning = np.asarray(d['tuning'])
    pref = np.argmax(tuning, axis=1); order = np.argsort(pref)
    fig = plt.figure(figsize=(17, 15)); gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.34)

    S.bar3(fig.add_subplot(gs[0, 0]), [g('std', 'spk'), g('elif', 'spk'), g('rm', 'spk')],
           'spikes / trial', '(a)  Firing rate (RM matched)')
    for ci, (k, lab) in enumerate([('raster_std', 'LIF'), ('raster_elif', 'eLIF'), ('raster_rm', 'RM')]):
        r = d.get(k)
        if r is not None:
            _raster(fig.add_subplot(gs[0, ci + 1]), r, NE, f'({chr(98+ci)})  {lab} raster (example trial)')

    ax = fig.add_subplot(gs[1, 0])
    _tun = np.asarray(tuning[order], float)
    im, sm = _rgb_heat(ax, _tun, 'viridis', float(np.nanmin(_tun)), float(np.nanmax(_tun)),
                       aspect='auto', origin='lower')
    ax.set_xlabel('stimulus'); ax.set_ylabel('neuron (sorted by preference)')
    ax.set_title('(e)  Population tuning', loc='left')
    _vector_cbar(plt.colorbar(sm, ax=ax, fraction=0.046, label='tuning weight'))
    ax = fig.add_subplot(gs[1, 1]); mr = d['ex_neuron_mean_rate']; xs = np.arange(len(np.asarray(mr['std'])))
    for k, c, lab in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        ax.plot(xs, np.asarray(mr[k]), 'o-', color=c, label=lab, mfc='white')
    ax.set_xlabel('stimulus'); ax.set_ylabel('firing rate (Hz)')
    ax.set_title('(f)  Example-cell tuning', loc='left'); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(pref, bins=np.arange(tuning.shape[1] + 1) - 0.5, color=COL_STD, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('preferred stimulus'); ax.set_ylabel('neuron count')
    ax.set_title('(g)  Population coverage', loc='left')

    for ri, (tag, letter0) in enumerate([('lda', 104), ('svm', 107)]):
        axk = None; im = None
        for ci, (mk, lab) in enumerate([('std', 'LIF'), ('elif', 'eLIF'), ('rm', 'RM')]):
            C = d.get(f'confusion_{tag}_{mk}')
            if C is not None:
                axk = fig.add_subplot(gs[2 + ri, ci])
                im = _confusion(axk, np.asarray(C) / NT, f'({chr(letter0+ci)})  {tag.upper()} confusion — {lab}')
        if axk is not None:
            _vector_cbar(fig.colorbar(im, ax=axk, fraction=0.046, pad=0.04))

    fig.suptitle(f'Figure {num}  |  Decoding supplement: rate-matched firing control, rasters, population & '
                 f'single-cell tuning, LDA and SVM confusion matrices (supports Figure 3)',
                 fontsize=12.5, fontweight='bold', y=0.995)
    _save(fig, num)


def figure_supp_sparse(num='S4'):
    src = pickle.load(open(os.path.join(HERE, 'source_data', 'sparse_source.pkl'), 'rb'))
    spr = src['sparsity_levels'] * 100; nz = src['noise_levels']; chance = src['chance']
    ns = src['n_seeds']; bys = src['acc_by_seed']
    cf = src.get('lda_confusion'); ctag = 'LDA' if cf is not None else 'NC'
    if cf is None:
        cf = src['confusion']
    fig = plt.figure(figsize=(17, 9)); gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.36)

    nlo = 0; nhi = int(np.argmin(np.abs(np.asarray(nz) - 0.8)))
    ax = fig.add_subplot(gs[0, 0])
    for k, c, lab in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        lo = bys[k][:, :, nlo]; hi = bys[k][:, :, nhi]
        ax.errorbar(spr, lo.mean(0), yerr=lo.std(0, ddof=1) / np.sqrt(ns), fmt='o-',
                    color=c, label=lab, capsize=2, mfc='white', lw=1.6)
        ax.errorbar(spr, hi.mean(0), yerr=hi.std(0, ddof=1) / np.sqrt(ns), fmt='o--',
                    color=c, capsize=2, mfc=c, lw=1.2, alpha=0.75)
    ax.plot([], [], 'k-', label=f'noise {nz[nlo]:.1f} (low)')
    ax.plot([], [], 'k--', label=f'noise {nz[nhi]:.1f} (high)')
    ax.axhline(chance, ls='--', color='gray', lw=1.0)
    ax.set_xlabel('active neurons (%)'); ax.set_ylabel('accuracy (%)')
    ax.set_title('(a)  Accuracy vs sparsity', loc='left'); ax.legend(fontsize=6.8, ncol=1)

    ax = fig.add_subplot(gs[0, 1])
    for k, c, lab in [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'), ('rm', COL_RM, 'RM')]:
        lo = bys[k][:, 0, :]
        hi = bys[k][:, -1, :]
        ax.errorbar(nz, lo.mean(0), yerr=lo.std(0, ddof=1) / np.sqrt(ns), fmt='o-',
                    color=c, label=lab, capsize=2, mfc='white', lw=1.6)
        ax.errorbar(nz, hi.mean(0), yerr=hi.std(0, ddof=1) / np.sqrt(ns), fmt='o--',
                    color=c, capsize=2, mfc=c, lw=1.2, alpha=0.75)
    ax.plot([], [], 'k-', label=f'sparse ({spr[0]:.0f}%)')
    ax.plot([], [], 'k--', label=f'dense ({spr[-1]:.0f}%)')
    ax.axhline(chance, ls='--', color='gray', lw=1.0)
    ax.set_xlabel('noise level'); ax.set_ylabel('accuracy (%)')
    ax.set_title('(b)  Noise robustness', loc='left')
    ax.legend(fontsize=6.8, ncol=1)

    ax = fig.add_subplot(gs[0, 2])
    data = [src['mean'][k] for k in ('std', 'elif', 'rm')]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, c in zip(parts['bodies'], COLORS3):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    for ci, d in enumerate(data):
        ax.scatter(np.full(len(d), ci + 1) + (np.random.default_rng(ci).random(len(d)) - 0.5) * 0.2,
                   d, s=10, color='k', alpha=0.45, zorder=3)
    ax.axhline(chance, ls='--', color='gray', lw=1.0)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(LABELS3); ax.set_ylabel('mean accuracy (%)')
    ax.set_title(f'(c)  Per-seed accuracy (n={ns})', loc='left')
    _sig_brackets(ax, data, [1, 2, 3])

    im = None
    for ci, (k, lab) in enumerate([('std', 'LIF'), ('elif', 'eLIF'), ('rm', 'RM')]):
        im = _confusion(fig.add_subplot(gs[1, ci]), cf[k], f'({chr(100+ci)})  {ctag} confusion — {lab}', annot=True)
    cax = fig.add_subplot(gs[1, 3]); cax.axis('off')
    _vector_cbar(fig.colorbar(im, ax=cax, fraction=0.5, label='P(predicted | true)'))

    fig.suptitle(f'Figure {num}  |  Sparse-coding supplement: sparsity/noise slices, '
                 f'per-seed accuracy spread, and LDA confusion matrices (supports Figure 4)',
                 fontsize=12.5, fontweight='bold', y=0.99)
    _save(fig, num)


if __name__ == '__main__':
    figure4(); figure5(); figure_supp_decoding(); figure_supp_sparse()
