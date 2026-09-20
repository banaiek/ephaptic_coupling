# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator, ScalarFormatter
import matplotlib.text as _mtext

COL_STD = '#4d4d4d'
COL_ELIF = '#d1495b'
COL_RM = '#2a9d8f'
COL_TH = '#1f6fb2'
COLORS3 = [COL_STD, COL_ELIF, COL_RM]
LABELS3 = ['LIF', 'eLIF', 'RM']
COL = {'std': COL_STD, 'elif': COL_ELIF, 'rm': COL_RM}
LAB = {'std': 'LIF', 'elif': 'eLIF', 'rm': 'RM (rate-matched)'}

RC = {
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300, 'figure.dpi': 120, 'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5, 'axes.edgecolor': '#222222', 'axes.titleweight': 'bold',
    'axes.titlesize': 8, 'axes.labelsize': 8, 'font.size': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 8,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 1.2,
}

FIG_SCALE = 0.62
FONT_MAX = 8


def apply():
    plt.rcParams.update(RC)


def _nice_ticks(lo, hi, maxn=4):
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return None
    for step in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        start = math.ceil(lo / step - 1e-9) * step
        ticks, v = [], start
        while v <= hi + 1e-9:
            ticks.append(round(v, 1)); v += step
        if 2 <= len(ticks) <= maxn:
            return ticks
    return None


def _tfmt(v, _):
    s = f'{v:g}'
    return '0' if s in ('-0', '0') else s


def _looks_categorical(axis):
    labs = [t.get_text() for t in axis.get_ticklabels() if t.get_text() != '']
    if not labs:
        return False
    for l in labs:
        s = l.replace('−', '-').replace('%', '').strip()
        try:
            float(s)
        except ValueError:
            return True
    return False


def _cap_axis(axis, lo, hi):
    if (axis.get_scale() == 'log' or _looks_categorical(axis)
            or isinstance(axis.get_major_locator(), NullLocator)
            or len(axis.get_majorticklocs()) == 0):
        return
    lo, hi = sorted((lo, hi))
    tks = _nice_ticks(lo, hi)
    if tks:
        axis.set_major_locator(FixedLocator(tks))
        axis.set_major_formatter(FuncFormatter(_tfmt))
    else:
        axis.set_major_locator(MaxNLocator(nbins=3))
        sf = ScalarFormatter(useMathText=True); sf.set_scientific(True); sf.set_powerlimits((-1, 4))
        axis.set_major_formatter(sf)


def cap(ax):
    _cap_axis(ax.xaxis, *ax.get_xlim())
    _cap_axis(ax.yaxis, *ax.get_ylim())


def finalize(fig, scale=FIG_SCALE):
    for ax in fig.get_axes():
        cap(ax)
    for t in fig.findobj(_mtext.Text):
        try:
            if t.get_fontsize() > FONT_MAX:
                t.set_fontsize(FONT_MAX)
        except Exception:
            pass
    if scale and scale != 1.0:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scale, h * scale)


def stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def holm_pairwise(groups, labels, paired=True):
    from scipy.stats import wilcoxon, mannwhitneyu
    import itertools
    res = []
    idx = list(itertools.combinations(range(len(groups)), 2))
    for i, j in idx:
        a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        test = 'n/a'; stat = np.nan; p = 1.0
        try:
            if paired and len(a) == len(b) and len(a) > 1 and np.any(a - b != 0):
                w = wilcoxon(a, b); test = 'Wilcoxon signed-rank'; stat = float(w.statistic); p = float(w.pvalue)
            else:
                u = mannwhitneyu(a, b, alternative='two-sided'); test = 'Mann-Whitney U'; stat = float(u.statistic); p = float(u.pvalue)
        except Exception:
            pass
        res.append({'i': i, 'j': j, 'pair': f'{labels[i]} vs {labels[j]}',
                    'test': test, 'stat': stat, 'p_raw': float(p),
                    'n': int(min(len(a), len(b))),
                    'mean_i': float(np.nanmean(a)), 'mean_j': float(np.nanmean(b))})
    order = sorted(range(len(res)), key=lambda k: res[k]['p_raw'])
    m = len(res)
    prev = 0.0
    for rank, k in enumerate(order):
        adj = min(1.0, (m - rank) * res[k]['p_raw'])
        adj = max(adj, prev); prev = adj
        res[k]['p_adj'] = adj; res[k]['sig'] = stars(adj)
    return res


def bar3(ax, groups, ylabel, title, fmt='{:.0f}', chance=None, paired=True,
         annotate_sig=True):
    means = [np.nanmean(g) for g in groups]
    sems = [np.nanstd(g, ddof=1) / np.sqrt(max(1, np.sum(np.isfinite(g)))) for g in groups]
    x = np.arange(3)
    ax.bar(x, means, yerr=sems, color=COLORS3, width=0.66, edgecolor='k',
           linewidth=0.6, capsize=4, error_kw={'lw': 1.1}, zorder=1)
    jrng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        gv = np.asarray(g, float); gv = gv[np.isfinite(gv)]
        if gv.size > 1:
            ax.scatter(i + (jrng.random(gv.size) - 0.5) * 0.30, gv, s=9, color='k',
                       alpha=0.32, linewidths=0, zorder=3)
    for i, mn in enumerate(means):
        ax.text(i, mn + sems[i], fmt.format(mn), ha='center', va='bottom', fontsize=9, zorder=4)
    if chance is not None:
        ax.axhline(chance, ls='--', color='gray', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(LABELS3)
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left')
    if annotate_sig:
        comps = holm_pairwise(groups, ['LIF', 'eLIF', 'RM'], paired=paired)
        top = max(m + s for m, s in zip(means, sems))
        rng = top - min(means) + 1e-9
        h = 0.08 * rng
        lvl = 0
        for cmp in comps:
            if cmp['sig'] == 'n.s.':
                continue
            i, j = cmp['i'], cmp['j']
            y = top + h * (1.5 + 1.6 * lvl)
            ax.plot([i, i, j, j], [y, y + h * 0.4, y + h * 0.4, y], lw=1.0, color='k')
            ax.text((i + j) / 2, y + h * 0.4, cmp['sig'], ha='center', va='bottom', fontsize=8.5)
            lvl += 1
        if lvl:
            ax.set_ylim(top=top + h * (1.8 + 1.6 * lvl))
    return means, sems
