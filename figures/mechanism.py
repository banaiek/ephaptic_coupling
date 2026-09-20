# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import style as S
from style import COL_STD, COL_ELIF, COL_RM, COL_TH, holm_pairwise
from style import stars as _stars_full


def stars(p):
    mark = _stars_full(p)
    return '***' if mark.startswith('****') else mark

RES = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures_out')
COL_OPEN = COL_TH
COL_GAP = '#7d4f9e'
COL_BIO = '#111111'
COL_SUB = '#9a9a9a'

S.apply()


def load(name):
    with open(os.path.join(RES, name + '.pkl'), 'rb') as fh:
        return pickle.load(fh)


def save(fig, name):
    S.finalize(fig)
    os.makedirs(OUT, exist_ok=True)
    extra = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight',
                bbox_extra_artists=extra, facecolor='white')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight',
                bbox_extra_artists=extra, facecolor='white')
    plt.close(fig)
    print(f'  saved {name}.pdf', flush=True)


def panel_label(ax, letter, dx=-0.22, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def _bars(ax, values, labels, colors, ylabel, chance=None, paired=True,
          fmt='{:.2f}', annotate=True, extra_pairs=()):
    means = [np.nanmean(v) for v in values]
    sems = [np.nanstd(v, ddof=1) / np.sqrt(max(1, np.sum(np.isfinite(v))))
            for v in values]
    x = np.arange(len(values))
    ax.bar(x, means, yerr=sems, color=colors, width=0.68, edgecolor='k',
           linewidth=0.5, capsize=3, error_kw={'lw': 0.8}, zorder=1)
    rng = np.random.default_rng(0)
    for i, v in enumerate(values):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size > 1:
            ax.scatter(i + (rng.random(v.size) - 0.5) * 0.28, v, s=6, color='k',
                       alpha=0.35, linewidths=0, zorder=3)
    if chance is not None:
        ax.axhline(chance, ls='--', color='gray', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    if annotate and len(values) > 1:
        comps = holm_pairwise(values, labels, paired=paired)
        vs_ref = [c for c in comps if c['i'] == 0]
        comps = holm_pairwise([values[0]] + [values[j] for j in range(1, len(values))],
                              [labels[0]] + labels[1:], paired=paired)
        comps = [c for c in comps if c['i'] == 0]
        order = sorted(range(len(comps)), key=lambda k: comps[k]['p_raw'])
        prev = 0.0
        for rank, k in enumerate(order):
            adj = max(min(1.0, (len(comps) - rank) * comps[k]['p_raw']), prev)
            prev = adj
            comps[k]['p_adj'] = adj
            comps[k]['sig'] = stars(adj)
        for pair in extra_pairs:
            for c in holm_pairwise(values, labels, paired=paired):
                if (c['i'], c['j']) == tuple(pair):
                    c = dict(c)
                    c['p_adj'] = c['p_raw']
                    c['sig'] = stars(c['p_raw'])
                    comps.append(c)
        top = max(m + s for m, s in zip(means, sems))
        base = min(min(means), 0.0)
        h = 0.07 * (top - base + 1e-9)
        lvl = 0
        for cmp in comps:
            if cmp['sig'] == 'n.s.':
                continue
            y = top + h * (1.2 + 1.5 * lvl)
            ax.plot([cmp['i'], cmp['i'], cmp['j'], cmp['j']],
                    [y, y + h * 0.35, y + h * 0.35, y], lw=0.8, color='k')
            ax.text((cmp['i'] + cmp['j']) / 2, y + h * 0.35, cmp['sig'],
                    ha='center', va='bottom', fontsize=7)
            lvl += 1
        if lvl:
            ax.set_ylim(top=top + h * (1.6 + 1.5 * lvl))
    return means, sems


def _bracket(ax, x0, x1, y, h, text):
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, color='k')
    ax.text((x0 + x1) / 2, y + h, text, ha='center', va='bottom', fontsize=7)


def _annotate_pairs(ax, comps, span, bars=()):
    from scipy.stats import wilcoxon
    out = []
    for x0, x1, a, b in comps:
        a, b = np.asarray(a, float), np.asarray(b, float)
        try:
            p = float(wilcoxon(a, b).pvalue)
        except Exception:
            p = 1.0
        sem = lambda v: v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
        base = max(a.mean() + sem(a), b.mean() + sem(b))
        spanned = [ht for bx, ht in bars if x0 - 1e-9 <= bx <= x1 + 1e-9]
        out.append([x0, x1, p, 1.0, max([base] + spanned)])
    order = sorted(range(len(out)), key=lambda k: out[k][2])
    prev = 0.0
    for rank, k in enumerate(order):
        adj = max(min(1.0, (len(out) - rank) * out[k][2]), prev)
        prev = adj
        out[k][3] = adj
    h = 0.05 * span
    placed, hi = [], -np.inf
    for x0, x1, _p, adj, base in out:
        if stars(adj) == 'n.s.':
            continue
        y = base + h
        while any(not (x1 < px0 or x0 > px1) and abs(y - py) < h * 2.1
                  for px0, px1, py in placed):
            y += h * 2.1
        _bracket(ax, x0, x1, y, h * 0.45, stars(adj))
        placed.append((x0, x1, y))
        hi = max(hi, y)
    if placed:
        ax.set_ylim(top=max(ax.get_ylim()[1], hi + h * 1.6))
    return len(placed)


def _collect(rows, path):
    out = []
    for r in rows:
        node = r
        for key in path:
            node = node[key]
        out.append(node)
    return np.asarray(out, float)


def _pair_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for x, y, col, lab in ((0.26, 0.52, COL_ELIF, 'source'),
                           (0.74, 0.36, COL_STD, 'neighbour')):
        ax.add_patch(Circle((x, y), 0.07, facecolor=col, edgecolor='k', lw=0.6,
                            zorder=3))
        ax.text(x, y - 0.15, lab, ha='center', fontsize=7)
    for rad in (0.12, 0.18, 0.24):
        ax.add_patch(Circle((0.26, 0.52), rad, facecolor='none',
                            edgecolor=COL_ELIF, lw=0.5, alpha=0.55, ls='--'))
    ax.add_patch(FancyArrowPatch((0.36, 0.47), (0.66, 0.39), lw=0.8,
                                 arrowstyle='-|>', mutation_scale=7,
                                 color=COL_ELIF))
    ax.text(0.5, 0.14, r'$\Phi_i = \alpha\,M_{ij}(V_j - V_{rest})$', fontsize=7,
            ha='center', color=COL_ELIF)
    ax.text(0.02, 0.96, 'the field one cell puts\non its neighbour', ha='left',
            va='top', fontsize=7)


def figure3():
    pair = load('pair')
    mech = load('mechanism')
    gap = load('gap')
    rep = load('replay_rm_clean')

    fig = plt.figure(figsize=(13.2, 7.0))
    top = fig.add_gridspec(1, 3, width_ratios=[2.9, 6.0, 4.2],
                           top=0.92, bottom=0.60, wspace=0.46)
    bot = fig.add_gridspec(1, 6, top=0.44, bottom=0.09, wspace=0.72)

    ax = fig.add_subplot(top[0, 0])
    _pair_schematic(ax)
    panel_label(ax, 'a', dx=-0.06, dy=1.12)

    ex = pair['example']
    t = ex['t']
    sel = (t >= 20) & (t <= 170)
    sub = top[0, 1].subgridspec(2, 1, hspace=0.12, height_ratios=[1, 1])
    ax_v = fig.add_subplot(sub[0])
    ax_p = fig.add_subplot(sub[1], sharex=ax_v)
    ax_v.plot(t[sel], ex['V_source'][sel], color=COL_ELIF, lw=0.9)
    spk = np.where(ex['spikes'] & sel)[0]
    ax_v.scatter(t[spk], np.full(spk.size, -46.5), marker='|', s=22, color='0.35',
                 linewidths=0.8)
    ax_v.text(t[sel][0], -45.2, 'spike times (the model has no spike waveform)',
              fontsize=5.2, color='0.35', va='bottom')
    ax_p.plot(t[sel], ex['phi']['sub'][sel] * 1e3, color=COL_STD, lw=0.9)
    for s_ in spk:
        for a_ in (ax_v, ax_p):
            a_.axvspan(t[s_], t[s_] + 2.0, color='0.86', lw=0, zorder=0)
    ax_v.set_ylabel('$V$ (mV)')
    ax_v.set_ylim(-62, -43)
    ax_v.tick_params(labelbottom=False)
    ax_p.set_ylabel(r'$\Phi$ ($\mu$V)')
    ax_p.set_xlabel('time (ms)')
    ax_v.set_title('one integrate-and-fire pair', loc='left', fontsize=7,
                   fontweight='normal')
    panel_label(ax_v, 'b', dx=-0.115, dy=1.34)

    ax = fig.add_subplot(top[0, 2])
    sta = pair['sta']
    for key, col, lab, lw in (('ap', COL_BIO, 'leak + spike', 1.0),
                              ('sub', COL_ELIF, 'leak', 1.4),
                              ('hold', COL_RM, 'held', 1.0)):
        ax.plot(sta['lag'], sta['phi'][key] * 1e3, color=col, lw=lw, label=lab,
                ls='--' if key == 'hold' else '-')
    ax.axvspan(0, 2.0, color='0.88', lw=0, zorder=0)
    ax.set_yscale('symlog', linthresh=20)
    ax.set_ylim(-900, 900)
    ax.axvline(0, ls=':', color='0.6', lw=0.6)
    ax.axhline(0, ls=':', color='0.6', lw=0.6)
    ax.set_xlabel('lag from spike (ms)')
    ax.set_ylabel(r'$\Phi$ ($\mu$V)')
    for k, (col, lab) in enumerate(((COL_ELIF, 'leak'),
                                    (COL_RM, 'held'),
                                    (COL_BIO, 'leak + spike'))):
        ax.text(0.04, 0.985 - 0.078 * k, lab, transform=ax.transAxes, color=col,
                fontsize=6.0, va='top', fontweight='bold')
    panel_label(ax, 'c', dx=-0.36)

    apf = load('apfield')
    _g40 = [k for k, q in enumerate(apf[0]['ap_gain']) if q['gain'] == 40][0]
    _hold = [k for k, q in enumerate(apf[0]['field_mode']) if q['mode'] == 'hold'][0]
    ax = fig.add_subplot(bot[0, 0])
    _bars(ax, [np.array([r['ap_gain'][0]['lif']['noise_corr'] for r in apf]),
               np.array([r['ap_gain'][0]['elif']['noise_corr'] for r in apf]),
               np.array([r['ap_gain'][_g40]['elif']['noise_corr'] for r in apf]),
               np.array([r['field_mode'][_hold]['elif']['noise_corr'] for r in apf])],
          ['LIF', 'leak', 'spike', 'held'],
          [COL_STD, COL_ELIF, COL_BIO, COL_RM], 'noise correlation',
          extra_pairs={(1, 2), (1, 3)})
    ax.tick_params(axis='x', labelsize=5.2)
    panel_label(ax, 'd', dx=-0.44)

    ax = fig.add_subplot(bot[0, 1])
    ax.set_title('network of 500 cells', loc='left', fontsize=7, fontweight='normal')
    pairs = ((_collect(mech, ['subthreshold', 'lif', 'v_corr']),
              _collect(mech, ['subthreshold', 'elif', 'v_corr'])),
             (_collect(mech, ['volt', 'lif', 'v_corr']),
              _collect(mech, ['volt', 'elif', 'v_corr'])))
    w = 0.34
    xs = np.array([0, 1])
    for off, idx, col, lab in ((-w / 2, 0, COL_STD, 'LIF'), (w / 2, 1, COL_ELIF, 'eLIF')):
        vals = [p[idx] for p in pairs]
        ax.bar(xs + off, [v.mean() for v in vals],
               yerr=[v.std(ddof=1) / np.sqrt(v.size) for v in vals], width=w,
               color=col, edgecolor='k', linewidth=0.5, capsize=3,
               error_kw={'lw': 0.8}, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(['spiking\noff', 'spiking\non'])
    ax.set_ylabel('voltage correlation')
    ax.tick_params(axis='x', labelsize=6.4)
    ax.legend(fontsize=6, handlelength=1.2)
    _annotate_pairs(ax, [(xs[g] - w / 2, xs[g] + w / 2, pairs[g][0], pairs[g][1])
                         for g in (0, 1)], ax.get_ylim()[1])
    panel_label(ax, 'e', dx=-0.34)

    ax = fig.add_subplot(bot[0, 2])
    keys = ['lif', 'elif', 'rm']
    shared = [_collect(mech, ['volt', k, 'var_shared']) for k in keys]
    priv = [_collect(mech, ['volt', k, 'var_private']) for k in keys]
    xs = np.arange(3)
    for off, vals, hatch in ((-0.19, shared, None), (0.19, priv, '///')):
        ax.bar(xs + off, [v.mean() for v in vals],
               yerr=[v.std(ddof=1) / np.sqrt(v.size) for v in vals], width=0.36,
               color=['0.25', COL_ELIF, COL_RM], edgecolor='k', linewidth=0.5,
               hatch=hatch, capsize=3, error_kw={'lw': 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels(['LIF', 'eLIF', 'RM'])
    ax.set_ylabel(r'voltage variance (mV$^2$)')
    ax.set_title('solid: shared      hatched: private', loc='left', fontsize=7,
                 fontweight='normal')
    _sem = lambda v: v.std(ddof=1) / np.sqrt(v.size)
    _annotate_pairs(ax, [(xs[0] - 0.19, xs[1] - 0.19, shared[0], shared[1]),
                         (xs[0] + 0.19, xs[1] + 0.19, priv[0], priv[1])],
                    ax.get_ylim()[1],
                    bars=[(xs[i] + off, v[i].mean() + _sem(v[i]))
                          for off, v in ((-0.19, shared), (0.19, priv))
                          for i in range(3)])
    panel_label(ax, 'f', dx=-0.40)

    labels = ['LIF', 'eLIF\n(RM)', 'replay\n(RM)']
    cols = [COL_STD, COL_ELIF, COL_OPEN]
    keys = ('lif', 'elif_rm', 'replay_same_rm')
    ax = fig.add_subplot(bot[0, 3])
    _bars(ax, [_collect(rep, [k, 'noise_corr']) for k in keys], labels, cols,
          'noise correlation', extra_pairs={(1, 2)})
    ax.tick_params(axis='x', labelsize=6)
    panel_label(ax, 'g', dx=-0.34)
    ax = fig.add_subplot(bot[0, 4])
    _bars(ax, [_collect(rep, [k, 'accuracy']) for k in keys], labels, cols,
          'decoding accuracy (%)', chance=10.0, extra_pairs={(1, 2)})
    ax.tick_params(axis='x', labelsize=6)
    panel_label(ax, 'h', dx=-0.36)

    ax = fig.add_subplot(bot[0, 5])
    g_ref = 2
    _bars(ax, [_collect(gap, ['lif', 'noise_corr']),
               _collect(gap, ['elif_rm', 'noise_corr']),
               np.array([r['gap'][g_ref]['rm']['noise_corr'] for r in gap])],
          ['LIF', 'eLIF\n(RM)', 'gap\n(RM)'],
          [COL_STD, COL_ELIF, COL_GAP], 'noise correlation')
    ax.tick_params(axis='x', labelsize=6)
    panel_label(ax, 'i', dx=-0.34)

    save(fig, 'Figure3')


def figure_s3():
    ap = load('apfield')
    pair = load('pair')
    mech = load('mechanism')
    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(3, 4, hspace=0.72, wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    loc = mech[0]['locality']
    extent_um = 1000.0
    sig = [q['sigma'] * extent_um for q in loc if q['sigma'] is not None]
    v = np.array([[q['metrics']['noise_corr'] for q in r['locality']
                   if q['sigma'] is not None] for r in mech])
    uni = np.array([[q['metrics']['noise_corr'] for q in r['locality']
                     if q['sigma'] is None] for r in mech]).ravel()
    y = np.concatenate([v.mean(0), [uni.mean()]])
    e = np.concatenate([v.std(0, ddof=1) / np.sqrt(len(mech)),
                        [uni.std(ddof=1) / np.sqrt(uni.size)]])
    x = np.arange(len(y))
    lifv = _collect(mech, ['lif', 'noise_corr'])
    ax.axhline(lifv.mean(), ls='--', color=COL_STD, lw=0.9)
    ax.text(0.03, lifv.mean(), ' LIF', color=COL_STD, fontsize=5.6, va='bottom')
    ax.errorbar(x, y, yerr=e, color=COL_ELIF, marker='o', ms=3.5, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s_:.0f}' for s_ in sig] + ['uniform'], fontsize=5.4)
    ax.set_xlim(-0.4, len(y) - 0.6)
    ax.set_ylim(0, lifv.mean() * 1.35)
    ax.set_xlabel(r'field kernel width ($\mu$m)')
    ax.set_ylabel('noise correlation')
    panel_label(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    tmpl = np.asarray(ap[0]['ap_template']['w'])
    tt = np.asarray(ap[0]['ap_template']['t'])
    ax.fill_between(tt, 0, tmpl, where=tmpl > 0, color=COL_BIO, alpha=0.16, lw=0)
    ax.fill_between(tt, 0, tmpl, where=tmpl < 0, color=COL_BIO, alpha=0.08, lw=0)
    ax.plot(tt, tmpl, color=COL_BIO, lw=1.3, label='field a neighbour receives')
    try:
        from network.biophysics import two_compartment_spike
        t_s, V_s, _, _ = two_compartment_spike(I_app=40.0)
        dV = np.gradient(V_s, t_s[1] - t_s[0])
        k = int(np.argmax(V_s))
        on = np.where(dV[:k] > 10.0)[0]
        t0 = t_s[on[0]] if len(on) else t_s[k]
        mm = (t_s >= t0) & (t_s <= t0 + tt[-1])
        v_ = V_s[mm]
        v_ = (v_ - v_.min()) / (v_.max() - v_.min())
        ax.plot(t_s[mm] - t0, v_, color='0.55', lw=0.9, ls='--',
                label='source action potential')
    except Exception:
        pass
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('time from spike (ms)')
    ax.set_ylabel('normalised')
    ax.set_ylim(-0.75, 1.65)
    ax.legend(fontsize=4.6, handlelength=1.0, loc='upper right', framealpha=0.9)
    panel_label(ax, 'b')

    xs = [q['gain'] for q in ap[0]['ap_gain']]
    rows = [r['ap_gain'] for r in ap]
    try:
        from network.ap_source import AP_CALIBRATED as cal
    except Exception:
        cal = 60.0
    wide_x = [q['gain'] for q in ap[0].get('ap_gain_wide', [])]
    wide = [r.get('ap_gain_wide', []) for r in ap]
    for j, (key, ylab, chance, lab) in enumerate(
            (('accuracy', 'decoding accuracy (%)', 10.0, 'c'),
             ('noise_corr', 'noise correlation', None, 'd'))):
        ax = fig.add_subplot(gs[0, j + 2])
        _sweep_panel(ax, rows, xs, key, 'source weight', ylab,
                     chance=chance)
        if wide_x:
            w = np.array([[p['elif'][key] for p in r] for r in wide if r])
            ax.errorbar(wide_x, w.mean(0), yerr=w.std(0, ddof=1) / np.sqrt(w.shape[0]),
                        color=COL_ELIF, marker='s', ms=3, lw=1.0, ls='--', mfc='white',
                        label='eLIF, wide kernel')
        ax.axvline(cal, ls=':', color='0.6', lw=0.7)
        if j == 0:
            ax.legend(fontsize=5.4, handlelength=1.2)
        panel_label(ax, lab)

    for j, (key, ylab, chance, lab) in enumerate(
            (('noise_corr', 'noise correlation', None, 'e'),
             ('accuracy', 'decoding accuracy (%)', 10.0, 'f'))):
        ax = fig.add_subplot(gs[1, j])
        vals = [_collect(ap, ['sign', k, key])
                for k in ('lif', 'negative', 'negative_rm')]
        _bars(ax, vals, ['LIF', r'$\alpha<0$', r'$\alpha<0$' '\n' '(RM)'],
              [COL_STD, COL_OPEN, COL_RM], ylab, chance=chance)
        ax.tick_params(axis='x', labelsize=5.4)
        panel_label(ax, lab)

    xs = [q['tau_ref'] for q in ap[0]['refractory']]
    rows = [r['refractory'] for r in ap]
    ax = fig.add_subplot(gs[1, 2])
    _sweep_panel(ax, rows, xs, 'accuracy', 'refractory period (ms)',
                 'decoding accuracy (%)', chance=10.0)
    ax.legend(fontsize=5.4, handlelength=1.2)
    panel_label(ax, 'g')
    ax = fig.add_subplot(gs[2, 0])
    _sweep_panel(ax, rows, xs, 'noise_corr', 'refractory period (ms)',
                 'noise correlation')
    panel_label(ax, 'h')

    xs = [q['V_reset'] for q in ap[0]['reset']]
    rows = [r['reset'] for r in ap]
    ax = fig.add_subplot(gs[2, 1])
    _sweep_panel(ax, rows, xs, 'accuracy', 'reset potential (mV)',
                 'decoding accuracy (%)', chance=10.0)
    ax.axvline(-60.0, ls=':', color='0.6', lw=0.7)
    panel_label(ax, 'i')
    ax = fig.add_subplot(gs[2, 2])
    _sweep_panel(ax, rows, xs, 'noise_corr', 'reset potential (mV)',
                 'noise correlation')
    ax.axvline(-60.0, ls=':', color='0.6', lw=0.7)
    ax.text(-60.0, ax.get_ylim()[1], ' rest', fontsize=5.5, color='0.4',
            va='top', ha='left')
    panel_label(ax, 'j', dy=1.14)

    save(fig, 'FigureS3')

def _sweep_panel(ax, rows, xs, key, xlabel, ylabel, chance=None):
    for name, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                           ('rm', COL_RM, 'RM')):
        v = np.array([[p[name][key] for p in r] for r in rows])
        ax.errorbar(xs, v.mean(0), yerr=v.std(0, ddof=1) / np.sqrt(v.shape[0]),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    if chance is not None:
        ax.axhline(chance, ls='--', color='0.6', lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def figure_s4():
    mech = load('mechanism')
    gap = load('gap')
    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.72)

    spec = gap[0].get('spectra')
    ax = fig.add_subplot(gs[0, 0])
    if spec:
        n = min(60, len(spec['gain_eph']))
        ax.plot(np.arange(n), np.asarray(spec['gain_eph'])[:n], color=COL_ELIF,
                lw=1.1, label='ephaptic')
        ax.plot(np.arange(n), np.asarray(spec['gain_gap'])[:n], color=COL_GAP,
                lw=1.1, label='gap junction')
    ax.axhline(1, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('spatial mode')
    ax.set_ylabel('mode gain')
    ax.legend(fontsize=6, loc='upper right')
    panel_label(ax, 'a', dy=1.16)

    gs_vals = [d['g'] for d in gap[0]['gap']]
    ax = fig.add_subplot(gs[0, 1])
    raw = np.array([[d['raw']['noise_corr'] for d in r['gap']] for r in gap])
    ax.errorbar(gs_vals, raw.mean(0), yerr=raw.std(0, ddof=1) / np.sqrt(len(gap)),
                color=COL_GAP, marker='s', ms=3, lw=1.1, ls=':', mfc='white',
                label='gap junction')
    nc = np.array([[d['rm']['noise_corr'] for d in r['gap']] for r in gap])
    ax.errorbar(gs_vals, nc.mean(0), yerr=nc.std(0, ddof=1) / np.sqrt(len(gap)),
                color=COL_GAP, marker='o', ms=3, lw=1.1, label='gap junction, RM')
    ax.axhline(np.mean(_collect(gap, ['lif', 'noise_corr'])), ls='--',
               color=COL_STD, lw=0.9, label='LIF')
    ax.axhline(np.mean(_collect(gap, ['elif_rm', 'noise_corr'])), ls='--',
               color=COL_ELIF, lw=0.9, label='eLIF (RM)')
    ax.set_xlabel('gap-junction coupling $g$')
    ax.set_ylabel('noise correlation')
    ax.legend(fontsize=5.6)
    panel_label(ax, 'b')

    ax = fig.add_subplot(gs[0, 2])
    araw = np.array([[d['raw']['accuracy'] for d in r['gap']] for r in gap])
    ax.errorbar(gs_vals, araw.mean(0), yerr=araw.std(0, ddof=1) / np.sqrt(len(gap)),
                color=COL_GAP, marker='s', ms=3, lw=1.1, ls=':', mfc='white',
                label='gap junction')
    acc = np.array([[d['rm']['accuracy'] for d in r['gap']] for r in gap])
    ax.errorbar(gs_vals, acc.mean(0), yerr=acc.std(0, ddof=1) / np.sqrt(len(gap)),
                color=COL_GAP, marker='o', ms=3, lw=1.1, label='gap junction, RM')
    ax.axhline(np.mean(_collect(gap, ['lif', 'accuracy'])), ls='--', color=COL_STD,
               lw=0.9, label='LIF')
    ax.axhline(np.mean(_collect(gap, ['elif_rm', 'accuracy'])), ls='--',
               color=COL_ELIF, lw=0.9, label='eLIF (RM)')
    ax.set_xlabel('gap-junction coupling $g$')
    ax.set_ylabel('decoding accuracy (%)')
    ax.legend(fontsize=6)
    panel_label(ax, 'c', dx=-0.30)

    vd = gap[0].get('vdist')
    ax = fig.add_subplot(gs[0, 3])
    if vd:
        for key, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                              ('gap', COL_GAP, 'gap junction')):
            ax.plot(np.asarray(vd['d']) * 1000.0, vd[key], color=col,
                    marker='o', ms=3, lw=1.1, label=lab)
        ax.legend(fontsize=6)
    ax.set_xlabel(r'inter-neuron distance ($\mu$m)')
    ax.set_ylabel('voltage correlation')
    panel_label(ax, 'd')

    ax = fig.add_subplot(gs[1, 0])
    dim = np.array([[d['rm']['dimensionality'] for d in r['gap']] for r in gap])
    ax.errorbar(gs_vals, dim.mean(0), yerr=dim.std(0, ddof=1) / np.sqrt(len(gap)),
                color=COL_GAP, marker='o', ms=3, lw=1.1, label='gap junction')
    ax.axhline(np.mean(_collect(gap, ['lif', 'dimensionality'])), ls='--',
               color=COL_STD, lw=0.9, label='LIF')
    ax.axhline(np.mean(_collect(gap, ['elif_rm', 'dimensionality'])), ls='--',
               color=COL_ELIF, lw=0.9, label='eLIF (RM)')
    ax.set_xlabel('gap-junction coupling $g$')
    ax.set_ylabel('dimensionality (PR)')
    ax.legend(fontsize=6)
    panel_label(ax, 'e')

    scales = [d['scale'] for d in mech[0]['inhibition']]
    ax = fig.add_subplot(gs[1, 1])
    for key, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                          ('rm', COL_RM, 'RM')):
        if key not in mech[0]['inhibition'][0]:
            continue
        vals = np.array([[d[key]['noise_corr'] for d in r['inhibition']] for r in mech])
        ax.errorbar(scales, vals.mean(0), yerr=vals.std(0, ddof=1) / np.sqrt(len(mech)),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    ax.set_xlabel('inhibitory weight scale')
    ax.set_ylabel('noise correlation')
    ax.legend(fontsize=6)
    panel_label(ax, 'f')

    ax = fig.add_subplot(gs[1, 2])
    for key, col, lab in (('elif', COL_ELIF, 'eLIF - LIF'), ('rm', COL_RM, 'RM - LIF')):
        if key not in mech[0]['inhibition'][0]:
            continue
        d_nc = np.array([[d[key]['noise_corr'] - d['lif']['noise_corr']
                          for d in r['inhibition']] for r in mech])
        ax.errorbar(scales, d_nc.mean(0), yerr=d_nc.std(0, ddof=1) / np.sqrt(len(mech)),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    ax.legend(fontsize=6)
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('inhibitory weight scale')
    ax.set_ylabel(r'$\Delta$ noise correlation')
    panel_label(ax, 'g')

    ax = fig.add_subplot(gs[1, 3])
    spk = np.array([[d['lif']['spikes'] for d in r['inhibition']] for r in mech])
    ax.errorbar(scales, spk.mean(0), yerr=spk.std(0, ddof=1) / np.sqrt(len(mech)),
                color=COL_STD, marker='o', ms=3, lw=1.1)
    ax.set_xlabel('inhibitory weight scale')
    ax.set_ylabel('spikes per trial')
    panel_label(ax, 'h', dy=1.16)

    save(fig, 'FigureS4')


def _het_sweep(ax, het, levels, key, ylabel, chance=None):
    for model, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                            ('rm', COL_RM, 'RM')):
        y = np.array([[q[model][key] for q in r['points']] for r in het])
        ax.errorbar(levels, y.mean(0), yerr=y.std(0, ddof=1) / np.sqrt(len(het)),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    if chance is not None:
        ax.axhline(chance, ls='--', color='0.6', lw=0.7)
    ax.set_xlabel('heterogeneity level')
    ax.set_ylabel(ylabel)


def figure_s5_heterogeneity():
    het = load('heterogeneity')
    levels = het[0]['levels']
    n = len(het)
    fig = plt.figure(figsize=(10.6, 3.1))
    gs = fig.add_gridspec(1, 4, wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    for model, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                            ('rm', COL_RM, 'RM')):
        cv = np.array([[q[model]['rate_cv'] for q in r['points']] for r in het])
        ax.errorbar(levels, cv.mean(0), yerr=cv.std(0, ddof=1) / np.sqrt(n),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    ax.set_xlabel('heterogeneity level')
    ax.set_ylabel('firing-rate CV')
    ax.legend(fontsize=6)
    panel_label(ax, 'k')

    ax = fig.add_subplot(gs[0, 1])
    target = float(np.mean([r['target_spikes'] for r in het]))
    for model, col, lab in (('lif', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF'),
                            ('rm', COL_RM, 'RM')):
        y = np.array([[q[model]['spikes'] for q in r['points']] for r in het])
        ax.errorbar(levels, y.mean(0), yerr=y.std(0, ddof=1) / np.sqrt(n),
                    color=col, marker='o', ms=3, lw=1.1, label=lab)
    ax.axhline(target, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('heterogeneity level')
    ax.set_ylabel('spikes per trial')
    ax.legend(fontsize=6)
    panel_label(ax, 'l')

    ax = fig.add_subplot(gs[0, 2])
    _het_sweep(ax, het, levels, 'noise_corr', 'noise correlation')
    ax.legend(fontsize=6)
    panel_label(ax, 'm')

    ax = fig.add_subplot(gs[0, 3])
    _het_sweep(ax, het, levels, 'accuracy', 'decoding accuracy (%)', chance=10.0)
    panel_label(ax, 'n')

    save(fig, 'FigureS5_added_heterogeneity')

def _column_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for x0, lab in ((0.06, 'column A'), (0.62, 'column B')):
        ax.add_patch(Rectangle((x0, 0.2), 0.3, 0.6, facecolor='0.94',
                               edgecolor='0.4', lw=0.6))
        ax.text(x0 + 0.15, 0.12, lab, ha='center', fontsize=7)
        rng = np.random.default_rng(1)
        px = x0 + 0.04 + rng.random(16) * 0.22
        py = 0.28 + rng.random(16) * 0.44
        ax.scatter(px, py, s=6, color=COL_STD, zorder=3)
        for xx, yy in zip(px[:6], py[:6]):
            ax.add_patch(Circle((xx, yy), 0.035, facecolor='none',
                                edgecolor=COL_ELIF, lw=0.4, alpha=0.7))
    ax.annotate('', xy=(0.60, 0.5), xytext=(0.38, 0.5),
                arrowprops=dict(arrowstyle='<|-|>', lw=0.9, color=COL_TH))
    ax.text(0.49, 0.62, 'dipole field', ha='center', fontsize=7, color=COL_TH)
    ax.text(0.49, 0.34, r'$\propto \cos\theta$', ha='center', fontsize=7,
            color=COL_TH)
    ax.text(0.5, 0.03, 'local field (red rings)', ha='center', fontsize=6.5,
            color=COL_ELIF)


def _beta_lines(ax, betas, series, ylabel, chance=None):
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    span = (max(betas) - min(betas)) or 1.0
    for vals, col, lab in series:
        m = vals.mean(0)
        e = vals.std(0, ddof=1) / np.sqrt(vals.shape[0])
        for k, b in enumerate(betas):
            ax.scatter(b + (rng.random(vals.shape[0]) - 0.5) * 0.045 * span,
                       vals[:, k], s=5, color=col, alpha=0.30, linewidths=0, zorder=2)
        ax.errorbar(betas, m, yerr=e, color=col, marker='o', ms=3.2, lw=1.2,
                    capsize=2, label=lab, zorder=3)
    a, b_ = series[0][0], series[1][0]
    lo, hi = ax.get_ylim()
    pad = 0.06 * (hi - lo)
    for k, bb in enumerate(betas):
        mark = stars(wilcoxon(a[:, k], b_[:, k]).pvalue)
        if not mark or mark.startswith('n'):
            continue
        top = max(v.mean(0)[k] + v.std(0, ddof=1)[k] / np.sqrt(v.shape[0])
                  for v, _, _ in series)
        ax.text(bb, top + pad, mark, ha='center', va='bottom', fontsize=6)
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    if chance is not None:
        ax.axhline(chance, ls='--', color='gray', lw=0.8)
    ax.set_xlabel(r'cross-column coupling $\beta$')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5.6, handlelength=1.2, loc='best')


def figure_s2_added():
    meso = load('mesoscopic')
    cols = meso['columns']
    betas = cols[0]['betas']
    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    _column_schematic(ax)
    panel_label(ax, 'g', dx=-0.05)

    ax = fig.add_subplot(gs[0, 1])
    uni = meso['uniformity']
    x = np.asarray(uni['x_um'])
    for R, prof in sorted(uni['profile'].items()):
        ax.plot(x, np.asarray(prof), lw=1.1,
                label=f'{int(R)} ' + r'$\mu$m away, CV ' + f"{uni['cv'][R]:.2f}")
    lcv = meso.get('local_cv')
    if lcv is not None:
        ax.axhline(1.0, ls=':', color=COL_ELIF, lw=0.9)
        ax.text(0.02, 0.06, f'local field across cells: CV {lcv:.2f}',
                transform=ax.transAxes, color=COL_ELIF, fontsize=6)
    ax.set_xlabel(r'position across the patch ($\mu$m)')
    ax.set_ylabel('field / field at centre')
    ax.legend(fontsize=5.8, handlelength=1.2)
    panel_label(ax, 'h')

    ax = fig.add_subplot(gs[0, 2])
    wc = np.array([[p['within']['noise_corr'] for p in r['points']] for r in cols])
    ax.errorbar(betas, wc.mean(0), yerr=wc.std(0, ddof=1) / np.sqrt(len(cols)),
                color=COL_ELIF, marker='o', ms=3, lw=1.1, label='within column')
    bc = np.array([[p['between_pairs'] for p in r['points']] for r in cols])
    ax.errorbar(betas, bc.mean(0), yerr=bc.std(0, ddof=1) / np.sqrt(len(cols)),
                color=COL_TH, marker='s', ms=3, lw=1.1, label='between columns')
    nl = cols[0].get('no_local')
    if nl:
        xs = [d['beta'] for d in nl]
        v = np.array([[r['no_local'][i]['within']['noise_corr'] for i in range(len(nl))]
                      for r in cols])
        ax.errorbar(xs, v.mean(0), yerr=v.std(0, ddof=1) / np.sqrt(len(cols)),
                    color=COL_STD, marker='^', ms=3, lw=1.0, ls='--',
                    label='within column,\nno local field')
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel(r'cross-column coupling $\beta$')
    ax.set_ylabel('noise correlation')
    ax.legend(fontsize=5.4, handlelength=1.2, labelspacing=0.25)
    panel_label(ax, 'i')

    ax = fig.add_subplot(gs[0, 3])
    wa = np.array([[0.5 * (p['within']['accuracy'] + p['within_B']['accuracy'])
                    for p in r['points']] for r in cols])
    ax.errorbar(betas, wa.mean(0), yerr=wa.std(0, ddof=1) / np.sqrt(len(cols)),
                color=COL_ELIF, marker='o', ms=3, lw=1.1)
    ax.set_xlabel(r'cross-column coupling $\beta$')
    ax.set_ylabel('within-column\ndecoding accuracy (%)')
    panel_label(ax, 'j')

    slow = meso['slow']
    ax = fig.add_subplot(gs[1, 0])
    f = np.asarray(slow[0]['f'])
    m = (f > 0) & (f < 3)
    b_max = max(d['beta'] for d in slow)
    for beta, col, lab in ((0.0, '0.5', 'uncoupled'),
                           (b_max, COL_TH, 'coupled, aligned')):
        P = np.array([d['P_B'] for d in slow
                      if d['beta'] == beta and d['cos_theta'] == 1.0])
        ax.semilogy(f[m], P.mean(0)[m], color=col, lw=1.1, label=lab)
    ax.axvline(slow[0]['f_drive'], ls=':', color=COL_ELIF, lw=0.8)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power of the undriven' '\n' 'column rate (a.u.)')
    ax.legend(fontsize=6)
    panel_label(ax, 'k')

    ax = fig.add_subplot(gs[1, 1])
    bet = sorted({d['beta'] for d in slow})
    name = {1.0: ('aligned', COL_TH), 0.0: ('orthogonal', '0.55')}
    for ct, (lab, col) in name.items():
        m, e = [], []
        for beta in bet:
            v = np.array([d['power_B_at_drive'] for d in slow
                          if d['beta'] == beta and d['cos_theta'] == ct])
            m.append(v.mean())
            e.append(v.std(ddof=1) / np.sqrt(max(1, v.size)))
        ax.errorbar(bet, m, yerr=e, color=col, marker='o', ms=3, lw=1.1, label=lab)
    ax.axhline(1.0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel(r'cross-column coupling $\beta$')
    ax.set_ylabel('power at the driving\nfrequency (relative)')
    ax.legend(fontsize=6)
    panel_label(ax, 'l')

    ent = load('meso_entrain')
    ebet = ent[0]['betas']
    def _pull(local, field):
        return np.array([[next(p['receiving_B'][field] for p in r['points']
                               if bool(p['local_field']) == local and p['beta'] == b)
                          for b in ebet] for r in ent])

    for slot, field, ylab, lo in ((gs[1, 2], 'noise_corr', 'noise correlation', None),
                                  (gs[1, 3], 'accuracy', 'decoding accuracy (%)', 10.0)):
        ax = fig.add_subplot(slot)
        _beta_lines(ax, ebet,
                    [(_pull(False, field), COL_STD, 'entrained'),
                     (_pull(True, field), COL_ELIF, 'entrained + ephaptic')],
                    ylab, chance=lo)
        panel_label(ax, 'm' if field == 'noise_corr' else 'n')


    save(fig, 'FigureS2')


MODEL_LABEL = {'lif': 'integrate-and-fire', 'het': 'heterogeneous LIF'}


def figure_s5_noise():
    data = load('operating')
    order = [k for k in ('lif', 'het') if k in data]
    fig = plt.figure(figsize=(10.0, 6.4))
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.35)

    for j, model in enumerate(order):
        rows = data[model]
        fracs = rows[0]['fracs']
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        _sweep_panel(ax, [r['points'] for r in rows], fracs, 'noise_corr',
                     'shared fraction of noise power', 'noise correlation')
        ax.set_title('   ' + MODEL_LABEL[model], loc='left', fontsize=7.5,
                     fontweight='normal')
        if j == 0:
            ax.legend(fontsize=6)
        panel_label(ax, 'fg'[j], dy=1.16)

    ax = fig.add_subplot(gs[0, 4:6])
    for model, col in zip(order, (COL_STD, COL_RM)):
        rows = data[model]
        fracs = rows[0]['fracs']
        d = np.array([[p['elif']['accuracy'] - p['lif']['accuracy'] for p in r['points']]
                      for r in rows])
        ax.errorbar(fracs, d.mean(0), yerr=d.std(0, ddof=1) / np.sqrt(len(rows)),
                    color=col, marker='o', ms=3, lw=1.1, label=MODEL_LABEL[model])
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('shared fraction of noise power')
    ax.set_ylabel('accuracy gain (points)')
    ax.legend(fontsize=6)
    panel_label(ax, 'h', dy=1.16)

    palette = {'lif': COL_STD, 'het': COL_RM}
    per = {}
    for model in order:
        b, dn, da = [], [], []
        for r in data[model]:
            for p in r['points']:
                b.append(p['lif']['noise_corr'])
                dn.append(p['elif']['noise_corr'] - p['lif']['noise_corr'])
                da.append(p['elif']['accuracy'] - p['lif']['accuracy'])
        per[model] = (np.array(b), np.array(dn), np.array(da))

    for k, (idx, ylab, letter) in enumerate(
            ((1, r'$\Delta$ noise correlation' '\n' '(eLIF - LIF)', 'i'),
             (2, 'accuracy gain (points)', 'j'))):
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        ax.axhline(0, ls=':', color='0.6', lw=0.7)
        rr = {}
        for model in order:
            b, y = per[model][0], per[model][idx]
            ax.scatter(b, y, s=9, color=palette[model], alpha=0.7, linewidths=0)
            if b.size > 2 and np.ptp(b) > 1e-9:
                fit = np.polyfit(b, y, 1)
                xx = np.linspace(b.min(), b.max(), 50)
                ax.plot(xx, np.polyval(fit, xx), color=palette[model], lw=1.0,
                        ls='--')
                r = np.corrcoef(b, y)[0, 1]
                rr[model] = r
        for n_, model in enumerate(order):
            if model in rr:
                ax.text(0.02 + 0.5 * n_, 1.04,
                        f'{MODEL_LABEL[model]}, r = {rr[model]:.2f}',
                        transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=5.6, color=palette[model], fontweight='bold')
        ax.set_xlabel('field-free noise correlation')
        ax.set_ylabel(ylab)
        panel_label(ax, letter)

    save(fig, 'FigureS5_added_noise')


def figure_s7_added():
    pair = load('pair')
    bio = pair['biophysical']
    meas = pair.get('measured')
    fig = plt.figure(figsize=(10.4, 3.0))
    gs = fig.add_gridspec(1, 3, wspace=0.52)

    tb = bio['t']
    m = (tb > -2) & (tb < 6)
    sub = gs[0, 0].subgridspec(2, 1, hspace=0.12)
    ax_s = fig.add_subplot(sub[0])
    ax_i = fig.add_subplot(sub[1], sharex=ax_s)
    ax_s.plot(tb[m], bio['V_soma'][m], color=COL_BIO, lw=1.0)
    ax_s.set_ylabel('soma $V$\n(mV)')
    ax_s.tick_params(labelbottom=False)
    ax_i.plot(tb[m], bio['I_nA'][m], color=COL_TH, lw=0.9)
    ax_i.axhline(0, ls=':', color='0.6', lw=0.6)
    ax_i.set_ylabel('membrane\ncurrent (nA)')
    ax_i.set_xlabel('time (ms)')
    ax_s.set_title('two-compartment HH source', loc='left', fontsize=6.4,
                   fontweight='normal')
    panel_label(ax_s, 'g', dx=-0.30, dy=1.40)

    ax = fig.add_subplot(gs[0, 1])
    dist = pair['distance']
    d = np.asarray(dist['d_um'])
    amp = np.abs(np.asarray(dist['peak_mV'])) * 1e3
    kern = np.exp(-d ** 2 / (2 * 150.0 ** 2))
    ref = int(np.argmin(np.abs(d - 50.0)))
    ax.plot(d, amp / amp[ref], color=COL_BIO, lw=1.1, label='dipole model')
    ax.plot(d, kern / kern[ref], color=COL_ELIF, lw=1.1, ls='--',
            label=r'kernel, $\sigma$ = 150 $\mu$m')
    ax.axvline(50.0, ls=':', color='0.6', lw=0.7)
    ax.axhline(1.0, ls=':', color='0.6', lw=0.7)
    i150 = int(np.argmin(np.abs(d - 150.0)))
    ax.annotate(f'{amp[i150] / amp[ref]:.2f}', xy=(150.0, amp[i150] / amp[ref]),
                xytext=(5, -9), textcoords='offset points', fontsize=5.4, color=COL_BIO)
    ax.annotate(f'{kern[i150] / kern[ref]:.2f}', xy=(150.0, kern[i150] / kern[ref]),
                xytext=(5, 4), textcoords='offset points', fontsize=5.4, color=COL_ELIF)
    ax.set_yscale('log')
    ax.set_xlim(20, 400)
    ax.set_xlabel(r'inter-soma distance ($\mu$m)')
    ax.set_ylabel(r'field, relative to 50 $\mu$m')
    ax.set_title('dotted: the close-pair distance', loc='left', fontsize=6.2,
                 fontweight='normal')
    ax.legend(fontsize=5.4, handlelength=1.4, loc='lower left', framealpha=0.9)
    panel_label(ax, 'h', dy=1.16)

    ax = fig.add_subplot(gs[0, 2])
    cmf = pair.get('dipole_cmf')
    exc = np.asarray(meas['excess_uV']) if meas is not None else None
    if exc is not None:
        ax.plot(meas['lag_ms'], exc, color=COL_RM, lw=1.6, label='measured')
    if cmf is not None:
        tc = np.asarray(cmf['t'])
        w = (tc > -2) & (tc < 6)
        base = np.asarray(cmf['excess_uV'])[w]
        k = np.abs(exc).max() / np.abs(base).max() if exc is not None else 1.0
        ax.plot(tc[w], base * k, color=COL_BIO, lw=1.2,
                label=f'two-compartment source')
        ax.text(0.03, 0.03, f'model scaled by {k:.1f}', transform=ax.transAxes,
                fontsize=5.2, color=COL_BIO, va='bottom')
    ax.axvline(0, ls=':', color='0.6', lw=0.6)
    ax.axhline(0, ls=':', color='0.6', lw=0.6)
    ax.set_xlim(-2, 6)
    ax.set_xlabel('lag from spike (ms)')
    ax.set_ylabel(r'close minus far ($\mu$V)')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.26 * (hi - lo))
    ax.legend(fontsize=5.6, handlelength=1.6, framealpha=0.9, loc='upper right')
    panel_label(ax, 'i')
    save(fig, 'FigureS7_added_panels_g-i')


FIGURES = {'3': figure3, 's3': figure_s3, 's4': figure_s4,
           's2': figure_s2_added, 's5noise': figure_s5_noise, 's5het': figure_s5_heterogeneity,
           's7': figure_s7_added}


def main(names):
    if not names or names == ['all']:
        names = list(FIGURES)
    for n in names:
        FIGURES[n.lower()]()


if __name__ == '__main__':
    main(sys.argv[1:])
