# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, pickle, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figures.panels as FV
from figures.panels import COL_STD, COL_ELIF, COL_RM, COL_TH
from figures.common import _compute_vel_tang

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300, 'figure.dpi': 120, 'savefig.facecolor': 'white',
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 8,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))
V02 = os.path.join(HERE, 'source_data')
OUT = os.path.join(HERE, 'output')
PY = sys.executable
BAR = [COL_STD, COL_ELIF, COL_RM]
TLAB = ['LIF', 'eLIF', 'RM']

RC8 = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
       'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 9}

import math as _math
from matplotlib.ticker import FixedLocator as _FixedLocator, FuncFormatter as _FuncFormatter


def _nice_ticks(lo, hi, maxn=4):
    span = hi - lo
    if span <= 0:
        return None
    for step in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
        start = _math.ceil(lo / step - 1e-9) * step
        ticks, v = [], start
        while v <= hi + 1e-9:
            ticks.append(round(v, 1)); v += step
        if 2 <= len(ticks) <= maxn:
            return ticks
    return None


def _tfmt(v, _):
    s = f'{v:g}'
    return '0' if s in ('-0', '0') else s


def _cap(ax, cbar=None):
    for axis, getlim in ((ax.xaxis, ax.get_xlim), (ax.yaxis, ax.get_ylim)):
        if axis.get_scale() == 'log':
            continue
        lo, hi = sorted(getlim())
        tks = _nice_ticks(lo, hi)
        if tks:
            axis.set_major_locator(_FixedLocator(tks))
        axis.set_major_formatter(_FuncFormatter(_tfmt))
    if cbar is not None:
        lo, hi = cbar.mappable.get_clim()
        tks = _nice_ticks(lo, hi)
        if tks:
            cbar.set_ticks(tks)
        cbar.ax.yaxis.set_major_formatter(_FuncFormatter(_tfmt))


def _save(fig, num):
    import style as _S
    _S.finalize(fig)
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f'Figure{num}')
    fig.savefig(base + '.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved Figure{num}.pdf + .png', flush=True)


def _bars(ax, vals, ylabel, title, chance=None, fmt='{:.0f}'):
    ax.bar(range(3), vals, color=BAR, width=0.66, edgecolor='k', linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha='center', va='bottom', fontsize=9.5)
    if chance is not None:
        ax.axhline(chance, ls='--', color='gray', lw=1.0, label='chance')
        ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks(range(3)); ax.set_xticklabels(TLAB)
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left', fontweight='bold')


def _traj(ax, scores, title):
    sc = np.asarray(scores)
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, sc.shape[1]))
    for s in range(sc.shape[1]):
        ax.plot(sc[:, s, 0], sc[:, s, 1], '-', color=cmap[s], lw=1.4, alpha=0.95)
        ax.scatter(sc[0, s, 0], sc[0, s, 1], color=cmap[s], s=16, zorder=3)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(title, loc='left', fontweight='bold')


def _mech_data():
    p = os.path.join(HERE, 'source_data', 'mech_source.pkl')
    if os.path.exists(p):
        data = pickle.load(open(p, 'rb'))
    else:
        data = pickle.load(open(f'{V02}/mech.pkl', 'rb')).data
    s2p = os.path.join(HERE, 'source_data', 'sigma2d.pkl')
    if os.path.exists(s2p):
        try:
            data['sigma2d'] = pickle.load(open(s2p, 'rb'))
        except Exception:
            pass
    return data


def _radial_kernels(geom, NE, nbins=20, dmax=1.0):
    from scipy.spatial.distance import cdist
    posE = np.asarray(geom.pos[:NE])
    D = cdist(posE, posE)
    W = np.asarray(geom.W_eph_E, float)
    GE = np.asarray(geom.G[:NE], float); IC = GE @ GE.T
    iu, ju = np.triu_indices(NE, k=1)
    dd = D[iu, ju]; wv = W[iu, ju]; iv = IC[iu, ju]
    edges = np.linspace(0.0, dmax, nbins + 1); ctr = 0.5 * (edges[:-1] + edges[1:])

    def _prof(v):
        out = np.full(nbins, np.nan)
        for b in range(nbins):
            m = (dd >= edges[b]) & (dd < edges[b + 1])
            if m.any():
                out[b] = v[m].mean()
        return out

    eb = _prof(wv); ib = _prof(iv)
    return ctr, eb / np.nanmax(eb), ib / np.nanmax(ib)


def _model_schematic(ax, _schletter='c'):
    from matplotlib.patches import Ellipse
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.5, 0.92, r'$\tau_m\,\dot V = -(V - V_{rest}) + I$', ha='center', va='center',
            fontsize=9, color='#333')
    ax.text(0.5, 0.80, r'$+\ \Phi$    (the ephaptic field)', ha='center', va='center',
            fontsize=9, color=COL_ELIF, fontweight='bold')
    nx = [0.28, 0.5, 0.72]; ny = 0.58
    ax.scatter(nx, [ny] * 3, s=200, c='#dcdcdc', edgecolors='#555', linewidths=0.8, zorder=3)
    ax.add_patch(Ellipse((0.5, 0.37), 0.62, 0.12, facecolor=COL_ELIF, alpha=0.15,
                         edgecolor=COL_ELIF, lw=1.0, zorder=1))
    ax.text(0.5, 0.37, r'shared field  $\Phi$', ha='center', va='center', fontsize=8.5, color=COL_ELIF, zorder=4)
    for x in nx:
        ax.annotate('', xy=(x, 0.43), xytext=(x, ny - 0.05),
                    arrowprops=dict(arrowstyle='<->', color=COL_ELIF, lw=1.0))
    ax.text(0.5, 0.17, 'coupling  ×  nearby depolarization', ha='center', fontsize=7.5, color='#555')
    ax.text(0.5, 0.05, r'$\Phi = \alpha\,\sum_j M_{ij}\,(V_j - V_{rest})$', ha='center', fontsize=8.5, color=COL_ELIF)
    ax.set_title(f'({_schletter})  Ephaptic field model', loc='left', fontweight='bold')


def figure1(num=1):
    import style as S
    from config import Config
    from geometry import NetworkGeometry
    d = _mech_data()
    cfg = Config.from_defaults(); cfg.geometry.eph_sparsify = 0.0
    geom = NetworkGeometry.build(cfg)
    pos = np.asarray(geom.pos); NE = cfg.network.NE
    with plt.rc_context(RC8):
        fig = plt.figure(figsize=(17.5, 8.6))
        gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.48)

        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(pos[:NE, 0], pos[:NE, 1], s=11, c='#555', label='excitatory', alpha=0.8, edgecolors='none')
        ax.scatter(pos[NE:, 0], pos[NE:, 1], s=20, c=COL_TH, marker='^', label='inhibitory', alpha=0.9, edgecolors='none')
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('(a)  Network geometry', loc='left', fontweight='bold'); ax.legend(loc='upper right'); _cap(ax)

        _model_schematic(fig.add_subplot(gs[0, 1]), _schletter='b')

        ax = fig.add_subplot(gs[0, 2])
        kc, keb, kib = _radial_kernels(geom, NE, nbins=20, dmax=1.0)
        ax.plot(kc, keb, 'o-', color=COL_ELIF, ms=3, label='ephaptic field', mfc='white')
        ax.plot(kc, kib, 's-', color='#888', ms=3, label='input correlation', mfc='white')
        ax.set_xlabel('distance'); ax.set_ylabel('coupling (normalized)')
        ax.set_title('(c)  Spatial coupling kernels', loc='left', fontweight='bold'); ax.legend(); _cap(ax)

        ax = fig.add_subplot(gs[1, 1])
        rc = d.get('rate_control')
        pdpath = os.path.join(HERE, 'source_data', 'decoding_source.pkl')
        if rc is not None:
            S.bar3(ax, [rc['std'], rc['elif'], rc['rm']], 'spikes (stim window)', '(e)  Rate-matched control')
        elif os.path.exists(pdpath):
            M = pickle.load(open(pdpath, 'rb'))['metrics']
            S.bar3(ax, [M['std']['spk'], M['elif']['spk'], M['rm']['spk']],
                   'spikes / trial', '(e)  Rate-matched control')
        else:
            ax.axis('off')

        ax = fig.add_subplot(gs[1, 2])
        c = d['correlations']; bc = np.asarray(c['bin_centers'])
        series = [('std', 'LIF', COL_STD), ('elif', 'eLIF', COL_ELIF)]
        if 'corr_binned_rm' in c:
            series.append(('rm', 'RM (rate-matched)', COL_RM))
        for key, lab, col in series:
            m_ = np.asarray(c[f'corr_binned_{key}']); s_ = np.asarray(c[f'corr_sem_{key}'])
            ax.plot(bc, m_, 'o-', color=col, label=lab, mfc='white')
            ax.fill_between(bc, m_ - s_, m_ + s_, color=col, alpha=0.18, lw=0)
        if 'input_corr_binned' in c:
            ax.plot(bc, np.asarray(c['input_corr_binned']), ':', color='#888', label='input corr (ref)')
        ax.set_xlabel('pairwise distance'); ax.set_ylabel('voltage correlation')
        ax.set_ylim(bottom=0)
        ax.set_title('(f)  Voltage correlation', loc='left', fontweight='bold'); ax.legend(); _cap(ax)

        try:
            rA = pickle.load(open(os.path.join(HERE, 'results', '_cache', 'V3_A.pkl'), 'rb'))
            ax = fig.add_subplot(gs[1, 0])
            modes = np.array(rA['modes'], float); kth = 2 * np.pi * modes
            al = rA['alpha']; sig = rA['sigma_eph']
            Hk = 1.0 / (1.0 - al * np.exp(-sig ** 2 * kth ** 2 / 2.0))
            gk = rA['kernels'].get('gaussian_norm', list(rA['kernels'].values())[0])
            ve = np.array(gk['Vgain_elif']); ve_se = np.array(gk.get('Vgain_elif_sem', np.zeros_like(ve)))
            vs = np.array(gk['Vgain_std']); vs_se = np.array(gk.get('Vgain_std_sem', np.zeros_like(vs)))
            ax.plot(modes, Hk, '-', color=COL_TH, lw=2.4, label=r'theory $H(k)$', zorder=1)
            ax.errorbar(modes, ve, yerr=ve_se, fmt='o', color=COL_ELIF, ms=5, mfc='white', capsize=2, label='eLIF (sim)', zorder=3)
            ax.errorbar(modes, vs, yerr=vs_se, fmt='s', color=COL_STD, ms=4, mfc='white', capsize=2, label='LIF (sim)', zorder=2)
            ax.axhline(1.0, ls=':', color='gray', lw=0.9)
            ax.set_xlabel('spatial mode  m  (cycles/unit)'); ax.set_ylabel('subthreshold voltage gain')
            ax.set_title('(d)  Field couples to coarse spatial modes', loc='left', fontweight='bold')
            ax.legend(loc='upper right'); _cap(ax)
        except FileNotFoundError:
            pass

        fig.suptitle(f'Figure {num}  |  Model and mechanism', fontsize=9, fontweight='bold', y=1.0)
        _save(fig, num)


def figureS1(num='S1'):
    import style as S
    import figures.coding as FP
    from config import Config
    from geometry import NetworkGeometry
    d = _mech_data()
    cfg = Config.from_defaults(); cfg.geometry.eph_sparsify = 0.0
    geom = NetworkGeometry.build(cfg)
    NE = cfg.network.NE
    posE = geom.pos[:NE]
    with plt.rc_context(RC8):
        fig = plt.figure(figsize=(16.5, 12)); gs = fig.add_gridspec(3, 3, wspace=0.5, hspace=0.5)

        def _foot(ax, w, title, mark0=False, star=None):
            w = np.asarray(w, float); w = w / (np.nanmax(w) or 1.0)
            sc = ax.scatter(posE[:, 0], posE[:, 1], c=w, s=15, cmap='viridis', edgecolors='none', vmin=0, vmax=1)
            if mark0:
                ax.scatter([posE[0, 0]], [posE[0, 1]], s=70, facecolors='white',
                           edgecolors='red', linewidths=1.4, zorder=5)
            if star is not None:
                ax.scatter([star[0]], [star[1]], s=130, marker='*', c='red',
                           edgecolors='white', linewidths=0.6, zorder=5)
            ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title, loc='left')
            cb = FP._vector_cbar(fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04))
            cb.set_label('relative')
            _cap(ax, cbar=cb)

        ax = fig.add_subplot(gs[0, 0])
        _foot(ax, geom.W_eph_E[0], '(a)  Ephaptic coupling, neuron 0', mark0=True)
        GE = np.asarray(geom.G[:NE]); ic0 = (GE @ GE.T)[0]
        ax = fig.add_subplot(gs[0, 1])
        _foot(ax, ic0, '(b)  Input-noise correlation, neuron 0', mark0=True)
        ksrc = int(np.argmin(np.linalg.norm(geom.source_pos - posE[0][None, :], axis=1)))
        ax = fig.add_subplot(gs[0, 2])
        _foot(ax, GE[:, ksrc], '(c)  One external noise source', star=geom.source_pos[ksrc])

        ax = fig.add_subplot(gs[1, 0])
        kc, keb, kib = _radial_kernels(geom, NE, nbins=20, dmax=1.0)
        ax.plot(kc, keb, 'o-', color=COL_ELIF, ms=3, mfc='white', label='ephaptic')
        ax.plot(kc, kib, 's-', color='#888', ms=3, mfc='white', label='input corr')
        ax.set_xlabel('distance'); ax.set_ylabel('coupling (relative)')
        ax.set_title('(d)  Radial kernels', loc='left'); ax.legend(); _cap(ax)

        try:
            A = pickle.load(open(os.path.join(HERE, 'results', '_cache', 'V3_A.pkl'), 'rb'))
            ax = fig.add_subplot(gs[1, 1]); modes = np.array(A['modes'], float)
            styles = {'gaussian_norm': ('-', 'o'), 'coulomb': ('--', '^')}
            for kk in A['kernels']:
                r = A['kernels'][kk]; ls, mk = styles.get(kk, ('-', 'o'))
                ratio = np.array(r['rate_gain_elif']) / np.maximum(np.array(r['rate_gain_std']), 1e-9)
                ax.plot(modes, ratio, ls + mk, color=COL_ELIF, mfc='white', label=f'eLIF/LIF ({kk})')
            ax.axhline(1.0, ls=':', color='gray')
            ax.set_xlabel('spatial mode m'); ax.set_ylabel('rate gain eLIF/LIF')
            ax.set_title('(e)  Transfer fn (kernel-robust)', loc='left'); ax.legend(); _cap(ax)
        except FileNotFoundError:
            pass

        lf = d['lfp']; t = np.asarray(lf['t'])
        lfp_series = [('std', COL_STD, 'LIF'), ('elif', COL_ELIF, 'eLIF')]
        if 'local_lfps_rm' in lf:
            lfp_series.append(('rm', COL_RM, 'RM'))
        ax = fig.add_subplot(gs[1, 2])
        for key, col, lab in lfp_series:
            ax.plot(t, np.asarray(lf[f'local_lfps_{key}'])[0], color=col, lw=1.0, label=lab)
        ax.set_xlabel('time (ms)'); ax.set_ylabel('local field (mV)')
        ax.set_title('(f)  Local LFP (one site)', loc='left'); ax.legend(); _cap(ax)

        ax = fig.add_subplot(gs[2, 0])
        al = d['alpha']
        ax.plot(al['alpha_vals'], al['noise_corr'], 'o-', color=COL_ELIF, mfc='white')
        ax.set_xlabel('α'); ax.set_ylabel('noise correlation'); ax.set_ylim(bottom=0)
        ax.set_title('(g)  Coupling sensitivity', loc='left'); _cap(ax)

        s2 = d.get('sigma2d')
        for ci, (dkey, lab, letter) in enumerate([('diff_elif', 'eLIF minus LIF', 'h'),
                                                  ('diff_rm', 'RM minus LIF', 'i')]):
            ax = fig.add_subplot(gs[2, ci + 1])
            if s2 is not None:
                M = np.asarray(s2[dkey]); se = np.asarray(s2['sigma_eph_vals']); si = np.asarray(s2['sigma_input_vals'])
                vmax = max(float(np.nanmax(np.abs(M))), 1e-3)
                im = FP._grid_cells(ax, si, se, M, 'RdBu_r', -vmax, vmax)
                cb = FP._vector_cbar(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)); cb.set_label(r'$\Delta$ corr')
                ax.set_xlabel(r'$\sigma_{input}$'); ax.set_ylabel(r'$\sigma_{eph}$')
                ax.set_title(f'({letter})  {lab}', loc='left'); _cap(ax, cbar=cb)
            else:
                ax.axis('off'); ax.set_title(f'({letter})  {lab}', loc='left')

        fig.suptitle(f'Figure {num}  |  Mechanism supplement', fontsize=9, fontweight='bold', y=0.995)
        _save(fig, num)


def figure_S2(num='S2'):
    from config import Config
    from geometry import generate_positions
    from scipy.stats import mannwhitneyu
    from matplotlib.patches import Patch
    import figures.coding as FP
    C = os.path.join(HERE, 'results', '_cache')
    D = pickle.load(open(os.path.join(C, 'V3_DEF.pkl'), 'rb'))
    cfg = Config.from_defaults(); fp = cfg.field_mode
    pos = generate_positions(cfg.network.N, cfg.geometry.layout, cfg.seed); NE = cfg.network.NE
    src = np.array(fp.src_pos, float); lam = fp.src_lambda
    dist = np.linalg.norm(pos[:NE] - src[None, :], axis=1); g = np.exp(-dist / lam)
    alphas = np.array(D['alphas'], float)
    a_lo = int(np.argmin(np.abs(alphas - 0.2))); a_hi = int(np.argmin(np.abs(alphas - 0.8)))
    CONDS = [('std', 'LIF', COL_STD), ('elif', 'eLIF', COL_ELIF), ('rm', 'RM', COL_RM),
             ('exo', 'exo', COL_TH), ('exo_rm', 'exo(RM)', '#8bb7df')]

    def _st(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''

    fig = plt.figure(figsize=(17.5, 9)); gs = fig.add_gridspec(2, 3, wspace=0.33, hspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(pos[:NE, 0], pos[:NE, 1], c=g, s=8, cmap='magma')
    ax.scatter([src[0]], [src[1]], marker='*', s=130, color='cyan', edgecolor='k', zorder=5, label='source')
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('(a)  Injected-field amplitude', loc='left', fontweight='bold')
    cb = FP._vector_cbar(fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)); cb.set_label('g(d)')
    ax.legend(fontsize=7, loc='upper right'); _cap(ax, cbar=cb)

    ax = fig.add_subplot(gs[0, 1]); dd = np.linspace(0, dist.max(), 100)
    ax.plot(dd, np.exp(-dd / lam), '-', color=COL_TH, lw=2)
    ax.set_xlabel('distance from source'); ax.set_ylabel('field amplitude g(d)')
    ax.set_title(f'(b)  Distance attenuation (λ={lam})', loc='left', fontweight='bold'); _cap(ax)

    def _mviolin(ax, key, ylab, letter, title):
        std_ps = {ai: np.asarray(D['std'][key]['per_seed'][ai], float) for ai in (a_lo, a_hi)}
        tests = [(m, ai) for m, _, _ in CONDS[1:] for ai in (a_lo, a_hi)]
        praw = []
        for m, ai in tests:
            xx = np.asarray(D[m][key]['per_seed'][ai], float)
            try:
                praw.append(float(mannwhitneyu(xx, std_ps[ai], alternative='two-sided').pvalue))
            except Exception:
                praw.append(1.0)
        praw = np.array(praw); order = np.argsort(praw); adj = np.ones_like(praw); prev = 0.0
        for rank, k in enumerate(order):
            v = min(1.0, (praw.size - rank) * praw[k]); v = max(v, prev); prev = v; adj[k] = v
        padj = {tests[i]: adj[i] for i in range(len(tests))}
        centers = []
        for ci, (m, lab, col) in enumerate(CONDS):
            base = ci * 2.5; centers.append(base + 1.475)
            for j, ai in enumerate((a_lo, a_hi)):
                p = base + 1 + j * 0.95
                data = np.asarray(D[m][key]['per_seed'][ai], float)
                vp = ax.violinplot([data], positions=[p], showmeans=True, showextrema=False, widths=0.82)
                b = vp['bodies'][0]; b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.28 if j == 0 else 0.62)
                vp['cmeans'].set_color(col); vp['cmeans'].set_linewidth(1.0)
                rng = np.random.default_rng(ci * 7 + j)
                ax.scatter(np.full(len(data), p) + (rng.random(len(data)) - 0.5) * 0.28,
                           data, s=4, color='k', alpha=0.3, edgecolors='none')
                if m != 'std':
                    s = _st(padj[(m, ai)])
                    if s:
                        ax.text(p, data.max() + 0.03 * (np.ptp(data) + 1e-9), s,
                                ha='center', va='bottom', fontsize=7, color=col)
        ax.set_ylabel(ylab); ax.set_title(f'({letter})  {title}', loc='left', fontweight='bold')
        ymin, ymax = ax.get_ylim(); ax.set_ylim(top=ymax + 0.12 * (ymax - ymin))
        ax.set_xticks(centers); ax.set_xticklabels([t[1] for t in CONDS], fontsize=7)
        ax.tick_params(axis='x', length=0)

    axc = fig.add_subplot(gs[0, 2])
    _mviolin(axc, 'noise_corr', 'noise correlation', 'c', 'Per-network noise corr')
    axc.legend(handles=[Patch(facecolor='gray', alpha=0.28, label='α = 0.2'),
                        Patch(facecolor='gray', alpha=0.62, label='α = 0.8')],
               loc='upper left', fontsize=7, frameon=False)
    _mviolin(fig.add_subplot(gs[1, 0]), 'nc', 'decoding accuracy (%)', 'd', 'Per-network decoding')
    _mviolin(fig.add_subplot(gs[1, 1]), 'dim', 'dimensionality', 'e', 'Per-network dimensionality')
    _mviolin(fig.add_subplot(gs[1, 2]), 'bits_per_spike', 'bits / spike', 'f', 'Per-network coding efficiency')
    fig.suptitle(f'Figure {num}  |  Endogenous/exogenous supplement: injected-field profile and '
                 'per-network coding at low vs high coupling (supports Figure 2)',
                 fontsize=12.5, fontweight='bold', y=0.99)
    _save(fig, num)


def _reuse_v3(v3key, fn, num, suffix):
    r = pickle.load(open(os.path.join(HERE, 'results', '_cache', f'{v3key}.pkl'), 'rb'))
    fn(r, os.path.join(HERE, 'results'), fignum=str(num))
    for ext in ('pdf', 'png'):
        src = os.path.join(HERE, 'results', f'{suffix}.{ext}')
        if os.path.exists(src):
            os.replace(src, os.path.join(OUT, f'Figure{num}.{ext}'))
    print(f'  Figure{num} <- {v3key}', flush=True)


def make_all(include_geom=True):
    import figures.coding as FP
    os.makedirs(OUT, exist_ok=True)
    if include_geom:
        figure1(1)
        figureS1('S1')
    try:
        rH = pickle.load(open(os.path.join(HERE, 'results', '_cache', 'V3_H.pkl'), 'rb'))
        rDEF = pickle.load(open(os.path.join(HERE, 'results', '_cache', 'V3_DEF.pkl'), 'rb'))
        FV.fig_control_parameter(rDEF, rH, os.path.join(HERE, 'results'), fignum='2')
        for ext in ('pdf', 'png'):
            src = os.path.join(HERE, 'results', f'V3_control_parameter.{ext}')
            if os.path.exists(src):
                os.replace(src, os.path.join(OUT, f'Figure2.{ext}'))
        print('  Figure2 <- V3_H + V3_DEF (control parameter; H(k) now in Figure 1)', flush=True)
    except FileNotFoundError as e:
        print(f'  [skip] Figure 2: {e}', flush=True)
    try:
        figure_S2('S2')
    except FileNotFoundError as e:
        print(f'  [skip] Figure S2: {e}', flush=True)
    for name, fn, arg in [('Figure 3 (decoding)', FP.figure4, 3),
                          ('Figure S3 (decoding supp)', FP.figure_supp_decoding, 'S3'),
                          ('Figure 4 (sparse)', FP.figure5, 4),
                          ('Figure S4 (sparse supp)', FP.figure_supp_sparse, 'S4')]:
        try:
            fn(arg)
        except FileNotFoundError as e:
            print(f'  [skip] {name}: source data not ready ({e})', flush=True)
    print('\n(Figure 5 + Figure S5 = real data: render via multipatch/figure.py & supplement.py)', flush=True)
    print('Figure set in', OUT, flush=True)


if __name__ == '__main__':
    make_all()