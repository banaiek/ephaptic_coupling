# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'figure.dpi': 120,
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 8,
    'axes.linewidth': 0.5, 'lines.linewidth': 1.2, 'lines.markersize': 5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titleweight': 'normal', 'figure.facecolor': 'white',
})

COL_STD = '#4d4d4d'
COL_ELIF = '#d1495b'
COL_RM = '#2a9d8f'
COL_TH = '#1f6fb2'
LAB = {'std': 'LIF', 'elif': 'eLIF', 'rm': 'RM (rate-matched)'}
COL = {'std': COL_STD, 'elif': COL_ELIF, 'rm': COL_RM}


def _stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''


def _paired_holm(per_a, per_b):
    from scipy.stats import wilcoxon
    raw, stats = [], []
    for a, b in zip(per_a, per_b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        try:
            w = wilcoxon(a, b); raw.append(float(w.pvalue)); stats.append(float(w.statistic))
        except Exception:
            raw.append(1.0); stats.append(np.nan)
    m = len(raw); order = sorted(range(m), key=lambda k: raw[k]); adj = [1.0] * m; prev = 0.0
    for rank, k in enumerate(order):
        v = min(1.0, (m - rank) * raw[k]); v = max(v, prev); prev = v; adj[k] = v
    return adj, stats, raw


def _sweep_stars(ax, x, per_a, per_b, y, color='k'):
    if per_a is None or per_b is None:
        return
    adj, _, _ = _paired_holm(per_a, per_b)
    yr = np.asarray(y, float)
    for xi, p, yi in zip(x, adj, yr):
        s = _stars(p)
        if s:
            ax.text(xi, yi, s, ha='center', va='bottom', fontsize=8, color=color)


def _field_stars(ax, x, defres, key):
    from scipy.stats import mannwhitneyu
    conds = [(m, col) for m, col in
             (('elif', COL_ELIF), ('rm', COL_RM), ('exo', COL_TH), ('exo_rm', '#8bb7df'))
             if m in defres]
    ps = defres['std'][key].get('per_seed')
    if not ps or not conds:
        return
    na = len(x); nc = len(conds)
    mean = {m: np.asarray(defres[m][key]['mean'], float) for m, _ in conds}
    sem = {m: np.asarray(defres[m][key]['sem'], float) for m, _ in conds}
    raw = np.ones((nc, na))
    for ci, (m, _) in enumerate(conds):
        pm = defres[m][key].get('per_seed')
        for ai in range(na):
            try:
                raw[ci, ai] = float(mannwhitneyu(np.asarray(pm[ai], float),
                                                 np.asarray(ps[ai], float),
                                                 alternative='two-sided').pvalue)
            except Exception:
                raw[ci, ai] = 1.0
    flat = raw.ravel(); order = np.argsort(flat); adj = np.ones_like(flat); prev = 0.0
    mtot = flat.size
    for rank, k in enumerate(order):
        v = min(1.0, (mtot - rank) * flat[k]); v = max(v, prev); prev = v; adj[k] = v
    adj = adj.reshape(nc, na)
    ymin, ymax = ax.get_ylim(); off = 0.03 * (ymax - ymin)
    for ai in range(na):
        sig = [ci for ci in range(nc) if adj[ci, ai] < 0.05]
        if len(sig) == nc:
            s = _stars(float(adj[:, ai].max()))
            if s:
                ytop = max(mean[m][ai] + sem[m][ai] for m, _ in conds)
                ax.text(x[ai], ytop + off, s, ha='center', va='bottom', fontsize=8, color='k')
        else:
            for ci in sig:
                m, col = conds[ci]; s = _stars(float(adj[ci, ai]))
                if s:
                    ax.text(x[ai], mean[m][ai] + sem[m][ai] + off, s,
                            ha='center', va='bottom', fontsize=8, color=col)


def _save(fig, name, save_dir):
    import style as _S
    _S.finalize(fig)
    fig.savefig(f"{save_dir}/{name}.png", bbox_inches='tight', facecolor='white')
    fig.savefig(f"{save_dir}/{name}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {save_dir}/{name}.png + .pdf", flush=True)


def _band(ax, x, d, color, label, marker='o', ms=None, mew=1.4):
    m = np.array(d['mean'], float); s = np.array(d['sem'], float)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, m, marker + '-', color=color, label=label, mfc='white', mew=mew,
            **({'ms': ms} if ms is not None else {}))


def _std_band(ax, d, label='LIF, α-independent'):
    m, s = d['mean'], max(d['sem'], 1e-9)
    ax.axhspan(m - s, m + s, color=COL_STD, alpha=0.18, linewidth=0)
    ax.axhline(m, ls='--', color=COL_STD, lw=1.6, label=label)


def _methods(ax, info, fs=8.6):
    ax.axis('off')
    order = [('task', 'TASK'), ('sweep', 'SWEEP'), ('predict', 'PREDICTION'),
             ('model', 'MODEL'), ('readout', 'READOUT'), ('stats', 'STATISTICS')]
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor='#f6f6f6', edgecolor='#dddddd', lw=1.0))
    y = 0.965
    if info.get('title'):
        for line in textwrap.wrap(info['title'], 46):
            ax.text(0.04, y, line, transform=ax.transAxes, fontsize=fs + 1.3,
                    fontweight='bold', va='top'); y -= 0.052
        y -= 0.015
    for key, head in order:
        if key not in info:
            continue
        ax.text(0.04, y, head, transform=ax.transAxes, fontsize=fs - 1.0,
                fontweight='bold', color='#7a1f2b' if key == 'predict' else '#444', va='top')
        y -= 0.045
        for line in textwrap.wrap(info[key], 52):
            ax.text(0.06, y, line, transform=ax.transAxes, fontsize=fs, va='top')
            y -= 0.042
        y -= 0.018


def fig_transfer(result, save_dir='results', fignum='2', field=None):
    modes = np.array(result['modes'], float)
    kth = 2 * np.pi * modes
    alpha = result['alpha']; sigma = result['sigma_eph']
    Mhat = np.exp(-sigma ** 2 * kth ** 2 / 2.0)
    Hk = 1.0 / (1.0 - alpha * Mhat)
    kernels = list(result['kernels'].keys())
    gk = result['kernels'].get('gaussian_norm', result['kernels'][kernels[0]])
    ve = np.array(gk['Vgain_elif']); ve_se = np.array(gk.get('Vgain_elif_sem', np.zeros_like(ve)))
    vs = np.array(gk['Vgain_std']); vs_se = np.array(gk.get('Vgain_std_sem', np.zeros_like(vs)))

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.34, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(modes, Hk, '-', color=COL_TH, lw=2.6, label=r'theory $H(k)=1/(1-\alpha\hat M(k))$', zorder=1)
    ax.errorbar(modes, ve, yerr=ve_se, fmt='o', color=COL_ELIF, ms=7, label='eLIF (sim)', zorder=3, mfc='white', mew=1.6, capsize=2)
    ax.errorbar(modes, vs, yerr=vs_se, fmt='s', color=COL_STD, ms=6, label='LIF (sim)', zorder=2, mfc='white', mew=1.4, capsize=2)
    ax.axhline(1.0, ls=':', color='gray', lw=0.9)
    ax.set_xlabel('spatial mode  m  (cycles / unit)'); ax.set_ylabel('subthreshold voltage gain')
    ax.set_title(f'(a)  The field is a low-pass filter   (α = {alpha:.2f})', loc='left'); ax.legend(loc='upper right', fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='#f6f6f6', edgecolor='#ddd'))
    ax.text(0.5, 0.94, 'Same field, two regimes', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.80, 'EXOGENOUS  (open loop)', fontsize=9.5, fontweight='bold', color=COL_TH, transform=ax.transAxes)
    for i, line in enumerate([r'$\Phi=\alpha M\,u_{ext}$  — imposed field,', 'independent of network state',
                              r'$\Rightarrow$ entrains  (H$_{sync}$: corr↑, dim↓)']):
        ax.text(0.07, 0.73 - i * 0.055, line, fontsize=8.8, transform=ax.transAxes)
    ax.text(0.05, 0.46, 'ENDOGENOUS  (closed loop)', fontsize=9.5, fontweight='bold', color=COL_ELIF, transform=ax.transAxes)
    for i, line in enumerate([r'$\Phi=\alpha M\,(V(t)-V_{rest})$  — reads', 'the live voltages of other neurons',
                              r'$\Rightarrow$ decorrelates  (H$_{endo}$: corr↓, dim↑)']):
        ax.text(0.07, 0.39 - i * 0.055, line, fontsize=8.8, transform=ax.transAxes)
    ax.text(0.05, 0.13, 'Exo = field injected from a source,', fontsize=8.6, style='italic', transform=ax.transAxes)
    ax.text(0.05, 0.06, 'amplitude attenuating with distance.', fontsize=8.6, style='italic', transform=ax.transAxes)

    if field is not None:
        a = np.array(field['alphas'], float)
        series = [('std', 'Std (no field)', COL_STD, 'x', '--'),
                  ('endo', 'endogenous (closed loop)', COL_ELIF, 'o', '-'),
                  ('endo_rm', 'endogenous, rate-matched', COL_RM, '^', '-'),
                  ('exo', 'exogenous injected (open loop)', COL_TH, 's', '-'),
                  ('exo_rm', 'exogenous, rate-matched', '#8bb7df', 'v', '--')]
        panels = [(0, 2, 'noise_corr', 'mean noise correlation', '(c)  Noise correlation'),
                  (1, 0, 'sync', 'synchrony  χ', '(d)  Population synchrony'),
                  (1, 1, 'dim', 'dimensionality', '(e)  Dimensionality')]
        for r, c, key, ylab, title in panels:
            ax = fig.add_subplot(gs[r, c])
            for ck, lab, col, mk, ls in series:
                if ck not in field:
                    continue
                d = field[ck][key]; m = np.array(d['mean'], float); s = np.array(d['sem'], float)
                ax.fill_between(a, m - s, m + s, color=col, alpha=0.13, lw=0)
                ax.plot(a, m, mk + ls, color=col, mfc='white', label=lab, lw=1.6, ms=5)
            em = np.array(field['endo'][key]['mean']); es = np.array(field['endo'][key]['sem'])
            xm = np.array(field['exo'][key]['mean']) + np.array(field['exo'][key]['sem'])
            _sweep_stars(ax, a, field['endo'][key].get('per_seed'), field['exo'][key].get('per_seed'),
                         np.maximum(em + es, xm) + 0.02 * (np.nanmax(xm) - np.nanmin(em) + 1e-9), color='k')
            ax.set_xlabel('α  (field amplitude)'); ax.set_ylabel(ylab); ax.set_title(title, loc='left')
            if r == 0 and c == 2:
                ax.legend(loc='best', fontsize=7)
    else:
        ax = fig.add_subplot(gs[1, 0])
        styles = {'gaussian_norm': ('-', 'o'), 'coulomb': ('--', '^')}
        for kernel in kernels:
            r = result['kernels'][kernel]; ls, mk = styles.get(kernel, ('-', 'o'))
            ratio = np.array(r['rate_gain_elif']) / np.maximum(np.array(r['rate_gain_std']), 1e-9)
            ax.plot(modes, ratio, ls + mk, color=COL_ELIF, mfc='white', label=f'eLIF/LIF — {kernel}')
        ax.axhline(1.0, ls=':', color='gray', lw=0.9)
        ax.set_xlabel('spatial mode  m'); ax.set_ylabel('firing-rate gain, eLIF/LIF')
        ax.set_title('(c)  Functional gain — kernel-robust', loc='left'); ax.legend(fontsize=8.5)

    _methods(fig.add_subplot(gs[:, 2]) if field is None else fig.add_subplot(gs[1, 2]),
             field.get('info', {}) if field is not None else result.get('info', _transfer_info(result)))
    fig.suptitle(f'Figure {fignum}  |  Endogenous vs exogenous field coupling: an imposed coherent field entrains '
                 'the network, but the self-generated (closed-loop) field decorrelates it',
                 fontsize=13, fontweight='bold', y=1.0)
    _save(fig, 'V3_A_transfer_function', save_dir)


def _transfer_info(result):
    return {'title': 'spatial transfer function H(k): endogenous & exogenous',
            'task': 'drive a spatial sinusoid cos(2 pi m x) (an exogenous field) and read the response gain at mode m.',
            'sweep': f"m = {result['modes']} cycles/unit; kernels {list(result['kernels'].keys())}.",
            'predict': ('H1 synchronizer: gain=1/(1-alpha*Mhat(k)) falls with k. '
                        'H2 sharpener: gain rises with k. Data adjudicates -> H1.'),
            'model': f"alpha = {result['alpha']}, sigma_eph = {result['sigma_eph']}.",
            'readout': 'subthreshold gain with spiking disabled (exact linear response); rate gain with spikes.',
            'stats': 'gain mean +/- SEM over noise realizations.'}


def fig_alpha(result, save_dir='results', fignum='3'):
    a = np.array(result['alphas'], float); std = result['std']
    fig = plt.figure(figsize=(17, 9.2))
    gs = fig.add_gridspec(2, 4, wspace=0.34, hspace=0.34)
    panels = [
        (0, 0, 'nc', 'decoding accuracy (%)', '(a)  Population decoding', 'nc', True),
        (0, 1, 'noise_corr', 'mean noise correlation', '(b)  Noise correlation', 'noise_corr', False),
        (0, 2, 'sync', 'synchrony  χ', '(c)  Population synchrony', 'sync', False),
        (1, 0, 'dim', 'dimensionality', '(d)  Effective dimensionality', 'dim', False),
        (1, 1, 'bits_per_spike', 'bits / spike', '(e)  Coding efficiency', 'bits_per_spike', False),
        (1, 2, 'spk', 'spikes / trial', '(f)  Firing rate (RM matched)', 'spk', False),
    ]
    for r, c, key, ylab, title, sk, chance in panels:
        ax = fig.add_subplot(gs[r, c])
        _band(ax, a, result['std'][key], COL_STD, 'LIF', 'x')
        _band(ax, a, result['elif'][key], COL_ELIF, 'eLIF', 'o')
        _band(ax, a, result['rm'][key], COL_RM, 'RM (rate-matched)', '^')
        if chance:
            ax.axhline(result['chance'], ls=':', color='gray', lw=0.9, label='chance')
        pe = result['elif'][key].get('per_seed'); ps = result['std'][key].get('per_seed')
        if pe and ps:
            em = np.array(result['elif'][key]['mean']); es = np.array(result['elif'][key]['sem'])
            yoff = (em.max() - em.min() + 1e-9) * 0.04
            _sweep_stars(ax, a, pe, ps, em + es + yoff, color=COL_ELIF)
        ax.set_xlabel('α  (fraction of critical coupling)'); ax.set_ylabel(ylab)
        ax.set_title(title, loc='left')
        if r == 0 and c == 0:
            ax.legend(loc='lower right')
    _methods(fig.add_subplot(gs[:, 3]), result.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Approaching criticality, the field decorrelates, desynchronizes, expands dimensionality, '
                 'and improves decoding at matched rate',
                 fontsize=13, fontweight='bold', y=0.98)
    note = (f"α = 0 is Standard LIF (no coupling), where all three coincide. Each point is an "
            f"independent network ({result.get('n_seeds','?')} per α); mean ± SEM. RM = eLIF with "
            f"baseline lowered to match LIF spike count, isolating the field effect from firing rate.")
    fig.text(0.5, 0.005, note, ha='center', fontsize=9, color='#444')
    _save(fig, 'V3_DEF_alpha_sweep', save_dir)


def _field_schematic(ax):
    from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    nx = [0.20, 0.40, 0.60, 0.80]; rN = 0.042

    def neuron(x, y):
        ax.add_patch(plt.Circle((x, y), rN, fc='#d9dde3', ec='k', lw=1.0, zorder=4))

    yT = 0.70
    ax.add_patch(FancyBboxPatch((0.37, 0.88), 0.26, 0.085, boxstyle='round,pad=0.008',
                                fc='#e8f1fb', ec=COL_TH, lw=1.3))
    ax.text(0.5, 0.9225, 'external field', ha='center', va='center', fontsize=7.6, color=COL_TH, fontweight='bold')
    for x in nx:
        ax.add_patch(FancyArrowPatch((0.5, 0.875), (x, yT + rN), arrowstyle='-|>',
                                     mutation_scale=9, color=COL_TH, lw=1.2, shrinkA=0, shrinkB=1))
        neuron(x, yT)
    ax.text(0.5, 0.585, 'EXOGENOUS  (open loop)', ha='center', fontsize=8.4, fontweight='bold', color=COL_TH)
    ax.text(0.5, 0.54, 'imposed field → entrains  (corr ↑)', ha='center', fontsize=7.4, color='#444')

    ax.plot([0.06, 0.94], [0.505, 0.505], ls=':', color='#bbb', lw=1.0)

    yB = 0.35; fc = (0.5, 0.135)
    ax.text(0.5, 0.47, 'ENDOGENOUS  (closed loop)', ha='center', fontsize=8.4, fontweight='bold', color=COL_ELIF)
    ax.text(0.5, 0.425, 'self-generated field → decorrelates  (corr ↓)', ha='center', fontsize=7.4, color='#444')
    ax.add_patch(Ellipse(fc, 0.34, 0.11, fc='#fdeef0', ec=COL_ELIF, lw=1.3, zorder=2))
    ax.text(fc[0], fc[1], 'shared field  Φ', ha='center', va='center', fontsize=8, color=COL_ELIF, fontweight='bold')
    for x in nx:
        tx = 0.5 + (x - 0.5) * 0.55
        ax.add_patch(FancyArrowPatch((x, yB - rN), (tx, fc[1] + 0.055), arrowstyle='<|-|>',
                                     mutation_scale=8, color=COL_ELIF, lw=1.1, shrinkA=1, shrinkB=1))
        neuron(x, yB)


def fig_control_parameter(defres, field, save_dir='results', fignum='2'):
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1.1, 1.0], wspace=0.36, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    _field_schematic(ax)
    ax.set_title('(a)  Same field, two regimes', loc='left')

    a = np.array(field['alphas'], float)
    series = [('std', 'Std (no field)', COL_STD, 'x', '--'),
              ('endo', 'endogenous (closed loop)', COL_ELIF, 'o', '-'),
              ('endo_rm', 'endogenous, rate-matched', COL_RM, '^', '-'),
              ('exo', 'exogenous (open loop)', COL_TH, 's', '-'),
              ('exo_rm', 'exogenous, rate-matched', '#8bb7df', 'v', '--')]
    ax = fig.add_subplot(gs[0, 1])
    for ck, lab, col, mk, ls in series:
        if ck not in field:
            continue
        d = field[ck]['noise_corr']; m = np.array(d['mean'], float); s = np.array(d['sem'], float)
        ax.fill_between(a, m - s, m + s, color=col, alpha=0.13, lw=0)
        ax.plot(a, m, mk + ls, color=col, mfc='white', label=lab, lw=1.6, ms=3, mew=0.9)
    em = np.array(field['endo']['noise_corr']['mean']); es = np.array(field['endo']['noise_corr']['sem'])
    xm = np.array(field['exo']['noise_corr']['mean']) + np.array(field['exo']['noise_corr']['sem'])
    _sweep_stars(ax, a, field['endo']['noise_corr'].get('per_seed'), field['exo']['noise_corr'].get('per_seed'),
                 np.maximum(em + es, xm) + 0.02 * (np.nanmax(xm) - np.nanmin(em) + 1e-9), color='k')
    ax.set_xlabel('α  (fraction of critical coupling)'); ax.set_ylabel('mean noise correlation')
    ax.set_title('(b)  Decorrelate vs entrain', loc='left'); ax.legend(fontsize=6.8, loc='best')

    ad = np.array(defres['alphas'], float)
    panels = [(0, 2, 'spk', 'spikes / trial', '(c)  Firing rate (RM matched)', False, 'upper left'),
              (1, 0, 'nc', 'decoding accuracy (%)', '(d)  Population decoding', True, 'lower right'),
              (1, 1, 'dim', 'dimensionality', '(e)  Effective dimensionality', False, None),
              (1, 2, 'bits_per_spike', 'bits / spike', '(f)  Coding efficiency', False, None)]
    has_exo = 'exo' in defres
    for r, c, key, ylab, title, chance, legloc in panels:
        ax = fig.add_subplot(gs[r, c])
        _band(ax, ad, defres['std'][key], COL_STD, 'LIF', 'x', ms=3, mew=1.0)
        _band(ax, ad, defres['elif'][key], COL_ELIF, 'eLIF (endo)', 'o', ms=3, mew=1.0)
        _band(ax, ad, defres['rm'][key], COL_RM, 'RM', '^', ms=3, mew=1.0)
        if has_exo:
            _band(ax, ad, defres['exo'][key], COL_TH, 'exo', 's', ms=3, mew=1.0)
            if 'exo_rm' in defres:
                _band(ax, ad, defres['exo_rm'][key], '#8bb7df', 'exo (RM)', 'v', ms=3, mew=1.0)
        ch = None
        if chance:
            ch = ax.axhline(defres['chance'], ls=':', color='gray', lw=0.9, label='chance')
        _field_stars(ax, ad, defres, key)
        ax.set_xlabel('α  (fraction of critical coupling)'); ax.set_ylabel(ylab); ax.set_title(title, loc='left')
        if legloc:
            ax.legend([ch], ['chance'], fontsize=8, loc=legloc) if ch is not None \
                else ax.legend(fontsize=6.5, loc=legloc)

    _methods(fig.add_subplot(gs[:, 3]), defres.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Ephaptic coupling as a control parameter: the self-generated (closed-loop) field '
                 'decorrelates rather than entrains, and improves coding at matched firing rate',
                 fontsize=12.5, fontweight='bold', y=1.0)
    _save(fig, 'V3_control_parameter', save_dir)


def fig_alpha_supp(result, noise=None, save_dir='results', fignum='S3'):
    a = np.array(result['alphas'], float); std = result['std']
    ns = result.get('n_seeds', '?')
    fig = plt.figure(figsize=(19, 9)); gs = fig.add_gridspec(2, 4, wspace=0.36, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0]); ax.axhline(0, color='k', lw=0.8)
    sd0 = np.array(std['nc']['mean'])
    for mdl, col, mk in (('elif', COL_ELIF, 'o'), ('rm', COL_RM, '^')):
        adv = np.array(result[mdl]['nc']['mean']) - sd0
        sem = np.array(result[mdl]['nc']['sem'])
        ax.errorbar(a, adv, yerr=sem, fmt=mk + '-', color=col, mfc='white', capsize=2,
                    label=f'{LAB[mdl].split(" ")[0]} − LIF')
    pe = result['elif']['nc'].get('per_seed'); ps = result['std']['nc'].get('per_seed')
    if pe and ps:
        adv = np.array(result['elif']['nc']['mean']) - sd0
        _sweep_stars(ax, a, pe, ps, adv + np.array(result['elif']['nc']['sem']) + 0.6, color=COL_ELIF)
    ax.set_xlabel('α'); ax.set_ylabel('Δ decoding (points)')
    ax.set_title('(a)  Decoding advantage vs LIF', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    _band(ax, a, std['lda'], COL_STD, 'LIF', 'x')
    _band(ax, a, result['elif']['lda'], COL_ELIF, 'eLIF', 'o')
    _band(ax, a, result['rm']['lda'], COL_RM, 'RM', '^')
    ax.set_xlabel('α'); ax.set_ylabel('LDA accuracy (%)')
    ax.set_title('(b)  LDA decoder agrees', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2]); ax.axhline(1.0, color='k', lw=0.8, ls=':')
    d0 = np.maximum(np.array(std['dim']['mean']), 1e-9)
    ax.plot(a, np.array(result['elif']['dim']['mean']) / d0, 'o-', color=COL_ELIF, mfc='white', label='eLIF')
    ax.plot(a, np.array(result['rm']['dim']['mean']) / d0, '^-', color=COL_RM, mfc='white', label='RM')
    ax.set_xlabel('α'); ax.set_ylabel('dimensionality / LIF')
    ax.set_title('(c)  Dimensionality expansion', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 3]); ax.axhline(1.0, color='k', lw=0.8, ls=':')
    b0 = np.maximum(np.array(std['bits_per_spike']['mean']), 1e-9)
    ax.plot(a, np.array(result['elif']['bits_per_spike']['mean']) / b0, 'o-', color=COL_ELIF, mfc='white', label='eLIF')
    ax.plot(a, np.array(result['rm']['bits_per_spike']['mean']) / b0, '^-', color=COL_RM, mfc='white', label='RM')
    ax.set_xlabel('α'); ax.set_ylabel('bits/spike / LIF')
    ax.set_title('(d)  Coding-efficiency gain', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    pe = result['elif']['nc'].get('per_seed')
    if pe:
        bp = ax.boxplot([np.asarray(x, float) for x in pe], positions=range(len(a)),
                        widths=0.6, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=COL_ELIF, alpha=0.4)
        ax.set_xticks(range(len(a))); ax.set_xticklabels([f'{x:.2f}' for x in a], rotation=45, fontsize=7)
    ax.plot(range(len(a)), np.array(std['nc']['mean']), '--', color=COL_STD, lw=1.4, label='Std')
    ax.legend(fontsize=8)
    ax.set_xlabel('α'); ax.set_ylabel('eLIF decoding (%)')
    ax.set_title(f'(e)  Per-network decoding (n={ns})', loc='left')

    if noise is not None:
        sg = np.array(noise['sigma_noise_levels'], float)
        ax = fig.add_subplot(gs[1, 1])
        for k in ('std', 'elif', 'rm'):
            _band(ax, sg, noise['nc'][k], COL[k], LAB[k], {'std': 's', 'elif': 'o', 'rm': '^'}[k])
        em = np.array(noise['nc']['elif']['mean']); es = np.array(noise['nc']['elif']['sem'])
        _sweep_stars(ax, sg, noise['nc']['elif'].get('per_seed'), noise['nc']['std'].get('per_seed'),
                     em + es + 1.5, color=COL_ELIF)
        ax.axhline(noise['chance'], ls=':', color='gray', lw=0.9)
        ax.set_xlabel('noise correlation length  σ'); ax.set_ylabel('phase decoding (%)')
        ax.set_title(f'(f)  Noise structure (α = {noise["alpha"]:.2f})', loc='left'); ax.legend(fontsize=7)

        ax = fig.add_subplot(gs[1, 2])
        adv, adv_se, pe2, ps2 = _paired_adv(noise, 'nc', 'elif', 'std')
        advr, advr_se, _, _ = _paired_adv(noise, 'nc', 'rm', 'std')
        ax.axhline(0, color='k', lw=0.8)
        ax.fill_between(sg, adv - adv_se, adv + adv_se, color=COL_ELIF, alpha=0.18, lw=0)
        ax.errorbar(sg, adv, yerr=adv_se, fmt='o-', color=COL_ELIF, mfc='white', capsize=2, label='eLIF − LIF')
        ax.errorbar(sg, advr, yerr=advr_se, fmt='^-', color=COL_RM, mfc='white', capsize=2, label='RM − LIF')
        _sweep_stars(ax, sg, pe2, ps2, adv + adv_se + 0.4, color=COL_ELIF)
        nccorr = np.array(noise['noise_corr']); ncse = np.array(noise.get('noise_corr_sem', np.zeros_like(nccorr)))
        ax2 = ax.twinx(); ax2.fill_between(sg, nccorr - ncse, nccorr + ncse, color='#bbb', alpha=0.4, lw=0)
        ax2.plot(sg, nccorr, ':', color='#888', lw=1.6); ax2.set_ylabel('noise corr (LIF)', color='#888')
        ax2.spines['top'].set_visible(False)
        ax.set_xlabel('noise correlation length  σ'); ax.set_ylabel('Δ accuracy (points)')
        ax.set_title('(g)  Benefit where corr. removable', loc='left'); ax.legend(fontsize=7, loc='upper left')

    _methods(fig.add_subplot(gs[1, 3]), result.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Figure 2 supplement: alpha-bifurcation detail and the noise-structure '
                 f'dependence of the coding benefit (supports Figure 2)',
                 fontsize=12.5, fontweight='bold', y=0.99)
    _save(fig, 'V3_DEF_alpha_sweep_supp', save_dir)


def fig_spatial_freq(result, save_dir='results', fignum='S1'):
    m = np.array(result['modes'], float)
    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    for k in ('std', 'elif', 'rm'):
        _band(ax, m, result['nc'][k], COL[k], LAB[k], {'std': 's', 'elif': 'o', 'rm': '^'}[k])
    ax.axhline(result['chance'], ls=':', color='gray', lw=0.9, label='chance')
    ax.set_xlabel('stimulus spatial mode  m  (cycles/unit)'); ax.set_ylabel('decoding accuracy (%)')
    ax.set_title(f'(a)  Phase discrimination   (α = {result["alpha"]:.2f})', loc='left'); ax.legend(ncol=1, loc='lower left')

    ax = fig.add_subplot(gs[0, 1])
    adv = np.array(result['nc']['elif']['mean']) - np.array(result['nc']['std']['mean'])
    advr = np.array(result['nc']['rm']['mean']) - np.array(result['nc']['std']['mean'])
    se = np.hypot(np.array(result['nc']['elif']['sem']), np.array(result['nc']['std']['sem']))
    ax.axhline(0, color='k', lw=0.8)
    ax.fill_between(m, adv - se, adv + se, color=COL_ELIF, alpha=0.18, lw=0)
    ax.plot(m, adv, 'o-', color=COL_ELIF, mfc='white', label='eLIF − LIF')
    ax.plot(m, advr, '^-', color=COL_RM, mfc='white', label='RM − LIF')
    ax.set_xlabel('stimulus spatial mode  m'); ax.set_ylabel('Δ accuracy (points)')
    ax.set_title('(b)  Effect ≈ 0 at all m (gain-invariant)', loc='left'); ax.legend()
    _methods(fig.add_subplot(gs[0, 2]), result.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Same-wavenumber discrimination is gain-invariant — no eLIF effect, as the theory predicts',
                 fontsize=13, fontweight='bold', y=1.02)
    _save(fig, 'V3_B_spatial_frequency', save_dir)


def _paired_adv(result, dec, a, b):
    pa = result[dec][a].get('per_seed'); pb = result[dec][b].get('per_seed')
    if pa and pb:
        d = [np.asarray(x, float) - np.asarray(y, float) for x, y in zip(pa, pb)]
        mean = np.array([di.mean() for di in d])
        sem = np.array([di.std(ddof=1) / np.sqrt(len(di)) for di in d])
        return mean, sem, pa, pb
    mean = np.array(result[dec][a]['mean']) - np.array(result[dec][b]['mean'])
    sem = np.hypot(np.array(result[dec][a]['sem']), np.array(result[dec][b]['sem']))
    return mean, sem, None, None


def fig_noise_coherence(result, save_dir='results', fignum='6'):
    sg = np.array(result['sigma_noise_levels'], float)
    ns = result.get('n_seeds', '?')
    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    for k in ('std', 'elif', 'rm'):
        _band(ax, sg, result['nc'][k], COL[k], LAB[k], {'std': 's', 'elif': 'o', 'rm': '^'}[k])
    em = np.array(result['nc']['elif']['mean']); es = np.array(result['nc']['elif']['sem'])
    _sweep_stars(ax, sg, result['nc']['elif'].get('per_seed'), result['nc']['std'].get('per_seed'),
                 em + es + 1.5, color=COL_ELIF)
    ax.axhline(result['chance'], ls=':', color='gray', lw=0.9)
    ax.set_xlabel('noise spatial correlation length  σ'); ax.set_ylabel('decoding accuracy (%)')
    ax.set_title(f'(a)  Phase decoding   (α = {result["alpha"]:.2f})', loc='left'); ax.legend(loc='upper center')

    ax = fig.add_subplot(gs[0, 1])
    adv, adv_se, pe, ps = _paired_adv(result, 'nc', 'elif', 'std')
    advr, advr_se, _, _ = _paired_adv(result, 'nc', 'rm', 'std')
    ax.axhline(0, color='k', lw=0.8)
    ax.fill_between(sg, adv - adv_se, adv + adv_se, color=COL_ELIF, alpha=0.18, lw=0)
    ax.errorbar(sg, adv, yerr=adv_se, fmt='o-', color=COL_ELIF, mfc='white', capsize=2, label='eLIF − LIF')
    ax.errorbar(sg, advr, yerr=advr_se, fmt='^-', color=COL_RM, mfc='white', capsize=2, label='RM − LIF')
    _sweep_stars(ax, sg, pe, ps, adv + adv_se + 0.4, color=COL_ELIF)
    nccorr = np.array(result['noise_corr']); ncse = np.array(result.get('noise_corr_sem', np.zeros_like(nccorr)))
    ax2 = ax.twinx()
    ax2.fill_between(sg, nccorr - ncse, nccorr + ncse, color='#bbb', alpha=0.4, lw=0)
    ax2.plot(sg, nccorr, ':', color='#888', lw=1.6, label='noise corr (LIF)')
    ax2.set_ylabel('mean noise correlation (LIF)', color='#888'); ax2.spines['top'].set_visible(False)
    ax.set_xlabel('noise spatial correlation length  σ'); ax.set_ylabel('Δ accuracy (points)')
    ax.set_title('(b)  Benefit where correlation is removable', loc='left'); ax.legend(loc='upper left')
    _methods(fig.add_subplot(gs[0, 2]), result.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Field decorrelation helps when noise carries removable spatial structure; '
                 f'the coherent-noise cost is rate-rescued (RM)   (n = {ns} seeds; paired Wilcoxon, Holm)',
                 fontsize=12.5, fontweight='bold', y=1.02)
    _save(fig, 'V3_C_noise_coherence', save_dir)


def fig_noise_coherence_supp(result, save_dir='results', fignum='S6'):
    sg = np.array(result['sigma_noise_levels'], float)
    ns = result.get('n_seeds', '?')
    fig = plt.figure(figsize=(16.5, 9))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    for k in ('std', 'elif', 'rm'):
        _band(ax, sg, result['lda'][k], COL[k], LAB[k], {'std': 's', 'elif': 'o', 'rm': '^'}[k])
    em = np.array(result['lda']['elif']['mean']); es = np.array(result['lda']['elif']['sem'])
    _sweep_stars(ax, sg, result['lda']['elif'].get('per_seed'), result['lda']['std'].get('per_seed'),
                 em + es + 1.5, color=COL_ELIF)
    ax.axhline(result['chance'], ls=':', color='gray', lw=0.9)
    ax.set_xlabel('noise σ'); ax.set_ylabel('LDA accuracy (%)')
    ax.set_title('(a)  LDA decoder (agrees with NC)', loc='left'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    nccorr = np.array(result['noise_corr']); ncse = np.array(result.get('noise_corr_sem', np.zeros_like(nccorr)))
    ax.fill_between(sg, nccorr - ncse, nccorr + ncse, color=COL_STD, alpha=0.2, lw=0)
    ax.plot(sg, nccorr, 's-', color=COL_STD, mfc='white')
    ax.set_xlabel('noise σ'); ax.set_ylabel('mean noise correlation (LIF)')
    ax.set_title('(b)  Removable correlation grows with σ', loc='left')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(sg, np.array(result['dI']), '^-', color=COL_RM, mfc='white')
    ax.set_xlabel('noise σ'); ax.set_ylabel('rate-match offset  ΔI (mV)')
    ax.set_title('(c)  RM baseline correction', loc='left')

    ax = fig.add_subplot(gs[1, 0])
    pe = result['nc']['elif'].get('per_seed'); ps = result['nc']['std'].get('per_seed')
    if pe and ps:
        d = [np.asarray(a, float) - np.asarray(b, float) for a, b in zip(pe, ps)]
        bp = ax.boxplot(d, positions=range(len(sg)), widths=0.6, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=COL_ELIF, alpha=0.4)
        ax.set_xticks(range(len(sg))); ax.set_xticklabels([f'{s:.1f}' for s in sg])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('noise σ'); ax.set_ylabel('eLIF − LIF (points)')
    ax.set_title(f'(d)  Seed-level advantage (n={ns})', loc='left')

    ax = fig.add_subplot(gs[1, 1])
    em = np.array(result['nc']['elif']['mean']); sm = np.array(result['nc']['std']['mean'])
    ratio = 100.0 * (em - sm) / np.maximum(sm, 1e-6)
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(sg, ratio, 'o-', color=COL_ELIF, mfc='white')
    ax.set_xlabel('noise σ'); ax.set_ylabel('(eLIF − LIF)/LIF  (%)')
    ax.set_title('(e)  Relative accuracy gain', loc='left')

    _methods(fig.add_subplot(gs[1, 2]), result.get('info', {}))
    fig.suptitle(f'Figure {fignum}  |  Noise-coherence supplement: decoder agreement, correlation structure, '
                 f'rate-match cost, and seed-level reliability (supports Figure 2)',
                 fontsize=12.5, fontweight='bold', y=0.98)
    _save(fig, 'V3_C_noise_coherence_supp', save_dir)
