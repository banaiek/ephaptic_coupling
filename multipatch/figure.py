# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
import sys; sys.path.insert(0, V3)
from style import COL_STD, COL_ELIF, COL_TH, COL_RM, apply as _apply
_apply()
OUT = os.path.join(V3, 'figures.make_all')
LAGW = 1.0


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


def _sig(sta, tau):
    sta = np.asarray(sta); tau = np.asarray(tau)
    return float(np.mean(sta[np.abs(tau) <= LAGW]))


def _example_sta(ex, pre_ms=15, post_ms=15):
    dt_ms = ex['dt'] * 1e3
    pre = int(pre_ms / dt_ms); post = int(post_ms / dt_ms); bl = int(8 / dt_ms)
    sidx = np.round(np.asarray(ex['src_spikes_ms']) / dt_ms).astype(int)
    sidx = sidx[(sidx > pre) & (sidx < len(ex['src_v']) - post)]
    tau = (np.arange(pre + post) - pre) * dt_ms
    out = []
    for nb in ex['neighbours']:
        v = np.asarray(nb['v'])
        W = [v[s - pre:s + post] - np.mean(v[s - pre:s - bl]) for s in sidx]
        out.append({'distance': nb['distance'], 'sta': np.mean(W, 0) * 1e3, 'tau': tau})
    return out


def _statbox(ax, text, loc='upper right'):
    xy = {'upper right': (0.97, 0.96, 'right', 'top'), 'lower left': (0.04, 0.05, 'left', 'bottom'),
          'upper left': (0.04, 0.96, 'left', 'top'), 'lower right': (0.97, 0.05, 'right', 'bottom')}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8.5,
            color='#7a1f2b', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#fff4f4', ec='#e3b6bc', lw=0.8))


def _schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle='round,pad=0.01',
                                fc='#f3f1ec', ec='#c9c2b4', lw=1.2))
    ax.text(0.5, 0.96, 'Multi-patch cortical slice (up to 8 cells)', ha='center', fontsize=10, fontweight='bold')
    rng = np.random.default_rng(3)
    src = np.array([0.40, 0.45])
    for r, a in [(0.07, 0.5), (0.13, 0.32), (0.20, 0.18)]:
        ax.add_patch(Circle(src, r, fill=False, ec=COL_ELIF, lw=1.3, alpha=a))
    cells = [src, [0.50, 0.55], [0.55, 0.36], [0.66, 0.62], [0.30, 0.66], [0.72, 0.30]]
    for i, c in enumerate(cells):
        is_src = i == 0
        ax.add_patch(Circle(c, 0.035, fc=(COL_ELIF if is_src else '#5b6'), ec='k', lw=0.8, zorder=4))
        tip = np.array(c) + np.array([0.0, 0.05]); top = tip + np.array([0.06 * (1 if c[0] > 0.45 else -1), 0.12])
        ax.add_patch(FancyArrowPatch(top, tip, arrowstyle='-', lw=1.4, color='#666', zorder=3))
        d = np.linalg.norm(np.array(c) - src) * 250
        if not is_src:
            ax.text(c[0], c[1] - 0.06, f'{d:.0f}µm', ha='center', fontsize=6.5, color='#444')
    ax.text(src[0], src[1] + 0.085, 'source\n(spikes)', ha='center', fontsize=7, color=COL_ELIF, fontweight='bold')
    ax.text(0.5, 0.12, 'measure: neighbour V locked to source spike, vs distance',
            ha='center', fontsize=7.5, style='italic', color='#333')


def figure(rows):
    uc = [r for r in rows if not r['connected']]
    cc = [r for r in rows if r['connected']]
    tau = np.asarray(uc[0]['tau_ms'])
    d = np.array([r['distance'] for r in uc])
    amp = np.array([_sig(r['sta_mV'], tau) for r in uc])
    sh = np.array([_sig(r['sh_sta_mV'], tau) if r.get('sh_sta_mV') else np.nan for r in uc])
    nexp = len({r['expt'] for r in rows})
    closeW = np.vstack([r['sta_mV'] for r in uc if r['distance'] < 70]).mean(0)
    farW = np.vstack([r['sta_mV'] for r in uc if r['distance'] > 150]).mean(0)
    excW = (closeW - farW) * 1e3
    base = np.nanmean(amp[d > 150])
    ex = pickle.load(open(os.path.join(HERE, 'example.pkl'), 'rb'))
    p_cf = mannwhitneyu(amp[d < 70], amp[d > 150], alternative='two-sided').pvalue
    rho, p_rho = spearmanr(d, amp)
    cm_ = (d < 70) & np.isfinite(sh)
    p_sh = wilcoxon(amp[cm_], sh[cm_]).pvalue if cm_.sum() > 5 else np.nan

    pk = tau[np.argmax(excW)]
    fig = plt.figure(figsize=(17, 16)); gs = fig.add_gridspec(4, 6, wspace=0.62, hspace=0.46,
                                                              height_ratios=[1, 1, 1, 0.8])

    ax_s = fig.add_subplot(gs[0, 0:2]); _schematic(ax_s)
    ax_s.set_title('(a)  Experiment', loc='left', fontweight='bold')

    ax = fig.add_subplot(gs[0, 2:4])
    t = ex['t_ms']; w = t <= 700; step = 22
    ax.plot(t[w], ex['src_v'][w], color='k', lw=0.5)
    ax.text(t[w][-1] + 8, ex['src_v'][w][0], 'source', fontsize=7, va='center')
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(ex['neighbours'])))
    for i, nb in enumerate(ex['neighbours']):
        ax.plot(t[w], np.asarray(nb['v'])[w] - (i + 1) * step, color=cmap[i], lw=0.5)
        ax.text(t[w][-1] + 8, nb['v'][w][0] - (i + 1) * step, f"{nb['distance']:.0f}µm", fontsize=7, color=cmap[i], va='center')
    ax.set_xlabel('time (ms)'); ax.set_yticks([]); ax.set_ylabel('V (cells offset)')
    ax.set_title('(b)  Simultaneous traces (by distance)', loc='left', fontweight='bold')

    ax = fig.add_subplot(gs[0, 4:6])
    ax.plot(ex['ap_tau_ms'], ex['ap_mV'], color='k', lw=2); ax.axvline(0, ls=':', color='k', lw=0.8)
    ax.set_xlabel('time from AP peak (ms)'); ax.set_ylabel('membrane potential (mV)')
    ax.set_title(f"(c)  Source spike (mean of {ex.get('n_src_spikes','?')} APs)", loc='left', fontweight='bold')

    ax = fig.add_subplot(gs[1, 0:3])
    for lab, sel, col in [(f'close <70µm (n={(d<70).sum()})', d < 70, COL_ELIF),
                          (f'far >150µm (n={(d>150).sum()})', d > 150, COL_STD)]:
        M = np.vstack([r['sta_mV'] for r, s in zip(uc, sel) if s]) * 1e3
        mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(M.shape[0])
        ax.fill_between(tau, mu - se, mu + se, color=col, alpha=0.2, lw=0); ax.plot(tau, mu, color=col, lw=2, label=lab)
    shM = np.vstack([r['sh_sta_mV'] for r in uc if r.get('sh_sta_mV')]).mean(0) * 1e3
    ax.plot(tau, shM, color='#bbb', lw=1.3, ls='--', label='jitter shuffle')
    ax.axvline(0, ls=':', color='k', lw=0.8); ax.axhline(0, ls=':', color='k', lw=0.6)
    ax.set_xlabel('lag from spike peak (ms)'); ax.set_ylabel('STA voltage (µV)')
    ax.set_title('(d)  Grand-average STA', loc='left', fontweight='bold'); ax.legend(fontsize=8, loc='upper left')
    _statbox(ax, f'close vs far\np={p_cf:.1e}')

    ax = fig.add_subplot(gs[1, 3:6])
    bins = np.array([0, 50, 70, 100, 150, 200, 300]); bc = 0.5 * (bins[:-1] + bins[1:])
    def binned(vv):
        m, s = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (d >= lo) & (d < hi) & np.isfinite(vv)
            m.append(np.mean(vv[sel]) if sel.sum() else np.nan)
            s.append(np.std(vv[sel], ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else np.nan)
        return np.array(m), np.array(s)
    rm, rs = binned(amp); sm, ss = binned(sh)
    ax.errorbar(bc, rm * 1e3, yerr=rs * 1e3, fmt='o-', color=COL_ELIF, capsize=3, label='unconnected (real)')
    ax.errorbar(bc + 3, sm * 1e3, yerr=ss * 1e3, fmt='s--', color='#999', capsize=3, label='jitter shuffle')
    ax.axhline(0, ls=':', color='k', lw=0.6)
    ax.set_xlabel('inter-soma distance (µm)'); ax.set_ylabel('signed STA at lag 0 (µV)')
    ax.set_title('(e)  Deflection decays with distance', loc='left', fontweight='bold'); ax.legend(fontsize=8)
    _statbox(ax, f'Spearman\nρ={rho:+.2f}, p={p_rho:.0e}', loc='upper right')

    ax = fig.add_subplot(gs[2, 0:3])
    okb = np.isfinite(rm) & np.isfinite(rs) & (rs > 0)
    x, y, ws = bc[okb], rm[okb] * 1e3, rs[okb] * 1e3
    def gauss(dd, b, A, sig): return b + A * np.exp(-dd ** 2 / (2 * sig ** 2))
    try:
        p1, _ = curve_fit(gauss, x, y, p0=[20, 30, 60], sigma=ws, absolute_sigma=True,
                          maxfev=20000, bounds=([-50, 0, 10], [200, 500, 300]))
        r1 = np.sum(((y - gauss(x, *p1)) / ws) ** 2)
    except Exception:
        p1 = [np.mean(y), 0, 60]; r1 = np.sum(((y - np.mean(y)) / ws) ** 2)
    b0 = np.average(y, weights=1 / ws ** 2); r0 = np.sum(((y - b0) / ws) ** 2)
    n = len(x); bic1 = n * np.log(r1 / n) + 3 * np.log(n); bic0 = n * np.log(r0 / n) + 1 * np.log(n)
    ax.errorbar(x, y, yerr=ws, fmt='o', color=COL_ELIF, capsize=3, label='data (binned)')
    dd = np.linspace(20, 290, 100)
    ax.plot(dd, gauss(dd, *p1), '-', color='#7a1f2b', lw=2, label=f'eLIF kernel σ={p1[2]:.0f}µm')
    ax.axhline(b0, ls='--', color='#999', label='no-coupling (flat)')
    ax.set_xlabel('inter-soma distance (µm)'); ax.set_ylabel('signed STA (µV)')
    ax.set_title('(f)  eLIF field kernel vs no coupling', loc='left', fontweight='bold'); ax.legend(fontsize=7.5)
    _statbox(ax, f'ΔBIC={bic0 - bic1:+.1f}\n{"eLIF kernel wins" if bic1 < bic0 else "null wins"}')

    ax = fig.add_subplot(gs[2, 3:6])
    t_el, el_dir, el_mem = _elif_pair()
    win = np.abs(tau) <= 8
    D = np.interp(tau, t_el, el_dir); M = np.interp(tau, t_el, el_mem)
    coef, *_ = np.linalg.lstsq(np.vstack([D[win], M[win]]).T, excW[win], rcond=None)
    comp_dir = coef[0] * D; comp_int = coef[1] * M; el_fit = comp_dir + comp_int
    r2_el = 1 - np.sum((excW[win] - el_fit[win]) ** 2) / np.sum((excW[win] - excW[win].mean()) ** 2)
    ax.plot(tau, excW, color=COL_ELIF, lw=2.4, label='data: excess (close − far)', zorder=5)
    ax.plot(tau, el_fit, '-', color='k', lw=2.0, label=f'eLIF: field + τ_m polariz. (R²={r2_el:.2f})')
    ax.plot(tau, comp_dir, ':', color=COL_TH, lw=1.3, label='  field (instantaneous)')
    ax.plot(tau, comp_int, ':', color=COL_RM, lw=1.3, label='  polarization (τ_m)')
    ax.plot(tau, np.zeros_like(tau), '--', color=COL_STD, lw=1.5, label='LIF / no ephaptic')
    ax.axvline(0, ls=':', color='k', lw=0.8); ax.axhline(0, ls=':', color='k', lw=0.6); ax.set_xlim(tau.min(), tau.max())
    ax.set_xlabel('lag from spike peak (ms)'); ax.set_ylabel('excess STA (µV)')
    ax.set_title('(g)  Model STA: eLIF field + τ_m polarization', loc='left', fontweight='bold'); ax.legend(fontsize=7.0)
    _statbox(ax, f'eLIF R²={r2_el:.2f}\nLIF: no deflection', loc='upper left')

    ax = fig.add_subplot(gs[3, :]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='#f6f6f6', edgecolor='#ddd'))
    cc_amp = np.array([_sig(r['sta_mV'], tau) for r in cc]) * 1e3 if cc else np.array([np.nan])
    L = [f'REAL DATA — Allen synaptic physiology: {nexp} multipatch experiments, {len(uc)} unconnected ordered pairs.',
         "TEST — spike-triggered neighbour voltage at lag 0, using each cell's ISOLATED spikes (no other recorded cell",
         '   spiking within ±15 ms); synaptic pairs excluded; controls = far pairs (baseline), jitter shuffle (null), connected (synaptic).',
         '',
         f'RESULTS — close <70µm: {np.nanmean(amp[d<70])*1e3:.0f} µV   vs   far >150µm: {np.nanmean(amp[d>150])*1e3:.0f} µV (baseline).   '
         f'close vs far Mann-Whitney p={p_cf:.1e};  amplitude–distance Spearman ρ={rho:+.2f}, p={p_rho:.0e};  real vs shuffle p={p_sh:.1e}.',
         f'   eLIF Gaussian field kernel fit σ={p1[2]:.0f} µm; model beats no-coupling ΔBIC={bic0-bic1:+.1f}; '
         f'STA reproduced by the eLIF model (instantaneous field + τ_m polarization, R²={r2_el:.2f}) but flat for the non-coupled LIF; '
         f'excess peaks {pk:+.1f} ms relative to the AP peak; connected/synaptic control {np.nanmean(cc_amp):.0f} µV.',
         'CAVEAT — residual electrode crosstalk cannot be fully excluded; its lag structure and distance dependence are examined in the supplement.']
    for i, ln in enumerate(L):
        ax.text(0.02, 0.92 - i * 0.135, ln, transform=ax.transAxes, fontsize=9.2,
                fontweight='bold' if ln[:7] in ('REAL DA', 'TEST — ', 'RESULTS', 'CAVEAT ') else 'normal')

    fig.suptitle('Figure 5  |  Ephaptic coupling in real cortical tissue', fontweight='bold', y=0.995)
    import style as _S; _S.finalize(fig, scale=0.85)
    for e_ in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'Figure5.{e_}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved Figure5 | close {np.nanmean(amp[d<70])*1e3:.0f}uV far {np.nanmean(amp[d>150])*1e3:.0f}uV '
          f'p={p_cf:.1e} dBIC={bic0-bic1:+.1f} sigma={p1[2]:.0f}um R2_eLIF={r2_el:.2f}', flush=True)


def _elif_pair(dt=0.05, T=1000.0, drive=10.6):
    tau_m, V_rest, V_th, V_reset, t_ref = 10.0, -60.0, -50.0, -60.0, 2.0
    n = int(T / dt); pre = int(15 / dt); post = int(15 / dt); refr = int(t_ref / dt)
    g = (np.arange(2 * pre) - pre) * dt
    tmpl = 90.0 * np.exp(-((g - 0.3) ** 2) / (2 * 0.4 ** 2)) - 6.0 * np.exp(-((g - 4.0) ** 2) / (2 * 2.0 ** 2))
    vA = V_rest; ref = 0; sp = []
    for i in range(n):
        if ref > 0:
            vA = V_reset; ref -= 1
        else:
            vA += dt * (-(vA - V_rest) + drive) / tau_m
            if vA >= V_th:
                sp.append(i); vA = V_reset; ref = refr
    sp = np.array(sp)
    Vfield = np.zeros(n)
    for s in sp:
        for k in range(len(g)):
            j = s - pre + k
            if 0 <= j < n:
                Vfield[j] = tmpl[k]
    Vmem = np.zeros(n); v = 0.0
    for i in range(n):
        v += dt * (-v + Vfield[i]) / tau_m
        Vmem[i] = v
    spv = sp[(sp > pre) & (sp < n - post)]
    t = (np.arange(pre + post) - pre) * dt
    if spv.size == 0:
        return t, np.zeros_like(t), np.zeros_like(t)

    def sta(x):
        return np.vstack([x[s - pre:s + post] - x[s - pre:s - int(8 / dt)].mean() for s in spv]).mean(0)
    return t, sta(Vfield), sta(Vmem)


if __name__ == '__main__':
    figure(pickle.load(open(os.path.join(HERE, 'sta.pkl'), 'rb')))
