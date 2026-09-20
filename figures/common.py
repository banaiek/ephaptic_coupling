# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300, 'figure.dpi': 110,
    'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5, 'axes.edgecolor': '#222222',
    'axes.titleweight': 'bold',
    'xtick.color': '#222222', 'ytick.color': '#222222',
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False,
    'lines.linewidth': 1.2,
})

from matplotlib import ticker as _mticker
_orig_scale_range = _mticker.scale_range
def _safe_scale_range(vmin, vmax, n=1, threshold=100):
    try:
        return _orig_scale_range(vmin, vmax, n=n, threshold=threshold)
    except ZeroDivisionError:
        return 1.0, 0.0
_mticker.scale_range = _safe_scale_range

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

COL_STD = '#4d4d4d'
COL_ELIF = '#d1495b'
COL_RM = '#2a9d8f'
COL_E = (0.30, 0.69, 0.29)
COL_I = (0.60, 0.20, 0.60)
COL_STIM = (1.0, 0.92, 0.80)
LABELS_2 = ['Standard', 'eLIF']
LABELS_3 = ['Standard', 'eLIF', 'eLIF (RM)']
COLORS_2 = [COL_STD, COL_ELIF]
COLORS_3 = [COL_STD, COL_ELIF, COL_RM]

def _save(fig, name, save_dir='.', dpi=300):
    for ax in fig.get_axes():
        try:
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            if xlim[1] - xlim[0] == 0:
                ax.set_xlim(xlim[0] - 1e-3, xlim[1] + 1e-3)
            if ylim[1] - ylim[0] == 0:
                ax.set_ylim(ylim[0] - 1e-3, ylim[1] + 1e-3)
        except Exception:
            pass

    path_png = os.path.join(save_dir, f"{name}.png")
    path_pdf = os.path.join(save_dir, f"{name}.pdf")
    def _try_save(path, **kw):
        try:
            fig.savefig(path, facecolor='white', bbox_inches='tight', **kw)
            return True
        except Exception:
            try:
                fig.savefig(path, facecolor='white', **kw)
                return True
            except Exception as e2:
                import traceback
                print(f"  WARNING: failed to save {path}: {e2}")
                traceback.print_exc(limit=25)
                return False
    ok_png = _try_save(path_png, dpi=dpi)
    ok_pdf = _try_save(path_pdf)
    if ok_png and ok_pdf:
        print(f"  Saved {path_png}  +  {path_pdf}")
    elif ok_png:
        print(f"  Saved {path_png}  (PDF save failed)")
    plt.close(fig)

def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _sig_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

def _add_sig_bracket(ax, x1, x2, y, p, h=None):
    stars = _sig_stars(p)
    if stars == 'n.s.': return
    if h is None: h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    y_bar = y + h
    ax.plot([x1, x1, x2, x2], [y_bar-h*0.3, y_bar, y_bar, y_bar-h*0.3], 'k-', lw=0.8)
    ax.text((x1+x2)/2, y_bar, stars, ha='center', va='bottom', fontsize=9, fontweight='bold')

def _bar(ax, values, labels, colors, ylabel='', title='', errs=None):
    x = np.arange(len(values))
    ax.bar(x, values, color=colors[:len(values)], width=0.6)
    if errs is not None:
        ax.errorbar(x, values, yerr=errs[:len(values)], fmt='none', ecolor='k', capsize=4, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(values)], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title: ax.set_title(title, fontsize=9.5)
    _clean(ax)

def _raster(ax, spikes, t, NE, title='', stim_range=None):
    N = spikes.shape[0]
    if stim_range:
        ax.axvspan(stim_range[0], stim_range[1], color=COL_STIM, zorder=0)
    for n in range(0, N, max(1, N // 300)):
        spk_t = t[spikes[n] > 0]
        if len(spk_t) > 0:
            c = COL_E if n < NE else COL_I
            ax.plot(spk_t, np.full_like(spk_t, n), '.', color=c, markersize=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, N)
    ax.set_ylabel('Neuron', fontsize=9)
    ax.set_title(title, fontsize=9.5)
    _clean(ax)


def _compute_vel_tang(scores, n_classes):
    sc = scores[:, :, :3]
    vel = np.diff(sc, axis=0)
    eps = 0.1
    pairs = [(s1, s2) for s1 in range(n_classes)
             for s2 in range(s1 + 1, n_classes)]
    nT = sc.shape[0]; n_p = len(pairs)
    va = np.zeros((nT - 1, n_p)); tg = np.zeros((nT - 1, n_p))
    for pi, (s1, s2) in enumerate(pairs):
        for tt in range(nT - 1):
            v1 = vel[tt, s1]; v2 = vel[tt, s2]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                va[tt, pi] = np.degrees(np.arccos(np.clip(
                    np.dot(v1, v2) / (n1 * n2), -1, 1)))
            pd_ = sc[tt, s1] - sc[tt, s2]
            vd_ = vel[tt, s1] - vel[tt, s2]
            tg[tt, pi] = np.sum(vd_ ** 2) / (np.sum(pd_ ** 2) + eps)
    return va, tg


def _draw_traj_simple(fig, gs_slice, traj, traj_rm, n_classes,
                       stim_on, stim_off, panel_letters,
                       per_trial_spikes=None):
    from scipy.stats import mannwhitneyu
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    L = panel_letters
    has_traj = traj is not None and traj.get('scores_std') is not None \
        and traj['scores_std'].shape[2] >= 3
    has_rm = traj_rm is not None and traj_rm.get('scores_elif') is not None \
        and traj_rm['scores_elif'].shape[2] >= 3

    ax3d = fig.add_subplot(gs_slice[0, 0], projection='3d')
    if has_traj:
        def _plot3(sc, col, lbl):
            for s in range(n_classes):
                ax3d.plot(sc[:, s, 0], sc[:, s, 1], sc[:, s, 2],
                          '-', color=col, lw=0.8, alpha=0.55)
            ax3d.plot([], [], [], '-', color=col, lw=1.5, label=lbl)
        _plot3(traj['scores_std'][:, :, :3], COL_STD, 'Std')
        _plot3(traj['scores_elif'][:, :, :3], COL_ELIF, 'eLIF')
        if has_rm:
            _plot3(traj_rm['scores_elif'][:, :, :3], COL_RM, 'RM')
        ax3d.set_xlabel('PC1', fontsize=8); ax3d.set_ylabel('PC2', fontsize=8)
        ax3d.set_zlabel('PC3', fontsize=8)
        ax3d.set_title(f'{L[0]}. 3D PCA (all models)', fontsize=9)
        ax3d.legend(fontsize=8, loc='upper left')
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax3d.axis('off'); ax3d.set_title(f'{L[0]}. 3D PCA (n/a)', fontsize=9)

    tang_pairs = {}; t_vel = None
    if has_traj:
        tbins = traj['time_bins']
        t_vel = tbins[:-1] + (tbins[1] - tbins[0]) / 2
        _, tang_pairs['std']  = _compute_vel_tang(traj['scores_std'],  n_classes)
        _, tang_pairs['elif'] = _compute_vel_tang(traj['scores_elif'], n_classes)
        if has_rm:
            _, tang_pairs['rm'] = _compute_vel_tang(traj_rm['scores_elif'], n_classes)

    ax = fig.add_subplot(gs_slice[0, 1])
    if t_vel is not None:
        for key, lbl, col in [('std', 'Std', COL_STD),
                              ('elif', 'eLIF', COL_ELIF),
                              ('rm', 'RM', COL_RM)]:
            if key in tang_pairs:
                v = tang_pairs[key]
                mu = v.mean(axis=1); se = v.std(axis=1) / np.sqrt(v.shape[1])
                ax.plot(t_vel, mu, '-', color=col, lw=1.5, label=lbl)
                ax.fill_between(t_vel, np.maximum(0, mu - se), mu + se,
                                color=col, alpha=0.15)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('Tangling Q', fontsize=9)
        ax.set_title(f'{L[1]}. Tangling (mean±SEM)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title(f'{L[1]}. Tangling (n/a)', fontsize=9)

    ax = fig.add_subplot(gs_slice[0, 2])
    if t_vel is not None and tang_pairs:
        stim_mask = (t_vel >= stim_on) & (t_vel < stim_off)
        if stim_mask.sum() == 0:
            stim_mask = np.ones_like(t_vel, dtype=bool)
        groups = []; lbls = []; cols = []
        for key, col in [('std', COL_STD), ('elif', COL_ELIF), ('rm', COL_RM)]:
            if key in tang_pairs:
                groups.append(tang_pairs[key][stim_mask].flatten())
                lbls.append({'std': 'Std', 'elif': 'eLIF', 'rm': 'RM'}[key])
                cols.append(col)
        bp = ax.boxplot(groups, labels=lbls, patch_artist=True, widths=0.55,
                        showfliers=False)
        for i, c in enumerate(cols):
            bp['boxes'][i].set_facecolor('none'); bp['boxes'][i].set_edgecolor(c)
            bp['medians'][i].set_color(c)
            bp['whiskers'][2*i].set_color(c); bp['whiskers'][2*i+1].set_color(c)
            bp['caps'][2*i].set_color(c); bp['caps'][2*i+1].set_color(c)
        try:
            if len(groups) >= 2:
                _, p_se = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                ymax = max(np.max(g) for g in groups) * 1.05
                _add_sig_bracket(ax, 1, 2, ymax, p_se)
            if len(groups) >= 3:
                _, p_sr = mannwhitneyu(groups[0], groups[2], alternative='two-sided')
                _add_sig_bracket(ax, 1, 3, max(np.max(g) for g in groups) * 1.18, p_sr)
        except Exception: pass
        ax.set_ylabel('Tangling Q', fontsize=9)
        ax.set_title(f'{L[2]}. Tangling BOX (stim)', fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title(f'{L[2]}. Tangling BOX (n/a)', fontsize=9)

    ax = fig.add_subplot(gs_slice[0, 3])
    if per_trial_spikes is not None and all(
            k in per_trial_spikes for k in ('std', 'elif', 'rm')):
        s_arr = np.asarray(per_trial_spikes['std']).flatten()
        e_arr = np.asarray(per_trial_spikes['elif']).flatten()
        r_arr = np.asarray(per_trial_spikes['rm']).flatten()
        means = [s_arr.mean(), e_arr.mean(), r_arr.mean()]
        sems = [s_arr.std()/np.sqrt(len(s_arr)),
                e_arr.std()/np.sqrt(len(e_arr)),
                r_arr.std()/np.sqrt(len(r_arr))]
        xp = np.arange(3)
        ax.bar(xp, means, 0.6, color=[COL_STD, COL_ELIF, COL_RM])
        ax.errorbar(xp, means, yerr=sems, fmt='none', ecolor='k',
                    capsize=3, lw=1)
        ax.set_xticks(xp); ax.set_xticklabels(['Std', 'eLIF', 'RM'], fontsize=9)
        ax.set_ylabel('Spikes / trial', fontsize=9)
        ax.set_title(f'{L[3]}. Firing rate (mean±SEM)', fontsize=9)
        try:
            _, p_se = mannwhitneyu(s_arr, e_arr, alternative='two-sided')
            _, p_sr = mannwhitneyu(s_arr, r_arr, alternative='two-sided')
            ymax = max(means) * 1.05
            _add_sig_bracket(ax, 1, 2, ymax, p_se)
            _add_sig_bracket(ax, 1, 3, max(means) * 1.18, p_sr)
            ax.text(0.02, 0.97,
                    f'S-E p={p_se:.2g} {_sig_stars(p_se)}\n'
                    f'S-R p={p_sr:.2g} {_sig_stars(p_sr)}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=8)
        except Exception: pass
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title(f'{L[3]}. Firing rate (n/a)', fontsize=9)


def _draw_traj_dynamics(fig, gs_slice, traj, traj_rm, n_classes,
                         stim_on, stim_off, panel_letters,
                         title_prefix=''):
    from scipy.stats import mannwhitneyu
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    sc_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    has_traj = traj is not None and traj.get('scores_std') is not None \
        and traj['scores_std'].shape[2] >= 2
    has_rm = traj_rm is not None and traj_rm.get('scores_elif') is not None \
        and traj_rm['scores_elif'].shape[2] >= 2

    def _traj_2d(ax, sc, title):
        for s in range(n_classes):
            ax.plot(sc[:, s, 0], sc[:, s, 1], '-', color=sc_colors[s],
                    lw=1.2, alpha=0.8)
            ax.plot(sc[0, s, 0], sc[0, s, 1], 'o', color=sc_colors[s], ms=5)
            ax.plot(sc[-1, s, 0], sc[-1, s, 1], 's', color=sc_colors[s], ms=5)
        ax.set_xlabel('PC1', fontsize=9); ax.set_ylabel('PC2', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        _clean(ax)

    L = panel_letters
    ax = fig.add_subplot(gs_slice[0, 0])
    if has_traj:
        _traj_2d(ax, traj['scores_std'][:, :, :2],
                 f'{L[0]}. {title_prefix}Std PC1-2')
    else:
        ax.axis('off'); ax.set_title(f'{L[0]}. Std PC1-2 (n/a)', fontsize=9)

    ax = fig.add_subplot(gs_slice[0, 1])
    if has_traj:
        _traj_2d(ax, traj['scores_elif'][:, :, :2],
                 f'{L[1]}. {title_prefix}eLIF PC1-2')
    else:
        ax.axis('off'); ax.set_title(f'{L[1]}. eLIF PC1-2 (n/a)', fontsize=9)

    ax = fig.add_subplot(gs_slice[0, 2])
    if has_rm:
        _traj_2d(ax, traj_rm['scores_elif'][:, :, :2],
                 f'{L[2]}. {title_prefix}RM PC1-2')
    else:
        ax.axis('off'); ax.set_title(f'{L[2]}. RM PC1-2 (n/a)', fontsize=9)

    ax3d = fig.add_subplot(gs_slice[0, 3], projection='3d')
    if has_traj and traj['scores_std'].shape[2] >= 3:
        def _plot3(sc, col, lbl):
            for s in range(n_classes):
                ax3d.plot(sc[:, s, 0], sc[:, s, 1], sc[:, s, 2],
                          '-', color=col, lw=0.8, alpha=0.55)
            ax3d.plot([], [], [], '-', color=col, lw=1.5, label=lbl)
        _plot3(traj['scores_std'][:, :, :3], COL_STD, 'Std')
        _plot3(traj['scores_elif'][:, :, :3], COL_ELIF, 'eLIF')
        if has_rm and traj_rm['scores_elif'].shape[2] >= 3:
            _plot3(traj_rm['scores_elif'][:, :, :3], COL_RM, 'RM')
        ax3d.set_xlabel('PC1', fontsize=8); ax3d.set_ylabel('PC2', fontsize=8)
        ax3d.set_zlabel('PC3', fontsize=8)
        ax3d.set_title(f'{L[3]}. {title_prefix}3D PCA', fontsize=9)
        ax3d.legend(fontsize=8, loc='upper left')
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax3d.axis('off'); ax3d.set_title(f'{L[3]}. 3D PCA (n/a)', fontsize=9)

    va_pairs = {}; tang_pairs = {}; t_vel = None
    if has_traj and traj['scores_std'].shape[2] >= 3:
        tbins = traj['time_bins']
        t_vel = tbins[:-1] + (tbins[1] - tbins[0]) / 2
        va_pairs['std'],  tang_pairs['std']  = _compute_vel_tang(traj['scores_std'], n_classes)
        va_pairs['elif'], tang_pairs['elif'] = _compute_vel_tang(traj['scores_elif'], n_classes)
        if has_rm and traj_rm['scores_elif'].shape[2] >= 3:
            va_pairs['rm'], tang_pairs['rm'] = _compute_vel_tang(traj_rm['scores_elif'], n_classes)

    ax = fig.add_subplot(gs_slice[1, 0])
    if t_vel is not None:
        for key, lbl, col in [('std', 'Std', COL_STD),
                              ('elif', 'eLIF', COL_ELIF),
                              ('rm', 'RM', COL_RM)]:
            if key in va_pairs:
                v = va_pairs[key]
                mu = v.mean(axis=1); se = v.std(axis=1) / np.sqrt(v.shape[1])
                ax.plot(t_vel, mu, '-', color=col, lw=1.5, label=lbl)
                ax.fill_between(t_vel, mu - se, mu + se, color=col, alpha=0.15)
        ax.axhline(90, ls='--', color='k', lw=0.6)
        ax.set_xlabel('Time (ms)', fontsize=9); ax.set_ylabel('Angle (deg)', fontsize=9)
        ax.set_title(f'{L[4]}. Vel divergence (mean±SEM)', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title(f'{L[4]}. Vel divergence (n/a)', fontsize=9)

    ax = fig.add_subplot(gs_slice[1, 1])
    if t_vel is not None:
        for key, lbl, col in [('std', 'Std', COL_STD),
                              ('elif', 'eLIF', COL_ELIF),
                              ('rm', 'RM', COL_RM)]:
            if key in tang_pairs:
                v = tang_pairs[key]
                mu = v.mean(axis=1); se = v.std(axis=1) / np.sqrt(v.shape[1])
                ax.plot(t_vel, mu, '-', color=col, lw=1.5, label=lbl)
                ax.fill_between(t_vel, np.maximum(0, mu - se), mu + se,
                                color=col, alpha=0.15)
        ax.set_xlabel('Time (ms)', fontsize=9); ax.set_ylabel('Tangling Q', fontsize=9)
        ax.set_title(f'{L[5]}. Tangling (mean±SEM)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title(f'{L[5]}. Tangling (n/a)', fontsize=9)

    def _box_stim(ax, pairs_dict, ylab, title):
        if t_vel is None or not pairs_dict:
            ax.axis('off'); ax.set_title(title + ' (n/a)', fontsize=9); return
        stim_mask = (t_vel >= stim_on) & (t_vel < stim_off)
        if stim_mask.sum() == 0:
            stim_mask = np.ones_like(t_vel, dtype=bool)
        groups = []; lbls = []; cols = []
        for key, col in [('std', COL_STD), ('elif', COL_ELIF), ('rm', COL_RM)]:
            if key in pairs_dict:
                groups.append(pairs_dict[key][stim_mask].flatten())
                lbls.append({'std': 'Std', 'elif': 'eLIF', 'rm': 'RM'}[key])
                cols.append(col)
        bp = ax.boxplot(groups, labels=lbls, patch_artist=True, widths=0.55,
                        showfliers=False)
        for i, c in enumerate(cols):
            bp['boxes'][i].set_facecolor('none'); bp['boxes'][i].set_edgecolor(c)
            bp['medians'][i].set_color(c)
            bp['whiskers'][2*i].set_color(c); bp['whiskers'][2*i+1].set_color(c)
            bp['caps'][2*i].set_color(c); bp['caps'][2*i+1].set_color(c)
        try:
            if len(groups) >= 2:
                _, p_se = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                ymax = max(np.max(g) for g in groups) * 1.05
                _add_sig_bracket(ax, 1, 2, ymax, p_se)
            if len(groups) >= 3:
                _, p_sr = mannwhitneyu(groups[0], groups[2], alternative='two-sided')
                ymax2 = max(np.max(g) for g in groups) * 1.18
                _add_sig_bracket(ax, 1, 3, ymax2, p_sr)
        except Exception:
            pass
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _clean(ax)

    _box_stim(fig.add_subplot(gs_slice[1, 2]), va_pairs, 'Angle (deg)',
              f'{L[6]}. Vel div BOX (stim)')
    _box_stim(fig.add_subplot(gs_slice[1, 3]), tang_pairs, 'Tangling Q',
              f'{L[7]}. Tangling BOX (stim)')

def fig4_decoding(result, save_dir='.'):
    from scipy.stats import mannwhitneyu
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    m=result.metrics; d=result.data; NS=d['confusion_std'].shape[0]; NT=d['counts_std'].shape[1]; has_rm='accuracy_elif_rate_matched' in m
    traj=d.get('trajectories')

    for mode in ['2way','3way']:
        if mode=='3way' and not has_rm: continue
        if mode=='3way':
            _fig4_decoding_3way(result, save_dir, mode)
            continue
        fig=plt.figure(figsize=(20,16))
        L=LABELS_3 if mode=='3way' else LABELS_2; C=COLORS_3 if mode=='3way' else COLORS_2

        ax=fig.add_subplot(4,4,1); im=ax.imshow(d['confusion_std']/NT,cmap='gray_r',vmin=0,vmax=1); ax.set_title(f"Std ({m['accuracy_std']:.1f}%)",fontsize=9); ax.set_xlabel('Pred',fontsize=9); ax.set_ylabel('True',fontsize=9); plt.colorbar(im,ax=ax,fraction=0.046)
        ax=fig.add_subplot(4,4,2); im=ax.imshow(d['confusion_elif']/NT,cmap='gray_r',vmin=0,vmax=1); ax.set_title(f"eLIF ({m['accuracy_elif']:.1f}%)",fontsize=9); ax.set_xlabel('Pred',fontsize=9); plt.colorbar(im,ax=ax,fraction=0.046)
        sc_colors=plt.cm.tab10(np.linspace(0,1,NS))
        if mode=='3way':
            conf_rm=d.get('confusion_elif_rm')
            if conf_rm is not None:
                ax=fig.add_subplot(4,4,3); im=ax.imshow(conf_rm/NT,cmap='gray_r',vmin=0,vmax=1); ax.set_title(f"RM ({m.get('accuracy_elif_rate_matched',0):.1f}%)",fontsize=9); ax.set_xlabel('Pred',fontsize=9); plt.colorbar(im,ax=ax,fraction=0.046)
            else: fig.add_subplot(4,4,3).axis('off')
            if traj is not None:
                ax3d=fig.add_subplot(4,4,4,projection='3d'); sc=traj['scores_elif']
                for s in range(min(NS,sc.shape[1])):
                    if sc.shape[2]>=3: ax3d.plot(sc[:,s,0],sc[:,s,1],sc[:,s,2],'-',color=sc_colors[s],lw=1.5); ax3d.scatter(sc[0,s,0],sc[0,s,1],sc[0,s,2],color=sc_colors[s],s=30,marker='o')
                ax3d.set_xlabel('PC1',fontsize=9); ax3d.set_ylabel('PC2',fontsize=9); ax3d.set_zlabel('PC3',fontsize=9); ax3d.set_title('eLIF PCA',fontsize=9)
            else: fig.add_subplot(4,4,4).axis('off')
        else:
            if traj is not None:
                for key,panel in [('scores_std',3),('scores_elif',4)]:
                    ax3d=fig.add_subplot(4,4,panel,projection='3d'); sc=traj[key]
                    for s in range(min(NS,sc.shape[1])):
                        if sc.shape[2]>=3: ax3d.plot(sc[:,s,0],sc[:,s,1],sc[:,s,2],'-',color=sc_colors[s],lw=1.5); ax3d.scatter(sc[0,s,0],sc[0,s,1],sc[0,s,2],color=sc_colors[s],s=30,marker='o')
                    ax3d.set_xlabel('PC1',fontsize=9); ax3d.set_ylabel('PC2',fontsize=9); ax3d.set_zlabel('PC3',fontsize=9)
                    ax3d.set_title('Std PCA' if 'std' in key else 'eLIF PCA',fontsize=9)
            else: fig.add_subplot(4,4,3).axis('off'); fig.add_subplot(4,4,4).axis('off')

        if mode=='2way': av=[m['accuracy_std'],m['accuracy_elif']]; lv=[m['lda_accuracy_std'],m['lda_accuracy_elif']]
        else: av=[m['accuracy_std'],m['accuracy_elif'],m['accuracy_elif_rate_matched']]; lv=[m['lda_accuracy_std'],m['lda_accuracy_elif'],m['lda_accuracy_elif_rate_matched']]
        ax=fig.add_subplot(4,4,5); _bar(ax,av,L,C,'%','Nearest Centroid'); ax.axhline(m['chance_level'],ls='--',color='k',lw=0.8)
        ax=fig.add_subplot(4,4,6); _bar(ax,lv,L,C,'%','LDA (5-fold CV)'); ax.axhline(m['chance_level'],ls='--',color='k',lw=0.8)

        ax=fig.add_subplot(4,4,7)
        mask_upper=np.triu(np.ones((NS,NS),dtype=bool),k=1)
        dp_s=d['dprime_std'][mask_upper]; dp_e=d['dprime_elif'][mask_upper]
        from scipy.stats import wilcoxon
        if mode=='3way':
            dp_rm_vals = d.get('dprime_elif_rm')
            if dp_rm_vals is not None and hasattr(dp_rm_vals, '__len__'):
                dp_r = dp_rm_vals[mask_upper] if dp_rm_vals.shape == d['dprime_std'].shape else dp_e * (m.get('dprime_elif_rate_matched',1) / max(m['mean_dprime_elif'],0.01))
            else:
                ratio = m.get('dprime_elif_rate_matched', m['mean_dprime_elif']) / max(m['mean_dprime_elif'], 0.01)
                dp_r = dp_e * ratio
            box_data=[dp_s, dp_e, dp_r]; box_labels=['Std','eLIF','eLIF(RM)']; box_colors=[COL_STD,COL_ELIF,COL_RM]
        else:
            box_data=[dp_s, dp_e]; box_labels=['Std','eLIF']; box_colors=[COL_STD,COL_ELIF]
        bp=ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
        for i,c in enumerate(box_colors):
            bp['boxes'][i].set_facecolor('none'); bp['boxes'][i].set_edgecolor(c); bp['medians'][i].set_color(c)
            bp['whiskers'][2*i].set_color(c); bp['whiskers'][2*i+1].set_color(c)
            bp['caps'][2*i].set_color(c); bp['caps'][2*i+1].set_color(c)
        from scipy.stats import mannwhitneyu
        try: _,pval_se=mannwhitneyu(dp_s,dp_e,alternative='two-sided')
        except: pval_se=1.0
        ymax=max(np.max(dp_e),np.max(dp_s))*1.1
        _add_sig_bracket(ax,1,2,ymax,pval_se)
        if mode=='3way' and len(box_data)==3:
            try: _,pval_sr=mannwhitneyu(dp_s,dp_r,alternative='two-sided')
            except: pval_sr=1.0
            _add_sig_bracket(ax,1,3,ymax*1.12,pval_sr)
        ax.set_ylabel("d'",fontsize=9)
        ax.set_title(f"d': Std={m['mean_dprime_std']:.2f} vs eLIF={m['mean_dprime_elif']:.2f}",fontsize=9)
        _clean(ax)

        ax=fig.add_subplot(4,4,8)
        nc_s=d.get('noise_corr_std',np.array([])); nc_e=d.get('noise_corr_elif',np.array([]))
        if len(nc_s)>0:
            ns=nc_s[np.isfinite(nc_s)]; ne=nc_e[np.isfinite(nc_e)]
            ax.hist(ns,bins=30,color=COL_STD,alpha=0.5,label=f"Std ({m['noise_corr_std']:.3f})",density=True)
            ax.hist(ne,bins=30,color=COL_ELIF,alpha=0.5,label=f"eLIF ({m['noise_corr_elif']:.3f})",density=True)
            if mode=='3way':
                nc_rm_data=d.get('noise_corr_elif_rm',None)
                if nc_rm_data is not None:
                    nr=nc_rm_data[np.isfinite(nc_rm_data)] if hasattr(nc_rm_data,'__len__') else np.array([])
                    if len(nr)>0:
                        ax.hist(nr,bins=30,color=COL_RM,alpha=0.5,label=f"RM ({m.get('noise_corr_elif_rate_matched',0):.3f})",density=True)
            ax.axvline(0,ls='--',color='k',lw=0.5); ax.legend(fontsize=9)
        ax.set_title('Noise Corr Distribution',fontsize=9); ax.set_xlabel('r'); _clean(ax)

        if traj is not None and traj['scores_std'].shape[2]>=3:
            sc_std=traj['scores_std'][:,:,:3]; sc_elif=traj['scores_elif'][:,:,:3]
            tbins=traj['time_bins']; nT=len(tbins)
            traj_rm=d.get('trajectories_rm')
            has_rm_traj = traj_rm is not None and traj_rm['scores_elif'].shape[2]>=3
            if has_rm_traj:
                sc_rm=traj_rm['scores_elif'][:,:,:3]

            def compute_vel_tang_pairs(sc, NS, nT):
                vel=np.diff(sc,axis=0); eps=0.1
                pairs = [(s1,s2) for s1 in range(NS) for s2 in range(s1+1,NS)]
                n_p = len(pairs)
                va_all=np.zeros((nT-1,n_p)); tang_all=np.zeros((nT-1,n_p))
                for pi,(s1,s2) in enumerate(pairs):
                    for tt in range(nT-1):
                        v1=vel[tt,s1]; v2=vel[tt,s2]; n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
                        if n1>1e-6 and n2>1e-6:
                            va_all[tt,pi]=np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)))
                        pd=sc[tt,s1]-sc[tt,s2]; vd=vel[tt,s1]-vel[tt,s2]
                        tang_all[tt,pi]=np.sum(vd**2)/(np.sum(pd**2)+eps)
                return va_all, tang_all

            va_pairs_std, tang_pairs_std = compute_vel_tang_pairs(sc_std, NS, nT)
            va_pairs_elif, tang_pairs_elif = compute_vel_tang_pairs(sc_elif, NS, nT)
            if has_rm_traj:
                va_pairs_rm, tang_pairs_rm = compute_vel_tang_pairs(sc_rm, NS, nT)
            t_vel=tbins[:-1]+(tbins[1]-tbins[0])/2

            ax=fig.add_subplot(4,4,9)
            va_m_s=va_pairs_std.mean(axis=1); va_se_s=va_pairs_std.std(axis=1)/max(1,np.sqrt(va_pairs_std.shape[1]))
            va_m_e=va_pairs_elif.mean(axis=1); va_se_e=va_pairs_elif.std(axis=1)/max(1,np.sqrt(va_pairs_elif.shape[1]))
            ax.plot(t_vel,va_m_s,'-',color=COL_STD,lw=2,label='Std')
            ax.fill_between(t_vel,va_m_s-va_se_s,va_m_s+va_se_s,color=COL_STD,alpha=0.15)
            ax.plot(t_vel,va_m_e,'-',color=COL_ELIF,lw=2,label='eLIF')
            ax.fill_between(t_vel,va_m_e-va_se_e,va_m_e+va_se_e,color=COL_ELIF,alpha=0.15)
            if mode=='3way' and has_rm_traj:
                va_m_r=va_pairs_rm.mean(axis=1); va_se_r=va_pairs_rm.std(axis=1)/max(1,np.sqrt(va_pairs_rm.shape[1]))
                ax.plot(t_vel,va_m_r,'-',color=COL_RM,lw=2,label='eLIF(RM)')
                ax.fill_between(t_vel,va_m_r-va_se_r,va_m_r+va_se_r,color=COL_RM,alpha=0.15)
            ax.axhline(90,ls='--',color='k',lw=0.8)
            ax.set_xlabel('Time (ms)',fontsize=9); ax.set_ylabel('Angle (deg)',fontsize=9)
            ax.set_title('Velocity Divergence (mean\u00B1SEM)',fontsize=9); ax.legend(fontsize=9); _clean(ax)

            ax=fig.add_subplot(4,4,10)
            tg_m_s=tang_pairs_std.mean(axis=1); tg_se_s=tang_pairs_std.std(axis=1)/max(1,np.sqrt(tang_pairs_std.shape[1]))
            tg_m_e=tang_pairs_elif.mean(axis=1); tg_se_e=tang_pairs_elif.std(axis=1)/max(1,np.sqrt(tang_pairs_elif.shape[1]))
            ax.plot(t_vel,tg_m_s,'-',color=COL_STD,lw=2,label='Std')
            ax.fill_between(t_vel,np.maximum(0,tg_m_s-tg_se_s),tg_m_s+tg_se_s,color=COL_STD,alpha=0.15)
            ax.plot(t_vel,tg_m_e,'-',color=COL_ELIF,lw=2,label='eLIF')
            ax.fill_between(t_vel,np.maximum(0,tg_m_e-tg_se_e),tg_m_e+tg_se_e,color=COL_ELIF,alpha=0.15)
            if mode=='3way' and has_rm_traj:
                tg_m_r=tang_pairs_rm.mean(axis=1); tg_se_r=tang_pairs_rm.std(axis=1)/max(1,np.sqrt(tang_pairs_rm.shape[1]))
                ax.plot(t_vel,tg_m_r,'-',color=COL_RM,lw=2,label='eLIF(RM)')
                ax.fill_between(t_vel,np.maximum(0,tg_m_r-tg_se_r),tg_m_r+tg_se_r,color=COL_RM,alpha=0.15)
            ax.set_xlabel('Time (ms)',fontsize=9); ax.set_ylabel('Tangling (Q)',fontsize=9)
            ax.set_title('Trajectory Tangling (mean\u00B1SEM)',fontsize=9); ax.legend(fontsize=9); _clean(ax)

            ax=fig.add_subplot(4,4,11)
            for s in range(NS):
                ax.plot(sc_std[:,s,0],sc_std[:,s,1],'-',color=sc_colors[s],lw=1.5,alpha=0.7)
                ax.plot(sc_std[0,s,0],sc_std[0,s,1],'o',color=sc_colors[s],ms=6)
                ax.plot(sc_std[-1,s,0],sc_std[-1,s,1],'s',color=sc_colors[s],ms=6)
            ax.set_xlabel('PC1',fontsize=9); ax.set_ylabel('PC2',fontsize=9)
            sep=traj.get('separation',{})
            ax.set_title(f'Std Traj PC1-2 (sep={sep.get("std",0):.1f})',fontsize=9); _clean(ax)

            ax=fig.add_subplot(4,4,12)
            for s in range(NS):
                ax.plot(sc_elif[:,s,0],sc_elif[:,s,1],'-',color=sc_colors[s],lw=1.5,alpha=0.7)
                ax.plot(sc_elif[0,s,0],sc_elif[0,s,1],'o',color=sc_colors[s],ms=6)
                ax.plot(sc_elif[-1,s,0],sc_elif[-1,s,1],'s',color=sc_colors[s],ms=6)
            ax.set_xlabel('PC1',fontsize=9); ax.set_ylabel('PC2',fontsize=9)
            ax.set_title(f'eLIF Traj PC1-2 (sep={sep.get("elif",0):.1f})',fontsize=9); _clean(ax)
        else:
            for p in [9,10,11,12]: fig.add_subplot(4,4,p).axis('off')

        if mode=='2way': nv=[m['noise_corr_std'],m['noise_corr_elif']]; sv=[m['signal_corr_std'],m['signal_corr_elif']]; spv=[m['spikes_per_trial_std'],m['spikes_per_trial_elif']]
        else: nv=[m['noise_corr_std'],m['noise_corr_elif'],m['noise_corr_elif_rate_matched']]; sv=[m['signal_corr_std'],m['signal_corr_elif'],m['signal_corr_elif']]; spv=[m['spikes_per_trial_std'],m['spikes_per_trial_elif'],m.get('spikes_rate_matched',0)]

        ax=fig.add_subplot(4,4,13); _bar(ax,nv,L,C,'r','Mean Noise Corr')

        ax=fig.add_subplot(4,4,14)
        if traj is not None and 'va_pairs_std' in dir():
            va_mean_s=va_pairs_std.mean(); va_se_s=va_pairs_std.mean(axis=1).std()/max(1,np.sqrt(nT-1))
            va_mean_e=va_pairs_elif.mean(); va_se_e=va_pairs_elif.mean(axis=1).std()/max(1,np.sqrt(nT-1))
            x_pos=np.arange(2); vals=[va_mean_s,va_mean_e]; errs=[va_se_s,va_se_e]; cols_b=[COL_STD,COL_ELIF]; lbls=['Std','eLIF']
            if mode=='3way' and has_rm_traj:
                va_mean_r=va_pairs_rm.mean(); va_se_r=va_pairs_rm.mean(axis=1).std()/max(1,np.sqrt(nT-1))
                x_pos=np.arange(3); vals.append(va_mean_r); errs.append(va_se_r); cols_b.append(COL_RM); lbls.append('RM')
            ax.bar(x_pos,vals,color=cols_b,width=0.6)
            ax.errorbar(x_pos,vals,yerr=errs,fmt='none',ecolor='k',capsize=4,lw=1.5)
            ax.set_xticks(x_pos); ax.set_xticklabels(lbls,fontsize=9)
            ax.set_ylabel('Angle (deg)',fontsize=9); ax.set_title('Mean Vel. Divergence',fontsize=9)
        else:
            _bar(ax,sv,L,C,'r','Signal Corr')
        _clean(ax)

        ax=fig.add_subplot(4,4,15)
        if traj is not None and 'tang_pairs_std' in dir():
            tg_mean_s=tang_pairs_std.mean(); tg_se_s=tang_pairs_std.mean(axis=1).std()/max(1,np.sqrt(nT-1))
            tg_mean_e=tang_pairs_elif.mean(); tg_se_e=tang_pairs_elif.mean(axis=1).std()/max(1,np.sqrt(nT-1))
            x_pos=np.arange(2); vals=[tg_mean_s,tg_mean_e]; errs=[tg_se_s,tg_se_e]; cols_b=[COL_STD,COL_ELIF]; lbls=['Std','eLIF']
            if mode=='3way' and has_rm_traj:
                tg_mean_r=tang_pairs_rm.mean(); tg_se_r=tang_pairs_rm.mean(axis=1).std()/max(1,np.sqrt(nT-1))
                x_pos=np.arange(3); vals.append(tg_mean_r); errs.append(tg_se_r); cols_b.append(COL_RM); lbls.append('RM')
            ax.bar(x_pos,vals,color=cols_b,width=0.6)
            ax.errorbar(x_pos,vals,yerr=errs,fmt='none',ecolor='k',capsize=4,lw=1.5)
            ax.set_xticks(x_pos); ax.set_xticklabels(lbls,fontsize=9)
            ax.set_ylabel('Tangling (Q)',fontsize=9); ax.set_title('Mean Tangling',fontsize=9)
        else:
            _bar(ax,spv,L,C,'Spikes','Spikes/Trial')
        _clean(ax)

        ax=fig.add_subplot(4,4,16); ax.set_xlim(0,400); ax.set_ylim(0,4); ax.axis('off')
        ax.set_title('Decoding Task Design',fontsize=9.5,fontweight='bold')
        ax.axvspan(0,100,color='#f0f0f0',transform=ax.transData); ax.axvspan(100,300,color=COL_STIM,transform=ax.transData)
        ax.text(200,3.5,f'{NS} stimuli, {NT} trials each',ha='center',fontsize=9)
        ax.text(50,2.8,'Baseline',ha='center',fontsize=9)
        ax.text(200,2.8,'Stimulus',ha='center',fontsize=9,fontweight='bold')
        ax.text(200,2.0,'Shared drive (all neurons)',ha='center',fontsize=9,color='gray')
        ax.text(200,1.3,'+ tuning modulation (small)',ha='center',fontsize=9,color='blue')
        ax.text(200,0.6,'+ shared noise + private noise',ha='center',fontsize=9,color='red')
        ax.text(350,2.8,'Post',ha='center',fontsize=9)

        sf='_3way' if mode=='3way' else ''
        fig.suptitle(f'Figure 4: Stimulus Decoding ({NS} stim x {NT} trials) [{mode}]',fontsize=13,fontweight='bold')
        fig.tight_layout(rect=[0,0,1,0.96]); _save(fig,f'Fig4_Decoding{sf}',save_dir)

    if d.get('I_sweep') is not None or d.get('conn_sweep') is not None:
        fig_decoding_sweeps(result, save_dir)

def fig_decoding_sweeps(result, save_dir='.'):
    d = result.data
    chance = result.metrics.get('chance_level', 10.0)
    sweeps = [('I_sweep', d.get('I_sweep'), 'Baseline current I (mV)'),
              ('conn_sweep', d.get('conn_sweep'), 'Connection probability p$_{conn}$')]
    sweeps = [(k, s, x) for k, s, x in sweeps if s is not None]
    if not sweeps:
        return
    nrows = len(sweeps)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.6 * nrows), squeeze=False)
    mcolors = {'std': COL_STD, 'elif': COL_ELIF, 'rm': COL_RM}
    mlabels = {'std': 'Std', 'elif': 'eLIF', 'rm': 'RM'}
    for ri, (key, swp, xlabel) in enumerate(sweeps):
        x = np.array(swp['values'], dtype=float)
        ax = axes[ri][0]
        for mk in ('std', 'elif', 'rm'):
            ax.plot(x, swp['nc'][mk], '-o', color=mcolors[mk], label=mlabels[mk], lw=2, ms=5)
        ax.axhline(chance, ls='--', color='k', lw=0.8, label=f'chance ({chance:.0f}%)')
        ax.set_ylabel('Accuracy (%)'); ax.set_title('Nearest-centroid accuracy', fontsize=10)
        ax.set_xlabel(xlabel); ax.legend(fontsize=9); _clean(ax)
        ax = axes[ri][1]
        for mk in ('std', 'elif', 'rm'):
            ax.plot(x, swp['spk'][mk], '-o', color=mcolors[mk], label=mlabels[mk], lw=2, ms=5)
        ax.set_ylabel('Spikes / trial'); ax.set_title('Firing (RM matched to Std)', fontsize=10)
        ax.set_xlabel(xlabel); _clean(ax)
    fig.suptitle('Figure 4 (suppl.): Nearest-centroid accuracy vs baseline current & connectivity',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'Fig4_Decoding_Sweeps', save_dir)

def _fig4_decoding_3way(result, save_dir, mode):
    from scipy.stats import mannwhitneyu
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    m = result.metrics; d = result.data
    NS = d['confusion_std'].shape[0]; NT = d['counts_std'].shape[1]
    traj = d.get('trajectories'); traj_rm = d.get('trajectories_rm')
    sc_colors = plt.cm.tab10(np.linspace(0, 1, NS))
    stim_labels = [f'S{i}' for i in range(NS)]
    stim_on = d.get('stim_onset', 100.0); stim_off = d.get('stim_offset', 300.0)

    fig = plt.figure(figsize=(17, 22))
    gs = fig.add_gridspec(7, 5, hspace=0.95, wspace=0.55)

    def _ticks(ax, nx=4, ny=4, dec=2):
        ax.xaxis.set_major_locator(MaxNLocator(nx))
        ax.yaxis.set_major_locator(MaxNLocator(ny))
        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{dec}f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{dec}f'))

    def _cbar_fmt(cb, n=4, dec=2):
        cb.ax.yaxis.set_major_locator(MaxNLocator(n))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{dec}f'))

    def _plot_conf(ax, C_raw, title):
        if C_raw is None:
            ax.axis('off'); ax.set_title(title, fontsize=9); return
        rs = C_raw.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        M = C_raw / rs
        im = ax.imshow(M, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
        tick_idx = np.linspace(0, NS - 1, min(NS, 4)).astype(int)
        ax.set_xticks(tick_idx); ax.set_yticks(tick_idx)
        ax.set_xticklabels([stim_labels[i] for i in tick_idx], fontsize=8)
        ax.set_yticklabels([stim_labels[i] for i in tick_idx], fontsize=8)
        ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
        ax.set_title(title, fontsize=9)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); _cbar_fmt(cb)

    def _whole_raster(ax, spikes, NE, title):
        if spikes is None:
            ax.axis('off'); ax.set_title(title + ' (n/a)', fontsize=9); return
        N, T = spikes.shape
        t_full = d.get('t_trial')
        t_ms = np.asarray(t_full) if t_full is not None and len(t_full) == T \
               else np.arange(T) * 0.1
        ax.axvspan(stim_on, stim_off, color=COL_STIM, alpha=0.5, zorder=0,
                   label='Stim')
        step = max(1, N // 200)
        for n in range(0, N, step):
            st = t_ms[spikes[n] > 0]
            if len(st) > 0:
                col = COL_E if n < NE else COL_I
                ax.plot(st, np.full_like(st, n), '|', color=col,
                        markersize=1.2, markeredgewidth=0.5)
        ax.set_xlim(0, t_ms[-1]); ax.set_ylim(0, N)
        ax.set_xlabel('Time (ms)', fontsize=9); ax.set_ylabel('Neuron', fontsize=9)
        ax.set_title(f'{title} ({int(spikes.sum())} spk)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, handlelength=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)

    def _hist_with_median(ax, vals_s, vals_e, vals_r, bins, xlab, title):
        vs = vals_s[np.isfinite(vals_s)]; ve = vals_e[np.isfinite(vals_e)]
        vr = vals_r[np.isfinite(vals_r)] if vals_r is not None else None
        ax.hist(vs, bins=bins, color=COL_STD, alpha=0.5, density=True,
                label=f'Std (med={np.median(vs):.2f})')
        ax.hist(ve, bins=bins, color=COL_ELIF, alpha=0.5, density=True,
                label=f'eLIF (med={np.median(ve):.2f})')
        if vr is not None and len(vr) > 0:
            ax.hist(vr, bins=bins, color=COL_RM, alpha=0.5, density=True,
                    label=f'RM (med={np.median(vr):.2f})')
        y_top = ax.get_ylim()[1] * 0.96
        for med, col in [(np.median(vs), COL_STD), (np.median(ve), COL_ELIF)]:
            ax.plot(med, y_top, marker='v', color=col, ms=7, mec='k', mew=0.5,
                    zorder=6, clip_on=False)
        if vr is not None and len(vr) > 0:
            ax.plot(np.median(vr), y_top, marker='v', color=COL_RM, ms=7,
                    mec='k', mew=0.5, zorder=6, clip_on=False)
        try:
            _, p_se = mannwhitneyu(vs, ve, alternative='two-sided')
            txt = f'S-E: p={p_se:.2g} {_sig_stars(p_se)}'
            if vr is not None and len(vr) > 0:
                _, p_sr = mannwhitneyu(vs, vr, alternative='two-sided')
                _, p_er = mannwhitneyu(ve, vr, alternative='two-sided')
                txt += (f'\nS-R: p={p_sr:.2g} {_sig_stars(p_sr)}'
                        f'\nE-R: p={p_er:.2g} {_sig_stars(p_er)}')
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha='left',
                    va='top', fontsize=8)
        except Exception:
            pass
        ax.axvline(0, ls='--', color='k', lw=0.5)
        ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel('Density', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)

    ex_idx = d.get('ex_neuron_idx')
    ex_pref = d.get('ex_neuron_pref')
    ex_tun = d.get('ex_neuron_tuning')
    ex_rate = d.get('ex_neuron_mean_rate', {})
    ex_rasters = d.get('ex_neuron_rasters', {})

    ax = fig.add_subplot(gs[0, 0])
    if ex_tun is not None:
        ax.plot(np.arange(NS), ex_tun, '-o', color='k', lw=1.3, ms=4)
        ax.set_xlabel('Stimulus index', fontsize=9)
        ax.set_ylabel('Tuning weight', fontsize=9)
        ax.set_title(f'A. Tuning of E-cell #{ex_idx} (pref={ex_pref})', fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title('A. Tuning (n/a)', fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    if ex_rate:
        for key, col, lbl in [('std', COL_STD, 'Std'),
                               ('elif', COL_ELIF, 'eLIF'),
                               ('rm', COL_RM, 'RM')]:
            if key in ex_rate:
                ax.plot(np.arange(NS), ex_rate[key], '-o', color=col, lw=1.3,
                        ms=4, label=lbl)
        ax.set_xlabel('Stimulus index', fontsize=9)
        ax.set_ylabel('Mean rate (Hz)', fontsize=9)
        ax.set_title(f'B. E-cell #{ex_idx} rate per stim', fontsize=9)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title('B. Mean rate (n/a)', fontsize=9)

    def _plot_neuron_raster(ax, traces_per_stim, title, col):
        if not traces_per_stim or all(len(t) == 0 for t in traces_per_stim):
            ax.axis('off'); ax.set_title(title + ' (n/a)', fontsize=9); return
        t_full = d.get('t_trial')
        T = len(traces_per_stim[0][0]) if traces_per_stim[0] else 0
        t_ms = np.asarray(t_full) if t_full is not None and len(t_full) == T \
               else np.arange(T) * 0.1
        ax.axvspan(stim_on, stim_off, color=COL_STIM, alpha=0.4, zorder=0)
        row = 0
        stim_ticks = []
        for s in range(len(traces_per_stim)):
            stim_start_row = row
            for trace in traces_per_stim[s]:
                spk_t = t_ms[np.asarray(trace) > 0]
                if len(spk_t) > 0:
                    ax.plot(spk_t, np.full_like(spk_t, row), '|', color=col,
                            markersize=2.5, markeredgewidth=0.6)
                row += 1
            if s < len(traces_per_stim) - 1:
                ax.axhline(row - 0.5, color='gray', lw=0.4, ls=':')
            stim_ticks.append((stim_start_row + row - 1) / 2.0)
        ax.set_xlim(0, t_ms[-1])
        ax.set_ylim(-0.5, row - 0.5)
        ax.set_yticks(stim_ticks)
        ax.set_yticklabels([f'S{s}' for s in range(len(traces_per_stim))],
                            fontsize=8)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)

    _plot_neuron_raster(fig.add_subplot(gs[0, 2]),
                        ex_rasters.get('std'),
                        f'C. Cell #{ex_idx} raster: Std (rows=stim x trial)',
                        COL_STD)
    _plot_neuron_raster(fig.add_subplot(gs[0, 3]),
                        ex_rasters.get('elif'),
                        f'D. Cell #{ex_idx} raster: eLIF',
                        COL_ELIF)
    _plot_neuron_raster(fig.add_subplot(gs[0, 4]),
                        ex_rasters.get('rm'),
                        f'E. Cell #{ex_idx} raster: RM',
                        COL_RM)

    ax = fig.add_subplot(gs[1, 0])
    _plot_conf(ax, d.get('confusion_std'),
               f"F. NC Std ({m['accuracy_std']:.1f}%)")
    ax = fig.add_subplot(gs[1, 1])
    _plot_conf(ax, d.get('confusion_elif'),
               f"G. NC eLIF ({m['accuracy_elif']:.1f}%)")
    ax = fig.add_subplot(gs[1, 2])
    _plot_conf(ax, d.get('confusion_elif_rm'),
               f"H. NC RM ({m.get('accuracy_elif_rate_matched', 0):.1f}%)")

    nc_s = d.get('noise_corr_std', np.array([]))
    nc_e = d.get('noise_corr_elif', np.array([]))
    nc_r = d.get('noise_corr_elif_rm', None)
    ax = fig.add_subplot(gs[1, 3])
    _hist_with_median(ax, nc_s, nc_e, nc_r,
                      bins=np.linspace(-0.3, 0.6, 40),
                      xlab='Noise correlation r',
                      title='I. Noise corr (median + MW)')

    ax = fig.add_subplot(gs[1, 4])
    mask_upper = np.triu(np.ones((NS, NS), dtype=bool), k=1)
    dp_s = d['dprime_std'][mask_upper]; dp_e = d['dprime_elif'][mask_upper]
    dp_rm_mat = d.get('dprime_elif_rm')
    if dp_rm_mat is not None and hasattr(dp_rm_mat, 'shape') \
            and dp_rm_mat.shape == d['dprime_std'].shape:
        dp_r = dp_rm_mat[mask_upper]
    else:
        dp_r = dp_e * (m.get('dprime_elif_rate_matched', m['mean_dprime_elif'])
                       / max(m['mean_dprime_elif'], 0.01))
    bp = ax.boxplot([dp_s, dp_e, dp_r], labels=['Std', 'eLIF', 'RM'],
                    patch_artist=True, widths=0.55)
    for i, c in enumerate([COL_STD, COL_ELIF, COL_RM]):
        bp['boxes'][i].set_facecolor('none'); bp['boxes'][i].set_edgecolor(c)
        bp['medians'][i].set_color(c)
        bp['whiskers'][2*i].set_color(c); bp['whiskers'][2*i+1].set_color(c)
        bp['caps'][2*i].set_color(c); bp['caps'][2*i+1].set_color(c)
    try: _, p_se = mannwhitneyu(dp_s, dp_e, alternative='two-sided')
    except: p_se = 1.0
    try: _, p_sr = mannwhitneyu(dp_s, dp_r, alternative='two-sided')
    except: p_sr = 1.0
    ymax = max(np.max(dp_s), np.max(dp_e), np.max(dp_r)) * 1.1
    _add_sig_bracket(ax, 1, 2, ymax, p_se)
    _add_sig_bracket(ax, 1, 3, ymax * 1.12, p_sr)
    ax.set_ylabel("d'", fontsize=9)
    ax.set_title(f"J. d' (med S/E/R = {np.median(dp_s):.2f}/{np.median(dp_e):.2f}/{np.median(dp_r):.2f})",
                 fontsize=8.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    _clean(ax)

    ax = fig.add_subplot(gs[2, 0])
    _plot_conf(ax, d.get('confusion_lda_std'),
               f"K. LDA Std ({m['lda_accuracy_std']:.1f}%)")
    ax = fig.add_subplot(gs[2, 1])
    _plot_conf(ax, d.get('confusion_lda_elif'),
               f"L. LDA eLIF ({m['lda_accuracy_elif']:.1f}%)")
    ax = fig.add_subplot(gs[2, 2])
    _plot_conf(ax, d.get('confusion_lda_rm'),
               f"M. LDA RM ({m.get('lda_accuracy_elif_rate_matched', 0):.1f}%)")

    decoders = ['NC', 'LDA', 'SVM']
    vals_std = [m['accuracy_std'], m['lda_accuracy_std'], m.get('svm_accuracy_std', 0)]
    vals_el  = [m['accuracy_elif'], m['lda_accuracy_elif'], m.get('svm_accuracy_elif', 0)]
    vals_rm  = [m['accuracy_elif_rate_matched'], m['lda_accuracy_elif_rate_matched'],
                m.get('svm_accuracy_elif_rate_matched', 0)]
    ax = fig.add_subplot(gs[2, 3])
    xp = np.arange(len(decoders)); w = 0.27
    ax.bar(xp - w, vals_std, w, color=COL_STD, label='Std')
    ax.bar(xp,     vals_el,  w, color=COL_ELIF, label='eLIF')
    ax.bar(xp + w, vals_rm,  w, color=COL_RM, label='RM')
    ax.axhline(m['chance_level'], ls='--', color='k', lw=0.6, label='chance')
    ax.set_xticks(xp); ax.set_xticklabels(decoders, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('N. Accuracy by decoder', fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _clean(ax)

    ax = fig.add_subplot(gs[2, 4])
    sc_s = d.get('signal_corr_std_vec', np.array([]))
    sc_e = d.get('signal_corr_elif_vec', np.array([]))
    if len(sc_s) > 0 and len(sc_e) > 0:
        _hist_with_median(ax, sc_s, sc_e, None,
                          bins=np.linspace(-1.0, 1.0, 40),
                          xlab='Signal correlation r',
                          title='O. Signal corr (median + MW)')
    else:
        ax.axis('off'); ax.set_title('O. Signal corr (n/a)', fontsize=9)

    ax = fig.add_subplot(gs[3, 0])
    _plot_conf(ax, d.get('confusion_svm_std'),
               f"P. SVM Std ({m.get('svm_accuracy_std', 0):.1f}%)")
    ax = fig.add_subplot(gs[3, 1])
    _plot_conf(ax, d.get('confusion_svm_elif'),
               f"Q. SVM eLIF ({m.get('svm_accuracy_elif', 0):.1f}%)")
    ax = fig.add_subplot(gs[3, 2])
    _plot_conf(ax, d.get('confusion_svm_rm'),
               f"R. SVM RM ({m.get('svm_accuracy_elif_rate_matched', 0):.1f}%)")

    ax = fig.add_subplot(gs[3, 3])
    imp_el = [vals_el[i] - vals_std[i] for i in range(len(decoders))]
    imp_rm = [vals_rm[i] - vals_std[i] for i in range(len(decoders))]
    xp = np.arange(len(decoders)); w = 0.38
    ax.bar(xp - w/2, imp_el, w, color=COL_ELIF, label='eLIF − Std')
    ax.bar(xp + w/2, imp_rm, w, color=COL_RM, label='RM − Std')
    ax.axhline(0, ls='-', color='k', lw=0.8)
    ax.set_xticks(xp); ax.set_xticklabels(decoders, fontsize=9)
    ax.set_ylabel(r'$\Delta$ accuracy (pp)', fontsize=9)
    ax.set_title('S. Improvement over Standard', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    _clean(ax)

    ax = fig.add_subplot(gs[3, 4])
    ns_v = nc_s[np.isfinite(nc_s)]; ne_v = nc_e[np.isfinite(nc_e)]
    nr_v = nc_r[np.isfinite(nc_r)] if (nc_r is not None and hasattr(nc_r, '__len__')) else np.array([])
    means = [np.mean(ns_v) if len(ns_v) else 0,
             np.mean(ne_v) if len(ne_v) else 0,
             np.mean(nr_v) if len(nr_v) else 0]
    sems = [np.std(ns_v)/max(1, np.sqrt(len(ns_v))) if len(ns_v) else 0,
            np.std(ne_v)/max(1, np.sqrt(len(ne_v))) if len(ne_v) else 0,
            np.std(nr_v)/max(1, np.sqrt(len(nr_v))) if len(nr_v) else 0]
    xp = np.arange(3)
    ax.bar(xp, means, 0.6, color=[COL_STD, COL_ELIF, COL_RM])
    ax.errorbar(xp, means, yerr=sems, fmt='none', ecolor='k', capsize=3, lw=1)
    ax.axhline(0, ls='--', color='k', lw=0.5)
    ax.set_xticks(xp); ax.set_xticklabels(['Std', 'eLIF', 'RM'], fontsize=9)
    ax.set_ylabel('Mean noise corr', fontsize=9)
    ax.set_title('T. Mean noise corr (±SEM)', fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    _clean(ax)

    NE_est = d.get('tuning').shape[0] if d.get('tuning') is not None else 400
    _whole_raster(fig.add_subplot(gs[4, 0]), d.get('raster_std'), NE_est,
                  f"U. Whole raster Std (stim S{d.get('raster_stim_idx', 0)})")
    _whole_raster(fig.add_subplot(gs[4, 1]), d.get('raster_elif'), NE_est,
                  f"V. Whole raster eLIF")
    _whole_raster(fig.add_subplot(gs[4, 2]), d.get('raster_rm'), NE_est,
                  f"W. Whole raster RM")

    ax = fig.add_subplot(gs[4, 3])
    X_std = d.get('X_std'); X_el = d.get('X_elif'); X_rm = d.get('X_elif_rm')
    per_trial_s = X_std.sum(axis=1) if X_std is not None else None
    per_trial_e = X_el.sum(axis=1) if X_el is not None else None
    per_trial_r = X_rm.sum(axis=1) if X_rm is not None else None
    spk_vals = [m['spikes_per_trial_std'], m['spikes_per_trial_elif'],
                m.get('spikes_rate_matched', 0)]
    if per_trial_s is not None:
        sems_sp = [np.std(per_trial_s)/np.sqrt(len(per_trial_s)),
                   np.std(per_trial_e)/np.sqrt(len(per_trial_e)),
                   np.std(per_trial_r)/np.sqrt(len(per_trial_r))]
    else:
        sems_sp = [0, 0, 0]
    xp = np.arange(3)
    ax.bar(xp, spk_vals, 0.6, color=[COL_STD, COL_ELIF, COL_RM])
    ax.errorbar(xp, spk_vals, yerr=sems_sp, fmt='none', ecolor='k', capsize=3, lw=1)
    ax.set_xticks(xp); ax.set_xticklabels(['Std', 'eLIF', 'RM'], fontsize=9)
    ax.set_ylabel('Spikes / trial', fontsize=9)
    ax.set_title('X. Stim-window spike count (mean±SEM)', fontsize=8.5)
    if per_trial_s is not None:
        try:
            _, p_se = mannwhitneyu(per_trial_s, per_trial_e, alternative='two-sided')
            _, p_sr = mannwhitneyu(per_trial_s, per_trial_r, alternative='two-sided')
            ax.text(0.02, 0.97,
                    f'S-E p={p_se:.2g} {_sig_stars(p_se)}\n'
                    f'S-R p={p_sr:.2g} {_sig_stars(p_sr)}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=8)
        except Exception: pass
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _clean(ax)

    ax = fig.add_subplot(gs[4, 4])
    if per_trial_s is not None:
        all_v = np.concatenate([per_trial_s, per_trial_e, per_trial_r])
        bins = np.linspace(np.min(all_v), np.max(all_v), 30)
        _hist_with_median(ax, per_trial_s, per_trial_e, per_trial_r,
                          bins=bins, xlab='Spikes / trial',
                          title='Y. Spike count distribution')
    else:
        ax.axis('off'); ax.set_title('Y. Spike count dist (n/a)', fontsize=9)

    va_pairs = {}; tang_pairs = {}; t_vel = None; tbins = None
    if traj is not None and traj['scores_std'].shape[2] >= 3:
        sc_std = traj['scores_std'][:, :, :3]
        sc_el  = traj['scores_elif'][:, :, :3]
        sc_rm  = (traj_rm['scores_elif'][:, :, :3]
                  if traj_rm is not None and 'scores_elif' in traj_rm
                  and traj_rm['scores_elif'].shape[2] >= 3 else None)
        tbins = traj['time_bins']; nT = len(tbins)
        t_vel = tbins[:-1] + (tbins[1] - tbins[0]) / 2

        def _vel_tang(sc):
            vel = np.diff(sc, axis=0); eps = 0.1
            pairs = [(s1, s2) for s1 in range(NS) for s2 in range(s1+1, NS)]
            n_p = len(pairs)
            va = np.zeros((nT-1, n_p)); tg = np.zeros((nT-1, n_p))
            for pi, (s1, s2) in enumerate(pairs):
                for tt in range(nT-1):
                    v1 = vel[tt, s1]; v2 = vel[tt, s2]
                    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        va[tt, pi] = np.degrees(np.arccos(np.clip(
                            np.dot(v1, v2)/(n1*n2), -1, 1)))
                    pd_ = sc[tt, s1] - sc[tt, s2]
                    vd_ = vel[tt, s1] - vel[tt, s2]
                    tg[tt, pi] = np.sum(vd_**2) / (np.sum(pd_**2) + eps)
            return va, tg

        va_pairs['std'], tang_pairs['std'] = _vel_tang(sc_std)
        va_pairs['elif'], tang_pairs['elif'] = _vel_tang(sc_el)
        if sc_rm is not None:
            va_pairs['rm'], tang_pairs['rm'] = _vel_tang(sc_rm)

    ax = fig.add_subplot(gs[5, 0])
    if t_vel is not None:
        for key, lbl, col in [('std', 'Std', COL_STD), ('elif', 'eLIF', COL_ELIF),
                              ('rm', 'RM', COL_RM)]:
            if key in tang_pairs:
                v = tang_pairs[key]
                mu = v.mean(axis=1); se = v.std(axis=1)/np.sqrt(v.shape[1])
                ax.plot(t_vel, mu, '-', color=col, lw=1.5, label=lbl)
                ax.fill_between(t_vel, np.maximum(0, mu - se), mu + se,
                                color=col, alpha=0.15)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('Tangling Q', fontsize=9)
        ax.set_title('Z. Tangling (mean±SEM)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title('Z. Tangling (n/a)', fontsize=9)

    cnt_window_s = 0.18
    pref_target = 5
    cell_pref = np.argmax(d.get('tuning', np.zeros((1, NS))), axis=1)
    s5_idx = np.where(cell_pref == pref_target)[0]
    counts_std_arr  = d.get('counts_std')
    counts_elif_arr = d.get('counts_elif')
    counts_rm_arr   = d.get('counts_elif_rm') if 'counts_elif_rm' in d else None
    if counts_rm_arr is None and 'X_elif_rm' in d:
        Xrm = d['X_elif_rm']
        nT_per_stim = Xrm.shape[0] // NS
        counts_rm_arr = Xrm.reshape(NS, nT_per_stim, -1).transpose(2, 1, 0)
    ax = fig.add_subplot(gs[5, 1])
    if len(s5_idx) > 0 and counts_std_arr is not None:
        tun_pop = d['tuning'][s5_idx].mean(axis=0)
        ax.plot(np.arange(NS), tun_pop, '-o', color='k', lw=1.5, ms=4)
        ax.fill_between(np.arange(NS),
                        d['tuning'][s5_idx].mean(0) - d['tuning'][s5_idx].std(0),
                        d['tuning'][s5_idx].mean(0) + d['tuning'][s5_idx].std(0),
                        color='gray', alpha=0.2, label='±SD')
        ax.set_xlabel('Stimulus index', fontsize=9)
        ax.set_ylabel('Tuning weight', fontsize=9)
        ax.set_title(f'AA. Pop tuning (n={len(s5_idx)} cells, pref=5)', fontsize=9)
        ax.legend(fontsize=8, loc='lower center')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title('AA. Pop tuning (n/a)', fontsize=9)

    ax = fig.add_subplot(gs[5, 2])
    if len(s5_idx) > 0 and counts_std_arr is not None:
        def _mean_rate(carr):
            if carr is None or carr.shape[0] < max(s5_idx) + 1:
                return None
            return carr[s5_idx].mean(axis=(0, 1)) / cnt_window_s
        rate_std_v  = _mean_rate(counts_std_arr)
        rate_elif_v = _mean_rate(counts_elif_arr)
        rate_rm_v   = _mean_rate(counts_rm_arr)
        for vec, col, lbl in [(rate_std_v, COL_STD, 'Std'),
                              (rate_elif_v, COL_ELIF, 'eLIF'),
                              (rate_rm_v, COL_RM, 'RM')]:
            if vec is not None:
                ax.plot(np.arange(NS), vec, '-o', color=col, lw=1.4, ms=4, label=lbl)
        ax.set_xlabel('Stimulus index', fontsize=9)
        ax.set_ylabel('Mean rate (Hz)', fontsize=9)
        ax.set_title(f'BB. Pop rate per stim (n={len(s5_idx)} S5-pref cells)',
                     fontsize=8.5)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        _clean(ax)
    else:
        ax.axis('off'); ax.set_title('BB. Pop rate (n/a)', fontsize=9)

    def _box_stim(ax, pairs_dict, ylab, title):
        if t_vel is None or not pairs_dict:
            ax.axis('off'); ax.set_title(title + ' (n/a)', fontsize=9); return
        stim_mask = (t_vel >= stim_on) & (t_vel < stim_off)
        if stim_mask.sum() == 0:
            stim_mask = np.ones_like(t_vel, dtype=bool)
        groups = []
        labels_box = []
        cols = []
        for key, col in [('std', COL_STD), ('elif', COL_ELIF), ('rm', COL_RM)]:
            if key in pairs_dict:
                groups.append(pairs_dict[key][stim_mask].flatten())
                labels_box.append({'std': 'Std', 'elif': 'eLIF', 'rm': 'RM'}[key])
                cols.append(col)
        bp = ax.boxplot(groups, labels=labels_box, patch_artist=True, widths=0.55,
                        showfliers=False)
        for i, c in enumerate(cols):
            bp['boxes'][i].set_facecolor('none'); bp['boxes'][i].set_edgecolor(c)
            bp['medians'][i].set_color(c)
            bp['whiskers'][2*i].set_color(c); bp['whiskers'][2*i+1].set_color(c)
            bp['caps'][2*i].set_color(c); bp['caps'][2*i+1].set_color(c)
        try:
            if len(groups) >= 2:
                _, p_se = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                ymax = max(np.max(g) for g in groups) * 1.05
                _add_sig_bracket(ax, 1, 2, ymax, p_se)
            if len(groups) >= 3:
                _, p_sr = mannwhitneyu(groups[0], groups[2], alternative='two-sided')
                ymax2 = max(np.max(g) for g in groups) * 1.18
                _add_sig_bracket(ax, 1, 3, ymax2, p_sr)
        except Exception: pass
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _clean(ax)

    _box_stim(fig.add_subplot(gs[5, 3]), tang_pairs, 'Tangling Q',
              'CC. Tangling BOX (stim window)')

    ax = fig.add_subplot(gs[5, 4]); ax.axis('off')
    ax.set_xlim(0, 400); ax.set_ylim(0, 4)
    ax.set_title('DD. Decoding task', loc='left', fontweight='bold', fontsize=9)
    ax.axvspan(0, 100, color='#f0f0f0'); ax.axvspan(100, 300, color=COL_STIM)
    ax.text(200, 3.5, f'{NS} stimuli x {NT} trials', ha='center', fontsize=8.5)
    ax.text(50, 2.8, 'Baseline', ha='center', fontsize=8.5)
    ax.text(200, 2.8, 'Stimulus', ha='center', fontsize=8.5, fontweight='bold')
    ax.text(200, 2.0, 'Shared drive', ha='center', fontsize=8, color='gray')
    ax.text(200, 1.3, '+ tuning modulation', ha='center', fontsize=8, color='blue')
    ax.text(200, 0.6, '+ shared + private noise', ha='center', fontsize=8, color='red')
    ax.text(350, 2.8, 'Post', ha='center', fontsize=8.5)

    def _traj_2d(ax, sc, title, sep_val=None):
        for s in range(NS):
            ax.plot(sc[:, s, 0], sc[:, s, 1], '-', color=sc_colors[s],
                    lw=1.2, alpha=0.8)
            ax.plot(sc[0, s, 0], sc[0, s, 1], 'o', color=sc_colors[s], ms=5)
            ax.plot(sc[-1, s, 0], sc[-1, s, 1], 's', color=sc_colors[s], ms=5)
        ax.set_xlabel('PC1', fontsize=9); ax.set_ylabel('PC2', fontsize=9)
        ttl = title + (f' (sep={sep_val:.1f})' if sep_val is not None else '')
        ax.set_title(ttl, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        _clean(ax)

    sep = traj.get('separation', {}) if traj is not None else {}
    if traj is not None and traj['scores_std'].shape[2] >= 2:
        _traj_2d(fig.add_subplot(gs[6, 0]), traj['scores_std'][:, :, :2],
                 'EE. Std PC1-2', sep.get('std'))
        _traj_2d(fig.add_subplot(gs[6, 1]), traj['scores_elif'][:, :, :2],
                 'FF. eLIF PC1-2', sep.get('elif'))
        if traj_rm is not None and traj_rm['scores_elif'].shape[2] >= 2:
            _traj_2d(fig.add_subplot(gs[6, 2]),
                     traj_rm['scores_elif'][:, :, :2],
                     'GG. RM PC1-2',
                     traj_rm.get('separation', {}).get('elif'))
        else:
            ax = fig.add_subplot(gs[6, 2]); ax.axis('off')
            ax.set_title('GG. RM PC1-2 (n/a)', fontsize=9)
    else:
        for c in (0, 1, 2):
            fig.add_subplot(gs[6, c]).axis('off')

    ax3d = fig.add_subplot(gs[6, 3], projection='3d')
    if traj is not None and traj['scores_std'].shape[2] >= 3:
        def _plot3(sc, col, lbl):
            for s in range(min(NS, sc.shape[1])):
                ax3d.plot(sc[:, s, 0], sc[:, s, 1], sc[:, s, 2],
                          '-', color=col, lw=0.8, alpha=0.55)
            ax3d.plot([], [], [], '-', color=col, lw=1.5, label=lbl)
        _plot3(traj['scores_std'][:, :, :3], COL_STD, 'Std')
        _plot3(traj['scores_elif'][:, :, :3], COL_ELIF, 'eLIF')
        if traj_rm is not None and traj_rm['scores_elif'].shape[2] >= 3:
            _plot3(traj_rm['scores_elif'][:, :, :3], COL_RM, 'RM')
        ax3d.set_xlabel('PC1', fontsize=8); ax3d.set_ylabel('PC2', fontsize=8)
        ax3d.set_zlabel('PC3', fontsize=8)
        ax3d.set_title('HH. 3D PCA (all models)', fontsize=9)
        ax3d.legend(fontsize=8, loc='upper left')
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax3d.axis('off'); ax3d.set_title('HH. 3D PCA (n/a)', fontsize=9)

    ax = fig.add_subplot(gs[6, 4]); ax.axis('off')
    ax.set_title('II. Summary', loc='left', fontweight='bold', fontsize=9)
    summary = (
        f"Decoding\n"
        f"  Std={m['accuracy_std']:.1f}%  eLIF={m['accuracy_elif']:.1f}%  RM={m['accuracy_elif_rate_matched']:.1f}%\n"
        f"\nLDA\n"
        f"  Std={m['lda_accuracy_std']:.1f}%  eLIF={m['lda_accuracy_elif']:.1f}%  RM={m['lda_accuracy_elif_rate_matched']:.1f}%\n"
        f"\nSVM\n"
        f"  Std={m.get('svm_accuracy_std', 0):.1f}%  eLIF={m.get('svm_accuracy_elif', 0):.1f}%  RM={m.get('svm_accuracy_elif_rate_matched', 0):.1f}%\n"
        f"\nDelta_I = {m.get('delta_I_for_rate_match', 0):.3f} mV\n"
        f"Spikes/trial: Std={m['spikes_per_trial_std']:.0f}\n"
        f"  eLIF={m['spikes_per_trial_elif']:.0f}, RM={m.get('spikes_rate_matched', 0):.0f}"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes, ha='left', va='top',
            fontsize=8, family='monospace')

    fig.suptitle(f'Figure 4: Stimulus Decoding ({NS} stim x {NT} trials) [3-way, expanded]',
                 fontsize=10, fontweight='bold', y=0.995)
    _save(fig, 'Fig4_Decoding_3way', save_dir)


def fig4f_temporal(result, save_dir='.'):
    from scipy.stats import wilcoxon as _wsr
    from matplotlib.ticker import MaxNLocator
    m = result.metrics; d = result.data
    nl = d.get('noise_levels', np.array([]))
    as_ = np.asarray(d.get('accuracy_std', []))
    ae = np.asarray(d.get('accuracy_elif', []))
    ar = np.asarray(d.get('accuracy_elif_rm', []))
    lda_s = np.asarray(d.get('lda_std', np.zeros_like(as_)))
    lda_e = np.asarray(d.get('lda_elif', np.zeros_like(as_)))
    lda_r = np.asarray(d.get('lda_elif_rm', np.zeros_like(as_)))
    cs = d.get('confusion_std'); ce = d.get('confusion_elif'); cr = d.get('confusion_elif_rm')
    pr = d.get('profiles'); tn = d.get('t_norm')
    pat_names = d.get('pattern_names', ['Early-Peak', 'Late-Peak', 'Sustained', 'Double-Peak'])
    chance = m.get('chance_level', 25.0)
    has_rm = ar is not None and len(ar) > 0

    def _set_ticks(ax, nx=4, ny=4):
        ax.xaxis.set_major_locator(MaxNLocator(nx))
        ax.yaxis.set_major_locator(MaxNLocator(ny))

    def _stars_annot(ax, groups):
        try:
            stats_lines = []
            for label, a, b in groups:
                _, p = _wsr(a, b)
                stats_lines.append(f'{label}: p={p:.2g} {_sig_stars(p)}')
            ax.text(0.98, 0.02, '\n'.join(stats_lines),
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8)
        except Exception:
            pass

    for mode in ['2way', '3way']:
        if mode == '3way' and not has_rm:
            continue
        L = LABELS_3 if mode == '3way' else LABELS_2
        C = COLORS_3 if mode == '3way' else COLORS_2
        fig = plt.figure(figsize=(14, 20))
        gs_main = fig.add_gridspec(4, 4, top=0.965, bottom=0.41,
                                    hspace=0.75, wspace=0.55)
        axes = np.empty((4, 4), dtype=object)
        for r in range(4):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gs_main[r, c])

        ax = axes[0, 0]
        if len(as_) > 0:
            ax.plot(nl * 100, as_, '-o', color=COL_STD, lw=1.3, ms=4, label='Std')
            ax.plot(nl * 100, ae, '-s', color=COL_ELIF, lw=1.3, ms=4, label='eLIF')
            if mode == '3way' and has_rm:
                ax.plot(nl * 100, ar, '-^', color=COL_RM, lw=1.3, ms=4, label='eLIF(RM)')
            ax.axhline(chance, ls='--', color='k', lw=0.6)
            groups = [('Std vs eLIF', as_, ae)]
            if mode == '3way': groups.append(('Std vs RM', as_, ar))
            _stars_annot(ax, groups)
        ax.set_xlabel('Noise (%)'); ax.set_ylabel('NC accuracy (%)')
        ax.set_title('A. NC accuracy vs noise')
        ax.set_ylim(0, 105); ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax)

        ax = axes[0, 1]
        if len(lda_s) > 0:
            ax.plot(nl * 100, lda_s, '-o', color=COL_STD, lw=1.3, ms=4, label='Std')
            ax.plot(nl * 100, lda_e, '-s', color=COL_ELIF, lw=1.3, ms=4, label='eLIF')
            if mode == '3way' and len(lda_r) > 0:
                ax.plot(nl * 100, lda_r, '-^', color=COL_RM, lw=1.3, ms=4, label='eLIF(RM)')
            ax.axhline(chance, ls='--', color='k', lw=0.6)
            groups = [('Std vs eLIF', lda_s, lda_e)]
            if mode == '3way': groups.append(('Std vs RM', lda_s, lda_r))
            _stars_annot(ax, groups)
        ax.set_xlabel('Noise (%)'); ax.set_ylabel('LDA accuracy (%)')
        ax.set_title('B. LDA accuracy vs noise')
        ax.set_ylim(0, 105); ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax)

        ax = axes[0, 2]
        if len(as_) > 0:
            x = np.arange(len(nl)); w = 0.35
            ax.bar(x - w / 2, ae - as_, w, color=COL_ELIF, label='eLIF - Std')
            if mode == '3way' and has_rm:
                ax.bar(x + w / 2, ar - as_, w, color=COL_RM, label='RM - Std')
            ax.axhline(0, ls='--', color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{n*100:.0f}' for n in nl], fontsize=8, rotation=45)
        ax.set_xlabel('Noise (%)'); ax.set_ylabel(r'$\Delta$ NC accuracy (pp)')
        ax.set_title('C. Per-noise NC improvement')
        ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax, ny=4)

        ax = axes[0, 3]
        if len(lda_s) > 0:
            x = np.arange(len(nl)); w = 0.35
            ax.bar(x - w / 2, lda_e - lda_s, w, color=COL_ELIF, label='eLIF - Std')
            if mode == '3way' and len(lda_r) > 0:
                ax.bar(x + w / 2, lda_r - lda_s, w, color=COL_RM, label='RM - Std')
            ax.axhline(0, ls='--', color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{n*100:.0f}' for n in nl], fontsize=8, rotation=45)
        ax.set_xlabel('Noise (%)'); ax.set_ylabel(r'$\Delta$ LDA accuracy (pp)')
        ax.set_title('D. Per-noise LDA improvement')
        ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax, ny=4)

        ax = axes[1, 0]
        nc_means = [m.get('mean_accuracy_std', 0), m.get('mean_accuracy_elif', 0)]
        nc_cols = [COL_STD, COL_ELIF]; nc_lbls = ['Std', 'eLIF']
        if mode == '3way':
            nc_means.append(m.get('mean_accuracy_elif_rm', 0))
            nc_cols.append(COL_RM); nc_lbls.append('RM')
        xi = np.arange(len(nc_means))
        ax.bar(xi, nc_means, color=nc_cols, width=0.6)
        ax.set_xticks(xi); ax.set_xticklabels(nc_lbls, fontsize=9)
        ax.set_ylabel('Mean NC accuracy (%)')
        ax.set_title('E. Mean NC accuracy')
        ax.axhline(chance, ls='--', color='k', lw=0.6)
        try:
            _, p1 = _wsr(as_, ae)
            ymax = max(nc_means) * 1.05
            _add_sig_bracket(ax, 0, 1, ymax, p1)
            if mode == '3way':
                _, p2 = _wsr(as_, ar)
                _add_sig_bracket(ax, 0, 2, ymax * 1.1, p2)
        except Exception: pass
        _clean(ax); _set_ticks(ax, nx=len(nc_means), ny=4)

        ax = axes[1, 1]
        lda_means = [m.get('mean_lda_std', 0), m.get('mean_lda_elif', 0)]
        if mode == '3way': lda_means.append(m.get('mean_lda_elif_rm', 0))
        ax.bar(xi, lda_means, color=nc_cols, width=0.6)
        ax.set_xticks(xi); ax.set_xticklabels(nc_lbls, fontsize=9)
        ax.set_ylabel('Mean LDA accuracy (%)')
        ax.set_title('F. Mean LDA accuracy')
        ax.axhline(chance, ls='--', color='k', lw=0.6)
        try:
            _, p1 = _wsr(lda_s, lda_e)
            ymax = max(lda_means) * 1.05
            _add_sig_bracket(ax, 0, 1, ymax, p1)
            if mode == '3way':
                _, p2 = _wsr(lda_s, lda_r)
                _add_sig_bracket(ax, 0, 2, ymax * 1.1, p2)
        except Exception: pass
        _clean(ax); _set_ticks(ax, nx=len(lda_means), ny=4)

        ax = axes[1, 2]
        if mode == '2way':
            cv = [m.get('accuracy_std_clean', 0), m.get('accuracy_elif_clean', 0)]
            nv = [m.get('accuracy_std_noisy', 0), m.get('accuracy_elif_noisy', 0)]
        else:
            cv = [m.get('accuracy_std_clean', 0), m.get('accuracy_elif_clean', 0),
                  m.get('accuracy_elif_rm_clean', 0)]
            nv = [m.get('accuracy_std_noisy', 0), m.get('accuracy_elif_noisy', 0),
                  m.get('accuracy_elif_rm_noisy', 0)]
        x2 = np.arange(len(cv)); w = 0.35
        ax.bar(x2 - w/2, cv, w, color=nc_cols, alpha=0.9, label='Clean')
        ax.bar(x2 + w/2, nv, w, color=nc_cols, alpha=0.4, label='Noisy')
        ax.set_xticks(x2); ax.set_xticklabels(nc_lbls, fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('G. Clean vs noisy')
        ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax, nx=len(cv), ny=4)

        ax = axes[1, 3]
        if len(as_) > 0:
            ds = as_[0] - as_; de = ae[0] - ae
            ax.plot(nl * 100, ds, '-o', color=COL_STD, lw=1.3, ms=4, label='Std')
            ax.plot(nl * 100, de, '-s', color=COL_ELIF, lw=1.3, ms=4, label='eLIF')
            if mode == '3way' and has_rm:
                dr = ar[0] - ar
                ax.plot(nl * 100, dr, '-^', color=COL_RM, lw=1.3, ms=4, label='eLIF(RM)')
        ax.set_xlabel('Noise (%)'); ax.set_ylabel('Accuracy drop (pp)')
        ax.set_title('H. Robustness vs noise')
        ax.legend(fontsize=8, handlelength=1.5)
        _clean(ax); _set_ticks(ax)

        def _mean_conf(c):
            if c is None: return None
            if c.ndim == 3:
                avg = c.mean(axis=0)
                rs_ = avg.sum(axis=1, keepdims=True); rs_[rs_ == 0] = 1
                return avg / rs_
            return c

        conf_std = _mean_conf(cs)
        conf_elif = _mean_conf(ce)
        conf_rm = _mean_conf(cr)
        mean_std = m.get('mean_accuracy_std', 0)
        mean_elif = m.get('mean_accuracy_elif', 0)
        mean_rm = m.get('mean_accuracy_elif_rm', 0)

        def _plot_conf(ax, M, title):
            if M is None:
                ax.axis('off'); ax.set_title(title); return
            im = ax.imshow(M, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(pat_names)))
            ax.set_yticks(range(len(pat_names)))
            ax.set_xticklabels(pat_names, fontsize=8, rotation=35, ha='right')
            ax.set_yticklabels(pat_names, fontsize=8)
            ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
            ax.set_title(title)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    col = 'white' if M[i, j] < 0.5 else 'black'
                    ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                            fontsize=7.5, color=col)
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))

        _plot_conf(axes[2, 0], conf_std,
                   f'I. Std confusion ({mean_std:.1f}%)')
        _plot_conf(axes[2, 1], conf_elif,
                   f'J. eLIF confusion ({mean_elif:.1f}%)')
        if mode == '3way':
            _plot_conf(axes[2, 2], conf_rm,
                       f'K. RM confusion ({mean_rm:.1f}%)')
        else:
            axes[2, 2].axis('off')
            axes[2, 2].set_title('K. RM confusion (not available)')

        ax = axes[2, 3]
        if pr is not None and tn is not None:
            cols_pat = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
            for i in range(min(pr.shape[0], 4)):
                ax.plot(tn, pr[i], lw=1.3,
                        color=cols_pat[i % 4], label=pat_names[i])
            ax.set_xlabel('Time (normalized)')
            ax.set_ylabel('Amplitude')
            ax.set_title('L. Temporal patterns')
            ax.legend(fontsize=8, handlelength=1.5, loc='upper right')
        _clean(ax); _set_ticks(ax)

        def _plot_conf_diff(ax, A, B, title):
            if A is None or B is None:
                ax.axis('off'); ax.set_title(title); return
            D = A - B
            vabs = max(abs(D.min()), abs(D.max()), 1e-3)
            im = ax.imshow(D, cmap='RdBu_r', vmin=-vabs, vmax=+vabs, aspect='auto')
            ax.set_xticks(range(len(pat_names)))
            ax.set_yticks(range(len(pat_names)))
            ax.set_xticklabels(pat_names, fontsize=8, rotation=35, ha='right')
            ax.set_yticklabels(pat_names, fontsize=8)
            ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
            ax.set_title(title)
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    col = 'black' if abs(D[i, j]) < vabs * 0.6 else 'white'
                    ax.text(j, i, f'{D[i,j]:+.2f}', ha='center', va='center',
                            fontsize=7.5, color=col)
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))

        _plot_conf_diff(axes[3, 0], conf_elif, conf_std, 'M. eLIF - Std')
        if mode == '3way':
            _plot_conf_diff(axes[3, 1], conf_rm, conf_std, 'N. RM - Std')
            _plot_conf_diff(axes[3, 2], conf_elif, conf_rm, 'O. eLIF - RM')
        else:
            axes[3, 1].axis('off'); axes[3, 1].set_title('N. RM - Std (N/A)')
            axes[3, 2].axis('off'); axes[3, 2].set_title('O. eLIF - RM (N/A)')

        ax = axes[3, 3]; ax.axis('off')
        delta_I = m.get('delta_I', 0.0)
        summary = (
            f"P. Summary\n"
            f"chance: {chance:.0f}%\n"
            f"N_patterns: {len(pat_names)}\n"
            f"N_trials per pattern: {int(as_.size) if len(as_) > 0 else 0}\n"
            f"N_noise levels: {len(nl)}\n\n"
            f"Mean NC acc:\n"
            f"  Std: {mean_std:.1f}%\n"
            f"  eLIF: {mean_elif:.1f}%\n"
            f"  RM: {mean_rm:.1f}%\n\n"
            f"delta_I (RM): {delta_I:.3f}\n"
        )
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
                fontsize=8, family='monospace', va='top', ha='left')

        traj = d.get('trajectories'); traj_rm = d.get('trajectories_rm')
        cstd_arr = d.get('counts_std'); celif_arr = d.get('counts_elif')
        crm_arr = d.get('counts_elif_rm')
        per_trial = None
        if cstd_arr is not None and celif_arr is not None:
            per_trial = {
                'std':  cstd_arr.sum(axis=-1).flatten(),
                'elif': celif_arr.sum(axis=-1).flatten(),
                'rm':   crm_arr.sum(axis=-1).flatten() if crm_arr is not None else None,
            }
        gs_traj = fig.add_gridspec(1, 4, top=0.36, bottom=0.06,
                                    hspace=0.85, wspace=0.55)
        _draw_traj_simple(
            fig, gs_traj, traj, traj_rm, n_classes=len(pat_names),
            stim_on=d.get('stim_onset', 0.0),
            stim_off=d.get('stim_offset', 1.0),
            panel_letters=['Q', 'R', 'S', 'T'],
            per_trial_spikes=per_trial)

        sf = '_3way' if mode == '3way' else ''
        fig.suptitle(f'Figure 4F: Temporal Pattern Classification [{mode}]',
                     fontsize=11, fontweight='bold')
        _save(fig, f'Fig4F_Temporal{sf}', save_dir)

def fig4g_sparse(result, save_dir='.'):
    from scipy.stats import wilcoxon as _wsr
    from matplotlib.ticker import MaxNLocator
    m = result.metrics; d = result.data
    as_ = d.get('accuracy_std'); ae = d.get('accuracy_elif'); ar = d.get('accuracy_elif_rm')
    lda_s_grid = d.get('lda_std'); lda_e_grid = d.get('lda_elif'); lda_r_grid = d.get('lda_elif_rm')
    imp_ = d.get('improvement'); imr = d.get('improvement_rm')
    nl = d.get('noise_levels', np.array([])); sl = d.get('sparsity_levels', np.array([]))
    conf_std = d.get('confusion_std'); conf_elif = d.get('confusion_elif')
    conf_rm = d.get('confusion_elif_rm')
    pat_names = d.get('pattern_names', ['Early-Decay', 'Late-Rise', 'Center-Peak', 'Sustained'])
    pr = d.get('profiles'); tn = d.get('t_norm')
    has_rm = ar is not None

    def _set_ticks(ax, nx=4, ny=4):
        ax.xaxis.set_major_locator(MaxNLocator(nx))
        ax.yaxis.set_major_locator(MaxNLocator(ny))

    for mode in ['2way', '3way']:
        if mode == '3way' and not has_rm:
            continue
        L = LABELS_3 if mode == '3way' else LABELS_2
        C = COLORS_3 if mode == '3way' else COLORS_2
        fig = plt.figure(figsize=(14, 21))
        gs_main = fig.add_gridspec(6, 4, top=0.95, bottom=0.05,
                                    hspace=0.75, wspace=0.55)
        axes = np.empty((6, 4), dtype=object)
        for r in range(6):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gs_main[r, c])

        ax = axes[0, 0]; ax.set_xlim(0, 150); ax.set_ylim(0, 3)
        ax.axvspan(0, 35, color='#f0f0f0'); ax.axvspan(35, 115, color=COL_STIM)
        ax.axvspan(115, 150, color='#f0f0f0')
        ax.text(17, 2.5, 'Baseline', ha='center', fontsize=9)
        ax.text(75, 2.5, 'Stimulus', ha='center', fontsize=9)
        ax.text(75, 1.5, 'Shared drive +', ha='center', fontsize=9, color='gray')
        ax.text(75, 0.8, 'Sparse pattern', ha='center', fontsize=9, color='blue')
        ax.set_xlabel('Time (ms)'); ax.set_yticks([])
        ax.set_title('A. Task timeline')
        _clean(ax); _set_ticks(ax, nx=4, ny=2)

        ax = axes[0, 1]
        if pr is not None and tn is not None:
            cols_pat = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
            for p in range(min(pr.shape[0], 4)):
                ax.plot(tn, pr[p], lw=1.3, color=cols_pat[p], label=pat_names[p])
            ax.set_xlabel('Time (norm)'); ax.set_ylabel('Amplitude')
            ax.legend(fontsize=8, handlelength=1.5)
        ax.set_title('B. Temporal patterns')
        _clean(ax); _set_ticks(ax)

        ax = axes[0, 2]
        for sp_idx, sp_pct in enumerate([5, 15, 25]):
            n_on = int(400 * sp_pct / 100)
            y_base = 2 - sp_idx * 0.8
            for i in range(40):
                c = COL_E if i < int(40 * sp_pct / 100) else '#e0e0e0'
                ax.add_patch(plt.Rectangle((i * 0.024, y_base), 0.02, 0.5, color=c))
            ax.text(1.05, y_base + 0.25, f'{sp_pct}%\n({n_on}N)', fontsize=8, va='center')
        ax.set_xlim(0, 1.3); ax.set_ylim(0, 2.8)
        ax.set_title('C. Sparsity levels')
        ax.set_xlabel('Neurons (green=active)')
        ax.set_yticks([])
        _clean(ax); _set_ticks(ax, nx=4, ny=2)

        ax = axes[0, 3]
        ms_v = np.mean(as_) if as_ is not None else 0
        me_v = np.mean(ae) if ae is not None else 0
        mr_v = np.mean(ar) if ar is not None else 0
        vals = [ms_v, me_v]; cols = [COL_STD, COL_ELIF]; lbls = ['Std', 'eLIF']
        if mode == '3way':
            vals.append(mr_v); cols.append(COL_RM); lbls.append('RM')
        xi = np.arange(len(vals))
        ax.bar(xi, vals, color=cols, width=0.6)
        ax.set_xticks(xi); ax.set_xticklabels(lbls, fontsize=9)
        ax.axhline(25, ls='--', color='k', lw=0.6, label='chance')
        ax.set_ylabel('Mean NC accuracy (%)')
        ax.set_title('D. Mean NC accuracy')
        if as_ is not None and ae is not None:
            try:
                _, p1 = _wsr(np.asarray(as_).ravel(), np.asarray(ae).ravel())
                _add_sig_bracket(ax, 0, 1, max(vals) * 1.05, p1)
                if mode == '3way':
                    _, p2 = _wsr(np.asarray(as_).ravel(), np.asarray(ar).ravel())
                    _add_sig_bracket(ax, 0, 2, max(vals) * 1.15, p2)
            except Exception: pass
        _clean(ax); _set_ticks(ax, nx=len(vals), ny=4)

        if as_ is not None and len(nl) > 1 and len(sl) > 1:
            ext = [nl[0] * 100, nl[-1] * 100, sl[0] * 100, sl[-1] * 100]

            def _grid_panel(ax, M, title, cmap='viridis', vmin=0, vmax=100,
                            clabel='acc (%)', cbar_ticks=4):
                im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                               origin='lower', extent=ext,
                               interpolation='bicubic')
                ax.set_xlabel('Noise (%)'); ax.set_ylabel('Sparsity (%)')
                ax.set_title(title)
                cb = plt.colorbar(im, ax=ax, fraction=0.046, label=clabel)
                cb.ax.yaxis.set_major_locator(MaxNLocator(cbar_ticks))
                _set_ticks(ax)

            _grid_panel(axes[1, 0], as_, f'E. Std NC ({np.mean(as_):.1f}%)')
            _grid_panel(axes[1, 1], ae, f'F. eLIF NC ({np.mean(ae):.1f}%)')
            if mode == '3way':
                _grid_panel(axes[1, 2], ar, f'G. RM NC ({np.mean(ar):.1f}%)')
            else:
                axes[1, 2].axis('off'); axes[1, 2].set_title('G. RM NC (N/A)')
            rr = m.get('rate_ratio', 1.0)
            sps = m.get('spikes_per_trial_std', 0); spe = m.get('spikes_per_trial_elif', 0)
            ax = axes[1, 3]
            if mode == '3way':
                bars = [sps, spe, sps]
                ax.bar(np.arange(3), bars, color=[COL_STD, COL_ELIF, COL_RM])
                ax.set_xticks(np.arange(3)); ax.set_xticklabels(['Std', 'eLIF', 'RM'], fontsize=9)
            else:
                bars = [sps, spe]
                ax.bar(np.arange(2), bars, color=[COL_STD, COL_ELIF])
                ax.set_xticks(np.arange(2)); ax.set_xticklabels(['Std', 'eLIF'], fontsize=9)
            ax.set_ylabel('Spikes per trial')
            ax.set_title(f'H. Firing rate (x{rr:.2f})')
            _clean(ax); _set_ticks(ax, nx=len(bars), ny=4)
        for col in range(4):
            _clean(axes[1, col])

        if as_ is not None:
            ax = axes[2, 0]
            mns = np.mean(as_, axis=0); mne = np.mean(ae, axis=0)
            ax.plot(nl * 100, mns, '-o', color=COL_STD, lw=1.3, ms=4, label='Std')
            ax.plot(nl * 100, mne, '-s', color=COL_ELIF, lw=1.3, ms=4, label='eLIF')
            if mode == '3way' and ar is not None:
                ax.plot(nl * 100, np.mean(ar, axis=0), '-^', color=COL_RM,
                        lw=1.3, ms=4, label='RM')
            ax.axhline(25, ls='--', color='k', lw=0.6)
            ax.set_xlabel('Noise (%)'); ax.set_ylabel('NC accuracy (%)')
            ax.set_title('I. NC accuracy vs noise')
            ax.legend(fontsize=8, handlelength=1.5)
            _clean(ax); _set_ticks(ax)

            ax = axes[2, 1]
            mss = np.mean(as_, axis=1); mse = np.mean(ae, axis=1)
            ax.plot(sl * 100, mss, '-o', color=COL_STD, lw=1.3, ms=4, label='Std')
            ax.plot(sl * 100, mse, '-s', color=COL_ELIF, lw=1.3, ms=4, label='eLIF')
            if mode == '3way' and ar is not None:
                ax.plot(sl * 100, np.mean(ar, axis=1), '-^', color=COL_RM,
                        lw=1.3, ms=4, label='RM')
            ax.set_xlabel('Sparsity (%)'); ax.set_ylabel('NC accuracy (%)')
            ax.set_title('J. NC accuracy vs sparsity')
            ax.legend(fontsize=8, handlelength=1.5)
            _clean(ax); _set_ticks(ax)

            ax = axes[2, 2]
            ibn = np.mean(ae - as_, axis=0)
            x = np.arange(len(nl)); w = 0.35
            ax.bar(x - w/2, ibn, w, color=COL_ELIF, label='eLIF - Std')
            if mode == '3way' and ar is not None:
                ibrn = np.mean(ar - as_, axis=0)
                ax.bar(x + w/2, ibrn, w, color=COL_RM, label='RM - Std')
            ax.axhline(0, ls='--', color='k', lw=0.5)
            ax.set_xticks(x); ax.set_xticklabels([f'{n*100:.0f}' for n in nl],
                                                 fontsize=8, rotation=45)
            ax.set_xlabel('Noise (%)'); ax.set_ylabel('Improvement (pp)')
            ax.set_title('K. NC improvement by noise')
            ax.legend(fontsize=8, handlelength=1.5)
            _clean(ax); _set_ticks(ax, nx=len(nl), ny=4)

            ax = axes[2, 3]
            ibs = np.mean(ae - as_, axis=1)
            x = np.arange(len(sl)); w = 0.35
            ax.bar(x - w/2, ibs, w, color=COL_ELIF, label='eLIF - Std')
            if mode == '3way' and ar is not None:
                ibrs = np.mean(ar - as_, axis=1)
                ax.bar(x + w/2, ibrs, w, color=COL_RM, label='RM - Std')
            ax.axhline(0, ls='--', color='k', lw=0.5)
            ax.set_xticks(x); ax.set_xticklabels([f'{s*100:.0f}' for s in sl],
                                                 fontsize=8, rotation=45)
            ax.set_xlabel('Sparsity (%)'); ax.set_ylabel('Improvement (pp)')
            ax.set_title('L. NC improvement by sparsity')
            ax.legend(fontsize=8, handlelength=1.5)
            _clean(ax); _set_ticks(ax, nx=len(sl), ny=4)

        if lda_s_grid is not None and lda_e_grid is not None and len(nl) > 1:
            ext = [nl[0] * 100, nl[-1] * 100, sl[0] * 100, sl[-1] * 100]

            def _lda_grid(ax, M, title):
                im = ax.imshow(M, cmap='viridis', vmin=0, vmax=100, aspect='auto',
                               origin='lower', extent=ext, interpolation='bicubic')
                ax.set_xlabel('Noise (%)'); ax.set_ylabel('Sparsity (%)')
                ax.set_title(title)
                cb = plt.colorbar(im, ax=ax, fraction=0.046, label='LDA acc (%)')
                cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))
                _set_ticks(ax)

            _lda_grid(axes[3, 0], lda_s_grid, f'M. Std LDA ({np.mean(lda_s_grid):.1f}%)')
            _lda_grid(axes[3, 1], lda_e_grid, f'N. eLIF LDA ({np.mean(lda_e_grid):.1f}%)')
            if mode == '3way' and lda_r_grid is not None:
                _lda_grid(axes[3, 2], lda_r_grid, f'O. RM LDA ({np.mean(lda_r_grid):.1f}%)')
            else:
                axes[3, 2].axis('off')
                axes[3, 2].set_title('O. RM LDA (N/A)')
            ax = axes[3, 3]
            lda_means = [m.get('mean_lda_std', 0), m.get('mean_lda_elif', 0)]
            lbls_lda = ['Std', 'eLIF']; cols_lda = [COL_STD, COL_ELIF]
            if mode == '3way':
                lda_means.append(m.get('mean_lda_elif_rm', 0))
                lbls_lda.append('RM'); cols_lda.append(COL_RM)
            xi = np.arange(len(lda_means))
            ax.bar(xi, lda_means, color=cols_lda, width=0.6)
            ax.set_xticks(xi); ax.set_xticklabels(lbls_lda, fontsize=9)
            ax.axhline(25, ls='--', color='k', lw=0.6)
            ax.set_ylabel('Mean LDA accuracy (%)')
            ax.set_title('P. Mean LDA accuracy')
            try:
                _, p1 = _wsr(np.asarray(lda_s_grid).ravel(),
                             np.asarray(lda_e_grid).ravel())
                _add_sig_bracket(ax, 0, 1, max(lda_means) * 1.05, p1)
                if mode == '3way':
                    _, p2 = _wsr(np.asarray(lda_s_grid).ravel(),
                                 np.asarray(lda_r_grid).ravel())
                    _add_sig_bracket(ax, 0, 2, max(lda_means) * 1.15, p2)
            except Exception: pass
            _clean(ax); _set_ticks(ax, nx=len(lda_means), ny=4)
        else:
            for col in range(4):
                axes[3, col].axis('off')
                axes[3, col].set_title('LDA N/A')

        def _plot_conf(ax, M, title):
            if M is None:
                ax.axis('off'); ax.set_title(title); return
            im = ax.imshow(M, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(pat_names)))
            ax.set_yticks(range(len(pat_names)))
            ax.set_xticklabels(pat_names, fontsize=8, rotation=35, ha='right')
            ax.set_yticklabels(pat_names, fontsize=8)
            ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
            ax.set_title(title)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    col = 'white' if M[i, j] < 0.5 else 'black'
                    ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                            fontsize=7.5, color=col)
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))

        mean_std_v = m.get('mean_accuracy_std', 0)
        mean_elif_v = m.get('mean_accuracy_elif', 0)
        mean_rm_v = m.get('mean_accuracy_elif_rm', 0)
        _plot_conf(axes[4, 0], conf_std, f'Q. Std confusion ({mean_std_v:.1f}%)')
        _plot_conf(axes[4, 1], conf_elif, f'R. eLIF confusion ({mean_elif_v:.1f}%)')
        if mode == '3way':
            _plot_conf(axes[4, 2], conf_rm, f'S. RM confusion ({mean_rm_v:.1f}%)')
        else:
            axes[4, 2].axis('off'); axes[4, 2].set_title('S. RM confusion (N/A)')

        ax = axes[4, 3]
        if imp_ is not None and len(nl) > 1 and len(sl) > 1:
            ext = [nl[0] * 100, nl[-1] * 100, sl[0] * 100, sl[-1] * 100]
            vm = max(abs(imp_.min()), abs(imp_.max()), 1)
            im = ax.imshow(imp_, cmap='RdBu_r', vmin=-vm, vmax=vm,
                           aspect='auto', origin='lower', extent=ext,
                           interpolation='bicubic')
            ax.set_xlabel('Noise (%)'); ax.set_ylabel('Sparsity (%)')
            ax.set_title('T. eLIF NC improvement grid')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, label='pp')
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))
            _set_ticks(ax)

        def _plot_conf_diff(ax, A, B, title):
            if A is None or B is None:
                ax.axis('off'); ax.set_title(title); return
            D = A - B
            vabs = max(abs(D.min()), abs(D.max()), 1e-3)
            im = ax.imshow(D, cmap='RdBu_r', vmin=-vabs, vmax=+vabs, aspect='auto')
            ax.set_xticks(range(len(pat_names)))
            ax.set_yticks(range(len(pat_names)))
            ax.set_xticklabels(pat_names, fontsize=8, rotation=35, ha='right')
            ax.set_yticklabels(pat_names, fontsize=8)
            ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
            ax.set_title(title)
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    col = 'black' if abs(D[i, j]) < vabs * 0.6 else 'white'
                    ax.text(j, i, f'{D[i,j]:+.2f}', ha='center', va='center',
                            fontsize=7.5, color=col)
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))

        _plot_conf_diff(axes[5, 0], conf_elif, conf_std, 'U. Conf diff: eLIF - Std')
        if mode == '3way':
            _plot_conf_diff(axes[5, 1], conf_rm, conf_std, 'V. Conf diff: RM - Std')
            _plot_conf_diff(axes[5, 2], conf_elif, conf_rm, 'W. Conf diff: eLIF - RM')
        else:
            axes[5, 1].axis('off'); axes[5, 1].set_title('V. RM - Std (N/A)')
            axes[5, 2].axis('off'); axes[5, 2].set_title('W. eLIF - RM (N/A)')
        ax = axes[5, 3]
        if imr is not None and len(nl) > 1 and len(sl) > 1:
            ext = [nl[0] * 100, nl[-1] * 100, sl[0] * 100, sl[-1] * 100]
            vm = max(abs(imr.min()), abs(imr.max()), 1)
            im = ax.imshow(imr, cmap='RdBu_r', vmin=-vm, vmax=vm,
                           aspect='auto', origin='lower', extent=ext,
                           interpolation='bicubic')
            ax.set_xlabel('Noise (%)'); ax.set_ylabel('Sparsity (%)')
            ax.set_title('X. RM NC improvement grid')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, label='pp')
            cb.ax.yaxis.set_major_locator(plt.LinearLocator(4))
            _set_ticks(ax)
        else:
            ax.axis('off'); ax.set_title('X. RM improvement (N/A)')


        sf = '_3way' if mode == '3way' else ''
        fig.suptitle(f'Figure 5  |  Sparse Coding Benchmark  [{mode}]',
                     fontsize=13, fontweight='bold')
        _save(fig, f'Fig4G_Sparse{sf}', save_dir)

def fig_spatiotemporal(result, save_dir='.'):
    from matplotlib.gridspec import GridSpec
    m = result.metrics; d = result.data
    phi = np.asarray(d['phi']); tau = np.asarray(d['tau_table'])
    NE = phi.shape[0]
    N_stim = d['N_stim']; chance = d['chance']
    t_ms = np.asarray(d['t_ms']); st0 = d['stim_start']; st1 = d['stim_end']
    delta = d['delta']; sigma = d['sigma']; period = d['period']
    rasters = d.get('rasters', {})
    conf = d.get('confusion', {})
    nstim_ex = int(d.get('n_example_stim', 4))
    sel = d.get('sel')
    sel = np.asarray(sel) if sel is not None else np.ones((NE, N_stim), bool)
    n_sel = int(d.get('n_selective_stim', sel[0].sum() if sel.size else 5))
    ex_s = 0
    sel_ex = np.where(sel[:, ex_s])[0]
    coding_order = sel_ex[np.argsort(phi[sel_ex])]

    def _cl(ax):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig = plt.figure(figsize=(19, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ex_neurons = coding_order[np.linspace(0, len(coding_order) - 1, 6).astype(int)] \
        if len(coding_order) >= 6 else coding_order
    cols = plt.cm.viridis(np.linspace(0, 1, len(ex_neurons)))
    for ci, n in enumerate(ex_neurons):
        ax.plot(np.arange(N_stim), tau[n], '-', color=cols[ci], lw=0.8, alpha=0.4)
        pref = np.where(sel[n])[0]
        ax.plot(pref, tau[n, pref], 'o', color=cols[ci], ms=4)
    ax.set_xlabel('Stimulus'); ax.set_ylabel('Activation time τ (ms)')
    ax.set_title(f'Temporal code (Δ={delta:.0f} ms, σ={sigma:.0f} ms)\n'
                 f'dots = preferred stimuli ({n_sel}/{N_stim})', fontsize=9.5)
    _cl(ax)

    nc = len(coding_order)
    for ci, mk in enumerate(['std', 'elif', 'rm']):
        ax = fig.add_subplot(gs[0, ci + 1])
        rs = rasters.get(mk, {})
        spk = rs.get(ex_s)
        if spk is not None and nc:
            E = spk[coding_order]
            yy, xx = np.nonzero(E)
            tt = xx * (t_ms[1] - t_ms[0])
            inwin = (tt >= st0) & (tt < st1)
            ax.scatter(tt[inwin], yy[inwin], s=3.0, c='k', marker='.', alpha=0.6)
        ax.plot(st0 + tau[coding_order, ex_s], np.arange(nc), '-', color='tab:red', lw=1.0, alpha=0.7)
        ax.set_xlim(st0, st1); ax.set_ylim(0, max(1, nc))
        ax.set_xlabel('Time (ms)');
        if ci == 0: ax.set_ylabel(f'Selective neuron (stim {ex_s}), by φ')
        ax.set_title(f'{["Std","eLIF","RM"][ci]} raster (stim {ex_s})', fontsize=9.5)
        _cl(ax)

    labels3 = ['Std', 'eLIF', 'RM']
    colors3 = [COL_STD, COL_ELIF, COL_RM]
    ax = fig.add_subplot(gs[1, 0])
    vals = [m['lda_std'], m['lda_elif'], m['lda_rm']]
    ax.bar(labels3, vals, color=colors3)
    ax.axhline(chance, ls='--', color='k', lw=0.8)
    for i, v in enumerate(vals): ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_ylabel('Accuracy (%)'); ax.set_title(f'LDA decoding (chance={chance:.0f}%)', fontsize=9.5)
    ax.set_ylim(0, max(100, max(vals) * 1.15)); _cl(ax)

    ax = fig.add_subplot(gs[1, 1])
    vals = [m['nc_std'], m['nc_elif'], m['nc_rm']]
    ax.bar(labels3, vals, color=colors3)
    ax.axhline(chance, ls='--', color='k', lw=0.8)
    for i, v in enumerate(vals): ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Nearest-centroid decoding', fontsize=9.5)
    ax.set_ylim(0, max(100, max(vals) * 1.15)); _cl(ax)

    ax = fig.add_subplot(gs[1, 2])
    vals = [m['code_spk_std'], m['code_spk_elif'], m['code_spk_rm']]
    ax.bar(labels3, vals, color=colors3)
    for i, v in enumerate(vals): ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Spikes / trial'); ax.set_title(f'Firing (RM err {m["rm_err_pct"]:+.1f}%)', fontsize=9.5)
    _cl(ax)

    ax = fig.add_subplot(gs[1, 3])
    vals = [m['noise_corr_std'], m['noise_corr_elif'], m['noise_corr_rm']]
    ax.bar(labels3, vals, color=colors3)
    for i, v in enumerate(vals): ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Noise corr'); ax.set_title('Coding-window noise corr', fontsize=9.5)
    _cl(ax)

    for ci, mk in enumerate(['std', 'elif', 'rm']):
        ax = fig.add_subplot(gs[2, ci])
        C = conf.get(mk)
        if C is not None and np.asarray(C).size:
            C = np.asarray(C, dtype=float)
            Cn = C / np.clip(C.sum(axis=1, keepdims=True), 1, None)
            im = ax.imshow(Cn, cmap='gray_r', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel('Predicted');
        if ci == 0: ax.set_ylabel('True')
        accv = [m['lda_std'], m['lda_elif'], m['lda_rm']][ci]
        ax.set_title(f'{labels3[ci]} confusion ({accv:.0f}%)', fontsize=9.5)

    ax = fig.add_subplot(gs[2, 3]); ax.axis('off')
    summary = (
        'TEMPORAL CODING (distributed selectivity)\n'
        f'{N_stim} stimuli, {d["N_trials"]} trials each\n'
        f'each neuron selective to {n_sel}/{N_stim} stim\n'
        f'(rest fire as noise); all neurons active\n'
        f'Δ={delta:.0f} ms, σ={sigma:.0f} ms (σ>Δ → overlap)\n'
        f'bin={d["decode_bin_ms"]:.0f} ms, chance={chance:.0f}%\n\n'
        f'LDA: Std {m["lda_std"]:.1f}%  eLIF {m["lda_elif"]:.1f}%\n'
        f'      RM {m["lda_rm"]:.1f}%\n'
        f'NC:  Std {m["nc_std"]:.1f}%  eLIF {m["nc_elif"]:.1f}%\n'
        f'      RM {m["nc_rm"]:.1f}%\n\n'
        f'Rate match: err {m["rm_err_pct"]:+.1f}%, '
        f'MW p={m["mw_p_std_rm"]:.2g}\n'
        f'Noise corr: Std {m["noise_corr_std"]:.3f},\n'
        f'  eLIF {m["noise_corr_elif"]:.3f}, RM {m["noise_corr_rm"]:.3f}\n\n'
        'Stimulus identity carried by SPIKE\n'
        'TIMING. RM = rate-matched eLIF, so\n'
        'any RM vs Std gap is not a rate effect.\n'
        'Only the neuron model differs.'
    )
    ax.text(0.0, 0.98, summary, transform=ax.transAxes, fontsize=7.6,
            va='top', ha='left', family='monospace')

    fig.suptitle('Figure: Temporal-coding decoding — spike-timing code (Std / eLIF / rate-matched eLIF)',
                 fontsize=11, fontweight='bold', y=0.995)
    _save(fig, 'Fig_Spatiotemporal', save_dir)


def fig_geometry(geometry, config, save_dir='.'):
    fig,axes=plt.subplots(2,4,figsize=(18,8)); NE=geometry.NE; pos=geometry.pos
    from scipy.spatial.distance import cdist

    axes[0,0].scatter(pos[:NE,0],pos[:NE,1],s=3,color=COL_E,alpha=0.7,label=f'E ({NE})')
    axes[0,0].scatter(pos[NE:,0],pos[NE:,1],s=8,color=COL_I,alpha=0.7,label=f'I ({geometry.NI})')
    axes[0,0].set_title('Neuron Layout (Halton)',fontsize=9.5); axes[0,0].set_aspect('equal')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y'); axes[0,0].legend(fontsize=9)

    axes[0,1].scatter(pos[:NE,0],pos[:NE,1],s=2,color=COL_E,alpha=0.3)
    axes[0,1].scatter(geometry.source_pos[:,0],geometry.source_pos[:,1],s=40,c='red',marker='x',zorder=5,label=f'Noise sources ({len(geometry.source_pos)})')
    axes[0,1].set_title('Spatial Noise Sources',fontsize=9.5); axes[0,1].set_aspect('equal')
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y'); axes[0,1].legend(fontsize=9)

    c=geometry.W_eph_E[0]; sc=axes[0,2].scatter(pos[:NE,0],pos[:NE,1],s=5,c=c,cmap='hot')
    axes[0,2].plot(pos[0,0],pos[0,1],'k*',ms=12,zorder=5)
    axes[0,2].set_title(f'Ephaptic coupling FROM neuron 0\n(sigma_eph={config.geometry.sigma_eph})',fontsize=9)
    axes[0,2].set_aspect('equal'); axes[0,2].set_xlabel('x'); plt.colorbar(sc,ax=axes[0,2])

    G=geometry.G[:NE]; nc=G@G.T; c0=nc[0]
    sc=axes[0,3].scatter(pos[:NE,0],pos[:NE,1],s=5,c=c0,cmap='hot')
    axes[0,3].plot(pos[0,0],pos[0,1],'k*',ms=12,zorder=5)
    axes[0,3].set_title(f'Input correlation FROM neuron 0\n(sigma_input={config.geometry.sigma_input})',fontsize=9)
    axes[0,3].set_aspect('equal'); axes[0,3].set_xlabel('x'); plt.colorbar(sc,ax=axes[0,3])

    D=cdist(pos[:NE],pos[:NE]); d0=D[0]

    axes[1,0].scatter(d0,c,s=2,color=COL_ELIF,alpha=0.5)
    axes[1,0].set_xlabel('Distance from neuron 0'); axes[1,0].set_ylabel('Ephaptic weight')
    axes[1,0].set_title('Ephaptic vs Distance',fontsize=9.5); _clean(axes[1,0])

    axes[1,1].scatter(d0,c0,s=2,color='orange',alpha=0.5)
    axes[1,1].set_xlabel('Distance from neuron 0'); axes[1,1].set_ylabel('Input correlation')
    axes[1,1].set_title('Input Corr vs Distance',fontsize=9.5); _clean(axes[1,1])

    axes[1,2].scatter(d0,c,s=2,color=COL_ELIF,alpha=0.4,label='Ephaptic')
    axes[1,2].scatter(d0,c0,s=2,color='orange',alpha=0.4,label='Input corr')
    axes[1,2].set_xlabel('Distance'); axes[1,2].set_ylabel('Weight / Correlation')
    axes[1,2].set_title('Double Geometric Factor',fontsize=9.5); axes[1,2].legend(fontsize=9); _clean(axes[1,2])

    ax=axes[1,3]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title('eLIF Model',fontsize=10,fontweight='bold')
    ax.text(5,9,'Standard LIF:',ha='center',fontsize=9.5,fontweight='bold')
    ax.text(5,8.2,r'$\tau_m \frac{dV}{dt} = -(V-V_{rest}) + I_{input} + I_{syn}$',ha='center',fontsize=9.5)
    ax.text(5,6.8,'eLIF (ephaptic):',ha='center',fontsize=9.5,fontweight='bold',color=COL_ELIF)
    ax.text(5,6,r'$\tau_m \frac{dV}{dt} = -(V-V_{rest}) + I_{input} + I_{syn} + \Phi_i$',ha='center',fontsize=9.5,color=COL_ELIF)
    ax.text(5,4.8,r'$\Phi_i = \alpha \sum_j w_{ij}(V_j - V_{rest}) / \sum_j w_{ij}$',ha='center',fontsize=9,color=COL_ELIF)
    ax.text(5,3.6,r'$w_{ij} = \exp(-d_{ij}^2 / 2\sigma_{eph}^2)$',ha='center',fontsize=9,color='gray')
    ax.text(5,2.2,f'Parameters:',ha='center',fontsize=9,fontweight='bold')
    ax.text(5,1.4,f'alpha={config.elif_params.alpha}, sigma_eph={config.geometry.sigma_eph}, sigma_input={config.geometry.sigma_input}',ha='center',fontsize=9)
    ax.text(5,0.6,f'N={config.network.N} ({config.network.NE}E, {config.network.NI}I), K={config.geometry.K_sources} sources',ha='center',fontsize=9)

    for ax in axes.flat: _clean(ax)
    fig.suptitle('Network Geometry & eLIF Model',fontsize=14,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,'Fig_Geometry',save_dir)

def fig_mechanism(result, save_dir='.'):
    d=result.data
    fig,axes=plt.subplots(2,3,figsize=(18,10))

    lfp=d.get('lfp',{})
    if lfp:
        t_m=lfp['t']; llfp_s=lfp['local_lfps_std']; llfp_e=lfp['local_lfps_elif']
        mean_s=np.mean(llfp_s,axis=0); mean_e=np.mean(llfp_e,axis=0)
        sem_s=np.std(llfp_s,axis=0)/np.sqrt(llfp_s.shape[0])
        sem_e=np.std(llfp_e,axis=0)/np.sqrt(llfp_e.shape[0])
        ax=axes[0,0]
        ax.plot(t_m,mean_s,color=COL_STD,lw=1,label='Std')
        ax.fill_between(t_m,mean_s-sem_s,mean_s+sem_s,color=COL_STD,alpha=0.15)
        ax.plot(t_m,mean_e,color=COL_ELIF,lw=1,label='eLIF')
        ax.fill_between(t_m,mean_e-sem_e,mean_e+sem_e,color=COL_ELIF,alpha=0.15)
        ax.axvspan(100,300,color=COL_STIM,alpha=0.2)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('LFP (mV from rest)')
        ax.set_title('Local LFP (mean+/-SEM, 4 sites)',fontsize=9.5); ax.legend(fontsize=9); _clean(ax)

        ax=axes[0,1]
        f=lfp['freq']; mask_f=(f>=0.1)&(f<=100)
        ps_m=lfp['psd_std_mean']; ps_se=lfp['psd_std_sem']
        pe_m=lfp['psd_elif_mean']; pe_se=lfp['psd_elif_sem']
        ax.semilogy(f[mask_f],ps_m[mask_f],color=COL_STD,lw=1.5,label='Std')
        ax.fill_between(f[mask_f],np.maximum(ps_m[mask_f]-ps_se[mask_f],1e-12),ps_m[mask_f]+ps_se[mask_f],color=COL_STD,alpha=0.15)
        ax.semilogy(f[mask_f],pe_m[mask_f],color=COL_ELIF,lw=1.5,label='eLIF')
        ax.fill_between(f[mask_f],np.maximum(pe_m[mask_f]-pe_se[mask_f],1e-12),pe_m[mask_f]+pe_se[mask_f],color=COL_ELIF,alpha=0.15)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (mV^2/Hz)')
        ax.set_title('LFP Power Spectrum (0.1-100 Hz, Hamming)',fontsize=9.5); ax.legend(fontsize=9); _clean(ax)

    corr_data=d.get('correlations',{})
    if corr_data:
        ax=axes[0,2]
        bc=corr_data['bin_centers']; n_b=len(bc)
        cs=corr_data['corr_binned_std']; ce=corr_data['corr_binned_elif']
        ses=corr_data['corr_sem_std']; see=corr_data['corr_sem_elif']
        pvals=corr_data['bin_pvals']; edges=corr_data['bin_edges']
        x=np.arange(n_b); w=0.35
        ax.bar(x-w/2,cs,w,color=COL_STD,label='Std',yerr=ses,capsize=3,error_kw={'lw':1})
        ax.bar(x+w/2,ce,w,color=COL_ELIF,label='eLIF',yerr=see,capsize=3,error_kw={'lw':1})
        for bi in range(n_b):
            if pvals[bi]<0.05:
                ymax=max(cs[bi]+ses[bi],ce[bi]+see[bi])
                _add_sig_bracket(ax,x[bi]-w/2,x[bi]+w/2,ymax,pvals[bi])
        inp_corr=corr_data.get('input_corr_binned')
        if inp_corr is not None:
            ax2c=ax.twinx()
            ax2c.plot(x,inp_corr,'--',color='gray',lw=1.5,ms=5,label='Input noise corr')
            ax2c.set_ylabel('Input corr',fontsize=9,color='gray')
            ax2c.tick_params(axis='y',labelsize=6,colors='gray')
        ax.axhline(0,ls='--',color='k',lw=0.5)
        labels_d=[f'{edges[i]:.2f}-{edges[i+1]:.2f}' for i in range(n_b)]
        ax.set_xticks(x); ax.set_xticklabels(labels_d,fontsize=9,rotation=30)
        ax.set_xlabel('Distance'); ax.set_ylabel('V correlation')
        ax.set_title('Pairwise Correlation vs Distance (all pairs)',fontsize=9); ax.legend(fontsize=9); _clean(ax)

    sc=d.get('sigma_corr',{})
    if sc:
        ax=axes[1,0]
        sv=sc['sigma_vals']; nc=sc['mean_corr']; nse=sc['sem_corr']
        ax.errorbar(sv,nc,yerr=nse,fmt='-o',color=COL_ELIF,lw=2,ms=8,capsize=4,label='eLIF')
        ax.axhline(sc.get('std_mean_corr',0),ls='--',color=COL_STD,lw=2,label=f'Std (alpha=0): {sc.get("std_mean_corr",0):.3f}')
        ax.set_xlabel('sigma_eph (all tasks use 0.15)'); ax.set_ylabel('Mean V correlation')
        ax.set_title('Correlation vs Ephaptic Scale (all pairs)',fontsize=9.5); ax.legend(fontsize=9); _clean(ax)

    kern=d.get('kernels',{})
    if kern:
        ax=axes[1,1]
        bc2=kern['bin_centers']; eph=kern['eph_binned']; inp=kern['inp_binned']
        eph_n=eph/max(eph.max(),1e-10); inp_n=inp/max(inp.max(),1e-10)
        ax.plot(bc2,eph_n,'-o',color=COL_ELIF,lw=2,ms=8,label='Ephaptic coupling')
        ax.plot(bc2,inp_n,'-s',color='orange',lw=2,ms=8,label='Input correlation')
        ax.set_xlabel('Distance'); ax.set_ylabel('Normalized weight')
        ax.set_title('Spatial Kernels: Ephaptic vs Input',fontsize=9.5); ax.legend(fontsize=9); _clean(ax)
        ax.annotate('sig_eph=0.15, sig_inp=0.15',xy=(0.95,0.95),xycoords='axes fraction',ha='right',va='top',fontsize=9,color='gray')

    al=d.get('alpha',{})
    if al:
        ax=axes[1,2]
        avals=al['alpha_vals']; acc_a=al['accuracy']; nc_a=al['noise_corr']
        ax.plot(avals,acc_a,'-o',color=COL_ELIF,lw=2,ms=8,label='Accuracy (%)')
        ax.set_xlabel('alpha (coupling strength)'); ax.set_ylabel('Accuracy (%)',color=COL_ELIF)
        ax.axvline(0.2,ls=':',color='gray',lw=1)
        ax.text(0.21,acc_a.max()*0.95,'default',fontsize=9,color='gray')
        ax2=ax.twinx()
        ax2.plot(avals,nc_a,'-s',color='orange',lw=2,ms=8,label='Noise corr')
        ax2.set_ylabel('Noise correlation',color='orange')
        ax.set_title('Alpha Sensitivity (decoding)',fontsize=9.5)
        lines1,labels1=ax.get_legend_handles_labels()
        lines2,labels2=ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2,labels1+labels2,fontsize=9); _clean(ax)

    fig.suptitle('eLIF Mechanism: LFP, Correlations, Spatial Scales, Coupling',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,'Fig_Mechanism',save_dir)

def fig_summary(all_results, save_dir='.'):
    tn=[]; a_s=[]; a_e=[]
    for n,r in all_results.items():
        mv=r.metrics
        if 'accuracy_std' in mv and 'accuracy_elif' in mv: tn.append(n); a_s.append(mv['accuracy_std']); a_e.append(mv['accuracy_elif'])
    if not tn: return
    fig,axes=plt.subplots(1,2,figsize=(12,5)); x=np.arange(len(tn)); w=0.35
    axes[0].bar(x-w/2,a_s,w,color=COL_STD,label='Std'); axes[0].bar(x+w/2,a_e,w,color=COL_ELIF,label='eLIF'); axes[0].set_xticks(x); axes[0].set_xticklabels(tn,rotation=45,fontsize=9); axes[0].set_ylabel('Acc %'); axes[0].set_title('Accuracy'); axes[0].legend(fontsize=9)
    imp=[(e-s)/max(0.1,s)*100 for s,e in zip(a_s,a_e)]; axes[1].bar(range(len(imp)),imp,color=COL_ELIF); axes[1].axhline(0,ls='--',color='k'); axes[1].set_xticks(range(len(imp))); axes[1].set_xticklabels(tn,rotation=45,fontsize=9); axes[1].set_ylabel('Improvement %'); axes[1].set_title('eLIF vs Std')
    for ax in axes: _clean(ax)
    fig.suptitle('Summary',fontsize=14,fontweight='bold'); fig.tight_layout(rect=[0,0,1,0.92]); _save(fig,'Fig_Summary',save_dir)
