# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from network import common as C
from network.integrator import Intrinsics, homogeneous, simulate

ALL_PARAMS = ('tau_m', 'gain', 'V_rest', 'V_th', 'V_reset', 'tau_ref')
INTRINSIC = ('tau_m', 'V_rest', 'V_th', 'V_reset', 'tau_ref')

VARIANTS = {
    'all': (ALL_PARAMS, False),
    'intrinsic': (INTRINSIC, False),
    'voltage': (('V_rest', 'V_th', 'V_reset'), False),
    'tau': (('tau_m',), False),
    'gain': (('gain',), False),
    'intrinsic_rate_matched': (INTRINSIC, True),
    'all_rate_matched': (ALL_PARAMS, True),
}


def compensate_rates(task, cfg, intr, r_target, iters=3, n_stim=5, n_trials=8):
    reach0 = cfg.neuron.V_th - cfg.neuron.V_rest
    intr.gain = intr.gain * (intr.V_th - intr.V_rest) / reach0
    for _ in range(iters):
        tot = np.zeros(cfg.network.N)
        for s in range(n_stim):
            for tr in range(n_trials):
                r = simulate(cfg, task.geom, task.input(s, tr), C.T_TRIAL,
                             use_elif=False, intr=intr)
                tot += r.spikes[:, task.count_mask].sum(1)
        cur = tot / (n_stim * n_trials)
        factor = np.clip(((r_target + 1.0) / (cur + 1.0)) ** 0.35, 0.8, 1.25)
        intr.gain = np.clip(intr.gain * factor, 0.05, 20.0)
    return intr


def heterogeneous_subset(config, level, seed, which):
    n = config.network.N
    p = config.neuron
    rng = np.random.default_rng(seed)
    if level <= 0:
        return homogeneous(config)
    sig = np.sqrt(np.log(1.0 + level ** 2))
    spread = level * (p.V_th - p.V_rest)

    def lognorm():
        return np.exp(rng.normal(-0.5 * sig ** 2, sig, n))

    draw_tau, draw_gain = lognorm(), lognorm()
    draw_rest = rng.normal(0.0, spread, n)
    draw_reach = rng.normal(0.0, spread, n)
    draw_reset = rng.normal(0.0, spread, n)
    draw_ref = lognorm()
    one = np.ones(n)

    tau_m = p.tau_m * (draw_tau if 'tau_m' in which else one)
    gain = draw_gain if 'gain' in which else one.copy()
    V_rest = one * p.V_rest + (draw_rest if 'V_rest' in which else 0.0)
    reach = one * (p.V_th - p.V_rest) + (draw_reach if 'V_th' in which else 0.0)
    V_th = V_rest + np.maximum(reach, 2.0)
    V_reset = V_rest + (p.V_reset - p.V_rest) + (draw_reset if 'V_reset' in which else 0.0)
    V_reset = np.clip(V_reset, None, V_th - 1.0)
    tau_ref = p.tau_ref * (draw_ref if 'tau_ref' in which else one)
    return Intrinsics(tau_m, V_rest, V_th, np.asarray(V_reset, float),
                      np.clip(tau_ref, 0.1, None), gain)


def band_cells(counts_ref, n_exc=400, frac=0.5):
    r = counts_ref[:n_exc].mean(axis=(1, 2))
    lo, hi = np.quantile(r, [(1 - frac) / 2, 1 - (1 - frac) / 2])
    return np.where((r >= lo) & (r <= hi))[0]


def nc_on_cells(counts, idx, n_pairs=4000, seed=0):
    if len(idx) < 2:
        return float('nan')
    resid = counts - counts.mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    i = idx[rng.integers(0, len(idx), n_pairs)]
    j = idx[rng.integers(0, len(idx), n_pairs)]
    keep = i != j
    i, j = i[keep], j[keep]
    vals = []
    for s in range(counts.shape[2]):
        a, b = resid[i, :, s], resid[j, :, s]
        sa, sb = a.std(1) + 1e-12, b.std(1) + 1e-12
        vals.append((a * b).mean(1) / (sa * sb))
    return float(np.nanmean(vals))


def _score(counts, idx, n_stim, n_trials, n_exc=400):
    r = counts[:n_exc].mean(axis=(1, 2))
    return {'nc_all': C.pair_noise_correlation(counts),
            'nc_band': nc_on_cells(counts, idx),
            'accuracy': float(C.decode_nearest_centroid_loo(counts, n_stim, n_trials)[0]),
            'spikes': float(counts.sum() / (n_stim * n_trials)),
            'rate_cv': float(r.std() / (r.mean() + 1e-12)),
            'quiet_frac': float((r < 0.2 * np.median(r)).mean())}


def run_seed(seed, alpha=C.ALPHA, n_trials=10, levels=(0.1, 0.2, 0.4),
             variants=None):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    out = {'seed': seed, 'levels': list(levels), 'variants': {}}

    c_lif0 = task.run(cfg, use_elif=False)
    target = c_lif0.sum() / (task.n_stim * n_trials)
    r_target = c_lif0.mean(axis=(1, 2))
    idx0 = band_cells(c_lif0)
    out['homogeneous'] = {
        'lif': _score(c_lif0, idx0, task.n_stim, n_trials),
        'elif': _score(task.run(ce, use_elif=True), idx0, task.n_stim, n_trials)}

    chosen = variants or list(VARIANTS)
    for name in chosen:
        which, rate_matched = VARIANTS[name]
        rows = []
        for lev in levels:
            intr = heterogeneous_subset(cfg, lev, seed * 17 + 3, which)
            if rate_matched:
                intr = compensate_rates(task, cfg, intr, r_target)
            dI = task.match_rate(target, cfg, use_elif=False, lo=-2.0, hi=8.0, intr=intr)
            c_lif = task.run(cfg, use_elif=False, dI=dI, intr=intr)
            c_el = task.run(ce, use_elif=True, dI=dI, intr=intr)
            idx = band_cells(c_lif)
            rows.append({'level': lev, 'dI': dI, 'n_band': int(len(idx)),
                         'lif': _score(c_lif, idx, task.n_stim, n_trials),
                         'elif': _score(c_el, idx, task.n_stim, n_trials)})
        out['variants'][name] = rows
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247), nproc=None, variants=None):
    from functools import partial
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), 8)
    with Pool(nproc, maxtasksperchild=1) as pool:
        return pool.map(partial(run_seed, variants=variants), list(seeds))
