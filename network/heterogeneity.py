# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from network import common as C
from network.integrator import heterogeneous
from network.heterogeneity_bands import band_cells, nc_on_cells

def _score(counts, idx, n_stim, n_trials, tag, n_exc=400):
    r = counts[:n_exc].mean(axis=(1, 2))
    out = C.code_metrics(counts, n_stim, n_trials, tag)
    out['nc_band'] = nc_on_cells(counts, idx)
    out['rate_cv'] = float(r.std() / (r.mean() + 1e-12))
    out['quiet_frac'] = float((r < 0.2 * np.median(r)).mean())
    return out


def run_seed(seed, alpha=C.ALPHA, n_trials=20,
             levels=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8)):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    out = {'seed': seed, 'alpha': alpha, 'levels': list(levels), 'points': []}

    c_ref = task.run(cfg, use_elif=False, intr=heterogeneous(cfg, 0.0))
    target = c_ref.sum() / (task.n_stim * n_trials)
    out['target_spikes'] = float(target)

    for lev in levels:
        intr = heterogeneous(cfg, lev, seed=seed * 17 + 3)
        dI_lif = (0.0 if lev <= 0 else
                  task.match_rate(target, cfg, use_elif=False,
                                  lo=-3.0, hi=8.0, intr=intr))
        c_lif = task.run(cfg, use_elif=False, dI=dI_lif, intr=intr)
        c_el = task.run(ce, use_elif=True, dI=dI_lif, intr=intr)
        dI_el = task.match_rate(target, ce, use_elif=True,
                                lo=dI_lif, hi=dI_lif + 6.0, intr=intr)
        c_rm = task.run(ce, use_elif=True, dI=dI_el, intr=intr)
        idx = band_cells(c_lif)
        out['points'].append({
            'level': lev, 'n_band': int(len(idx)),
            'tau_m': intr.tau_m.tolist(),
            'V_rest': intr.V_rest.tolist(),
            'V_th': intr.V_th.tolist(),
            'lif': _score(c_lif, idx, task.n_stim, n_trials, 'LIF'),
            'elif': _score(c_el, idx, task.n_stim, n_trials, 'eLIF'),
            'rm': _score(c_rm, idx, task.n_stim, n_trials, 'RM'),
            'dI_lif': dI_lif, 'dI': dI_el})
    return out


def run(seeds=tuple(6242 + i for i in range(20)), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), 8)
    with Pool(nproc, maxtasksperchild=1) as pool:
        return pool.map(run_seed, list(seeds))
