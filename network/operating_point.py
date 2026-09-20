# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from geometry import NetworkGeometry
from network import common as C
from network.integrator import heterogeneous


def _split_noise(total, frac):
    return total * np.sqrt(frac), total * np.sqrt(1.0 - frac)


def run_seed(seed, alpha=C.ALPHA, n_trials=15,
             fracs=(0.0, 0.15, 0.35, 0.6, 1.0), total_noise=5.66,
             het_level=0.0):
    out = {'seed': seed, 'alpha': alpha, 'het_level': het_level,
           'fracs': list(fracs), 'points': []}
    for frac in fracs:
        shared, private = _split_noise(total_noise, frac)
        over = {'alpha_sweep.shared_noise_amp': float(shared),
                'alpha_sweep.private_noise_amp': float(private)}
        task = C.DecodingTask(seed=seed, n_trials=n_trials)
        cfg = C.clone(task.cfg, **over)
        task.cfg = cfg
        geom = NetworkGeometry.build(cfg)
        geom.W_E, geom.W_I = task.geom.W_E, task.geom.W_I
        geom.G, geom.source_pos = task.geom.G, task.geom.source_pos
        ce = C.clone(cfg, **{'elif_params.alpha': alpha})

        kw = ({} if het_level <= 0
              else {'intr': heterogeneous(cfg, het_level, seed=seed * 17 + 3)})

        c_lif = task.run(cfg, geom=geom, use_elif=False, **kw)
        c_el = task.run(ce, geom=geom, use_elif=True, **kw)
        target = c_lif.sum() / (task.n_stim * n_trials)
        excess = c_el.sum() / (task.n_stim * n_trials)
        dI = (task.match_rate(target, ce, geom=geom, use_elif=True, lo=0.0, **kw)
              if excess > target * 1.01 else 0.0)
        c_rm = task.run(ce, geom=geom, use_elif=True, dI=dI, **kw)
        out['points'].append({
            'frac_shared': frac, 'shared': float(shared), 'private': float(private),
            'dI': dI,
            'lif': C.code_metrics(c_lif, task.n_stim, n_trials, 'LIF'),
            'elif': C.code_metrics(c_el, task.n_stim, n_trials, 'eLIF'),
            'rm': C.code_metrics(c_rm, task.n_stim, n_trials, 'RM')})
    return out


def _lif_job(seed):
    return run_seed(seed)


def _het_job(seed):
    return run_seed(seed, het_level=0.2)


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), 8)
    with Pool(nproc, maxtasksperchild=1) as pool:
        lif = pool.map(_lif_job, list(seeds))
        het = pool.map(_het_job, list(seeds))
    return {'lif': lif, 'het': het}
