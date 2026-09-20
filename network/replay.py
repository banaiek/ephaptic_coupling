# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os

import numpy as np

from network import common as C
from network.integrator import simulate


def run_seed(seed, alpha=C.ALPHA, n_trials=20):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    geom, mask = task.geom, task.count_mask
    half, shift = task.n_stim // 2, n_trials // 2
    N = cfg.network.N

    lif = task.run(cfg, use_elif=False)
    target = lif.sum() / (task.n_stim * n_trials)
    out = {'seed': seed, 'lif': C.code_metrics(lif, task.n_stim, n_trials, 'LIF')}

    dI_e = task.match_rate(target, ce, use_elif=True, lo=0.0)
    banks, el = [], np.zeros((N, n_trials, task.n_stim))
    for s in range(task.n_stim):
        phi = []
        for tr in range(n_trials):
            r = simulate(ce, geom, task.input(s, tr, dI_e), C.T_TRIAL,
                         use_elif=True, record_phi=True)
            phi.append(r.phi_trace.astype(np.float32))
            el[:, tr, s] = r.spikes[:, mask].sum(1)
        banks.append(phi)
    out['elif_rm'] = C.code_metrics(el, task.n_stim, n_trials, 'eLIF rate-matched')
    out['elif_rm']['dI'] = float(dI_e)

    src = {'same': lambda s, tr: banks[s][(tr + shift) % n_trials],
           'xstim': lambda s, tr: banks[(s + half) % task.n_stim][tr]}
    for key, f in src.items():
        dI_r = _match(task, cfg, geom, target, f)
        counts = np.zeros((N, n_trials, task.n_stim))
        for s in range(task.n_stim):
            for tr in range(n_trials):
                r = simulate(cfg, geom, task.input(s, tr, dI_r), C.T_TRIAL,
                             use_elif=False, phi_replay=f(s, tr))
                counts[:, tr, s] = r.spikes[:, mask].sum(1)
        name = f'replay_{key}_rm'
        out[name] = C.code_metrics(counts, task.n_stim, n_trials, name)
        out[name]['dI'] = float(dI_r)
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))


def _match(task, cfg, geom, target, phi_for, lo=-6.0, hi=6.0,
           passes=9, n_probe=4, n_stim_probe=3):
    n_s = min(n_stim_probe, task.n_stim)

    def probe(dI):
        tot = 0.0
        for s in range(n_s):
            for tr in range(n_probe):
                r = simulate(cfg, geom, task.input(s, tr, dI), C.T_TRIAL,
                             use_elif=False, phi_replay=phi_for(s, tr))
                tot += r.spikes[:, task.count_mask].sum()
        return tot / (n_s * n_probe)

    for _ in range(passes):
        mid = (lo + hi) / 2
        if probe(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
