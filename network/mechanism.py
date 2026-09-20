# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np
from copy import deepcopy

from geometry import NetworkGeometry
from network import common as C
from network.integrator import simulate


def _voltage_stats(task, cfg, geom, use_elif, dI=0.0, n_trials=6):
    NE = cfg.network.NE
    vc, vs, vp, rate = [], [], [], []
    for tr in range(n_trials):
        I = task.input(0, tr, dI)
        r = simulate(cfg, geom, I, C.T_TRIAL, use_elif=use_elif, record_voltage=True)
        vc.append(C.voltage_correlation(r.V_trace, NE, task.stim_t))
        a, b = C.shared_private_variance(r.V_trace, NE, task.stim_t)
        vs.append(a)
        vp.append(b)
        rate.append(r.spikes[:NE][:, task.stim_t].sum() / NE
                    / (task.stim_t.sum() * C.DT / 1000.0))
    return dict(v_corr=float(np.mean(vc)), var_shared=float(np.mean(vs)),
                var_private=float(np.mean(vp)), rate=float(np.mean(rate)))


def run_seed(seed, alpha=C.ALPHA, n_trials=20):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    geom = task.geom
    out = {'seed': seed, 'alpha': alpha}

    counts_lif = task.run(cfg, use_elif=False)
    counts_el = task.run(ce, use_elif=True)
    target = counts_lif.sum() / (task.n_stim * task.n_trials)
    dI = task.match_rate(target, ce, use_elif=True, lo=0.0)
    counts_rm = task.run(ce, use_elif=True, dI=dI)
    out['lif'] = C.code_metrics(counts_lif, task.n_stim, task.n_trials, 'LIF')
    out['elif'] = C.code_metrics(counts_el, task.n_stim, task.n_trials, 'eLIF')
    out['rm'] = C.code_metrics(counts_rm, task.n_stim, task.n_trials, 'RM')
    out['dI'] = dI

    out['volt'] = {
        'lif': _voltage_stats(task, cfg, geom, False),
        'elif': _voltage_stats(task, ce, geom, True),
        'rm': _voltage_stats(task, ce, geom, True, dI=dI),
    }

    sub = C.clone(cfg, **{'neuron.V_th': 1e6})
    sub_e = C.clone(ce, **{'neuron.V_th': 1e6})
    out['subthreshold'] = {
        'lif': _voltage_stats(task, sub, geom, False),
        'elif': _voltage_stats(task, sub_e, geom, True),
    }

    counts_same = np.zeros_like(counts_lif)
    counts_other = np.zeros_like(counts_lif)
    shift = task.n_trials // 2
    for s in range(task.n_stim):
        bank = [simulate(ce, geom, task.input(s, tr), C.T_TRIAL, use_elif=True,
                         record_phi=True).phi_trace.astype(np.float32)
                for tr in range(task.n_trials)]
        for tr in range(task.n_trials):
            I = task.input(s, tr)
            r_same = simulate(cfg, geom, I, C.T_TRIAL, use_elif=False,
                              phi_replay=bank[tr])
            counts_same[:, tr, s] = r_same.spikes[:, task.count_mask].sum(1)
            r_open = simulate(cfg, geom, I, C.T_TRIAL, use_elif=False,
                              phi_replay=bank[(tr + shift) % task.n_trials])
            counts_other[:, tr, s] = r_open.spikes[:, task.count_mask].sum(1)
    out['replay_same'] = C.code_metrics(counts_same, task.n_stim, task.n_trials,
                                        'replay (same trial)')
    out['replay_other'] = C.code_metrics(counts_other, task.n_stim, task.n_trials,
                                         'replay (other trial)')

    out['inhibition'] = []
    for scale in (0.0, 0.5, 1.0, 1.5):
        gm = deepcopy(geom)
        gm.W_I = geom.W_I * scale
        c_l = task.run(cfg, geom=gm, use_elif=False)
        m_l = C.code_metrics(c_l, task.n_stim, task.n_trials, f'LIF gI={scale}')
        m_e = C.code_metrics(task.run(ce, geom=gm, use_elif=True),
                             task.n_stim, task.n_trials, f'eLIF gI={scale}')
        target = c_l.sum() / (task.n_stim * task.n_trials)
        dI = task.match_rate(target, ce, geom=gm, use_elif=True, lo=0.0)
        m_r = C.code_metrics(task.run(ce, geom=gm, use_elif=True, dI=dI),
                             task.n_stim, task.n_trials, f'RM gI={scale}')
        out['inhibition'].append({'scale': scale, 'lif': m_l, 'elif': m_e,
                                  'rm': m_r, 'dI': dI})

    out['locality'] = []
    for kernel, sigma in (('global', None), ('gaussian_norm', 0.05),
                          ('gaussian_norm', 0.15), ('gaussian_norm', 0.45)):
        cg = C.clone(ce, **{'geometry.eph_kernel': kernel})
        if sigma is not None:
            cg.geometry.sigma_eph = sigma
        gk = NetworkGeometry.build(cg)
        gk.W_E, gk.W_I, gk.G, gk.source_pos = geom.W_E, geom.W_I, geom.G, geom.source_pos
        m = C.code_metrics(task.run(cg, geom=gk, use_elif=True),
                           task.n_stim, task.n_trials,
                           f'{kernel}' + (f' s={sigma}' if sigma else ''))
        out['locality'].append({'kernel': kernel, 'sigma': sigma, 'metrics': m})
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
