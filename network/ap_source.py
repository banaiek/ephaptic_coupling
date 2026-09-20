# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from field import build_kernel
from network import common as C
from network.biophysics import ap_field_template

AP_GAINS = (0.0, 13.0, 40.0, 80.0, 160.0)
AP_CALIBRATED = 40.0
AP_SIGMA_RATIO = 1.0 / 3.0


def _pair(task, cfg_lif, cfg_elif, base_counts, tag, **sim_kwargs):
    n = task.n_stim * task.n_trials
    target = base_counts.sum() / n
    c_el = task.run(cfg_elif, use_elif=True, **sim_kwargs)
    excess = c_el.sum() / n
    dI = (task.match_rate(target, cfg_elif, use_elif=True, lo=0.0, **sim_kwargs)
          if excess > target * 1.01 else 0.0)
    c_rm = task.run(cfg_elif, use_elif=True, dI=dI, **sim_kwargs)
    return {'tag': tag, 'dI': dI,
            'lif': C.code_metrics(base_counts, task.n_stim, task.n_trials, 'LIF'),
            'elif': C.code_metrics(c_el, task.n_stim, task.n_trials, 'eLIF'),
            'rm': C.code_metrics(c_rm, task.n_stim, task.n_trials, 'RM')}


def run_seed(seed, alpha=C.ALPHA, n_trials=20):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    out = {'seed': seed, 'alpha': alpha}
    base = task.run(cfg, use_elif=False)

    out['refractory'] = []
    for tref in (0.0, 1.0, 2.0, 3.0, 4.0):
        cl = C.clone(cfg, **{'neuron.tau_ref': tref})
        cel = C.clone(ce, **{'neuron.tau_ref': tref})
        b = base if tref == cfg.neuron.tau_ref else task.run(cl, use_elif=False)
        out['refractory'].append({'tau_ref': tref,
                                  **_pair(task, cl, cel, b, f'tau_ref={tref}')})

    out['reset'] = []
    for vres in (-64.0, -62.0, -60.0, -58.0, -56.0):
        cl = C.clone(cfg, **{'neuron.V_reset': vres})
        cel = C.clone(ce, **{'neuron.V_reset': vres})
        b = base if vres == cfg.neuron.V_reset else task.run(cl, use_elif=False)
        out['reset'].append({'V_reset': vres,
                             **_pair(task, cl, cel, b, f'V_reset={vres}')})

    out['field_mode'] = [{'mode': m, **_pair(task, cfg, ce, base, f'field={m}',
                                             field_mode=m)}
                         for m in ('sub', 'hold')]

    tmpl = ap_field_template(C.DT)
    ap_op = build_kernel(task.geom.pos, cfg.geometry.eph_kernel,
                         cfg.geometry.sigma_eph * AP_SIGMA_RATIO,
                         sparsify=cfg.geometry.eph_sparsify)
    out['ap_gain'] = []
    for gain in AP_GAINS:
        out['ap_gain'].append({'gain': gain,
                               **_pair(task, cfg, ce, base, f'ap gain={gain}',
                                       field_mode='ap', ap_gain=gain,
                                       ap_template=tmpl, ap_operator=ap_op)})
    out['ap_gain_wide'] = [{'gain': gain,
                            **_pair(task, cfg, ce, base, f'ap wide gain={gain}',
                                    field_mode='ap', ap_gain=gain, ap_template=tmpl)}
                           for gain in AP_GAINS]
    out['ap_template'] = {'t': (np.arange(len(tmpl)) * C.DT).tolist(),
                          'w': tmpl.tolist(), 'sigma_ratio': AP_SIGMA_RATIO}

    cneg = C.clone(cfg, **{'elif_params.alpha': -alpha})
    c_neg = task.run(cneg, use_elif=True)
    target = base.sum() / (task.n_stim * task.n_trials)
    dI_neg = task.match_rate(target, cneg, use_elif=True, lo=-6.0, hi=0.0)
    c_neg_rm = task.run(cneg, use_elif=True, dI=dI_neg)
    out['sign'] = {'lif': C.code_metrics(base, task.n_stim, task.n_trials, 'LIF'),
                   'negative': C.code_metrics(c_neg, task.n_stim, task.n_trials, 'alpha<0'),
                   'negative_rm': C.code_metrics(c_neg_rm, task.n_stim, task.n_trials,
                                                 'alpha<0 RM'),
                   'dI': dI_neg}
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
