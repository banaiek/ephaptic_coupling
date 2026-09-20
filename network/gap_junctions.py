# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from network import common as C
from network.integrator import build_gap_operator, simulate


def operator_spectra(pos, sigma, alpha, g):
    from field import build_kernel
    import scipy.sparse as sp
    M = build_kernel(pos, 'gaussian_norm', sigma, sparsify=0.0)
    lam = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
    gap = build_gap_operator(pos, sigma, 1.0, sparsify=0.0)
    mu = np.sort(np.real(np.linalg.eigvals(gap.toarray())))[::-1]
    return {'lam': lam, 'gain_eph': 1.0 / (1.0 - alpha * lam),
            'mu': mu, 'gain_gap': 1.0 / (1.0 - g * mu)}


def _voltage_by_distance(task, cfg, geom, n_trials=6, n_bins=10, **kw):
    NE = cfg.network.NE
    pos = geom.pos[:NE]
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(NE, 1)
    dv = d[iu]
    edges = np.linspace(0, dv.max(), n_bins + 1)
    acc = np.zeros(n_bins)
    for tr in range(n_trials):
        r = simulate(cfg, geom, task.input(0, tr), C.T_TRIAL, record_voltage=True, **kw)
        X = r.V_trace[:NE][:, task.stim_t]
        X = X - X.mean(1, keepdims=True)
        X = X / (X.std(1, keepdims=True) + 1e-12)
        Ccorr = (X @ X.T) / X.shape[1]
        cv = Ccorr[iu]
        for b in range(n_bins):
            sel = (dv >= edges[b]) & (dv < edges[b + 1])
            acc[b] += np.nanmean(cv[sel]) if sel.any() else np.nan
    return 0.5 * (edges[:-1] + edges[1:]), acc / n_trials


def run_seed(seed, alpha=C.ALPHA, n_trials=20, gaps=(0.05, 0.1, 0.2, 0.4)):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    ce = C.clone(cfg, **{'elif_params.alpha': alpha})
    geom = task.geom
    sigma = cfg.geometry.sigma_eph
    out = {'seed': seed, 'alpha': alpha}

    c_lif = task.run(cfg, use_elif=False)
    target = c_lif.sum() / (task.n_stim * task.n_trials)
    out['lif'] = C.code_metrics(c_lif, task.n_stim, task.n_trials, 'LIF')

    c_el = task.run(ce, use_elif=True)
    dI = task.match_rate(target, ce, use_elif=True, lo=0.0)
    out['elif'] = C.code_metrics(c_el, task.n_stim, task.n_trials, 'eLIF')
    out['elif_rm'] = C.code_metrics(task.run(ce, use_elif=True, dI=dI),
                                    task.n_stim, task.n_trials, 'eLIF RM')

    out['gap'] = []
    for g in gaps:
        op = build_gap_operator(geom.pos, sigma, g)
        c_g = task.run(cfg, use_elif=False, gap_operator=op)
        dIg = task.match_rate(target, cfg, use_elif=False, gap_operator=op)
        c_grm = task.run(cfg, use_elif=False, dI=dIg, gap_operator=op)
        out['gap'].append({'g': g, 'dI': dIg,
                           'raw': C.code_metrics(c_g, task.n_stim, task.n_trials, f'gap g={g}'),
                           'rm': C.code_metrics(c_grm, task.n_stim, task.n_trials,
                                                f'gap g={g} RM')})

    if seed == 4242:
        op = build_gap_operator(geom.pos, sigma, 0.2)
        d, v_lif = _voltage_by_distance(task, cfg, geom, use_elif=False)
        _, v_el = _voltage_by_distance(task, ce, geom, use_elif=True)
        _, v_gap = _voltage_by_distance(task, cfg, geom, use_elif=False, gap_operator=op)
        out['vdist'] = {'d': d.tolist(), 'lif': v_lif.tolist(),
                        'elif': v_el.tolist(), 'gap': v_gap.tolist()}
        out['spectra'] = {k: np.asarray(v).tolist() for k, v in
                          operator_spectra(geom.pos, sigma, alpha, 0.2).items()}
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
