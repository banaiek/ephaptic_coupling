# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os

import numpy as np

from network import common as C
from network.mesoscopic import _pop_rate, cross_column_pairs, simulate_pair
from network.integrator import simulate

F_DRIVE = 0.5
DRIVE_AMP = 2.5
BETAS = (0.0, 0.1, 0.2, 0.3)


def run_seed(seed, alpha=C.ALPHA, n_trials=15, betas=BETAS):
    taskA = C.DecodingTask(seed=seed, n_trials=n_trials)
    taskB = C.DecodingTask(seed=seed + 500, n_trials=n_trials)
    cfgA, geomA, cfgB, geomB = taskA.cfg, taskA.geom, taskB.cfg, taskB.geom
    ceA = C.clone(cfgA, **{'elif_params.alpha': alpha})
    ceB = C.clone(cfgB, **{'elif_params.alpha': alpha})
    dt = cfgA.sim.dt
    t = np.arange(taskA.T_steps) * dt
    mask = taskA.count_mask
    rng = np.random.default_rng(seed + 31)
    phases = rng.uniform(0, 2 * np.pi, size=(taskA.n_stim, n_trials))

    def drive(s, tr):
        return DRIVE_AMP * np.sin(2 * np.pi * F_DRIVE * t / 1000.0 + phases[s, tr])

    def mean_deflection(use_local):
        r = simulate(ceA if use_local else cfgA, geomA,
                     taskA.input(0, 0) + drive(0, 0)[None, :], C.T_TRIAL,
                     use_elif=use_local, record_voltage=True)
        return float((r.V_trace[:, taskA.stim_t] - cfgA.neuron.V_rest).mean())

    out = {'seed': seed, 'alpha': alpha, 'betas': list(betas),
           'f_drive': F_DRIVE, 'points': []}
    for use_local in (False, True):
        u_ref = mean_deflection(use_local)
        for beta in betas:
            cA = np.zeros((cfgA.network.N, n_trials, taskA.n_stim))
            cB = np.zeros_like(cA)
            rateA, rateB = [], []
            for s in range(taskA.n_stim):
                for tr in range(n_trials):
                    sA, sB = simulate_pair(
                        ceA if use_local else cfgA, geomA,
                        ceB if use_local else cfgB, geomB,
                        taskA.input(s, tr) + drive(s, tr)[None, :],
                        taskB.input(s, tr), C.T_TRIAL, beta, 1.0,
                        use_local_field=use_local, u_ref=u_ref)
                    cA[:, tr, s] = sA[:, mask].sum(1)
                    cB[:, tr, s] = sB[:, mask].sum(1)
                    if s == 0:
                        rateA.append(_pop_rate(sA[:cfgA.network.NE][:, taskA.stim_t], dt))
                        rateB.append(_pop_rate(sB[:cfgB.network.NE][:, taskB.stim_t], dt))
            out['points'].append({
                'beta': beta, 'local_field': use_local,
                'driven_A': C.code_metrics(cA, taskA.n_stim, n_trials, 'column A'),
                'receiving_B': C.code_metrics(cB, taskB.n_stim, n_trials, 'column B'),
                'between_pairs': cross_column_pairs(cA, cB),
                'between_rate': float(np.corrcoef(np.concatenate(rateA),
                                                  np.concatenate(rateB))[0, 1])})
    return out


def run(seeds=(4242, 4243, 4244, 4245), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
