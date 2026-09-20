# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from geometry import NetworkGeometry
from network import common as C


def run_seed(seed, alpha=C.ALPHA, n_trials=20):
    task = C.DecodingTask(seed=seed, n_trials=n_trials)
    cfg = task.cfg
    out = {'seed': seed, 'alpha': alpha}
    c_lif = task.run(cfg, use_elif=False)
    target = c_lif.sum() / (task.n_stim * n_trials)
    out['lif'] = C.code_metrics(c_lif, task.n_stim, n_trials, 'LIF')

    for tag, pop in (('split_EI', 'split_EI'), ('all', 'all'), ('all_to_E', 'all')):
        ce = C.clone(cfg, **{'elif_params.alpha': alpha,
                             'geometry.field_population': pop})
        geom = NetworkGeometry.build(ce)
        geom.W_E, geom.W_I = task.geom.W_E, task.geom.W_I
        geom.G, geom.source_pos = task.geom.G, task.geom.source_pos
        if tag == 'all_to_E':
            M = geom.W_eph_all
            M = M.toarray() if hasattr(M, 'toarray') else np.array(M)
            M[cfg.network.NE:, :] = 0.0
            import scipy.sparse as sp
            geom.W_eph_all = sp.csr_matrix(M)
        counts = task.run(ce, geom=geom, use_elif=True)
        dI = task.match_rate(target, ce, geom=geom, use_elif=True, lo=0.0)
        out[tag] = C.code_metrics(counts, task.n_stim, n_trials, tag)
        out[tag + '_rm'] = C.code_metrics(task.run(ce, geom=geom, use_elif=True, dI=dI),
                                          task.n_stim, n_trials, tag + ' RM')
    return out


def run(seeds=(4242, 4243, 4244, 4245, 4246, 4247), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
