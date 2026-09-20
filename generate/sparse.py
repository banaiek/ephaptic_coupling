# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle, time
import numpy as np
from config import Config
from geometry import NetworkGeometry

N_SEEDS = 20
N_TRIALS = 40


def _seed(args):
    si, base = args
    from tasks.sparse import SparseCodingTask
    cfg = Config.from_defaults()
    cfg.seed = base + si * 101
    cfg.sparse.N_trials = N_TRIALS
    geom = NetworkGeometry.build(cfg)
    res = SparseCodingTask(cfg, geom).run(seed=cfg.seed)
    d, m = res.data, res.metrics
    return {
        'seed': cfg.seed,
        'sparsity_levels': np.array(d['sparsity_levels']),
        'noise_levels': np.array(d['noise_levels']),
        'acc': {'std': np.array(d['accuracy_std']), 'elif': np.array(d['accuracy_elif']),
                'rm': np.array(d['accuracy_elif_rm'])},
        'mean': {'std': m['mean_accuracy_std'], 'elif': m['mean_accuracy_elif'],
                 'rm': m['mean_accuracy_elif_rm']},
        'confusion': {'std': np.array(d['confusion_std']), 'elif': np.array(d['confusion_elif']),
                      'rm': np.array(d['confusion_elif_rm'])},
        'lda_confusion': {'std': np.array(d['lda_confusion_std']), 'elif': np.array(d['lda_confusion_elif']),
                          'rm': np.array(d['lda_confusion_elif_rm'])},
        'spk': {'std': m['spikes_per_trial_std'], 'elif': m['spikes_per_trial_elif']},
        'chance': m['chance_level'],
    }


def main():
    from multiprocessing import Pool
    os.makedirs('source_data', exist_ok=True)
    args = [(si, 2024) for si in range(N_SEEDS)]
    nproc = min(N_SEEDS, max(1, (os.cpu_count() or 4) - 2))
    print(f"Sparse source: {N_SEEDS} seeds x {N_TRIALS} trials (full grid) on {nproc} cores", flush=True)
    t0 = time.time(); res = []
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_seed, args):
            res.append(r)
            print(f"  seed {r['seed']}: mean acc Std/eLIF/RM="
                  f"{r['mean']['std']:.0f}/{r['mean']['elif']:.0f}/{r['mean']['rm']:.0f} "
                  f"[{len(res)}/{N_SEEDS}, {time.time()-t0:.0f}s]", flush=True)
    res.sort(key=lambda d: d['seed'])
    src = {'task': 'sparse', 'n_seeds': N_SEEDS, 'n_trials': N_TRIALS,
           'sparsity_levels': res[0]['sparsity_levels'], 'noise_levels': res[0]['noise_levels'],
           'chance': res[0]['chance'], 'seeds': res,
           'mean': {k: np.array([r['mean'][k] for r in res]) for k in ('std', 'elif', 'rm')},
           'acc_grid': {k: np.mean([r['acc'][k] for r in res], axis=0) for k in ('std', 'elif', 'rm')},
           'acc_by_seed': {k: np.array([r['acc'][k] for r in res]) for k in ('std', 'elif', 'rm')},
           'confusion': {k: np.mean([r['confusion'][k] for r in res], axis=0) for k in ('std', 'elif', 'rm')},
           'lda_confusion': {k: np.mean([r['lda_confusion'][k] for r in res], axis=0) for k in ('std', 'elif', 'rm')},
           'spk': {k: np.array([r['spk'][k] for r in res]) for k in ('std', 'elif')}}
    with open('source_data/sparse_source.pkl', 'wb') as f:
        pickle.dump(src, f)
    print(f"Saved source_data/sparse_source.pkl ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == '__main__':
    main()
