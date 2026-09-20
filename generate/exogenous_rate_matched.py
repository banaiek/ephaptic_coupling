# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pickle, time
from copy import deepcopy
from config import Config
import tasks.alpha_sweep as A
from tasks.threemodel import ThreeModel, aggregate_seeds

MK = ('nc', 'lda', 'spk', 'bits_per_spike', 'dim', 'sync', 'noise_corr')


def _point_exorm(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    alpha, config, pt_seed, target = args
    ap = config.alpha_sweep
    cfg = deepcopy(config); cfg.seed = pt_seed
    geom, count_mask, stim_t, build_input, exo_field = A._setup(cfg)
    tm = ThreeModel(cfg, geom)

    def build_input_exo(s, trial, dI=0.0):
        return build_input(s, trial, dI) + exo_field(s, trial, float(alpha))

    dI = tm.rate_match(build_input_exo, ap.N_stim, count_mask, ap.T_trial, float(target),
                       pt_seed, passes=9, use_elif=False, bidirectional=True)
    c, r = tm.simulate_counts(build_input_exo, ap.N_stim, ap.N_trials, count_mask,
                              ap.T_trial, use_elif=False, dI=dI, store_spikes=True)
    m = A._metrics(cfg, c, r, stim_t, ap.N_stim, ap.N_trials)
    m['dI'] = dI
    return float(alpha), pt_seed, m


def main():
    from multiprocessing import Pool
    t0 = time.time()
    cfg = Config.from_defaults(); ap = cfg.alpha_sweep
    D = pickle.load(open('results/_cache/V3_DEF.pkl', 'rb'))
    alphas = list(ap.alphas)
    tgt = {round(a, 6): D['std']['spk']['mean'][i] for i, a in enumerate(alphas)}
    seed = 6000
    args = [(a, cfg, seed + ai * 9973 + si * 137, tgt[round(a, 6)])
            for ai, a in enumerate(alphas) for si in range(ap.n_seeds)]
    nproc = min(len(args), max(1, (os.cpu_count() or 4) - 2))
    print(f"Recomputing exo_rm (bidirectional) {len(alphas)}x{ap.n_seeds} on {nproc} cores...", flush=True)
    raw = {round(a, 6): [] for a in alphas}
    with Pool(nproc) as pool:
        for (a, sd, m) in pool.imap_unordered(_point_exorm, args):
            raw[round(a, 6)].append(m)
    exo_rm = {k: {'mean': [], 'sem': [], 'per_seed': []} for k in MK + ('dI',)}
    for a in alphas:
        agg = aggregate_seeds(raw[round(a, 6)], MK + ('dI',))
        for k in MK:
            exo_rm[k]['mean'].append(agg[k]['mean']); exo_rm[k]['sem'].append(agg[k]['sem'])
            exo_rm[k]['per_seed'].append([float(d[k]) for d in raw[round(a, 6)]])
        exo_rm['dI']['mean'].append(agg['dI']['mean']); exo_rm['dI']['sem'].append(agg['dI']['sem'])
        print(f"  [alpha={a:.2f}] exo_rm spk={agg['spk']['mean']:.0f} (target {tgt[round(a,6)]:.0f}) "
              f"nc={agg['nc']['mean']:.0f} dim={agg['dim']['mean']:.0f} dI={agg['dI']['mean']:.2f}", flush=True)
    D['exo_rm'] = exo_rm
    pickle.dump(D, open('results/_cache/V3_DEF.pkl', 'wb'))
    print(f"Merged bidirectional exo_rm into V3_DEF ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
