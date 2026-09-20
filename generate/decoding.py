# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle, time
import numpy as np
from copy import deepcopy
from config import Config
from geometry import NetworkGeometry
from tasks.threemodel import ThreeModel
from analysis import decode_nearest_centroid_loo, decode_sklearn, dprime_matrix, noise_correlations

N_SEEDS = 20
N_TRIALS = 150
N_STIM = 10


def _metrics(counts, N_stim, N_trials):
    nc_acc, conf = decode_nearest_centroid_loo(counts, N_stim, N_trials)
    N = counts.shape[0]
    X = np.zeros((N_trials * N_stim, N)); y = np.zeros(N_trials * N_stim, int)
    for s in range(N_stim):
        idx = slice(s * N_trials, (s + 1) * N_trials); X[idx] = counts[:, :, s].T; y[idx] = s
    try:
        lda = decode_sklearn(X, y, method='lda')['accuracy']
    except Exception:
        lda = 100.0 / N_stim
    try:
        svm = decode_sklearn(X, y, method='svm')['accuracy']
    except Exception:
        svm = 100.0 / N_stim
    D = dprime_matrix(counts, y, N_stim)
    mask = np.triu(np.ones((N_stim, N_stim), bool), 1)
    nccorr, _ = noise_correlations(counts, n_pairs=500, seed=999)
    spk = X.sum() / (N_stim * N_trials)
    return {'nc': float(nc_acc), 'lda': float(lda), 'svm': float(svm),
            'dprime': float(np.nanmean(D[mask])), 'noise_corr': float(np.nanmean(nccorr)),
            'spk': float(spk), 'confusion': conf / N_trials}


def _seed(args):
    si, base = args
    cfg = Config.from_defaults()
    cfg.seed = base + si * 101
    p = cfg.decoding
    dt = cfg.sim.dt
    N, NE = cfg.network.N, cfg.network.NE
    geom = NetworkGeometry.build(cfg)
    tm = ThreeModel(cfg, geom)
    T_steps = int(round(p.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    stim_t = (t >= p.stim_onset) & (t < p.stim_offset)
    count_mask = (t >= p.count_start) & (t < p.count_end)
    rng = np.random.default_rng(cfg.seed + 5)
    pref = rng.integers(0, N_STIM, size=NE)
    tuning = np.zeros((NE, N_STIM))
    for n in range(NE):
        for s in range(N_STIM):
            d = min(abs(pref[n] - s), N_STIM - abs(pref[n] - s))
            tuning[n, s] = 0.3 + 0.7 * np.exp(-d ** 2 / (2 * p.tuning_width ** 2))

    def build_input(s, trial, dI=0.0):
        rng_n = np.random.default_rng(cfg.seed * 17 + s * N_TRIALS + trial)
        sn = tm._tm_shared(T_steps, p.shared_noise_amp, rng_n, dt) if hasattr(tm, '_tm_shared') else _shared(geom, T_steps, p.shared_noise_amp, rng_n, dt)
        rng_p = np.random.default_rng(cfg.seed * 17 + 900000 + s * N_TRIALS + trial)
        I = (p.I_baseline - dI) + sn + p.private_noise_amp * rng_p.standard_normal((N, T_steps))
        I[:NE][:, stim_t] += p.stim_amp_shared
        I[:NE][:, stim_t] += (p.stim_amp_signal * tuning[:, s])[:, None]
        return I

    r = tm.run_three(build_input, N_STIM, N_TRIALS, count_mask, p.T_trial, base_seed=cfg.seed)
    out = {'seed': cfg.seed, 'dI': r['dI']}
    for k in ('std', 'elif', 'rm'):
        out[k] = _metrics(r[k], N_STIM, N_TRIALS)
    return out


def _shared(geom, T_steps, amp, rng, dt, smooth_ms=50.0):
    K = geom.G.shape[1]
    xi = rng.standard_normal((K, T_steps))
    kw = max(1, int(round(smooth_ms / dt))); kb = np.ones(kw) / kw
    for k in range(K):
        xi[k] = np.convolve(xi[k], kb, mode='same')
    return amp * (geom.G @ xi)


def main():
    from multiprocessing import Pool
    os.makedirs('source_data', exist_ok=True)
    args = [(si, 2024) for si in range(N_SEEDS)]
    nproc = min(N_SEEDS, max(1, (os.cpu_count() or 4) - 2))
    print(f"Decoding source: {N_SEEDS} seeds x {N_TRIALS} trials x {N_STIM} stim on {nproc} cores", flush=True)
    t0 = time.time(); res = []
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_seed, args):
            res.append(r)
            print(f"  seed {r['seed']}: NC Std/eLIF/RM="
                  f"{r['std']['nc']:.0f}/{r['elif']['nc']:.0f}/{r['rm']['nc']:.0f} "
                  f"corr={r['std']['noise_corr']:.3f}/{r['elif']['noise_corr']:.3f}/{r['rm']['noise_corr']:.3f} "
                  f"[{len(res)}/{N_SEEDS}, {time.time()-t0:.0f}s]", flush=True)
    res.sort(key=lambda d: d['seed'])
    src = {'task': 'decoding', 'n_seeds': N_SEEDS, 'n_trials': N_TRIALS, 'n_stim': N_STIM,
           'chance': 100.0 / N_STIM, 'seeds': res,
           'metrics': {k: {m: np.array([r[k][m] for r in res]) for m in ('nc', 'lda', 'svm', 'dprime', 'noise_corr', 'spk')}
                       for k in ('std', 'elif', 'rm')},
           'confusion': {k: np.mean([r[k]['confusion'] for r in res], axis=0) for k in ('std', 'elif', 'rm')}}
    with open('source_data/decoding_source.pkl', 'wb') as f:
        pickle.dump(src, f)
    print(f"Saved source_data/decoding_source.pkl ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == '__main__':
    main()
