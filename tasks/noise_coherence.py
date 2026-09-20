# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np
from scipy.spatial.distance import cdist
from config import Config
from geometry import NetworkGeometry
from tasks.threemodel import ThreeModel


def _smoother(pos, sg):
    if sg <= 0:
        S = np.eye(pos.shape[0])
    else:
        S = np.exp(-cdist(pos, pos) ** 2 / (2 * sg ** 2))
    norm = np.sqrt((S ** 2).sum(axis=1, keepdims=True))
    return S, norm


def _point(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    li, sg, config, seed = args
    config.elif_params.alpha = getattr(config.noise_coherence, 'alpha', config.elif_params.alpha)
    nc = config.noise_coherence
    dt = config.sim.dt
    N, NE = config.network.N, config.network.NE
    geom = NetworkGeometry.build(config)
    tm = ThreeModel(config, geom)
    x = geom.pos[:, 0]
    T_steps = int(round(nc.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    stim_t = (t >= nc.stim_onset) & (t < nc.stim_offset)
    count_mask = (t >= nc.count_start) & (t < nc.count_end)
    S, norm = _smoother(geom.pos, sg)
    m = nc.signal_mode
    patterns = np.stack([nc.stim_amp * np.cos(2 * np.pi * m * x[:NE] + 2 * np.pi * s / nc.N_stim)
                         for s in range(nc.N_stim)])

    def build_input(s, trial, dI=0.0):
        rng = np.random.default_rng(seed + 100000 * li + 1000 * s + trial)
        white = rng.standard_normal((N, T_steps))
        noise = nc.noise_amp * (S @ white) / norm
        I = (nc.baseline - dI) + noise
        I[:NE][:, stim_t] += patterns[s][:, None]
        return I

    r = tm.run_three(build_input, nc.N_stim, nc.N_trials, count_mask, nc.T_trial, base_seed=seed)
    res = {'li': li, 'sigma': sg, 'dI': r['dI'], 'noise_corr': _mean_noise_corr(r['std'])}
    for k in ('std', 'elif', 'rm'):
        a, l = ThreeModel.decode(r[k], nc.N_stim, nc.N_trials)
        res[k] = {'nc': a, 'lda': l}
    return res


def run(config: Config, seed=5500, nproc=None):
    from multiprocessing import Pool
    from tasks.threemodel import mean_sem
    nc = config.noise_coherence
    seeds = [seed + 137 * i for i in range(getattr(nc, 'n_seeds', 1))]
    args = [(li, sg, config, sd) for li, sg in enumerate(nc.sigma_noise_levels) for sd in seeds]
    nproc = nproc or min(len(args), max(1, (os.cpu_count() or 4) - 2))
    print(f"  Noise-coherence: {len(nc.sigma_noise_levels)} levels x {len(seeds)} seeds, "
          f"alpha={getattr(nc,'alpha',config.elif_params.alpha)}, on {nproc} cores...", flush=True)
    raw = {li: [] for li in range(len(nc.sigma_noise_levels))}
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_point, args):
            raw[r['li']].append(r)
    out = {'sigma_noise_levels': list(nc.sigma_noise_levels), 'N_stim': nc.N_stim,
           'chance': 100.0 / nc.N_stim, 'signal_mode': nc.signal_mode,
           'alpha': getattr(nc, 'alpha', config.elif_params.alpha), 'n_seeds': len(seeds),
           'nc': {k: {'mean': [], 'sem': [], 'per_seed': []} for k in ('std', 'elif', 'rm')},
           'lda': {k: {'mean': [], 'sem': [], 'per_seed': []} for k in ('std', 'elif', 'rm')},
           'noise_corr': [], 'noise_corr_sem': [], 'dI': [], 'info': _info(config)}
    for li in range(len(nc.sigma_noise_levels)):
        rs = raw[li]
        for dec in ('nc', 'lda'):
            for k in ('std', 'elif', 'rm'):
                vals = [r[k][dec] for r in rs]
                mu, se = mean_sem(vals)
                out[dec][k]['mean'].append(mu); out[dec][k]['sem'].append(se)
                out[dec][k]['per_seed'].append([float(v) for v in vals])
        ncv = [r['noise_corr'] for r in rs]
        out['noise_corr'].append(float(np.mean(ncv)))
        out['noise_corr_sem'].append(float(np.std(ncv, ddof=1) / np.sqrt(max(1, len(ncv)))))
        out['dI'].append(float(np.mean([r['dI'] for r in rs])))
        print(f"  [sigma={nc.sigma_noise_levels[li]:.2f}] NC Std/eLIF/RM="
              f"{out['nc']['std']['mean'][-1]:.0f}/{out['nc']['elif']['mean'][-1]:.0f}/"
              f"{out['nc']['rm']['mean'][-1]:.0f} corr={out['noise_corr'][-1]:.3f}", flush=True)
    return out


def _info(config):
    nc = config.noise_coherence; net = config.network; geo = config.geometry
    return {
        'title': 'noise spatial structure: where field decorrelation helps vs hurts',
        'task': (f"{nc.N_stim} phase stimuli at low wavenumber m={nc.signal_mode} "
                 f"(coherent signal, amp {nc.stim_amp}). Decode phase from spike "
                 f"counts."),
        'sweep': (f"noise spatial correlation length sigma_noise = "
                  f"{list(nc.sigma_noise_levels)} (total power fixed at "
                  f"{nc.noise_amp}). 0=private/white, large=shared/coherent."),
        'predict': ("eLIF helps where noise carries REMOVABLE spatial correlation "
                    "(intermediate sigma, where Std is hardest); the cost appears "
                    "only for strongly coherent noise (amplified) and is escaped by "
                    "the rate-matched control RM."),
        'model': (f"N={net.N}, alpha={getattr(nc,'alpha',config.elif_params.alpha)} "
                  f"(fraction of critical), kernel={geo.eph_kernel}, sigma_eph={geo.sigma_eph}."),
        'stats': f"{nc.N_trials} trials x {getattr(nc,'n_seeds',1)} seeds, mean +/- SEM.",
    }


def _mean_noise_corr(counts, n_pairs=400, seed=7):
    N, T, S = counts.shape
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, N, (n_pairs, 2))
    cc = []
    for s in range(S):
        z = counts[:, :, s] - counts[:, :, s].mean(1, keepdims=True)
        sd = z.std(1) + 1e-9
        for a, b in pairs:
            if a == b: continue
            cc.append(np.mean(z[a] * z[b]) / (sd[a] * sd[b]))
    return float(np.nanmean(cc))
