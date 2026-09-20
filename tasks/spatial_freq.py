# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np
from config import Config
from geometry import NetworkGeometry
from tasks.threemodel import ThreeModel


def _shared_noise(geom, T_steps, amp, rng, dt, smooth_ms=50.0):
    K = geom.G.shape[1]
    xi = rng.standard_normal((K, T_steps))
    kw = max(1, int(round(smooth_ms / dt))); kb = np.ones(kw) / kw
    for k in range(K):
        xi[k] = np.convolve(xi[k], kb, mode='same')
    return amp * (geom.G @ xi)


def _point(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    m, config, seed = args
    config.elif_params.alpha = getattr(config.spatial_freq, 'alpha', config.elif_params.alpha)
    sp = config.spatial_freq
    dt = config.sim.dt
    N, NE = config.network.N, config.network.NE
    geom = NetworkGeometry.build(config)
    tm = ThreeModel(config, geom)
    x = geom.pos[:, 0]
    T_steps = int(round(sp.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    stim_t = (t >= sp.stim_onset) & (t < sp.stim_offset)
    count_mask = (t >= sp.count_start) & (t < sp.count_end)
    patterns = np.stack([sp.stim_amp * np.cos(2 * np.pi * m * x[:NE] + 2 * np.pi * s / sp.N_stim)
                         for s in range(sp.N_stim)])

    def build_input(s, trial, dI=0.0):
        rng = np.random.default_rng(seed + 100000 * m + 1000 * s + trial)
        sn = _shared_noise(geom, T_steps, sp.shared_noise_amp, rng, dt)
        rng_p = np.random.default_rng(seed + 7 + 100000 * m + 1000 * s + trial)
        I = (sp.baseline - dI) + sn + sp.private_noise_amp * rng_p.standard_normal((N, T_steps))
        I[:NE][:, stim_t] += patterns[s][:, None]
        return I

    r = tm.run_three(build_input, sp.N_stim, sp.N_trials, count_mask, sp.T_trial, base_seed=seed)
    res = {'m': m, 'dI': r['dI']}
    for k in ('std', 'elif', 'rm'):
        nc, lda = ThreeModel.decode(r[k], sp.N_stim, sp.N_trials)
        res[k] = {'nc': nc, 'lda': lda, 'spk': ThreeModel.window_total(r[k], sp.N_stim, sp.N_trials)}
    return res


def run(config: Config, seed=5000, nproc=None):
    from multiprocessing import Pool
    from tasks.threemodel import mean_sem
    sp = config.spatial_freq
    seeds = [seed + 137 * i for i in range(getattr(sp, 'n_seeds', 1))]
    args = [(m, config, sd) for m in sp.modes for sd in seeds]
    nproc = nproc or min(len(args), max(1, (os.cpu_count() or 4) - 2))
    print(f"  Spatial-frequency: {len(sp.modes)} modes x {len(seeds)} seeds, "
          f"alpha={getattr(sp,'alpha',config.elif_params.alpha)}, on {nproc} cores...", flush=True)
    raw = {m: [] for m in sp.modes}
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_point, args):
            raw[r['m']].append(r)
    out = {'modes': list(sp.modes), 'N_stim': sp.N_stim, 'chance': 100.0 / sp.N_stim,
           'sigma_eph': config.geometry.sigma_eph, 'alpha': getattr(sp, 'alpha', config.elif_params.alpha),
           'n_seeds': len(seeds),
           'nc': {k: {'mean': [], 'sem': []} for k in ('std', 'elif', 'rm')},
           'lda': {k: {'mean': [], 'sem': []} for k in ('std', 'elif', 'rm')},
           'spk': {k: [] for k in ('std', 'elif', 'rm')}, 'dI': [], 'info': _info(config)}
    for m in sp.modes:
        rs = raw[m]
        for dec in ('nc', 'lda'):
            for k in ('std', 'elif', 'rm'):
                mu, se = mean_sem([r[k][dec] for r in rs])
                out[dec][k]['mean'].append(mu); out[dec][k]['sem'].append(se)
        for k in ('std', 'elif', 'rm'):
            out['spk'][k].append(float(np.mean([r[k]['spk'] for r in rs])))
        out['dI'].append(float(np.mean([r['dI'] for r in rs])))
        print(f"  [m={m}] NC Std/eLIF/RM={out['nc']['std']['mean'][-1]:.0f}/"
              f"{out['nc']['elif']['mean'][-1]:.0f}/{out['nc']['rm']['mean'][-1]:.0f}", flush=True)
    return out


def _info(config):
    sp = config.spatial_freq; net = config.network; geo = config.geometry
    return {
        'title': 'spatial-frequency discrimination: a gain-invariance control',
        'task': (f"{sp.N_stim} stimuli = phase shifts of a spatial sinusoid at "
                 f"wavenumber m (cycles/unit). Decode phase from population spike "
                 f"counts. Weak signal (amp {sp.stim_amp}) in private noise "
                 f"(amp {sp.private_noise_amp})."),
        'sweep': f"m = {list(sp.modes)} cycles/unit. Std vs eLIF vs RM.",
        'predict': ("signal and noise occupy the SAME wavenumber m, so the "
                    "spatial filter H(m) scales both equally and the SNR (hence "
                    "decoding) is invariant to the field -> expect NO eLIF effect. "
                    "Confirms the alpha-sweep benefit is decorrelation, not gain."),
        'model': (f"N={net.N}, alpha={getattr(sp,'alpha',config.elif_params.alpha)} "
                  f"(fraction of critical), kernel={geo.eph_kernel}, sigma_eph={geo.sigma_eph}."),
        'stats': f"{sp.N_trials} trials x {getattr(sp,'n_seeds',1)} seeds, mean +/- SEM.",
    }
