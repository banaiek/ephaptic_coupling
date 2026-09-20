# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np
from copy import deepcopy
from config import Config
from geometry import NetworkGeometry
from analysis import decode_nearest_centroid_loo
from simulation import simulate_network
from tasks.threemodel import (ThreeModel, participation_ratio, synchrony_chi,
                              confusion_mutual_information)


def _build_tuning(NE, N_stim, width, seed=1000):
    rng = np.random.default_rng(seed)
    pref = rng.integers(0, N_stim, size=NE)
    tuning = np.zeros((NE, N_stim))
    for n in range(NE):
        for s in range(N_stim):
            d = min(abs(pref[n] - s), N_stim - abs(pref[n] - s))
            tuning[n, s] = 0.3 + 0.7 * np.exp(-d ** 2 / (2 * width ** 2))
    return tuning


def _setup(config):
    ap = config.alpha_sweep
    dt = config.sim.dt
    seed = config.seed
    N, NE = config.network.N, config.network.NE
    geom = NetworkGeometry.build(config)
    T_steps = int(round(ap.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    stim_t = (t >= ap.stim_onset) & (t < ap.stim_offset)
    count_mask = (t >= ap.count_start) & (t < ap.count_end)
    tuning = _build_tuning(NE, ap.N_stim, ap.tuning_width, seed=seed + 1)

    def build_input(s, trial, dI=0.0):
        rng = np.random.default_rng(seed + 1000 * s + trial)
        K = geom.G.shape[1]
        xi = rng.standard_normal((K, T_steps))
        kw = max(1, int(round(50.0 / dt))); kb = np.ones(kw) / kw
        for k in range(K):
            xi[k] = np.convolve(xi[k], kb, mode='same')
        sn = ap.shared_noise_amp * (geom.G @ xi)
        rng_p = np.random.default_rng(seed + 7 + 1000 * s + trial)
        I = (ap.baseline - dI) + sn + ap.private_noise_amp * rng_p.standard_normal((N, T_steps))
        I[:NE][:, stim_t] += ap.stim_amp_shared
        I[:NE][:, stim_t] += (ap.stim_amp_signal * tuning[:, s])[:, None]
        return I

    fp = config.field_mode
    src = np.asarray(fp.src_pos, float)
    gprof = np.exp(-np.linalg.norm(geom.pos - src[None, :], axis=1) / fp.src_lambda)
    kw = max(1, int(round(50.0 / dt))); kb = np.ones(kw) / kw

    def exo_field(s, trial, alpha):
        rng_c = np.random.default_rng(seed * 31 + 5101 + s * ap.N_trials + trial)
        eta = np.convolve(rng_c.standard_normal(T_steps), kb, mode='same')
        eta = eta / (eta.std() + 1e-9)
        return (alpha * fp.exo_field_amp) * gprof[:, None] * eta[None, :]

    return geom, count_mask, stim_t, build_input, exo_field


def _metrics(config, counts, raster, stim_t, N_stim, N_trials):
    nc_acc, conf = decode_nearest_centroid_loo(counts, N_stim, N_trials)
    _, lda = ThreeModel.decode(counts, N_stim, N_trials)
    spk = ThreeModel.window_total(counts, N_stim, N_trials)
    mi = confusion_mutual_information(conf)
    NE = config.network.NE
    chi = synchrony_chi(raster[:NE][:, stim_t], config.sim.dt) if raster is not None else np.nan
    return {'nc': float(nc_acc), 'lda': float(lda), 'spk': spk,
            'bits_per_spike': mi / max(spk, 1e-9), 'mi': mi,
            'dim': participation_ratio(counts), 'sync': chi, 'noise_corr': _ncorr(counts)}


def _point(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    alpha, config, pt_seed = args
    ap = config.alpha_sweep
    cfg = deepcopy(config); cfg.seed = pt_seed
    geom, count_mask, stim_t, build_input, exo_field = _setup(cfg)
    tm = ThreeModel(cfg, geom)
    c_s, r_s = tm.simulate_counts(build_input, ap.N_stim, ap.N_trials, count_mask,
                                  ap.T_trial, use_elif=False, store_spikes=True)
    sm = _metrics(cfg, c_s, r_s, stim_t, ap.N_stim, ap.N_trials)
    target_spk = ThreeModel.window_total(c_s, ap.N_stim, ap.N_trials)
    cfg_e = deepcopy(cfg); cfg_e.elif_params.alpha = float(alpha)
    tm_e = ThreeModel(cfg_e, geom)
    c_el, r_el = tm_e.simulate_counts(build_input, ap.N_stim, ap.N_trials, count_mask,
                                      ap.T_trial, use_elif=True, store_spikes=True)
    el = ThreeModel.window_total(c_el, ap.N_stim, ap.N_trials)
    dI = tm_e.rate_match(build_input, ap.N_stim, count_mask, ap.T_trial, target_spk, pt_seed,
                         passes=9) if el > target_spk * 1.01 else 0.0
    c_rm, r_rm = tm_e.simulate_counts(build_input, ap.N_stim, ap.N_trials, count_mask,
                                      ap.T_trial, use_elif=True, dI=dI, store_spikes=True)
    em = _metrics(cfg_e, c_el, r_el, stim_t, ap.N_stim, ap.N_trials)
    rm = _metrics(cfg_e, c_rm, r_rm, stim_t, ap.N_stim, ap.N_trials)
    rm['dI'] = dI; em['dI'] = 0.0; sm['dI'] = 0.0
    def build_input_exo(s, trial, dI=0.0):
        return build_input(s, trial, dI) + exo_field(s, trial, float(alpha))
    c_ex, r_ex = tm.simulate_counts(build_input_exo, ap.N_stim, ap.N_trials, count_mask,
                                    ap.T_trial, use_elif=False, store_spikes=True)
    ex = _metrics(cfg, c_ex, r_ex, stim_t, ap.N_stim, ap.N_trials)
    dI_ex = tm.rate_match(build_input_exo, ap.N_stim, count_mask, ap.T_trial, target_spk,
                          pt_seed, passes=9, use_elif=False, bidirectional=True)
    c_exr, r_exr = tm.simulate_counts(build_input_exo, ap.N_stim, ap.N_trials, count_mask,
                                      ap.T_trial, use_elif=False, dI=dI_ex, store_spikes=True)
    exr = _metrics(cfg, c_exr, r_exr, stim_t, ap.N_stim, ap.N_trials)
    ex['dI'] = 0.0; exr['dI'] = dI_ex
    return float(alpha), pt_seed, sm, em, rm, ex, exr


MKEYS = ('nc', 'lda', 'spk', 'bits_per_spike', 'dim', 'sync', 'noise_corr')


def run(config: Config, seed=6000, nproc=None):
    from multiprocessing import Pool
    from tasks.threemodel import aggregate_seeds
    ap = config.alpha_sweep
    args = [(a, config, seed + ai * 9973 + si * 137)
            for ai, a in enumerate(ap.alphas) for si in range(ap.n_seeds)]
    nproc = nproc or min(len(args), max(1, (os.cpu_count() or 4) - 2))
    print(f"  Sweeping {len(ap.alphas)} alphas x {ap.n_seeds} seeds (new network per point) "
          f"on {nproc} cores...", flush=True)
    MODELS = ('std', 'elif', 'rm', 'exo', 'exo_rm')
    DI_MODELS = ('elif', 'rm', 'exo', 'exo_rm')
    raw = {round(a, 6): {m: [] for m in MODELS} for a in ap.alphas}
    with Pool(nproc) as pool:
        for (a, sd, sm, em, rm, ex, exr) in pool.imap_unordered(_point, args):
            r = raw[round(a, 6)]
            r['std'].append(sm); r['elif'].append(em); r['rm'].append(rm)
            r['exo'].append(ex); r['exo_rm'].append(exr)

    out = {'alphas': list(ap.alphas), 'chance': 100.0 / ap.N_stim, 'n_seeds': ap.n_seeds,
           'std': {k: {'mean': [], 'sem': [], 'per_seed': []} for k in MKEYS},
           'info': _info(config)}
    for m in DI_MODELS:
        out[m] = {k: {'mean': [], 'sem': [], 'per_seed': []} for k in MKEYS + ('dI',)}
    for a in ap.alphas:
        ra_ = raw[round(a, 6)]
        aggs = {'std': aggregate_seeds(ra_['std'], MKEYS)}
        for m in DI_MODELS:
            aggs[m] = aggregate_seeds(ra_[m], MKEYS + ('dI',))
        for m in MODELS:
            for k in MKEYS:
                out[m][k]['mean'].append(aggs[m][k]['mean']); out[m][k]['sem'].append(aggs[m][k]['sem'])
                out[m][k]['per_seed'].append([float(d[k]) for d in ra_[m]])
        for m in DI_MODELS:
            out[m]['dI']['mean'].append(aggs[m]['dI']['mean']); out[m]['dI']['sem'].append(aggs[m]['dI']['sem'])
        sa, ea, xa = aggs['std'], aggs['elif'], aggs['exo']
        print(f"  [alpha={a:.2f}] NC std/eLIF/exo={sa['nc']['mean']:.0f}/{ea['nc']['mean']:.0f}/{xa['nc']['mean']:.0f} "
              f"corr std/eLIF/exo={sa['noise_corr']['mean']:.3f}/{ea['noise_corr']['mean']:.3f}/{xa['noise_corr']['mean']:.3f}", flush=True)
    return out


def _info(config):
    ap = config.alpha_sweep; net = config.network; geo = config.geometry
    return {
        'title': 'alpha bifurcation: ephaptic coupling as a control parameter',
        'task': (f"{ap.N_stim} broadly-tuned stimuli (Gaussian tuning width "
                 f"{ap.tuning_width}); decode population spike counts in "
                 f"[{ap.count_start:.0f},{ap.count_end:.0f}] ms. Shared + private "
                 f"noise (amp {ap.shared_noise_amp}/{ap.private_noise_amp})."),
        'sweep': (f"alpha = 0..{max(ap.alphas)} (fraction of critical coupling; "
                  f"alpha_c=1). Endogenous eLIF and its rate-matched RM, plus an "
                  f"open-loop exogenous field (exo) and its rate-matched control, "
                  f"vs Std=alpha 0."),
        'model': (f"N={net.N} ({net.NE}E/{net.NI}I), kernel={geo.eph_kernel}, "
                  f"sigma_eph={geo.sigma_eph}, field={geo.field_population}."),
        'stats': f"{ap.N_trials} trials x {getattr(ap,'n_seeds',1)} seeds, mean +/- SEM.",
        'readout': ('decoding (nearest-centroid + LDA), pairwise noise corr, '
                    'population synchrony chi, participation-ratio dimensionality, '
                    'bits/spike. RM isolates the field effect from firing rate.'),
    }


def _ncorr(counts, n_pairs=400, seed=7):
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
