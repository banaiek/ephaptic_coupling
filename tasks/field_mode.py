# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np
from copy import deepcopy
from config import Config
from geometry import NetworkGeometry
from simulation import simulate_network
from tasks.threemodel import participation_ratio, synchrony_chi


def _ncorr(counts, n_pairs=400, seed=7):
    N, T, S = counts.shape
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, N, (n_pairs, 2))
    cc = []
    for s in range(S):
        z = counts[:, :, s] - counts[:, :, s].mean(1, keepdims=True)
        sd = z.std(1) + 1e-9
        for a, b in pairs:
            if a == b:
                continue
            cc.append(np.mean(z[a] * z[b]) / (sd[a] * sd[b]))
    return float(np.nanmean(cc))


def _metrics(cfg, counts, raster, stim_t):
    NE = cfg.network.NE
    chi = synchrony_chi(raster[:NE][:, stim_t], cfg.sim.dt) if raster is not None else np.nan
    return {'noise_corr': _ncorr(counts), 'sync': float(chi), 'dim': participation_ratio(counts)}


def _build_setup(cfg, seed):
    cfg.seed = seed
    fp = cfg.field_mode
    dt = cfg.sim.dt
    N, NE = cfg.network.N, cfg.network.NE
    geom = NetworkGeometry.build(cfg)
    T_steps = int(round(fp.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    stim_t = (t >= fp.stim_onset) & (t < fp.stim_offset)
    count_mask = (t >= fp.count_start) & (t < fp.count_end)
    rng = np.random.default_rng(seed + 5)
    pref = rng.integers(0, fp.N_stim, size=NE)
    tuning = np.zeros((NE, fp.N_stim))
    for n in range(NE):
        for s in range(fp.N_stim):
            d = min(abs(pref[n] - s), fp.N_stim - abs(pref[n] - s))
            tuning[n, s] = 0.3 + 0.7 * np.exp(-d ** 2 / (2 * fp.tuning_width ** 2))
    kw = max(1, int(round(50.0 / dt))); kb = np.ones(kw) / kw

    def build_input(s, trial, dI=0.0):
        rng_n = np.random.default_rng(seed * 17 + s * fp.N_trials + trial)
        K = geom.G.shape[1]
        xi = rng_n.standard_normal((K, T_steps))
        for k in range(K):
            xi[k] = np.convolve(xi[k], kb, mode='same')
        sn = fp.shared_noise_amp * (geom.G @ xi)
        rng_p = np.random.default_rng(seed * 17 + 900000 + s * fp.N_trials + trial)
        I = (fp.baseline - dI) + sn + fp.private_noise_amp * rng_p.standard_normal((N, T_steps))
        I[:NE][:, stim_t] += fp.stim_amp_shared
        I[:NE][:, stim_t] += (fp.stim_amp_signal * tuning[:, s])[:, None]
        return I

    src = np.asarray(fp.src_pos, float)
    dist = np.linalg.norm(geom.pos - src[None, :], axis=1)
    gprof = np.exp(-dist / fp.src_lambda)

    def exo_field(s, trial, alpha):
        rng_c = np.random.default_rng(seed * 31 + 5101 + s * fp.N_trials + trial)
        eta = np.convolve(rng_c.standard_normal(T_steps), kb, mode='same')
        eta = eta / (eta.std() + 1e-9)
        amp = alpha * fp.exo_field_amp
        return amp * gprof[:, None] * eta[None, :]

    return geom, count_mask, stim_t, build_input, exo_field


def _v0(cfg, seed):
    n = cfg.network.N
    rng = np.random.default_rng(seed)
    return rng.uniform(cfg.neuron.V_reset, cfg.neuron.V_th, n)


def _counts(cfg, geom, build_input, exo_field, count_mask, use_elif, with_exo,
            alpha, dI, v0_seed, fp, want_raster=False):
    N = cfg.network.N
    counts = np.zeros((N, fp.N_trials, fp.N_stim)); raster = None
    for s in range(fp.N_stim):
        for tr in range(fp.N_trials):
            I = build_input(s, tr, dI)
            if with_exo:
                I = I + exo_field(s, tr, alpha)
            v0 = _v0(cfg, v0_seed + s * fp.N_trials + tr)
            r = simulate_network(cfg, geom, I, fp.T_trial, use_elif=use_elif, v_init=v0)
            counts[:, tr, s] = r.spikes[:, count_mask].sum(1)
            if want_raster and s == 0 and tr == 0:
                raster = r.spikes
    return counts, raster


def _match_dI(cfg, geom, build_input, exo_field, count_mask, use_elif, with_exo,
              alpha, target_E, v0_seed, fp):
    NE = cfg.network.NE; ns = min(fp.N_stim, 2); nt = min(fp.N_trials, 3)

    def probe(dI):
        tot = 0.0; c = 0
        for s in range(ns):
            for tr in range(nt):
                I = build_input(s, tr, dI)
                if with_exo:
                    I = I + exo_field(s, tr, alpha)
                v0 = _v0(cfg, v0_seed + s * fp.N_trials + tr)
                r = simulate_network(cfg, geom, I, fp.T_trial, use_elif=use_elif, v_init=v0)
                tot += r.spikes[:NE, count_mask].sum(); c += 1
        return tot / max(c, 1)

    if probe(0.0) <= target_E * 1.01:
        return 0.0
    lo, hi = 0.0, 5.0
    for _ in range(7):
        mid = (lo + hi) / 2
        if probe(mid) > target_E: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def _target_E(cfg, geom, build_input, count_mask, v0_seed, fp):
    NE = cfg.network.NE; ns = min(fp.N_stim, 2); nt = min(fp.N_trials, 3)
    tot = 0.0; c = 0
    for s in range(ns):
        for tr in range(nt):
            v0 = _v0(cfg, v0_seed + s * fp.N_trials + tr)
            r = simulate_network(cfg, geom, build_input(s, tr, 0.0), fp.T_trial, use_elif=False, v_init=v0)
            tot += r.spikes[:NE, count_mask].sum(); c += 1
    return tot / max(c, 1)


CONDS = ('std', 'endo', 'endo_rm', 'exo', 'exo_rm')


def _point(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    alpha, config, pt_seed = args
    cfg = deepcopy(config)
    geom, count_mask, stim_t, build_input, exo_field = _build_setup(cfg, pt_seed)
    cfg_e = deepcopy(cfg); cfg_e.elif_params.alpha = alpha
    vS = pt_seed * 13 + 1
    out = {}
    c, r = _counts(cfg, geom, build_input, exo_field, count_mask, False, False, alpha, 0.0, vS, cfg.field_mode, True)
    out['std'] = _metrics(cfg, c, r, stim_t)
    tgt = _target_E(cfg, geom, build_input, count_mask, vS, cfg.field_mode)
    c, r = _counts(cfg_e, geom, build_input, exo_field, count_mask, True, False, alpha, 0.0, vS, cfg.field_mode, True)
    out['endo'] = _metrics(cfg, c, r, stim_t)
    dI = _match_dI(cfg_e, geom, build_input, exo_field, count_mask, True, False, alpha, tgt, vS, cfg.field_mode)
    c, r = _counts(cfg_e, geom, build_input, exo_field, count_mask, True, False, alpha, dI, vS, cfg.field_mode, True)
    out['endo_rm'] = _metrics(cfg, c, r, stim_t)
    c, r = _counts(cfg, geom, build_input, exo_field, count_mask, False, True, alpha, 0.0, vS, cfg.field_mode, True)
    out['exo'] = _metrics(cfg, c, r, stim_t)
    dI = _match_dI(cfg, geom, build_input, exo_field, count_mask, False, True, alpha, tgt, vS, cfg.field_mode)
    c, r = _counts(cfg, geom, build_input, exo_field, count_mask, False, True, alpha, dI, vS, cfg.field_mode, True)
    out['exo_rm'] = _metrics(cfg, c, r, stim_t)
    return float(alpha), pt_seed, out


def run(config: Config, seed=8000, nproc=None):
    from multiprocessing import Pool
    from tasks.threemodel import aggregate_seeds
    fp = config.field_mode
    MK = ('noise_corr', 'sync', 'dim')
    args = [(a, config, seed + ai * 9973 + si * 137)
            for ai, a in enumerate(fp.alphas) for si in range(fp.n_seeds)]
    nproc = nproc or min(len(args), max(1, (os.cpu_count() or 4) - 2))
    print(f"  Endo vs Exo: {len(fp.alphas)} alphas x {fp.n_seeds} seeds (new network each) "
          f"on {nproc} cores...", flush=True)
    raw = {}
    with Pool(nproc) as pool:
        for (a, sd, res) in pool.imap_unordered(_point, args):
            raw.setdefault(round(a, 6), {c: [] for c in CONDS})
            for c in CONDS:
                raw[round(a, 6)][c].append(res[c])

    out = {'alphas': list(fp.alphas), 'n_seeds': fp.n_seeds, 'info': _info(config)}
    for c in CONDS:
        out[c] = {k: {'mean': [], 'sem': [], 'per_seed': []} for k in MK}
    for a in fp.alphas:
        for c in CONDS:
            agg = aggregate_seeds(raw[round(a, 6)][c], MK)
            for k in MK:
                out[c][k]['mean'].append(agg[k]['mean']); out[c][k]['sem'].append(agg[k]['sem'])
                out[c][k]['per_seed'].append([m[k] for m in raw[round(a, 6)][c]])
        g = {c: aggregate_seeds(raw[round(a, 6)][c], MK) for c in CONDS}
        print(f"  [alpha={a:.2f}] corr std/endo/exo="
              f"{g['std']['noise_corr']['mean']:.3f}/{g['endo']['noise_corr']['mean']:.3f}/"
              f"{g['exo']['noise_corr']['mean']:.3f}  dim "
              f"{g['std']['dim']['mean']:.0f}/{g['endo']['dim']['mean']:.0f}/{g['exo']['dim']['mean']:.0f}", flush=True)
    return out


def _info(config):
    fp = config.field_mode; net = config.network; geo = config.geometry
    return {
        'title': 'endogenous vs exogenous field: synchronize or decorrelate?',
        'task': (f"{fp.N_stim} tuned stimuli x {fp.N_trials} trials; pairwise noise "
                 f"correlation, synchrony, dimensionality."),
        'sweep': f"field amplitude alpha = {list(fp.alphas)} (0 = no field; all conditions coincide).",
        'predict': ('exogenous injected field (open-loop, source-attenuated) ENTRAINS '
                    '-> corr up, dim down; endogenous (closed-loop, reads V_j) DECORRELATES.'),
        'model': (f"N={net.N}, kernel {geo.eph_kernel}; NEW network per seed + random "
                  f"initial voltages. exo: field from a source, decay length {fp.src_lambda}."),
        'stats': f"{fp.n_seeds} networks/alpha, mean +/- SEM; paired Wilcoxon (endo vs exo).",
    }
