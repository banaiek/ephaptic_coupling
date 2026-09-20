# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from copy import deepcopy

from config import Config
from geometry import NetworkGeometry
from analysis import decode_nearest_centroid_loo
from network.integrator import simulate as simulate_rev
from tasks.threemodel import participation_ratio, synchrony_chi

DT = 0.1
T_TRIAL = 400.0
STIM_ON, STIM_OFF = 100.0, 300.0
COUNT_ON, COUNT_OFF = 120.0, 300.0
N_STIM = 10
ALPHA = 0.2


def build_tuning(NE, n_stim, width, seed):
    rng = np.random.default_rng(seed)
    pref = rng.integers(0, n_stim, size=NE)
    tun = np.zeros((NE, n_stim))
    for n in range(NE):
        for s in range(n_stim):
            d = min(abs(pref[n] - s), n_stim - abs(pref[n] - s))
            tun[n, s] = 0.3 + 0.7 * np.exp(-d ** 2 / (2 * width ** 2))
    return tun, pref


class DecodingTask:

    def __init__(self, seed=4242, n_stim=N_STIM, n_trials=20, config=None):
        self.seed = seed
        self.n_stim = n_stim
        self.n_trials = n_trials
        cfg = config if config is not None else Config.from_defaults()
        cfg.seed = seed
        self.cfg = cfg
        self.geom = NetworkGeometry.build(cfg)
        self.T_steps = int(round(T_TRIAL / DT)) + 1
        t = np.arange(self.T_steps) * DT
        self.t = t
        self.stim_t = (t >= STIM_ON) & (t < STIM_OFF)
        self.count_mask = (t >= COUNT_ON) & (t < COUNT_OFF)
        self.tuning, self.pref = build_tuning(cfg.network.NE, n_stim,
                                              cfg.alpha_sweep.tuning_width, seed + 1)
        kw = max(1, int(round(50.0 / DT)))
        self._boxcar = np.ones(kw) / kw

    def input(self, s, trial, dI=0.0):
        ap = self.cfg.alpha_sweep
        N, NE = self.cfg.network.N, self.cfg.network.NE
        rng = np.random.default_rng(self.seed + 1000 * s + trial)
        K = self.geom.G.shape[1]
        xi = rng.standard_normal((K, self.T_steps))
        for k in range(K):
            xi[k] = np.convolve(xi[k], self._boxcar, mode='same')
        shared = ap.shared_noise_amp * (self.geom.G @ xi)
        rp = np.random.default_rng(self.seed + 7 + 1000 * s + trial)
        I = (ap.baseline - dI) + shared + ap.private_noise_amp * rp.standard_normal((N, self.T_steps))
        I[:NE][:, self.stim_t] += ap.stim_amp_shared
        I[:NE][:, self.stim_t] += (ap.stim_amp_signal * self.tuning[:, s])[:, None]
        return I

    def run(self, config=None, geom=None, use_elif=False, dI=0.0, extra=None,
            simulate=None, store_raster=False, **sim_kwargs):
        cfg = config if config is not None else self.cfg
        gm = geom if geom is not None else self.geom
        sim = simulate if simulate is not None else simulate_rev
        counts = np.zeros((cfg.network.N, self.n_trials, self.n_stim))
        raster = None
        for s in range(self.n_stim):
            for tr in range(self.n_trials):
                I = self.input(s, tr, dI)
                if extra is not None:
                    I = I + extra(s, tr)
                r = sim(cfg, gm, I, T_TRIAL, use_elif=use_elif, **sim_kwargs)
                counts[:, tr, s] = r.spikes[:, self.count_mask].sum(1)
                if store_raster and s == 0 and tr == 0:
                    raster = r.spikes.astype(np.uint8)
        return (counts, raster) if store_raster else counts

    def match_rate(self, target_spk, config, geom=None, use_elif=True, extra=None,
                   simulate=None, lo=-6.0, hi=6.0, passes=9, n_probe=4,
                   n_stim_probe=3, **sim_kwargs):
        sim = simulate if simulate is not None else simulate_rev

        n_s = min(n_stim_probe, self.n_stim)

        def probe(dI):
            tot = 0.0
            for s in range(n_s):
                for tr in range(n_probe):
                    I = self.input(s, tr, dI)
                    if extra is not None:
                        I = I + extra(s, tr)
                    r = sim(config, geom if geom is not None else self.geom, I, T_TRIAL,
                            use_elif=use_elif, **sim_kwargs)
                    tot += r.spikes[:, self.count_mask].sum()
            return tot / (n_s * n_probe)
        for _ in range(passes):
            mid = (lo + hi) / 2
            if probe(mid) > target_spk:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def pair_noise_correlation(counts, n_pairs=4000, n_exc=400, seed=0):
    resid = counts - counts.mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_exc, n_pairs)
    j = rng.integers(0, n_exc, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    vals = []
    for s in range(counts.shape[2]):
        a, b = resid[i, :, s], resid[j, :, s]
        sa, sb = a.std(1) + 1e-12, b.std(1) + 1e-12
        vals.append((a * b).mean(1) / (sa * sb))
    return float(np.nanmean(vals))


def voltage_correlation(V, n_exc, mask, n_pairs=3000, seed=0):
    X = V[:n_exc][:, mask]
    X = X - X.mean(1, keepdims=True)
    sd = X.std(1) + 1e-12
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_exc, n_pairs)
    j = rng.integers(0, n_exc, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return float(np.mean((X[i] * X[j]).mean(1) / (sd[i] * sd[j])))


def shared_private_variance(V, n_exc, mask):
    X = V[:n_exc][:, mask]
    m = X.mean(0)
    return float(m.var()), float((X - m[None, :]).var(1).mean())


def code_metrics(counts, n_stim, n_trials, tag=''):
    acc, conf = decode_nearest_centroid_loo(counts, n_stim, n_trials)
    return {'tag': tag,
            'accuracy': float(acc),
            'noise_corr': pair_noise_correlation(counts),
            'dimensionality': participation_ratio(counts),
            'spikes': float(counts.sum() / (n_stim * n_trials)),
            'confusion': conf}


def clone(cfg, **overrides):
    out = deepcopy(cfg)
    for key, val in overrides.items():
        obj = out
        parts = key.split('.')
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    return out
