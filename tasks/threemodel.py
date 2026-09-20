# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from simulation import simulate_network
from analysis import decode_nearest_centroid_loo, decode_sklearn


def participation_ratio(counts):
    N, T, S = counts.shape
    X = counts.reshape(N, T * S).T.astype(float)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return 1.0
    C = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return 1.0
    return float((ev.sum() ** 2) / np.sum(ev ** 2))


def synchrony_chi(spikes_pop, dt, bin_ms=5.0):
    n, T = spikes_pop.shape
    b = max(1, int(round(bin_ms / dt)))
    nb = T // b
    if nb < 4:
        return np.nan
    r = spikes_pop[:, :nb * b].reshape(n, nb, b).sum(2).astype(float)
    pop = r.mean(0)
    var_pop = pop.var()
    var_cells = r.var(1).mean()
    return float(var_pop / (var_cells + 1e-12))


def mean_sem(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), 0.0
    return float(v.mean()), float(v.std(ddof=0) / max(1.0, np.sqrt(v.size)))


def aggregate_seeds(dicts, keys):
    out = {}
    for k in keys:
        m, s = mean_sem([d[k] for d in dicts])
        out[k] = {'mean': m, 'sem': s}
    return out


def confusion_mutual_information(confusion):
    C = np.asarray(confusion, float)
    tot = C.sum()
    if tot <= 0:
        return 0.0
    P = C / tot
    pt = P.sum(1, keepdims=True); pd = P.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = P * (np.log2(P) - np.log2(pt) - np.log2(pd))
    return float(np.nansum(terms))


class ThreeModel:
    def __init__(self, config, geometry):
        self.config = config
        self.geom = geometry
        self.N = config.network.N
        self.NE = config.network.NE
        self.dt = config.sim.dt

    def simulate_counts(self, build_input, N_stim, N_trials, count_mask, T_trial,
                        use_elif, dI=0.0, base_seed=0, store_spikes=False):
        counts = np.zeros((self.N, N_trials, N_stim))
        raster = None
        for s in range(N_stim):
            for tr in range(N_trials):
                I = build_input(s, tr, dI)
                r = simulate_network(self.config, self.geom, I, T_trial, use_elif=use_elif)
                counts[:, tr, s] = r.spikes[:, count_mask].sum(axis=1)
                if store_spikes and s == 0 and tr == 0:
                    raster = r.spikes.astype(np.uint8)
        return counts, raster

    def rate_match(self, build_input, N_stim, count_mask, T_trial, target_total,
                   base_seed=0, n_probe_stim=3, n_probe_trials=5, passes=8, use_elif=True,
                   bidirectional=False):
        def probe(dI):
            tot = 0.0; cnt = 0
            for s in range(min(n_probe_stim, N_stim)):
                for tr in range(n_probe_trials):
                    I = build_input(s, tr, dI)
                    r = simulate_network(self.config, self.geom, I, T_trial, use_elif=use_elif)
                    tot += r.spikes[:, count_mask].sum(); cnt += 1
            return tot / max(1, cnt)
        lo, hi = (-8.0 if bidirectional else 0.0), 8.0
        for _ in range(passes):
            mid = (lo + hi) / 2
            if probe(mid) > target_total:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def decode(counts, N_stim, N_trials):
        nc, _ = decode_nearest_centroid_loo(counts, N_stim, N_trials)
        N = counts.shape[0]
        X = np.zeros((N_trials * N_stim, N)); y = np.zeros(N_trials * N_stim, int)
        for s in range(N_stim):
            idx = slice(s * N_trials, (s + 1) * N_trials)
            X[idx] = counts[:, :, s].T; y[idx] = s
        try:
            lda = decode_sklearn(X, y, method='lda')['accuracy']
        except Exception:
            lda = 100.0 / N_stim
        return float(nc), float(lda)

    @staticmethod
    def window_total(counts, N_stim, N_trials):
        return float(counts.sum() / (N_stim * N_trials))

    def run_three(self, build_input, N_stim, N_trials, count_mask, T_trial,
                  base_seed=0, store_spikes=False):
        c_std, r_std = self.simulate_counts(build_input, N_stim, N_trials, count_mask,
                                            T_trial, False, 0.0, base_seed, store_spikes)
        c_el, r_el = self.simulate_counts(build_input, N_stim, N_trials, count_mask,
                                          T_trial, True, 0.0, base_seed, store_spikes)
        tgt = self.window_total(c_std, N_stim, N_trials)
        el = self.window_total(c_el, N_stim, N_trials)
        dI = self.rate_match(build_input, N_stim, count_mask, T_trial, tgt,
                             base_seed) if el > tgt * 1.01 else 0.0
        c_rm, r_rm = self.simulate_counts(build_input, N_stim, N_trials, count_mask,
                                          T_trial, True, dI, base_seed, store_spikes)
        return {'std': c_std, 'elif': c_el, 'rm': c_rm, 'dI': dI,
                'raster': {'std': r_std, 'elif': r_el, 'rm': r_rm}}
