# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import pickle, time
import numpy as np
from config import Config
from geometry import NetworkGeometry
from simulation import simulate_network
from tasks import Task

SIGMA_EPH = [0.1, 0.2, 0.3, 0.4, 0.5]
SIGMA_INP = [0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS = 50
T_MS = 300.0
BASE_SEED = 20240

cfg0 = Config.from_defaults(); cfg0.geometry.eph_sparsify = 0.0
N, NE = cfg0.network.N, cfg0.network.NE
dt = cfg0.sim.dt
pd = cfg0.decoding
T_steps = int(round(T_MS / dt)) + 1
t = np.arange(T_steps) * dt
stim_mask = (t >= 100) & (t < 300)
iu, ju = np.triu_indices(NE, k=1)

ne, ni = len(SIGMA_EPH), len(SIGMA_INP)


def build_geom(s_eph, s_inp):
    c = Config.from_defaults()
    c.geometry.eph_sparsify = 0.0
    c.geometry.sigma_eph = s_eph
    c.geometry.sigma_input = s_inp
    return c, NetworkGeometry.build(c)


GEO = [[build_geom(se, si) for si in SIGMA_INP] for se in SIGMA_EPH]


def build_I(noise_seed, j):
    cfg_ref, geo_ref = GEO[0][j]
    task = Task(cfg_ref, geo_ref)
    rn = np.random.default_rng(noise_seed)
    sn = task._build_shared_noise(T_steps, pd.shared_noise_amp, rn, 50.0)
    rp = np.random.default_rng(noise_seed + 100000)
    I = np.empty((N, T_steps))
    for ti in range(T_steps):
        I[:, ti] = pd.I_baseline + sn[:, ti] + pd.private_noise_amp * rp.standard_normal(N)
        if 100 <= t[ti] < 300:
            I[:NE, ti] += pd.stim_amp_shared + pd.stim_amp_signal
    return I


from scipy.ndimage import binary_dilation
_WMASK = int(round(2.0 / dt))


def mean_corr(res):
    V = res.V_trace[:NE, stim_mask].astype(float).copy()
    sp = res.spikes[:NE, stim_mask].astype(bool)
    V[binary_dilation(sp, structure=np.ones((1, 2 * _WMASK + 1)))] = np.nan
    Vd = V[:, ::10]
    Vc = Vd - np.nanmean(Vd, axis=1, keepdims=True)
    valid = (~np.isnan(Vd)).astype(float)
    Vc = np.nan_to_num(Vc)
    num = Vc @ Vc.T
    den = np.sqrt((Vc ** 2) @ valid.T) * np.sqrt(valid @ (Vc ** 2).T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        C = num / den
    return float(np.nanmean(C[iu, ju]))


def spikes(cfg, geo, I, off, use_elif):
    return float(simulate_network(cfg, geo, I - off, T_MS, use_elif=use_elif).spikes[:NE, stim_mask].sum())


def main():
    t0 = time.time()
    print("Rate-matching RM dI per grid point...", flush=True)
    dI = np.zeros((ne, ni))
    I0 = [build_I(BASE_SEED, j) for j in range(ni)]
    for j in range(ni):
        cfg_ref, geo_ref = GEO[0][j]
        tgt = spikes(cfg_ref, geo_ref, I0[j], 0.0, False)
        for i in range(ne):
            cfg_e, geo_e = GEO[i][j]
            if spikes(cfg_e, geo_e, I0[j], 0.0, True) > tgt * 1.01:
                lo, hi = 0.0, 4.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    if spikes(cfg_e, geo_e, I0[j], mid, True) > tgt:
                        lo = mid
                    else:
                        hi = mid
                dI[i, j] = (lo + hi) / 2
        print(f"  sigma_input={SIGMA_INP[j]}: dI(sigma_eph)={np.round(dI[:, j], 2)}", flush=True)

    acc_el = np.zeros((ne, ni)); acc_rm = np.zeros((ne, ni))
    acc_std = np.zeros((ne, ni)); acc_elif_abs = np.zeros((ne, ni)); acc_rm_abs = np.zeros((ne, ni))
    for s in range(N_SEEDS):
        for j in range(ni):
            I = build_I(BASE_SEED + 1 + s * 131 + j, j)
            cfg_ref, geo_ref = GEO[0][j]
            c_std = mean_corr(simulate_network(cfg_ref, geo_ref, I, T_MS, use_elif=False, record_voltage=True))
            for i in range(ne):
                cfg_e, geo_e = GEO[i][j]
                c_el = mean_corr(simulate_network(cfg_e, geo_e, I, T_MS, use_elif=True, record_voltage=True))
                c_rm = mean_corr(simulate_network(cfg_e, geo_e, I - dI[i, j], T_MS, use_elif=True, record_voltage=True))
                acc_el[i, j] += c_el - c_std
                acc_rm[i, j] += c_rm - c_std
                acc_std[i, j] += c_std; acc_elif_abs[i, j] += c_el; acc_rm_abs[i, j] += c_rm
        print(f"  seed {s + 1}/{N_SEEDS} done ({time.time() - t0:.0f}s)", flush=True)

    out = {
        'sigma_eph_vals': np.array(SIGMA_EPH), 'sigma_input_vals': np.array(SIGMA_INP),
        'diff_elif': acc_el / N_SEEDS, 'diff_rm': acc_rm / N_SEEDS,
        'corr_std': acc_std / N_SEEDS, 'corr_elif': acc_elif_abs / N_SEEDS,
        'corr_rm': acc_rm_abs / N_SEEDS, 'dI': dI, 'n_seeds': N_SEEDS, 'T_ms': T_MS,
    }
    os.makedirs('source_data', exist_ok=True)
    with open('source_data/sigma2d.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved source_data/sigma2d.pkl ({(time.time() - t0) / 60:.1f} min)")
    print("diff_elif range:", np.round(out['diff_elif'].min(), 3), np.round(out['diff_elif'].max(), 3))
    print("diff_rm range:  ", np.round(out['diff_rm'].min(), 3), np.round(out['diff_rm'].max(), 3))


if __name__ == '__main__':
    main()
