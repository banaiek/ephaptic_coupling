# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
import numpy as np

from config import Config
from geometry import NetworkGeometry
from network import common as C
from network.integrator import simulate, homogeneous


def dipole_uniformity(patch_um=300.0, distances_um=(300.0, 1000.0, 3000.0),
                      n=41, dipole_um=1000.0):
    x = np.linspace(-patch_um / 2, patch_um / 2, n)
    out, cv = {}, {}
    for R in distances_um:
        r = np.abs(R + x)
        v = 1.0 / r - 1.0 / np.sqrt(r ** 2 + dipole_um ** 2)
        out[R] = (v / v[n // 2]).tolist()
        cv[R] = float(v.std() / v.mean())
    return {'x_um': x.tolist(), 'patch_um': patch_um, 'profile': out, 'cv': cv}


def beta_scale(separation_um=3000.0, dipole_um=1000.0, sigma_e=0.3):
    cfg = Config.from_defaults()
    geom = NetworkGeometry.build(cfg)
    pos_um = geom.pos * cfg.geometry.extent_um if hasattr(cfg.geometry, 'extent_um') \
        else geom.pos * 1000.0

    def g(d):
        d = np.maximum(d, 1.0)
        return (1.0 / (4 * np.pi * sigma_e)) * (1.0 / d - 1.0 / np.sqrt(d ** 2 + dipole_um ** 2))

    d = np.sqrt(((pos_um[:, None, :] - pos_um[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    local = float(np.median(g(d).sum(1)))
    far = float(len(pos_um) * g(np.array([separation_um]))[0])
    return {'separation_um': separation_um, 'local_per_mV': local,
            'far_per_mV': far, 'ratio': far / local}


def local_field_dispersion(seed=4242, alpha=C.ALPHA, n_trials=4):
    task = C.DecodingTask(seed=seed, n_trials=1)
    cfg = C.clone(task.cfg, **{'elif_params.alpha': alpha})
    vals = []
    for tr in range(n_trials):
        r = simulate(cfg, task.geom, task.input(0, tr), C.T_TRIAL, use_elif=True,
                     record_phi=True)
        phi = r.phi_trace[:cfg.network.NE][:, task.stim_t]
        vals.append(float(np.mean(phi.std(0) / (np.abs(phi.mean(0)) + 1e-12))))
    return float(np.mean(vals))


def _column(seed, alpha, adapt_cfg=None):
    cfg = Config.from_defaults()
    cfg.seed = seed
    cfg.elif_params.alpha = alpha
    geom = NetworkGeometry.build(cfg)
    return cfg, geom


def simulate_pair(cfgA, geomA, cfgB, geomB, I_A, I_B, T_ms, beta, cos_theta,
                  use_local_field=True, adapt=None, u_ref=0.0):
    import scipy.sparse as sp
    dt = cfgA.sim.dt
    N, NE = cfgA.network.N, cfgA.network.NE
    p = homogeneous(cfgA)
    syn = cfgA.synapse
    decay_e, decay_i = np.exp(-dt / syn.tau_e), np.exp(-dt / syn.tau_i)
    T_steps = int(round(T_ms / dt)) + 1

    def op(geom):
        M = geom.W_eph_all
        return M if M is not None else sp.block_diag((geom.W_eph_E, geom.W_eph_I),
                                                     format='csr')

    M = [op(geomA), op(geomB)]
    cfgs = [cfgA, cfgB]
    geoms = [geomA, geomB]
    I_in = [I_A, I_B]

    V = [p.V_rest.copy(), p.V_rest.copy()]
    hE = [np.zeros(N), np.zeros(N)]
    hI = [np.zeros(N), np.zeros(N)]
    ref = [np.zeros(N), np.zeros(N)]
    spikes = [np.zeros((N, T_steps)), np.zeros((N, T_steps))]
    if adapt is not None:
        tau_w, b_w = adapt
        w = [np.zeros(N), np.zeros(N)]
        decay_w = np.exp(-dt / tau_w)

    for ti in range(1, T_steps):
        u = [V[0] - p.V_rest, V[1] - p.V_rest]
        cross = [beta * cos_theta * (u[1].mean() - u_ref),
                 beta * cos_theta * (u[0].mean() - u_ref)]
        for c in (0, 1):
            hE[c] = hE[c] * decay_e + geoms[c].W_E @ spikes[c][:NE, ti - 1]
            hI[c] = hI[c] * decay_i + geoms[c].W_I @ spikes[c][NE:, ti - 1]
            I_tot = I_in[c][:, ti] + hE[c] + hI[c] + cross[c]
            if use_local_field:
                I_tot = I_tot + cfgs[c].elif_params.alpha * (M[c] @ u[c])
            if adapt is not None:
                w[c] *= decay_w
                I_tot = I_tot - w[c]
            act = ref[c] <= 0
            dV = (-(V[c] - p.V_rest) + I_tot) / p.tau_m
            V[c][act] += dt * dV[act]
            fired = V[c] >= p.V_th
            spikes[c][fired, ti] = 1.0
            V[c][fired] = p.V_reset[fired]
            ref[c][fired] = p.tau_ref[fired]
            if adapt is not None:
                w[c][fired] += b_w
            ref[c] = np.maximum(0.0, ref[c] - dt)
    return spikes


def _pop_rate(spk, dt, bin_ms=10.0):
    b = max(1, int(round(bin_ms / dt)))
    nb = spk.shape[1] // b
    return spk[:, :nb * b].reshape(spk.shape[0], nb, b).sum(2).mean(0)


def cross_column_pairs(cA, cB, n_pairs=4000, n_exc=400, seed=0):
    rA = cA - cA.mean(1, keepdims=True)
    rB = cB - cB.mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_exc, n_pairs)
    j = rng.integers(0, n_exc, n_pairs)
    vals = []
    for s in range(cA.shape[2]):
        a, b = rA[i, :, s], rB[j, :, s]
        sa, sb = a.std(1) + 1e-12, b.std(1) + 1e-12
        vals.append((a * b).mean(1) / (sa * sb))
    return float(np.nanmean(vals))


def run_seed(seed, alpha=C.ALPHA, n_trials=15,
             betas=(0.0, 0.05, 0.1, 0.2, 0.4), beta_ref=0.2):
    taskA = C.DecodingTask(seed=seed, n_trials=n_trials)
    taskB = C.DecodingTask(seed=seed + 500, n_trials=n_trials)
    cfgA, geomA = taskA.cfg, taskA.geom
    cfgB, geomB = taskB.cfg, taskB.geom
    ceA = C.clone(cfgA, **{'elif_params.alpha': alpha})
    ceB = C.clone(cfgB, **{'elif_params.alpha': alpha})
    out = {'seed': seed, 'alpha': alpha, 'betas': list(betas), 'points': [], 'orient': []}
    dt = cfgA.sim.dt
    mask = taskA.count_mask

    def mean_deflection(use_local):
        r = simulate(ceA if use_local else cfgA, geomA, taskA.input(0, 0),
                     C.T_TRIAL, use_elif=use_local, record_voltage=True)
        return float((r.V_trace[:, taskA.stim_t] - cfgA.neuron.V_rest).mean())

    def score(beta, cos_theta, use_local, dI=0.0, u_ref=0.0):
        cA = np.zeros((cfgA.network.N, n_trials, taskA.n_stim))
        cB = np.zeros_like(cA)
        rateA, rateB = [], []
        for s in range(taskA.n_stim):
            for tr in range(n_trials):
                sA, sB = simulate_pair(ceA if use_local else cfgA, geomA,
                                       ceB if use_local else cfgB, geomB,
                                       taskA.input(s, tr, dI), taskB.input(s, tr, dI),
                                       C.T_TRIAL, beta, cos_theta,
                                       use_local_field=use_local, u_ref=u_ref)
                cA[:, tr, s] = sA[:, mask].sum(1)
                cB[:, tr, s] = sB[:, mask].sum(1)
                if s == 0:
                    rateA.append(_pop_rate(sA[:cfgA.network.NE][:, taskA.stim_t], dt))
                    rateB.append(_pop_rate(sB[:cfgB.network.NE][:, taskB.stim_t], dt))
        rA = np.concatenate(rateA)
        rB = np.concatenate(rateB)
        return {'beta': beta, 'cos_theta': cos_theta, 'local_field': use_local, 'dI': dI,
                'within': C.code_metrics(cA, taskA.n_stim, n_trials, 'column A'),
                'within_B': C.code_metrics(cB, taskB.n_stim, n_trials, 'column B'),
                'between_pairs': cross_column_pairs(cA, cB),
                'between_rate': float(np.corrcoef(rA, rB)[0, 1])}

    def match_pair_rate(target, beta, cos_theta, use_local, u_ref,
                        lo=-4.0, hi=4.0, passes=7, n_probe=3):
        for _ in range(passes):
            mid = 0.5 * (lo + hi)
            tot = 0.0
            for st in range(min(3, taskA.n_stim)):
                for tr in range(n_probe):
                    sA, sB = simulate_pair(ceA if use_local else cfgA, geomA,
                                           ceB if use_local else cfgB, geomB,
                                           taskA.input(st, tr, mid),
                                           taskB.input(st, tr, mid),
                                           C.T_TRIAL, beta, cos_theta,
                                           use_local_field=use_local, u_ref=u_ref)
                    tot += sA[:, mask].sum() + sB[:, mask].sum()
            if tot / (min(3, taskA.n_stim) * n_probe * 2) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    for use_local in (True, False):
        u_ref = mean_deflection(use_local)
        base = score(0.0, 1.0, use_local, u_ref=u_ref)
        target = base['within']['spikes']
        rows = [base]
        for beta in betas[1:]:
            dI = match_pair_rate(target, beta, 1.0, use_local, u_ref)
            rows.append(score(beta, 1.0, use_local, dI=dI, u_ref=u_ref))
        out['points' if use_local else 'no_local'] = rows
        if use_local:
            out['u_ref'] = u_ref
            for cos_theta in (1.0, 0.0, -1.0):
                dI = match_pair_rate(target, beta_ref, cos_theta, True, u_ref)
                out['orient'].append(score(beta_ref, cos_theta, True, dI=dI,
                                           u_ref=u_ref))
    return out


def _slow_job(args):
    from scipy.signal import csd, welch
    (rep, beta, cos_theta, seed, alpha, T_ms, f_drive, drive_amp,
     baseline, noise) = args
    cfgA = Config.from_defaults(); cfgA.seed = seed; cfgA.elif_params.alpha = alpha
    cfgB = Config.from_defaults(); cfgB.seed = seed + 500; cfgB.elif_params.alpha = alpha
    geomA, geomB = NetworkGeometry.build(cfgA), NetworkGeometry.build(cfgB)
    dt = cfgA.sim.dt
    T_steps = int(round(T_ms / dt)) + 1
    N = cfgA.network.N
    t = np.arange(T_steps) * dt
    drive = drive_amp * np.sin(2 * np.pi * f_drive * t / 1000.0)
    rngA = np.random.default_rng(1000 + rep)
    rngB = np.random.default_rng(2000 + rep)
    I_A = baseline + drive[None, :] + noise * rngA.standard_normal((N, T_steps))
    I_B = baseline + noise * rngB.standard_normal((N, T_steps))
    n_probe = int(round(4000.0 / dt)) + 1
    probe = simulate(cfgA, geomA, I_A[:, :n_probe], 4000.0, use_elif=True,
                     record_voltage=True)
    u_ref = float((probe.V_trace - cfgA.neuron.V_rest).mean())
    sA, sB = simulate_pair(cfgA, geomA, cfgB, geomB, I_A, I_B, T_ms, beta,
                           cos_theta, u_ref=u_ref)
    rA = _pop_rate(sA[:cfgA.network.NE], dt, bin_ms=20.0)
    rB = _pop_rate(sB[:cfgB.network.NE], dt, bin_ms=20.0)
    fs = 50.0
    f, PB = welch(rB - rB.mean(), fs=fs, nperseg=512, noverlap=256)
    k = int(np.argmin(np.abs(f - f_drive)))
    band = (f > 0.2) & (f < 1.5) & (np.abs(f - f_drive) > 0.05)
    _, Pxy = csd(rA - rA.mean(), rB - rB.mean(), fs=fs, nperseg=512, noverlap=256)
    return {'rep': rep, 'beta': beta, 'cos_theta': cos_theta, 'f_drive': f_drive,
            'u_ref': u_ref,
            'rate_A': float(rA.mean()), 'rate_B': float(rB.mean()),
            'power_B_at_drive': float(PB[k] / PB[band].mean()),
            'phase': float(np.angle(Pxy[k])),
            'rate_corr': float(np.corrcoef(rA, rB)[0, 1]),
            'f': f.tolist(), 'P_B': PB.tolist()}


def run_slow(seed=4242, alpha=C.ALPHA, T_ms=40000.0,
             betas=(0.0, 0.1, 0.2, 0.3), cos_thetas=(1.0, 0.0, -1.0),
             f_drive=0.5, drive_amp=2.5, baseline=10.0, noise=2.0, n_rep=5,
             nproc=None):
    from multiprocessing import Pool
    jobs = [(rep, b, c, seed, alpha, T_ms, f_drive, drive_amp, baseline, noise)
            for rep in range(n_rep) for b in betas for c in cos_thetas]
    nproc = nproc or min(len(jobs), 8)
    with Pool(nproc, maxtasksperchild=1) as pool:
        return pool.map(_slow_job, jobs)


def run(seeds=(4242, 4243, 4244, 4245), nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(len(seeds), max(1, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        return pool.map(run_seed, list(seeds))
