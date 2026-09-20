# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from copy import deepcopy
from config import Config
from geometry import NetworkGeometry
from simulation import simulate_network
import field


def _spatial_amp(response, x, m):
    if m == 0:
        return float(np.mean(response))
    c = np.cos(2 * np.pi * m * x); s = np.sin(2 * np.pi * m * x)
    A = np.column_stack([np.ones_like(x), c, s])
    coef, *_ = np.linalg.lstsq(A, response, rcond=None)
    return float(np.hypot(coef[1], coef[2]))


def _build_geom(base_cfg, kernel):
    cfg = deepcopy(base_cfg)
    cfg.geometry.eph_kernel = kernel
    cfg.geometry.field_population = 'all' if kernel == 'coulomb' else 'split_EI'
    cfg.geometry.eph_sparsify = 0.0
    return cfg, NetworkGeometry.build(cfg)


def run(config: Config, seed=4000):
    tp = config.transfer
    dt = config.sim.dt
    N = config.network.N
    alpha = config.elif_params.alpha
    sigma = config.geometry.sigma_eph
    Vrest, Vth = config.neuron.V_rest, config.neuron.V_th
    modes = list(tp.modes)
    T_steps = int(round(tp.T_trial / dt)) + 1
    t = np.arange(T_steps) * dt
    win = t >= tp.settle
    dur = win.sum() * dt / 1000.0
    sub_base = (Vth - Vrest) - 3.0

    out = {'modes': modes, 'alpha': alpha, 'sigma_eph': sigma, 'kernels': {}}
    kth = 2 * np.pi * np.array(modes, float)
    out['theory_gaussian'] = list(field.transfer_theory(kth, alpha, sigma))

    for kernel in tp.kernels:
        cfg, geom = _build_geom(config, kernel)
        x = geom.pos[:, 0]

        def make_input(m, baseline, trial, dI=0.0):
            rng = np.random.default_rng(seed + 1000 * m + trial)
            spatial = (baseline - dI) + tp.drive_amp * np.cos(2 * np.pi * m * x)
            I = np.tile(spatial[:, None], (1, T_steps))
            I += tp.private_noise_amp * rng.standard_normal((N, T_steps))
            return I

        res = {k: [] for k in ('Vgain_std', 'Vgain_elif', 'rate_gain_std',
                               'rate_gain_elif', 'rate_gain_rm',
                               'rate_std', 'rate_elif', 'rate_rm',
                               'Vgain_std_sem', 'Vgain_elif_sem',
                               'rate_gain_std_sem', 'rate_gain_elif_sem', 'rate_gain_rm_sem')}

        cfg_lin = deepcopy(cfg); cfg_lin.neuron.V_th = 1e12
        for m in modes:
            vs_l, ve_l = [], []
            for trial in range(max(2, tp.n_trials // 2)):
                I = make_input(m, 0.0, trial)
                rs = simulate_network(cfg_lin, geom, I, tp.T_trial, use_elif=False, record_voltage=True)
                re = simulate_network(cfg_lin, geom, I, tp.T_trial, use_elif=True, record_voltage=True)
                vs_l.append(_spatial_amp(rs.V_trace[:, win].mean(1) - Vrest, x, m) / tp.drive_amp)
                ve_l.append(_spatial_amp(re.V_trace[:, win].mean(1) - Vrest, x, m) / tp.drive_amp)
            res['Vgain_std'].append(float(np.mean(vs_l)))
            res['Vgain_elif'].append(float(np.mean(ve_l)))
            res['Vgain_std_sem'].append(float(np.std(vs_l, ddof=1) / np.sqrt(len(vs_l))) if len(vs_l) > 1 else 0.0)
            res['Vgain_elif_sem'].append(float(np.std(ve_l, ddof=1) / np.sqrt(len(ve_l))) if len(ve_l) > 1 else 0.0)

        mref = modes[len(modes) // 2]

        def probe_total(use_elif, dI):
            tot = 0.0
            for trial in range(3):
                I = make_input(mref, tp.baseline, trial, dI)
                tot += simulate_network(cfg, geom, I, tp.T_trial, use_elif=use_elif).spikes[:, win].sum()
            return tot / 3.0

        tgt, el0 = probe_total(False, 0.0), probe_total(True, 0.0)
        dI = 0.0
        if el0 > tgt * 1.01:
            lo, hi = 0.0, 3.0
            for _ in range(7):
                mid = (lo + hi) / 2
                if probe_total(True, mid) > tgt: lo = mid
                else: hi = mid
            dI = (lo + hi) / 2

        for m in modes:
            gs, ge, gr, rs_, re_, rr_ = [], [], [], [], [], []
            for trial in range(tp.n_trials):
                I = make_input(m, tp.baseline, trial, 0.0)
                Irm = make_input(m, tp.baseline, trial, dI)
                frs = simulate_network(cfg, geom, I, tp.T_trial, use_elif=False).spikes[:, win].sum(1) / dur
                fre = simulate_network(cfg, geom, I, tp.T_trial, use_elif=True).spikes[:, win].sum(1) / dur
                frr = simulate_network(cfg, geom, Irm, tp.T_trial, use_elif=True).spikes[:, win].sum(1) / dur
                gs.append(_spatial_amp(frs, x, m)); ge.append(_spatial_amp(fre, x, m)); gr.append(_spatial_amp(frr, x, m))
                rs_.append(frs.mean()); re_.append(fre.mean()); rr_.append(frr.mean())
            res['rate_gain_std'].append(float(np.mean(gs)))
            res['rate_gain_elif'].append(float(np.mean(ge)))
            res['rate_gain_rm'].append(float(np.mean(gr)))
            res['rate_gain_std_sem'].append(float(np.std(gs, ddof=1) / np.sqrt(len(gs))) if len(gs) > 1 else 0.0)
            res['rate_gain_elif_sem'].append(float(np.std(ge, ddof=1) / np.sqrt(len(ge))) if len(ge) > 1 else 0.0)
            res['rate_gain_rm_sem'].append(float(np.std(gr, ddof=1) / np.sqrt(len(gr))) if len(gr) > 1 else 0.0)
            res['rate_std'].append(float(np.mean(rs_)))
            res['rate_elif'].append(float(np.mean(re_)))
            res['rate_rm'].append(float(np.mean(rr_)))
        res['dI'] = dI
        out['kernels'][kernel] = res
        print(f"  [transfer:{kernel}] dI={dI:.2f}  Vgain eLIF "
              f"m{modes[0]}={res['Vgain_elif'][0]:.2f} -> m{modes[-1]}={res['Vgain_elif'][-1]:.2f} "
              f"(theory {out['theory_gaussian'][0]:.2f}->{out['theory_gaussian'][-1]:.2f})", flush=True)
    return out
