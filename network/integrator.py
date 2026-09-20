# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from dataclasses import dataclass


@dataclass
class SimResult:
    spikes: np.ndarray
    t: np.ndarray
    V_trace: np.ndarray = None
    phi_trace: np.ndarray = None


@dataclass
class Intrinsics:
    tau_m: np.ndarray
    V_rest: np.ndarray
    V_th: np.ndarray
    V_reset: np.ndarray
    tau_ref: np.ndarray
    gain: np.ndarray
    bias: np.ndarray = None


def homogeneous(config):
    n = config.network.N
    p = config.neuron
    one = np.ones(n)
    return Intrinsics(one * p.tau_m, one * p.V_rest, one * p.V_th,
                      one * p.V_reset, one * p.tau_ref, one.copy())


V_SPREAD_MV = 3.0

TAU_M_RANGE = (2.0, 40.0)
TAU_REF_RANGE = (0.5, 5.0)


def heterogeneous(config, level, seed=0):
    n = config.network.N
    p = config.neuron
    one = np.ones(n)
    if level <= 0:
        return homogeneous(config)
    rng = np.random.default_rng(seed)
    sig = np.sqrt(np.log(1.0 + level ** 2))

    def lognorm():
        return np.exp(rng.normal(-0.5 * sig ** 2, sig, n))

    tau_m = np.clip(p.tau_m * lognorm(), *TAU_M_RANGE)
    spread = level * V_SPREAD_MV
    V_rest = one * p.V_rest + rng.normal(0.0, spread, n)
    reach = np.maximum((p.V_th - p.V_rest) + rng.normal(0.0, spread, n), 2.0)
    V_th = V_rest + reach
    V_reset = np.clip(V_rest + (p.V_reset - p.V_rest) + rng.normal(0.0, spread, n),
                      None, V_th - 1.0)
    tau_ref = np.clip(p.tau_ref * lognorm(), *TAU_REF_RANGE)
    return Intrinsics(tau_m, V_rest, V_th, V_reset, tau_ref, one.copy())


def biphasic_ap(dt, dur_ms=3.0, peak=1.0, width=0.35, ahp_ratio=0.30,
                ahp_width=0.8, ahp_lag=1.0):
    n = max(1, int(round(dur_ms / dt)))
    tt = np.arange(n) * dt
    return peak * (np.exp(-(tt ** 2) / (2 * width ** 2))
                   - ahp_ratio * np.exp(-((tt - ahp_lag) ** 2) / (2 * ahp_width ** 2)))


def build_gap_operator(pos, sigma, g, sparsify=0.02):
    from scipy.spatial.distance import cdist
    import scipy.sparse as sp
    D = cdist(pos, pos)
    A = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    if sparsify > 0:
        A[A < sparsify * A.max(axis=1, keepdims=True)] = 0.0
    rs = A.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    A = A / rs
    L = np.diag(A.sum(1)) - A
    return sp.csr_matrix(-g * L)


def simulate(config, geometry, I_ext, T_ms, use_elif=True, record_voltage=False,
             record_phi=False, intr=None, field_mode='sub', ap_gain=1.0,
             ap_template=None, ap_operator=None, gap_operator=None, phi_replay=None,
             v_init=None, adapt=None):
    dt = config.sim.dt
    NE, N = config.network.NE, config.network.N
    syn = config.synapse
    alpha = config.elif_params.alpha
    p = intr if intr is not None else homogeneous(config)

    T_steps = int(round(T_ms / dt)) + 1
    t = np.arange(T_steps) * dt
    decay_e = np.exp(-dt / syn.tau_e)
    decay_i = np.exp(-dt / syn.tau_i)

    V = p.V_rest.copy() if v_init is None else np.asarray(v_init, float).copy()
    h_E = np.zeros(N)
    h_I = np.zeros(N)
    ref = np.zeros(N)
    spikes = np.zeros((N, T_steps))
    V_trace = np.zeros((N, T_steps)) if record_voltage else None
    phi_trace = np.zeros((N, T_steps)) if record_phi else None
    if record_voltage:
        V_trace[:, 0] = V

    W_E, W_I = geometry.W_E, geometry.W_I
    M = geometry.W_eph_all
    if M is None:
        import scipy.sparse as sp
        M = sp.block_diag((geometry.W_eph_E, geometry.W_eph_I), format='csr')

    if adapt is not None:
        tau_w, b_w = float(adapt[0]), float(adapt[1])
        w = np.zeros(N)
        decay_w = np.exp(-dt / tau_w)

    use_ap = field_mode == 'ap'
    if use_ap:
        tmpl = ap_template if ap_template is not None else biphasic_ap(dt)
        L_ap = len(tmpl)
        ap_phase = np.full(N, -1, dtype=np.int64)
    src_hold = np.zeros(N)

    for ti in range(1, T_steps):
        h_E = h_E * decay_e + W_E @ spikes[:NE, ti - 1]
        h_I = h_I * decay_i + W_I @ spikes[NE:, ti - 1]
        I_total = p.gain * I_ext[:, ti] + h_E + h_I
        if p.bias is not None:
            I_total = I_total + p.bias
        if adapt is not None:
            w *= decay_w
            I_total = I_total - w

        phi = None
        if use_elif:
            live = V - p.V_rest
            src = live
            if field_mode == 'hold':
                inref = ref > 0
                src = np.where(inref, src_hold, live)
                src_hold = np.where(inref, src_hold, live)
            elif use_ap:
                on = ap_phase >= 0
                ap_src = np.zeros(N)
                if on.any():
                    ap_src[on] = ap_gain * tmpl[np.minimum(ap_phase[on], L_ap - 1)]
                if ap_operator is None:
                    src = src + ap_src
            phi = alpha * (M @ src)
            if use_ap and ap_operator is not None:
                phi = phi + alpha * (ap_operator @ ap_src)
            I_total = I_total + phi
        if gap_operator is not None:
            I_total = I_total + gap_operator @ (V - p.V_rest)
        if phi_replay is not None:
            I_total = I_total + phi_replay[:, ti]
        if record_phi:
            phi_trace[:, ti] = phi if phi is not None else 0.0

        active = ref <= 0
        dV = (-(V - p.V_rest) + I_total) / p.tau_m
        V[active] += dt * dV[active]

        fired = V >= p.V_th
        spikes[fired, ti] = 1.0
        V[fired] = p.V_reset[fired]
        ref[fired] = p.tau_ref[fired]
        if adapt is not None:
            w[fired] += b_w
        if use_ap:
            ap_phase[ap_phase >= 0] += 1
            ap_phase[ap_phase >= L_ap] = -1
            ap_phase[fired] = 0
        ref = np.maximum(0.0, ref - dt)
        if record_voltage:
            V_trace[:, ti] = V

    return SimResult(spikes=spikes, t=t, V_trace=V_trace, phi_trace=phi_trace)
