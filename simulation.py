# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from dataclasses import dataclass
from typing import Optional
from config import Config
from geometry import NetworkGeometry


@dataclass
class SimulationResult:
    spikes: np.ndarray
    t: np.ndarray
    V_trace: Optional[np.ndarray] = None
    phi_trace: Optional[np.ndarray] = None


def simulate_network(config, geometry, I_ext, T_ms, use_elif=True,
                     record_voltage=False, record_phi=False, seed=None,
                     v_init=None):
    dt = config.sim.dt
    neuron = config.neuron
    syn = config.synapse
    NE = config.network.NE
    NI = config.network.NI
    N = config.network.N
    alpha = config.elif_params.alpha
    use_geo = config.elif_params.use_geometry

    T_steps = int(round(T_ms / dt)) + 1
    t = np.arange(T_steps) * dt

    decay_e = np.exp(-dt / syn.tau_e)
    decay_i = np.exp(-dt / syn.tau_i)

    V_rest = neuron.V_rest
    V_th = neuron.V_th
    V_reset = neuron.V_reset
    tau_m = neuron.tau_m
    tau_ref = neuron.tau_ref

    spikes = np.zeros((N, T_steps), dtype=np.float64)
    V = np.full(N, V_rest, dtype=np.float64) if v_init is None else np.asarray(v_init, float).copy()
    h_E = np.zeros(N, dtype=np.float64)
    h_I = np.zeros(N, dtype=np.float64)
    ref_timer = np.zeros(N, dtype=np.float64)

    V_trace = np.zeros((N, T_steps), dtype=np.float64) if record_voltage else None
    phi_trace = np.zeros((N, T_steps), dtype=np.float64) if record_phi else None

    if record_voltage:
        V_trace[:, 0] = V

    W_E = geometry.W_E
    W_I = geometry.W_I

    W_eph_E = geometry.W_eph_E if (use_elif and use_geo) else None
    W_eph_I = geometry.W_eph_I if (use_elif and use_geo) else None
    W_eph_all = getattr(geometry, 'W_eph_all', None)
    field_pop = getattr(geometry, 'field_population', 'split_EI')
    use_all_field = (use_elif and use_geo and field_pop == 'all'
                     and W_eph_all is not None)

    field_spike = bool(getattr(config.elif_params, 'field_spike', False)) and use_elif
    if field_spike:
        spike_gain = float(getattr(config.elif_params, 'spike_gain', 1.0))
        sp_peak = float(getattr(config.elif_params, 'spike_peak', 30.0))
        sp_dur = float(getattr(config.elif_params, 'spike_dur_ms', 2.0))
        L_ap = max(1, int(round(sp_dur / dt)))
        _tap = np.arange(L_ap) * dt
        _amp = sp_peak - V_rest
        ap_tmpl = (_amp * np.exp(-((_tap - 0.3) ** 2) / (2 * 0.25 ** 2))
                   - 0.05 * _amp * np.exp(-((_tap - 0.75) ** 2) / (2 * 0.3 ** 2)))
        ap_phase = np.full(N, -1, dtype=np.int64)

    for ti in range(1, T_steps):
        I_total = I_ext[:, ti].copy()

        h_E *= decay_e
        h_E += W_E @ spikes[:NE, ti - 1]
        h_I *= decay_i
        h_I += W_I @ spikes[NE:, ti - 1]
        I_total += h_E + h_I

        if use_elif:
            Vsrc = V - V_rest
            if field_spike:
                inap = ap_phase >= 0
                if inap.any():
                    Vsrc = Vsrc.copy()
                    Vsrc[inap] += spike_gain * ap_tmpl[np.minimum(ap_phase[inap], L_ap - 1)]
            if use_all_field:
                phi = alpha * (W_eph_all @ Vsrc)
                I_total += phi
                if record_phi:
                    phi_trace[:, ti] = phi
            elif use_geo:
                phi_E = alpha * (W_eph_E @ Vsrc[:NE])
                phi_I = alpha * (W_eph_I @ Vsrc[NE:])
                I_total[:NE] += phi_E
                I_total[NE:] += phi_I
                if record_phi:
                    phi_trace[:NE, ti] = phi_E
                    phi_trace[NE:, ti] = phi_I
            else:
                phi_E = alpha * np.mean(Vsrc[:NE])
                phi_I = alpha * np.mean(Vsrc[NE:])
                I_total[:NE] += phi_E
                I_total[NE:] += phi_I
                if record_phi:
                    phi_trace[:NE, ti] = phi_E
                    phi_trace[NE:, ti] = phi_I

        active = ref_timer <= 0
        dV = (-(V - V_rest) + I_total) / tau_m
        V[active] += dt * dV[active]

        spiked = V >= V_th
        spikes[:, ti] = spiked.astype(np.float64)
        V[spiked] = V_reset
        ref_timer[spiked] = tau_ref

        if field_spike:
            ongoing = ap_phase >= 0
            ap_phase[ongoing] += 1
            ap_phase[ap_phase >= L_ap] = -1
            ap_phase[spiked] = 0

        ref_timer = np.maximum(0.0, ref_timer - dt)

        if record_voltage:
            V_trace[:, ti] = V

    return SimulationResult(
        spikes=spikes, t=t,
        V_trace=V_trace, phi_trace=phi_trace
    )


def simulate_pair(config, I_pre, I_post_bg, T_ms, use_elif=True):
    dt = config.sim.dt
    n = config.neuron
    syn = config.synapse
    alpha = config.elif_params.alpha

    T_steps = int(round(T_ms / dt)) + 1

    V_pre = np.full(T_steps, n.V_rest)
    V_post = np.full(T_steps, n.V_rest)
    spk_pre = np.zeros(T_steps)
    spk_post = np.zeros(T_steps)
    h_syn = np.zeros(T_steps)
    phi = np.zeros(T_steps)
    ref_pre = 0.0
    ref_post = 0.0

    for i in range(1, T_steps):
        if use_elif:
            phi[i] = alpha * (V_post[i - 1] - n.V_rest)

        ref_pre = max(0.0, ref_pre - dt)
        if ref_pre <= 0:
            dV = (-(V_pre[i - 1] - n.V_rest) + I_pre[i - 1] + (phi[i] if use_elif else 0)) / n.tau_m
            V_pre[i] = V_pre[i - 1] + dt * dV
            if V_pre[i] >= n.V_th:
                spk_pre[i] = 1
                V_pre[i] = n.V_reset
                ref_pre = n.tau_ref
        else:
            V_pre[i] = n.V_reset

        h_syn[i] = h_syn[i - 1] * np.exp(-dt / syn.tau_e)
        if spk_pre[i] == 1:
            h_syn[i] += syn.J_ee

        ref_post = max(0.0, ref_post - dt)
        if ref_post <= 0:
            dV = (-(V_post[i - 1] - n.V_rest) + h_syn[i] + I_post_bg[i]) / n.tau_m
            V_post[i] = V_post[i - 1] + dt * dV
            if V_post[i] >= n.V_th:
                spk_post[i] = 1
                V_post[i] = n.V_reset
                ref_post = n.tau_ref
        else:
            V_post[i] = n.V_reset

    return {
        'V_pre': V_pre, 'V_post': V_post,
        'spk_pre': spk_pre, 'spk_post': spk_post,
        'h_syn': h_syn, 'phi': phi,
        't': np.arange(T_steps) * dt
    }
