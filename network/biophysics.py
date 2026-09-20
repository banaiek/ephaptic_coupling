# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np

SIGMA_E = 0.3

TEMP_C = 12.0
E_NA, E_K, E_L = 50.0, -77.0, -54.4
G_NA, G_K, G_L = 120.0, 36.0, 0.3
C_M = 1.0


def _rates(V, phi):
    an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    return phi * an, phi * bn, phi * am, phi * bm, phi * ah, phi * bh


def two_compartment_spike(dt=0.005, T=25.0, I_app=12.0, t_on=5.0, t_off=6.0,
                          p_soma=0.5, g_c=1.5, temp_c=TEMP_C, area_soma=1.0e-5):
    q10 = 3.0 ** ((temp_c - 6.3) / 10.0)
    n_steps = int(round(T / dt)) + 1
    t = np.arange(n_steps) * dt
    Vs = np.full(n_steps, -65.0)
    Vd = np.full(n_steps, -65.0)
    Is = np.zeros(n_steps)
    v = -65.0
    an, bn, am, bm, ah, bh = _rates(v, q10)
    n, m, h = an / (an + bn), am / (am + bm), ah / (ah + bh)
    vs, vd = v, v
    for i in range(1, n_steps):
        an, bn, am, bm, ah, bh = _rates(vs, q10)
        n += dt * (an * (1 - n) - bn * n)
        m += dt * (am * (1 - m) - bm * m)
        h += dt * (ah * (1 - h) - bh * h)
        i_ion = (G_NA * m ** 3 * h * (vs - E_NA) + G_K * n ** 4 * (vs - E_K)
                 + G_L * (vs - E_L))
        i_ax = g_c * (vd - vs) / p_soma
        drive = I_app if (t_on <= t[i] < t_off) else 0.0
        vs = vs + dt * (-i_ion + i_ax + drive) / C_M
        i_ion_d = G_L * (vd - E_L)
        vd = vd + dt * (-i_ion_d + g_c * (vs - vd) / (1 - p_soma)) / C_M
        Vs[i], Vd[i] = vs, vd
        Is[i] = i_ion + C_M * (Vs[i] - Vs[i - 1]) / dt - drive
    return t, Vs, Vd, Is * area_soma * 1e3


def dipole_potential(I_nA, r_um, h_um=200.0, sigma=SIGMA_E):
    r = np.asarray(r_um, float) * 1e-6
    h = h_um * 1e-6
    geom = 1.0 / r - 1.0 / np.sqrt(r ** 2 + h ** 2)
    return np.outer(I_nA * 1e-9, geom) / (4 * np.pi * sigma) * 1e3


def membrane_polarization(V_e, dt, tau_m=10.0, kappa=0.15):
    out = np.zeros_like(V_e)
    v = 0.0
    for i in range(1, len(V_e)):
        v += dt * (-v - kappa * V_e[i]) / tau_m
        out[i] = v
    return out


def lif_source_trace(dt=0.005, T=25.0, t_spike=5.5, tau_m=10.0, V_rest=-60.0,
                     V_th=-50.0, drive=10.6, tau_ref=2.0):
    n_steps = int(round(T / dt)) + 1
    t = np.arange(n_steps) * dt
    V = np.full(n_steps, V_rest)
    ref = 0.0
    v = V_rest
    spikes = []
    for i in range(1, n_steps):
        if ref > 0:
            v = V_rest
            ref -= dt
        else:
            v += dt * (-(v - V_rest) + drive) / tau_m
            if v >= V_th:
                spikes.append(t[i])
                v = V_rest
                ref = tau_ref
        V[i] = v
    return t, V, np.array(spikes)


def ap_current_template(t, t_spike, width=0.35, ahp_ratio=0.12, ahp_width=1.2,
                        ahp_lag=1.1):
    g = t - t_spike
    fast = np.exp(-(g ** 2) / (2 * width ** 2))
    slow = ahp_ratio * np.exp(-((g - ahp_lag) ** 2) / (2 * ahp_width ** 2))
    return fast - slow

def ap_field_template(dt, dur_ms=4.0, taper_ms=0.0, r_um=50.0, temp_c=TEMP_C, I_app=40.0):
    t, V_s, _, I = two_compartment_spike(temp_c=temp_c, I_app=I_app)
    drive = -dipole_potential(I, r_um).ravel()
    k_peak = int(np.argmax(V_s))
    dV = np.gradient(V_s, t[1] - t[0])
    fast = np.where(dV[:k_peak] > 10.0)[0]
    t0 = t[fast[0]] if len(fast) else t[k_peak]
    n = max(1, int(round(dur_ms / dt)))
    w = np.interp(t0 + np.arange(n) * dt, t, drive)
    if taper_ms > 0:
        m = min(n, max(1, int(round(taper_ms / dt))))
        w[-m:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, m)))
    return w / np.abs(w).max()
