# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np

from network.biophysics import ap_field_template
from network import biophysics as B

DT = 0.05
V_REST, V_TH, V_RESET, TAU_M, TAU_REF = -60.0, -50.0, -60.0, 10.0, 2.0


def _source_cell(T_ms, drive, noise_amp=0.0, seed=0):
    n = int(round(T_ms / DT)) + 1
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    V = np.full(n, V_REST)
    spikes = np.zeros(n, bool)
    ref = np.zeros(n, bool)
    v, r = V_REST, 0.0
    for i in range(1, n):
        I = drive + (noise_amp * rng.standard_normal() if noise_amp else 0.0)
        if r > 0:
            v = V_REST
            r -= DT
            ref[i] = True
        else:
            v += DT * (-(v - V_REST) + I) / TAU_M
            if v >= V_TH:
                spikes[i] = True
                v = V_RESET
                r = TAU_REF
        V[i] = v
    return t, V, spikes, ref


def field_sources(V, spikes, ref, ap_gain=3.0):
    sub = V - V_REST
    hold = sub.copy()
    last = 0.0
    for i in range(len(sub)):
        if ref[i] or spikes[i]:
            hold[i] = last
        else:
            last = sub[i]
    tmpl = ap_field_template(DT)
    ap = sub.copy()
    idx = np.where(spikes)[0]
    for s in idx:
        end = min(len(ap), s + len(tmpl))
        ap[s:end] += ap_gain * tmpl[:end - s]
    return {'sub': sub, 'hold': hold, 'ap': ap}


def neighbour_response(source_term, alpha, weight=1.0):
    out = np.zeros_like(source_term)
    v = 0.0
    drive = alpha * weight * source_term
    for i in range(1, len(out)):
        v += DT * (-v + drive[i]) / TAU_M
        out[i] = v
    return out


def pair_example(T_ms=200.0, drive=10.9, alpha=0.2, weight=1.0 / 56.5, ap_gain=150.0,
                 noise_amp=1.5, seed=3):
    t, V, spikes, ref = _source_cell(T_ms, drive, noise_amp=noise_amp, seed=seed)
    src = field_sources(V, spikes, ref, ap_gain=ap_gain)
    phi = {k: alpha * weight * v for k, v in src.items()}
    resp = {k: neighbour_response(v, alpha, weight) for k, v in src.items()}
    return {'t': t, 'V_source': V, 'spikes': spikes, 'refractory': ref,
            'phi': phi, 'neighbour': resp, 'alpha': alpha, 'weight': weight}


def spike_aligned(T_ms=200.0, pre_ms=5.0, post_ms=10.0, **kw):
    ex = pair_example(T_ms=T_ms, **kw)
    pre, post = int(pre_ms / DT), int(post_ms / DT)
    idx = np.where(ex['spikes'])[0]
    idx = idx[(idx > pre) & (idx < len(ex['t']) - post)]
    lags = (np.arange(pre + post) - pre) * DT
    out = {'lag': lags}
    for group in ('phi', 'neighbour'):
        out[group] = {}
        for k, v in ex[group].items():
            seg = np.vstack([v[s - pre:s + post] for s in idx])
            seg = seg - seg[:, :int(2.0 / DT)].mean(1, keepdims=True)
            out[group][k] = seg.mean(0)
    return out


def measured_excess_sta(path=None, close_um=70.0, far_um=150.0):
    import os
    import pickle
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'multipatch', 'sta.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as fh:
        rows = pickle.load(fh)
    rows = [r for r in rows if not r['connected'] and r.get('sta_mV') is not None]
    lag = np.asarray(rows[0]['tau_ms'], float)
    close = np.array([r['sta_mV'] for r in rows if r['distance'] < close_um])
    far = np.array([r['sta_mV'] for r in rows if r['distance'] > far_um])
    excess = (close.mean(0) - far.mean(0)) * 1e3
    return {'lag_ms': lag, 'excess_uV': excess,
            'n_close': int(close.shape[0]), 'n_far': int(far.shape[0])}


def dipole_close_minus_far(path=None, close_um=70.0, far_um=150.0, h_um=400.0,
                           tau_m=TAU_M, kappa=0.15):
    import os
    import pickle
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'multipatch', 'sta.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as fh:
        rows = pickle.load(fh)
    rows = [r for r in rows if not r['connected'] and r.get('sta_mV') is not None]
    d = np.array([r['distance'] for r in rows], float)

    def geom(r_um):
        r = np.asarray(r_um, float) * 1e-6
        h = h_um * 1e-6
        return 1.0 / r - 1.0 / np.sqrt(r ** 2 + h ** 2)

    close, far = d[d < close_um], d[d > far_um]
    dg = geom(close).mean() - geom(far).mean()
    t, Vs, Vd, I_nA = B.two_compartment_spike(dt=0.005, I_app=40.0,
                                              temp_c=B.TEMP_C)
    V_e = I_nA * 1e-9 * dg / (4 * np.pi * B.SIGMA_E) * 1e3
    pol = B.membrane_polarization(V_e, 0.005, tau_m=tau_m, kappa=kappa)
    peak = t[np.argmax(Vs)]
    return {'t': t - peak, 'excess_uV': -(V_e + pol) * 1e3,
            'n_close': int(close.size), 'n_far': int(far.size)}


def biophysical_reference(r_um=50.0, h_um=400.0, tau_m=TAU_M, kappa=0.15):
    t, Vs, Vd, I_nA = B.two_compartment_spike(dt=0.005, I_app=40.0,
                                              temp_c=B.TEMP_C)
    Ve = B.dipole_potential(I_nA, [r_um], h_um=h_um)[:, 0]
    pol = B.membrane_polarization(Ve, 0.005, tau_m=tau_m, kappa=kappa)
    peak = t[np.argmax(Vs)]
    return {'t': t - peak, 'V_soma': Vs, 'I_nA': I_nA, 'V_e_mV': Ve,
            'polarization_mV': pol}


def distance_profile(distances_um=np.linspace(5, 400, 60), h_um=400.0):
    t, Vs, Vd, I_nA = B.two_compartment_spike(dt=0.005, I_app=40.0,
                                              temp_c=B.TEMP_C)
    Ve = B.dipole_potential(I_nA, distances_um, h_um=h_um)
    peak = Ve[np.argmin(Ve[:, 0]), :]
    return {'d_um': np.asarray(distances_um), 'peak_mV': peak}
