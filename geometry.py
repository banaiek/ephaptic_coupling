# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from scipy.spatial.distance import cdist
from config import Config


def _halton_sequence(n, base):
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result[i] = r
    return result


def generate_positions(N, method='halton', seed=0):
    if method == 'halton':
        x = _halton_sequence(N, 2)
        y = _halton_sequence(N, 3)
        pos = np.column_stack([x, y])
    elif method == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0.05, 0.95, side)
        ys = np.linspace(0.05, 0.95, side)
        xx, yy = np.meshgrid(xs, ys)
        pos = np.column_stack([xx.ravel(), yy.ravel()])[:N]
        rng = np.random.default_rng(seed)
        pos += rng.normal(0, 0.01, pos.shape)
        pos = np.clip(pos, 0, 1)
    elif method == 'random':
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0, 1, (N, 2))
    else:
        raise ValueError(f"Unknown layout method: {method}")
    return pos


def build_ephaptic_matrix(pos, sigma):
    D = cdist(pos, pos)
    W = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


def build_noise_correlation_matrix(neuron_pos, K, sigma_input, seed=0):
    rng = np.random.default_rng(seed)
    source_pos = rng.uniform(0, 1, (K, 2))
    D = cdist(neuron_pos, source_pos)
    G = np.exp(-D ** 2 / (2 * sigma_input ** 2))
    row_sums = G.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    G /= row_sums
    return G, source_pos


def build_synaptic_connectivity(N, NE, NI, p_conn, w_ee, w_ee_std,
                                w_ie, w_ie_std, seed=100,
                                pos=None, sigma_syn=None):
    rng = np.random.default_rng(seed)

    if sigma_syn is not None and sigma_syn > 0 and pos is not None:
        pos_E = pos[:NE]
        pos_I = pos[NE:]
        d2_E = ((pos[:, None, :] - pos_E[None, :, :]) ** 2).sum(axis=2)
        d2_I = ((pos[:, None, :] - pos_I[None, :, :]) ** 2).sum(axis=2)
        kE = np.exp(-d2_E / (2.0 * sigma_syn ** 2))
        kI = np.exp(-d2_I / (2.0 * sigma_syn ** 2))
        for i in range(NE):
            kE[i, i] = 0.0
        rmE = kE.mean(axis=1, keepdims=True); rmE[rmE == 0] = 1.0
        rmI = kI.mean(axis=1, keepdims=True); rmI[rmI == 0] = 1.0
        pE = np.clip(kE / rmE * p_conn, 0.0, 1.0)
        pI = np.clip(kI / rmI * p_conn, 0.0, 1.0)
        conn_E = (rng.uniform(0, 1, (N, NE)) < pE).astype(float)
        conn_I = (rng.uniform(0, 1, (N, NI)) < pI).astype(float)
    else:
        conn_E = (rng.uniform(0, 1, (N, NE)) < p_conn).astype(float)
        conn_I = (rng.uniform(0, 1, (N, NI)) < p_conn).astype(float)

    weights_E = w_ee + w_ee_std * rng.standard_normal((N, NE))
    W_E = conn_E * weights_E
    for i in range(NE):
        W_E[i, i] = 0.0

    weights_I = w_ie + w_ie_std * rng.standard_normal((N, NI))
    W_I = conn_I * weights_I

    return W_E, W_I


class NetworkGeometry:

    def __init__(self, pos, W_eph_E, W_eph_I, G, source_pos, W_E, W_I, NE, NI,
                 W_eph_all=None, eph_kernel='gaussian_norm', field_population='split_EI'):
        self.pos = pos
        self.W_eph_E = W_eph_E
        self.W_eph_I = W_eph_I
        self.W_eph_all = W_eph_all
        self.G = G
        self.source_pos = source_pos
        self.W_E = W_E
        self.W_I = W_I
        self.NE = NE
        self.NI = NI
        self.eph_kernel = eph_kernel
        self.field_population = field_population

    @classmethod
    def build(cls, config: Config):
        from field import build_kernel
        N = config.network.N
        NE = config.network.NE
        NI = config.network.NI
        geo = config.geometry
        syn = config.synapse
        net = config.network

        import scipy.sparse as sp
        kind = getattr(geo, 'eph_kernel', 'gaussian_norm')
        fpop = getattr(geo, 'field_population', 'split_EI')
        ceps = getattr(geo, 'coulomb_eps', 0.02)
        spf = getattr(geo, 'eph_sparsify', 0.0)

        pos = generate_positions(N, method=geo.layout, seed=config.seed)
        pos_E = pos[:NE]
        pos_I = pos[NE:]

        W_eph_E = build_kernel(pos_E, kind, geo.sigma_eph, ceps, sparsify=spf)
        W_eph_I = build_kernel(pos_I, kind, geo.sigma_eph, ceps, sparsify=spf)
        W_eph_all = build_kernel(pos, kind, geo.sigma_eph, ceps, sparsify=spf) if fpop == 'all' else None

        G, source_pos = build_noise_correlation_matrix(
            pos, geo.K_sources, geo.sigma_input, seed=config.seed + 1
        )

        sigma_syn = getattr(geo, 'sigma_syn', 0.0)
        W_E, W_I = build_synaptic_connectivity(
            N, NE, NI, net.p_conn,
            syn.w_ee, syn.w_ee_std, syn.w_ie, syn.w_ie_std,
            seed=100,
            pos=pos if sigma_syn > 0 else None,
            sigma_syn=sigma_syn if sigma_syn > 0 else None,
        )

        if spf > 0:
            W_E = sp.csr_matrix(W_E); W_I = sp.csr_matrix(W_I)

        return cls(pos, W_eph_E, W_eph_I, G, source_pos, W_E, W_I, NE, NI,
                   W_eph_all=W_eph_all, eph_kernel=kind, field_population=fpop)
