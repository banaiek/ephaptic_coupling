# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist


def _spectral_radius(M):
    if np.allclose(M, M.T, atol=1e-12):
        ev = np.linalg.eigvalsh(M)
        return float(np.max(np.abs(ev)))
    ev = np.linalg.eigvals(M)
    return float(np.max(np.abs(ev)))


def build_kernel(pos, kind='gaussian_norm', sigma=0.15, coulomb_eps=0.02,
                 normalize='auto', sparsify=0.0):
    D = cdist(pos, pos)
    if kind == 'gaussian_norm':
        W = np.exp(-D ** 2 / (2 * sigma ** 2)); np.fill_diagonal(W, 0.0)
        norm = 'row' if normalize == 'auto' else normalize
    elif kind == 'gaussian_spec':
        W = np.exp(-D ** 2 / (2 * sigma ** 2)); np.fill_diagonal(W, 0.0)
        norm = 'spectral' if normalize == 'auto' else normalize
    elif kind == 'coulomb':
        W = 1.0 / np.sqrt(D ** 2 + coulomb_eps ** 2); np.fill_diagonal(W, 0.0)
        norm = 'spectral' if normalize == 'auto' else normalize
    elif kind == 'global':
        N = pos.shape[0]
        W = (np.ones((N, N)) - np.eye(N))
        norm = 'row' if normalize == 'auto' else normalize
    else:
        raise ValueError(f"unknown kernel kind: {kind}")

    if sparsify > 0:
        rowmax = W.max(axis=1, keepdims=True)
        W[W < sparsify * rowmax] = 0.0

    if norm == 'row':
        rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        W = W / rs
    elif norm == 'spectral':
        sr = _spectral_radius(W)
        if sr > 0:
            W = W / sr
    if sparsify > 0:
        return sp.csr_matrix(W)
    return W


def mode_gains(M, alpha):
    ev = np.linalg.eigvals(M)
    lam = np.real(ev)
    gains = 1.0 / (1.0 - alpha * lam)
    order = np.argsort(-lam)
    return lam[order], gains[order]


def Mhat_gaussian(k, sigma):
    return np.exp(-(sigma ** 2) * (k ** 2) / 2.0)


def transfer_theory(k, alpha, sigma):
    return 1.0 / (1.0 - alpha * Mhat_gaussian(k, sigma))


def critical_alpha(M):
    lam = np.real(np.linalg.eigvals(M))
    lmax = np.max(lam)
    return np.inf if lmax <= 0 else 1.0 / lmax
