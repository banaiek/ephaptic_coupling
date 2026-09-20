# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve


def spike_counts(spikes, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    return spikes[:, mask].sum(axis=1)


def firing_rate(spikes, dt, window_ms=20.0):
    pop = spikes.sum(axis=0)
    kernel_len = max(1, int(round(window_ms / dt)))
    kernel = np.ones(kernel_len) / kernel_len
    rate = fftconvolve(pop, kernel, mode='same') * (1000.0 / dt) / spikes.shape[0]
    return rate


def psth(spikes, dt, bin_ms=25.0):
    N, T = spikes.shape
    T_ms = T * dt
    bin_steps = max(1, int(round(bin_ms / dt)))
    n_bins = T // bin_steps
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    rates = np.zeros(n_bins)
    for b in range(n_bins):
        s = b * bin_steps
        e = min(s + bin_steps, T)
        rates[b] = spikes[:, s:e].sum() / N / (bin_ms / 1000.0)
    return centers, rates


def dprime(responses_A, responses_B):
    mu_a, mu_b = np.mean(responses_A), np.mean(responses_B)
    var_a, var_b = np.var(responses_A, ddof=1), np.var(responses_B, ddof=1)
    pooled_std = np.sqrt(0.5 * (var_a + var_b) + 1e-12)
    return (mu_a - mu_b) / pooled_std


def dprime_matrix(counts, labels, N_stim):
    templates = np.mean(counts, axis=1)
    D = np.zeros((N_stim, N_stim))

    for s1 in range(N_stim):
        for s2 in range(s1 + 1, N_stim):
            mu1 = templates[:, s1]
            mu2 = templates[:, s2]
            diff = mu2 - mu1
            norm_diff = np.linalg.norm(diff) + 1e-12

            proj1 = (counts[:, :, s1].T @ diff) / norm_diff
            proj2 = (counts[:, :, s2].T @ diff) / norm_diff

            d = abs(np.mean(proj1) - np.mean(proj2)) / \
                np.sqrt(0.5 * (np.var(proj1, ddof=1) + np.var(proj2, ddof=1)) + 1e-12)
            D[s1, s2] = d
            D[s2, s1] = d

    return D


def compute_roc(signal_resp, noise_resp, n_criteria=100):
    all_resp = np.concatenate([signal_resp, noise_resp])
    criteria = np.linspace(all_resp.min() - 1, all_resp.max() + 1, n_criteria)
    hit_rate = np.array([np.mean(signal_resp > c) for c in criteria])
    fa_rate = np.array([np.mean(noise_resp > c) for c in criteria])
    auc = -np.trapz(hit_rate, fa_rate)
    return fa_rate, hit_rate, auc


def noise_correlations(counts, n_pairs=500, seed=999):
    N, N_trials, N_stim = counts.shape
    rng = np.random.default_rng(seed)
    pair_idx = np.column_stack([rng.integers(0, N, n_pairs),
                                rng.integers(0, N, n_pairs)])
    same = pair_idx[:, 0] == pair_idx[:, 1]
    pair_idx[same, 1] = (pair_idx[same, 1] + 1) % N

    corrs = np.zeros(n_pairs)
    for p in range(n_pairs):
        n1, n2 = pair_idx[p]
        rvals = []
        for s in range(N_stim):
            r1 = counts[n1, :, s] - np.mean(counts[n1, :, s])
            r2 = counts[n2, :, s] - np.mean(counts[n2, :, s])
            if np.std(r1) > 0 and np.std(r2) > 0:
                rvals.append(np.corrcoef(r1, r2)[0, 1])
        corrs[p] = np.nanmean(rvals) if rvals else 0.0

    return corrs, pair_idx


def signal_correlations(counts, n_pairs=500, seed=999):
    N, N_trials, N_stim = counts.shape
    rng = np.random.default_rng(seed)
    pair_idx = np.column_stack([rng.integers(0, N, n_pairs),
                                rng.integers(0, N, n_pairs)])
    same = pair_idx[:, 0] == pair_idx[:, 1]
    pair_idx[same, 1] = (pair_idx[same, 1] + 1) % N

    corrs = np.zeros(n_pairs)
    for p in range(n_pairs):
        n1, n2 = pair_idx[p]
        sig1 = np.mean(counts[n1, :, :], axis=0)
        sig2 = np.mean(counts[n2, :, :], axis=0)
        if np.std(sig1) > 0 and np.std(sig2) > 0:
            corrs[p] = np.corrcoef(sig1, sig2)[0, 1]

    return corrs


def compute_pca_trajectories(spk_examples, t_trial, stim_onset, stim_offset,
                             NE, n_time_bins=30, bin_width=30.0, dt=0.1,
                             n_example_trials=20):
    from sklearn.decomposition import PCA

    N_stim = len(spk_examples['std'])
    time_bins = np.linspace(stim_onset, stim_offset, n_time_bins)
    Nt = len(t_trial)

    pop_traj = {}
    for cond in ['std', 'elif']:
        traj = np.zeros((NE, n_time_bins, N_stim))
        for s in range(N_stim):
            trials = spk_examples[cond][s]
            n_trials = len(trials)
            for trial_spk in trials:
                for tb in range(n_time_bins):
                    t_center = time_bins[tb]
                    t_start = max(0, int(round((t_center - bin_width / 2) / dt)))
                    t_end = min(Nt, int(round((t_center + bin_width / 2) / dt)))
                    traj[:, tb, s] += trial_spk[:NE, t_start:t_end].sum(axis=1) / n_trials
        pop_traj[cond] = traj

    kernel = np.exp(-np.arange(-3, 4) ** 2 / 2)
    kernel /= kernel.sum()
    for cond in ['std', 'elif']:
        for s in range(N_stim):
            for n in range(NE):
                pop_traj[cond][n, :, s] = np.convolve(
                    pop_traj[cond][n, :, s], kernel, mode='same')

    combined = np.hstack([
        pop_traj['std'].reshape(NE, -1),
        pop_traj['elif'].reshape(NE, -1)
    ])

    pca = PCA(n_components=min(10, NE))
    scores = pca.fit_transform(combined.T)

    n_pts = n_time_bins * N_stim
    scores_std = scores[:n_pts].reshape(n_time_bins, N_stim, -1)
    scores_elif = scores[n_pts:].reshape(n_time_bins, N_stim, -1)

    sep = {'std': 0.0, 'elif': 0.0}
    n_pairs = 0
    for s1 in range(N_stim):
        for s2 in range(s1 + 1, N_stim):
            d_std = np.sqrt(((scores_std[:, s1, :3] - scores_std[:, s2, :3]) ** 2).sum(axis=1))
            d_elif = np.sqrt(((scores_elif[:, s1, :3] - scores_elif[:, s2, :3]) ** 2).sum(axis=1))
            sep['std'] += d_std.mean()
            sep['elif'] += d_elif.mean()
            n_pairs += 1
    sep['std'] /= max(1, n_pairs)
    sep['elif'] /= max(1, n_pairs)

    return {
        'scores_std': scores_std,
        'scores_elif': scores_elif,
        'explained': pca.explained_variance_ratio_ * 100,
        'time_bins': time_bins,
        'separation': sep,
        'pop_traj': pop_traj
    }


def trajectory_tangling(traj_pc, N_stim, n_time_bins):
    eps = 0.1
    vel = np.diff(traj_pc[:, :, :3], axis=0)
    tangling = np.zeros(n_time_bins - 1)

    for t in range(n_time_bins - 1):
        max_Q = 0.0
        for s1 in range(N_stim):
            for s2 in range(s1 + 1, N_stim):
                pos_diff = traj_pc[t, s1, :3] - traj_pc[t, s2, :3]
                vel_diff = vel[t, s1, :] - vel[t, s2, :]
                Q = np.sum(vel_diff ** 2) / (np.sum(pos_diff ** 2) + eps)
                max_Q = max(max_Q, Q)
        tangling[t] = max_Q

    return tangling


def wilcoxon_test(x, y):
    if np.allclose(x, y):
        return {'statistic': 0.0, 'p_value': 1.0}
    stat, p = stats.wilcoxon(x, y)
    return {'statistic': stat, 'p_value': p}


def mannwhitney_test(x, y):
    stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
    return {'statistic': stat, 'p_value': p}


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) /
                         (nx + ny - 2) + 1e-12)
    return (np.mean(x) - np.mean(y)) / pooled_std


def bootstrap_ci(data, statistic=np.mean, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.array([statistic(rng.choice(data, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [100 * alpha, 100 * (1 - alpha)])


def decode_nearest_centroid_loo(counts, N_stim, N_trials):
    N = counts.shape[0]
    correct = 0
    confusion = np.zeros((N_stim, N_stim), dtype=int)

    for s in range(N_stim):
        for trial in range(N_trials):
            test = counts[:, trial, s]

            templates = np.zeros((N, N_stim))
            for s2 in range(N_stim):
                if s2 == s:
                    mask = np.ones(N_trials, dtype=bool)
                    mask[trial] = False
                    templates[:, s2] = counts[:, mask, s2].mean(axis=1)
                else:
                    templates[:, s2] = counts[:, :, s2].mean(axis=1)

            dists = np.sum((templates - test[:, None]) ** 2, axis=0)
            pred = np.argmin(dists)
            confusion[s, pred] += 1
            if pred == s:
                correct += 1

    accuracy = 100.0 * correct / (N_stim * N_trials)
    return accuracy, confusion


def decode_sklearn(X, y, method='lda', k_folds=5, seed=42, return_confusion=False):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix

    if method == 'lda':
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    elif method == 'svm':
        clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual=False))
    else:
        raise ValueError(f"Unknown method: {method}")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_accs = []
    n_classes = int(np.max(y)) + 1
    conf_mat = np.zeros((n_classes, n_classes), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        fold_accs.append(np.mean(y_pred == y[test_idx]) * 100)
        if return_confusion:
            conf_mat += confusion_matrix(y[test_idx], y_pred,
                                         labels=list(range(n_classes)))

    out = {
        'accuracy': np.mean(fold_accs),
        'std': np.std(fold_accs),
        'fold_accuracies': fold_accs,
        'ci': bootstrap_ci(np.array(fold_accs))
    }
    if return_confusion:
        out['confusion'] = conf_mat
    return out
