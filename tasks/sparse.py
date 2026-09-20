# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import numpy as np
from tasks import Task, TaskResult
from simulation import simulate_network
from analysis import compute_pca_trajectories


class SparseCodingTask(Task):

    def run(self, seed=5000):
        p = self.config.sparse
        T_steps = int(round(p.T_trial / self.dt)) + 1
        t_trial = np.arange(T_steps) * self.dt

        dur_steps = int(round(p.duration / self.dt))
        t_norm = np.linspace(0.0, 1.0, dur_steps)

        profiles = np.zeros((p.N_patterns, dur_steps))
        profiles[0] = np.exp(-2.0 * t_norm)
        profiles[1] = np.exp(-2.0 * (1.0 - t_norm))
        profiles[2] = np.exp(-8.0 * (t_norm - 0.5) ** 2)
        profiles[3] = 0.7 * np.ones(dur_steps)
        for k in range(p.N_patterns):
            mx = profiles[k].max()
            if mx > 0:
                profiles[k] /= mx

        start_idx = int(round(p.start / self.dt))
        end_idx = start_idx + dur_steps

        resp_start = p.start
        resp_end = p.start + p.duration + 20.0
        count_mask = (t_trial >= resp_start) & (t_trial < resp_end)

        noise_levels = np.asarray(p.noise_levels)
        sparsity_levels = np.asarray(p.sparsity_levels)
        n_noise = len(noise_levels)
        n_sparsity = len(sparsity_levels)

        accuracy_std = np.zeros((n_sparsity, n_noise))
        accuracy_elif = np.zeros((n_sparsity, n_noise))
        accuracy_elif_rm = np.zeros((n_sparsity, n_noise))
        lda_std = np.zeros((n_sparsity, n_noise))
        lda_elif = np.zeros((n_sparsity, n_noise))
        lda_elif_rm = np.zeros((n_sparsity, n_noise))
        confusion_std = np.zeros((p.N_patterns, p.N_patterns))
        confusion_elif = np.zeros((p.N_patterns, p.N_patterns))
        confusion_elif_rm = np.zeros((p.N_patterns, p.N_patterns))
        lda_conf_std = np.zeros((p.N_patterns, p.N_patterns))
        lda_conf_elif = np.zeros((p.N_patterns, p.N_patterns))
        lda_conf_rm = np.zeros((p.N_patterns, p.N_patterns))
        n_cells_conf = 0

        all_counts_s = {}
        all_counts_e = {}
        all_counts_r = {}

        traj_si, traj_ni = min(2, n_sparsity - 1), min(2, n_noise - 1)
        n_traj_trials = min(15, p.N_trials)
        spk_examples = {'std':  [[] for _ in range(p.N_patterns)],
                        'elif': [[] for _ in range(p.N_patterns)],
                        'rm':   [[] for _ in range(p.N_patterns)]}

        print(f"  Sparse: {n_sparsity} sparsity x {n_noise} noise = "
              f"{n_sparsity * n_noise} conditions, "
              f"{p.N_patterns} patterns x {p.N_trials} trials each")

        for si, sparsity in enumerate(sparsity_levels):
            n_active = max(1, int(round(self.NE * sparsity)))

            rng_pat = np.random.default_rng(seed + si * 1000)
            pattern_neurons = [
                rng_pat.choice(self.NE, size=n_active, replace=False)
                for _ in range(p.N_patterns)
            ]

            for ni, noise_level in enumerate(noise_levels):
                signal_strength = p.signal_amp * (1.0 - noise_level)
                print(f"    sparsity={sparsity:.2f}, noise={noise_level:.1f} "
                      f"(n_active={n_active}, signal={signal_strength:.3f})")

                counts_s = np.zeros((p.N_patterns, p.N_trials, self.N))
                counts_e = np.zeros((p.N_patterns, p.N_trials, self.N))

                for pat in range(p.N_patterns):
                    active_idx = pattern_neurons[pat]

                    for trial in range(p.N_trials):
                        trial_seed = (seed + 20000
                                      + si * 100000
                                      + ni * 10000
                                      + pat * p.N_trials
                                      + trial)

                        rng_noise = np.random.default_rng(trial_seed)
                        shared_noise = self._build_shared_noise(
                            T_steps, p.shared_noise_amp, rng_noise, smooth_ms=50.0)

                        rng_priv = np.random.default_rng(trial_seed + 600000)
                        I_ext = (p.baseline * np.ones((self.N, T_steps))
                                 + shared_noise
                                 + p.private_noise_amp
                                 * rng_priv.standard_normal((self.N, T_steps)))

                        stim_amp_shared = 2.0
                        I_ext[:self.NE, start_idx:end_idx] += stim_amp_shared
                        if signal_strength > 0:
                            I_ext[active_idx, start_idx:end_idx] += (
                                signal_strength * profiles[pat][np.newaxis, :])

                        res_std = simulate_network(
                            self.config, self.geometry, I_ext, p.T_trial,
                            use_elif=False)
                        counts_s[pat, trial] = (
                            res_std.spikes[:, count_mask].sum(axis=1))

                        res_elif = simulate_network(
                            self.config, self.geometry, I_ext, p.T_trial,
                            use_elif=True)
                        counts_e[pat, trial] = (
                            res_elif.spikes[:, count_mask].sum(axis=1))

                        if si == traj_si and ni == traj_ni and trial < n_traj_trials:
                            spk_examples['std'][pat].append(res_std.spikes)
                            spk_examples['elif'][pat].append(res_elif.spikes)

                all_counts_s[(si, ni)] = counts_s.copy()
                all_counts_e[(si, ni)] = counts_e.copy()

                accuracy_std[si, ni], cm_s = self._classify_loo(
                    counts_s, p.N_patterns, p.N_trials, return_confusion=True)
                accuracy_elif[si, ni], cm_e = self._classify_loo(
                    counts_e, p.N_patterns, p.N_trials, return_confusion=True)
                confusion_std += cm_s
                confusion_elif += cm_e
                n_cells_conf += 1

                from analysis import decode_sklearn as _ds
                X_s = counts_s.reshape(p.N_patterns * p.N_trials, self.N)
                X_e = counts_e.reshape(p.N_patterns * p.N_trials, self.N)
                y_lbl = np.repeat(np.arange(p.N_patterns), p.N_trials)
                try:
                    r_s = _ds(X_s, y_lbl, method='lda', k_folds=5, return_confusion=True)
                    r_e = _ds(X_e, y_lbl, method='lda', k_folds=5, return_confusion=True)
                    lda_std[si, ni] = r_s['accuracy']; lda_elif[si, ni] = r_e['accuracy']
                    lda_conf_std += r_s['confusion']; lda_conf_elif += r_e['confusion']
                except Exception:
                    lda_std[si, ni] = 0.0
                    lda_elif[si, ni] = 0.0

        n_total = n_sparsity * n_noise * p.N_patterns * p.N_trials
        spk_std_total = sum(c.sum() for c in all_counts_s.values()) / n_total
        spk_elif_total = sum(c.sum() for c in all_counts_e.values()) / n_total

        rate_ratio = spk_elif_total / max(1, spk_std_total)
        print(f"  Firing rate: Std={spk_std_total:.0f}, eLIF={spk_elif_total:.0f} "
              f"(ratio={rate_ratio:.3f}, +{(rate_ratio-1)*100:.1f}%)")

        SKIP_RATE_MATCH = False
        if not SKIP_RATE_MATCH and spk_elif_total > spk_std_total * 1.005:
            print("  Finding rate-matched delta_I...")

            N_REFINE_TRIALS = 4

            pat_neurons_per_si = []
            for si_, sp_ in enumerate(sparsity_levels):
                n_active_ = max(1, int(round(self.NE * sp_)))
                rng_p_ = np.random.default_rng(seed + si_ * 1000)
                pat_neurons_per_si.append([
                    rng_p_.choice(self.NE, size=n_active_, replace=False)
                    for _ in range(p.N_patterns)
                ])

            def _refine_sample(dI):
                tot = 0.0; cnt = 0
                for si_ in range(n_sparsity):
                    for ni_ in range(n_noise):
                        sig_ = p.signal_amp * (1.0 - noise_levels[ni_])
                        for pat in range(p.N_patterns):
                            active_idx = pat_neurons_per_si[si_][pat]
                            for trial in range(N_REFINE_TRIALS):
                                ts = (seed + 20000 + si_ * 100000
                                      + ni_ * 10000 + pat * p.N_trials + trial)
                                rngn = np.random.default_rng(ts)
                                sn = self._build_shared_noise(
                                    T_steps, p.shared_noise_amp, rngn, smooth_ms=50.0)
                                rngp = np.random.default_rng(ts + 600000)
                                I_ext_rm = ((p.baseline - dI) * np.ones((self.N, T_steps))
                                            + sn
                                            + p.private_noise_amp
                                            * rngp.standard_normal((self.N, T_steps)))
                                I_ext_rm[:self.NE, start_idx:end_idx] += 2.0
                                if sig_ > 0:
                                    I_ext_rm[active_idx, start_idx:end_idx] += (
                                        sig_ * profiles[pat][np.newaxis, :])
                                rrm = simulate_network(
                                    self.config, self.geometry, I_ext_rm,
                                    p.T_trial, use_elif=True)
                                tot += rrm.spikes[:, count_mask].sum()
                                cnt += 1
                return tot / max(1, cnt)

            target_mid = float(spk_std_total)
            print(f"    Target Std rate (global avg across all 25 conditions) = {target_mid:.1f}")
            print(f"    eLIF (dI=0) rate (global avg)                         = {spk_elif_total:.1f}")
            lo, hi = 0.0, 3.0
            for bisect_pass in range(10):
                mid = (lo + hi) / 2
                r = _refine_sample(mid)
                err = (r - target_mid) / max(1, target_mid)
                print(f"    bisect {bisect_pass+1}: Δ_I={mid:.4f} "
                      f"rate={r:.1f} target={target_mid:.1f} err={100*err:+.2f}%")
                if abs(err) < 0.002:
                    break
                if r > target_mid:
                    lo = mid
                else:
                    hi = mid
            delta_I = (lo + hi) / 2
            print(f"  Bisection-converged delta_I = {delta_I:.4f}")

            print(f"  Re-running {n_sparsity} sparsity x {n_noise} noise (rate-matched)...")
            for si, sparsity in enumerate(sparsity_levels):
                n_active = max(1, int(round(self.NE * sparsity)))
                rng_pat = np.random.default_rng(seed + si * 1000)
                pattern_neurons = [
                    rng_pat.choice(self.NE, size=n_active, replace=False)
                    for _ in range(p.N_patterns)
                ]

                for ni, noise_level in enumerate(noise_levels):
                    signal_strength = p.signal_amp * (1.0 - noise_level)
                    counts_rm = np.zeros((p.N_patterns, p.N_trials, self.N))

                    for pat in range(p.N_patterns):
                        active_idx = pattern_neurons[pat]
                        for trial in range(p.N_trials):
                            trial_seed = (seed + 20000
                                          + si * 100000
                                          + ni * 10000
                                          + pat * p.N_trials
                                          + trial)
                            rng_noise = np.random.default_rng(trial_seed)
                            shared_noise = self._build_shared_noise(
                                T_steps, p.shared_noise_amp, rng_noise, smooth_ms=50.0)
                            rng_priv = np.random.default_rng(trial_seed + 600000)
                            I_ext_rm = ((p.baseline - delta_I) * np.ones((self.N, T_steps))
                                        + shared_noise
                                        + p.private_noise_amp
                                        * rng_priv.standard_normal((self.N, T_steps)))
                            I_ext_rm[:self.NE, start_idx:end_idx] += 2.0
                            if signal_strength > 0:
                                I_ext_rm[active_idx, start_idx:end_idx] += (
                                    signal_strength * profiles[pat][np.newaxis, :])

                            res_rm = simulate_network(
                                self.config, self.geometry, I_ext_rm, p.T_trial,
                                use_elif=True)
                            counts_rm[pat, trial] = (
                                res_rm.spikes[:, count_mask].sum(axis=1))
                            if si == traj_si and ni == traj_ni and trial < n_traj_trials:
                                spk_examples['rm'][pat].append(res_rm.spikes)

                    accuracy_elif_rm[si, ni], cm_rm = self._classify_loo(
                        counts_rm, p.N_patterns, p.N_trials, return_confusion=True)
                    confusion_elif_rm += cm_rm
                    all_counts_r[(si, ni)] = counts_rm.copy()
                    X_rm = counts_rm.reshape(p.N_patterns * p.N_trials, self.N)
                    try:
                        r_rm = _ds(X_rm, y_lbl, method='lda', k_folds=5, return_confusion=True)
                        lda_elif_rm[si, ni] = r_rm['accuracy']; lda_conf_rm += r_rm['confusion']
                    except Exception:
                        lda_elif_rm[si, ni] = 0.0
        else:
            delta_I = 0.0
            accuracy_elif_rm = accuracy_elif.copy()
            lda_elif_rm = lda_elif.copy()
            confusion_elif_rm = confusion_elif.copy()
            lda_conf_rm = lda_conf_elif.copy()

        improvement = accuracy_elif - accuracy_std
        improvement_rm = accuracy_elif_rm - accuracy_std
        chance = 100.0 / p.N_patterns

        print(f"  Mean accuracy:  Std={accuracy_std.mean():.1f}%, "
              f"eLIF={accuracy_elif.mean():.1f}%, "
              f"eLIF_rm={accuracy_elif_rm.mean():.1f}% (chance={chance:.0f}%)")
        print(f"  Mean improvement (eLIF - Std): {improvement.mean():.1f}%")
        print(f"  Mean improvement (eLIF_rm - Std): {improvement_rm.mean():.1f}%")
        print(f"  Max improvement:  {improvement.max():.1f}% at "
              f"sparsity={sparsity_levels[np.unravel_index(improvement.argmax(), improvement.shape)[0]]:.2f}, "
              f"noise={noise_levels[np.unravel_index(improvement.argmax(), improvement.shape)[1]]:.1f}")
        print(f"  delta_I = {delta_I:.2f}")

        if n_cells_conf > 0:
            confusion_std /= n_cells_conf
            confusion_elif /= n_cells_conf
            confusion_elif_rm /= n_cells_conf

        def _rownorm(cm):
            cm = np.asarray(cm, float); rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            return cm / rs
        lda_confusion_std = _rownorm(lda_conf_std)
        lda_confusion_elif = _rownorm(lda_conf_elif)
        lda_confusion_elif_rm = _rownorm(lda_conf_rm)

        if not any(spk_examples['rm']):
            spk_examples['rm'] = [list(s) for s in spk_examples['elif']]

        def _flatten_counts(d_dict):
            if not d_dict:
                return np.array([])
            arrs = []
            for (si, ni), c in d_dict.items():
                arrs.append(c.sum(axis=-1).flatten())
            return np.concatenate(arrs)

        per_trial_spikes = {
            'std':  _flatten_counts(all_counts_s),
            'elif': _flatten_counts(all_counts_e),
            'rm':   _flatten_counts(all_counts_r) if all_counts_r else None,
        }
        if per_trial_spikes['rm'] is None or len(per_trial_spikes['rm']) == 0:
            per_trial_spikes['rm'] = per_trial_spikes['elif'].copy()

        traj = compute_pca_trajectories(
            {'std': spk_examples['std'], 'elif': spk_examples['elif']},
            t_trial, p.start, p.start + p.duration,
            self.NE, n_time_bins=30, bin_width=20.0, dt=self.dt,
            n_example_trials=n_traj_trials)
        traj_rm = compute_pca_trajectories(
            {'std': spk_examples['std'], 'elif': spk_examples['rm']},
            t_trial, p.start, p.start + p.duration,
            self.NE, n_time_bins=30, bin_width=20.0, dt=self.dt,
            n_example_trials=n_traj_trials)

        return TaskResult(
            name='sparse_coding',
            metrics={
                'mean_accuracy_std': float(accuracy_std.mean()),
                'mean_accuracy_elif': float(accuracy_elif.mean()),
                'mean_accuracy_elif_rm': float(accuracy_elif_rm.mean()),
                'mean_lda_std': float(lda_std.mean()),
                'mean_lda_elif': float(lda_elif.mean()),
                'mean_lda_elif_rm': float(lda_elif_rm.mean()),
                'mean_improvement': float(improvement.mean()),
                'mean_improvement_rm': float(improvement_rm.mean()),
                'max_improvement': float(improvement.max()),
                'chance_level': chance,
                'delta_I': delta_I,
                'spikes_per_trial_std': float(spk_std_total),
                'spikes_per_trial_elif': float(spk_elif_total),
                'rate_ratio': float(rate_ratio),
            },
            data={
                'sparsity_levels': sparsity_levels,
                'noise_levels': noise_levels,
                'accuracy_std': accuracy_std,
                'accuracy_elif': accuracy_elif,
                'accuracy_elif_rm': accuracy_elif_rm,
                'lda_std': lda_std,
                'lda_elif': lda_elif,
                'lda_elif_rm': lda_elif_rm,
                'improvement': improvement,
                'improvement_rm': improvement_rm,
                'lda_improvement': lda_elif - lda_std,
                'lda_improvement_rm': lda_elif_rm - lda_std,
                'confusion_std': confusion_std,
                'confusion_elif': confusion_elif,
                'confusion_elif_rm': confusion_elif_rm,
                'lda_confusion_std': lda_confusion_std,
                'lda_confusion_elif': lda_confusion_elif,
                'lda_confusion_elif_rm': lda_confusion_elif_rm,
                'pattern_names': ['Early-Decay', 'Late-Rise', 'Center-Peak', 'Sustained'],
                'profiles': profiles,
                't_norm': t_norm,
                'trajectories': traj,
                'trajectories_rm': traj_rm,
                't_trial': t_trial,
                'stim_onset': p.start,
                'stim_offset': p.start + p.duration,
                'per_trial_spikes': per_trial_spikes,
            }
        )

    @staticmethod
    def _classify_loo(counts, N_patterns, N_trials, return_confusion=False):
        N = counts.shape[2]
        n_correct = 0
        total = 0
        confusion = np.zeros((N_patterns, N_patterns), dtype=float)

        for pat in range(N_patterns):
            for trial in range(N_trials):
                centroids = np.zeros((N_patterns, N))
                for k in range(N_patterns):
                    if k == pat:
                        mask = np.ones(N_trials, dtype=bool)
                        mask[trial] = False
                        centroids[k] = counts[k, mask].mean(axis=0)
                    else:
                        centroids[k] = counts[k].mean(axis=0)

                test_vec = counts[pat, trial]
                dists = np.linalg.norm(centroids - test_vec[np.newaxis, :],
                                       axis=1)
                pred = np.argmin(dists)
                confusion[pat, pred] += 1
                if pred == pat:
                    n_correct += 1
                total += 1

        acc = 100.0 * n_correct / total
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        confusion = confusion / row_sums
        if return_confusion:
            return acc, confusion
        return acc
