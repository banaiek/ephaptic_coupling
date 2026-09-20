# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import copy
import numpy as np
from tasks import Task, TaskResult
from simulation import simulate_network
from geometry import NetworkGeometry
from analysis import (decode_nearest_centroid_loo, decode_sklearn,
                      dprime_matrix, noise_correlations, signal_correlations,
                      compute_pca_trajectories)


class DecodingTask(Task):

    def _sim_counts(self, geom, tuning, I_baseline, dI, use_elif,
                    N_trials, N_stim, T_steps, t_trial, count_mask, p, seed_base):
        counts = np.zeros((self.N, N_trials, N_stim))
        stim_t = (t_trial >= p.stim_onset) & (t_trial < p.stim_offset)
        for s in range(N_stim):
            tw = tuning[:, s]
            for trial in range(N_trials):
                ts = seed_base + s * N_trials + trial
                rng_n = np.random.default_rng(ts)
                sn = self._build_shared_noise(T_steps, p.shared_noise_amp, rng_n, 50.0)
                rng_p = np.random.default_rng(ts + 100000)
                I = np.full((self.N, T_steps), I_baseline - dI, dtype=float)
                I += sn
                I += p.private_noise_amp * rng_p.standard_normal((self.N, T_steps))
                I[:self.NE][:, stim_t] += p.stim_amp_shared
                I[:self.NE][:, stim_t] += (p.stim_amp_signal * tw)[:, None]
                r = simulate_network(self.config, geom, I, p.T_trial, use_elif=use_elif)
                counts[:, trial, s] = r.spikes[:, count_mask].sum(axis=1)
        return counts

    def _match_dI(self, geom, tuning, target_total, N_stim, T_steps,
                  t_trial, count_mask, p, seed_base):
        stim_t = (t_trial >= p.stim_onset) & (t_trial < p.stim_offset)

        def sample(dI):
            tot = 0.0; cnt = 0
            for s in range(min(2, N_stim)):
                tw = tuning[:, s]
                for trial in range(5):
                    ts = seed_base + s * 1000 + trial
                    rng_n = np.random.default_rng(ts)
                    sn = self._build_shared_noise(T_steps, p.shared_noise_amp, rng_n, 50.0)
                    rng_p = np.random.default_rng(ts + 100000)
                    I = np.full((self.N, T_steps), p.I_baseline - dI, dtype=float)
                    I += sn
                    I += p.private_noise_amp * rng_p.standard_normal((self.N, T_steps))
                    I[:self.NE][:, stim_t] += p.stim_amp_shared
                    I[:self.NE][:, stim_t] += (p.stim_amp_signal * tw)[:, None]
                    r = simulate_network(self.config, geom, I, p.T_trial, use_elif=True)
                    tot += r.spikes[:, count_mask].sum(); cnt += 1
            return tot / max(1, cnt)

        lo, hi = 0.0, 3.0
        for _ in range(6):
            mid = (lo + hi) / 2
            if sample(mid) > target_total:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _acc(self, counts, N_stim, N_trials):
        nc, _ = decode_nearest_centroid_loo(counts, N_stim, N_trials)
        X = np.zeros((N_trials * N_stim, self.N))
        lab = np.zeros(N_trials * N_stim, dtype=int)
        for s in range(N_stim):
            idx = slice(s * N_trials, (s + 1) * N_trials)
            X[idx] = counts[:, :, s].T
            lab[idx] = s
        lda = decode_sklearn(X, lab, method='lda')['accuracy']
        spk = X.sum() / (N_stim * N_trials)
        return nc, lda, spk

    def _run_sweep(self, param, values, tuning, N_stim, N_trials,
                   T_steps, t_trial, count_mask, p, seed_base):
        out = {'param': param, 'values': list(values),
               'nc': {'std': [], 'elif': [], 'rm': []},
               'lda': {'std': [], 'elif': [], 'rm': []},
               'spk': {'std': [], 'elif': [], 'rm': []},
               'delta_I': []}
        for vi, val in enumerate(values):
            if param == 'I_baseline':
                geom = self.geometry
                I_base = float(val)
                old_I = p.I_baseline; p.I_baseline = I_base
            else:
                cfg2 = copy.deepcopy(self.config)
                cfg2.network.p_conn = float(val)
                geom = NetworkGeometry.build(cfg2)
                I_base = p.I_baseline
                old_I = None
            sb = seed_base + vi * 777
            c_std = self._sim_counts(geom, tuning, I_base, 0.0, False,
                                     N_trials, N_stim, T_steps, t_trial, count_mask, p, sb)
            c_el = self._sim_counts(geom, tuning, I_base, 0.0, True,
                                    N_trials, N_stim, T_steps, t_trial, count_mask, p, sb)
            nc_s, lda_s, spk_s = self._acc(c_std, N_stim, N_trials)
            nc_e, lda_e, spk_e = self._acc(c_el, N_stim, N_trials)
            dI = 0.0
            if spk_e > spk_s * 1.01:
                dI = self._match_dI(geom, tuning, spk_s, N_stim, T_steps,
                                    t_trial, count_mask, p, sb + 333_000)
            c_rm = self._sim_counts(geom, tuning, I_base, dI, True,
                                    N_trials, N_stim, T_steps, t_trial, count_mask, p, sb)
            nc_r, lda_r, spk_r = self._acc(c_rm, N_stim, N_trials)
            if old_I is not None:
                p.I_baseline = old_I
            out['nc']['std'].append(nc_s); out['nc']['elif'].append(nc_e); out['nc']['rm'].append(nc_r)
            out['lda']['std'].append(lda_s); out['lda']['elif'].append(lda_e); out['lda']['rm'].append(lda_r)
            out['spk']['std'].append(spk_s); out['spk']['elif'].append(spk_e); out['spk']['rm'].append(spk_r)
            out['delta_I'].append(dI)
            print(f"    {param}={val:g}: NC Std={nc_s:.0f}/eLIF={nc_e:.0f}/RM={nc_r:.0f}  "
                  f"LDA Std={lda_s:.0f}/eLIF={lda_e:.0f}/RM={lda_r:.0f}  "
                  f"spk Std={spk_s:.0f}/eLIF={spk_e:.0f}/RM={spk_r:.0f} (dI={dI:.2f})")
        return out

    def run(self, seed=1000):
        p = self.config.decoding
        T_steps = int(round(p.T_trial / self.dt)) + 1
        t_trial = np.arange(T_steps) * self.dt

        rng = np.random.default_rng(seed)
        neuron_preferred = rng.integers(0, p.N_stim, size=self.NE)
        tuning = np.zeros((self.NE, p.N_stim))
        for n in range(self.NE):
            for s in range(p.N_stim):
                dist = min(abs(neuron_preferred[n] - s),
                           p.N_stim - abs(neuron_preferred[n] - s))
                tuning[n, s] = 0.3 + 0.7 * np.exp(-dist ** 2 / (2 * p.tuning_width ** 2))

        count_mask = (t_trial >= p.count_start) & (t_trial < p.count_end)

        counts_std = np.zeros((self.N, p.N_trials, p.N_stim))
        counts_elif = np.zeros((self.N, p.N_trials, p.N_stim))

        n_example = min(20, p.N_trials)
        spk_examples = {'std': [[] for _ in range(p.N_stim)],
                        'elif': [[] for _ in range(p.N_stim)]}

        TARGET_STIM = 5
        NCANDIDATES = 12
        cand_order = np.argsort(-tuning[:, TARGET_STIM])[:NCANDIDATES]
        ex_candidates = [int(i) for i in cand_order]
        ex_neuron_rasters_cand = {
            f'{m}_{c}': [[] for _ in range(p.N_stim)]
            for m in ('std', 'elif', 'rm') for c in ex_candidates
        }
        n_ex_raster = min(10, p.N_trials)

        print(f"  Running {p.N_stim} stimuli x {p.N_trials} trials...")

        for s in range(p.N_stim):
            if (s + 1) % 2 == 0:
                print(f"    Stimulus {s + 1}/{p.N_stim}")

            tuning_weights = tuning[:, s]

            for trial in range(p.N_trials):
                trial_seed = 5000 + s * p.N_trials + trial

                rng_noise = np.random.default_rng(trial_seed)
                shared_noise = self._build_shared_noise(
                    T_steps, p.shared_noise_amp, rng_noise, smooth_ms=50.0)

                rng_private = np.random.default_rng(trial_seed + 100000)
                I_ext = np.zeros((self.N, T_steps))
                for ti in range(T_steps):
                    I_ext[:, ti] = p.I_baseline
                    I_ext[:, ti] += shared_noise[:, ti]
                    I_ext[:, ti] += p.private_noise_amp * rng_private.standard_normal(self.N)
                    if p.stim_onset <= t_trial[ti] < p.stim_offset:
                        I_ext[:self.NE, ti] += p.stim_amp_shared
                        I_ext[:self.NE, ti] += p.stim_amp_signal * tuning_weights

                res_std = simulate_network(
                    self.config, self.geometry, I_ext, p.T_trial, use_elif=False)
                counts_std[:, trial, s] = res_std.spikes[:, count_mask].sum(axis=1)

                res_elif = simulate_network(
                    self.config, self.geometry, I_ext, p.T_trial, use_elif=True)
                counts_elif[:, trial, s] = res_elif.spikes[:, count_mask].sum(axis=1)

                if trial < n_example:
                    spk_examples['std'][s].append(res_std.spikes)
                    spk_examples['elif'][s].append(res_elif.spikes)

                if trial < n_ex_raster:
                    for c in ex_candidates:
                        ex_neuron_rasters_cand[f'std_{c}'][s].append(res_std.spikes[c])
                        ex_neuron_rasters_cand[f'elif_{c}'][s].append(res_elif.spikes[c])

        print("  Simulations complete. Running analysis...")

        acc_std,  conf_std  = decode_nearest_centroid_loo(counts_std,  p.N_stim, p.N_trials)
        acc_elif, conf_elif = decode_nearest_centroid_loo(counts_elif, p.N_stim, p.N_trials)

        X_std  = np.zeros((p.N_trials * p.N_stim, self.N))
        X_elif = np.zeros((p.N_trials * p.N_stim, self.N))
        labels = np.zeros(p.N_trials * p.N_stim, dtype=int)
        for s in range(p.N_stim):
            idx = slice(s * p.N_trials, (s + 1) * p.N_trials)
            X_std[idx]  = counts_std[:, :, s].T
            X_elif[idx] = counts_elif[:, :, s].T
            labels[idx] = s

        lda_std  = decode_sklearn(X_std,  labels, method='lda', return_confusion=True)
        lda_elif = decode_sklearn(X_elif, labels, method='lda', return_confusion=True)
        svm_std  = decode_sklearn(X_std,  labels, method='svm', return_confusion=True)
        svm_elif = decode_sklearn(X_elif, labels, method='svm', return_confusion=True)

        D_std  = dprime_matrix(counts_std,  labels, p.N_stim)
        D_elif = dprime_matrix(counts_elif, labels, p.N_stim)
        mask_upper = np.triu(np.ones((p.N_stim, p.N_stim), dtype=bool), k=1)
        mean_dp_std  = D_std[mask_upper].mean()
        mean_dp_elif = D_elif[mask_upper].mean()

        nc_std,  _ = noise_correlations(counts_std,  n_pairs=500, seed=999)
        nc_elif, _ = noise_correlations(counts_elif, n_pairs=500, seed=999)
        sc_std  = signal_correlations(counts_std,  n_pairs=500, seed=999)
        sc_elif = signal_correlations(counts_elif, n_pairs=500, seed=999)

        traj = compute_pca_trajectories(
            spk_examples, t_trial, p.stim_onset, p.stim_offset,
            self.NE, n_time_bins=30, bin_width=30.0, dt=self.dt,
            n_example_trials=n_example)

        spk_per_trial_std  = X_std.sum() / (p.N_stim * p.N_trials)
        spk_per_trial_elif = X_elif.sum() / (p.N_stim * p.N_trials)
        chance = 100.0 / p.N_stim

        print("  Running rate-matched control (input-adjusted)...")
        rate_std_mean = spk_per_trial_std

        delta_I = 0.0
        if spk_per_trial_elif > rate_std_mean * 1.01:
            lo, hi = 0.0, 3.0
            for _ in range(6):
                mid = (lo + hi) / 2
                sample_spikes = 0
                sample_count = 0
                for s_test in range(min(2, p.N_stim)):
                    for t_test in range(min(5, p.N_trials)):
                        ts = 5000 + s_test * p.N_trials + t_test
                        rng_n = np.random.default_rng(ts)
                        sn = self._build_shared_noise(T_steps, p.shared_noise_amp, rng_n, 50.0)
                        rng_p = np.random.default_rng(ts + 100000)
                        I_test = np.zeros((self.N, T_steps))
                        tw = tuning[:, s_test]
                        for ti_t in range(T_steps):
                            I_test[:, ti_t] = (p.I_baseline - mid)
                            I_test[:, ti_t] += sn[:, ti_t]
                            I_test[:, ti_t] += p.private_noise_amp * rng_p.standard_normal(self.N)
                            if p.stim_onset <= t_trial[ti_t] < p.stim_offset:
                                I_test[:self.NE, ti_t] += p.stim_amp_shared
                                I_test[:self.NE, ti_t] += p.stim_amp_signal * tw
                        r = simulate_network(self.config, self.geometry, I_test, p.T_trial, use_elif=True)
                        sample_spikes += r.spikes[:, count_mask].sum()
                        sample_count += 1
                est_rate = sample_spikes / sample_count
                if est_rate > rate_std_mean:
                    lo = mid
                else:
                    hi = mid
            delta_I = (lo + hi) / 2

        def _run_rm_sweep(dI):
            counts = np.zeros((self.N, p.N_trials, p.N_stim))
            spk_ex = [[] for _ in range(p.N_stim)]
            ex_neuron_rm_cand = {c: [[] for _ in range(p.N_stim)] for c in ex_candidates}
            for s in range(p.N_stim):
                tw = tuning[:, s]
                for trial in range(p.N_trials):
                    ts = 5000 + s * p.N_trials + trial
                    rng_n = np.random.default_rng(ts)
                    sn = self._build_shared_noise(T_steps, p.shared_noise_amp, rng_n, 50.0)
                    rng_p = np.random.default_rng(ts + 100000)
                    I_ext_rm = np.zeros((self.N, T_steps))
                    for ti_t in range(T_steps):
                        I_ext_rm[:, ti_t] = (p.I_baseline - dI)
                        I_ext_rm[:, ti_t] += sn[:, ti_t]
                        I_ext_rm[:, ti_t] += p.private_noise_amp * rng_p.standard_normal(self.N)
                        if p.stim_onset <= t_trial[ti_t] < p.stim_offset:
                            I_ext_rm[:self.NE, ti_t] += p.stim_amp_shared
                            I_ext_rm[:self.NE, ti_t] += p.stim_amp_signal * tw
                    r_rm = simulate_network(self.config, self.geometry, I_ext_rm,
                                             p.T_trial, use_elif=True)
                    counts[:, trial, s] = r_rm.spikes[:, count_mask].sum(axis=1)
                    if trial < n_example:
                        spk_ex[s].append(r_rm.spikes)
                    if trial < n_ex_raster:
                        for c in ex_candidates:
                            ex_neuron_rm_cand[c][s].append(r_rm.spikes[c])
            return counts, spk_ex, ex_neuron_rm_cand

        counts_elif_rm, spk_examples_rm, ex_neuron_rm_cand = _run_rm_sweep(delta_I)
        for c in ex_candidates:
            for s in range(p.N_stim):
                key = f'rm_{c}'
                if key not in ex_neuron_rasters_cand:
                    ex_neuron_rasters_cand[key] = [[] for _ in range(p.N_stim)]
            for s in range(p.N_stim):
                ex_neuron_rasters_cand[f'rm_{c}'][s] = ex_neuron_rm_cand[c][s]
        spk_rm_pass1 = counts_elif_rm.sum() / (p.N_stim * p.N_trials)
        print(f"  RM pass 1: dI={delta_I:.3f} rate={spk_rm_pass1:.1f} "
              f"target={rate_std_mean:.1f} err={100*(spk_rm_pass1-rate_std_mean)/rate_std_mean:+.2f}%")

        prev_dI, prev_rate = 0.0, spk_per_trial_elif
        cur_dI,  cur_rate  = delta_I, spk_rm_pass1
        for refine_pass in (2, 3):
            rel_err = (cur_rate - rate_std_mean) / max(1.0, rate_std_mean)
            if abs(rel_err) <= 0.003:
                break
            slope = (cur_rate - prev_rate) / (cur_dI - prev_dI)
            if abs(slope) < 1e-3:
                break
            new_dI = max(0.0, cur_dI + (rate_std_mean - cur_rate) / slope)
            counts_elif_rm, spk_examples_rm, ex_neuron_rm_cand = _run_rm_sweep(new_dI)
            for c in ex_candidates:
                for s in range(p.N_stim):
                    ex_neuron_rasters_cand[f'rm_{c}'][s] = ex_neuron_rm_cand[c][s]
            new_rate = counts_elif_rm.sum() / (p.N_stim * p.N_trials)
            print(f"  RM pass {refine_pass}: dI={new_dI:.3f} rate={new_rate:.1f} "
                  f"err={100*(new_rate-rate_std_mean)/rate_std_mean:+.2f}%")
            prev_dI, prev_rate = cur_dI, cur_rate
            cur_dI,  cur_rate  = new_dI, new_rate
        delta_I = cur_dI

        acc_elif_rm, _ = decode_nearest_centroid_loo(counts_elif_rm, p.N_stim, p.N_trials)
        X_elif_rm = np.zeros_like(X_elif)
        for s in range(p.N_stim):
            idx = slice(s * p.N_trials, (s + 1) * p.N_trials)
            X_elif_rm[idx] = counts_elif_rm[:, :, s].T
        lda_elif_rm = decode_sklearn(X_elif_rm, labels, method='lda', return_confusion=True)
        svm_elif_rm = decode_sklearn(X_elif_rm, labels, method='svm', return_confusion=True)
        spk_rm = X_elif_rm.sum() / (p.N_stim * p.N_trials)

        D_elif_rm = dprime_matrix(counts_elif_rm, labels, p.N_stim)
        mean_dp_elif_rm = D_elif_rm[mask_upper].mean()
        nc_elif_rm, _ = noise_correlations(counts_elif_rm, n_pairs=500, seed=999)

        spk_examples_with_rm = {
            'std': spk_examples['std'],
            'elif': spk_examples_rm,
        }
        traj_rm = compute_pca_trajectories(
            spk_examples_with_rm, t_trial, p.stim_onset, p.stim_offset,
            self.NE, n_time_bins=30, bin_width=30.0, dt=self.dt,
            n_example_trials=n_example)

        efficiency_std  = acc_std  / max(1, spk_per_trial_std)
        efficiency_elif = acc_elif / max(1, spk_per_trial_elif)

        print(f"  Nearest centroid: Std={acc_std:.1f}%, eLIF={acc_elif:.1f}% (chance={chance:.0f}%)")
        print(f"  LDA (5-fold CV):  Std={lda_std['accuracy']:.1f}%, eLIF={lda_elif['accuracy']:.1f}%")
        print(f"  SVM (5-fold CV):  Std={svm_std['accuracy']:.1f}%, eLIF={svm_elif['accuracy']:.1f}%")
        print(f"  Mean d-prime:     Std={mean_dp_std:.2f}, eLIF={mean_dp_elif:.2f}")
        print(f"  Noise corr:       Std={np.nanmean(nc_std):.3f}, eLIF={np.nanmean(nc_elif):.3f}")
        print(f"  --- CONTROLS ---")
        print(f"  Rate-matched eLIF (delta_I={delta_I:.2f}): "
              f"NC={acc_elif_rm:.1f}%, LDA={lda_elif_rm['accuracy']:.1f}%, "
              f"SVM={svm_elif_rm['accuracy']:.1f}% (spk: {spk_rm:.0f} vs std {spk_per_trial_std:.0f})")
        print(f"  Rate-matched d-prime: {mean_dp_elif_rm:.2f}")
        print(f"  Rate-matched noise corr: {np.nanmean(nc_elif_rm):.3f}")
        print(f"  Efficiency (acc/spk): Std={efficiency_std:.4f}, eLIF={efficiency_elif:.4f}")

        stim_ex = 0
        raster_std  = spk_examples['std'][stim_ex][0]  if spk_examples['std'][stim_ex]  else None
        raster_elif = spk_examples['elif'][stim_ex][0] if spk_examples['elif'][stim_ex] else None
        raster_rm   = spk_examples_rm[stim_ex][0]      if spk_examples_rm[stim_ex]      else None

        cnt_window_s = (p.count_end - p.count_start) / 1000.0
        pop_rate = (counts_std[:self.NE].mean()
                    + counts_elif[:self.NE].mean()
                    + counts_elif_rm[:self.NE].mean()) / (3.0 * cnt_window_s)
        best_c = ex_candidates[0]; best_score = float('inf')
        for c in ex_candidates:
            r_std = counts_std[c].mean() / cnt_window_s
            r_rm = counts_elif_rm[c].mean() / cnt_window_s
            cell_rate = (counts_std[c].mean() + counts_elif[c].mean()
                         + counts_elif_rm[c].mean()) / (3.0 * cnt_window_s)
            score = abs(r_std - r_rm) + 0.1 * abs(cell_rate - pop_rate)
            if score < best_score:
                best_score = score; best_c = c
        ex_neuron = best_c
        print(f"  Example cell selected: #{ex_neuron} "
              f"(pref={int(np.argmax(tuning[ex_neuron]))}, "
              f"Std={(counts_std[ex_neuron].mean()/cnt_window_s):.2f} Hz, "
              f"RM={(counts_elif_rm[ex_neuron].mean()/cnt_window_s):.2f} Hz, "
              f"eLIF={(counts_elif[ex_neuron].mean()/cnt_window_s):.2f} Hz; "
              f"pop mean={pop_rate:.2f} Hz)")

        ex_neuron_rasters = {
            'std':  ex_neuron_rasters_cand[f'std_{ex_neuron}'],
            'elif': ex_neuron_rasters_cand[f'elif_{ex_neuron}'],
            'rm':   ex_neuron_rasters_cand[f'rm_{ex_neuron}'],
        }
        mean_rate_ex = {
            'std':  np.array([counts_std[ex_neuron, :, s].mean() / cnt_window_s for s in range(p.N_stim)]),
            'elif': np.array([counts_elif[ex_neuron, :, s].mean() / cnt_window_s for s in range(p.N_stim)]),
            'rm':   np.array([counts_elif_rm[ex_neuron, :, s].mean() / cnt_window_s for s in range(p.N_stim)]),
        }

        I_sweep = conn_sweep = None
        if getattr(p, 'run_sweeps', False):
            swp_NT = getattr(p, 'sweep_N_trials', 25)
            print(f"  --- SWEEPS (N_trials={swp_NT}/stim) ---")
            print("  I_baseline sweep [Std/eLIF/RM]...")
            I_sweep = self._run_sweep(
                'I_baseline', p.I_sweep_values, tuning, p.N_stim, swp_NT,
                T_steps, t_trial, count_mask, p, seed_base=20000)
            print("  Connectivity (p_conn) sweep [Std/eLIF/RM]...")
            conn_sweep = self._run_sweep(
                'p_conn', p.conn_sweep_values, tuning, p.N_stim, swp_NT,
                T_steps, t_trial, count_mask, p, seed_base=40000)

        return TaskResult(
            name='decoding',
            metrics={
                'accuracy_std': acc_std, 'accuracy_elif': acc_elif,
                'lda_accuracy_std': lda_std['accuracy'],
                'lda_accuracy_elif': lda_elif['accuracy'],
                'svm_accuracy_std': svm_std['accuracy'],
                'svm_accuracy_elif': svm_elif['accuracy'],
                'mean_dprime_std': mean_dp_std, 'mean_dprime_elif': mean_dp_elif,
                'noise_corr_std': np.nanmean(nc_std),
                'noise_corr_elif': np.nanmean(nc_elif),
                'signal_corr_std': np.nanmean(sc_std),
                'signal_corr_elif': np.nanmean(sc_elif),
                'spikes_per_trial_std': spk_per_trial_std,
                'spikes_per_trial_elif': spk_per_trial_elif,
                'chance_level': chance,
                'accuracy_elif_rate_matched': acc_elif_rm,
                'lda_accuracy_elif_rate_matched': lda_elif_rm['accuracy'],
                'svm_accuracy_elif_rate_matched': svm_elif_rm['accuracy'],
                'dprime_elif_rate_matched': mean_dp_elif_rm,
                'noise_corr_elif_rate_matched': np.nanmean(nc_elif_rm),
                'spikes_rate_matched': spk_rm,
                'delta_I_for_rate_match': delta_I,
                'efficiency_std': efficiency_std,
                'efficiency_elif': efficiency_elif,
            },
            data={
                'counts_std': counts_std, 'counts_elif': counts_elif,
                'confusion_std': conf_std, 'confusion_elif': conf_elif,
                'dprime_std': D_std, 'dprime_elif': D_elif, 'dprime_elif_rm': D_elif_rm,
                'confusion_elif_rm': decode_nearest_centroid_loo(counts_elif_rm, p.N_stim, p.N_trials)[1],
                'confusion_lda_std':  lda_std.get('confusion'),
                'confusion_lda_elif': lda_elif.get('confusion'),
                'confusion_lda_rm':   lda_elif_rm.get('confusion'),
                'confusion_svm_std':  svm_std.get('confusion'),
                'confusion_svm_elif': svm_elif.get('confusion'),
                'confusion_svm_rm':   svm_elif_rm.get('confusion'),
                'noise_corr_std': nc_std, 'noise_corr_elif': nc_elif, 'noise_corr_elif_rm': nc_elif_rm,
                'signal_corr_std_vec': sc_std, 'signal_corr_elif_vec': sc_elif,
                'X_std': X_std, 'X_elif': X_elif, 'X_elif_rm': X_elif_rm, 'labels': labels,
                'lda_std': lda_std, 'lda_elif': lda_elif,
                'trajectories': traj,
                'trajectories_rm': traj_rm,
                't_trial': t_trial, 'tuning': tuning,
                'raster_std': raster_std, 'raster_elif': raster_elif, 'raster_rm': raster_rm,
                'raster_stim_idx': stim_ex,
                'stim_onset': p.stim_onset, 'stim_offset': p.stim_offset,
                'ex_neuron_idx': int(ex_neuron),
                'ex_neuron_pref': int(neuron_preferred[ex_neuron]),
                'ex_neuron_tuning': tuning[ex_neuron].copy(),
                'ex_neuron_rasters': ex_neuron_rasters,
                'ex_neuron_mean_rate': mean_rate_ex,
                'ex_neuron_pop_rate': float(pop_rate),
                'I_sweep': I_sweep,
                'conn_sweep': conn_sweep,
            }
        )
