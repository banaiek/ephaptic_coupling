# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

from dataclasses import dataclass, field


@dataclass
class NeuronParams:
    tau_m: float = 10.0
    V_rest: float = -60.0
    V_th: float = -50.0
    V_reset: float = -60.0
    tau_ref: float = 2.0


@dataclass
class SynapticParams:
    tau_e: float = 3.0
    tau_i: float = 8.0
    J_ee: float = 4.0
    J_ie: float = -8.0
    w_ee: float = 0.4
    w_ee_std: float = 0.1
    w_ie: float = -1.5
    w_ie_std: float = 0.3


@dataclass
class NetworkParams:
    N: int = 500
    NE: int = 400
    NI: int = 100
    p_conn: float = 0.15


@dataclass
class GeometryParams:
    sigma_eph: float = 0.15
    sigma_input: float = 0.15
    sigma_syn: float = 0.0
    K_sources: int = 50
    layout: str = 'halton'
    eph_kernel: str = 'gaussian_norm'
    field_population: str = 'split_EI'
    coulomb_eps: float = 0.02
    eph_sparsify: float = 0.02


@dataclass
class ELIFParams:
    alpha: float = 0.2
    use_geometry: bool = True
    field_spike: bool = False
    spike_gain: float = 1.0
    spike_peak: float = 30.0
    spike_dur_ms: float = 1.0


@dataclass
class SimParams:
    dt: float = 0.1


@dataclass
class DecodingParams:
    N_stim: int = 10
    N_trials: int = 100
    T_trial: float = 400.0
    stim_onset: float = 100.0
    stim_offset: float = 300.0
    count_start: float = 120.0
    count_end: float = 300.0
    tuning_width: float = 8.0
    I_baseline: float = 10.0
    stim_amp_shared: float = 2.0
    stim_amp_signal: float = 2.0
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 4.0
    run_sweeps: bool = False
    sweep_N_trials: int = 100
    I_sweep_values: list = field(default_factory=lambda: [10.0 + 0.5 * i for i in range(11)])
    conn_sweep_values: list = field(default_factory=lambda: [0.05 + 0.0125 * i for i in range(13)])


@dataclass
class AttentionParams:
    attn_boost: float = 2.0
    target_amp: float = 3.0
    N_trials: int = 80


@dataclass
class WorkingMemoryParams:
    T_trial: float = 1000.0
    cue_onset: float = 100.0
    cue_offset: float = 300.0
    cue_amp: float = 5.0
    delay_ends: list = field(default_factory=lambda: [400, 600, 800])
    N_trials: int = 40
    N_stim: int = 4


@dataclass
class DetectionParams:
    T_trial: float = 500.0
    signal_onset: float = 150.0
    signal_offset: float = 350.0
    N_signal_neurons: int = 100
    snr_levels: list = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    N_trials: int = 40
    noise_amp: float = 4.0


@dataclass
class TemporalParams:
    T_trial: float = 300.0
    N_patterns: int = 4
    N_trials: int = 80
    pattern_duration: float = 100.0
    pattern_start: float = 150.0
    noise_levels: list = field(default_factory=lambda: [i * 0.1 for i in range(11)])
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 4.0
    baseline: float = 10.0
    signal_amp: float = 1.0
    sparsity: float = 0.25


@dataclass
class SpatiotemporalParams:
    T_trial: float = 350.0
    stim_start: float = 100.0
    stim_end: float = 300.0
    period: float = 200.0
    N_stim: int = 10
    N_trials: int = 80
    n_selective_stim: int = 2
    delta: float = 12.0
    sigma: float = 40.0
    bump_amp: float = 2.5
    distractor_amp: float = 2.0
    baseline: float = 8.0
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 10.0
    sigma_syn: float = 0.0
    decode_bin_ms: float = 10.0
    decode_pca: int = 40
    n_example_stim: int = 4


@dataclass
class SparseParams:
    T_trial: float = 150.0
    N_patterns: int = 4
    N_trials: int = 50
    duration: float = 80.0
    start: float = 35.0
    noise_levels: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    sparsity_levels: list = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25])
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 4.0
    baseline: float = 10.0
    signal_amp: float = 1.0


@dataclass
class TransferParams:
    T_trial: float = 600.0
    settle: float = 200.0
    modes: list = field(default_factory=lambda: list(range(0, 13)))
    drive_amp: float = 2.0
    baseline: float = 11.0
    n_trials: int = 12
    private_noise_amp: float = 1.0
    kernels: list = field(default_factory=lambda: ['gaussian_norm', 'coulomb'])


@dataclass
class SpatialFreqParams:
    T_trial: float = 400.0
    stim_onset: float = 100.0
    stim_offset: float = 300.0
    count_start: float = 120.0
    count_end: float = 300.0
    modes: list = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 10, 12])
    N_stim: int = 8
    N_trials: int = 40
    n_seeds: int = 5
    alpha: float = 0.2
    baseline: float = 10.0
    stim_amp: float = 0.15
    shared_noise_amp: float = 2.0
    private_noise_amp: float = 9.0


@dataclass
class NoiseCoherenceParams:
    T_trial: float = 400.0
    stim_onset: float = 100.0
    stim_offset: float = 300.0
    count_start: float = 120.0
    count_end: float = 300.0
    sigma_noise_levels: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    N_stim: int = 8
    N_trials: int = 20
    n_seeds: int = 100
    alpha: float = 0.2
    baseline: float = 10.0
    stim_amp: float = 0.15
    noise_amp: float = 9.0
    signal_mode: int = 2


@dataclass
class AlphaSweepParams:
    alphas: list = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9])
    n_seeds: int = 15
    T_trial: float = 400.0
    stim_onset: float = 100.0
    stim_offset: float = 300.0
    count_start: float = 120.0
    count_end: float = 300.0
    N_stim: int = 10
    N_trials: int = 50
    baseline: float = 10.0
    stim_amp_signal: float = 2.0
    stim_amp_shared: float = 2.0
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 4.0
    tuning_width: float = 8.0


@dataclass
class FieldModeParams:
    alphas: list = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9])
    n_seeds: int = 12
    exo_field_amp: float = 8.0
    src_pos: tuple = (0.5, 0.5)
    src_lambda: float = 0.5
    T_trial: float = 400.0
    stim_onset: float = 100.0
    stim_offset: float = 300.0
    count_start: float = 120.0
    count_end: float = 300.0
    N_stim: int = 4
    N_trials: int = 15
    baseline: float = 10.0
    stim_amp_signal: float = 2.0
    stim_amp_shared: float = 2.0
    shared_noise_amp: float = 4.0
    private_noise_amp: float = 4.0
    tuning_width: float = 8.0


@dataclass
class Config:
    neuron: NeuronParams = field(default_factory=NeuronParams)
    synapse: SynapticParams = field(default_factory=SynapticParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    elif_params: ELIFParams = field(default_factory=ELIFParams)
    sim: SimParams = field(default_factory=SimParams)
    decoding: DecodingParams = field(default_factory=DecodingParams)
    attention: AttentionParams = field(default_factory=AttentionParams)
    wm: WorkingMemoryParams = field(default_factory=WorkingMemoryParams)
    detection: DetectionParams = field(default_factory=DetectionParams)
    temporal: TemporalParams = field(default_factory=TemporalParams)
    spatiotemporal: SpatiotemporalParams = field(default_factory=SpatiotemporalParams)
    sparse: SparseParams = field(default_factory=SparseParams)
    transfer: TransferParams = field(default_factory=TransferParams)
    spatial_freq: SpatialFreqParams = field(default_factory=SpatialFreqParams)
    noise_coherence: NoiseCoherenceParams = field(default_factory=NoiseCoherenceParams)
    alpha_sweep: AlphaSweepParams = field(default_factory=AlphaSweepParams)
    field_mode: FieldModeParams = field(default_factory=FieldModeParams)
    seed: int = 2024

    @classmethod
    def from_defaults(cls):
        return cls()
