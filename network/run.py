# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import time
import pickle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'data')

SEEDS = (4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249)
SEEDS_WIDE = tuple(4242 + i for i in range(16))


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + '.pkl')
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)
    print(f'  wrote {path}', flush=True)


def mechanism():
    from network import mechanism
    _save('mechanism', mechanism.run(SEEDS_WIDE, nproc=8))


def apfield():
    from network import ap_source
    _save('apfield', ap_source.run(SEEDS_WIDE, nproc=8))


def heterogeneity():
    from network import heterogeneity
    _save('heterogeneity', heterogeneity.run())


def gap():
    from network import gap_junctions
    _save('gap', gap_junctions.run(SEEDS_WIDE, nproc=8))


def meso():
    from network import mesoscopic
    _save('mesoscopic', {'columns': mesoscopic.run(SEEDS[:4]),
                         'uniformity': mesoscopic.dipole_uniformity(),
                         'local_cv': mesoscopic.local_field_dispersion(),
                         'beta_scale': {R: mesoscopic.beta_scale(R)
                                        for R in (300.0, 1000.0, 3000.0)},
                         'slow': mesoscopic.run_slow()})


def operating():
    from network import operating_point
    _save('operating', operating_point.run(SEEDS_WIDE[:12]))
def fieldpop():
    from network import field_population
    _save('fieldpop', field_population.run(SEEDS_WIDE, nproc=8))


def pair():
    from network import pair
    _save('pair', {'example': pair.pair_example(),
                   'sta': pair.spike_aligned(T_ms=4000.0),
                   'biophysical': pair.biophysical_reference(),
                   'distance': pair.distance_profile(),
                   'measured': pair.measured_excess_sta(),
                   'dipole_cmf': pair.dipole_close_minus_far()})


def rmclean():
    from network import replay
    _save('replay_rm_clean', replay.run(SEEDS_WIDE, nproc=8))


def entrain():
    from network import entrainment
    _save('meso_entrain', entrainment.run(SEEDS_WIDE, nproc=8))


TASKS = {'mechanism': mechanism, 'entrain': entrain, 'rmclean': rmclean, 'apfield': apfield,
         'heterogeneity': heterogeneity, 'gap': gap,
         'meso': meso, 'pair': pair, 'operating': operating,
         'fieldpop': fieldpop}


def main(names):
    if not names or names == ['all']:
        names = list(TASKS)
    for name in names:
        if name not in TASKS:
            print(f'unknown experiment: {name}')
            continue
        t0 = time.time()
        print(f'[{time.strftime("%H:%M:%S")}] {name} ...', flush=True)
        TASKS[name]()
        print(f'[{time.strftime("%H:%M:%S")}] {name} done in {time.time() - t0:.0f}s',
              flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])