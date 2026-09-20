# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pickle, time
from config import Config
from geometry import NetworkGeometry
from tasks.mechanism import MechanismAnalysis


def main():
    os.makedirs('source_data', exist_ok=True)
    cfg = Config.from_defaults()
    cfg.geometry.eph_sparsify = 0.0
    geom = NetworkGeometry.build(cfg)
    t0 = time.time()
    print("Mechanism (3-way Std/eLIF/RM) source...", flush=True)
    res = MechanismAnalysis(cfg, geom).run(seed=2024)
    with open('source_data/mech_source.pkl', 'wb') as f:
        pickle.dump(res.data, f)
    print(f"Saved source_data/mech_source.pkl ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
