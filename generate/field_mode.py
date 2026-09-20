# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pickle, time
from config import Config
import tasks.field_mode as m


def main():
    t0 = time.time()
    cfg = Config.from_defaults()
    print('field_mode (endo/exo) alphas =', list(cfg.field_mode.alphas), flush=True)
    res = m.run(cfg)
    pickle.dump(res, open('results/_cache/V3_H.pkl', 'wb'))
    print(f'Saved results/_cache/V3_H.pkl ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
