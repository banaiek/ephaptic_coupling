# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pickle, time
from config import Config
import tasks.alpha_sweep as A


def main():
    t0 = time.time()
    cfg = Config.from_defaults()
    print('Alpha sweep WITH exo/exo_rm (std/elif/rm/exo/exo_rm)...', flush=True)
    res = A.run(cfg)
    pickle.dump(res, open('results/_cache/V3_DEF.pkl', 'wb'))
    print('models saved:', [k for k in res if k in ('std', 'elif', 'rm', 'exo', 'exo_rm')])
    print(f'Saved results/_cache/V3_DEF.pkl ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
