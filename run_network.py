# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, pickle, time, warnings, sys
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from geometry import NetworkGeometry
import figures.panels


def _run_transfer(cfg):
    import tasks.transfer as m
    return m.run(cfg), figures.panels.fig_transfer

def _run_spatial_freq(cfg):
    import tasks.spatial_freq as m
    return m.run(cfg), figures.panels.fig_spatial_freq

def _run_noise(cfg):
    import tasks.noise_coherence as m
    return m.run(cfg), figures.panels.fig_noise_coherence

def _run_alpha(cfg):
    import tasks.alpha_sweep as m
    return m.run(cfg), figures.panels.fig_alpha

def _run_field_mode(cfg):
    import tasks.field_mode as m
    return m.run(cfg), (lambda r, save_dir, **k: None)

def _run_decoding(cfg):
    from figures import common as figures
    from tasks.decoding import DecodingTask
    res = DecodingTask(cfg, NetworkGeometry.build(cfg)).run(seed=1000)
    return res, (lambda r, save_dir: figures.fig4_decoding(r, save_dir=save_dir))

def _run_sparse(cfg):
    from figures import common as figures
    from tasks.sparse import SparseCodingTask
    res = SparseCodingTask(cfg, NetworkGeometry.build(cfg)).run(seed=9000)
    return res, (lambda r, save_dir: figures.fig4g_sparse(r, save_dir=save_dir))


PARTS = {
    'A':   ('Spatial transfer function (A/G)', _run_transfer),
    'B':   ('Spatial-frequency discrimination (B)', _run_spatial_freq),
    'C':   ('Noise-coherence sweep (C)', _run_noise),
    'DEF': ('Alpha bifurcation (D/E/F)', _run_alpha),
    'H':   ('Endogenous vs exogenous field (Fig 2)', _run_field_mode),
    '4a':  ('Stimulus decoding', _run_decoding),
    '4f':  ('Sparse coding', _run_sparse),
}
ORDER = ['A', 'B', 'C', 'DEF', 'H', '4a', '4f']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', default=['all'])
    ap.add_argument('--alpha', type=float, default=None, help='override ephaptic alpha')
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--cache-only', action='store_true')
    args = ap.parse_args()

    parts = ORDER if args.parts == ['all'] else args.parts
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, '_cache'); os.makedirs(cache_dir, exist_ok=True)

    for key in parts:
        if key not in PARTS:
            print(f'  (skip unknown part {key})'); continue
        name, runner = PARTS[key]
        print('=' * 64); print(f'  {name}'); print('=' * 64)
        cpath = os.path.join(cache_dir, f'V3_{key}.pkl')
        t0 = time.time()
        try:
            if args.cache_only and os.path.exists(cpath):
                with open(cpath, 'rb') as f:
                    result, figfn = pickle.load(f), PARTS_FIG[key]
            else:
                cfg = Config.from_defaults()
                if args.alpha is not None:
                    cfg.elif_params.alpha = args.alpha
                result, figfn = runner(cfg)
                with open(cpath, 'wb') as f:
                    pickle.dump(result, f)
                print(f'  cached -> {cpath}')
            print(f'  done in {time.time() - t0:.1f}s')
            if not args.no_figures:
                figfn(result, save_dir=args.output_dir)
        except Exception as e:
            import traceback; print(f'  ERROR: {e}'); traceback.print_exc()
    print('\nV3 complete. Figures in', args.output_dir)


PARTS_FIG = {
    'A': figures.panels.fig_transfer, 'B': figures.panels.fig_spatial_freq,
    'C': figures.panels.fig_noise_coherence, 'DEF': figures.panels.fig_alpha,
}


if __name__ == '__main__':
    main()
