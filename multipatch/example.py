# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, pickle, glob
import numpy as np
import sqlite3
from multipatch import patch_io
from neuroanalysis.miesnwb import MiesNwb
from multipatch.detect import _expt_meta, DB, SPIKE_V

HERE = os.path.dirname(os.path.abspath(__file__))
DS = 25


def _scan_nwb(path, con, best):
    ext = os.path.basename(path).replace('expt_', '').replace('.nwb', '')
    meta = _expt_meta(con, ext)
    if meta is None:
        return best
    pos = meta['pos']
    try:
        nwb = MiesNwb(path)
    except Exception:
        return best
    for sr in nwb.contents[:120]:
        ic = {}
        for rec in sr.recordings:
            if getattr(rec, 'clamp_mode', None) != 'ic':
                continue
            dev = rec.device_id
            if dev not in meta['dev2cell']:
                continue
            try:
                pri = rec['primary']; v = np.asarray(pri.data, float); dt = float(pri.dt)
            except Exception:
                continue
            if v.size > 5000 and 0 < dt < 1:
                ic[meta['dev2cell'][dev]] = (v, dt)
        if len(ic) < 4:
            continue
        nsp = {c: int(np.sum((v[:-1] < SPIKE_V) & (v[1:] >= SPIKE_V))) for c, (v, dt) in ic.items()}
        src = max(nsp, key=nsp.get)
        if nsp[src] < 5 or src not in pos:
            continue
        nbrs = [c for c in ic if c != src and c in pos]
        dists = [float(np.linalg.norm(pos[c] - pos[src]) * 1e6) for c in nbrs]
        if len(nbrs) < 3 or not dists or min(dists) > 80:
            continue
        score = nsp[src] * len(nbrs) * (1 + 3 * (min(dists) < 60))
        if best is None or score > best['score']:
            v, dt = ic[src]
            nb = []
            for c, d in zip(nbrs, dists):
                vc, _ = ic[c]
                nb.append({'cell': c, 'distance': d, 'v': (vc[::DS] * 1e3)})
            nb.sort(key=lambda z: z['distance'])
            spk_idx = np.where((v[:-1] < SPIKE_V) & (v[1:] >= SPIKE_V))[0]
            wpk = max(1, int(round(0.002 / dt)))
            if spk_idx.size:
                spk_idx = np.array([c + int(np.argmax(v[c:min(c + wpk + 1, len(v))])) for c in spk_idx], dtype=int)
            spk = spk_idx * dt * 1e3
            pa, po = int(0.003 / dt), int(0.005 / dt)
            si = spk_idx[(spk_idx > pa) & (spk_idx < len(v) - po)]
            ap = np.mean([v[s - pa:s + po] for s in si], 0) * 1e3 if si.size else np.zeros(pa + po)
            ap_tau = (np.arange(pa + po) - pa) * dt * 1e3
            best = {'score': score, 'src': src, 'ext': ext, 'dt': dt * DS,
                    't_ms': np.arange(len(v[::DS])) * dt * DS * 1e3,
                    'src_v': v[::DS] * 1e3, 'src_spikes_ms': spk, 'neighbours': nb,
                    'ap_mV': ap, 'ap_tau_ms': ap_tau, 'n_src_spikes': int(si.size)}
    return best


def main():
    con = sqlite3.connect(DB)
    best = None
    nwbs = sorted(glob.glob(os.path.join(HERE, 'expt_*.nwb')), key=os.path.getsize, reverse=True)[:14]
    for p in nwbs:
        best = _scan_nwb(p, con, best)
        if best:
            print(f"  scanned {os.path.basename(p)}; best so far: src dists "
                  f"{[round(n['distance']) for n in best['neighbours']]}um", flush=True)
    if best is None:
        print('no suitable sweep found'); return
    pickle.dump(best, open(os.path.join(HERE, 'example.pkl'), 'wb'))
    print(f"\nsaved example: expt {best['ext']} source {best['src']}, {len(best['neighbours'])} neighbours, "
          f"{len(best['src_spikes_ms'])} source spikes, distances "
          f"{[round(n['distance']) for n in best['neighbours']]} um", flush=True)


if __name__ == '__main__':
    main()
