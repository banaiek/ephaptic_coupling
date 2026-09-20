# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, sys, glob, pickle
import numpy as np
from scipy.signal import butter, filtfilt
import sqlite3
from multipatch import patch_io
from neuroanalysis.miesnwb import MiesNwb

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, 'synphys_r2.1_medium.sqlite')
SPIKE_V = -0.020
HP_HZ = 2.0
DS = 10


def _expt_meta(con, ext_id):
    cur = con.cursor()
    row = cur.execute("select id from experiment where ext_id like ?", (ext_id.split('.')[0] + '%',)).fetchone()
    if row is None:
        return None
    eid = row[0]
    import json
    dev2cell, pos = {}, {}
    rows = cur.execute("select c.id, e.device_id, c.position from cell c "
                       "join electrode e on c.electrode_id=e.id where c.experiment_id=?", (eid,)).fetchall()
    for cid, dev, p in rows:
        if dev is None or p is None:
            continue
        try:
            xyz = json.loads(p) if isinstance(p, str) else p
            dev2cell[dev] = cid; pos[cid] = np.array(xyz, float)
        except Exception:
            pass
    pairs = {}
    for pre, post, dist, hs, he in cur.execute(
            "select pre_cell_id, post_cell_id, distance, has_synapse, has_electrical "
            "from pair where experiment_id=?", (eid,)).fetchall():
        if dist is not None:
            pairs[(pre, post)] = (dist * 1e6, bool(hs), bool(he))
    return {'eid': eid, 'dev2cell': dev2cell, 'pos': pos, 'pairs': pairs}


def _clean_trace(v, dt):
    v = np.asarray(v, float)
    fs = 1.0 / dt
    b, a = butter(2, HP_HZ / (fs / 2), 'high')
    vf = filtfilt(b, a, v)
    spk = np.where(v > SPIKE_V)[0]
    if spk.size:
        w = int(round(0.003 / dt))
        m = np.zeros(len(v), bool)
        for s in spk:
            m[max(0, s - w):s + w] = True
        vf[m] = np.nan
    return vf[::DS]


def process_experiment(nwb_path, con, max_sweeps=60):
    ext_id = os.path.basename(nwb_path).replace('expt_', '').replace('.nwb', '')
    meta = _expt_meta(con, ext_id)
    if meta is None or len(meta['dev2cell']) < 2:
        return []
    nwb = MiesNwb(nwb_path)
    recs = nwb.contents[:max_sweeps]
    from collections import defaultdict
    acc = defaultdict(list)
    for sr in recs:
        traces = {}
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
            if v.size < 1000 or not np.isfinite(dt) or dt <= 0:
                continue
            traces[meta['dev2cell'][dev]] = _clean_trace(v, dt)
        cells = list(traces)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                ci, cj = cells[i], cells[j]
                key = (ci, cj) if (ci, cj) in meta['pairs'] else (cj, ci)
                if key not in meta['pairs']:
                    continue
                a, b = traces[ci], traces[cj]
                n = min(len(a), len(b)); a, b = a[:n], b[:n]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 500:
                    continue
                r = np.corrcoef(a[ok], b[ok])[0, 1]
                if np.isfinite(r):
                    acc[key].append(r)
    out = []
    for key, rs in acc.items():
        dist, hs, he = meta['pairs'][key]
        out.append({'expt': ext_id, 'pair': key, 'distance': dist,
                    'connected': hs or he, 'has_synapse': hs, 'has_electrical': he,
                    'corr': float(np.mean(rs)), 'n_sweeps': len(rs),
                    'traces_cells': key})
    return out


def main():
    con = sqlite3.connect(DB)
    nwbs = sorted(glob.glob(os.path.join(HERE, 'expt_*.nwb')))
    print(f"processing {len(nwbs)} experiments", flush=True)
    allrows = []
    for p in nwbs:
        try:
            rows = process_experiment(p, con)
            allrows += rows
            uc = [r for r in rows if not r['connected']]
            print(f"  {os.path.basename(p)}: {len(rows)} pairs ({len(uc)} unconnected), "
                  f"mean unconn corr={np.mean([r['corr'] for r in uc]):.4f}" if uc else f"  {os.path.basename(p)}: {len(rows)} pairs", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAIL {p}: {e}", flush=True)
    with open(os.path.join(HERE, 'pairs.pkl'), 'wb') as f:
        pickle.dump(allrows, f)
    uc = [r for r in allrows if not r['connected']]
    cc = [r for r in allrows if r['connected']]
    print(f"\nTOTAL: {len(allrows)} pairs; unconnected={len(uc)}, connected={len(cc)}")
    if uc:
        d = np.array([r['distance'] for r in uc]); c = np.array([r['corr'] for r in uc])
        print(f"  unconnected corr: mean={c.mean():.4f} median={np.median(c):.4f}")
        near = c[d < 80]; far = c[d > 150]
        print(f"  near(<80um) n={near.size} mean={near.mean():.4f}  far(>150um) n={far.size} mean={far.mean():.4f}")
    if cc:
        print(f"  connected corr: mean={np.mean([r['corr'] for r in cc]):.4f}")


if __name__ == '__main__':
    main()
