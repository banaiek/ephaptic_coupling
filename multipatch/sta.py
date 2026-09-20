# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import os, glob, pickle
import numpy as np
import sqlite3
from multipatch import patch_io
from neuroanalysis.miesnwb import MiesNwb
from multipatch.detect import _expt_meta, DB

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE_V = -0.020
PRE, POST = 0.015, 0.015
BL0, BL1 = -0.015, -0.008
AMP0, AMP1 = -0.002, 0.002
B_SPIKE_EXCL = -0.040
ISO_MS = 15.0


def _spike_times(v, dt, refr=0.002):
    cross = np.where((v[:-1] < SPIKE_V) & (v[1:] >= SPIKE_V))[0]
    if cross.size == 0:
        return cross
    w = max(1, int(round(0.002 / dt)))
    peaks = np.array([c + int(np.argmax(v[c:min(c + w + 1, len(v))])) for c in cross], dtype=int)
    keep = [peaks[0]]
    rs = int(refr / dt)
    for p in peaks[1:]:
        if p - keep[-1] > rs:
            keep.append(p)
    return np.array(keep)


TAU = np.arange(-PRE, POST, 1e-4)
_BLM = (TAU >= BL0) & (TAU <= BL1)


def _isolated(spkA, others, iso):
    spkA = np.sort(spkA)
    if others.size == 0:
        return spkA
    others = np.sort(others)
    pos = np.searchsorted(others, spkA)
    lo = np.where(pos > 0, spkA - others[np.clip(pos - 1, 0, len(others) - 1)], iso + 1)
    hi = np.where(pos < len(others), others[np.clip(pos, 0, len(others) - 1)] - spkA, iso + 1)
    return spkA[np.minimum(np.abs(lo), np.abs(hi)) > iso]


def _sta_from(vB, dt, spk, rng=None):
    pre, post = int(PRE / dt), int(POST / dt)
    if spk.size == 0:
        return None
    if rng is not None:
        K = 50; jlo, jhi = int(0.020 / dt), int(0.060 / dt)
        s2 = np.concatenate([spk + rng.integers(jlo, jhi, size=spk.size) * rng.choice([-1, 1], size=spk.size)
                             for _ in range(K)])
    else:
        s2 = spk
    s2 = s2[(s2 > pre) & (s2 < len(vB) - post)]
    if s2.size == 0:
        return None
    wtau = (np.arange(pre + post) - pre) * dt
    ssum = np.zeros(len(TAU)); n = 0
    for s in s2:
        w = vB[s - pre:s + post]
        if w.size != pre + post:
            continue
        wi = np.interp(TAU, wtau, w)
        ssum += wi - np.mean(wi[_BLM]); n += 1
    return (ssum, n) if n else None


def process(nwb_path, con, max_sweeps=80):
    ext_id = os.path.basename(nwb_path).replace('expt_', '').replace('.nwb', '')
    meta = _expt_meta(con, ext_id)
    if meta is None or len(meta['dev2cell']) < 2:
        return []
    nwb = MiesNwb(nwb_path)
    from collections import defaultdict
    accum = defaultdict(lambda: {'sum': None, 'n': 0, 'sh_sum': None, 'sh_n': 0, 'tau': None})
    rng = np.random.default_rng(0)
    for sr in nwb.contents[:max_sweeps]:
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
            if v.size < 2000 or not (0 < dt < 1):
                continue
            traces[meta['dev2cell'][dev]] = (v, dt)
        cells = list(traces)
        if len(cells) < 2:
            continue
        dts = [traces[c][1] for c in cells]
        if max(dts) - min(dts) > 1e-9:
            continue
        dt = dts[0]; L = min(len(traces[c][0]) for c in cells); iso = int(ISO_MS / 1000 / dt)
        spks = {c: _spike_times(traces[c][0][:L], dt) for c in cells}
        isol = {}
        for c in cells:
            others = np.concatenate([spks[o] for o in cells if o != c]) if len(cells) > 1 else np.array([])
            isol[c] = _isolated(spks[c], others, iso)
        for ci in cells:
            if isol[ci].size == 0:
                continue
            for cj in cells:
                if ci == cj:
                    continue
                key = (ci, cj) if (ci, cj) in meta['pairs'] else ((cj, ci) if (cj, ci) in meta['pairs'] else None)
                if key is None:
                    continue
                vB = traces[cj][0][:L]
                r = _sta_from(vB, dt, isol[ci])
                if r is None:
                    continue
                a = accum[(ci, cj)]
                a['sum'] = r[0] if a['sum'] is None else a['sum'] + r[0]
                a['n'] += r[1]; a['tau'] = TAU
                rs = _sta_from(vB, dt, isol[ci], rng=rng)
                if rs is not None:
                    a['sh_sum'] = rs[0] if a['sh_sum'] is None else a['sh_sum'] + rs[0]
                    a['sh_n'] += rs[1]
    out = []
    for (ci, cj), a in accum.items():
        if a['n'] < 50 or a['sum'] is None:
            continue
        key = (ci, cj) if (ci, cj) in meta['pairs'] else (cj, ci)
        dist, hs, he = meta['pairs'][key]
        sta = a['sum'] / a['n']; tau = a['tau']
        amp_win = (tau >= AMP0) & (tau <= AMP1)
        amp = float(sta[amp_win][np.argmax(np.abs(sta[amp_win]))])
        sh = (a['sh_sum'] / a['sh_n']) if a['sh_n'] else None
        sh_amp = float(sh[amp_win][np.argmax(np.abs(sh[amp_win]))]) if sh is not None else np.nan
        out.append({'expt': ext_id, 'A': ci, 'B': cj, 'distance': dist,
                    'connected': hs or he, 'has_synapse': hs, 'has_electrical': he,
                    'n_spikes': a['n'], 'amp_mV': amp * 1e3, 'shuffle_amp_mV': sh_amp * 1e3,
                    'sta_mV': (sta * 1e3).tolist(), 'tau_ms': (tau * 1e3).tolist(),
                    'sh_sta_mV': ((sh * 1e3).tolist() if sh is not None else None)})
    return out


def main():
    con = sqlite3.connect(DB)
    nwbs = sorted(glob.glob(os.path.join(HERE, 'expt_*.nwb')))
    print(f"STA over {len(nwbs)} experiments", flush=True)
    rows = []
    for p in nwbs:
        try:
            r = process(p, con); rows += r
            uc = [x for x in r if not x['connected']]
            msg = f"  {os.path.basename(p)}: {len(r)} ordered pairs"
            if uc:
                msg += f", unconn |amp| mean={np.mean([abs(x['amp_mV']) for x in uc]):.4f} mV"
            print(msg, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
    pickle.dump(rows, open(os.path.join(HERE, 'sta.pkl'), 'wb'))
    uc = [x for x in rows if not x['connected']]; cc = [x for x in rows if x['connected']]
    print(f"\nTOTAL ordered pairs: {len(rows)} (unconn {len(uc)}, conn {len(cc)})")
    if uc:
        amp = np.array([abs(x['amp_mV']) for x in uc]); d = np.array([x['distance'] for x in uc])
        sh = np.array([abs(x['shuffle_amp_mV']) for x in uc])
        print(f"  unconnected |STA amp|: mean={amp.mean():.4f} mV  shuffle={np.nanmean(sh):.4f} mV")
        for lo, hi in [(0, 60), (60, 100), (100, 150), (150, 1e9)]:
            m = (d >= lo) & (d < hi)
            if m.sum():
                print(f"   {lo}-{hi}um: n={m.sum()} |amp|={amp[m].mean():.4f} mV  shuf={np.nanmean(sh[m]):.4f}")
    if cc:
        print(f"  connected |STA amp|: mean={np.mean([abs(x['amp_mV']) for x in cc]):.4f} mV")


if __name__ == '__main__':
    main()
