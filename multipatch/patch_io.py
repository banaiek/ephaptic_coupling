# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

from collections import OrderedDict
import numpy as np
from neuroanalysis.miesnwb import MiesNwb


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _notebook(self):
    if self._notebook is None:
        sweep_entries = OrderedDict(); tp_entries = []
        device = list(self.hdf['general/devices'].keys())[0].split('_', 1)[-1]
        ln = self.hdf['general']['labnotebook'][device]
        nb_keys = [_dec(k) for k in ln['numericalKeys'][0]]
        nb_fields = OrderedDict([(k, i) for i, k in enumerate(nb_keys)])
        nb = np.array(ln['numericalValues'])
        est = nb_fields.get('EntrySourceType', None)
        it = iter(range(nb.shape[0]))
        for i in it:
            rec = nb[i]; sweep_num = rec[0, 0]
            is_tp = False; is_sweep = False
            if est is not None and not np.isnan(rec[est][0]):
                is_sweep = rec[est][0] == 0; is_tp = not is_sweep
            elif i < nb.shape[0] - 1:
                if any(np.isfinite(rec[nb_fields['TP Peak Resistance']])):
                    if any(np.isfinite(nb[i + 1][nb_fields['TP Pulse Duration']])):
                        next(it); is_tp = True
                if not is_tp:
                    is_sweep = np.isfinite(sweep_num)
            if is_tp:
                rec = np.array(rec); next(it); rec2 = np.array(nb[i + 1])
                m = ~np.isnan(rec2); rec[m] = rec2[m]; tp_entries.append(rec)
            elif is_sweep:
                sn = int(sweep_num)
                if sn not in sweep_entries:
                    sweep_entries[sn] = np.array(rec)
                else:
                    m = ~np.isnan(rec); sweep_entries[sn][m] = rec[m]
        for swid, entry in sweep_entries.items():
            m = ~np.isnan(entry[:, 8]); entry[m] = entry[:, 8:9][m]
            entry[:4] = entry[:4, 0:1]
            for i, k in enumerate(nb_keys):
                if k.startswith('Async AD '):
                    entry[i] = entry[i, 0]
            meta = []
            for i in range(entry.shape[1]):
                tm = entry[:, i]
                meta.append(OrderedDict([(nb_keys[j], (None if np.isnan(tm[j]) else tm[j]))
                                         for j in range(len(nb_keys))]))
            sweep_entries[swid] = meta
        text_keys = [_dec(k) for k in ln['textualKeys'][0]]
        text_fields = OrderedDict([(k, i) for i, k in enumerate(text_keys)])
        _tv = ln['textualValues']
        if hasattr(_tv, 'asstr'):
            text_nb = _tv.asstr()[()]
        else:
            text_nb = np.asarray(_tv[()] if hasattr(_tv, '__getitem__') else _tv)
        if text_nb.dtype.kind == 'S' or (text_nb.size and isinstance(text_nb.flat[0], (bytes, bytearray))):
            text_nb = np.vectorize(_dec)(text_nb)
        est2 = text_fields.get('EntrySourceType', None)
        for rec in text_nb:
            if est2 is None:
                stype = 0
            else:
                try:
                    stype = int(rec[est2, 0])
                except (ValueError, TypeError):
                    continue
            if stype != 0:
                continue
            try:
                sweep_id = int(rec[0, 0])
            except (ValueError, TypeError):
                continue
            se = sweep_entries.get(sweep_id)
            if se is None:
                continue
            for k, i in text_fields.items():
                for j, val in enumerate(rec[i, :-1]):
                    if k in se[j] or val == '':
                        continue
                    se[j][k] = val
        self._notebook = sweep_entries
        self._tp_notebook = tp_entries
        self._notebook_keys = nb_fields
        self._tp_entries = None
    return self._notebook


MiesNwb.notebook = _notebook
