# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

import pickle, os, urllib.request, sys
nwb = pickle.load(open('nwb_index.pkl','rb'))
have = {f.replace('expt_','').split('.')[0] for f in os.listdir('.') if f.startswith('expt_')}
todo = [(e,u) for e,u in nwb if e.split('.')[0] not in have][:55]
print(f'downloading {len(todo)} experiments', flush=True)
for i,(e,u) in enumerate(todo):
    out=f'expt_{e.split(".")[0]}.nwb'
    try:
        urllib.request.urlretrieve(u, out)
        print(f'  [{i+1}/{len(todo)}] {out} {os.path.getsize(out)/1e9:.2f}GB', flush=True)
    except Exception as ex:
        print(f'  FAIL {e}: {ex}', flush=True)
print('DL_BATCH_DONE', flush=True)
