# Self-generated ephaptic fields improve the neural population code

Simulation and analysis code for Banaie Boroujeni and Kastner.

Neurons share a resistive extracellular medium, so the field an active
population generates feeds back onto its own membranes. This code builds
spiking networks with and without that feedback, derives the closed-form
response of the coupled network, and tests the pair-level prediction against
multi-patch cortical recordings.

## Layout

| path | contents |
| --- | --- |
| `config.py`, `geometry.py`, `simulation.py`, `field.py`, `analysis.py` | network model, coupling operator, integrator, population measures |
| `tasks/` | decoding, sparse-coding, transfer-function and sweep protocols |
| `network/` | mechanism experiments: replay, gap junctions, source models, heterogeneity, two-column mesoscopic simulation |
| `multipatch/` | spike detection, spike-triggered averages and figures for the cortical recordings |
| `figures/` | figure rendering |
| `generate/` | scripts that build the cached source data |
| `data/`, `source_data/` | cached simulation output, one file per experiment |

## Running

```
pip install -r requirements.txt
python run_network.py --parts all
python -m network.run all
python -m figures.make_all
python -m figures.mechanism all
```

Figures read cached results from `data/` and `source_data/`, so they can be
redrawn without repeating the simulations. Deleting a cache file and re-running
the corresponding task regenerates it.

## Multi-patch recordings

The recordings are the Allen Institute for Brain Science Synaptic Physiology
dataset, release 2.1, from
https://portal.brain-map.org/explore/connectivity/synaptic-physiology. The raw
NWB files run to tens of gigabytes and are not distributed here.
`multipatch/download.py` retrieves them, and `detect.py`, `sta.py` and
`example.py` reduce them to the three cached files in `multipatch/`, which are
what the figures use.
