# Shared detuning inputs and renderer

This directory is the one canonical source for main Fig. 3 and SI
Fig. S10. The SI wrapper and output are in `Figures/SI/FigS10/`.
The legacy top-level `Fig5/` tree was moved byte-for-byte to `cache/`; its
superseded combined renderer and README are retained in `archive/`.

Active cached inputs are:

- 13 `cache/width_N=*_Gammac=44.15mev-*.txt` linewidth tables;
- for each legacy cache key `1.8, 1.85, ..., 2.2`, the files
  `resp1st.w1`, `resp1st_im.w`, and `Aw_sc_fullpol.txt` below
  `cache/N=4_Gammac=44.15mev/`;
- the already organized endpoint values in `Figures/common/endpoint_linewidths.json`.

The cache keys in the nine spectrum directories predate the final energy
convention.  They map in order to physical cavity frequencies
`1.83, 1.88, ..., 2.23 eV`, as confirmed by each cached SCBA parameter file.

Plot-only commands, runnable from any working directory, are:

```bash
python3 Figures/Main/Fig3/plot_Fig3.py
python3 Figures/SI/FigS10/plot_FigS10.py
```

Both wrappers import `detuning_plot.py`.  The shared renderer has no solver
entry point and records an aggregate SHA-256 checksum of its 40 active inputs
in each figure's `validation.json`.

