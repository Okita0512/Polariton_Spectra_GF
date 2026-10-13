# Shared discrete-mode spectra

This is the single active cache root used by main Fig. 4 and final SI
Figs. S4/S5. Their publication wrappers and outputs are in `Figures/SI/FigS4/`
and `Figures/SI/FigS5/`.
It contains two validated grids:

- `zero_zero/`: cavity tuned to `omega_00=2.00 eV` (former `FigS5/` cache);
- `vertical/`: cavity tuned to `omega_0=omega_00+S nu_0` (former `FigS6/`
  cache).

Each grid contains 15 `.npz` cases: outside-cavity and `N=1,2,4,8`, each at
`S=0.5,1,2`. The active generation uses `eta=0.5 meV`, a 0.05-meV SCBA grid,
and 5000/4000 production/check Lanczos lengths. The original migration
digests and the hashes of the current regenerated caches are recorded in
`cache_manifest.json`. The per-grid `summary.csv` and calculation validation
accompany the caches.

Figure-specific plotting and calculation wrappers must resolve this location
from the project root. Active figure commands must not read the archived
Command-22 Figure 4 assets.
