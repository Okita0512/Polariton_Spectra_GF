# Main Figure 2: collective linewidth narrowing

This folder owns the two fixed-`g_c` spectral panels and the fixed-collective-
splitting LP linewidth panel used as main Fig. 2. Panels (a,b) retain their
legacy cached HEOM/SCBA spectra. Panel (c) uses the command30 SCBA production
tables: `eta=0.01 meV` for `N<=64` and `eta=0.001 meV` for `N=128` in both
baths. The latter solutions use a `0.0005 meV` internal grid.

Plot-only rebuild from the project root:

```bash
python3 Figures/Main/Fig2/plot_Fig2.py
```

The script writes `Fig2.pdf` and `figure_validation.txt`. It checks the retained
fit windows/exponents and the area matching of every displayed spectral pair;
it launches no solver or benchmark calculation. The reproducible SCBA driver,
full complex endpoint solutions, comparison tables, and diagnostic plots are
under `../../common/validation/eta001_scaling/`, with the driver at
`../../common/validate_eta001_scaling.py` and the audit at
`../../common/audit_scaling_eta30.py`.
