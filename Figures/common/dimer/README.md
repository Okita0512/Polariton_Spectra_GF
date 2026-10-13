# Shared dimer calculations

This subtree holds reusable code and the single-copy numerical inputs behind
Supplementary Figs. S6--S8. Their publication-matched folders are
`Figures/SI/FigS6/`, `Figures/SI/FigS7/`, and `Figures/SI/FigS8/`.

## Layout

- `single_mode/workflow.py`: common archive loading, numerical checks,
  Wasserstein evaluation, and plotting support for the merged Fig. S6.
- `single_mode/data/source_manifest.json`: project-relative routing for the
  active `eta/nu0=0.005` figure profile.
- `single_mode/generate_eta0p005.py` and `single_mode/data/eta0p005/`: stored
  figure-profile generator, 2,214-point mesh, seven baselines, eight A-D
  archives, and validation.
- `single_mode/source/`: common ED/BA/SCBA solvers plus the preserved broader
  historical `eta/nu0=0.01` scan and its audits.
- `notes/single_mode_dimer_scba_theory.{tex,pdf}`: derivation note used for the
  continued-fraction comparison.
- `drude/benchmark.py`: reusable HEOM/SCBA loading, conversion, common-support,
  clipping-accounting, and area-normalization checks for Fig. S7.
- `drude/raw/heom/` and `drude/raw/scba/`: the sole active copies of the nine
  signed-J data sets.
- `drude/archive/`: earlier HEOM-generation and plotting files; it contains no
  duplicate spectra used by the manuscript.

Figure-local drivers, processed tables, manifests, and validation reports stay
in `Figures/SI/FigS6/` and `Figures/SI/FigS7/`; these render Figs. S6 and S7.
The vector diagram source for Fig. S8 is in `Figures/SI/FigS8/`. The
former separate scan-and-cut figure is unpublished;
its pre-merge files are preserved under
`Figures/archive/revision28/old_S6_S7/`. All paths are project relative.

## Maintained commands

Run from the project root:

```bash
python3 Figures/common/dimer/single_mode/generate_eta0p005.py --workers 4
python3 Figures/common/dimer/single_mode/validate_targeted_convergence.py
python3 Figures/SI/FigS6/plot_FigS6.py
python3 Figures/SI/FigS7/plot_FigS7.py
bash Figures/SI/FigS8/build_FigS8.sh
```

The first command reproducibly generates or resumes the active single-mode
profile. The second performs the three-case `N_vib=10 -> 12` and
`Delta x=0.002 -> 0.001` audit without overwriting the stored spectra. The
figure drivers use those checked arrays and do not retune physical
parameters.
