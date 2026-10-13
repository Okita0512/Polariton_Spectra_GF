# Drude-dimer raw benchmark archive

This is the single-copy raw archive used by
`Figures/SI/FigS7/plot_FigS7.py`.

## `heom/`

The nine complete `J_*_meV` case directories were moved from
`test-2-sites/heom_reference/` without modification.  Complete directories
were retained so the solver inputs, returned time/frequency responses, logs,
case metadata, and input checksums stay together.  The active `manifest.csv`
and `heom_run_validation.txt` moved with them.

Fig. S7 reads only `case.json`, `resp1st.w1`, and `resp1st_im.w` from each
case, but the remaining files preserve numerical provenance.  Input-generation
helpers and failed historical submissions are archived under
`../archive/heom_generation/` and are not figure dependencies.

## `scba/`

The nine `dimer_scba_J_*_meV.txt` spectra, scan summary, and validation report
were moved from `test-2-sites/results/` without modification.  The spectra
contain the energy grid, bright and dark channel densities, local density of
states, and site self-energies; Fig. S7 reads the first two columns.

Do not independently normalize or rewrite these raw files.  Common-support
processing and figure-specific checks belong in
`Figures/common/dimer/drude/benchmark.py` and `Figures/SI/FigS7/results/`,
respectively.
