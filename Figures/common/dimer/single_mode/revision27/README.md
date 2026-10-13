# Revision-27 focused dimer audit

This directory is separate from the base data in `../data/eta0p005/`. Its
script performs source-level checks used by the merged SI Fig. S6 without
overwriting any CSV or NPZ file there.

Run from the project root:

```bash
python3 Figures/common/dimer/single_mode/revision27/audit_dimer_revision27.py
```

Fixed physical settings are `|J|=25 meV`, `T=0`, and `eta/nu0=0.005`.
The base grid is `x in [-32,24]`, `Delta x=0.002`, with `w` evaluated
on `[-18,18]`.  The audit repeats selected points at `Delta x=0.001` and
`0.0005`, shifts the numerical grid by half a base-grid cell, raises the ED
cutoff from `N_vib=10` to `12`, checks `[-16,16]`, `[-18,18]`, and `[-20,20]`
metric supports, and calculates intermediate `r` values.

Outputs:

- `data/refined_r_points.csv`: 34 new raw calculations used to refine the
  plotted Fig. S6 lines; no interpolated values;
- `data/convergence.csv`: eight cases and six numerical variants per case;
- `data/peak_basin_metrics.csv` and `data/cdf_region_metrics.csv`: peak,
  region-weight, and exact CDF-integral diagnostics near `r~2`;
- `data/parity_checks.csv` and `data/even_branch_tracking.csv`: exact
  commutator, parity-block, source-overlap, and opposite-parity crossing tests;
- `data/*.npz`: fine-grid diagnostic spectra and the CDF difference entering
  the stored Wasserstein definition;
- `plots/r2_spectral_cdf_diagnostics.pdf`: audit-only vector plot, not an SI
  publication figure;
- `data/summary.json` and `data/pipeline_audit.json`: machine-readable settings,
  maxima, hashes, sorting/duplicate/overlap checks, and output routing.
- `data/diagnosis.txt`: concise quantitative diagnosis and explicit remaining
  diagrammatic/selection-rule uncertainty.

The audit reproduces the stored `w` values at the selected points to better
than `9e-14`. The base mesh and cut-table hashes remain
`6cd6eb...9772` and `ac34e9...e88`, respectively.
