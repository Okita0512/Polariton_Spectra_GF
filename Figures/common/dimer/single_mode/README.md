# Shared single-mode dimer calculation

This directory is the common numerical source for the merged SI Fig. S6. Its
publication folder is `Figures/SI/FigS6/`. The former separate figure is no
longer published and is preserved in the revision-28 archive.

- `workflow.py` contains shared validation, Wasserstein, archive-loading, and
  plotting utilities.
- `data/source_manifest.json` is the project-relative routing authority for
  the active arrays.
- `generate_eta0p005.py` reproducibly generates the active figure profile.
- `validate_targeted_convergence.py` reruns the requested ED-cutoff and
  grid-halving checks without changing the stored spectra.
- `revision27/` contains the additional midpoint-in-r, `Delta x=0.001` and
  `0.0005`, half-cell grid-phase, cutoff, support, CDF/peak-weight, and exact
  parity audit. Its 34 calculated refined-`r` points are merged into the
  Fig. S6 scan curves; the base arrays remain unchanged.
- `data/eta0p005/` contains the fresh 2,214-point finite-J mesh, seven J=0
  baselines, eight A-D archives, hashes, and numerical validation.
- `source/` preserves the broader historical `eta/nu0=0.01` scan
  and the common ED/SCBA solver implementations reused by the fresh profile.
- `../notes/` contains the current theory note source and PDF.

The historical calculation subtree is not duplicated or overwritten.
`Figures/SI/FigS6/manifest.csv` records hashes and project-relative data paths.
The pre-merge S6/S7 files are preserved under
`Figures/archive/revision28/old_S6_S7/`.

From the project root, reproduce or resume the active calculation with:

```bash
python3 Figures/common/dimer/single_mode/generate_eta0p005.py --workers 4
```

The figure driver is:

```bash
python3 Figures/SI/FigS6/plot_FigS6.py
```

The driver checks the 2,248 calculated scan points, fixed `eta/nu0=0.005`,
raw and normalized spectral areas, scan/spectrum distance agreement, and input
hashes before writing the figure and its sidecars.

## Numerical profile

The active arrays use `x=(E-E_B)/nu0` on the internal support `[-32,24]`
with `Delta x=0.002` (28,001 points); Wasserstein metrics use the common
analysis support `[-18,18]` (18,001 points). The finite-J curves in Fig. S6
use the factorized ED cutoff `N_vib=10`, i.e. a
`2*N_vib=20` difference-mode matrix. The seven J=0 inset baselines use
`N_vib=10,10,10,12,16,20,24` for `S=0.02,0.05,0.10,0.25,0.50,1,2`.

At `T=0`, the stored SCBA calculation uses the causal ascending-frequency sweep, so all
stored cases finish in one pass and do not use mixing. The guarded iterative
fallback has absolute/relative/Green-change tolerances `1e-9 eV`, `1e-8`,
and `1e-8`, a 5,000-iteration limit, and adaptive mixing initially `0.08`
within `[0.002,0.40]`: it halves the mixing if the residual grows by more
than `1.10` and multiplies it by `1.05` if the residual falls below `0.70`
of its previous value. The retarded check requires nonpositive
`Im Sigma` within `5e-10 eV`; the profile additionally checks fixed-point
residual, channel closure, spectral positivity, and raw-area mismatch
(`<=3e-6`).

For `w`, ED and SCBA are restricted to the identical `x in [-18,18]`
support and independently normalized by trapezoidal integration. Weighted
one-dimensional Wasserstein evaluation uses trapezoidal point weights and is
equivalent to integrating the absolute CDF difference. No smoothing or peak
normalization enters `w`. Worker exceptions abort generation, and validation
requires exactly 2,214 finite-J rows plus seven J=0 baselines, so failed or
nonconverged points cannot be silently omitted.

Run the compact convergence audit with:

```bash
python3 Figures/common/dimer/single_mode/validate_targeted_convergence.py
```

It checks `J=0, S=0.10`, the measured positive-J maximum
`(S,r)=(0.10,0.800)`, and the off-integer positive-J reference
`(S,r)=(0.10,1.350)` using `N_vib=12` and `Delta x=0.001`. Results are in
`data/eta0p005/targeted_convergence.{csv,json,txt}`.
