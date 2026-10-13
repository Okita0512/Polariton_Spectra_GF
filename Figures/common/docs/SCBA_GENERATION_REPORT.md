# G-only SCBA generation report

Updated: 2026-10-04

## Scope

- `Figures/SI/FigS9/`: 1 SCBA case for SI Fig. S9
- `Figures/Main/Fig2/`: 35 SCBA cases
- `Figures/shared/detuning/cache/`: 176 SCBA cases supporting main Fig. 3
  and SI Fig. S10 (`FigS10`)
- `Figures/SI/FigS12/`: 21 SCBA cases for SI Fig. S12
- Total: 233 frequency-coupled G-only SCBA calculations
- `Figures/SI/FigS2/`: 12 continuous-Drude single-shot BA and G-only SCBA
  cases, plus the numerically identical zero-density G--D control

Every active `EGF_sc_fullpol.py` retains its original case parameters and
delegates to `Figures/common/scba_g_only.py`. The obsolete pointwise fixed-point loops
have been removed from the active sources.

## Numerical procedure

The production self-energy is

\[
\Sigma^R(\omega)=\sum_qc_q^2\left[
(\overline{n}(\omega_q)+1)G_{\mathrm{loc}}^R(\omega-\omega_q)
+\overline{n}(\omega_q)G_{\mathrm{loc}}^R(\omega+\omega_q)\right],
\]

where \(\overline{n}(\omega)=(e^{\beta\omega}-1)^{-1}\).

with

\[
G_{\mathrm{loc}}^R=(G_B^R+(N-1)G_D^R)/N.
\]

All frequencies are iterated together. The bare thermal phonon propagator is
held fixed, as required in the zero-exciton linear-response reference state.
The production calculation uses the continuous Drude integral as a
zero-padded linear FFT convolution.  The global grid spans -1--4 eV, with
0.05 meV spacing for the 0.1 meV retarded half-width cases and 0.01 meV
spacing for the 0.01 meV cases.  The omitted Drude emission tail is restored
analytically.  The maximum residual tolerance is
\(10^{-8}E_{\rm h}=2.72\times10^{-7}\) eV.
The former 1000-mode tangent grid remains selectable as an alternative
quadrature and regression check.

## Validation

- All 233 shared calculations converged in 5--20 global iterations.
- Largest final residual: 0.00026407 meV.
- All 12 standalone Fig. S2 G-only calculations converged with the more
  conservative fixed mixing; the G--D controls reproduce them to within
  \(9.46\times10^{-9}\) after area normalization.
- All retarded spectra and self-energies pass the causality sign check.
- All 11 derived SCBA linewidth tables are finite; no peak widths are
  missing.
- The uncoupled-molecule limit reproduces the standalone Frenkel G-only
  benchmark.
- Half-spacing convolution checks over four representative cases changed no
  peak position on the plotting grid, changed FWHM values by at most
  \(7.10\times10^{-6}\) meV, and changed area-normalized spectra by at most
  \(4.55\times10^{-7}\) in integrated \(L^1\) distance.
- The alternative \(N_b=2000\), \(N_\omega=8000\) tangent-grid check changed
  peak positions by at most 0.025 meV, FWHM values by at most 0.039 meV, and
  area-normalized spectra by at most \(3.94\times10^{-3}\) in integrated
  \(L^1\) distance.

Detailed production convergence is in
`Figures/common/validation/scba_generation_summary.csv`, and the refinement and
alternative-quadrature comparisons are in
`Figures/common/validation/scba_convolution_validation.csv`.

## Generated figures

- `Figures/Main/Fig2/Fig2.pdf`
- `Figures/Main/Fig3/Fig3.pdf`
- `Figures/Main/Fig4/Fig4.pdf`
- Fig. S1 is drawn directly in `SI.tex` from `Figures/SI/FigS1/FigS1.tex`.
- `Figures/SI/FigS2/FigS2.pdf`
- `Figures/SI/FigS3/FigS3.pdf`
- `Figures/SI/FigS4/FigS4.pdf`
- `Figures/SI/FigS5/FigS5.pdf`
- `Figures/SI/FigS6/FigS6.pdf`
- `Figures/SI/FigS7/FigS7.pdf`
- `Figures/SI/FigS8/FigS8.pdf`
- `Figures/SI/FigS9/FigS9.pdf`
- `Figures/SI/FigS10/FigS10.pdf`
- `Figures/SI/FigS11/FigS11.pdf`
- `Figures/SI/FigS12/FigS12.pdf`

The publication plotting scripts use the names `SCBA` and `Single-shot BA`.
Their PDFs embed Helvetica for ordinary text.

## Backup

The complete pre-cleanup state is preserved outside this project directory in
`../test_new_pre_cleanup_2026-07-27.tar.gz`.

SHA-256:
`18c703a47a513d411ac3259fa30536815393c9873ad9363f6f9a6ea4c3247cfe`
