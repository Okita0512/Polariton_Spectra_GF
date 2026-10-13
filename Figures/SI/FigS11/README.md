# Fig. S11 — bright–dark near-degeneracy diagnostic

`diagnose_scba.py` is an independent numerical driver for the anomalous
exciton-rich UP linewidth in the detuning analysis. It imports the shared
frequency-coupled SCBA kernel and the common direct-FWHM routine, but does not
import the detuning plotting or linewidth-table drivers.

The horizontal isolated-molecule reference is read from the shared canonical
endpoint source `Figures/common/endpoint_linewidths.json`. Its 88.146 meV value is
audited from the raw strict-\(g_c=0\) cache and is the same disconnected SCBA
endpoint used by the detuning analysis. The underlying direct-FWHM result retained in
`data/isolated_molecule.csv` is 88.146370607 meV.

Regenerate the publication PDF from the retained CSV tables with this
plot-only command from the project root:

```bash
python3 Figures/SI/FigS11/plot_FigS11.py
```

The plotting command writes the publication output to
`Figures/SI/FigS11/FigS11.pdf`. The former root-level
`Figures/FigS6.pdf` is preserved outside the active numbered tree as
`Figures/archive/revision23/previous_outputs/SI_FigS6_legacy.pdf`.

The calculation includes:

- the strict \(g_c=0\) normalization and multiplicity test;
- the UP–dark gap scan for \(N=1\) through the \(N=10^4\) collective-limit
  implementation;
- nonlinear fixed-point and bidirectional-continuation tests;
- separate bright and dark self-energy contributions;
- single-shot BA and frozen-molecular-dark controls;
- temperature scans; and
- bath, frequency-grid, padding, broadening, tolerance, and mixing checks.

Machine-readable results are written to `data/`. Compressed converged
self-energies are retained in `cache/` so interrupted or repeated scans can
resume without recomputing completed cases. Use `--force` only when the
diagnostic equations or numerical settings have changed.

`data/correlation_summary.csv` records the Pearson product-moment
correlation coefficient `r_P` between the linewidth excess and excess dark
damping for every displayed \(N=2\)--64 scan. Each coefficient uses all 12
retained gaps from 0.02 to 100 meV; subtracting the decoupled dark baseline
does not change the coefficient.

For the molecular endpoint, the independent route uses a 1000-point tangent
bath, a 4000-point internal self-energy grid over 0.8--3.2 eV, a 4000-point
output grid over 0.9--3.15 eV, and 0.01 meV numerical broadening. The SCBA
stopping tolerance is \(10^{-8}\) Hartree
(\(2.7211397\times10^{-7}\) eV); the archive labels both units explicitly.
The direct-FWHM analysis uses the unsmoothed 1.65--2.35 eV line support and linear
half-height interpolation. The FFT-convolution calculation archived for
Fig. 5 gives 88.349285 meV; its 0.202915 meV (0.2302%) difference from the
independent result is retained in `Figures/common/endpoint_linewidths.json` as a
cross-route diagnostic rather than as a second manuscript molecular limit.

The publication figure has three panels. Panel (a) shows the UP linewidth
and isolated-molecule limit over the full gap scan for the seven retained
series \(N=1,2,4,8,16,64,\infty\). It imports the explicit \(N\)-color
dictionary from `Figures/common/si_plot_style.py`, which exactly reproduces
the main-Fig.-3/current-Fig.-S10 mapping; omitted \(N\) values therefore
cannot shift the colors. The former vertical detuning guide and its sentence
were removed because the SI text already gives the corresponding
\(\delta_{+D}\simeq24\) meV connection.

Panel (b) reuses the same \(N=4\) and 64 colors and distinguishes dark (solid,
circles) from bright (dashed, squares) channels. Panel (c) uses the shared
ordered cool-to-warm temperature palette for 0, 100, 200, and 300 K. All
markers are the actual sampled CSV points, with no interpolation or smoothing.
The Command-28 layout places all panel letters in bands above the axes and
moves the compact keys above the high curves. The pale \(N=1\) and 2 series
retain their established colors but use darker marker edges for legibility.
`plot_FigS11.py` reads the existing CSV files only and performs no numerical
recalculation.

The expensive `diagnose_scba.py` calculation driver and its resumable caches
are retained for provenance, but the plot-only command does not invoke them.
The immediately preceding Command-27 PDF and plot driver are preserved under
`Figures/archive/revision27/FigS13/`.
