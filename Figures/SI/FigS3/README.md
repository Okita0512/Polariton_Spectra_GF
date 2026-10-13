# Final Fig. S3 — Brownian benchmark at zero temperature

This standalone directory compares single-shot BA and G-only SCBA with the
exact independent-boson cumulant for the underdamped-Brownian spectral density

\[
J(\omega)=
\frac{2\lambda\Gamma\omega_s^2\omega}
{(\omega^2-\omega_s^2)^2+\Gamma^2\omega^2},
\qquad
\frac{1}{\pi}\int_0^\infty\frac{J(\omega)}{\omega}\,d\omega=\lambda .
\]

The area-normalized linear figure has 16 panels using
\(\omega_s=0.2\) eV, \(\lambda=50,100,200,400\) meV, and
\(\Gamma=0.05,0.1,0.2,0.4\) eV at \(T=0\) K.  The peak-normalized and
area-normalized logarithmic variants retain the original three
\(\Gamma=0.1,0.2,0.4\) eV rows.  As in the archived zero-temperature
Drude benchmark under `bkup/Unused/FigS1_T=0`, \(E_{00}=2.0\) eV and the
retarded half-width is 0.1 meV.

No discrete bath is used. SCBA evaluates its continuous
Volterra convolution on a 0.025 meV uniform grid from -20 to 4.4 eV and
adds the first three analytic terms of the omitted Brownian tail.  The
independent audit halves the step to 0.0125 meV and expands the range to
-40--4.8 eV.

The exact cumulant is also continuous.  Its direct Ohmic transform would
converge slowly at long times, so `brownian.py` subtracts
\(h_0\omega_s^4/(\omega^2+\omega_s^2)^2\) from \(J(\omega)/\omega\).
That reference is analytic; the residual is \(O(\omega^2)\) at the origin
and \(O(\omega^{-6})\) at high frequency.  At
\(\Gamma=2\omega_s=0.4\) eV the residual vanishes identically.  The
90-ps response window and 6-eV Nyquist range are audited by doubling the
window, halving the time step, and doubling the cumulant-transform
padding.

Regenerate the publication figure and retained display variants from the
cached spectra with this plot-only command from the project root:

```bash
python3 Figures/SI/FigS3/plot_FigS3.py
```

The supplementary output is `Figures/SI/FigS3/FigS3.pdf`, generated from the
area-normalized linear variant.  The standalone peak-normalized and
area-normalized linear/logarithmic variants are also retained in this
directory.  Raw spectra, BA and SCBA
self-energies, summary metrics, and numerical convergence reports are
stored in `exact_cumulant/`, `results/`, and `convergence/`.
`generate.py` remains the full numerical regeneration/audit pipeline and is
not invoked by the plot-only command.

All four retained renderings use `Figures/common/si_plot_style.py` as of
revision 28.  The publication panel uses black-solid exact, neutral-gray
dashed BA, red-dashed SCBA, shared typography/ticks/panel letters, and no
decorative parameter banner; the physical settings remain in the caption.
Letters occupy a reserved band above the axes, only the first column carries
y tick numbers, and the reduced major-tick set avoids neighboring labels.

The completed audit gives worst area-normalized L1 changes of
\(4.34\times10^{-5}\) for the exact calculation and
\(2.72\times10^{-5}\) for the SCBA frequency grid; the refined-grid SCBA
peak shift is zero at 0.0125 meV resolution.  A separate adaptive
quadrature check of the cumulant agrees within \(6.8\times10^{-8}\).
The large SCBA--exact differences in the main panels are therefore
physical approximation errors, not unresolved bath or frequency grids:
SCBA retains a narrow threshold pole while the exact Ohmic cumulant
transfers increasing weight into a broad edge/continuum.
