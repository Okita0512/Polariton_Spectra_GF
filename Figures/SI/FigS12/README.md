# Fig. S12 — collective-limit benchmark

Regenerate the publication PDF from the project root with:

```bash
python3 Figures/SI/FigS12/plot_FigS12.py
```

This plot-only command reads the cached spectra and linewidth tables in this
folder and writes `Figures/SI/FigS12/FigS12.pdf`. The Command-28 renderer places
panel letters in the shared band above each axes. The stacked spectra use
light-blue MF-HEOM shading and red dashed SCBA curves rather than parameter
rainbow colors, and the ordinate explicitly reads `Normalized absorption +
offset`. LP and UP remain blue and red in the linewidth panels; solid curves,
open markers, and light dashed guides distinguish SCBA, MF-HEOM, and the
Hopfield-weighted reference. All linewidth columns and guides are displayed in
meV by exact multiplication of the retained eV values by 1000. The PDF driver
uses the shared Helvetica/STIX settings and does not emit Type-3 glyphs.
The immediately preceding plot driver and PDF are preserved under
`Figures/archive/revision27/FigS1_collective/`.

The MF-HEOM files store the susceptibility spectral function
`-Im[P(omega)/E(omega)]` in eV\(^{-1}\), reconstructed with the propagation's
`exp(-i omega t)` discrete Fourier convention. Rebuild them from the retained
propagations before rebuilding linewidth tables or the figure:

```bash
python3 Figures/common/process_mf_heom.py Figures/SI/FigS12 Figures/SI/FigS12
python3 Figures/common/rebuild_analysis.py
```

The retained MF-HEOM and SCBA data use \(N=10^4\), collective coupling
\(G=\sqrt{N}g_c=0.1\) eV, and cavity spectral FWHM \(\kappa=20\) meV.
Thus the resonant collective Rabi splitting shown in the SI is
\(\Omega_R=2G=0.2\) eV. The diagonal Hopfield-weighted guides use the
canonical uncoupled HEOM molecular linewidth stored under `common/`.
