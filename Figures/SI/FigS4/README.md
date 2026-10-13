# Final Fig. S4 — 0--0-resonant cavity with fixed molecular coupling

This directory reproduces `Figures/SI/FigS4/FigS4.pdf`. It uses a 20-meV
Holstein--Tavis--Cummings benchmark in a 5-by-3 panel layout, with the cavity
resonant with the 0--0 transition, `omega_c = omega_00 = 2 eV`.

The three columns use Huang--Rhys factors `S = 0.5, 1, 2`. The five rows show
the molecule outside the cavity and the cavity calculations for
`N = 1, 2, 4, 8`. Inside the cavity, the single-molecule coupling is fixed at
`g_c = 35.36 meV`; consequently `sqrt(N) g_c` grows as 35.36, 50.01, 70.72,
and 100.01 meV. Other parameters are `omega_00 = 2 eV`, `nu_0 = 20 meV`,
`eta = 0.5 meV` (`eta/nu_0 = 0.025`), and `T = 0`. The three columns therefore
have vertical transition energies `omega_0 = 2.01, 2.02, 2.04 eV`,
reorganization energies `S nu_0 = 10, 20, 40 meV`, and vibronic couplings
`nu_0 sqrt(S) = 14.142, 20.000, 28.284 meV`.

Black curves are exact results: an analytic Franck--Condon progression outside
the cavity and exact diagonalization in the permutation-symmetric bright
sector inside it. Red dashed curves are the self-consistent Born approximation
(SCBA). Each exact curve is peak-normalized, and its SCBA counterpart is area
matched over the displayed 1.8--2.3 eV interval.

The revision-28 renderer uses the SI-only conventions in
`Figures/common/si_plot_style.py`.  The axes occupy 89% of the PDF width;
panel letters and row identifiers occupy a reserved band above the axes, and
the shared labels and legend are centered on the axes grid. Three bottom-row
major ticks prevent labels from colliding between columns. Vector paths are exported without
Matplotlib path simplification.  `FigS4.png` is the matching 220-dpi preview.

The cavity exact calculation uses a total vibrational cutoff `Q = 28`, checked
against `Q = 30`. The `N = 1, 2` matrices are diagonalized densely; `N = 4, 8`
use 5000 projected Lanczos steps, checked against 4000. The SCBA grid has a
0.05-meV spacing, so the 20-meV frequency shift is exactly 400 grid points.

Replot the committed caches only (no solver run) from the project root with:

```bash
python3 Figures/SI/FigS4/plot_FigS4.py
```

To regenerate the expensive caches explicitly, run
`python3 Figures/SI/FigS4/calculate_FigS4.py` first. Numerical checks,
summaries, and independent panel archives are stored under
`Figures/shared/discrete_mode/zero_zero/`; the local
`figure_validation.txt` records plot validation. Retired assets are preserved
under `Figures/archive/revision23/`.
