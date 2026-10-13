# Fig. S9 — single-molecule BA/SCBA convergence diagnostics

From the project root, rebuild the publication PDF using only the cached
HEOM, BA, SCBA, self-energy, and convergence data under `N=1/`:

```bash
python3 Figures/SI/FigS9/plot_FigS9.py
```

The legacy driver name records this figure's earlier main-text number. The
active output is `Figures/SI/FigS9/FigS9.pdf`. Its three panels show the
HEOM/BA/SCBA spectrum comparison, aligned real and imaginary frequency-domain
self-energies, and the actual whole-grid unmixed SCBA fixed-point residual.
The driver imports the SI-only typography and method encodings from
`Figures/common/si_plot_style.py`: HEOM is light-blue shading, single-shot BA
is neutral gray short-dashed, and SCBA is red long-dashed. The convergence
panel uses a thin neutral line with open markers and labels only the tolerance.
The Command-28 layout reserves a band above each axes for its panel label,
places the real-self-energy key above the stacked axes, and labels the tolerance
at the vacant left end of the threshold, away from the residual curve. The
precise endpoint is intentionally kept out of the axis and retained in the SI
caption and source text. The last panel reads column `max_residual_au` from
`N=1/scba_convergence.txt` and shows the implemented `1e-8 E_h` threshold;
the final recorded residual is `7.600615852486004e-10 E_h` at iteration 6.

The plot-only command does not run the HEOM or Green's-function calculations.
The superseded four-panel script and output are preserved under
`Figures/archive/revision26/FigS12_original_diagnostics/`.
The immediately preceding Command-27 PDF and driver are preserved under
`Figures/archive/revision27/FigS12/`.
