# Fig. S10 — additional detuning-dependent results

Plot-only rebuild from shared cached spectra and linewidth tables:

```bash
python3 Figures/SI/FigS10/plot_FigS10.py
```

The command writes `Figures/SI/FigS10/FigS10.pdf` and `validation.json`.
Panel (a) reads the nine cached `N=4` HEOM/SCBA spectrum pairs and uses the
SI-wide method encoding: light-blue HEOM shading and red dashed SCBA curves.
Panel (b) reads the shared linewidth tables, preserves all eight ensemble
sizes and their explicit main-text color mapping, and distinguishes SCBA from
HEOM/MF-HEOM with filled lines and open markers. The Hopfield-weighted result
remains a neutral-gray reference. Panel letters occupy the reserved band above
the axes. The local renderer imports the shared numerical loaders but does not
change the shared renderer used by the frozen main-text figure. It performs
only loading, normalization, and rendering; it cannot launch HEOM or SCBA
calculations.

