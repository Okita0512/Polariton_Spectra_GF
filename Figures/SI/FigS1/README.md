# Final Fig. S1 — Fan/SCBA diagrammatics

`FigS1.tex` is the editable TikZ body input directly by `SI.tex`. Rebuild it
as part of the supplement from the project root:

```bash
latexmk -pdf SI.tex
```

It has no numerical cache and does not launch a calculation.
The explanatory verdicts use neutral black text; blue is reserved for the
phonon propagator and its $D_0$ label.
Revision 28 gives this inline diagram the same sans-serif typography and exact
phonon-blue color as the dimer-specific diagram; the topology is unchanged.
The immediately preceding TikZ source and a baseline rendered SI page are
preserved under `Figures/archive/revision27/FigS2_inline/`.
