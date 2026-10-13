# Main Figure 4: resolved vibronic spectra

`plot_Fig4.py` is a plot-only pipeline for the resolved-vibronic comparison in
main Fig. 4. It reads the validated discrete-mode caches shared with SI Figs.
S5 and S6; it does not import or launch either calculation driver.
All curves use the common retarded half-width `eta = 0.5 meV`.

The panels are:

- (a) the common outside-cavity `S=2` reference followed by 0--0 tuning at
  `N=1,2,4,8`, from S5(c,f,i,l,o), top to bottom;
- (b) the same outside-cavity reference followed by vertical tuning at
  `N=1,2,4,8`, from S6(c,f,i,l,o), top to bottom.

The outside-cavity arrays in S5(c) and S6(c) are exactly identical. The nine
unique physical cases therefore appear as ten displayed exact/SCBA pairs,
five in each panel.

Each exact spectrum is divided by its own maximum over 1.8--2.3 eV. Its SCBA
partner receives one area-matching factor over the same interval and is then
divided by that exact maximum. The SCBA peak is never independently set to
unity. A compact common row spacing is derived from the largest cavity curve
among both methods and both tuning protocols; the taller outside-cavity pair
occupies the top row independently. The same row-to-offset mapping is used in
panels (a) and (b). All cached samples in the displayed interval are plotted,
with Matplotlib path simplification disabled.

From the project root, regenerate only the vector figure and validation log:

```bash
python3 Figures/Main/Fig4/plot_Fig4.py
```

Output: `Figures/Main/Fig4/Fig4.pdf`.

Audit: `Figures/Main/Fig4/figure_validation.txt`, including source hashes,
metadata checks, display factors, common offsets, and matched-area ratios.
