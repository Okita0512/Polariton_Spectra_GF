# Main Fig. 3: lower-polariton detuning dependence

Plot-only rebuild from the shared cached linewidth tables:

```bash
python3 Figures/Main/Fig3/plot_Fig3.py
```

The command writes `Figures/Main/Fig3/Fig3.pdf` and `validation.json`.
It loads `Figures/shared/detuning/cache/width_*.txt`; it does not invoke a
spectral solver or recalculate linewidths.

