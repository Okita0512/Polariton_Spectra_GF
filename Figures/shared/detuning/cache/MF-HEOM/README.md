# Detuning-analysis MF-HEOM reference data

Each `wc=*` directory is named by its physical cavity frequency in eV and
contains the unmodified MF-HEOM propagation output and input provenance.
The processed `absorption_pulse_2eV.txt` spectrum stores
`-Im[P(omega)/E(omega)]` in eV\(^{-1}\), without the additional frequency
factor used for absorbed power. It is generated with:

```bash
python3 Figures/common/process_mf_heom.py Figures/shared/detuning/cache/MF-HEOM Figures/shared/detuning/cache/MF-HEOM
```

`Figures/common/rebuild_analysis.py` combines these organized reference runs with
the retained collective-limit MF-HEOM spectra when rebuilding the shared
detuning linewidth and peak tables used by main Fig. 3 and SI Fig. S10
(`FigS10`).
