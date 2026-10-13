Created on 10/12/2026

# Polariton spectra from Green's-function calculations

This repository contains the main-text and Supporting Information figure
folders, numerical drivers, processed data, validation records, and
figure-reproduction code for the Green's-function polariton-spectra
calculations.

All published external figures can be rebuilt from the committed inputs
without rerunning the expensive solvers. The SCBA, exact-diagonalization,
cumulant, spectral-analysis, and validation code is also included so that the
numerical results can be regenerated or audited independently.

## Repository layout

- `Figures/Main/Fig1`--`Fig4`: main-text figures and figure-local documentation.
- `Figures/SI/FigS1`--`FigS12`: Supporting Information figures and documentation.
- `Figures/common`: shared SCBA, linewidth, normalization, dimer, plotting, and
  validation code.
- `Figures/shared`: compact data used by more than one figure.
- `MANIFEST.sha256`: checksums for the public reproducibility package.

The editable TikZ sources for main Fig. 1 and SI Fig. S1 are retained in their
numbered owner folders. The other publication figures are stored there as
vector PDFs together with their plotting inputs.

## Reference environment

The figures were validated with Python 3.8.10 and the package versions in
`requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A LaTeX installation with TikZ is required for the vector diagram sources.

## Rebuild every figure

From the repository root, run:

```bash
python3 Figures/common/generate_all.py --figures-only
```

This is a plot-only operation. It reads the committed processed inputs and
does not launch SCBA, HEOM, exact-diagonalization, or cumulant calculations.
Each numbered folder also has a local `README.md` describing its parameters,
inputs, validation, and individual rebuild command.

To rebuild only the analysis tables before plotting, use:

```bash
python3 Figures/common/rebuild_analysis.py
python3 Figures/common/generate_all.py --figures-only
```

## Regenerate numerical results

The shared continuous-Drude SCBA implementation is
`Figures/common/scba_g_only.py`. The documented full-SCBA production route is:

```bash
python3 Figures/common/generate_all.py
```

The fixed-collective-coupling scaling audit, including the `N=128`,
`eta=0.001 meV` calculation, can be reproduced with:

```bash
python3 Figures/common/validate_eta001_scaling.py eta0001
python3 Figures/common/validate_eta001_scaling.py eta0001_grid
python3 Figures/common/validate_eta001_scaling.py eta00005
python3 Figures/common/audit_scaling_eta30.py
```

The expensive single-mode and bright--dark calculations have explicit drivers
in their figure folders. See the local READMEs for exact commands and numerical
settings.

## HEOM data and storage policy

Processed HEOM spectra consumed by the plotting scripts are committed. The
external HEOM propagation program itself is not distributed here. Multi-GB raw
trajectories (`prop-rhom.dat`), resumable solver checkpoints
(`internal_solution.npz`), scheduler logs, Python caches, and superseded figure
archives are omitted because they are not required by the plot-only rebuild.
The retained validation tables record the numerical settings and checks for
the public data.

## Numerical convention

The molecular 0--0 transition is denoted by `omega_00`, and the vertical
electronic energy is `omega_0 = omega_00 + lambda`. The production SCBA uses a
frequency-coupled self-energy on the full grid with zero-padded linear FFT
convolution for the continuous Drude bath. Additional implementation and
validation details are in `Figures/common/docs/SCBA_MODEL.md` and
`Figures/common/docs/SCBA_GENERATION_REPORT.md`.
