# Shared Drude-dimer utilities

`benchmark.py` provides the reusable, figure-independent part of the
finite-temperature Frenkel-dimer benchmark: manifest loading, metadata and
upstream-validation checks, atomic-unit conversion, common-support handling,
trapezoidal integration, and unit-area normalization.

The large raw arrays were moved, without copying, into the following shared
layout:

- `raw/heom/`: the nine complete HEOM case directories plus their active
  manifest and returned-run validation;
- `raw/scba/`: the nine SCBA spectra, scan summary, and SCBA validation.

A figure-specific manifest selects them with project-relative paths; currently
that manifest is in the publication folder at
`Figures/SI/FigS7/data_manifest.csv`. Fig. S7 has no dependency on
`test-2-sites/`.

Earlier HEOM-generation and plotting files are kept under `archive/`. They
contain no duplicate spectrum arrays used by the manuscript.

The HEOM Fourier transform has a validated, tiny negative far-tail ringing
floor.  The utility records its signed raw area and removed negative mass,
then clips negative samples before constructing the nonnegative, unit-area
spectrum used in the montage.  It never extrapolates spectral mass and only
interpolates when native source grids differ.
