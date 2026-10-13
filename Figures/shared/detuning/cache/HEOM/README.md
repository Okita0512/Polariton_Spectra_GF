# Fig. 5 small-N HEOM reference data

Each `N=*/wc=*` directory contains the unmodified HEOM response files and
input provenance for an additional physical cavity frequency. The cavity
frequency in the directory name is the value used directly in the HEOM
Hamiltonian; unlike the legacy mixed SCBA/HEOM directories, no reorganization
energy offset is added to this name.

The `wc=1.28` calculations use \(g_c=0.1362/\sqrt{N}\) eV, so the total
collective coupling is fixed across \(N=1,2,4,8\). The current response files
correspond to the corrected second jobs (`7291396`--`7291399`), with four
Padé correction poles, hierarchy depth 20, and filtering threshold
\(10^{-6}\). Earlier job logs are retained only as provenance.

Run

```text
python3 Figures/common/rebuild_analysis.py
```

from the project root to merge these spectra with the retained legacy HEOM
references and rebuild the Fig. 5 linewidth and peak tables.
