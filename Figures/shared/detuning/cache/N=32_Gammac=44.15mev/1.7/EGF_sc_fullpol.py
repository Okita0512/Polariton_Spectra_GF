# Delegated to the shared frequency-coupled G-only SCBA solver.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    for _root in _Path(__file__).resolve().parents:
        if (_root / "Figures" / "common" / "scba_g_only.py").is_file():
            _sys.path.insert(0, str(_root / "Figures" / "common"))
            from scba_g_only import run_legacy_script as _run_scba
            _run_scba(_Path(__file__).resolve())
            raise SystemExit
    raise RuntimeError("Could not locate Figures/common/scba_g_only.py")

import numpy as np

conv = 27.211397                            # 1 a.u. = 27.211397 eV
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
mm2au = 18897261.257078                     # 1 mm = 18897261.257078 a.u.
c2au = 137.0359895

def Bose(x):
    beta = au_to_K / 300
    return 1 / (np.exp(beta * x) - 1)

def bathParam(lam, wc, ndof):

    c = np.zeros((ndof))
    w = np.zeros((ndof))
    for d in range(ndof):
        w[d] = wc * np.tan(np.pi * (1 - (d + 1) / (ndof + 1)) / 2)
        c[d] = np.sqrt(lam * w[d] / (ndof + 1))

    return c, w

def omega_LP(wc):
    return 0.5 * (w0 + wc) - 0.5 * np.sqrt((wc - w0) ** 2 + 4 * Omega_R ** 2)

def omega_UP(wc):
    return 0.5 * (w0 + wc) + 0.5 * np.sqrt((wc - w0) ** 2 + 4 * Omega_R ** 2)

wc = 1.73 / conv                   # cavity frequency
kappa = 5 * 0.00883 / conv / 2        # cavity loss rate (imaginary part)
wc = wc - 1.0j * kappa             # include cavity loss
wc_real = np.real(wc)             # use real part for Hopfield coefficients
w0 = 2.00 / conv                   # exciton frequency
gamma = 0.0248 / conv             # characteristic frequency
ndof = 1000                       # number of bath DOFs

Nexc = 32
Nplot = 4000
w_plot = np.linspace(1.5 / conv, 2.5 / conv, Nplot)      # discretizing the polaritons
# Omega_R = 0.0681 * np.sqrt(Nexc) / conv                              # value of \sqrt(N) gc
Omega_R = 0.0681 * 2 / conv                              # value of \sqrt(N) gc
gammak = 0.00001 / conv                                    # value of 0_+

# case 1
lam = 0.030 / conv                 # reorganization energy
w0 = w0 + lam                      # further adding the reorganization energy

# The shared solver reads the physical parameters above and performs the global SCBA iteration.
