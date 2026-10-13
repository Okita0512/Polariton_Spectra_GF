# Delegated to the shared continuous-convolution single-shot BA solver.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    for _root in _Path(__file__).resolve().parents:
        if (_root / "Figures" / "common" / "scba_g_only.py").is_file():
            _sys.path.insert(0, str(_root / "Figures" / "common"))
            from scba_g_only import run_legacy_ba_script as _run_ba
            _run_ba(_Path(__file__).resolve())
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

wc = 2.03 / conv                   # cavity frequency
kappa = 0 # 5 * 0.00883 / conv / 2        # cavity loss rate (imaginary part)
wc = wc - 1.0j * kappa             # include cavity loss
wc_real = np.real(wc)             # use real part for Hopfield coefficients
w0 = 2.00 / conv                   # exciton frequency
gamma = 0.0248 / conv             # characteristic frequency
ndof = 10000                      # number of bath DOFs

Nexc = 1
Nplot = 4000
w_plot = np.linspace(1.5 / conv, 2.5 / conv, Nplot)      # discretizing the polaritons
Omega_R = 0.0681 * np.sqrt(Nexc) / conv                              # value of \sqrt(N) gc
gammak = 0.0001 / conv                                    # value of 0_+

# case 1
lam = 0.030 / conv                 # reorganization energy
w0 = w0 + lam                      # further adding the reorganization energy
cb, wb = bathParam(lam, gamma, ndof)

target_w_ev = 2.0
target_w = target_w_ev / conv
target_j = int(np.argmin(np.abs(w_plot - target_w)))

Sigma = np.zeros((len(w_plot)), dtype=complex)
Gw = np.zeros((len(w_plot)), dtype=float)

sin_2_theta = 0.5 + 0.5 * (wc_real - w0) / np.sqrt((wc_real - w0) ** 2 + 4 * Omega_R ** 2)
cos_2_theta = 0.5 - 0.5 * (wc_real - w0) / np.sqrt((wc_real - w0) ** 2 + 4 * Omega_R ** 2)

sin_cos = np.sqrt(sin_2_theta * cos_2_theta)

# Calculate the non-self-consistent, single-shot Born self-energy.
for j in range(len(w_plot)):

    # UP contribution
    Sigma[j] += cos_2_theta * (
        np.sum(
            cb ** 2 * (1 + Bose(wb)) / (w_plot[j] - omega_UP(wc) - wb + 1.0j * gammak)
            + cb ** 2 * Bose(wb) / (w_plot[j] - omega_UP(wc) + wb + 1.0j * gammak)
        )
    ) / Nexc
    # LP contribution
    Sigma[j] += sin_2_theta * (
        np.sum(
            cb ** 2 * (1 + Bose(wb)) / (w_plot[j] - omega_LP(wc) - wb + 1.0j * gammak)
            + cb ** 2 * Bose(wb) / (w_plot[j] - omega_LP(wc) + wb + 1.0j * gammak)
        )
    ) / Nexc
    # DS contribution
    Sigma[j] += (
        np.sum(
            cb ** 2 * (1 + Bose(wb)) / (w_plot[j] - w0 - wb + 1.0j * gammak)
            + cb ** 2 * Bose(wb) / (w_plot[j] - w0 + wb + 1.0j * gammak)
        )
        * (Nexc - 1)
        / Nexc
    )

sigma_target = Sigma[target_j]
sigma_out = np.array([[sigma_target.real, sigma_target.imag]], dtype=float)
target_w_ev_actual = w_plot[target_j] * conv
np.savetxt(
    "Sigma_single_shot_ba_2.0eV.txt",
    sigma_out,
    header=f"sigma_real sigma_imag (w={target_w_ev_actual:.6f} eV)",
)

sigma_dist = np.zeros((Nplot, 3), dtype=float)
sigma_dist[:, 0] = w_plot
sigma_dist[:, 1] = Sigma.real
sigma_dist[:, 2] = Sigma.imag
np.savetxt(
    "Sigma_single_shot_ba_fullpol.txt",
    sigma_dist,
    header="w sigma_real sigma_imag",
)

# calculate spectrum function with full polariton Green's matrix
for j in range(len(w_plot)):
    w = w_plot[j]
    g0_inv = np.array(
        [
            [w - omega_UP(wc) + 1.0j * gammak, 0.0 + 0.0j],
            [0.0 + 0.0j, w - omega_LP(wc) + 1.0j * gammak],
        ]
    )

    sigma_mat = Sigma[j] * np.array(
        [
            [cos_2_theta, -sin_cos],
            [-sin_cos, sin_2_theta],
        ]
    )

    gmat = np.linalg.inv(g0_inv - sigma_mat)

    gw = (
        cos_2_theta * gmat[0, 0]
        + sin_2_theta * gmat[1, 1]
        - 2.0 * sin_cos * gmat[0, 1]
    )

    Gw[j] = np.imag(gw)

Aw = np.zeros((Nplot, 2), dtype=float)
Aw[:, 0] = w_plot
Aw[:, 1] = Gw

np.savetxt("Aw_single_shot_ba_fullpol.txt", Aw)
