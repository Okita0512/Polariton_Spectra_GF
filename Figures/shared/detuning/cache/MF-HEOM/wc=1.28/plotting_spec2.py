import numpy as np
import matplotlib.pyplot as plt

# ================= constants ====================
fs_to_au = 41.341          # 1 fs = 41.341 a.u.
au_to_eV = 27.211397       # 1 a.u. energy = 27.211397 eV

# ================= load data ====================
# prop-rhom.dat: col0 = time (a.u.), then pairs of Re/Im for rho_ij
data1 = np.loadtxt("prop-rhom.dat", dtype=float)

time_au = data1[:, 0]
NSteps = len(time_au)
dt_au = time_au[1] - time_au[0]
time_fs = time_au / fs_to_au

# ================= reconstruct density matrix ====================
NStates = 2   # adjust if needed

rhot = np.zeros((NStates, NStates, NSteps), dtype=complex)
for i in range(NStates):
    for j in range(NStates):
        re_col = 2 * (NStates * i + j) + 1
        im_col = re_col + 1
        rhot[i, j, :] = data1[:, re_col] + 1j * data1[:, im_col]

# ================= dipole operator & μ(t) ====================
# μ = |g><e| + |e><g| in basis {|g>, |e>}
mu_op = np.zeros((NStates, NStates), dtype=complex)
mu_op[0, 1] = 1.0
mu_op[1, 0] = 1.0

# μ(t) = Tr[ μ ρ(t) ]
mu_t = np.einsum("ij,ijt->t", mu_op, rhot).real

# ================= remove background & window ====================
n_tail = max(NSteps // 5, 1)
mu_eq = mu_t[-n_tail:].mean()
Ct = mu_t - mu_eq                       # Δμ(t)

window = np.hanning(NSteps)
Ct_win = Ct * window

# (optional) extra exponential damping to smooth peaks:
eta = 0.0       # in a.u.; try e.g. 5e-4 for slight broadening
if eta != 0.0:
    Ct_win *= np.exp(-eta * time_au)

# ================= plot time-domain signal ====================
plt.figure()
plt.plot(time_fs, Ct_win)
plt.xlabel("Time (fs)")
plt.ylabel(r"$\Delta \mu(t)$ (arb. units)")
plt.title("Time-domain dipole response")
plt.tight_layout()
plt.show()

# ================= Fourier transform (one-sided) ====================
Ct_w = np.fft.rfft(Ct_win)                 # complex spectrum
freq_au = np.fft.rfftfreq(NSteps, d=dt_au)
omega_au = 2.0 * np.pi * freq_au           # angular frequency (a.u.)

# Linear absorption ~ Re[χ(ω)]
absorption = Ct_w.real    # overall sign is arbitrary; multiply by -1 if you prefer

energy_eV = omega_au * au_to_eV

# ================= save & plot spectrum ====================
np.savetxt(
    "absorption_eV_from_mu_real.txt",
    np.column_stack((energy_eV, absorption)),
    header="Energy_eV  Absorption(arb.units)"
)

plt.figure()
plt.plot(energy_eV, absorption)
plt.xlabel("Energy (eV)")
plt.ylabel("Absorption (arb. units)")
plt.title("Linear absorption spectrum from μ(t)")
plt.xlim(1.8, 2.2)  # adjust as needed
plt.tight_layout()
plt.show()
