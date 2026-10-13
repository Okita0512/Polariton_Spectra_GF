import numpy as np
import matplotlib.pyplot as plt

# ================= constants ====================
fs_to_au = 41.341                  # 1 fs = 41.341 a.u.
au_to_eV = 27.211397               # 1 a.u. energy = 27.211397 eV

# ================= load data ====================
# prop-rhom.dat: col 0 = time (a.u.), col 1 = μ(t)
data1 = np.loadtxt("prop-rhom.dat", dtype=float)

time_au = data1[:, 0]
mu_t = data1[:, 1].copy()

NSteps = len(time_au)
dt_au = time_au[1] - time_au[0]
time_fs = time_au / fs_to_au

# ================= preprocess signal ====================
# subtract background (DC offset)
mu_eq = mu_t[-100:].mean()
Ct = mu_t - mu_eq                  # connected signal Δμ(t)

# window to suppress FFT ringing
window = np.hanning(NSteps)
Ct *= window

# ================= plot time-domain ====================
plt.figure()
plt.plot(time_fs, Ct)
plt.xlabel("Time (fs)")
plt.ylabel(r"$\Delta \mu(t)$ (arb.)")
plt.title("Time-domain dipole response")
plt.tight_layout()
plt.show()

# ================= Fourier transform ====================
Ct_w = np.fft.fft(Ct)
freq_au = np.fft.fftfreq(NSteps, d=dt_au)
omega_au = 2.0 * np.pi * freq_au   # angular frequency (a.u.)

# keep positive frequencies
mask = omega_au > 0
omega_au = omega_au[mask]
Ct_w = Ct_w[mask]

# linear absorption ∝ ω Im[χ(ω)]
absorption = omega_au * Ct_w.imag

# convert to eV
energy_eV = omega_au * au_to_eV

# ================= save spectrum ====================
np.savetxt(
    "absorption_eV.txt",
    np.column_stack((energy_eV, absorption)),
    header="Energy_eV  Absorption(arb.units)"
)

# ================= plot spectrum ====================
plt.figure()
plt.plot(energy_eV, absorption)
plt.xlabel("Energy (eV)")
plt.ylabel("Absorption (arb. units)")
plt.title("Linear absorption spectrum")
plt.xlim(0, 3.5)   # change if needed
plt.tight_layout()
plt.show()

