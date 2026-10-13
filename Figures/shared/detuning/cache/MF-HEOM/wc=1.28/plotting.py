import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from gen_input import parameters
import scipy.constants as sc

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================

data1 = np.loadtxt("prop-rhom.dat", dtype = float)
data2 = np.loadtxt("prop-alpha.dat", dtype = float)

plt.plot(data1[:,0] / fs_to_au, data1[:,1], "-", color = 'red', label = "|G>")
# plt.plot(data1[:,0] / fs_to_au, data1[:,7], "-", color = 'blue', label = "|E>")

plt.vlines(2 * np.pi / (0.2 / 0.658211951), 0, 1, color = 'green', linestyles = "--", linewidths = 2.0)

alpha = data2[:, 1] + 1.0j * data2[:, 2]
N_ph = np.abs(alpha)**2

plt.plot(data2[:,0] / fs_to_au, N_ph, "-", color = 'black', label = "N_ph")

plt.xlim(0, 200)
#plt.ylim(-2.1, 5.1)

plt.legend()

plt.show()



# ==============================================================================================
NStates = 2
Qs1 = np.zeros((NStates, NStates), dtype = complex)
# Qs1[0, 1] = 1.0
Qs1[1, 0] = 1.0
time = data1[:, 0]
NSteps = len(time)
NSkip = 50
dt = 0.01 * fs_to_au

rhot = np.zeros((NStates, NStates, NSteps), dtype = complex)
for i in range(NStates):
    for j in range(NStates):
        rhot[i, j, :] = data1[:, 2 * (NStates * i + j) + 1] + data1[:, 2 * (NStates * i + j) + 2] * 1.0j

Ct = np.zeros((NSteps), dtype = complex)
for i in range(NSteps):
    #Ct[i] = np.trace(np.matmul(Qs1, rhot[:, :, i]))
    #Ct[i] = np.trace(np.matmul(rhot[:, :, i], Qs1))
    #Ct[i] = rhot[0, 0, i]
    Ct[i] = data1[i, 1] - 1.0 

ωmax, ωmin = 24000, 0        # in cm⁻¹
ω_max, ω_min = ωmax*2*np.pi*sc.c/(10**13), ωmin*2*np.pi*sc.c/(10**13)
lenω = 10001
ω = np.linspace(ω_min, ω_max, lenω)
dω = ω[1] - ω[0]

def savStpRem(NSteps, nskip):
    if NSteps%nskip == 0:
            pl = 0
    else :
        pl = 1
    return pl

# use only short time TCF
# ctime = 40000       # 80000 will be nice
# R1t = Ct[:ctime] - np.conjugate(Ct[:ctime])
#R1t = Ct - np.conjugate(Ct)
R1t = N_ph
R1ω = np.zeros(len(ω), dtype = np.complex128)

pl = savStpRem(NSteps, NSkip)
R1dim = (NSteps//NSkip) + pl
time = np.arange(0, dt * len(R1t), dt) # time axis in au

"""First fourier transform over the t3 axis"""
for omega in range(lenω):
#    print(ω[omega], " cm-1")
    exp_fact = np.exp(1j * ω[omega] * time / fs_to_au)
    cos_fact = np.cos(np.pi * time / (2 * np.max(time)))        # smooth function
    int_func = exp_fact * R1t # * cos_fact
    R1ω[omega] = -2 * np.sum(int_func) * dω

np.savetxt("absorption.txt", R1ω)

pω = np.linspace(ωmin, ωmax, lenω)
plt.plot(pω, - R1ω.real)
# plt.plot(pω, R1ω.imag)
# plt.savefig("absorption.png", dpi = 200)
plt.show()

