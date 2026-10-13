import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from gen_input import parameters

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341374575751                  # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================

data = np.loadtxt("./resp1st.w1", dtype = float)
data2 = np.loadtxt("./resp1st_im.w", dtype = float)
norm = np.max(data2)
plt.plot(data * conv, 100 * data2 / norm, '-', label = "Spectra")

# plt.xlim(1.0, 3.0)
plt.ylim(-5, 105)


plt.legend(frameon = False)



plt.savefig("IR.pdf")

