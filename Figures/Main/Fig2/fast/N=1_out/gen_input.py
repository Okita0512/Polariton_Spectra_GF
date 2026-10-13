from scipy import integrate
import json
import numpy as np
import armadillo as arma
from bath_gen_Drude_PSD import generate

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================
#                                       Auxiliary functions     
# ==============================================================================================

def delta(m, n):
    return 1 if m == n else 0

def bose(temp, w):
    return 1 / (np.exp(w / temp) - 1)

def get_Hs(NStates, eps, wc, gc, lam):

    hams = np.zeros((NStates + 2, NStates + 2), dtype = complex)

    hams[0, 0] = wc
    hams[1, 1] = 0.0
    for i in range(2, NStates + 2):
        hams[i, 0] = gc
    for j in range(2, NStates + 2):
        hams[0, j] = gc
        hams[j, j] = eps + lam

    return hams

def get_dc(NStates):

    dc = np.zeros((NStates + 2, NStates + 2), dtype = float)
    for i in range(2, NStates + 2):
        dc[i, 1] = 1.0
    for j in range(2, NStates + 2):
        dc[1, j] = 1.0

    return dc

def get_Lb1(NStates):  # a

    Lb1 = np.zeros((NStates + 2, NStates + 2), dtype = complex)
    Lb1[1, 0] = 1.0

    return Lb1

def get_Lb2(NStates):  # a^+

    Lb2 = np.zeros((NStates + 2, NStates + 2), dtype = complex)
    Lb2[0, 1] = 1.0

    return Lb2

def get_rho0(NStates):

    rho0 = np.zeros((NStates + 2, NStates + 2), dtype = complex)
    dc = get_dc(NStates)
    rho0[1, 1] = 1.0
    
    return np.dot(dc, rho0)

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================
class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.01 * fs_to_au
    t = 40 * fs_to_au    # plateau time as 20ps for HEOM
    nt = int(t / dt) 
    nskip = 1

    lmax = 20
    nmax = 1000000
    ferr = 2.0e-05

    # ===== number of system states =====
    NStates = 1                        # total number of matter states
    eps = 2.0 / conv

    # ===== Cavity parameters =====
    omega_c = 2.0 / conv
    gc = 0.0 # 0.0681 / conv

    # ===== Drude-Lorentz model =====
    temp    = 300 / au_to_K                             # temperature
    nmod    = NStates                                   # number of dissipation modes
    nmod2   = 2                                         # number of Lindblad dissipators

    # Bath I parameters, Drude-Lorentz model
    gamma_1   = 0.0248 / conv
    lambda_1 = 0.03 / conv

    # PSD scheme
    pade    = 2                            # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 3                            # number of Pade terms

    # Bath II parameters, Brownian Oscillator
    Gamma_c = 0.00883 / conv * 10
    lbds = np.zeros((nmod2), dtype=float)
    lbds[0] = Gamma_c * (1 + bose(temp, omega_c))  # coefficient associated with Lb1
    lbds[1] = Gamma_c * bose(temp, omega_c)        # coefficient associated with Lb2, satisfying detailed balance

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    hams = get_Hs(NStates, eps, omega_c, gc, lambda_1)
    rho0 = get_rho0(NStates)
    Lb1 = get_Lb1(NStates)
    Lb2 = get_Lb2(NStates)

# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # cavity
    omega_c = parameters.omega_c
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    nmod2 = parameters.nmod2
    lambda_1 = parameters.lambda_1
    gamma_1 = parameters.gamma_1
    pade = parameters.pade
    npsd = parameters.npsd
    # system
    NStates = parameters.NStates
    gc = parameters.gc
    hams = parameters.hams
    rho0 = parameters.rho0
    # system-bath
    Lb1 = parameters.Lb1
    Lb2 = parameters.Lb2
    lbds = parameters.lbds
    # DEOM
    dt = parameters.dt
    nt = parameters.nt
    nskip = parameters.nskip
    lmax = parameters.lmax
    nmax = parameters.nmax
    ferr = parameters.ferr

# ==============================================================================================================================
    # hidx
    ini['hidx']['trun'] = 0
    ini['hidx']['lmax'] = lmax
    ini['hidx']['nmax'] = nmax
    ini['hidx']['ferr'] = ferr

	# bath PSD
    ini['bath']['temp'] = temp
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd
    ini['bath']['jomg'] = [{"jdru":[(lambda_1, gamma_1)]} for i in range(NStates)]
                                                                               
    jomg = ini['bath']['jomg']
    nind = 0
    for m in range(nmod):       # one mode is treated by PFD
        try:
            ndru = len(jomg[m]['jdru'])
        except:
            ndru = 0
        try:
            nsdr = len(jomg[m]['jsdr'])
        except:
            nsdr = 0
        nper = ndru + 2 * nsdr + npsd
        nind += nper
                                                                               
    etal, etar, etaa, expn, delr = generate (temp, npsd, pade, jomg)

    mode = np.zeros((nind), dtype = int)
    for i in range(NStates):
        mode[(npsd + 1) * i] = i
        for j in range(npsd + 1):
            mode[(npsd + 1) * i + j] = i

    arma.arma_write(mode, 'inp_mode.mat')
    arma.arma_write(delr, 'inp_delr.mat')
    arma.arma_write(etal, 'inp_etal.mat')
    arma.arma_write(etar, 'inp_etar.mat')
    arma.arma_write(etaa, 'inp_etaa.mat')
    arma.arma_write(expn, 'inp_expn.mat')

    # dissipation modes
    qmds = np.zeros((nmod, NStates + 2, NStates + 2), dtype = complex)
    for i in range(2, NStates + 2):
        qmds[i-2, i, i] = 1.0

    """
    The Lindblad dissipator here for cavity loss:

        L[Ï(t)] = lbds[0] * [L Ï(t) L^* - 0.5 {L^* L, Ï(t)}]
                + lbds[1] * [L^* Ï(t) L - 0.5 {L L^*, Ï(t)}]
    """

    # bath II with Lindblad
    lbld = np.zeros((nmod2, NStates + 2, NStates + 2), dtype = complex)
    lbld[0,:,:] = Lb1       # relaxation term
    lbld[1,:,:] = Lb2       # excitation term

    arma.arma_write (hams,ini['syst']['hamsFile'])
    arma.arma_write (qmds,ini['syst']['qmdsFile'])
    arma.arma_write (lbld,ini['syst']['lbldFile'])
    arma.arma_write (lbds,ini['syst']['lbdsFile'])
    arma.arma_write (rho0,'inp_rho0.mat')

    # real time dynamics
#    jsonInit = {"deom":ini,
#                "rhot":{
#                    "dt": dt,
#                    "nt": nt,
#                    "nk": nskip,
#					"xpflag": 1,
#					"staticErr": 0,
#                    "rho0File": "inp_rho0.mat",
#                    "sdipFile": "inp_sdip.mat",
#                    "pdipFile": "inp_pdip.mat",
#					"bdipFile": "inp_bdip.mat"
#                },
#            }
    
    # real time dynamics
    jsonInit = {"deom":ini,
                "spec":{
                    "w1max": 6.0 / conv,
                    "nt1": nt,
                    "dt": dt,
                    "nk": nskip,
                    "staticErr": 2e-05,
                    "rho0File": "inp_rho0.mat",
                    "sdipFile": "inp_sdip.mat",
                    "pdipFile": "inp_pdip.mat",
                    "bdipFile": "inp_bdip.mat"
                },
            }

# ==============================================================================================================================
# ==============================================================================================================================

    # dipoles
    sdip = get_dc(NStates)
    arma.arma_write(sdip,'inp_sdip.mat')

    pdip = np.zeros((nmod, NStates + 2, NStates + 2),dtype=float)
    pdip[0,:,:] = np.identity(NStates + 2)
    arma.arma_write(pdip,'inp_pdip.mat')

    bdip = np.zeros(nmod * len(expn),dtype=complex)
#    bdip[0]=-complex(5.00000000e-01,8.66025404e-01)
#    bdip[1]=-complex(5.00000000e-01,-8.66025404e-01)
#    bdip[2]=-complex(7.74596669e+00,0.00000000e+00)
    arma.arma_write(bdip,'inp_bdip.mat')

    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
