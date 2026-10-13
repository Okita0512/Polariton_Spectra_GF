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

def get_Hsm(NStates, w0, lambda_1):

    hamsm = np.zeros((NStates, NStates), dtype = complex)

    hamsm[1, 1] = w0 + lambda_1

    return hamsm

def get_Qsm(NStates):

    Qsm = np.zeros((NStates, NStates), dtype = complex)

    Qsm[1, 1] = 1.0

    return Qsm

def get_rho0m(N, NStates):

    rho0m = np.zeros((NStates, NStates), dtype = complex)

    psim = np.array([1, 0], dtype = complex)
    psim = psim / np.sqrt(np.dot(psim.conjugate(), psim))
    rho0m = np.outer(psim.conjugate(), psim)
    
    return rho0m

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================
class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.01 * fs_to_au
    t = 4000 * fs_to_au    # plateau time as 20ps for HEOM
    nt = int(t / dt)
    nskip = 1

    lmaxm = 20
    nmaxm = 1000000
    ferrm = 1.0e-07
    use_pulse = True

    # ===== number of system states =====
    NStates = 2                        # total number of matter states
    N = 10000                            # number of molecules
    w0 = 2.0 / conv                     # exciton site energy

    # ===== Cavity parameters =====
    omega_c = 2.3 / conv            # cavity frequency
    g = 0.100 / conv               # keep the total Rabi splitting: 2g
    kappa = 0.02 / conv                    # cavity spectral FWHM
    alpha0 = 0.0 + 0.0j            # initial cavity amplitude
    # pulse (set use_pulse=True to enable)
    pulse_type = "chirped"            # "mono" or "chirped"
    pulse_ampl = 0.001
    pulse_freq = 2.0 / conv
    pulse_sigm = 1 * fs_to_au
    pulse_varp = 0.0
    pulse_t0   = 5 * fs_to_au

    # ===== Drude-Lorentz model =====
    tempm = 300 / au_to_K                             # temperature
    nmodm = 1                                         # number of dissipation modes
    
    # Bath I parameters, Drude-Lorentz model
    gamma_1 = 24.8 / (1000 * conv)
    lambda_1 = 30.0 / (1000 * conv)
    
    # PSD scheme
    padem    = 2                            # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsdm    = 4                            # number of Pade terms

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    hamsm = get_Hsm(NStates, w0, lambda_1)
    Qsm = get_Qsm(NStates)
    rho0m = get_rho0m(N, NStates)

# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # cavity
    omega_c = parameters.omega_c
    kappa = parameters.kappa
    alpha0 = parameters.alpha0
    # bath
    tempm = parameters.tempm
    nmodm = parameters.nmodm
    lambda_1 = parameters.lambda_1
    gamma_1 = parameters.gamma_1
    padem = parameters.padem
    npsdm = parameters.npsdm
    # system
    NStates = parameters.NStates
    hamsm = parameters.hamsm
    rho0m = parameters.rho0m
    g = parameters.g
    N = parameters.N
    use_pulse = parameters.use_pulse
    pulse_type = parameters.pulse_type
    pulse_ampl = parameters.pulse_ampl
    pulse_freq = parameters.pulse_freq
    pulse_sigm = parameters.pulse_sigm
    pulse_varp = parameters.pulse_varp
    pulse_t0   = parameters.pulse_t0
    # system-bath
    Qsm = parameters.Qsm
    # DEOM
    dt = parameters.dt
    nt = parameters.nt
    nskip = parameters.nskip
    lmaxm = parameters.lmaxm
    nmaxm = parameters.nmaxm
    ferrm = parameters.ferrm

# ==============================================================================================================================
    # hidx
    ini['hidx']['trun'] = 0
    ini['hidx']['lmaxm'] = lmaxm
    ini['hidx']['nmaxm'] = nmaxm
    ini['hidx']['ferrm'] = ferrm

	# bath II with PSD
    ini['bath']['tempm'] = tempm
    ini['bath']['nmodm'] = nmodm
    ini['bath']['padem'] = padem
    ini['bath']['npsdm'] = npsdm
    ini['bath']['jomgm'] = [{"jdru":[(lambda_1, gamma_1)]}] 
                                                                               
    jomgm = ini['bath']['jomgm']
    nindm = 0
    for m in range(nmodm):
        try:
            ndrum = len(jomgm[m]['jdru'])
        except:
            ndrum = 0
        try:
            nsdrm = len(jomgm[m]['jsdr'])
        except:
            nsdrm = 0
        nperm = ndrum + 2 * nsdrm + npsdm
        nindm += nperm
                                                                               
    etalm, etarm, etaam, expnm, delrm = generate (tempm, npsdm, padem, jomgm)
    modem = np.zeros((nindm), dtype = int)

    arma.arma_write(modem, 'inp_modem.mat')
    arma.arma_write(delrm, 'inp_delrm.mat')
    arma.arma_write(etalm, 'inp_etalm.mat')
    arma.arma_write(etarm, 'inp_etarm.mat')
    arma.arma_write(etaam, 'inp_etaam.mat')
    arma.arma_write(expnm, 'inp_expnm.mat')

    # two dissipation modes
    qmdsm = np.zeros((nmodm, NStates, NStates), dtype = complex)
    qmdsm[0,:,:] = Qsm                           # the molecular bath

    # pulse-related dipoles (zeros by default; enable use_pulse to populate rhot.pulse)
    sdip = np.zeros((NStates, NStates), dtype=float)
    sdip[0,1] = 1.0
    sdip[1,0] = 1.0
    pdip = np.zeros((nmodm, NStates, NStates), dtype=float)
    bdip = np.zeros((nindm), dtype=complex)

    # MF terms
    intm = np.zeros((2, NStates, NStates), dtype = complex)
    intm[0,1,0] = 1.0
    intm[1,0,1] = 1.0

    ini['syst']['g'] = g
    ini['syst']['wc'] = omega_c
    ini['syst']['N'] = N
    ini['syst']['kappa'] = kappa
    ini['syst']['alpha0Re'] = np.real(alpha0)
    ini['syst']['alpha0Im'] = np.imag(alpha0)
    arma.arma_write (hamsm,ini['syst']['hamsmFile'])
    arma.arma_write (qmdsm,ini['syst']['qmdsmFile'])
    arma.arma_write (intm,ini['syst']['intmFile'])
    arma.arma_write (rho0m,'inp_rho0m.mat')
    arma.arma_write (sdip,'inp_sdip.mat')
    arma.arma_write (pdip,'inp_pdip.mat')
    arma.arma_write (bdip,'inp_bdip.mat')

    rhot_block = {
        "dt": dt,
        "nt": nt,
        "nk": nskip,
        "xpflag": 1,
        "staticErr": 0,
        "rhomFile": "inp_rho0m.mat",
    }
    if use_pulse:
        rhot_block["pulse"] = {
            "type": pulse_type,
            "ampl": pulse_ampl,
            "freq": pulse_freq,
            "sigm": pulse_sigm,
            "varp": pulse_varp,
            "t0":   pulse_t0,
        }
        rhot_block["sdipFile"] = "inp_sdip.mat"
        rhot_block["pdipFile"] = "inp_pdip.mat"
        rhot_block["bdipFile"] = "inp_bdip.mat"

    jsonInit = {
        "deom": ini,
        "rhot": rhot_block,
    }
    
    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
