"""Physical and numerical parameters for the T=0 Brownian benchmark."""

from __future__ import annotations


EV_PER_AU = 27.211397
TEMPERATURE_K = 0.0
ENERGY_00_EV = 2.0
ETA_EV = 0.0001

CENTER_FREQUENCY_EV = 0.2
LAMBDAS_MEV = (50.0, 100.0, 200.0, 400.0)
GAMMAS_EV = (0.05, 0.1, 0.2, 0.4)
STANDARD_FIGURE_GAMMAS_EV = (0.1, 0.2, 0.4)


def gamma_tag(gamma_eV: float) -> str:
    """Format Gamma without colliding 0.05 eV with the 0.1 eV case."""
    return f"{gamma_eV:.6f}".rstrip("0").rstrip(".")

# The continuous SCBA grid gives eight points across the bare 2*eta FWHM.
# Its large negative-energy padding is accompanied by an analytic Brownian
# high-frequency tail.  The audit halves the spacing and doubles the padding.
CONTINUUM_OMEGA_MIN_EV = -20.0
CONTINUUM_OMEGA_MAX_EV = 4.4
CONTINUUM_STEP_EV = 0.000025
SCBA_AUDIT_OMEGA_MIN_EV = -40.0
SCBA_AUDIT_OMEGA_MAX_EV = 4.8
SCBA_AUDIT_STEP_EV = 0.0000125
SCBA_AUDIT_CASES = ((400.0, 0.05), (400.0, 0.1), (400.0, 0.4))
MAX_SCBA_GRID_L1_CHANGE = 7.5e-4
MAX_SCBA_GRID_POINTWISE_CHANGE = 1.5e-2
MAX_SCBA_GRID_PEAK_SHIFT_EV = 1.25e-5

OUTPUT_OMEGA_MIN_EV = 1.6
OUTPUT_OMEGA_MAX_EV = 3.8
N_OUTPUT_OMEGA = 88001

MAX_ITERATIONS = 400
MIXING_G = 0.20
TOLERANCE_G = 1.0e-8

# The optical-response grid spans about 90 ps and has a 6 eV Nyquist
# frequency.  The auxiliary cumulant transform uses a four-times smaller
# internal step (24 eV range) and twice the requested time window.
TIME_STEP_AU = 14.247854151519221
N_TIME = 262144
N_RESPONSE_FFT = 524288
CUMULANT_SUBSTEPS = 4
CUMULANT_FFT_MULTIPLIER = 2
CUMULANT_AUDIT_MULTIPLIER = 4
EXACT_AUDIT_CASES = (
    (400.0, 0.05),
    (50.0, 0.1),
    (400.0, 0.2),
    (400.0, 0.4),
)
MAX_EXACT_L1_CHANGE = 2.0e-4
MAX_EXACT_POINTWISE_CHANGE = 2.0e-3
