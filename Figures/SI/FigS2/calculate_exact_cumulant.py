#!/usr/bin/env python3
"""Exact independent-boson cumulant spectra for the FigS2 Drude baths."""

from __future__ import annotations

from pathlib import Path

import numpy as np


EV_PER_AU = 27.211397
K_PER_AU = 3.1577464e5
TEMPERATURE_K = 300.0
ENERGY_00_EV = 2.0
LAMBDAS_MEV = (15.0, 30.0, 45.0, 60.0)
CUTOFFS_MEV = (12.4, 24.8, 37.2)
N_MATSUBARA = 2000


def drude_lineshape(
    time: np.ndarray,
    lam: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    r"""Return the exact Gaussian-bath lineshape function ``g(t)``.

    The convention is
    ``J(w) = 2 lambda gamma w / (w^2 + gamma^2)`` and
    ``g(t) = (1/pi) int J(nu)/nu^2 [(1-cos(nu*t))*coth(beta*nu/2)
    + i*(sin(nu*t)-nu*t)] dnu``.  The response carrier is the vertical
    energy ``E_00 + lambda``; the linear imaginary term restores the
    relaxed 0--0 origin.
    """
    c0 = lam * gamma * (1.0 / np.tan(0.5 * beta * gamma) - 1.0j)
    result = c0 * (np.exp(-gamma * time) + gamma * time - 1.0) / gamma**2

    prefactor = 4.0 * lam * gamma / beta
    for k in range(1, N_MATSUBARA + 1):
        nu = 2.0 * np.pi * k / beta
        coefficient = prefactor * nu / (nu**2 - gamma**2)
        result += coefficient * (np.exp(-nu * time) + nu * time - 1.0) / nu**2
    return result


def exact_spectrum(
    lambda_mev: float,
    cutoff_mev: float,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lam = lambda_mev * 1.0e-3 / EV_PER_AU
    gamma = cutoff_mev * 1.0e-3 / EV_PER_AU
    beta = K_PER_AU / TEMPERATURE_K
    vertical_energy = (ENERGY_00_EV + lambda_mev * 1.0e-3) / EV_PER_AU
    lineshape = drude_lineshape(time, lam, gamma, beta)
    response = np.exp(-1.0j * vertical_energy * time - lineshape)

    dt = float(time[1] - time[0])
    transform = np.fft.ifft(response) * len(time) * dt
    # The FFT rectangle rule gives the t=0 endpoint full weight.  A one-sided
    # response integral requires the trapezoidal half weight; without this
    # correction the real spectrum has a small constant baseline.
    transform -= 0.5 * dt * response[0]
    transform += 0.5 * dt * response[-1] * np.exp(
        2.0j * np.pi * np.arange(len(time)) * (len(time) - 1) / len(time)
    )
    omega = 2.0 * np.pi * np.fft.fftfreq(len(time), d=dt)
    positive = omega >= 0.0
    return omega[positive] * EV_PER_AU, transform.real[positive]


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "exact_cumulant"
    output_dir.mkdir(exist_ok=True)

    for cutoff in CUTOFFS_MEV:
        for lam in LAMBDAS_MEV:
            case = root / "heom_reference" / f"{lam:.0f}_{cutoff:.1f}"
            time = np.loadtxt(case / "resp1st.t1")
            omega, spectrum = exact_spectrum(lam, cutoff, time)
            output = output_dir / f"cumulant_lambda_{lam:.0f}_cutoff_{cutoff:.1f}.txt"
            np.savetxt(
                output,
                np.column_stack((omega, spectrum)),
                header="omega_eV  exact_cumulant_absorption",
            )
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
