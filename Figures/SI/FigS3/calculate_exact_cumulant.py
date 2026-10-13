#!/usr/bin/env python3
"""Generate exact T=0 Brownian independent-boson spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brownian import exact_brownian_lineshape
from config import (
    ENERGY_00_EV,
    ETA_EV,
    EV_PER_AU,
    GAMMAS_EV,
    LAMBDAS_MEV,
    N_RESPONSE_FFT,
    N_TIME,
    OUTPUT_OMEGA_MAX_EV,
    OUTPUT_OMEGA_MIN_EV,
    TIME_STEP_AU,
    gamma_tag,
)


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else values.copy()


def spectrum_from_lineshape(
    time: np.ndarray,
    lineshape: np.ndarray,
    lambda_mev: float,
    *,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertical = (ENERGY_00_EV + lambda_mev * 1.0e-3) / EV_PER_AU
    eta = ETA_EV / EV_PER_AU
    response = np.exp(
        -1.0j * vertical * time - lineshape - eta * time
    )
    step = float(time[1] - time[0])
    if n_fft < len(time):
        raise ValueError("response FFT is shorter than the time grid")
    transform = np.fft.ifft(response, n=n_fft) * n_fft * step
    transform -= 0.5 * step * response[0]
    transform -= (
        0.5
        * step
        * response[-1]
        * np.exp(
            2.0j
            * np.pi
            * np.arange(n_fft)
            * (len(time) - 1)
            / n_fft
        )
    )
    omega = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=step)
    positive = omega >= 0.0
    return (
        omega[positive] * EV_PER_AU,
        transform.real[positive] / EV_PER_AU,
    )


def exact_spectrum(
    lambda_mev: float,
    gamma_eV: float,
    time: np.ndarray,
    *,
    response_n_fft: int | None = None,
    cumulant_fft_multiplier: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    kwargs = {}
    if cumulant_fft_multiplier is not None:
        kwargs["fft_multiplier"] = cumulant_fft_multiplier
    lineshape = exact_brownian_lineshape(
        time, lambda_mev, gamma_eV, **kwargs
    )
    if response_n_fft is None:
        response_n_fft = max(N_RESPONSE_FFT, 2 * len(time))
    energy, spectrum = spectrum_from_lineshape(
        time, lineshape, lambda_mev, n_fft=response_n_fft
    )
    endpoint = float(
        np.exp(
            -lineshape[-1].real
            - ETA_EV / EV_PER_AU * time[-1]
        )
    )
    return energy, spectrum, endpoint


def output_path(root: Path, lambda_mev: float, gamma_eV: float) -> Path:
    return (
        root
        / "exact_cumulant"
        / (
            f"cumulant_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "exact_cumulant"
    output_dir.mkdir(exist_ok=True)
    time = np.arange(N_TIME, dtype=float) * TIME_STEP_AU
    for gamma_eV in GAMMAS_EV:
        for lambda_mev in LAMBDAS_MEV:
            print(
                f"exact cumulant: lambda={lambda_mev:.0f} meV, "
                f"Gamma={gamma_tag(gamma_eV)} eV",
                flush=True,
            )
            energy, spectrum, endpoint = exact_spectrum(
                lambda_mev, gamma_eV, time
            )
            selected = (
                (energy >= OUTPUT_OMEGA_MIN_EV)
                & (energy <= OUTPUT_OMEGA_MAX_EV)
            )
            data_energy = energy[selected]
            data_spectrum = spectrum[selected]
            path = output_path(root, lambda_mev, gamma_eV)
            np.savetxt(
                path,
                np.column_stack(
                    (
                        data_energy,
                        data_spectrum,
                        normalize(data_spectrum),
                    )
                ),
                fmt="%.12e",
                header=(
                    "omega_eV  exact_cumulant_absorption_per_eV  "
                    "exact_cumulant_peak_normalized; "
                    f"T=0_K eta={ETA_EV * 1e3:.6g}_meV "
                    f"endpoint_response={endpoint:.6e}"
                ),
            )
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
