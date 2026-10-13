"""Continuous-bath zero-temperature SCBA for a Brownian spectral density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from brownian import spectral_density_eV
from config import (
    CENTER_FREQUENCY_EV,
    CONTINUUM_OMEGA_MAX_EV,
    CONTINUUM_OMEGA_MIN_EV,
    CONTINUUM_STEP_EV,
    ENERGY_00_EV,
    ETA_EV,
    MAX_ITERATIONS,
    MIXING_G,
    TOLERANCE_G,
)


@dataclass
class ScbaResult:
    energy_eV: np.ndarray
    sigma_eV: np.ndarray
    green_per_eV: np.ndarray
    iterations: int
    error_g: float
    fixed_point_residual_eV: float
    converged: bool

    @property
    def spectrum_per_eV(self) -> np.ndarray:
        return -self.green_per_eV.imag


def make_continuum_grid(
    step_eV: float = CONTINUUM_STEP_EV,
    omega_min_eV: float = CONTINUUM_OMEGA_MIN_EV,
    omega_max_eV: float = CONTINUUM_OMEGA_MAX_EV,
) -> np.ndarray:
    if step_eV <= 0.0 or omega_min_eV >= omega_max_eV:
        raise ValueError("invalid continuous-SCBA grid")
    count = int(round((omega_max_eV - omega_min_eV) / step_eV)) + 1
    grid = omega_min_eV + step_eV * np.arange(count, dtype=float)
    if not np.isclose(grid[-1], omega_max_eV, atol=0.1 * step_eV):
        raise RuntimeError("continuous grid misses its upper boundary")
    return grid


def electronic_green_eV(
    energy_eV: np.ndarray,
    sigma_eV: np.ndarray,
    lambda_eV: float,
) -> np.ndarray:
    vertical_eV = ENERGY_00_EV + lambda_eV
    return 1.0 / (
        energy_eV - vertical_eV + 1.0j * ETA_EV - sigma_eV
    )


def asymptotic_brownian_tail_eV(
    energy_eV: np.ndarray,
    lambda_eV: float,
    gamma_eV: float,
    lower_boundary_eV: float,
) -> np.ndarray:
    r"""Integrate three omitted large-``q`` terms analytically."""
    center = CENTER_FREQUENCY_EV
    amplitude = 2.0 * lambda_eV * gamma_eV * center**2
    quadratic = gamma_eV**2 - 2.0 * center**2
    q_max = energy_eV - lower_boundary_eV
    a_all = (
        energy_eV - (ENERGY_00_EV + lambda_eV) + 1.0j * ETA_EV
    )
    result = np.zeros_like(energy_eV, dtype=complex)
    selected = (
        (q_max > max(2.0, 5.0 * max(center, gamma_eV)))
        & (q_max > 2.0 * np.abs(a_all))
    )
    q = q_max[selected]
    a = a_all[selected]
    result[selected] = (
        -amplitude / (3.0 * np.pi * q**3)
        - amplitude * a / (4.0 * np.pi * q**4)
        - amplitude
        * (a**2 - quadratic)
        / (5.0 * np.pi * q**5)
    )
    return result


def sigma_map_eV(
    energy_eV: np.ndarray,
    sigma_eV: np.ndarray,
    lambda_eV: float,
    gamma_eV: float,
) -> np.ndarray:
    r"""Evaluate ``Sigma(E)=int_0^inf J(q)G(E-q)dq/pi``."""
    spacing = np.diff(energy_eV)
    step_eV = float(spacing[0])
    if not np.allclose(spacing, step_eV, rtol=0.0, atol=1.0e-12):
        raise ValueError("continuous-SCBA energy grid must be uniform")
    green_eV = electronic_green_eV(
        energy_eV, sigma_eV, lambda_eV
    )
    frequency_eV = step_eV * np.arange(len(energy_eV), dtype=float)
    density_eV = spectral_density_eV(
        frequency_eV, lambda_eV, gamma_eV
    )
    mapped = (
        fftconvolve(green_eV, density_eV, mode="full")[
            : len(energy_eV)
        ]
        * step_eV
        / np.pi
    )
    # J(0)=0; the opposite trapezoid endpoint is x=energy_min.
    mapped -= (
        0.5
        * step_eV
        * green_eV[0]
        * density_eV
        / np.pi
    )
    mapped += asymptotic_brownian_tail_eV(
        energy_eV,
        lambda_eV,
        gamma_eV,
        float(energy_eV[0]),
    )
    return mapped


def solve_continuum_scba(
    lambda_mev: float,
    gamma_eV: float,
    *,
    energy_eV: np.ndarray | None = None,
    initial_sigma_eV: np.ndarray | None = None,
) -> ScbaResult:
    lambda_eV = lambda_mev * 1.0e-3
    if energy_eV is None:
        energy_eV = make_continuum_grid()
    else:
        energy_eV = np.asarray(energy_eV, dtype=float)
    if (
        energy_eV.ndim != 1
        or len(energy_eV) < 2
        or not np.all(np.diff(energy_eV) > 0.0)
    ):
        raise ValueError("invalid continuous-SCBA energy grid")
    if initial_sigma_eV is None:
        sigma_eV = np.zeros_like(energy_eV, dtype=complex)
    else:
        sigma_eV = np.asarray(initial_sigma_eV, dtype=complex).copy()
        if sigma_eV.shape != energy_eV.shape:
            raise ValueError("restart self-energy has the wrong shape")

    green_old = electronic_green_eV(
        energy_eV, sigma_eV, lambda_eV
    )
    error_g = np.inf
    for iteration in range(1, MAX_ITERATIONS + 1):
        mapped = sigma_map_eV(
            energy_eV, sigma_eV, lambda_eV, gamma_eV
        )
        sigma_new = (
            (1.0 - MIXING_G) * sigma_eV + MIXING_G * mapped
        )
        green_new = electronic_green_eV(
            energy_eV, sigma_new, lambda_eV
        )
        error_g = float(
            np.max(np.abs(green_new - green_old))
            / (np.max(np.abs(green_new)) + 1.0e-15)
        )
        sigma_eV = sigma_new
        green_old = green_new
        if iteration == 1 or iteration % 25 == 0:
            print(
                f"iteration {iteration:3d}: epsilon_G={error_g:.3e}",
                flush=True,
            )
        if error_g < TOLERANCE_G:
            break

    final_map = sigma_map_eV(
        energy_eV, sigma_eV, lambda_eV, gamma_eV
    )
    fixed_point_residual = float(
        np.max(np.abs(final_map - sigma_eV))
    )
    converged = error_g < TOLERANCE_G
    for name, values in (
        ("self-energy", sigma_eV),
        ("Green function", green_old),
        ("fixed-point map", final_map),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} contains NaN or Inf")
    if float(np.max(sigma_eV.imag)) > 1.0e-9:
        raise RuntimeError("SCBA self-energy violates causality")
    if float(np.min(-green_old.imag)) < -1.0e-9:
        raise RuntimeError("SCBA spectrum violates causality")
    return ScbaResult(
        energy_eV=energy_eV,
        sigma_eV=sigma_eV,
        green_per_eV=green_old,
        iterations=iteration,
        error_g=error_g,
        fixed_point_residual_eV=fixed_point_residual,
        converged=converged,
    )

