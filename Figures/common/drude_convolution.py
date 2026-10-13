#!/usr/bin/env python3
"""Continuous finite-temperature Drude convolution on a uniform grid.

The production Fan map is

    Sigma(w) = int_0^inf dnu J(nu) [
        (nbar(nu) + 1) G(w - nu) + nbar(nu) G(w + nu)
    ] / pi.

The two one-sided integrals are evaluated as zero-padded linear
convolutions.  No periodic wrapping is allowed.  The lower and finite-grid
endpoints use trapezoidal half weights.  A caller-supplied high-frequency
expansion of the bare propagator can be used to restore the omitted
emission tail below the lower electronic-grid boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft, ifft, next_fast_len


def drude_spectral_density(
    frequency: np.ndarray,
    reorganization: float,
    cutoff: float,
) -> np.ndarray:
    """Return ``J(nu)=2 lambda gamma nu/(nu^2+gamma^2)``."""
    frequency = np.asarray(frequency, dtype=float)
    return (
        2.0
        * reorganization
        * cutoff
        * frequency
        / (frequency**2 + cutoff**2)
    )


def thermal_drude_kernels(
    frequency: np.ndarray,
    reorganization: float,
    cutoff: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the emission and absorption kernels, including ``nu=0``."""
    frequency = np.asarray(frequency, dtype=float)
    if frequency.ndim != 1 or len(frequency) < 2:
        raise ValueError("frequency must be a one-dimensional grid")
    if frequency[0] != 0.0 or np.any(frequency < 0.0):
        raise ValueError("frequency grid must begin at zero")
    if reorganization < 0.0 or cutoff <= 0.0 or beta <= 0.0:
        raise ValueError("invalid Drude or temperature parameter")

    density = drude_spectral_density(
        frequency, reorganization, cutoff
    )
    absorption = np.zeros_like(frequency)
    positive = frequency > 0.0
    argument = beta * frequency[positive]
    finite = argument < 700.0
    occupation = np.zeros_like(argument)
    occupation[finite] = 1.0 / np.expm1(argument[finite])
    absorption[positive] = density[positive] * occupation

    # J(nu)nbar(nu) has a finite Ohmic limit at the origin.
    zero_limit = 2.0 * reorganization / (beta * cutoff)
    absorption[0] = zero_limit
    emission = absorption + density
    emission[0] = zero_limit
    return emission, absorption


def asymptotic_emission_tail(
    cutoff_frequency: np.ndarray,
    coefficients: list[np.ndarray],
    reorganization: float,
    cutoff: float,
    *,
    minimum_cutoff: float,
    density_terms: int = 8,
) -> np.ndarray:
    r"""Restore the omitted Drude emission tail.

    For the bare propagator below the lower grid boundary,

    ``G(w-q) = -sum_m a_m(w)/q^(m+1)``.

    Combining this expansion with the large-``q`` Drude series gives an
    analytic tail.  The correction is applied only where ``q_min`` exceeds
    ``minimum_cutoff`` so that the propagator expansion is controlled.
    """
    q_min = np.asarray(cutoff_frequency, dtype=float)
    if not coefficients:
        return np.zeros_like(q_min, dtype=complex)
    for coefficient in coefficients:
        if np.asarray(coefficient).shape != q_min.shape:
            raise ValueError("tail coefficients must match the grid")

    selected = q_min >= minimum_cutoff
    tail = np.zeros_like(q_min, dtype=complex)
    if not np.any(selected) or reorganization == 0.0:
        return tail

    q = q_min[selected]
    prefactor = 2.0 * reorganization * cutoff / np.pi
    cutoff_squared = cutoff**2
    for order, coefficient in enumerate(coefficients):
        integral = np.zeros_like(q)
        cutoff_power = 1.0
        sign = 1.0
        for density_order in range(density_terms):
            power = order + 1 + 2 * density_order
            integral += (
                sign
                * cutoff_power
                * q ** (-power)
                / power
            )
            sign *= -1.0
            cutoff_power *= cutoff_squared
        tail[selected] -= (
            np.asarray(coefficient)[selected]
            * prefactor
            * integral
        )
    return tail


@dataclass
class LinearDrudeConvolver:
    """Reusable zero-padded convolution operator for one Drude bath."""

    grid: np.ndarray
    reorganization: float
    cutoff: float
    beta: float
    _step: float = field(init=False, repr=False)
    _emission: np.ndarray = field(init=False, repr=False)
    _absorption: np.ndarray = field(init=False, repr=False)
    _fft_length: int = field(init=False, repr=False)
    _emission_fft: np.ndarray = field(init=False, repr=False)
    _absorption_fft: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float)
        if self.grid.ndim != 1 or len(self.grid) < 2:
            raise ValueError("electronic grid must be one-dimensional")
        spacing = np.diff(self.grid)
        self._step = float(spacing[0])
        if self._step <= 0.0 or not np.allclose(
            spacing, self._step, rtol=1.0e-10, atol=1.0e-14
        ):
            raise ValueError("linear convolution requires a uniform grid")

        frequency = self._step * np.arange(len(self.grid), dtype=float)
        self._emission, self._absorption = thermal_drude_kernels(
            frequency,
            self.reorganization,
            self.cutoff,
            self.beta,
        )
        self._fft_length = next_fast_len(2 * len(self.grid) - 1)
        self._emission_fft = fft(
            self._emission, self._fft_length
        )
        self._absorption_fft = fft(
            self._absorption, self._fft_length
        )

    @property
    def step(self) -> float:
        return self._step

    @property
    def frequency(self) -> np.ndarray:
        return self._step * np.arange(len(self.grid), dtype=float)

    def apply(
        self,
        green: np.ndarray,
        *,
        tail_coefficients: list[np.ndarray] | None = None,
        tail_minimum_cutoff: float | None = None,
    ) -> np.ndarray:
        """Apply the continuous Fan map to a sampled retarded propagator."""
        green = np.asarray(green, dtype=complex)
        if green.shape != self.grid.shape:
            raise ValueError("Green function must match the convolution grid")

        green_fft = fft(green, self._fft_length)
        emission = ifft(
            green_fft * self._emission_fft
        )[: len(self.grid)]

        reversed_fft = fft(green[::-1], self._fft_length)
        absorption = ifft(
            reversed_fft * self._absorption_fft
        )[: len(self.grid)][::-1]

        scale = self._step / np.pi
        result = scale * (emission + absorption)

        # Trapezoidal half weights at nu=0 and at the two finite electronic
        # grid boundaries.  The latter delimit the sampled part of G.
        result -= 0.5 * scale * (
            self._emission[0] + self._absorption[0]
        ) * green
        result -= (
            0.5 * scale * self._emission * green[0]
        )
        result -= (
            0.5 * scale * self._absorption[::-1] * green[-1]
        )

        if tail_coefficients is not None:
            if tail_minimum_cutoff is None:
                raise ValueError(
                    "tail_minimum_cutoff is required with tail coefficients"
                )
            result += asymptotic_emission_tail(
                self.grid - self.grid[0],
                tail_coefficients,
                self.reorganization,
                self.cutoff,
                minimum_cutoff=tail_minimum_cutoff,
            )
        return result
