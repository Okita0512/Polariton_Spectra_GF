#!/usr/bin/env python3
"""Discrete-mode Born and self-consistent Born solvers for a Frenkel dimer.

The physical model and all stored energies use eV with ``hbar = 1``.  Two
independent identical site oscillators give a diagonal site self-energy.  For
the exchange-symmetric dimer the production equation is

    Sigma(w) = g^2 [(n0 + 1) G_loc(w - omega0)
                     + n0 G_loc(w + omega0)],

where ``G_loc = (G_B + G_D) / 2``.  A separate full two-by-two site solver is
retained as a regression test and is deliberately started from unequal site
self-energies.

Shifted Green functions are evaluated by sampling the *self-energy* and then
applying Dyson's equation at the requested shifted frequency.  Linear
interpolation is nonperiodic and preserves the sign of a causal imaginary
part.  Outside the finite grid a causal first-Born 1/w tail is used; no
periodic wrapping or broadened phonon line is introduced.

This file can be executed from its directory or from the repository root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:  # Support both ``python file.py`` and import as part of a package.
    from .single_mode_common import (
        GridSpec,
        ModelParams,
        SPECTRA,
        default_grid_spec,
        encode_metadata,
        ensure_directories,
        finite_or_raise,
        model_dict,
        signed_tag,
    )
except ImportError:  # pragma: no cover - exercised by the documented CLI.
    module_directory = str(Path(__file__).resolve().parent)
    if module_directory not in sys.path:
        sys.path.insert(0, module_directory)
    from single_mode_common import (
        GridSpec,
        ModelParams,
        SPECTRA,
        default_grid_spec,
        encode_metadata,
        ensure_directories,
        finite_or_raise,
        model_dict,
        signed_tag,
    )


HISTORY_COLUMNS = (
    "iteration",
    "fixed_point_abs_residual_eV",
    "fixed_point_relative_residual",
    "green_relative_change",
    "mixing",
    "max_Im_Sigma_eV",
)


@dataclass(frozen=True)
class ScbaNumerics:
    """Iteration controls for the finite-temperature fixed point."""

    max_iterations: int = 5000
    initial_mixing: float = 0.08
    minimum_mixing: float = 0.002
    maximum_mixing: float = 0.40
    mixing_growth: float = 1.05
    residual_growth_factor: float = 1.10
    rapid_decrease_factor: float = 0.70
    absolute_residual_tolerance_eV: float = 1.0e-9
    relative_residual_tolerance: float = 1.0e-8
    green_change_tolerance: float = 1.0e-8
    consecutive_converged: int = 3
    causality_tolerance: float = 5.0e-10

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 0.0 < self.minimum_mixing <= self.initial_mixing:
            raise ValueError("Require 0 < minimum_mixing <= initial_mixing")
        if not self.initial_mixing <= self.maximum_mixing <= 1.0:
            raise ValueError("Require initial_mixing <= maximum_mixing <= 1")
        if self.mixing_growth <= 1.0:
            raise ValueError("mixing_growth must exceed one")
        if self.residual_growth_factor <= 1.0:
            raise ValueError("residual_growth_factor must exceed one")
        if not 0.0 < self.rapid_decrease_factor < 1.0:
            raise ValueError("rapid_decrease_factor must lie between zero and one")
        if self.absolute_residual_tolerance_eV <= 0.0:
            raise ValueError("absolute residual tolerance must be positive")
        if self.relative_residual_tolerance <= 0.0:
            raise ValueError("relative residual tolerance must be positive")
        if self.green_change_tolerance <= 0.0:
            raise ValueError("Green-function tolerance must be positive")
        if self.consecutive_converged < 1:
            raise ValueError("consecutive_converged must be positive")
        if self.causality_tolerance <= 0.0:
            raise ValueError("causality_tolerance must be positive")


@dataclass
class ScbaResult:
    """One BA or SCBA solution on the complete internal frequency grid."""

    mode: str
    solver: str
    omega_eV: np.ndarray
    sigma_eV: np.ndarray
    green_bright_per_eV: np.ndarray
    green_dark_per_eV: np.ndarray
    green_local_per_eV: np.ndarray
    sigma_from_bright_emission_eV: np.ndarray
    sigma_from_bright_absorption_eV: np.ndarray
    sigma_from_dark_emission_eV: np.ndarray
    sigma_from_dark_absorption_eV: np.ndarray
    iterations: int
    converged: bool
    fixed_point_abs_residual_eV: float
    fixed_point_relative_residual: float
    green_relative_change: float
    final_mixing: float
    max_positive_im_sigma_eV: float
    minimum_bright_spectrum_per_eV: float
    minimum_dark_spectrum_per_eV: float
    history: np.ndarray
    seed_label: str

    @property
    def energy_eV(self) -> np.ndarray:
        """ED-compatible alias for the common frequency grid."""

        return self.omega_eV

    @property
    def green_bright(self) -> np.ndarray:
        return self.green_bright_per_eV

    @property
    def green_dark(self) -> np.ndarray:
        return self.green_dark_per_eV

    @property
    def green_local(self) -> np.ndarray:
        return self.green_local_per_eV

    @property
    def spectrum_bright_per_eV(self) -> np.ndarray:
        return -np.imag(self.green_bright_per_eV)

    @property
    def spectrum_dark_per_eV(self) -> np.ndarray:
        return -np.imag(self.green_dark_per_eV)

    @property
    def spectrum_local_per_eV(self) -> np.ndarray:
        return -np.imag(self.green_local_per_eV)

    @property
    def spectrum_bright(self) -> np.ndarray:
        return self.spectrum_bright_per_eV

    @property
    def spectrum_dark(self) -> np.ndarray:
        return self.spectrum_dark_per_eV

    @property
    def spectrum_local(self) -> np.ndarray:
        return self.spectrum_local_per_eV

    @property
    def sigma_channel_sum_eV(self) -> np.ndarray:
        return (
            self.sigma_from_bright_emission_eV
            + self.sigma_from_bright_absorption_eV
            + self.sigma_from_dark_emission_eV
            + self.sigma_from_dark_absorption_eV
        )

    def diagnostic_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "solver": self.solver,
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "fixed_point_abs_residual_eV": float(
                self.fixed_point_abs_residual_eV
            ),
            "fixed_point_relative_residual": float(
                self.fixed_point_relative_residual
            ),
            "green_relative_change": float(self.green_relative_change),
            "final_mixing": float(self.final_mixing),
            "max_positive_im_sigma_eV": float(
                self.max_positive_im_sigma_eV
            ),
            "minimum_bright_spectrum_per_eV": float(
                self.minimum_bright_spectrum_per_eV
            ),
            "minimum_dark_spectrum_per_eV": float(
                self.minimum_dark_spectrum_per_eV
            ),
            "sigma_channel_sum_residual_eV": float(
                np.max(np.abs(self.sigma_channel_sum_eV - self.sigma_eV))
            ),
            "seed_label": self.seed_label,
        }


@dataclass
class FullMatrixRegression:
    """Result and discrepancies from the asymmetric full-matrix solve."""

    sigma_site1_eV: np.ndarray
    sigma_site2_eV: np.ndarray
    green_bright_per_eV: np.ndarray
    green_dark_per_eV: np.ndarray
    green_local_per_eV: np.ndarray
    history: np.ndarray
    iterations: int
    converged: bool
    fixed_point_abs_residual_eV: float
    fixed_point_relative_residual: float
    green_relative_change: float
    final_mixing: float
    max_site_sigma_difference_eV: float
    relative_site_sigma_difference: float
    max_mean_sigma_minus_reduced_eV: float
    relative_mean_sigma_difference: float
    relative_bright_green_difference: float
    relative_dark_green_difference: float
    symmetric_map_identity_max_error_eV: float
    symmetric_green_identity_max_error_per_eV: float

    def diagnostic_dict(self) -> Dict[str, object]:
        return {
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "fixed_point_abs_residual_eV": float(
                self.fixed_point_abs_residual_eV
            ),
            "fixed_point_relative_residual": float(
                self.fixed_point_relative_residual
            ),
            "green_relative_change": float(self.green_relative_change),
            "final_mixing": float(self.final_mixing),
            "max_site_sigma_difference_eV": float(
                self.max_site_sigma_difference_eV
            ),
            "relative_site_sigma_difference": float(
                self.relative_site_sigma_difference
            ),
            "max_mean_sigma_minus_reduced_eV": float(
                self.max_mean_sigma_minus_reduced_eV
            ),
            "relative_mean_sigma_difference": float(
                self.relative_mean_sigma_difference
            ),
            "relative_bright_green_difference": float(
                self.relative_bright_green_difference
            ),
            "relative_dark_green_difference": float(
                self.relative_dark_green_difference
            ),
            "symmetric_map_identity_max_error_eV": float(
                self.symmetric_map_identity_max_error_eV
            ),
            "symmetric_green_identity_max_error_per_eV": float(
                self.symmetric_green_identity_max_error_per_eV
            ),
        }


def _validate_grid(grid: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("The SCBA grid must be a one-dimensional array")
    if not np.all(np.isfinite(grid)):
        raise ValueError("The SCBA grid contains NaN or Inf")
    differences = np.diff(grid)
    if np.any(differences <= 0.0):
        raise ValueError("The SCBA grid must be strictly increasing")
    step = float(differences[0])
    if not np.allclose(differences, step, rtol=2.0e-10, atol=2.0e-14):
        raise ValueError("The SCBA solver requires a uniform frequency grid")
    return step


def validate_grid_spec(grid_spec: GridSpec, grid: np.ndarray) -> None:
    """Check that the finite analysis window lies within the internal grid."""

    _validate_grid(grid)
    if not grid_spec.analysis_min_eV < grid_spec.analysis_max_eV:
        raise ValueError("Analysis bounds are unordered")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(grid)))
    if grid_spec.analysis_min_eV < grid[0] - tolerance:
        raise ValueError("analysis_min_eV lies below the internal grid")
    if grid_spec.analysis_max_eV > grid[-1] + tolerance:
        raise ValueError("analysis_max_eV lies above the internal grid")


def reduced_green(
    omega_eV: np.ndarray,
    sigma_eV: np.ndarray,
    model: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bright, dark, and site-local retarded Green functions."""

    common_inverse = (
        np.asarray(omega_eV)
        - model.vertical_energy_eV
        + 1.0j * model.eta_eV
        - np.asarray(sigma_eV)
    )
    bright = 1.0 / (common_inverse - model.j_eV)
    dark = 1.0 / (common_inverse + model.j_eV)
    local = 0.5 * (bright + dark)
    return bright, dark, local


def bare_local_green(omega_eV: np.ndarray, model: ModelParams) -> np.ndarray:
    """Bare site-diagonal propagator, expressed through bright and dark poles."""

    zeros = np.zeros_like(np.asarray(omega_eV), dtype=complex)
    return reduced_green(np.asarray(omega_eV), zeros, model)[2]


def asymptotic_sigma(omega_eV: np.ndarray, model: ModelParams) -> np.ndarray:
    """Causal first-Born extension with the correct vanishing 1/w tail."""

    points = np.asarray(omega_eV, dtype=float)
    if model.coupling_squared_eV2 == 0.0:
        return np.zeros_like(points, dtype=complex)
    occupation = model.bose_occupation
    emission = bare_local_green(points - model.omega0_eV, model)
    absorption = bare_local_green(points + model.omega0_eV, model)
    return model.coupling_squared_eV2 * (
        (occupation + 1.0) * emission + occupation * absorption
    )


def interpolate_sigma_with_tail(
    points_eV: np.ndarray,
    grid_eV: np.ndarray,
    sigma_eV: np.ndarray,
    model: ModelParams,
    use_asymptotic_tail: bool = True,
) -> np.ndarray:
    """Linearly sample Sigma without wrapping.

    Self-consistent iterations use the causal first-Born tail outside the
    finite grid.  A one-shot BA evaluation passes ``False`` so that its input
    is exactly the bare Green function, including at shifted edge points.
    """

    points = np.asarray(points_eV, dtype=float)
    grid = np.asarray(grid_eV, dtype=float)
    sigma = np.asarray(sigma_eV, dtype=complex)
    if sigma.shape != grid.shape:
        raise ValueError("Self-energy and frequency grid shapes differ")
    result = np.empty(points.shape, dtype=complex)
    inside = (points >= grid[0]) & (points <= grid[-1])
    if np.any(inside):
        result[inside] = np.interp(points[inside], grid, sigma.real) + 1.0j * np.interp(
            points[inside], grid, sigma.imag
        )
    if np.any(~inside):
        if use_asymptotic_tail:
            result[~inside] = asymptotic_sigma(points[~inside], model)
        else:
            result[~inside] = 0.0
    return result


class DiscreteModeKernel:
    """Cached shifted grids and Fan maps for one model/grid pair."""

    def __init__(self, model: ModelParams, omega_eV: np.ndarray):
        model.validate()
        self.model = model
        self.omega_eV = np.asarray(omega_eV, dtype=float)
        self.step_eV = _validate_grid(self.omega_eV)
        self.minus_eV = self.omega_eV - model.omega0_eV
        self.plus_eV = self.omega_eV + model.omega0_eV
        shift_float = model.omega0_eV / self.step_eV
        shift_integer = int(round(shift_float))
        tolerance = 2.0e-10 * max(1.0, abs(shift_float))
        self.shift_index = (
            shift_integer if abs(shift_float - shift_integer) <= tolerance else None
        )

    def shifted_sigma(
        self,
        sigma_eV: np.ndarray,
        sign: int,
        use_asymptotic_tail: bool = True,
    ) -> np.ndarray:
        """Return Sigma(w + sign*omega0), explicitly replacing edge values."""

        sigma = np.asarray(sigma_eV, dtype=complex)
        if sigma.shape != self.omega_eV.shape:
            raise ValueError("Self-energy and kernel grid shapes differ")
        points = self.plus_eV if sign > 0 else self.minus_eV
        shift = self.shift_index
        if shift is None or shift == 0:
            return interpolate_sigma_with_tail(
                points,
                self.omega_eV,
                sigma,
                self.model,
                use_asymptotic_tail=use_asymptotic_tail,
            )

        result = np.empty_like(sigma)
        count = len(sigma)
        if shift >= count:
            if use_asymptotic_tail:
                return asymptotic_sigma(points, self.model)
            return np.zeros_like(sigma)
        if sign > 0:
            result[:-shift] = sigma[shift:]
            if use_asymptotic_tail:
                result[-shift:] = asymptotic_sigma(
                    points[-shift:], self.model
                )
            else:
                result[-shift:] = 0.0
        else:
            result[shift:] = sigma[:-shift]
            if use_asymptotic_tail:
                result[:shift] = asymptotic_sigma(
                    points[:shift], self.model
                )
            else:
                result[:shift] = 0.0
        return result

    def fan_map_reduced(
        self,
        sigma_eV: np.ndarray,
        return_channels: bool = False,
        use_asymptotic_tail: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Apply the symmetry-reduced finite-temperature Fan functional."""

        sigma_minus = self.shifted_sigma(
            sigma_eV, -1, use_asymptotic_tail=use_asymptotic_tail
        )
        sigma_plus = self.shifted_sigma(
            sigma_eV, +1, use_asymptotic_tail=use_asymptotic_tail
        )
        bright_minus, dark_minus, _ = reduced_green(
            self.minus_eV, sigma_minus, self.model
        )
        bright_plus, dark_plus, _ = reduced_green(
            self.plus_eV, sigma_plus, self.model
        )
        prefactor = 0.5 * self.model.coupling_squared_eV2
        occupation = self.model.bose_occupation
        channels = {
            "bright_emission": prefactor
            * (occupation + 1.0)
            * bright_minus,
            "bright_absorption": prefactor * occupation * bright_plus,
            "dark_emission": prefactor * (occupation + 1.0) * dark_minus,
            "dark_absorption": prefactor * occupation * dark_plus,
        }
        mapped = (
            channels["bright_emission"]
            + channels["bright_absorption"]
            + channels["dark_emission"]
            + channels["dark_absorption"]
        )
        if return_channels:
            return mapped, channels
        return mapped

    def fan_map_full(
        self,
        sigma_site1_eV: np.ndarray,
        sigma_site2_eV: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the site-local Fan map using the complete 2x2 Dyson matrix."""

        s1_minus = self.shifted_sigma(sigma_site1_eV, -1)
        s2_minus = self.shifted_sigma(sigma_site2_eV, -1)
        s1_plus = self.shifted_sigma(sigma_site1_eV, +1)
        s2_plus = self.shifted_sigma(sigma_site2_eV, +1)
        g11_minus, g22_minus, _, _ = full_site_green(
            self.minus_eV, s1_minus, s2_minus, self.model
        )
        g11_plus, g22_plus, _, _ = full_site_green(
            self.plus_eV, s1_plus, s2_plus, self.model
        )
        occupation = self.model.bose_occupation
        coupling_squared = self.model.coupling_squared_eV2
        mapped1 = coupling_squared * (
            (occupation + 1.0) * g11_minus + occupation * g11_plus
        )
        mapped2 = coupling_squared * (
            (occupation + 1.0) * g22_minus + occupation * g22_plus
        )
        return mapped1, mapped2


def full_site_green(
    omega_eV: np.ndarray,
    sigma_site1_eV: np.ndarray,
    sigma_site2_eV: np.ndarray,
    model: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return G11, G22, G12, G21 from the complete site-basis Dyson equation."""

    common = np.asarray(omega_eV) - model.vertical_energy_eV + 1.0j * model.eta_eV
    inverse11 = common - np.asarray(sigma_site1_eV)
    inverse22 = common - np.asarray(sigma_site2_eV)
    determinant = inverse11 * inverse22 - model.j_eV**2
    g11 = inverse22 / determinant
    g22 = inverse11 / determinant
    g12 = model.j_eV / determinant
    return g11, g22, g12, g12.copy()


def full_bright_dark_green(
    omega_eV: np.ndarray,
    sigma_site1_eV: np.ndarray,
    sigma_site2_eV: np.ndarray,
    model: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the full site Green matrix onto bright, dark, and local channels."""

    g11, g22, g12, g21 = full_site_green(
        omega_eV, sigma_site1_eV, sigma_site2_eV, model
    )
    bright = 0.5 * (g11 + g22 + g12 + g21)
    dark = 0.5 * (g11 + g22 - g12 - g21)
    local = 0.5 * (g11 + g22)
    return bright, dark, local


def _relative_residual(
    residual: np.ndarray,
    mapped: np.ndarray,
    current: np.ndarray,
    eta_eV: float,
) -> float:
    denominator = max(
        eta_eV,
        float(np.max(np.abs(mapped))),
        float(np.max(np.abs(current))),
    )
    return float(np.max(np.abs(residual)) / denominator)


def _relative_array_change(new: np.ndarray, old: np.ndarray) -> float:
    denominator = max(1.0, float(np.max(np.abs(new))), float(np.max(np.abs(old))))
    return float(np.max(np.abs(new - old)) / denominator)


def _causality_diagnostics(
    sigma_eV: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
) -> Tuple[float, float, float]:
    return (
        float(max(0.0, np.max(np.imag(sigma_eV)))),
        float(np.min(-np.imag(bright))),
        float(np.min(-np.imag(dark))),
    )


def _validate_causality(
    name: str,
    sigma_eV: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
    tolerance: float,
) -> Tuple[float, float, float]:
    max_positive_imag, min_bright, min_dark = _causality_diagnostics(
        sigma_eV, bright, dark
    )
    if max_positive_imag > tolerance:
        raise RuntimeError(
            "%s violates retarded self-energy causality: max Im Sigma=%.6e eV"
            % (name, max_positive_imag)
        )
    spectral_scale = max(
        1.0, float(np.max(-np.imag(bright))), float(np.max(-np.imag(dark)))
    )
    # Spectral functions and self-energies have reciprocal units, so use a
    # scale-relative spectral tolerance rather than reusing the eV tolerance.
    spectral_tolerance = max(1.0e-10 * spectral_scale, 1.0e-10)
    if min_bright < -spectral_tolerance or min_dark < -spectral_tolerance:
        raise RuntimeError(
            "%s violates spectral causality: min A_B=%.6e, min A_D=%.6e"
            % (name, min_bright, min_dark)
        )
    return max_positive_imag, min_bright, min_dark


def _make_result(
    mode: str,
    solver: str,
    model: ModelParams,
    kernel: DiscreteModeKernel,
    sigma_eV: np.ndarray,
    channels: Dict[str, np.ndarray],
    iterations: int,
    converged: bool,
    fixed_point_abs_residual_eV: float,
    fixed_point_relative_residual: float,
    green_relative_change: float,
    final_mixing: float,
    history: np.ndarray,
    seed_label: str,
    causality_tolerance: float,
) -> ScbaResult:
    bright, dark, local = reduced_green(kernel.omega_eV, sigma_eV, model)
    finite_or_raise(
        mode,
        [
            sigma_eV,
            bright,
            dark,
            local,
            channels["bright_emission"],
            channels["bright_absorption"],
            channels["dark_emission"],
            channels["dark_absorption"],
        ],
    )
    max_positive_imag, min_bright, min_dark = _validate_causality(
        mode, sigma_eV, bright, dark, causality_tolerance
    )
    return ScbaResult(
        mode=mode,
        solver=solver,
        omega_eV=kernel.omega_eV.copy(),
        sigma_eV=np.asarray(sigma_eV).copy(),
        green_bright_per_eV=bright,
        green_dark_per_eV=dark,
        green_local_per_eV=local,
        sigma_from_bright_emission_eV=channels["bright_emission"],
        sigma_from_bright_absorption_eV=channels["bright_absorption"],
        sigma_from_dark_emission_eV=channels["dark_emission"],
        sigma_from_dark_absorption_eV=channels["dark_absorption"],
        iterations=iterations,
        converged=converged,
        fixed_point_abs_residual_eV=fixed_point_abs_residual_eV,
        fixed_point_relative_residual=fixed_point_relative_residual,
        green_relative_change=green_relative_change,
        final_mixing=final_mixing,
        max_positive_im_sigma_eV=max_positive_imag,
        minimum_bright_spectrum_per_eV=min_bright,
        minimum_dark_spectrum_per_eV=min_dark,
        history=np.asarray(history, dtype=float),
        seed_label=seed_label,
    )


def solve_single_shot_ba(
    model: ModelParams,
    grid_spec: Optional[GridSpec] = None,
    numerics: Optional[ScbaNumerics] = None,
) -> ScbaResult:
    """Evaluate the discrete Fan kernel once with the bare Green function."""

    model.validate()
    controls = numerics if numerics is not None else ScbaNumerics()
    controls.validate()
    selected_grid = grid_spec if grid_spec is not None else default_grid_spec(model)
    omega = selected_grid.grid()
    validate_grid_spec(selected_grid, omega)
    kernel = DiscreteModeKernel(model, omega)
    zeros = np.zeros_like(omega, dtype=complex)
    sigma_ba, channels = kernel.fan_map_reduced(
        zeros,
        return_channels=True,
        use_asymptotic_tail=False,
    )
    mapped_again = kernel.fan_map_reduced(sigma_ba)
    residual = mapped_again - sigma_ba
    residual_abs = float(np.max(np.abs(residual)))
    residual_relative = _relative_residual(
        residual, mapped_again, sigma_ba, model.eta_eV
    )
    history = np.asarray(
        [[1.0, residual_abs, residual_relative, 0.0, 1.0, np.max(sigma_ba.imag)]],
        dtype=float,
    )
    return _make_result(
        "single_shot_ba",
        "one_fan_map_from_bare_G",
        model,
        kernel,
        sigma_ba,
        channels,
        1,
        True,
        residual_abs,
        residual_relative,
        0.0,
        1.0,
        history,
        "bare_G",
        controls.causality_tolerance,
    )


def solve_ba(
    model: ModelParams,
    grid_spec: Optional[GridSpec] = None,
    numerics: Optional[ScbaNumerics] = None,
) -> ScbaResult:
    """Short alias used by the scan driver."""

    return solve_single_shot_ba(model, grid_spec, numerics)


def solve_zero_temperature_sweep(
    model: ModelParams,
    grid_spec: Optional[GridSpec] = None,
    numerics: Optional[ScbaNumerics] = None,
) -> ScbaResult:
    """Solve the T=0 one-way recursion by an ascending-frequency sweep.

    At zero temperature Sigma(w) depends only on Sigma(w-omega0).  With a
    grid finer than omega0 all interpolation support is already known while
    sweeping upward.  Aligned shifts are evaluated in vector blocks; the
    nonaligned fallback uses the two previously completed grid points.
    """

    model.validate()
    if model.temperature_K != 0.0:
        raise ValueError("The ascending SCBA sweep is valid only at T=0")
    controls = numerics if numerics is not None else ScbaNumerics()
    controls.validate()
    selected_grid = grid_spec if grid_spec is not None else default_grid_spec(model)
    omega = selected_grid.grid()
    validate_grid_spec(selected_grid, omega)
    kernel = DiscreteModeKernel(model, omega)
    if kernel.step_eV >= model.omega0_eV:
        raise ValueError(
            "The T=0 ascending sweep requires grid step smaller than omega0"
        )

    sigma = np.empty_like(omega, dtype=complex)
    shift = kernel.shift_index
    if shift is not None and shift > 0:
        for start in range(0, len(omega), shift):
            stop = min(start + shift, len(omega))
            shifted_points = kernel.minus_eV[start:stop]
            if start < shift:
                sampled_sigma = asymptotic_sigma(shifted_points, model)
            else:
                sampled_sigma = sigma[start - shift : stop - shift]
            _, _, local = reduced_green(shifted_points, sampled_sigma, model)
            sigma[start:stop] = model.coupling_squared_eV2 * local
    else:
        step = kernel.step_eV
        origin = float(omega[0])
        for index, shifted_point in enumerate(kernel.minus_eV):
            if shifted_point < origin:
                sampled = complex(asymptotic_sigma(np.asarray([shifted_point]), model)[0])
            else:
                coordinate = (shifted_point - origin) / step
                lower = int(np.floor(coordinate + 8.0 * np.finfo(float).eps))
                fraction = coordinate - lower
                if lower + 1 >= index:
                    raise RuntimeError(
                        "T=0 interpolation requested an unfinished frequency; refine the grid"
                    )
                sampled = (1.0 - fraction) * sigma[lower] + fraction * sigma[lower + 1]
            _, _, local = reduced_green(
                np.asarray([shifted_point]), np.asarray([sampled]), model
            )
            sigma[index] = model.coupling_squared_eV2 * local[0]

    mapped, channels = kernel.fan_map_reduced(sigma, return_channels=True)
    residual = mapped - sigma
    residual_abs = float(np.max(np.abs(residual)))
    residual_relative = _relative_residual(residual, mapped, sigma, model.eta_eV)
    history = np.asarray(
        [[1.0, residual_abs, residual_relative, 0.0, 1.0, np.max(sigma.imag)]],
        dtype=float,
    )
    sweep_roundoff = max(
        100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(sigma)))),
        1.0e-13,
    )
    sweep_abs_tolerance = max(
        sweep_roundoff, controls.absolute_residual_tolerance_eV
    )
    if (
        residual_abs > sweep_abs_tolerance
        or residual_relative > controls.relative_residual_tolerance
    ):
        raise RuntimeError(
            "T=0 ascending sweep failed its fixed-point check: %.6e eV "
            "(relative %.6e)" % (residual_abs, residual_relative)
        )
    return _make_result(
        "scba",
        "zero_temperature_ascending_sweep",
        model,
        kernel,
        sigma,
        channels,
        1,
        True,
        residual_abs,
        residual_relative,
        0.0,
        1.0,
        history,
        "causal_asymptotic_tail",
        controls.causality_tolerance,
    )


def _coerce_seed(
    seed: Optional[Union[np.ndarray, ScbaResult]],
    model: ModelParams,
    omega_eV: np.ndarray,
    ba: ScbaResult,
) -> Tuple[np.ndarray, str]:
    if seed is None:
        return ba.sigma_eV.copy(), "single_shot_ba"
    if isinstance(seed, ScbaResult):
        values = interpolate_sigma_with_tail(
            omega_eV, seed.omega_eV, seed.sigma_eV, model
        )
        label = "%s:%s" % (seed.mode, seed.solver)
    else:
        values = np.asarray(seed, dtype=complex)
        if values.shape != omega_eV.shape:
            raise ValueError(
                "An array SCBA seed must have the same shape as the target grid"
            )
        values = values.copy()
        label = "user_array"
    finite_or_raise("SCBA seed", [values])
    if float(np.max(values.imag)) > 5.0e-8:
        raise ValueError("SCBA seed has a noncausal positive imaginary part")
    return values, label


def solve_scba(
    model: ModelParams,
    grid_spec: Optional[GridSpec] = None,
    numerics: Optional[ScbaNumerics] = None,
    seed: Optional[Union[np.ndarray, ScbaResult]] = None,
    use_zero_temperature_sweep: bool = True,
    verbose: bool = False,
) -> ScbaResult:
    """Solve the reduced discrete-mode SCBA equation on the full grid."""

    model.validate()
    controls = numerics if numerics is not None else ScbaNumerics()
    controls.validate()
    selected_grid = grid_spec if grid_spec is not None else default_grid_spec(model)
    if model.temperature_K == 0.0 and use_zero_temperature_sweep and seed is None:
        return solve_zero_temperature_sweep(model, selected_grid, controls)

    ba = solve_single_shot_ba(model, selected_grid, controls)
    omega = ba.omega_eV
    kernel = DiscreteModeKernel(model, omega)
    sigma, seed_label = _coerce_seed(seed, model, omega, ba)
    bright, dark, _ = reduced_green(omega, sigma, model)
    old_green = np.concatenate((bright, dark))
    previous_residual = None  # type: Optional[float]
    previous_green_change = 1.0
    mixing = controls.initial_mixing
    stable_iterations = 0
    history_rows = []

    for iteration in range(1, controls.max_iterations + 1):
        mapped = kernel.fan_map_reduced(sigma)
        residual = mapped - sigma
        absolute_residual = float(np.max(np.abs(residual)))
        relative_residual = _relative_residual(
            residual, mapped, sigma, model.eta_eV
        )

        history_rows.append(
            [
                float(iteration),
                absolute_residual,
                relative_residual,
                previous_green_change,
                mixing,
                float(np.max(sigma.imag)),
            ]
        )
        residual_ok = (
            absolute_residual < controls.absolute_residual_tolerance_eV
            and relative_residual < controls.relative_residual_tolerance
        )
        green_ok = previous_green_change < controls.green_change_tolerance
        stable_iterations = stable_iterations + 1 if residual_ok and green_ok else 0
        if stable_iterations >= controls.consecutive_converged:
            final_map, channels = kernel.fan_map_reduced(
                sigma, return_channels=True
            )
            final_residual = final_map - sigma
            final_abs = float(np.max(np.abs(final_residual)))
            final_relative = _relative_residual(
                final_residual, final_map, sigma, model.eta_eV
            )
            return _make_result(
                "scba",
                "adaptive_damped_picard",
                model,
                kernel,
                sigma,
                channels,
                iteration,
                True,
                final_abs,
                final_relative,
                previous_green_change,
                mixing,
                np.asarray(history_rows),
                seed_label,
                controls.causality_tolerance,
            )

        if previous_residual is not None:
            if absolute_residual > controls.residual_growth_factor * previous_residual:
                mixing = max(controls.minimum_mixing, 0.5 * mixing)
            elif absolute_residual < controls.rapid_decrease_factor * previous_residual:
                mixing = min(controls.maximum_mixing, controls.mixing_growth * mixing)

        sigma_new = sigma + mixing * residual
        bright_new, dark_new, _ = reduced_green(omega, sigma_new, model)
        new_green = np.concatenate((bright_new, dark_new))
        previous_green_change = _relative_array_change(new_green, old_green)
        sigma = sigma_new
        old_green = new_green
        previous_residual = absolute_residual

        if verbose and (iteration == 1 or iteration % 100 == 0):
            print(
                "SCBA iteration %d: abs=%.3e eV rel=%.3e dG=%.3e mix=%.4f"
                % (
                    iteration,
                    absolute_residual,
                    relative_residual,
                    previous_green_change,
                    mixing,
                )
            )

    raise RuntimeError(
        "SCBA failed to converge after %d iterations; abs residual %.6e eV, "
        "relative residual %.6e, dG %.6e, mixing %.5f"
        % (
            controls.max_iterations,
            absolute_residual,
            relative_residual,
            previous_green_change,
            mixing,
        )
    )


def solve_full_matrix_regression(
    model: ModelParams,
    reduced: ScbaResult,
    numerics: Optional[ScbaNumerics] = None,
    seed_asymmetry: float = 1.0e-3,
    verbose: bool = False,
) -> FullMatrixRegression:
    """Solve both site self-energies from a deliberately asymmetric seed."""

    controls = numerics if numerics is not None else ScbaNumerics()
    controls.validate()
    if not 0.0 < seed_asymmetry < 0.5:
        raise ValueError("seed_asymmetry must lie between zero and 0.5")
    omega = reduced.omega_eV
    kernel = DiscreteModeKernel(model, omega)
    base = reduced.sigma_eV
    if float(np.max(np.abs(base))) == 0.0:
        perturbation = -1.0j * seed_asymmetry * model.eta_eV
        sigma1 = base + perturbation
        sigma2 = base + 2.0 * perturbation
    else:
        sigma1 = (1.0 + seed_asymmetry) * base
        sigma2 = (1.0 - seed_asymmetry) * base

    bright, dark, _ = full_bright_dark_green(omega, sigma1, sigma2, model)
    old_green = np.concatenate((bright, dark))
    previous_residual = None  # type: Optional[float]
    previous_green_change = 1.0
    mixing = controls.initial_mixing
    stable_iterations = 0
    history_rows = []

    for iteration in range(1, controls.max_iterations + 1):
        mapped1, mapped2 = kernel.fan_map_full(sigma1, sigma2)
        residual1 = mapped1 - sigma1
        residual2 = mapped2 - sigma2
        absolute_residual = float(
            max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
        )
        relative_residual = max(
            _relative_residual(residual1, mapped1, sigma1, model.eta_eV),
            _relative_residual(residual2, mapped2, sigma2, model.eta_eV),
        )
        history_rows.append(
            [
                float(iteration),
                absolute_residual,
                relative_residual,
                previous_green_change,
                mixing,
                float(max(np.max(sigma1.imag), np.max(sigma2.imag))),
            ]
        )
        residual_ok = (
            absolute_residual < controls.absolute_residual_tolerance_eV
            and relative_residual < controls.relative_residual_tolerance
        )
        green_ok = previous_green_change < controls.green_change_tolerance
        stable_iterations = stable_iterations + 1 if residual_ok and green_ok else 0
        if stable_iterations >= controls.consecutive_converged:
            break

        if previous_residual is not None:
            if absolute_residual > controls.residual_growth_factor * previous_residual:
                mixing = max(controls.minimum_mixing, 0.5 * mixing)
            elif absolute_residual < controls.rapid_decrease_factor * previous_residual:
                mixing = min(controls.maximum_mixing, controls.mixing_growth * mixing)
        sigma1_new = sigma1 + mixing * residual1
        sigma2_new = sigma2 + mixing * residual2
        bright_new, dark_new, _ = full_bright_dark_green(
            omega, sigma1_new, sigma2_new, model
        )
        new_green = np.concatenate((bright_new, dark_new))
        previous_green_change = _relative_array_change(new_green, old_green)
        sigma1 = sigma1_new
        sigma2 = sigma2_new
        old_green = new_green
        previous_residual = absolute_residual

        if verbose and (iteration == 1 or iteration % 100 == 0):
            print(
                "Full regression %d: abs=%.3e rel=%.3e dG=%.3e mix=%.4f"
                % (
                    iteration,
                    absolute_residual,
                    relative_residual,
                    previous_green_change,
                    mixing,
                )
            )
    else:
        raise RuntimeError(
            "Full matrix SCBA failed to converge after %d iterations"
            % controls.max_iterations
        )

    bright, dark, local = full_bright_dark_green(omega, sigma1, sigma2, model)
    finite_or_raise("full matrix SCBA", [sigma1, sigma2, bright, dark, local])
    _validate_causality(
        "full matrix site 1",
        sigma1,
        bright,
        dark,
        controls.causality_tolerance,
    )
    _validate_causality(
        "full matrix site 2",
        sigma2,
        bright,
        dark,
        controls.causality_tolerance,
    )
    mean_sigma = 0.5 * (sigma1 + sigma2)
    sigma_scale = max(
        model.eta_eV,
        float(np.max(np.abs(mean_sigma))),
        float(np.max(np.abs(reduced.sigma_eV))),
    )
    max_site_difference = float(np.max(np.abs(sigma1 - sigma2)))
    max_mean_difference = float(np.max(np.abs(mean_sigma - reduced.sigma_eV)))
    relative_site_difference = max_site_difference / sigma_scale
    relative_mean_difference = max_mean_difference / sigma_scale
    relative_bright_difference = _relative_array_change(
        bright, reduced.green_bright_per_eV
    )
    relative_dark_difference = _relative_array_change(
        dark, reduced.green_dark_per_eV
    )

    # This checks the algebra of the two implementations at exactly the same
    # symmetric input, independently of the nonlinear stopping tolerance.
    symmetric_map1, symmetric_map2 = kernel.fan_map_full(
        reduced.sigma_eV, reduced.sigma_eV
    )
    symmetric_reduced_map = kernel.fan_map_reduced(reduced.sigma_eV)
    symmetric_map_error = float(
        max(
            np.max(np.abs(symmetric_map1 - symmetric_reduced_map)),
            np.max(np.abs(symmetric_map2 - symmetric_reduced_map)),
        )
    )
    symmetric_bright, symmetric_dark, _ = full_bright_dark_green(
        omega, reduced.sigma_eV, reduced.sigma_eV, model
    )
    symmetric_green_error = float(
        max(
            np.max(np.abs(symmetric_bright - reduced.green_bright_per_eV)),
            np.max(np.abs(symmetric_dark - reduced.green_dark_per_eV)),
        )
    )
    roundoff_scale = 500.0 * np.finfo(float).eps
    if symmetric_map_error > roundoff_scale * max(
        1.0, float(np.max(np.abs(symmetric_reduced_map)))
    ):
        raise RuntimeError("Full/reduced Fan maps fail their algebraic identity")
    if symmetric_green_error > roundoff_scale * max(
        1.0,
        float(np.max(np.abs(reduced.green_bright_per_eV))),
        float(np.max(np.abs(reduced.green_dark_per_eV))),
    ):
        raise RuntimeError("Full/reduced Dyson projections fail their algebraic identity")

    allowed = max(
        5.0e-6,
        50.0 * controls.relative_residual_tolerance,
        50.0 * controls.green_change_tolerance,
    )
    if max(
        relative_site_difference,
        relative_mean_difference,
        relative_bright_difference,
        relative_dark_difference,
    ) > allowed:
        raise RuntimeError(
            "Full/reduced SCBA regression failed: maximum relative difference "
            "%.6e exceeds %.6e"
            % (
                max(
                    relative_site_difference,
                    relative_mean_difference,
                    relative_bright_difference,
                    relative_dark_difference,
                ),
                allowed,
            )
        )

    return FullMatrixRegression(
        sigma_site1_eV=sigma1,
        sigma_site2_eV=sigma2,
        green_bright_per_eV=bright,
        green_dark_per_eV=dark,
        green_local_per_eV=local,
        history=np.asarray(history_rows),
        iterations=iteration,
        converged=True,
        fixed_point_abs_residual_eV=absolute_residual,
        fixed_point_relative_residual=relative_residual,
        green_relative_change=previous_green_change,
        final_mixing=mixing,
        max_site_sigma_difference_eV=max_site_difference,
        relative_site_sigma_difference=relative_site_difference,
        max_mean_sigma_minus_reduced_eV=max_mean_difference,
        relative_mean_sigma_difference=relative_mean_difference,
        relative_bright_green_difference=relative_bright_difference,
        relative_dark_green_difference=relative_dark_difference,
        symmetric_map_identity_max_error_eV=symmetric_map_error,
        symmetric_green_identity_max_error_per_eV=symmetric_green_error,
    )


def load_seed(path: Path, model: ModelParams, target_grid_eV: np.ndarray) -> np.ndarray:
    """Load a saved SCBA self-energy and safely transfer it to a new grid."""

    with np.load(str(path), allow_pickle=False) as archive:
        if "omega_eV" not in archive or "scba_sigma_eV" not in archive:
            raise ValueError(
                "Seed archive must contain omega_eV and scba_sigma_eV"
            )
        old_grid = np.asarray(archive["omega_eV"], dtype=float)
        old_sigma = np.asarray(archive["scba_sigma_eV"], dtype=complex)
    _validate_grid(old_grid)
    return interpolate_sigma_with_tail(
        target_grid_eV, old_grid, old_sigma, model
    )


def _result_arrays(prefix: str, result: ScbaResult) -> Dict[str, np.ndarray]:
    return {
        prefix + "_sigma_eV": result.sigma_eV,
        prefix + "_green_bright_per_eV": result.green_bright_per_eV,
        prefix + "_green_dark_per_eV": result.green_dark_per_eV,
        prefix + "_green_local_per_eV": result.green_local_per_eV,
        prefix + "_spectrum_bright_per_eV": result.spectrum_bright_per_eV,
        prefix + "_spectrum_dark_per_eV": result.spectrum_dark_per_eV,
        prefix + "_spectrum_local_per_eV": result.spectrum_local_per_eV,
        prefix
        + "_sigma_from_bright_emission_eV": result.sigma_from_bright_emission_eV,
        prefix
        + "_sigma_from_bright_absorption_eV": result.sigma_from_bright_absorption_eV,
        prefix
        + "_sigma_from_dark_emission_eV": result.sigma_from_dark_emission_eV,
        prefix
        + "_sigma_from_dark_absorption_eV": result.sigma_from_dark_absorption_eV,
        prefix + "_history": result.history,
    }


def save_case(
    output_path: Path,
    model: ModelParams,
    grid_spec: GridSpec,
    numerics: ScbaNumerics,
    ba: ScbaResult,
    scba: ScbaResult,
    full_regression: Optional[FullMatrixRegression] = None,
) -> Path:
    """Write a self-describing compressed BA/SCBA archive."""

    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "omega_eV": scba.omega_eV,
        "analysis_mask": grid_spec.analysis_mask(scba.omega_eV),
        "history_columns": np.asarray(HISTORY_COLUMNS),
    }
    arrays.update(_result_arrays("ba", ba))
    arrays.update(_result_arrays("scba", scba))
    metadata = {
        "format": "single_mode_dimer_ba_scba_v1",
        "model": model_dict(model),
        "grid": asdict(grid_spec),
        "numerics": asdict(numerics),
        "tail_extension": "causal first-Born asymptote, no periodic wrap",
        "phonon_linewidth": "none; eta is only the electronic retarded HWHM",
        "raw_bright_sum_rule_target": float(np.pi),
        "ba": ba.diagnostic_dict(),
        "scba": scba.diagnostic_dict(),
        "full_matrix_regression": (
            None if full_regression is None else full_regression.diagnostic_dict()
        ),
    }
    if full_regression is not None:
        arrays.update(
            {
                "full_sigma_site1_eV": full_regression.sigma_site1_eV,
                "full_sigma_site2_eV": full_regression.sigma_site2_eV,
                "full_green_bright_per_eV": full_regression.green_bright_per_eV,
                "full_green_dark_per_eV": full_regression.green_dark_per_eV,
                "full_green_local_per_eV": full_regression.green_local_per_eV,
                "full_history": full_regression.history,
            }
        )
    arrays["metadata_json"] = encode_metadata(metadata)
    np.savez_compressed(str(output), **arrays)
    return output.resolve()


def default_output_path(model: ModelParams) -> Path:
    name = (
        "single_mode_scba_w%smeV_S%s_J%smeV_T%sK_eta%smeV.npz"
        % (
            signed_tag(model.omega0_meV),
            signed_tag(model.huang_rhys),
            signed_tag(model.j_meV),
            signed_tag(model.temperature_K),
            signed_tag(model.eta_meV),
        )
    )
    return SPECTRA / name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute discrete-mode BA and SCBA spectra for a Frenkel dimer."
    )
    parser.add_argument("--energy-00-ev", type=float, default=2.0)
    parser.add_argument("--omega0-mev", type=float, default=24.8)
    coupling = parser.add_mutually_exclusive_group()
    coupling.add_argument("--huang-rhys", type=float, default=None)
    coupling.add_argument("--lambda-mev", type=float, default=None)
    parser.add_argument("--j-mev", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--eta-mev", type=float, default=0.5)
    parser.add_argument("--step-mev", type=float, default=0.0)
    parser.add_argument("--padding-ev", type=float, default=0.55)
    parser.add_argument("--analysis-min-ev", type=float, default=None)
    parser.add_argument("--analysis-max-ev", type=float, default=None)
    parser.add_argument("--internal-min-ev", type=float, default=None)
    parser.add_argument("--internal-max-ev", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--mixing", type=float, default=0.08)
    parser.add_argument("--minimum-mixing", type=float, default=0.002)
    parser.add_argument("--maximum-mixing", type=float, default=0.40)
    parser.add_argument("--abs-tolerance-ev", type=float, default=1.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--green-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=Path, default=None)
    parser.add_argument(
        "--force-iteration",
        action="store_true",
        help="At T=0, iterate instead of using the ascending reference sweep.",
    )
    parser.add_argument(
        "--skip-full-regression",
        action="store_true",
        help="Skip the independently iterated two-site Dyson regression.",
    )
    parser.add_argument("--seed-asymmetry", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _grid_from_args(model: ModelParams, args: argparse.Namespace) -> GridSpec:
    base = default_grid_spec(model, args.step_mev, args.padding_ev)
    return GridSpec(
        analysis_min_eV=(
            base.analysis_min_eV
            if args.analysis_min_ev is None
            else args.analysis_min_ev
        ),
        analysis_max_eV=(
            base.analysis_max_eV
            if args.analysis_max_ev is None
            else args.analysis_max_ev
        ),
        internal_min_eV=(
            base.internal_min_eV
            if args.internal_min_ev is None
            else args.internal_min_ev
        ),
        internal_max_eV=(
            base.internal_max_eV
            if args.internal_max_ev is None
            else args.internal_max_ev
        ),
        step_meV=base.step_meV,
    )


def main() -> None:
    args = _build_parser().parse_args()
    huang_rhys = (
        30.0 / args.omega0_mev
        if args.huang_rhys is None and args.lambda_mev is None
        else (
            args.lambda_mev / args.omega0_mev
            if args.lambda_mev is not None
            else args.huang_rhys
        )
    )
    model = ModelParams(
        energy_00_eV=args.energy_00_ev,
        omega0_meV=args.omega0_mev,
        huang_rhys=float(huang_rhys),
        j_meV=args.j_mev,
        temperature_K=args.temperature_k,
        eta_meV=args.eta_mev,
    )
    model.validate()
    grid_spec = _grid_from_args(model, args)
    numerics = ScbaNumerics(
        max_iterations=args.max_iterations,
        initial_mixing=args.mixing,
        minimum_mixing=args.minimum_mixing,
        maximum_mixing=args.maximum_mixing,
        absolute_residual_tolerance_eV=args.abs_tolerance_ev,
        relative_residual_tolerance=args.relative_tolerance,
        green_change_tolerance=args.green_tolerance,
    )
    numerics.validate()
    omega = grid_spec.grid()
    validate_grid_spec(grid_spec, omega)
    seed = None
    if args.seed is not None:
        seed = load_seed(args.seed, model, omega)

    ba = solve_single_shot_ba(model, grid_spec, numerics)
    scba = solve_scba(
        model,
        grid_spec,
        numerics,
        seed=seed,
        use_zero_temperature_sweep=not args.force_iteration,
        verbose=args.verbose,
    )
    full = None
    if not args.skip_full_regression:
        full = solve_full_matrix_regression(
            model,
            scba,
            numerics,
            seed_asymmetry=args.seed_asymmetry,
            verbose=args.verbose,
        )

    ensure_directories()
    output = args.output if args.output is not None else default_output_path(model)
    saved = save_case(output, model, grid_spec, numerics, ba, scba, full)
    analysis = grid_spec.analysis_mask(scba.omega_eV)
    ba_area = float(np.trapz(ba.spectrum_bright_per_eV[analysis], scba.omega_eV[analysis]))
    scba_area = float(
        np.trapz(scba.spectrum_bright_per_eV[analysis], scba.omega_eV[analysis])
    )
    summary = {
        "output": str(saved),
        "n_omega": int(len(scba.omega_eV)),
        "grid_step_meV": float(
            (scba.omega_eV[1] - scba.omega_eV[0]) * 1.0e3
        ),
        "ba_area_analysis": ba_area,
        "scba_area_analysis": scba_area,
        "scba": scba.diagnostic_dict(),
        "full_matrix_regression": (
            None if full is None else full.diagnostic_dict()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
