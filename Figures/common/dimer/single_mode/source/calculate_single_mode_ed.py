#!/usr/bin/env python3
"""Factorized exact diagonalization for the single-mode Frenkel dimer.

The two identical local oscillators are transformed to common and difference
modes.  In the one-excitation sector the Hamiltonian then separates into an
exactly displaced common oscillator and a ``2 * N_vib`` difference-mode
Hamiltonian.  This reduces the production diagonalization from dimension
``2 * N_vib**2`` to a single small real-symmetric eigenproblem.

Finite-temperature bright and dark Green functions are evaluated from the
Lehmann representation.  Pole weights of the displaced common oscillator are
analytic; difference-mode pole weights come from dense diagonalization.  The
two positive pole measures are combined by conservative linear deposition on
a uniform grid and broadened by a *linear* (non-circular) FFT convolution with
the retarded Lorentz kernel.  The Lorentz half-width is the same numerical
resolution parameter used by SCBA, not a phonon damping rate.

All energies are in eV internally and hbar=1.  The module can be imported by
the scan driver or run directly, from this directory or the repository root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from math import comb
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import fftconvolve
from scipy.special import eval_genlaguerre, gammaln


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from single_mode_common import (  # noqa: E402
    GridSpec,
    KB_EV_PER_K,
    ModelParams,
    ROOT,
    SPECTRA,
    default_grid_spec,
    encode_metadata,
    ensure_directories,
    finite_or_raise,
    model_dict,
    recommended_n_vib,
    signed_tag,
    thermal_cutoff,
)


@dataclass(frozen=True)
class PoleMeasure:
    """A positive discrete spectral measure."""

    energy_eV: np.ndarray
    weight: np.ndarray
    discarded_weight: float = 0.0

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.weight, dtype=np.float64))


@dataclass
class EDResult:
    """Broadened Green functions and convergence diagnostics for one case."""

    energy_eV: np.ndarray
    green_bright: np.ndarray
    green_dark: np.ndarray
    green_local: np.ndarray
    diagnostics: Dict[str, Any]

    @property
    def omega_eV(self) -> np.ndarray:
        """SCBA-compatible name for the common energy grid."""

        return self.energy_eV

    @property
    def green_bright_per_eV(self) -> np.ndarray:
        return self.green_bright

    @property
    def green_dark_per_eV(self) -> np.ndarray:
        return self.green_dark

    @property
    def green_local_per_eV(self) -> np.ndarray:
        return self.green_local

    @property
    def spectrum_bright(self) -> np.ndarray:
        return -np.imag(self.green_bright)

    @property
    def spectrum_dark(self) -> np.ndarray:
        return -np.imag(self.green_dark)

    @property
    def spectrum_local(self) -> np.ndarray:
        return -np.imag(self.green_local)

    @property
    def spectrum_bright_per_eV(self) -> np.ndarray:
        return self.spectrum_bright

    @property
    def spectrum_dark_per_eV(self) -> np.ndarray:
        return self.spectrum_dark

    @property
    def spectrum_local_per_eV(self) -> np.ndarray:
        return self.spectrum_local

    def diagnostic_dict(self) -> Dict[str, Any]:
        return dict(self.diagnostics)


def _thermal_probabilities(
    model: ModelParams,
    n_thermal: int,
) -> Tuple[np.ndarray, float, float]:
    """Return unrenormalized one-mode probabilities and joint omission."""

    if n_thermal < 1:
        raise ValueError("n_thermal must be at least one")
    if model.temperature_K <= 0.0:
        if n_thermal != 1:
            # Extra zero-probability states are harmless, but accepting them
            # tends to conceal a caller error and wastes transition storage.
            raise ValueError("Use n_thermal=1 at zero temperature")
        return np.ones(1), 0.0, 0.0

    q = float(
        np.exp(-model.omega0_eV / (KB_EV_PER_K * model.temperature_K))
    )
    probabilities = (1.0 - q) * q ** np.arange(n_thermal, dtype=float)
    retained_one_mode = float(1.0 - q**n_thermal)
    omitted_joint = float(1.0 - retained_one_mode**2)
    return probabilities, q, omitted_joint


def displaced_number_overlap_squared(
    initial_n: int,
    final_m: np.ndarray,
    displacement_squared: float,
) -> np.ndarray:
    r"""Return ``|<m|D(alpha)|n>|^2`` for real ``alpha``.

    ``displacement_squared`` is :math:`\alpha^2`.  The generalized-Laguerre
    expression is the infinite-oscillator result, so the common mode has no
    artificial hard-wall eigenvalue error.  ``N_vib`` only truncates the set
    of final states retained in its Lehmann sum.
    """

    if initial_n < 0:
        raise ValueError("initial_n must be nonnegative")
    final = np.asarray(final_m, dtype=int)
    if np.any(final < 0):
        raise ValueError("final oscillator indices must be nonnegative")
    if displacement_squared < 0.0 or not np.isfinite(displacement_squared):
        raise ValueError("displacement_squared must be finite and nonnegative")

    if displacement_squared == 0.0:
        return (final == initial_n).astype(float)

    low = np.minimum(final, initial_n)
    high = np.maximum(final, initial_n)
    order = high - low
    laguerre = eval_genlaguerre(low, order, displacement_squared)
    log_prefactor = (
        -displacement_squared
        + gammaln(low + 1.0)
        - gammaln(high + 1.0)
        + order * np.log(displacement_squared)
    )
    weights = np.exp(log_prefactor) * np.square(laguerre)
    # Roundoff can only produce a negative value through an upstream NaN;
    # keep that detectable instead of silently clipping a real failure.
    if not np.all(np.isfinite(weights)) or float(np.min(weights)) < 0.0:
        raise RuntimeError("Non-finite displaced-oscillator overlap weight")
    return weights


def common_mode_measure(
    model: ModelParams,
    n_vib: int,
    thermal_probabilities: np.ndarray,
) -> PoleMeasure:
    """Analytic common-mode transition measure.

    Its Huang--Rhys factor is ``S/2`` and its excited-sector eigenenergies are
    ``m*omega0 - lambda/2``.  Initial ground-sector energies ``n*omega0`` are
    subtracted explicitly in the Lehmann transition energies.
    """

    if n_vib < len(thermal_probabilities):
        raise ValueError("n_vib must not be smaller than the thermal cutoff")
    final = np.arange(n_vib, dtype=int)
    energies: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    displacement_squared = 0.5 * model.huang_rhys
    shift = -0.5 * model.reorganization_eV

    for initial_n, probability in enumerate(thermal_probabilities):
        overlap = displaced_number_overlap_squared(
            initial_n,
            final,
            displacement_squared,
        )
        energies.append(
            shift + (final.astype(float) - float(initial_n)) * model.omega0_eV
        )
        weights.append(float(probability) * overlap)

    return PoleMeasure(np.concatenate(energies), np.concatenate(weights))


def difference_mode_eigensystem(
    model: ModelParams,
    n_vib: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the bright/dark difference-mode Hamiltonian."""

    if n_vib < 1:
        raise ValueError("n_vib must be positive")
    number = np.diag(np.arange(n_vib, dtype=float))
    raising = np.diag(np.sqrt(np.arange(1, n_vib, dtype=float)), 1)
    coordinate = raising + raising.T
    identity_e = np.eye(2)
    identity_v = np.eye(n_vib)
    tau_z = np.diag([1.0, -1.0])
    tau_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    coupling = model.coupling_eV / np.sqrt(2.0)

    hamiltonian = (
        np.kron(
            model.vertical_energy_eV * identity_e + model.j_eV * tau_z,
            identity_v,
        )
        + np.kron(identity_e, model.omega0_eV * number)
        + coupling * np.kron(tau_x, coordinate)
    )
    hermiticity_error = float(
        np.max(np.abs(hamiltonian - hamiltonian.T.conjugate()))
    )
    if hermiticity_error > 1.0e-13:
        raise RuntimeError(
            "Difference-mode Hamiltonian is not Hermitian: "
            f"{hermiticity_error:.3e}"
        )
    eigenvalues, eigenvectors = eigh(
        hamiltonian,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    return eigenvalues, eigenvectors


def difference_mode_measures(
    model: ModelParams,
    n_vib: int,
    thermal_probabilities: np.ndarray,
) -> Tuple[PoleMeasure, PoleMeasure, Dict[str, float]]:
    """Return bright and dark transition measures of the difference mode."""

    if n_vib < len(thermal_probabilities):
        raise ValueError("n_vib must not be smaller than the thermal cutoff")
    eigenvalues, eigenvectors = difference_mode_eigensystem(model, n_vib)
    n_thermal = len(thermal_probabilities)
    initial = np.arange(n_thermal)
    transition_energy = (
        eigenvalues[None, :]
        - model.omega0_eV * initial.astype(float)[:, None]
    )
    bright_weight = (
        thermal_probabilities[:, None]
        * np.square(np.abs(eigenvectors[initial, :]))
    )
    dark_weight = (
        thermal_probabilities[:, None]
        * np.square(np.abs(eigenvectors[n_vib + initial, :]))
    )
    expected = float(np.sum(thermal_probabilities))
    diagnostics = {
        "difference_bright_weight": float(np.sum(bright_weight)),
        "difference_dark_weight": float(np.sum(dark_weight)),
        "difference_expected_weight": expected,
        "difference_bright_completeness_error": float(
            abs(np.sum(bright_weight) - expected)
        ),
        "difference_dark_completeness_error": float(
            abs(np.sum(dark_weight) - expected)
        ),
        "difference_lowest_eigenvalue_eV": float(eigenvalues[0]),
        "difference_highest_eigenvalue_eV": float(eigenvalues[-1]),
    }
    return (
        PoleMeasure(transition_energy.ravel(), bright_weight.ravel()),
        PoleMeasure(transition_energy.ravel(), dark_weight.ravel()),
        diagnostics,
    )


def _trim_measure(measure: PoleMeasure, tolerance: float) -> PoleMeasure:
    """Drop at most ``tolerance`` total weight, never per-pole thresholding."""

    if tolerance <= 0.0 or measure.weight.size == 0:
        return measure
    if np.any(measure.weight < 0.0):
        raise ValueError("A pole measure contains negative weights")
    order = np.argsort(measure.weight, kind="stable")
    cumulative = np.cumsum(measure.weight[order], dtype=np.float64)
    number_to_drop = int(np.searchsorted(cumulative, tolerance, side="right"))
    if number_to_drop == 0:
        return measure
    dropped = order[:number_to_drop]
    keep = np.ones(measure.weight.size, dtype=bool)
    keep[dropped] = False
    discarded = float(np.sum(measure.weight[dropped], dtype=np.float64))
    return PoleMeasure(
        measure.energy_eV[keep],
        measure.weight[keep],
        measure.discarded_weight + discarded,
    )


def _validate_uniform_grid(energy: np.ndarray) -> float:
    if energy.ndim != 1 or energy.size < 3:
        raise ValueError("The ED energy grid must be one-dimensional")
    differences = np.diff(energy)
    # Use the end-to-end spacing rather than one floating-point subtraction.
    # GridSpec constructs points as origin + index*step; the former recovers
    # that lattice spacing without accumulating the rounding error of the
    # first adjacent pair across tens of thousands of samples.
    step = float((energy[-1] - energy[0]) / (energy.size - 1))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(step))
    if step <= 0.0 or float(np.max(np.abs(differences - step))) > tolerance:
        raise ValueError("The ED energy grid must be strictly uniform")
    return step


def _product_energy_bounds(
    common: PoleMeasure,
    differences: Sequence[PoleMeasure],
) -> Tuple[float, float]:
    if common.energy_eV.size == 0 or any(x.energy_eV.size == 0 for x in differences):
        raise ValueError("Cannot broaden an empty pole measure")
    lower = min(
        float(np.min(common.energy_eV) + np.min(measure.energy_eV))
        for measure in differences
    )
    upper = max(
        float(np.max(common.energy_eV) + np.max(measure.energy_eV))
        for measure in differences
    )
    return lower, upper


def _full_lattice(
    requested_grid: np.ndarray,
    pole_bounds: Tuple[float, float],
    refinement: int,
) -> Tuple[np.ndarray, int]:
    """Return a grid-aligned lattice containing poles and requested samples."""

    requested_step = _validate_uniform_grid(requested_grid)
    if refinement < 1:
        raise ValueError("deposition refinement must be positive")
    step = requested_step / refinement
    relative_lower = (pole_bounds[0] - requested_grid[0]) / step
    relative_upper = (pole_bounds[1] - requested_grid[0]) / step
    lower_index = min(0, int(np.floor(relative_lower)) - 1)
    upper_index = max(
        refinement * (requested_grid.size - 1),
        int(np.ceil(relative_upper)) + 1,
    )
    full = requested_grid[0] + step * np.arange(
        lower_index,
        upper_index + 1,
        dtype=float,
    )
    requested_start = -lower_index
    if not np.allclose(
        full[
            requested_start : requested_start
            + refinement * requested_grid.size : refinement
        ],
        requested_grid,
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps * max(1.0, abs(requested_grid[-1])),
    ):
        raise RuntimeError("Failed to align the pole and requested grids")
    return full, requested_start


def _deposit_product_measure(
    common: PoleMeasure,
    difference: PoleMeasure,
    lattice: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Conservatively deposit all pairwise sums with linear interpolation."""

    step = float(lattice[1] - lattice[0])
    mass = np.zeros(lattice.size, dtype=float)
    for start in range(0, common.energy_eV.size, batch_size):
        stop = min(start + batch_size, common.energy_eV.size)
        energies = (
            common.energy_eV[start:stop, None]
            + difference.energy_eV[None, :]
        ).ravel()
        weights = (
            common.weight[start:stop, None] * difference.weight[None, :]
        ).ravel()
        coordinate = (energies - lattice[0]) / step
        lower = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - lower
        if int(np.min(lower)) < 0 or int(np.max(lower + 1)) >= lattice.size:
            raise RuntimeError("Pole deposition exceeded the auxiliary lattice")
        mass += np.bincount(
            lower,
            weights=weights * (1.0 - fraction),
            minlength=lattice.size,
        )[: lattice.size]
        mass += np.bincount(
            lower + 1,
            weights=weights * fraction,
            minlength=lattice.size,
        )[: lattice.size]
    return mass


def _linear_fft_retarded_green(
    mass: np.ndarray,
    step_eV: float,
    eta_eV: float,
) -> np.ndarray:
    r"""Return ``sum_j mass_j/(omega-E_j+i eta)`` on the mass lattice."""

    number = mass.size
    offsets = np.arange(-(number - 1), number, dtype=float) * step_eV
    kernel = 1.0 / (offsets + 1.0j * eta_eV)
    # Full convolution followed by the analytically determined slice is a
    # zero-padded linear convolution.  No periodic image can enter G.
    convolved = fftconvolve(mass, kernel, mode="full")
    start = number - 1
    return convolved[start : start + number]


def broaden_factorized_measure(
    common: PoleMeasure,
    difference: PoleMeasure,
    requested_grid: np.ndarray,
    eta_eV: float,
    pole_bounds: Optional[Tuple[float, float]] = None,
    deposition_refinement: int = 4,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Deposit, linearly broaden, and return a factorized pole measure."""

    if eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")
    if pole_bounds is None:
        pole_bounds = _product_energy_bounds(common, (difference,))
    lattice, requested_start = _full_lattice(
        requested_grid,
        pole_bounds,
        deposition_refinement,
    )
    mass = _deposit_product_measure(common, difference, lattice)
    green_full = _linear_fft_retarded_green(
        mass,
        float(lattice[1] - lattice[0]),
        eta_eV,
    )
    green = green_full[
        requested_start : requested_start
        + deposition_refinement * requested_grid.size : deposition_refinement
    ]
    expected_deposited = common.total_weight * difference.total_weight
    diagnostics = {
        "deposited_pole_weight": float(np.sum(mass)),
        "expected_deposited_pole_weight": expected_deposited,
        "deposition_weight_error": float(abs(np.sum(mass) - expected_deposited)),
        "auxiliary_grid_min_eV": float(lattice[0]),
        "auxiliary_grid_max_eV": float(lattice[-1]),
        "auxiliary_grid_samples": int(lattice.size),
        "deposition_refinement": int(deposition_refinement),
        "deposition_step_eV": float(lattice[1] - lattice[0]),
    }
    return green, float(np.sum(mass)), diagnostics


def direct_lorentz_green(
    energy_eV: np.ndarray,
    pole_energy_eV: np.ndarray,
    pole_weight: np.ndarray,
    eta_eV: float,
    chunk_size: int = 64,
) -> np.ndarray:
    """Reference Lehmann sum used only by small regression calculations."""

    result = np.zeros(energy_eV.size, dtype=complex)
    for start in range(0, pole_energy_eV.size, chunk_size):
        stop = min(start + chunk_size, pole_energy_eV.size)
        result += np.sum(
            pole_weight[None, start:stop]
            / (
                energy_eV[:, None]
                - pole_energy_eV[None, start:stop]
                + 1.0j * eta_eV
            ),
            axis=1,
        )
    return result


def _combined_measure(
    common: PoleMeasure,
    difference: PoleMeasure,
) -> PoleMeasure:
    """Materialize pairwise poles for small direct-summation tests only."""

    energies = (
        common.energy_eV[:, None] + difference.energy_eV[None, :]
    ).ravel()
    weights = (common.weight[:, None] * difference.weight[None, :]).ravel()
    return PoleMeasure(energies, weights)


def _product_raw_moments(
    first: PoleMeasure,
    second: PoleMeasure,
    maximum_order: int = 3,
) -> List[float]:
    """Normalized unbroadened moments of the convolution measure."""

    raw_first = [
        float(np.sum(first.weight * first.energy_eV**order))
        for order in range(maximum_order + 1)
    ]
    raw_second = [
        float(np.sum(second.weight * second.energy_eV**order))
        for order in range(maximum_order + 1)
    ]
    product = []
    for order in range(maximum_order + 1):
        value = 0.0
        for left_order in range(order + 1):
            coefficient = float(comb(order, left_order))
            value += (
                coefficient
                * raw_first[left_order]
                * raw_second[order - left_order]
            )
        product.append(value)
    if product[0] <= 0.0:
        raise RuntimeError("The ED pole measure has zero total weight")
    return [value / product[0] for value in product]


def solve_ed(
    model: ModelParams,
    grid_spec: GridSpec,
    n_vib: Optional[int] = None,
    thermal_tolerance: float = 1.0e-10,
    n_thermal: Optional[int] = None,
    stick_weight_tolerance: float = 1.0e-14,
    deposition_refinement: int = 4,
) -> EDResult:
    """Calculate finite-temperature bright, dark, and local ED spectra."""

    model.validate()
    if stick_weight_tolerance < 0.0 or not np.isfinite(stick_weight_tolerance):
        raise ValueError("stick_weight_tolerance must be finite and nonnegative")
    if deposition_refinement < 1:
        raise ValueError("deposition_refinement must be at least one")
    energy = grid_spec.grid()
    _validate_uniform_grid(energy)
    if n_thermal is None:
        n_thermal, thermal_omitted = thermal_cutoff(model, thermal_tolerance)
    else:
        if n_thermal < 1:
            raise ValueError("n_thermal must be positive")
        _, _, thermal_omitted = _thermal_probabilities(model, n_thermal)
    if n_vib is None:
        n_vib = recommended_n_vib(model, n_thermal)
    if n_vib < n_thermal:
        raise ValueError(
            f"n_vib={n_vib} is smaller than n_thermal={n_thermal}"
        )
    probabilities, q, thermal_omitted_direct = _thermal_probabilities(
        model,
        n_thermal,
    )
    if abs(thermal_omitted - thermal_omitted_direct) > 1.0e-14:
        raise RuntimeError("Inconsistent thermal omitted-weight calculation")

    common_full = common_mode_measure(model, n_vib, probabilities)
    difference_bright_full, difference_dark_full, difference_diagnostics = (
        difference_mode_measures(model, n_vib, probabilities)
    )
    expected_one_mode_weight = float(np.sum(probabilities))
    expected_joint_weight = expected_one_mode_weight**2
    raw_bright_weight = common_full.total_weight * difference_bright_full.total_weight
    raw_dark_weight = common_full.total_weight * difference_dark_full.total_weight

    bright_moments = _product_raw_moments(
        common_full,
        difference_bright_full,
    )
    dark_moments = _product_raw_moments(
        common_full,
        difference_dark_full,
    )

    common = _trim_measure(common_full, stick_weight_tolerance / 3.0)
    difference_bright = _trim_measure(
        difference_bright_full,
        stick_weight_tolerance / 3.0,
    )
    difference_dark = _trim_measure(
        difference_dark_full,
        stick_weight_tolerance / 3.0,
    )
    bounds = _product_energy_bounds(
        common,
        (difference_bright, difference_dark),
    )
    green_bright, deposited_bright, bright_deposition = broaden_factorized_measure(
        common,
        difference_bright,
        energy,
        model.eta_eV,
        pole_bounds=bounds,
        deposition_refinement=deposition_refinement,
    )
    green_dark, deposited_dark, dark_deposition = broaden_factorized_measure(
        common,
        difference_dark,
        energy,
        model.eta_eV,
        pole_bounds=bounds,
        deposition_refinement=deposition_refinement,
    )
    green_local = 0.5 * (green_bright + green_dark)
    finite_or_raise(
        "ED Green functions",
        (green_bright, green_dark, green_local),
    )

    analysis_mask = grid_spec.analysis_mask(energy)
    if np.count_nonzero(analysis_mask) < 2:
        raise ValueError("The analysis window contains fewer than two grid points")
    spectrum_bright = -np.imag(green_bright)
    spectrum_dark = -np.imag(green_dark)
    spectrum_local = -np.imag(green_local)
    analysis_area_bright = float(np.trapz(spectrum_bright[analysis_mask], energy[analysis_mask]))
    analysis_area_dark = float(np.trapz(spectrum_dark[analysis_mask], energy[analysis_mask]))
    internal_area_bright = float(np.trapz(spectrum_bright, energy))
    internal_area_dark = float(np.trapz(spectrum_dark, energy))

    expected_bright_mean = model.vertical_energy_eV + model.j_eV
    expected_dark_mean = model.vertical_energy_eV - model.j_eV
    diagnostics: Dict[str, Any] = {
        "method": "factorized_exact_common_plus_truncated_difference_ed",
        "n_vib": int(n_vib),
        "n_thermal": int(n_thermal),
        "thermal_q": q,
        "thermal_omitted_probability": thermal_omitted,
        "thermal_retained_joint_probability": expected_joint_weight,
        "common_retained_weight": common_full.total_weight,
        "common_final_basis_omitted_weight": float(
            expected_one_mode_weight - common_full.total_weight
        ),
        "raw_bright_pole_weight": raw_bright_weight,
        "raw_dark_pole_weight": raw_dark_weight,
        "raw_bright_pole_area": float(np.pi * raw_bright_weight),
        "raw_dark_pole_area": float(np.pi * raw_dark_weight),
        "raw_bright_missing_from_retained_thermal": float(
            expected_joint_weight - raw_bright_weight
        ),
        "raw_dark_missing_from_retained_thermal": float(
            expected_joint_weight - raw_dark_weight
        ),
        "common_discarded_for_fft": common.discarded_weight,
        "difference_bright_discarded_for_fft": difference_bright.discarded_weight,
        "difference_dark_discarded_for_fft": difference_dark.discarded_weight,
        "deposited_bright_weight": deposited_bright,
        "deposited_dark_weight": deposited_dark,
        "analysis_area_bright": analysis_area_bright,
        "analysis_area_dark": analysis_area_dark,
        "internal_grid_area_bright": internal_area_bright,
        "internal_grid_area_dark": internal_area_dark,
        "bright_stick_moment_1_eV": bright_moments[1],
        "bright_stick_moment_2_eV2": bright_moments[2],
        "bright_stick_moment_3_eV3": bright_moments[3],
        "dark_stick_moment_1_eV": dark_moments[1],
        "dark_stick_moment_2_eV2": dark_moments[2],
        "dark_stick_moment_3_eV3": dark_moments[3],
        "bright_first_moment_expected_eV": expected_bright_mean,
        "dark_first_moment_expected_eV": expected_dark_mean,
        "bright_first_moment_error_eV": float(
            bright_moments[1] - expected_bright_mean
        ),
        "dark_first_moment_error_eV": float(
            dark_moments[1] - expected_dark_mean
        ),
        "minimum_bright_spectrum": float(np.min(spectrum_bright)),
        "minimum_dark_spectrum": float(np.min(spectrum_dark)),
        "minimum_local_spectrum": float(np.min(spectrum_local)),
        "grid_step_eV": float(energy[1] - energy[0]),
        "grid_samples": int(energy.size),
        "analysis_samples": int(np.count_nonzero(analysis_mask)),
        "stick_weight_tolerance": stick_weight_tolerance,
        "deposition_refinement": int(deposition_refinement),
    }
    diagnostics.update(
        {f"bright_{key}": value for key, value in bright_deposition.items()}
    )
    diagnostics.update(
        {f"dark_{key}": value for key, value in dark_deposition.items()}
    )
    diagnostics.update(difference_diagnostics)

    causality_tolerance = 1.0e-10 * max(
        1.0,
        float(np.max(spectrum_bright)),
        float(np.max(spectrum_dark)),
    )
    if min(float(np.min(spectrum_bright)), float(np.min(spectrum_dark))) < -causality_tolerance:
        raise RuntimeError("ED FFT broadening produced a noncausal spectrum")
    if difference_diagnostics["difference_bright_completeness_error"] > 1.0e-12:
        raise RuntimeError("Bright difference-mode eigenvectors are incomplete")
    if difference_diagnostics["difference_dark_completeness_error"] > 1.0e-12:
        raise RuntimeError("Dark difference-mode eigenvectors are incomplete")

    return EDResult(
        energy_eV=energy,
        green_bright=green_bright,
        green_dark=green_dark,
        green_local=green_local,
        diagnostics=diagnostics,
    )


# Public synonym used by some scan scripts.
calculate_ed = solve_ed


def validate_direct_lorentz_regression() -> Dict[str, float]:
    """Compare conservative FFT broadening with a direct Lehmann sum."""

    model = ModelParams(
        omega0_meV=24.8,
        huang_rhys=0.5,
        j_meV=12.4,
        temperature_K=0.0,
        eta_meV=1.0,
    )
    grid_spec = GridSpec(1.80, 2.40, 1.80, 2.40, 0.05)
    energy = grid_spec.grid()
    probabilities = np.ones(1)
    common = common_mode_measure(model, 16, probabilities)
    bright, _, _ = difference_mode_measures(model, 16, probabilities)
    combined = _combined_measure(common, bright)
    direct = direct_lorentz_green(
        energy,
        combined.energy_eV,
        combined.weight,
        model.eta_eV,
    )
    fft_green, _, deposition = broaden_factorized_measure(
        common,
        bright,
        energy,
        model.eta_eV,
    )
    relative_complex_error = float(
        np.max(np.abs(fft_green - direct)) / np.max(np.abs(direct))
    )
    direct_spectrum = -np.imag(direct)
    fft_spectrum = -np.imag(fft_green)
    direct_area = float(np.trapz(direct_spectrum, energy))
    fft_area = float(np.trapz(fft_spectrum, energy))
    normalized_l1 = float(
        np.trapz(
            np.abs(fft_spectrum / fft_area - direct_spectrum / direct_area),
            energy,
        )
    )
    if relative_complex_error > 2.0e-4 or normalized_l1 > 2.0e-4:
        raise RuntimeError(
            "Direct Lorentz regression failed: "
            f"complex={relative_complex_error:.3e}, L1={normalized_l1:.3e}"
        )
    return {
        "relative_complex_error": relative_complex_error,
        "unit_area_l1": normalized_l1,
        "deposition_weight_error": deposition["deposition_weight_error"],
    }


def validate_zero_coupling() -> Dict[str, float]:
    """Validate the analytic bright and dark poles when S=0."""

    model = ModelParams(
        omega0_meV=24.8,
        huang_rhys=0.0,
        j_meV=24.8,
        temperature_K=0.0,
        eta_meV=1.0,
    )
    grid_spec = GridSpec(1.80, 2.20, 1.80, 2.20, 0.05)
    result = solve_ed(model, grid_spec, n_vib=8)
    expected_bright = 1.0 / (
        result.energy_eV
        - (model.vertical_energy_eV + model.j_eV)
        + 1.0j * model.eta_eV
    )
    expected_dark = 1.0 / (
        result.energy_eV
        - (model.vertical_energy_eV - model.j_eV)
        + 1.0j * model.eta_eV
    )
    bright_error = float(np.max(np.abs(result.green_bright - expected_bright)))
    dark_error = float(np.max(np.abs(result.green_dark - expected_dark)))
    bright_relative = bright_error / float(np.max(np.abs(expected_bright)))
    dark_relative = dark_error / float(np.max(np.abs(expected_dark)))
    if max(bright_relative, dark_relative) > 1.0e-9:
        raise RuntimeError(
            "S=0 validation failed: "
            f"B={bright_relative:.3e}, D={dark_relative:.3e}"
        )
    return {
        "bright_max_abs_error": bright_error,
        "dark_max_abs_error": dark_error,
        "bright_relative_error": bright_relative,
        "dark_relative_error": dark_relative,
    }


def _franck_condon_green(
    energy_eV: np.ndarray,
    model: ModelParams,
    tail_tolerance: float = 1.0e-15,
) -> Tuple[np.ndarray, float, int]:
    """Analytic zero-temperature independent-boson Green function."""

    weights = [float(np.exp(-model.huang_rhys))]
    cumulative = weights[0]
    sideband = 0
    while 1.0 - cumulative > tail_tolerance:
        sideband += 1
        weights.append(weights[-1] * model.huang_rhys / sideband)
        cumulative += weights[-1]
        if sideband > 10000:
            raise RuntimeError("Franck-Condon tail failed to converge")
    indices = np.arange(len(weights), dtype=float)
    poles = model.energy_00_eV + indices * model.omega0_eV
    green = direct_lorentz_green(
        energy_eV,
        poles,
        np.asarray(weights),
        model.eta_eV,
    )
    return green, float(1.0 - cumulative), len(weights)


def validate_j_zero_franck_condon() -> Dict[str, float]:
    """Validate J=0 against the exact one-site Franck-Condon progression."""

    model = ModelParams(
        omega0_meV=24.8,
        huang_rhys=1.2,
        j_meV=0.0,
        temperature_K=0.0,
        eta_meV=1.0,
    )
    grid_spec = GridSpec(1.75, 2.45, 1.75, 2.45, 0.05)
    result = solve_ed(model, grid_spec, n_vib=24)
    exact, tail, sidebands = _franck_condon_green(result.energy_eV, model)
    spectrum_ed = result.spectrum_bright
    spectrum_exact = -np.imag(exact)
    area_ed = float(np.trapz(spectrum_ed, result.energy_eV))
    area_exact = float(np.trapz(spectrum_exact, result.energy_eV))
    normalized_l1 = float(
        np.trapz(
            np.abs(spectrum_ed / area_ed - spectrum_exact / area_exact),
            result.energy_eV,
        )
    )
    peak_error = float(
        abs(
            result.energy_eV[int(np.argmax(spectrum_ed))]
            - result.energy_eV[int(np.argmax(spectrum_exact))]
        )
    )
    if normalized_l1 > 2.0e-3 or peak_error > grid_spec.step_eV:
        raise RuntimeError(
            "J=0 Franck-Condon validation failed: "
            f"L1={normalized_l1:.3e}, peak={peak_error:.3e} eV"
        )
    return {
        "unit_area_l1": normalized_l1,
        "dominant_peak_error_eV": peak_error,
        "analytic_tail_weight": tail,
        "analytic_sidebands": int(sidebands),
    }


def validate_signed_j_symmetry() -> Dict[str, float]:
    """Validate A_B(J)=A_D(-J) and the even local spectrum."""

    positive = ModelParams(
        omega0_meV=24.8,
        huang_rhys=1.0,
        j_meV=12.4,
        temperature_K=100.0,
        eta_meV=1.0,
    )
    negative = replace(positive, j_meV=-positive.j_meV)
    grid_spec = GridSpec(1.65, 2.45, 1.65, 2.45, 0.10)
    n_thermal, _ = thermal_cutoff(positive)
    n_vib = max(24, n_thermal + 12)
    plus = solve_ed(positive, grid_spec, n_vib=n_vib)
    minus = solve_ed(negative, grid_spec, n_vib=n_vib)
    scale = max(
        1.0,
        float(np.max(np.abs(plus.green_bright))),
        float(np.max(np.abs(plus.green_local))),
    )
    bright_dark_error = float(
        np.max(np.abs(plus.green_bright - minus.green_dark)) / scale
    )
    dark_bright_error = float(
        np.max(np.abs(plus.green_dark - minus.green_bright)) / scale
    )
    local_error = float(
        np.max(np.abs(plus.green_local - minus.green_local)) / scale
    )
    if max(bright_dark_error, dark_bright_error, local_error) > 1.0e-10:
        raise RuntimeError(
            "Signed-J symmetry validation failed: "
            f"BD={bright_dark_error:.3e}, DB={dark_bright_error:.3e}, "
            f"local={local_error:.3e}"
        )
    return {
        "bright_J_vs_dark_minusJ_relative_error": bright_dark_error,
        "dark_J_vs_bright_minusJ_relative_error": dark_bright_error,
        "local_even_relative_error": local_error,
    }


def _total_shell_operators(
    maximum_total_quanta: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Return two annihilation matrices in a rotation-invariant shell."""

    if maximum_total_quanta < 0:
        raise ValueError("maximum_total_quanta must be nonnegative")
    basis = [
        (first, total - first)
        for total in range(maximum_total_quanta + 1)
        for first in range(total + 1)
    ]
    lookup = {state: index for index, state in enumerate(basis)}
    first_annihilation = np.zeros((len(basis), len(basis)))
    second_annihilation = np.zeros_like(first_annihilation)
    for column, (first, second) in enumerate(basis):
        if first > 0:
            first_annihilation[lookup[(first - 1, second)], column] = np.sqrt(first)
        if second > 0:
            second_annihilation[lookup[(first, second - 1)], column] = np.sqrt(second)
    return basis, first_annihilation, second_annihilation


def _projected_channel_green(
    hamiltonian: np.ndarray,
    initial: np.ndarray,
    energy_eV: np.ndarray,
    eta_eV: float,
) -> Tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = eigh(
        hamiltonian,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    weights = np.square(np.abs(eigenvectors.T.conjugate() @ initial))
    green = direct_lorentz_green(
        energy_eV,
        eigenvalues,
        weights,
        eta_eV,
    )
    return eigenvalues, green


def validate_representation_equivalence(
    maximum_total_quanta: int = 7,
) -> Dict[str, float]:
    """Compare local and common/difference Hamiltonians in one exact shell.

    A rectangular cutoff per mode is not preserved by the oscillator rotation.
    The total-number shell used here *is* preserved, so this is a genuine
    finite-dimensional equivalence test rather than an asymptotic comparison.
    """

    model = ModelParams(
        omega0_meV=50.0,
        huang_rhys=0.7,
        j_meV=17.0,
        temperature_K=0.0,
        eta_meV=1.0,
    )
    basis, annihilation_1, annihilation_2 = _total_shell_operators(
        maximum_total_quanta
    )
    phonon_dimension = len(basis)
    identity_v = np.eye(phonon_dimension)
    identity_e = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    tau_z = np.diag([1.0, -1.0])
    tau_x = sigma_x.copy()
    projector_1 = np.diag([1.0, 0.0])
    projector_2 = np.diag([0.0, 1.0])
    number_total = np.diag(
        np.asarray([first + second for first, second in basis], dtype=float)
    )
    coordinate_1 = annihilation_1 + annihilation_1.T
    coordinate_2 = annihilation_2 + annihilation_2.T

    hamiltonian_local = (
        np.kron(
            model.vertical_energy_eV * identity_e + model.j_eV * sigma_x,
            identity_v,
        )
        + np.kron(identity_e, model.omega0_eV * number_total)
        + model.coupling_eV
        * (
            np.kron(projector_1, coordinate_1)
            + np.kron(projector_2, coordinate_2)
        )
    )
    coupling_normal = model.coupling_eV / np.sqrt(2.0)
    hamiltonian_normal = (
        np.kron(
            model.vertical_energy_eV * identity_e + model.j_eV * tau_z,
            identity_v,
        )
        + np.kron(identity_e, model.omega0_eV * number_total)
        + coupling_normal
        * (
            np.kron(identity_e, coordinate_1)
            + np.kron(tau_x, coordinate_2)
        )
    )

    vacuum_index = basis.index((0, 0))
    bright_local = np.zeros(2 * phonon_dimension)
    bright_local[vacuum_index] = 1.0 / np.sqrt(2.0)
    bright_local[phonon_dimension + vacuum_index] = 1.0 / np.sqrt(2.0)
    bright_normal = np.zeros_like(bright_local)
    bright_normal[vacuum_index] = 1.0
    energy = np.linspace(1.5, 2.8, 5001)
    eigen_local, green_local = _projected_channel_green(
        hamiltonian_local.copy(),
        bright_local,
        energy,
        model.eta_eV,
    )
    eigen_normal, green_normal = _projected_channel_green(
        hamiltonian_normal.copy(),
        bright_normal,
        energy,
        model.eta_eV,
    )
    site_1 = np.zeros_like(bright_local)
    site_2 = np.zeros_like(bright_local)
    site_1[vacuum_index] = 1.0
    site_2[phonon_dimension + vacuum_index] = 1.0
    _, green_site_1 = _projected_channel_green(
        hamiltonian_local.copy(), site_1, energy, model.eta_eV
    )
    _, green_site_2 = _projected_channel_green(
        hamiltonian_local.copy(), site_2, energy, model.eta_eV
    )
    eigenvalue_error = float(np.max(np.abs(eigen_local - eigen_normal)))
    spectrum_scale = max(1.0, float(np.max(np.abs(green_local))))
    green_error = float(np.max(np.abs(green_local - green_normal)) / spectrum_scale)
    site_scale = max(
        1.0,
        float(np.max(np.abs(green_site_1))),
        float(np.max(np.abs(green_site_2))),
    )
    site_exchange_error = float(
        np.max(np.abs(green_site_1 - green_site_2)) / site_scale
    )
    if (
        eigenvalue_error > 2.0e-12
        or green_error > 2.0e-10
        or site_exchange_error > 2.0e-10
    ):
        raise RuntimeError(
            "Local/common-difference representation check failed: "
            f"eigen={eigenvalue_error:.3e} eV, G={green_error:.3e}, "
            f"site exchange={site_exchange_error:.3e}"
        )
    return {
        "maximum_total_quanta": int(maximum_total_quanta),
        "phonon_dimension": int(phonon_dimension),
        "matrix_dimension": int(2 * phonon_dimension),
        "max_eigenvalue_error_eV": eigenvalue_error,
        "relative_green_error": green_error,
        "site_1_site_2_relative_green_error": site_exchange_error,
    }


def run_validation_suite() -> Dict[str, Dict[str, float]]:
    """Run the inexpensive ED regression suite."""

    return {
        "direct_lorentz": validate_direct_lorentz_regression(),
        "zero_coupling": validate_zero_coupling(),
        "j_zero_franck_condon": validate_j_zero_franck_condon(),
        "signed_j_symmetry": validate_signed_j_symmetry(),
        "representation_equivalence": validate_representation_equivalence(),
    }


def save_ed_result(
    path: Path,
    model: ModelParams,
    grid_spec: GridSpec,
    result: EDResult,
) -> None:
    """Save one ED calculation without pickle-dependent metadata."""

    metadata: Dict[str, Any] = {
        "model": model_dict(model),
        "grid": {key: float(value) for key, value in asdict(grid_spec).items()},
        "diagnostics": result.diagnostics,
        "spectrum_convention": "A_X(energy) = -Im G_X^R(energy)",
        "raw_full_line_area_convention": "pi times retained Lehmann pole weight",
        "eta_is_physical_damping": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        energy_eV=result.energy_eV,
        green_bright=result.green_bright,
        green_dark=result.green_dark,
        green_local=result.green_local,
        spectrum_bright=result.spectrum_bright,
        spectrum_dark=result.spectrum_dark,
        spectrum_local=result.spectrum_local,
        metadata_json=encode_metadata(metadata),
    )


def _default_output_name(model: ModelParams) -> str:
    return (
        "ed_single_mode_"
        f"S_{signed_tag(model.huang_rhys)}_"
        f"r_{signed_tag(model.resonance_ratio)}_"
        f"w_{signed_tag(model.omega0_meV)}meV_"
        f"T_{signed_tag(model.temperature_K)}K_"
        f"eta_{signed_tag(model.eta_meV)}meV.npz"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-00-ev", type=float, default=2.0)
    parser.add_argument("--omega0-mev", type=float, default=24.8)
    parser.add_argument("--huang-rhys", type=float, default=30.0 / 24.8)
    parser.add_argument("--j-mev", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--eta-mev", type=float, default=0.5)
    parser.add_argument(
        "--n-vib",
        type=int,
        default=0,
        help="states per common/difference mode; 0 selects a conservative value",
    )
    parser.add_argument("--n-thermal", type=int, default=0)
    parser.add_argument("--thermal-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--stick-weight-tolerance", type=float, default=1.0e-14)
    parser.add_argument(
        "--deposition-refinement",
        type=int,
        default=4,
        help="internal pole-deposition refinement relative to the returned grid",
    )
    parser.add_argument("--step-mev", type=float, default=0.0)
    parser.add_argument("--analysis-min-ev", type=float)
    parser.add_argument("--analysis-max-ev", type=float)
    parser.add_argument("--padding-ev", type=float, default=0.55)
    parser.add_argument(
        "--output",
        type=Path,
        help="output NPZ; a relative path is resolved below single-mode-bath/",
    )
    parser.add_argument(
        "--run-validations",
        action="store_true",
        help="run analytic and representation regressions after the case",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run regressions without calculating or saving the CLI case",
    )
    return parser


def _cli_grid(args: argparse.Namespace, model: ModelParams) -> GridSpec:
    grid = default_grid_spec(model, step_meV=args.step_mev, padding_eV=args.padding_ev)
    if (args.analysis_min_ev is None) != (args.analysis_max_ev is None):
        raise ValueError("Specify both --analysis-min-ev and --analysis-max-ev")
    if args.analysis_min_ev is not None:
        if not args.analysis_min_ev < args.analysis_max_ev:
            raise ValueError("The analysis window bounds are unordered")
        grid = GridSpec(
            analysis_min_eV=args.analysis_min_ev,
            analysis_max_eV=args.analysis_max_ev,
            internal_min_eV=args.analysis_min_ev - args.padding_ev,
            internal_max_eV=args.analysis_max_ev + args.padding_ev,
            step_meV=grid.step_meV,
        )
    return grid


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        validations = run_validation_suite()
        for name, values in validations.items():
            formatted = ", ".join(f"{key}={value:.6g}" for key, value in values.items())
            print(f"PASS {name}: {formatted}")
        return 0

    model = ModelParams(
        energy_00_eV=args.energy_00_ev,
        omega0_meV=args.omega0_mev,
        huang_rhys=args.huang_rhys,
        j_meV=args.j_mev,
        temperature_K=args.temperature_k,
        eta_meV=args.eta_mev,
    )
    model.validate()
    grid_spec = _cli_grid(args, model)
    n_vib = args.n_vib if args.n_vib > 0 else None
    n_thermal = args.n_thermal if args.n_thermal > 0 else None
    result = solve_ed(
        model,
        grid_spec,
        n_vib=n_vib,
        thermal_tolerance=args.thermal_tolerance,
        n_thermal=n_thermal,
        stick_weight_tolerance=args.stick_weight_tolerance,
        deposition_refinement=args.deposition_refinement,
    )
    ensure_directories()
    if args.output is None:
        output = SPECTRA / _default_output_name(model)
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
    save_ed_result(output, model, grid_spec, result)
    diagnostics = result.diagnostics
    print(f"Saved ED result: {output}")
    print(
        "ED: "
        f"N_vib={diagnostics['n_vib']}, "
        f"N_thermal={diagnostics['n_thermal']}, "
        f"P_thermal_omitted={diagnostics['thermal_omitted_probability']:.3e}, "
        f"W_B={diagnostics['raw_bright_pole_weight']:.12f}, "
        f"W_D={diagnostics['raw_dark_pole_weight']:.12f}"
    )
    print(
        "Raw first moments: "
        f"B error={diagnostics['bright_first_moment_error_eV'] * 1.0e3:.3e} meV, "
        f"D error={diagnostics['dark_first_moment_error_eV'] * 1.0e3:.3e} meV"
    )

    if args.run_validations:
        validations = run_validation_suite()
        for name, values in validations.items():
            formatted = ", ".join(f"{key}={value:.6g}" for key, value in values.items())
            print(f"PASS {name}: {formatted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
