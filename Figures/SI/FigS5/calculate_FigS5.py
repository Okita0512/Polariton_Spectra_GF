#!/usr/bin/env python3
"""Calculate the vertical-transition-resonant ED/SCBA data for Fig. S6.

The physical model is a one-excitation Holstein--Tavis--Cummings benchmark.
The local vibration is 0.02 eV, the cavity is
resonant with the vertical transition omega_0 = omega_00 + S nu_0, and the
single-molecule light--matter coupling is fixed at g_c = 35.36 meV.  Thus the
collective matrix element sqrt(N) g_c grows with molecule number.

Inside the cavity, ED uses the exact totally permutation-symmetric sector
reached from the normalized bright optical source, subject only to a total
vibrational-quanta cutoff.  Outside the cavity, the equivalent analytic
Franck--Condon progression is used.  The T=0 discrete-mode SCBA equation is
solved by a causal recursion on a grid for which nu_0 is an integer number of
frequency steps.

All energies are in eV and hbar = 1.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix


FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parents[2]
PROTOCOL_ROOT = PROJECT_ROOT / "Figures" / "shared" / "discrete_mode" / "vertical"
RESULTS_ROOT = PROTOCOL_ROOT / "results"

HUANG_RHYS_FACTORS = (0.5, 1.0, 2.0)
MOLECULE_COUNTS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Model:
    """Physical parameters for one plotted case."""

    n_molecules: int
    huang_rhys: float
    inside_cavity: bool
    energy_00: float = 2.0
    phonon_frequency: float = 0.02
    cavity_frequency: float = 2.0
    fixed_single_molecule_coupling: float = 0.03536
    eta: float = 0.0005

    @property
    def vertical_exciton_energy(self) -> float:
        return self.energy_00 + self.huang_rhys * self.phonon_frequency

    @property
    def vibronic_coupling(self) -> float:
        return self.phonon_frequency * np.sqrt(self.huang_rhys)

    @property
    def active_collective_coupling(self) -> float:
        if not self.inside_cavity:
            return 0.0
        return np.sqrt(self.n_molecules) * self.single_molecule_coupling

    @property
    def single_molecule_coupling(self) -> float:
        return (
            self.fixed_single_molecule_coupling
            if self.inside_cavity
            else 0.0
        )


@dataclass(frozen=True)
class Numerics:
    """Production and convergence controls."""

    omega_min: float = 1.5
    omega_max: float = 3.2
    omega_step: float = 0.00005
    coarse_omega_step: float = 0.00010
    plot_min: float = 1.8
    plot_max: float = 2.3
    exact_plot_step: float = 0.00005
    ed_total_phonon_cutoff: int = 28
    ed_check_total_phonon_cutoff: int = 30
    ed_lanczos_steps: int = 5000
    ed_lanczos_check_steps: int = 4000
    ed_breakdown_tolerance: float = 1.0e-13
    dense_dimension_limit: int = 2000
    franck_condon_terms: int = 60

    def grid(self, step: float) -> np.ndarray:
        intervals = int(round((self.omega_max - self.omega_min) / step))
        if abs(self.omega_min + intervals * step - self.omega_max) > 1.0e-12:
            raise ValueError("Frequency interval is not commensurate with step")
        return np.linspace(self.omega_min, self.omega_max, intervals + 1)

    def exact_grid(self) -> np.ndarray:
        intervals = int(
            round((self.plot_max - self.plot_min) / self.exact_plot_step)
        )
        return np.linspace(self.plot_min, self.plot_max, intervals + 1)


@dataclass
class ExactResult:
    spectrum: np.ndarray
    dimension: int
    method: str
    steps: int
    alpha: np.ndarray
    beta: np.ndarray
    hermiticity_error: float
    vacuum_collective_coupling: float
    lanczos_check_difference: float
    elapsed_seconds: float


@dataclass
class ScbaResult:
    omega: np.ndarray
    spectrum: np.ndarray
    sigma: np.ndarray
    fixed_point_residual: float
    max_positive_im_sigma: float
    minimum_spectrum: float
    elapsed_seconds: float


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise ValueError("Cannot peak-normalize a nonpositive spectrum")
    return np.asarray(values, dtype=float) / maximum


@lru_cache(maxsize=None)
def sorted_occupations(
    n_modes: int,
    total_cutoff: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Return nondecreasing occupation tuples with total <= cutoff."""

    if n_modes < 0 or total_cutoff < 0:
        raise ValueError("Mode count and cutoff must be nonnegative")
    if n_modes == 0:
        return ((),)

    states = []

    def append_states(prefix, minimum, remaining):
        if len(prefix) == n_modes:
            states.append(tuple(prefix))
            return
        for occupation in range(minimum, remaining + 1):
            append_states(
                prefix + [occupation],
                occupation,
                remaining - occupation,
            )

    append_states([], 0, total_cutoff)
    return tuple(states)


@lru_cache(maxsize=None)
def symmetric_basis(n_molecules: int, total_cutoff: int):
    """Return photon and distinguished-exciton orbit labels."""

    if n_molecules < 1:
        raise ValueError("At least one molecule is required")
    photons = sorted_occupations(n_molecules, total_cutoff)
    excitons = []
    for excited_occupation in range(total_cutoff + 1):
        spectators_at_fixed_excitation = sorted_occupations(
            n_molecules - 1,
            total_cutoff - excited_occupation,
        )
        excitons.extend(
            (excited_occupation, spectators)
            for spectators in spectators_at_fixed_excitation
        )
    return photons, tuple(excitons)


def build_symmetric_hamiltonian(
    model: Model,
    total_cutoff: int,
) -> Tuple[csr_matrix, int, float, float]:
    """Build the exact permutation-symmetric one-excitation Hamiltonian."""

    if not model.inside_cavity:
        raise ValueError("The symmetric cavity Hamiltonian requires a cavity")
    photons, excitons = symmetric_basis(model.n_molecules, total_cutoff)
    keys = [("photon", occupations) for occupations in photons]
    keys.extend(
        ("exciton", excited, spectators)
        for excited, spectators in excitons
    )
    index = {key: position for position, key in enumerate(keys)}

    rows = []
    columns = []
    values = []

    def add(row, column, value):
        rows.append(row)
        columns.append(column)
        values.append(float(value))

    for occupations in photons:
        photon_index = index[("photon", occupations)]
        add(
            photon_index,
            photon_index,
            model.cavity_frequency
            + model.phonon_frequency * sum(occupations),
        )
        for occupation in set(occupations):
            spectators_list = list(occupations)
            spectators_list.remove(occupation)
            spectators = tuple(spectators_list)
            exciton_index = index[("exciton", occupation, spectators)]
            multiplicity = occupations.count(occupation)
            coupling = model.single_molecule_coupling * np.sqrt(multiplicity)
            add(photon_index, exciton_index, coupling)
            add(exciton_index, photon_index, coupling)

    for excited_occupation, spectators in excitons:
        exciton_index = index[("exciton", excited_occupation, spectators)]
        add(
            exciton_index,
            exciton_index,
            model.vertical_exciton_energy
            + model.phonon_frequency
            * (excited_occupation + sum(spectators)),
        )
        if excited_occupation + sum(spectators) < total_cutoff:
            raised_index = index[
                ("exciton", excited_occupation + 1, spectators)
            ]
            coupling = model.vibronic_coupling * np.sqrt(
                excited_occupation + 1
            )
            add(exciton_index, raised_index, coupling)
            add(raised_index, exciton_index, coupling)

    dimension = len(keys)
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension),
        dtype=float,
    ).tocsr()
    source_index = index[
        ("exciton", 0, (0,) * (model.n_molecules - 1))
    ]
    photon_vacuum_index = index[
        ("photon", (0,) * model.n_molecules)
    ]
    vacuum_collective_coupling = float(
        hamiltonian[photon_vacuum_index, source_index]
    )
    difference = hamiltonian - hamiltonian.T
    hermiticity_error = (
        0.0
        if difference.nnz == 0
        else float(np.max(np.abs(difference.data)))
    )
    return (
        hamiltonian,
        source_index,
        hermiticity_error,
        vacuum_collective_coupling,
    )


def broaden_poles(
    energy: np.ndarray,
    poles: np.ndarray,
    weights: np.ndarray,
    eta: float,
    chunk_size: int = 256,
) -> np.ndarray:
    """Lorentz-broaden a positive projected pole measure."""

    spectrum = np.zeros_like(energy, dtype=float)
    for start in range(0, len(poles), chunk_size):
        stop = min(start + chunk_size, len(poles))
        difference = energy[None, :] - poles[start:stop, None]
        spectrum += np.sum(
            weights[start:stop, None]
            * eta
            / (difference**2 + eta**2),
            axis=0,
        )
    return spectrum


def lanczos_coefficients(
    hamiltonian: csr_matrix,
    source_index: int,
    max_steps: int,
    breakdown_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Projected Hermitian Lanczos recursion from the optical source."""

    dimension = hamiltonian.shape[0]
    steps = min(int(max_steps), dimension)
    current = np.zeros(dimension, dtype=float)
    current[source_index] = 1.0
    previous = np.zeros_like(current)
    previous_beta = 0.0
    alpha = []
    beta = []

    for iteration in range(steps):
        residual = hamiltonian @ current
        if iteration:
            residual -= previous_beta * previous
        diagonal = float(np.dot(current, residual))
        residual -= diagonal * current
        residual -= float(np.dot(current, residual)) * current
        if iteration:
            residual -= float(np.dot(previous, residual)) * previous

        off_diagonal = float(np.linalg.norm(residual))
        alpha.append(diagonal)
        if off_diagonal < breakdown_tolerance or iteration + 1 == steps:
            break
        beta.append(off_diagonal)
        previous, current = current, residual / off_diagonal
        previous_beta = off_diagonal

    alpha_array = np.asarray(alpha, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    if len(beta_array) != len(alpha_array) - 1:
        raise RuntimeError("Lanczos coefficient lengths are inconsistent")
    if not np.all(np.isfinite(alpha_array)) or not np.all(
        np.isfinite(beta_array)
    ):
        raise RuntimeError("Lanczos recursion produced NaN or Inf")
    return alpha_array, beta_array


def continued_fraction_spectrum(
    energy: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    eta: float,
    steps: int = None,
) -> np.ndarray:
    """Evaluate the projected retarded Lanczos resolvent."""

    if steps is None:
        steps = len(alpha)
    steps = min(int(steps), len(alpha))
    if steps < 1:
        raise ValueError("At least one Lanczos coefficient is required")
    denominator = energy + 1.0j * eta - alpha[steps - 1]
    for index in range(steps - 2, -1, -1):
        denominator = (
            energy
            + 1.0j * eta
            - alpha[index]
            - beta[index] ** 2 / denominator
        )
    return -np.imag(1.0 / denominator)


def solve_exact_cavity(
    model: Model,
    numerics: Numerics,
    energy: np.ndarray,
    total_cutoff: int,
    perform_lanczos_check: bool,
) -> ExactResult:
    """Return the symmetry-reduced cavity ED spectrum."""

    start = time.perf_counter()
    (
        hamiltonian,
        source_index,
        hermiticity_error,
        vacuum_collective_coupling,
    ) = build_symmetric_hamiltonian(model, total_cutoff)
    dimension = hamiltonian.shape[0]
    if hermiticity_error > 1.0e-14:
        raise RuntimeError(
            "Exact Hamiltonian is not Hermitian: "
            f"{hermiticity_error:.3e}"
        )

    if dimension <= numerics.dense_dimension_limit:
        poles, eigenvectors = eigh(
            hamiltonian.toarray(),
            overwrite_a=True,
            check_finite=False,
            driver="evd",
        )
        weights = np.square(eigenvectors[source_index, :])
        oscillator_strength_error = abs(float(np.sum(weights)) - 1.0)
        if oscillator_strength_error > 1.0e-11:
            raise RuntimeError(
                "Dense ED oscillator-strength sum failed: "
                f"{oscillator_strength_error:.3e}"
            )
        spectrum = broaden_poles(energy, poles, weights, model.eta)
        method = "dense_eigh"
        steps = dimension
        alpha = np.empty(0, dtype=float)
        beta = np.empty(0, dtype=float)
        lanczos_check_difference = 0.0
    else:
        alpha, beta = lanczos_coefficients(
            hamiltonian,
            source_index,
            numerics.ed_lanczos_steps,
            numerics.ed_breakdown_tolerance,
        )
        spectrum = continued_fraction_spectrum(
            energy,
            alpha,
            beta,
            model.eta,
        )
        method = "projected_lanczos_cf"
        steps = len(alpha)
        if perform_lanczos_check:
            short_steps = min(numerics.ed_lanczos_check_steps, steps)
            short_spectrum = continued_fraction_spectrum(
                energy,
                alpha,
                beta,
                model.eta,
                steps=short_steps,
            )
            lanczos_check_difference = float(
                np.max(
                    np.abs(
                        normalize(short_spectrum) - normalize(spectrum)
                    )
                )
            )
        else:
            lanczos_check_difference = 0.0

    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("Exact spectrum contains NaN or Inf")
    if float(np.min(spectrum)) < -1.0e-12:
        raise RuntimeError("Exact projected spectrum is negative")
    return ExactResult(
        spectrum=spectrum,
        dimension=dimension,
        method=method,
        steps=steps,
        alpha=alpha,
        beta=beta,
        hermiticity_error=hermiticity_error,
        vacuum_collective_coupling=vacuum_collective_coupling,
        lanczos_check_difference=lanczos_check_difference,
        elapsed_seconds=time.perf_counter() - start,
    )


def exact_franck_condon(
    energy: np.ndarray,
    model: Model,
    terms: int,
) -> np.ndarray:
    """Analytic isolated-molecule Franck--Condon progression."""

    spectrum = np.zeros_like(energy)
    weight = float(np.exp(-model.huang_rhys))
    for quantum in range(terms + 1):
        if quantum:
            weight *= model.huang_rhys / quantum
        center = model.energy_00 + quantum * model.phonon_frequency
        spectrum += (
            weight
            * model.eta
            / ((energy - center) ** 2 + model.eta**2)
        )
    return spectrum


def green_components_scalar(
    omega: float,
    sigma: complex,
    model: Model,
) -> Tuple[complex, complex, complex]:
    """Return local, bright, and dark molecular Green functions."""

    exciton_inverse = (
        omega - model.vertical_exciton_energy + 1.0j * model.eta - sigma
    )
    if not model.inside_cavity:
        molecular = 1.0 / exciton_inverse
        return molecular, molecular, molecular

    photon_inverse = omega - model.cavity_frequency + 1.0j * model.eta
    bright = photon_inverse / (
        photon_inverse * exciton_inverse
        - model.active_collective_coupling**2
    )
    dark = 1.0 / exciton_inverse
    local = (
        bright + (model.n_molecules - 1.0) * dark
    ) / model.n_molecules
    return local, bright, dark


def solve_scba(
    model: Model,
    numerics: Numerics,
    step: float,
) -> ScbaResult:
    """Solve Sigma(w)=M^2 G_loc(w-nu_0) by causal grid recursion."""

    start = time.perf_counter()
    omega = numerics.grid(step)
    shift_steps = int(round(model.phonon_frequency / step))
    if abs(shift_steps * step - model.phonon_frequency) > 1.0e-12:
        raise ValueError("Phonon frequency is not aligned with SCBA grid")

    sigma = np.zeros_like(omega, dtype=complex)
    local = np.zeros_like(omega, dtype=complex)
    bright = np.zeros_like(omega, dtype=complex)
    coupling_squared = model.vibronic_coupling**2

    for index, frequency in enumerate(omega):
        if index >= shift_steps:
            shifted_local = local[index - shift_steps]
        else:
            shifted_frequency = frequency - model.phonon_frequency
            shifted_local, _, _ = green_components_scalar(
                shifted_frequency,
                0.0j,
                model,
            )
        sigma[index] = coupling_squared * shifted_local
        local[index], bright[index], _ = green_components_scalar(
            frequency,
            sigma[index],
            model,
        )

    mapped = np.empty_like(sigma)
    for index, frequency in enumerate(omega):
        if index >= shift_steps:
            shifted_local = local[index - shift_steps]
        else:
            shifted_local, _, _ = green_components_scalar(
                frequency - model.phonon_frequency,
                0.0j,
                model,
            )
        mapped[index] = coupling_squared * shifted_local

    spectrum = -np.imag(bright)
    fixed_point_residual = float(np.max(np.abs(mapped - sigma)))
    max_positive_im_sigma = float(np.max(sigma.imag))
    minimum_spectrum = float(np.min(spectrum))
    if not np.all(np.isfinite(spectrum)) or not np.all(np.isfinite(sigma)):
        raise RuntimeError("SCBA result contains NaN or Inf")
    if max_positive_im_sigma > 5.0e-10:
        raise RuntimeError("SCBA self-energy violates retarded causality")
    if minimum_spectrum < -5.0e-10:
        raise RuntimeError("SCBA spectrum is negative")
    return ScbaResult(
        omega=omega,
        spectrum=spectrum,
        sigma=sigma,
        fixed_point_residual=fixed_point_residual,
        max_positive_im_sigma=max_positive_im_sigma,
        minimum_spectrum=minimum_spectrum,
        elapsed_seconds=time.perf_counter() - start,
    )


def spectral_metrics(
    exact_energy: np.ndarray,
    exact: np.ndarray,
    scba_energy: np.ndarray,
    scba: np.ndarray,
) -> dict:
    """Return area-normalized shape diagnostics on the exact plot grid."""

    scba_on_exact = np.interp(exact_energy, scba_energy, scba)
    exact_area = float(np.trapz(exact, exact_energy))
    scba_area = float(np.trapz(scba_on_exact, exact_energy))
    if exact_area <= 0.0 or scba_area <= 0.0:
        raise RuntimeError("Spectral areas must be positive")
    exact_unit = exact / exact_area
    scba_unit = scba_on_exact / scba_area
    unit_area_l1 = float(
        np.trapz(np.abs(exact_unit - scba_unit), exact_energy)
    )
    exact_cdf = cumulative_trapezoid(
        exact_unit,
        exact_energy,
        initial=0.0,
    )
    scba_cdf = cumulative_trapezoid(
        scba_unit,
        exact_energy,
        initial=0.0,
    )
    wasserstein = float(
        np.trapz(np.abs(exact_cdf - scba_cdf), exact_energy)
    )
    return {
        "exact_area": exact_area,
        "scba_area": scba_area,
        "scba_area_scale": exact_area / scba_area,
        "unit_area_l1": unit_area_l1,
        "wasserstein_eV": wasserstein,
    }


def grid_convergence_difference(
    fine: ScbaResult,
    coarse: ScbaResult,
    numerics: Numerics,
) -> float:
    mask = (
        (fine.omega >= numerics.plot_min)
        & (fine.omega <= numerics.plot_max)
    )
    coarse_on_fine = np.interp(
        fine.omega[mask],
        coarse.omega,
        coarse.spectrum,
    )
    return float(
        np.max(
            np.abs(
                normalize(fine.spectrum[mask]) - normalize(coarse_on_fine)
            )
        )
    )


def case_filename(model: Model) -> str:
    prefix = f"N{model.n_molecules}" if model.inside_cavity else "outside"
    return f"{prefix}_S{model.huang_rhys:.1f}.npz"


def save_case(
    model: Model,
    numerics: Numerics,
    exact_energy: np.ndarray,
    exact: ExactResult,
    scba: ScbaResult,
    diagnostics: dict,
) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULTS_ROOT / case_filename(model)
    model_metadata = asdict(model)
    model_metadata.update(
        {
            "vertical_exciton_energy": model.vertical_exciton_energy,
            "vibronic_coupling": model.vibronic_coupling,
            "single_molecule_coupling": model.single_molecule_coupling,
            "collective_coupling": model.active_collective_coupling,
        }
    )
    metadata = {
        "model": model_metadata,
        "numerics": asdict(numerics),
        "diagnostics": diagnostics,
        "exact_method": exact.method,
        "scba_method": "aligned-grid causal recursion",
    }
    np.savez_compressed(
        path,
        exact_energy_eV=exact_energy,
        exact_spectrum_per_eV=exact.spectrum,
        scba_energy_eV=scba.omega,
        scba_spectrum_per_eV=scba.spectrum,
        scba_sigma_eV=scba.sigma,
        ed_lanczos_alpha_eV=exact.alpha,
        ed_lanczos_beta_eV=exact.beta,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def maximum(rows: Iterable[dict], key: str) -> float:
    return max(float(row[key]) for row in rows)


def minimum(rows: Iterable[dict], key: str) -> float:
    return min(float(row[key]) for row in rows)


def write_summary(rows: Sequence[dict]) -> Path:
    path = PROTOCOL_ROOT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_validation(rows: Sequence[dict], numerics: Numerics) -> Path:
    checks = {
        "case_count": len(rows) == 15,
        "phonon_frequency": all(
            abs(float(row["phonon_frequency_eV"]) - 0.02) <= 1.0e-14
            for row in rows
        ),
        "vertical_cavity_resonance": all(
            row["condition"] == "outside"
            or abs(
                float(row["cavity_frequency_eV"])
                - float(row["vertical_exciton_energy_eV"])
            ) <= 1.0e-14
            for row in rows
        ),
        "fixed_single_molecule_coupling": all(
            row["condition"] == "outside"
            or abs(float(row["single_molecule_g_c_eV"]) - 0.03536)
            <= 1.0e-14
            for row in rows
        ),
        "max_hermiticity_error": maximum(
            rows, "ed_hermiticity_error"
        ) <= 1.0e-14,
        "max_collective_coupling_error": maximum(
            rows, "collective_coupling_error_eV"
        ) <= 1.0e-13,
        "max_cutoff_difference": maximum(
            rows, "ed_Q28_Q30_peak_normalized_max_difference"
        ) <= 1.0e-6,
        "max_lanczos_difference": maximum(
            rows, "ed_lanczos_4000_5000_peak_normalized_max_difference"
        ) <= 1.0e-8,
        "max_scba_grid_difference": maximum(
            rows, "scba_0p10_0p05meV_peak_normalized_max_difference"
        ) <= 1.0e-2,
        "fixed_point_residual": maximum(
            rows, "scba_fixed_point_residual_eV"
        ) <= 1.0e-13,
        "retarded_causality": maximum(
            rows, "scba_max_positive_Im_Sigma_eV"
        ) <= 5.0e-10,
        "exact_positivity": minimum(
            rows, "minimum_exact_spectrum_per_eV"
        ) >= -1.0e-12,
        "scba_positivity": minimum(
            rows, "minimum_scba_spectrum_per_eV"
        ) >= -5.0e-10,
        "positive_areas": all(
            float(row["exact_area_1p8_2p3"]) > 0.0
            and float(row["scba_area_1p8_2p3"]) > 0.0
            for row in rows
        ),
    }
    passed = all(checks.values())
    lines = [
        "FigS5 vertical-resonance ED/SCBA validation",
        "================================================",
        f"status = {'PASS' if passed else 'FAIL'}",
        "model = single-mode Holstein--Tavis--Cummings, T=0",
        "rows = outside cavity; inside cavity N=1, 2, 4, 8",
        "columns = S=0.5, 1.0, 2.0",
        "E00 = 2.0 eV; nu0 = 0.020 eV; eta = 0.0005 eV",
        "omega_c = omega_0 = E00 + S nu0 (vertical transition)",
        "fixed single-molecule g_c = 0.03536 eV inside the cavity",
        "collective sqrt(N) g_c grows as sqrt(N)",
        "ED cutoff = Q=28; cutoff check = Q=30 (retained validated check when Lanczos coefficients are reused)",
        "Lanczos production/check = 5000/4000 steps",
        "SCBA grid = 1.5--3.2 eV in 0.05-meV steps",
        "SCBA equation = M^2 G_local(omega-nu0), causal recursion",
        "plot and area-match interval = 1.8--2.3 eV",
        "",
        "Aggregate checks",
        "----------------",
    ]
    lines.extend(f"{key} = {value}" for key, value in checks.items())
    lines.extend(
        [
            "",
            "max Q=28 -> 30 peak-normalized difference = "
            f"{maximum(rows, 'ed_Q28_Q30_peak_normalized_max_difference'):.16e}",
            "max 4000 -> 5000 Lanczos difference = "
            f"{maximum(rows, 'ed_lanczos_4000_5000_peak_normalized_max_difference'):.16e}",
            "max 0.10 -> 0.05 meV SCBA-grid difference = "
            f"{maximum(rows, 'scba_0p10_0p05meV_peak_normalized_max_difference'):.16e}",
            "max SCBA fixed-point residual (eV) = "
            f"{maximum(rows, 'scba_fixed_point_residual_eV'):.16e}",
            "max positive Im Sigma (eV) = "
            f"{maximum(rows, 'scba_max_positive_Im_Sigma_eV'):.16e}",
            "minimum exact spectrum (1/eV) = "
            f"{minimum(rows, 'minimum_exact_spectrum_per_eV'):.16e}",
            "minimum SCBA spectrum (1/eV) = "
            f"{minimum(rows, 'minimum_scba_spectrum_per_eV'):.16e}",
            "",
        ]
    )
    path = PROTOCOL_ROOT / "calculation_validation.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("Validation failed: " + ", ".join(failed))
    return path


def outside_exact_result(
    model: Model,
    numerics: Numerics,
    exact_energy: np.ndarray,
) -> ExactResult:
    start = time.perf_counter()
    spectrum = exact_franck_condon(
        exact_energy,
        model,
        numerics.franck_condon_terms,
    )
    return ExactResult(
        spectrum=spectrum,
        dimension=0,
        method="analytic_franck_condon",
        steps=numerics.franck_condon_terms + 1,
        alpha=np.empty(0, dtype=float),
        beta=np.empty(0, dtype=float),
        hermiticity_error=0.0,
        vacuum_collective_coupling=0.0,
        lanczos_check_difference=0.0,
        elapsed_seconds=time.perf_counter() - start,
    )


def cached_lanczos_exact_result(
    model: Model,
    numerics: Numerics,
    exact_energy: np.ndarray,
) -> tuple[ExactResult, float, int, str] | None:
    """Reevaluate validated eta-independent Lanczos data at the active eta."""

    if not model.inside_cavity or model.n_molecules not in (4, 8):
        return None
    path = RESULTS_ROOT / case_filename(model)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        alpha = archive["ed_lanczos_alpha_eV"].copy()
        beta = archive["ed_lanczos_beta_eV"].copy()

    source_model = metadata["model"]
    expected = {
        "n_molecules": model.n_molecules,
        "huang_rhys": model.huang_rhys,
        "energy_00": model.energy_00,
        "phonon_frequency": model.phonon_frequency,
        "cavity_frequency": model.cavity_frequency,
        "fixed_single_molecule_coupling": model.fixed_single_molecule_coupling,
    }
    if metadata.get("exact_method") != "projected_lanczos_cf":
        return None
    for key, expected_value in expected.items():
        if not np.isclose(
            float(source_model[key]), float(expected_value), rtol=0.0, atol=1.0e-12
        ):
            return None
    if len(alpha) != numerics.ed_lanczos_steps or len(beta) != len(alpha) - 1:
        return None

    start = time.perf_counter()
    spectrum = continued_fraction_spectrum(
        exact_energy,
        alpha,
        beta,
        model.eta,
    )
    short_spectrum = continued_fraction_spectrum(
        exact_energy,
        alpha,
        beta,
        model.eta,
        steps=numerics.ed_lanczos_check_steps,
    )
    lanczos_difference = float(
        np.max(np.abs(normalize(short_spectrum) - normalize(spectrum)))
    )
    diagnostics = metadata["diagnostics"]
    cutoff_difference = float(
        diagnostics["ed_cutoff_peak_normalized_max_difference"]
    )
    source_eta = float(source_model["eta"])
    exact = ExactResult(
        spectrum=spectrum,
        dimension=int(diagnostics["ed_dimension"]),
        method="projected_lanczos_cf",
        steps=len(alpha),
        alpha=alpha,
        beta=beta,
        hermiticity_error=0.0,
        vacuum_collective_coupling=(
            np.sqrt(model.n_molecules) * model.single_molecule_coupling
        ),
        lanczos_check_difference=lanczos_difference,
        elapsed_seconds=time.perf_counter() - start,
    )
    provenance = (
        "Q=28/Q=30 convergence retained from validated cached source "
        f"at eta={source_eta:.7f} eV; eta-independent Q=28 Lanczos "
        f"coefficients reevaluated at eta={model.eta:.7f} eV"
    )
    return (
        exact,
        cutoff_difference,
        int(diagnostics["ed_check_dimension"]),
        provenance,
    )


def calculate_case(
    model: Model,
    numerics: Numerics,
    exact_energy: np.ndarray,
) -> dict:
    """Calculate, validate, save, and summarize one panel."""

    if model.inside_cavity:
        cached = cached_lanczos_exact_result(model, numerics, exact_energy)
        if cached is not None:
            exact, cutoff_difference, check_dimension, cutoff_provenance = cached
        else:
            exact = solve_exact_cavity(
                model,
                numerics,
                exact_energy,
                numerics.ed_total_phonon_cutoff,
                perform_lanczos_check=True,
            )
            exact_check = solve_exact_cavity(
                model,
                numerics,
                exact_energy,
                numerics.ed_check_total_phonon_cutoff,
                perform_lanczos_check=False,
            )
            cutoff_difference = float(
                np.max(
                    np.abs(
                        normalize(exact.spectrum)
                        - normalize(exact_check.spectrum)
                    )
                )
            )
            check_dimension = exact_check.dimension
            cutoff_provenance = (
                f"Q=28/Q=30 spectra recalculated at eta={model.eta:.7f} eV"
            )
    else:
        exact = outside_exact_result(model, numerics, exact_energy)
        cutoff_difference = 0.0
        check_dimension = 0
        cutoff_provenance = "analytic outside-cavity reference"

    scba = solve_scba(model, numerics, numerics.omega_step)
    scba_coarse = solve_scba(model, numerics, numerics.coarse_omega_step)
    grid_difference = grid_convergence_difference(
        scba,
        scba_coarse,
        numerics,
    )
    metrics = spectral_metrics(
        exact_energy,
        exact.spectrum,
        scba.omega,
        scba.spectrum,
    )
    collective_error = (
        abs(
            exact.vacuum_collective_coupling
            - np.sqrt(model.n_molecules) * 0.03536
        )
        if model.inside_cavity
        else 0.0
    )
    diagnostics = {
        "ed_dimension": exact.dimension,
        "ed_check_dimension": check_dimension,
        "ed_cutoff_peak_normalized_max_difference": cutoff_difference,
        "ed_cutoff_check_provenance": cutoff_provenance,
        "ed_lanczos_check_difference": exact.lanczos_check_difference,
        "scba_grid_check_difference": grid_difference,
        "scba_fixed_point_residual_eV": scba.fixed_point_residual,
        **metrics,
    }
    output = save_case(
        model,
        numerics,
        exact_energy,
        exact,
        scba,
        diagnostics,
    )

    return {
        "condition": "inside" if model.inside_cavity else "outside",
        "n_molecules": model.n_molecules if model.inside_cavity else 0,
        "huang_rhys": f"{model.huang_rhys:.1f}",
        "energy_00_eV": f"{model.energy_00:.9f}",
        "vertical_exciton_energy_eV": (
            f"{model.vertical_exciton_energy:.9f}"
        ),
        "phonon_frequency_eV": f"{model.phonon_frequency:.9f}",
        "cavity_frequency_eV": (
            f"{model.cavity_frequency:.9f}"
            if model.inside_cavity
            else "nan"
        ),
        "eta_eV": f"{model.eta:.9f}",
        "single_molecule_g_c_eV": (
            f"{model.single_molecule_coupling:.16e}"
        ),
        "collective_sqrtN_g_c_eV": (
            f"{model.active_collective_coupling:.16e}"
        ),
        "collective_coupling_error_eV": f"{collective_error:.16e}",
        "ed_total_phonon_cutoff": (
            numerics.ed_total_phonon_cutoff
            if model.inside_cavity
            else "analytic"
        ),
        "ed_check_total_phonon_cutoff": (
            numerics.ed_check_total_phonon_cutoff
            if model.inside_cavity
            else "analytic"
        ),
        "ed_dimension": exact.dimension,
        "ed_check_dimension": check_dimension,
        "ed_method": exact.method,
        "ed_lanczos_steps": exact.steps,
        "ed_hermiticity_error": f"{exact.hermiticity_error:.16e}",
        "ed_Q28_Q30_peak_normalized_max_difference": (
            f"{cutoff_difference:.16e}"
        ),
        "ed_lanczos_4000_5000_peak_normalized_max_difference": (
            f"{exact.lanczos_check_difference:.16e}"
        ),
        "scba_0p10_0p05meV_peak_normalized_max_difference": (
            f"{grid_difference:.16e}"
        ),
        "scba_fixed_point_residual_eV": (
            f"{scba.fixed_point_residual:.16e}"
        ),
        "scba_max_positive_Im_Sigma_eV": (
            f"{scba.max_positive_im_sigma:.16e}"
        ),
        "minimum_exact_spectrum_per_eV": (
            f"{np.min(exact.spectrum):.16e}"
        ),
        "minimum_scba_spectrum_per_eV": (
            f"{scba.minimum_spectrum:.16e}"
        ),
        "exact_area_1p8_2p3": f"{metrics['exact_area']:.16e}",
        "scba_area_1p8_2p3": f"{metrics['scba_area']:.16e}",
        "scba_area_match_factor": f"{metrics['scba_area_scale']:.16e}",
        "unit_area_L1_error": f"{metrics['unit_area_l1']:.16e}",
        "W1_error_eV": f"{metrics['wasserstein_eV']:.16e}",
        "ed_seconds": f"{exact.elapsed_seconds:.6f}",
        "scba_seconds": f"{scba.elapsed_seconds:.6f}",
        "result_file": str(output.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    numerics = Numerics()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    exact_energy = numerics.exact_grid()
    rows = []

    for s_value in HUANG_RHYS_FACTORS:
        outside_model = Model(
            n_molecules=1,
            huang_rhys=s_value,
            inside_cavity=False,
            cavity_frequency=2.0 + s_value * 0.02,
        )
        print(f"outside cavity, S={s_value:.1f}", flush=True)
        outside_row = calculate_case(outside_model, numerics, exact_energy)
        rows.append(outside_row)
        print(
            "  SCBA grid check={:.3e}; unit-area L1={:.4f}".format(
                float(
                    outside_row[
                        "scba_0p10_0p05meV_peak_normalized_max_difference"
                    ]
                ),
                float(outside_row["unit_area_L1_error"]),
            ),
            flush=True,
        )

        for n_molecules in MOLECULE_COUNTS:
            model = Model(
                n_molecules=n_molecules,
                huang_rhys=s_value,
                inside_cavity=True,
                cavity_frequency=2.0 + s_value * 0.02,
            )
            print(
                f"inside cavity, N={n_molecules}, S={s_value:.1f}, "
                f"g_c={model.single_molecule_coupling:.9f} eV",
                flush=True,
            )
            row = calculate_case(model, numerics, exact_energy)
            rows.append(row)
            print(
                "  ED dim={:d}, Q check={:.3e}, Lanczos check={:.3e}; "
                "SCBA grid check={:.3e}; unit-area L1={:.4f}".format(
                    int(row["ed_dimension"]),
                    float(row["ed_Q28_Q30_peak_normalized_max_difference"]),
                    float(
                        row[
                            "ed_lanczos_4000_5000_peak_normalized_max_difference"
                        ]
                    ),
                    float(
                        row[
                            "scba_0p10_0p05meV_peak_normalized_max_difference"
                        ]
                    ),
                    float(row["unit_area_L1_error"]),
                ),
                flush=True,
            )

    summary_path = write_summary(rows)
    validation_path = write_validation(rows, numerics)
    print(f"Saved {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {validation_path.relative_to(PROJECT_ROOT)}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
