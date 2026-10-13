#!/usr/bin/env python3
"""Shifted-frequency G-only SCBA for the HTC figure calculations.

The legacy figure scripts contain the physical parameters for each case.
``run_legacy_script`` reads those parameters, solves the SCBA Dyson equation
on one padded frequency grid, and writes files in the legacy format expected
by the plotting and linewidth-analysis scripts.

All internal energies are in atomic units.  The equilibrium phonon line is
kept bare: in the zero-exciton reference state the connected density
polarization vanishes, so there is no phonon Dyson loop.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from drude_convolution import LinearDrudeConvolver


EV_PER_AU = 27.211397
K_PER_AU = 3.1577464e5


@dataclass(frozen=True)
class Case:
    script: Path
    vertical_energy: float
    omega_c: float
    cavity_halfwidth: float
    reorganization: float
    bath_cutoff: float
    collective_g: float
    eta: float
    n_molecules: int
    output_grid: np.ndarray
    legacy_n_bath: int
    temperature_K: float = 300.0

    @property
    def energy_00(self) -> float:
        """Relaxed 0--0 transition used throughout the manuscript."""
        return self.vertical_energy - self.reorganization


@dataclass
class Result:
    bath_method: str
    self_energy_mode: str
    grid: np.ndarray
    sigma: np.ndarray
    output_sigma: np.ndarray
    output_green_bright: np.ndarray
    history: np.ndarray
    iterations: int
    final_error: float
    final_mixing: float


def _evaluate_legacy_assignments(script: Path) -> Dict[str, object]:
    """Evaluate only the simple top-level parameter assignments."""
    source = script.read_text()
    tree = ast.parse(source, filename=str(script))
    wanted = {
        "conv",
        "wc",
        "kappa",
        "w0",
        "gamma",
        "ndof",
        "Nexc",
        "Nplot",
        "w_plot",
        "Omega_R",
        "gammak",
        "lam",
    }
    values: Dict[str, object] = {"np": np}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        expression = ast.Expression(body=node.value)
        ast.fix_missing_locations(expression)
        values[target.id] = eval(
            compile(expression, str(script), "eval"),
            {"__builtins__": {}},
            values,
        )
    missing = wanted.difference(values)
    if missing:
        raise ValueError(
            "{} is missing required assignments: {}".format(
                script, ", ".join(sorted(missing))
            )
        )
    return values


def parse_legacy_case(script: Path) -> Case:
    script = Path(script).resolve()
    values = _evaluate_legacy_assignments(script)
    conv = float(values["conv"])
    omega_c_complex = complex(values["wc"])
    return Case(
        script=script,
        vertical_energy=float(values["w0"]),
        omega_c=float(omega_c_complex.real),
        cavity_halfwidth=max(0.0, float(-omega_c_complex.imag)),
        reorganization=float(values["lam"]),
        bath_cutoff=float(values["gamma"]),
        collective_g=float(values["Omega_R"]),
        eta=float(values["gammak"]),
        n_molecules=int(values["Nexc"]),
        output_grid=np.asarray(values["w_plot"], dtype=float),
        legacy_n_bath=int(values["ndof"]),
    )


def bose(omega: np.ndarray, temperature_K: float) -> np.ndarray:
    x = K_PER_AU * np.asarray(omega) / temperature_K
    result = np.zeros_like(x, dtype=float)
    mask = x < 700.0
    result[mask] = 1.0 / np.expm1(x[mask])
    return result


def tangent_bath(
    reorganization: float,
    cutoff: float,
    n_bath: int,
) -> Tuple[np.ndarray, np.ndarray]:
    index = np.arange(1, n_bath + 1, dtype=float)
    omega_q = cutoff * np.tan(
        0.5 * np.pi * (1.0 - index / (n_bath + 1.0))
    )
    coupling = np.sqrt(reorganization * omega_q / (n_bath + 1.0))
    return coupling, omega_q


def interpolate_complex(
    points: np.ndarray,
    grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    shape = points.shape
    flat = points.ravel()
    real = np.interp(flat, grid, values.real, left=0.0, right=0.0)
    imag = np.interp(flat, grid, values.imag, left=0.0, right=0.0)
    return (real + 1.0j * imag).reshape(shape)


def hopfield_transform(case: Case) -> Tuple[np.ndarray, np.ndarray]:
    """Return R and the matter-amplitude vector for (P_+, P_-).

    The input ordering is (a, B), the polariton ordering is (P_+, P_-), and
    p_pol = R p_aB.  The angle diagonalizes the lossless bare Hamiltonian
    containing the vertical molecular energy.
    """
    detuning = case.omega_c - case.vertical_energy
    splitting = np.sqrt(detuning**2 + 4.0 * case.collective_g**2)
    if splitting == 0.0:
        sine = 0.0
        cosine = 1.0
    else:
        cosine_squared = 0.5 * (1.0 - detuning / splitting)
        sine_squared = 0.5 * (1.0 + detuning / splitting)
        cosine = float(np.sqrt(max(0.0, cosine_squared)))
        sine = float(np.sqrt(max(0.0, sine_squared)))
    rotation = np.asarray(
        [[sine, cosine], [cosine, -sine]], dtype=float
    )
    matter = np.asarray([cosine, -sine], dtype=float)
    return rotation, matter


def _invert_two_by_two(
    matrix_00: np.ndarray,
    matrix_01: np.ndarray,
    matrix_10: np.ndarray,
    matrix_11: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    determinant = matrix_00 * matrix_11 - matrix_01 * matrix_10
    return (
        matrix_11 / determinant,
        -matrix_01 / determinant,
        -matrix_10 / determinant,
        matrix_00 / determinant,
    )


def polariton_inverse(
    omega: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    self_energy_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four components of inverse G_pol in (P_+, P_-) ordering.

    The bare inverse is the exact rotation of the finite-loss cavity--bright
    block.  ``polariton_diagonal`` changes only the off-diagonal projection
    of the matter self-energy; it does not diagonalize the bare inverse.
    """
    rotation, matter = hopfield_transform(case)
    sine, cosine = rotation[0, 0], rotation[0, 1]
    photon = (
        omega
        - case.omega_c
        + 1.0j * (case.cavity_halfwidth + case.eta)
    )
    bare_exciton = omega - case.vertical_energy + 1.0j * case.eta
    bare_00 = (
        sine**2 * photon
        + cosine**2 * bare_exciton
        - 2.0 * sine * cosine * case.collective_g
    )
    bare_11 = (
        cosine**2 * photon
        + sine**2 * bare_exciton
        + 2.0 * sine * cosine * case.collective_g
    )
    bare_01 = (
        sine * cosine * (photon - bare_exciton)
        + (sine**2 - cosine**2) * case.collective_g
    )
    inverse_00 = bare_00 - sigma * matter[0] ** 2
    inverse_11 = bare_11 - sigma * matter[1] ** 2
    if self_energy_mode == "full":
        inverse_01 = bare_01 - sigma * matter[0] * matter[1]
    elif self_energy_mode == "polariton_diagonal":
        inverse_01 = bare_01
    else:
        raise ValueError(
            "self_energy_mode must be 'full' or 'polariton_diagonal'"
        )
    return inverse_00, inverse_01, inverse_01, inverse_11


def polariton_green(
    omega: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    self_energy_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four components of G_pol in (P_+, P_-) ordering."""
    inverse_00, inverse_01, inverse_10, inverse_11 = polariton_inverse(
        omega, sigma, case, self_energy_mode
    )
    return _invert_two_by_two(
        inverse_00, inverse_01, inverse_10, inverse_11
    )


def local_green(
    omega: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    self_energy_mode: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local, bright, and dark retarded exciton Green functions."""
    photon = (
        omega
        - case.omega_c
        + 1.0j * (case.cavity_halfwidth + case.eta)
    )
    bare_exciton = omega - case.vertical_energy + 1.0j * case.eta
    dark = 1.0 / (bare_exciton - sigma)
    if self_energy_mode == "full":
        determinant = photon * (bare_exciton - sigma) - case.collective_g**2
        bright = photon / determinant
    elif self_energy_mode == "polariton_diagonal":
        _, matter = hopfield_transform(case)
        g00, g01, g10, g11 = polariton_green(
            omega, sigma, case, self_energy_mode
        )
        bright = (
            matter[0] ** 2 * g00
            + matter[0] * matter[1] * (g01 + g10)
            + matter[1] ** 2 * g11
        )
    else:
        raise ValueError(
            "self_energy_mode must be 'full' or 'polariton_diagonal'"
        )
    local = (
        bright + (case.n_molecules - 1.0) * dark
    ) / case.n_molecules
    return local, bright, dark


def sigma_map_tangent(
    grid: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    coupling: np.ndarray,
    omega_q: np.ndarray,
    occupation: np.ndarray,
    bath_chunk: int,
    self_energy_mode: str,
) -> np.ndarray:
    r"""Apply Sigma(w)=sum_q c_q^2[(nbar(wq)+1)Gloc(w-wq)+nbar(wq)Gloc(w+wq)]."""
    result = np.zeros_like(sigma)
    row = grid[None, :]
    for start in range(0, len(omega_q), bath_chunk):
        stop = min(start + bath_chunk, len(omega_q))
        q = omega_q[start:stop, None]
        c2 = coupling[start:stop, None] ** 2
        nq = occupation[start:stop, None]
        emit = row - q
        absorb = row + q
        sigma_emit = interpolate_complex(emit, grid, sigma)
        sigma_absorb = interpolate_complex(absorb, grid, sigma)
        green_emit, _, _ = local_green(
            emit, sigma_emit, case, self_energy_mode
        )
        green_absorb, _, _ = local_green(
            absorb, sigma_absorb, case, self_energy_mode
        )
        result += np.sum(
            c2 * ((1.0 + nq) * green_emit + nq * green_absorb),
            axis=0,
        )
    return result


# Backward-compatible name for independent diagnostic scripts that retain the
# tangent-grid implementation as an explicitly selected alternative.
sigma_map = sigma_map_tangent


def bare_local_tail_coefficients(
    grid: np.ndarray,
    case: Case,
    order: int = 7,
) -> list[np.ndarray]:
    """Return coefficients of the bare local-G expansion below the grid."""
    if order < 1:
        raise ValueError("tail expansion order must be positive")

    exciton_pole = case.vertical_energy - 1.0j * case.eta
    cavity_pole = case.omega_c - 1.0j * (
        case.cavity_halfwidth + case.eta
    )
    a00 = grid - cavity_pole
    a01 = -case.collective_g * np.ones_like(grid, dtype=complex)
    a10 = a01.copy()
    a11 = grid - exciton_pole

    p00 = np.ones_like(grid, dtype=complex)
    p01 = np.zeros_like(grid, dtype=complex)
    p10 = np.zeros_like(grid, dtype=complex)
    p11 = np.ones_like(grid, dtype=complex)
    dark_power = np.ones_like(grid, dtype=complex)
    dark_factor = grid - exciton_pole
    coefficients: list[np.ndarray] = []
    for _ in range(order):
        coefficients.append(
            (
                p11
                + (case.n_molecules - 1.0) * dark_power
            )
            / case.n_molecules
        )
        new00 = p00 * a00 + p01 * a10
        new01 = p00 * a01 + p01 * a11
        new10 = p10 * a00 + p11 * a10
        new11 = p10 * a01 + p11 * a11
        p00, p01, p10, p11 = new00, new01, new10, new11
        dark_power = dark_power * dark_factor
    return coefficients


def sigma_map_convolution(
    grid: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    convolver: LinearDrudeConvolver,
    self_energy_mode: str,
) -> np.ndarray:
    """Apply the continuous Drude Fan kernel by linear FFT convolution."""
    green, _, _ = local_green(
        grid, sigma, case, self_energy_mode
    )
    return convolver.apply(
        green,
        tail_coefficients=bare_local_tail_coefficients(grid, case),
        tail_minimum_cutoff=1.8 / EV_PER_AU,
    )


def make_frequency_grid(
    case: Case,
    bath_method: str,
    n_omega: int | None,
    omega_step_meV: float | None,
    omega_min_eV: float,
    omega_max_eV: float,
) -> np.ndarray:
    """Construct the global self-energy grid for the selected bath method."""
    if omega_min_eV >= omega_max_eV:
        raise ValueError("invalid SCBA frequency interval")
    if bath_method == "convolution":
        if n_omega is None:
            if omega_step_meV is None:
                eta_meV = case.eta * EV_PER_AU * 1.0e3
                omega_step_meV = min(0.05, eta_meV)
            if omega_step_meV <= 0.0:
                raise ValueError("omega_step_meV must be positive")
            intervals = int(
                round(
                    (omega_max_eV - omega_min_eV)
                    * 1.0e3
                    / omega_step_meV
                )
            )
            n_omega = intervals + 1
        if n_omega < 2:
            raise ValueError("n_omega must be at least two")
    elif bath_method == "tangent":
        if n_omega is None:
            n_omega = 4000
    else:
        raise ValueError(
            "bath_method must be 'convolution' or 'tangent'"
        )
    return np.linspace(
        omega_min_eV / EV_PER_AU,
        omega_max_eV / EV_PER_AU,
        n_omega,
    )


def solve(
    case: Case,
    n_bath: int = 1000,
    n_omega: int | None = None,
    omega_min_eV: float = -1.0,
    omega_max_eV: float = 4.0,
    omega_step_meV: float | None = None,
    max_iterations: int = 200,
    tolerance: float = 1.0e-8,
    bath_chunk: int = 50,
    self_energy_mode: str = "full",
    bath_method: str = "convolution",
) -> Result:
    """Solve all frequencies together, using the one-shot BA as the seed."""
    grid = make_frequency_grid(
        case,
        bath_method,
        n_omega,
        omega_step_meV,
        omega_min_eV,
        omega_max_eV,
    )
    zeros = np.zeros_like(grid, dtype=complex)
    if bath_method == "convolution":
        convolver = LinearDrudeConvolver(
            grid,
            case.reorganization,
            case.bath_cutoff,
            K_PER_AU / case.temperature_K,
        )

        def apply_map(current_sigma: np.ndarray) -> np.ndarray:
            return sigma_map_convolution(
                grid,
                current_sigma,
                case,
                convolver,
                self_energy_mode,
            )

    else:
        coupling, omega_q = tangent_bath(
            case.reorganization, case.bath_cutoff, n_bath
        )
        occupation = bose(omega_q, case.temperature_K)

        def apply_map(current_sigma: np.ndarray) -> np.ndarray:
            return sigma_map_tangent(
                grid,
                current_sigma,
                case,
                coupling,
                omega_q,
                occupation,
                bath_chunk,
                self_energy_mode,
            )

    sigma = apply_map(zeros)

    target = 2.0 / EV_PER_AU
    target_value = interpolate_complex(
        np.array([target]), grid, sigma
    )[0]
    history: List[Tuple[float, float, float, float, float]] = [
        (0.0, target_value.real, target_value.imag, np.nan, 1.0)
    ]
    previous_error = np.inf
    mixing = 1.0
    final_error = np.inf

    for iteration in range(1, max_iterations + 1):
        mapped = apply_map(sigma)
        residual = mapped - sigma
        final_error = float(np.max(np.abs(residual)))
        if final_error > 1.25 * previous_error:
            mixing = max(0.2, 0.5 * mixing)
        sigma = sigma + mixing * residual
        target_value = interpolate_complex(
            np.array([target]), grid, sigma
        )[0]
        history.append(
            (
                float(iteration),
                target_value.real,
                target_value.imag,
                final_error,
                mixing,
            )
        )
        if final_error < tolerance:
            break
        previous_error = final_error
    else:
        raise RuntimeError(
            "SCBA did not converge for {} (last error {:.6e} E_h)".format(
                case.script, final_error
            )
        )

    max_positive_imag = float(np.max(sigma.imag))
    if max_positive_imag > 10.0 * tolerance:
        raise RuntimeError(
            "Retarded-causality check failed for {}: max Im Sigma={:.6e}".format(
                case.script, max_positive_imag
            )
        )

    output_sigma = interpolate_complex(case.output_grid, grid, sigma)
    _, output_bright, _ = local_green(
        case.output_grid, output_sigma, case, self_energy_mode
    )
    return Result(
        bath_method=bath_method,
        self_energy_mode=self_energy_mode,
        grid=grid,
        sigma=sigma,
        output_sigma=output_sigma,
        output_green_bright=output_bright,
        history=np.asarray(history, dtype=float),
        iterations=iteration,
        final_error=final_error,
        final_mixing=mixing,
    )


def solve_single_shot_ba(
    case: Case,
    n_bath: int = 1000,
    n_omega: int | None = None,
    omega_min_eV: float = -1.0,
    omega_max_eV: float = 4.0,
    omega_step_meV: float | None = None,
    bath_chunk: int = 50,
    self_energy_mode: str = "full",
    bath_method: str = "convolution",
) -> Result:
    """Evaluate the Fan map once with the bare on-site propagator."""
    grid = make_frequency_grid(
        case,
        bath_method,
        n_omega,
        omega_step_meV,
        omega_min_eV,
        omega_max_eV,
    )
    zeros = np.zeros_like(grid, dtype=complex)
    if bath_method == "convolution":
        convolver = LinearDrudeConvolver(
            grid,
            case.reorganization,
            case.bath_cutoff,
            K_PER_AU / case.temperature_K,
        )
        sigma = sigma_map_convolution(
            grid,
            zeros,
            case,
            convolver,
            self_energy_mode,
        )
    elif bath_method == "tangent":
        coupling, omega_q = tangent_bath(
            case.reorganization, case.bath_cutoff, n_bath
        )
        occupation = bose(omega_q, case.temperature_K)
        sigma = sigma_map_tangent(
            grid,
            zeros,
            case,
            coupling,
            omega_q,
            occupation,
            bath_chunk,
            self_energy_mode,
        )
    else:
        raise ValueError(
            "bath_method must be 'convolution' or 'tangent'"
        )

    max_positive_imag = float(np.max(sigma.imag))
    if max_positive_imag > 1.0e-8:
        raise RuntimeError(
            "Retarded-causality check failed for {}: max Im Sigma={:.6e}".format(
                case.script, max_positive_imag
            )
        )
    output_sigma = interpolate_complex(case.output_grid, grid, sigma)
    _, output_bright, _ = local_green(
        case.output_grid, output_sigma, case, self_energy_mode
    )
    target = 2.0 / EV_PER_AU
    target_value = interpolate_complex(
        np.array([target]), grid, sigma
    )[0]
    return Result(
        bath_method=bath_method,
        self_energy_mode=self_energy_mode,
        grid=grid,
        sigma=sigma,
        output_sigma=output_sigma,
        output_green_bright=output_bright,
        history=np.asarray(
            [(0.0, target_value.real, target_value.imag, 0.0, 1.0)],
            dtype=float,
        ),
        iterations=1,
        final_error=0.0,
        final_mixing=1.0,
    )


def numerical_header(result: Result, n_bath: int) -> str:
    """Describe the bath integration and global frequency grid."""
    step_meV = (
        (result.grid[1] - result.grid[0]) * EV_PER_AU * 1.0e3
    )
    if result.bath_method == "convolution":
        bath_description = (
            "bath_method=continuous_linear_fft "
            "tail=asymptotic_Drude"
        )
    else:
        bath_description = (
            f"bath_method=tangent n_bath={n_bath}"
        )
    return (
        f"{bath_description} n_omega={len(result.grid)} "
        f"omega_step_meV={step_meV:.8f} "
        "omega_grid_eV=[{:.6f},{:.6f}]"
    ).format(
        result.grid[0] * EV_PER_AU,
        result.grid[-1] * EV_PER_AU,
    )


def write_result(case: Case, result: Result, n_bath: int) -> None:
    directory = case.script.parent
    if result.self_energy_mode == "full":
        spectrum_name = "Aw_sc_fullpol.txt"
        sigma_name = "Sigma_sc_fullpol.txt"
        iteration_name = "Sigma_iter_2.0eV.txt"
        convergence_name = "scba_convergence.txt"
    else:
        spectrum_name = "Aw_sc_polariton_diagonal.txt"
        sigma_name = "Sigma_sc_polariton_diagonal.txt"
        iteration_name = "Sigma_iter_polariton_diagonal_2.0eV.txt"
        convergence_name = "scba_convergence_polariton_diagonal.txt"
    aw = np.column_stack(
        (case.output_grid, np.imag(result.output_green_bright))
    )
    np.savetxt(
        str(directory / spectrum_name),
        aw,
        header=(
            "omega_au Im_G_bright_au^-1; frequency-coupled SCBA; "
            f"self_energy_mode={result.self_energy_mode}; "
            f"{numerical_header(result, n_bath)}"
        ),
    )
    sigma_output = np.column_stack(
        (
            case.output_grid,
            result.output_sigma.real,
            result.output_sigma.imag,
        )
    )
    np.savetxt(
        str(directory / sigma_name),
        sigma_output,
        header=(
            "omega_au Re_Sigma_au Im_Sigma_au; G-only SCBA; "
            f"{numerical_header(result, n_bath)}"
        ),
    )
    np.savetxt(
        str(directory / iteration_name),
        result.history[:, :3],
        header="iteration Re_Sigma_au Im_Sigma_au at 2.0 eV",
    )
    np.savetxt(
        str(directory / convergence_name),
        result.history,
        header=(
            "iteration Re_Sigma_au Im_Sigma_au "
            "max_residual_au mixing\n{}".format(
                numerical_header(result, n_bath)
            )
        ),
    )


def write_single_shot_ba_result(
    case: Case,
    result: Result,
    n_bath: int,
) -> None:
    """Write the legacy Fig. 3 BA files from the shared continuum kernel."""
    directory = case.script.parent
    np.savetxt(
        str(directory / "Aw_single_shot_ba_fullpol.txt"),
        np.column_stack(
            (case.output_grid, np.imag(result.output_green_bright))
        ),
        header=(
            "omega_au Im_G_bright_au^-1; single-shot BA; "
            f"{numerical_header(result, n_bath)}"
        ),
    )
    np.savetxt(
        str(directory / "Sigma_single_shot_ba_fullpol.txt"),
        np.column_stack(
            (
                case.output_grid,
                result.output_sigma.real,
                result.output_sigma.imag,
            )
        ),
        header=(
            "omega_au Re_Sigma_au Im_Sigma_au; single-shot BA; "
            f"{numerical_header(result, n_bath)}"
        ),
    )
    target = interpolate_complex(
        np.asarray([2.0 / EV_PER_AU]),
        case.output_grid,
        result.output_sigma,
    )[0]
    np.savetxt(
        str(directory / "Sigma_single_shot_ba_2.0eV.txt"),
        np.asarray([[target.real, target.imag]]),
        header="sigma_real_au sigma_imag_au at 2.0 eV",
    )


def run_legacy_script(
    script: Path,
    n_bath: int = None,
    n_omega: int = None,
    self_energy_mode: str = None,
    bath_method: str = None,
    omega_step_meV: float = None,
) -> Dict[str, object]:
    """Entry point used by every revised ``EGF_sc_fullpol.py``."""
    if n_bath is None:
        n_bath = int(os.environ.get("SCBA_N_BATH", "1000"))
    if self_energy_mode is None:
        self_energy_mode = os.environ.get("SCBA_SELF_ENERGY_MODE", "full")
    if bath_method is None:
        bath_method = os.environ.get(
            "SCBA_BATH_METHOD", "convolution"
        )
    if n_omega is None and "SCBA_N_OMEGA" in os.environ:
        n_omega = int(os.environ["SCBA_N_OMEGA"])
    if omega_step_meV is None and "SCBA_OMEGA_STEP_MEV" in os.environ:
        omega_step_meV = float(os.environ["SCBA_OMEGA_STEP_MEV"])
    case = parse_legacy_case(Path(script))
    result = solve(
        case,
        n_bath=n_bath,
        n_omega=n_omega,
        self_energy_mode=self_energy_mode,
        bath_method=bath_method,
        omega_step_meV=omega_step_meV,
    )
    write_result(case, result, n_bath)
    summary = {
        "script": str(case.script),
        "n_molecules": case.n_molecules,
        "omega_c_eV": case.omega_c * EV_PER_AU,
        "iterations": result.iterations,
        "residual_meV": result.final_error * EV_PER_AU * 1.0e3,
        "mixing": result.final_mixing,
        "n_bath": n_bath,
        "n_omega": len(result.grid),
        "omega_step_meV": (
            (result.grid[1] - result.grid[0])
            * EV_PER_AU
            * 1.0e3
        ),
        "bath_method": bath_method,
        "self_energy_mode": self_energy_mode,
    }
    print(
        "{script}: {self_energy_mode} converged in {iterations} iterations; "
        "residual={residual_meV:.3e} meV".format(**summary)
    )
    return summary


def run_legacy_ba_script(
    script: Path,
    n_bath: int = None,
    n_omega: int = None,
    bath_method: str = None,
    omega_step_meV: float = None,
) -> Dict[str, object]:
    """Generate the Fig. 3 single-shot BA files with the shared kernel."""
    if n_bath is None:
        n_bath = int(os.environ.get("SCBA_N_BATH", "1000"))
    if bath_method is None:
        bath_method = os.environ.get(
            "SCBA_BATH_METHOD", "convolution"
        )
    if n_omega is None and "SCBA_N_OMEGA" in os.environ:
        n_omega = int(os.environ["SCBA_N_OMEGA"])
    if omega_step_meV is None and "SCBA_OMEGA_STEP_MEV" in os.environ:
        omega_step_meV = float(os.environ["SCBA_OMEGA_STEP_MEV"])
    case = parse_legacy_case(Path(script))
    result = solve_single_shot_ba(
        case,
        n_bath=n_bath,
        n_omega=n_omega,
        bath_method=bath_method,
        omega_step_meV=omega_step_meV,
    )
    write_single_shot_ba_result(case, result, n_bath)
    summary = {
        "script": str(case.script),
        "n_molecules": case.n_molecules,
        "omega_c_eV": case.omega_c * EV_PER_AU,
        "iterations": 1,
        "residual_meV": 0.0,
        "mixing": 1.0,
        "n_bath": n_bath,
        "n_omega": len(result.grid),
        "omega_step_meV": (
            (result.grid[1] - result.grid[0])
            * EV_PER_AU
            * 1.0e3
        ),
        "bath_method": bath_method,
        "self_energy_mode": "full",
    }
    print(
        "{script}: single-shot BA evaluated with {bath_method}; "
        "n_omega={n_omega}".format(**summary)
    )
    return summary
