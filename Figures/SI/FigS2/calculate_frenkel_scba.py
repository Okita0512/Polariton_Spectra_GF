#!/usr/bin/env python3
"""
G-only and coupled G-D SCBA benchmarks for the FigS2 Frenkel model.

The physical state is 300 K thermal equilibrium for the Drude phonon bath
and the zero-excitation vacuum for the electronic transition.  The latter
implies G_xx^< = 0 and therefore a vanishing connected Holstein density
polarization.  The explicit G-D Dyson loop consequently returns D = D0; it
must not manufacture phonon screening without an exciton population.

Atomic units are used internally.  Energies written to disk are in eV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """Locate the repository from stable project markers after figure moves."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "main.tex").is_file()
            and (candidate / "SI.tex").is_file()
            and (candidate / "Figures" / "common" / "drude_convolution.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate the project root above {start}; expected "
        "main.tex, SI.tex, and Figures/common/drude_convolution.py"
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))
from drude_convolution import LinearDrudeConvolver


EV_PER_AU = 27.211397
K_PER_AU = 3.1577464e5
LAMBDAS_MEV = (15.0, 30.0, 45.0, 60.0)
CUTOFFS_MEV = (12.4, 24.8, 37.2)


@dataclass(frozen=True)
class Model:
    lambda_mev: float
    cutoff_mev: float
    temperature_K: float = 300.0
    energy_00_eV: float = 2.0
    eta_eV: float = 0.0001
    exciton_occupation: float = 0.0

    @property
    def reorganization(self) -> float:
        return self.lambda_mev * 1.0e-3 / EV_PER_AU

    @property
    def cutoff(self) -> float:
        return self.cutoff_mev * 1.0e-3 / EV_PER_AU

    @property
    def vertical_energy(self) -> float:
        return (
            self.energy_00_eV + self.lambda_mev * 1.0e-3
        ) / EV_PER_AU

    @property
    def eta(self) -> float:
        return self.eta_eV / EV_PER_AU


@dataclass(frozen=True)
class Numerics:
    bath_method: str = "convolution"
    n_bath: int = 1000
    bath_chunk: int = 50
    omega_min_eV: float = -1.0
    omega_max_eV: float = 4.0
    omega_step_meV: float = 0.05
    n_omega: int = 100001
    output_min_eV: float = 0.8
    output_max_eV: float = 3.2
    n_output: int = 4000
    phonon_min_eV: float = -3.2
    phonon_max_eV: float = 3.2
    n_phonon: int = 8001
    convolution_chunk: int = 32
    max_iterations: int = 300
    mixing_g: float = 0.25
    mixing_d: float = 0.25
    tolerance_g: float = 1.0e-8
    tolerance_d: float = 1.0e-11
    error_floor: float = 1.0e-15


@dataclass
class ScbaResult:
    mode: str
    omega: np.ndarray
    sigma: np.ndarray
    green: np.ndarray
    nu: np.ndarray
    phonon: np.ndarray
    polarization: np.ndarray
    iterations: int
    error_g: float
    error_d: float
    converged: bool
    bath_method: str

    @property
    def spectrum(self) -> np.ndarray:
        return -np.imag(self.green)


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else values.copy()


def bose(omega: np.ndarray, model: Model) -> np.ndarray:
    """Stable Bose-Einstein factor for strictly positive frequencies."""
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0.0):
        raise ValueError("Bath frequencies must be strictly positive")
    if model.temperature_K < 0.0:
        raise ValueError("Temperature must be nonnegative")
    if model.temperature_K == 0.0:
        return np.zeros_like(omega)
    x = K_PER_AU * omega / model.temperature_K
    result = np.zeros_like(x, dtype=float)
    mask = x < 700.0
    result[mask] = 1.0 / np.expm1(x[mask])
    return result


def tangent_bath(
    model: Model,
    numerics: Numerics,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretize J(w)=2 lambda gamma w/(w^2+gamma^2).

    With this tangent grid, sum_q c_q^2 f(w_q) approaches
    int_0^infinity dw J(w) f(w)/pi.
    """
    index = np.arange(1, numerics.n_bath + 1, dtype=float)
    omega_q = model.cutoff * np.tan(
        0.5
        * np.pi
        * (1.0 - index / (numerics.n_bath + 1.0))
    )
    coupling = np.sqrt(
        model.reorganization * omega_q / (numerics.n_bath + 1.0)
    )
    return coupling, omega_q


def interpolate_complex(
    points: np.ndarray,
    grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with the zero self-energy asymptote off-grid."""
    shape = points.shape
    flat = points.ravel()
    real = np.interp(flat, grid, values.real, left=0.0, right=0.0)
    imag = np.interp(flat, grid, values.imag, left=0.0, right=0.0)
    return (real + 1.0j * imag).reshape(shape)


def electronic_green(
    omega: np.ndarray,
    sigma: np.ndarray,
    model: Model,
) -> np.ndarray:
    """Retarded Frenkel-transition Green function, normalized as 1/omega."""
    return 1.0 / (
        omega - model.vertical_energy + 1.0j * model.eta - sigma
    )


def collective_bare_phonon(
    nu: np.ndarray,
    model: Model,
) -> np.ndarray:
    r"""
    Retarded propagator of X=sum_q c_q(b_q+b_q^dagger).

    For the Drude density,

        D_X,0^R(nu) = -2 lambda gamma/(gamma-i nu),
        -2 Im D_X,0^R(nu) = 2 J(nu).

    D_X has units of energy because the couplings c_q are included.
    """
    lam = model.reorganization
    gamma = model.cutoff
    return -2.0 * lam * gamma / (gamma - 1.0j * nu)


def phonon_dyson(
    nu: np.ndarray,
    polarization: np.ndarray,
    model: Model,
) -> np.ndarray:
    """Solve D_X^R=[(D_X,0^R)^(-1)-Pi_nn^R]^(-1)."""
    d0 = collective_bare_phonon(nu, model)
    if not np.any(polarization):
        return d0.copy()
    return 1.0 / (1.0 / d0 - polarization)


def bare_sigma_map(
    grid: np.ndarray,
    sigma: np.ndarray,
    model: Model,
    coupling: np.ndarray,
    omega_q: np.ndarray,
    occupation: np.ndarray,
    numerics: Numerics,
) -> np.ndarray:
    r"""
    Existing frequency-coupled G-only skeleton map.

    Sigma^R(w)=sum_q c_q^2[(nbar(w_q)+1)G^R(w-w_q)+nbar(w_q)G^R(w+w_q)].
    """
    result = np.zeros_like(sigma)
    grid_row = grid[None, :]

    for start in range(0, numerics.n_bath, numerics.bath_chunk):
        stop = min(start + numerics.bath_chunk, numerics.n_bath)
        q = omega_q[start:stop, None]
        c2 = coupling[start:stop, None] ** 2
        nq = occupation[start:stop, None]
        emit = grid_row - q
        absorb = grid_row + q
        sigma_emit = interpolate_complex(emit, grid, sigma)
        sigma_absorb = interpolate_complex(absorb, grid, sigma)
        g_emit = electronic_green(emit, sigma_emit, model)
        g_absorb = electronic_green(absorb, sigma_absorb, model)
        result += np.sum(
            c2 * ((1.0 + nq) * g_emit + nq * g_absorb),
            axis=0,
        )
    return result


def continuous_sigma_map(
    grid: np.ndarray,
    sigma: np.ndarray,
    model: Model,
    convolver: LinearDrudeConvolver,
) -> np.ndarray:
    """Evaluate the continuous Drude Fan map by linear FFT convolution."""
    green = electronic_green(grid, sigma, model)
    pole = model.vertical_energy - 1.0j * model.eta
    coefficients = [
        (grid - pole) ** order for order in range(7)
    ]
    return convolver.apply(
        green,
        tail_coefficients=coefficients,
        tail_minimum_cutoff=1.8 / EV_PER_AU,
    )


def make_convolver(
    omega: np.ndarray,
    model: Model,
) -> LinearDrudeConvolver:
    return LinearDrudeConvolver(
        omega,
        model.reorganization,
        model.cutoff,
        K_PER_AU / model.temperature_K,
    )


def dressed_sigma_map(
    grid: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    phonon: np.ndarray,
    model: Model,
    numerics: Numerics,
) -> np.ndarray:
    r"""
    Full equilibrium convolution for a dressed collective bath line.

    Sigma^R(w)=int_0^infinity dnu/(2pi) B_X(nu)
      [(nbar(nu)+1)G^R(w-nu)+nbar(nu) G^R(w+nu)].

    Direct quadrature avoids circular wrapping.  Shifted self-energies use
    nonperiodic interpolation and their zero high-frequency asymptote.
    """
    positive = nu > 0.0
    frequencies = nu[positive]
    spectral = -2.0 * np.imag(phonon[positive])
    if float(np.min(spectral)) < -1.0e-12:
        raise RuntimeError(
            "Dressed bath has negative positive-frequency spectral weight"
        )

    delta_nu = float(nu[1] - nu[0])
    weights = spectral * delta_nu / (2.0 * np.pi)
    threshold = max(float(np.max(np.abs(weights))) * 1.0e-12, 1.0e-30)
    selected = np.abs(weights) > threshold
    frequencies = frequencies[selected]
    weights = weights[selected]
    occupation = bose(frequencies, model)

    result = np.zeros_like(sigma)
    for start in range(0, len(frequencies), numerics.convolution_chunk):
        stop = min(start + numerics.convolution_chunk, len(frequencies))
        q = frequencies[start:stop, None]
        weight = weights[start:stop, None]
        nq = occupation[start:stop, None]
        emit = grid[None, :] - q
        absorb = grid[None, :] + q
        g_emit = electronic_green(
            emit,
            interpolate_complex(emit, grid, sigma),
            model,
        )
        g_absorb = electronic_green(
            absorb,
            interpolate_complex(absorb, grid, sigma),
            model,
        )
        result += np.sum(
            weight * ((1.0 + nq) * g_emit + nq * g_absorb),
            axis=0,
        )
    return result


def vacuum_phonon_polarization(
    nu: np.ndarray,
    omega: np.ndarray,
    green_retarded: np.ndarray,
    model: Model,
) -> np.ndarray:
    r"""
    Connected density polarization for the actual benchmark state.

    The excitation is a restricted optical transition in its vacuum, not an
    occupied fermion band.  Thus G_xx^<(w)=0 even though the phonon bath is at
    300 K.  At skeleton level,

        Pi^<(nu) ~ -i int dw/(2pi) G^<(w)G^>(w-nu) = 0,
        Pi^>(nu) ~ -i int dw/(2pi) G^>(w)G^<(w-nu) = 0,

    and Pi^R=0.  A nonzero result would require an explicitly defined pumped
    or finite-density electronic distribution.
    """
    if model.exciton_occupation != 0.0:
        raise NotImplementedError(
            "Finite-density phonon polarization requires a specified "
            "electronic distribution or pump model."
        )
    if omega.shape != green_retarded.shape:
        raise ValueError("Frequency grid and retarded Green function must match")
    return np.zeros_like(nu, dtype=complex)


def relative_change(
    new: np.ndarray,
    old: np.ndarray,
    floor: float,
) -> float:
    return float(
        np.max(np.abs(new - old)) / (np.max(np.abs(new)) + floor)
    )


def report_iteration(
    mode: str,
    iteration: int,
    error_g: float,
    error_d: float,
    final: bool = False,
) -> None:
    if final or iteration == 1 or iteration % 25 == 0:
        print(
            f"{mode:7s} iteration {iteration:3d}: "
            f"epsilon_G={error_g:.3e}, epsilon_D={error_d:.3e}"
        )


def make_grids(numerics: Numerics) -> tuple[np.ndarray, np.ndarray]:
    omega = np.linspace(
        numerics.omega_min_eV / EV_PER_AU,
        numerics.omega_max_eV / EV_PER_AU,
        numerics.n_omega,
    )
    nu = np.linspace(
        numerics.phonon_min_eV / EV_PER_AU,
        numerics.phonon_max_eV / EV_PER_AU,
        numerics.n_phonon,
    )
    return omega, nu


def solve_single_shot_ba(
    model: Model,
    numerics: Numerics,
) -> ScbaResult:
    """Evaluate the finite-temperature Fan kernel once with bare G."""
    omega, nu = make_grids(numerics)
    d0 = collective_bare_phonon(nu, model)
    zeros = np.zeros_like(omega, dtype=complex)
    if numerics.bath_method == "convolution":
        sigma_ba = continuous_sigma_map(
            omega, zeros, model, make_convolver(omega, model)
        )
    elif numerics.bath_method == "tangent":
        coupling, omega_q = tangent_bath(model, numerics)
        occupation = bose(omega_q, model)
        sigma_ba = bare_sigma_map(
            omega,
            zeros,
            model,
            coupling,
            omega_q,
            occupation,
            numerics,
        )
    else:
        raise ValueError(
            "bath_method must be 'convolution' or 'tangent'"
        )
    green_ba = electronic_green(omega, sigma_ba, model)
    for name, values in (
        ("self-energy", sigma_ba),
        ("Green function", green_ba),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Single-shot BA {name} contains NaN or Inf")
    if float(np.max(sigma_ba.imag)) > 1.0e-8:
        raise RuntimeError("Single-shot BA self-energy violates causality")
    if float(np.min(-green_ba.imag)) < -1.0e-8:
        raise RuntimeError("Single-shot BA spectrum violates causality")
    return ScbaResult(
        "single_shot_ba",
        omega,
        sigma_ba,
        green_ba,
        nu,
        d0,
        np.zeros_like(d0),
        1,
        0.0,
        0.0,
        True,
        numerics.bath_method,
    )


def solve_g_only(
    model: Model,
    numerics: Numerics,
    omega_grid: np.ndarray | None = None,
    initial_sigma: np.ndarray | None = None,
) -> ScbaResult:
    default_omega, nu = make_grids(numerics)
    if omega_grid is None:
        omega = default_omega
    else:
        omega = np.asarray(omega_grid, dtype=float)
        if omega.ndim != 1 or len(omega) < 2:
            raise ValueError("omega_grid must be a one-dimensional grid")
        if not np.all(np.isfinite(omega)) or not np.all(np.diff(omega) > 0.0):
            raise ValueError("omega_grid must be finite and strictly increasing")
    d0 = collective_bare_phonon(nu, model)
    if numerics.bath_method == "convolution":
        convolver = make_convolver(omega, model)

        def apply_map(current_sigma: np.ndarray) -> np.ndarray:
            return continuous_sigma_map(
                omega, current_sigma, model, convolver
            )

    elif numerics.bath_method == "tangent":
        coupling, omega_q = tangent_bath(model, numerics)
        occupation = bose(omega_q, model)

        def apply_map(current_sigma: np.ndarray) -> np.ndarray:
            return bare_sigma_map(
                omega,
                current_sigma,
                model,
                coupling,
                omega_q,
                occupation,
                numerics,
            )

    else:
        raise ValueError(
            "bath_method must be 'convolution' or 'tangent'"
        )

    if initial_sigma is None:
        sigma = apply_map(
            np.zeros_like(omega, dtype=complex)
        )
    else:
        sigma = np.asarray(initial_sigma, dtype=complex).copy()
        if sigma.shape != omega.shape or not np.all(np.isfinite(sigma)):
            raise ValueError(
                "initial_sigma must be finite and match omega_grid"
            )
    g_old = electronic_green(omega, sigma, model)

    for iteration in range(1, numerics.max_iterations + 1):
        sigma_map = apply_map(sigma)
        sigma_new = (
            (1.0 - numerics.mixing_g) * sigma
            + numerics.mixing_g * sigma_map
        )
        g_new = electronic_green(omega, sigma_new, model)
        error_g = relative_change(g_new, g_old, numerics.error_floor)
        error_d = 0.0
        report_iteration("g_only", iteration, error_g, error_d)
        sigma = sigma_new
        g_old = g_new
        if error_g < numerics.tolerance_g:
            report_iteration(
                "g_only", iteration, error_g, error_d, final=True
            )
            return ScbaResult(
                "g_only",
                omega,
                sigma,
                g_new,
                nu,
                d0,
                np.zeros_like(d0),
                iteration,
                error_g,
                error_d,
                True,
                numerics.bath_method,
            )

    raise RuntimeError(
        f"G-only SCBA failed for lambda={model.lambda_mev}, "
        f"cutoff={model.cutoff_mev} meV"
    )


def solve_gd_scba(
    model: Model,
    numerics: Numerics,
    initial: ScbaResult,
) -> ScbaResult:
    omega = initial.omega
    nu = initial.nu
    d0 = collective_bare_phonon(nu, model)
    if numerics.bath_method == "convolution":
        convolver = make_convolver(omega, model)
    elif numerics.bath_method == "tangent":
        coupling, omega_q = tangent_bath(model, numerics)
        occupation = bose(omega_q, model)
    else:
        raise ValueError(
            "bath_method must be 'convolution' or 'tangent'"
        )

    sigma = initial.sigma.copy()
    polarization = np.zeros_like(d0)
    phonon = d0.copy()
    g_old = electronic_green(omega, sigma, model)
    d_old = phonon.copy()

    for iteration in range(1, numerics.max_iterations + 1):
        if np.max(np.abs(polarization)) <= numerics.error_floor:
            if numerics.bath_method == "convolution":
                sigma_map = continuous_sigma_map(
                    omega, sigma, model, convolver
                )
            else:
                sigma_map = bare_sigma_map(
                    omega,
                    sigma,
                    model,
                    coupling,
                    omega_q,
                    occupation,
                    numerics,
                )
        else:
            sigma_map = dressed_sigma_map(
                omega,
                sigma,
                nu,
                phonon,
                model,
                numerics,
            )
        sigma_new = (
            (1.0 - numerics.mixing_g) * sigma
            + numerics.mixing_g * sigma_map
        )
        g_new = electronic_green(omega, sigma_new, model)

        polarization_map = vacuum_phonon_polarization(
            nu, omega, g_new, model
        )
        polarization_new = (
            (1.0 - numerics.mixing_d) * polarization
            + numerics.mixing_d * polarization_map
        )
        d_new = phonon_dyson(nu, polarization_new, model)

        error_g = relative_change(g_new, g_old, numerics.error_floor)
        error_d = relative_change(d_new, d_old, numerics.error_floor)
        report_iteration("gd_scba", iteration, error_g, error_d)

        sigma = sigma_new
        polarization = polarization_new
        phonon = d_new
        g_old = g_new
        d_old = d_new
        if (
            error_g < numerics.tolerance_g
            and error_d < numerics.tolerance_d
        ):
            report_iteration(
                "gd_scba", iteration, error_g, error_d, final=True
            )
            return ScbaResult(
                "gd_scba",
                omega,
                sigma,
                g_new,
                nu,
                phonon,
                polarization,
                iteration,
                error_g,
                error_d,
                True,
                numerics.bath_method,
            )

    raise RuntimeError(
        f"G-D SCBA failed for lambda={model.lambda_mev}, "
        f"cutoff={model.cutoff_mev} meV"
    )


def normalized_l1_to_reference(
    omega: np.ndarray,
    spectrum: np.ndarray,
    reference_omega: np.ndarray,
    reference_spectrum: np.ndarray,
) -> float:
    mask = (
        (reference_omega >= 1.6)
        & (reference_omega <= 2.4)
    )
    x = reference_omega[mask]
    ref = normalize(reference_spectrum[mask])
    calculated = normalize(
        np.interp(x, omega, spectrum, left=0.0, right=0.0)
    )
    denominator = float(np.trapz(np.abs(ref), x))
    return float(np.trapz(np.abs(calculated - ref), x) / denominator)


def peak_positions(omega: np.ndarray, spectrum: np.ndarray) -> str:
    energy = omega * EV_PER_AU
    mask = (energy >= 1.6) & (energy <= 2.4)
    normalized = normalize(spectrum[mask])
    indices, properties = find_peaks(normalized, prominence=0.02)
    if len(indices) == 0:
        return ""
    prominences = properties["prominences"]
    chosen = indices[np.argsort(prominences)[-6:]]
    peaks = np.sort(energy[mask][chosen])
    return ";".join(f"{peak:.6f}" for peak in peaks)


def phonon_peak_and_fwhm(
    nu: np.ndarray,
    phonon: np.ndarray,
) -> tuple[float, float]:
    positive = nu > 0.0
    x = nu[positive] * EV_PER_AU
    y = -2.0 * np.imag(phonon[positive])
    peak_index = int(np.argmax(y))
    peak = float(x[peak_index])
    half = 0.5 * float(y[peak_index])
    above = np.flatnonzero(y >= half)
    if len(above) < 2:
        return peak, float("nan")
    return peak, float(x[above[-1]] - x[above[0]])


def existing_g_only_regression(
    root: Path,
    model: Model,
    result: ScbaResult,
) -> float:
    path = (
        root
        / "results"
        / f"scba_lambda_{model.lambda_mev:.0f}_"
        f"cutoff_{model.cutoff_mev:.1f}.txt"
    )
    if not path.exists():
        return float("nan")
    previous = np.loadtxt(path)
    current = np.interp(
        previous[:, 0],
        result.omega * EV_PER_AU,
        normalize(result.spectrum),
    )
    return float(
        np.max(np.abs(current - normalize(previous[:, 1])))
    )


def validate(
    model: Model,
    ba: ScbaResult,
    g_only: ScbaResult,
    gd: ScbaResult,
) -> dict[str, float]:
    d0 = collective_bare_phonon(gd.nu, model)
    b = -2.0 * np.imag(gd.phonon)
    positive = gd.nu > 0.0
    checks = {
        "min_minus_im_g": float(
            min(
                np.min(ba.spectrum),
                np.min(g_only.spectrum),
                np.min(gd.spectrum),
            )
        ),
        "max_positive_im_sigma": float(
            max(
                np.max(ba.sigma.imag),
                np.max(g_only.sigma.imag),
                np.max(gd.sigma.imag),
            )
        ),
        "max_abs_pi": float(np.max(np.abs(gd.polarization))),
        "max_abs_d_minus_d0": float(
            np.max(np.abs(gd.phonon - d0))
        ),
        "phonon_odd_error": float(np.max(np.abs(b + b[::-1]))),
        "min_positive_phonon_weight": float(np.min(b[positive])),
    }
    tolerance = 1.0e-8
    if checks["min_minus_im_g"] < -tolerance:
        raise RuntimeError(
            f"Electronic causality failed: {checks['min_minus_im_g']:.3e}"
        )
    if checks["max_positive_im_sigma"] > tolerance:
        raise RuntimeError(
            "Electronic damping failed: "
            f"{checks['max_positive_im_sigma']:.3e}"
        )
    if checks["phonon_odd_error"] > tolerance:
        raise RuntimeError(
            f"Bath odd-frequency symmetry failed: "
            f"{checks['phonon_odd_error']:.3e}"
        )
    if checks["min_positive_phonon_weight"] < -tolerance:
        raise RuntimeError(
            "Bath positive-frequency spectral weight became negative"
        )
    return checks


def save_electronic(
    root: Path,
    model: Model,
    result: ScbaResult,
) -> Path:
    path = (
        root
        / "results"
        / f"frenkel_{result.mode}_lambda_{model.lambda_mev:.0f}_"
        f"cutoff_{model.cutoff_mev:.1f}.txt"
    )
    output_omega = np.linspace(
        0.8,
        3.2,
        4000,
    )
    internal_omega = result.omega * EV_PER_AU
    output_spectrum = np.interp(
        output_omega, internal_omega, result.spectrum
    )
    output_sigma = interpolate_complex(
        output_omega / EV_PER_AU,
        result.omega,
        result.sigma,
    )
    np.savetxt(
        path,
        np.column_stack(
            (
                output_omega,
                output_spectrum / EV_PER_AU,
                normalize(output_spectrum),
                output_sigma.real * EV_PER_AU,
                output_sigma.imag * EV_PER_AU,
            )
        ),
        header=(
            "omega_eV  minus_Im_G_raw_per_eV  minus_Im_G_normalized  "
            "Re_Sigma_eV  Im_Sigma_eV; "
            f"bath_method={result.bath_method} "
            "tail=asymptotic_Drude"
        ),
    )
    print(f"Saved {path}")
    return path


def save_phonon(
    root: Path,
    model: Model,
    result: ScbaResult,
) -> Path:
    path = (
        root
        / "results"
        / f"frenkel_phonon_gd_scba_lambda_{model.lambda_mev:.0f}_"
        f"cutoff_{model.cutoff_mev:.1f}.txt"
    )
    d0 = collective_bare_phonon(result.nu, model)
    np.savetxt(
        path,
        np.column_stack(
            (
                result.nu * EV_PER_AU,
                -2.0 * np.imag(d0) * EV_PER_AU,
                -2.0 * np.imag(result.phonon) * EV_PER_AU,
                result.phonon.real * EV_PER_AU,
                result.phonon.imag * EV_PER_AU,
                result.polarization.real / EV_PER_AU,
                result.polarization.imag / EV_PER_AU,
            )
        ),
        header=(
            "nu_eV  B_X0_eV  B_X_gd_scba_eV  Re_D_X_eV  Im_D_X_eV  "
            "Re_Pi_nn_per_eV  Im_Pi_nn_per_eV"
        ),
    )
    print(f"Saved {path}")
    return path


def zero_coupling_check(numerics: Numerics) -> float:
    model = Model(lambda_mev=0.0, cutoff_mev=12.4)
    omega, nu = make_grids(numerics)
    zeros = np.zeros_like(omega, dtype=complex)
    if numerics.bath_method == "convolution":
        sigma = continuous_sigma_map(
            omega, zeros, model, make_convolver(omega, model)
        )
    else:
        coupling, omega_q = tangent_bath(model, numerics)
        occupation = bose(omega_q, model)
        sigma = bare_sigma_map(
            omega,
            zeros,
            model,
            coupling,
            omega_q,
            occupation,
            numerics,
        )
    g0 = electronic_green(
        omega, np.zeros_like(omega, dtype=complex), model
    )
    polarization = vacuum_phonon_polarization(nu, omega, g0, model)
    d0 = collective_bare_phonon(nu, model)
    d = phonon_dyson(nu, polarization, model)
    return float(
        max(
            np.max(np.abs(sigma)),
            np.max(np.abs(polarization)),
            np.max(np.abs(d - d0)),
            np.max(
                np.abs(
                    electronic_green(omega, sigma, model) - g0
                )
            ),
        )
    )


def load_references(
    root: Path,
    model: Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    case = (
        root
        / "heom_reference"
        / f"{model.lambda_mev:.0f}_{model.cutoff_mev:.1f}"
    )
    heom_omega = np.loadtxt(case / "resp1st.w1") * EV_PER_AU
    heom = np.loadtxt(case / "resp1st_im.w")
    cumulant = np.loadtxt(
        root
        / "exact_cumulant"
        / f"cumulant_lambda_{model.lambda_mev:.0f}_"
        f"cutoff_{model.cutoff_mev:.1f}.txt"
    )
    return heom_omega, heom, cumulant[:, 0], cumulant[:, 1]


def write_summary(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "results" / "frenkel_scba_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return path


def write_validation(
    root: Path,
    rows: list[dict[str, object]],
    zero_error: float,
) -> Path:
    path = root / "results" / "frenkel_scba_validation.txt"
    values = {
        "max_abs_pi_per_eV": max(
            float(row["max_abs_Pi_nn_per_eV"]) for row in rows
        ),
        "max_abs_d_change_eV": max(
            float(row["max_abs_D_X_minus_D_X0_eV"]) for row in rows
        ),
        "max_spectral_difference": max(
            float(row["max_normalized_G_only_GD_difference"])
            for row in rows
        ),
        "max_previous_regression": max(
            float(row["previous_G_only_regression_max_abs"])
            for row in rows
        ),
    }
    lines = [
        "Frenkel BA, G-only, and G-D SCBA validation",
        "phonon_state = 300 K equilibrium Drude bath",
        "electronic_state = zero-excitation vacuum",
        "bath_method = continuous zero-padded linear FFT convolution",
        f"zero_coupling_max_error = {zero_error:.16e}",
        f"max_abs_Pi_nn_per_eV = {values['max_abs_pi_per_eV']:.16e}",
        f"max_abs_D_X_minus_D_X0_eV = {values['max_abs_d_change_eV']:.16e}",
        (
            "max_normalized_G_only_GD_difference = "
            f"{values['max_spectral_difference']:.16e}"
        ),
        (
            "previous_G_only_regression_max_abs = "
            f"{values['max_previous_regression']:.16e}"
        ),
        "phonon_update_result = D_X equals D_X0 because Pi_nn is zero",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")
    return path


def main() -> None:
    root = SCRIPT_DIR
    (root / "results").mkdir(exist_ok=True)
    bath_method = os.environ.get("SCBA_BATH_METHOD", "convolution")
    if bath_method == "convolution":
        numerics = Numerics()
    elif bath_method == "tangent":
        numerics = Numerics(
            bath_method="tangent",
            omega_min_eV=0.8,
            omega_max_eV=3.2,
            n_omega=4000,
        )
    else:
        raise ValueError(
            "SCBA_BATH_METHOD must be 'convolution' or 'tangent'"
        )

    zero_error = zero_coupling_check(numerics)
    if zero_error > 1.0e-12:
        raise RuntimeError(
            f"Zero-coupling regression failed: {zero_error:.3e}"
        )

    rows: list[dict[str, object]] = []
    for cutoff in CUTOFFS_MEV:
        for lam in LAMBDAS_MEV:
            model = Model(lambda_mev=lam, cutoff_mev=cutoff)
            print(
                f"\nlambda={lam:.0f} meV, cutoff={cutoff:.1f} meV"
            )
            ba = solve_single_shot_ba(model, numerics)
            g_only = solve_g_only(model, numerics)
            previous_error = existing_g_only_regression(
                root, model, g_only
            )
            gd = solve_gd_scba(model, numerics, g_only)

            save_electronic(root, model, ba)
            save_electronic(root, model, g_only)
            save_electronic(root, model, gd)
            save_phonon(root, model, gd)

            checks = validate(model, ba, g_only, gd)
            heom_w, heom, cumulant_w, cumulant = load_references(
                root, model
            )
            bath_peak, bath_fwhm = phonon_peak_and_fwhm(
                gd.nu, gd.phonon
            )
            rows.append(
                {
                    "lambda_meV": f"{lam:.1f}",
                    "cutoff_meV": f"{cutoff:.1f}",
                    "phonon_state": "300K_equilibrium_Drude",
                    "electronic_state": "zero_exciton_vacuum",
                    "single_shot_ba_converged": ba.converged,
                    "g_only_converged": g_only.converged,
                    "gd_scba_converged": gd.converged,
                    "single_shot_ba_iterations": ba.iterations,
                    "g_only_iterations": g_only.iterations,
                    "gd_scba_iterations": gd.iterations,
                    "g_only_final_electronic_error": (
                        f"{g_only.error_g:.16e}"
                    ),
                    "gd_final_electronic_error": f"{gd.error_g:.16e}",
                    "gd_final_phonon_error": f"{gd.error_d:.16e}",
                    "single_shot_ba_HEOM_normalized_L1_error": (
                        f"{normalized_l1_to_reference(ba.omega * EV_PER_AU, ba.spectrum, heom_w, heom):.16e}"
                    ),
                    "g_only_HEOM_normalized_L1_error": (
                        f"{normalized_l1_to_reference(g_only.omega * EV_PER_AU, g_only.spectrum, heom_w, heom):.16e}"
                    ),
                    "gd_HEOM_normalized_L1_error": (
                        f"{normalized_l1_to_reference(gd.omega * EV_PER_AU, gd.spectrum, heom_w, heom):.16e}"
                    ),
                    "single_shot_ba_cumulant_normalized_L1_error": (
                        f"{normalized_l1_to_reference(ba.omega * EV_PER_AU, ba.spectrum, cumulant_w, cumulant):.16e}"
                    ),
                    "g_only_cumulant_normalized_L1_error": (
                        f"{normalized_l1_to_reference(g_only.omega * EV_PER_AU, g_only.spectrum, cumulant_w, cumulant):.16e}"
                    ),
                    "gd_cumulant_normalized_L1_error": (
                        f"{normalized_l1_to_reference(gd.omega * EV_PER_AU, gd.spectrum, cumulant_w, cumulant):.16e}"
                    ),
                    "max_normalized_G_only_GD_difference": (
                        f"{np.max(np.abs(normalize(g_only.spectrum) - normalize(gd.spectrum))):.16e}"
                    ),
                    "max_abs_Pi_nn_per_eV": (
                        f"{checks['max_abs_pi'] / EV_PER_AU:.16e}"
                    ),
                    "max_abs_D_X_minus_D_X0_eV": (
                        f"{checks['max_abs_d_minus_d0'] * EV_PER_AU:.16e}"
                    ),
                    "single_shot_ba_dominant_peak_positions_eV": (
                        peak_positions(ba.omega, ba.spectrum)
                    ),
                    "g_only_dominant_peak_positions_eV": peak_positions(
                        g_only.omega, g_only.spectrum
                    ),
                    "gd_dominant_peak_positions_eV": peak_positions(
                        gd.omega, gd.spectrum
                    ),
                    "bath_spectral_peak_eV": f"{bath_peak:.8f}",
                    "bath_spectral_FWHM_eV": f"{bath_fwhm:.8f}",
                    "phonon_meaningfully_changed": (
                        checks["max_abs_d_minus_d0"] * EV_PER_AU
                        > 1.0e-10
                    ),
                    "previous_G_only_regression_max_abs": (
                        f"{previous_error:.16e}"
                    ),
                    "min_minus_Im_G_au": (
                        f"{checks['min_minus_im_g']:.16e}"
                    ),
                    "max_positive_Im_Sigma_au": (
                        f"{checks['max_positive_im_sigma']:.16e}"
                    ),
                    "phonon_odd_symmetry_error_au": (
                        f"{checks['phonon_odd_error']:.16e}"
                    ),
                    "min_positive_phonon_weight_au": (
                        f"{checks['min_positive_phonon_weight']:.16e}"
                    ),
                }
            )

    write_summary(root, rows)
    write_validation(root, rows, zero_error)


if __name__ == "__main__":
    main()
