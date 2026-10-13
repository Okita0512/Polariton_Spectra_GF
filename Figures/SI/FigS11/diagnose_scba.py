#!/usr/bin/env python3
"""Independent SCBA diagnostics near the bright--dark degeneracy.

This script imports the shared frequency-coupled SCBA kernel, but constructs
its own cases, nonlinear scans, self-energy decomposition, and output tables.
The only linewidth extractor used here is the validated common direct-FWHM
routine.  No detuning plotting or linewidth-analysis driver is imported.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import find_peaks, savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """Locate the repository from stable project markers after figure moves."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "main.tex").is_file()
            and (candidate / "SI.tex").is_file()
            and (candidate / "Figures" / "common" / "scba_g_only.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate the project root above {start}; expected "
        "main.tex, SI.tex, and Figures/common/scba_g_only.py"
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
COMMON_DIR = PROJECT_ROOT / "Figures" / "common"
sys.path.insert(0, str(COMMON_DIR))

from scba_g_only import (  # noqa: E402
    Case,
    EV_PER_AU,
    bose,
    interpolate_complex,
    local_green,
    sigma_map as shared_sigma_map,
    tangent_bath,
)
from spectral_analysis import (  # noqa: E402
    BranchMetrics,
    analyze_single_peak,
    analyze_two_peaks,
)


KAPPA_EEV = 0.04415
VERTICAL_ENERGY_EV = 2.03
REORGANIZATION_EV = 0.03
BATH_CUTOFF_EV = 0.0248
COLLECTIVE_G_EV = 0.1362
ETA_EV = 1.0e-5
TEMPERATURE_K = 300.0
DEFAULT_N = (1, 2, 4, 8, 16, 32, 64, 10000)
DECOUPLED_N = (1, 2, 4, 8, 16, 32, 64)
GAPS_MEV = (100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
DATA_DIR = SCRIPT_DIR / "data"
CACHE_DIR = SCRIPT_DIR / "cache"


@dataclass(frozen=True)
class Settings:
    n_bath: int = 1000
    n_omega: int = 4000
    omega_min_eV: float = 0.8
    omega_max_eV: float = 3.2
    max_iterations: int = 300
    tolerance: float = 1.0e-8
    bath_chunk: int = 50
    initial_mixing: float = 1.0
    adaptive_mixing: bool = True


@dataclass
class DiagnosticResult:
    grid: np.ndarray
    sigma: np.ndarray
    sigma_bright: np.ndarray
    sigma_dark: np.ndarray
    output_sigma: np.ndarray
    output_green_bright: np.ndarray
    history: np.ndarray
    iterations: int
    final_error: float
    final_mixing: float
    method: str
    seed_name: str


def base_case(
    n_molecules: int,
    omega_c_eV: float,
    collective_g_eV: float = COLLECTIVE_G_EV,
    temperature_K: float = TEMPERATURE_K,
    eta_eV: float = ETA_EV,
) -> Case:
    return Case(
        script=Path(__file__).resolve(),
        vertical_energy=VERTICAL_ENERGY_EV / EV_PER_AU,
        omega_c=omega_c_eV / EV_PER_AU,
        cavity_halfwidth=0.5 * KAPPA_EEV / EV_PER_AU,
        reorganization=REORGANIZATION_EV / EV_PER_AU,
        bath_cutoff=BATH_CUTOFF_EV / EV_PER_AU,
        collective_g=collective_g_eV / EV_PER_AU,
        eta=eta_eV / EV_PER_AU,
        n_molecules=n_molecules,
        output_grid=np.linspace(
            0.9 / EV_PER_AU, 3.15 / EV_PER_AU, 4000
        ),
        legacy_n_bath=1000,
        temperature_K=temperature_K,
    )


def gap_to_cavity_eV(gap_meV: float) -> float:
    gap_eV = gap_meV * 1.0e-3
    detuning_eV = gap_eV - COLLECTIVE_G_EV**2 / gap_eV
    return VERTICAL_ENERGY_EV + detuning_eV


def up_exciton_fraction(gap_meV: float) -> float:
    gap_eV = gap_meV * 1.0e-3
    detuning_eV = gap_eV - COLLECTIVE_G_EV**2 / gap_eV
    splitting = np.sqrt(detuning_eV**2 + 4.0 * COLLECTIVE_G_EV**2)
    return float(0.5 * (1.0 - detuning_eV / splitting))


def thermal_occupation(omega_q: np.ndarray, temperature_K: float) -> np.ndarray:
    if temperature_K <= 0.0:
        return np.zeros_like(omega_q)
    return bose(omega_q, temperature_K)


def sigma_components(
    grid: np.ndarray,
    sigma: np.ndarray,
    case: Case,
    coupling: np.ndarray,
    omega_q: np.ndarray,
    occupation: np.ndarray,
    bath_chunk: int,
    frozen_dark: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bright/N and (N-1)dark/N Born-kernel contributions."""
    bright_result = np.zeros_like(sigma)
    dark_result = np.zeros_like(sigma)
    row = grid[None, :]
    bright_weight = 1.0 / case.n_molecules
    dark_weight = (case.n_molecules - 1.0) / case.n_molecules

    for start in range(0, len(omega_q), bath_chunk):
        stop = min(start + bath_chunk, len(omega_q))
        q = omega_q[start:stop, None]
        c2 = coupling[start:stop, None] ** 2
        nq = occupation[start:stop, None]
        emit = row - q
        absorb = row + q
        sigma_emit = interpolate_complex(emit, grid, sigma)
        sigma_absorb = interpolate_complex(absorb, grid, sigma)
        _, bright_emit, dark_emit = local_green(
            emit, sigma_emit, case, "full"
        )
        _, bright_absorb, dark_absorb = local_green(
            absorb, sigma_absorb, case, "full"
        )
        if frozen_dark is not None:
            dark_emit = interpolate_complex(emit, grid, frozen_dark)
            dark_absorb = interpolate_complex(absorb, grid, frozen_dark)
        bright_result += bright_weight * np.sum(
            c2 * ((1.0 + nq) * bright_emit + nq * bright_absorb),
            axis=0,
        )
        dark_result += dark_weight * np.sum(
            c2 * ((1.0 + nq) * dark_emit + nq * dark_absorb),
            axis=0,
        )
    return bright_result, dark_result


def solve_diagnostic(
    case: Case,
    settings: Settings,
    method: str = "full",
    initial_sigma: np.ndarray | None = None,
    seed_name: str = "BA",
    frozen_dark: np.ndarray | None = None,
) -> DiagnosticResult:
    """Solve full SCBA, one-shot BA, or the frozen-molecular-dark closure."""
    grid = np.linspace(
        settings.omega_min_eV / EV_PER_AU,
        settings.omega_max_eV / EV_PER_AU,
        settings.n_omega,
    )
    coupling, omega_q = tangent_bath(
        case.reorganization, case.bath_cutoff, settings.n_bath
    )
    occupation = thermal_occupation(omega_q, case.temperature_K)
    zeros = np.zeros_like(grid, dtype=complex)
    ba_sigma = shared_sigma_map(
        grid,
        zeros,
        case,
        coupling,
        omega_q,
        occupation,
        settings.bath_chunk,
        "full",
    )

    if method == "ba":
        sigma = ba_sigma
        mapped = shared_sigma_map(
            grid,
            sigma,
            case,
            coupling,
            omega_q,
            occupation,
            settings.bath_chunk,
            "full",
        )
        final_error = float(np.max(np.abs(mapped - sigma)))
        history = np.asarray(
            [[0.0, final_error, settings.initial_mixing]], dtype=float
        )
        iterations = 1
        mixing = settings.initial_mixing
    else:
        if initial_sigma is None:
            sigma = ba_sigma.copy()
        else:
            if initial_sigma.shape != grid.shape:
                raise ValueError("Initial self-energy grid does not match")
            sigma = np.asarray(initial_sigma, dtype=complex).copy()
        if method == "frozen_dark" and frozen_dark is None:
            raise ValueError("frozen_dark method requires a fixed dark line")
        history_rows: list[tuple[float, float, float]] = []
        previous_error = np.inf
        mixing = settings.initial_mixing
        final_error = np.inf
        for iteration in range(1, settings.max_iterations + 1):
            if method == "full":
                mapped = shared_sigma_map(
                    grid,
                    sigma,
                    case,
                    coupling,
                    omega_q,
                    occupation,
                    settings.bath_chunk,
                    "full",
                )
            elif method == "frozen_dark":
                mapped_b, mapped_d = sigma_components(
                    grid,
                    sigma,
                    case,
                    coupling,
                    omega_q,
                    occupation,
                    settings.bath_chunk,
                    frozen_dark=frozen_dark,
                )
                mapped = mapped_b + mapped_d
            else:
                raise ValueError(f"Unknown method: {method}")
            residual = mapped - sigma
            final_error = float(np.max(np.abs(residual)))
            if (
                settings.adaptive_mixing
                and final_error > 1.25 * previous_error
            ):
                mixing = max(0.1, 0.5 * mixing)
            sigma = sigma + mixing * residual
            history_rows.append((float(iteration), final_error, mixing))
            if final_error < settings.tolerance:
                break
            previous_error = final_error
        else:
            raise RuntimeError(
                f"{method} did not converge for N={case.n_molecules}, "
                f"wc={case.omega_c * EV_PER_AU:.9g} eV; "
                f"residual={final_error:.6e} E_h"
            )
        history = np.asarray(history_rows, dtype=float)
        iterations = iteration

    if float(np.max(sigma.imag)) > 10.0 * settings.tolerance:
        raise RuntimeError("Retarded-causality check failed")

    # For one-shot BA the internal lines are noninteracting; decompose the
    # map evaluated at Sigma=0, not the next Born iteration evaluated with
    # the already generated BA self-energy.
    component_sigma = zeros if method == "ba" else sigma
    sigma_bright, sigma_dark = sigma_components(
        grid,
        component_sigma,
        case,
        coupling,
        omega_q,
        occupation,
        settings.bath_chunk,
        frozen_dark=frozen_dark if method == "frozen_dark" else None,
    )
    output_sigma = interpolate_complex(case.output_grid, grid, sigma)
    _, output_bright, _ = local_green(
        case.output_grid, output_sigma, case, "full"
    )
    return DiagnosticResult(
        grid=grid,
        sigma=sigma,
        sigma_bright=sigma_bright,
        sigma_dark=sigma_dark,
        output_sigma=output_sigma,
        output_green_bright=output_bright,
        history=history,
        iterations=iterations,
        final_error=final_error,
        final_mixing=mixing,
        method=method,
        seed_name=seed_name,
    )


def cache_file(tag: str) -> Path:
    return CACHE_DIR / f"{tag}.npz"


def save_cached(tag: str, result: DiagnosticResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file(tag),
        grid=result.grid,
        sigma=result.sigma,
        sigma_bright=result.sigma_bright,
        sigma_dark=result.sigma_dark,
        output_sigma=result.output_sigma,
        output_green_bright=result.output_green_bright,
        history=result.history,
        iterations=np.asarray(result.iterations),
        final_error=np.asarray(result.final_error),
        final_mixing=np.asarray(result.final_mixing),
        method=np.asarray(result.method),
        seed_name=np.asarray(result.seed_name),
    )


def load_cached(tag: str) -> DiagnosticResult:
    data = np.load(cache_file(tag), allow_pickle=False)
    return DiagnosticResult(
        grid=data["grid"],
        sigma=data["sigma"],
        sigma_bright=data["sigma_bright"],
        sigma_dark=data["sigma_dark"],
        output_sigma=data["output_sigma"],
        output_green_bright=data["output_green_bright"],
        history=data["history"],
        iterations=int(data["iterations"]),
        final_error=float(data["final_error"]),
        final_mixing=float(data["final_mixing"]),
        method=str(data["method"]),
        seed_name=str(data["seed_name"]),
    )


def run_cached(
    tag: str,
    case: Case,
    settings: Settings,
    method: str = "full",
    initial_sigma: np.ndarray | None = None,
    seed_name: str = "BA",
    frozen_dark: np.ndarray | None = None,
    force: bool = False,
) -> DiagnosticResult:
    if cache_file(tag).exists() and not force:
        return load_cached(tag)
    result = solve_diagnostic(
        case,
        settings,
        method=method,
        initial_sigma=initial_sigma,
        seed_name=seed_name,
        frozen_dark=frozen_dark,
    )
    save_cached(tag, result)
    return result


def count_peaks(
    energy_eV: np.ndarray,
    spectrum: np.ndarray,
    x_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(energy_eV)
        & np.isfinite(spectrum)
        & (energy_eV >= x_range[0])
        & (energy_eV <= x_range[1])
    )
    x = energy_eV[mask]
    y = spectrum[mask]
    y = y / np.max(y)
    smooth = savgol_filter(y, 11, 3)
    peaks, props = find_peaks(smooth, prominence=0.01, distance=10)
    return x[peaks], props["prominences"]


def measure_up(case: Case, result: DiagnosticResult) -> dict[str, float | int]:
    energy = case.output_grid * EV_PER_AU
    spectrum = -np.imag(result.output_green_bright)
    peak_positions, prominences = count_peaks(
        energy, spectrum, (0.9, 2.5)
    )
    resolved_peaks = len(peak_positions)

    if resolved_peaks >= 2:
        metrics_two = analyze_two_peaks(
            energy, spectrum, x_range=(0.9, 2.5)
        )
        metrics = metrics_two.upper
        up_positions, _ = count_peaks(
            energy, spectrum, (metrics_two.valley, 2.5)
        )
        up_window_peaks = len(up_positions)
    else:
        metrics = analyze_single_peak(
            energy, spectrum, x_range=(1.65, 2.35)
        )
        up_positions, _ = count_peaks(
            energy, spectrum, (1.65, 2.35)
        )
        up_window_peaks = len(up_positions)

    center_au = metrics.center / EV_PER_AU
    sigma_at_peak = interpolate_complex(
        np.asarray([center_au]), result.grid, result.sigma
    )[0]
    bright_at_peak = interpolate_complex(
        np.asarray([center_au]), result.grid, result.sigma_bright
    )[0]
    dark_at_peak = interpolate_complex(
        np.asarray([center_au]), result.grid, result.sigma_dark
    )[0]
    return {
        "up_peak_eV": metrics.center,
        "fwhm_meV": metrics.fwhm * 1.0e3,
        "left_halfwidth_meV": metrics.left_half_width * 1.0e3,
        "right_halfwidth_meV": metrics.right_half_width * 1.0e3,
        "resolved_peaks": resolved_peaks,
        "up_window_peaks": up_window_peaks,
        "prominence": metrics.prominence,
        "reliable": int(metrics.reliable),
        "minus2ImSigma_meV": -2.0 * sigma_at_peak.imag * EV_PER_AU * 1.0e3,
        "ReSigma_meV": sigma_at_peak.real * EV_PER_AU * 1.0e3,
        "minus2ImSigmaB_meV": (
            -2.0 * bright_at_peak.imag * EV_PER_AU * 1.0e3
        ),
        "minus2ImSigmaD_meV": (
            -2.0 * dark_at_peak.imag * EV_PER_AU * 1.0e3
        ),
    }


def result_row(
    case: Case,
    result: DiagnosticResult,
    n_molecules: int,
    gap_meV: float | None,
    label: str,
    bare_fwhm_meV: float,
) -> dict[str, object]:
    measured = measure_up(case, result)
    row: dict[str, object] = {
        "label": label,
        "method": result.method,
        "seed": result.seed_name,
        "N": n_molecules,
        "temperature_K": case.temperature_K,
        "gap_meV": np.nan if gap_meV is None else gap_meV,
        "omega_c_eV": case.omega_c * EV_PER_AU,
        "exciton_fraction": (
            1.0 if gap_meV is None else up_exciton_fraction(gap_meV)
        ),
        **measured,
        "excess_meV": float(measured["fwhm_meV"]) - bare_fwhm_meV,
        "iterations": result.iterations,
        "residual_meV": result.final_error * EV_PER_AU * 1.0e3,
        "mixing": result.final_mixing,
    }
    return row


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def default_cache_tag(
    prefix: str,
    n_molecules: int,
    gap_meV: float | None,
    temperature_K: float,
    method: str,
    settings: Settings,
    eta_eV: float = ETA_EV,
    seed: str = "BA",
) -> str:
    gap_text = "decoupled" if gap_meV is None else f"gap{gap_meV:g}"
    return (
        f"{prefix}_N{n_molecules}_{gap_text}_T{temperature_K:g}_"
        f"{method}_{seed}_b{settings.n_bath}_w{settings.n_omega}_"
        f"lo{settings.omega_min_eV:g}_hi{settings.omega_max_eV:g}_"
        f"tol{settings.tolerance:.0e}_eta{eta_eV:.0e}_"
        f"mix{settings.initial_mixing:g}_ad{int(settings.adaptive_mixing)}"
    )


def compute_decoupled(
    settings: Settings, force: bool
) -> tuple[DiagnosticResult, float, list[dict[str, object]]]:
    results: list[tuple[int, Case, DiagnosticResult]] = []
    for n_molecules in DECOUPLED_N:
        case = base_case(
            n_molecules=n_molecules,
            omega_c_eV=VERTICAL_ENERGY_EV,
            collective_g_eV=0.0,
        )
        tag = default_cache_tag(
            "g0", n_molecules, None, TEMPERATURE_K, "full", settings
        )
        result = run_cached(tag, case, settings, force=force)
        results.append((n_molecules, case, result))
        print(
            f"g=0 N={n_molecules}: {result.iterations} iterations, "
            f"residual={result.final_error * EV_PER_AU * 1e3:.3e} meV",
            flush=True,
        )

    molecule_result = results[0][2]
    molecule_case = results[0][1]
    molecule_metrics = measure_up(molecule_case, molecule_result)
    bare_fwhm = float(molecule_metrics["fwhm_meV"])
    rows = [
        result_row(
            case,
            result,
            n_molecules,
            None,
            "strict_g0",
            bare_fwhm,
        )
        for n_molecules, case, result in results
    ]
    reference_spectrum = -np.imag(results[0][2].output_green_bright)
    reference_sigma = results[0][2].sigma
    for row, (_, _, result) in zip(rows, results):
        row["max_spectrum_difference"] = float(
            np.max(
                np.abs(
                    -np.imag(result.output_green_bright)
                    - reference_spectrum
                )
            )
        )
        row["max_sigma_difference_meV"] = float(
            np.max(np.abs(result.sigma - reference_sigma))
            * EV_PER_AU
            * 1.0e3
        )
    write_csv(DATA_DIR / "decoupled_limit.csv", rows)
    write_csv(DATA_DIR / "isolated_molecule.csv", [rows[0]])
    return molecule_result, bare_fwhm, rows


def _independent_gap_worker(
    payload: tuple[int, float, float, Settings, float, bool]
) -> dict[str, object]:
    n_molecules, gap_meV, temperature_K, settings, bare_fwhm, force = payload
    case = base_case(
        n_molecules=n_molecules,
        omega_c_eV=gap_to_cavity_eV(gap_meV),
        temperature_K=temperature_K,
    )
    tag = default_cache_tag(
        "gap", n_molecules, gap_meV, temperature_K, "full", settings
    )
    result = run_cached(tag, case, settings, force=force)
    return result_row(
        case, result, n_molecules, gap_meV, "independent_BA_seed", bare_fwhm
    )


def compute_gap_scan(
    settings: Settings,
    bare_fwhm: float,
    workers: int,
    force: bool,
) -> list[dict[str, object]]:
    payloads = [
        (n, gap, TEMPERATURE_K, settings, bare_fwhm, force)
        for n in DEFAULT_N
        for gap in GAPS_MEV
    ]
    rows: list[dict[str, object]] = []
    if workers == 1:
        for payload in payloads:
            row = _independent_gap_worker(payload)
            rows.append(row)
            print(
                f"gap scan N={row['N']} delta={row['gap_meV']} meV: "
                f"FWHM={row['fwhm_meV']:.4f} meV",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_independent_gap_worker, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"gap scan N={row['N']} delta={row['gap_meV']} meV: "
                    f"FWHM={row['fwhm_meV']:.4f} meV",
                    flush=True,
                )
    rows.sort(key=lambda row: (int(row["N"]), -float(row["gap_meV"])))
    write_csv(DATA_DIR / "gap_scan.csv", rows)
    return rows


def gap_result(
    n_molecules: int,
    gap_meV: float,
    settings: Settings,
    force: bool = False,
) -> tuple[Case, DiagnosticResult]:
    case = base_case(
        n_molecules=n_molecules,
        omega_c_eV=gap_to_cavity_eV(gap_meV),
    )
    tag = default_cache_tag(
        "gap", n_molecules, gap_meV, TEMPERATURE_K, "full", settings
    )
    return case, run_cached(tag, case, settings, force=force)


def compute_fixed_points(
    settings: Settings,
    molecule_result: DiagnosticResult,
    bare_fwhm: float,
    force: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_molecules in (4, 16, 64):
        for target_gap, larger_gap, smaller_gap in (
            (0.5, 1.0, 0.2),
            (0.2, 0.5, 0.1),
            (0.02, 0.05, 0.01),
        ):
            case, ba_result = gap_result(
                n_molecules, target_gap, settings, force=force
            )
            _, larger_result = gap_result(
                n_molecules, larger_gap, settings, force=force
            )
            _, smaller_result = gap_result(
                n_molecules, smaller_gap, settings, force=force
            )
            seeds = (
                ("BA", None),
                ("molecule", molecule_result.sigma),
                ("from_larger_gap", larger_result.sigma),
                ("from_smaller_gap", smaller_result.sigma),
            )
            for seed_name, initial_sigma in seeds:
                if seed_name == "BA":
                    result = ba_result
                else:
                    tag = default_cache_tag(
                        "fixedpoint",
                        n_molecules,
                        target_gap,
                        TEMPERATURE_K,
                        "full",
                        settings,
                        seed=seed_name,
                    )
                    result = run_cached(
                        tag,
                        case,
                        settings,
                        initial_sigma=initial_sigma,
                        seed_name=seed_name,
                        force=force,
                    )
                rows.append(
                    result_row(
                        case,
                        result,
                        n_molecules,
                        target_gap,
                        "fixed_point_test",
                        bare_fwhm,
                    )
                )
    write_csv(DATA_DIR / "fixed_point_scan.csv", rows)
    return rows


def compute_methods(
    settings: Settings,
    molecule_result: DiagnosticResult,
    bare_fwhm: float,
    force: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bare_exciton_denominator = (
        molecule_result.grid
        - VERTICAL_ENERGY_EV / EV_PER_AU
        + 1.0j * ETA_EV / EV_PER_AU
        - molecule_result.sigma
    )
    frozen_dark = 1.0 / bare_exciton_denominator
    for n_molecules in (4, 16, 64):
        for gap_meV in (20.0, 5.0, 1.0, 0.2):
            case, full_result = gap_result(
                n_molecules, gap_meV, settings, force=force
            )
            rows.append(
                result_row(
                    case,
                    full_result,
                    n_molecules,
                    gap_meV,
                    "method_comparison",
                    bare_fwhm,
                )
            )
            for method in ("ba", "frozen_dark"):
                tag = default_cache_tag(
                    "method",
                    n_molecules,
                    gap_meV,
                    TEMPERATURE_K,
                    method,
                    settings,
                )
                result = run_cached(
                    tag,
                    case,
                    settings,
                    method=method,
                    frozen_dark=frozen_dark if method == "frozen_dark" else None,
                    force=force,
                )
                rows.append(
                    result_row(
                        case,
                        result,
                        n_molecules,
                        gap_meV,
                        "method_comparison",
                        bare_fwhm,
                    )
                )
    write_csv(DATA_DIR / "method_comparison.csv", rows)
    return rows


def compute_temperature(
    settings: Settings,
    bare_fwhm_300K: float,
    workers: int,
    force: bool,
) -> list[dict[str, object]]:
    temperatures = (0.0, 100.0, 200.0, 300.0)
    gaps = (20.0, 5.0, 1.0, 0.2)
    molecule_widths: dict[float, float] = {}
    for temperature in temperatures:
        case = base_case(
            n_molecules=1,
            omega_c_eV=VERTICAL_ENERGY_EV,
            collective_g_eV=0.0,
            temperature_K=temperature,
        )
        tag = default_cache_tag(
            "temperature_molecule",
            1,
            None,
            temperature,
            "full",
            settings,
        )
        result = run_cached(tag, case, settings, force=force)
        molecule_widths[temperature] = float(measure_up(case, result)["fwhm_meV"])
    payloads = [
        (
            n,
            gap,
            temperature,
            settings,
            molecule_widths[temperature],
            force,
        )
        for temperature in temperatures
        for n in (4, 16, 64)
        for gap in gaps
    ]
    rows: list[dict[str, object]] = []
    if workers == 1:
        rows = [_independent_gap_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_independent_gap_worker, payloads))
    for row in rows:
        row["label"] = "temperature_scan"
        row["molecule_fwhm_at_temperature_meV"] = molecule_widths[
            float(row["temperature_K"])
        ]
    rows.sort(
        key=lambda row: (
            float(row["temperature_K"]),
            int(row["N"]),
            -float(row["gap_meV"]),
        )
    )
    write_csv(DATA_DIR / "temperature_scan.csv", rows)
    temperature_rows = [
        {
            "temperature_K": temperature,
            "molecule_fwhm_meV": width,
            "difference_from_300K_meV": width - bare_fwhm_300K,
        }
        for temperature, width in molecule_widths.items()
    ]
    write_csv(DATA_DIR / "temperature_molecule.csv", temperature_rows)
    return rows


def compute_robustness(
    settings: Settings,
    bare_fwhm: float,
    force: bool,
) -> list[dict[str, object]]:
    variants = (
        ("production", settings, ETA_EV),
        ("double_bath", replace(settings, n_bath=2000), ETA_EV),
        ("double_frequency", replace(settings, n_omega=8000), ETA_EV),
        (
            "wider_padding",
            replace(
                settings,
                n_omega=5334,
                omega_min_eV=0.4,
                omega_max_eV=3.6,
            ),
            ETA_EV,
        ),
        ("half_eta", settings, 0.5 * ETA_EV),
        ("tight_tolerance", replace(settings, tolerance=1.0e-9), ETA_EV),
        (
            "fixed_mixing_0.3",
            replace(
                settings,
                initial_mixing=0.3,
                adaptive_mixing=False,
            ),
            ETA_EV,
        ),
        (
            "fixed_mixing_0.6",
            replace(
                settings,
                initial_mixing=0.6,
                adaptive_mixing=False,
            ),
            ETA_EV,
        ),
        (
            "fixed_mixing_1.0",
            replace(
                settings,
                initial_mixing=1.0,
                adaptive_mixing=False,
            ),
            ETA_EV,
        ),
    )
    rows: list[dict[str, object]] = []
    for n_molecules in (4, 64):
        gap_meV = 20.0
        for variant, variant_settings, eta_eV in variants:
            case = base_case(
                n_molecules=n_molecules,
                omega_c_eV=gap_to_cavity_eV(gap_meV),
                eta_eV=eta_eV,
            )
            tag = default_cache_tag(
                f"robust_{variant}",
                n_molecules,
                gap_meV,
                TEMPERATURE_K,
                "full",
                variant_settings,
                eta_eV=eta_eV,
            )
            result = run_cached(
                tag, case, variant_settings, force=force
            )
            row = result_row(
                case,
                result,
                n_molecules,
                gap_meV,
                variant,
                bare_fwhm,
            )
            row["n_bath"] = variant_settings.n_bath
            row["n_omega"] = variant_settings.n_omega
            row["omega_min_eV"] = variant_settings.omega_min_eV
            row["omega_max_eV"] = variant_settings.omega_max_eV
            row["eta_meV"] = eta_eV * 1.0e3
            row["tolerance_hartree"] = variant_settings.tolerance
            row["tolerance_eV"] = (
                variant_settings.tolerance * EV_PER_AU
            )
            rows.append(row)
            print(
                f"robustness N={n_molecules} {variant}: "
                f"FWHM={row['fwhm_meV']:.4f} meV",
                flush=True,
            )
    write_csv(DATA_DIR / "robustness.csv", rows)
    return rows


def write_summaries(
    decoupled_rows: Sequence[dict[str, object]],
    gap_rows: Sequence[dict[str, object]],
    fixed_rows: Sequence[dict[str, object]],
    method_rows: Sequence[dict[str, object]],
    temperature_rows: Sequence[dict[str, object]],
    robustness_rows: Sequence[dict[str, object]],
) -> None:
    smallest_gap = min(float(row["gap_meV"]) for row in gap_rows)
    asymptotic = [
        row for row in gap_rows if float(row["gap_meV"]) == smallest_gap
    ]
    write_csv(DATA_DIR / "asymptotic_summary.csv", asymptotic)

    # The manuscript denotes these product-moment correlations by r_P so
    # they cannot be confused with the dimer scan coordinate r=2|J|/nu_0.
    correlation_summary: list[dict[str, object]] = []
    for n_molecules in (2, 4, 8, 16, 32, 64):
        subset = sorted(
            (
                row
                for row in gap_rows
                if int(row["N"]) == n_molecules
            ),
            key=lambda row: float(row["gap_meV"]),
        )
        decoupled = next(
            row
            for row in decoupled_rows
            if int(row["N"]) == n_molecules
        )
        dark_baseline = float(decoupled["minus2ImSigmaD_meV"])
        linewidth_excess = np.asarray(
            [float(row["excess_meV"]) for row in subset]
        )
        dark_damping_excess = np.asarray(
            [
                float(row["minus2ImSigmaD_meV"]) - dark_baseline
                for row in subset
            ]
        )
        pearson_r_p = float(
            np.corrcoef(linewidth_excess, dark_damping_excess)[0, 1]
        )
        correlation_summary.append(
            {
                "statistic": "Pearson product-moment correlation",
                "notation": "r_P",
                "N": n_molecules,
                "points": len(subset),
                "gap_min_meV": min(float(row["gap_meV"]) for row in subset),
                "gap_max_meV": max(float(row["gap_meV"]) for row in subset),
                "dark_baseline_meV": dark_baseline,
                "pearson_r_P": pearson_r_p,
            }
        )
    write_csv(DATA_DIR / "correlation_summary.csv", correlation_summary)

    fixed_groups: list[dict[str, object]] = []
    for n_molecules in (4, 16, 64):
        for gap_meV in (0.5, 0.2, 0.02):
            subset = [
                row
                for row in fixed_rows
                if int(row["N"]) == n_molecules
                and float(row["gap_meV"]) == gap_meV
            ]
            widths = np.asarray([float(row["fwhm_meV"]) for row in subset])
            minus2im = np.asarray(
                [float(row["minus2ImSigma_meV"]) for row in subset]
            )
            fixed_groups.append(
                {
                    "N": n_molecules,
                    "gap_meV": gap_meV,
                    "max_width_seed_difference_meV": float(np.ptp(widths)),
                    "max_minus2ImSigma_seed_difference_meV": float(
                        np.ptp(minus2im)
                    ),
                }
            )
    write_csv(DATA_DIR / "fixed_point_summary.csv", fixed_groups)

    production_robust = [
        row for row in robustness_rows if row["label"] == "production"
    ]
    convergence_summary: list[dict[str, object]] = []
    for n_molecules in (4, 64):
        reference = next(
            row
            for row in production_robust
            if int(row["N"]) == n_molecules
        )
        subset = [
            row for row in robustness_rows if int(row["N"]) == n_molecules
        ]
        convergence_summary.append(
            {
                "N": n_molecules,
                "reference_fwhm_meV": reference["fwhm_meV"],
                "max_fwhm_variation_meV": max(
                    abs(
                        float(row["fwhm_meV"])
                        - float(reference["fwhm_meV"])
                    )
                    for row in subset
                ),
                "max_peak_variation_meV": 1.0e3
                * max(
                    abs(
                        float(row["up_peak_eV"])
                        - float(reference["up_peak_eV"])
                    )
                    for row in subset
                ),
                "all_reliable": int(
                    all(int(row["reliable"]) for row in subset)
                ),
                "all_single_up_peak": int(
                    all(int(row["up_window_peaks"]) == 1 for row in subset)
                ),
            }
        )
    write_csv(DATA_DIR / "robustness_summary.csv", convergence_summary)

    overall = [
        {
            "isolated_molecule_fwhm_meV": decoupled_rows[0]["fwhm_meV"],
            "max_g0_width_difference_meV": max(
                abs(
                    float(row["fwhm_meV"])
                    - float(decoupled_rows[0]["fwhm_meV"])
                )
                for row in decoupled_rows
            ),
            "max_g0_sigma_difference_meV": max(
                float(row["max_sigma_difference_meV"])
                for row in decoupled_rows
            ),
            "smallest_gap_meV": smallest_gap,
            "max_asymptotic_excess_meV": max(
                abs(float(row["excess_meV"])) for row in asymptotic
            ),
            "min_pearson_r_P": min(
                float(row["pearson_r_P"])
                for row in correlation_summary
            ),
            "max_pearson_r_P": max(
                float(row["pearson_r_P"])
                for row in correlation_summary
            ),
            "max_fixed_point_width_difference_meV": max(
                float(row["max_width_seed_difference_meV"])
                for row in fixed_groups
            ),
            "max_robustness_width_variation_meV": max(
                float(row["max_fwhm_variation_meV"])
                for row in convergence_summary
            ),
            "all_gap_points_reliable": int(
                all(int(row["reliable"]) for row in gap_rows)
            ),
            "all_gap_up_windows_single_peaked": int(
                all(int(row["up_window_peaks"]) == 1 for row in gap_rows)
            ),
            "method_rows": len(method_rows),
            "temperature_rows": len(temperature_rows),
        }
    ]
    write_csv(DATA_DIR / "overall_summary.csv", overall)


def validate_outputs(
    decoupled_rows: Sequence[dict[str, object]],
    gap_rows: Sequence[dict[str, object]],
    fixed_rows: Sequence[dict[str, object]],
    method_rows: Sequence[dict[str, object]],
    temperature_rows: Sequence[dict[str, object]],
    robustness_rows: Sequence[dict[str, object]],
) -> None:
    bare_width = float(decoupled_rows[0]["fwhm_meV"])
    assert max(
        abs(float(row["fwhm_meV"]) - bare_width)
        for row in decoupled_rows
    ) < 1.0e-9
    assert max(
        float(row["max_sigma_difference_meV"])
        for row in decoupled_rows
    ) < 1.0e-10
    assert all(int(row["reliable"]) for row in gap_rows)
    assert all(int(row["up_window_peaks"]) == 1 for row in gap_rows)
    assert max(float(row["residual_meV"]) for row in gap_rows) < 1.0e-3
    assert max(
        abs(
            float(row["minus2ImSigma_meV"])
            - float(row["minus2ImSigmaB_meV"])
            - float(row["minus2ImSigmaD_meV"])
        )
        for row in gap_rows
    ) < 1.0e-3
    smallest_gap = min(float(row["gap_meV"]) for row in gap_rows)
    assert max(
        abs(float(row["excess_meV"]))
        for row in gap_rows
        if float(row["gap_meV"]) == smallest_gap
    ) < 0.1

    for n_molecules in (2, 4, 8, 16, 32, 64):
        subset = [
            row for row in gap_rows if int(row["N"]) == n_molecules
        ]
        decoupled = next(
            row
            for row in decoupled_rows
            if int(row["N"]) == n_molecules
        )
        dark_baseline = float(decoupled["minus2ImSigmaD_meV"])
        pearson_r_p = float(
            np.corrcoef(
                [float(row["excess_meV"]) for row in subset],
                [
                    float(row["minus2ImSigmaD_meV"]) - dark_baseline
                    for row in subset
                ],
            )[0, 1]
        )
        assert 0.89 < pearson_r_p < 0.92

    fixed_groups: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in fixed_rows:
        key = (int(row["N"]), float(row["gap_meV"]))
        fixed_groups.setdefault(key, []).append(row)
    assert max(
        np.ptp([float(row["fwhm_meV"]) for row in rows])
        for rows in fixed_groups.values()
    ) < 1.0e-3

    for method in ("full", "frozen_dark"):
        subset = [row for row in method_rows if row["method"] == method]
        assert all(int(row["reliable"]) for row in subset)
        assert all(int(row["up_window_peaks"]) == 1 for row in subset)

    for n_molecules in (4, 64):
        subset = [
            row
            for row in robustness_rows
            if int(row["N"]) == n_molecules
        ]
        production = next(
            row for row in subset if row["label"] == "production"
        )
        assert max(
            abs(
                float(row["fwhm_meV"])
                - float(production["fwhm_meV"])
            )
            for row in subset
        ) < 0.05
        assert all(int(row["up_window_peaks"]) == 1 for row in subset)

    for n_molecules in (4, 16, 64):
        subset = sorted(
            (
                row
                for row in temperature_rows
                if int(row["N"]) == n_molecules
                and float(row["gap_meV"]) == 5.0
            ),
            key=lambda row: float(row["temperature_K"]),
        )
        damping = [float(row["minus2ImSigmaD_meV"]) for row in subset]
        assert all(
            later > earlier
            for earlier, later in zip(damping, damping[1:])
        )
    print("Fig. S13 validation: all diagnostic assertions passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "decoupled",
            "gap",
            "fixed",
            "methods",
            "temperature",
            "robustness",
        ),
        default="all",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    settings = Settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    molecule_result, bare_fwhm, decoupled_rows = compute_decoupled(
        settings, args.force
    )
    if args.stage == "decoupled":
        return

    if args.stage in ("all", "gap"):
        gap_rows = compute_gap_scan(
            settings, bare_fwhm, args.workers, args.force
        )
    else:
        gap_rows = read_csv(DATA_DIR / "gap_scan.csv")
    if args.stage == "gap":
        return

    if args.stage in ("all", "fixed"):
        fixed_rows = compute_fixed_points(
            settings, molecule_result, bare_fwhm, args.force
        )
    else:
        fixed_rows = read_csv(DATA_DIR / "fixed_point_scan.csv")
    if args.stage == "fixed":
        return

    if args.stage in ("all", "methods"):
        method_rows = compute_methods(
            settings, molecule_result, bare_fwhm, args.force
        )
    else:
        method_rows = read_csv(DATA_DIR / "method_comparison.csv")
    if args.stage == "methods":
        return

    if args.stage in ("all", "temperature"):
        temperature_rows = compute_temperature(
            settings, bare_fwhm, args.workers, args.force
        )
    else:
        temperature_rows = read_csv(DATA_DIR / "temperature_scan.csv")
    if args.stage == "temperature":
        return

    if args.stage in ("all", "robustness"):
        robustness_rows = compute_robustness(
            settings, bare_fwhm, args.force
        )
    else:
        robustness_rows = read_csv(DATA_DIR / "robustness.csv")
    if args.stage == "robustness":
        return

    write_summaries(
        decoupled_rows,
        gap_rows,
        fixed_rows,
        method_rows,
        temperature_rows,
        robustness_rows,
    )
    validate_outputs(
        decoupled_rows,
        gap_rows,
        fixed_rows,
        method_rows,
        temperature_rows,
        robustness_rows,
    )
    print(f"Saved diagnostic tables in {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
