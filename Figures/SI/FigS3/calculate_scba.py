#!/usr/bin/env python3
"""Generate the continuous Brownian SCBA spectra."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np

from config import (
    CENTER_FREQUENCY_EV,
    CONTINUUM_STEP_EV,
    GAMMAS_EV,
    LAMBDAS_MEV,
    N_OUTPUT_OMEGA,
    OUTPUT_OMEGA_MAX_EV,
    OUTPUT_OMEGA_MIN_EV,
    gamma_tag,
)
from continuum_scba import (
    electronic_green_eV,
    make_continuum_grid,
    solve_continuum_scba,
)


ROOT = Path(__file__).resolve().parent


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else values.copy()


def interpolate_complex(
    target: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(target, source, values.real) + 1.0j * np.interp(
        target, source, values.imag
    )


def spectral_change(
    energy_eV: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
) -> tuple[float, float, float, float]:
    reference_area = float(np.trapz(reference, energy_eV))
    comparison_area = float(np.trapz(comparison, energy_eV))
    if reference_area <= 0.0 or comparison_area <= 0.0:
        raise RuntimeError("cannot compare nonpositive spectra")
    l1 = float(
        np.trapz(
            np.abs(
                reference / reference_area
                - comparison / comparison_area
            ),
            energy_eV,
        )
    )
    pointwise = float(
        np.max(np.abs(comparison - reference))
        / np.max(comparison)
    )
    peak_shift = float(
        abs(
            energy_eV[np.argmax(comparison)]
            - energy_eV[np.argmax(reference)]
        )
    )
    area_change = abs(comparison_area - reference_area) / comparison_area
    return l1, pointwise, peak_shift, area_change


def result_path(lambda_mev: float, gamma_eV: float) -> Path:
    return (
        ROOT
        / "results"
        / (
            f"scba_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )


def load_restart(
    path: Path, solver_energy_eV: np.ndarray
) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.loadtxt(path)
    return interpolate_complex(
        solver_energy_eV,
        data[:, 0],
        data[:, 3] + 1.0j * data[:, 4],
    )


def calculate_case(case: tuple[float, float]) -> dict[str, object]:
    lambda_mev, gamma_eV = case
    print(
        f"\ncontinuous Brownian SCBA: lambda={lambda_mev:.0f} meV, "
        f"Gamma={gamma_tag(gamma_eV)} eV",
        flush=True,
    )
    solver_energy = make_continuum_grid()
    path = result_path(lambda_mev, gamma_eV)
    result = solve_continuum_scba(
        lambda_mev,
        gamma_eV,
        energy_eV=solver_energy,
        initial_sigma_eV=load_restart(path, solver_energy),
    )
    if not result.converged:
        raise RuntimeError(f"SCBA failed to converge for {case}")

    output_energy = np.linspace(
        OUTPUT_OMEGA_MIN_EV,
        OUTPUT_OMEGA_MAX_EV,
        N_OUTPUT_OMEGA,
    )
    output_sigma = interpolate_complex(
        output_energy, result.energy_eV, result.sigma_eV
    )
    output_green = electronic_green_eV(
        output_energy, output_sigma, lambda_mev * 1.0e-3
    )
    output_spectrum = -output_green.imag
    np.savetxt(
        path,
        np.column_stack(
            (
                output_energy,
                output_spectrum,
                normalize(output_spectrum),
                output_sigma.real,
                output_sigma.imag,
            )
        ),
        fmt="%.12e",
        header=(
            "omega_eV  minus_Im_G_raw_per_eV  "
            "minus_Im_G_peak_normalized  Re_Sigma_eV  Im_Sigma_eV"
        ),
    )

    exact = np.loadtxt(
        ROOT
        / "exact_cumulant"
        / (
            f"cumulant_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )
    exact_interp = np.interp(output_energy, exact[:, 0], exact[:, 1])
    l1, pointwise, peak_shift, area_change = spectral_change(
        output_energy, exact_interp, output_spectrum
    )
    return {
        "lambda_meV": f"{lambda_mev:.1f}",
        "Gamma_eV": gamma_tag(gamma_eV),
        "omega_s_eV": f"{CENTER_FREQUENCY_EV:.1f}",
        "bath_treatment": "continuous_Brownian_FFT_Volterra",
        "converged": result.converged,
        "iterations": result.iterations,
        "final_electronic_error": f"{result.error_g:.16e}",
        "fixed_point_residual_eV": (
            f"{result.fixed_point_residual_eV:.16e}"
        ),
        "spectral_area": (
            f"{float(np.trapz(output_spectrum, output_energy)):.16e}"
        ),
        "min_spectrum_per_eV": f"{float(np.min(output_spectrum)):.16e}",
        "max_positive_Im_Sigma_eV": (
            f"{float(np.max(output_sigma.imag)):.16e}"
        ),
        "SCBA_exact_area_normalized_L1_error": f"{l1:.16e}",
        "SCBA_exact_max_pointwise_relative_error": (
            f"{pointwise:.16e}"
        ),
        "SCBA_exact_peak_shift_meV": f"{peak_shift * 1e3:.16e}",
        "SCBA_exact_area_relative_change": f"{area_change:.16e}",
        "solver_omega_min_eV": f"{solver_energy[0]:.8f}",
        "solver_omega_max_eV": f"{solver_energy[-1]:.8f}",
        "solver_step_meV": f"{CONTINUUM_STEP_EV * 1e3:.8f}",
        "solver_grid_points": len(solver_energy),
    }


def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    cases = [
        (lambda_mev, gamma_eV)
        for gamma_eV in GAMMAS_EV
        for lambda_mev in LAMBDAS_MEV
    ]
    workers = max(
        1, int(os.environ.get("FIGS1_BROWNIAN_WORKERS", "2"))
    )
    if workers == 1:
        rows = [calculate_case(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(calculate_case, cases))
    summary = ROOT / "results" / "scba_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
