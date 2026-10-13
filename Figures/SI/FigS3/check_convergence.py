#!/usr/bin/env python3
"""Independent time/frequency and SCBA-grid convergence audits."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from calculate_exact_cumulant import exact_spectrum
from calculate_scba import (
    interpolate_complex,
    result_path,
    spectral_change,
)
from config import (
    CUMULANT_AUDIT_MULTIPLIER,
    CUMULANT_FFT_MULTIPLIER,
    EXACT_AUDIT_CASES,
    MAX_EXACT_L1_CHANGE,
    MAX_EXACT_POINTWISE_CHANGE,
    MAX_SCBA_GRID_L1_CHANGE,
    MAX_SCBA_GRID_PEAK_SHIFT_EV,
    MAX_SCBA_GRID_POINTWISE_CHANGE,
    N_RESPONSE_FFT,
    N_TIME,
    OUTPUT_OMEGA_MAX_EV,
    OUTPUT_OMEGA_MIN_EV,
    SCBA_AUDIT_CASES,
    SCBA_AUDIT_OMEGA_MAX_EV,
    SCBA_AUDIT_OMEGA_MIN_EV,
    SCBA_AUDIT_STEP_EV,
    TIME_STEP_AU,
    gamma_tag,
)
from continuum_scba import (
    electronic_green_eV,
    make_continuum_grid,
    solve_continuum_scba,
)


ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_spectra(
    reference: tuple[np.ndarray, np.ndarray],
    comparison: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float, float]:
    ref_energy, ref_spectrum = reference
    selected = (
        (ref_energy >= OUTPUT_OMEGA_MIN_EV)
        & (ref_energy <= OUTPUT_OMEGA_MAX_EV)
    )
    energy = ref_energy[selected]
    ref_values = ref_spectrum[selected]
    values = np.interp(energy, comparison[0], comparison[1])
    return spectral_change(energy, ref_values, values)


def exact_audit_case(
    case: tuple[float, float],
) -> list[dict[str, object]]:
    lambda_mev, gamma_eV = case
    production_time = np.arange(N_TIME, dtype=float) * TIME_STEP_AU
    double_time = np.arange(2 * N_TIME, dtype=float) * TIME_STEP_AU
    half_step_time = (
        np.arange(2 * N_TIME - 1, dtype=float)
        * (0.5 * TIME_STEP_AU)
    )
    calculations = {
        "production": (
            production_time,
            N_RESPONSE_FFT,
            CUMULANT_FFT_MULTIPLIER,
        ),
        "double_time_window": (
            double_time,
            2 * N_RESPONSE_FFT,
            CUMULANT_FFT_MULTIPLIER,
        ),
        "half_time_step": (
            half_step_time,
            2 * N_RESPONSE_FFT,
            CUMULANT_FFT_MULTIPLIER,
        ),
        "double_cumulant_padding": (
            production_time,
            N_RESPONSE_FFT,
            CUMULANT_AUDIT_MULTIPLIER,
        ),
    }
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    endpoints: dict[str, float] = {}
    for name, (time, response_n_fft, cumulant_multiplier) in (
        calculations.items()
    ):
        print(
            f"exact audit {case}: {name}",
            flush=True,
        )
        energy, spectrum, endpoint = exact_spectrum(
            lambda_mev,
            gamma_eV,
            time,
            response_n_fft=response_n_fft,
            cumulant_fft_multiplier=cumulant_multiplier,
        )
        spectra[name] = energy, spectrum
        endpoints[name] = endpoint

    rows = []
    for name in (
        "double_time_window",
        "half_time_step",
        "double_cumulant_padding",
    ):
        l1, pointwise, peak_shift, area_change = compare_spectra(
            spectra["production"], spectra[name]
        )
        if l1 > MAX_EXACT_L1_CHANGE:
            raise RuntimeError(
                f"exact {name} L1 audit failed for {case}: {l1:.3e}"
            )
        if pointwise > MAX_EXACT_POINTWISE_CHANGE:
            raise RuntimeError(
                f"exact {name} pointwise audit failed for {case}: "
                f"{pointwise:.3e}"
            )
        rows.append(
            {
                "lambda_meV": f"{lambda_mev:.1f}",
                "Gamma_eV": gamma_tag(gamma_eV),
                "test": name,
                "area_normalized_L1_change": f"{l1:.16e}",
                "max_pointwise_relative_change": f"{pointwise:.16e}",
                "peak_shift_meV": f"{peak_shift * 1e3:.16e}",
                "spectral_area_relative_change": (
                    f"{area_change:.16e}"
                ),
                "production_endpoint_response": (
                    f"{endpoints['production']:.16e}"
                ),
                "test_endpoint_response": f"{endpoints[name]:.16e}",
            }
        )
    return rows


def scba_grid_audit_case(
    case: tuple[float, float],
) -> dict[str, object]:
    lambda_mev, gamma_eV = case
    production = np.loadtxt(result_path(lambda_mev, gamma_eV))
    audit_energy = make_continuum_grid(
        step_eV=SCBA_AUDIT_STEP_EV,
        omega_min_eV=SCBA_AUDIT_OMEGA_MIN_EV,
        omega_max_eV=SCBA_AUDIT_OMEGA_MAX_EV,
    )
    initial_sigma = interpolate_complex(
        audit_energy,
        production[:, 0],
        production[:, 3] + 1.0j * production[:, 4],
    )
    print(f"SCBA grid audit {case}", flush=True)
    result = solve_continuum_scba(
        lambda_mev,
        gamma_eV,
        energy_eV=audit_energy,
        initial_sigma_eV=initial_sigma,
    )
    if not result.converged:
        raise RuntimeError(f"refined SCBA failed to converge for {case}")

    comparison_energy = production[:, 0]
    production_spectrum = production[:, 1]
    audit_sigma = interpolate_complex(
        comparison_energy, result.energy_eV, result.sigma_eV
    )
    audit_spectrum = -electronic_green_eV(
        comparison_energy, audit_sigma, lambda_mev * 1e-3
    ).imag
    l1, pointwise, peak_shift, area_change = spectral_change(
        comparison_energy, production_spectrum, audit_spectrum
    )
    if l1 > MAX_SCBA_GRID_L1_CHANGE:
        raise RuntimeError(
            f"SCBA grid L1 audit failed for {case}: {l1:.3e}"
        )
    if pointwise > MAX_SCBA_GRID_POINTWISE_CHANGE:
        raise RuntimeError(
            f"SCBA grid pointwise audit failed for {case}: "
            f"{pointwise:.3e}"
        )
    if peak_shift > MAX_SCBA_GRID_PEAK_SHIFT_EV + 1.0e-12:
        raise RuntimeError(
            f"SCBA grid peak audit failed for {case}: "
            f"{peak_shift * 1e3:.6g} meV"
        )
    return {
        "lambda_meV": f"{lambda_mev:.1f}",
        "Gamma_eV": gamma_tag(gamma_eV),
        "production_step_meV": "0.025",
        "audit_step_meV": f"{SCBA_AUDIT_STEP_EV * 1e3:.8g}",
        "audit_omega_min_eV": f"{SCBA_AUDIT_OMEGA_MIN_EV:.1f}",
        "audit_omega_max_eV": f"{SCBA_AUDIT_OMEGA_MAX_EV:.1f}",
        "audit_grid_points": len(audit_energy),
        "iterations": result.iterations,
        "final_electronic_error": f"{result.error_g:.16e}",
        "area_normalized_L1_change": f"{l1:.16e}",
        "max_pointwise_relative_change": f"{pointwise:.16e}",
        "peak_shift_meV": f"{peak_shift * 1e3:.16e}",
        "spectral_area_relative_change": f"{area_change:.16e}",
    }


def main() -> None:
    output = ROOT / "convergence"
    output.mkdir(exist_ok=True)
    exact_rows = [
        row
        for case in EXACT_AUDIT_CASES
        for row in exact_audit_case(case)
    ]
    write_csv(output / "exact_cumulant_convergence.csv", exact_rows)
    grid_rows = [scba_grid_audit_case(case) for case in SCBA_AUDIT_CASES]
    write_csv(output / "scba_grid_convergence.csv", grid_rows)

    max_exact_l1 = max(
        float(row["area_normalized_L1_change"]) for row in exact_rows
    )
    max_exact_pointwise = max(
        float(row["max_pointwise_relative_change"]) for row in exact_rows
    )
    max_grid_l1 = max(
        float(row["area_normalized_L1_change"]) for row in grid_rows
    )
    max_grid_pointwise = max(
        float(row["max_pointwise_relative_change"]) for row in grid_rows
    )
    max_grid_peak = max(
        float(row["peak_shift_meV"]) for row in grid_rows
    )
    report = output / "convergence_report.txt"
    report.write_text(
        "\n".join(
            (
                "Fig. S4 Brownian convergence audit",
                "production_bath_treatment = continuous_Brownian_integral",
                "discrete_mode_count = not_applicable",
                f"exact_audit_case_count = {len(EXACT_AUDIT_CASES)}",
                (
                    "max_exact_area_normalized_L1_change = "
                    f"{max_exact_l1:.16e}"
                ),
                (
                    "max_exact_pointwise_relative_change = "
                    f"{max_exact_pointwise:.16e}"
                ),
                f"SCBA_grid_audit_case_count = {len(SCBA_AUDIT_CASES)}",
                (
                    "max_SCBA_grid_area_normalized_L1_change = "
                    f"{max_grid_l1:.16e}"
                ),
                (
                    "max_SCBA_grid_pointwise_relative_change = "
                    f"{max_grid_pointwise:.16e}"
                ),
                f"max_SCBA_grid_peak_shift_meV = {max_grid_peak:.16e}",
                "status = PASS",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
